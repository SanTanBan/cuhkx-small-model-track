"""
Reproducible inference for the packaged CUHK-X Small Model Track solution.

    python infer.py --data /path/to/small_model_track_test --out predictions.csv \
                    --ckpt ../checkpoints/model.pth

Reads raw clips in the released layout (<data>/<clip>/<modality>/...), applies
exactly the preprocessing that built the training shards (prep.py: the person
detector, the crop, the JPEG round trip), runs every ensemble member stored in
the single checkpoint with deterministic test-time augmentation (centre +
mirror), fuses modalities with the stored out-of-fold weights, and writes one
prediction per clip. Nothing is random, so the same inputs always yield the
same CSV — which is what the organisers' reproduction check compares against.

Checkpoint layout: {"models": {member: state_dict}, "meta": {"weights",
"members"}, "detector": state_dict}. A member's tensors are fp16, or int8 with
a per-channel scale (cuhkx.quantize_int8); the detector is stored unconverted
(fp32) so test crops come from the very network that made the training crops.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import prep  # noqa: E402
from cuhkx import (COMPOSITES, NUM_CLASSES, VIDEO_ARCHS, Cfg, IMUNet,  # noqa: E402
                   SkeletonNet, VideoNet, VideoNet3D, dequantize, norm_stats)

IMAGE = dict(frames=16, stored=160, crop=True, quality=90)
SKEL = dict(frames=32)
MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]
PERSON_CROP = "_PC"                    # packed from person-box crops of the raw frames
FULL_FRAME = (-1, -1, -1, -1, 1, 1)    # the detector found nobody: keep the whole frame


# ------------------------------------------------------------ preprocessing

def base_mod(mod):
    """Raw folder a packed modality was built from: Depth_Color_PC -> Depth_Color."""
    return mod[:-len(PERSON_CROP)] if mod.endswith(PERSON_CROP) else mod


def parts_of(mod):
    """[(packed modality, channels)] a member reads: a composite's parts, else itself."""
    return list(COMPOSITES[mod]) if mod in COMPOSITES else [(mod, 3)]


def person_box(det, ir_dir, device):
    """
    The clip's crop box from its IR frames, computed exactly as the Kaggle
    detection kernel computed the training boxes: prep.det_jpegs ->
    det_input -> detector -> best_person -> person_box.
    """
    res = prep.det_jpegs(ir_dir) if os.path.isdir(ir_dir) else None
    if res is None:
        return FULL_FRAME
    bufs, (h, w) = res
    xs = [prep.det_input(b) for b in bufs]
    ok = [x for x in xs if x is not None]
    outs = iter(det([x.to(device) for x in ok]) if ok else [])
    fb = [prep.best_person({k: v.cpu() for k, v in next(outs).items()}) if x is not None
          else None for x in xs]
    box = prep.person_box(fb, (h, w))
    return FULL_FRAME if box is None else (*box, h, w)


def stored_frames(clip_dir, modality, box=None):
    """Raw frames -> the T frames exactly as packed for training (person box or
    foreground crop, resize to 160, JPEG q90), decoded to uint8 BGR."""
    args = (clip_dir, modality, IMAGE["frames"], IMAGE["stored"], IMAGE["crop"],
            IMAGE["quality"])
    res = prep.encode_clip(args + ((box,) if box is not None else ()))
    if res is None:
        return None
    return [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in res[1]]


def member_input(frames_by_part, mod, size):
    """
    ClipDataset(train=False) for one member: a composite stacks its parts
    (RGB depth, then one IR channel; a missing part stays black), then the 90%
    centre crop and resize -> [T,C,size,size] float in [0,1].
    """
    if mod in COMPOSITES:
        ref = next(f for f in frames_by_part if f is not None)
        H, W = ref[0].shape[:2]
        layers = []
        for (_, ch), fr in zip(COMPOSITES[mod], frames_by_part):
            if fr is None:
                layers.append([np.zeros((H, W, ch), np.uint8)] * len(ref))
            else:
                layers.append([f[:, :, :1] if ch == 1 else cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                               for f in fr])
        frames = [np.concatenate([l[k] if l[k].shape[:2] == (H, W)
                                  else cv2.resize(l[k], (W, H)).reshape(H, W, -1)
                                  for l in layers], axis=2)
                  for k in range(len(ref))]
    else:
        frames = frames_by_part[0]
    H, W = frames[0].shape[:2]
    side = int(min(H, W) * 0.90)
    y0, x0 = (H - side) // 2, (W - side) // 2
    out = [cv2.resize(im[y0:y0 + side, x0:x0 + side], (size, size),
                      interpolation=cv2.INTER_LINEAR) for im in frames]
    return torch.from_numpy(np.stack(out).astype(np.float32) / 255.0).permute(0, 3, 1, 2)


def skeleton_clip(clip_dir):
    """Raw pose JSON -> [T,17,3], with the fp16 storage round trip of training."""
    res = prep.encode_skeleton((clip_dir, SKEL["frames"]))
    if res is None:
        return None
    _, seq = res
    seq = prep.normalize_pose(seq).astype(np.float16).astype(np.float32)
    T = SKEL["frames"]
    if seq.shape[0] != T:
        seq = seq[np.round(np.linspace(0, seq.shape[0] - 1, T)).astype(int)]
    return torch.from_numpy(np.ascontiguousarray(seq))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def imu_clip(imu_dir, T):
    """Raw IMU CSVs -> [7*D, T] tensor, exactly as scripts/imu_pack.py packed training."""
    if not os.path.isdir(imu_dir):
        return None
    bufs = [open(os.path.join(imu_dir, f), "rb").read() for f in sorted(os.listdir(imu_dir))
            if f.lower().endswith(".csv")]
    x = prep.encode_imu(bufs, T)
    return None if x is None else torch.from_numpy(x)


def mirror_pose(x):
    x = x.clone()
    x[..., 0] *= -1
    idx = list(range(x.shape[-2]))
    for a, b in MIRROR_PAIRS:
        idx[a], idx[b] = b, a
    return x[..., idx, :]


# ------------------------------------------------------------------ models

def member_mod(key):
    tag = key.split(":", 1)[-1]
    return tag[:-5] if tag.endswith("_full") else tag.rsplit("_f", 1)[0]


def build_members(ckpt, device):
    """
    {stream: (data_modality, [(net, opts)])}. A stream is one fused input —
    two skeleton recipes are two streams over the same Skeleton data — and
    opts is that member's training-time config: architecture, input size and
    normalisation, skeleton height handling.
    """
    info = ckpt.get("meta", {}).get("members", {})
    streams = {}
    for key, sd in ckpt["models"].items():
        opts = dict(info.get(key, {}))
        mod = opts.get("modality") or member_mod(key)
        stream = opts.get("stream") or (key.split(":", 1)[0] if ":" in key else mod)
        opts["size"] = int(opts.get("size", 144))
        if mod.upper().startswith("IMU"):
            net = IMUNet(int(opts.get("imu_ch", 35)), width=int(opts.get("imu_width", 48)))
        elif mod.lower().startswith("skel"):
            net = SkeletonNet()
        else:
            cfg = Cfg(modality=mod, frames=IMAGE["frames"], size=opts["size"],
                      arch=opts.get("arch", "resnet18"))
            net = (VideoNet3D(cfg, pretrained=False) if cfg.arch in VIDEO_ARCHS
                   else VideoNet(cfg, pretrained=False))
        net.load_state_dict(dequantize(sd))
        streams.setdefault(stream, (mod, []))[1].append((net.to(device).eval(), opts))
    return streams


def normalise(x, norm):
    """[B,T,C,H,W] in [0,1] -> that member's training-time input normalisation."""
    if norm in ("kinetics", "imagenet"):
        mean, std = norm_stats(norm, x.shape[2])
        return ((x - mean.view(1, 1, -1, 1, 1).to(x.device))
                / std.view(1, 1, -1, 1, 1).to(x.device))
    dims = tuple(range(1, x.dim()))            # per clip, as ClipDataset does
    return (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True).clamp_min(1e-4)


@torch.no_grad()
def predict_imu(net, x, device):
    """IMU members see one deterministic view, as in training's test predictions."""
    return net(x.to(device)).float().softmax(-1).cpu().numpy()


@torch.no_grad()
def predict_member(net, opts, x, skel, device):
    """Mean softmax of one member over the two deterministic views."""
    x = x.to(device)
    if skel:
        if opts.get("center_z"):
            # this member was trained with the root's height removed per frame
            x = x.clone()
            x[..., 2] = x[..., 2] - x[:, :, 0:1, 2]
        views = (x, mirror_pose(x))
    else:
        x = normalise(x, opts.get("norm", "clip"))
        views = (x, x.flip(-1))
    return (sum(net(v).float().softmax(-1) for v in views) / 2).cpu().numpy()


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding <clip>/<modality>/")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "checkpoints", "model.pth"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N clips only")
    ap.add_argument("--save-probs", help="optional .npy of fused probabilities")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    # full-precision convolutions/matmuls: the same numbers on any GPU
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    weights = ckpt.get("meta", {}).get("weights", {})
    streams = build_members(ckpt, device)
    if not weights:
        weights = {s: 1.0 for s in streams}
    det = None
    if any(p.endswith(PERSON_CROP) for mod, _ in streams.values() for p, _ in parts_of(mod)):
        if "detector" not in ckpt:
            raise SystemExit("person-crop members need the packaged person detector")
        det = prep.make_detector(ckpt["detector"]).to(device)
    print(f"device {device} | streams " +
          ", ".join(f"{s}({m})x{len(n)}" for s, (m, n) in streams.items()) +
          f" | weights {weights} | detector {'yes' if det is not None else 'no'}", flush=True)

    root = os.path.normpath(args.data)
    clips = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)) and not prep.is_junk(d)
                   and not d.startswith("."))
    if args.limit:
        clips = clips[:args.limit]

    probs = {s: np.full((len(clips), NUM_CLASSES), np.nan) for s in streams}
    boxes, t0 = {}, time.time()
    for s0 in range(0, len(clips), args.batch):
        chunk = list(range(s0, min(s0 + args.batch, len(clips))))
        cache = {}                     # (clip, packed modality) -> stored frames

        def frames_for(i, pm):
            if (i, pm) not in cache:
                raw = os.path.join(root, clips[i], base_mod(pm))
                box = None
                if pm.endswith(PERSON_CROP):
                    if i not in boxes:
                        boxes[i] = person_box(det, os.path.join(root, clips[i], "IR"), device)
                    box = boxes[i]
                cache[(i, pm)] = (stored_frames(raw, base_mod(pm), box)
                                  if os.path.isdir(raw) else None)
            return cache[(i, pm)]

        for stream, (mod, nets) in streams.items():
            skel = mod.lower().startswith("skel")
            imu = mod.upper().startswith("IMU")
            inputs, keep = [], []
            for i in chunk:
                if imu:
                    x = imu_clip(os.path.join(root, clips[i], "IMU"),
                                 int(nets[0][1].get("frames", 40)))
                elif skel:
                    d = os.path.join(root, clips[i], mod)
                    x = skeleton_clip(d) if os.path.isdir(d) else None
                else:
                    fbp = [frames_for(i, pm) for pm, _ in parts_of(mod)]
                    x = None if all(f is None for f in fbp) else fbp
                if x is not None:
                    inputs.append(x)
                    keep.append(i)
            if not keep:
                continue
            p = 0
            for net, opts in nets:
                x = (torch.stack(inputs) if skel or imu else
                     torch.stack([member_input(f, mod, opts["size"]) for f in inputs]))
                p = p + (predict_imu(net, x, device) if imu else
                         predict_member(net, opts, x, skel, device))
            probs[stream][keep] = p / len(nets)
        print(f"  {chunk[-1] + 1:>4}/{len(clips)} clips  {time.time() - t0:6.1f}s", flush=True)

    # Streams are fused by the rule their weights were fitted for: a linear
    # mixture, or a geometric pool (p ∝ Π_s p_s^w_s). A stream missing for a
    # clip is left out of that clip's fusion.
    fusion = ckpt.get("meta", {}).get("fusion", "linear")
    num = np.zeros((len(clips), NUM_CLASSES))
    den = np.zeros(len(clips))
    for stream, p in probs.items():
        w = float(weights.get(stream, 0.0))
        ok = ~np.isnan(p).any(1)
        if w > 0:
            num[ok] += w * (np.log(np.clip(p[ok], 1e-8, 1)) if fusion == "loglin" else p[ok])
            den[ok] += w
        print(f"  {stream:<14} {ok.sum():>4}/{len(clips)} clips  weight {w:.2f}", flush=True)
    fused = np.zeros_like(num)
    has = den > 0
    fused[has] = softmax(num[has]) if fusion == "loglin" else num[has] / den[has, None]

    orphan = ~has
    on, od = np.zeros_like(num), np.zeros(len(clips))
    for stream, p in probs.items():            # clip seen only by 0-weight streams
        ok = orphan & ~np.isnan(p).any(1)
        on[ok] += p[ok]
        od[ok] += 1
    if (orphan & (od == 0)).any():
        missing = [clips[i] for i in np.where(orphan & (od == 0))[0]]
        raise SystemExit(f"no usable modality for {len(missing)} clips: {missing[:5]}")
    fused[orphan] = on[orphan] / od[orphan, None]
    steps = int(ckpt.get("meta", {}).get("balance_steps", 0))
    if steps:
        # the package's label-free test-time rebalancing towards the training class mix
        prior = np.asarray(ckpt["meta"]["class_prior"], dtype=np.float64)
        for _ in range(steps):
            fused = fused / (fused.sum(0) / (len(fused) * prior))[None]
            fused = fused / fused.sum(1, keepdims=True)

    prefix = os.path.basename(root)
    pd.DataFrame({"path": [f"{prefix}/{c}/" for c in clips],
                  "prediction": fused.argmax(1).astype(int)}).to_csv(args.out, index=False)
    if args.save_probs:
        np.save(args.save_probs, fused)
    dt = time.time() - t0
    print(f"wrote {args.out}: {len(clips)} clips in {dt:.1f}s "
          f"({1000 * dt / max(len(clips), 1):.0f} ms/clip on {device}); "
          f"{int(orphan.sum())} clips filled from zero-weight streams; "
          f"person boxes for {sum(b[0] >= 0 for b in boxes.values())}/{len(boxes)} clips")


if __name__ == "__main__":
    main()
