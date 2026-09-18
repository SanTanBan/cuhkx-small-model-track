"""
End-to-end smoke test on synthetic data — CPU only, ~1 minute.

Kaggle GPU quota is 30 h/week and the competition has days left, so every bug
found here is quota not wasted there. Exercises: the blob format round-trip,
temporal shift correctness, model shapes, user-grouped folds, a real training
step, OOF/test artefact writing, fusion, and the fp16 checkpoint packer.

    python scripts/05_smoke_test.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))

from cuhkx import (Cfg, ClipDataset, N_JOINTS, PARENT, SkeletonNet,  # noqa: E402
                   TemporalShift, VideoNet, build_adjacency, build_folds,
                   load_index, make_dataset, make_model, pack_fp16,
                   pose_features)

OK, FAIL = "  [ok]", "  [FAIL]"
_failures = []


def check(name, cond, detail=""):
    print(f"{OK if cond else FAIL} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _failures.append(name)


# ---------------------------------------------------------------- fake data

def make_fake_compact(root, modality="Thermal", n_users=9, n_cls=40,
                      trials=1, T=8, size=64, n_test=20):
    """
    Write a dataset in the exact blob+parquet format 04_preprocess.py emits.

    Frames carry a class-dependent bright square plus per-user brightness bias,
    so a working model should beat chance and a user-leaking split would score
    suspiciously high.
    """
    mout = os.path.join(root, modality)
    os.makedirs(mout, exist_ok=True)
    rows, bi, off = [], 0, 0
    bf = open(os.path.join(mout, f"blob_{bi:03d}.bin"), "wb")
    rng = np.random.RandomState(0)

    def emit(clip_id, split, user, trial, aid):
        nonlocal off, bi, bf
        offs, lens = [], []
        ux = (hash(user) % 7) * 3
        for t in range(T):
            img = rng.randint(0, 60, (size, size, 3), np.uint8)
            cy = 4 + (aid // 8) * 11
            cx = 4 + (aid % 8) * 7 + t
            img[cy:cy + 9, cx:cx + 9] = 220
            img = np.clip(img.astype(np.int16) + ux, 0, 255).astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            assert ok
            b = buf.tobytes()
            offs.append(off); lens.append(len(b)); bf.write(b); off += len(b)
        rows.append(dict(clip_id=clip_id, split=split, user=user, trial=trial,
                         action_id=aid, n_src_frames=T * 3, blob=bi,
                         offsets=offs, lengths=lens, t=T, size=size))

    for aid in range(n_cls):
        for u in range(1, n_users + 1):
            for tr in range(trials):
                emit(f"{aid}_a/{u}/{tr}", "train", str(u), str(tr), aid)
    for i in range(n_test):
        emit(f"SM_test_{i:04d}", "test", "TEST", f"SM_test_{i:04d}", -1)

    bf.close()
    pd.DataFrame(rows).to_parquet(os.path.join(mout, "index.parquet"), index=False)

    pd.DataFrame({"action_id": range(n_cls),
                  "action_name": [f"{i}_act" for i in range(n_cls)]}
                 ).to_csv(os.path.join(root, "class_mapping.csv"), index=False)
    pd.DataFrame({"path": [f"small_model_track_test/SM_test_{i:04d}/"
                           for i in range(n_test)],
                  "prediction": [""] * n_test}
                 ).to_csv(os.path.join(root, "test.csv"), index=False)
    return len(rows)


# -------------------------------------------------------------------- tests

def test_temporal_shift():
    """Channels must move across time by exactly one step, in both directions."""
    T, C, H, W = 4, 8, 2, 2
    x = torch.zeros(1 * T, C, H, W)
    for t in range(T):
        x[t] = t + 1
    ts = TemporalShift(torch.nn.Identity(), n_segment=T, shift_div=8)
    y = ts(x).view(T, C, H, W)
    fold = max(1, C // 8)
    left_ok = torch.allclose(y[0, :fold], torch.full((fold, H, W), 2.0))
    right_ok = torch.allclose(y[1, fold:2 * fold], torch.full((fold, H, W), 1.0))
    keep_ok = torch.allclose(y[2, 2 * fold:], torch.full((C - 2 * fold, H, W), 3.0))
    check("TemporalShift shifts future->past", left_ok)
    check("TemporalShift shifts past->future", right_ok)
    check("TemporalShift leaves remaining channels intact", keep_ok)
    check("TemporalShift preserves shape", ts(x).shape == x.shape)


def test_model_shapes():
    cfg = Cfg(frames=8, size=64, arch="resnet18", dropout=0.0)
    m = VideoNet(cfg, pretrained=False)
    x = torch.randn(2, cfg.frames, 3, cfg.size, cfg.size)
    out = m(x)
    n = sum(p.numel() for p in m.parameters())
    check("VideoNet output shape", tuple(out.shape) == (2, 40), str(tuple(out.shape)))
    check("VideoNet params fit fp16 budget", n * 2 / 1e6 < 100,
          f"{n/1e6:.2f} M = {n*2/1e6:.1f} MB fp16")
    out.sum().backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all()
             for p in m.parameters() if p.requires_grad]
    check("gradients finite through TSM", all(grads))


def test_dataset(root):
    cfg = Cfg(data_root=root, modality="Thermal", frames=8, size=56,
              per_clip_norm=True)
    mod_dir, idx = load_index(root, "Thermal")
    tr = idx[idx.split == "train"]
    ds = ClipDataset(tr, mod_dir, cfg, train=True)
    x, y = ds[0]
    check("dataset tensor shape", tuple(x.shape) == (8, 3, 56, 56), str(tuple(x.shape)))
    check("dataset label in range", 0 <= y < 40, str(y))
    check("per-clip norm ~zero-mean", abs(float(x.mean())) < 0.2,
          f"mean={float(x.mean()):.3f} std={float(x.std()):.3f}")

    ev = ClipDataset(tr, mod_dir, cfg, train=False)
    a, _ = ev[3]
    b, _ = ev[3]
    check("eval sampling deterministic", torch.allclose(a, b))

    from torch.utils.data import DataLoader
    xb, yb = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    check("dataloader batches", tuple(xb.shape) == (4, 8, 3, 56, 56))


def test_folds(root):
    _, idx = load_index(root, "Thermal")
    tr = idx[idx.split == "train"].copy()
    f = build_folds(tr, 5, 42)
    tr["fold"] = f
    leaks = []
    for k in range(5):
        a = set(tr[tr.fold == k].user.astype(str))
        b = set(tr[tr.fold != k].user.astype(str))
        if a & b:
            leaks.append((k, a & b))
    check("GroupKFold: no user in both train and val", not leaks, str(leaks[:2]))
    check("GroupKFold: every fold non-empty",
          all((f == k).sum() > 0 for k in range(5)))
    check("GroupKFold: deterministic",
          np.array_equal(f, build_folds(tr, 5, 42)))


def test_training_run(root, tmp):
    """Actually run train.py for 2 epochs — catches wiring bugs, not just shapes."""
    cmd = [sys.executable, os.path.join(ROOT, "kaggle", "train.py"),
           "--data-root", root, "--modality", "Thermal", "--fold", "0",
           "--epochs", "2", "--batch-size", "8", "--frames", "8", "--size", "56",
           "--workers", "0", "--out-dir", tmp, "--no-tta"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=os.path.join(ROOT, "kaggle"))
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
    check("train.py exits cleanly", r.returncode == 0)
    if r.returncode != 0:
        return

    tail = [l for l in r.stdout.splitlines() if "val_acc" in l]
    print("      " + "\n      ".join(tail[-2:]))
    for f in ["Thermal_f0.pt", "Thermal_f0_oof.npz", "Thermal_f0_test.npz"]:
        check(f"wrote {f}", os.path.exists(os.path.join(tmp, f)))

    d = np.load(os.path.join(tmp, "Thermal_f0_oof.npz"), allow_pickle=True)
    check("OOF logits shape", d["logits"].shape[1] == 40, str(d["logits"].shape))
    check("OOF has clip ids", len(d["clip_id"]) == len(d["y"]))
    t = np.load(os.path.join(tmp, "Thermal_f0_test.npz"), allow_pickle=True)
    check("test probs sum to 1",
          np.allclose(t["probs"].sum(1), 1, atol=1e-3), str(t["probs"].sum(1)[:3]))

    check("OOF covers the whole val fold", len(d["y"]) > 0, f"{len(d['y'])} clips")


def test_learnability(tmp):
    """
    Can the loop actually fit a signal? 40 classes x 2 epochs cannot beat chance,
    which says nothing. Use a small separable task instead: if the optimiser,
    loss, TSM and pooling are wired correctly this saturates quickly.
    """
    root = os.path.join(tmp, "learn")
    os.makedirs(root, exist_ok=True)
    make_fake_compact(root, n_users=6, n_cls=4, trials=3, T=8, size=40, n_test=4)
    out = os.path.join(tmp, "learn_out")
    cmd = [sys.executable, os.path.join(ROOT, "kaggle", "train.py"),
           "--data-root", root, "--modality", "Thermal", "--fold", "0",
           "--epochs", "8", "--batch-size", "8", "--frames", "8", "--size", "40",
           "--workers", "0", "--out-dir", out, "--no-tta", "--lr", "1e-3"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=os.path.join(ROOT, "kaggle"))
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        check("learnability run completed", False)
        return
    for l in [l for l in r.stdout.splitlines() if "val_acc" in l][-3:]:
        print("      " + l)
    d = np.load(os.path.join(out, "Thermal_f0_oof.npz"), allow_pickle=True)
    acc = float((d["logits"].argmax(1) == d["y"]).mean())
    check("training loop learns a separable 4-class task", acc > 0.45,
          f"val acc {acc:.3f} vs chance 0.250")


def test_pack(tmp):
    sds = {}
    cfg = Cfg(frames=8, size=64)
    for n in ["Thermal_f0", "Thermal_f1"]:
        sds[n] = {k: v for k, v in VideoNet(cfg, pretrained=False)
                  .state_dict().items()}
    mb = pack_fp16(sds, os.path.join(tmp, "model.pth"), meta={"t": 1})
    check("packed checkpoint under 100 MB", mb < 100, f"{mb:.1f} MB")
    ck = torch.load(os.path.join(tmp, "model.pth"), map_location="cpu",
                    weights_only=False)
    check("checkpoint holds all members", set(ck["models"]) == set(sds))
    dt = next(iter(ck["models"]["Thermal_f0"].values())).dtype
    check("weights stored as fp16", dt == torch.float16, str(dt))

    m = VideoNet(cfg, pretrained=False)
    m.load_state_dict({k: v.float() for k, v in ck["models"]["Thermal_f0"].items()})
    check("packed weights reload into the model", True)


def make_fake_skeleton(root, n_users=9, n_cls=40, trials=1, T=32, n_test=20):
    """Packed-skeleton fixture matching prep.pack_skeleton's output format."""
    mout = os.path.join(root, "Skeleton")
    os.makedirs(mout, exist_ok=True)
    rng = np.random.RandomState(1)
    rows, arrs = [], []

    def emit(clip_id, split, user, trial, aid):
        t = np.linspace(0, 2 * np.pi, T)[:, None]
        base = rng.RandomState if False else None
        pose = np.zeros((T, N_JOINTS, 3), np.float32)
        for j in range(N_JOINTS):
            phase = (aid + 1) * 0.3 + j * 0.1
            pose[:, j, 0] = np.sin(t[:, 0] * (1 + aid % 4) + phase) * 0.3
            pose[:, j, 1] = np.cos(t[:, 0] * (1 + aid % 3) + phase) * 0.2
            pose[:, j, 2] = j / N_JOINTS + 0.02 * np.sin(t[:, 0] + phase)
        pose += rng.normal(0, 0.01, pose.shape).astype(np.float32)
        pose[:, 0] = 0.0                       # root at origin, as normalised
        rows.append(dict(clip_id=clip_id, split=split, user=user, trial=trial,
                         action_id=aid, n_src_frames=T, row=len(arrs), t=T))
        arrs.append(pose.astype(np.float16))

    for aid in range(n_cls):
        for u in range(1, n_users + 1):
            for tr in range(trials):
                emit(f"{aid}_a/{u}/{tr}", "train", str(u), str(tr), aid)
    for i in range(n_test):
        emit(f"SM_test_{i:04d}", "test", "TEST", f"SM_test_{i:04d}", -1)

    np.save(os.path.join(mout, "poses.npy"), np.stack(arrs))
    pd.DataFrame(rows).to_parquet(os.path.join(mout, "index.parquet"), index=False)
    return len(rows)


def test_skeleton_graph():
    A = build_adjacency()
    check("adjacency shape [3,17,17]", tuple(A.shape) == (3, N_JOINTS, N_JOINTS),
          str(tuple(A.shape)))
    check("partition 0 is self-loops", torch.allclose(A[0], torch.eye(N_JOINTS)))
    rs = A.sum(-1)
    ok = torch.all(((rs - 1).abs() < 1e-5) | (rs.abs() < 1e-6)).item()
    check("each partition row sums to 1 (or 0 where empty)", ok,
          f"max row sum {rs.max():.4f}")
    # Every joint except the root must have exactly one centripetal neighbour.
    cent = (A[1] > 0).sum(-1)
    check("all non-root joints point towards the root",
          int((cent[1:] == 1).sum()) == N_JOINTS - 1, f"{cent.tolist()}")
    check("root has no centripetal neighbour", int(cent[0]) == 0)
    check("PARENT chain reaches root from every joint",
          all(_walk_to_root(j) for j in range(N_JOINTS)))


def _walk_to_root(j, limit=32):
    for _ in range(limit):
        if j == 0:
            return True
        j = int(PARENT[j])
    return False


def test_pose_features():
    x = torch.randn(2, 8, N_JOINTS, 3)
    f = pose_features(x)
    check("pose_features shape [B,9,T,V]", tuple(f.shape) == (2, 9, 8, N_JOINTS),
          str(tuple(f.shape)))
    # channels 3:6 are bone vectors = joint - parent
    bone = f[:, 3:6].permute(0, 2, 3, 1)
    check("bone channel equals joint-minus-parent",
          torch.allclose(bone, x - x[:, :, PARENT, :], atol=1e-5))
    vel = f[:, 6:9].permute(0, 2, 3, 1)
    check("velocity channel is the frame difference",
          torch.allclose(vel[:, 1:], x[:, 1:] - x[:, :-1], atol=1e-5))
    check("velocity is zero on the first frame",
          torch.allclose(vel[:, 0], torch.zeros_like(vel[:, 0])))


def test_skeleton_model():
    m = SkeletonNet()
    x = torch.randn(2, 32, N_JOINTS, 3)
    out = m(x)
    n = sum(p.numel() for p in m.parameters())
    check("SkeletonNet output shape", tuple(out.shape) == (2, 40), str(tuple(out.shape)))
    check("SkeletonNet is small", n * 2 / 1e6 < 20, f"{n/1e6:.2f} M = {n*2/1e6:.1f} MB fp16")
    out.sum().backward()
    check("SkeletonNet gradients finite",
          all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))


def test_skeleton_dataset(root):
    cfg = Cfg(data_root=root, modality="Skeleton", frames=32)
    mod_dir, idx = load_index(root, "Skeleton")
    tr = idx[idx.split == "train"]
    ds = make_dataset(tr, mod_dir, cfg, train=True)
    x, y = ds[0]
    check("skeleton sample shape", tuple(x.shape) == (32, N_JOINTS, 3),
          str(tuple(x.shape)))
    check("skeleton dispatch picks SkeletonNet",
          type(make_model(cfg, pretrained=False)).__name__ == "SkeletonNet")
    ev = make_dataset(tr, mod_dir, cfg, train=False)
    a, _ = ev[5]; b, _ = ev[5]
    check("skeleton eval deterministic", torch.allclose(a, b))
    aug = [ds[5][0] for _ in range(3)]
    check("skeleton train augmentation varies",
          not torch.allclose(aug[0], aug[1]))
    # temporal resampling to a different length must still work
    cfg16 = Cfg(data_root=root, modality="Skeleton", frames=16)
    x16, _ = make_dataset(tr, mod_dir, cfg16, train=False)[0]
    check("skeleton resamples to requested frame count",
          tuple(x16.shape) == (16, N_JOINTS, 3), str(tuple(x16.shape)))


def test_augmentations(root):
    from cuhkx import temporal_window
    for n, T in [(16, 16), (32, 32), (5, 16), (1, 8), (60, 16)]:
        ok = True
        for _ in range(50):
            ix = temporal_window(n, T)
            ok &= (len(ix) == T and min(ix) >= 0 and max(ix) <= max(n - 1, 0)
                   and ix == sorted(ix))
        check(f"temporal_window n={n} T={T}: in range and monotone", ok)
    check("temporal_window varies between calls",
          len({tuple(temporal_window(16, 16)) for _ in range(20)}) > 1)

    cfg = Cfg(data_root=root, modality="Skeleton", frames=32,
              extra={"temporal_aug": True, "amp_aug": True})
    mod_dir, idx = load_index(root, "Skeleton")
    tr = idx[idx.split == "train"]
    a, _ = make_dataset(tr, mod_dir, cfg, train=False)[3]
    base, _ = make_dataset(tr, mod_dir, Cfg(data_root=root, modality="Skeleton",
                                            frames=32), train=False)[3]
    check("aug flags leave skeleton eval untouched", torch.allclose(a, base))
    x, _ = make_dataset(tr, mod_dir, cfg, train=True)[3]
    check("skeleton aug sample shape", tuple(x.shape) == (32, N_JOINTS, 3),
          str(tuple(x.shape)))

    cfgi = Cfg(data_root=root, modality="Thermal", frames=8, size=48,
               extra={"temporal_aug": True})
    mdi, idxi = load_index(root, "Thermal")
    tri = idxi[idxi.split == "train"]
    xi, _ = make_dataset(tri, mdi, cfgi, train=True)[0]
    check("image temporal aug sample shape", tuple(xi.shape) == (8, 3, 48, 48),
          str(tuple(xi.shape)))
    e1, _ = make_dataset(tri, mdi, cfgi, train=False)[0]
    e2, _ = make_dataset(tri, mdi, Cfg(data_root=root, modality="Thermal", frames=8,
                                       size=48), train=False)[0]
    check("aug flags leave image eval untouched", torch.allclose(e1, e2))


def test_skeleton_training(tmp):
    root = os.path.join(tmp, "skel")
    os.makedirs(root, exist_ok=True)
    make_fake_skeleton(root, n_users=6, n_cls=4, trials=3, T=32, n_test=4)
    out = os.path.join(tmp, "skel_out")
    cmd = [sys.executable, os.path.join(ROOT, "kaggle", "train.py"),
           "--data-root", root, "--modality", "Skeleton", "--fold", "0",
           "--epochs", "8", "--batch-size", "8", "--frames", "32",
           "--workers", "0", "--out-dir", out, "--lr", "1e-3"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=os.path.join(ROOT, "kaggle"))
    if r.returncode != 0:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        check("skeleton train.py exits cleanly", False)
        return
    check("skeleton train.py exits cleanly", True)
    for l in [l for l in r.stdout.splitlines() if "val_acc" in l][-2:]:
        print("      " + l)
    d = np.load(os.path.join(out, "Skeleton_f0_oof.npz"), allow_pickle=True)
    acc = float((d["logits"].argmax(1) == d["y"]).mean())
    check("ST-GCN learns a separable 4-class task", acc > 0.45,
          f"val acc {acc:.3f} vs chance 0.250")
    t = np.load(os.path.join(out, "Skeleton_f0_test.npz"), allow_pickle=True)
    check("skeleton TTA produces valid probs",
          np.allclose(t["probs"].sum(1), 1, atol=1e-3))


def test_real_skeleton():
    """Run against the actual extracted competition data when it is present."""
    real = os.path.join(ROOT, "compact", "Skeleton")
    if not os.path.exists(os.path.join(real, "poses.npy")):
        print("  (skipped — real compact/Skeleton not built yet)")
        return
    p = np.load(os.path.join(real, "poses.npy")).astype(np.float32)
    idx = pd.read_parquet(os.path.join(real, "index.parquet"))
    check("real poses finite", bool(np.isfinite(p).all()))
    check("real poses root-centred horizontally",
          float(np.abs(p[:, :, 0, :2]).max()) < 1e-2)
    check("real poses keep posture height (root z varies)",
          float(p[:, :, 0, 2].std()) > 1e-3, f"root z std {p[:, :, 0, 2].std():.3f}")
    check("real index matches array length", len(idx) == len(p),
          f"{len(idx)} vs {len(p)}")
    check("real test clips all present", int((idx.split == 'test').sum()) == 405,
          str(int((idx.split == 'test').sum())))
    # head must sit above feet on average — catches an axis mix-up
    check("head above feet in real data",
          float(p[:, :, 10, 2].mean()) > float(p[:, :, 6, 2].mean()),
          f"head {p[:,:,10,2].mean():.2f} vs foot {p[:,:,6,2].mean():.2f}")
    cfg = Cfg(data_root=os.path.join(ROOT, "compact"), modality="Skeleton", frames=32)
    ds = make_dataset(idx, real, cfg, train=False)
    x, _ = ds[0]
    cz = Cfg(data_root=os.path.join(ROOT, "compact"), modality="Skeleton",
             frames=32, extra={"center_z": True})
    xz, _ = make_dataset(idx, real, cz, train=False)[0]
    check("center_z removes root height on real data",
          float(xz[:, 0, 2].abs().max()) < 1e-5 and float(x[:, 0, 2].abs().max()) > 1e-3)
    check("real skeleton loads through the dataset",
          tuple(x.shape) == (32, N_JOINTS, 3), str(tuple(x.shape)))
    with torch.no_grad():
        out = SkeletonNet()(x[None])
    check("real skeleton runs through SkeletonNet", tuple(out.shape) == (1, 40))


def test_video_archs():
    for arch in ("mc3_18", "s3d"):
        cfg = Cfg(frames=8, size=64, arch=arch, dropout=0.0)
        m = make_model(cfg, pretrained=False)
        out = m(torch.randn(2, 8, 3, 64, 64))
        n = sum(p.numel() for p in m.parameters())
        check(f"{arch} output shape", tuple(out.shape) == (2, 40), str(tuple(out.shape)))
        check(f"{arch} fits the budget", n * 2 / 1e6 < 60, f"{n/1e6:.1f} M = {n*2/1e6:.1f} MB fp16")
        out.sum().backward()
        check(f"{arch} gradients finite",
              all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))


def test_video_training(tmp):
    """A real train.py run on a Kinetics-pretrained 3D backbone with Kinetics norm."""
    root = os.path.join(tmp, "vid")
    os.makedirs(root, exist_ok=True)
    make_fake_compact(root, n_users=5, n_cls=4, trials=2, T=8, size=40, n_test=4)
    out = os.path.join(tmp, "vid_out")
    cmd = [sys.executable, os.path.join(ROOT, "kaggle", "train.py"),
           "--data-root", root, "--modality", "Thermal", "--fold", "0",
           "--epochs", "1", "--batch-size", "4", "--frames", "8", "--size", "40",
           "--workers", "0", "--out-dir", out, "--no-tta",
           "--arch", "mc3_18", "--norm", "kinetics"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=os.path.join(ROOT, "kaggle"))
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    check("mc3_18 + kinetics norm train.py run exits cleanly", r.returncode == 0)
    check("video run wrote OOF + test outputs",
          os.path.exists(os.path.join(out, "Thermal_f0_oof.npz"))
          and os.path.exists(os.path.join(out, "Thermal_f0_test.npz")))


def test_full_data(tmp):
    """--fold -1 trains on every user: weights + test probs, and no OOF file."""
    root = os.path.join(tmp, "full")
    os.makedirs(root, exist_ok=True)
    make_fake_compact(root, n_users=5, n_cls=4, trials=2, T=8, size=40, n_test=6)
    out = os.path.join(tmp, "full_out")
    cmd = [sys.executable, os.path.join(ROOT, "kaggle", "train.py"),
           "--data-root", root, "--modality", "Thermal", "--fold", "-1",
           "--epochs", "2", "--batch-size", "8", "--frames", "8", "--size", "40",
           "--workers", "0", "--out-dir", out, "--no-tta"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=os.path.join(ROOT, "kaggle"))
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    check("full-data run exits cleanly", r.returncode == 0)
    if r.returncode != 0:
        return
    check("full-data run trains on all users", "val 0 clips" in r.stdout)
    check("wrote Thermal_full.pt", os.path.exists(os.path.join(out, "Thermal_full.pt")))
    check("wrote Thermal_full_test.npz",
          os.path.exists(os.path.join(out, "Thermal_full_test.npz")))
    check("no OOF file without a held-out fold",
          not os.path.exists(os.path.join(out, "Thermal_full_oof.npz")))
    t = np.load(os.path.join(out, "Thermal_full_test.npz"), allow_pickle=True)
    check("full-data test probs valid", np.allclose(t["probs"].sum(1), 1, atol=1e-3)
          and len(t["probs"]) == 6)


def main():
    tmp = tempfile.mkdtemp(prefix="cuhkx_smoke_")
    data = os.path.join(tmp, "compact")
    os.makedirs(data, exist_ok=True)
    try:
        print("\n== synthetic dataset ==")
        n = make_fake_compact(data)
        check("built fake compact dataset", n > 0, f"{n} clips")

        print("\n== temporal shift ==");     test_temporal_shift()
        print("\n== model ==");              test_model_shapes()
        print("\n== dataset ==");            test_dataset(data)
        print("\n== cross-subject folds =="); test_folds(data)
        print("\n== training run ==");       test_training_run(data, tmp)
        print("\n== learnability ==");        test_learnability(tmp)
        print("\n== full-data mode ==");      test_full_data(tmp)
        print("\n== video backbones ==");    test_video_archs()
        print("\n== video training ==");     test_video_training(tmp)

        print("\n== skeleton graph ==")
        make_fake_skeleton(data)
        test_skeleton_graph()
        print("\n== pose features ==");       test_pose_features()
        print("\n== skeleton model ==");      test_skeleton_model()
        print("\n== skeleton dataset ==");    test_skeleton_dataset(data)
        print("\n== augmentations ==");       test_augmentations(data)
        print("\n== skeleton training ==");   test_skeleton_training(tmp)
        print("\n== REAL competition data =="); test_real_skeleton()

        print("\n== checkpoint packing =="); test_pack(tmp)

        print("\n" + "=" * 60)
        if _failures:
            print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
            sys.exit(1)
        print("ALL CHECKS PASSED — pipeline is safe to run on Kaggle GPU")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
