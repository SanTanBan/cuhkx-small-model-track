"""
Package + submission with fixed fusion weights, for members that may have no
out-of-fold predictions (full-data students).

    python scripts/pack_fixed.py --weights runs/fusion_report_day4b.json --balance 1 \
        --out-csv runs/x.csv --out-pth runs/x.pth \
        v2=runs/skel-a/Skeleton_f0,runs/skel-b/Skeleton_f2,runs/skel-c/Skeleton_f4 \
        hkept=runs/skelh-a/Skeleton_f0,runs/skelh-b/Skeleton_f2,runs/skelz-f4-aug/Skeleton_f4 \
        th_s3d=runs/th-s3d/Thermal_full imu=runs/imu-a/IMU_full \
        r2p144=runs/r2p-r2/DepthIR_PC_full [--weight r2p144=0.7767]

Each member is a path prefix: <prefix>.pt (weights) and <prefix>_test.npz (its
test probabilities). `kaggle_ops.py fuse` builds the fusion from out-of-fold
predictions and silently drops a stream that has none; this script takes the
weights as given, so a full-data student cannot fall out of its slot. Members
are stored int8, the person detector fp16. The metadata carries the weights, each
member's training config, the geometric fusion rule, and the class prior plus
rebalancing steps that infer.py applies. The CSV is the package's own fusion of
its members' test predictions.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from cuhkx import COMPOSITES, quantize_int8  # noqa: E402
from kaggle_ops import member_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("streams", nargs="+", help="name=prefix[,prefix...]")
    ap.add_argument("--weights", required=True, help="fusion report whose weights to use")
    ap.add_argument("--weight", action="append", default=[], help="override, e.g. r2p144=0.78")
    ap.add_argument("--balance", type=int, default=1)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-pth")
    ap.add_argument("--out-probs", help="npz of the fused, rebalanced test probabilities")
    a = ap.parse_args()

    w = dict(json.load(open(os.path.join(ROOT, a.weights)))["weights"])
    for o in a.weight:
        k, v = o.split("=")
        w[k] = float(v)
    tmpl = pd.read_csv(os.path.join(ROOT, "extracted", "test.csv"))
    ids = [p.strip("/").split("/")[-1] for p in tmpl.path]
    z, den = np.zeros((len(ids), 40)), np.zeros(len(ids))
    sds, members, weights = {}, {}, {}
    for spec in a.streams:
        name, prefixes = spec.split("=", 1)
        prefixes = prefixes.split(",")
        if name not in w:
            raise SystemExit(f"no weight for stream {name}")
        weights[name] = float(w[name])
        acc = None
        for pre in prefixes:
            d = np.load(os.path.join(ROOT, pre + "_test.npz"), allow_pickle=True)
            df = pd.DataFrame(d["probs"], index=d["clip_id"].astype(str)).reindex(ids)
            acc = df if acc is None else acc + df
            if a.out_pth:
                tag = os.path.basename(pre)
                key = f"{name}:{tag}"
                pt = os.path.join(ROOT, pre + ".pt")
                sds[key] = quantize_int8(torch.load(pt, map_location="cpu"), bits=8)
                members[key] = dict(member_cfg(pt, tag), stream=name, int8=True, bits=8)
        p = (acc / len(prefixes)).values
        ok = ~np.isnan(p).any(1)
        z[ok] += weights[name] * np.log(np.clip(p[ok], 1e-8, 1))
        den[ok] += weights[name]
        print(f"  {name:<8} weight {weights[name]:.4f}  members {len(prefixes)}  test clips {int(ok.sum())}")
    if (den == 0).any():
        raise SystemExit(f"{int((den == 0).sum())} test clips have no stream")
    z -= z.max(1, keepdims=True)
    final = np.exp(z)
    final /= final.sum(1, keepdims=True)

    idx = pd.read_parquet(os.path.join(ROOT, "compact", "Depth_Color_PC", "index.parquet"))
    prior = np.bincount(idx[idx.split == "train"].action_id.astype(int), minlength=40) + 1.0
    prior /= prior.sum()
    before = final.argmax(1)
    for _ in range(a.balance):
        final = final / (final.sum(0) / (len(final) * prior))[None]
        final /= final.sum(1, keepdims=True)
    pred = final.argmax(1)
    pd.DataFrame({"path": tmpl.path, "prediction": pred.astype(int)}).to_csv(
        os.path.join(ROOT, a.out_csv), index=False)
    if a.out_probs:
        np.savez(os.path.join(ROOT, a.out_probs), probs=final.astype(np.float32),
                 clip_id=np.asarray(ids, dtype=object))
    print(f"wrote {a.out_csv}: {len(set(pred))}/40 classes; rebalancing changed "
          f"{int((pred != before).sum())}")

    if a.out_pth:
        meta = dict(weights=weights, members=members, fusion="loglin",
                    class_prior=prior.tolist(), balance_steps=a.balance)
        ckpt = {"models": sds, "meta": meta}
        if any(p.endswith("_PC") for m in members.values()
               for p, _ in COMPOSITES.get(m["modality"], ((m["modality"], 3),))):
            hub = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
            det = torch.load(glob.glob(os.path.join(hub, "ssdlite320*.pth"))[0], map_location="cpu")
            ckpt["detector"] = {k: v.half() if v.is_floating_point() else v for k, v in det.items()}
        out = os.path.join(ROOT, a.out_pth)
        torch.save(ckpt, out)
        mb = os.path.getsize(out) / 1e6
        print(f"package {a.out_pth}: {mb:.1f} MB, {len(sds)} members"
              f"{', + fp16 detector' if 'detector' in ckpt else ''}")
        if mb >= 100:
            raise SystemExit("package is over 100 MB")


if __name__ == "__main__":
    main()
