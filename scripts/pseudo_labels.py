"""
Pseudo-labels for self-training, which the organisers allow (Kaggle topic 729989).

    python scripts/pseudo_labels.py --report runs/fusion_report_day3d.json \
        --threshold 0.6 --fold 0 --test-out kaggle/pseudo_test.npz \
        --fold-out kaggle/pseudo_f0.npz \
        v2=runs/skel-a,runs/skel-b,runs/skel-c hkept=runs/skelh-a,runs/skelh-b,runs/skelz-f4-aug \
        th_s3d=runs/th-s3d r2p=runs/r2p-a imu=runs/imu-a

The teacher is the geometric fusion of the given streams with the report's
weights. Test clips use the fused test predictions (a stream's members
averaged); the validation fold uses the fused out-of-fold predictions for its
held-out users, so a student trained with them can still be scored on true
labels it never saw. Only clips whose fused top-1 probability reaches the
threshold are kept; a stream missing for a clip is left out of its fusion.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
from cuhkx import build_folds  # noqa: E402

C = 40


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def stream_probs(dirs, suffix):
    """{clip_id: probs}: test members averaged; out-of-fold rows (one per clip)."""
    acc, n, meta = {}, {}, {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*" + suffix))):
            z = np.load(f, allow_pickle=True)
            p = z["probs"] if "probs" in z else softmax(z["logits"].astype(np.float64))
            for i, c in enumerate(z["clip_id"].astype(str)):
                if suffix == "_oof.npz":
                    acc[c], n[c] = p[i], 1
                    meta[c] = (int(z["y"][i]), str(z["user"][i]))
                else:
                    acc[c] = acc.get(c, 0) + p[i]
                    n[c] = n.get(c, 0) + 1
    return {c: acc[c] / n[c] for c in acc}, meta


def fuse(streams, weights, suffix):
    per, meta = {}, {}
    for name, dirs in streams.items():
        per[name], m = stream_probs(dirs, suffix)
        meta.update(m)
    clips = sorted(set().union(*[set(v) for v in per.values()]))
    z = np.zeros((len(clips), C))
    for name, probs in per.items():
        w = float(weights.get(name, 0.0))
        for i, c in enumerate(clips):
            if w > 0 and c in probs:
                z[i] += w * np.log(np.clip(probs[c], 1e-8, 1))
    return clips, softmax(z), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("streams", nargs="+", help="name=run_dir[,run_dir...]")
    ap.add_argument("--report", required=True, help="fusion_report.json with loglin weights")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--fold-out", required=True)
    ap.add_argument("--soft-out", help="also write distillation targets (all clips, no threshold)")
    a = ap.parse_args()
    rep = json.load(open(os.path.join(ROOT, a.report) if not os.path.isabs(a.report) else a.report))
    assert rep.get("fusion") == "loglin", "teacher weights must come from a geometric fusion"
    streams = {s.split("=", 1)[0]: s.split("=", 1)[1].split(",") for s in a.streams}
    weights = rep["weights"]
    print("teacher weights:", {k: weights.get(k) for k in streams})

    clips, p, _ = fuse(streams, weights, "_test.npz")
    conf, lab = p.max(1), p.argmax(1)
    keep = conf >= a.threshold
    np.savez(a.test_out, clip_id=np.array(clips)[keep], label=lab[keep], prob=conf[keep])
    print(f"test: {int(keep.sum())}/{len(clips)} clips at p >= {a.threshold} "
          f"({len(set(lab[keep]))} classes) -> {a.test_out}")

    clips, p, meta = fuse(streams, weights, "_oof.npz")
    users = pd.DataFrame({"user": [meta[c][1] for c in clips]})
    fold = build_folds(users, 5, 42)
    y = np.array([meta[c][0] for c in clips])
    sel = fold == a.fold
    conf, lab = p.max(1), p.argmax(1)
    keep = sel & (conf >= a.threshold)
    np.savez(a.fold_out, clip_id=np.array(clips)[keep], label=lab[keep], prob=conf[keep])
    if a.soft_out:
        # Distillation targets: the fused out-of-fold probabilities of every training
        # clip (from models that never saw its subject) and the fused test probabilities.
        tc, tp, _ = fuse(streams, weights, "_test.npz")
        np.savez(a.soft_out, clip_id=np.array(list(tc) + list(clips)),
                 probs=np.concatenate([tp, p]).astype(np.float32))
        print(f"soft targets: {len(tc)} test + {len(clips)} training clips -> {a.soft_out}")
    print(f"fold {a.fold}: {int(keep.sum())}/{int(sel.sum())} held-out clips at p >= {a.threshold}; "
          f"pseudo-label accuracy {float((lab[keep] == y[keep]).mean()):.3f} "
          f"(teacher accuracy on the whole fold {float((lab[sel] == y[sel]).mean()):.3f}) -> {a.fold_out}")


if __name__ == "__main__":
    main()
