"""
Round-2 teacher: the nine-stream #11 fusion stacked with the self-trained student.

    python scripts/teacher2.py --threshold 0.6

Two log-linear weights (nine-stream fusion, student) are fitted on fold 0 —
the only fold where the student has held-out predictions — and applied to the
test clips. Writes kaggle/pl2_test.npz (test clips the stacked teacher labels
at p >= threshold) and kaggle/kd2.npz (distillation targets: nine-stream
out-of-fold probabilities for training clips, stacked-teacher probabilities
for test clips), probabilities stored as uint8 so the kernel payload stays
under Kaggle's script-size limit.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pseudo_labels as PL  # noqa: E402

STREAMS = {"v2": ["runs/skel-a", "runs/skel-b", "runs/skel-c"],
           "hkept": ["runs/skelh-a", "runs/skelh-b", "runs/skelz-f4-aug"],
           "de_s3d": ["runs/de-s3d"], "th_s3d": ["runs/th-s3d"],
           "th_tsm": ["runs/thermal-gpu"], "ir_tsm": ["runs/ir-gpu"],
           "r2p": ["runs/r2p-a"], "r2p144": ["runs/r2p-b"], "imu": ["runs/imu-a"]}


def fit_loglin(L, y, l2=1e-3, steps=3000, lr=0.05):
    S, N, _ = L.shape
    w = np.full(S, 1.0 / S)
    m1 = m2 = np.zeros(S)
    rows = np.arange(N)
    for t in range(1, steps + 1):
        q = PL.softmax(np.tensordot(w, L, axes=1))
        g = (L[:, rows, y] - (q[None] * L).sum(2)).mean(1) - l2 * w
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g * g
        w = np.maximum(0, w + lr * (m1 / (1 - 0.9 ** t)) / (np.sqrt(m2 / (1 - 0.999 ** t)) + 1e-8))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/fusion_report_day4a.json")
    ap.add_argument("--student", default="runs/r2p-pl")
    ap.add_argument("--threshold", type=float, default=0.6)
    a = ap.parse_args()
    import json
    weights = json.load(open(os.path.join(ROOT, a.report)))["weights"]

    clips_t, p9_t, _ = PL.fuse(STREAMS, weights, "_test.npz")
    clips_o, p9_o, meta = PL.fuse(STREAMS, weights, "_oof.npz")
    io = {c: i for i, c in enumerate(clips_o)}
    so = np.load(os.path.join(ROOT, a.student, "DepthIR_PC_f0_oof.npz"), allow_pickle=True)
    st = np.load(os.path.join(ROOT, a.student, "DepthIR_PC_full_test.npz"), allow_pickle=True)

    sid = [c for c in so["clip_id"].astype(str) if c in io]
    sel = [i for i, c in enumerate(so["clip_id"].astype(str)) if c in io]
    ps = PL.softmax(so["logits"].astype(np.float64))[sel]
    p9 = p9_o[[io[c] for c in sid]]
    y = so["y"].astype(int)[sel]
    L = np.stack([np.log(np.clip(p9, 1e-8, 1)), np.log(np.clip(ps, 1e-8, 1))])
    w2 = fit_loglin(L, y)
    fused = np.tensordot(w2, L, axes=1).argmax(1)
    print(f"fold 0 ({len(y)} clips): nine-stream {float((p9.argmax(1) == y).mean()):.4f} | "
          f"student {float((ps.argmax(1) == y).mean()):.4f} | stacked (in-sample) "
          f"{float((fused == y).mean()):.4f} | weights nine-stream {w2[0]:.3f} student {w2[1]:.3f}")

    ps_t = pd.DataFrame(st["probs"], index=st["clip_id"].astype(str)).reindex(clips_t).values
    ok = ~np.isnan(ps_t).any(1)
    zt = w2[0] * np.log(np.clip(p9_t, 1e-8, 1))
    zt[ok] += w2[1] * np.log(np.clip(ps_t[ok], 1e-8, 1))
    t2 = PL.softmax(zt)
    conf, lab = t2.max(1), t2.argmax(1)
    keep = conf >= a.threshold
    np.savez(os.path.join(ROOT, "kaggle", "pl2_test.npz"), clip_id=np.array(clips_t)[keep],
             label=lab[keep], prob=conf[keep])
    agree = float((lab == p9_t.argmax(1)).mean())
    print(f"test: {int(keep.sum())}/{len(clips_t)} clips at p >= {a.threshold} "
          f"({len(set(lab[keep]))} classes); stacked teacher agrees with the nine-stream "
          f"fusion on {agree:.3f} of test clips")

    probs = np.concatenate([t2, p9_o])
    q = np.round(probs * 255).astype(np.uint8)
    empty = q.sum(1) == 0
    q[empty, probs[empty].argmax(1)] = 255
    ids = np.array(list(clips_t) + list(clips_o))
    np.savez_compressed(os.path.join(ROOT, "kaggle", "kd2.npz"), clip_id=ids, probs=q)
    print(f"kd2.npz: {len(ids)} clips, {os.path.getsize(os.path.join(ROOT, 'kaggle', 'kd2.npz')) // 1024} KB; "
          f"pl2_test.npz {os.path.getsize(os.path.join(ROOT, 'kaggle', 'pl2_test.npz')) // 1024} KB")


if __name__ == "__main__":
    main()
