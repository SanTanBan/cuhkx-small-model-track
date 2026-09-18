"""
Geometric fusion with fixed stream weights and explicit test members.

    python scripts/fuse_fixed.py --report runs/fusion_report_day4b.json \
        --out runs/submission_x.csv \
        v2=runs/skel-a/Skeleton_f0_test.npz,runs/skel-b/Skeleton_f2_test.npz \
        r2p144=runs/r2p-pl/DepthIR_PC_full_test.npz ...

Each stream averages its members' test probabilities; streams are fused as
softmax(sum_s w_s log p_s) over the streams present for each clip, with w
taken from a fusion report. Used to swap a better model into a stream's slot
without refitting the weights.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("streams", nargs="+", help="name=test_npz[,test_npz...]")
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", help="overrides, e.g. r2p144=0.5,imu=0.2")
    a = ap.parse_args()
    w = dict(json.load(open(os.path.join(ROOT, a.report)))["weights"])
    if a.weights:
        w.update({k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))})
    tmpl = pd.read_csv(os.path.join(ROOT, "extracted", "test.csv"))
    ids = [p.strip("/").split("/")[-1] for p in tmpl.path]
    z = np.zeros((len(ids), 40))
    den = np.zeros(len(ids))
    for spec in a.streams:
        name, files = spec.split("=", 1)
        files = files.split(",")
        acc = None
        for f in files:
            d = np.load(os.path.join(ROOT, f), allow_pickle=True)
            df = pd.DataFrame(d["probs"], index=d["clip_id"].astype(str)).reindex(ids)
            acc = df if acc is None else acc + df
        p = (acc / len(files)).values
        ok = ~np.isnan(p).any(1)
        wi = float(w[name])
        z[ok] += wi * np.log(np.clip(p[ok], 1e-8, 1))
        den[ok] += wi
        print(f"  {name:<8} weight {wi:.4f}  members {len(files)}  clips {int(ok.sum())}")
    assert (den > 0).all(), "a test clip has no stream"
    z -= z.max(1, keepdims=True)
    pr = np.exp(z)
    pr /= pr.sum(1, keepdims=True)
    sub = pd.DataFrame({"path": tmpl.path, "prediction": pr.argmax(1).astype(int)})
    out = os.path.join(ROOT, a.out)
    sub.to_csv(out, index=False)
    np.save(out.replace(".csv", "_probs.npy"), pr)
    print(f"wrote {a.out}: 405 rows, {sub.prediction.nunique()}/40 classes")


if __name__ == "__main__":
    main()
