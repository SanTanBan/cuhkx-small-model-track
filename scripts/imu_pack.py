"""
Pack the body-worn IMU recordings into compact/IMU: training clips straight out
of the split archive, test clips from the extracted folders.

    python scripts/imu_pack.py [--T 40]

Output: compact/IMU/imu.npy [N, 7*D, T] float32 (prep.encode_imu) and
index.parquet (clip_id, split, user, trial, action_id, row), with clip ids and
user names matching every other modality, so folds and fusion line up.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
import prep  # noqa: E402

VOL_DIR = os.path.join(ROOT, "data", "Small-Model-Track", "Training", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=prep.IMU_T)
    ap.add_argument("--out", default=os.path.join(ROOT, "compact", "IMU"))
    a = ap.parse_args()

    e = pd.read_parquet(os.path.join(ROOT, "logs", "har_entries.parquet"))
    g = e[(e["mod"] == "IMU") & e.name.str.lower().str.endswith(".csv")]
    vz = prep.ZipVolumes(VOL_DIR)
    rows, arrs, empty, t0 = [], [], [], time.time()
    for (action, user, trial), grp in g.groupby(["action", "user", "trial"], sort=True):
        head = str(action).split("_")[0]
        if not head.isdigit():
            continue
        cid = f"{action}/{user}/{trial}"
        x = prep.encode_imu([vz.entry(int(r.disk), int(r.loff), int(r.csize))
                             for r in grp.itertuples()], a.T)
        if x is None:
            empty.append(cid)
            continue
        rows.append(dict(clip_id=cid, split="train", user=str(user), trial=str(trial),
                         action_id=int(head), row=len(arrs)))
        arrs.append(x)
    n_train = len(rows)

    _, test_root = prep.find_roots(os.path.join(ROOT, "extracted"))
    for c in sorted(os.listdir(test_root)):
        d = os.path.join(test_root, c, "IMU")
        if not c.startswith("SM_test") or not os.path.isdir(d):
            continue
        bufs = [open(os.path.join(d, f), "rb").read() for f in sorted(os.listdir(d))
                if f.lower().endswith(".csv")]
        x = prep.encode_imu(bufs, a.T)
        if x is None:
            empty.append(c)
            continue
        rows.append(dict(clip_id=c, split="test", user="test", trial="", action_id=-1,
                         row=len(arrs)))
        arrs.append(x)

    os.makedirs(a.out, exist_ok=True)
    X = np.stack(arrs).astype(np.float32)
    np.save(os.path.join(a.out, "imu.npy"), X)
    idx = pd.DataFrame(rows)
    idx.to_parquet(os.path.join(a.out, "index.parquet"), index=False)
    D = len(prep.IMU_DEVICES)
    pres = X[:, 6 * D:, 0]
    summary = dict(modality="IMU", T=a.T, channels=int(X.shape[1]), train=n_train,
                   test=len(rows) - n_train, empty_clips=len(empty),
                   users=int(idx[idx.split == "train"].user.nunique()),
                   classes=int(idx[idx.split == "train"].action_id.nunique()),
                   device_presence={dv: round(float(pres[:, i].mean()), 4)
                                    for i, dv in enumerate(prep.IMU_DEVICES)},
                   minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary))
    print("empty clips:", empty[:8], "..." if len(empty) > 8 else "")


if __name__ == "__main__":
    main()
