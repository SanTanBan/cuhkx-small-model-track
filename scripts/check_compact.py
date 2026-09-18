"""
Gate before upload: refuse to ship a compact modality that is incomplete or
leaks test subjects into training.

    python scripts/check_compact.py Skeleton Thermal      # exit 1 on any failure
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_USERS = {"user10", "user11", "user25", "user26"}
# Test clips measured on the real test set (Thermal is missing on 10 clips and
# 4 IR clips are zero-filled files).
EXPECT_TEST = {"Skeleton": 405, "Depth_Color": 405, "IR": 401, "Thermal": 395,
               "Depth_Color_PC": 405, "IR_PC": 401, "IRdet": 401}


def main():
    bad = []
    for m in sys.argv[1:]:
        d = os.path.join(ROOT, "compact", m)
        idx = pd.read_parquet(os.path.join(d, "index.parquet"))
        trn = idx[idx.split == "train"]
        tst = idx[idx.split == "test"]
        users = set(trn.user.astype(str))
        leak = sorted(TEST_USERS & users)
        dup = int(idx.clip_id.duplicated().sum())
        ok = (len(trn) >= 2800 and len(users) == 18 and trn.action_id.nunique() == 40
              and len(tst) >= EXPECT_TEST.get(m, 0) and not leak and dup == 0
              and trn.action_id.between(0, 39).all())
        print(f"{m:<12} train={len(trn)} test={len(tst)} users={len(users)} "
              f"classes={trn.action_id.nunique()} dup={dup} leak={leak} "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            bad.append(m)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
