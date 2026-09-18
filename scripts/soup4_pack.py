"""
Package candidates from the round-3 weight averages (kaggle/soup.py phase D).

    python scripts/soup4_pack.py [--pull runs/soup2] [--pth S4]

Members: P = r2p-pl (128 px), R = r2p-r2 (144 px), R3A / R3B = r2p-r3 (144 /
128 px), and for S6 the teacher-era A = r2p-a (128 px) and B = r2p-b (144 px) at
half weight. For each average the kernel scored, this lays out runs/soup-<name>/
like a training run (the fp16 weighted average, R's training log since the
average runs at 144 px, and its test probabilities), fuses it in the
#15/#16/#20 slot with #12's weights and one rebalancing step
(scripts/pack_fixed.py), and reports:
  * the round-3 students on their own: agreement with P and R, and their fused
    submissions against #15/#16 (are they as strong and as different?),
  * each average's test agreement with its members and with the four-student
    prediction ensemble (four big models, so a reference, not a valid package),
  * how its submission differs from #15, #16, #20 and that ensemble's.
--pth <name> also writes that average's package.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from soup import soup  # noqa: E402
from soup_pack import REF, diff, pack, probs  # noqa: E402

REFS = dict(REF, **{"#20": "runs/submission_soup_avg144_b1.csv"})
MEMBERS = {"P": "runs/r2p-pl/DepthIR_PC_full", "R": "runs/r2p-r2/DepthIR_PC_full",
           "R3A": "runs/r2p-r3/DepthIR_R3A_full", "R3B": "runs/r2p-r3/DepthIR_R3B_full",
           "A": "runs/r2p-a/DepthIR_PC_full", "B": "runs/r2p-b/DepthIR_PC_full"}
MIXES = {"S4": {"P": 1, "R": 1, "R3A": 1, "R3B": 1},
         "S6": {"P": 2, "R": 2, "R3A": 2, "R3B": 2, "A": 1, "B": 1}}
STUDENTS = ["P", "R", "R3A", "R3B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", default="runs/soup2")
    ap.add_argument("--pth", action="append", default=[])
    a = ap.parse_args()
    pull = os.path.join(ROOT, a.pull)

    for name in ("soupC_results.json", "soupD_results.json"):
        f = os.path.join(pull, name)
        if os.path.exists(f):
            print(f"== {name}\n{pd.DataFrame(json.load(open(f))).T.to_string()}\n")

    ids = probs(MEMBERS["P"] + "_test.npz").index
    pr = {k: probs(v + "_test.npz").reindex(ids).values for k, v in MEMBERS.items()}
    top = {k: v.argmax(1) for k, v in pr.items()}
    print("students' test top-1 agreement:")
    print(pd.DataFrame([[round((top[x] == top[z]).mean(), 3) for z in STUDENTS] for x in STUDENTS],
                       index=STUDENTS, columns=STUDENTS).to_string())
    for k in ("R3A", "R3B"):
        csv = f"runs/submission_student_{k.lower()}_b1.csv"
        pack(MEMBERS[k], csv)
        print(f"{k} alone in the slot: {diff(csv, REFS['#15'])} predictions differ from #15, "
              f"{diff(csv, REFS['#16'])} from #16; mean top p {pr[k].max(1).mean():.3f}")

    ens = sum(pr[k] for k in STUDENTS) / len(STUDENTS)
    os.makedirs(os.path.join(ROOT, "runs", "ens-4"), exist_ok=True)
    np.savez(os.path.join(ROOT, "runs", "ens-4", "DepthIR_PC_full_test.npz"),
             probs=ens.astype(np.float32), clip_id=np.asarray(ids, dtype=object))
    ens_csv = "runs/reference_ens4_b1.csv"
    pack("runs/ens-4/DepthIR_PC_full", ens_csv)
    print(f"four-student prediction ensemble (reference only): differs from "
          + ", ".join(f"{r} on {diff(ens_csv, p)}" for r, p in REFS.items()) + "\n")

    rows = []
    for name, w in MIXES.items():
        src = os.path.join(pull, f"DepthIR_PC_soup_{name}_avg144_test.npz")
        if not os.path.exists(src):
            print(f"{name}: no test probabilities in {a.pull}")
            continue
        run = os.path.join(ROOT, "runs", f"soup-{name}")
        os.makedirs(run, exist_ok=True)
        sd = soup(*[torch.load(os.path.join(ROOT, MEMBERS[m] + ".pt"), map_location="cpu") for m in w],
                  weights=list(w.values()))
        torch.save(sd, os.path.join(run, "DepthIR_PC_full.pt"))
        del sd
        shutil.copy(os.path.join(ROOT, "runs", "r2p-r2", "log_DepthIR_PC_f-1.txt"),
                    os.path.join(run, "log_DepthIR_PC_f-1.txt"))
        shutil.copy(src, os.path.join(run, "DepthIR_PC_full_test.npz"))
        sp = probs(os.path.relpath(src, ROOT)).reindex(ids).values
        csv = f"runs/submission_soup_{name}_b1.csv"
        pth = f"runs/model_soup_{name}_b1.pth" if name in a.pth else None
        for line in pack(f"runs/soup-{name}/DepthIR_PC_full", csv, pth):
            print(f"  {name}: {line}")
        t = sp.argmax(1)
        row = dict(average=name, **{f"agree_{m}": round((t == top[m]).mean(), 3) for m in STUDENTS})
        row.update(agree_ens4=round((t == ens.argmax(1)).mean(), 3),
                   mean_top_p=round(sp.max(1).mean(), 3),
                   **{f"fused_vs_{r[1:]}": diff(csv, p) for r, p in REFS.items()},
                   fused_vs_ens4=diff(csv, ens_csv))
        rows.append(row)
    if rows:
        print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
