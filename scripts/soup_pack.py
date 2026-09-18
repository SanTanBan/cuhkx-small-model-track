"""
Package candidates from the weight-averaged R(2+1)D-34 (kaggle/soup.py outputs).

    python scripts/soup_pack.py [--pull runs/soup] [--pth re144]

For every soup variant the kernel produced (re144, avg144, re128, avg128) this
lays out runs/soup-<variant>/ the way a training run looks: DepthIR_PC_full.pt
(the fp16 average of the two self-trained students, carrying the re-estimated
BatchNorm statistics for re*), the matching student's training log (so the
package records the input size the soup runs at), and DepthIR_PC_full_test.npz.
It then fuses the soup in the #15/#16 slot with #12's weights and one
rebalancing step (scripts/pack_fixed.py), and reports:
  * whether the kernel's view protocol reproduces P's and R's saved test probabilities,
  * each soup's test agreement with P, R and their prediction ensemble,
  * how its submission differs from #15, #16 and the P+R prediction ensemble
    (two big models, so that one is a reference, not a valid package).
--pth <variant> also writes that variant's package.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
from soup import soup  # noqa: E402

P_RUN, R_RUN = "runs/r2p-pl", "runs/r2p-r2"
BASE = ["v2=runs/skel-a/Skeleton_f0,runs/skel-b/Skeleton_f2,runs/skel-c/Skeleton_f4",
        "hkept=runs/skelh-a/Skeleton_f0,runs/skelh-b/Skeleton_f2,runs/skelz-f4-aug/Skeleton_f4",
        "th_s3d=runs/th-s3d/Thermal_full", "imu=runs/imu-a/IMU_full"]
REF = {"#15": "runs/submission_day4f_student_balanced.csv",
       "#16": "runs/submission_r2_pc_b1.csv"}


def probs(path):
    d = np.load(os.path.join(ROOT, path), allow_pickle=True)
    return pd.DataFrame(d["probs"].astype(np.float64), index=d["clip_id"].astype(str))


def pack(member, csv, pth=None):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "pack_fixed.py"), *BASE,
           f"r2p144={member}", "--weights", "runs/fusion_report_day4b.json",
           "--balance", "1", "--out-csv", csv] + (["--out-pth", pth] if pth else [])
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    if out.returncode:
        print(out.stdout[-2000:], out.stderr[-2000:])
        raise SystemExit(f"pack_fixed failed for {member}")
    return [l for l in out.stdout.splitlines() if l.startswith(("wrote", "package"))]


def diff(a, b):
    return int((pd.read_csv(os.path.join(ROOT, a)).prediction
                != pd.read_csv(os.path.join(ROOT, b)).prediction).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", default="runs/soup")
    ap.add_argument("--pth", action="append", default=[])
    a = ap.parse_args()
    pull = os.path.join(ROOT, a.pull)

    for name in ("soupA_results.json", "soupB_results.json"):
        f = os.path.join(pull, name)
        if os.path.exists(f):
            print(f"== {name}\n{pd.DataFrame(json.load(open(f))).T.to_string()}\n")

    Pp = probs(f"{P_RUN}/DepthIR_PC_full_test.npz")
    Rp = probs(f"{R_RUN}/DepthIR_PC_full_test.npz").reindex(Pp.index)
    for name, ref, size in (("R", Rp, 144), ("P", Pp, 128)):
        f = os.path.join(a.pull, f"DepthIR_PC_repro_{name}{size}_test.npz")
        if os.path.exists(os.path.join(ROOT, f)):
            x = probs(f).reindex(Pp.index).values
            print(f"protocol check {name}: top-1 agreement with its saved test probabilities "
                  f"{(x.argmax(1) == ref.values.argmax(1)).mean():.4f}, "
                  f"max |dp| {np.abs(x - ref.values).max():.4f}")

    check = "runs/_check16.csv"
    pack(f"{R_RUN}/DepthIR_PC_full", check)
    d16 = diff(check, REF["#16"])
    os.remove(os.path.join(ROOT, check))
    if d16:
        raise SystemExit(f"rebuilding #16 gives {d16} different predictions: fix before use")
    print("#16 rebuilds exactly (0 differences)")

    os.makedirs(os.path.join(ROOT, "runs", "ens-PR"), exist_ok=True)
    np.savez(os.path.join(ROOT, "runs", "ens-PR", "DepthIR_PC_full_test.npz"),
             probs=((Pp.values + Rp.values) / 2).astype(np.float32),
             clip_id=Pp.index.values.astype(object))
    ens_csv = "runs/reference_ens_PR_b1.csv"
    pack("runs/ens-PR/DepthIR_PC_full", ens_csv)
    ens_top = (Pp.values + Rp.values).argmax(1)
    print(f"P+R prediction ensemble (reference only): {diff(ens_csv, REF['#15'])} predictions "
          f"differ from #15, {diff(ens_csv, REF['#16'])} from #16\n")

    S = soup(torch.load(os.path.join(ROOT, P_RUN, "DepthIR_PC_full.pt"), map_location="cpu"),
             torch.load(os.path.join(ROOT, R_RUN, "DepthIR_PC_full.pt"), map_location="cpu"))
    rows = []
    for v in ("re144", "avg144", "re128", "avg128"):
        src = os.path.join(pull, f"DepthIR_PC_soup_{v}_test.npz")
        if not os.path.exists(src):
            continue
        size = int(v[-3:])
        run = os.path.join(ROOT, "runs", f"soup-{v}")
        os.makedirs(run, exist_ok=True)
        sd = dict(S)
        if v.startswith("re"):
            bn = torch.load(os.path.join(pull, f"soup_re{size}_bn.pt"), map_location="cpu")
            unknown = [k for k in bn if k not in sd]
            if unknown or not bn:
                raise SystemExit(f"BatchNorm file for {v} does not match the model: {unknown[:3]}")
            sd.update(bn)
        torch.save(sd, os.path.join(run, "DepthIR_PC_full.pt"))
        shutil.copy(os.path.join(ROOT, R_RUN if size == 144 else P_RUN, "log_DepthIR_PC_f-1.txt"),
                    os.path.join(run, "log_DepthIR_PC_f-1.txt"))
        shutil.copy(src, os.path.join(run, "DepthIR_PC_full_test.npz"))
        sp = probs(os.path.relpath(src, ROOT)).reindex(Pp.index).values
        csv = f"runs/submission_soup_{v}_b1.csv"
        pth = f"runs/model_soup_{v}_b1.pth" if v in a.pth else None
        for line in pack(f"runs/soup-{v}/DepthIR_PC_full", csv, pth):
            print(f"  {v}: {line}")
        top = sp.argmax(1)
        rows.append(dict(variant=v, agree_P=round((top == Pp.values.argmax(1)).mean(), 3),
                         agree_R=round((top == Rp.values.argmax(1)).mean(), 3),
                         agree_PR_ens=round((top == ens_top).mean(), 3),
                         mean_top_p=round(sp.max(1).mean(), 3),
                         fused_vs_15=diff(csv, REF["#15"]), fused_vs_16=diff(csv, REF["#16"]),
                         fused_vs_ens=diff(csv, ens_csv)))
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print(f"\nstudents alone: P vs R top-1 agreement {(Pp.values.argmax(1) == Rp.values.argmax(1)).mean():.3f}; "
          f"mean top p P {Pp.values.max(1).mean():.3f}, R {Rp.values.max(1).mean():.3f}")


if __name__ == "__main__":
    main()
