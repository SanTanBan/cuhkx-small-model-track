"""
Build valid package candidates from the round-2 students (no submission).

    python scripts/round2_packages.py [--run runs/r2p-r2] [--balance 1]

For each student found in the run (DepthIR_PC_full = self-trained 144 px,
DepthIR_KDT_full = distilled 144 px) this runs `kaggle_ops.py fuse` with #12's
weights, the student in the R(2+1)D-34 slot, the package members of #15
(skeleton folds 0/2/4 of both streams, Thermal S3D, IMU, fp16 detector) and the
class rebalancing step, then saves the submission, report and checkpoint under
runs/ with a suffix and prints how the predictions differ from #15.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = ["v2=runs/skel-a,runs/skel-b,runs/skel-c",
        "hkept=runs/skelh-a,runs/skelh-b,runs/skelz-f4-aug",
        "th_s3d=runs/th-s3d", "imu=runs/imu-a"]
PACK = ["v2:Skeleton_f[024]", "hkept:Skeleton_f[024]", "th_s3d:*_full", "imu:*_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/r2p-r2")
    ap.add_argument("--balance", type=int, default=1)
    ap.add_argument("--weights", default="runs/fusion_report_day4b.json")
    a = ap.parse_args()
    ref = pd.read_csv(os.path.join(ROOT, "runs", "submission_day4f_student_balanced.csv"))
    tags = sorted(os.path.basename(p)[:-len("_test.npz")]
                  for p in glob.glob(os.path.join(ROOT, a.run, "*_full_test.npz")))
    if not tags:
        raise SystemExit(f"no *_full_test.npz in {a.run}")
    for tag in tags:
        has_pt = os.path.exists(os.path.join(ROOT, a.run, f"{tag}.pt"))
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "kaggle_ops.py"), "fuse", *BASE,
               f"r2p144={a.run}", "--weights-json", a.weights, "--balance-steps", str(a.balance),
               "--pack-only", *PACK, f"r2p144:{tag}",
               "--int8", "v2:*", "hkept:*", "th_s3d:*", "imu:*", "r2p144:*",
               "--detector-fp16", "--test-as-packed"] + ([] if has_pt else ["--no-pack"])
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
        lines = [l for l in out.stdout.splitlines()
                 if any(k in l for k in ("weights", "rebalancing", "checkpoint", "wrote", "OVER"))]
        if out.returncode:
            print(out.stdout[-1500:], out.stderr[-1500:])
            raise SystemExit(f"fuse failed for {tag}")
        suffix = tag.replace("DepthIR_", "").replace("_full", "").lower()
        sub = os.path.join(ROOT, "runs", f"submission_r2_{suffix}_b{a.balance}.csv")
        shutil.copy(os.path.join(ROOT, "runs", "submission.csv"), sub)
        shutil.copy(os.path.join(ROOT, "runs", "fusion_report.json"),
                    os.path.join(ROOT, "runs", f"fusion_report_r2_{suffix}_b{a.balance}.json"))
        if has_pt:
            shutil.copy(os.path.join(ROOT, "runs", "model.pth"),
                        os.path.join(ROOT, "runs", f"model_r2_{suffix}_b{a.balance}.pth"))
        d = int((pd.read_csv(sub).prediction != ref.prediction).sum())
        print(f"== {tag}: " + " | ".join(l.strip() for l in lines))
        print(f"   -> {os.path.relpath(sub, ROOT)}  ({d}/405 predictions differ from #15; "
              f"{'package saved' if has_pt else 'no .pt pulled yet: submission only'})")


if __name__ == "__main__":
    main()
