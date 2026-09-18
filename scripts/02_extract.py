"""
Extract the downloaded archives.

Training data is a 9-part split zip (HAR.z01..HAR.z08 + HAR.zip). Python's
zipfile cannot read split archives, so we shell out to 7-Zip, which resolves
the sibling volumes automatically when pointed at the LAST volume (HAR.zip).

Usage:
    python scripts/02_extract.py --part test
    python scripts/02_extract.py --part train
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Small-Model-Track")
OUT = os.path.join(ROOT, "extracted")

SEVENZIP_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    shutil.which("7z") or "",
    shutil.which("7za") or "",
]


def find_7z():
    for c in SEVENZIP_CANDIDATES:
        if c and os.path.exists(c):
            return c
    sys.exit("7-Zip not found. Install with:  winget install --id 7zip.7zip -e")


def run_7z(archive, dest):
    sevenzip = find_7z()
    os.makedirs(dest, exist_ok=True)
    print(f"extract {archive}\n     -> {dest}", flush=True)
    # -bsp1 streams progress to stdout so long extractions are observable.
    proc = subprocess.run(
        [sevenzip, "x", archive, f"-o{dest}", "-y", "-bsp1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace",
    )
    tail = [l for l in proc.stdout.splitlines() if l.strip()][-12:]
    print("\n".join(tail))
    if proc.returncode != 0:
        sys.exit(f"7-Zip failed (exit {proc.returncode})")
    print("OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["test", "train", "all"], default="test")
    args = ap.parse_args()

    jobs = []
    if args.part in ("test", "all"):
        jobs.append(
            (os.path.join(DATA, "Testing", "data", "small_model_track_test.zip"), OUT)
        )
    if args.part in ("train", "all"):
        # Point 7-Zip at the final volume; it pulls in .z01...z08 itself.
        jobs.append((os.path.join(DATA, "Training", "data", "HAR.zip"), OUT))

    for archive, dest in jobs:
        if not os.path.exists(archive):
            sys.exit(f"missing archive: {archive}\nrun scripts/01_download.py first")
        run_7z(archive, dest)

    # Copy the small csvs next to the extracted trees for convenience.
    for src in [
        os.path.join(DATA, "class_mapping.csv"),
        os.path.join(DATA, "Testing", "test_file", "test.csv"),
        os.path.join(DATA, "Testing", "test_file", "sample_submission.csv"),
    ]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, os.path.basename(src)))
            print(f"copied {os.path.basename(src)}")


if __name__ == "__main__":
    main()
