"""
Download the CUHK-X Small Model Track dataset from HuggingFace.

Auth: run `hf auth login` first (token is read from the local HF credential store).

Usage:
    python scripts/01_download.py --part test      # 2.6 GB  test zip + csvs   (do this first)
    python scripts/01_download.py --part train     # 42.5 GB training zips
    python scripts/01_download.py --part all

Resumable: re-running skips files already fully downloaded.
"""
import argparse
import os
import sys
import time

# hf_transfer is faster on fat pipes but CANNOT resume a partial file. On a slow
# link a dropped 5 GB transfer would restart from zero, so it is opt-in only.
if "--fast" in sys.argv:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
else:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError  # noqa: E402

REPO = "Kevin-Pal/CUHK-X_Small_Model_Track"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data")

PATTERNS = {
    # Small files first: test clips + the csvs we need to build a submission.
    "test": [
        "Small-Model-Track/Testing/**",
        "Small-Model-Track/class_mapping.csv",
        "README.md",
        "LICENSE",
    ],
    # The 9-part split zip: HAR.z01 .. HAR.z08 + HAR.zip (central directory).
    "train": ["Small-Model-Track/Training/**"],
}
PATTERNS["all"] = PATTERNS["test"] + PATTERNS["train"]


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def tree_size(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["test", "train", "all"], default="test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fast", action="store_true",
                    help="use hf_transfer (faster, but no resume on failure)")
    args = ap.parse_args()

    os.makedirs(DEST, exist_ok=True)
    before = tree_size(DEST)
    t0 = time.time()

    print(f"repo    : {REPO}")
    print(f"dest    : {DEST}")
    print(f"part    : {args.part}")
    print(f"already : {human(before)} on disk")
    print("-" * 60, flush=True)

    try:
        snapshot_download(
            repo_id=REPO,
            repo_type="dataset",
            local_dir=DEST,
            allow_patterns=PATTERNS[args.part],
            max_workers=args.workers,
            resume_download=True,
        )
    except GatedRepoError:
        sys.exit(
            "\nGATED: accept the dataset terms at\n"
            f"  https://huggingface.co/datasets/{REPO}\n"
            "then run `hf auth login` and retry."
        )
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            sys.exit("\nAUTH: run `hf auth login` with a read token, then retry.")
        raise

    after = tree_size(DEST)
    dt = time.time() - t0
    print("-" * 60)
    print(f"downloaded {human(after - before)} in {dt/60:.1f} min "
          f"({human((after - before) / max(dt, 1))}/s)")
    print(f"total on disk: {human(after)}")


if __name__ == "__main__":
    main()
