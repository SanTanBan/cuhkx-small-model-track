"""
Compact the extracted dataset into Kaggle-uploadable shards (local runner).

All logic lives in kaggle/prep.py so the local run and the Kaggle notebook
produce byte-identical output.

    python scripts/04_preprocess.py --modalities Skeleton
    python scripts/04_preprocess.py --modalities Depth_Color IR Thermal Skeleton
    python scripts/04_preprocess.py --modalities Skeleton --limit 200   # smoke
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))

from prep import (SKELETON, discover_test, discover_train, find_roots,  # noqa: E402
                  pack_images, pack_skeleton)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", nargs="+",
                    default=["Skeleton", "Depth_Color", "IR", "Thermal"])
    ap.add_argument("--frames", type=int, default=16, help="image frames per clip")
    ap.add_argument("--skel-frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extracted", default=os.path.join(ROOT, "extracted"))
    ap.add_argument("--out", default=os.path.join(ROOT, "compact"))
    args = ap.parse_args()

    train_root, test_root = find_roots(args.extracted)
    print(f"train_root: {train_root}\ntest_root : {test_root}")
    print(f"frames={args.frames} skel_frames={args.skel_frames} size={args.size} "
          f"crop={not args.no_crop} workers={args.workers}\n")
    if not train_root and not test_root:
        sys.exit("nothing extracted — run scripts/02_extract.py first")

    os.makedirs(args.out, exist_ok=True)
    summary = {}

    for mod in args.modalities:
        recs = (discover_train(train_root, mod) if train_root else []) + \
               (discover_test(test_root, mod) if test_root else [])
        if args.limit:
            recs = recs[:args.limit]
        ntr = sum(r["split"] == "train" for r in recs)
        print(f"=== {mod}: {len(recs)} clips ({ntr} train / {len(recs)-ntr} test)")
        if not recs:
            print("  no clips — skipping\n")
            continue

        if mod == SKELETON:
            df = pack_skeleton(recs, args.out, args.skel_frames, args.workers)
        else:
            df = pack_images(mod, recs, args.out, args.frames, args.size,
                             not args.no_crop, args.quality, args.workers)
        if len(df):
            summary[mod] = dict(
                clips=int(len(df)),
                train=int((df.split == "train").sum()),
                test=int((df.split == "test").sum()),
                users=sorted(df[df.split == "train"].user.astype(str).unique()),
            )
        print()

    with open(os.path.join(args.out, "prep_config.json"), "w") as f:
        json.dump(dict(vars(args), summary=summary), f, indent=2, default=str)
    print(json.dumps(summary, indent=2)[:1500])


if __name__ == "__main__":
    main()
