"""
Probe the extracted dataset: report the real directory layout, modalities,
users, per-clip frame counts, image resolutions and CSV schemas.

Everything downstream (preprocessing, model input shapes, fusion) depends on
these facts, so run this before writing any training code against assumptions.

Usage:
    python scripts/03_probe.py
    python scripts/03_probe.py --root extracted --deep
"""
import argparse
import collections
import csv
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def shallow_tree(path, max_depth=4, max_entries=8, _depth=0, _prefix=""):
    """Print the top of the directory tree, truncating wide levels."""
    if _depth > max_depth or not os.path.isdir(path):
        return
    try:
        entries = sorted(os.listdir(path))
    except OSError:
        return
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if not os.path.isdir(os.path.join(path, e))]

    for d in dirs[:max_entries]:
        print(f"{_prefix}{d}/")
        shallow_tree(os.path.join(path, d), max_depth, max_entries,
                     _depth + 1, _prefix + "    ")
    if len(dirs) > max_entries:
        print(f"{_prefix}... (+{len(dirs) - max_entries} more dirs)")
    for f in files[:3]:
        print(f"{_prefix}{f}")
    if len(files) > 3:
        print(f"{_prefix}... (+{len(files) - 3} more files)")


def img_size(p):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size, im.mode
    except Exception:
        try:
            import cv2
            a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            return (a.shape[1], a.shape[0]), str(a.dtype) + f"x{a.ndim}"
        except Exception:
            return None, None


def csv_schema(p, n=3):
    try:
        with open(p, "r", errors="replace", newline="") as f:
            rows = [r for _, r in zip(range(n + 1), csv.reader(f))]
        return rows
    except Exception as e:
        return [[f"<unreadable: {e}>"]]


def probe_train(train_root, deep):
    print("=" * 78)
    print(f"TRAIN ROOT: {train_root}")
    print("=" * 78)
    if not os.path.isdir(train_root):
        print("  (not extracted yet)")
        return

    modalities = sorted(
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    )
    print(f"\nmodalities ({len(modalities)}): {modalities}\n")

    for mod in modalities:
        mroot = os.path.join(train_root, mod)
        actions = sorted(
            d for d in os.listdir(mroot) if os.path.isdir(os.path.join(mroot, d))
        )
        users, clips, frame_counts = set(), 0, []
        samples = []
        for a in actions:
            aroot = os.path.join(mroot, a)
            for u in os.listdir(aroot):
                uroot = os.path.join(aroot, u)
                if not os.path.isdir(uroot):
                    continue
                users.add(u)
                for t in os.listdir(uroot):
                    troot = os.path.join(uroot, t)
                    if not os.path.isdir(troot):
                        continue
                    clips += 1
                    if deep or clips <= 400:
                        try:
                            fs = os.listdir(troot)
                        except OSError:
                            continue
                        frame_counts.append(len(fs))
                        if len(samples) < 4 and fs:
                            samples.append(os.path.join(troot, fs[0]))

        print(f"--- {mod}")
        print(f"    actions : {len(actions)}   e.g. {actions[:3]}")
        print(f"    users   : {len(users)}   {sorted(users, key=str)[:24]}")
        print(f"    clips   : {clips}")
        if frame_counts:
            fc = sorted(frame_counts)
            print(f"    frames/clip: min={fc[0]} p50={fc[len(fc)//2]} "
                  f"max={fc[-1]} mean={sum(fc)/len(fc):.1f}  (n={len(fc)})")
        for s in samples[:2]:
            ext = os.path.splitext(s)[1].lower()
            if ext in IMG_EXT:
                size, mode = img_size(s)
                print(f"    sample  : {os.path.basename(s)}  size={size} mode={mode}")
            else:
                print(f"    sample  : {os.path.basename(s)}")
                for row in csv_schema(s):
                    print(f"        {row[:12]}")
        print()


def probe_test(test_root):
    print("=" * 78)
    print(f"TEST ROOT: {test_root}")
    print("=" * 78)
    if not os.path.isdir(test_root):
        print("  (not extracted yet)")
        return
    clips = sorted(
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    )
    print(f"clips: {len(clips)}   e.g. {clips[:3]}")

    mod_presence = collections.Counter()
    frames_by_mod = collections.defaultdict(list)
    for c in clips:
        croot = os.path.join(test_root, c)
        for m in os.listdir(croot):
            mroot = os.path.join(croot, m)
            if os.path.isdir(mroot):
                mod_presence[m] += 1
                try:
                    frames_by_mod[m].append(len(os.listdir(mroot)))
                except OSError:
                    pass

    print(f"\nmodality coverage across {len(clips)} test clips:")
    for m, n in mod_presence.most_common():
        fc = sorted(frames_by_mod[m]) or [0]
        print(f"    {m:<14} present in {n:>4}/{len(clips)} "
              f"({100*n/len(clips):5.1f}%)   frames: min={fc[0]} "
              f"p50={fc[len(fc)//2]} max={fc[-1]}")

    # Show one full clip's layout.
    if clips:
        print(f"\nlayout of {clips[0]}:")
        shallow_tree(os.path.join(test_root, clips[0]), max_depth=2, max_entries=10,
                     _prefix="    ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "extracted"))
    ap.add_argument("--deep", action="store_true",
                    help="count frames for every clip (slow, accurate)")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"not found: {args.root}\nrun scripts/02_extract.py first")

    print("TOP-LEVEL LAYOUT")
    print("-" * 78)
    shallow_tree(args.root, max_depth=3, max_entries=8, _prefix="  ")
    print()

    # Locate the HAR/<modality> root and the test clip root wherever they landed.
    train_root = None
    test_root = None
    for dirpath, dirnames, _ in os.walk(args.root):
        base = os.path.basename(dirpath)
        if base == "data" and "HAR" in dirpath and train_root is None:
            train_root = dirpath
        if base == "small_model_track_test" and test_root is None:
            test_root = dirpath
        if dirpath.count(os.sep) - args.root.count(os.sep) > 4:
            dirnames[:] = []

    probe_test(test_root or os.path.join(args.root, "small_model_track_test"))
    probe_train(train_root or os.path.join(args.root, "HAR", "data"))

    for name in ("class_mapping.csv", "test.csv", "sample_submission.csv"):
        p = os.path.join(args.root, name)
        if os.path.exists(p):
            print(f"\n--- {name}")
            for row in csv_schema(p, 4):
                print(f"    {row}")


if __name__ == "__main__":
    main()
