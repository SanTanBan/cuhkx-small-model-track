"""
Build compact/<modality>/ for training + test, reading training frames straight
out of the split archive (no extraction).

Training clips come from logs/har_entries.parquet (written by zip_listing.py);
test clips come from the already-extracted test folder. Both land in the same
blob set, so train.py sees one index with split = train/test.

    python scripts/stream_prep.py Skeleton Thermal
    python scripts/stream_prep.py Depth_Color --workers 4
    python scripts/stream_prep.py Thermal --limit 50          # quick check
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))

import prep  # noqa: E402

VOL_DIR = os.path.join(ROOT, "data", "Small-Model-Track", "Training", "data")
ENTRIES = os.path.join(ROOT, "logs", "har_entries.parquet")
EXPECTED = {d: 5120 * 1024 * 1024 for d in range(8)}      # HAR.z01..z08


def volumes_ready(disks, vz):
    """Which of the needed volumes are fully downloaded."""
    missing = []
    for d in sorted(disks):
        p = vz.path(d)
        if not os.path.exists(p) or (d in EXPECTED and os.path.getsize(p) < EXPECTED[d]):
            missing.append(os.path.basename(p))
    return missing


def train_recs(entries, modality):
    g = entries[entries["mod"] == modality]
    if modality == prep.SKELETON:
        g = g[g.name.str.lower().str.endswith(".json")]
    else:
        g = g[g.name.str.lower().str.endswith(tuple(prep.IMG_EXT))]
    recs = []
    for (action, user, trial), grp in g.groupby(["action", "user", "trial"], sort=True):
        head = str(action).split("_")[0]
        if not head.isdigit():
            continue
        rows = sorted(zip(grp.name, grp.disk, grp.loff, grp.csize),
                      key=lambda r: prep.frame_key(os.path.basename(r[0])))
        frames = [(int(d), int(o), int(c)) for _, d, o, c in rows]
        recs.append(dict(frames=frames, split="train", user=str(user),
                         trial=str(trial), action_id=int(head),
                         clip_id=f"{action}/{user}/{trial}"))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modalities", nargs="+")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--skel-frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "compact"))
    ap.add_argument("--boxes", help="person_boxes.parquet from the detection kernel: crop "
                    "every clip to its detected person instead of the foreground heuristic")
    ap.add_argument("--suffix", default="", help="output folder = modality + suffix")
    args = ap.parse_args()

    entries = pd.read_parquet(ENTRIES)
    _, test_root = prep.find_roots(os.path.join(ROOT, "extracted"))
    vz = prep.ZipVolumes(VOL_DIR)
    boxes = None
    if args.boxes:
        b = pd.read_parquet(args.boxes)
        boxes = {c: tuple(int(v) for v in r)
                 for c, *r in zip(b.clip_id, b.x0, b.y0, b.x1, b.y1, b.h, b.w)}
        print(f"person boxes: {sum(v[0] >= 0 for v in boxes.values())}/{len(boxes)} "
              f"clips with a person", flush=True)

    for mod in args.modalities:
        t0 = time.time()
        recs = train_recs(entries, mod)
        g = entries[entries["mod"] == mod]
        # Every volume this modality's entries start or end in (the listing
        # records where each entry's bytes finish, so spill-over is exact
        # rather than assumed for every volume).
        disks = set(g.disk.astype(int)) | set(g.disk_end.astype(int))
        missing = volumes_ready(disks, vz)
        if missing:
            print(f"[{mod}] waiting on volumes {missing} — skipped")
            continue
        test = prep.discover_test(test_root, mod) if test_root else []
        if args.limit:
            recs = recs[:args.limit]
        ntr, users = len(recs), sorted({r["user"] for r in recs})
        print(f"[{mod}] {ntr} train clips / {len(users)} users + {len(test)} test clips "
              f"| volumes {sorted(disks)}", flush=True)

        allr = recs + test
        if mod == prep.SKELETON:
            df = prep.pack_skeleton(allr, args.out, args.skel_frames, args.workers,
                                    vol_dir=VOL_DIR)
        else:
            df = prep.pack_images(mod, allr, args.out, args.frames, args.size,
                                  not args.no_crop, args.quality, args.workers,
                                  vol_dir=VOL_DIR, boxes=boxes, name=mod + args.suffix)
        name = mod if mod == prep.SKELETON else mod + args.suffix
        trn = df[df.split == "train"]
        summary = dict(modality=name, clips=int(len(df)), train=int(len(trn)),
                       test=int((df.split == "test").sum()),
                       users=sorted(trn.user.astype(str).unique().tolist()),
                       classes=int(trn.action_id.nunique()),
                       minutes=round((time.time() - t0) / 60, 1))
        with open(os.path.join(args.out, name, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[{mod}] done: {summary['train']} train + {summary['test']} test, "
              f"{summary['classes']}/40 classes, {len(summary['users'])} users, "
              f"{summary['minutes']} min\n", flush=True)


if __name__ == "__main__":
    main()
