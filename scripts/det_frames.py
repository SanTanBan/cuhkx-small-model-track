"""
Pack a few native-resolution IR frames per clip for person detection on Kaggle.

The person fills only ~12% of the 640x480 frame, so the image models see a
hand holding a phone as a handful of pixels. Detecting the person and cropping
Depth + IR to it (the two are pixel-aligned: same sensor) puts ~3x more pixels
on the subject. Detection is too slow on this laptop (~1.6 s/frame), so the
frames go to a Kaggle CPU kernel; only the boxes come back.

Output (flat dataset folder, same blob format as prep.pack_images):
    compact/IRdet/blob_NNN.bin   JPEG bytes, native size (no crop, no resize)
    compact/IRdet/index.parquet  clip_id, split, user, trial, action_id,
                                 blob, offsets, lengths, t, h, w

    python scripts/det_frames.py --frames 4
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

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from stream_prep import train_recs  # noqa: E402

VOL_DIR = os.path.join(ROOT, "data", "Small-Model-Track", "Training", "data")
ENTRIES = os.path.join(ROOT, "logs", "har_entries.parquet")


def native_jpegs(a):
    """Worker: one clip's detector input frames (prep.det_jpegs — infer.py
    builds the test-time input with the same function)."""
    i, (src, T, quality) = a
    try:
        return i, prep.det_jpegs(src, T, quality)
    except Exception:
        return i, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "compact", "IRdet"))
    args = ap.parse_args()

    entries = pd.read_parquet(ENTRIES)
    _, test_root = prep.find_roots(os.path.join(ROOT, "extracted"))
    recs = train_recs(entries, "IR") + prep.discover_test(test_root, "IR")
    print(f"IR clips: {sum(r['split'] == 'train' for r in recs)} train + "
          f"{sum(r['split'] == 'test' for r in recs)} test", flush=True)

    prep._fresh_dir(args.out)
    jobs = [(i, (prep._src(r), args.frames, args.quality)) for i, r in enumerate(recs)]
    rows, bi, off, t0 = [], 0, 0, time.time()
    bf = open(os.path.join(args.out, f"blob_{bi:03d}.bin"), "wb")
    try:
        with prep.ProcessPoolExecutor(max_workers=args.workers,
                                      initializer=prep.init_worker,
                                      initargs=(VOL_DIR,)) as ex:
            for n, fut in enumerate(prep.as_completed([ex.submit(native_jpegs, j)
                                                       for j in jobs]), 1):
                i, res = fut.result()
                prep._progress(n, len(jobs), t0)
                if res is None:
                    continue
                bufs, (h, w) = res
                r = recs[i]
                if len(rows) and len(rows) % prep.SHARD_CLIPS == 0:
                    bf.close(); bi += 1; off = 0
                    bf = open(os.path.join(args.out, f"blob_{bi:03d}.bin"), "wb")
                offs, lens = [], []
                for b in bufs:
                    offs.append(off); lens.append(len(b)); bf.write(b); off += len(b)
                rows.append(dict(clip_id=r["clip_id"], split=r["split"], user=r["user"],
                                 trial=r["trial"], action_id=r["action_id"], blob=bi,
                                 offsets=offs, lengths=lens, t=len(bufs), h=h, w=w))
    finally:
        bf.close()

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(args.out, "index.parquet"), index=False)
    gb = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)
             if f.endswith(".bin")) / 1e9
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(dict(clips=int(len(df)), train=int((df.split == "train").sum()),
                       test=int((df.split == "test").sum()), frames=args.frames,
                       gb=round(gb, 3)), fh, indent=2)
    print(f"[IRdet] {len(df)} clips, {bi + 1} blobs, {gb:.2f} GB, "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
