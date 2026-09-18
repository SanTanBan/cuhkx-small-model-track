"""
Dataset compaction — single source of truth for local and Kaggle runs.

Two output families:

  image modalities (Depth_Color, IR, Thermal)
      <out>/<mod>/blob_NNN.bin      concatenated JPEG bytes
      <out>/<mod>/index.parquet     clip_id, split, user, trial, action_id,
                                    blob, offsets, lengths, t, size

  Skeleton
      <out>/Skeleton/poses.npy      float16 [N, T, 17, 3]
      <out>/Skeleton/index.parquet  clip_id, split, user, trial, action_id, row

A clip's frames can come from a directory on disk, or straight out of the
split training archive: every entry in HAR.z01..HAR.zip is stored uncompressed,
so a frame is a contiguous byte range at a known (volume, offset). Reading only
the sampled frames avoids extracting ~45 GB and lets a modality be processed as
soon as the volumes that hold it have arrived.

Measured facts this is built around (from the real data):
  * ~2,930 training clips per modality over 18 users; 405 test clips over 4.
  * Depth_Color / IR / Skeleton are present for 100% of test clips and are
    frame-aligned 1:1. Thermal covers 97.5% at ~2.2x the frame rate. Radar is
    empty for 207/405 test clips, so it is not worth modelling.
  * Skeleton is Human3.6M 17-joint 3D, root-centred in x/y with z as height.
  * Clips are short: median 20 depth frames, min 2.
  * The archives carry macOS cruft (__MACOSX/, .DS_Store) and a stray .claude/
    directory inside the test root — all must be filtered out or they are
    silently ingested as clips. 4 IR test clips are zero-filled files.
"""
from __future__ import annotations

import json
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor as _PoolExecutor
from concurrent.futures import as_completed as _as_completed

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)


# -------------------------------------------------- memory-aware execution
#
# Each spawned worker re-imports cv2 / numpy / pandas (~0.75 GB of commit on
# Windows). On a small, busy machine a pool of 2-4 exhausted the commit limit
# and died with a native crash (0xC000070A) and no traceback. The worker count
# is therefore capped by free memory, and a cap of 1 runs jobs lazily in this
# process: no spawn, and only one clip's frames held in memory at a time.

def _free_gb():
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _MS()
            m.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return min(m.ullAvailPhys, m.ullAvailPageFile) / 1e9
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None


def safe_workers(requested):
    free = _free_gb()
    if free is None:
        return max(1, requested)
    return max(1, min(requested, int((free - 1.0) / 0.75)))


class _LazyFuture:
    def __init__(self, fn, args):
        self._fn, self._args = fn, args

    def result(self):
        return self._fn(*self._args)


class _InlineExecutor:
    def __init__(self, initializer=None, initargs=()):
        if initializer:
            initializer(*initargs)

    def submit(self, fn, *args):
        return _LazyFuture(fn, args)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def ProcessPoolExecutor(max_workers=None, initializer=None, initargs=()):
    n = safe_workers(max_workers or 1)
    free = _free_gb()
    print(f"    workers: {n} (requested {max_workers}; "
          f"{free if free is not None else float('nan'):.1f} GB free)", flush=True)
    if n <= 1:
        return _InlineExecutor(initializer, initargs)
    return _PoolExecutor(max_workers=n, initializer=initializer, initargs=initargs)


def as_completed(fs):
    if fs and isinstance(fs[0], _LazyFuture):
        return iter(fs)          # results are computed on .result(), in order
    return _as_completed(fs)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SHARD_CLIPS = 512
IMAGE_MODALITIES = ("Depth_Color", "IR", "Thermal")
SKELETON = "Skeleton"
N_JOINTS = 17

# Human3.6M 17-joint bone list, inferred from the measured joint heights
# (0 root at x=y=0, 3/6 feet at z~0, 10 head at z~1.23).
H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),            # right leg
    (0, 4), (4, 5), (5, 6),            # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),   # spine -> head
    (8, 11), (11, 12), (12, 13),       # left arm
    (8, 14), (14, 15), (15, 16),       # right arm
]


def is_junk(name):
    return name in ("__MACOSX", ".DS_Store", ".claude") or name.startswith("._")


def frame_key(name):
    """Numeric-aware sort key: frame_2 < frame_10."""
    dig = "".join(c if c.isdigit() else " " for c in name).split()
    return ([int(x) for x in dig], name)


def list_frames(d):
    try:
        names = [f for f in os.listdir(d)
                 if os.path.splitext(f)[1].lower() in IMG_EXT and not is_junk(f)]
    except OSError:
        return []
    return [os.path.join(d, f) for f in sorted(names, key=frame_key)]


def sample_idx(n, t):
    if n <= 0:
        return []
    if n <= t:
        return list(range(n)) + [n - 1] * (t - n)
    return np.linspace(0, n - 1, t).round().astype(int).tolist()


# ------------------------------------------------------- split-zip reader

class ZipVolumes:
    """Random access to STORED entries of a split zip (HAR.z01.. + HAR.zip)."""

    def __init__(self, vol_dir, stem="HAR", last_disk=8):
        self.vol_dir, self.stem, self.last = vol_dir, stem, last_disk
        self._fh, self._sz = {}, {}

    def path(self, disk):
        ext = "zip" if disk == self.last else f"z{disk + 1:02d}"
        return os.path.join(self.vol_dir, f"{self.stem}.{ext}")

    def _size(self, disk):
        if disk not in self._sz:
            self._sz[disk] = os.path.getsize(self.path(disk))
        return self._sz[disk]

    def _read(self, disk, off, n):
        # an offset past the end of a volume continues into the next one
        while off >= self._size(disk):
            off -= self._size(disk)
            disk += 1
        out = bytearray()
        while n > 0:
            f = self._fh.get(disk)
            if f is None:
                f = self._fh[disk] = open(self.path(disk), "rb")
            f.seek(off)
            chunk = f.read(n)
            out += chunk
            n -= len(chunk)
            if n > 0:
                if disk >= self.last:
                    raise EOFError("read past the final volume")
                disk, off = disk + 1, 0
        return bytes(out)

    def entry(self, disk, loff, csize):
        hdr = self._read(disk, loff, 30)
        if hdr[:4] != b"PK\x03\x04":
            raise ValueError(f"no local header at disk {disk} offset {loff}")
        nlen, xlen = struct.unpack("<HH", hdr[26:30])
        return self._read(disk, loff + 30 + nlen + xlen, csize)


_ZIP = None   # per-process reader, set by init_worker


def init_worker(vol_dir):
    global _ZIP
    _ZIP = ZipVolumes(vol_dir) if vol_dir else None


# ------------------------------------------------------------------- images

def to_bgr8(img):
    """Any decoded frame (uint16, float, gray, BGRA) -> uint8 BGR."""
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype == np.uint16:
        lo, hi = np.percentile(img, [1, 99])
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None if img is None else to_bgr8(img)


def decode_img(buf):
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    return None if img is None else to_bgr8(img)


def foreground_box(img, modality):
    """Cheap deterministic person localisation; None means 'keep full frame'."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(g, (5, 5), 0)
    if modality.lower().startswith("thermal"):
        mask = (g >= np.percentile(g, 92)).astype(np.uint8) * 255
    else:
        _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.01 * g.size:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return x, y, x + w, y + h


def union_box(boxes, shape, pad=0.15):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    H, W = shape
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(max(x1 - x0, y1 - y0) * (1 + 2 * pad), 0.30 * min(H, W))
    x0 = int(max(0, cx - side / 2)); x1 = int(min(W, cx + side / 2))
    y0 = int(max(0, cy - side / 2)); y1 = int(min(H, cy + side / 2))
    return None if (x1 - x0 < 16 or y1 - y0 < 16) else (x0, y0, x1, y1)


def person_box(frame_boxes, shape, margin=1.3, min_side=0.30):
    """
    One square crop per clip from per-frame person detections.

    frame_boxes: per frame, (x0, y0, x1, y1, score) for the best person, or
    None. The union over frames covers the whole movement, and `margin` scales
    its larger side so hands and held objects at the body's edge stay inside.
    The square is shifted — never truncated — to lie within the frame.
    Returns (x0, y0, x1, y1) in pixels, or None when no frame had a person
    (the caller then keeps the full frame). Used for both the training
    crops (Kaggle detection kernel) and test-time crops (infer.py), so the two
    cannot drift apart.
    """
    boxes = [b for b in frame_boxes if b is not None]
    if not boxes:
        return None
    H, W = shape
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(max(x1 - x0, y1 - y0) * margin, min_side * min(H, W))
    side = min(side, min(H, W))
    x0 = min(max(0.0, cx - side / 2), W - side)
    y0 = min(max(0.0, cy - side / 2), H - side)
    return int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side))


def _frame_source(src):
    """(frames, getter) for a clip given as a directory or a list of zip entries."""
    if isinstance(src, str):
        return list_frames(src), load_img
    return list(src), lambda e: decode_img(_ZIP.entry(*e))


def encode_clip(a):
    src, modality, T, size, crop, quality = a[:6]
    box = a[6] if len(a) > 6 else None
    frames, get = _frame_source(src)
    if not frames:
        return None

    # A few recordings contain corrupt frames. Dropping a whole clip over one
    # bad frame would forfeit a test prediction outright, so substitute a
    # neighbour and only give up if nothing in the clip decodes.
    imgs, last = [], None
    for i in sample_idx(len(frames), T):
        try:
            im = get(frames[i])
        except Exception:
            im = None
        if im is None:
            im = last
        else:
            last = im
        imgs.append(im)
    if all(im is None for im in imgs):
        return None
    if imgs[0] is None:                       # backfill any leading failures
        first = next(im for im in imgs if im is not None)
        imgs = [first if im is None else im for im in imgs]
    h, w = imgs[0].shape[:2]
    imgs = [im if im.shape[:2] == (h, w) else cv2.resize(im, (w, h)) for im in imgs]

    if box is not None:
        # A person box from the detector (person_box), in the pixel grid of the
        # frames it was found on (bh x bw). IR and Depth_Color share one
        # sensor, so a box found on IR crops both; rescale only if this
        # modality's frames differ in size. x0 < 0 means no person was found:
        # keep the full frame, exactly as test time does.
        x0, y0, x1, y1, bh, bw = box
        if x0 >= 0:
            sx, sy = w / bw, h / bh
            x0, x1 = int(round(x0 * sx)), int(round(x1 * sx))
            y0, y1 = int(round(y0 * sy)), int(round(y1 * sy))
            imgs = [im[y0:y1, x0:x1] for im in imgs]
    elif crop:
        probe = imgs[:: max(1, len(imgs) // 4)][:4]
        fbox = union_box([foreground_box(im, modality) for im in probe],
                         imgs[0].shape[:2])
        if fbox:
            x0, y0, x1, y1 = fbox
            imgs = [im[y0:y1, x0:x1] for im in imgs]
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    out = []
    for im in imgs:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, enc)
        if not ok:
            return None
        out.append(buf.tobytes())
    return len(frames), out


# ----------------------------------------------------------------- skeleton

def parse_pose_bytes(raw):
    """
    One frame's JSON -> [17,3] float32, or None.

    The file holds a list of detected people; keep the one with the best mean
    keypoint score (ties broken by vertical extent) so multi-person frames do
    not inject a bystander's pose.
    """
    try:
        people = json.loads(raw)
    except Exception:
        return None
    if not isinstance(people, list) or not people:
        return None

    best, best_score = None, -1.0
    for p in people:
        kp = p.get("keypoints") if isinstance(p, dict) else None
        if not kp or len(kp) < N_JOINTS:
            continue
        a = np.asarray(kp, dtype=np.float32)[:N_JOINTS]
        if a.ndim != 2:
            continue
        if a.shape[1] < 3:
            a = np.concatenate([a, np.zeros((N_JOINTS, 3 - a.shape[1]), np.float32)], 1)
        sc = p.get("keypoint_scores") or []
        score = float(np.mean(sc)) if len(sc) else 0.0
        score += 0.01 * float(np.ptp(a[:, 2]))     # prefer the fuller skeleton
        if score > best_score:
            best, best_score = a[:, :3], score
    return best


def read_pose_json(path):
    try:
        with open(path, "rb") as f:
            return parse_pose_bytes(f.read())
    except OSError:
        return None


def encode_skeleton(a):
    src, T = a
    if isinstance(src, str):
        pdir = os.path.join(src, "predictions")
        root = pdir if os.path.isdir(pdir) else src
        try:
            names = sorted((f for f in os.listdir(root)
                            if f.endswith(".json") and not is_junk(f)), key=frame_key)
        except OSError:
            return None
        files = [os.path.join(root, f) for f in names]
        get = read_pose_json
    else:
        files = list(src)
        get = lambda e: parse_pose_bytes(_ZIP.entry(*e))
    if not files:
        return None

    poses, last = [], None
    for i in sample_idx(len(files), T):
        try:
            p = get(files[i])
        except Exception:
            p = None
        if p is None:
            p = last if last is not None else np.zeros((N_JOINTS, 3), np.float32)
        last = p
        poses.append(p)
    return len(files), np.stack(poses).astype(np.float32)


def normalize_pose(seq):
    """
    Make the sequence subject-invariant without discarding posture height.

    The pose estimator already centres the root horizontally (x = y = 0) and
    grounds the skeleton vertically (feet at z ~ 0), so z carries hip and head
    height — exactly what separates sit down / stand up / squat / lie down, and
    the vertical bounce of jogging or jumping jacks. Re-centre only x/y on the
    root each frame (a no-op on clean frames, a repair on drifted ones), keep
    z, then divide by the subject's own torso length so people of different
    builds share one scale.
    """
    seq = seq.astype(np.float32).copy()
    seq[:, :, :2] -= seq[:, 0:1, :2]
    torso = np.linalg.norm(seq[:, 8, :] - seq[:, 0, :], axis=-1)   # root->thorax
    scale = np.median(torso[torso > 1e-3]) if np.any(torso > 1e-3) else 1.0
    return seq / max(float(scale), 1e-3)


# ---------------------------------------------------------------- discovery

def discover_train(train_root, modality):
    """HAR/data/<modality>/<action_id>_<name>/<user>/<trial>/"""
    mroot = os.path.join(train_root, modality)
    if not os.path.isdir(mroot):
        return []
    recs = []
    for action in sorted(os.listdir(mroot)):
        if is_junk(action):
            continue
        aroot = os.path.join(mroot, action)
        if not os.path.isdir(aroot):
            continue
        head = action.split("_")[0]
        if not head.isdigit():
            continue
        aid = int(head)
        for user in sorted(os.listdir(aroot)):
            if is_junk(user):
                continue
            uroot = os.path.join(aroot, user)
            if not os.path.isdir(uroot):
                continue
            for trial in sorted(os.listdir(uroot)):
                if is_junk(trial):
                    continue
                troot = os.path.join(uroot, trial)
                if os.path.isdir(troot):
                    recs.append(dict(clip_dir=troot, split="train", user=user,
                                     trial=trial, action_id=aid,
                                     clip_id=f"{action}/{user}/{trial}"))
    return recs


def discover_test(test_root, modality):
    """small_model_track_test/SM_test_XXXX/<modality>/"""
    if not os.path.isdir(test_root):
        return []
    recs = []
    for clip in sorted(os.listdir(test_root)):
        if not clip.startswith("SM_test"):      # skips .claude, .DS_Store, ...
            continue
        mdir = os.path.join(test_root, clip, modality)
        if os.path.isdir(mdir):
            recs.append(dict(clip_dir=mdir, split="test", user="TEST",
                             trial=clip, action_id=-1, clip_id=clip))
    return recs


def find_roots(extracted):
    """Locate HAR/data and the test clip root, ignoring the __MACOSX shadow tree."""
    train_root = test_root = None
    for dirpath, dirnames, _ in os.walk(extracted):
        dirnames[:] = [d for d in dirnames if not is_junk(d)]
        if "__MACOSX" in dirpath:
            continue
        base = os.path.basename(dirpath)
        if base == "data" and os.path.basename(os.path.dirname(dirpath)) == "HAR":
            train_root = train_root or dirpath
        if base == "small_model_track_test":
            test_root = test_root or dirpath
    return train_root, test_root


# ------------------------------------------------------------------ packing

def _progress(done, total, t0, every=500):
    if done % every == 0 or done == total:
        r = done / max(time.time() - t0, 1e-9)
        print(f"    {done}/{total}  {r:.1f} clips/s  "
              f"eta {(total-done)/max(r,1e-9)/60:.1f} min", flush=True)


def _fresh_dir(mout):
    """
    Remove previous outputs by unlinking rather than truncating: staged dataset
    folders hardlink these files, and truncating in place would corrupt a copy
    that may be mid-upload.
    """
    os.makedirs(mout, exist_ok=True)
    for f in os.listdir(mout):
        p = os.path.join(mout, f)
        if os.path.isfile(p):
            os.remove(p)


def _src(rec):
    return rec["frames"] if "frames" in rec else rec["clip_dir"]


def _encode_indexed(a):
    i, rest = a
    try:
        return i, encode_clip(rest)
    except Exception:
        return i, None


def _skel_indexed(a):
    i, rest = a
    try:
        return i, encode_skeleton(rest)
    except Exception:
        return i, None


def pack_images(modality, recs, out_root, T, size, crop, quality, workers,
                vol_dir=None, boxes=None, name=None):
    """
    boxes: optional {clip_id: (x0, y0, x1, y1, h, w)} person boxes. When given
    they replace the foreground crop for every clip, and a clip without one
    keeps its full frame. name: output folder (default: the modality).
    """
    mout = os.path.join(out_root, name or modality)
    _fresh_dir(mout)
    full = (-1, -1, -1, -1, 1, 1)
    clip_box = [boxes.get(r["clip_id"], full) if boxes is not None else None
                for r in recs]
    jobs = [(i, (_src(r), modality, T, size, crop, quality, clip_box[i]))
            for i, r in enumerate(recs)]

    rows, bi, off, done, failed = [], 0, 0, 0, 0
    t0 = time.time()
    bf = open(os.path.join(mout, f"blob_{bi:03d}.bin"), "wb")
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                 initargs=(vol_dir,)) as ex:
            for fut in as_completed([ex.submit(_encode_indexed, j) for j in jobs]):
                i, res = fut.result()
                done += 1
                _progress(done, len(jobs), t0)
                if res is None:
                    failed += 1
                    continue
                nsrc, bufs = res
                r = recs[i]
                if len(rows) and len(rows) % SHARD_CLIPS == 0:
                    bf.close(); bi += 1; off = 0
                    bf = open(os.path.join(mout, f"blob_{bi:03d}.bin"), "wb")
                offs, lens = [], []
                for b in bufs:
                    offs.append(off); lens.append(len(b)); bf.write(b); off += len(b)
                row = dict(clip_id=r["clip_id"], split=r["split"],
                           user=r["user"], trial=r["trial"],
                           action_id=r["action_id"], n_src_frames=nsrc,
                           blob=bi, offsets=offs, lengths=lens,
                           t=len(bufs), size=size)
                if clip_box[i] is not None:
                    row["box"] = [int(v) for v in clip_box[i][:4]]
                rows.append(row)
    finally:
        bf.close()

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(mout, "index.parquet"), index=False)
    gb = sum(os.path.getsize(os.path.join(mout, f)) for f in os.listdir(mout)
             if f.endswith(".bin")) / 1e9
    print(f"  [{modality}] {len(df)} clips ({failed} undecodable), {bi+1} blobs, "
          f"{gb:.2f} GB, {(time.time()-t0)/60:.1f} min", flush=True)
    return df


def pack_skeleton(recs, out_root, T, workers, vol_dir=None):
    mout = os.path.join(out_root, SKELETON)
    _fresh_dir(mout)
    jobs = [(i, (_src(r), T)) for i, r in enumerate(recs)]

    rows, arrs, done, failed = [], [], 0, 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(vol_dir,)) as ex:
        for fut in as_completed([ex.submit(_skel_indexed, j) for j in jobs]):
            i, res = fut.result()
            done += 1
            _progress(done, len(jobs), t0, every=1000)
            if res is None:
                failed += 1
                continue
            nsrc, seq = res
            r = recs[i]
            rows.append(dict(clip_id=r["clip_id"], split=r["split"],
                             user=r["user"], trial=r["trial"],
                             action_id=r["action_id"], n_src_frames=nsrc,
                             row=len(arrs), t=T))
            arrs.append(normalize_pose(seq).astype(np.float16))

    if not arrs:
        print(f"  [{SKELETON}] nothing packed")
        return pd.DataFrame()
    poses = np.stack(arrs)
    np.save(os.path.join(mout, "poses.npy"), poses)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(mout, "index.parquet"), index=False)
    print(f"  [{SKELETON}] {len(df)} clips ({failed} failed), array {poses.shape} "
          f"{poses.nbytes/1e6:.1f} MB, {(time.time()-t0)/60:.1f} min", flush=True)
    return df
