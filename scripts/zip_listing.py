"""
List the split training archive from its final volume alone.

A split zip keeps its central directory at the end of the last volume
(HAR.zip). Parsing it directly — before the 40 GB of earlier volumes arrive —
yields the real training clip counts per modality (to budget GPU hours) and
which volumes hold each modality's bytes (to start extracting a modality as
soon as its volumes land instead of waiting for all nine).

    python scripts/zip_listing.py [path/to/HAR.zip]
"""
import collections
import json
import os
import struct
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "data", "Small-Model-Track", "Training", "data", "HAR.zip")
VOLUME = 5120 * 1024 * 1024        # each HAR.z0N is exactly 5120 MiB


def volume_path(zip_path, disk, last_disk):
    return zip_path if disk == last_disk else f"{zip_path[:-4]}.z{disk + 1:02d}"


def read_central_directory(zip_path):
    size = os.path.getsize(zip_path)
    with open(zip_path, "rb") as f:
        tail = min(size, 65536 + 22)
        f.seek(size - tail)
        buf = f.read(tail)
        i = buf.rfind(b"PK\x05\x06")
        if i < 0:
            raise SystemExit("no end-of-central-directory record — file incomplete?")
        (_, disk_no, cd_disk, _, n_total, cd_size, cd_off, _) = \
            struct.unpack("<IHHHHIIH", buf[i:i + 22])

        j = buf.rfind(b"PK\x06\x07", 0, i)          # ZIP64 locator
        if j >= 0:
            _, z64_disk, z64_off, n_disks = struct.unpack("<IIQI", buf[j:j + 20])
            if z64_disk != disk_no:
                raise SystemExit(f"ZIP64 record sits on disk {z64_disk}; need that volume")
            f.seek(z64_off)
            rec = f.read(56)
            (sig, _, _, _, disk_no, cd_disk, _, n_total, cd_size, cd_off) = \
                struct.unpack("<IQHHIIQQQQ", rec)
            assert sig == 0x06064B50, "bad ZIP64 EOCD"

        last_disk = disk_no
        if cd_disk != last_disk:
            raise SystemExit(f"central directory starts on disk {cd_disk} "
                             f"({volume_path(zip_path, cd_disk, last_disk)}) — "
                             f"that volume is needed too")
        f.seek(cd_off)
        cd = f.read(cd_size)

    entries, p = [], 0
    for _ in range(n_total):
        (sig, _, _, flags, comp, _, _, _, csize, usize, nlen, xlen, clen,
         dstart, _, _, loff) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        if sig != 0x02014B50:
            raise SystemExit(f"bad central directory entry at {p}")
        name = cd[p + 46:p + 46 + nlen]
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, loff) or dstart == 0xFFFF:
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    d, k = extra[q + 4:q + 4 + hlen], 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", d[k:k + 8])[0]; k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", d[k:k + 8])[0]; k += 8
                    if loff == 0xFFFFFFFF:
                        loff = struct.unpack("<Q", d[k:k + 8])[0]; k += 8
                    if dstart == 0xFFFF:
                        dstart = struct.unpack("<I", d[k:k + 4])[0]
                    break
                q += 4 + hlen
        name = name.decode("utf-8" if flags & 0x800 else "cp437", "replace")
        start = dstart * VOLUME + loff
        entries.append((name, comp, csize, usize, dstart, loff, start))
        p += 46 + nlen + xlen + clen
    return entries, last_disk


def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    entries, last_disk = read_central_directory(zip_path)
    df = pd.DataFrame(entries, columns=["name", "comp", "csize", "usize",
                                        "disk", "loff", "gstart"])
    df = df[~df.name.str.contains("__MACOSX|/\\._|\\.DS_Store", regex=True)]
    df = df[~df.name.str.endswith("/")]
    parts = df.name.str.split("/", expand=True)
    # HAR/data/<modality>/<action>/<user>/<trial>/<...file>
    df["mod"], df["action"], df["user"], df["trial"] = parts[2], parts[3], parts[4], parts[5]
    df["gend"] = df.gstart + df.csize + 256
    df["disk_end"] = (df.gend // VOLUME).clip(upper=last_disk)

    print(f"{zip_path}\n{len(entries)} entries, {last_disk + 1} volumes, "
          f"compression methods {sorted(df.comp.unique().tolist())}\n")
    print(f"{'modality':<14}{'clips':>7}{'files':>9}{'GB zip':>8}{'GB raw':>8}"
          f"{'users':>7}{'actions':>9}   volumes")
    summary = {}
    for mod, g in df.groupby("mod"):
        clips = g.groupby(["action", "user", "trial"]).size()
        vols = sorted(set(range(int(g.disk.min()), int(g.disk_end.max()) + 1)))
        summary[mod] = dict(clips=int(len(clips)), files=int(len(g)),
                            gb_zip=round(g.csize.sum() / 1e9, 2),
                            gb_raw=round(g.usize.sum() / 1e9, 2),
                            users=sorted(g.user.unique().tolist(), key=lambda s: (len(s), s)),
                            actions=int(g.action.nunique()),
                            volumes=vols,
                            files_per_clip_p50=float(clips.median()))
        s = summary[mod]
        print(f"{mod:<14}{s['clips']:>7}{s['files']:>9}{s['gb_zip']:>8}{s['gb_raw']:>8}"
              f"{len(s['users']):>7}{s['actions']:>9}   {vols}")

    print()
    for mod, s in summary.items():
        print(f"{mod:<14} users {s['users']}")
    sub = collections.Counter(tuple(n.split("/")[6:7]) for n in
                              df[df["mod"] == "Skeleton"].name.head(2000))
    print(f"\nSkeleton sub-paths (sample): {dict(sub.most_common(5))}")

    out = os.path.join(ROOT, "logs", "har_listing.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    df.drop(columns=["gend"]).to_parquet(os.path.join(ROOT, "logs", "har_entries.parquet"),
                                         index=False)
    print(f"\nwrote {out} and logs/har_entries.parquet")


if __name__ == "__main__":
    main()
