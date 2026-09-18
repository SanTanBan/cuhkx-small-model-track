"""
Operate the Kaggle side of the pipeline from this laptop (which has no GPU).

Every action shells out to the official `kaggle` CLI, authenticated through
~/.kaggle/access_token — the same operations a person performs in the web UI.

Datasets are uploaded flat (no sub-directories, so the CLI's directory
archiving never applies) and each carries a `modality.json` marker; kernels
find datasets by that marker and symlink them into one data root. Keeping each
modality in its own private dataset means Skeleton can train while Thermal is
still being preprocessed, and re-versioning one modality never re-uploads the
others.

    python scripts/kaggle_ops.py stage   meta skeleton thermal
    python scripts/kaggle_ops.py upload  meta skeleton thermal
    python scripts/kaggle_ops.py kernel  envcheck --datasets meta skeleton thermal --envcheck
    python scripts/kaggle_ops.py kernel  skel --datasets meta skeleton --jobs Skeleton:0 Skeleton:1
    python scripts/kaggle_ops.py push    skel
    python scripts/kaggle_ops.py wait    skel
    python scripts/kaggle_ops.py pull    skel
    python scripts/kaggle_ops.py fuse
    python scripts/kaggle_ops.py submit  runs/submission.csv -m "skeleton 5-fold"
    python scripts/kaggle_ops.py subs | lb
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "kaggle")
COMPACT = os.path.join(ROOT, "compact")
DS_DIR = os.path.join(ROOT, "ds")
KERNELS = os.path.join(ROOT, "kernels")
RUNS = os.path.join(ROOT, "runs")
EXTRACTED = os.path.join(ROOT, "extracted")

COMP = "cuhk-x-competition-small-model-track"
OWNER = os.environ.get("KAGGLE_OWNER", "santanubanerjee9")
RESNET18 = "resnet18-f37072fd.pth"
NUM_CLASSES = 40

# short name -> (dataset slug, compact/ sub-folder; None = shared metadata)
DATASETS = {
    "meta":     ("cuhkx-meta", None),
    "skeleton": ("cuhkx-skeleton", "Skeleton"),
    "thermal":  ("cuhkx-thermal", "Thermal"),
    "depth":    ("cuhkx-depth", "Depth_Color"),
    "ir":       ("cuhkx-ir", "IR"),
    "irdet":    ("cuhkx-irdet", "IRdet"),     # native IR frames for person detection
    "depthpc":  ("cuhkx-depth-pc", "Depth_Color_PC"),   # Depth_Color cropped to the person
    "irpc":     ("cuhkx-ir-pc", "IR_PC"),         # IR cropped to the person
    "imu":      ("cuhkx-imu", "IMU"),             # body-worn IMUs (scripts/imu_pack.py)
}

# Per-modality training defaults; a job spec may override epochs and add a
# wall-clock cap: "Mod:fold[:epochs[:cap_minutes]]".
JOB_DEFAULTS = {
    "Skeleton": ["--epochs", "40", "--batch-size", "32", "--frames", "32",
                 "--lr", "1e-3", "--workers", "2"],
    "_image":   ["--epochs", "20", "--batch-size", "16", "--frames", "16",
                 "--size", "144", "--lr", "3e-4", "--workers", "3"],
}


# ------------------------------------------------------------------ helpers

def kaggle(*args, check=True):
    exe = shutil.which("kaggle") or "kaggle"
    r = subprocess.run([exe, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    # The CLI prints a urllib3/chardet version warning to stderr on every call.
    # Filter stderr on its own: stdout often lacks a trailing newline, so
    # concatenating the two streams glues the real answer onto the warning
    # line, and the filter would then silently delete it.
    err = "\n".join(l for l in (r.stderr or "").splitlines()
                    if "RequestsDependencyWarning" not in l
                    and "warnings.warn(" not in l)
    out = "\n".join(x for x in ((r.stdout or "").rstrip("\n"), err) if x)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed ({r.returncode}):\n{out}")
    return r.returncode, out


def folder_bytes(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
               if os.path.isfile(os.path.join(path, f)))


def _link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)          # hardlink: instant and free on the same volume
    except OSError:
        shutil.copy2(src, dst)


def _first(*cands):
    return next((c for c in cands if c and os.path.exists(c)), None)


def kernel_ref(name):
    return f"{OWNER}/cuhkx-{name}"


# -------------------------------------------------------------------- stage

def cmd_stage(a):
    for n in a.names:
        slug, mod = DATASETS[n]
        out = os.path.join(DS_DIR, n)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        marker = {"modality": mod or "_meta"}

        if mod is None:
            data_sm = os.path.join(ROOT, "data", "Small-Model-Track")
            files = [
                _first(os.path.join(EXTRACTED, "class_mapping.csv"),
                       os.path.join(data_sm, "class_mapping.csv")),
                _first(os.path.join(EXTRACTED, "test.csv"),
                       os.path.join(data_sm, "Testing", "test_file", "test.csv")),
                _first(os.path.join(EXTRACTED, "sample_submission.csv"),
                       os.path.join(data_sm, "Testing", "test_file",
                                    "sample_submission.csv")),
                _first(os.path.join(os.path.expanduser("~"), ".cache", "torch",
                                    "hub", "checkpoints", RESNET18)),
            ]
            if any(f is None for f in files):
                raise SystemExit(f"meta: missing inputs {files}")
            # Kinetics video weights ride along so kernels never need internet.
            hub = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub",
                               "checkpoints")
            files += sorted(p for p in glob.glob(os.path.join(hub, "*.pth"))
                            if os.path.basename(p).split("-")[0]
                            in ("s3d", "mc3_18", "r2plus1d_18", "r3d_18"))
        else:
            src = os.path.join(COMPACT, mod)
            files = [os.path.join(src, f) for f in sorted(os.listdir(src))
                     if os.path.isfile(os.path.join(src, f))] if os.path.isdir(src) else []
            if not files:
                raise SystemExit(f"{src} is empty — run 04_preprocess.py first")
            import pandas as pd
            idx = pd.read_parquet(os.path.join(src, "index.parquet"))
            trn = idx[idx.split == "train"]
            marker.update(clips=int(len(idx)), train=int(len(trn)),
                          test=int((idx.split == "test").sum()),
                          users=sorted(trn.user.astype(str).unique().tolist()),
                          classes=int(trn.action_id.nunique()))

        for f in files:
            _link(f, os.path.join(out, os.path.basename(f)))
        with open(os.path.join(out, "modality.json"), "w") as fh:
            json.dump(marker, fh, indent=2)
        with open(os.path.join(out, "dataset-metadata.json"), "w") as fh:
            json.dump({"title": slug, "id": f"{OWNER}/{slug}",
                       "licenses": [{"name": "other"}]}, fh, indent=2)
        print(f"staged {n:<9} {len(os.listdir(out)):>3} files "
              f"{folder_bytes(out)/1e6:9.1f} MB   "
              f"{ {k: v for k, v in marker.items() if k != 'users'} }")


# ------------------------------------------------------------------- upload

def ds_status(slug):
    rc, out = kaggle("datasets", "status", f"{OWNER}/{slug}", check=False)
    return out.strip().splitlines()[-1].strip().lower() if rc == 0 and out.strip() else None


def cmd_upload(a):
    for n in a.names:
        slug, _ = DATASETS[n]
        path = os.path.join(DS_DIR, n)
        if not os.path.isdir(path):
            raise SystemExit(f"stage {n} first")
        size = folder_bytes(path)
        existing = ds_status(slug)
        t = time.time()
        if existing:
            rc, out = kaggle("datasets", "version", "-p", path, "-m", a.message,
                             "-r", "skip", check=False)
        else:
            rc, out = kaggle("datasets", "create", "-p", path, "-r", "skip",
                             check=False)
        dt = max(time.time() - t, 1e-6)
        tail = [l for l in out.splitlines() if l.strip()][-4:]
        print("\n".join("   " + l for l in tail))
        print(f"{'version' if existing else 'create '} {n:<9} rc={rc}  "
              f"{size/1e6:8.1f} MB in {dt/60:5.1f} min  "
              f"= {size/1e6/dt:.2f} MB/s", flush=True)
        if rc != 0:
            raise SystemExit(rc)


def cmd_dsstatus(a):
    names = a.names or list(DATASETS)
    deadline = time.time() + a.wait * 60
    while True:
        states = {n: ds_status(DATASETS[n][0]) for n in names}
        print("  ".join(f"{n}={s}" for n, s in states.items()), flush=True)
        if not a.wait or all(s == "ready" for s in states.values()) \
                or time.time() > deadline:
            return
        time.sleep(20)


# ------------------------------------------------------------------- kernel

def parse_job(spec):
    parts = spec.split(":")
    mod = parts[0]
    fold = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    args = list(JOB_DEFAULTS["Skeleton" if mod == "Skeleton" else "_image"])
    if len(parts) > 2 and parts[2]:
        args[args.index("--epochs") + 1] = parts[2]
    cap = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
    if len(parts) > 4 and parts[4]:          # flags, e.g. temporal-aug+amp-aug
        args += ["--" + f for f in parts[4].split("+") if f]
    return dict(modality=mod, fold=fold, args=args, cap_min=cap)


def cmd_kernel(a):
    slug = f"cuhkx-{a.name}"
    kdir = os.path.join(KERNELS, a.name)
    shutil.rmtree(kdir, ignore_errors=True)
    os.makedirs(kdir)

    jobs = [parse_job(s) for s in (a.jobs or [])]
    src = {f: base64.b64encode(open(os.path.join(SRC_DIR, f), "rb").read()).decode()
           for f in (a.embed or ("cuhkx.py", "train.py"))}
    inject = {
        "@@JOBS@@": f"JOBS = {jobs!r}",
        "@@SRC@@": f"SRC = {src!r}",
        "@@BUDGET@@": f"BUDGET_H = {float(a.budget_h)!r}",
        "@@ENVCHECK@@": f"ENVCHECK = {bool(a.envcheck)!r}",
    }
    lines = []
    with open(os.path.join(SRC_DIR, a.template), encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            for marker, repl in inject.items():
                if marker in line:
                    line = repl
                    break
            lines.append(line)
    code_file = f"{slug}.py"
    with open(os.path.join(kdir, code_file), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    meta = {
        "id": f"{OWNER}/{slug}", "title": slug, "code_file": code_file,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": not a.cpu, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{OWNER}/{DATASETS[d][0]}" for d in a.datasets],
        "competition_sources": [],
        # earlier kernels whose outputs (e.g. trained checkpoints) this one reads
        "kernel_sources": [k if "/" in k else f"{OWNER}/cuhkx-{k}"
                           for k in (a.kernel_sources or [])],
        "model_sources": [],
    }
    with open(os.path.join(kdir, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    compile(open(os.path.join(kdir, code_file), encoding="utf-8").read(), code_file, "exec")
    print(f"built {kdir}  ({os.path.getsize(os.path.join(kdir, code_file))/1e3:.0f} KB)")
    for j in jobs:
        print(f"   job {j['modality']:<12} fold {j['fold']}  "
              f"{' '.join(j['args'])}  cap={j['cap_min'] or '-'}")
    print(f"   datasets: {meta['dataset_sources']}  envcheck={bool(a.envcheck)}  "
          f"gpu={meta['enable_gpu']}  budget={a.budget_h} h")


def cmd_push(a):
    """
    Push a kernel version. The CLI reports some rejections as a printed
    "Kernel push error" with exit code 0, so success is judged on the text.
    With --retry, a full GPU session pool (Kaggle allows 2 concurrent batch GPU
    sessions per account) or a used-up weekly GPU quota (30 h, shared by every
    project on the account) is waited out rather than treated as failure.
    Transient network errors are always retried (for up to 30 min without
    --retry); before re-pushing after one, the kernel's status is checked so a
    push that landed but lost its reply is never launched twice.
    """
    kdir = os.path.join(KERNELS, a.name)
    args = ["kernels", "push", "-p", kdir]
    acc = a.accelerator
    if acc is None:
        with open(os.path.join(kdir, "kernel-metadata.json")) as fh:
            if json.load(fh).get("enable_gpu"):
                # Kaggle's default GPU is a P100 (sm_60), which the image's
                # current PyTorch (cu128) no longer supports: the first CUDA op
                # fails with "no kernel image is available". Always ask for a T4.
                acc = "NvidiaTeslaT4"
    if acc:
        args += ["--accelerator", acc]
    t0 = time.time()
    limit = a.retry_hours * 3600 if a.retry else 1800
    while True:
        rc, out = kaggle(*args, check=False)
        text = out.strip()
        busy = "session count" in text.lower()
        quota = "quota" in text.lower()       # weekly GPU hours used up: wait for the reset
        net = is_net_error(text)
        failed = rc != 0 or "error" in text.lower()
        if not failed:
            print(text)
            return
        if net:
            state, _ = kernel_state(a.name)
            if state in ("queued", "running"):
                print(f"push reply lost to a network error, but {a.name} is {state} — done")
                return
        if ((busy or quota) and a.retry or net) and time.time() - t0 < limit:
            why = ("GPU pool full" if busy else
                   "weekly GPU quota used up" if quota else "network error")
            wait_min = (a.quota_wait if quota else a.retry) if (busy or quota) else 1
            print(f"[{time.strftime('%H:%M:%S')}] {why} — retrying in {wait_min:g} min "
                  f"({text.splitlines()[-1][:120] if text else ''})", flush=True)
            time.sleep(wait_min * 60)
            continue
        print(text)
        raise SystemExit(rc or 1)


NET_ERRORS = ("httpsconnectionpool", "nameresolution", "max retries exceeded",
              "connectionerror", "connection aborted", "connection reset",
              "timed out", "getaddrinfo", "name resolution", "remote end closed",
              "bad gateway", "service unavailable", "gateway time")


def is_net_error(text):
    low = (text or "").lower()
    return any(s in low for s in NET_ERRORS)


def kernel_state(name):
    rc, out = kaggle("kernels", "status", kernel_ref(name), check=False)
    low = out.lower()
    # A transient network failure is not a kernel state. Its traceback contains
    # "Error", which would otherwise read as a finished failed run and end a
    # watcher or a push queue during a brief outage.
    if rc != 0 and is_net_error(out):
        return "unknown", out.strip()
    # A kernel not pushed yet answers with a 404 whose text contains "error";
    # that must read as "not there yet", never as a finished failed run.
    if any(s in low for s in ("404", "not found", "forbidden", "denied")):
        return "missing", out.strip()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in low:
            return s, out.strip()
    return "unknown", out.strip()


def cmd_status(a):
    for n in a.names:
        s, raw = kernel_state(n)
        print(f"{n:<14} {s:<9} {raw.splitlines()[-1] if raw else ''}")


def cmd_wait(a):
    t0, last = time.time(), None
    while True:
        s, raw = kernel_state(a.name)
        if s != last:
            print(f"[{(time.time()-t0)/60:6.1f}m] {a.name}: {s}", flush=True)
            last = s
        if s in ("complete", "error", "cancel"):
            return
        if a.timeout and time.time() - t0 > a.timeout * 60:
            print("wait timed out; kernel still", s)
            return
        time.sleep(a.poll)


def print_log(path, tail):
    """Kaggle kernel logs are a JSON array of {stream_name, time, data}."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            rows = json.load(fh)
        text = "".join(r.get("data", "") for r in rows)
    except Exception:
        text = open(path, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    print(f"--- {os.path.basename(path)} (last {tail} of {len(lines)} lines) ---")
    print("\n".join(lines[-tail:]))


def cmd_pull(a):
    out_dir = os.path.join(RUNS, a.name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    while True:
        args = ["kernels", "output", kernel_ref(a.name), "-p", out_dir, "-o"]
        if a.pattern:                  # e.g. "*.npz" — fetch the small files first
            args += ["--file-pattern", a.pattern]
        rc, out = kaggle(*args, check=False)
        if rc == 0 and not is_net_error(out):
            break
        # Outputs stay on Kaggle, so waiting is free; a 75-min outage outlived
        # the original 30-min window, hence a generous one.
        if is_net_error(out) and time.time() - t0 < 3 * 3600:
            print(f"[{time.strftime('%H:%M:%S')}] network error pulling {a.name} "
                  f"— retrying in 1 min", flush=True)
            time.sleep(60)
            continue
        print(out)
        raise SystemExit(rc or 1)
    files = sorted(os.listdir(out_dir))
    print(f"pulled {len(files)} files -> {out_dir}")
    for f in files:
        print(f"   {f:<34} {os.path.getsize(os.path.join(out_dir, f))/1e6:8.2f} MB")
    for lf in glob.glob(os.path.join(out_dir, "*.log")):
        print_log(lf, a.tail)
    res = os.path.join(out_dir, "results.txt")
    if os.path.exists(res):
        print("--- results.txt ---\n" + open(res).read())


# Runs whose skeleton models were trained on dataset v2, where normalize_pose
# removed the root's full position (height included) inside the data itself.
# Their inference input is therefore today's v3 representation + center_z.
V2_SKELETON_RUNS = {"skel-a", "skel-b", "skel-c"}


def member_cfg(pt_path, tag):
    """
    Inference-relevant training config of one ensemble member, recovered from
    its job log (train.py prints its full Cfg as JSON at the top), so that
    infer.py preprocesses every member exactly as it was trained.
    """
    if tag.endswith("_full"):
        mod, fold = tag[:-5], "-1"
    else:
        mod, fold = tag.rsplit("_f", 1)
    run = os.path.basename(os.path.dirname(pt_path))
    cfg = {}
    log = os.path.join(os.path.dirname(pt_path), f"log_{mod}_f{fold}.txt")
    if os.path.exists(log):
        text = open(log, encoding="utf-8", errors="replace").read()
        start, depth = text.find("{"), 0
        if start >= 0:
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            cfg = json.loads(text[start:i + 1])
                        except ValueError:
                            cfg = {}
                        break
    extra = cfg.get("extra") or {}
    skel = mod.lower().startswith("skel")
    return dict(modality=mod, run=run,
                frames=int(cfg.get("frames", 32 if skel else 16)),
                size=int(cfg.get("size", 144)),
                arch=str(cfg.get("arch", "resnet18")),
                norm=extra.get("norm") or ("clip" if cfg.get("per_clip_norm", True)
                                           else "imagenet"),
                center_z=bool(extra.get("center_z")) or (skel and run in V2_SKELETON_RUNS),
                **{k: extra[k] for k in ("imu_ch", "imu_width") if k in extra})


def tag_mod(tag):
    """Data modality of a result tag: 'Depth_Color_f3' -> 'Depth_Color', 'IR_full' -> 'IR'."""
    return tag[:-5] if tag.endswith("_full") else tag.rsplit("_f", 1)[0]


def parse_streams(specs):
    """
    A spec is a run directory, or `name=dir1,dir2,...`.

    Streams are the units the fusion weighs. A bare directory contributes one
    stream per data modality it holds; a named spec pools several runs into one
    stream, so two recipes on the same modality (e.g. two pose normalisations)
    become two streams whose weights are fitted on held-out folds like any
    other input.
    """
    out = []
    for s in specs:
        if "=" in s:
            name, dirs = s.split("=", 1)
            out += [(name, d) for d in dirs.split(",") if d]
        else:
            out.append((None, s))
    return out


def cmd_fuse(a):
    import itertools
    from fnmatch import fnmatch

    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, SRC_DIR)
    from cuhkx import COMPOSITES, quantize_int8

    specs = a.runs or sorted(d for d in glob.glob(os.path.join(RUNS, "*"))
                             if os.path.isdir(d))
    files = {"_oof.npz": {}, "_test.npz": {}, ".pt": {}}      # {(stream, tag): path}
    for name, d in parse_streams(specs):
        for suffix, bucket in files.items():
            for f in glob.glob(os.path.join(d, "*" + suffix)):
                tag = os.path.basename(f)[: -len(suffix)]
                if any(tag.startswith(x) for x in a.exclude):
                    continue
                key = (name or tag_mod(tag), tag)
                if key not in bucket or os.path.getmtime(f) > os.path.getmtime(bucket[key]):
                    bucket[key] = f
    oof_f, test_f, pt_f = files["_oof.npz"], files["_test.npz"], files[".pt"]
    if not oof_f:
        raise SystemExit("no *_oof.npz in the given runs — pull a finished kernel first")

    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    C = list(range(NUM_CLASSES))
    per_stream = {}
    for (stream, tag), f in sorted(oof_f.items()):
        d = np.load(f, allow_pickle=True)
        df = pd.DataFrame(softmax(d["logits"].astype(np.float64)), columns=C)
        df["clip_id"] = d["clip_id"].astype(str)
        df["y"] = d["y"].astype(int)
        df["user"] = d["user"].astype(str)
        per_stream.setdefault(stream, []).append(df)
    oof = {s: pd.concat(v, ignore_index=True).drop_duplicates("clip_id", keep="last")
           for s, v in per_stream.items()}

    report = {"members": sorted(f"{s}:{t}" for s, t in oof_f), "streams": {}}
    for s, df in sorted(oof.items()):
        acc = float((df[C].values.argmax(1) == df.y.values).mean())
        report["streams"][s] = dict(oof_clips=int(len(df)), users=int(df.user.nunique()),
                                    oof_acc=round(acc, 4))
        print(f"{s:<16} OOF {len(df):>5} clips / {df.user.nunique():>2} users   "
              f"acc {acc:.4f}")

    mods = sorted(oof)
    common = sorted(set.intersection(*[set(oof[m].clip_id) for m in mods]))
    stack = np.stack([oof[m].set_index("clip_id").loc[common, C].values for m in mods])
    y = oof[mods[0]].set_index("clip_id").loc[common, "y"].values
    rows = np.arange(len(y))

    # Fit weights on out-of-fold predictions only.
    method = "fixed" if a.weights_json else (a.method or "loglin")
    if method == "fixed":
        # Geometric-pool weights from an earlier fusion report: swaps a better model
        # into a stream's slot without refitting on the few clips it has held-out
        # predictions for.
        given = json.load(open(a.weights_json))["weights"]
        w = np.array([float(given[m]) for m in mods])
        Lg = np.log(np.clip(stack, 1e-8, 1))
        p = softmax(np.tensordot(w, Lg, axes=1))
        fused_acc = float((p.argmax(1) == y).mean())
        neg_nll = float(np.log(np.clip(p[rows, y], 1e-12, 1)).mean())
    elif method == "grid":
        # Exhaustive simplex grid (11^N points at step 0.1): exact for few
        # streams. Ties on accuracy go to the better-calibrated mix.
        best = None
        grid = np.round(np.arange(0, 1 + 1e-9, a.step), 4)
        for w in itertools.product(grid, repeat=len(mods)):
            tot = float(sum(w))
            if tot <= 0:
                continue
            w = np.asarray(w) / tot
            p = np.tensordot(w, stack, axes=1)
            acc = float((p.argmax(1) == y).mean())
            nll = float(-np.log(np.clip(p[rows, y], 1e-9, 1)).mean())
            key = (round(acc, 6), -nll)
            if best is None or key > best[0]:
                best = (key, w)
        (fused_acc, neg_nll), w = best
    elif method == "loglin":
        # Log-linear (geometric) pool, p ∝ Π_s p_s^w_s: streams that agree
        # reinforce each other and a weak stream can still veto classes it rules
        # out. w >= 0 maximises the out-of-fold log-likelihood (concave in w).
        # Nested user-grouped CV on the Day-2 streams: 0.615 vs 0.593 for the
        # linear mixture below, better on all five folds.
        Lg = np.log(np.clip(stack, 1e-8, 1))
        w = np.full(len(mods), 1.0 / len(mods))
        m1 = m2 = np.zeros_like(w)
        for t in range(1, 3001):                               # Adam, projected to w >= 0
            q = softmax(np.tensordot(w, Lg, axes=1))
            g = (Lg[:, rows, y] - (q[None] * Lg).sum(2)).mean(1) - 1e-3 * w
            m1 = 0.9 * m1 + 0.1 * g
            m2 = 0.999 * m2 + 0.001 * g * g
            w = np.maximum(0, w + 0.05 * (m1 / (1 - 0.9 ** t))
                           / (np.sqrt(m2 / (1 - 0.999 ** t)) + 1e-8))
        p = softmax(np.tensordot(w, Lg, axes=1))
        fused_acc = float((p.argmax(1) == y).mean())
        neg_nll = float(np.log(np.clip(p[rows, y], 1e-12, 1)).mean())
    else:
        # EM for mixture weights: maximises the out-of-fold log-likelihood of
        # the weighted mixture. The objective is concave in the weights, so
        # this reaches the global optimum in seconds for any number of
        # streams, and fits noise less than an accuracy search does.
        P = np.clip(stack[:, rows, y], 1e-12, 1)              # [M, N]
        w = np.full(len(mods), 1.0 / len(mods))
        for _ in range(5000):
            r = w[:, None] * P
            r /= r.sum(0, keepdims=True)
            w_new = r.mean(1)
            done = np.abs(w_new - w).max() < 1e-9
            w = w_new
            if done:
                break
        p = np.tensordot(w, stack, axes=1)
        fused_acc = float((p.argmax(1) == y).mean())
        neg_nll = float(np.log(np.clip(p[rows, y], 1e-12, 1)).mean())
    print(f"weight fit: {method}")
    print(f"\nfusion on {len(common)} clips shared by {mods}")
    print(f"weights {dict(zip(mods, np.round(w, 2).tolist()))}")
    print(f"FUSED OOF cross-subject accuracy {fused_acc:.4f}  (NLL {-neg_nll:.4f})")
    report.update(weights=dict(zip(mods, [round(float(x), 4) for x in w])),
                  fused_oof_acc=fused_acc, fused_clips=len(common), fusion=method)
    loglin = method in ("loglin", "fixed")

    def combine(w, arr):
        """Fused probabilities of stream probabilities arr [M,N,C] under the fitted rule."""
        if loglin:
            return softmax(np.tensordot(w, np.log(np.clip(arr, 1e-8, 1)), axes=1))
        return np.tensordot(w, arr, axes=1)

    names = {}
    cm = _first(os.path.join(EXTRACTED, "class_mapping.csv"))
    if cm:
        m_ = pd.read_csv(cm)
        names = dict(zip(m_.action_id, m_.action_name))
    pred = combine(w, stack).argmax(1)
    worst = []
    for c in range(NUM_CLASSES):
        sel = y == c
        if not sel.any():
            continue
        wrong = pd.Series(pred[sel][pred[sel] != c]).value_counts()
        worst.append((float((pred[sel] == c).mean()), names.get(c, c), int(sel.sum()),
                      names.get(int(wrong.index[0]), "-") if len(wrong) else "-"))
    worst.sort()
    print("\nweakest classes (recall, class, n, most-confused-with):")
    for r in worst[:10]:
        print(f"   {r[0]:.2f}  {r[1]:<32} n={r[2]:<4} -> {r[3]}")
    report["weakest"] = worst[:10]

    # ---- test: members of a stream are averaged, streams are weighted
    tests = {}
    for (stream, tag), f in test_f.items():
        # --test-as-packed: average only the members the checkpoint will hold,
        # so the leaderboard scores what the package predicts
        if stream in mods and (not a.test_as_packed or not a.pack_only or
                               any(fnmatch(f"{stream}:{tag}", p) for p in a.pack_only)):
            d = np.load(f, allow_pickle=True)
            tests.setdefault(stream, []).append(
                pd.DataFrame(d["probs"], index=d["clip_id"].astype(str)))
    tmpl = pd.read_csv(_first(os.path.join(EXTRACTED, "test.csv")))
    ids = [p.strip("/").split("/")[-1] for p in tmpl.path]
    avg = {m: (sum(x.reindex(ids) for x in v) / len(v)) for m, v in tests.items()}

    # A stream missing for a clip (no Thermal on 10 test clips, no IR on 4) is
    # left out of that clip's fusion — for the geometric pool this is the same
    # as that stream giving a uniform distribution.
    num = np.zeros((len(ids), NUM_CLASSES))
    den = np.zeros(len(ids))
    for wi, m in zip(w, mods):
        if m not in avg or wi <= 0:
            continue
        v = avg[m]
        ok = ~v.isna().any(axis=1).values
        num[ok] += wi * (np.log(np.clip(v.values[ok], 1e-8, 1)) if loglin else v.values[ok])
        den[ok] += wi
    final = np.zeros_like(num)
    has = den > 0
    final[has] = softmax(num[has]) if loglin else num[has] / den[has, None]
    # A clip covered only by zero-weighted streams still needs a prediction.
    orphan = ~has
    if orphan.any():
        on, od = np.zeros_like(num), np.zeros(len(ids))
        for m, v in avg.items():
            ok = orphan & ~v.isna().any(axis=1).values
            on[ok] += v.values[ok]
            od[ok] += 1
        if (orphan & (od == 0)).any():
            raise SystemExit(f"{int((orphan & (od == 0)).sum())} test clips have no prediction")
        final[orphan] = on[orphan] / od[orphan, None]
    prior = None
    if a.balance_steps:
        # Label-free test-time rebalancing: rescale each class so the predicted class
        # mix moves towards the training class mix. infer.py repeats it from the
        # prior stored in the package.
        idx_pc = pd.read_parquet(os.path.join(COMPACT, "Depth_Color_PC", "index.parquet"))
        tr_pc = idx_pc[idx_pc.split == "train"]
        prior = np.bincount(tr_pc.action_id.astype(int), minlength=NUM_CLASSES) + 1.0
        prior = prior / prior.sum()
        before = final.argmax(1)
        for _ in range(a.balance_steps):
            final = final / (final.sum(0) / (len(final) * prior))[None]
            final = final / final.sum(1, keepdims=True)
        print(f"class rebalancing ({a.balance_steps} step): {int((final.argmax(1) != before).sum())} "
              f"predictions changed")
        report.update(balance_steps=a.balance_steps)
    sub = pd.DataFrame({"path": tmpl.path, "prediction": final.argmax(1).astype(int)})

    assert len(sub) == 405 and sub.prediction.between(0, 39).all()
    os.makedirs(RUNS, exist_ok=True)
    sub_path = os.path.join(RUNS, "submission.csv")
    sub.to_csv(sub_path, index=False)
    np.save(os.path.join(RUNS, "test_probs.npy"), final)
    print(f"\nwrote {sub_path}: {len(sub)} rows, {sub.prediction.nunique()}/40 classes "
          f"predicted, orphans filled {int(orphan.sum())}")
    report["test_coverage"] = {m: int((~v.isna().any(axis=1)).sum()) for m, v in avg.items()}

    live = {m for m, wi in zip(mods, w) if wi > 0}
    used = sorted(k for k in pt_f if k[0] in live and
                  (not a.pack_only or any(fnmatch(f"{k[0]}:{k[1]}", p) for p in a.pack_only)))
    if used and not a.no_pack:
        # One member at a time: fp16, or per-channel int8 for the large video
        # nets (R(2+1)D-34 is 127 MB in fp16, ~64 MB in int8).
        sds, members = {}, {}
        for s, t in used:
            key = f"{s}:{t}"
            sd = torch.load(pt_f[(s, t)], map_location="cpu")
            # an --int8 pattern may carry its own width: "r2p:*=6"
            bits = next((int(p.split("=")[1]) if "=" in p else a.bits for p in a.int8
                         if fnmatch(key, p.split("=")[0])), 16)
            int8 = bits < 16
            sds[key] = (quantize_int8(sd, bits=bits) if int8 else
                        {k: v.half() if v.is_floating_point() else v for k, v in sd.items()})
            members[key] = dict(member_cfg(pt_f[(s, t)], t), stream=s, int8=int8, bits=bits)
        ckpt = {"models": sds, "meta": dict(weights=report["weights"], fused_oof_acc=fused_acc,
                                            members=members,
                                            fusion="loglin" if loglin else "linear")}
        if prior is not None:
            ckpt["meta"].update(class_prior=prior.tolist(), balance_steps=a.balance_steps)
        # Person-crop members need the detector that made their crops, unconverted.
        if any(p.endswith("_PC") for m in members.values()
               for p, _ in COMPOSITES.get(m["modality"], ((m["modality"], 3),))):
            hub = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
            det = torch.load(glob.glob(os.path.join(hub, "ssdlite320*.pth"))[0],
                             map_location="cpu")
            # fp16 halves the detector (13.4 -> 6.7 MB). On the 401 test clips its
            # boxes equal the fp32 ones for 348 and are within 2 px for 395.
            ckpt["detector"] = ({k: v.half() if v.is_floating_point() else v
                                 for k, v in det.items()} if a.detector_fp16 else det)
        path = os.path.join(RUNS, "model.pth")
        torch.save(ckpt, path)
        mb = os.path.getsize(path) / 1e6
        print(f"checkpoint: {path}  {mb:.1f} MB  ({len(sds)} models"
              f"{', int8: ' + str(sum(m['int8'] for m in members.values())) if a.int8 else ''}"
              f"{', + detector' if 'detector' in ckpt else ''})")
        if mb >= 100:
            print("!! OVER the 100 MB limit — pack fewer members or quantise more (--int8)")
        report["members_packed"] = members
        report["checkpoint_mb"] = round(mb, 1)
    with open(os.path.join(RUNS, "fusion_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)


# ------------------------------------------------------------ competition

def _csv_rows(text):
    lines = text.splitlines()
    # CLI 2.x header is "ref,fileName,date,...", so match the column, not the prefix
    start = next((i for i, l in enumerate(lines) if "fileName" in l.split(",")), None)
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def cmd_submit(a):
    import pandas as pd
    sub = pd.read_csv(a.file)
    tmpl = pd.read_csv(_first(os.path.join(EXTRACTED, "test.csv")))
    if list(sub.columns) != ["path", "prediction"] or len(sub) != len(tmpl) \
            or list(sub.path) != list(tmpl.path) \
            or not sub.prediction.between(0, 39).all():
        raise SystemExit("submission does not match the test.csv template — refusing")
    rc, out = kaggle("competitions", "submit", COMP, "-f", a.file, "-m", a.message,
                     check=False)
    print(out.strip())
    if rc != 0:
        raise SystemExit(rc)
    for _ in range(40):
        time.sleep(15)
        _, out = kaggle("competitions", "submissions", COMP, "-v", check=False)
        rows = _csv_rows(out)
        if rows:
            r = rows[0]
            print(f"   {r.get('status')}  public={r.get('publicScore')}  "
                  f"{r.get('description')}", flush=True)
            if any(s in str(r.get("status", "")).lower() for s in ("complete", "error")):
                return


def cmd_subs(a):
    _, out = kaggle("competitions", "submissions", COMP, check=False)
    print(out.strip())


def cmd_lb(a):
    _, out = kaggle("competitions", "leaderboard", COMP, "-s", check=False)
    print("\n".join(l for l in out.splitlines() if not l.startswith("Next Page")))


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("stage"); p.add_argument("names", nargs="+", choices=list(DATASETS))
    p.set_defaults(fn=cmd_stage)
    p = sp.add_parser("upload"); p.add_argument("names", nargs="+", choices=list(DATASETS))
    p.add_argument("-m", "--message", default="update"); p.set_defaults(fn=cmd_upload)
    p = sp.add_parser("dsstatus"); p.add_argument("names", nargs="*")
    p.add_argument("--wait", type=float, default=0, help="minutes to wait for ready")
    p.set_defaults(fn=cmd_dsstatus)

    p = sp.add_parser("kernel"); p.add_argument("name")
    p.add_argument("--datasets", nargs="+", default=["meta"], choices=list(DATASETS))
    p.add_argument("--jobs", nargs="*")
    p.add_argument("--envcheck", action="store_true")
    p.add_argument("--budget-h", type=float, default=8.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--template", default="runner_template.py",
                   help="kernel script template in kaggle/")
    p.add_argument("--embed", nargs="*",
                   help="kaggle/ modules to embed (default: cuhkx.py train.py)")
    p.add_argument("--kernel-sources", nargs="*", default=[],
                   help="kernels whose outputs to attach, e.g. r2p-a (= OWNER/cuhkx-r2p-a)")
    p.set_defaults(fn=cmd_kernel)
    p = sp.add_parser("push"); p.add_argument("name")
    p.add_argument("--accelerator")
    p.add_argument("--retry", type=float, default=0,
                   help="minutes between retries while the GPU pool is full")
    p.add_argument("--retry-hours", type=float, default=12)
    p.add_argument("--quota-wait", type=float, default=10,
                   help="with --retry: minutes between retries while the weekly "
                        "GPU quota is used up (the push starts once it resets)")
    p.set_defaults(fn=cmd_push)
    p = sp.add_parser("status"); p.add_argument("names", nargs="+")
    p.set_defaults(fn=cmd_status)
    p = sp.add_parser("wait"); p.add_argument("name")
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--timeout", type=float, default=0, help="minutes")
    p.set_defaults(fn=cmd_wait)
    p = sp.add_parser("pull"); p.add_argument("name")
    p.add_argument("--tail", type=int, default=40)
    p.add_argument("--pattern", help="download only output files matching this pattern")
    p.set_defaults(fn=cmd_pull)

    p = sp.add_parser("fuse"); p.add_argument("runs", nargs="*")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--method", choices=["loglin", "em", "grid"],
                   help="stream fusion: loglin = geometric pool (default), em / grid = "
                        "linear mixture with EM-fitted / grid-searched weights")
    p.add_argument("--no-pack", action="store_true",
                   help="write the submission only; skip packing model.pth")
    p.add_argument("--pack-only", nargs="*", default=[],
                   help="fnmatch patterns of stream:tag members to pack "
                        "(default: every member of a weighted stream)")
    p.add_argument("--int8", nargs="*", default=[],
                   help="fnmatch patterns of members stored as per-channel integers")
    p.add_argument("--bits", type=int, default=8, choices=[4, 5, 6, 7, 8],
                   help="bit width for the --int8 members (below 8 is bit-packed)")
    p.add_argument("--weights-json", help="use this fusion report's weights instead of fitting")
    p.add_argument("--balance-steps", type=int, default=0,
                   help="test-time class rebalancing steps towards the training class mix")
    p.add_argument("--detector-fp16", action="store_true",
                   help="store the person detector in fp16 (6.7 MB instead of 13.4)")
    p.add_argument("--test-as-packed", action="store_true",
                   help="submission averages only the --pack-only members' test predictions")
    p.set_defaults(fn=cmd_fuse)
    p = sp.add_parser("submit"); p.add_argument("file")
    p.add_argument("-m", "--message", required=True); p.set_defaults(fn=cmd_submit)
    p = sp.add_parser("subs"); p.set_defaults(fn=cmd_subs)
    p = sp.add_parser("lb"); p.set_defaults(fn=cmd_lb)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
