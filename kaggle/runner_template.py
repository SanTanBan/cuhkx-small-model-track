"""
CUHK-X Kaggle job runner.

Generated per kernel by scripts/kaggle_ops.py, which replaces the four marked
lines below. Writes the embedded library, links every attached cuhkx-* dataset
into one data root, optionally benchmarks the hardware, then runs the queued
(modality, fold) jobs inside a wall-clock budget so that a Kaggle session
timeout can never throw away work that already finished.
"""
import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import time

JOBS = []          # @@JOBS@@
SRC = {}           # @@SRC@@
BUDGET_H = 8.0     # @@BUDGET@@
ENVCHECK = False   # @@ENVCHECK@@

T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
# Symlinks go outside /kaggle/working so Kaggle does not sweep the datasets
# into the saved kernel output.
DATA = "/tmp/cuhkx"
os.chdir(WORK)


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f}m]", *a, flush=True)


def sh(cmd):
    log("$", cmd)
    subprocess.run(cmd, shell=True)


# ---------------------------------------------------------------- library
for name, b64 in SRC.items():
    with open(os.path.join(WORK, name), "wb") as f:
        f.write(base64.b64decode(b64))
sys.path.insert(0, WORK)

# ------------------------------------------------------------ environment
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
import torch  # noqa: E402
import torchvision  # noqa: E402

log("torch", torch.__version__, "| torchvision", torchvision.__version__,
    "| cuda", torch.cuda.is_available(), "| cpus", os.cpu_count())


def _gpu_supported():
    """Same major compute capability, minor <= the device's, has a cubin."""
    maj, mnr = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    for a in arches:
        digits = "".join(c for c in a[3:] if c.isdigit()) if a.startswith("sm_") else ""
        if len(digits) >= 2 and int(digits[:-1]) == maj and int(digits[-1]) <= mnr:
            return True, maj, mnr, arches
    return not arches, maj, mnr, arches


if torch.cuda.is_available():
    ok, maj, mnr, arches = _gpu_supported()
    log("gpu", torch.cuda.get_device_name(0), f"sm_{maj}{mnr}", "| build", " ".join(arches))
    if not ok:
        # e.g. a P100 (sm_60) under a cu128 build that starts at sm_70: every
        # CUDA op would fail with "no kernel image". Stop with the reason.
        raise SystemExit(f"unsupported GPU sm_{maj}{mnr} for this PyTorch build {arches}; "
                         "re-push with --accelerator NvidiaTeslaT4")

# --------------------------------------------------------------- datasets
os.makedirs(DATA, exist_ok=True)
found = {}
for mj in sorted(glob.glob("/kaggle/input/**/modality.json", recursive=True)):
    src = os.path.dirname(mj)
    mod = json.load(open(mj))["modality"]
    if mod == "_meta":
        for f in os.listdir(src):
            p = os.path.join(src, f)
            if f.endswith(".csv"):
                shutil.copy2(p, os.path.join(DATA, f))
            elif f.endswith(".pth"):
                # Pre-seed the torchvision cache so ImageNet weights never
                # depend on the kernel's internet access.
                hub = os.path.join(torch.hub.get_dir(), "checkpoints")
                os.makedirs(hub, exist_ok=True)
                if not os.path.exists(os.path.join(hub, f)):
                    shutil.copy2(p, os.path.join(hub, f))
        found["_meta"] = src
        continue
    dst = os.path.join(DATA, mod)
    if not os.path.lexists(dst):
        os.symlink(src, dst)
    found[mod] = src

for k, v in found.items():
    log(f"dataset {k:<12} <- {v}  ({len(os.listdir(v))} files)")
if not found:
    log("!! no cuhkx datasets found under /kaggle/input")
    sh("find /kaggle/input -maxdepth 4 | head -60")


# -------------------------------------------------------------- benchmark
def benchmark():
    """Measure data-loading and GPU throughput per attached modality."""
    from torch.utils.data import DataLoader
    import cuhkx

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    report = {}
    for mod in sorted(m for m in found if m != "_meta"):
        img = not mod.lower().startswith("skel")
        cfg = cuhkx.Cfg(data_root=DATA, modality=mod,
                        frames=16 if img else 32, size=144)
        mdir, idx = cuhkx.load_index(DATA, mod)
        ds = cuhkx.make_dataset(idx, mdir, cfg, train=True)
        bs = 16 if img else 32
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=3,
                        drop_last=True, pin_memory=True)

        t, n = time.time(), 0
        for i, (x, _) in enumerate(dl):
            n += x.size(0)
            if i == 12:
                break
        data_cps = n / max(time.time() - t, 1e-6)

        model = cuhkx.make_model(cfg, pretrained=img).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
        xb, _ = next(iter(dl))
        xb = xb.to(dev)
        yb = torch.randint(0, 40, (xb.size(0),), device=dev)
        for i in range(25):
            if i == 5:
                if cuda:
                    torch.cuda.synchronize()
                t = time.time()
            with torch.autocast("cuda", enabled=cuda):
                loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if cuda:
            torch.cuda.synchronize()
        gpu_cps = 20 * xb.size(0) / max(time.time() - t, 1e-6)
        mem = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0

        report[mod] = dict(clips=int(len(idx)), data_clips_per_s=round(data_cps, 1),
                           gpu_clips_per_s=round(gpu_cps, 1),
                           bottleneck="data" if data_cps < gpu_cps else "gpu",
                           peak_mem_gb=round(mem, 2))
        log(f"BENCH {mod:<12} data {data_cps:7.1f} clips/s | "
            f"gpu {gpu_cps:7.1f} clips/s | peak {mem:.2f} GB")
        del model, opt
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    with open(os.path.join(WORK, "benchmark.json"), "w") as f:
        json.dump(report, f, indent=2)


# ------------------------------------------------------------------- jobs
def _attached(mod):
    """A composite input needs every part's dataset."""
    import cuhkx
    parts = [p for p, _ in cuhkx.COMPOSITES.get(mod, ())] or [mod]
    return all(p in found for p in parts)


def _pump(p, lf, prefix):
    for line in p.stdout:
        lf.write(line)
        print(prefix + line, end="", flush=True)
    lf.close()


def run_jobs():
    """
    Kaggle's T4 session has two GPUs and a job uses one, so jobs run one per
    GPU, started in list order as GPUs free up. Each job's training cap is the
    budget left when it starts, less the time to write its predictions.
    """
    import threading
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    slots = list(range(max(ngpu, 1)))
    queue, running, results = list(JOBS), {}, []     # running: slot -> (proc, job, t0, thread)
    log(f"{len(queue)} jobs on {len(slots)} {'GPU' if ngpu else 'CPU'} slot(s)")

    def save():
        with open(os.path.join(WORK, "runner_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    while queue or running:
        for s in [s for s, r in running.items() if r[0].poll() is not None]:
            p, job, t, th = running.pop(s)
            th.join()
            mins = (time.time() - t) / 60
            results.append(dict(modality=job["modality"], fold=job["fold"],
                                rc=p.returncode, minutes=round(mins, 1), gpu=s))
            log(f"DONE {job['modality']} f{job['fold']} rc={p.returncode} "
                f"in {mins:.1f} min (gpu {s})")
            save()
        free = [s for s in slots if s not in running]
        while queue and free:
            job = queue.pop(0)
            mod, fold = job["modality"], job["fold"]
            left = BUDGET_H * 60 - (time.time() - T0) / 60
            if left < 20:
                log(f"SKIP {mod} f{fold}: {left:.0f} min of budget left")
                results.append(dict(modality=mod, fold=fold, rc=None, skipped="budget"))
                continue
            if not _attached(mod):
                log(f"SKIP {mod} f{fold}: dataset not attached")
                results.append(dict(modality=mod, fold=fold, rc=None, skipped="no data"))
                continue

            # Stop training early enough to still write OOF + test predictions.
            cap = left - 12
            if job.get("cap_min"):
                cap = min(cap, job["cap_min"])
            s = free.pop(0)
            cmd = [sys.executable, "train.py", "--data-root", DATA, "--modality", mod,
                   "--fold", str(fold), "--out-dir", WORK,
                   "--max-minutes", f"{cap:.0f}"] + list(job["args"])
            env = dict(os.environ)
            if ngpu:
                env["CUDA_VISIBLE_DEVICES"] = str(s)
            log(f"RUN gpu {s}:", " ".join(cmd[1:]))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, env=env)
            lf = open(os.path.join(WORK, f"log_{mod}_f{fold}.txt"), "w")
            th = threading.Thread(target=_pump, daemon=True,
                                  args=(p, lf, f"[g{s}] " if len(slots) > 1 else ""))
            th.start()
            running[s] = (p, job, time.time(), th)
        time.sleep(5)
    save()
    return results


if ENVCHECK:
    benchmark()
if JOBS:
    import cuhkx
    for arch in sorted({a.split("=", 1)[1] for j in JOBS for a in j["args"]
                        if a.startswith("--arch=")}):
        log("prefetch", arch)
        cuhkx.prefetch(arch)
    run_jobs()
log("runner finished")
