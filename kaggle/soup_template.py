"""
CUHK-X model-soup check, run as a Kaggle kernel; kaggle/soup.py does the work.

    python scripts/kaggle_ops.py kernel soup --template soup_template.py \
        --embed cuhkx.py soup.py [soup_phases.json] --datasets depthpc irpc \
        --kernel-sources r2p-a r2p-b r2p-pl r2p-r2

Writes the embedded modules, links the attached cuhkx-* datasets into one data
root, and runs two phases of soup.py on one GPU each: A and B by default, or the
pair listed in an embedded soup_phases.json (e.g. ["C", "D"]). The checkpoints
come from the attached training kernels' outputs.
"""
import base64
import glob
import json
import os
import subprocess
import sys
import threading
import time

SRC = {}           # @@SRC@@

T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
DATA = "/tmp/cuhkx"
os.chdir(WORK)


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f}m]", *a, flush=True)


for name, b64 in SRC.items():
    with open(os.path.join(WORK, name), "wb") as f:
        f.write(base64.b64decode(b64))
PHASES = ["A", "B"]
if os.path.exists(os.path.join(WORK, "soup_phases.json")):
    with open(os.path.join(WORK, "soup_phases.json")) as f:
        PHASES = json.load(f)
log("phases", PHASES)

subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
import torch  # noqa: E402

os.makedirs(DATA, exist_ok=True)
for mj in sorted(glob.glob("/kaggle/input/**/modality.json", recursive=True)):
    mod = json.load(open(mj))["modality"]
    dst = os.path.join(DATA, mod)
    if mod != "_meta" and not os.path.lexists(dst):
        os.symlink(os.path.dirname(mj), dst)
        log(f"dataset {mod:<16} <- {os.path.dirname(mj)}")
for p in sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True)):
    log("checkpoint", p)


def run(phase, gpu):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, "soup.py", "--phase", phase, "--data-root", DATA,
           "--ckpt-root", "/kaggle/input", "--out-dir", WORK]
    log(f"RUN gpu {gpu}:", " ".join(cmd[1:]))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    def pump():
        with open(os.path.join(WORK, f"log_soup{phase}.txt"), "w") as lf:
            for line in p.stdout:
                lf.write(line)
                print(f"[{phase}] " + line, end="", flush=True)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    return phase, p, th


ngpu = torch.cuda.device_count()
codes = {}
if ngpu >= 2:
    for phase, p, th in [run(ph, i) for i, ph in enumerate(PHASES[:2])]:
        codes[phase] = p.wait()
        th.join()
else:
    for ph in reversed(PHASES):    # the later phase holds what a submission needs
        phase, p, th = run(ph, 0 if ngpu else None)
        codes[phase] = p.wait()
        th.join()
with open(os.path.join(WORK, "soup_runner.json"), "w") as f:
    json.dump(codes, f)
log("soup check finished", codes)
