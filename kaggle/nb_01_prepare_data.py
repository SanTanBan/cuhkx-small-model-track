# %% [markdown]
# # CUHK-X Small Model Track — Step 1: build the compact dataset
#
# Runs entirely on Kaggle, so your own connection never carries the 44 GB.
# Downloads the source from HuggingFace at Kaggle's bandwidth, extracts **one
# modality at a time** (peak disk stays under the ~73 GB limit), packs
# uniformly-sampled JPEG frames into blob shards plus a 3D-pose array, and
# writes a few GB to `/kaggle/working`.
#
# **Before running:**
# 1. Accept the terms at
#    https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track
# 2. Add-ons → Secrets → add `HF_TOKEN` (a HuggingFace **read** token) and
#    attach it to this notebook.
# 3. Settings → Accelerator **None** (this step is I/O bound — save GPU quota),
#    Internet **On**, Persistence **Files**.
#
# Then **Save Version → Save & Run All (Commit)** and let it run unattended.

# %%
import os, sys, subprocess, shutil, time, json, glob

T0 = time.time()

def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")

def disk():
    t, u, f = shutil.disk_usage("/kaggle")
    return f"disk: {u/1e9:.1f} GB used / {f/1e9:.1f} GB free"

def tsize(p):
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(p) for f in fs) / 1e9

print(disk())

# %% [markdown]
# ## Config
#
# Measured from the real test set: Depth_Color and IR are 640x480 and present
# for 100% of clips; Thermal is 320x240 at 97.5%; Skeleton is 3D pose present
# for 100%; **Radar is empty for 207/405 clips so it is excluded**. Clips are
# short — median 20 depth frames — so 16 sampled frames covers most of them.

# %%
MODALITIES  = ["Skeleton", "Depth_Color", "IR", "Thermal"]
FRAMES      = 16     # image frames per clip
SKEL_FRAMES = 32     # pose is cheap, so sample it denser
SIZE        = 160    # stored; training random-crops to 144
QUALITY     = 90
CROP        = True   # foreground person crop

WORK    = "/kaggle/working"
SCRATCH = "/kaggle/temp"          # not persisted — holds the 44 GB of archives
RAW     = f"{SCRATCH}/raw"
EXTRACT = f"{SCRATCH}/extracted"
OUT     = f"{WORK}/compact"
for d in (RAW, EXTRACT, OUT):
    os.makedirs(d, exist_ok=True)

sh("pip install -q huggingface_hub hf_transfer pyarrow 2>&1 | tail -2", check=False)
sh("apt-get -qq install -y p7zip-full 2>&1 | tail -2", check=False)

# %% [markdown]
# ## Preprocessing module
#
# Identical code to the local `kaggle/prep.py`, so a local run and this notebook
# produce the same shards.

# %%
#!writefile prep.py

# %% [markdown]
# ## Download

# %%
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

REPO = "Kevin-Pal/CUHK-X_Small_Model_Track"
t = time.time()
snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=RAW,
                  allow_patterns=["Small-Model-Track/**"], max_workers=8,
                  token=os.environ["HF_TOKEN"])
print(f"\ndownloaded {tsize(RAW):.1f} GB in {(time.time()-t)/60:.1f} min\n{disk()}")

SRC      = f"{RAW}/Small-Model-Track"
HAR_ZIP  = f"{SRC}/Training/data/HAR.zip"      # last volume of HAR.z01..z08
TEST_ZIP = f"{SRC}/Testing/data/small_model_track_test.zip"
sh(f"ls -la {SRC}/Training/data {SRC}/Testing/data")

# %% [markdown]
# ## Extract the test set and confirm the layout

# %%
sh(f"7z x '{TEST_ZIP}' -o'{EXTRACT}' -y -bso0 -bsp0")

import prep
train_root, test_root = prep.find_roots(EXTRACT)
print(f"train_root: {train_root}\ntest_root : {test_root}")

clips = sorted(d for d in os.listdir(test_root) if d.startswith("SM_test"))
print(f"test clips: {len(clips)}   (expected 405)")

from collections import Counter
pres = Counter()
for c in clips:
    for m in os.listdir(os.path.join(test_root, c)):
        if os.path.isdir(os.path.join(test_root, c, m)):
            pres[m] += 1
for m, n in pres.most_common():
    print(f"  {m:<14} {n:>4}/{len(clips)}  ({100*n/len(clips):5.1f}%)")

# %% [markdown]
# ## Extract → pack → delete, one modality at a time
#
# Selective extraction (`-ir!`) is what keeps peak disk inside Kaggle's budget:
# the archives stay on disk but only one modality is ever unpacked at a time.

# %%
WORKERS = os.cpu_count() or 4
summary = {}

for mod in MODALITIES:
    print(f"\n{'='*70}\n{mod}\n{'='*70}", flush=True)
    mdir = f"{EXTRACT}/HAR/data/{mod}"

    if not os.path.isdir(mdir):
        t = time.time()
        sh(f"7z x '{HAR_ZIP}' -o'{EXTRACT}' -y -bso0 -bsp0 -ir'!HAR/data/{mod}/*'",
           check=False)
        print(f"  extracted in {(time.time()-t)/60:.1f} min  |  {disk()}")
    if not os.path.isdir(mdir):
        print(f"  !! {mdir} missing — inspect the archive layout:")
        sh(f"7z l '{HAR_ZIP}' -ba | head -20", check=False)
        continue

    train_root, test_root = prep.find_roots(EXTRACT)
    recs = prep.discover_train(train_root, mod) + prep.discover_test(test_root, mod)
    ntr = sum(r["split"] == "train" for r in recs)
    print(f"  clips: {len(recs)}  ({ntr} train / {len(recs)-ntr} test)", flush=True)
    if not recs:
        continue

    if mod == "Skeleton":
        df = prep.pack_skeleton(recs, OUT, SKEL_FRAMES, WORKERS)
    else:
        df = prep.pack_images(mod, recs, OUT, FRAMES, SIZE, CROP, QUALITY, WORKERS)

    if len(df):
        trn = df[df.split == "train"]
        summary[mod] = dict(clips=int(len(df)), train=int(len(trn)),
                            test=int((df.split == "test").sum()),
                            users=sorted(trn.user.astype(str).unique()),
                            classes=int(trn.action_id.nunique()))
        print(f"  users: {summary[mod]['users']}")
        print(f"  classes covered: {summary[mod]['classes']}/40")

    shutil.rmtree(mdir, ignore_errors=True)     # reclaim before the next modality
    print(f"  freed {mod}  |  {disk()}", flush=True)

# %% [markdown]
# ## Ship the label map and submission template

# %%
for src, dst in [(f"{SRC}/class_mapping.csv", "class_mapping.csv"),
                 (f"{SRC}/Testing/test_file/test.csv", "test.csv"),
                 (f"{SRC}/Testing/test_file/sample_submission.csv",
                  "sample_submission.csv")]:
    if os.path.exists(src):
        shutil.copy2(src, f"{OUT}/{dst}")

with open(f"{OUT}/prep_config.json", "w") as fh:
    json.dump(dict(modalities=MODALITIES, frames=FRAMES, skel_frames=SKEL_FRAMES,
                   size=SIZE, quality=QUALITY, crop=CROP, summary=summary),
              fh, indent=2)

print(json.dumps(summary, indent=2)[:2500])
print(f"\ncompact output: {tsize(OUT):.2f} GB")
print(f"total runtime : {(time.time()-T0)/60:.1f} min")
sh(f"du -sh {OUT}/* | sort -h", check=False)

# %% [markdown]
# ## Check before moving on
#
# * **train users must be 1–9 and 16–24**, and none of 10, 11, 25, 26 —
#   the test subjects. If a test user shows up in training, the split is wrong.
# * every modality should cover **40/40 classes**
# * test counts should be 405 (395 for Thermal, which is genuinely missing on
#   10 clips)
#
# Then `Save Version → Save & Run All (Commit)`; attach the output as the input
# dataset for Step 2.
