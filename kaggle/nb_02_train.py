# %% [markdown]
# # CUHK-X Small Model Track — Step 2: train
#
# Trains TSM-ResNet18 per modality with **GroupKFold on user**, so validation
# measures the thing the competition actually scores: generalisation to people
# the model has never seen.
#
# **Setup:** attach the `compact` dataset produced by Step 1. Accelerator:
# **GPU P100** (or T4 x2). Internet: On (for the ImageNet weights).
#
# Budget guide on a P100 — roughly 6–10 min/epoch for ~15k clips at 16x144.
# One modality x 20 epochs ≈ 2–3 h. You have 30 GPU-h/week, so plan the order:
# **Thermal first** (strongest single modality in the source paper: 92.6%),
# then Depth_Color, then IR.

# %%
import os, sys, json, time, glob, subprocess
DATA = "/kaggle/input/cuhkx-compact/compact"     # <- adjust to your dataset path
if not os.path.isdir(DATA):
    cands = glob.glob("/kaggle/input/*/compact") + glob.glob("/kaggle/input/*")
    print("compact/ not at the default path. Candidates:")
    for c in cands[:20]: print("  ", c)
    DATA = cands[0] if cands else DATA
print("DATA =", DATA)
print(sorted(os.listdir(DATA))[:10])
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)

# %% [markdown]
# ## Library

# %%
#!writefile cuhkx.py

# %%
#!writefile train.py

# %%
import importlib, cuhkx
importlib.reload(cuhkx)
import pandas as pd, numpy as np, torch
from cuhkx import load_index, build_folds

# Sanity-check the split before spending GPU hours on it.
mod0 = json.load(open(f"{DATA}/prep_config.json"))["modalities"][0] \
       if os.path.exists(f"{DATA}/prep_config.json") else "Thermal"
_, idx = load_index(DATA, mod0)
tr = idx[idx.split == "train"]
print(f"{mod0}: {len(idx)} clips  ({len(tr)} train, {(idx.split=='test').sum()} test)")
print(f"train users ({tr.user.nunique()}): {sorted(tr.user.astype(str).unique())}")
print(f"\nclass balance: min={tr.action_id.value_counts().min()} "
      f"max={tr.action_id.value_counts().max()} over "
      f"{tr.action_id.nunique()} classes")

f = build_folds(tr, 5, 42)
print("\nfold -> held-out users (these must be disjoint from training users):")
for k in range(5):
    u = sorted(tr[f == k].user.astype(str).unique())
    print(f"  fold {k}: {len(tr[f==k]):>5} clips  users {u}")

# %% [markdown]
# ## Train
#
# Run one cell per (modality, fold). Each writes `{mod}_f{fold}.pt`,
# `_oof.npz` and `_test.npz` into `/kaggle/working`.
#
# Start with a single fold to measure the real cross-subject accuracy and the
# per-epoch time before committing GPU quota to the rest.

# %% [markdown]
# ### Skeleton first — minutes, not hours
#
# The 3D-pose ST-GCN is ~2 M params and trains in a few minutes on ~1 MB of
# data. Run it first: it costs almost no quota, is present for 100% of clips,
# and gives you a real cross-subject number to calibrate everything else
# against. 3D pose also discards appearance entirely, so it is the single most
# subject-invariant signal available.

# %%
!python train.py --data-root "$DATA" --modality Skeleton --fold 0 \
    --epochs 40 --batch-size 32 --frames 32 --lr 1e-3 \
    --workers 2 --out-dir /kaggle/working

# %% [markdown]
# ### Then the image modalities
#
# Depth_Color first (100% coverage, 640x480), then Thermal (strongest single
# modality in the source paper at 92.6%, but missing on 10 test clips), then IR.

# %%
MODALITY = "Depth_Color"
FOLD     = 0
EPOCHS   = 20

!python train.py --data-root "$DATA" --modality $MODALITY --fold $FOLD \
    --epochs $EPOCHS --batch-size 16 --frames 16 --size 144 \
    --arch resnet18 --lr 3e-4 --workers 2 --out-dir /kaggle/working

# %% [markdown]
# ### Remaining runs
#
# Uncomment as quota allows. Two folds per modality is usually enough — the
# ensemble gain past that is small relative to the budget it consumes.

# %%
# !python train.py --data-root "$DATA" --modality Thermal     --fold 0 --epochs 20 --out-dir /kaggle/working
# !python train.py --data-root "$DATA" --modality IR          --fold 0 --epochs 20 --out-dir /kaggle/working
# !python train.py --data-root "$DATA" --modality Skeleton    --fold 1 --epochs 40 --frames 32 --batch-size 32 --lr 1e-3 --out-dir /kaggle/working
# !python train.py --data-root "$DATA" --modality Depth_Color --fold 1 --epochs 20 --out-dir /kaggle/working
# !python train.py --data-root "$DATA" --modality Thermal     --fold 1 --epochs 20 --out-dir /kaggle/working

# %%
if os.path.exists("/kaggle/working/results.txt"):
    print(open("/kaggle/working/results.txt").read())
for f in sorted(glob.glob("/kaggle/working/*.pt")):
    print(f"{os.path.basename(f):<28} {os.path.getsize(f)/1e6:7.1f} MB")

# %% [markdown]
# ## Read the result honestly
#
# The validation number here is **cross-subject** and is the closest thing you
# have to the private leaderboard *and* to the Selection Stage re-test. If val
# says 0.78 and the public LB says 0.90, trust the 0.78 — the Zoom verification
# disqualifies teams whose accuracy drops more than 10 points on fresh subjects.
#
# Next: Step 3 fuses the per-modality probabilities and writes the submission.
