# %% [markdown]
# # CUHK-X Small Model Track — Step 3: fuse, pack, submit
#
# Three jobs:
# 1. Fit modality fusion weights **on out-of-fold predictions**, never on the
#    leaderboard. Tuning weights against the public LB is how a team ends up
#    with a great public score and a failed Selection Stage.
# 2. Pack every ensemble member into **one fp16 checkpoint under 100 MB**, which
#    is what the rules require.
# 3. Write `submission.csv` in exactly the expected format.
#
# **Setup:** attach the Step 2 notebook's output (the `.pt` / `.npz` files) and
# the `compact` dataset. GPU only needed if you re-run inference here.

# %%
import os, glob, json, itertools
import numpy as np, pandas as pd, torch

WORK = "/kaggle/working"
ART  = None                      # directory holding *_oof.npz / *_test.npz / *.pt
for c in glob.glob("/kaggle/input/*"):
    if glob.glob(f"{c}/*_oof.npz"):
        ART = c; break
ART = ART or WORK
DATA = next((p for p in glob.glob("/kaggle/input/*/compact") if os.path.isdir(p)),
            "/kaggle/input/cuhkx-compact/compact")
print("artifacts:", ART, "\ndata     :", DATA)
print(sorted(os.path.basename(p) for p in glob.glob(f"{ART}/*.np*"))[:20])

# %% [markdown]
# ## Load OOF predictions
#
# Each `{mod}_f{k}_oof.npz` covers a disjoint set of users, so concatenating the
# folds of one modality gives a full out-of-fold prediction over all training
# subjects — an unbiased estimate of cross-subject accuracy.

# %%
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

oof = {}      # modality -> DataFrame(clip_id, user, y, p0..p39)
for f in sorted(glob.glob(f"{ART}/*_oof.npz")):
    base = os.path.basename(f)[:-8]           # "Thermal_f0"
    mod = base.rsplit("_f", 1)[0]
    d = np.load(f, allow_pickle=True)
    p = softmax(d["logits"].astype(np.float64))
    df = pd.DataFrame(p, columns=[f"p{i}" for i in range(p.shape[1])])
    df.insert(0, "y", d["y"]); df.insert(0, "user", d["user"].astype(str))
    df.insert(0, "clip_id", d["clip_id"].astype(str))
    oof.setdefault(mod, []).append(df)

for m in list(oof):
    oof[m] = pd.concat(oof[m], ignore_index=True)
    P = oof[m][[f"p{i}" for i in range(40)]].values
    acc = (P.argmax(1) == oof[m].y.values).mean()
    print(f"{m:<14} {len(oof[m]):>6} OOF clips  "
          f"{oof[m].user.nunique():>2} users  acc {acc:.4f}")

if not oof:
    raise SystemExit("No *_oof.npz found — run Step 2 first.")

# %% [markdown]
# ## Fit fusion weights on OOF
#
# Grid search over simplex weights. With few modalities this is exact and far
# more robust than learning a stacker on so little data.

# %%
mods = sorted(oof)
common = set(oof[mods[0]].clip_id)
for m in mods[1:]:
    common &= set(oof[m].clip_id)
common = sorted(common)
print(f"clips shared by all {len(mods)} modalities: {len(common)}")

cols = [f"p{i}" for i in range(40)]
stack, y = [], None
for m in mods:
    d = oof[m].set_index("clip_id").loc[common]
    stack.append(d[cols].values)
    y = d.y.values
stack = np.stack(stack)                       # [M, N, 40]

best_w, best_acc = None, -1
grid = np.arange(0, 1.01, 0.05)
for w in itertools.product(grid, repeat=len(mods)):
    s = sum(w)
    if s <= 0: continue
    w = np.array(w) / s
    acc = ((w[:, None, None] * stack).sum(0).argmax(1) == y).mean()
    if acc > best_acc:
        best_acc, best_w = acc, w

for m in mods:
    d = oof[m].set_index("clip_id").loc[common]
    print(f"  {m:<14} solo {(d[cols].values.argmax(1)==y).mean():.4f}")
print(f"\nbest fusion weights: {dict(zip(mods, best_w.round(3)))}")
print(f"FUSED OOF cross-subject accuracy: {best_acc:.4f}")

# %% [markdown]
# ### Where is it failing?
#
# Per-class recall points at what to fix next. Confusions here are almost always
# between the semantically adjacent classes (*Check the time* / *Use a mobile
# phone* / *Make a phone call*), which is a spatial-detail problem, not a
# temporal one.

# %%
fused = (best_w[:, None, None] * stack).sum(0)
pred = fused.argmax(1)
cm_names = pd.read_csv(f"{DATA}/class_mapping.csv") \
    if os.path.exists(f"{DATA}/class_mapping.csv") else None
name = dict(zip(cm_names.action_id, cm_names.action_name)) if cm_names is not None else {}

rows = []
for c in range(40):
    m_ = y == c
    if m_.sum() == 0: continue
    wrong = pred[m_][pred[m_] != c]
    top = pd.Series(wrong).value_counts().head(1)
    rows.append(dict(cls=c, name=name.get(c, c), n=int(m_.sum()),
                     recall=float((pred[m_] == c).mean()),
                     top_confusion=name.get(int(top.index[0]), int(top.index[0]))
                     if len(top) else "-"))
worst = pd.DataFrame(rows).sort_values("recall").head(12)
print(worst.to_string(index=False))

# %% [markdown]
# ## Build the submission

# %%
test_p, test_ids = {}, None
for f in sorted(glob.glob(f"{ART}/*_test.npz")):
    base = os.path.basename(f)[:-9]
    mod = base.rsplit("_f", 1)[0]
    d = np.load(f, allow_pickle=True)
    ids = d["clip_id"].astype(str)
    df = pd.DataFrame(d["probs"], index=ids)
    test_p.setdefault(mod, []).append(df)

agg = {}
for m, lst in test_p.items():
    # average the folds of a modality, then align every modality to one index
    agg[m] = sum(x.reindex(lst[0].index) for x in lst) / len(lst)
    print(f"{m:<14} test clips {len(agg[m])}")

all_ids = sorted(set().union(*[set(v.index) for v in agg.values()]))
num, den = np.zeros((len(all_ids), 40)), np.zeros((len(all_ids), 1))
for w, m in zip(best_w, mods):
    if m not in agg: continue
    v = agg[m].reindex(all_ids)
    mask = ~v.isna().all(axis=1).values          # clips missing this modality
    num[mask] += w * np.nan_to_num(v.values[mask])
    den[mask] += w                                # renormalise over what exists
final = num / np.maximum(den, 1e-9)
pred = final.argmax(1)

sub = pd.DataFrame({"path": [f"small_model_track_test/{i}/" for i in all_ids],
                    "prediction": pred.astype(int)})

# Conform to the official template: same rows, same order, no gaps.
tmpl = pd.read_csv(f"{DATA}/test.csv")
sub = tmpl[["path"]].merge(sub, on="path", how="left")
missing = sub.prediction.isna().sum()
if missing:
    print(f"!! {missing} clips had no prediction — filling with the modal class")
    sub["prediction"] = sub.prediction.fillna(pd.Series(pred).mode()[0])
sub["prediction"] = sub.prediction.astype(int)

sub.to_csv(f"{WORK}/submission.csv", index=False)
print(f"\nwrote {len(sub)} rows -> submission.csv")
print(sub.head())
print(f"\nprediction spread: {sub.prediction.nunique()}/40 classes used")
print(sub.prediction.value_counts().head(5))

# %% [markdown]
# ## Pack the competition checkpoint
#
# Rule: *every* weight loaded at inference goes into one file under 100 MB.

# %%
#!writefile cuhkx.py

# %%
from cuhkx import pack_fp16
sds = {}
for p in sorted(glob.glob(f"{ART}/*.pt")):
    sds[os.path.basename(p)[:-3]] = torch.load(p, map_location="cpu")
if sds:
    mb = pack_fp16(sds, f"{WORK}/model.pth",
                   meta=dict(modalities=mods, weights=best_w.tolist(),
                             oof_acc=float(best_acc), frames=16, size=144,
                             arch="tsm_resnet18"))
    print(f"members: {list(sds)}")
else:
    print("no .pt files found to pack")

# %% [markdown]
# ## Submit
#
# `Save Version -> Save & Run All`, then from the notebook's Output tab click
# **Submit to Competition**.
#
# Track both numbers every time: the fused **OOF cross-subject accuracy** above
# and the public LB. If the LB runs far ahead of OOF, you are fitting the test
# set and will lose the seat at the Selection Stage — the pass criterion there
# is a drop of no more than 10 points on fresh subjects.
