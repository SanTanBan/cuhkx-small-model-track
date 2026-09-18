> Archived. This was the plan written at the start of the competition, kept for
> the record: the data findings still hold, but the final solution differs (see
> the top-level README.md and SUBMISSION_LOG.md).

# CUHK-X Small Model Track — solution

40-class cross-subject human activity recognition from privacy-preserving
sensors. Kaggle deadline **15 Sep 2026**; code + report deadline **22 Sep 2026**.

---

## What actually wins this

The Kaggle leaderboard is a **gate, not the prize**. For finalists the score is:

| Component | Weight |
|---|---|
| On-site private test | 30% |
| Technical report | 20% |
| **Kaggle private LB** | **20%** |
| Presentation | 10% |
| Reproducibility | 10% |
| Model efficiency | 10% |

And between Kaggle and the finals sits the **Selection Stage** (16–30 Sep): a
45-minute Zoom where organizers hand you a fresh dataset containing unseen
subjects, you have 2 hours to return predictions, and **a drop of more than 10
points from your Kaggle score disqualifies you**. Your slot goes to the next
team.

A confirmed leak existed early on (test skeleton filenames were matchable to
public metadata via timestamps); the repo was pulled and organizers said they
would expand the Stage-2 pool toward "teams that demonstrate genuine progress."

> **A robust 0.88 beats a fragile 0.97.** Everything here optimises
> cross-subject generalisation, and validates on held-out *people* rather than
> held-out clips.

### Leaderboard targets

Public LB = 50% of test = 201 clips, so one clip is worth 0.4975.

| Goal | Score | Meaning |
|---|---|---|
| **Top 15 → Finalist** | **0.826** | the gate that matters |
| Top 15% (~rank 42) | ~0.751 | "Excellence" |
| Top 30% (~rank 85) | ~0.70 | "Distinction" |

Current top three: 0.985 / 0.980 / 0.975. Public notebooks: 0.667–0.80.

---

## Rules worth exploiting

All confirmed by host replies in the competition discussions:

- ✅ **ImageNet-pretrained ResNet18** and similar small CNNs — only LLM/VLM
  foundation models are banned
- ✅ **Knowledge distillation** from larger models
- ✅ **Ensembles** — bundle every weight into *one* file < 100 MB;
  **fp16/int8 explicitly encouraged**
- ✅ **Pseudo-labelling the test set** (with a generalisation warning)
- ✅ **External public datasets** (must be public and disclosed)
- ✅ Deterministic per-sample test preprocessing

100 MB in fp16 holds roughly **8 ResNet18s**. Size is not the binding
constraint — cross-subject generalisation is.

---

## What the data actually contains

Measured from the real 405-clip test set, not assumed:

| Modality | Coverage | Format | Verdict |
|---|---|---|---|
| **Skeleton** | **405/405** | 17-joint **3D pose**, JSON per frame | **Highest value** |
| Depth_Color | 405/405 | 640×480 RGB PNG | Backbone |
| IR | 401/405 | 640×480 grayscale PNG | Backbone |
| Thermal | 395/405 | 320×240 JPG, ~2.2× frame rate | Strong, incomplete |
| Radar | 198/405 non-empty | CSV | **Excluded — 207 clips have zero rows** |
| IMU | ~37–53 rows | 2 CSVs, mixed encodings | Marginal |

Findings that shaped the design:

- **Skeleton is Human3.6M 17-joint 3D**, confirmed by joint-height ordering
  (feet 3/6 lowest → head 10 highest), root-centred in x/y with z as height.
  3D pose discards appearance, clothing, body temperature and lighting — the
  exact nuisance variables that break cross-subject models. It is present for
  100% of clips and costs 4.1 MB.
- Clips are **short**: median 20 depth frames, min 2. 16 sampled frames covers
  most of them.
- 4 IR clips (`SM_test_0012/0014/0154/0194`) are **zero-filled and
  unrecoverable**. Fusion renormalises over present modalities so these still
  get a prediction.
- Archives carry `__MACOSX/`, `.DS_Store` and a stray `.claude/` directory
  inside the test root — all silently ingested as clips unless filtered.

### The 40 classes split into three regimes

- **0–11** fine-grained hand/object (*Brush teeth*, *Peel fruits*) — needs
  spatial detail
- **17–27** desk/media, highly confusable (*Check the time* vs *Use a mobile
  phone* vs *Make a phone call*)
- **28–36** whole-body motion (*Squats*, *Lunges*, *Walk*) — needs temporal
  dynamics; skeleton dominates here

This is why person-cropping is worth ~4 points in the public notebooks, and why
a pose model and an appearance model make a genuinely complementary pair.

---

## Approach

Per-modality models, late-fused on probabilities.

| Modality | Model | Params |
|---|---|---|
| Skeleton | **ST-GCN** on the H36M graph, 3 spatial partitions, joint+bone+velocity channels | 2.05 M (4.1 MB fp16) |
| Depth / IR / Thermal | **TSM-ResNet18** — Temporal Shift Modules in the residual branches, attention-pooled over time | 11.3 M (22.5 MB fp16) |

Generalisation measures, all aimed at the cross-subject gap:

- **GroupKFold on `user`** — no person appears in both train and val
- **Per-clip intensity normalisation** — absolute thermal/IR level encodes body
  temperature and session, i.e. subject identity
- **Torso-length pose normalisation** — removes body-size differences between
  people
- Yaw rotation, left/right mirroring, scale and jitter on pose; random resized
  crop, flip, brightness and erasing on images
- Mixup, label smoothing, EMA with a warmup ramp, cosine schedule
- 3-view TTA; fusion weights fitted **on out-of-fold predictions only**

---

## Runbook

Local machine is an i3-10110U with no CUDA, so **all training runs on Kaggle**
(free P100/T4×2, 30 GPU-h/week). Local is used only for validating code.

### 0. One-time — only you can do these

1. **Register** at <https://openaiotlab.github.io/CUHK-X-Challenge/> — required
   for certificates and finalist notification.
2. **Join** the Kaggle competition; the Kaggle team name must match the
   registered one **exactly**.
3. Accept the dataset terms at
   <https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track>.
4. On Kaggle: **Add-ons → Secrets → `HF_TOKEN`** = a HuggingFace read token.

### 1. Build the compact dataset (Kaggle, CPU, ~1–2 h unattended)

Import `notebooks/01_prepare_data.ipynb`. Accelerator **None**, Internet
**On**. `Save Version → Save & Run All (Commit)`.

Downloads 44 GB at Kaggle's bandwidth, extracts one modality at a time to stay
inside the ~73 GB disk budget, and writes ~5–8 GB of shards.

**Verify before continuing:** training users must be **1–9 and 16–24** with
none of 10/11/25/26, and every modality should cover 40/40 classes.

### 2. Train (Kaggle, GPU)

Import `notebooks/02_train.ipynb` and attach Step 1's output.

Run **Skeleton first** — ~2 M params on ~65 MB of data, a few minutes, and it
gives you a real cross-subject number to calibrate against before spending
quota. Then Depth_Color, Thermal, IR.

Budget: roughly 2–3 h per image modality per fold on a P100. With 30 GPU-h/week
plan for ~4–6 image runs plus several cheap skeleton runs.

### 3. Fuse and submit

Import `notebooks/03_submit.ipynb`. Fits fusion weights on OOF, prints
per-class recall and top confusions, writes `submission.csv`, and packs all
members into one fp16 `model.pth` with a size check.

### Local development

```bash
python scripts/05_smoke_test.py          # 45 checks, ~4 min, CPU only
python scripts/01_download.py --part test
python scripts/02_extract.py  --part test
python scripts/04_preprocess.py --modalities Skeleton
python scripts/make_notebooks.py         # rebuild .ipynb after editing kaggle/*.py
```

---

## Layout

```
kaggle/
  prep.py     compaction: discovery, JPEG blobs, pose packing   (shared)
  cuhkx.py    Cfg, datasets, TSM-ResNet18, ST-GCN, EMA, packer  (shared)
  train.py    one modality x one fold -> weights + OOF + test logits
  nb_0*.py    notebook sources (# %% cells; #!writefile inlines a module)
notebooks/    generated .ipynb — import these into Kaggle
scripts/      local runners: download, extract, probe, preprocess, smoke test
compact/      preprocessed shards
```

`kaggle/prep.py` and `kaggle/cuhkx.py` are the single source of truth; the
notebooks inline them via `#!writefile` at build time, so local and Kaggle runs
cannot drift.

---

## Reading results honestly

Track two numbers on every submission: **fused OOF cross-subject accuracy** and
the **public LB**. If the LB runs well ahead of OOF, you are fitting the test
set — which wins nothing here, because the Selection Stage re-tests you on
fresh subjects with a 10-point tolerance.

Deliverables due 22 Sep: `code/`, `checkpoints/model.pth`, `inference.sh`,
`README.md`, `honor_declaration.pdf`.
