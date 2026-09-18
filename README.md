# [CUHK-X Challenge 2026 — Small Model Track (team InSociEUP)](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track)

40-class cross-subject human activity recognition from privacy-preserving
sensors (depth, IR, thermal, skeleton, IMU; no RGB), under a **100 MB total
model budget**. Everything that runs at inference lives in one checkpoint.

**Final standing: private leaderboard 0.75980, rank 60 of 326 (top 18.4%,
Distinction tier).** Public leaderboard 0.72636, rank 79.

| Submission | Public | Private | |
|---|---|---|---|
| S6 — six-model weight average | 0.72636 | 0.75980 | the score that counted |
| R3A — round-3 student alone | 0.71641 | **0.77450** | our best; would have ranked 46 (Excellence) |
| S4 — four-student weight average | 0.71641 | 0.75980 | |
| Two-student weight average | 0.71144 | 0.75490 | |
| R3B — round-3 student (128 px) | 0.70646 | 0.74509 | |
| Round-2 student alone | 0.71641 | 0.74019 | |
| Round-1 student alone (day 4) | 0.71641 | — | best public before the last day |

The public half is 201 clips, so one clip is worth 0.5 points there. Nothing in
the public spread of these submissions (0.706–0.726, i.e. four clips) predicted
the private spread (0.740–0.775, seven clips). See
[SUBMISSION_LOG.md](SUBMISSION_LOG.md) for the day-by-day reasoning, including
the decisions that turned out wrong.

## The solution

Nine members, fused as a weighted geometric mean of their class probabilities,
packed into a single 92.9 MB checkpoint.

| Member | Input | Notes |
|---|---|---|
| R(2+1)D-34 | 4-channel Depth_Color + IR person crops, 16 frames | IG-65M → Kinetics init (ruled a permitted pretrained CNN by the hosts, topic 738333); the one big model, int8 |
| ST-GCN × 6 | 17-joint 3D skeleton | folds 0/2/4 of two pose normalisations |
| S3D | Thermal | |
| 1-D ResNet | 5 body-worn IMUs, accelerometer + gyroscope | |
| SSDlite320 | 4 IR frames | person detector for the crops, fp16 |

What mattered, in rough order of value:

1. **Person crops.** Boxes from a small detector on IR frames, applied to the
   pixel-aligned depth and IR streams.
2. **Self-training.** Test clips the fused ensemble labelled confidently
   (p ≥ 0.6) were added to training. Round 1 moved a held-out-user fold from
   0.644 to 0.712 and the public score from 0.687 to 0.711.
3. **Weight averaging ("model soup").** Four students trained from the same
   initialisation and recipe, averaged into one model, which keeps most of an
   ensemble's gain inside the size budget. Verified on held-out users first:
   averages matched their prediction ensembles, and beat their parents when the
   parents were comparably strong.
4. **Class rebalancing.** One Sinkhorn step towards the training class mix.
   A second step lost clips both times it was tried.
5. **Cross-subject validation throughout.** GroupKFold on `user`, never on
   clips; fusion weights fitted on out-of-fold predictions only.

Quantisation: members int8 per-channel with a clipping search, detector fp16.
int6 was tried and rejected — it flipped four of 32 test predictions.

## Layout

```
kaggle/      library and Kaggle-side entry points
  cuhkx.py     Cfg, datasets, models (R(2+1)D-34, ST-GCN, S3D, TSM, IMUNet), EMA, int8 packer
  prep.py      compaction: person boxes, JPEG blob packing, IMU parsing
  train.py     one modality × one fold → weights + out-of-fold + test predictions
  infer.py     package → predictions.csv (the reproduction entry point)
  soup.py      weight averaging and its held-out checks
  *_template.py  Kaggle kernel templates (runner, detector, soup, IMU)
scripts/     local orchestration: kaggle_ops.py (datasets/kernels/fusion/submit),
             pack_fixed.py (package builder), pseudo_labels.py, teacher2.py, soup_pack.py
notebooks/   generated notebooks for the data-preparation step
runs/        per-run training logs, out-of-fold reports, every submission CSV
docs/        the plan written at the start, kept for the record
```

## Reproducing a submission

```bash
pip install -r requirements.txt
# checkpoint: https://www.kaggle.com/datasets/santanubanerjee9/cuhkx-small-model-track-insocieup
mkdir -p checkpoints && mv model_soup_S6_b1.pth checkpoints/model.pth
./inference.sh /path/to/small_model_track_test final_submission.csv
```

`inference.sh` calls `kaggle/infer.py`, which reads the raw test folders
(`SM_test_0001/`, …), runs the packaged detector and every member, fuses them
with the weights stored in the checkpoint, applies the rebalancing step and
writes the CSV. It needs no internet and no downloads: every weight it uses is
in the checkpoint. Measured on two CPU threads (i3-10110U): 13.0 s per clip and
1.5 GB peak host RAM, so about 1.5 hours for all 405 clips. Nearly all of that
is model compute, so a T4 run should land in the 5–15 minute range.

One caveat worth stating: the submitted CSVs were produced from per-member test
probabilities computed during training with a three-view augmentation average,
while `infer.py` averages two deterministic views (identity and horizontal
flip). The predictions are therefore very close but not guaranteed identical.

## Training

Training ran entirely on Kaggle (T4 ×2, 30 GPU-h/week); the local machine has
no GPU. `scripts/kaggle_ops.py` stages the compact datasets, builds and pushes
kernels, pulls results, fits the fusion and submits. For example, the round-3
students:

```bash
python scripts/kaggle_ops.py kernel r2p-r3 --datasets meta depthpc irpc \
  --embed cuhkx.py train.py pl3_test.npz \
  --jobs "DepthIR_R3A:-1:30::temporal-aug+arch=r2plus1d_34+norm=kinetics+lr=1.5e-4+mixup-prob=0+label-smoothing=0+pseudo=pl3_test.npz+size=144" \
         "DepthIR_R3B:-1:30::temporal-aug+arch=r2plus1d_34+norm=kinetics+lr=1.5e-4+mixup-prob=0+label-smoothing=0+pseudo=pl3_test.npz+size=128"
python scripts/kaggle_ops.py push r2p-r3
```

The full sequence of runs, with timings and scores, is in
[SUBMISSION_LOG.md](SUBMISSION_LOG.md).

## Data, licence, attribution

- **No competition data is in this repository.** The CUHK-X dataset is released
  under the CUHK-X License v2.0 and must be obtained from the organisers
  ([Hugging Face](https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track)).
- The video trunk is fine-tuned from **R(2+1)D-34 pretrained on IG-65M and
  Kinetics** (Facebook VMZ weights, non-commercial research licence), so the
  published checkpoints are for research and competition verification only.
- The person detector is torchvision's SSDlite320 MobileNetV3-Large.
- Code in this repository is shared for verification and research use.

## Further reading

- [docs/approach-and-decisions.md](docs/approach-and-decisions.md) — why each
  choice was made, the ideas rejected and why, and the retrospective on the
  final-pick mistake that cost a certificate tier.
- [SUBMISSION_LOG.md](SUBMISSION_LOG.md) — the day-by-day log, including the
  packaging bug and the deadline misread.
- [docs/reproduction-environment.md](docs/reproduction-environment.md) — the
  answers given to the organisers' reproduction survey, with the measurements
  behind them.
- [docs/initial-plan.md](docs/initial-plan.md) — the plan written at the start,
  kept for the record.

## Links

- Checkpoints: <https://www.kaggle.com/datasets/santanubanerjee9/cuhkx-small-model-track-insocieup>
- Notebook that opens the package and lists its members:
  <https://www.kaggle.com/code/santanubanerjee9/cuhk-x-small-model-track-insocieup-solution>
- Competition: <https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track>
