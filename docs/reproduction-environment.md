# Reproduction environment (answers given to the organisers)

Submitted on 18 Sep 2026 to the CUHK-X Reproduction Environment Survey for the
Small Model Track (Kaggle discussion 740972), by team **InSociEUP**. Recorded
here so the repository states the same thing the organisers were told.

| Question | Answer |
|---|---|
| A1 Kaggle team name | InSociEUP |
| B1 Peak GPU memory, one inference process | ≤8 GB |
| B2 GPUs per inference run | 1 |
| C1 Container image | No — `requirements.txt` in this repository |
| C2 PyTorch version | 2.10.0 (developed and tested); anything ≥2.4 should work |
| C3 CUDA version | 12.6 or newer (the Kaggle image used cu128) |
| C4 Special dependencies | None of the above |
| C5 Development GPU | NVIDIA Tesla T4 16 GB (Kaggle, T4 ×2 session, one GPU per run) |
| D1 One full inference run | 5–15 min on a T4 |
| D2 Peak system RAM | ≤16 GB |
| E2 Team time zone | UTC+5:30 |

## Measurements behind those answers

Timed on the local CPU (i3-10110U, two threads), running the published
`inference.sh` path with `runs/model_soup_S6_b1.pth` over six real test clips:

- **13.0 s per clip**, so about 1.5 hours for all 405 clips on CPU.
- **1.5 GB peak host RSS** at batch 4.
- Person boxes found for 6/6 clips; all five streams contributed.

Almost all of that time is model compute, which a T4 does 10–30× faster, giving
roughly 3–8 minutes plus model loading for the full test set — hence the
5–15 minute answer. GPU memory was not measured directly (no local GPU), but
the same harness peaked at 1.73 GB for the thermal model and 0.21 GB for the
skeleton model during *training*, and the big model trained on a 15 GB T4 at
this batch size and resolution with gradients and optimiser state, none of
which inference carries.

## Notes given in the free-text field

- Code, logs and `inference.sh`: <https://github.com/SanTanBan/cuhkx-small-model-track>
- Checkpoints: <https://www.kaggle.com/datasets/santanubanerjee9/cuhkx-small-model-track-insocieup>
  (`model_soup_S6_b1.pth` is the submission that counted; `model_student_r3a_b1.pth`
  scored higher on the private split.)
- One self-contained 92.9 MB checkpoint holds every weight used at inference,
  including the person detector. No internet access and no pre-downloaded
  weights are needed. Members are stored int8 and dequantised at load.
- No custom CUDA/C++ extensions, no FlashAttention, no Triton, single GPU. Runs
  CPU-only as well.
- Input expected: the extracted test directory with `SM_test_XXXX/<modality>/`
  subfolders. The four clips with zero-filled IR are handled by renormalising
  the fusion over the modalities actually present.
- **Caveat:** the submitted CSVs came from a three-view test-time average
  computed during training, while `infer.py` averages two deterministic views
  (identity and horizontal flip). A re-run lands very close but is not
  guaranteed bit-identical; the three-view variant can be supplied if exact
  reproduction of the submitted CSV is required.
- Licence: the video trunk is fine-tuned from IG-65M/Kinetics R(2+1)D-34
  (Facebook VMZ weights, non-commercial research licence), so the published
  checkpoints are for research and verification only. No competition data is
  redistributed here.
