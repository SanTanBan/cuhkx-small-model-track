# Submission log — CUHK-X Small Model Track

One entry per leaderboard submission: what changed, why, and what it taught.
"CV" is always **cross-subject** accuracy — user-grouped folds, so no person is
ever in both train and validation. Public LB = 50% of the 405 test clips (201),
so one clip ≈ 0.5 pt.

Top-15 (Finalist) public cutoff at the time of writing: **0.836**.

---

## Day 1 — 2026-09-10 (UTC)

### #1 · Skeleton ST-GCN, single fold → **public 0.40298** (rank ~190)

- **What:** 2 M-parameter ST-GCN on 3D pose (Human3.6M 17 joints; joint,
  bone and velocity channels), one fold (held-out users 17, 18, 21), 30 epochs
  on Kaggle CPU.
- **Why this first:** both GPU slots were busy with other runs, while pose is
  present for 100% of clips, tiny, and the most subject-invariant signal. It
  is the cheapest honest anchor for how CV maps to the leaderboard.
- **Learned:** CV 0.564 vs LB 0.403. A large gap, and one fold is noisy.

### #2 · Skeleton ST-GCN, 5-fold ensemble → **public 0.44776**

- **What:** the same model trained on all 5 user-grouped folds; test
  probabilities averaged across the five.
- **Why:** averaging models that each saw different people reduces variance
  under subject shift, and pooling five folds gives an honest CV over all 18
  training users rather than one lucky split.
- **Learned:** pooled CV 0.534; LB **+4.5 pts** from ensembling alone; the
  CV→LB gap shrinks to 8.6 pts — a real but steady cross-subject shift.

### Investigated and rejected (no submission spent)

| Hypothesis | Test | Verdict |
|---|---|---|
| Test clips are shorter (median 20 vs 24 frames, p90 41 vs 56) and that causes the gap | OOF accuracy by clip length | Short clips are the model's *easiest* (0.59 vs 0.49) — not the cause |
| Label shift: test class mix differs, fix with EM prior re-estimation | EM on each held-out fold (each fold = different people, different class mix) | **Hurts** −1 to −7 pts on every fold — rejected |
| Keeping body height in the pose normalisation helps (sit / stand / squat) | Fold 4, 30 epochs, same seed: height kept vs removed | **Hurts**: 0.535 vs 0.564. Absolute hip height from the pose estimator varies by subject and setup, so it behaves as identity, not signal. Kept as a flag (`--center-z` restores the original) rather than reverting the data |

### Skeleton A/B on fold 4 (same held-out users 17/18/21, same seed)

Controlled arms, one change at a time, to pick the Day-2 skeleton recipe:

| Arm | Best epoch | Final epoch | Reading |
|---|---|---|---|
| Original normalisation (height removed), 30 ep | 0.564 | 0.564 | baseline |
| Height kept, 30 ep | 0.535 | 0.528 | height hurts (−3) |
| Height kept, 60 ep | 0.568 | 0.547 | longer training +2–3, peaks ~ep 50 |
| Height kept, 60 ep **+ temporal & amplitude aug** | **0.589** | **0.579** | aug +2–3 **and** removes the late decline |
| Height removed, 60 ep + aug | 0.581 | 0.568 | just below height kept once augmented (≈5 clips, within noise) — the amplitude augmentation takes over the job height removal was doing |

Settled: **60 epochs + augmentation**. Open: which normalisation pairs best with
it. Hedged on CPU (free, no GPU quota): folds 0–1 are training with height
removed, folds 2–3 with height kept. Whichever wins gets completed; if they are
close, both are kept as two skeleton streams and the fusion weights them on
held-out folds — two differently normalised views of the same pose usually
ensemble better than either alone.

### In flight (next upgrades)

1. **Pose normalisation bug fix.** The skeleton is grounded (feet z≈0), so z
   carries hip/head height — what separates sit / stand / squat / lie down.
   The original normalisation subtracted the root's full position every frame
   and zeroed that height. A/B on fold 4 (vs 0.564): height kept at 30 epochs,
   at 60 epochs, and with augmentation.
2. **Temporal + motion-amplitude augmentation.** Training clips so far saw one
   fixed tempo per epoch; random 60–100% windows and 0.6–1.15× motion scaling
   simulate truncated, lower-energy performances like the test subjects'.
3. **Image models** — TSM-ResNet18 on Thermal, Depth and IR (5 folds + a
   full-data model each, temporal augmentation on). Needed to reach the
   cutoff: pose alone lacks the hand/object detail that separates *Check the
   time* from *Use a mobile phone*. Queued; they start as GPU slots free up
   (never pre-empting other runs).

---

## Day 2 — 2026-09-11 (UTC)

### #3 · Skeleton, two fused streams → **public 0.45273**

- **What:** two complete 5-fold skeleton ensembles fused with weights fitted on
  all 2,931 out-of-fold clips — the new recipe (height kept, 60 epochs,
  temporal + amplitude augmentation) at 0.59, the original at 0.41.
- **Why:** on fold 4 alone (477 clips) the case for combining them was too
  thin to act on; with full out-of-fold predictions the fusion weighs them
  honestly. The new recipe wins on average (0.541 vs 0.534) but not on every
  fold (fold 0: 0.505 vs 0.549) — the situation where two streams pay off.
- **Result:** CV 0.534 → **0.551** (+1.7); LB 0.448 → **0.453** (+0.5, one clip).
- **Learned:** pose is near its ceiling — about 0.55 cross-subject, ~0.45 on
  the leaderboard. The weakest classes are all fine hand/object actions
  (*Watch TV* 0.00, *Write* 0.03, *Play games* 0.10, *Wipe bowls* 0.14, *Use a
  mobile phone* 0.23): what appearance can see and pose cannot.

### Incidents (no score impact)

- **Laptop network outage, 06:33–07:48.** Kaggle runs were unaffected, but
  local watchers read the DNS failure as "kernel error" and push loops would
  have exited. Fixed: network errors now mean "retry", pulls wait up to 3 h,
  and a push that loses its reply checks the kernel's status before retrying
  so nothing is launched twice.
- **P100 incompatibility.** Kaggle's default GPU (Tesla P100, sm_60) is not
  supported by the image's PyTorch 2.10 / CUDA 12.8 — the Thermal kernel died
  on its first CUDA op. Every GPU kernel is now pushed on a T4 by default, and
  the runner stops with a clear message on any unsupported GPU.

### In flight

- **Thermal, Depth, IR** — TSM-ResNet18, 5 cross-subject folds plus one
  full-data model each, temporal augmentation, on T4 GPUs. They started only
  after the other runs on the account had finished; IR is queued behind them.
  These are the models the weakest classes need.

### #4 · Pose + appearance fusion → **public 0.46766**

- **What:** the two skeleton streams plus the first image models (Thermal and
  IR: TSM-ResNet18, 5 folds + a full-data model each), weighted on 2,786
  shared out-of-fold clips — original skeleton 0.35, height-kept 0.25,
  Thermal 0.20, IR 0.20.
- **Why:** the image models are weak alone (Thermal 0.374, IR 0.307
  cross-subject) but they fail on different clips than pose does, so the
  fusion can still use them — a question the held-out predictions answer at
  no GPU cost.
- **Result:** CV 0.551 → **0.567** (+1.6); LB 0.453 → **0.468** (+1.5, three clips).
- **Learned:** appearance is complementary even when weak. But the 2D
  ImageNet recipe generalises poorly across people: in both modalities
  training loss keeps falling while held-out accuracy flattens by epoch 8–12,
  which points at the backbone rather than the data.
- **Caveat:** this ensemble is 312 MB (22 models) — fine for exploring, but
  the final submission must come from a checkpoint under 100 MB, so members
  will be pruned and quantised (and re-scored on held-out folds) before the end.

### In flight

- **Kinetics-400 video backbones** — S3D (8 M params) and MC3-18 (11.7 M),
  pretrained on human-action video, A/B-tested on fold 0 against the TSM
  recipe: Thermal/S3D running, Depth/S3D and Thermal/MC3-18 queued. The winner
  gets full 5-fold runs on Thermal, Depth and IR.

### #5 · Depth S3D joins the fusion → **public 0.53233**

- **What:** Kinetics-400 video backbones replace the ImageNet 2D trunk for
  appearance. Depth and Thermal S3D (40 epochs, 5 folds + a full-data model
  each) join the two skeleton streams and the older TSM models. Weights are
  fitted by EM on the out-of-fold log-likelihood: Depth S3D 0.37, skeleton
  (original) 0.29, skeleton (height kept) 0.27, Thermal S3D 0.07 — EM gave the
  older Thermal/IR TSM models ~zero weight, dropping them on its own.
- **Why:** held-out accuracy showed the 2D recipe memorising people (training
  loss falling, held-out flat from epoch ~8). Fold-0 A/B on Thermal: TSM 0.366
  → S3D 0.412 → MC3-18 0.442 (20 epochs). Full runs: Thermal S3D-40 mean
  0.436 (+6.7 over TSM); **Depth S3D-40 mean 0.556 — the strongest single
  input, ahead of pose**.
- **Why EM:** the exhaustive weight grid grows 11^N (1.8 M points for six
  streams); EM maximises the mixture's out-of-fold log-likelihood — a concave
  problem, solved exactly in seconds — and fits noise less than an accuracy
  search.
- **Result:** CV 0.567 → **0.595** (+2.8); LB 0.468 → **0.532** (+6.5, 13
  clips). The CV→LB gap narrowed to 6.3 pts (≈10 with pose alone):
  appearance carries over to the test subjects better than pose.
- **Still weak:** Watch TV 0.00, Play games 0.03, Write 0.03, Use a mobile
  phone 0.11 — small hand-held objects, where more pixels on the hands would
  help.

### In flight — the strongest public recipe, rebuilt inside our pipeline

- **What the top public notebooks do.** The two person-crop notebooks (public
  0.667 with one fold, 0.711 with two) share one recipe: R(2+1)D-34
  pretrained on IG-65M (65 M Instagram videos) then Kinetics; Depth (3
  channels) and IR (1 channel) stacked into one 4-channel clip cropped to the
  person; 16 frames at 128 px, lr 5e-5, 30 epochs, EMA; int8/int6 weights to
  fit 100 MB. Our S3D starts from far weaker features (Kinetics only, 8 M
  params), which is where most of the 0.53 → 0.71 gap should come from.
- **Person crops (done).** torchvision SSDlite320-MobileNetV3 (COCO; BSD,
  chosen over AGPL YOLO) on 4 IR frames per clip, run as a Kaggle CPU job:
  a person in 94.5% of clips (test 378/401), one square box per clip (union
  of the frames × 1.3, median side 82% of the frame height). Depth and IR
  share one sensor, so one box crops both. `infer.py` recomputes the boxes
  from raw frames with the same `prep.py` functions — identical on 6/6 test
  clips checked.
- **Training (queued).** R(2+1)D-34 IG-65M on the 4-channel crops: 5
  cross-subject folds (for honest out-of-fold weights) + one full-data model.
  Kaggle T4 sessions carry two GPUs and the runner had been using one; it now
  runs a job per GPU, halving every session.
- **Package plan.** int8 R(2+1)D-34 (~64 MB) + fp32 detector (13 MB) + pose
  models → under 100 MB. `infer.py` now runs detector → crop → 4-channel
  input; on existing members it reproduces the old path exactly.
- **Capacity:** both GPU slots and all 5 CPU slots on the account were busy
  with other work today, so jobs wait their turn; this one uses a single GPU
  slot.

### Incident — weekly GPU quota used up

- At ~12:10 UTC Kaggle refused the R(2+1)D-34 run: "Maximum weekly GPU quota
  of 30.00 hours reached" — the 30 h/week is shared by every project on the
  account. No GPU run of any kind can start until it resets. The push now
  waits out a spent quota the way it waits out a full session pool, so the
  run starts by itself when GPU hours return.

### #6 · Geometric fusion of the same six streams → **public 0.57213**

- **What:** no new model — the six streams of #5 fused as a geometric
  (log-linear) pool, p ∝ Π_s p_s^w_s, instead of a weighted average; w fitted
  by maximum likelihood on out-of-fold predictions: skeleton 0.38 / 0.25,
  Depth S3D 0.37, Thermal S3D 0.23, Thermal TSM 0.09, IR TSM 0.01.
- **Why:** in an average, one confident-but-wrong stream can outvote the
  rest; in a product, streams that agree reinforce each other and a weak
  stream can still veto the classes it rules out — so the weak thermal
  models now contribute instead of being zeroed.
- **Validated before submitting:** weights fitted on 4 user-folds and scored
  on the 5th — 0.615 vs 0.593 for the linear mix, better on all 5 folds (142
  vs 81 discordant clips, p < 0.001).
- **Result:** CV 0.595 → **0.616**; LB 0.532 → **0.572** (+4.0, ~8 clips); 65 of
  405 test predictions changed.
- **Learned:** here the fusion rule is worth as much as a new model. Every
  ensemble from now on uses the geometric pool; `infer.py` reproduces it
  exactly (checked against a hand computation).
- **Tested, not adopted:** rebalancing the test predictions towards the
  training class mix (Sinkhorn scaling) — at best +0.6 to +0.8 points on
  held-out user folds, and stronger balancing loses points. Too small and too
  setting-sensitive to justify a step that depends on the test set; re-test
  once R(2+1)D-34 is in the fusion.


## Day 3 — 2026-09-12 (UTC)

### #7 · R(2+1)D-34 on Depth+IR person crops → **public 0.66169** (rank ~128)

- **What:** a new stream — R(2+1)D-34 (IG-65M → Kinetics) on the 4-channel
  Depth+IR person crops, 5 cross-subject folds + a full-data model (30
  epochs, 128 px, no mixup), fused geometrically with the six existing
  streams. Weights: R(2+1)D-34 0.41, skeleton 0.25 / 0.19, Thermal S3D 0.12,
  Depth S3D 0.08, Thermal TSM 0.04, IR TSM 0.
- **Single stream:** out-of-fold 0.692 (folds 0.644 / 0.671 / 0.749 / 0.669 /
  0.715) — the previous best input, Depth S3D, was 0.558.
- **Result:** CV 0.616 → **0.704** (nested CV, weights fitted on other folds:
  0.704); LB 0.572 → **0.662** (+9.0). The GPU quota reset overnight; the run
  used ~5.4 session hours on two T4s at once.
- **Still weak:** Play games 0.07, Write 0.08, Watch TV 0.09, Use a mobile
  phone 0.24 — small hand-held objects at a few pixels.
- **Team:** confirmed on the leaderboard as InSociEUP (matches the organiser
  registration).

### #8 · Package probe → **public 0.66666**

- **What:** only the members that fit the final ≤100 MB checkpoint —
  R(2+1)D-34 full-data model (6-bit), Depth and Thermal S3D full-data models
  (8-bit), both 5-fold skeleton streams (8-bit), plus the person detector:
  99.8 MB. Streams re-weighted without the two TSM streams (CV 0.703).
- **Why:** the final submission must be reproducible from one checkpoint, so
  the leaderboard should score exactly what the package predicts, not the
  larger fold ensembles.
- **Result:** LB **0.66666** vs 0.66169 for the full ensemble (+1 clip):
  packaging costs nothing. Full-data models are at least as good as the
  fold averages.

### Feedback from Santanu → strategy for the last three days

- **GPU:** use at most half of the remaining weekly GPU hours at any time.
- **Two tracks via the spare daily submission slots:** the IG-65M track
  (strongest) and a small-backbone track that uses only Kinetics-400
  torchvision models, as a hedge on the organisers' "no large pretrained
  backbones" wording. Final two picks: one of each.
- **In flight:** a second R(2+1)D-34 at 144 px (IG-65M track); then
  R(2+1)D-18 (Kinetics-400 only, 31 M) on the same 4-channel crops (small
  track), one GPU slot at a time.
- **Tested, marginal:** class rebalancing with the new stream +0.4–0.5 points
  on held-out folds — kept in reserve for a spare slot.

### Day 3 evening — a sensor we had not used: IMU

- **Found:** every clip also carries two CSVs from five body-worn IMUs (left
  and right arm, chest, left and right leg): acceleration and angular
  velocity at ~11 Hz. All five are present in ~99% of clips, train and test.
  Earlier data checks had covered the cameras, pose and radar only.
- **Why it matters:** our weakest classes (Write, Play games, Use a mobile
  phone, Watch TV) are distinguished by what the hands do, which the cameras
  see as a few pixels — but an arm sensor measures the wrist motion directly.
- **Built:** shared parsing in `prep.py` (training CSVs have Chinese headers,
  test CSVs English ones, same column order; millisecond stamps are not
  zero-padded; rows are slightly out of time order), packed to 40 steps per
  clip (2,863 train + 404 test clips, 18 MB); a 0.9 M-parameter 1-D ResNet
  trained on the same user folds as every other stream. After only 2 epochs
  it reached 0.19 on held-out users (chance 0.025). `infer.py` reproduces its
  predictions from the raw CSVs exactly.
- **Budget:** the model is tiny and trains on CPU, so it costs no GPU hours,
  and it belongs to both tracks (no pretrained backbone).
- **In flight:** 5 folds + full-data model on Kaggle CPU; then fusion,
  nested CV, and a submission for each track if it helps.

### #9 · Small-backbone track + IMU → **public 0.60696**

- **What:** the fallback track — only small Kinetics-400 / ImageNet backbones
  and models trained from scratch, no IG-65M — i.e. the #6 streams plus the
  IMU 1-D ResNet (out-of-fold 0.326 alone), fused geometrically, and scored
  as exactly the members of its 66 MB all-int8 package (no detector needed).
- **Why:** Santanu suggested using spare daily slots to keep a strong
  entry that stays inside the strictest reading of "no large pretrained
  backbones", as one of the two final picks.
- **Result:** nested CV 0.610 → 0.622; LB 0.572 → **0.607** (+3.5). IMU adds
  little to the IG-65M ensemble (+0.2 nested CV) but a lot here, where no
  stream sees fine hand motion well.

### Quantisation check (no submission)

- R(2+1)D-34 full model on 32 test clips, predictions vs fp16: int8 keeps
  31/32 (mean probability shift 0.011); int6 keeps 28/32 (0.071), int5 28/32.
  A per-channel clipping search lifted int6 only to 29/32. **Decision:** the
  big model ships in int8 (63.9 MB); the size budget is met elsewhere — fp16
  detector (7.1 MB; boxes identical on 348/401 test clips, within 2 px on
  395), Thermal S3D, 3 folds per skeleton stream, IMU (0.9 MB): 92.9 MB.
- #8's 99.8 MB int6 package is therefore retired as a final candidate.

### #10 · IG-65M track, final-package composition → **public 0.66666**

- **What:** exactly the 92.9 MB checkpoint planned above — R(2+1)D-34
  full-data model (int8), Thermal S3D full-data model, skeleton folds 0/2/4
  of both streams, the IMU full-data model, and the fp16 person detector —
  fused geometrically (R(2+1)D-34 0.42, IMU 0.26, skeleton 0.24 / 0.18,
  Thermal 0.12; CV 0.708).
- **Result:** LB **0.66666**, equal to #8 and above the unconstrained #7
  (0.66169): the valid package loses nothing. 39 of 405 predictions differ
  from #8 with the same public score, so the changes net out on the public
  half — the private half decides.
- **Next:** regenerate this submission with `infer.py` from the package itself
  (int8 weights, fp16-detector crops) so the final pick is byte-for-byte what
  the organisers' re-run produces.

### Organiser rulings (Kaggle forum) → strategy change

- **R(2+1)D-34 from IG-65M is explicitly allowed:** "Yes, this is acceptable.
  It counts as a permitted pretrained CNN, not a prohibited large backbone"
  (topic 738333, 7 Sep). Small pretrained person detectors for deterministic
  cropping are allowed too. The ban targets LLMs / large vision-language
  foundation models (topic 711665).
- **Size rule:** every weight loaded at inference, ensembles included, in one
  checkpoint file under 100 MB on disk; int8 and lower encouraged (729056,
  740749). Our 92.9 MB int8 package complies.
- **Pseudo-labelling test data is allowed** (self-training, 729989), with a
  caution to keep generalising to unseen subjects.
- **Consequences:** the small-backbone hedge is no longer needed for rule
  risk, so both final picks can come from the IG-65M track (#9 stays as a
  fallback). The queued R(2+1)D-18 run was cancelled before it reached Kaggle
  (pure insurance, and too large for the package), saving ~3.5 GPU hours.
  **Next lever:** one self-training round for R(2+1)D-34 — test clips the
  ensemble labels confidently join training — validated first on held-out
  fold 0 (true labels unseen), in the same GPU session as the full-data
  student.

## Day 4 — 2026-09-13 (UTC)

### Overnight

- The 144 px R(2+1)D-34 run finished: out-of-fold 0.693 (folds 0.655 /
  0.678 / 0.736 / 0.665 / 0.732) vs 0.690 at 128 px — better on three folds,
  worse on two, so the two resolutions disagree usefully.
- Coordinated with the Large Model Track sessions (same Kaggle account):
  they plan no GPU use, so the self-training run goes ahead. Their HAR data
  covers exactly our 18 training subjects (no test subjects) and our 40
  classes + 4 more, i.e. the same recordings — no new data for us.
- Selection mechanics (from their forum reading): the private leaderboard
  scores the better of the up-to-2 submissions **selected** on the
  Submissions page; without a selection Kaggle picks by public score.

### #11 · Experiment: all nine streams, both R(2+1)D-34 → **public 0.68656**

- **What:** geometric fusion of every stream — R(2+1)D-34 128 px (0.27) and
  144 px (0.19), IMU (0.25), skeleton (0.21 / 0.16), Thermal S3D (0.08),
  Depth S3D (0.02); the TSM streams get zero weight. CV 0.719.
- **Result:** LB 0.667 → **0.687** (+2.0), a new best — but not a valid
  package: two int8 R(2+1)D-34 alone are 128 MB.
- **Next:** knowledge distillation (allowed by the organisers, topic 711665)
  — train one R(2+1)D-34 on the ensemble's out-of-fold probabilities for
  training clips and its test probabilities for test clips, so a single
  package-sized model carries the ensemble's accuracy.

### In flight (day 4 morning)

- **Self-training (`r2p-pl`, running on Kaggle):** teacher = the #11 fusion.
  Two jobs side by side: fold 0 retrained with 374 of its 613 held-out
  clips pseudo-labelled at p ≥ 0.6 (86.4% of those labels are correct),
  scored on their true labels against the plain fold-0 model (0.644); and a
  full-data student that adds the 216 test clips labelled at p ≥ 0.6.
- **Distillation (`r2p-kd`, queued behind a CPU smoke test):** two
  R(2+1)D-34 students trained on the #11 ensemble's probabilities —
  out-of-fold probabilities for training clips (from models that never saw
  that subject). One also trains on the ensemble's test-clip probabilities, a
  soft form of the allowed pseudo-labelling. Goal: one package-sized model
  that carries the two-model ensemble's gain.
- **#12:** the valid package with the 144 px model in place of the 128 px one
  (being built).
- **Housekeeping:** the laptop is shared with other Claude sessions, so local
  CPU work is capped at two jobs. Kaggle hosts only `test.csv` and
  `sample_submission.csv`, so package-exact inference runs locally, once, on
  the final package.

### #12 · Valid package with the 144 px R(2+1)D-34 → **public 0.68656**

- **What:** the #10 composition with the 144 px full-data model in place of
  the 128 px one: R(2+1)D-34 144 px (int8) + Thermal S3D + skeleton folds
  0/2/4 of both streams + IMU + fp16 detector — 92.9 MB. Weights
  R(2+1)D-34 0.40, IMU 0.25, skeleton 0.23 / 0.21, Thermal 0.12; CV 0.706.
- **Result:** LB **0.68656** — equal to the unconstrained nine-stream #11 and
  +2.0 over #10. A single, valid ≤100 MB package now matches our best score,
  so it is the leading final pick.
- **Next:** package-exact `infer.py` run on this checkpoint (running locally
  with 2 threads), then self-training and distillation results decide
  whether a better single model replaces the 144 px one.

### Self-training result (validated on held-out users)

- **Fold 0** (539 clips of 4 unseen subjects, scored on true labels):
  plain R(2+1)D-34 0.644 (128 px) / 0.655 (144 px) → **0.712** when its 374
  confidently pseudo-labelled clips (86% correct labels, from the #11
  ensemble) join training — +6.9 points, above the teacher's own 0.690 on
  that fold.
- **Full-data student:** trained with the 216 test clips the ensemble labels
  at p ≥ 0.6.
- The distillation session never reached Kaggle: its script embedded 1.3 MB
  of soft targets and Kaggle rejected the save (400 Bad Request). Targets are
  now stored as uint8 (~5× smaller).

### #13 · Valid package with the self-trained student → **public 0.71144**

- **What:** #12's composition and weights with the self-trained student in
  the R(2+1)D-34 slot (92.9 MB package).
- **Result:** LB 0.687 → **0.711** (+2.5), our best, and the first valid
  package at the Distinction cutoff seen on the leaderboard.

### #14 · Same members, weights refit on fold 0 → **public 0.71144**

- **What:** stream weights fitted only on fold 0 (the student's held-out
  fold): skeleton 0.44 / 0.00, student 0.29, IMU 0.21, Thermal 0.10.
- **Result:** 10 test predictions differ from #13, same public score — the
  weighting is not what limits us.

### In flight

- **Round 2 (Kaggle, 144 px):** a stronger teacher — the nine-stream fusion
  stacked with the student (weights fitted on fold 0) — gives new test
  pseudo-labels and distillation targets. Two full-data students train side by
  side: self-trained (pseudo-labels) and distilled (soft targets on training
  and test clips). Either can replace the student in the package tomorrow.

### #15 · #13 + one-step class rebalancing → **public 0.71641**

- **What:** #13's fused probabilities, rescaled once per class so the
  predicted class mix moves towards the training class mix (a label-free,
  deterministic test-time step; the test set is ~10 clips per class). 17 of
  405 predictions change; the one class #13 never predicted gets clips again.
- **Why:** it gained +0.5 points on 4 of 5 held-out folds earlier, and at a
  crowded cutoff a single clip matters.
- **Result:** LB 0.711 → **0.716** (+1 clip) — best so far, a valid package,
  and above the 0.71144 Distinction cutoff seen on the leaderboard.
- **Next:** the rebalancing step goes into the package itself (class mix
  stored in the checkpoint, applied by `infer.py`), so the pick is
  reproducible.
- **Done:** `kaggle_ops.py fuse --weights-json … --balance-steps 1` rebuilds
  #15 from its members exactly (0 of 405 predictions differ) and writes the
  92.9 MB checkpoint `runs/model_day4f_student_balanced.pth`, whose metadata
  carries the class prior and the rebalancing step that `infer.py` applies.
- **Round 2 running on Kaggle** (pushed 16:31 UTC, two 144 px students).
  The stacked teacher gives the student only a small weight (0.06 vs 0.86 on
  fold 0) and agrees with the nine-stream fusion on 97% of test clips; it
  labels 219 test clips at p ≥ 0.6.

### Investigated and not used: recording order of test clips

- **Idea (from the Large Model Track session):** frame filenames carry
  recording timestamps, so clips can be ordered into recording sessions and
  neighbours' labels exploited.
- **Measured on training data:** the protocol is scripted — within a session,
  consecutive clips are *different* actions (1.2% share a label vs 2.5% by
  chance), in the same order every repetition block. Order would therefore
  carry real label information.
- **Organisers' position:** the forum's Leakage Report (topic 714827) showed
  test labels recoverable by matching skeleton filename timestamps; the
  organisers took the source repository offline, pointed to anti-cheating
  mechanisms, and said Stage 2 favours "genuine progress in solving the HAR
  cross-subject challenge".
- **Decision:** not used. Scores should come from the sensor data, and a
  timestamp-based step would mirror the leak they closed.

### Round 2 finished (19:43 UTC)

- **Self-trained student, 144 px:** full data + 219 test clips pseudo-labelled
  by the stacked teacher; 163 min on a T4.
- **Distilled student, 144 px:** trained on the #11 ensemble's probabilities
  for all 3,336 clips (out-of-fold for training clips, ensemble for test
  clips); 190 min.
- Both go into the #15 package slot (with #12's weights and one rebalancing
  step) for tomorrow's first two submissions. GPU this week: ~18 of 30 h.
- **Queued for 00:02 UTC Sep 14** (the day's limit was reached): #16 = the
  self-trained 144 px student in the #15 package; #17 = the distilled student
  in the same slot. Both keep #12's weights and one rebalancing step.
- **Prepared for the next slots:** (a) the distilled student is much less
  confident (mean top probability 0.67 vs 0.89 for the self-trained one), and
  #12's slot weight was fitted for a sharper model. Matching its sharpness to
  the 144 px model's test confidence gives slot weight 0.78 instead of 0.40.
  (b) The self-trained 144 px package with two rebalancing steps. (c) #15
  with two rebalancing steps (6 predictions change, all 40 classes predicted).

### Incident — package builder dropped the big model (caught before any submission)

- **What happened:** building the round-2 packages with `kaggle_ops.py fuse`
  silently left out the R(2+1)D-34 stream. The fusion is assembled from
  out-of-fold predictions, and the round-2 students are full-data models with
  none. The result was a 21.9 MB "package" without the big model (141 of 405
  predictions off #15). The builder copied it over the two files queued for
  the 00:02 UTC submissions.
- **Caught:** by the size and the prediction-difference check, before
  anything was submitted. The queued files were rebuilt from the saved
  fused probabilities, and they match the earlier numbers exactly (56 and 69
  predictions differ from #15). The wrong artifacts were deleted.
- **Fix:** `scripts/pack_fixed.py` takes the weights as given, so a member
  without held-out predictions cannot fall out of its slot. It must rebuild
  #15 exactly before building the round-2 packages, and every package's CSV
  must equal its queued submission file.
- **Verified (19:51 UTC):** `pack_fixed.py` rebuilds #15 exactly (same
  members, weights, detector, rebalancing; 0 of 405 predictions differ). The
  round-2 packages — self-trained, distilled, and distilled with matched slot
  weight — are 92.9 MB each, and each one's predictions equal its queued
  submission file (0 differences). The two-step rebalancing variants also
  match.
- **Sep 14 queue (all valid packages):** 00:02 UTC #16 self-trained 144 px,
  #17 distilled; from 00:10 UTC #18 distilled with matched weight, #19 #15
  with two rebalancing steps, #20 self-trained 144 px with two steps.

## Day 5 — Sep 14

### Round-2 results (00:06 UTC)

- **#16 self-trained 144 px student: 0.71641.** Ties #15, although 56 of 405
  predictions differ: two equally strong, fairly different models.
- **#17 distilled student: 0.65174.** 13 public clips behind #15. Training on
  the ensemble's soft probabilities gave a less confident and weaker member, so
  it is out of the running for the final picks.
- **Queue change:** the distilled student with a matched slot weight (planned
  #18) was cancelled before it fired. Re-weighting the same member cannot
  plausibly win back 13 clips, so its slot is kept for a stronger candidate.
  The two rebalancing probes go in now: #18 = #15 with two rebalancing steps,
  #19 = #16 with two steps.

### Rebalancing probes (00:13 UTC)

- **#18 = #15 with two rebalancing steps: 0.71144** (1 public clip below #15;
  only 6 predictions changed).
- **#19 = #16 with two rebalancing steps: 0.70149** (3 clips below #16).
- **Conclusion:** one step is the right strength. A second step pushes the
  predicted class mix too hard towards the training mix. Every remaining
  candidate keeps one step.

### Weight-averaging check (kernel pushed 00:28 UTC)

- **Why:** #15's student (128 px) and #16's student (144 px) score the same
  but disagree on 56 fused predictions. That is where an ensemble gains; on
  fold 0 the two teacher-era models (0.644 and 0.655) ensemble to 0.670. Two
  R(2+1)D-34s do not fit in 100 MB, but the students share their IG-65M
  initialisation and recipe, so their weight average may keep part of the
  gain in one model.
- **Risk:** averaging fails when the two sit on opposite sides of a loss
  barrier. Their weights differ by 28% of their norm, so this is tested
  rather than assumed.
- **Test (inference only):**
  - Phase A: average the fold-0 models and score the averages on the fold-0
    held-out users, with averaged and with re-estimated BatchNorm statistics,
    against the endpoints and their prediction ensembles.
  - Phase B: the full-student average — its fit on training clips, and test
    probabilities with the students' own three-view protocol, checked
    against their saved predictions.
- **Rule for use:** only if the fold-0 averages beat their endpoints.

### Weight-averaging results (kernel finished 00:36 UTC, 7 min on two T4s)

- **Held-out users (fold 0, 539 clips): averages behave like prediction
  ensembles.**
  - a (128 px, 0.644) and b (144 px, 0.655) average to 0.674 at 144 px; their
    prediction ensemble scores 0.668.
  - With the stronger self-trained p (0.712), averages land near p: p+b
    0.703–0.705, a+p 0.714 (ensembles 0.701 and 0.711).
  - Averaged BatchNorm statistics did as well as or better than
    re-estimated ones. 144 px gave the lowest loss (NLL 1.48 for a+b vs 1.62
    at 128 px).
- **Full students, no loss barrier:** the average keeps 98.4% training fit
  (students 98.9%, NLL 0.057 vs 0.03).
- **Protocol check:** the kernel reproduces both students' saved test
  probabilities (100% top-1 agreement, max difference 0.004), so the average's
  test probabilities are directly comparable.
- **Candidate:** the average at 144 px with averaged BatchNorm statistics.
  - It agrees with the two students' prediction ensemble on 91% of test
    clips; the students agree with each other on 85%.
  - Packaged at 92.9 MB (9 members + fp16 detector). Its fused predictions
    differ from #15 on 40 clips and from #16 on 36.
  - Submitted as #20 in today's last slot.

### #20 result (00:40 UTC)

- **#20 weight-averaged students: 0.71144.** One public clip below #15/#16. The
  public set has 201 clips, so the standard error is about 6 clips and this
  is noise. The held-out-user evidence still favours the average.
- **Day 5 slots used:** 5 of 5.
- **Next (GPU, within half of the remaining weekly hours):**
  - Averaging gains most from comparably strong members, so train two more
    self-trained students with the same initialisation and recipe (144 px
    and 128 px). Their pseudo-labels come from the #20 fusion.
  - Then average all four students, and check on held-out users whether the
    weaker teacher-era models help as well.
- **Round-3 pseudo-labels:** built from the #20 fusion (its probabilities
  reproduce the #20 submission exactly). 268 of 405 test clips reach
  p >= 0.6, against 216 and 219 in rounds 1 and 2, and every clip the older
  sets also labelled keeps the same label. Saved as `kaggle/pl3_test.npz`.

### Round 3 training (pushed 00:47 UTC, ~3 h on two T4s)

- **Jobs:** two more self-trained R(2+1)D-34 students with round 2's exact
  recipe and IG-65M initialisation: R3A at 144 px and R3B at 128 px, both on
  the round-3 pseudo-labels.
- **Why two:** averaging gains most from comparably strong members. The data
  order differs from rounds 1–2 (a different pseudo-label count), the same kind
  of difference under which P and R averaged without a loss barrier.
- **Next:** one inference kernel —
  - Fold 0: does averaging weaker models into a strong one help (a+b+p)?
  - Full data: the four-student average, and a variant with the teacher-era
    models added at half weight. Fit check and three-view test predictions.
- **Peer:** the Large Model Track session confirmed it needs no GPU now.
- **Round 3 finished (03:45 UTC):** R3A (144 px) in 175 min and R3B (128 px) in
  148 min, each trained with the 268 round-3 pseudo-labels. GPU this week:
  about 21 h of 30.
- **Diagnostics on test clips:**
  - Every pair of the four students (P, R, R3A, R3B) agrees on 85–88% of
    clips, so the new students add as much diversity as P and R have between
    them.
  - The new students are a little more confident (mean top probability 0.90–0.91
    vs 0.87–0.89), which fits their larger pseudo-label set.
  - Alone in the package slot, R3A's submission differs from #15/#16 on 56/49
    clips and R3B's on 49/43.
  - The four-student prediction ensemble (too big to package; reference only)
    differs from #15 on 35 clips, #16 on 36 and #20 on 26.
- **Averaging kernel pushed 03:48 UTC.**

### Averaging kernel 2 results (finished 03:52 UTC, 2.5 min)

- **Held-out users (fold 0): weaker models do not add accuracy.** Averaging
  the teacher-era a (0.644) and b (0.655) into the self-trained p (0.712)
  gives 0.699–0.705 uniformly and 0.709–0.712 with p counted twice. Log-loss
  improves (1.41–1.60 vs 2.10 for p), and the averages again track their
  prediction ensembles (0.701 / 0.707) within a clip.
- **Consequence:** averaging pays when members are comparably strong, so the
  four-student average (S4) is the candidate. The variant adding the
  teacher-era models at half weight (S6) is not used.
- **Training fit:** S4 98.0% (NLL 0.08); S6 97.0% (NLL 0.10); the
  two-student average was 98.4%. Only mild smoothing, as expected for
  averages — no loss barrier.
- **S4 package:** `runs/model_soup_S4_b1.pth`, 92.9 MB (9 members + fp16
  detector). On test clips S4 agrees with the four-student prediction ensemble
  on 93.3% of clips (the two-student average: 91.1% with its pair). Its
  submission differs from #15 on 47 clips, #16 on 50 and #20 on 24.

### Sep 15 queue (submitter fires 00:02 UTC; each CSV only if its package exists)

- **#21 S4:** the four-student average — the main candidate.
- **#22 R3A alone (144 px)** and **#23 R3B alone (128 px):** do the larger
  round-3 pseudo-label set and the extra training make a stronger single
  student? Also diverse final-pick options.
- **#24 S6:** four students + teacher-era models at half weight. On held-out
  users this kind of average added no accuracy but much better calibration,
  which can matter inside the geometric fusion.
- **#25:** decided in the morning from #21–#24.
- **Final picks:** recommended to Santanu by ~12:00 UTC, before the 15:55 UTC
  deadline. If the queued submitter dies (session restart), submit the same
  files by hand in the same order.
- **Packages verified (04:05 UTC):** R3A, R3B and S6 packages are 92.9 MB
  each. Every package's own CSV equals its queued submission file (0
  differences). The big-model slot records the right input size (R3A 144 px,
  R3B 128 px, S4/S6 144 px).

## Day 6 — Sep 15 (final day)

- **The queued submitter never fired.** The laptop and session were off
  overnight, and the job died (exit code 4) before 00:02 UTC. The five
  submissions went in by hand at 15:47–15:49 UTC, minutes before the 15:55 UTC
  deadline.
- **Scores:**
  - #21 S4, four-student average: 0.71641
  - #22 R3A alone (144 px): 0.71641
  - #23 R3B alone (128 px): 0.70646
  - **#24 S6, four students + teacher-era models at half weight: 0.72636 —
    best public score.**
  - #25 distilled student with matched slot weight: 0.66169
- **Reading:** on held-out users the six-model kind of average was the
  best-calibrated (log-loss 1.41 vs 2.10 for the strongest single model),
  without extra accuracy. Calibration matters inside the geometric fusion,
  which fits S6 beating every single student and S4 by 2 clips.
- **Final picks recommended (15:50 UTC):**
  - #24 S6 (0.72636).
  - #16, the round-2 student (0.71641). Among the 0.71641 packages it
    differs most from S6 (55 predictions), which gives the private
    leaderboard's better-of-two rule the most room.
  - Both are valid 92.9 MB packages.

### Final selection review (16:05 UTC Sep 15)

- **Correction:** Kaggle closes at **23:59 UTC Sep 15**, not 15:55 UTC. The
  day's five submissions were already used, and the daily reset falls after
  the close, so no further submissions are possible.
- **Selection state:** Santanu's saved Submissions page (15:53 UTC) showed 0/2
  selected.
- **Public leaderboard (326 teams):** InSociEUP is rank 79 at 0.72636.
  - Top 15: 0.87562.
  - Excellence (top 15%, rank 48): 0.78109.
  - Distinction (top 30%, rank 97): 0.71144, 3 clips below us, with 95 teams
    at 0.71641 or better.
  - The private half (~204 clips) can move a submission by several points,
    so the Distinction margin is not safe with one pick.
- **Predictions that differ from #24 (of 405):**
  - #21 S4: 9
  - #20: 27
  - #23: 41
  - #22 R3A: 45
  - #15 P: 50
  - #16 R: 55
- **Recommended final picks:** #24 (best public; best-calibrated average on
  held-out users) and #16 (0.71641, the most different strong package).
  - Auto-select is not recommended: it would pair #24 with an unknown one of
    four 0.71641 ties, which could be #21 — nearly a copy of #24.
- **Reproduction Setup Form (topic 740972):** unscored, with no deadline. It
  only matters for teams re-run in the Selection Stage (private Top 15), so it
  is optional for InSociEUP. Draft answers were given to Santanu.
- **Official registration:** InSociEUP was not on the organisers' Sep 14 list
  of Kaggle teams without a matching registration.

## Final result (private leaderboard, read 2026-09-18)

- **InSociEUP: private 0.75980, rank 60 of 326 (top 18.4%) — Distinction tier.**
  Public was 0.72636 at rank 79. Excellence needed top 15%: rank 48, 0.76960.
- **Private scores of the day-6 submissions:**

  | Submission | Public | Private | Private rank it would give |
  |---|---|---|---|
  | #22 R3A student alone | 0.71641 | **0.77450** | 46 — Excellence |
  | #24 S6 six-model average | 0.72636 | 0.75980 | 56–60 (counted) |
  | #21 S4 four-student average | 0.71641 | 0.75980 | same |
  | #20 two-student average | 0.71144 | 0.75490 | |
  | #23 R3B student alone | 0.70646 | 0.74509 | |
  | #16 round-2 student | 0.71641 | 0.74019 | 78 |
  | #25 distilled + matched weight | 0.66169 | 0.74019 | |
  | #17 distilled student | 0.65174 | 0.68627 | |

- **The selection cost one tier.** I recommended #24 + #16 and both were
  beaten by #22, which was never in the running because its public score tied
  with three others at 0.71641 while #24 led by two clips. Picking #22 would
  have been Excellence.
- **What the numbers say in hindsight:**
  - Public and private disagreed at the scale the decision needed. The public
    spread of the six candidate packages was four clips; the private spread
    was seven, and the ordering barely correlated.
  - The held-out-user evidence did point the right way on averaging (the
    averages beat the single round-1/round-2 students privately, 0.7598 and
    0.7549 vs 0.7402), but the newest single student beat everything, and no
    cross-subject number existed for it — the round-3 students were trained on
    all users, so they had no held-out fold.
  - The honest lesson: when a candidate has no validation number of its own,
    a two-clip public lead is not evidence. Pairing two *diverse* picks was the
    right instinct; the pair should have spanned the *model families*
    (an average and the newest student) rather than two neighbours in the
    public ranking.
