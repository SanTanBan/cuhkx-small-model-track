# Approach, decisions, and what they cost

A compressed record of how this solution was built, why each choice was made,
and which ones turned out wrong. [SUBMISSION_LOG.md](../SUBMISSION_LOG.md) has
the day-by-day detail; this file is the reasoning behind it.

Development used an AI coding assistant (Claude Code). The host clarified that
the "no LLMs for development" rule concerns LLMs inside the solution and
LLM-based labelling, not coding assistants (Kaggle discussion 724942). No LLM
is part of the model, and no test sample was labelled by hand or by an LLM.

## The constraints that shaped everything

- **≤100 MB for every weight loaded at inference**, ensembles included
  (host ruling, topic 729056). This is the constraint that killed the obvious
  path of "train five strong models and average their predictions".
- **Cross-subject test split.** Train users 1–9 and 16–24, test users 10–11 and
  25–26. Anything that fits a person rather than an action is wasted capacity.
- **Certificates come from the private leaderboard**; only the private Top 15
  advance to a Selection Stage where organisers re-run the code. That made
  robustness, not public rank, the target — with one important exception
  discussed in the retrospective.

## The line of work

| Step | What | Public |
|---|---|---|
| 1 | ST-GCN on 3D skeleton, one fold | 0.40298 |
| 2 | Skeleton two-stream + thermal/IR TSM-ResNet18, EM-weighted fusion | 0.53233 |
| 3 | Geometric (log-linear) fusion of six streams, weights fitted on out-of-fold predictions | 0.57213 |
| 4 | + R(2+1)D-34 (IG-65M → Kinetics) on 4-channel Depth+IR **person crops** | 0.66169 |
| 5 | Packaging discipline: int8 members, fp16 detector, one 92.9 MB file | 0.66666 |
| 6 | Nine streams incl. IMU; 144 px video model | 0.68656 |
| 7 | **Self-training round 1** (216 confident test clips as pseudo-labels) | 0.71144 |
| 8 | + one-step class rebalancing towards the training class mix | 0.71641 |
| 9 | Self-training rounds 2 and 3, and **weight averaging** of the students | 0.72636 |

Final: **private 0.75980, rank 60/326 (Distinction)**; public 0.72636, rank 79.

### Why geometric fusion

Streams disagree on which clips are hard. A log-linear pool
(p ∝ Π pₛ^wₛ) lets a confident stream veto, which matched the data better than
a linear mixture in nested cross-validation (0.615 vs 0.593). Weights were
fitted by maximising out-of-fold log-likelihood, never on the leaderboard.

### Why person crops

The 40 classes split into fine-grained hand work, confusable desk activity and
whole-body motion. Cropping to the person removes room background that differs
per subject. Boxes come from torchvision SSDlite320 run on four IR frames, and
the same box is applied to depth and IR, which are pixel-aligned.

### Why self-training worked

The test subjects are unseen, so their appearance is exactly what the model
lacks. Adding confidently-labelled test clips (p ≥ 0.6) to training moved a
held-out-user fold from 0.644 to 0.712 — the single largest gain in the
project. The organisers explicitly allow it (topic 729989). Rounds 2 and 3
added little on the public split but produced the diversity that averaging
later exploited.

### Why weight averaging instead of ensembling

Two R(2+1)D-34s do not fit in 100 MB; one does. Models fine-tuned from the same
initialisation often lie in one basin, so their weight average behaves like
their ensemble at the size of a single model. This was tested before use, on
held-out users:

- Averaging two comparably strong fold-0 models (0.644 and 0.655) gave 0.674,
  slightly better than averaging their predictions (0.668).
- Averaging a strong model with a weaker one landed between them, exactly as an
  ensemble would.
- No loss barrier: the four-student average kept 98.0% training-set accuracy
  against 98.9% for single students.
- Averaged BatchNorm statistics did as well as re-estimated ones, so no
  recalibration pass was needed.

## Ideas rejected, and why

- **Recording order from file timestamps.** Frame filenames carry timestamps,
  and the protocol is scripted: consecutive clips in a session are different
  actions (1.2% share a label vs 2.5% by chance), in the same order every
  block. Ordering test clips would therefore leak labels. The organisers had
  already treated timestamp matching as a leak (topic 714827: repository taken
  offline, "anti-cheating mechanisms"). **Not used.** Scores should come from
  the sensors.
- **int6 quantisation.** Would have freed room for a second big model, but it
  flipped four of 32 test predictions against int8's one, even with a clipping
  search for the scales. Rejected; int8 everywhere, space freed elsewhere.
- **Knowledge distillation** into a student trained on the ensemble's soft
  targets. Public 0.65174, private 0.68627 — the worst of the late
  submissions. The soft targets made a less confident, weaker member.
- **A small-backbone hedge** (no IG-65M) in case the pretrained trunk was
  ruled a "large pretrained backbone". The host ruled it permitted
  (topic 738333), so the hedge was dropped after one submission (0.60696).
- **A second class-rebalancing step.** Tried on two different packages; cost 1
  and 3 public clips. One step is the right strength.
- **Adding the weaker teacher-era models into the average at half weight.** On
  held-out users this improved calibration a lot (log-loss 2.10 → 1.48) without
  improving accuracy. It was submitted anyway because calibration matters
  inside a geometric fusion, and it did score best on the public split
  (0.72636) — but privately it tied the plain four-student average.
- **Refitting fusion weights on the student's held-out fold.** No change
  (0.71144 both ways).

## Validation methodology

- **GroupKFold on `user`**, never on clips. Every number quoted internally is
  cross-subject.
- **Fusion weights fitted only on out-of-fold predictions.** A stream with no
  out-of-fold predictions (a full-data student) cannot be weighted by fitting,
  so those packages use the previous round's weights, taken as given.
- **Package-exactness.** Every submitted CSV is the fusion of the members
  actually stored in the checkpoint, at their stored precision, so the
  leaderboard number belongs to a real 92.9 MB artefact.
- **Protocol reproduction.** Before trusting the averaged models' test
  probabilities, the inference kernel re-derived two students' saved
  predictions and matched them exactly (100% top-1 agreement).

## Engineering notes

- **All training on Kaggle** (T4 ×2, 30 GPU-h/week shared with other projects);
  the local machine is a CPU-only i3. `scripts/kaggle_ops.py` handles datasets,
  kernel generation, pushes with quota-aware retries, pulls, fusion and
  submission. One training job per GPU, both GPUs in one session.
- **A silent packaging bug cost half a day.** The fusion builder dropped any
  stream without out-of-fold predictions, which silently removed the big model
  from two packages (21.9 MB instead of 92.9 MB, 141 predictions changed). It
  was caught by a size and prediction-difference check before submission, and
  fixed with `scripts/pack_fixed.py`, which takes weights as given and cannot
  drop a member. Every package since is verified by rebuilding a known
  submission and requiring zero differences.
- **Scheduled work does not survive a laptop being shut down.** A submitter
  queued for 00:02 UTC died overnight; the day's five submissions went in by
  hand with hours to spare only because the deadline was later than believed.
- **The deadline was misread once** (15:55 UTC from the organiser site versus
  Kaggle's actual 23:59 UTC). Trust the competition's own API for the close.

## Retrospective

The final picks were #24 (six-model average, private 0.75980) and #16 (round-2
student, private 0.74019). The best submission we owned was #22, the round-3
student on its own: **private 0.77450, which would have ranked 46 — the
Excellence tier instead of Distinction.**

Why it was missed, and what to do differently:

1. **A two-clip public lead is not evidence.** The public split is 201 clips;
   the six candidate packages spanned four clips there and seven privately,
   with barely related ordering. #22 tied three others publicly, so it never
   looked special.
2. **Every new model needs its own held-out number.** The round-3 students were
   trained on all users to maximise data, so no cross-subject score existed for
   them — and the one signal that could have flagged #22 was therefore missing.
   Training one fold-holdout variant of each new recipe is cheap insurance.
3. **Spread the two final picks across model families, not across neighbours in
   the public ranking.** An average and the newest single student would have
   covered both hypotheses; two neighbours differing by 55 predictions did not.
4. Time the inference path early. It was measured only after the competition
   (13 s/clip on two CPU threads), when it was needed for the organisers' form.
