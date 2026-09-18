# Draft forum post (ready to paste)

Post it at
<https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion>
→ **New Topic**. Kaggle's API cannot create discussion topics, so this has to
be pasted by hand.

---

**Title:** Full solution open-sourced: code, weights and write-up (private 0.75980, rank 60/326)

**Body:**

Sharing everything from team InSociEUP, in case it's useful to others working
under the 100 MB rule.

- **Code, run logs, every submission CSV, `inference.sh`, and a decision record:**
  https://github.com/SanTanBan/cuhkx-small-model-track
- **Checkpoints (two 92.9 MB packages):**
  https://www.kaggle.com/datasets/santanubanerjee9/cuhkx-small-model-track-insocieup
- **Notebook that opens the package and shows what's inside:**
  https://www.kaggle.com/code/santanubanerjee9/cuhk-x-small-model-track-insocieup-solution

**The solution.** Nine members fused as a weighted geometric mean, packed into
one 92.9 MB checkpoint: R(2+1)D-34 (IG-65M → Kinetics) on 4-channel Depth+IR
person crops, six ST-GCN skeleton folds across two pose normalisations, a
Thermal S3D, an IMU 1-D ResNet, and the SSDlite320 detector that produces the
person boxes. Members int8, detector fp16, fusion weights fitted on
out-of-fold predictions only.

**What moved the score:**

1. Person crops from a small detector on IR, applied to the pixel-aligned
   depth and IR frames.
2. Self-training: test clips the ensemble labelled at p ≥ 0.6 joined training.
   A held-out-user fold went 0.644 → 0.712, the single biggest gain.
3. Weight averaging ("model soup"): several students fine-tuned from the same
   initialisation, averaged into one model. That is how you get an ensemble's
   gain while staying under 100 MB. Checked on held-out users before use —
   averages matched their prediction ensembles, and beat their parents when
   the parents were comparably strong.
4. One Sinkhorn step of class rebalancing towards the training class mix. A
   second step always lost clips.

**What did not work:** distillation into a student (private 0.686, our worst
late submission), int6 quantisation (flipped four of 32 test predictions
against int8's one), and refitting the fusion weights on a single held-out
fold (no change).

**One thing we deliberately did not do:** frame filenames carry recording
timestamps, and the capture protocol is scripted, so ordering test clips would
leak labels. The hosts had already treated timestamp matching as a leak
(discussion 714827), so we left it alone. Everything above comes from the
sensor data.

**The mistake worth sharing.** Our final picks were the six-model average
(private 0.75980) and a single earlier student. The best submission we owned
was a round-3 student on its own, private 0.77450, which would have ranked 46
instead of 60. It tied three other submissions on the public split, so it never
looked special — and because it was trained on all users, it had no
cross-subject score of its own to argue with. On a 201-clip public split, a
two-clip lead is noise; if a candidate has no held-out number, don't let public
rank stand in for one.

Happy to answer questions about any part of it.
