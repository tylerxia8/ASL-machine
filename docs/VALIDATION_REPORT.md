# Validation Report — Wave 1

> **Status: validated on the `wave1-semlex-full-v8` release, but not yet pilot-ready.**
> The auto-metrics block below was populated by `ml/eval.py` from the release checkpoint. Accuracy is currently low, so the model should be treated as an integration/demo model until more data and retraining improve signer-disjoint performance.

## Pilot scope

This report covers the controlled production pilot of the ASL Learning with Computer Vision system. Scope and constraints from the rubric:

- **Vocabulary trained:** 25 signs from [`content/wave1_signs.csv`](../content/wave1_signs.csv), drawn from a 103-sign reference vocabulary in [`content/vocabulary.csv`](../content/vocabulary.csv).
- **Vocabulary prompted (reference only):** the remaining 78 signs are shown to learners with handshape/movement/location references but are not evaluated by the model.
- **Browser-first inference:** all recognition runs locally via ONNX Runtime Web. Raw video never leaves the device.

## Model approach

- **Architecture:** [`SignClipCNN3D`](../ml/model.py) — three Conv3d blocks (32→64→128 channels) with BatchNorm + ReLU, followed by a fixed-kernel AvgPool3d(3,8,8) and a 256-unit fully-connected head. ~3.57M parameters total.
- **Training from scratch:** all weights initialized with Kaiming-normal. No pretrained backbones, no pretrained landmark/pose detectors, no pretrained sign classifiers. See [`NO_PRETRAINED_MODELS.md`](NO_PRETRAINED_MODELS.md) for the attestation.
- **Input:** 24 RGB frames at 160×160, captured over a 2-second window (≈12 fps effective). Center-cropped to a square; pixel values normalized to [0, 1].
- **Loss / optimizer:** cross-entropy, Adam-family training from scratch through the Wave 1 GitHub Actions workflow. The release metadata reports checkpoint validation accuracy of 8.79%.
- **Augmentations applied during training:** _none in the v1 pipeline._ If signer-disjoint accuracy is weak, add horizontal flip, temporal jitter ±2 frames, and ±15% brightness/contrast (see [`ml/dataset.py`](../ml/dataset.py)).

## Dataset

| Property | Value |
|---|---|
| Signers | Sem-Lex signer IDs |
| Clips per sign per signer | Variable from Sem-Lex availability |
| Total clips | Release artifact exports 220 signer-disjoint test clips; full train/val manifest was generated in CI but not bundled in this repo |
| Train/Val/Test split | Signer-disjoint when ≥3 signers; otherwise random 70/15/15 within signers |
| Lighting conditions | Sem-Lex conditions vary |
| Background | Sem-Lex conditions vary |
| Capture sessions | Sem-Lex source sessions |

See [`CONTROLLED_CONDITIONS.md`](CONTROLLED_CONDITIONS.md) for the operational protocol.

### Per-sign Sem-Lex coverage (before training)

Counted from `semlex_metadata.csv` filtered to our Wave 1 sign roster:

| Sign | Sem-Lex clips (asllex + freetext) | Notes |
|---|---|---|
| `eat` | 597 | Mapped via `eat`, `eat_1`, `eat_2` variants |
| `water` | 408 | |
| `sleep` | 211 | |
| `nice` | 165 | |
| `friend` | 138 | |
| `help` | 136 | |
| `where` | 130 | |
| `deaf` | 103 | Mapped via `deaf`, `deaf_1`, `deaf_2` |
| `what` | 92 | Mapped via `what`, `what_1`, `what_2` |
| `name` | 86 | |
| `who` | 85 | |
| `no` | 76 | |
| `yes` | 66 | |
| `one` | 65 | |
| `two` | 49 | |
| `sorry` | 39 | |
| `thank_you` | 36 | |
| `hello` | 33 | |
| `goodbye` | 28 | Mapped via `goodbye`/`bye`/`good_bye` (Sem-Lex labels mostly `bye`) |
| `please` | 27 | |
| `meet` | 56 | |
| `how` | 21 | low |
| `three` | 21 | low |
| `four` | **8** | very low — model accuracy will be weak on this sign |
| `five` | **1** | **near-zero coverage in Sem-Lex.** Recognition will not work reliably. Treat as a training-data gap; the sign stays in the prompted roster for content completeness but the validation report below will show low recall. Mitigation: supplement with self-recorded `learner_samples` clips. |

Per-sign cap (`semlex_clips_per_sign` workflow input, default 50) trims the high-count classes to keep the dataset balanced. The above counts are *available*, not necessarily *trained-on*.

### Known infrastructure limitation: Sem-Lex Drive quota

Sem-Lex distributes the three video tarballs (`train.tar.gz` 23.7 GB, `val.tar.gz`, `test.tar.gz`) via Google Drive. Public Drive files have a per-file daily download quota; repeated dispatches of the training workflow (especially partial downloads) can exhaust it, locking the file for ~24 hours. When this happens the fetcher logs a `⚠ QUOTA on Sem-Lex {role}` warning and continues with remaining splits instead of aborting the whole run — but the trained-on dataset shrinks accordingly.

A more reliable distribution channel (Hugging Face Hub, S3 with a signed mirror, etc.) would remove this risk. For the controlled pilot we accept the quota constraint and re-train when it resets.

## Threshold policy

The browser inference layer uses three confidence bands ([`apps/web/src/lib/threshold.ts`](../apps/web/src/lib/threshold.ts)):

| Outcome | Condition |
|---|---|
| **Pass** | top-1 = prompted sign AND confidence ≥ 0.90 |
| **Retry (uncertain)** | confidence in [0.70, 0.90) — model declines to commit |
| **Fail** | top-1 ≠ prompted sign OR confidence < 0.70 |

The 0.90 pass bar is deliberately strict — Requirement 9 says "avoid marking uncertain predictions as correct." Tune downward only if validation shows the bar suppresses too many true passes.

<!-- AUTO-METRICS:START -->

> This block is overwritten by `python ml/eval.py`. Edit the narrative
> sections above/below; do not edit between these markers.

**Model version:** `wave1-semlex-full-v8`
**Test accuracy (clip-level, signer-disjoint):** 14.09%
**Classes:** 25
**Checkpoint val accuracy:** 0.08791208791208792
**Confusion matrix shape:** 25×25

### Per-class metrics

| Sign | Precision | Recall | F1 | Support |
|------|-----------|--------|------|---------|
| deaf | 0.00 | 0.00 | 0.00 | 13 |
| eat | 0.00 | 0.00 | 0.00 | 12 |
| five | 0.00 | 0.00 | 0.00 | 0 |
| four | 0.00 | 0.00 | 0.00 | 2 |
| friend | 0.00 | 0.00 | 0.00 | 19 |
| goodbye | 0.00 | 0.00 | 0.00 | 1 |
| hello | 0.00 | 0.00 | 0.00 | 3 |
| help | 0.00 | 0.00 | 0.00 | 19 |
| how | 0.00 | 0.00 | 0.00 | 4 |
| meet | 0.00 | 0.00 | 0.00 | 8 |
| name | 0.00 | 0.00 | 0.00 | 10 |
| nice | 0.00 | 0.00 | 0.00 | 15 |
| no | 0.00 | 0.00 | 0.00 | 6 |
| one | 0.00 | 0.00 | 0.00 | 7 |
| please | 0.00 | 0.00 | 0.00 | 4 |
| sleep | 0.00 | 0.00 | 0.00 | 9 |
| sorry | 0.00 | 0.00 | 0.00 | 1 |
| thank_you | 0.00 | 0.00 | 0.00 | 2 |
| three | 0.00 | 0.00 | 0.00 | 1 |
| two | 0.00 | 0.00 | 0.00 | 3 |
| water | 0.00 | 0.00 | 0.00 | 12 |
| what | 0.00 | 0.00 | 0.00 | 10 |
| where | 0.14 | 1.00 | 0.25 | 31 |
| who | 0.00 | 0.00 | 0.00 | 23 |
| yes | 0.00 | 0.00 | 0.00 | 5 |
| **macro avg** | 0.01 | 0.04 | 0.01 | 220 |

### Most-confused pairs (top 10)

| True → | Predicted | Count |
|--------|-----------|-------|
| who | where | 23 |
| help | where | 19 |
| friend | where | 18 |
| nice | where | 14 |
| deaf | where | 13 |
| water | where | 12 |
| eat | where | 12 |
| what | where | 10 |
| name | where | 10 |
| sleep | where | 9 |

<!-- AUTO-METRICS:END -->

## Confidence calibration

The `wave1-semlex-full-v8` release artifact does not include per-clip confidence values, so a calibration histogram cannot be reconstructed for that bundled model. [`ml/eval.py`](../ml/eval.py) now writes per-clip predictions plus 0.1-wide confidence bins into `eval_metrics.json` and the auto-metrics block.

The follow-up `wave1-semlex-full-v10-hardened` run produced calibration data, but it did not beat v8 and should not replace the bundled model. Its predictions were mostly low-confidence:

| Confidence bin | Clips | Correct | Accuracy |
|---|---:|---:|---:|
| 0.0-0.1 | 204 | 10 | 4.9% |
| 0.1-0.2 | 16 | 0 | 0.0% |

There were no predictions above 0.2 confidence in that run.

If the model is overconfident on errors (a common failure mode of from-scratch CNNs on small datasets), consider:

- Raising the pass threshold to 0.95
- Adding temperature scaling in `ml/eval.py` before the threshold check
- Collecting more clips for whichever class drives the overconfident errors

## Known limitations

Concrete observations from `wave1-semlex-full-v8`:

- **Current model is not pilot-quality.** Test accuracy is 14.09% with macro F1 0.01. The model collapses heavily toward `where`, so it is useful for end-to-end app integration but not reliable learner assessment.
- **Same-data retraining did not fix collapse.** `wave1-semlex-full-v9` tied v8 at 14.09% accuracy but had slightly lower macro F1; `wave1-semlex-full-v9-small` dropped to 8.18% accuracy; `wave1-semlex-full-v10-hardened` dropped to 4.55% accuracy with macro F1 0.01141; `wave1-semlex-full-v11-lr3e4-small` dropped to 3.18% accuracy with macro F1 0.00561. The next meaningful lever is deeper diagnosis and architecture/training improvement on the Sem-Lex subset, not another whole-frame CNN run with minor hyperparameter changes.
- **`five` has zero test support in this release.** Accuracy for that class is unknown until the evaluation split includes held-out clips.
- **Highest observed confusion pattern:** many classes are predicted as `where` (`who`, `help`, `friend`, `nice`, `deaf`, `water`, `eat`, `what`, `name`, `sleep`).

General limitations:

- **Small-sample generalization.** Two signers cannot cover the variation of all hand shapes, sizes, skin tones, and signing styles. Held-out signer accuracy is the relevant metric, not in-signer accuracy.
- **No pose/landmark backbone.** Whole-frame 3D CNN trained from scratch on ~1,500 clips will plateau well below research benchmarks that use pretrained MediaPipe/OpenPose features. The pilot is not classroom-assessment-grade.
- **Confusable pairs already identified in the 25-sign set:**
  - **please / sorry** — both chest circles; differ only in handshape (flat-B vs S-fist).
  - **what / where** — both shaking motions; differ in handshape (5-open vs 1-index).
  - **four / five** — only the thumb position differs.
  - **one / two / three** — static handshapes; sensitive to finger precision.
- **Operating envelope.** Accuracy is documented under the conditions in [`CONTROLLED_CONDITIONS.md`](CONTROLLED_CONDITIONS.md). Performance outside that envelope (poor lighting, off-axis camera, hands cropped) is undefined.

## Privacy

See [`PRIVACY.md`](PRIVACY.md). Inference is browser-local. The API receives only `{sign_id, outcome, confidence, predicted_label}` per attempt — no images, no audio, no biometric features. Raw `.webm` captures stay on the data-collector's local disk and are not transmitted by the practice flow.

## Pretrained-model attestation

See [`NO_PRETRAINED_MODELS.md`](NO_PRETRAINED_MODELS.md). Grep evidence and dependency audit demonstrate the model is trained from scratch end-to-end.

## Sign-off

| Field | Value |
|---|---|
| Report version | 0.2 (`wave1-semlex-full-v8`) |
| Last metrics run | 2026-05-26 release artifact |
| Trained checkpoint | GitHub Actions release `wave1-semlex-full-v8` |
| Exported ONNX | `apps/web/public/models/model.onnx` and release asset `wave1-semlex-full-v8/model.onnx` |
| Reviewer | _name + date_ |
