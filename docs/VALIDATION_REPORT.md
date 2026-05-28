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

**Model version:** `wave1-semlex-full-v18-hand-landmarks`  
**Test accuracy (clip-level, signer-disjoint):** 78.31%  
**Classes:** 25  
**Checkpoint val accuracy:** 0.8803418803418803  
**Confusion matrix shape:** 25×25

### Per-class metrics

| Sign | Precision | Recall | F1 | Support |
|------|-----------|--------|------|---------|
| deaf | 0.47 | 0.68 | 0.55 | 31 |
| eat | 0.92 | 0.87 | 0.89 | 38 |
| five | 0.00 | 0.00 | 0.00 | 1 |
| four | 0.00 | 0.00 | 0.00 | 2 |
| friend | 0.79 | 1.00 | 0.88 | 23 |
| goodbye | 0.30 | 0.75 | 0.43 | 8 |
| hello | 0.70 | 0.50 | 0.58 | 14 |
| help | 0.89 | 0.88 | 0.88 | 48 |
| how | 0.50 | 0.43 | 0.46 | 7 |
| meet | 1.00 | 0.85 | 0.92 | 20 |
| name | 0.97 | 0.97 | 0.97 | 30 |
| nice | 0.90 | 0.82 | 0.86 | 33 |
| no | 0.70 | 0.64 | 0.67 | 22 |
| one | 0.94 | 0.89 | 0.91 | 18 |
| please | 0.75 | 0.75 | 0.75 | 8 |
| sleep | 0.85 | 0.92 | 0.88 | 25 |
| sorry | 1.00 | 0.94 | 0.97 | 18 |
| thank_you | 0.50 | 0.89 | 0.64 | 9 |
| three | 0.50 | 0.80 | 0.62 | 5 |
| two | 1.00 | 0.63 | 0.77 | 19 |
| water | 0.95 | 1.00 | 0.97 | 37 |
| what | 1.00 | 0.69 | 0.82 | 29 |
| where | 0.67 | 0.84 | 0.75 | 49 |
| who | 0.67 | 0.25 | 0.36 | 32 |
| yes | 0.71 | 0.67 | 0.69 | 18 |
| **macro avg** | 0.71 | 0.71 | 0.69 | 544 |

### Most-confused pairs (top 10)

| True → | Predicted | Count |
|--------|-----------|-------|
| who | where | 11 |
| who | deaf | 11 |
| deaf | where | 4 |
| yes | friend | 3 |
| where | goodbye | 3 |
| where | deaf | 3 |
| two | three | 3 |
| no | yes | 3 |
| nice | goodbye | 3 |
| how | help | 3 |

### Confidence calibration

| Confidence | Clips | Correct | Accuracy |
|------------|-------|---------|----------|
| 0.0-0.1 | 0 | 0 | n/a |
| 0.1-0.2 | 2 | 1 | 50.00% |
| 0.2-0.3 | 6 | 0 | 0.00% |
| 0.3-0.4 | 15 | 3 | 20.00% |
| 0.4-0.5 | 21 | 9 | 42.86% |
| 0.5-0.6 | 21 | 12 | 57.14% |
| 0.6-0.7 | 35 | 18 | 51.43% |
| 0.7-0.8 | 25 | 11 | 44.00% |
| 0.8-0.9 | 45 | 26 | 57.78% |
| 0.9-1.0 | 374 | 346 | 92.51% |

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
- **Same-data retraining did not fix collapse.** `wave1-semlex-full-v9` tied v8 at 14.09% accuracy but had slightly lower macro F1; `wave1-semlex-full-v9-small` dropped to 8.18% accuracy; `wave1-semlex-full-v10-hardened` dropped to 4.55% accuracy with macro F1 0.01141; `wave1-semlex-full-v11-lr3e4-small` dropped to 3.18% accuracy with macro F1 0.00561; `wave1-semlex-full-v12-frame` reached 4.55% accuracy with macro F1 0.01601; `wave1-semlex-full-v13-tcn` reached 6.82% accuracy with macro F1 0.02255; `wave1-semlex-full-v14-tcn-balanced-split` reached 5.70% accuracy with macro F1 0.03441 after fixing zero-test-support split coverage; `wave1-semlex-full-v15-tcn-letterbox` reached 6.07% accuracy with macro F1 0.02845; `wave1-semlex-full-v16-motion-tcn` reached 5.51% accuracy with macro F1 0.02574; `wave1-semlex-full-v17-motion-tcn-fixed15` reached 5.70% accuracy with macro F1 0.04030. V17 is the best macro-F1 result so far and has better prediction diversity, but none beat the v8 bundled accuracy. The next meaningful lever is a stronger model/data strategy and a product fallback/self-check path, not another whole-frame CNN run with minor hyperparameter changes.
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
