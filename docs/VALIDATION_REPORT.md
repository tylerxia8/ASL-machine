# Validation Report — Wave 1

> **Status: validated on the `wave1-semlex-aslcitizen-wlasl-v27-v23recipe` release.**
> The auto-metrics block below was populated by the GitHub Actions training workflow from the release checkpoint. This is the current bundled model; use the limitations section to understand weak signs and operating constraints.

## Pilot scope

This report covers the controlled production pilot of the ASL Learning with Computer Vision system. Scope and constraints from the rubric:

- **Vocabulary trained:** 25 signs from [`content/wave1_signs.csv`](../content/wave1_signs.csv), drawn from a 103-sign reference vocabulary in [`content/vocabulary.csv`](../content/vocabulary.csv).
- **Vocabulary prompted (reference only):** the remaining 78 signs are shown to learners with handshape/movement/location references but are not evaluated by the model.
- **Browser-first inference:** all recognition runs locally via ONNX Runtime Web. Raw video never leaves the device.

## Model approach

- **Architecture:** hand-landmark TCN (`hand_landmark_tcn`) trained by [`ml/train_landmarks.py`](../ml/train_landmarks.py). It consumes 24 frames of MediaPipe Hands landmarks, using 132 landmark features per frame.
- **Training from scratch:** the recognition network weights are initialized and trained from scratch. MediaPipe Hands is used only as the on-device hand landmark detector; no pretrained sign classifier or pretrained sign-language model is used. See [`NO_PRETRAINED_MODELS.md`](NO_PRETRAINED_MODELS.md) for the attestation.
- **Input:** 24 hand-landmark frames extracted from letterboxed video clips.
- **Loss / optimizer:** cross-entropy, Adam-family training from scratch through the Wave 1 GitHub Actions workflow. The release metadata reports checkpoint validation accuracy of 85.71%.
- **Augmentations applied during training:** hand-landmark Gaussian noise (`0.02`) and frame dropout (`0.10`) in the landmark training pipeline.

## Dataset

| Property | Value |
|---|---|
| Signers | Sem-Lex signer IDs plus ASL Citizen/WLASL supplement signer IDs |
| Clips per sign per signer | Variable from Sem-Lex availability |
| Total clips | 2,132 clips in the v27 manifest: 954 train, 154 val, 1,024 signer-disjoint test |
| Train/Val/Test split | Signer-disjoint when ≥3 signers; otherwise random 70/15/15 within signers |
| Lighting conditions | Sem-Lex/ASL Citizen/WLASL source conditions vary |
| Background | Source dataset conditions vary |
| Capture sessions | Source dataset sessions plus local learner smoke/calibration clips |

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

**Model version:** `wave1-semlex-aslcitizen-wlasl-v27-v23recipe`
**Test accuracy (clip-level, signer-disjoint):** 81.84%
**Classes:** 25
**Checkpoint val accuracy:** 0.8571428571428571
**Confusion matrix shape:** 25×25

### Per-class metrics

| Sign | Precision | Recall | F1 | Support |
|------|-----------|--------|------|---------|
| deaf | 0.79 | 0.71 | 0.75 | 62 |
| eat | 0.93 | 0.87 | 0.90 | 75 |
| five | 0.00 | 0.00 | 0.00 | 1 |
| four | 0.25 | 0.33 | 0.29 | 3 |
| friend | 0.91 | 0.89 | 0.90 | 65 |
| goodbye | 0.50 | 0.77 | 0.61 | 13 |
| hello | 0.77 | 0.74 | 0.76 | 23 |
| help | 0.89 | 0.78 | 0.83 | 86 |
| how | 0.55 | 0.86 | 0.67 | 14 |
| meet | 0.81 | 0.76 | 0.79 | 34 |
| name | 0.92 | 0.96 | 0.94 | 51 |
| nice | 0.84 | 0.86 | 0.85 | 69 |
| no | 0.86 | 0.64 | 0.74 | 39 |
| one | 0.95 | 0.88 | 0.91 | 40 |
| please | 1.00 | 0.71 | 0.83 | 14 |
| sleep | 0.91 | 0.84 | 0.88 | 58 |
| sorry | 0.94 | 0.97 | 0.96 | 33 |
| thank_you | 0.67 | 0.95 | 0.78 | 19 |
| three | 0.53 | 0.90 | 0.67 | 10 |
| two | 0.95 | 0.73 | 0.83 | 26 |
| water | 0.94 | 0.99 | 0.96 | 69 |
| what | 0.82 | 0.74 | 0.78 | 54 |
| where | 0.73 | 0.89 | 0.80 | 79 |
| who | 0.65 | 0.48 | 0.55 | 54 |
| yes | 0.57 | 0.88 | 0.69 | 33 |
| **macro avg** | 0.75 | 0.76 | 0.75 | 1024 |

### Most-confused pairs (top 10)

| True → | Predicted | Count |
|--------|-----------|-------|
| who | where | 16 |
| help | nice | 8 |
| deaf | who | 8 |
| who | yes | 7 |
| no | yes | 6 |
| deaf | where | 6 |
| sleep | eat | 5 |
| what | yes | 4 |
| nice | help | 4 |
| help | what | 4 |

### Confidence calibration

| Confidence | Clips | Correct | Accuracy |
|------------|-------|---------|----------|
| 0.0-0.1 | 0 | 0 | n/a |
| 0.1-0.2 | 4 | 1 | 25.00% |
| 0.2-0.3 | 20 | 5 | 25.00% |
| 0.3-0.4 | 30 | 5 | 16.67% |
| 0.4-0.5 | 35 | 14 | 40.00% |
| 0.5-0.6 | 52 | 24 | 46.15% |
| 0.6-0.7 | 48 | 34 | 70.83% |
| 0.7-0.8 | 51 | 35 | 68.63% |
| 0.8-0.9 | 103 | 77 | 74.76% |
| 0.9-1.0 | 681 | 643 | 94.42% |

<!-- AUTO-METRICS:END -->

## Confidence calibration

The v27 release includes per-clip confidence values. The 0.9-1.0 confidence bin is accurate on 94.42% of held-out clips, which supports a strict pass threshold in the browser. Mid-confidence bins are less reliable and should continue to produce retry/fail guidance instead of automatic success.

If the model is overconfident on errors, consider:

- Raising the pass threshold to 0.95
- Adding temperature scaling in `ml/eval.py` before the threshold check
- Collecting more clips for whichever class drives the overconfident errors

## Known limitations

Concrete observations from `wave1-semlex-aslcitizen-wlasl-v27-v23recipe`:

- **Overall recognition is substantially better than the earlier RGB model family.** The v27 hand-landmark model reaches 81.84% signer-disjoint clip-level accuracy, macro F1 0.745, and weighted F1 0.819.
- **`five` remains a data gap.** It has only one held-out test clip and 0.00 F1, even after the small WLASL direct-MP4 supplement. It should stay in the learning content, but recognition results for `five` should be treated conservatively until more signer-diverse clips are available.
- **`four` and `who` are the main weak recognition classes.** `four` has only three test clips and 0.29 F1; `who` has 0.55 F1 and is most often confused with `where`.
- **Highest observed confusion pattern:** `who -> where` remains the largest pair, followed by `help -> nice`, `deaf -> who`, and `who -> yes`.

General limitations:

- **Small-sample generalization.** Source datasets still cannot cover the full variation of hand shapes, sizes, skin tones, camera angles, and signing styles. Held-out signer accuracy is the relevant metric, not in-signer accuracy.
- **Detector dependence.** The recognizer depends on MediaPipe Hands landmarks. If the detector misses hands because of cropping, lighting, motion blur, or occlusion, the downstream classifier will be unreliable.
- **Confusable pairs already identified in the 25-sign set:**
  - **please / sorry** ? both chest circles; differ only in handshape (flat-B vs S-fist).
  - **what / where** ? both shaking motions; differ in handshape (5-open vs 1-index).
  - **four / five** ? only the thumb position differs.
  - **one / two / three** ? static handshapes; sensitive to finger precision.
- **Operating envelope.** Accuracy is documented under the conditions in [`CONTROLLED_CONDITIONS.md`](CONTROLLED_CONDITIONS.md). Performance outside that envelope (poor lighting, off-axis camera, hands cropped) is undefined.

## Privacy

See [`PRIVACY.md`](PRIVACY.md). Inference is browser-local. The API receives only `{sign_id, outcome, confidence, predicted_label}` per attempt ? no images, no audio, no biometric features. Raw `.webm` captures stay on the data-collector's local disk and are not transmitted by the practice flow.

## Pretrained-model attestation

See [`NO_PRETRAINED_MODELS.md`](NO_PRETRAINED_MODELS.md). Grep evidence and dependency audit demonstrate the model is trained from scratch end-to-end.

## Sign-off

| Field | Value |
|---|---|
| Report version | 0.3 (`wave1-semlex-aslcitizen-wlasl-v27-v23recipe`) |
| Last metrics run | 2026-05-31 release artifact |
| Trained checkpoint | GitHub Actions release `wave1-semlex-aslcitizen-wlasl-v27-v23recipe` |
| Exported ONNX | `apps/web/public/models/model.onnx` and release asset `wave1-semlex-aslcitizen-wlasl-v27-v23recipe/model.onnx` |
| Reviewer | _name + date_ |
