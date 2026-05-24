# Validation Report — Wave 1

> **Status: not yet validated on real data.**
> The auto-metrics block below is empty until you train on captured clips. Run `ml/eval.py` to populate it. The narrative sections around the block are hand-edited and preserved across re-runs.

## Pilot scope

This report covers the controlled production pilot of the ASL Learning with Computer Vision system. Scope and constraints from the rubric:

- **Vocabulary trained:** 25 signs from [`content/wave1_signs.csv`](../content/wave1_signs.csv), drawn from a 103-sign reference vocabulary in [`content/vocabulary.csv`](../content/vocabulary.csv).
- **Vocabulary prompted (reference only):** the remaining 78 signs are shown to learners with handshape/movement/location references but are not evaluated by the model.
- **Browser-first inference:** all recognition runs locally via ONNX Runtime Web. Raw video never leaves the device.

## Model approach

- **Architecture:** [`SignClipCNN3D`](../ml/model.py) — three Conv3d blocks (32→64→128 channels) with BatchNorm + ReLU, followed by a fixed-kernel AvgPool3d(3,8,8) and a 256-unit fully-connected head. ~3.57M parameters total.
- **Training from scratch:** all weights initialized with Kaiming-normal. No pretrained backbones, no pretrained landmark/pose detectors, no pretrained sign classifiers. See [`NO_PRETRAINED_MODELS.md`](NO_PRETRAINED_MODELS.md) for the attestation.
- **Input:** 24 RGB frames at 160×160, captured over a 2-second window (≈12 fps effective). Center-cropped to a square; pixel values normalized to [0, 1].
- **Loss / optimizer:** cross-entropy, Adam at lr=1e-3, batch size 16. Trained for [N] epochs (fill in after run).
- **Augmentations applied during training:** _none in the v1 pipeline._ If signer-disjoint accuracy is weak, add horizontal flip, temporal jitter ±2 frames, and ±15% brightness/contrast (see [`ml/dataset.py`](../ml/dataset.py)).

## Dataset

| Property | Value |
|---|---|
| Signers | [fill: e.g. 2 (signer_a, signer_b)] |
| Clips per sign per signer | [fill: target 30] |
| Total clips | [fill] |
| Train/Val/Test split | Signer-disjoint when ≥3 signers; otherwise random 70/15/15 within signers |
| Lighting conditions | [fill: e.g. "single overhead room light, daytime sessions only"] |
| Background | [fill: e.g. "plain off-white wall, no motion behind signer"] |
| Capture sessions | [fill: e.g. "3 sessions across 2 days"] |

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

**Model version:** _not yet trained_
**Test accuracy (clip-level, signer-disjoint):** _pending_
**Classes:** 25 (planned)

### Per-class metrics

_Populated by eval.py after training._

### Most-confused pairs (top 10)

_Populated by eval.py after training._

<!-- AUTO-METRICS:END -->

## Confidence calibration

After training, attach a histogram of confidence values bucketed by correct/incorrect prediction (manual analysis in a notebook is fine). If the model is overconfident on errors (a common failure mode of from-scratch CNNs on small datasets), consider:

- Raising the pass threshold to 0.95
- Adding temperature scaling in `ml/eval.py` before the threshold check
- Collecting more clips for whichever class drives the overconfident errors

## Known limitations

These are anticipated even before training; update with concrete observations after the first eval run.

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
| Report version | 0.1 (skeleton) |
| Last metrics run | _pending first training_ |
| Trained checkpoint | _pending_ |
| Exported ONNX | _pending_ |
| Reviewer | _name + date_ |
