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
