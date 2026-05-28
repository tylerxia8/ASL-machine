# Product Model Pivot

The original rubric-strict model path avoided pretrained detectors and trained
directly on raw RGB clips. That path now has enough evidence to call it an
integration/demo model, not a working learner-assessment model.

## Decision

For product-quality recognition, this project now allows a pretrained hand
landmark detector:

- Training extraction: MediaPipe Hands in `ml/scripts/extract_hand_landmarks.py`
- Browser extraction: `@mediapipe/tasks-vision` in `apps/web/src/lib/handLandmarks.ts`
- Classifier: a project-owned ONNX temporal classifier trained on hand-landmark
  sequences, exported with `input_type: "hand_landmarks"`

This means the classifier is still ours, but the hand detector is pretrained.
That relaxes the old "no pretrained landmark detectors" rubric constraint.

## Why

The RGB-only Sem-Lex runs did not generalize well to signer-disjoint evaluation:

- Best bundled integration model: `wave1-semlex-full-v8`, 14.09% accuracy
- Best macro-F1 diagnostic run: `wave1-semlex-full-v17-motion-tcn-fixed15`,
  5.70% accuracy, macro F1 0.04030

The failure mode changed across architectures and preprocessing, but the
underlying issue remained: the model was spending too much capacity on camera,
background, framing, lighting, and signer appearance.

Hand landmarks remove much of that nuisance variation and feed the classifier
the geometry most relevant to beginner Wave 1 signs.

## Current Implementation Status

Implemented:

- Landmark feature extraction from manifest clips
- Landmark TCN training script
- Landmark evaluation script
- Landmark ONNX export script
- Train workflow branch for `model_size=hand_landmark_tcn`
- Browser-side MediaPipe hand-landmark inference path
- CI smoke coverage for landmark ONNX export and numerical equivalence

Still needed:

- Run the first full Sem-Lex landmark training job
- Compare signer-disjoint accuracy against v8
- If better, sync the release assets into `apps/web/public/models/`
- Run the Wave 1 dry run with recognition demo enabled

## Important Caveat

MediaPipe hand landmarks may miss hands when signing is fast, cropped, or
motion-blurred. If landmark frame coverage is low, the extractor fails instead
of training on mostly-empty features.
