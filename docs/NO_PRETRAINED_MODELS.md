# No Pretrained Models � Attestation

## Requirement

Per project rubric, the sign recognition system must **not** use pretrained weights for:

- Sign language classifiers
- Hand/pose landmark detectors
- Feature extractors
- General-purpose CV backbones (e.g., ResNet trained on ImageNet)

## This project

| Component | Approach |
|-----------|----------|
| Model | `SignClipCNN3D` in `ml/model.py` � PyTorch modules with default random initialization |
| Training | `ml/train.py` trains from scratch on team/synthetic clips |
| Inference | Exported ONNX consumed by `onnxruntime-web` in the browser |
| Input | Raw RGB frame tensors (160�160), no MediaPipe landmarks |

## Dependency audit

**ML (`ml/requirements.txt`):** torch, numpy, onnx, onnxruntime, pillow, opencv-python (I/O only), scikit-learn (metrics only).

**Explicitly excluded from pipeline:** mediapipe, ultralytics, timm, torchvision.models pretrained APIs, tensorflow hub, huggingface transformers for vision.

## Verification steps

1. Inspect `ml/model.py` � no `pretrained=True` or weight download URLs.
2. Run `grep -r "pretrained\|mediapipe\|torchvision.models" ml/` � should return no production usage.
3. Training logs record `model_version` and initialization note in `model_meta.json`.

## Frameworks allowed

PyTorch, ONNX, ONNX Runtime Web, OpenCV for resize/color � processing only, not recognition weights.
