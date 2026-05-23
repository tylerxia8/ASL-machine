# No Pretrained Models — Attestation

## Requirement (rubric Req 7)

The sign recognition system must **not** use pretrained weights for:

- Sign language classifiers (including SemLex's SL-GCN models)
- Hand or pose landmark detectors (MediaPipe Hands/Pose, OpenPose, MMPose)
- Feature extractors
- General-purpose CV backbones (e.g., ResNet/EfficientNet trained on ImageNet, I3D trained on Kinetics)

## This project's architecture

| Component | Approach |
|-----------|----------|
| Model | [`SignClipCNN3D`](../ml/model.py) — 3-block Conv3d → AvgPool3d → FC. All weights `kaiming_normal_` initialized at instantiation. ~3.57M params. |
| Training | [`ml/train.py`](../ml/train.py) initializes a fresh model each run; no checkpoint loading from external sources. |
| Inference | Exported ONNX consumed by `onnxruntime-web` in the browser. ONNX Runtime is an *execution framework*, not pretrained weights. |
| Input | Raw RGB frame tensors at (3, 24, 160, 160) directly into the CNN. **No pose/landmark features.** |

## Dataset attestation

The Wave 1 model is trained on a combination of:

1. **Self-recorded learner clips** in [`ml/data/learner_samples/`](../ml/data/learner_samples/), captured via the in-app `/capture` flow.
2. **Sem-Lex Benchmark videos** ([leekezar/SemLex](https://github.com/leekezar/SemLex), Apache-2.0). We use **only the raw video tarballs and the metadata CSV** from this dataset:

   **Used (4 files):**
   - `semlex_metadata.csv` — index mapping each video filename to gloss / signer / split
   - `train.tar.gz` — train-split raw videos
   - `val.tar.gz` — val-split raw videos
   - `test.tar.gz` — test-split raw videos

   **Explicitly NOT used (3 files):**
   - `sem-lex-train-poses.tar.gz`
   - `sem-lex-val-poses.tar.gz`
   - `sem-lex-test-poses.tar.gz`

   These pose archives contain pre-computed body/hand landmark features extracted by a **pretrained landmark detector** — exactly what rubric Req 7 forbids. Our pipeline does not download, extract, or reference these files at any point. [`ml/scripts/fetch_semlex.py`](../ml/scripts/fetch_semlex.py) accepts only the 4 file IDs above; the SEMLEX_DRIVE_FILES JSON schema has no key for poses. The Sem-Lex repository also distributes pretrained SL-GCN models for gloss/phoneme recognition — we don't touch those either.

External video data is permitted under rubric Req 6 (which speaks of "curating or collecting" the dataset). Req 7 is specifically about pretrained model **weights**, not source videos.

## Dependency audit

Production ML stack ([`ml/requirements.txt`](../ml/requirements.txt)): `torch`, `numpy`, `onnx`, `onnxruntime`, `pillow`, `opencv-python-headless`, `scikit-learn`, `tqdm`.

**Explicitly excluded from the recognition pipeline:**

- `mediapipe`, `mediapipe-tasks` (pretrained hand/pose landmarks)
- `torchvision.models.*` with `weights=...` arguments
- `timm`, `transformers`, `tensorflow_hub`
- `ultralytics`, `mmpose`, `openpose`
- SemLex's published SL-GCN checkpoints (raw videos only)

OpenCV is used **only for video decoding and pixel resize** ([`ml/scripts/import_captures.py`](../ml/scripts/import_captures.py)), not for any pretrained DNN module. scikit-learn is used **only for the eval metrics** (`classification_report`, `confusion_matrix`).

## Verification steps

You can reproduce the attestation locally:

```bash
# 1. No pretrained imports in production code
grep -rE "pretrained|from_pretrained|mediapipe|torchvision\.models|timm\.|transformers|tensorflow_hub|ultralytics" \
  ml/*.py ml/scripts/*.py apps/api/*.py apps/web/src/

# 2. Sem-Lex pretrained models AND pretrained-derived pose features never referenced
grep -rE "SL-GCN|SLGCN|sl_gcn|sl-gcn\.pt|gloss_recognition\.pt|sem-lex.*poses|semlex.*poses" \
  ml/ apps/ scripts/

# 3. Model file confirms random init
python -c "from ml.model import build_model; m = build_model(25); print([(n, p.requires_grad) for n, p in m.named_parameters()][:5])"

# 4. CI re-runs the smoke pipeline on every push (build → forward → ONNX → ORT load → numerical match)
#    See .github/workflows/ci.yml
```

## Frameworks allowed

| Framework | Role | Pretrained? |
|---|---|---|
| PyTorch | Model definition + training | No — only `nn.Module` primitives + `kaiming_normal_` init |
| ONNX Runtime / ONNX Runtime Web | Inference execution | No — execution framework only |
| OpenCV (headless) | Video decoding, resize | No — pure pixel I/O |
| scikit-learn | Metrics (precision/recall/F1/confusion matrix) | No — classical metrics |

## Sign-off

| Field | Value |
|---|---|
| Attestation version | 1.0 |
| Last verified | _(populate with the date of your last `grep` audit)_ |
| Reviewer | _(name + date)_ |
