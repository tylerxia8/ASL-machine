# Dataset and Training Process

## Collection

1. Use **Capture** mode in the web app (`/capture`) or `ml/scripts/record_from_webcam.py`.
2. Each clip: ~1�1.5 s, saved as frame sequence under `ml/data/clips/{sign_id}/{signer_id}/`.
3. Minimum target: **30 clips per sign** across **?2 signers**.
4. Register clips in `ml/data/manifest.json` via `python scripts/build_manifest.py`.

### Naming convention

```
{sign_id}/{signer_id}/{session_id}_{clip_index}.npz
```

Each `.npz` contains `frames` float32 array shape `(T, H, W, 3)` with T=24.

## Splits

- **Train/val/test:** 70/15/15 by clip.
- **Signer-disjoint test:** at least one signer held out entirely for final eval (`eval.py --signer-disjoint`).

## Training (from scratch)

```bash
cd ml
pip install -r requirements.txt
python train.py --manifest data/manifest.json --epochs 30 --model-version v1
python export_onnx.py --checkpoint checkpoints/best.pt
python eval.py --checkpoint checkpoints/best.pt --report ../docs/VALIDATION_REPORT.md
```

- Architecture: `SignClipCNN3D` � random init only (`weights=None` equivalent).
- **Forbidden:** torchvision pretrained, MediaPipe, YOLO pose, transfer learning from ImageNet/ASL Zoo.

## Synthetic bootstrap

For pipeline testing without real video:

```bash
python scripts/generate_synthetic_dataset.py --signs 20 --clips-per-sign 40
```

Replace with real clips before pilot submission; retrain and update validation report.

## Versioning

- Checkpoints: `ml/checkpoints/{model_version}/`
- Exported ONNX: `ml/exports/model.onnx`, `labels.json`, `model_meta.json`
- Web bundle: sync via `apps/web` script `npm run sync-model`
