# Dataset and Training Process

## Sources

The Wave 1 model is trained on a combination of:

| Source | Description | Stored in |
|---|---|---|
| **Self-recorded learner samples** | Captured via the in-app `/capture` flow. Currently 25 single-clip-per-sign baseline recordings from one signer. Used as a "learner-quality" calibration set. | `ml/data/learner_samples/` (committed to repo, ~12 MB) |
| **Sem-Lex Benchmark** | [leekezar/SemLex](https://github.com/leekezar/SemLex), Apache-2.0. 91,148 isolated-sign videos across 3,149 glosses from 41 deaf signers. We curate a subset matching our 25-sign Wave 1 roster. **Only the raw videos are used — Sem-Lex's pretrained SL-GCN models are not touched.** See [NO_PRETRAINED_MODELS.md](NO_PRETRAINED_MODELS.md). | Fetched on demand by [`ml/scripts/fetch_semlex.py`](../ml/scripts/fetch_semlex.py); never committed. In CI, downloads are cached via `actions/cache@v4` between runs. |

Access to Sem-Lex requires accepting [their terms of use](https://docs.google.com/forms/d/e/1FAIpQLSeFjIcbJcr2kWibgrEdFyLhNADo1ErnVGuQHtGeiDiqe4iteQ/viewform). The fetcher reads the index URL Sem-Lex provides via the `SEMLEX_INDEX_URL` env var or `--index-url` flag.

## Vocabulary

| Tier | Count | File | Used for |
|---|---|---|---|
| Reference vocabulary | 103 signs | [`content/vocabulary.csv`](../content/vocabulary.csv) | Practice prompts (78 of these are reference-only) |
| Wave 1 trained roster | 25 signs | [`content/wave1_signs.csv`](../content/wave1_signs.csv) | What the model is trained to recognize |

The Wave 1 roster was chosen for visual distinctness (avoiding confusable pairs like all-color signs sharing the same chin-touch handshape; see [VALIDATION_REPORT.md → Known limitations](VALIDATION_REPORT.md)).

## Per-sign hint content

[`content/hints/`](../content/hints/) contains a JSON file per vocabulary sign with `handshape / movement / location / orientation / framing / common_confusions`. The 25 trained signs have hand-written content (see [`scripts/generate_hints.py`](../scripts/generate_hints.py) `SPECIFIC` dict); the 78 reference-only signs have templated content.

## Clip format

Both Sem-Lex and learner-recorded videos go through the same pipeline:

1. **Decode** ([`ml/scripts/import_captures.py`](../ml/scripts/import_captures.py)): OpenCV `VideoCapture` reads frames from `.webm` or `.mp4`. Resamples to 24 frames evenly, center-crops to square, resizes to 160×160, normalizes to `[0, 1]` float32.
2. **Stored layout** on disk:
   ```
   ml/data/clips/{sign_id}/{signer_id}/clip_{NNNN}.npz
   ```
   Each `.npz` contains a single `frames` float32 array of shape `(24, 160, 160, 3)` in NHWC order.
3. **Manifest** ([`ml/scripts/build_manifest.py`](../ml/scripts/build_manifest.py)): walks `ml/data/clips/`, applies wave1 filter when `--wave1` is passed, assigns each clip to `train/val/test` per the split policy.

## Splits

`--signer-disjoint` is the rubric-honest evaluation:

- With ≥3 signers: holds out one entire signer for `test`.
- With 2 signers: holds out `signer_b` for `test`, `signer_a` clips split 85/15 train/val.
- With 1 signer only: every clip goes to `train` (no test signal). Eval will report zero test accuracy; you need more signers to claim generalization.

For Sem-Lex data, each Sem-Lex signer becomes `signer_id = semlex_<their_id>`, so signer-disjoint operates across Sem-Lex contributors too.

## Training (from scratch — no pretrained weights)

```bash
python ml/train.py \
  --manifest ml/data/manifest.json \
  --epochs 20 \
  --batch-size 16 \
  --model-version wave1-v1
```

- **Architecture:** [`SignClipCNN3D`](../ml/model.py) — 3 Conv3d blocks (32/64/128 ch) + AvgPool3d(3,8,8) + FC head. ~3.57M params. All weights `kaiming_normal_` initialized at instantiation. **No `weights=` arguments, no checkpoint loading from external sources.**
- **Loss / optimizer:** cross-entropy, Adam @ 1e-3, batch size 16.
- **Best checkpoint** saved to `ml/checkpoints/{model_version}/best.pt` (the epoch with highest val accuracy).

**Forbidden** (verified by CI grep in [NO_PRETRAINED_MODELS.md](NO_PRETRAINED_MODELS.md)): MediaPipe, MMPose, OpenPose, `torchvision.models.*(weights=...)`, `timm`, `transformers`, Sem-Lex's published SL-GCN checkpoints.

## Export to ONNX

```bash
python ml/export_onnx.py \
  --checkpoint ml/checkpoints/wave1-v1/best.pt
```

Writes a single self-contained file at `ml/exports/model.onnx` (~14 MB for 25 classes) plus `labels.json` and `model_meta.json`. Uses the legacy `dynamo=False` exporter so weights are inlined (the dynamo exporter splits them into a `.onnx.data` sidecar which the web loader doesn't expect).

For the browser:
```bash
cd apps/web && npm run sync-model
```
Copies the three files into `apps/web/public/models/`.

## Versioning

| Artifact | Naming |
|---|---|
| Training checkpoint | `ml/checkpoints/{model_version}/best.pt` |
| ONNX export | `ml/exports/model.onnx` (single file, ~14 MB) |
| Labels | `ml/exports/labels.json` with `model_version`, `sign_ids`, `label_to_idx`, `input_type=3d`, `num_frames=24`, `frame_size=160` |
| GitHub Release | tag = `{model_version}` (set via workflow input) |

Increment `model_version` per training run: `wave1-v1`, `wave1-v2`, `wave1-with-semlex-v1`, etc. The lobby "Model:" line reads the version from `labels.json` so testers can see what's deployed.

## Running training

| Path | When | How |
|---|---|---|
| **GitHub Actions** | Default. No local PyTorch or disk needed. | Actions tab → "Train Wave 1" → Run workflow. Set `dataset_source` and `SEMLEX_INDEX_URL` secret (if using SemLex). See [README.md](../README.md). |
| **Local** | Only if you can't reach Actions or want to iterate on dataset/code locally. | `scripts\continue-wave1.ps1` (Windows) or run the three Python commands above by hand. Requires ~3 GB free disk for PyTorch install. |

## Quality bar

Pilot quality (not classroom-assessment grade). Specific criteria, thresholds, and known limitations are tracked in [VALIDATION_REPORT.md](VALIDATION_REPORT.md). Re-running `ml/eval.py` after each training run overwrites only the `<!-- AUTO-METRICS:START -->`…`END` block; the surrounding narrative is preserved.
