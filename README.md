# ASL Learning with Computer Vision � Pilot

Browser-based ASL 1 vocabulary practice with local ONNX inference, learner accounts, pass/fail feedback, and rule-based hints.

## Structure

- `apps/web` � React + TypeScript practice UI and dataset capture
- `apps/api` � FastAPI progress and sign metadata API
- `ml/` � Dataset tools, from-scratch training, ONNX export, evaluation
- `content/` � Vocabulary, hints, label manifest
- `docs/` � Controlled conditions, privacy, validation, no-pretrained attestation
- `supabase/migrations/` � Database schema

## Quick start

### Prerequisites

- Node.js 20+
- Python 3.11+
- Supabase project (optional for local dev � API falls back to SQLite)

### ML (prototype model)

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python scripts/generate_synthetic_dataset.py --signs 20
python train.py --epochs 5
python export_onnx.py
```

### API

```bash
cd apps/api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Web

```bash
cd apps/web
npm install
npm run dev
```

**Windows:** Quote paths with spaces. Use `npm.cmd` if `npm` is blocked. Double-click `START-HERE.cmd` or run `.\scripts\start-api.cmd` from the project folder after `cd` with quotes.

**Port 8000 busy:** Another API may already be running. Open http://127.0.0.1:8000/health � or use `scripts/start-api.cmd` (falls back to port 8001; then set `VITE_API_URL=http://localhost:8001` in `apps/web/.env.local`).

Copy `ml/exports/model.onnx` and `ml/exports/labels.json` into `apps/web/public/models/` after training (or run `npm run sync-model`).

**No fallback:** the practice page hard-fails with "Recognition model unavailable" if `/models/model.onnx` is missing. Train the model before testing the practice flow.

**Smoke-test the canonical pipeline anytime:** `python ml/scripts/smoke_pipeline.py` verifies model build → forward → ONNX export → ONNXRuntime load → numerical match. No real data required.

## Wave 1 workflow

Two paths depending on whether you train locally or on GitHub Actions.

### Path A — GitHub Actions (recommended; no local disk needed)

Training runs on a free GitHub-hosted runner (4 cores, 7 GB RAM, 14 GB disk). Raw videos stay on the runner — your laptop only ever sees the final `model.onnx` (~14 MB) published as a Release asset.

1. **Accept Sem-Lex terms** at https://docs.google.com/forms/d/e/1FAIpQLSeFjIcbJcr2kWibgrEdFyLhNADo1ErnVGuQHtGeiDiqe4iteQ/viewform. You'll receive Google Drive links for 7 files. We only use 4 (the metadata CSV and the 3 raw video tarballs); the 3 `*-poses.tar.gz` files are skipped because they contain pretrained-landmark output that would violate Req 7 — see [docs/NO_PRETRAINED_MODELS.md](docs/NO_PRETRAINED_MODELS.md).

2. **Extract the 4 Drive file IDs** (the ID is the long string between `/d/` and `/view` in each link).

3. **Add a single repo secret** at GitHub → repo → Settings → Secrets and variables → Actions → New repository secret. Name: `SEMLEX_DRIVE_FILES`. Value:
   ```json
   {
     "metadata": "1pkX8_TzL3kdJytQvrU68QEAp6oUvt4rv",
     "train":    "1jiUasWSGv5lkrBUIRmtCXyMzliClCqXo",
     "val":      "1VvrbYgNZe_4fWS5ZdSsHyxOuWHmhisGq",
     "test":     "1uYoM1zNpw4oLpJe4LwtDVPBNgKmAi8CC"
   }
   ```
   (The IDs above are placeholders showing format — paste your actual IDs from the Drive URLs.)

4. **Dispatch the workflow:** Actions → "Train Wave 1" → Run workflow. Recommended first run:
   - `dataset_source` = `both`
   - `semlex_splits` = `val` (smallest archive; fastest first end-to-end run)
   - `semlex_clips_per_sign` = 50
   - `epochs` = 20
   - `model_version` = `wave1-semlex-val-v1`

   Once that works, repeat with `semlex_splits = train,val,test` for the full dataset.

5. **Download the release:** After the workflow finishes (~10–60 min depending on splits + clips/sign), a new GitHub Release appears tagged with your `model_version`. It contains `model.onnx`, `labels.json`, `model_meta.json`, `eval_metrics.json`, `VALIDATION_REPORT.md`. Drop the first three into `apps/web/public/models/` locally and `npm run dev`.

6. **Dry run** — [docs/WAVE1_DRY_RUN.md](docs/WAVE1_DRY_RUN.md) with 2–3 testers.

The workflow file: [.github/workflows/train_wave1.yml](.github/workflows/train_wave1.yml).
The Sem-Lex fetcher (streams tarballs, extracts only matched clips, never touches `*-poses.tar.gz`): [ml/scripts/fetch_semlex.py](ml/scripts/fetch_semlex.py).
Attestation that no Sem-Lex pretrained models or pose features are used: [docs/NO_PRETRAINED_MODELS.md](docs/NO_PRETRAINED_MODELS.md).

### Path B — Local recording + local training

For shakedown testing or if you can't run the CI workflow.

1. **Shakedown** — [docs/WAVE1_SHAKEDOWN.md](docs/WAVE1_SHAKEDOWN.md): record 5 clips/sign for 5 signs and run the full pipeline. ~20 min.
2. **Full collect** — http://localhost:5173/capture (25 trained signs, target 30 clips/sign × 2 signers). Move `.webm` downloads to `ml/data/incoming/` (or run `scripts\import-from-downloads.ps1`).
3. **Retrain** — `.\scripts\continue-wave1.ps1` from repo root.
4. **Practice** — Lobby → **Start Wave 1 session**.
5. **Dry run** — [docs/WAVE1_DRY_RUN.md](docs/WAVE1_DRY_RUN.md).

See [docs/WAVE1_DATA_COLLECTION.md](docs/WAVE1_DATA_COLLECTION.md) for full instructions and [docs/CONTROLLED_CONDITIONS.md](docs/CONTROLLED_CONDITIONS.md) for the operational envelope.

### Environment

See `.env.example` for Supabase and API URLs.

## Pilot constraints

See [docs/CONTROLLED_CONDITIONS.md](docs/CONTROLLED_CONDITIONS.md) and [docs/NO_PRETRAINED_MODELS.md](docs/NO_PRETRAINED_MODELS.md).
