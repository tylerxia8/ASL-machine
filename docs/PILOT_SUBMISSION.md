# Pilot Submission — Index

Maps the rubric's 15 requirements and 7 final deliverables to specific artifacts in this repo. Reviewers can use this as the entry-point document; each link points at the file that satisfies the line item.

Current readiness caveat: the bundled/best-available model is `wave1-semlex-full-v8`. It satisfies the technical deliverable of a trained, from-scratch ONNX model, but validation is not yet pilot-quality: signer-disjoint test accuracy is 14.09% with macro F1 0.01, and predictions collapse heavily toward `where`. Follow-up releases through `wave1-semlex-full-v15-tcn-letterbox` did not improve the bundled accuracy. Treat the current model as an end-to-end integration/demo artifact until additional diagnosis, data work, and retraining improve accuracy.

Project repo: https://github.com/tylerxia8/ASL-machine

## Final Deliverables (rubric §7)

| # | Deliverable | Location |
|---|---|---|
| 1 | Working browser-based ASL learning application | [`apps/web/`](../apps/web/) — Vite + React + TypeScript. Lives behind `/lobby`, `/practice`, `/progress`, `/capture`, `/dry-run`, `/login`. |
| 2 | Trained sign recognition model | Published as a GitHub Release: https://github.com/tylerxia8/ASL-machine/releases. Use `wave1-semlex-full-v8` for the current bundled/best-by-accuracy artifact; newer `wave1-*` tags are experiments unless docs say they replace v8. Drop `model.onnx`, `labels.json`, `model_meta.json` into `apps/web/public/models/`. |
| 3 | Dataset & training process docs (no-pretrained evidence) | [`docs/DATASET_AND_TRAINING.md`](DATASET_AND_TRAINING.md) + [`docs/NO_PRETRAINED_MODELS.md`](NO_PRETRAINED_MODELS.md) |
| 4 | Validation report with accuracy + conditions + limitations | [`docs/VALIDATION_REPORT.md`](VALIDATION_REPORT.md) (narrative hand-written; metrics block populated by `ml/eval.py`) |
| 5 | Learner accounts + progress tracking | [`apps/api/`](../apps/api/) (FastAPI) + [`supabase/migrations/001_initial_schema.sql`](../supabase/migrations/001_initial_schema.sql) |
| 6 | Practice interface (camera, pass/fail, hints, retry, saved progress) | [`apps/web/src/pages/PracticePage.tsx`](../apps/web/src/pages/PracticePage.tsx) |
| 7 | Privacy documentation | [`docs/PRIVACY.md`](PRIVACY.md) |

## Required Pilot Scope (rubric §4)

| Req | Title | Satisfied by |
|---|---|---|
| 1 | ASL-only pilot | No BSL or other-language support anywhere; only `content/vocabulary.csv` (ASL glosses) drives prompts. |
| 2 | 75–100 beginner vocabulary items | [`content/vocabulary.csv`](../content/vocabulary.csv) = 103 entries. Trained subset = 25 ([`content/wave1_signs.csv`](../content/wave1_signs.csv)); the other 78 surface as "reference" prompts and are not evaluated. |
| 3 | Browser-based application | [`apps/web/`](../apps/web/) Vite SPA, deployable to Vercel ([`apps/web/vercel.json`](../apps/web/vercel.json)). |
| 4 | Camera access + clear error handling | [`apps/web/src/lib/camera.ts`](../apps/web/src/lib/camera.ts) returns typed `CameraError` codes (`denied / unsupported / not_found / unknown`); [`PracticePage.tsx`](../apps/web/src/pages/PracticePage.tsx) maps each to user-facing copy + Retry button. |
| 5 | Browser-first CV inference | ONNX Runtime Web (WASM) in [`apps/web/src/lib/inference.ts`](../apps/web/src/lib/inference.ts). No server-side inference path is implemented or required. |
| 6 | Engineer-owned dataset + training | [`docs/DATASET_AND_TRAINING.md`](DATASET_AND_TRAINING.md). Curation logic: [`ml/scripts/fetch_semlex.py`](../ml/scripts/fetch_semlex.py) (Sem-Lex subset selection), [`ml/scripts/import_captures.py`](../ml/scripts/import_captures.py) (decode + normalize), [`ml/scripts/build_manifest.py`](../ml/scripts/build_manifest.py) (signer-disjoint splits). |
| 7 | No pretrained models | [`docs/NO_PRETRAINED_MODELS.md`](NO_PRETRAINED_MODELS.md) — full attestation + reproducible `grep` audit. |
| 8 | Controlled pilot quality | [`docs/CONTROLLED_CONDITIONS.md`](CONTROLLED_CONDITIONS.md) (camera/lighting/distance/framing/validation set/thresholds/limitations). |
| 9 | Pass/fail with confidence thresholds | [`apps/web/src/lib/threshold.ts`](../apps/web/src/lib/threshold.ts) — pass ≥ 0.90, retry 0.70–0.90, fail otherwise. Unit tests in [`apps/web/src/lib/threshold.test.ts`](../apps/web/src/lib/threshold.test.ts). |
| 10 | Targeted pedagogical hints | [`content/hints/`](../content/hints/) — per-sign JSON (handshape / movement / location / orientation / framing / common_confusions). 25 trained signs hand-written; surfaced post-failure by [`PracticePage.tsx`](../apps/web/src/pages/PracticePage.tsx). |
| 11 | Learner accounts | Supabase auth + [`apps/api/auth.py`](../apps/api/auth.py); dev fallback to SQLite + `dev-local-user`. |
| 12 | Saved progress | [`supabase/migrations/001_initial_schema.sql`](../supabase/migrations/001_initial_schema.sql) tables: `attempts`, `practice_sessions`, `sign_mastery`. UI: [`apps/web/src/pages/ProgressPage.tsx`](../apps/web/src/pages/ProgressPage.tsx). |
| 13 | Privacy-conscious video handling | [`docs/PRIVACY.md`](PRIVACY.md). Verifiable via DevTools Network tab: no media bytes leave the browser during practice. |
| 14 | Web app usability | Onboarding-friendly: countdown + REC indicator, "Show me the sign" reference panel, single-button Record, explicit retry/next, untrained signs marked as "reference" instead of failing silently. |
| 15 | Pilot documentation | This file. Full doc index in `docs/`. |

## Out-of-Scope items (rubric §5) — confirmed absent

The following are intentionally NOT implemented (and the code base reflects this — no half-built versions to mislead reviewers):

- Multi-language sign support
- Full ASL conversation / sentence translation
- Teacher or administrator portal
- Classroom rostering
- Google Classroom / Clever SSO
- Server-side video inference path
- Default upload of raw learner video
- Pretrained model weights of any kind (see attestation)
- Research-grade bias analysis
- Production-scale public deployment

## Pre-submission verification

Run these to convince yourself (and the reviewer) that everything still ticks.

### 1. Local web + tests
```powershell
cd apps/web
npm.cmd run lint     # tsc --noEmit, expect clean
npm.cmd test         # vitest, expect 4/4
npm.cmd run build    # production bundle
```

### 2. Local API tests
```powershell
cd apps/api
.\.venv\Scripts\python.exe -m pytest tests/ -q   # expect 5/5
```

### 3. ML pipeline smoke
```powershell
$env:PYTHONIOENCODING = "utf-8"
.\ml\.venv\Scripts\python.exe ml/scripts/smoke_pipeline.py
.\ml\.venv\Scripts\python.exe ml/scripts/test_eval_splice.py
```

### 4. No-pretrained grep audit (rubric Req 7)
```bash
grep -rE "pretrained|from_pretrained|mediapipe|torchvision\.models|timm\.|transformers|tensorflow_hub|ultralytics" \
  ml/*.py ml/scripts/*.py apps/api/*.py apps/web/src/
#   ^ should return zero hits in production code
grep -rE "SL-GCN|SLGCN|sl_gcn|sl-gcn\.pt" ml/ apps/ scripts/
#   ^ also zero
```

### 5. CI on GitHub Actions
Three workflows live in [`.github/workflows/`](../.github/workflows/):
- **CI** ([`ci.yml`](../.github/workflows/ci.yml)) — runs on every push: web lint+test, api pytest, ml-smoke (build → forward → ONNX export → ORT load → numerical match).
- **Train Wave 1** ([`train_wave1.yml`](../.github/workflows/train_wave1.yml)) — manual dispatch: fetches Sem-Lex + learner samples, decodes, trains, evaluates, exports ONNX, publishes as a Release.

Latest CI status: https://github.com/tylerxia8/ASL-machine/actions

### 6. Privacy assertion (rubric Req 13)
With a trained model deployed, open DevTools → Network tab → perform one Record → Evaluate cycle on `/practice`. Confirm:
- Two `POST`s only: `/attempts` (JSON metadata) and `/signs/{id}/hint` (text).
- Neither request body contains binary frame data.
- `/models/model.onnx` and `/models/labels.json` load **once** at page load (cached by browser).

## Pilot internal test (before broader rollout)

Walk [`docs/WAVE1_DRY_RUN.md`](WAVE1_DRY_RUN.md) with 2–3 testers:
- Pass criteria for each of 14 numbered steps.
- Issue log priorities (false-pass > false-fail > confusing hints > latency > UI).
- Network-tab privacy verification (step 14).

## Deployment

| Layer | Where | File |
|---|---|---|
| Web | Vercel | [`apps/web/vercel.json`](../apps/web/vercel.json) |
| API | Render or Railway | [`apps/api/Procfile`](../apps/api/Procfile), [`render.yaml`](../render.yaml) |
| DB + Auth | Supabase | [`supabase/migrations/001_initial_schema.sql`](../supabase/migrations/001_initial_schema.sql) |
| Model bundle | Static (in web bundle) | `apps/web/public/models/` populated from GitHub Release |

## Doc index

| File | What |
|---|---|
| [README.md](../README.md) | Top-level orientation + two training paths |
| [docs/REMAINING_WORK.md](REMAINING_WORK.md) | Remaining model-quality blockers and next training run checklist |
| [docs/CONTROLLED_CONDITIONS.md](CONTROLLED_CONDITIONS.md) | Operational envelope (lighting, framing, distance, thresholds) |
| [docs/DATASET_AND_TRAINING.md](DATASET_AND_TRAINING.md) | Data sources, clip format, splits, training command, ONNX export |
| [docs/NO_PRETRAINED_MODELS.md](NO_PRETRAINED_MODELS.md) | Rubric Req 7 attestation + reproducible audit commands |
| [docs/PRIVACY.md](PRIVACY.md) | Rubric Req 13 — what data leaves the browser, what doesn't, how to verify |
| [docs/VALIDATION_REPORT.md](VALIDATION_REPORT.md) | Narrative + auto-populated metrics block |
| [docs/WAVE1_CAPTURE_QUICKSTART.md](WAVE1_CAPTURE_QUICKSTART.md) | 5-minute capture page intro |
| [docs/WAVE1_DATA_COLLECTION.md](WAVE1_DATA_COLLECTION.md) | Full 25-sign × 30-clip × 2-signer recording plan |
| [docs/WAVE1_SHAKEDOWN.md](WAVE1_SHAKEDOWN.md) | 20-min pipeline-validation procedure before the full session |
| [docs/SELF_LABELING_GUIDE.md](SELF_LABELING_GUIDE.md) | Quality + consistency guidance for self-recording |
| [docs/WAVE1_DRY_RUN.md](WAVE1_DRY_RUN.md) | 30-min facilitator checklist with 2–3 testers |
| [docs/PILOT_SUBMISSION.md](PILOT_SUBMISSION.md) | _This file_ |
