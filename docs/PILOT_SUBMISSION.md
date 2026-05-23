# Pilot Submission Checklist

## Deliverables

| # | Item | Location |
|---|------|----------|
| 1 | Browser ASL learning app | `apps/web` |
| 2 | Sign recognition model | `ml/exports/model.onnx` (after training) |
| 3 | Dataset & training docs | `docs/DATASET_AND_TRAINING.md` |
| 4 | Validation report | `docs/VALIDATION_REPORT.md` |
| 5 | Learner accounts & progress | Supabase + `apps/api` |
| 6 | Practice UI | `/practice` |
| 7 | Privacy documentation | `docs/PRIVACY.md` |

## Run locally

1. API: `cd apps/api && pip install -r requirements.txt && uvicorn main:app --reload`
2. Web: `cd apps/web && npm install && npm run dev`
3. ML: `cd ml && pip install -r requirements.txt` then train/export per README

## Deploy

- Web: Vercel (`apps/web/vercel.json`)
- API: Railway/Render (`apps/api/Procfile`)
- DB/Auth: Supabase (`supabase/migrations/001_initial_schema.sql`)

## Internal pilot

Test with 5–10 users: login, 10-sign session, camera deny path, pass/fail/hint/retry, progress persistence.
