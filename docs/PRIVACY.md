# Privacy — Camera and Video Handling

Covers rubric Req 13 (privacy-conscious video handling) and Req 5 (browser-first inference).

## Default learner-facing behavior

- **Raw camera frames are never uploaded** during a normal practice session.
- All inference runs **in the browser** via ONNX Runtime Web (WebAssembly). The `model.onnx` weights ship in the static bundle at `/models/model.onnx` (~14 MB, cached after first load).
- After each attempt the API receives **metadata only** (no images, no audio, no biometric features):
  - `sign_id` (the prompted sign, e.g. `"hello"`)
  - `outcome` (`pass` / `fail` / `retry`)
  - `confidence` (a single float)
  - `predicted_label` (which class the model thought it saw)
  - `session_id` and timestamp

You can verify this directly in **DevTools → Network tab** during a Record → Evaluate cycle on the practice page. You should see exactly two POSTs: `/attempts` (the metadata above) and `/signs/{id}/hint` (text content from the hint files). No request body should contain a binary blob.

## Server-stored data

- `apps/api` writes to Supabase (or local SQLite in dev). Schema in [`supabase/migrations/001_initial_schema.sql`](../supabase/migrations/001_initial_schema.sql).
- Tables persisted: `profiles`, `practice_sessions`, `attempts`, `sign_mastery`. None hold pixel data.
- Row-level security is enabled per table — users can read/write only their own rows.
- The deployment must use HTTPS — `navigator.mediaDevices.getUserMedia` is gated by secure context.

## Dataset capture mode (the engineer flow)

The `/capture` page is for **engineers building the training set**, not for learners. It writes `.webm` files **locally to the engineer's Downloads folder** via the browser's download API. Files are then manually moved into `ml/data/incoming/` for processing. **No cloud upload is in the path.**

When the engineer is also the data subject (recording themselves), this is consistent with rubric Req 6 (engineer-owned dataset).

## Sem-Lex data

When training uses Sem-Lex videos:

- Videos are downloaded **only inside the GitHub Actions runner** that performs training (ephemeral; the runner is destroyed after the job).
- Cached temporarily via `actions/cache@v4` keyed on the wave1 sign list + per-sign cap; cache evicted per GitHub's 7-day policy.
- Videos are **never** committed to this repo, **never** served to the learner's browser, **never** logged to telemetry.
- Trained model weights are an abstract derivative of the videos — the model file does not embed or expose any original Sem-Lex frames.

Sem-Lex's own privacy posture is set by them (Apache-2.0 license, public dataset of consenting deaf participants per their terms). This project consumes the dataset under those terms but adds no new personal data exposure.

## Analytics

The web app emits lightweight `trackEvent` calls (camera_ok, camera_error, attempt, inference_error). These contain no image data and no PII beyond what's already in the API metadata above. In dev mode they're `console.debug` only; in production they're dispatched as `CustomEvent('asl-analytics', ...)` so the host page can ship them to an analytics backend if one is configured.

## What is intentionally NOT in scope for the pilot

- Server-side video inference path (rubric Req 5 says default must be browser; we have not built a server path either way).
- Uploading raw learner video for retraining (rubric Out-of-scope item).
- Long-term storage of learner attempt videos in any form.

If a future iteration proposes any of the above, it requires:
- Explicit opt-in UI copy (not a default).
- Separate consent record stored alongside the user profile.
- Documentation update separating "default pilot behavior" from "optional consent flow."
- Security review.

None of this is implemented in the current pilot build.

## Verification commands

```bash
# 1. No video/audio mime types are POSTed anywhere from the practice flow
grep -rn "image/\|video/\|audio/\|Blob\|FormData" apps/web/src/lib/ apps/web/src/pages/PracticePage.tsx

# 2. recordVideo() / Blob handling is confined to the capture page (engineer flow), never practice
grep -rn "recordVideo\|MediaRecorder\|application/blob" apps/web/src/pages/PracticePage.tsx
#   ^ expect zero matches

# 3. API receives only the documented metadata fields
grep -rn "AttemptCreate\|AttemptOut" apps/api/schemas.py
```
