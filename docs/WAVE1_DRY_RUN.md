# Wave 1 Internal Dry Run (30 minutes)

Run after retraining on **real** Wave 1 clips. The auto-metrics block in [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) must be populated (not "pending") before starting.

## Prerequisites

- Trained model: `model.onnx` + `labels.json` present in `apps/web/public/models/`. Lobby's "Model:" line shows a real version (e.g. `wave1-v1`), not `unavailable`.
- API on :8000 healthy, web on :5173 (or printed port) running.
- Browser: Chrome 120+ or Edge 120+ on desktop, with camera permission grantable.

## Participants

- **2–3 testers** with a webcam each (their own computer or yours).
- **1 facilitator** with this checklist, recording observations.

## Environment

- Controlled conditions per [`CONTROLLED_CONDITIONS.md`](CONTROLLED_CONDITIONS.md): plain background, even lighting, 2–3 feet from camera, eye-level webcam, no backlight.
- Don't tell testers the trained-sign list ahead of time — they should be able to discover it from the lobby/practice UI.

## Script

| # | Action | Pass criterion |
|---|--------|---------------|
| 1 | Open app at /; register or continue in dev mode | Reaches Lobby |
| 2 | Lobby → **Start Wave 1 session** | Lands on /practice with first sign visible |
| 3 | Allow camera; confirm preview + on-screen guide box | Camera feed visible; box overlaid |
| 4 | First sign: **Show me the sign** → read reference | Reference panel renders handshape/movement/location |
| 5 | Click **Record 2s clip** | 3-2-1 countdown shows, red REC border for 2s, then "Evaluating locally…" |
| 6 | Complete 5 signs with intentional **correct** signing | Pass on most (≥3/5). Latency under ~3s. |
| 7 | Sign #6 — deliberately perform the **wrong** sign (e.g. sign HELLO when prompt is GOODBYE) | Fail outcome + specific hint (handshape/movement detail, not "incorrect") |
| 8 | Sign #7 — perform a partial / mis-framed sign | Retry outcome (uncertain) OR Fail. Hint mentions framing if applicable. |
| 9 | Retry button works; "Skip to next" works | Both functional; next sign loads |
| 10 | Continue to a reference-only sign (Full catalog session) | "reference" chip visible; Record button hidden; "Next sign" works |
| 11 | Open **Progress**; confirm recent attempts listed | Sign IDs + outcomes + timestamps appear |
| 12 | Hard-refresh page (Ctrl+F5); progress still visible | DB state survives reload |
| 13 | In browser settings, deny camera permission; reload | Page shows "Camera access was denied…" with **Retry camera** button |
| 14 | Open DevTools Network tab during a Record → Evaluate cycle | **No POST containing a blob**. Only `/attempts`, `/signs/{id}/hint` (JSON payloads, no media). |

## Log these issues

- **False passes** (wrong sign accepted as pass) — most important to track; record `sign_id`, what they actually signed, confidence.
- **False fails** on textbook-correct signs — record `sign_id`, lighting/framing context.
- **Confusing hints** — too generic, wrong handshape advice, contradicts the reference panel.
- **Latency > 3 s** on evaluate — note CPU/browser.
- **UI blockers** — anything a beginner needed help understanding.

File these in your tracker. Fix the top 3 issues before inviting external ASL 1 learners.

## Success criteria

- A full 25-sign session is completable by a non-technical tester without facilitator help (after the initial walk-through).
- Pass/fail feels **mostly fair** under controlled conditions — false-pass rate well under 10% on known-good signs.
- Hints are specific enough to be actionable (e.g. "the snap at the end was missing" not "incorrect").
- Privacy verified: **no media in network traffic**, model loads from `/models/model.onnx` only.
- Progress persists across reloads and across browser sessions (same account).
