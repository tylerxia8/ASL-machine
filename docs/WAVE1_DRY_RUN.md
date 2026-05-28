# Wave 1 Internal Dry Run (30 minutes)

Run after the guided practice flow is deployed locally. The current recognition model is not pilot-quality, so this dry run treats guided self-check as the primary product path and recognition as an experimental demo path.

## Prerequisites

- Trained model: `model.onnx` + `labels.json` present in `apps/web/public/models/` for the optional recognition demo. Guided self-check must still work if the model is unavailable.
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
| 5 | Confirm **Guided self-check** is selected; click **Record & self-check** | Red REC border for 2s, then self-check controls appear |
| 6 | Complete 5 signs with intentional **correct** signing and click **Matched it** | Attempts log as pass; latency is immediate after self-check |
| 7 | Sign #6 — deliberately perform a weak/mis-framed sign and click **Needs practice** | Attempt logs as retry and shows a useful hint |
| 8 | Switch to **Recognition demo** for one trained sign | Record & evaluate path still runs locally, but result is treated as experimental |
| 9 | Retry button works; "Skip to next" works | Both functional; next sign loads |
| 10 | Continue to a reference-only sign (Full catalog session) | "reference" chip visible; guided self-check still works; recognition demo hides evaluation |
| 11 | Open **Progress**; confirm recent attempts listed | Sign IDs + outcomes + timestamps appear |
| 12 | Hard-refresh page (Ctrl+F5); progress still visible | DB state survives reload |
| 13 | In browser settings, deny camera permission; reload | Page shows "Camera access was denied…" with **Retry camera** button |
| 14 | Open DevTools Network tab during guided and recognition recording | **No POST containing a blob**. Only `/attempts`, `/signs/{id}/hint` (JSON payloads, no media). |

## Log these issues

- **Self-check confusion** — users unsure whether to click Matched it or Needs practice; record `sign_id` and wording issue.
- **Recognition false passes/fails** in demo mode — record `sign_id`, what they actually signed, confidence.
- **Confusing hints** — too generic, wrong handshape advice, contradicts the reference panel.
- **Latency > 3 s** on evaluate — note CPU/browser.
- **UI blockers** — anything a beginner needed help understanding.

File these in your tracker. Fix the top 3 issues before inviting external ASL 1 learners.

## Success criteria

- A full 25-sign session is completable by a non-technical tester without facilitator help (after the initial walk-through).
- Guided self-check feels clear and fair under controlled conditions.
- Hints are specific enough to be actionable (e.g. "the snap at the end was missing" not "incorrect").
- Privacy verified: **no media in network traffic**, model loads from `/models/model.onnx` only.
- Progress persists across reloads and across browser sessions (same account).
