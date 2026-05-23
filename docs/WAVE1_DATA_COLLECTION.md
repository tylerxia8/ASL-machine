# Wave 1 — Real Data Collection (25 signs)

Target signs: [`content/wave1_signs.csv`](../content/wave1_signs.csv) (25 entries).

## Setup (one time)

1. Read [`CONTROLLED_CONDITIONS.md`](./CONTROLLED_CONDITIONS.md). Use the same desk, lighting, and camera distance throughout each session.
2. Recruit **at least 2 signers** (`signer_a`, `signer_b`). A third signer (`signer_c`) lets you hold one out entirely for a signer-disjoint test set.
3. Per signer per sign target:
   - **Minimum to retrain:** 10 clips/sign
   - **Pilot target:** 30 clips/sign
4. Total for the recommended setup: 25 signs × 30 clips × 2 signers = **1,500 clips** (≈ 80–150 MB of webm files).

## Capture workflow

1. Start the API and web dev server (see [`../README.md`](../README.md)).
2. Open **Capture** in the app (e.g. http://localhost:5173/capture).
3. Pick your **Signer ID** (`signer_a`/`b`/`c`). The dropdown drives the filename and the train/test split.
4. The page shows the current sign with a **Reference** panel (handshape / movement / location / orientation / framing). Read it before recording.
5. Click **Record 2s clip**. You'll see:
   - A 3-second countdown (`3 → 2 → 1`) — get into starting position.
   - A red **● REC** border for 2 seconds — perform the sign.
   - The browser saves a `.webm` (~30–100 KB) to your Downloads folder, named `{sign_id}_{signer_id}_{timestamp}.webm`.
6. Page auto-advances when you reach 30 clips on a sign; use **Next sign** / **Previous sign** to navigate manually.
7. Goofed a take? Click **Undo last** — it decrements the count and tells you which file to delete from Downloads.

## Recording quality checklist

Re-record a take if any of these are true:
- Hands left the guide box at any point.
- Face was cropped or obscured.
- You started or stopped mid-sign because of the 2-second window — practice once so the sign fits cleanly.
- Background motion (someone walked past, screen behind you changed).
- Lighting changed mid-clip (cloud passing across a window, lamp flicker).
- You self-corrected ("oops, let me try that again") — record a fresh take instead.

## Importing into the training pipeline

Move all `.webm` files from your Downloads folder into:

```
ml/data/incoming/
```

Then run from the project root:

```powershell
.\scripts\continue-wave1.ps1
```

This script:
1. Decodes each `.webm` with OpenCV, resamples to 24 frames over the clip, center-crops to 160×160, normalizes.
2. Builds the train/val/test manifest (signer-disjoint when ≥ 3 signers are present).
3. Trains the from-scratch 3D CNN.
4. Evaluates on the held-out split, writes `docs/VALIDATION_REPORT.md`.
5. Exports `model.onnx` + `labels.json`, copies to `apps/web/public/models/`.

`scripts\retrain_wave1.ps1` does the same thing but assumes you already have clips imported.

## After training

1. Open the **Lobby** → **Start Wave 1 session**.
2. Practice each of the 25 trained signs. Note which ones the model gets wrong; record more clips of those and re-run `continue-wave1.ps1`.
3. Walk through [`WAVE1_DRY_RUN.md`](./WAVE1_DRY_RUN.md) before inviting external testers.

## Minimum to proceed to pilot testing

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Signs trained | 25 | 25 |
| Signers | 2 | 3 (hold one out) |
| Clips per sign per signer | 10 | 30 |
| Lighting/background conditions | 1 setup | 2 setups across sessions |
| Signer-disjoint test accuracy | n/a (record more) | ≥ 0.50 top-1 to pilot |
