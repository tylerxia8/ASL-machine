# Wave 1 Capture — Quick Start

## Before you start (5 min)

1. Read [CONTROLLED_CONDITIONS.md](./CONTROLLED_CONDITIONS.md) — use the same desk, lighting, and distance for every session.
2. Pick two signers: `signer_a` (you), `signer_b` (a teammate). A third signer (`signer_c`) is recommended so one can be held out as a signer-disjoint test set.
3. Have a per-sign reference handy. The capture page shows one automatically; the canonical source is [`content/wave1_signs.csv`](../content/wave1_signs.csv) → [`content/hints/`](../content/hints/).

## Record clips

1. API healthy: open http://127.0.0.1:8000/health → `{"status":"ok"}`
2. Web running: open http://localhost:5173/capture (port may differ).
3. Select your **Signer ID** in the dropdown (this drives the train/test split — be consistent).
4. Each clip:
   - Read the **Reference** panel (handshape / movement / location / orientation / framing).
   - Click **Record 2s clip** → 3-2-1 countdown → red REC border for 2 seconds → file saves to Downloads as `{sign_id}_{signer_id}_{timestamp}.webm` (≈ 30–100 KB).
   - If you flubbed the take, click **Undo last** and the page tells you which file to delete from Downloads.
5. Page auto-advances when a sign reaches **30 clips**. Use **Next sign** / **Previous sign** to navigate manually.

## Move files into the project

**Option A — PowerShell (easiest)**

```powershell
cd "c:\Users\tyler\Downloads\Gauntlet Superbuilders ASL"
.\scripts\import-from-downloads.ps1
```

The script reads the trained-sign list from `content/wave1_signs.csv` at runtime, so it stays in sync with [Phase 1's sign roster](../content/wave1_signs.csv).

**Option B — Manual**

Move every `.webm` matching `{sign_id}_signer_{x}_{timestamp}.webm` from your Downloads folder into:

```
ml\data\incoming\
```

## Retrain

```powershell
cd "c:\Users\tyler\Downloads\Gauntlet Superbuilders ASL"
.\scripts\continue-wave1.ps1
```

This decodes the `.webm` files (opencv), builds a signer-disjoint manifest, trains the 3D CNN from scratch, writes the validation report's auto-metrics block, exports `model.onnx`, and syncs to `apps/web/public/models/`.

After it completes, restart `npm.cmd run dev` (or just hard-refresh the page) and practice Wave 1.

## Minimum viable (if short on time)

| Level | Clips/sign/signer | Signs | Signers | Total |
|-------|-------------------|-------|---------|-------|
| Shakedown (validate the pipeline) | 5 | 5 | 1 | 25 |
| Minimum retrain | 10 | 25 | 2 | 500 |
| Pilot target | 30 | 25 | 2 | 1,500 |

Before committing to a full 1,500-clip session, run the [shakedown](./WAVE1_SHAKEDOWN.md) — it proves the webm → import → train → ONNX → browser-inference loop works end-to-end in ~20 minutes.
