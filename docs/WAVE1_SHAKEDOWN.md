# Wave 1 Pipeline Shakedown (≈ 20 minutes)

A tiny end-to-end run that proves the **capture → import → train → export → browser inference** loop works on real data **before** you commit to a full 1,500-clip recording session.

If anything in the pipeline is broken (a bad codec, a manifest path bug, a training crash on real `.webm` shapes), it surfaces here on 25 clips, not on 1,500.

## Goals

- One signer records 5 clips for each of 5 trained signs (25 clips total, ≈ 5 minutes of recording).
- Full pipeline runs locally to produce a `model.onnx` exported from those 25 clips.
- The browser practice page successfully loads the model and returns *some* result (pass/retry/fail) for a live attempt. Accuracy will be terrible — that's not the point.

## What this does NOT validate

- **Recognition quality** — 5 clips/sign cannot generalize. Expect chance-level accuracy.
- **Signer-disjoint generalization** — only one signer.
- **Confusable-sign behavior** — the 5 signs are picked specifically to be visually distinct.

## Picking the 5 signs

Choose signs with maximally different motion patterns so even an undertrained model produces non-degenerate outputs:

| Sign | Why it's distinct |
|------|-------------------|
| `hello` | Hand at temple, outward motion |
| `please` | Open-palm circular motion on chest |
| `five` | Static open palm at chest height |
| `eat` | Pinched fingers tapping at the mouth |
| `friend` | Two-handed interlocked X hooks |

These span: head-level, chest-level, mouth-level, one-hand, two-hand, static, motion, and circular. Different enough that the model has *something* to discriminate on.

## Procedure

### 1. Set up (2 minutes)

```powershell
# In one terminal:
.\scripts\start-api.cmd
# In another:
.\scripts\start-web.cmd
```

Confirm: `http://127.0.0.1:8000/health` returns `{"status":"ok"}` and the web app opens.

### 2. Record (≈ 5 minutes)

Open `http://localhost:5173/capture`:

1. Set **Signer ID** to `signer_a`.
2. For each of the 5 shakedown signs:
   - Use **Next sign** / **Previous sign** to navigate to it.
   - Read the **Reference** panel.
   - Record **5 clips** (click **Record 2s clip** → countdown → 2s record → repeat).
   - Vary slightly between takes: small differences in hand position, exact moment you start the sign inside the 2s window. Don't perform 5 identical robotic takes.
3. End: 25 `.webm` files in your Downloads folder.

### 3. Import (1 minute)

```powershell
.\scripts\import-from-downloads.ps1
```

Expected output: `Moved 25 file(s) to ml\data\incoming`. If any files were skipped as "sign not in wave1_signs.csv", check the filenames — they must start with one of the 25 trained sign IDs.

### 4. Train + export (≈ 10 minutes on CPU)

```powershell
.\scripts\continue-wave1.ps1
```

You should see, in order:
- `Imported {filename} -> ml/data/clips/...` (25 lines)
- `Wrote ml/data/manifest.json clips=25 splits={...}`
- Training progress: 20 epochs, val_acc rising then plateauing (it'll overfit fast on 25 clips — that's expected)
- `Exported ml/exports/model.onnx (≈ 14 MB, 25 classes)` — note: **even with only 5 signs recorded, the model still has 25 output heads** because labels come from `wave1_signs.csv`. The 20 unrecorded classes will have no training signal.
- `Wrote docs/VALIDATION_REPORT.md accuracy=…`

### 5. Verify in browser (2 minutes)

1. Hard-refresh the lobby page (Ctrl+F5). The "Model:" line should show the new `model_version` (e.g. `wave1-v1` or whatever `continue-wave1.ps1` set) and `25 signs`.
2. Click **Start Wave 1 session**.
3. When prompted with one of your 5 trained signs:
   - Perform it. You should get **a pass/fail/retry verdict** (accuracy is not the point — non-crash is).
   - Network tab: no POST containing media data. Only `/attempts` with JSON.
4. When prompted with one of the 20 unrecorded signs:
   - Sign will likely fail with low confidence. That's correct behavior — the model wasn't trained on that class.

### 6. Verify validation-report splice (30 seconds)

Open [`docs/VALIDATION_REPORT.md`](VALIDATION_REPORT.md). Confirm:
- The **Pilot scope**, **Model approach**, **Dataset**, and **Known limitations** narrative sections are still present (eval.py preserved them).
- The block between `<!-- AUTO-METRICS:START -->` and `<!-- AUTO-METRICS:END -->` now contains real accuracy numbers + per-class metrics + most-confused pairs.
- The `[fill]` placeholders in the Dataset table are still placeholders (those are hand-edited).

## Pass criteria

Shakedown passes if **all** of these are true:

- ✅ `import-from-downloads.ps1` moved 25 files without errors.
- ✅ `continue-wave1.ps1` completed without crashing; produced `ml/exports/model.onnx` and `ml/exports/labels.json`.
- ✅ Browser loaded the new model (lobby shows new model version, no `unavailable` banner).
- ✅ A Record-and-Evaluate cycle returned a verdict for at least one of the 5 trained signs.
- ✅ DevTools Network tab confirmed no video data was POSTed.
- ✅ Validation report has both narrative *and* fresh auto-metrics.

## If anything fails

| Symptom | First thing to check |
|---------|----------------------|
| `import-from-downloads.ps1` says "no matching files" | Filename format — the regex is `{sign_id}_signer_x_{ts}.webm`. The capture page enforces this; manual renames break it. |
| `cv2.VideoCapture could not open` during import | Browser saved an empty/zero-byte webm. Re-record. Confirm `.webm` files are > 5 KB. |
| `RuntimeError: stack expects each tensor to be equal size` during training | A `.webm` decoded to <24 frames. Check that capture wasn't interrupted mid-recording. |
| ONNX export error | Re-run `python ml/scripts/smoke_pipeline.py` — it should still pass. If it doesn't, the canonical pipeline regressed since last verified. |
| Browser loads model but every attempt is "uncertain" | Expected with 5 clips/class. Move on to the full 1,500-clip session. |

## After shakedown passes

Proceed to [`WAVE1_DATA_COLLECTION.md`](WAVE1_DATA_COLLECTION.md) for the full 1,500-clip session. The pipeline you just validated is exactly what `continue-wave1.ps1` runs on the full dataset.
