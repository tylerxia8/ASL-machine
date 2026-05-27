# Remaining Work

This project now has a runnable browser app, API, bundled Wave 1 model files, validation metrics, and rubric mapping. The remaining work is mostly model-quality work, not application plumbing.

## Remaining blocker

All signer data for the recognition model comes from Sem-Lex. Local `signer_a` clips are kept only as learner-quality smoke/calibration samples and should not be treated as the signer-diversity source.

Sem-Lex access is working: the v9 and v10 runs fetched, decoded, trained, evaluated, exported, uploaded artifacts, and published releases successfully. The remaining blocker is model/training quality, specifically poor signer-disjoint generalization and mode collapse.

## Training attempts already run

| Version | Settings | Result |
|---|---|---|
| `wave1-semlex-full-v8` | Full Sem-Lex run, default release before this pass | 14.09% test accuracy, macro F1 0.00996. Best available by macro F1 among the tied 14.09% runs. |
| `wave1-semlex-full-v9` | `both`, `train,val,test`, 100 clips/sign, 30 epochs, default model | 14.09% test accuracy, macro F1 0.00988. Did not improve over v8. |
| `wave1-semlex-full-v9-small` | `both`, `train,val,test`, 100 clips/sign, 15 epochs, small model | 8.18% test accuracy, macro F1 0.00610. Worse than v8/v9. |
| `wave1-semlex-full-v10-hardened` | `semlex`, `train,val,test`, 100 clips/sign, 20 epochs, default model, label smoothing + grad clipping | 4.55% test accuracy, macro F1 0.01141. Confidence bins were produced, but accuracy worsened; keep v8 as bundled best-by-accuracy model. |
| `wave1-semlex-probe-only-v2` | `semlex`, `val`, 20 clips/sign, `probe_only=true`, small model, dropout disabled inside probe | Probe passed: memorized 20 clips across 10 classes at 90% by epoch 17. No model release was created. |

GitHub CLI is authenticated on this machine as of 2026-05-26, and the `SEMLEX_DATA_URLS` / `SEMLEX_DRIVE_FILES` secrets exist.

The v9/v10 logs show:

- Total clips: 1,411
- Split counts: 1,009 train / 182 val / 220 test
- Classes with data: 25 / 25
- Train class counts: min 2, max 82, mean 40.4, zero-count classes 0
- Repeated `MODE-COLLAPSE` warnings: validation predictions collapse to 1-2 classes

The probe-only run (`26516283720`) used the Sem-Lex `val` split as a smaller diagnostic:

- Staged videos: 323
- Manifest clips: 322
- Split counts: 255 train / 53 val / 14 test
- Classes with data: 24 / 24 for that restricted split
- Overfit probe: 20 clips, 10 classes, 2 clips/class, small model, dropout disabled
- Result: PASS at 90% memorization accuracy by epoch 17

Local training hardening now added and verified in the v10 run:

- `ml/train.py` uses light label smoothing by default (`--label-smoothing 0.05`)
- Gradients are clipped by default (`--max-grad-norm 1.0`)
- Checkpoint selection now breaks validation-accuracy ties in favor of more distinct predicted classes
- `ml/eval.py` now exports per-clip predictions plus 0.1-wide confidence bins
- `ml/scripts/overfit_probe.py` can test whether the current model can memorize a tiny per-class subset before launching another full training job
- `.github/workflows/train_wave1.yml` runs the overfit probe by default before Sem-Lex-backed full training; set `run_overfit_probe=false` only when intentionally bypassing that diagnostic
- The training workflow also supports `probe_only=true`, which stops after fetch/decode/manifest/probe and skips train/export/release
- The training workflow exposes trainer hyperparameters (`learning_rate`, `weight_decay`, `label_smoothing`, `max_grad_norm`, `early_stop_patience`) so experiments do not require code edits

## Next training/diagnostic work

Do not spend more runs on the same settings. The v10 hardened run lowered confidence and reduced the single-class `where` collapse somewhat, but it did not improve accuracy. The probe-only run confirms the stack can memorize a tiny Sem-Lex subset, so the next useful work is generalization-focused diagnosis and architecture/training changes:

- Inspect label/video alignment for classes that collapse or have near-zero recall, but treat total label breakage as less likely because the tiny memorization probe passed.
- Try lower learning rates and class-balanced sampling/loss on Sem-Lex only. Suggested next run: `learning_rate=0.0003`, `label_smoothing=0.0`, `model_size=small`, `epochs=30`.
- Consider a stronger temporal architecture than the current whole-frame 3D CNN, while still training from scratch and avoiding pretrained pose/landmark models.
- Increase Sem-Lex clips per sign only after the overfit/alignment checks pass.

Example overfit probe command after a Sem-Lex manifest exists locally:

```powershell
.\ml\.venv\Scripts\python.exe ml/scripts/overfit_probe.py `
  --manifest ml/data/manifest.json `
  --split train `
  --samples-per-class 2 `
  --max-classes 10 `
  --epochs 40 `
  --model-size small `
  --disable-dropout
```

After training:

1. Download release assets: `model.onnx`, `labels.json`, `model_meta.json`, `eval_metrics.json`, `VALIDATION_REPORT.md`.
2. Put the first three into `apps/web/public/models/`.
3. Confirm `docs/VALIDATION_REPORT.md` includes confidence calibration bins from the updated `ml/eval.py`.
4. Run `docs/WAVE1_DRY_RUN.md` with 2-3 testers.

## Current blocker summary

The current bundled model, `wave1-semlex-full-v8`, is valid as an integration/demo artifact but not pilot-quality. Validation accuracy is 14.09% with macro F1 0.01, and predictions collapse heavily toward `where`. Follow-up runs (`wave1-semlex-full-v9`, `wave1-semlex-full-v9-small`, and `wave1-semlex-full-v10-hardened`) did not improve the bundled accuracy, so the next meaningful step is diagnostic work and architecture/training improvement on Sem-Lex.
