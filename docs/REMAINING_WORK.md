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

GitHub CLI is authenticated on this machine as of 2026-05-26, and the `SEMLEX_DATA_URLS` / `SEMLEX_DRIVE_FILES` secrets exist.

The v9/v10 logs show:

- Total clips: 1,411
- Split counts: 1,009 train / 182 val / 220 test
- Classes with data: 25 / 25
- Train class counts: min 2, max 82, mean 40.4, zero-count classes 0
- Repeated `MODE-COLLAPSE` warnings: validation predictions collapse to 1-2 classes

Local training hardening now added and verified in the v10 run:

- `ml/train.py` uses light label smoothing by default (`--label-smoothing 0.05`)
- Gradients are clipped by default (`--max-grad-norm 1.0`)
- Checkpoint selection now breaks validation-accuracy ties in favor of more distinct predicted classes
- `ml/eval.py` now exports per-clip predictions plus 0.1-wide confidence bins

## Next training/diagnostic work

Do not spend more runs on the same settings. The v10 hardened run lowered confidence and reduced the single-class `where` collapse somewhat, but it did not improve accuracy. The next useful work is diagnosis and architecture/training changes:

- Add a tiny per-class overfit check: can the model memorize 2-5 clips per sign?
- Inspect label/video alignment for classes that collapse or have near-zero recall.
- Try lower learning rates and class-balanced sampling/loss on Sem-Lex only.
- Consider a stronger temporal architecture than the current whole-frame 3D CNN, while still training from scratch and avoiding pretrained pose/landmark models.
- Increase Sem-Lex clips per sign only after the overfit/alignment checks pass.

After training:

1. Download release assets: `model.onnx`, `labels.json`, `model_meta.json`, `eval_metrics.json`, `VALIDATION_REPORT.md`.
2. Put the first three into `apps/web/public/models/`.
3. Confirm `docs/VALIDATION_REPORT.md` includes confidence calibration bins from the updated `ml/eval.py`.
4. Run `docs/WAVE1_DRY_RUN.md` with 2-3 testers.

## Current blocker summary

The current bundled model, `wave1-semlex-full-v8`, is valid as an integration/demo artifact but not pilot-quality. Validation accuracy is 14.09% with macro F1 0.01, and predictions collapse heavily toward `where`. Follow-up runs (`wave1-semlex-full-v9`, `wave1-semlex-full-v9-small`, and `wave1-semlex-full-v10-hardened`) did not improve the bundled accuracy, so the next meaningful step is diagnostic work and architecture/training improvement on Sem-Lex.
