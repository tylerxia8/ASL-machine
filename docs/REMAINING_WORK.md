# Remaining Work

This project now has a runnable browser app, API, bundled Wave 1 model files, validation metrics, and rubric mapping. The remaining work is mostly model-quality work, not application plumbing.

## Remaining blocker

All signer data for the recognition model comes from Sem-Lex. Local `signer_a` clips are kept only as learner-quality smoke/calibration samples and should not be treated as the signer-diversity source.

Sem-Lex access is working: the v9 runs fetched, decoded, trained, evaluated, exported, uploaded artifacts, and published releases successfully. The remaining blocker is model/training quality, specifically mode collapse toward `where`.

## Training attempts already run

| Version | Settings | Result |
|---|---|---|
| `wave1-semlex-full-v8` | Full Sem-Lex run, default release before this pass | 14.09% test accuracy, macro F1 0.00996. Best available by macro F1 among the tied 14.09% runs. |
| `wave1-semlex-full-v9` | `both`, `train,val,test`, 100 clips/sign, 30 epochs, default model | 14.09% test accuracy, macro F1 0.00988. Did not improve over v8. |
| `wave1-semlex-full-v9-small` | `both`, `train,val,test`, 100 clips/sign, 15 epochs, small model | 8.18% test accuracy, macro F1 0.00610. Worse than v8/v9. |

GitHub CLI is authenticated on this machine as of 2026-05-26, and the `SEMLEX_DATA_URLS` / `SEMLEX_DRIVE_FILES` secrets exist.

The v9 logs show:

- Total clips: 1,411
- Split counts: 1,009 train / 182 val / 220 test
- Classes with data: 25 / 25
- Train class counts: min 2, max 82, mean 40.4, zero-count classes 0
- Repeated `MODE-COLLAPSE` warnings: validation predictions collapse to 1-2 classes

Local training hardening now added:

- `ml/train.py` uses light label smoothing by default (`--label-smoothing 0.05`)
- Gradients are clipped by default (`--max-grad-norm 1.0`)
- Checkpoint selection now breaks validation-accuracy ties in favor of more distinct predicted classes

## Next training run

Recommended workflow inputs after committing/pushing the training hardening:

- `dataset_source`: `semlex`
- `semlex_splits`: `train,val,test`
- `semlex_clips_per_sign`: increase beyond the previous v8 cap if storage/time allow
- `epochs`: 20+
- `model_size`: `small` for low-data tests, `default` once the dataset is larger
- `model_version`: e.g. `wave1-semlex-full-v9`

After training:

1. Download release assets: `model.onnx`, `labels.json`, `model_meta.json`, `eval_metrics.json`, `VALIDATION_REPORT.md`.
2. Put the first three into `apps/web/public/models/`.
3. Confirm `docs/VALIDATION_REPORT.md` includes confidence calibration bins from the updated `ml/eval.py`.
4. Run `docs/WAVE1_DRY_RUN.md` with 2-3 testers.

## Current blocker summary

The current bundled model, `wave1-semlex-full-v8`, is valid as an integration/demo artifact but not pilot-quality. Validation accuracy is 14.09% with macro F1 0.01, and predictions collapse heavily toward `where`. Two follow-up training runs (`wave1-semlex-full-v9` and `wave1-semlex-full-v9-small`) did not improve the result, so the next meaningful step is training hardening or architecture improvement against mode collapse, using Sem-Lex as the signer data source.
