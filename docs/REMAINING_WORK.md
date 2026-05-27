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
| `wave1-semlex-full-v11-lr3e4-small` | `semlex`, `train,val,test`, 100 clips/sign, 30 epochs, small model, learning rate 0.0003, no label smoothing | 3.18% test accuracy, macro F1 0.00561. Probe passed first, but full training still failed to generalize; predictions concentrated on `what` and `no`. |
| `wave1-semlex-full-v12-frame` | `semlex`, `train,val,test`, 100 clips/sign, 30 epochs, `model_size=frame`, learning rate 0.0005, no label smoothing | 4.55% test accuracy, macro F1 0.01601. Better macro F1 than v8/v11 but still far below v8 accuracy; do not replace bundled model. |
| `wave1-semlex-full-v13-tcn` | `semlex`, `train,val,test`, 100 clips/sign, 30 epochs, `model_size=tcn`, learning rate 0.0005, no label smoothing | 6.82% test accuracy, macro F1 0.02255. Best macro F1 so far, but still below v8 accuracy; do not replace bundled model. |
| `wave1-semlex-full-v14-tcn-balanced-split` | `semlex`, `train,val,test`, 100 clips/sign, 30 epochs, `model_size=tcn`, learning rate 0.0005, no label smoothing, coverage-aware signer-disjoint split | 5.70% test accuracy, macro F1 0.03441. Best macro F1 so far and improved test coverage diagnostics, but still far below v8 accuracy; do not replace bundled model. |

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

A later probe-only workflow (`26536422324`) verified the manifest report step on real Sem-Lex data:

- Restricted input: Sem-Lex `val` split, 20 clips/sign cap
- Manifest clips: 322
- Signer-disjoint test split: OK
- Zero-test-support signs in that restricted diagnostic split: 13 / 24
- Overfit probe: PASS at 90% memorization accuracy

After changing the signer-disjoint selection to maximize sign coverage and balancing validation per sign, probe-only workflow `26538435250` on the same restricted setup showed:

- Manifest clips: 322
- Signer-disjoint test split: OK
- Zero-test-support signs: 0 / 24
- Zero-val-support signs: 1 / 24 (`four`, with only one non-test clip)
- Overfit probe: PASS at 95% memorization accuracy

The first full run after the split fix, `wave1-semlex-full-v14-tcn-balanced-split` (`26538861231`), showed the split fix worked on the full Sem-Lex-backed Wave 1 dataset but did not solve model quality:

- Manifest clips: 1,411
- Split counts: 750 train / 117 val / 544 test
- Signer-disjoint test split: OK
- Zero-test-support signs: 0 / 25
- Zero-val-support signs: 1 / 25 (`five`, with only 1 train clip and 1 test clip)
- Overfit probe: PASS
- Test result: 5.70% accuracy, macro F1 0.03441, weighted F1 0.04886
- Prediction spread improved compared with the old `where` collapse, but predictions still concentrated on a few labels (`nice`, `no`, `one`, `help`, `four`, `where`) and accuracy stayed below the v8 bundled model.

Local training hardening now added and verified in the v10 run:

- `ml/train.py` uses light label smoothing by default (`--label-smoothing 0.05`)
- Gradients are clipped by default (`--max-grad-norm 1.0`)
- Checkpoint selection now breaks validation-accuracy ties in favor of more distinct predicted classes
- `ml/eval.py` now exports per-clip predictions plus 0.1-wide confidence bins
- `ml/scripts/overfit_probe.py` can test whether the current model can memorize a tiny per-class subset before launching another full training job
- `.github/workflows/train_wave1.yml` runs the overfit probe by default before Sem-Lex-backed full training; set `run_overfit_probe=false` only when intentionally bypassing that diagnostic
- The training workflow also supports `probe_only=true`, which stops after fetch/decode/manifest/probe and skips train/export/release
- The training workflow exposes trainer hyperparameters (`learning_rate`, `weight_decay`, `label_smoothing`, `max_grad_norm`, `early_stop_patience`) so experiments do not require code edits
- `ml/scripts/manifest_report.py` reports split/sign/signer coverage before training and is uploaded as `manifest_report.json` with future releases
- `ml/scripts/build_manifest.py` now chooses signer-disjoint test signers by greedy Wave 1 sign coverage instead of sorted signer ID, reducing accidental zero-test-support signs
- After selecting held-out test signers, `build_manifest.py` also assigns validation clips per sign where at least two non-test clips remain, reducing accidental zero-val-support signs

## Next training/diagnostic work

Do not spend more runs on the same settings. The v10 hardened run lowered confidence and reduced the single-class `where` collapse somewhat, but it did not improve accuracy. The v11 lower-learning-rate small-model run passed the memorization gate, then still failed on signer-disjoint evaluation. The v12 frame-wise model improved macro F1 slightly but did not improve accuracy. The v13 TCN model improved macro F1, and v14 confirmed the split coverage fix on the full dataset, but neither is pilot-quality. The next useful work is preprocessing/visual diagnosis plus stronger architecture/training changes:

- Inspect Sem-Lex frame crops/contact sheets for low-recall and high-confusion signs. The current importer center-crops to a square, which may cut off hands or signing space for some videos.
- Inspect label/video alignment for classes that collapse or have near-zero recall, but treat total label breakage as less likely because the tiny memorization probe passed.
- Consider an even stronger temporal architecture or a better Sem-Lex preprocessing/alignment strategy, while still training from scratch and avoiding pretrained pose/landmark models.
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

The current bundled model, `wave1-semlex-full-v8`, is valid as an integration/demo artifact but not pilot-quality. Validation accuracy is 14.09% with macro F1 0.01, and predictions collapse heavily toward `where`. Follow-up runs (`wave1-semlex-full-v9`, `wave1-semlex-full-v9-small`, `wave1-semlex-full-v10-hardened`, `wave1-semlex-full-v11-lr3e4-small`, `wave1-semlex-full-v12-frame`, `wave1-semlex-full-v13-tcn`, and `wave1-semlex-full-v14-tcn-balanced-split`) did not improve the bundled accuracy, so the next meaningful step is preprocessing/visual diagnosis and stronger architecture/training improvement on Sem-Lex.
