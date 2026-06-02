# Post-Training Notes: v31 PopSign Supplement

Run: `wave1-semlex-aslcitizen-wlasl-popsign-v31`  
Date: 2026-06-02  
Release: https://github.com/tylerxia8/ASL-machine/releases/tag/wave1-semlex-aslcitizen-wlasl-popsign-v31

## Recipe

- Dataset source: `both`
- Sem-Lex splits: `train,val,test`
- ASL Citizen: enabled
- WLASL direct-video supplement: enabled
- PopSign supplement: enabled through `ml/scripts/fetch_popsign.py`
- ASLLVD: disabled
- Supplement signs: `five,four,who,goodbye,deaf,please,hello,thank_you,yes,no,where,sleep,water`
- Model: `hand_landmark_tcn`
- Preprocess: `letterbox`
- Epochs: 55
- Batch size: 32
- Label smoothing: 0.02
- Landmark noise/dropout: 0.02 / 0.10
- Early stop patience: 8
- Baseline: `wave1-semlex-aslcitizen-wlasl-v27-v23recipe`

## Result

Do not promote v31.

| Metric | v27 baseline | v31 candidate | Delta |
|---|---:|---:|---:|
| Accuracy | 0.8184 | 0.7680 | -0.0504 |
| Macro F1 | 0.7454 | 0.7536 | +0.0082 |
| Weighted F1 | 0.8191 | 0.7672 | -0.0518 |

v31 improved macro F1 and helped some weak/targeted signs:

- `five`: F1 0.000 -> 0.333, but support is only 1.
- `four`: F1 0.286 -> 0.571.
- `please`: F1 0.833 -> 0.913.
- `goodbye`: F1 0.606 -> 0.689.
- `hello`: F1 0.756 -> 0.824.

The blockers are larger than the gains:

- `deaf`: F1 0.746 -> 0.494.
- `sleep`: F1 0.875 -> 0.731.
- Overall accuracy dropped by 5.0 points.
- Weighted F1 dropped by 5.2 points.

## Notes

The PopSign fetch path is useful and should stay. It successfully streamed capped per-sign tar files in GitHub Actions without downloading the full 1.1 TB dataset. The next model experiment should avoid mixing all PopSign signs at once; try a narrower PopSign-only supplement for signs that improved here, or add class/source weighting so learner-style PopSign clips do not overpower Sem-Lex/native-signer generalization.
