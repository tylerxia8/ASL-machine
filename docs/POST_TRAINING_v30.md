# Post-training analysis - `wave1-semlex-aslcitizen-wlasl-asllvd-v30-weak5`

Run: https://github.com/tylerxia8/ASL-machine/actions/runs/26835853080  
Release: https://github.com/tylerxia8/ASL-machine/releases/tag/wave1-semlex-aslcitizen-wlasl-asllvd-v30-weak5

## Settings

- Dataset source: Sem-Lex + learner samples + ASL Citizen + WLASL + ASLLVD.
- Targeted online signs: `five,four,who,goodbye,deaf`.
- Model: `hand_landmark_tcn`
- Epochs: 55
- Batch size: 32
- Preprocess: `letterbox`
- Learning rate: `0.001`
- Label smoothing: `0.02`
- Landmark noise/dropout: `0.02` / `0.10`
- Baseline comparison: `wave1-semlex-aslcitizen-wlasl-v27-v23recipe`

## Metrics

| Metric | v27 baseline | v30 candidate | Delta |
|---|---:|---:|---:|
| Accuracy | 0.8184 | 0.8049 | -0.0135 |
| Macro F1 | 0.7454 | 0.7690 | +0.0236 |
| Weighted F1 | 0.8191 | 0.8018 | -0.0173 |

**Recommendation:** DO NOT PROMOTE.

## Weakest Classes

| Sign | F1 | Support | Notes |
|---|---:|---:|---|
| `five` | 0.000 | 1 | Still no held-out recognition signal. |
| `who` | 0.479 | 44 | More support, but lower F1 than v27. |
| `four` | 0.588 | 8 | Large improvement from v27. |
| `deaf` | 0.600 | 47 | Key blocker; F1 dropped 0.146 vs v27. |
| `where` | 0.653 | 39 | Non-target regression. |

## Interpretation

The targeted online source sweep helped macro F1 and meaningfully improved `four`, `three`, `how`, `yes`, `goodbye`, `thank_you`, and `meet`. It did not produce a better app model because `deaf`, `where`, `sleep`, `sorry`, and `who` regressed, and overall accuracy/weighted F1 stayed below v27.

Keep `wave1-semlex-aslcitizen-wlasl-v27-v23recipe` as the bundled app model. The v30 data sources are useful, but they need either cleaner per-sign curation, better split control, or more user-recorded examples before promotion.
