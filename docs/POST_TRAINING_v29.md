# Post-training analysis - `wave1-semlex-aslcitizen-wlasl-v29-bs16`

Run: https://github.com/tylerxia8/ASL-machine/actions/runs/26832179057  
Release: https://github.com/tylerxia8/ASL-machine/releases/tag/wave1-semlex-aslcitizen-wlasl-v29-bs16

## Settings

- Dataset source: Sem-Lex + learner samples + ASL Citizen + WLASL.
- ASLLVD: disabled, so this is a WLASL-only follow-up to the v28 ASLLVD experiment.
- Model: `hand_landmark_tcn`
- Epochs: 55
- Batch size: 16
- Preprocess: `letterbox`
- Learning rate: `0.001`
- Label smoothing: `0.02`
- Landmark noise/dropout: `0.02` / `0.10`
- Baseline comparison: `wave1-semlex-aslcitizen-wlasl-v27-v23recipe`

## Metrics

| Metric | v27 baseline | v29 candidate | Delta |
|---|---:|---:|---:|
| Accuracy | 0.8184 | 0.7685 | -0.0499 |
| Macro F1 | 0.7454 | 0.7133 | -0.0321 |
| Weighted F1 | 0.8191 | 0.7673 | -0.0517 |

**Recommendation:** DO NOT PROMOTE.

## Weakest Classes

| Sign | F1 | Support | Notes |
|---|---:|---:|---|
| `five` | 0.000 | 1 | Still no held-out recognition signal. |
| `four` | 0.222 | 4 | Thin support and low recall. |
| `who` | 0.406 | 40 | Key blocker; F1 dropped 0.147 vs v27. |
| `goodbye` | 0.538 | 14 | Moderate regression. |
| `deaf` | 0.590 | 38 | Key blocker; F1 dropped 0.156 vs v27. |

## Interpretation

Batch size 16 did not improve the deployable model. It helped a few signs (`three`, `yes`, `two`, `no`) but regressed too many high-value signs, especially `deaf`, `who`, `help`, `where`, `hello`, and `please`.

Keep `wave1-semlex-aslcitizen-wlasl-v27-v23recipe` as the bundled app model. Use v29 only as an experiment record showing that smaller landmark batches are not a promotion path without additional data or training changes.
