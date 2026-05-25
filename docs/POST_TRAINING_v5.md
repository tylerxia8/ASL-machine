# Post-training analysis — `wave1-semlex-full-v5`

**Overall test accuracy:** 1.12%  
**Macro F1:** 0.001  
**Total test clips:** 89  
**Classes with any test data:** 24 / 25

## Weakest 5 classes by F1 (with non-empty test set)

| Sign | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `deaf` | 0.00 | 0.00 | 0.00 | 4 |
| `eat` | 0.00 | 0.00 | 0.00 | 4 |
| `four` | 0.00 | 0.00 | 0.00 | 2 |
| `friend` | 0.00 | 0.00 | 0.00 | 1 |
| `goodbye` | 0.00 | 0.00 | 0.00 | 1 |

## 1 class(es) with zero test data

These signs are in the trained roster but the signer-disjoint test split happens to include no clips of them. Accuracy is unknown until the test set grows or `--signer-disjoint` is rebalanced.

`five`

## Confusable pairs (known a-priori from sign linguistics)

| Pair | Recall A | Recall B | A↔B confusions | A+B confusion rate |
|---|---|---|---|---|
| `please` vs `sorry` | 0.00 (n=4) | 1.00 (n=1) | 3/0 | 60.00% |
| `what` vs `where` | 0.00 (n=4) | 0.00 (n=7) | 0/0 | 0.00% |
| `four` vs `five` | 0.00 (n=2) | 0.00 (n=0) | 0/0 | 0.00% |
| `one` vs `two` | 0.00 (n=4) | 0.00 (n=3) | 0/0 | 0.00% |
| `two` vs `three` | 0.00 (n=3) | 0.00 (n=1) | 0/0 | 0.00% |
| `eat` vs `drink` | _not in trained set_ | | | |
| `hello` vs `goodbye` | 0.00 (n=1) | 0.00 (n=1) | 1/0 | 50.00% |

## Top 10 most-confused pairs (any, sorted by count)

| True → | Predicted | Count |
|---|---|---|
| `who` | `sorry` | 8 |
| `where` | `sorry` | 6 |
| `what` | `sorry` | 4 |
| `no` | `sorry` | 4 |
| `nice` | `sorry` | 4 |
| `name` | `sorry` | 4 |
| `meet` | `sorry` | 4 |
| `eat` | `sorry` | 4 |
| `who` | `goodbye` | 3 |
| `sleep` | `sorry` | 3 |

## Interpretation

- Overall accuracy is **near-chance** for a 25-class problem (random = 4%). Almost certainly insufficient training data per class, or test set is too sparse to evaluate. Action: increase `semlex_clips_per_sign`, or train with data augmentation enabled (already wired into `ml/dataset.py`).
