# Post-training analysis — `wave1-semlex-aslcitizen-wlasl-asllvd-v28-five`

**Overall test accuracy:** 80.47%  
**Macro F1:** 0.740  
**Total test clips:** 1060  
**Classes with any test data:** 25 / 25

## Weakest 5 classes by F1 (with non-empty test set)

| Sign | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `five` | 0.00 | 0.00 | 0.00 | 1 |
| `four` | 0.15 | 0.50 | 0.24 | 4 |
| `goodbye` | 0.35 | 0.57 | 0.43 | 14 |
| `who` | 0.63 | 0.48 | 0.55 | 56 |
| `how` | 0.50 | 0.86 | 0.63 | 14 |

## Confusable pairs (known a-priori from sign linguistics)

| Pair | Recall A | Recall B | A↔B confusions | A+B confusion rate |
|---|---|---|---|---|
| `please` vs `sorry` | 0.80 (n=15) | 0.97 (n=33) | 0/0 | 0.00% |
| `what` vs `where` | 0.66 (n=58) | 0.88 (n=84) | 0/0 | 0.00% |
| `four` vs `five` | 0.50 (n=4) | 0.00 (n=1) | 0/0 | 0.00% |
| `one` vs `two` | 0.88 (n=40) | 0.78 (n=27) | 0/1 | 1.49% |
| `two` vs `three` | 0.78 (n=27) | 1.00 (n=10) | 0/0 | 0.00% |
| `eat` vs `drink` | _not in trained set_ | | | |
| `hello` vs `goodbye` | 0.83 (n=23) | 0.57 (n=14) | 1/3 | 10.81% |

## Top 10 most-confused pairs (any, sorted by count)

| True → | Predicted | Count |
|---|---|---|
| `who` | `where` | 16 |
| `who` | `deaf` | 10 |
| `sleep` | `eat` | 9 |
| `help` | `who` | 8 |
| `help` | `how` | 8 |
| `nice` | `please` | 6 |
| `eat` | `sleep` | 6 |
| `deaf` | `where` | 6 |
| `what` | `thank_you` | 5 |
| `no` | `yes` | 5 |

## Interpretation

- Do **not** promote v28. It added usable ASLLVD `five` clips, but it regressed against v27: accuracy 80.47% vs 81.84%, macro F1 0.740 vs 0.745, weighted F1 0.809 vs 0.819.
- The model comparison artifact also blocked promotion because `please` F1 dropped by 0.202.
- Keep `wave1-semlex-aslcitizen-wlasl-v27-v23recipe` bundled until a candidate beats it on accuracy, macro F1, and weighted F1 without a major key-sign regression.
- ASLLVD is still useful as source coverage for `five`; the next attempt should either add more `five` signers or adjust splitting/training so the added clips improve generalization instead of only shifting the split.
