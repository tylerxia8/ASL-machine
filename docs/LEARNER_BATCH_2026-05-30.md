# Learner Batch Report - 2026-05-30

Imported 69 newly recorded clips from `Downloads` into `ml/data/clips/`.

## Clip Counts

| Sign | Clips |
|---|---:|
| `five` | 18 |
| `four` | 5 |
| `goodbye` | 5 |
| `how` | 19 |
| `please` | 5 |
| `thank_you` | 4 |
| `three` | 4 |
| `who` | 9 |

## Technical QA

- Import result: 69 imported, 0 failed.
- Manifest split: 49 train, 10 val, 10 test.
- Hand-landmark coverage: 1513/1656 frames, 91.4%.
- Visual contact sheet: `ml/reports/contact_sheets/learner_batch_2026-05-30/contact_sheet_train.jpg`.
- Caveat: all clips are `signer_a`, so local evaluation is not signer-disjoint.

## Current v23 Model Predictions

| Sign | Correct | Model predictions |
|---|---:|---|
| `five` | 0/18 | `thank_you` x9, `nice` x4, `four` x3, `goodbye` x2 |
| `four` | 5/5 | `four` x5 |
| `goodbye` | 5/5 | `goodbye` x5 |
| `how` | 9/19 | `how` x9, `sorry` x7, `friend` x1, `eat` x1, `what` x1 |
| `please` | 5/5 | `please` x5 |
| `thank_you` | 4/4 | `thank_you` x4 |
| `three` | 4/4 | `three` x4 |
| `who` | 0/9 | `where` x8, `sleep` x1 |

## Recommendation

This batch is useful and should be kept for the next retraining run. It should not replace the full 25-sign training set by itself because it currently covers 8 signs from one signer. The highest-value next captures are:

1. `who`: record 20-30 more clips, with more chin-area contrast and a clear difference from `where`.
2. `five`: record 10-15 more clips with varied hand position, distance, and lighting.
3. `how`: record 10-15 more clips, especially slower repetitions with the wrist-turn visible.
4. `four`: optional; the current model recognized the new clips, but more examples would help future retraining.

`please`, `thank_you`, and `three` looked good against the current model, so they are lower priority now.
