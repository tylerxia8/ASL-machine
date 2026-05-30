# Learner Batch Report - 2026-05-30

Imported 31 newly recorded clips from `Downloads` into `ml/data/clips/`.

## Clip Counts

| Sign | Clips |
|---|---:|
| `five` | 5 |
| `goodbye` | 5 |
| `how` | 4 |
| `please` | 5 |
| `thank_you` | 4 |
| `three` | 4 |
| `who` | 4 |

## Technical QA

- Import result: 31 imported, 0 failed.
- Manifest split: 17 train, 7 val, 7 test.
- Hand-landmark coverage: 672/744 frames, 90.3%.
- Visual contact sheet: `ml/reports/contact_sheets/learner_batch_2026-05-30/contact_sheet_train.jpg`.
- Caveat: all clips are `signer_a`, so local evaluation is not signer-disjoint.

## Current v23 Model Predictions

| Sign | Correct | Model predictions |
|---|---:|---|
| `five` | 0/5 | `thank_you` x2, `goodbye` x1, `nice` x1, `four` x1 |
| `goodbye` | 4/5 | `goodbye` x4, `no` x1 |
| `how` | 1/4 | `sorry` x2, `how` x1, `friend` x1 |
| `please` | 5/5 | `please` x5 |
| `thank_you` | 4/4 | `thank_you` x4 |
| `three` | 4/4 | `three` x4 |
| `who` | 0/4 | `where` x3, `sleep` x1 |

## Recommendation

This batch is useful and should be kept for the next retraining run. The highest-value next captures are:

1. `five`: record 20-30 more clips.
2. `who`: record 20-30 more clips, with more chin-area contrast and a clear difference from `where`.
3. `how`: record 15-20 more clips.
4. `goodbye`: record 5-10 more clips if time allows.

`please`, `thank_you`, and `three` looked good against the current model, so they are lower priority now.
