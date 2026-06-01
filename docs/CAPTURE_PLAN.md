# Capture Plan for `wave1-semlex-aslcitizen-wlasl-v27-v23recipe`

Total additional learner clips recommended: **24**

Prioritize rows from top to bottom. Existing clips are counted from `ml/data/learner_samples/` and `ml/data/incoming/`.

| Sign | F1 | Recall | Test clips | Existing learner clips | Target | Need | Reason | Top confusions |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `five` | 0.00 | 0.00 | 1 | 6 | 30 | 24 | near-zero test support | thank_you (1) |
| `four` | 0.29 | 0.33 | 3 | 39 | 30 | 0 | very weak F1 | hello (2) |
| `who` | 0.55 | 0.48 | 54 | 96 | 24 | 0 | weak F1 | where (16), yes (7), deaf (2) |
| `goodbye` | 0.61 | 0.77 | 13 | 55 | 18 | 0 | medium F1 | four (1), hello (1), three (1) |
| `three` | 0.67 | 0.90 | 10 | 49 | 18 | 0 | medium F1 | thank_you (1) |
| `how` | 0.67 | 0.86 | 14 | 51 | 18 | 0 | medium F1 | help (1), what (1) |
| `yes` | 0.69 | 0.88 | 33 | 78 | 18 | 0 | medium F1 | friend (3), goodbye (1) |
| `no` | 0.74 | 0.64 | 39 | 83 | 18 | 0 | medium F1 | yes (6), three (3), goodbye (2) |
| `deaf` | 0.75 | 0.71 | 62 | 114 | 18 | 0 | medium F1 | who (8), where (6), friend (1) |
