# Capture Plan for `wave1-semlex-full-v18-hand-landmarks`

Total additional learner clips recommended: **381**

Prioritize rows from top to bottom. Existing clips are counted from `ml/data/learner_samples/` and `ml/data/incoming/`.

| Sign | F1 | Recall | Test clips | Existing learner clips | Target | Need | Reason | Top confusions |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `five` | 0.00 | 0.00 | 1 | 1 | 30 | 29 | near-zero test support | thank_you (1) |
| `four` | 0.00 | 0.00 | 2 | 1 | 30 | 29 | near-zero test support | goodbye (2) |
| `who` | 0.36 | 0.25 | 32 | 1 | 30 | 29 | very weak F1 | deaf (11), where (11), friend (1) |
| `goodbye` | 0.43 | 0.75 | 8 | 1 | 24 | 23 | weak F1 | hello (1), how (1) |
| `how` | 0.46 | 0.43 | 7 | 1 | 24 | 23 | weak F1 | help (3), where (1) |
| `deaf` | 0.55 | 0.68 | 31 | 1 | 24 | 23 | weak F1 | where (4), goodbye (3), friend (2) |
| `hello` | 0.58 | 0.50 | 14 | 1 | 24 | 23 | weak F1 | deaf (2), sleep (2), nice (1) |
| `three` | 0.62 | 0.80 | 5 | 1 | 18 | 17 | medium F1 | thank_you (1) |
| `thank_you` | 0.64 | 0.89 | 9 | 1 | 18 | 17 | medium F1 | help (1) |
| `no` | 0.67 | 0.64 | 22 | 1 | 18 | 17 | medium F1 | yes (3), deaf (2), goodbye (2) |
| `yes` | 0.69 | 0.67 | 18 | 1 | 18 | 17 | medium F1 | friend (3), deaf (2), where (1) |
| `where` | 0.75 | 0.84 | 49 | 1 | 18 | 17 | medium F1 | deaf (3), goodbye (3), no (1) |
| `please` | 0.75 | 0.75 | 8 | 1 | 10 | 9 | learner baseline | deaf (2) |
| `two` | 0.77 | 0.63 | 19 | 1 | 10 | 9 | learner baseline | three (3), eat (2), deaf (1) |
| `what` | 0.82 | 0.69 | 29 | 1 | 10 | 9 | learner baseline | thank_you (2), hello (1), help (1) |
| `nice` | 0.86 | 0.82 | 33 | 1 | 10 | 9 | learner baseline | goodbye (3), hello (1), no (1) |
| `help` | 0.88 | 0.88 | 48 | 1 | 10 | 9 | learner baseline | nice (2), who (2), how (1) |
| `friend` | 0.88 | 1.00 | 23 | 1 | 10 | 9 | learner baseline |  |
| `sleep` | 0.88 | 0.92 | 25 | 1 | 10 | 9 | learner baseline | eat (1), thank_you (1) |
| `eat` | 0.89 | 0.87 | 38 | 1 | 10 | 9 | learner baseline | sleep (2), name (1), no (1) |
| `one` | 0.91 | 0.89 | 18 | 1 | 10 | 9 | learner baseline | deaf (1), where (1) |
| `meet` | 0.92 | 0.85 | 20 | 1 | 10 | 9 | learner baseline | how (1), thank_you (1), where (1) |
| `name` | 0.97 | 0.97 | 30 | 1 | 10 | 9 | learner baseline | who (1) |
| `sorry` | 0.97 | 0.94 | 18 | 1 | 10 | 9 | learner baseline | goodbye (1) |
| `water` | 0.97 | 1.00 | 37 | 1 | 10 | 9 | learner baseline |  |
