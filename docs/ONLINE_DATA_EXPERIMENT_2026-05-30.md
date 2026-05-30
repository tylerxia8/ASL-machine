# Online Data Experiment - 2026-05-30

## Source

Used the official Microsoft Research ASL Citizen archive through `ml/scripts/fetch_asl_citizen.py`. The script range-reads the ZIP directory and downloads only clips matching the Wave 1 roster, so the full archive is not stored locally.

Secondary sources considered:

- Sem-Lex remains the preferred native-signer source for the promoted model.
- WLASL/ASLLVD may contain extra `five` / `eat` examples, but they were not imported in this pass because ASL Citizen is already wired into the project pipeline and has cleaner isolated-sign coverage for most Wave 1 signs.

## Import Result

Command:

```bash
python ml/scripts/fetch_asl_citizen.py --clips-per-sign 20 --out-dir ml/data/incoming_online_aslcitizen
python ml/scripts/import_captures.py --in-dir ml/data/incoming_online_aslcitizen --resize-mode letterbox
python ml/scripts/build_manifest.py --wave1 --signer-disjoint
python ml/scripts/extract_hand_landmarks.py --manifest ml/data/manifest.json --out-dir ml/data/hand_landmarks
```

Results:

- ASL Citizen downloaded/imported: 460 clips, 0 import failures.
- Combined local dataset after adding learner recordings: 529 clips, 24 signs, 38 signers.
- Signer-disjoint split: 332 train, 48 val, 149 test.
- Hand landmark coverage: 6244/12696 frames, 49.2%.
- Missing from ASL Citizen Wave 1 fetch: `five`, `eat`.
- `five` is present only from learner recordings, so it has no signer-disjoint held-out test support.

## Experimental Training

Trained a local measurement-only model:

```bash
python ml/train_landmarks.py \
  --manifest ml/data/manifest.json \
  --feature-dir ml/data/hand_landmarks \
  --epochs 25 \
  --batch-size 32 \
  --model-version wave1-aslcitizen-learner-v24-local \
  --early-stop-patience 6 \
  --min-feature-coverage 0.10
```

Results:

- Validation accuracy: 77.08%.
- Signer-disjoint test accuracy: 72.48%.
- Macro F1: 0.699.
- Weighted F1: 0.726.
- `who` test recall improved to 0.67 in this experimental model, but this model is not promotable as-is because it has only 24 classes and excludes `eat`.

## Recommendation

Do not promote this local v24 experiment to the app. Use it as evidence that ASL Citizen is useful supplemental data, then run the next promotable training job with Sem-Lex plus ASL Citizen plus learner clips in one 25-class pipeline.

Most important remaining data gaps:

1. `five`: needs native/public signer diversity or more self-recorded variation.
2. `eat`: needs public signer clips from Sem-Lex or another compatible source.
3. `how`, `who`, `deaf`, `friend`: keep as watchlist signs because they still showed weaker held-out recall than the easier classes.
