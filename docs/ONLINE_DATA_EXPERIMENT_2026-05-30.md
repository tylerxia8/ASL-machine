# Online Data Experiment - 2026-05-30

## Source

Used the official Microsoft Research ASL Citizen archive through `ml/scripts/fetch_asl_citizen.py`. The script range-reads the ZIP directory and downloads only clips matching the Wave 1 roster, so the full archive is not stored locally.

Secondary sources considered:

- Sem-Lex remains the preferred native-signer source for the promoted model.
- WLASL contains a small public-index `five` set. The project fetcher downloads direct MP4 URLs only and skips YouTube entries unless a future pass adds a dedicated clipping workflow.

## Import Result

Command:

```bash
python ml/scripts/fetch_asl_citizen.py --clips-per-sign 20 --out-dir ml/data/incoming_online_aslcitizen
python ml/scripts/fetch_wlasl.py --signs five --clips-per-sign 20 --out-dir ml/data/incoming_online_wlasl
python ml/scripts/import_captures.py --in-dir ml/data/incoming_online_aslcitizen --resize-mode letterbox
python ml/scripts/import_captures.py --in-dir ml/data/incoming_online_wlasl --resize-mode letterbox
python ml/scripts/build_manifest.py --wave1 --signer-disjoint
python ml/scripts/extract_hand_landmarks.py --manifest ml/data/manifest.json --out-dir ml/data/hand_landmarks
```

Results:

- ASL Citizen downloaded/imported: 480 clips after adding the `EAT1` / `EAT2` gloss mapping.
- WLASL direct MP4 supplement: 3 `five` clips. The ASL SignBank URL failed TLS hostname validation, Handspeak/SigningSavvy returned 403, and the YouTube entry was skipped by default.
- Combined local dataset after adding learner recordings: 552 clips, 25 signs, 41 signers.
- Signer-disjoint split: 360 train, 55 val, 137 test.
- Hand landmark coverage: 6481/13248 frames, 48.9%.
- Missing from ASL Citizen Wave 1 fetch after the gloss-map fix: `five`.
- `five` now has a small WLASL supplement, but held-out support is still thin: 17 train, 3 val, 1 test.

## Experimental Training

Trained a local measurement-only model:

```bash
python ml/train_landmarks.py \
  --manifest ml/data/manifest.json \
  --feature-dir ml/data/hand_landmarks \
  --epochs 30 \
  --batch-size 32 \
  --model-version wave1-aslcitizen-wlasl-learner-v25-local \
  --early-stop-patience 7 \
  --min-feature-coverage 0.10
```

Results:

- Validation accuracy: 80.00%.
- Signer-disjoint test accuracy: 65.69%.
- Macro F1: 0.655.
- Weighted F1: 0.656.
- This run is source-complete for the 25-label Wave 1 roster, but it underperforms the current promoted v23 model and should not be promoted.

## Recommendation

Do not promote this local v25 experiment to the app. Use the source pipeline in the next promotable Sem-Lex plus ASL Citizen plus WLASL plus learner training job, then promote only if signer-disjoint metrics beat the current v23 baseline.

Most important remaining data gaps:

1. `five`: needs more native/public signer diversity or more self-recorded variation. The current WLASL direct-MP4 supplement is useful but too small.
2. `eat`: use the ASL Citizen `EAT1` / `EAT2` mapping plus Sem-Lex in the next full training run.
3. `how`, `who`, `deaf`, `friend`: keep as watchlist signs because they still showed weaker held-out recall than the easier classes.
