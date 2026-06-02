# Online Data Experiment - 2026-05-30

## Source

Used the official Microsoft Research ASL Citizen archive through `ml/scripts/fetch_asl_citizen.py`. The script range-reads the ZIP directory and downloads only clips matching the Wave 1 roster, so the full archive is not stored locally.

Secondary sources considered:

- Sem-Lex remains the preferred native-signer source for the promoted model.
- WLASL contains a small public-index `five` set. The project fetcher downloads direct MP4 URLs only and skips YouTube entries unless a future pass adds a dedicated clipping workflow.
- ASLLVD contains exact-gloss `FIVE` citation-form rows from four signers. The project fetcher trims the official scene videos locally from annotated frame ranges and keeps all raw/trimmed files out of git.

## Import Result

Command:

```bash
python ml/scripts/fetch_asl_citizen.py --clips-per-sign 20 --out-dir ml/data/incoming_online_aslcitizen
python ml/scripts/fetch_wlasl.py --signs five --clips-per-sign 20 --out-dir ml/data/incoming_online_wlasl
python ml/scripts/fetch_asllvd.py --signs five --clips-per-sign 20 --out-dir ml/data/incoming_online_asllvd
python ml/scripts/import_captures.py --in-dir ml/data/incoming_online_aslcitizen --resize-mode letterbox
python ml/scripts/import_captures.py --in-dir ml/data/incoming_online_wlasl --resize-mode letterbox
python ml/scripts/import_captures.py --in-dir ml/data/incoming_online_asllvd --resize-mode letterbox
python ml/scripts/build_manifest.py --wave1 --signer-disjoint
python ml/scripts/extract_hand_landmarks.py --manifest ml/data/manifest.json --out-dir ml/data/hand_landmarks
```

Results:

- ASL Citizen downloaded/imported: 480 clips after adding the `EAT1` / `EAT2` gloss mapping.
- WLASL direct MP4 supplement: 3 `five` clips. The ASL SignBank URL failed TLS hostname validation, Handspeak/SigningSavvy returned 403, and the YouTube entry was skipped by default.
- ASLLVD exact-gloss supplement: 4 trimmed `five` clips from Liz, Naomi, Brady, and Tyler.
- Combined local dataset after adding learner recordings and ASLLVD: 556 clips, 25 signs, 45 signers.
- Local signer-disjoint split after ASLLVD import: 343 train, 51 val, 162 test.
- Hand landmark coverage: 6481/13248 frames, 48.9%.
- Missing from ASL Citizen Wave 1 fetch after the gloss-map fix: `five`.
- `five` now has WLASL plus ASLLVD supplements, but held-out support is still thin in the local split: 21 train, 3 val, 1 test.

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
- This run is source-complete for the 25-label Wave 1 roster, but it underperformed the promoted v23 model that was current at the time and should not be promoted.

## Recommendation

Do not promote this local v25 experiment to the app. The follow-up GitHub Actions run `wave1-semlex-aslcitizen-wlasl-v27-v23recipe` used the same source pipeline with the stronger v23-style training recipe and should replace v23: accuracy 81.84%, macro F1 0.745, weighted F1 0.819, with the comparison script recommending PROMOTE.

The later ASLLVD-backed `wave1-semlex-aslcitizen-wlasl-asllvd-v28-five` run should **not** replace v27. It added usable ASLLVD `five` clips, but accuracy dropped to 80.47%, macro F1 dropped to 0.740, weighted F1 dropped to 0.809, and `please` F1 regressed enough for the comparison script to block promotion.

Most important remaining data gaps:

1. `five`: needs more native/public signer diversity or more self-recorded variation. WLASL plus ASLLVD helps, but held-out support is still thin.
2. `eat`: use the ASL Citizen `EAT1` / `EAT2` mapping plus Sem-Lex in the next full training run.
3. `how`, `who`, `deaf`, `friend`: keep as watchlist signs because they still showed weaker held-out recall than the easier classes.
