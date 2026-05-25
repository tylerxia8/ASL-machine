# Mirroring Sem-Lex to Hugging Face Hub

## Why

Google Drive's anonymous-download quota gets exhausted after a couple of partial dispatches of the 23.7 GB `train.tar.gz`. Once locked, the file is unreachable for ~24 hours. This is fine for an occasional one-off run but kills iteration — every dispatch costs quota, even on success.

Hugging Face Hub is the standard distribution channel for ML datasets. Public dataset files have no equivalent per-file daily quota, so once the four Sem-Lex files (1 metadata CSV + 3 video tarballs) are mirrored to a public HF dataset, the training workflow can hit them as many times as you want.

The mirroring is a **one-time operation** that takes about 60 minutes (limited by your home upload bandwidth and HF's API). After it's done, the workflow swaps a single secret value and proceeds without quota ever being a concern again.

We mirror **only the 4 files we actually use**. The 3 `*-poses.tar.gz` files are precomputed-pose-landmark output that rubric Req 7 forbids us from consuming, and we never touch them; mirroring them would be a waste of disk and a misleading signal that we use them.

## Prereqs

1. **~30 GB free disk** on your local machine (the script downloads one file at a time and deletes after each upload, so peak usage is just the largest single file: train.tar.gz at 23.7 GB).
2. **Hugging Face account** (free) — sign up at https://huggingface.co/join
3. **HF access token with WRITE scope** — https://huggingface.co/settings/tokens → "New token" → name it `asl-pilot-mirror`, role: write, copy the `hf_...` value.
4. **An empty HF dataset repo** — https://huggingface.co/new-dataset:
   - Owner: your username
   - Dataset name: e.g. `sem-lex-pilot-mirror`
   - Visibility: **Public** (so the CI runner can fetch without auth)
   - License: pick `apache-2.0` to match Sem-Lex's own license
   - Click Create dataset
5. **Sem-Lex Drive file IDs** — from the email you got after accepting Sem-Lex's terms-of-use form. The same JSON you would have put in `SEMLEX_DRIVE_FILES`:
   ```json
   {
     "metadata": "<file_id_of_semlex_metadata.csv>",
     "train":    "<file_id_of_train.tar.gz>",
     "val":      "<file_id_of_val.tar.gz>",
     "test":     "<file_id_of_test.tar.gz>"
   }
   ```
   (The 3 `*-poses.tar.gz` Drive IDs are NOT included.)

## Run the mirror

```powershell
# Make sure huggingface_hub and gdown are installed
.\ml\.venv\Scripts\pip.exe install --quiet huggingface_hub gdown

# Set token + run
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
.\ml\.venv\Scripts\python.exe scripts/mirror_drive_to_hf.py `
  --hf-repo your-username/sem-lex-pilot-mirror `
  --drive-files-json '{"metadata":"1pkX8_...","train":"1jiUasW...","val":"1Vvr...","test":"1uYoM..."}'
```

What happens:

1. For each of `metadata`, `train`, `val`, `test`:
   - Downloads from Drive via gdown (handles the confirm interstitial)
   - Uploads to `your-username/sem-lex-pilot-mirror/data/{filename}`
   - Deletes the local copy
2. Prints the final URLs as a JSON snippet at the end.

If something interrupts you (network drop, ctrl-C), re-run with `--skip-existing` to resume — already-uploaded files won't be re-downloaded.

## After mirroring: switch the workflow secret

1. Copy the JSON snippet the script printed:
   ```json
   {
     "metadata": "https://huggingface.co/datasets/your-username/sem-lex-pilot-mirror/resolve/main/data/semlex_metadata.csv",
     "train":    "https://huggingface.co/datasets/.../resolve/main/data/train.tar.gz",
     "val":      "https://huggingface.co/datasets/.../resolve/main/data/val.tar.gz",
     "test":     "https://huggingface.co/datasets/.../resolve/main/data/test.tar.gz"
   }
   ```
2. Go to https://github.com/tylerxia8/ASL-machine/settings/secrets/actions
3. Click **New repository secret**
4. Name: `SEMLEX_DATA_URLS`
5. Value: paste the JSON above
6. Save

The workflow checks `SEMLEX_DATA_URLS` first; if set, it uses the HF URLs and ignores `SEMLEX_DRIVE_FILES`. The Drive secret can stay around as a fallback (the fetcher won't touch it as long as the URLs secret exists).

## After switching

- Dispatch `Train Wave 1` as normal. The fetcher will log `Using SEMLEX_DATA_URLS (direct HTTPS, recommended)` at the top.
- No more `⚠ QUOTA on Sem-Lex …` errors.
- Iterate as much as you want.

## Cost

- HF Hub dataset storage: **free** for public datasets up to 300 GB (your mirror is ~30 GB).
- HF Hub bandwidth: **free**, no quota for public datasets.
- Sem-Lex licensing: Apache-2.0 permits redistribution as long as you preserve the license notice. The script writes a generic README to the repo on first upload that points back to the canonical Sem-Lex source.

## Reverting

If for some reason you want to go back to Drive, just delete the `SEMLEX_DATA_URLS` secret from GitHub. The workflow will fall back to `SEMLEX_DRIVE_FILES` automatically.
