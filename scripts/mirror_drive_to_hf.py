"""One-time helper: mirror the Sem-Lex Drive files to a Hugging Face Hub dataset.

Why: Google Drive's anonymous-download quota gets exhausted after a couple of
partial dispatches of the 23.7 GB train.tar.gz. Hugging Face Hub has no
equivalent throttle for public dataset files and is the standard distribution
channel for ML datasets, so mirroring once gives us a stable source URL that
the CI training workflow can hit reliably.

What this script does:
  1. For each role (metadata, train, val, test):
     - Downloads the file from Drive via gdown (handles the confirm interstitial)
     - Uploads it to {hf_repo}/data/{role}.{ext}
     - Deletes the local copy
  2. Prints a JSON snippet you paste into the SEMLEX_DATA_URLS GitHub secret.

What this script does NOT do:
  - Touch the 3 *-poses.tar.gz files. They contain pretrained-derived pose
    features that rubric Req 7 forbids; we never use them, so don't mirror them.
  - Create or modify any HF repo besides --hf-repo. The repo must exist and the
    auth token must have write access.

Prereqs:
  - A Hugging Face account + access token with WRITE scope:
        https://huggingface.co/settings/tokens
  - An empty HF *dataset* repo created in advance (the script does not create it):
        https://huggingface.co/new-dataset
    Make the dataset public so the CI runner can fetch without auth.
  - ~30 GB free local disk (the script downloads one file at a time and
    deletes after upload, so peak usage is just the largest single file).
  - `huggingface_hub` and `gdown` installed (run `pip install huggingface_hub
    gdown` if missing — they aren't in the project requirements because this
    is a one-off helper).

Usage:
    export HF_TOKEN=hf_xxxxxxxx
    python scripts/mirror_drive_to_hf.py \\
        --hf-repo your-username/sem-lex-pilot-mirror \\
        --drive-files-json '{"metadata":"1pkX8_...","train":"1jiUasW...","val":"1Vvr...","test":"1uYoM..."}'

After it finishes you'll see:

    Paste this into the SEMLEX_DATA_URLS GitHub secret:
    {
      "metadata": "https://huggingface.co/datasets/.../resolve/main/data/semlex_metadata.csv",
      "train":    "https://huggingface.co/datasets/.../resolve/main/data/train.tar.gz",
      "val":      "https://huggingface.co/datasets/.../resolve/main/data/val.tar.gz",
      "test":     "https://huggingface.co/datasets/.../resolve/main/data/test.tar.gz"
    }

Then in the Train Wave 1 workflow, the fetcher will see SEMLEX_DATA_URLS
and stream directly from HF instead of Drive — same `r|gz` tarfile pipeline,
no quota interstitials.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROLE_TO_FILENAME = {
    "metadata": "semlex_metadata.csv",
    "train": "train.tar.gz",
    "val": "val.tar.gz",
    "test": "test.tar.gz",
}


def _drive_download(file_id: str, dest: Path) -> None:
    import gdown
    print(f"  Downloading from Drive: {file_id} → {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest), quiet=False, resume=True, fuzzy=True)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"gdown produced empty/missing file at {dest}")
    print(f"  ✓ {dest.stat().st_size / 1e9:.2f} GB on disk")


def _hf_upload(local: Path, hf_repo: str, repo_path: str, token: str) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"  Uploading to HF: {hf_repo} → {repo_path}")
    t0 = time.monotonic()
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=repo_path,
        repo_id=hf_repo,
        repo_type="dataset",
    )
    dur = time.monotonic() - t0
    mb_per_s = (local.stat().st_size / 1e6) / max(dur, 1e-6)
    print(f"  ✓ Uploaded in {dur:.0f}s ({mb_per_s:.1f} MB/s)")
    return f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{repo_path}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hf-repo", required=True,
                   help="e.g. yourname/sem-lex-pilot-mirror (must exist + be writable by your token)")
    p.add_argument("--drive-files-json", default=os.environ.get("SEMLEX_DRIVE_FILES"),
                   help='JSON dict mapping role → Drive file ID. Defaults to env var SEMLEX_DRIVE_FILES.')
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF write token. Defaults to env var HF_TOKEN.")
    p.add_argument("--work-dir", default="ml/data/semlex_mirror_tmp",
                   help="Temp dir for one-file-at-a-time downloads. Deleted as we go.")
    p.add_argument("--keep-local", action="store_true",
                   help="Keep local files after upload (default: delete to free disk).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip files already in the HF repo (by path). Lets you resume after an interrupt.")
    args = p.parse_args()

    if not args.token:
        print("ERROR: --token (or HF_TOKEN env var) required.", file=sys.stderr)
        return 1
    if not args.drive_files_json:
        print("ERROR: --drive-files-json (or SEMLEX_DRIVE_FILES env var) required.", file=sys.stderr)
        return 1
    try:
        drive_files: dict[str, str] = json.loads(args.drive_files_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON: {e}", file=sys.stderr)
        return 1

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Check existing repo state if resume requested.
    existing_paths: set[str] = set()
    if args.skip_existing:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=args.token)
            existing_paths = set(api.list_repo_files(args.hf_repo, repo_type="dataset"))
            print(f"Found {len(existing_paths)} existing file(s) in {args.hf_repo}")
        except Exception as e:
            print(f"WARNING: couldn't list existing files: {e}; proceeding without skip.", file=sys.stderr)

    final_urls: dict[str, str] = {}
    for role, file_id in drive_files.items():
        if role not in ROLE_TO_FILENAME:
            print(f"Skipping unknown role {role!r}")
            continue
        filename = ROLE_TO_FILENAME[role]
        repo_path = f"data/{filename}"
        url = f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/{repo_path}"

        print(f"\n=== {role} ({filename}) ===")
        if args.skip_existing and repo_path in existing_paths:
            print(f"  ⏭ Already in HF repo, skipping download+upload")
            final_urls[role] = url
            continue

        local = work_dir / filename
        try:
            _drive_download(file_id, local)
            final_urls[role] = _hf_upload(local, args.hf_repo, repo_path, args.token)
        finally:
            if not args.keep_local and local.exists():
                local.unlink()
                print(f"  Removed local {local.name} to free disk")

    print("\n" + "=" * 70)
    print("MIRROR COMPLETE — paste this into the SEMLEX_DATA_URLS GitHub secret:")
    print("=" * 70)
    print(json.dumps(final_urls, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
