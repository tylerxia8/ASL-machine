param(
  [string]$BaselineTag = "wave1-semlex-aslcitizen-v23-augmented-landmarks",
  [switch]$SkipCompare
)

# Wave 1 continue: import any incoming captures, then train the hand-landmark model.
# Aborts loudly if there is no real capture data (synthetic seeding has been removed).
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Py = Join-Path $Root "ml\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  Write-Host "Creating ML venv..."
  C:\Users\tyler\.local\bin\python3.14.exe -m venv ml\.venv
  & (Join-Path $Root "ml\.venv\Scripts\pip.exe") install -r ml\requirements.txt -q
}

$incoming = @(Get-ChildItem "ml\data\incoming\*" -Include *.webm,*.mp4,*.mov,*.mkv,*.avi,*.json -ErrorAction SilentlyContinue)
if ($incoming.Count -gt 0) {
  Write-Host "==> Import $($incoming.Count) capture files"
  & $Py ml/scripts/import_captures.py --resize-mode letterbox
}

$npz = @(Get-ChildItem "ml\data\clips" -Recurse -Filter "*.npz" -ErrorAction SilentlyContinue)
if ($npz.Count -lt 50) {
  Write-Error "Only $($npz.Count) real clips found. Record at /capture (target: 30+ clips/sign/signer) before training."
  exit 1
}

Write-Host "==> Build manifest"
& $Py ml/scripts/build_manifest.py --wave1 --signer-disjoint

Write-Host "==> Extract hand landmarks"
& $Py ml/scripts/extract_hand_landmarks.py --manifest ml/data/manifest.json --out-dir ml/data/hand_landmarks

Write-Host "==> Train wave1-local-hand-landmarks"
& $Py ml/train_landmarks.py `
  --manifest ml/data/manifest.json `
  --feature-dir ml/data/hand_landmarks `
  --model-version wave1-local-hand-landmarks `
  --epochs 40 `
  --min-feature-coverage 0 `
  --landmark-noise-std 0.02 `
  --landmark-dropout-prob 0.10

Write-Host "==> Eval"
& $Py ml/eval_landmarks.py `
  --checkpoint ml/checkpoints/wave1-local-hand-landmarks/best.pt `
  --manifest ml/data/manifest.json `
  --feature-dir ml/data/hand_landmarks

Write-Host "==> Build recognition calibration and capture plan"
& $Py ml/scripts/build_recognition_calibration.py
& $Py ml/scripts/build_capture_plan.py --out docs/CAPTURE_PLAN.md

if (-not $SkipCompare -and $BaselineTag) {
  Write-Host "==> Compare against $BaselineTag"
  $BaselineDir = Join-Path $Root "ml\exports\baseline"
  New-Item -ItemType Directory -Force -Path $BaselineDir | Out-Null
  gh release download $BaselineTag --pattern eval_metrics.json --dir $BaselineDir --clobber
  & $Py ml/scripts/compare_models.py `
    --base (Join-Path $BaselineDir "eval_metrics.json") `
    --candidate ml/exports/eval_metrics.json `
    --out ml/exports/model_comparison.md
}

Write-Host "==> Export ONNX"
& $Py ml/export_landmark_onnx.py --checkpoint ml/checkpoints/wave1-local-hand-landmarks/best.pt

Set-Location (Join-Path $Root "apps\web")
npm.cmd run sync-model

Write-Host ""
Write-Host "Done. Restart web dev server if running, then:"
Write-Host "  Practice: http://localhost:5174/"
Write-Host "  Capture:  /capture - move downloaded videos to ml\data\incoming\ then re-run this script"
