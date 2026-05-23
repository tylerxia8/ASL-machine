# Wave 1: import captures -> manifest -> train 3D CNN -> export ONNX -> sync to web
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root "ml\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  Write-Host "Creating ML venv..."
  $pyGlobal = "C:\Users\tyler\.local\bin\python3.14.exe"
  if (-not (Test-Path $pyGlobal)) { $pyGlobal = "python" }
  Set-Location (Join-Path $Root "ml")
  & $pyGlobal -m venv .venv
  .\.venv\Scripts\pip install -r requirements.txt -q
  $Py = ".\.venv\Scripts\python.exe"
  Set-Location $Root
}

Set-Location $Root
Write-Host "==> Import captures from ml/data/incoming/"
& $Py ml/scripts/import_captures.py

Write-Host "==> Build manifest (Wave 1, signer-disjoint test split)"
& $Py ml/scripts/build_manifest.py --wave1 --signer-disjoint

$clipCount = (Get-ChildItem -Recurse ml/data/clips -Filter *.npz -ErrorAction SilentlyContinue | Measure-Object).Count
$signCount = (Import-Csv content/wave1_signs.csv | Measure-Object).Count
$target = $signCount * 30 * 2
if ($clipCount -lt 100) {
  Write-Warning "Only $clipCount .npz clips found. Target ~$target (30 x $signCount signs x 2 signers). Continuing anyway."
}

Write-Host "==> Train 3D CNN (from scratch, no pretrained weights)"
& $Py ml/train.py --manifest ml/data/manifest.json --model-version wave1-v1 --epochs 20

Write-Host "==> Eval on held-out signer"
& $Py ml/eval.py --checkpoint ml/checkpoints/wave1-v1/best.pt --manifest ml/data/manifest.json

Write-Host "==> Export ONNX for browser inference"
& $Py ml/export_onnx.py --checkpoint ml/checkpoints/wave1-v1/best.pt

Write-Host "==> Sync to web/public/models"
Set-Location (Join-Path $Root "apps/web")
npm run sync-model

Write-Host "Done. Start API + npm run dev, then walk docs/WAVE1_DRY_RUN.md"
