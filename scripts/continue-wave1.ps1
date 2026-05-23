# Wave 1 continue: import any incoming captures, then train 3D CNN end-to-end.
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

$incoming = @(Get-ChildItem "ml\data\incoming\*.json" -ErrorAction SilentlyContinue)
if ($incoming.Count -gt 0) {
  Write-Host "==> Import $($incoming.Count) capture files"
  & $Py ml/scripts/import_captures.py
}

$npz = @(Get-ChildItem "ml\data\clips" -Recurse -Filter "*.npz" -ErrorAction SilentlyContinue)
if ($npz.Count -lt 50) {
  Write-Error "Only $($npz.Count) real clips found. Record at /capture (target: 30+ clips/sign/signer) before training."
  exit 1
}

Write-Host "==> Build manifest"
& $Py ml/scripts/build_manifest.py --wave1 --signer-disjoint

Write-Host "==> Train wave1-v1 (3D CNN)"
& $Py ml/train.py --manifest ml/data/manifest.json --model-version wave1-v1 --epochs 20

Write-Host "==> Eval"
& $Py ml/eval.py --checkpoint ml/checkpoints/wave1-v1/best.pt --manifest ml/data/manifest.json

Write-Host "==> Export ONNX"
& $Py ml/export_onnx.py --checkpoint ml/checkpoints/wave1-v1/best.pt

Set-Location (Join-Path $Root "apps\web")
npm.cmd run sync-model

Write-Host ""
Write-Host "Done. Restart web dev server if running, then:"
Write-Host "  Practice: http://localhost:5173/"
Write-Host "  Capture:  /capture - save JSON to ml\data\incoming\ then re-run this script"
