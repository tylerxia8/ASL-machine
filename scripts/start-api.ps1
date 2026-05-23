# Start API on port 8001 if 8000 is busy (update VITE_API_URL if you change port)
$ErrorActionPreference = "Stop"
$ApiDir = Join-Path $PSScriptRoot "..\apps\api"
$Port = 8001
if (-not (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue)) {
  $Port = 8000
}
Set-Location $ApiDir
$env:PYTHONPATH = ".vendor"
$py = "C:\Users\tyler\.local\bin\python3.14.exe"
if (-not (Test-Path $py)) { $py = "python" }
Write-Host "API: http://127.0.0.1:$Port"
& $py -m uvicorn main:app --reload --port $Port
