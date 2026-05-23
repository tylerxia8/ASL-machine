# Move Wave-1 capture files from Downloads into ml/data/incoming.
# Source of truth for trained signs: content/wave1_signs.csv (read at runtime).
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Incoming = Join-Path $Root "ml\data\incoming"
$Downloads = Join-Path $env:USERPROFILE "Downloads"
New-Item -ItemType Directory -Force -Path $Incoming | Out-Null

$wave1Csv = Join-Path $Root "content\wave1_signs.csv"
if (-not (Test-Path $wave1Csv)) {
  Write-Error "Missing $wave1Csv"
  exit 1
}
$signIds = Import-Csv $wave1Csv | ForEach-Object { $_.sign_id }
$signAlt = ($signIds | ForEach-Object { [regex]::Escape($_) }) -join "|"
# Filenames written by the capture page: {sign_id}_{signer_id}_{timestamp}.{webm|json}
$pattern = "^($signAlt)_signer_[a-z0-9]+_\d+\.(webm|json)$"

$moved = 0
$skippedNoMatch = 0
Get-ChildItem $Downloads -File | Where-Object {
  $_.Extension -in @(".webm", ".json") -and $_.Name -match $pattern
} | ForEach-Object {
  $dest = Join-Path $Incoming $_.Name
  if (Test-Path $dest) {
    $suffix = "_" + [guid]::NewGuid().ToString("n").Substring(0, 6)
    $dest = Join-Path $Incoming ($_.BaseName + $suffix + $_.Extension)
  }
  Move-Item $_.FullName $dest -Force
  Write-Host "Moved $($_.Name)"
  $moved++
}

# Diagnostic: count downloads that look like captures but don't match the trained roster.
Get-ChildItem $Downloads -File | Where-Object {
  $_.Extension -in @(".webm", ".json") -and
  $_.Name -match "^.+_signer_[a-z0-9]+_\d+\.(webm|json)$" -and
  $_.Name -notmatch $pattern
} | ForEach-Object {
  $skippedNoMatch++
  Write-Host "Skipped (sign not in wave1_signs.csv): $($_.Name)"
}

if ($moved -eq 0) {
  Write-Host ""
  Write-Host "No matching capture files in Downloads."
  Write-Host "Expected name format: {sign_id}_signer_{x}_{timestamp}.webm"
  Write-Host "Trained signs ($($signIds.Count)): $($signIds -join ', ')"
  Write-Host "Target folder: $Incoming"
} else {
  Write-Host ""
  Write-Host "Moved $moved file(s) to $Incoming"
  if ($skippedNoMatch -gt 0) {
    Write-Host "Skipped $skippedNoMatch file(s) that don't match the trained-sign roster."
  }
  Write-Host "Next: .\scripts\continue-wave1.ps1"
}
