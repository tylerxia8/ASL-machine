# Uses npm.cmd to avoid PowerShell execution policy blocking npm.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\apps\web")
Write-Host "Web: http://localhost:5173"
& npm.cmd run dev
