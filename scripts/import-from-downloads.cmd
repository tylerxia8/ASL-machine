@echo off
cd /d "%~dp0.."
powershell -ExecutionPolicy Bypass -File "%~dp0import-from-downloads.ps1"
pause
