@echo off
cd /d "%~dp0.."
powershell -ExecutionPolicy Bypass -File "%~dp0continue-wave1.ps1"
pause
