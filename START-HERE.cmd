@echo off
REM Double-click this file from Explorer, or run from any shell:
REM   "c:\Users\tyler\Downloads\Gauntlet Superbuilders ASL\START-HERE.cmd"
cd /d "%~dp0"
echo.
echo === ASL Pilot - choose ===
echo 1) Start API
echo 2) Start Web
echo 3) Start both (two windows)
echo.
set /p choice=Enter 1, 2, or 3:
if "%choice%"=="1" goto api
if "%choice%"=="2" goto web
if "%choice%"=="3" goto both
goto api

:api
start "ASL API" cmd /k "%~dp0scripts\start-api.cmd"
goto eof

:web
start "ASL Web" cmd /k "%~dp0scripts\start-web.cmd"
goto eof

:both
start "ASL API" cmd /k "%~dp0scripts\start-api.cmd"
timeout /t 2 /nobreak >nul
start "ASL Web" cmd /k "%~dp0scripts\start-web.cmd"
goto eof

:eof
pause
