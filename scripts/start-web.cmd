@echo off
cd /d "%~dp0..\apps\web"
echo Web: http://localhost:5173
call npm.cmd run dev
