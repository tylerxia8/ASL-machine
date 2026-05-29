@echo off
cd /d "%~dp0..\apps\web"
echo Web: http://localhost:5173
echo If 5173 is busy, Vite will print the next available localhost URL.
call npm.cmd run dev
