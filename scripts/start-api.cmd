@echo off
cd /d "%~dp0..\apps\api"
set PYTHONPATH=.vendor
echo This window will stay on the API server. Open a NEW terminal for other commands.
echo Checking port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do (
  echo API already on 8000 PID %%a - http://127.0.0.1:8000/health
  goto :run8001
)
echo API: http://127.0.0.1:8000/health
C:\Users\tyler\.local\bin\python3.14.exe -m uvicorn main:app --reload --port 8000
goto :eof
:run8001
echo API: http://127.0.0.1:8001/health  (web .env.local should use port 8001)
C:\Users\tyler\.local\bin\python3.14.exe -m uvicorn main:app --reload --port 8001
