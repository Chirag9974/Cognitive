@echo off
setlocal enabledelayedexpansion
title Cognitive Load Predictor

rem One command for the whole app: server + dashboard + cognitive test.
rem On a fresh clone this also sets up the virtual environment and installs
rem requirements.txt, so the only prerequisite is Python itself.
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "PORT=8765"

call :ensure_venv
if errorlevel 1 goto :failed

rem Free the port if a previous server is still holding it.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8765" ^| findstr "LISTENING"') do (
  echo   Stopping previous server on port %PORT% -- PID %%P
  taskkill /PID %%P /F >nul 2>&1
)

echo.
echo   Cognitive Load Predictor V2
echo   =============================================
echo   Cognitive test : http://127.0.0.1:%PORT%/#/test
echo   Dashboard      : http://127.0.0.1:%PORT%/
echo   API docs       : http://127.0.0.1:%PORT%/docs
echo.
echo   Your browser opens on the test. Take the four tasks, then
echo   the dashboard shows your score, breakdown and suggestions.
echo.
echo   Closing this window (or Ctrl+C) stops the server.
echo.

rem Open the browser a few seconds later, once the server is up.
start "open-test" /min cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:%PORT%/#/test"

"%PY%" serve_api.py --port %PORT%

echo.
echo   Server stopped.
pause
exit /b 0


:ensure_venv
if exist "%PY%" (
  "%PY%" -c "import fastapi, sklearn" >nul 2>&1
  if not errorlevel 1 exit /b 0
  echo   Virtual environment found but incomplete -- installing requirements...
  call :pip_install
  exit /b !errorlevel!
)

echo.
echo   First run: setting up the virtual environment (a few minutes)...
echo.

set "BOOTSTRAP_PY="
where py >nul 2>&1 && set "BOOTSTRAP_PY=py -3"
if not defined BOOTSTRAP_PY where python >nul 2>&1 && set "BOOTSTRAP_PY=python"
if not defined BOOTSTRAP_PY (
  echo   [!] Python was not found on PATH.
  echo       Install it from https://www.python.org/downloads/ and tick
  echo       "Add python.exe to PATH", then run this file again.
  exit /b 1
)

echo   Using %BOOTSTRAP_PY%
%BOOTSTRAP_PY% -m venv .venv
if errorlevel 1 (
  echo   [!] Could not create the virtual environment.
  exit /b 1
)

call :pip_install
exit /b !errorlevel!


:pip_install
rem --upgrade pip first: old pip cannot see wheels for newer Python releases.
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo   [!] Dependency install failed. Check your internet connection and retry:
  echo       %PY% -m pip install -r requirements.txt
  exit /b 1
)
echo   Dependencies ready.
exit /b 0


:failed
echo.
echo   Setup did not complete -- see the message above.
pause
exit /b 1
