@echo off
REM Run the CLI without activating the virtualenv or typing its path.
REM
REM   fpl export-frontend --open
REM   fpl backtest
REM
REM A .cmd rather than a .ps1 on purpose: PowerShell's execution policy blocks
REM unsigned scripts on a default Windows install, and this has to just work.

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Virtualenv not found. Create it first:
  echo     python -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" -m src.cli %*
