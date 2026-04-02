@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Running setup first...
  powershell -ExecutionPolicy Bypass -File ".\scripts\setup.ps1"
  if errorlevel 1 exit /b %errorlevel%
)

powershell -ExecutionPolicy Bypass -File ".\scripts\run_local.ps1"
