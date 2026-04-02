$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Virtual environment not found. Run .\scripts\setup.ps1 first."
    exit 1
}

$env:PYTHONPATH = "$ProjectRoot\src"

Write-Host "Starting API in a new PowerShell window on http://localhost:8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot\src'; .\.venv\Scripts\python.exe scripts\run_api.py"

Start-Sleep -Seconds 2

Write-Host "Starting Streamlit in current window on http://localhost:8501 ..."
& ".venv\Scripts\python.exe" -m streamlit run "src\fakenews\app\streamlit_app.py"
