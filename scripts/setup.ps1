param(
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $ProjectRoot

Write-Host "[1/4] Creating virtual environment (.venv) if needed..."
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

Write-Host "[2/4] Installing dependencies..."
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "[3/4] Setting PYTHONPATH for this session..."
$env:PYTHONPATH = "$ProjectRoot\src"

if (-not $SkipTrain) {
    Write-Host "[4/4] Training model..."
    & ".venv\Scripts\python.exe" "scripts\train.py"
} else {
    Write-Host "[4/4] Training skipped (-SkipTrain)."
}

Write-Host "Setup complete."
Write-Host "Next: run .\scripts\run_local.ps1"
