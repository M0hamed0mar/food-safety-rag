
============================================================
Food Safety RAG - Development Server
============================================================
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\venv")) {
Write-Host "venv not found. Run .\scripts\setup.ps1 first." -ForegroundColor Red
exit 1
}

& ".\venv\Scripts\Activate.ps1"
Write-Host "Starting dev server at http://localhost:8000" -ForegroundColor Green
python -m app.scripts.run_api --reload
