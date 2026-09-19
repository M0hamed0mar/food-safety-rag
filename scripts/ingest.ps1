
============================================================
Food Safety RAG - Ingest documents
============================================================
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\venv")) {
Write-Host "venv not found. Run .\scripts\setup.ps1 first." -ForegroundColor Red
exit 1
}

& ".\venv\Scripts\Activate.ps1"
Write-Host "Ingesting documents from data/uploaded_docs/..." -ForegroundColor Green
python -m app.scripts.ingest_data
