
============================================================
Food Safety RAG - Local Setup
============================================================
Run once after cloning the repo.
$ErrorActionPreference = "Stop"

Write-Host "=== Food Safety RAG - Local Setup ===" -ForegroundColor Cyan
Write-Host ""

1. Check Python
pyVersion = python --version 2>&1 Write-Host "Python: pyVersion"

2. Create venv
if (-not (Test-Path ".\venv")) {
Write-Host "Creating virtual environment..."
python -m venv venv
}

3. Activate + install
Write-Host "Installing dependencies..."
& ".\venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt

4. Copy .env
if (-not (Test-Path "..env")) {
Write-Host "Creating .env from .env.example..."
Copy-Item "..env.example" "..env"
Write-Host ""
Write-Host "IMPORTANT: Edit .env and set GROQ_API_KEY" -ForegroundColor Yellow
}

5. Create runtime directories
New-Item -ItemType Directory -Force -Path ".\data\uploaded_docs" | Out-Null
New-Item -ItemType Directory -Force -Path ".\data\faiss_index" | Out-Null
New-Item -ItemType Directory -Force -Path ".\data\reports" | Out-Null
New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host " 1. Edit .env and set GROQ_API_KEY"
Write-Host " 2. Place source PDFs in data/uploaded_docs/"
Write-Host " 3. Run: .\scripts\ingest.ps1"
Write-Host " 4. Run: .\scripts\dev.ps1"
