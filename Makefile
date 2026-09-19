# ============================================================
# Food Safety RAG - Makefile
# ============================================================
# Usage: make <target>
# Windows: use `make` from Git Bash, WSL, or install via choco/scoop.
# ============================================================

.PHONY: help install run ingest test benchmark docker-build docker-up docker-down docker-logs clean lint format

# ---------- Meta ----------
help:
@echo "Food Safety RAG - Available targets:"
@echo ""
@echo "  make install       Install Python dependencies"
@echo "  make run           Start the FastAPI server (local)"
@echo "  make ingest        Ingest documents from data/uploaded_docs/"
@echo "  make test          Run tests"
@echo "  make benchmark     Run the factual accuracy benchmark"
@echo ""
@echo "  make docker-build  Build the Docker image"
@echo "  make docker-up     Start the stack via docker compose"
@echo "  make docker-down   Stop the stack"
@echo "  make docker-logs   Follow container logs"
@echo ""
@echo "  make clean         Remove __pycache__, .pytest_cache, etc."
@echo "  make lint          Run ruff (if installed)"
@echo "  make format        Run black + isort (if installed)"

# ---------- Local dev ----------
install:
pip install --upgrade pip
pip install -r requirements.txt

run:
python -m app.scripts.run_api --reload

ingest:
python -m app.scripts.ingest_data

test:
pytest tests/ -v

benchmark:
python -m tests.test_01_factual_accuracy

# ---------- Docker ----------
docker-build:
docker build -t food-safety-rag:latest .

docker-up:
docker compose up --build

docker-down:
docker compose down

docker-logs:
docker compose logs -f

# ---------- Housekeeping ----------
clean:
@echo "Cleaning caches..."
@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
@find . -type f -name "*.pyc" -delete 2>/dev/null || true
@echo "Done."

lint:
@ruff check app/ tests/ 2>/dev/null || echo "Install ruff: pip install ruff"

format:
@black app/ tests/ 2>/dev/null || echo "Install black: pip install black"
@isort app/ tests/ 2>/dev/null || echo "Install isort: pip install isort"