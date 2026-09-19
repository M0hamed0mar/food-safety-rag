# ============================================================
# Food Safety RAG - Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): install Python deps
#   Stage 2 (runtime): copy app + deps, minimal image
# ============================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in a virtualenv we can copy
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install torch CPU-only first (avoids ~2GB of CUDA libraries)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Then install the rest of the requirements (torch is already satisfied)
RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pre-download the embedding model at build time so the container
# starts fast and doesn't need network access at runtime.
# (Uncomment if you want models baked into the image.)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Copy application
COPY app/ ./app/
COPY tests/ ./tests/
COPY .env.example ./

# Runtime dirs
RUN mkdir -p /app/data/uploaded_docs /app/data/faiss_index /app/data/reports /app/logs

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "app.scripts.run_api", "--host", "0.0.0.0", "--port", "8000"]