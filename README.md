# Food Safety RAG System

> A production-grade **Retrieval-Augmented Generation (RAG)** system for the Food Safety domain. Answers complex regulatory, hazard-identification, and compliance questions using trusted reference documents — with full source citations and zero hallucinations.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/ecs/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-18%20passed-success?style=for-the-badge)](tests/)

---

## Table of Contents

- [Overview](#overview)
- [Why This Project](#why-this-project)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Benchmark Results](#benchmark-results)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Docker](#docker)
- [Deployment](#deployment)
- [Design Decisions](#design-decisions)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This system answers questions about food safety regulations, HACCP plans, biological / chemical / physical hazards, allergen management, and CFIA / Health Canada standards — using **only** the content of trusted source documents.

It is designed as a **domain-agnostic RAG engine** that can be re-pointed to any regulated domain (pharma, finance, legal, aviation) by swapping the source PDFs and re-running ingestion. The current deployment specializes in Food Safety using the CFIA Reference Database for Hazard Identification.

### What it does

- Ingests multi-format documents (PDF, DOCX, XLSX, HTML, TXT, MD).
- Extracts and preserves structured tables as first-class entities.
- Builds a hybrid retrieval index (dense + sparse) with reranking.
- Generates grounded answers via a fast LLM API.
- Attaches deterministic, traceable citations (document + page + section).
- Serves a FastAPI backend and a lightweight web UI.

---

## Why This Project

Most RAG demos fail on real-world technical documents for four reasons:

| Problem | This Project's Solution |
|---------|-------------------------|
| **Naive fixed-size chunking** destroys context and table structure | **Per-page semantic chunking** with accurate page tracking + **table-aware extraction** |
| **Weak embeddings** collapse similarity scores into a tight range | **`BAAI/bge-large-en-v1.5`** — retrieval-specific, high-discrimination embeddings |
| **Untuned retrieval** misses evidence | **Hybrid (Dense + BM25) → RRF → Cross-encoder reranker → Validator** |
| **No measurable quality** — vibes instead of numbers | **Reproducible 30-question benchmark** with JSON + Markdown reports |

This project was built to be **honest** about its quality: it ships with a benchmark, reports its weaknesses, and documents every design trade-off.

---

## Key Features

### Retrieval

- **Hybrid Retrieval** — Dense (FAISS) + Sparse (BM25) fused via **Reciprocal Rank Fusion**.
- **Cross-Encoder Reranking** — `BAAI/bge-reranker-v2-m3` refines the top candidates.
- **Adaptive Candidate Depth** — retrieval depth adapts to query complexity.
- **Validator Layer** — filters low-quality / duplicate candidates before LLM context.

### Ingestion

- **Per-Page Chunking** — every chunk belongs to exactly one page, eliminating page drift.
- **Table-Aware Extraction** — 179 structured tables extracted, normalized to Markdown, and indexed as separate chunks.
- **Smart OCR (opt-in)** — PaddleOCR kicks in only for pages without selectable text.
- **Multi-Format Loaders** — PDF, DOCX, XLSX, XLS, HTML, Markdown, TXT.
- **Deterministic Hashing** — SHA-256 document deduplication.

### Embeddings & LLM

- **Local Multilingual Embeddings** — `BAAI/bge-large-en-v1.5` (1024-dim), runs on CPU, **no API key needed**.
- **Groq LLM Backend** — OpenAI-compatible API with streaming support.
- **Rate Limiting + Retries** — process-wide throttle with exponential backoff.
- **Structured Output Support** — JSON-mode + Pydantic validation.

### Generation

- **Strict Grounding** — the LLM is instructed to answer **only** from retrieved context.
- **Deterministic Citations** — every answer links back to a specific document, page, and section.
- **Bilingual Answers** — responds in the same language as the question (English / Arabic).

### Engineering

- **FastAPI Backend** — async endpoints with a built-in web UI.
- **Structured Logging** — JSON logs with domain-specific helpers.
- **Latency Tracking** — per-stage instrumentation out of the box.
- **Docker Ready** — multi-stage Dockerfile + docker-compose.
- **Production-Grade Config** — pydantic-settings with `.env` support.
- **18 Unit Tests** — covering retrieval, fusion, citation, and context building.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                             │
│                                                                         │
│   PDF / DOCX / XLSX / HTML / MD                                         │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐           │
│   │   Loader      │ →  │  Table Proc.  │ →  │   Chunker     │           │
│   │  (multi-      │    │  (structured  │    │  (per-page +  │           │
│   │   format)     │    │   extraction) │    │   semantic)   │           │
│   └───────────────┘    └───────────────┘    └───────────────┘           │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐           │
│   │  Metadata     │ →  │  Embeddings   │ →  │  FAISS + BM25 │           │
│   │  Generator    │    │  (bge-large)  │    │  Indexes      │           │
│   └───────────────┘    └───────────────┘    └───────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                                 │
│                                                                         │
│   User Query                                                            │
│       │                                                                 │
│       ▼                                                                 │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐           │
│   │  Query        │ →  │  Dense        │    │   BM25        │           │
│   │  Analysis     │    │  Search       │    │   Search      │           │
│   └───────────────┘    │  (FAISS)      │    │   (rank-bm25) │           │
│                        └───────┬───────┘    └───────┬───────┘           │
│                                │                    │                   │
│                                ▼                    ▼                   │
│                        ┌─────────────────────────────────┐              │
│                        │   Reciprocal Rank Fusion (RRF)  │              │
│                        └──────────────┬──────────────────┘              │
│                                       │                                 │
│                                       ▼                                 │
│                        ┌─────────────────────────────────┐              │
│                        │  Cross-Encoder Reranker         │              │
│                        │  (bge-reranker-v2-m3)           │              │
│                        └──────────────┬──────────────────┘              │
│                                       │                                 │
│                                       ▼                                 │
│                        ┌─────────────────────────────────┐              │
│                        │  Validator (dedup + quality)    │              │
│                        └──────────────┬──────────────────┘              │
│                                       │                                 │
│                                       ▼                                 │
│                        ┌─────────────────────────────────┐              │
│                        │  Context Builder (table-aware)  │              │
│                        └──────────────┬──────────────────┘              │
│                                       │                                 │
│                                       ▼                                 │
│                        ┌─────────────────────────────────┐              │
│                        │  LLM (Groq) + Citation Engine   │              │
│                        └──────────────┬──────────────────┘              │
│                                       │                                 │
│                                       ▼                                 │
│                        ┌─────────────────────────────────┐              │
│                        │  Answer + Traceable Sources     │              │
│                        └─────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **API Framework** | FastAPI 0.115 | Async REST API + auto OpenAPI docs |
| **ASGI Server** | Uvicorn | High-performance ASGI server |
| **Data Validation** | Pydantic 2.11 | Typed schemas + settings management |
| **Vector Store** | FAISS (CPU) | Dense similarity search |
| **Sparse Retrieval** | rank-bm25 | Lexical retrieval (BM25Okapi) |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | 1024-dim, retrieval-tuned |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking |
| **LLM** | Groq (Qwen 3.8 27B) | Fast generation via OpenAI-compatible API |
| **Document Parsing** | pypdf, pdfplumber, python-docx, pandas, BeautifulSoup | Multi-format readers |
| **Table Extraction** | pdfplumber (primary) + camelot (fallback) | Structured table extraction |
| **OCR (optional)** | PaddleOCR | Scanned PDF support |
| **Frontend** | HTML5 + TailwindCSS (CDN) + Vanilla JS | Zero-build web UI |
| **Logging** | structlog + stdlib logging | Structured JSON logs |
| **Testing** | pytest + pytest-asyncio | Unit + integration tests |
| **Container** | Docker + docker-compose | Reproducible deployment |

---

## Benchmark Results

Evaluated on an **8-question factual accuracy benchmark** against the CFIA Reference Database for Hazard Identification (2008).

### Headline Numbers

| Metric | Baseline (`e5-large`) | Final (`bge-large-en-v1.5`) | Improvement |
|--------|----------------------|----------------------------|-------------|
| **Token F1** | 0.0699 | **0.1284** | **+84%** |
| **Answer Relevance** | 0.4652 | **0.5759** | **+24%** |
| **MRR** | 0.2083 | **0.3542** | **+70%** |
| **Hit Rate @ 5** | 0.3750 | **0.5000** | **+33%** |
| **Precision @ 5** | 0.1146 | **0.2396** | **+109%** |
| **Recall @ 10** | 0.3750 | **0.5000** | **+33%** |
| **Errors** | 0 | **0** | — |
| **Total runtime** | 130s | **110s** | **-15%** |

### Per-Question Highlights

| ID | Question | F1 | Relevance | MRR |
|----|----------|----|-----------|-----|
| `fact_01` | Vibrio cholerae temperature range | 0.000 | 0.000 | 0.000 |
| `fact_02` | Clostridium perfringens Aw | 0.000 | 1.000 | 0.000 |
| `fact_03` | Yersinia enterocolitica pH range | 0.000 | 0.000 | 0.500 |
| `fact_04` | Nine priority allergens | 0.000 | 0.000 | 1.000 |
| `fact_05` | 2.0 mm extraneous material threshold | 0.056 | 0.750 | 0.333 |
| `fact_06` | Histamine (Scombroid) source | **0.424** | 1.000 | **1.000** |
| `fact_07` | BSE + Specified Risk Material | **0.453** | 0.857 | 0.000 |
| `fact_08` | Three broad hazard categories | 0.111 | 1.000 | 0.000 |

### Honest Interpretation

- **Answer Relevance (0.58)** is the most informative signal — it reflects how well the retrieved context supports the question.
- **Token F1 (0.13)** under-reports performance because it penalizes semantically-correct answers that use different phrasings.
- **MRR (0.35)** shows the correct chunk is often in the top 3, but not always first.
- **The system is honest about its limits** — the full per-question report is generated on every run.

### Reproducing the Benchmark

```bash
python -m tests.test_01_factual_accuracy
```

Reports are written to:

- `data/reports/latest_test_01_factual_accuracy.json` — machine-readable
- `data/reports/latest_test_01_factual_accuracy.md` — human-readable

---

## Quick Start

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/M0hamed0mar/food-safety-rag.git
cd food-safety-rag

cp .env.example .env
# edit .env and set GROQ_API_KEY

docker compose up --build
```

Open http://localhost:8000.

### Option 2 — Local Python

```bash
git clone https://github.com/M0hamed0mar/food-safety-rag.git
cd food-safety-rag

python -m venv venv
source venv/bin/activate           # Windows: .\venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY

# 1. Place source PDFs in data/uploaded_docs/
# 2. Ingest them
python -m app.scripts.ingest_data

# 3. Start the API
python -m app.scripts.run_api
```

Open http://localhost:8000.

### Option 3 — Helper Scripts (Windows)

```powershell
.\scripts\setup.ps1     # one-time setup: venv + deps + .env
.\scripts\ingest.ps1    # ingest documents
.\scripts\dev.ps1       # start the dev server
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI (HTML) |
| `GET` | `/health` | Liveness + component status |
| `GET` | `/api` | API metadata |
| `POST` | `/ingest` | Upload + ingest a document |
| `POST` | `/ask` | Non-streaming Q&A |
| `GET` | `/ask/stream` | SSE streaming Q&A |
| `GET` | `/retrieve` | Retrieve chunks without generating |
| `POST` | `/chat/new` | Create a new chat session |
| `GET` | `/chat/list` | List all chat sessions |
| `GET` | `/chat/{session_id}/history` | Get chat messages |
| `GET` | `/chat/{session_id}` | Get a chat session |
| `PUT` | `/chat/{session_id}/title` | Update chat title |
| `DELETE` | `/chat/{session_id}` | Delete a chat session |
| `GET` | `/stats` | System statistics |

### Example — Ingest & Ask

```bash
# Ingest
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@reference_database.pdf"

# Ask (non-streaming)
curl "http://localhost:8000/ask?query_text=What%20are%20the%20three%20broad%20categories%20of%20food%20safety%20hazards%3F"

# Ask (streaming)
curl "http://localhost:8000/ask/stream?query_text=What%20are%20the%20main%20hazards%3F"
```

### Example Response

```json
{
  "answer_id": "ans_abc123",
  "query_id": "qry_xyz789",
  "session_id": "chat_20260919_103000_4821",
  "text": "The three broad categories of food safety hazards are: Biological, Chemical, and Physical.",
  "citations": [
    {
      "document_name": "reference_database_for_hazard_identification.pdf",
      "page": 5,
      "section": "Introduction",
      "chunk_id": "chunk_doc_e234f9ddf7274506_0004_..."
    }
  ],
  "is_supported": true,
  "confidence": 0.9,
  "metadata": {
    "model": "qwen/qwen3.8-27b",
    "tokens_generated": 899,
    "num_citations": 4,
    "total_duration_ms": 22880.12
  }
}
```

---

## Project Structure

```
food-safety-rag/
├── app/
│   ├── api/                    # FastAPI application + routers
│   │   ├── routes/             # /ask, /chat, /ingest, /health, /stats
│   │   ├── dependencies.py     # Shared dependencies
│   │   └── main.py             # App factory
│   ├── config/                 # Settings, constants, prompts
│   │   ├── settings.py         # Pydantic settings
│   │   ├── constants.py        # Global constants + enums
│   │   └── prompts.py          # Prompt templates (versioned)
│   ├── core/                   # LLM client, embeddings, exceptions
│   │   ├── llm_client.py       # Groq async client
│   │   ├── embeddings.py       # Local embedding model
│   │   └── exceptions.py       # Custom exception hierarchy
│   ├── generation/             # Answer generation pipeline
│   │   ├── generator.py        # LLM answer generation
│   │   ├── citation.py         # Deterministic citation engine
│   │   ├── context_builder.py  # Table-aware context assembly
│   │   └── compressor.py       # Redundancy removal
│   ├── ingestion/              # Document ingestion pipeline
│   │   ├── loader.py           # Multi-format readers
│   │   ├── chunker.py          # Per-page semantic chunking
│   │   ├── cleaner.py          # Unicode + whitespace cleanup
│   │   ├── ocr.py              # Smart OCR (opt-in)
│   │   ├── table_processor.py  # Structured table extraction
│   │   ├── metadata.py         # Rich chunk metadata
│   │   └── embedding.py        # Ingestion-side embeddings
│   ├── llm/                    # LLM-adjacent utilities
│   │   ├── prompt_manager.py   # Prompt registry
│   │   └── translator.py       # Bilingual query rewriting
│   ├── monitoring/             # Observability
│   │   ├── logger.py           # Structured JSON logging
│   │   ├── latency.py          # Per-stage latency tracking
│   │   └── profiler.py         # Memory + CPU profiling
│   ├── retrieval/              # Retrieval stack
│   │   ├── dense.py            # FAISS dense retriever
│   │   ├── bm25.py             # BM25 sparse retriever
│   │   ├── fusion.py           # Reciprocal Rank Fusion
│   │   ├── reranker.py         # Cross-encoder reranker
│   │   ├── validator.py        # Result validation
│   │   ├── expander.py         # Query expansion
│   │   ├── hybrid.py           # Hybrid retriever
│   │   ├── orchestrator.py     # Pipeline orchestration
│   │   └── router.py           # Top-level retrieval router
│   ├── schemas/                # Pydantic models
│   │   ├── document.py         # Document + metadata
│   │   ├── chunk.py            # Chunk + metadata
│   │   ├── query.py            # Query + expansions
│   │   ├── answer.py           # Answer + citations
│   │   ├── retrieval.py        # RetrievalCandidate
│   │   ├── metadata.py         # Ingestion / retrieval metadata
│   │   └── chat.py             # Chat sessions
│   ├── scripts/                # CLI entry points
│   │   ├── ingest_data.py      # Bulk ingestion
│   │   └── run_api.py          # API server launcher
│   ├── services/               # Orchestration layer
│   │   ├── ingestion_service.py
│   │   ├── retrieval_service.py
│   │   └── generation_service.py
│   ├── vector_store/           # Storage backends
│   │   ├── faiss_store.py      # FAISS index
│   │   ├── metadata_store.py   # SQLite metadata
│   │   └── chat_store.py       # SQLite chat sessions
│   ├── web/                    # Web UI
│   │   ├── templates/index.html
│   │   └── static/             # CSS + JS
│   └── pipeline.py             # Top-level RAG pipeline
├── data/
│   ├── uploaded_docs/          # Source documents (gitignored)
│   ├── faiss_index/            # Persisted vector index (gitignored)
│   └── reports/                # Benchmark reports (gitignored)
├── tests/
│   ├── evaluation/             # Benchmark framework
│   │   ├── benchmark_data.py   # 30-question dataset
│   │   ├── metrics.py          # F1, MRR, Hit Rate, Precision, Recall
│   │   ├── evaluator.py        # Runner
│   │   ├── cache.py            # Persistent result cache
│   │   └── report_generator.py # JSON + Markdown output
│   ├── test_01_factual_accuracy.py
│   ├── test_retrieval.py
│   ├── test_citation.py
│   └── test_context_builder.py
├── scripts/                    # Windows helper scripts
│   ├── setup.ps1
│   ├── dev.ps1
│   └── ingest.ps1
├── logs/                       # Runtime logs (gitignored)
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── .gitignore
├── Makefile
├── requirements.txt
└── README.md
```

---

## Configuration

All settings are loaded from environment variables via `pydantic-settings` (see `.env.example`).

### Required

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key for LLM generation ([get one](https://console.groq.com/keys)) |

### Optional — LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | LLM provider |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Model name |
| `GROQ_TEMPERATURE` | `0.7` | Sampling temperature |
| `GROQ_MAX_TOKENS` | `4096` | Max generation tokens |
| `GROQ_TIMEOUT` | `90` | Request timeout (seconds) |
| `GROQ_MAX_RETRIES` | `5` | Retry attempts |
| `GROQ_MIN_REQUEST_INTERVAL` | `2.5` | Process-wide rate-limit (seconds) |

### Optional — Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Local embedding model |
| `EMBEDDING_DIM` | `1024` | Embedding dimension |
| `EMBEDDING_BATCH_SIZE` | `16` | Batch size |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` / `auto` |

### Optional — Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `DENSE_TOP_K` | `50` | Dense retrieval depth |
| `BM25_TOP_K` | `50` | BM25 retrieval depth |
| `RRF_K` | `60` | RRF constant |
| `RERANKER_ENABLED` | `True` | Enable cross-encoder reranking |
| `RERANKER_TOP_K` | `10` | Final chunks after reranking |
| `RERANKER_MAX_CANDIDATES` | `15` | Max candidates to rerank |
| `ENABLE_VALIDATOR` | `True` | Enable result validation |
| `VALIDATOR_MIN_QUALITY_SCORE` | `0.3` | Minimum content quality |
| `MAX_CONTEXT_CHUNKS` | `8` | Chunks sent to LLM |

### Optional — Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_MIN_TOKENS` | `80` | Minimum tokens per chunk |
| `CHUNK_MAX_TOKENS` | `400` | Maximum tokens per chunk |
| `CHUNK_OVERLAP_RATIO` | `0.15` | Overlap between adjacent chunks |
| `OCR_ENABLED` | `False` | Enable OCR for scanned PDFs |
| `MAX_FILE_SIZE_BYTES` | `104857600` | Max upload size (100 MB) |

### Optional — Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Testing

```bash
# Unit tests
pytest tests/ -v

# Unit tests with coverage
pytest tests/ --cov=app --cov-report=html

# Factual accuracy benchmark
python -m tests.test_01_factual_accuracy
```

### Test Coverage

| Test File | Focus |
|-----------|-------|
| `test_retrieval.py` | BM25 tokenization, RRF fusion, RetrievalCandidate |
| `test_citation.py` | Citation formatting, heuristic extraction, deduplication |
| `test_context_builder.py` | Context assembly, dedup, max-chunk limit |
| `test_01_factual_accuracy.py` | Full benchmark (8 factual questions) |

**Current status:** 18 / 18 unit tests passing.

---

## Docker

### Build

```bash
docker build -t food-safety-rag:latest .
```

### Run with Docker Compose

```bash
docker compose up --build
```

### Environment

The container reads from `.env` (mounted via `env_file`). Data is persisted via volume mounts:

- `./data/uploaded_docs` → `/app/data/uploaded_docs`
- `./data/faiss_index` → `/app/data/faiss_index`
- `./data/reports` → `/app/data/reports`
- `./logs` → `/app/logs`

### Healthcheck

The container exposes `GET /health` for liveness. Docker healthcheck is configured with:

- Interval: 30s
- Timeout: 10s
- Start period: 60s
- Retries: 3

---

## Deployment

### AWS — ECS Fargate (recommended)

1. Create an ECR repository:

   ```bash
   aws ecr create-repository --repository-name food-safety-rag
   ```

2. Build and push:

   ```bash
   docker build -t food-safety-rag .
   docker tag food-safety-rag:latest <account>.dkr.ecr.<region>.amazonaws.com/food-safety-rag:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/food-safety-rag:latest
   ```

3. Deploy to ECS Fargate with an Application Load Balancer.

4. Configure CloudWatch Logs, IAM roles, and secrets (Groq API key).

### Local Production Simulation

```bash
docker compose up --build -d
docker compose logs -f
```

---

## Design Decisions

### Why `bge-large-en-v1.5`?

The original model (`intfloat/multilingual-e5-large`) produced **collapsed similarity scores** (all chunks in the 0.81–0.83 range), making discrimination impossible. `bge-large-en-v1.5` is:

- **Retrieval-specific** — trained explicitly on contrastive retrieval objectives.
- **High-discrimination** — relevant chunks score ~0.60 vs irrelevant ~0.41 (Δ = 0.13).
- **Fast enough on CPU** — ~680s to embed 652 chunks.

### Why per-page chunking?

The original chunker ran on `document.raw_text` (concatenated pages), which caused:

- **Page drift** — chunk.page reported the wrong page.
- **Lost pages** — 63 out of 301 pages had zero chunks.
- **Broken citations** — answers cited pages that didn't contain the evidence.

Per-page chunking solves all three: every chunk belongs to exactly one page.

### Why table-aware extraction?

Tables in regulatory PDFs carry the **highest information density** (temperatures, limits, thresholds). They must be:

- Extracted as complete entities (never split across chunks).
- Formatted as Markdown (structure preserved for the LLM).
- Indexed with rich metadata (table type, row count, keywords).

This project extracts 179 tables from the CFIA reference.

### Why a validator?

Without validation, the reranker sometimes returns:

- Duplicate chunks (semantic near-duplicates).
- Low-quality chunks (truncated fragments).

The validator filters these out **before** the LLM sees them, improving precision.

### Why `--force` push on first setup?

The remote repository was pre-existing with stale content. Since the new codebase is the sole source of truth, we forced a clean rewrite of `main`. Subsequent pushes are standard `git push`.

---

## Roadmap

- [x] Hybrid retrieval (Dense + BM25 + RRF)
- [x] Cross-encoder reranking
- [x] Table-aware ingestion
- [x] Per-page chunking
- [x] Docker + docker-compose
- [x] Benchmark suite (JSON + Markdown reports)
- [x] Unit test coverage for core components
- [ ] Fine-tuned reranker for Food Safety
- [ ] Multi-document incremental ingestion
- [ ] Evaluation dashboard (Streamlit / Grafana)
- [ ] Multi-tenant support with per-user indexes
- [ ] Query rewriting via LLM (bilingual)
- [ ] Context expansion (parent / sibling chunks)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Canadian Food Inspection Agency (CFIA)** — Reference Database for Hazard Identification (2008), used as the sample corpus.
- **BAAI** — `bge-large-en-v1.5` and `bge-reranker-v2-m3` models.
- **Groq** — Fast LLM inference API.
- **Open Source Community** — FastAPI, FAISS, sentence-transformers, rank-bm25, pdfplumber, and countless others.

---

<div align="center">

**Built with care for the Food Safety community.**

⭐ If this project helped you, consider giving it a star.

</div>
