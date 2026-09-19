# Food Safety RAG System

> A production-grade Retrieval-Augmented Generation (RAG) system for the Food Safety domain. Answers complex regulatory and hazard-identification questions from trusted reference documents with full source citations.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECS-orange.svg)](https://aws.amazon.com/ecs/)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Benchmark Results](#benchmark-results)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

This system answers questions about food safety regulations, HACCP plans, biological / chemical / physical hazards, allergen management, and CFIA / Health Canada standards — using **only** the content of trusted source documents, never hallucinating.

It is designed as a **domain-agnostic RAG engine** that can be re-pointed to any regulated domain (pharma, finance, legal) by swapping the source PDFs and re-running ingestion.

### Why this project?

Most RAG demos fail on real-world technical documents because of:

- **Poor chunking** - naive fixed-size splits destroy table structure and context.
- **Weak embeddings** - generic multilingual models collapse similar scores together.
- **Untuned retrieval** - missing fusion, reranking, and validation layers.
- **No measurable quality** - "it feels right" instead of hard numbers.

This project addresses all four, with a reproducible benchmark and an honest report card.

---

## Key Features

- **Hybrid Retrieval** - Dense (FAISS) + Sparse (BM25) fused via Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking** - `BAAI/bge-reranker-v2-m3` refines the candidate list.
- **Per-Page Chunking** - every chunk belongs to exactly one page, eliminating page drift.
- **Table-Aware Ingestion** - 179 structured tables extracted and indexed separately.
- **Local Multilingual Embeddings** - `BAAI/bge-large-en-v1.5` runs on CPU, no API key needed.
- **LLM Generation via Groq** - OpenAI-compatible API with streaming support.
- **Deterministic Citations** - every answer links back to a specific document + page.
- **FastAPI Backend** - async endpoints with a built-in web UI.
- **Production-Ready** - Docker, docker-compose, and AWS ECS deployment included.
- **Reproducible Benchmarks** - 30-question evaluation suite with JSON + Markdown reports.

---

## Architecture
+------------------+ +-------------------+ +------------------+
| PDF Documents | --> | Ingestion Layer | --> | Vector Store |
| (HACCP, CFIA, | | - Loader | | - FAISS |
| regulations) | | - OCR (opt-in) | | - SQLite meta |
+------------------+ | - Cleaner | +------------------+
| - Chunker | |
| - Table extract | |
| - Embeddings | |
+-------------------+ |
| |
v v
+------------------+ +-------------------+ +------------------+
| User Query | --> | Retrieval Layer | <-- | BM25 Index |
+------------------+ | - Dense search | +------------------+
| - BM25 search |
| - RRF fusion |
| - Reranker |
| - Validator |
+-------------------+
|
v
+-------------------+
| Context Builder |
+-------------------+
|
v
+-------------------+
| LLM (Groq) |
| + Citation Engine|
+-------------------+
|
v
+-------------------+
| Answer + Sources |
+-------------------+

text

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI 0.115 | Async REST API |
| **Vector Store** | FAISS (CPU) | Dense similarity search |
| **Sparse Retrieval** | rank-bm25 | Lexical matching |
| **Embeddings** | BAAI/bge-large-en-v1.5 | 1024-dim, local |
| **Reranker** | BAAI/bge-reranker-v2-m3 | Cross-encoder |
| **LLM** | Groq (Qwen 3.8 27B) | Answer generation |
| **Document Parsing** | pypdf, pdfplumber, python-docx, pandas | Multi-format |
| **Table Extraction** | pdfplumber + custom | Structured tables |
| **OCR (optional)** | PaddleOCR | Scanned PDFs |
| **Frontend** | HTML + TailwindCSS + Vanilla JS | Web UI |
| **Monitoring** | structlog + custom profiler | Observability |

---

## Benchmark Results

Evaluated on a 30-question benchmark against the CFIA Reference Database for Hazard Identification (2008).

### Factual Accuracy (8 questions)

| Metric | Baseline (e5-large) | Final (bge-large-en-v1.5) | Improvement |
|--------|---------------------|---------------------------|-------------|
| **Token F1** | 0.0699 | **0.1284** | **+84%** |
| **Answer Relevance** | 0.4652 | **0.5759** | **+24%** |
| **MRR** | 0.2083 | **0.3542** | **+70%** |
| **Hit Rate @ 5** | 0.3750 | **0.5000** | **+33%** |
| **Precision @ 5** | 0.1146 | **0.2396** | **+109%** |
| **Recall @ 10** | 0.3750 | **0.5000** | **+33%** |

**Total runtime:** 110s for 8 questions on CPU.

> **Note:** These scores reflect a deliberately strict evaluation on **open-ended Q&A**. Substring / exact-match metrics under-report performance on domain-specific phrasings, so Answer Relevance (0.58) is the most informative signal.

---

## Quick Start

### Option 1 - Docker (recommended)

```bash
git clone https://github.com/your-username/food-safety-rag.git
cd food-safety-rag

cp .env.example .env
# edit .env and set GROQ_API_KEY

docker compose up --build
Then open http://localhost:8000.

Option 2 - Local Python
bash
git clone https://github.com/your-username/food-safety-rag.git
cd food-safety-rag

python -m venv venv
source venv/bin/activate    # Windows: .\venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY

# Ingest sample documents
python -m app.scripts.ingest_data

# Start the API
python -m app.scripts.run_api
Then open http://localhost:8000.

API Reference
MethodEndpointDescription
GET/Web UI
GET/healthLiveness + component status
GET/apiAPI metadata
POST/ingestUpload + ingest a document
POST/askNon-streaming Q&A
GET/ask/streamSSE streaming Q&A
GET/retrieveRetrieve chunks without generating
POST/chat/newCreate a new chat session
GET/chat/listList all chat sessions
GET/chat/{session_id}/historyGet chat messages
DELETE/chat/{session_id}Delete a chat session
GET/statsSystem statistics
Example
bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@reference_database.pdf"

curl "http://localhost:8000/ask?query_text=What%20are%20the%20three%20broad%20categories%20of%20food%20safety%20hazards%3F"
Project Structure
text
food-safety-rag/
├── app/
│   ├── api/               # FastAPI endpoints + routers
│   ├── config/            # Settings, constants, prompts
│   ├── core/              # LLM client, embeddings, exceptions
│   ├── generation/        # Context builder, citation engine, generator
│   ├── ingestion/         # Loader, cleaner, chunker, table processor
│   ├── llm/               # Prompt manager, query translator
│   ├── monitoring/        # Structured logger, latency tracker, profiler
│   ├── retrieval/         # Dense, BM25, fusion, reranker, validator
│   ├── schemas/           # Pydantic models (Document, Chunk, Query, Answer)
│   ├── scripts/           # CLI entry points (ingest, run_api)
│   ├── services/          # High-level orchestration layer
│   ├── vector_store/      # FAISS + metadata + chat stores
│   ├── web/               # HTML templates + static assets
│   └── pipeline.py        # Main RAG pipeline orchestrator
├── data/
│   ├── uploaded_docs/     # Source documents (gitignored)
│   ├── faiss_index/       # Persisted vector index (gitignored)
│   └── reports/           # Benchmark reports
├── tests/
│   ├── evaluation/        # Metrics, benchmark data, report generator
│   └── test_01_factual_accuracy.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .env.example
└── README.md
Configuration
All settings are loaded from environment variables (see .env.example).

Required
VariableDescription
GROQ_API_KEYGroq API key for LLM generation
Optional (with sensible defaults)
VariableDefaultDescription
GROQ_MODELqwen/qwen3.8-27bLLM model name
EMBEDDING_MODELBAAI/bge-large-en-v1.5Local embedding model
EMBEDDING_DEVICEcpucpu / cuda / mps
DENSE_TOP_K50Dense retrieval depth
BM25_TOP_K50BM25 retrieval depth
RERANKER_ENABLEDTrueEnable cross-encoder reranking
RERANKER_TOP_K10Final chunks after reranking
MAX_CONTEXT_CHUNKS8Chunks sent to LLM
OCR_ENABLEDFalseEnable OCR for scanned PDFs
Testing
bash
# Run the factual accuracy benchmark
python -m tests.test_01_factual_accuracy

# Reports are written to data/reports/
Each run produces:

data/reports/latest_test_01_factual_accuracy.json - machine-readable

data/reports/latest_test_01_factual_accuracy.md - human-readable

Deployment
AWS (ECS Fargate)
Build the image and push to ECR:

bash
aws ecr create-repository --repository-name food-safety-rag
docker build -t food-safety-rag .
docker tag food-safety-rag:latest <account>.dkr.ecr.<region>.amazonaws.com/food-safety-rag:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/food-safety-rag:latest
Deploy with the provided task definition and ALB configuration.

Detailed instructions: see docs/deployment.md.

Roadmap
☑ Hybrid retrieval (Dense + BM25 + RRF)
☑ Cross-encoder reranking
☑ Table-aware ingestion
☑ Per-page chunking
☑ Docker + docker-compose
☑ Benchmark suite
□ Fine-tuned reranker for Food Safety
□ Multi-document incremental ingestion
□ Evaluation dashboard
□ Multi-tenant support
License
MIT - see LICENSE for details.

Acknowledgments
CFIA - Reference Database for Hazard Identification (2008)

BAAI - BGE embedding and reranker models

Groq - Fast LLM inference

Open Source Community - FastAPI, FAISS, sentence-transformers, and more
