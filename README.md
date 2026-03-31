# SAGE Enterprise Graph RAG

SAGE is an enterprise-focused Graph RAG project that:
- extracts structured information from documents and messages,
- stores relationships in Neo4j,
- answers questions using graph-aware retrieval and LLM reasoning.

## Current Repository Layout

- `app/`: main application code (core and active code paths)
- `data/`: datasets and input files
- `DOCUMENTATION/`: design notes, examples, and writeups
- `results/`: generated outputs/experiments (can be cleaned if distracting)
- `scripts/`: utility scripts used for evaluation/reporting and one-off workflows
- `tests/`: partial test coverage (not complete yet)
- `under_development/`: in-progress features that are not considered stable

## Tech Stack

- Python 3.9+
- Neo4j
- Streamlit
- FastAPI
- LangChain + Groq
- SentenceTransformers
- uv (environment and dependency management)

## Setup

This project uses `uv` and a local `.venv`.

```bash
uv venv .venv

# Windows activate .venv
.venv/Scripts/activate

# Linux/Mac activate .venv
source .venv/bin/activate

uv sync
```

## Environment Variables

Create a `.env` file with at least:

```env
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
NEO4J_DATABASE=...
GROQ_API_KEY=...
```

Important note:
- App modules load `.env` from the repo root.

## Running Interfaces

### 1) Streamlit Chat Interface (verified)

```bash
uv run streamlit run app/pipeline.py
```

Default URL: `http://localhost:8501`

### 2) FastAPI Backend

```bash
uv run python app/backend.py
```

Default URL: `http://localhost:8000`

Useful endpoints:
- `GET /api/health`
- `POST /api/chat`
- `POST /api/process-document`
- `GET /api/debug-graph`

## Data Notes

- UI document ingestion primarily uses `data/documents_ui/` if present, otherwise `data/documents/`.
- Uploaded files are stored in `data/uploads/`.
- Evaluation inputs live under `data/eval/`.

## Single Streamlit App

`app/pipeline.py` now contains the Streamlit chat, document processing, and message-ingestion flows in one place.

## Scripts

Common script entrypoints:

```bash
uv run python scripts/performance_comparison.py
uv run python scripts/run_performance_comparison.py
uv run python scripts/generate_report.py
```

Batch helpers (Windows):

```powershell
scripts\run_quick_comparison.bat
scripts\run_comprehensive_comparison.bat
```

## Tests

Run tests with:

```bash
uv run pytest
```

Current state: tests are present but coverage is incomplete.

## Status

- Core Graph RAG flow: available
- Streamlit interfaces: available
- FastAPI backend: available
- Under-development features: in `under_development/` (not production-ready)
