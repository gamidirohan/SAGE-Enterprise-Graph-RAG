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
uv run streamlit run app/graph_rag.py
```

Default URL: `http://localhost:8501`

### 2) Streamlit Document Processor

```bash
uv run streamlit run app/pipeline.py
```

### 3) FastAPI Backend

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

## Important Message Processor Commands

These are the main flag-based commands used for ingestion and evaluation workflows.

### From repo root (recommended)

```bash
# Ingest documents/messages only (no QA generation)
uv run python app/message_processor.py --directory data/documents_ui --skip-qa

# Ingest uploaded message files only (no QA generation)
uv run python app/message_processor.py --directory data/uploads --skip-qa

# Generate QA pairs only (skip ingestion)
uv run python app/message_processor.py --skip-processing --num-pairs 30 --output data/eval/qa_pairs.json

# Full run: ingest + generate QA pairs
uv run python app/message_processor.py --directory data/documents_ui --num-pairs 30 --output data/eval/qa_pairs.json
```

### From inside `app/` folder

```bash
# Equivalent to the root command above when current directory is app/
uv run python message_processor.py --directory ../data/documents_ui --skip-qa
```

### Useful flags

- `--directory <path>`: input folder of `.txt` files to process.
- `--skip-qa`: ingest data only, no QA generation.
- `--skip-processing`: generate QA only from existing graph data.
- `--num-pairs <n>`: number of QA pairs to generate.
- `--output <path>`: output JSON path for generated QA pairs.

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
