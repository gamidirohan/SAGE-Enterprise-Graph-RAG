# SAGE Enterprise Graph RAG

SAGE is an enterprise-focused Graph RAG project that:
- extracts structured information from documents and messages,
- stores relationships in Neo4j,
- answers questions using graph-aware retrieval and LLM reasoning,
- powers a companion Next.js chat UI with authenticated direct chats, group chats, and SAGE conversations.

## Runtime Overview

- `SAGE-Enterprise-Graph-RAG` is the canonical backend and graph-ingestion layer.
- `../ChatAppSAGE` is the companion Next.js frontend.
- Neo4j is the runtime source of truth for users, groups, conversations, messages, read state, and graph-ingested documents.
- `users.json`, `messages.json`, and `groups.json` in the frontend are seed inputs only. They bootstrap Neo4j on first use and are not the live runtime store.
- Backend-dependent features now hard-fail instead of returning mock chat or mock graph-debug data.

## Current Repository Layout

- `app/`: main application code (core and active code paths)
- `app/backend.py`: FastAPI app used by the Next.js frontend
- `app/chat_store.py`: canonical Neo4j-backed auth/chat/group/message storage
- `data/`: datasets and input files
- `DOCUMENTATION/`: design notes, examples, and writeups
- `results/`: generated outputs/experiments (can be cleaned if distracting)
- `scripts/`: utility scripts used for evaluation/reporting and one-off workflows
- `tests/`: backend and ingestion test coverage
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

If you need to verify existing bcrypt password hashes, install the optional legacy auth extra instead of the default dependency set:

```bash
uv sync --extra legacy-auth
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

Useful optional backend settings:

```env
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
SAIA_ENABLED=false
```

Important note:
- App modules load `.env` from the repo root.

## Running Interfaces

### 1) Streamlit Chat Interface

```bash
uv run streamlit run app/pipeline.py
```

Default URL: `http://localhost:8501`

### 2) FastAPI Backend

```bash
uv run python app/backend.py
```

Default URL: `http://localhost:8000`

If `uv` complains about `unrecognized subcommand 'app/backend.py'`, the missing piece is `run`. The default install uses the built-in PBKDF2 password fallback; use `uv sync --extra legacy-auth` only if you need legacy bcrypt-hash verification.

Useful endpoints:
- `GET /api/health`
- `POST /api/bootstrap`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/session`
- `PUT /api/profile`
- `GET /api/groups`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}/messages`
- `POST /api/conversations/{conversation_id}/messages`
- `POST /api/messages/{message_id}/read`
- `POST /api/chat`
- `POST /api/process-document`
- `GET /api/debug-graph`
- `POST /api/sync-user`
- `POST /api/sync-messages`

### 3) Companion Next.js Chat App

The frontend lives in the sibling repo/folder:

`../ChatAppSAGE`

Install and run it with:

```bash
cd ../ChatAppSAGE
npm install
npm run dev
```

For realtime notifications, run the websocket server in a second terminal:

```bash
cd ../ChatAppSAGE
npm run websocket
```

Or run both frontend services together:

```bash
cd ../ChatAppSAGE
npm run dev:all
```

Useful frontend environment variables:

```env
FASTAPI_URL=http://localhost:8000
SESSION_SECRET=replace-with-a-long-random-secret
WS_INTERNAL_SECRET=replace-with-a-long-random-secret
WS_BROADCAST_URL=http://localhost:8080/broadcast
NEXT_PUBLIC_WS_URL=ws://localhost:8080/ws
```

Notes:
- `SESSION_SECRET` is required for signed auth/session and websocket tokens.
- `WS_INTERNAL_SECRET` should match between the Next.js app and `websocket-server.js`.
- The frontend bootstraps Neo4j from its seed JSON files on first authenticated access.

## Data Notes

- UI document ingestion primarily uses `data/documents_ui/` if present, otherwise `data/documents/`.
- Uploaded files are stored in `data/uploads/`.
- Evaluation inputs live under `data/eval/`.
- Message and attachment ingestion supports `PDF`, `TXT`, and `DOCX`.
- Batch document ingestion skips already-ingested files by default using the document content hash as `doc_id`.
- Chat messages persisted through the canonical flow store sender, receiver or group, timestamp, source, attachment metadata, trace, and graph sync status.

## Single Streamlit App

`app/pipeline.py` now contains the Streamlit chat, document processing, and message-ingestion flows in one place.

## Scripts

Common script entrypoints:

```bash
uv run python scripts/batch_ingest_documents.py
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

Useful verification commands:

```bash
uv run pytest
uv run python -m py_compile app/backend.py app/chat_store.py app/services.py app/utils.py app/pipeline.py
```

Frontend verification:

```bash
cd ../ChatAppSAGE
npm run lint
npx tsc --noEmit
npm run build
node --check websocket-server.js
```

Note:
- `npm run lint` uses ESLint when the frontend has `eslint` and `eslint-config-next` installed.
- In restricted/offline environments, the frontend lint script falls back to Next.js build/type validation so there is still a non-interactive verification path.

## Status

- Core Graph RAG flow: available
- Streamlit interfaces: available
- FastAPI backend: available
- Companion Next.js chat app: available
- Neo4j-backed auth/chat/groups/provenance flow: available
- Under-development features: in `under_development/` (not production-ready)
