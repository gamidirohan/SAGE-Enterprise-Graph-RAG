# SAGE — Current Implementation (Existing Work)

**One-line summary:** SAGE is a Neo4j-backed Graph-RAG system that ingests documents/messages into a knowledge graph, performs vector + graph retrieval, and uses Groq LLMs to generate answers — with evaluation harness and reporting. ✅

---

## 🔧 Architecture & Components

- **Ingestion & Processing**
  - `scripts/pipeline.py` (document processor): PDF/text extraction, LLM-based entity extraction, chunking, embedding, Neo4j node/relationship creation.
  - `scripts/message_processor.py`: processes message uploads, extracts structured fields, and generates QA pairs.
- **Knowledge Graph (Neo4j)**
  - Node types: `Document`, `Chunk`, `Person` (senders/receivers), etc.
  - Relationships: `PART_OF`, `SENT`, `RECEIVED_BY`.
- **Embeddings & Retrieval**
  - SentenceTransformers embeddings (default: `all-mpnet-base-v2`).
  - Vector search implemented in Neo4j (gds.similarity.cosine) to rank `Chunk` embeddings and return top-K chunks.
- **LLM Layer**
  - Groq via `langchain_groq` (e.g., `deepseek-r1-distill-llama-70b`, `gemma2-9b-it`, `llama3-8b-8192` available in configs).
  - LLMs used for summarization, structured extraction (JSON), and final generation using prompt templates.
- **API & UI**
  - `backend.py`: FastAPI with endpoints `/api/chat`, `/api/process-document`, `/api/debug-graph`, `/api/health`.
  - `graph_rag.py` and Streamlit apps for Chat UI and Document Processor UI.
- **Evaluation & Reporting**
  - `scripts/performance_comparison.py`: multi-model evaluation harness (quality, latency, preference, similarity).
  - `scripts/generate_report.py`: HTML report generator, plots, and CSV outputs.
  - Batch scripts: `scripts/run_quick_comparison.bat`, `scripts/run_comprehensive_comparison.bat`.

---

## 🧭 Data Flow (Simplified)

1. Upload document or messages → text extraction (PyPDF2 / TXT) → LLM extracts Sender/Receivers/Content.
2. Chunking (NLTK sentence tokenize + overlap) → per-chunk summaries (LLM) → per-chunk embedding storage.
3. Store nodes and relationships in Neo4j (Document, Chunk, Person, PART_OF, SENT, RECEIVED_BY).
4. At query time: compute query embedding → run Neo4j vector similarity query → select top chunks → format context → call Groq LLM to generate final answer.
5. Evaluation pipeline compares SAGE vs. Traditional RAG and writes artifacts to `results/`.

---

## 📁 Notable Files (Quick Map)

- `backend.py` — FastAPI server, ingestion, query, Groq response pipeline.
- `graph_rag.py` — Streamlit chat interface + query pipeline.
- `scripts/pipeline.py` — Document ingestion, entity extraction, chunking (also used in the UI variant in `data/documents_ui/`).
- `scripts/message_processor.py` — Message ingestion and QA pair generation.
- `scripts/performance_comparison.py` — Evaluation harness and metrics.
- `scripts/generate_report.py` — Builds HTML report and visualizations from results JSON/CSV.
- `data/eval/qa_pairs.json`, `data/eval/test_queries.json` — Test queries and QA pairs used by the comparison pipeline.
- `results/` — evaluation outputs (JSON, CSV, HTML reports).

---

## 📊 Summary of Existing Results

- Example from `results/initial_comparison_results.json`:
  - SAGE Graph RAG shows **higher quality** (example: 7.60/10) vs Traditional RAG (6.30/10) on sample runs.
  - Latency is comparable (SAGE slightly higher in some runs).
  - The results include structured contexts (chunk summaries, doc IDs, relationships) and occasional LLM error traces (e.g., organization-restricted API errors).
- Artifacts to inspect: `initial_performance_report.html`, `performance_data_overall.csv`, and `initial_comparison_results.json` (detailed per-query traces).

---
