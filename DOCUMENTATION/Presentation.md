(Introduction — 30s)
Good afternoon. 
Hello — I’m Rohan Gamidi. I’m starting off with the presenting on behalf of our team: Vinitha Chowdary, Tejaswi Muppala, Ashish Babu and myself.

Title: “Neuro‑Symbolic RAG for Enterprise Applications: Bridging Inductive Tree‑Based Reasoning with Deductive Graph‑Structured Knowledge Representation.”


(Situation — 40s)
Enterprise RAG gaps — concrete consequences:

- Confident but ungrounded answers: firm responses with no verifiable link to source data.
- Document‑only retrieval: misses cross‑document links and multi‑hop evidence.
- Single‑pass reasoning fails on complex queries: one‑shot pipelines can’t reliably chain steps; early errors compound.
- No reliable provenance or audit trail: missing document IDs, graph node/edge references, and execution logs.
- Result: higher risk of incorrect, unverifiable, or non‑compliant outputs in enterprise settings.

Example questions (use in slides to illustrate failure modes vs SAGE):

- “Why was Project Orion delayed, and who was responsible for the final approval?”
- “List all procurement approvals above ₹10 lakhs in Q3 2025 that violated escalation policy.”


(Literature snapshot — 40s)
A few relevant works we build on or differ from:

Neural‑symbolic dual‑indexing architectures for scalable RAG (2025) — improves retrieval scale but lacks deterministic reasoning guarantees.
LLM‑based approaches for legal KG completion (2024) — focuses on triple completion, not explainable QA.
Systems combining LLMs with knowledge graphs for QA (2024) — improve accuracy but depend on query generation quality.
Multi‑agent LLM frameworks for complex tasks (2024) — good for coordination but lack structured grounding.
Knowledge‑driven multi‑agent systems in domain settings (2025) — domain specific, not enterprise‑grade audit trails.
(How SAGE addresses the gaps — 90s)
SAGE integrates neural and symbolic methods so answers are both flexible and verifiable. Key points:

Neuro‑symbolic integration: GAP planner and policy_guard make planning and execution schema‑aware — not just post‑hoc checks.
Provenance & explainability: every response includes a provenance bundle with plan JSON, validated Cypher paths, and doc/chunk IDs.
Systematic evaluation: built‑in metrics and ablation suites measure grounding rate, path accuracy, and hallucination rate with reproducible comparisons.
Retrieval selector: chooses BM25, dense embeddings, or hybrid per query to maximize recall and precision.
Graph‑first reasoning: the knowledge graph is the core substrate; G‑CT forces chain‑of‑thought steps to align with actual graph paths.
Robust multi‑hop & self‑correction: ISQA decomposes hard queries and the Critic triggers retries until confidence is sufficient.
Staleness & self‑adjustment: SAIA detects contradictory updates, computes impact radius, and incrementally re‑embeds or invalidates stale plans.
(Architecture — focus: Retrieval) — 90s
I’ll cover the query‑time and retrieval pieces — my portion of the split.

Orchestrator & Planner (GAP): the orchestrator receives a user query and the GAP planner emits a schema‑aware plan in JSON that references valid node types and edges.
Retrieval Selector: for each retrieval step, retrieval_selector.py picks BM25 for exact IDs or short queries, embeddings for conceptual matches, or a hybrid strategy when high recall is needed. The selector logs its decision and falls back if coverage is low.
Retrieval methods:
Vector search returns semantically similar chunks.
BM25 returns precise, lexical matches (IDs, dates, phrases).
Graph traversal finds multi‑hop, relationship‑preserving evidence.
Reranker: candidate passages from all modalities are reranked (cross‑encoder or LLM) before being passed to the reasoner.
Symbolic Reasoner (G‑CT): verifies that reasoning steps correspond to real Neo4j paths; invalid steps are rejected and the planner reworks the plan.
Critic & policy_guard: final audit gate — enforces rules, adds provenance, and triggers retries if checks fail.
(Suggested split for architecture explanation)

You (retrieval): cover Orchestrator → Retrieval Selector → Vector/BM25/Graph → Reranker → hand off to Reasoner.
Teammate (ingestion & maintenance): cover ingestion pipeline, SAIA, graph population, and evaluation harness.
(Core functionalities & evaluation) — 50s
What the system delivers:

Traceable answers with provenance bundles and Cypher‑validated paths.
Hybrid retrieval + reranking for higher recall and precision.
Iterative self‑querying for reliable multi‑hop reasoning.
SAIA for automated staleness detection and incremental re‑embedding.
Evaluation suite measuring grounding rate, hallucination rate, graph path accuracy, and latency; results are reproducible and auditable.