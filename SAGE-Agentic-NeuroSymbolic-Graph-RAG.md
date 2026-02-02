# How SAGE Graph RAG evolves into an intelligent, agent-driven, reasoning system

This document proposes a concrete methodology to upgrade SAGE from classic Graph RAG into an agentic, neuro-symbolic reasoning system suitable for enterprise use and Q1 journal publication. It outlines the architectural shift, agent orchestration, novel algorithms, safety/governance, metrics, and an evaluation harness—while preserving backward compatibility with the current repository.

---

## Big Picture (in very simple terms)

Classic RAG:

> Ask → Fetch documents → Stuff them into prompt → Answer

Agentic + Neuro-Symbolic Graph RAG:

> Ask → Plan → Think → Retrieve → Check rules → Answer → Verify

The system loops, thinks, checks itself, and proves its answers using:
- Neural intelligence (LLMs, embeddings)
- Symbolic intelligence (graphs, rules, constraints)

---

## Current System (Quick Summary)

- Retrieval and chat:
  - Streamlit UI and graph cosine similarity in [graph_rag.py](graph_rag.py)
  - LLM generation via Groq; uses chunk summaries and cosine similarity over stored embeddings
- Ingestion and graph population:
  - Document/message processing and chunking into Neo4j in [pipeline.py](pipeline.py)
- Backend API:
  - FastAPI endpoints for chat, document ingestion, and graph debugging in [backend.py](backend.py)

This is a strong foundation: embeddings + Neo4j + LLM. The upgrade introduces planning, tool use, constraints, and verification.

---

## 1️⃣ Architectural Shift: From Pipeline to Thinking Loop

Old (single pass):

```
User Query
   ↓
Retrieve Docs
   ↓
LLM Generates Answer
```

New (agentic loop):

```
User Query
   ↓
Planner Agent
   ↓
Execute Steps (retrieve / reason / query graph)
   ↓
Verify & Critique
   ↓
Final Answer (with proof)
```

What changes:
- No more single pass
- System plans, acts, checks, retries
- Specialized agents handle distinct responsibilities

---

## 2️⃣ Agentic Orchestration Layer (The Brain Controller)

Introduce a lightweight orchestrator (LangGraph or AutoGen). It controls who does what and in what order.

Agents (roles):
- Planner (GAP): break question into constrained steps
- Retriever: hybrid retrieval (dense + BM25 + graph)
- Symbolic Reasoner: apply organizational rules, time constraints, graph path validity
- Generator: synthesize answer + provenance
- Critic/Safety: verify grounding, citations, policies; trigger retries if weak

Implementation note: Keep existing endpoints intact; add an `agentic_mode` toggle wired through [backend.py](backend.py) and [graph_rag.py](graph_rag.py).

---

## 3️⃣ Novel Algorithms (Key Research Contributions)

### 🔹 GAP — Graph-Augmented Planner (Neuro-Symbolic)
Problem: LLM planners ignore graph schema and over-plan free-form.
Solution: Planner conditioned on Neo4j schema + constraints. Plans must reference valid node types and relations.

Plan schema (JSON):
```json
{
  "steps": [
    {"type": "graph_lookup", "node": "Employee", "filter": {"name": "Alice"}},
    {"type": "traverse", "edge": "REPORTS_TO", "depth": 1},
    {"type": "rule_check", "policy": "APPROVAL_POLICY_2024"}
  ],
  "constraints": {"allowed_edges": ["REPORTS_TO", "APPROVES"], "max_depth": 3}
}
```

Pseudo-code (LangGraph):
```
planner(query) -> plan_json
executor(plan_json):
  for step in steps:
    if step.type == graph_lookup: cypher(...)
    if step.type == traverse: cypher(PATH CONSTRAINTS)
    if step.type == rule_check: rules_engine(...)
collect artifacts -> context
```

### 🔹 ISQA — Iterative Self-Querying Agent
Problem: Complex multi-hop questions fail in one shot.
Solution: Split into sub-questions with a bounded loop; each sub-answer updates context. Stops when confidence ≥ τ or budget exhausted.

Loop outline:
```
subqs = decompose(query)
for q in subqs:
  ctx += retrieve(q)
  if not valid(ctx): refine(q)
if confidence(ctx) < τ: fallback/retry
```

### 🔹 G-CT — Graph-Constrained Chain-of-Thought
Problem: Hallucinated reasoning steps.
Solution: Force CoT to align with actual graph paths. Invalid path → step rejected.

Constraint pattern:
- If Cypher path does not exist, the step cannot be used
- CoT template requires citing node IDs + relationship types

Example Cypher for path validation:
```
MATCH (e:Employee {name:$name})-[:REPORTS_TO]->(m:Manager)
WITH e,m
MATCH (m)-[:APPROVES]->(p:Policy {id:"2024"})
RETURN e,m,p
```
If no rows, reject the step and replan.

---

## 4️⃣ Neuro-Symbolic Reasoning (Together)

Neural side (soft): embeddings, semantic similarity, language understanding.
Symbolic side (hard): org rules, effective dates, graph structure.

Workflow:
1) Neural retrieval finds candidates
2) Symbolic constraints prune invalid ones
3) Generator produces answer + proof traces (IDs, paths, rules)

---

## 5️⃣ Hybrid Retrieval (Stronger Search)

Combine three sources:
- Dense vectors (SentenceTransformers)
- BM25 keyword (Elastic/Whoosh/Pyserini; simple baseline ok)
- Graph traversal (Neo4j paths; typed relationships)

Then:
- Rerank (Cross-Encoder or LLM-based reranker under token budget)
- Deduplicate
- Structured context packing (fixed template + token accounting)

Benefits: Higher recall/precision, lower hallucination.

---

## 6️⃣ Safety, Governance, and Audit (Enterprise-ready)

Built-in protections:
- PII filters (HR/Legal)
- Policy allow/deny rules
- Tool access control and rate limits

Provenance (auditable outputs):
- Document IDs (from Neo4j `Document.doc_id`)
- Node IDs/labels, edge types
- Execution plan (JSON), tool calls, validated paths

---

## 7️⃣ Modular Tools (Composable)

Add simple tool wrappers. Agents decide when to call them.
- `graph_query.py`: Cypher/Gremlin queries + path validators
- `vector_search.py`: dense + BM25 blended retrieval
- `rerank.py`: cross-encoder/LLM reranking
- `policy_guard.py`: rule engine + policy checks
- `sql_query.py`: optional structured data integration
- `code_exec.py`: sandboxed execution for calculative tasks

---

## 8️⃣ Pipeline Integration (Low-Risk Upgrade)

Keep classic Graph RAG intact; add a toggle:
```
agentic_mode = true | false
```
- `false` → existing flow in [graph_rag.py](graph_rag.py) / [backend.py](backend.py)
- `true` → route to orchestrator loop (LangGraph/AutoGen), with fallback to classic if weak/failed

---

## 9️⃣ Self-Critique and Verification Loop

Before returning:
- Critic checks grounding, citations, rule satisfaction
- If weak, re-retrieve or re-generate under constraints
- Final emits provenance bundle

---

## 🔟 End-to-End Data Flow

1) Query enters system
2) Planner emits JSON execution plan (schema-aware)
3) Retrieval returns documents + graph paths
4) Symbolic reasoner applies rules + prunes invalid paths
5) Generator writes answer + cites artifacts
6) Critic verifies and may loop once
7) Final answer + provenance returned

---

## 1️⃣1️⃣ Evaluation & Regression Testing (Publishable)

Extend the existing framework in [performance_comparison.py](performance_comparison.py) with the following metrics:
- Grounding rate: % answers with valid supporting artifacts
- Hallucination rate: % steps rejected by G-CT validator
- Graph path accuracy: % claimed paths that exist in Neo4j
- Tool-call faithfulness: % tool outputs used consistently in CoT
- Rule satisfaction: % answers passing `policy_guard`
- Latency per stage: planner, retrieval, reasoning, generation, critique
- Rerank uplift: Δ quality with reranker vs. none

Harness design:
- Use [qa_pairs.json](qa_pairs.json) + [test_queries.json](test_queries.json)
- Create ablation suites: {dense-only, bm25-only, graph-only, hybrid}, {no-constraints vs. G-CT}, {no-critique vs. critic}
- Persist artifacts to `results/` (JSON + HTML report via [generate_report.py](generate_report.py))

Reporting:
- Confidence intervals via bootstrapping
- Path visualizations (Cypher → Mermaid diagrams) with error bars

---

## 1️⃣2️⃣ Backend & UI Integration

Backend:
- New endpoint: `/api/agent_chat` that streams plan → tool calls → constraints → answer → provenance
- Toggle in request body: `{ agentic_mode: true }`

Streamlit:
- Add “Agent Mode” toggle in [graph_rag.py](graph_rag.py)
- Live panes for: planner steps, tool calls, validated paths, critic verdicts

---

## 1️⃣3️⃣ Academic / Journal Value (Contributions)

Claimable contributions:
- Hallucination reduction via graph-constrained CoT (G-CT)
- Schema-aware Graph-Augmented Planning (GAP)
- Iterative Self-Querying Agent (ISQA) for multi-hop QA
- Neuro-symbolic integration with enterprise graph rules
- Auditable provenance under enterprise constraints

Formalization:
- Define GAP/ISQA/G-CT as repeatable, evaluatable patterns
- Provide pseudo-code and plan schemas; report metrics and ablations

---

## Suggested Implementation Roadmap (Non-breaking)

Short term (1–2 weeks):
- Add orchestrator module and `agentic_mode` toggle
- Implement GAP planner (schema-conditioned prompts)
- Implement graph path validator + minimal `policy_guard`
- Hybrid retrieval baseline (BM25 + dense + graph)

Medium term (2–4 weeks):
- Add reranker; integrate provenance packing
- Implement critic loop; wire fallbacks
- Extend evaluation metrics + ablations; generate HTML report

Long term (4–8 weeks):
- UI telemetry of plans/paths
- SQL/data warehouse tool integration
- Advanced rule engine and safety filters

---

## Appendix: Minimal Orchestration Sketch (LangGraph)

```
from langgraph import Graph

G = Graph()

@G.node
def planner_node(state):
  return plan_json

@G.node
def retrieve_node(state):
  return artifacts

@G.node
def reason_node(state):
  return pruned_artifacts

@G.node
def generate_node(state):
  return answer

@G.node
def critic_node(state):
  return verdict

G.edge(planner_node, retrieve_node)
G.edge(retrieve_node, reason_node)
G.edge(reason_node, generate_node)
G.edge(generate_node, critic_node)
G.edge(critic_node, generate_node, condition="retry")

result = G.run(query)
```

---

## Final One-Line Summary

This proposal upgrades SAGE Graph RAG from a simple document retriever into an agent-driven, neuro-symbolic reasoning system that plans, reasons over graphs, applies rules, verifies itself, and produces auditable, trustworthy answers—ready for enterprise and Q1-level publication.
