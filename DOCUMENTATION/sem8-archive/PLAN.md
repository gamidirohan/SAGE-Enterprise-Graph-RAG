# SAGE Phase 2 Plan: Examiner-Driven Research Hardening

## Summary

The core SAGE agentic backend is mostly functional: it has agent orchestration, hybrid retrieval selection, graph-aware retrieval, lightweight reranking, graph binding validation, answer payloads, and SAIA-based fact adjustment. The next phase should focus on the gaps examiners explicitly called out: explainability, policy enforcement, and quantitative evaluation.

The goal is to move SAGE from a working prototype to a defensible neuro-symbolic enterprise system with auditable reasoning, symbolic guardrails, stronger reranking, and measurable advantages over traditional RAG.

## Priority 1: Upgrade Policy Guard To A Symbolic Engine

Current state:

- `SAGE-Enterprise-Graph-RAG/app/policy_guard.py` performs minimal grounding/provenance checks.
- It can flag missing evidence or missing policy provenance, but it does not yet enforce hard graph constraints.

Planned changes:

- Implement graph-backed symbolic constraints.
- Encode enterprise rules as graph-checkable constraints, for example: `Service A requires Approval B`.
- Before final answer release, query Neo4j for required edges such as `APPROVED_BY`, `REQUIRES_APPROVAL`, `DEPENDS_ON`, or equivalent project schema edges.
- If a required edge/path is absent, return a guarded answer with explicit uncertainty instead of allowing the generator to infer it.
- Add policy verdicts to the agentic trace: `passed`, `failed`, `missing_edge`, `missing_approval`, `insufficient_policy_evidence`.
- Add a policy constraint test dataset with positive and negative examples.

PII filtering:

- Add a dedicated PII detection/redaction step before generator input.
- Start with deterministic regexes for email, phone, IDs, account-like numbers, and optionally add NER for names/locations if available.
- Store both raw evidence and redacted generator context separately so provenance remains auditable while model exposure is minimized.
- Add tests proving sensitive values are redacted from generator prompts but trace/provenance metadata can still reference protected source IDs safely.

Acceptance criteria:

- Policy-sensitive questions cannot pass solely on LLM judgment.
- At least one policy rule is enforced through a Neo4j graph query.
- PII-bearing evidence is redacted before generation.
- The critic can trigger retry or guarded refusal when symbolic constraints fail.

## Priority 2: Formalize Neuro-Symbolic Provenance

Current state:

- Answers include `answer_payload.evidence_refs`.
- Agentic traces include tool calls, evidence items, route history, and critic verdicts.
- This is useful, but not yet a formal provenance bundle.

Planned changes:

- Create a formal `provenance_bundle` object for each assistant answer.
- Include answer metadata: message id, user id, query type, timestamp, agentic mode.
- Include retrieval metadata: selected strategy, tool sequence, rerank status, scores.
- Include symbolic metadata: Cypher query or graph operation used, validated node/edge/path IDs, hop count, validation result.
- Include source metadata: document ids, chunk ids, fact ids, claim ids, SAIA run id where applicable.
- Include safety metadata: policy guard result, PII redaction summary, critic issues.
- Expose the bundle through the chat response and info/explainability panel.

Graph path visualization:

- Export the specific graph path used for the answer, not only a textual `evidence_ref`.
- Include the Cypher template/query name and the returned subgraph nodes/edges.
- Add graph path accuracy as an evaluation metric: percentage of answers whose cited path exists and supports the answer.

Deductive justification:

- Add an explicit reasoning summary field, for example:
  - `inductive_step`: what was inferred from semantic/fulltext evidence.
  - `symbolic_step`: what graph path or rule was checked.
  - `deductive_conclusion`: final answer grounded in validated evidence.
- Keep this explanation concise and user-readable in the UI.

Acceptance criteria:

- Every agentic answer can produce a provenance bundle.
- A manager can inspect the path from source node to answer-supporting fact.
- The bundle distinguishes semantic evidence, graph facts, symbolic constraints, and generated text.

## Priority 3: Expand Performance Harness

Current state:

- `scripts/performance_comparison.py` compares SAGE against traditional RAG with quality, latency, ROUGE, and LLM-based evaluation.
- It does not yet measure graph path accuracy, multi-hop success, tool faithfulness, rerank uplift, or SAIA freshness.

Planned changes:

- Add ablation studies:
  - traditional RAG only
  - SAGE without graph evidence
  - SAGE with graph evidence but no SAIA
  - SAGE with graph evidence plus SAIA
  - SAGE with and without reranking
  - SAGE with and without policy guard
- Add multi-hop success rate:
  - Build queries requiring 2-hop and 3-hop graph reasoning.
  - Measure whether the final answer identifies the correct node/fact and cites the correct graph path.
- Add graph path accuracy:
  - Verify cited paths exist in Neo4j.
  - Verify path endpoints match the answer entities.
- Add tool-call faithfulness:
  - Check whether final answer claims are supported by retrieved tool outputs.
- Add SAIA effectiveness:
  - Create conflicting-information trials, for example "Policy A is current" followed by "Policy B supersedes Policy A".
  - Measure stale citation rate: how often the system still cites superseded facts.
  - Measure adjustment latency: time from ingestion to updated fact availability.
  - Measure supersession correctness: one old current fact becomes superseded and one new fact becomes current.

Acceptance criteria:

- Evaluation report includes graph-specific metrics, not only generic LLM scores.
- Traditional RAG vs SAGE comparison includes at least one clear multi-hop failure/success contrast.
- SAIA metrics quantify freshness and conflict handling.

## Priority 4: Refine Agentic Tracing And Reranking

Current state:

- `app/rerank.py` is minimal and sorts by existing score.
- `app/agentic.py` creates a heuristic JSON plan and supports one retry when the critic requests stronger grounding.

Cross-encoder reranking:

- Add a stronger reranker after initial retrieval.
- Use a cross-encoder when available, with a safe fallback to current score sorting.
- Rerank by combining semantic relevance, lexical match, graph structural confidence, and policy/path validity.
- Record pre-rank and post-rank positions in the provenance bundle.

Recursive planning:

- Improve planner behavior when graph traversal returns zero nodes.
- Allow planner to revise query constraints, switch retrieval modality, or broaden/narrow graph filters.
- Record planner revisions in `trace.agentic.route_history`.
- Prevent unbounded loops with strict round and retry budgets.

Acceptance criteria:

- Reranking produces measurable uplift in evaluation.
- Planner can self-correct at least one failed initial traversal.
- Trace clearly shows why the planner changed route.

## Priority 5: Explainability Panel And Demo Readiness

Current state:

- Frontend explainability work exists conceptually through the info icon and answer trace needs.
- Backend already emits traces and `answer_payload`, but the formal bundle still needs to be standardized.

Explainability panel:

- Show answer metadata, retrieval route, graph path, evidence refs, policy verdict, SAIA status, and confidence.
- Keep it opt-in through the info icon so normal chat remains clean.
- Show global graph health separately from per-answer provenance.

Concise demos:

- Build a side-by-side demo:
  - Traditional RAG hallucinates or misses a dependency.
  - SAGE identifies the answer through a Neo4j path and displays the path in the provenance panel.
- Use a second demo for SAIA:
  - Old fact is added.
  - New conflicting fact arrives.
  - SAGE stops citing the outdated fact and shows the supersession trail.

Presentation strategy:

- Keep the demo under the 30-minute slot.
- Emphasize novelty clearly:
  - SAIA: self-adjustment on information addition with conflict supersession.
  - Neuro-symbolic hybrid search: semantic/fulltext retrieval combined with graph validation.
  - Formal provenance: path-backed, inspectable answer bundles.
  - Policy guard: symbolic constraints checked against graph structure.

## Implementation Order

1. Strengthen `policy_guard.py` with symbolic graph checks and PII redaction.
2. Standardize `provenance_bundle` in backend responses.
3. Add graph path export and path accuracy validation.
4. Upgrade reranking with cross-encoder or structured hybrid scoring fallback.
5. Extend the evaluation harness with ablations, multi-hop success, graph path accuracy, tool faithfulness, and SAIA effectiveness.
6. Wire the frontend info panel to the formal provenance bundle.
7. Prepare concise side-by-side demo scripts for presentation.

## Definition Of Done

- Policy-sensitive answers pass through graph-backed symbolic checks.
- Generator context is PII-redacted.
- Every agentic answer can produce a formal provenance bundle.
- Evaluation report includes graph/path/SAIA-specific metrics.
- Reranking is stronger than score sorting or explicitly falls back with trace metadata.
- Planner can recover from at least one failed graph traversal.
- Presentation demo clearly shows SAGE outperforming traditional RAG on a multi-hop graph dependency and stale-info SAIA case.
