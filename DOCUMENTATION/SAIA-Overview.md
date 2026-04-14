# SAIA Overview (Current Behavior)

## Purpose
SAIA extracts structured claims from each chat message, grounds people and times, and promotes confident facts into Neo4j so later answers can cite graph-backed evidence instead of free-form text.

## Message Processing Flow
1. **Ingest message** → persist raw chat message.
2. **SAIA run** → parse claims (assignments, reports-to, etc.), ground entities/timestamps, clean commitments (e.g., remove correction markers like "instead").
3. **Canonicalization** → if confidence is high, upsert/update a canonical fact in Neo4j and link it to the source message; otherwise leave the claim unresolved/ambiguous and do not change the graph.
4. **Storage** → claims/facts retain provenance back to the message; retrieval can fetch both text chunks and facts.
5. **Answering** → responses are built from graph + message evidence and emitted via `answer_payload` (summary, bullets, explanation, evidence_refs). Legacy `answer` mirrors the summary.

## Individual vs. Group Messages
- **Direct messages (one-to-one):** subjects/objects are resolved and promoted when confidence is sufficient.
- **Group messages:** if a unique recipient cannot be resolved, SAIA keeps the claim ambiguous, avoids promoting a canonical fact, and surfaces guidance about group ambiguity. Evidence may appear, but graph updates are withheld until disambiguated.

## After a Message is Sent
- Message is stored → SAIA extracts/grounds → canonical facts are promoted when confident → future queries read from graph + text with provenance.
- Person resolution prefers a single best `User:Person` record (name/email/display_name) to avoid duplicate nodes.

## Retrieval & Mode Signals
- Person lookups filter to names in the query and use conversation metadata (type/id/group) to scope evidence.
- Queries using "who/whom" are treated as direct lookups (short mode); broad/compare/audit/explain prompts select long mode, but long answers stay structured (summary + bullets).

## Operational Note
- If Neo4j is unreachable (e.g., Aura timeout), SAIA cannot promote facts. The `/api/bootstrap` endpoint now returns 503 in this state; health remains process-only. Retry once connectivity is restored.

<!-- TODO: Future planner/critic flows may replace the producer, but must continue writing stable `answer_payload` and Neo4j facts as described above. -->
