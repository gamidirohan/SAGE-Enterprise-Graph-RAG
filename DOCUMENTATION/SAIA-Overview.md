# SAIA Overview

## Purpose

SAIA, the Semantic Adaptive Information Agent, turns graph-worthy chat and upload content into structured graph evidence. It extracts claims, resolves people/groups/time, promotes reliable claims into canonical facts, and keeps provenance back to the original source so later answers can explain where the information came from.

SAIA is not intended to promote every sentence. It is a controlled graph mutation layer: add new facts when useful, confirm existing facts when repeated, and supersede current facts when newer conflicting information is observed.

## Current Processing Flow

1. Source content is stored first as a chat message, attachment document, or standalone document upload.
2. SAIA checks whether the source is eligible and whether the text contains a likely enterprise fact/request signal.
3. If there is no SAIA-worthy signal, the run completes as skipped/no-claims and does not create claims or canonical facts.
4. Deterministic extractors run first for supported claim shapes.
5. LLM-assisted extraction runs only as fallback for spans where deterministic rules found no claim and the span has an LLM fallback signal.
6. Candidate claims are normalized and deduped by canonical intent before graph mutation.
7. Noncanonical, ambiguous, unresolved, or low-confidence claims are retained as claims but are not promoted into canonical facts.
8. Promotable claims are inserted, confirmed, or used to supersede existing current facts.
9. Query/retrieval uses canonical facts plus source text evidence to answer with provenance.

## Triggering Rules

SAIA now has a lightweight trigger gate before expensive extraction.

- Eligible sources include chat messages, message attachments, and document uploads.
- AI-authored evidence is ignored unless explicitly allowed by configuration.
- Plain chat without enterprise fact/request content should not create claims or canonical facts.
- Requests such as "Can you send the deck?" can create REQUEST claims, but those remain noncanonical unless they resolve to graph-worthy factual evidence.
- Standalone uploads can run SAIA after duplicate-document hash checks and graph storage.

This keeps casual chat from polluting the graph.

## Deterministic-First Extraction

The extractor order is deterministic-first:

- REQUEST
- MANAGER assertions
- REPORTS_TO
- APPROVAL
- STATUS_UPDATE
- ASSIGNMENT
- MEETING_EVENT
- COMMITMENT

If deterministic extraction succeeds for a sentence span, SAIA suppresses LLM fallback for that same span. This avoids duplicate variants such as a precise meeting event plus a generic LLM meeting event for the same sentence.

LLM-assisted extraction remains supported through `SAIA_LLM_ASSISTED=true`, but it is fallback-only. It fills gaps; it should not compete with deterministic claims that already resolved the same span.

## Canonicalization And Mutation

SAIA uses a canonical key to decide how facts relate to existing graph state.

- `insert_new_fact`: no current fact exists for the canonical key, so SAIA creates one.
- `confirm_existing_fact`: an equivalent current fact already exists, so SAIA links the new claim as support and touches the existing fact.
- `supersede_current_fact`: a current fact exists for the same canonical key but the new claim conflicts with it, so SAIA creates the new fact and marks the old fact as `superseded`.
- `not_promoted`: the claim was extracted but is noncanonical, unresolved, ambiguous, or below confidence threshold.
- `pending_review`: multiple existing facts or unclear mutation state require review instead of automatic replacement.

For example:

- "Charlie Davis reports to Diana Wilson." creates one current `REPORTS_TO` fact with canonical key `reports_to::3`.
- "Charlie Davis now reports to Elijah Parker." creates one new current fact for `reports_to::3`, supersedes the Diana Wilson fact once, and records the replacement.

## Deduplication And Self-Replacement Protection

SAIA dedupes claim candidates before mutating the graph. Dedupe is based on canonical intent, not only raw text.

The dedupe key considers:

- claim type
- canonical key
- subject
- object
- normalized value
- temporal value where relevant
- scope for scoped events

Special handling:

- `REPORTS_TO` ignores temporal `now` as a separate fact identity. `now` means "update the current reporting relationship", not "create a second temporal reporting fact".
- `MEETING_EVENT` dedupes by scope, event value/signature, and normalized time.
- Deterministic candidates win over LLM candidates when both describe the same event/fact.
- Same-run duplicates should be ignored/confirmed rather than generating self-supersession.

## Individual vs Group Messages

Direct messages can usually resolve subjects and objects confidently, so graph-worthy facts can be promoted.

Group messages are stricter:

- If a request target is ambiguous, SAIA keeps the REQUEST claim but does not create a canonical fact.
- Group meeting facts can be promoted when the event, group scope, and time are clear.
- Group-scoped events use canonical keys such as `meeting::<group_id>::project-alpha-review`.

Example:

- "Can you send the deck?" in a group creates a REQUEST claim with `skipped_noncanonical` and `canonical_fact_count = 0`.
- "We have a Project Alpha review next Monday at 10am." in a group creates one `MEETING_EVENT` canonical fact for that group scope.

## Document Uploads

Document upload processing now has two checks:

1. Hash/document ID duplicate check.
2. SAIA processing if the document is new and SAIA is enabled.

If the document already exists, it should not be inserted into the graph again, and SAIA should not rerun for that duplicate upload.

Standalone uploads are supported as `document_upload` sources. Message attachments continue to use the attachment source path.

## Retrieval Behavior

SAIA improves retrieval by making stable facts available as graph evidence instead of only free-form chunks.

- Person lookups prefer canonical facts such as `REPORTS_TO`.
- Updated facts should answer from the current canonical fact, not from superseded facts.
- Superseded facts remain in the graph for audit/provenance but should not be treated as the current answer.
- Noncanonical claims can still be useful evidence, but they do not become canonical truth.

## Observed Bruno Results

The latest Bruno run used existing people from `ChatAppSAGE/src/data/users.json` and filtered output with `Select-String`.

Test 3:

- Message: "Charlie Davis reports to Diana Wilson."
- Result: one `REPORTS_TO` claim, one current canonical fact, no replacement.
- Mutation: `insert_new_fact`.

Test 4:

- Message: "Charlie Davis now reports to Elijah Parker."
- Result: one `REPORTS_TO` claim, current Elijah Parker fact, superseded Diana Wilson fact, one replacement.
- Mutation: `supersede_current_fact`.

Test 5:

- Message: "Can you send the deck?" in a group.
- Result: one REQUEST claim, `canonical_fact_count = 0`.
- Mutation: `not_promoted`.

Test 6:

- Message: "We have a Project Alpha review next Monday at 10am." in a group.
- Result: one `MEETING_EVENT` claim and one current meeting fact.
- Canonical time: `2026-05-11T10:00:00+00:00`.

## Current Acceptance Baseline

The current expected behavior is:

- SAIA triggers only when the source and content are eligible.
- Deterministic extraction runs before LLM extraction.
- LLM extraction only fills gaps.
- Duplicate claims are removed before graph mutation.
- Repeated equivalent information confirms/supports existing facts.
- Updated conflicting information supersedes the old current fact once.
- Ambiguous group requests do not become canonical facts.
- Meeting events produce one canonical event for the same scope/event/time.

## Known Boundaries

- Existing historical graph state can affect observed counts during manual testing. A fresh canonical key or clean test scope is required for exact insert-vs-supersede assertions.
- Retrieval failures for unrelated person lookup cases are separate from SAIA extraction/canonicalization behavior.
- Superseded facts are intentionally retained for audit history.
- The frontend info-icon rendering consumes SAIA output but is separate from backend extraction quality.
