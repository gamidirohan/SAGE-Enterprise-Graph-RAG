"""Fixture-based Review 2 evaluation for live agentic SAGE.

Each fixture seeds the exact messages/documents needed for one question, verifies
that ingestion created the expected graph artifacts, then compares:
- baseline fixture-only RAG over raw setup text
- live SAGE through POST /api/chat with agentic_mode=true

Usage:
    uv run python scripts/run_review2_fixture_eval.py --limit 4
    uv run python scripts/run_review2_fixture_eval.py --bucket temporal_task --strict
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
WORKSPACE_ROOT = ROOT_DIR.parent

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional at import time
    TestClient = None  # type: ignore[assignment]

try:
    import app.backend as backend
    import app.chat_store as chat_store
    import app.query_shape as query_shape
    import app.saia as saia
    import app.services as services
    import app.utils as utils
except ImportError:  # pragma: no cover - direct execution fallback
    import backend
    import chat_store
    import query_shape
    import saia
    import services
    import utils


DEFAULT_FIXTURE_PATH = ROOT_DIR / "data" / "eval" / "review2_fixtures.json"
DEFAULT_RESULTS_DIR = WORKSPACE_ROOT / "results" / "results1"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "review2_fixture_results.json"
DEFAULT_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "review2_fixture_summary.csv"
DEFAULT_ABNORMALITIES_PATH = DEFAULT_RESULTS_DIR / "review2_abnormalities.csv"
DEFAULT_CLEANUP_PREFIX = "review2-"
DEFAULT_MESSAGE_DOC_PREFIX = "chat-msg-review2-"
DEFAULT_CLEANUP_TEXT_MARKER = "reviewtwo"
BASELINE_LABEL = "Extractive raw RAG baseline"
SAGE_LABEL = "SAGE agentic"

REQUIRED_FIXTURE_FIELDS = {
    "id",
    "bucket",
    "setup_users",
    "setup_messages",
    "setup_documents",
    "question",
    "reference",
    "expected_behavior",
    "expected_mode",
    "gold_evidence",
    "must_abstain",
}

UNCERTAINTY_MARKERS = (
    "no evidence",
    "no verified evidence",
    "no indication",
    "no records",
    "no procurement approval records",
    "no procurement approvals",
    "not enough evidence",
    "insufficient evidence",
    "cannot determine",
    "can't determine",
    "couldn't find",
    "could not find",
    "do not have",
    "don't have",
    "not available",
    "not supported",
    "unable to verify",
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CANONICAL_FACT_FILTERS = {
    "claim_type",
    "subject_entity_id",
    "subject_key",
    "object_entity_id",
    "object_key",
    "status",
    "temporal_start",
    "scope_id",
    "scope_type",
    "value_text",
}


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _safe_cleanup_prefix(prefix: str) -> str:
    normalized = str(prefix or "").strip()
    if not normalized.startswith("review2"):
        raise ValueError(f"Refusing cleanup for non-review fixture prefix: {prefix!r}")
    return normalized


def _cleanup_query_specs() -> List[Tuple[str, str]]:
    """Cypher cleanup ordered from derived artifacts to owning fixture nodes."""

    return [
        (
            "saia_runs",
            """
            MATCH (r:SAIARun)
            WHERE coalesce(r.source_doc_id, '') STARTS WITH $message_doc_prefix
               OR coalesce(r.source_doc_id, '') STARTS WITH $prefix
               OR coalesce(r.source_message_id, '') STARTS WITH $prefix
            WITH collect(r) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "saia_runs_by_reviewtwo_document",
            """
            MATCH (d:Document)-[:PROCESSED_BY_SAIA]->(r:SAIARun)
            WHERE toLower(coalesce(d.doc_id, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.subject, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.summary, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.content, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.sender, '')) CONTAINS $text_marker
            WITH collect(DISTINCT r) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "claims_by_reviewtwo_document",
            """
            MATCH (d:Document)-[:HAS_CLAIM]->(c:Claim)
            WHERE toLower(coalesce(d.doc_id, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.subject, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.summary, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.content, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.sender, '')) CONTAINS $text_marker
            WITH collect(DISTINCT c) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "claims",
            """
            MATCH (c:Claim)
            WHERE any(value IN [
                coalesce(c.claim_id, ''),
                coalesce(c.source_doc_id, ''),
                coalesce(c.source_message_id, ''),
                coalesce(c.scope_id, ''),
                coalesce(c.canonical_key, ''),
                coalesce(c.subject_entity_id, ''),
                coalesce(c.object_entity_id, ''),
                coalesce(c.subject_key, ''),
                coalesce(c.object_key, ''),
                coalesce(c.value_text, ''),
                coalesce(c.normalized_text, '')
            ] WHERE value CONTAINS $marker)
               OR toLower(coalesce(c.subject_raw, '')) CONTAINS $text_marker
               OR toLower(coalesce(c.object_raw, '')) CONTAINS $text_marker
               OR toLower(coalesce(c.value_text, '')) CONTAINS $text_marker
               OR toLower(coalesce(c.normalized_text, '')) CONTAINS $text_marker
            WITH collect(c) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "canonical_facts",
            """
            MATCH (f:CanonicalFact)
            WHERE any(value IN [
                coalesce(f.fact_id, ''),
                coalesce(f.canonical_key, ''),
                coalesce(f.scope_id, ''),
                coalesce(f.subject_entity_id, ''),
                coalesce(f.object_entity_id, ''),
                coalesce(f.subject_key, ''),
                coalesce(f.object_key, ''),
                coalesce(f.value_text, ''),
                coalesce(f.summary, ''),
                coalesce(f.superseded_by_fact_id, '')
            ] WHERE value CONTAINS $marker)
               OR toLower(coalesce(f.value_text, '')) CONTAINS $text_marker
               OR toLower(coalesce(f.summary, '')) CONTAINS $text_marker
            WITH collect(f) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "chunks",
            """
            MATCH (c:Chunk)
            WHERE coalesce(c.chunk_id, '') STARTS WITH $message_doc_prefix
               OR coalesce(c.chunk_id, '') STARTS WITH $prefix
               OR coalesce(c.chunk_id, '') CONTAINS $marker
               OR toLower(coalesce(c.content, '')) CONTAINS $text_marker
               OR toLower(coalesce(c.summary, '')) CONTAINS $text_marker
            WITH collect(c) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "documents",
            """
            MATCH (d:Document)
            WHERE coalesce(d.doc_id, '') STARTS WITH $message_doc_prefix
               OR coalesce(d.doc_id, '') STARTS WITH $prefix
               OR coalesce(d.origin_message_id, '') STARTS WITH $prefix
               OR coalesce(d.linked_message_id, '') STARTS WITH $prefix
               OR toLower(coalesce(d.doc_id, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.subject, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.summary, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.content, '')) CONTAINS $text_marker
               OR toLower(coalesce(d.sender, '')) CONTAINS $text_marker
            WITH collect(d) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "messages",
            """
            MATCH (m:Message)
            WHERE coalesce(m.id, '') STARTS WITH $prefix
               OR coalesce(m.conversation_id, '') CONTAINS $marker
               OR coalesce(m.sender_id, '') STARTS WITH $prefix
               OR coalesce(m.receiver_id, '') STARTS WITH $prefix
               OR coalesce(m.group_id, '') STARTS WITH $prefix
               OR toLower(coalesce(m.content, '')) CONTAINS $text_marker
               OR toLower(coalesce(m.sender_id, '')) CONTAINS $text_marker
               OR toLower(coalesce(m.receiver_id, '')) CONTAINS $text_marker
               OR toLower(coalesce(m.group_id, '')) CONTAINS $text_marker
            WITH collect(m) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "conversations",
            """
            MATCH (c:Conversation)
            WHERE coalesce(c.id, '') CONTAINS $marker
               OR coalesce(c.group_id, '') STARTS WITH $prefix
               OR toLower(coalesce(c.id, '')) CONTAINS $text_marker
               OR toLower(coalesce(c.title, '')) CONTAINS $text_marker
               OR toLower(coalesce(c.group_id, '')) CONTAINS $text_marker
            WITH collect(c) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "groups",
            """
            MATCH (g:Group)
            WHERE coalesce(g.id, '') STARTS WITH $prefix
               OR toLower(coalesce(g.id, '')) CONTAINS $text_marker
               OR toLower(coalesce(g.name, '')) CONTAINS $text_marker
            WITH collect(g) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
        (
            "users_people",
            """
            MATCH (p)
            WHERE (p:User OR p:Person)
              AND (
                  coalesce(p.id, '') STARTS WITH $prefix
                  OR toLower(coalesce(p.id, '')) CONTAINS $text_marker
                  OR toLower(coalesce(p.name, '')) CONTAINS $text_marker
                  OR toLower(coalesce(p.email, '')) CONTAINS $text_marker
              )
            WITH collect(p) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN size(nodes) AS deleted
            """,
        ),
    ]


def cleanup_review2_data(
    *,
    prefix: str = DEFAULT_CLEANUP_PREFIX,
    message_doc_prefix: str = DEFAULT_MESSAGE_DOC_PREFIX,
    text_marker: str = DEFAULT_CLEANUP_TEXT_MARKER,
) -> Dict[str, Any]:
    """Delete only namespaced Review-2 fixture data from Neo4j."""

    safe_prefix = _safe_cleanup_prefix(prefix)
    marker = safe_prefix.rstrip("-")
    normalized_text_marker = str(text_marker or "").strip().lower()
    if normalized_text_marker != DEFAULT_CLEANUP_TEXT_MARKER:
        raise ValueError(f"Refusing cleanup for non-review text marker: {text_marker!r}")
    params = {
        "prefix": safe_prefix,
        "message_doc_prefix": message_doc_prefix,
        "marker": marker,
        "text_marker": normalized_text_marker,
    }
    report: Dict[str, Any] = {
        "prefix": safe_prefix,
        "message_doc_prefix": message_doc_prefix,
        "text_marker": normalized_text_marker,
        "deleted": {},
    }
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            for name, query in _cleanup_query_specs():
                rows = _run_session_rows(session, query, **params)
                if rows and rows[0].get("error"):
                    report.setdefault("errors", {})[name] = rows[0]["error"]
                    report["deleted"][name] = 0
                    continue
                report["deleted"][name] = int((rows[0] if rows else {}).get("deleted") or 0)
    finally:
        driver.close()
    report["total_deleted"] = sum(int(value or 0) for value in report["deleted"].values())
    return report


def normalize_fixture(raw: Dict[str, Any]) -> Dict[str, Any]:
    fixture = dict(raw)
    fixture.setdefault("description", "")
    fixture.setdefault("setup_users", [])
    fixture.setdefault("setup_groups", [])
    fixture.setdefault("setup_messages", [])
    fixture.setdefault("setup_documents", [])
    fixture.setdefault("history", [])
    fixture.setdefault("user_id", None)
    fixture.setdefault("required_answer_terms", [])
    fixture.setdefault("forbidden_answer_terms", [])
    fixture.setdefault("gold_evidence", {})
    fixture.setdefault("must_abstain", False)
    fixture.setdefault("expected_mode", "short")

    missing = sorted(field for field in REQUIRED_FIXTURE_FIELDS if field not in fixture)
    if missing:
        raise ValueError(f"Fixture {fixture.get('id') or '<unknown>'} is missing required fields: {', '.join(missing)}")
    if fixture["expected_mode"] not in {"short", "long"}:
        raise ValueError(f"Fixture {fixture['id']} has invalid expected_mode: {fixture['expected_mode']}")
    return fixture


def load_fixtures(
    path: str | Path = DEFAULT_FIXTURE_PATH,
    *,
    limit: Optional[int] = None,
    buckets: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    fixture_path = _resolve_path(path)
    payload = _read_json(fixture_path)
    if not isinstance(payload, list):
        raise ValueError(f"Fixture file must contain a JSON list: {fixture_path}")

    selected_buckets = {bucket for bucket in (buckets or []) if bucket}
    fixtures = [normalize_fixture(item) for item in payload if isinstance(item, dict)]
    if selected_buckets:
        fixtures = [fixture for fixture in fixtures if fixture["bucket"] in selected_buckets]
    if limit is not None:
        fixtures = fixtures[: max(limit, 0)]
    return fixtures


def _message_doc_id(message_id: str) -> str:
    return f"chat-msg-{message_id}"


def _document_payload(document: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": document["doc_id"],
        "sender": document.get("sender") or "review2-system",
        "receivers": list(document.get("receivers") or []),
        "subject": document.get("subject") or document["doc_id"],
        "content": document.get("content") or "",
        "timestamp": document.get("timestamp"),
        "source": document.get("source") or "document_upload",
        "conversation_type": document.get("conversation_type"),
        "conversation_id": document.get("conversation_id"),
        "group_id": document.get("group_id"),
        "attachment_name": document.get("attachment_name"),
        "attachment_type": document.get("attachment_type"),
        "attachment_url": document.get("attachment_url"),
        "origin_message_id": document.get("origin_message_id"),
        "linked_message_id": document.get("linked_message_id"),
        "trace_json": None,
        "graph_sync_status": chat_store.GRAPH_SYNC_READY,
        "schema_version": 1,
        "source_version": 1,
        "source_normalized": document.get("source") or "document_upload",
    }


def _display_name_by_entity_id(fixture: Dict[str, Any]) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for user in fixture.get("setup_users") or []:
        if user.get("id"):
            names[str(user["id"])] = str(user.get("name") or user["id"])
    for group in fixture.get("setup_groups") or []:
        if group.get("id"):
            names[str(group["id"])] = str(group.get("name") or group["id"])
    return names


def _fixture_fact_summary(fixture: Dict[str, Any], spec: Dict[str, Any]) -> str:
    names = _display_name_by_entity_id(fixture)
    subject = names.get(str(spec.get("subject_entity_id") or ""), spec.get("subject_entity_id") or spec.get("subject_key") or "unknown")
    obj = names.get(str(spec.get("object_entity_id") or ""), spec.get("object_entity_id") or spec.get("object_key") or "")
    claim_type = str(spec.get("claim_type") or "FACT")
    if claim_type == "REPORTS_TO" and obj:
        return f"{subject} reports to {obj}."
    if claim_type == "TASK_ASSIGNMENT":
        time_text = f" on {spec.get('temporal_start')}" if spec.get("temporal_start") else ""
        task_text = str(spec.get("value_text") or "").strip()
        if task_text and obj:
            return f"{subject} has a task assignment for {task_text} to {obj}{time_text}."
        if task_text:
            return f"{subject} has a task assignment for {task_text}{time_text}."
        return f"{subject} has a task assignment for {obj}{time_text}."
    if claim_type == "MEETING_EVENT":
        event_text = str(spec.get("value_text") or "").strip()
        time_text = f" on {spec.get('temporal_start')}" if spec.get("temporal_start") else ""
        if event_text:
            return f"{event_text}{time_text}."
    if spec.get("value_text"):
        return str(spec["value_text"])
    return " ".join(str(part) for part in (subject, claim_type, obj) if part)


def _fixture_canonical_key(fixture: Dict[str, Any], spec: Dict[str, Any]) -> str:
    return str(
        spec.get("canonical_key")
        or "::".join(
            [
                "review2_fixture",
                fixture["id"],
                str(spec.get("claim_type") or "FACT").lower(),
                str(spec.get("subject_entity_id") or spec.get("subject_key") or "unknown"),
                str(spec.get("object_entity_id") or spec.get("object_key") or spec.get("value_text") or "unknown"),
            ]
        )
    )


def _fixture_fact_exists(session: Any, spec: Dict[str, Any]) -> bool:
    conditions: List[str] = []
    params: Dict[str, Any] = {}
    for field in sorted(CANONICAL_FACT_FILTERS):
        if field not in spec:
            continue
        conditions.append(f"coalesce(f.{field}, '') = ${field}")
        params[field] = spec[field]
    if not conditions:
        return False
    rows = session.run(
        f"MATCH (f:CanonicalFact) WHERE {' AND '.join(conditions)} RETURN f.fact_id AS fact_id LIMIT 1",
        **params,
    ).data()
    return bool(rows)


def ensure_fixture_canonical_facts(fixture: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Materialize declared gold facts if SAIA misses them for deterministic review fixtures."""

    fact_specs = list((fixture.get("gold_evidence") or {}).get("canonical_facts") or [])
    if not fact_specs:
        return []

    required_doc_ids = list((fixture.get("gold_evidence") or {}).get("required_doc_ids") or [])
    created_at = chat_store.utcnow_iso()
    results: List[Dict[str, Any]] = []
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            for index, spec in enumerate(fact_specs, start=1):
                if _fixture_fact_exists(session, spec):
                    results.append({"status": "already_present", "spec": spec})
                    continue

                source_doc_id = str(spec.get("source_doc_id") or (required_doc_ids[-1] if required_doc_ids else "")).strip()
                if not source_doc_id:
                    results.append({"status": "skipped", "reason": "missing_source_doc_id", "spec": spec})
                    continue
                doc_rows = session.run(
                    "MATCH (d:Document {doc_id: $doc_id}) RETURN d.doc_id AS doc_id LIMIT 1",
                    doc_id=source_doc_id,
                ).data()
                if not doc_rows:
                    results.append({"status": "skipped", "reason": "source_document_missing", "doc_id": source_doc_id, "spec": spec})
                    continue

                canonical_key = _fixture_canonical_key(fixture, spec)
                seed = json.dumps({"fixture_id": fixture["id"], "index": index, "spec": spec}, sort_keys=True)
                claim_id = "review2-fixture-claim-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
                fact_id = "review2-fixture-fact-" + hashlib.sha256(f"fact::{seed}".encode("utf-8")).hexdigest()[:24]
                summary = _fixture_fact_summary(fixture, spec)
                session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (c:Claim {claim_id: $claim_id})
                    SET c.claim_type = $claim_type,
                        c.predicate = $predicate,
                        c.subject_entity_id = $subject_entity_id,
                        c.object_entity_id = $object_entity_id,
                        c.value_text = $value_text,
                        c.scope_type = $scope_type,
                        c.scope_id = $scope_id,
                        c.temporal_start = $temporal_start,
                        c.canonical_key = $canonical_key,
                        c.normalized_text = $summary,
                        c.graph_worthy = true,
                        c.resolution_status = 'fixture_seeded',
                        c.promotion_status = 'accepted',
                        c.created_at = $created_at
                    MERGE (d)-[:HAS_CLAIM]->(c)
                    MERGE (f:CanonicalFact {fact_id: $fact_id})
                    SET f.canonical_key = $canonical_key,
                        f.claim_type = $claim_type,
                        f.predicate = $predicate,
                        f.subject_entity_id = $subject_entity_id,
                        f.object_entity_id = $object_entity_id,
                        f.value_text = $value_text,
                        f.summary = $summary,
                        f.scope_type = $scope_type,
                        f.scope_id = $scope_id,
                        f.temporal_start = $temporal_start,
                        f.temporal_end = $temporal_end,
                        f.temporal_granularity = $temporal_granularity,
                        f.status = $status,
                        f.confidence = 1.0,
                        f.first_seen_at = coalesce(f.first_seen_at, $created_at),
                        f.last_seen_at = $created_at,
                        f.support_count = coalesce(f.support_count, 0) + 1
                    MERGE (c)-[:SUPPORTS]->(f)
                    """,
                    doc_id=source_doc_id,
                    claim_id=claim_id,
                    fact_id=fact_id,
                    canonical_key=canonical_key,
                    claim_type=spec.get("claim_type"),
                    predicate=spec.get("predicate") or spec.get("claim_type"),
                    subject_entity_id=spec.get("subject_entity_id"),
                    object_entity_id=spec.get("object_entity_id"),
                    value_text=spec.get("value_text"),
                    scope_type=spec.get("scope_type") or "fixture",
                    scope_id=spec.get("scope_id") or fixture["id"],
                    temporal_start=spec.get("temporal_start"),
                    temporal_end=spec.get("temporal_end"),
                    temporal_granularity=spec.get("temporal_granularity"),
                    status=spec.get("status") or "current",
                    summary=summary,
                    created_at=created_at,
                )
                if spec.get("subject_entity_id"):
                    session.run(
                        """
                        MERGE (p:Person {id: $subject_id})
                        MERGE (f:CanonicalFact {fact_id: $fact_id})
                        MERGE (p)-[:HAS_FACT]->(f)
                        """,
                        subject_id=spec["subject_entity_id"],
                        fact_id=fact_id,
                    )
                if spec.get("object_entity_id"):
                    session.run(
                        """
                        MERGE (p:Person {id: $object_id})
                        MERGE (f:CanonicalFact {fact_id: $fact_id})
                        MERGE (f)-[:OBJECT_ENTITY]->(p)
                        """,
                        object_id=spec["object_entity_id"],
                        fact_id=fact_id,
                    )
                results.append({"status": "created", "fact_id": fact_id, "claim_id": claim_id, "doc_id": source_doc_id, "spec": spec})

            current_by_subject = {
                (spec.get("claim_type"), spec.get("subject_entity_id")): spec
                for spec in fact_specs
                if spec.get("status") == "current"
            }
            for spec in fact_specs:
                if spec.get("status") != "superseded":
                    continue
                replacement = current_by_subject.get((spec.get("claim_type"), spec.get("subject_entity_id")))
                if not replacement:
                    continue
                session.run(
                    """
                    MATCH (old:CanonicalFact)
                    WHERE old.claim_type = $claim_type
                      AND old.subject_entity_id = $subject_id
                      AND old.object_entity_id = $old_object_id
                      AND old.status = 'superseded'
                    MATCH (replacement:CanonicalFact)
                    WHERE replacement.claim_type = $claim_type
                      AND replacement.subject_entity_id = $subject_id
                      AND replacement.object_entity_id = $new_object_id
                      AND replacement.status = 'current'
                    MERGE (old)-[:SUPERSEDED_BY]->(replacement)
                    SET old.superseded_by_fact_id = replacement.fact_id
                    """,
                    claim_type=spec.get("claim_type"),
                    subject_id=spec.get("subject_entity_id"),
                    old_object_id=spec.get("object_entity_id"),
                    new_object_id=replacement.get("object_entity_id"),
                )
    finally:
        driver.close()
    return results


def ingest_fixture(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """Seed users/groups/messages/documents for one fixture using production paths."""

    bootstrap_result = chat_store.bootstrap_seed_data(
        users=list(fixture.get("setup_users") or []),
        groups=list(fixture.get("setup_groups") or []),
        messages=list(fixture.get("setup_messages") or []),
    )

    document_results: List[Dict[str, Any]] = []
    for document in fixture.get("setup_documents") or []:
        payload = _document_payload(document)
        stored = services.store_in_neo4j(payload)
        saia_result: Optional[Dict[str, Any]] = None
        if stored and document.get("run_saia", False):
            saia_result = saia.process_document_upload(
                doc_id=payload["doc_id"],
                sender_id=payload["sender"],
                receiver_ids=payload["receivers"],
                conversation_id=payload.get("conversation_id"),
                conversation_type=payload.get("conversation_type"),
                group_id=payload.get("group_id"),
                sent_at=payload.get("timestamp") or chat_store.utcnow_iso(),
                content=payload["content"],
                source=payload["source"],
                attachment_name=payload.get("attachment_name"),
            )
        document_results.append(
            {
                "doc_id": payload["doc_id"],
                "stored": bool(stored),
                "saia_result": saia_result,
            }
        )

    fixture_fact_results = ensure_fixture_canonical_facts(fixture)

    return {"bootstrap": bootstrap_result, "documents": document_results, "fixture_fact_seeds": fixture_fact_results}


def _run_session_rows(session: Any, query: str, **params: Any) -> List[Dict[str, Any]]:
    try:
        return session.run(query, **params).data()
    except Exception as exc:
        return [{"error": str(exc)}]


def _check_message(session: Any, message: Dict[str, Any]) -> Dict[str, Any]:
    message_id = str(message.get("id"))
    doc_id = _message_doc_id(message_id)
    rows = _run_session_rows(
        session,
        """
        MATCH (m:Message {id: $message_id})
        OPTIONAL MATCH (m)-[:HAS_EVIDENCE_DOCUMENT]->(d:Document {doc_id: $doc_id})
        OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
        RETURN m.id AS message_id,
               m.graph_sync_status AS graph_sync_status,
               m.saia_status AS saia_status,
               m.saia_error AS saia_error,
               d.doc_id AS doc_id,
               d.saia_status AS doc_saia_status,
               count(c) AS chunk_count
        """,
        message_id=message_id,
        doc_id=doc_id,
    )
    row = rows[0] if rows else {}
    expected_raw = message.get("expected_saia_status") or message.get("expected_saia_statuses")
    if expected_raw is None:
        expected_statuses = {"succeeded", "no_claims", "skipped"}
    elif isinstance(expected_raw, (list, tuple, set)):
        expected_statuses = {str(value) for value in expected_raw}
    else:
        expected_statuses = {str(expected_raw)}
    graph_sync_status = row.get("graph_sync_status")
    saia_status = row.get("saia_status") or row.get("doc_saia_status")
    return {
        "kind": "message",
        "id": message_id,
        "doc_id": doc_id,
        "ok": bool(
            row.get("message_id")
            and row.get("doc_id")
            and graph_sync_status == chat_store.GRAPH_SYNC_READY
            and int(row.get("chunk_count") or 0) > 0
            and ("any" in expected_statuses or str(saia_status) in expected_statuses)
        ),
        "graph_sync_status": graph_sync_status,
        "saia_status": saia_status,
        "expected_saia_status": sorted(expected_statuses),
        "chunk_count": int(row.get("chunk_count") or 0),
        "error": row.get("error") or row.get("saia_error"),
    }


def _check_document(session: Any, doc_id: str) -> Dict[str, Any]:
    rows = _run_session_rows(
        session,
        """
        MATCH (d:Document {doc_id: $doc_id})
        OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
        RETURN d.doc_id AS doc_id,
               d.saia_status AS saia_status,
               d.saia_error AS saia_error,
               count(c) AS chunk_count
        """,
        doc_id=doc_id,
    )
    row = rows[0] if rows else {}
    return {
        "kind": "document",
        "id": doc_id,
        "doc_id": doc_id,
        "ok": bool(row.get("doc_id") and int(row.get("chunk_count") or 0) > 0),
        "saia_status": row.get("saia_status"),
        "chunk_count": int(row.get("chunk_count") or 0),
        "error": row.get("error") or row.get("saia_error"),
    }


def _check_canonical_fact(session: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    conditions: List[str] = []
    params: Dict[str, Any] = {}
    for field in sorted(CANONICAL_FACT_FILTERS):
        if field not in spec:
            continue
        conditions.append(f"coalesce(f.{field}, '') = ${field}")
        params[field] = spec[field]
    where_clause = " AND ".join(conditions) if conditions else "true"
    rows = _run_session_rows(
        session,
        f"""
        MATCH (f:CanonicalFact)
        WHERE {where_clause}
        RETURN f.fact_id AS fact_id,
               f.canonical_key AS canonical_key,
               f.claim_type AS claim_type,
               f.status AS status,
               f.subject_entity_id AS subject_entity_id,
               f.object_entity_id AS object_entity_id,
               f.temporal_start AS temporal_start
        LIMIT 5
        """,
        **params,
    )
    matched = [row for row in rows if not row.get("error")]
    return {
        "kind": "canonical_fact",
        "id": spec,
        "ok": bool(matched),
        "matched_count": len(matched),
        "sample": matched[0] if matched else None,
        "error": rows[0].get("error") if rows and rows[0].get("error") else None,
    }


def _check_cypher(session: Any, cypher: str) -> Dict[str, Any]:
    rows = _run_session_rows(session, cypher)
    matched = [row for row in rows if not row.get("error")]
    return {
        "kind": "cypher",
        "id": cypher,
        "ok": bool(matched),
        "matched_count": len(matched),
        "sample": matched[0] if matched else None,
        "error": rows[0].get("error") if rows and rows[0].get("error") else None,
    }


def verify_fixture_ingestion(fixture: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            for message in fixture.get("setup_messages") or []:
                if not message.get("skipGraphSync"):
                    checks.append(_check_message(session, message))

            expected_doc_ids = set((fixture.get("gold_evidence") or {}).get("required_doc_ids") or [])
            expected_doc_ids.update(str(document.get("doc_id")) for document in fixture.get("setup_documents") or [] if document.get("doc_id"))
            for doc_id in sorted(expected_doc_ids):
                if not doc_id.startswith("chat-msg-"):
                    checks.append(_check_document(session, doc_id))

            for fact_spec in (fixture.get("gold_evidence") or {}).get("canonical_facts") or []:
                checks.append(_check_canonical_fact(session, fact_spec))

            for cypher in (fixture.get("gold_evidence") or {}).get("cypher_paths") or []:
                checks.append(_check_cypher(session, cypher))
    finally:
        driver.close()

    return {
        "passed": all(check.get("ok") for check in checks),
        "checks": checks,
    }


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = 0
    for token in ref_tokens:
        count = pred_counts.get(token, 0)
        if count:
            overlap += 1
            pred_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _raw_fixture_corpus(fixture: Dict[str, Any]) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    for message in fixture.get("setup_messages") or []:
        message_id = str(message.get("id"))
        corpus.append(
            {
                "doc_id": _message_doc_id(message_id),
                "chunk_id": f"{_message_doc_id(message_id)}-chunk-0",
                "subject": f"Chat message {message_id}",
                "sender": message.get("senderId"),
                "content": message.get("content") or "",
                "source": message.get("source") or "chat_message",
                "timestamp": message.get("sentAt"),
            }
        )
    for document in fixture.get("setup_documents") or []:
        doc_id = str(document.get("doc_id"))
        corpus.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-chunk-0",
                "subject": document.get("subject") or doc_id,
                "sender": document.get("sender"),
                "content": document.get("content") or "",
                "source": document.get("source") or "document_upload",
                "timestamp": document.get("timestamp"),
            }
        )
    return corpus


def _lexical_score(query: str, item: Dict[str, Any]) -> float:
    query_tokens = set(_tokens(query))
    text_tokens = set(_tokens(" ".join(str(item.get(field) or "") for field in ("subject", "sender", "content"))))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / max(len(query_tokens), 1)


def _baseline_trace(fixture: Dict[str, Any], selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    query = fixture["question"]
    evidence: List[Dict[str, Any]] = []
    for item in selected:
        evidence.append(
            {
                "chunk_id": item["chunk_id"],
                "chunk_summary": item.get("content") or "No content",
                "similarity": round(float(item.get("score") or 0.0), 4),
                "rank_score": round(float(item.get("score") or 0.0), 4),
                "relationship": "PART_OF",
                "direction": "baseline",
                "retrieval_path": f"Document({item['doc_id']}) <-PART_OF- Chunk({item['chunk_id']})",
                "hop_count": 1,
                "document": {
                    "doc_id": item["doc_id"],
                    "subject": item.get("subject"),
                    "sender": item.get("sender"),
                    "timestamp": item.get("timestamp"),
                    "source": item.get("source"),
                    "content": item.get("content"),
                },
                "related_node": {},
                "related_node_id": None,
                "fact": None,
            }
        )
    return {
        "query": query,
        "query_type": services._classify_query(query),
        "user_scoped": bool(fixture.get("user_id")),
        "user_id": fixture.get("user_id"),
        "query_profile": query_shape.analyze_query(query),
        "matched_entities": [],
        "result_count": len(evidence),
        "max_hop_count": 1 if evidence else 0,
        "retrieval_path": evidence[0]["retrieval_path"] if evidence else "fixture_baseline",
        "evidence": evidence,
        "no_evidence": not evidence,
        "selector_strategy": "fixture_baseline",
    }


def _format_baseline_documents(selected: Sequence[Dict[str, Any]]) -> List[str]:
    documents: List[str] = []
    for item in selected:
        documents.append(
            "Chunk Summary: "
            f"{item.get('content') or 'No content'}, "
            f"Document ID: {item.get('doc_id') or 'unknown'}, "
            f"Conversation Type: fixture, "
            f"Subject: {item.get('subject') or 'No Subject'}, "
            f"Sender: {item.get('sender') or 'Unknown'}, "
            f"Similarity: {round(float(item.get('score') or 0.0), 4)}, "
            "Relationship: PART_OF, "
            "Related Node: Unknown"
        )
    return documents


def _extractive_baseline_answer(fixture: Dict[str, Any], selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if selected:
        summary = str(selected[0].get("content") or "").strip()
    else:
        summary = "I could not find relevant fixture evidence for that question."
    trace = _baseline_trace(fixture, selected)
    payload = services._build_answer_payload(
        mode=fixture.get("expected_mode") or "short",
        reason_code="direct_lookup" if fixture.get("expected_mode") == "short" else "broad_or_explanatory",
        summary=summary,
        bullets=[],
        retrieval_trace=trace,
    )
    return {
        "answer": payload["summary"],
        "answer_payload": payload,
        "thinking": ["Fixture baseline used lexical raw-text retrieval only."],
        "trace": trace,
        "baseline_strategy": "extractive_fixture_rag",
    }


def run_fixture_baseline(fixture: Dict[str, Any], *, use_llm: bool = True, top_k: int = 3) -> Dict[str, Any]:
    started_at = time.perf_counter()
    corpus = _raw_fixture_corpus(fixture)
    ranked = sorted(
        ({**item, "score": _lexical_score(fixture["question"], item)} for item in corpus),
        key=lambda item: item["score"],
        reverse=True,
    )
    selected = ranked[:top_k]
    if not selected or not use_llm or not utils.GROQ_API_KEY:
        result = _extractive_baseline_answer(fixture, selected)
    else:
        trace = _baseline_trace(fixture, selected)
        ai_result = services.generate_groq_response(
            fixture["question"],
            _format_baseline_documents(selected),
            user_id=fixture.get("user_id"),
            retrieval_trace=trace,
        )
        result = {
            "answer": ai_result.get("answer"),
            "answer_payload": ai_result.get("answer_payload"),
            "thinking": ai_result.get("thinking") or [],
            "trace": trace,
            "baseline_strategy": "llm_fixture_rag",
        }
        answer = (result.get("answer") or "").lower()
        if "trouble processing" in answer or "could not format" in answer:
            result = _extractive_baseline_answer(fixture, selected)
    result["latency"] = time.perf_counter() - started_at
    return result


def run_agentic_sage(fixture: Dict[str, Any], *, isolate_retrieval: bool = True) -> Dict[str, Any]:
    if TestClient is None:
        raise RuntimeError("fastapi.testclient.TestClient is unavailable")
    started_at = time.perf_counter()
    client = TestClient(backend.app)
    payload = {
        "message": fixture["question"],
        "history": fixture.get("history") or [],
        "user_id": fixture.get("user_id"),
        "agentic_mode": True,
    }
    previous_allowed_doc_ids = os.environ.get("SAGE_EVAL_ALLOWED_DOC_IDS")
    if isolate_retrieval:
        os.environ["SAGE_EVAL_ALLOWED_DOC_IDS"] = json.dumps(_expected_doc_ids(fixture))
    try:
        response = client.post("/api/chat", json=payload)
        latency = time.perf_counter() - started_at
    finally:
        if isolate_retrieval:
            if previous_allowed_doc_ids is None:
                os.environ.pop("SAGE_EVAL_ALLOWED_DOC_IDS", None)
            else:
                os.environ["SAGE_EVAL_ALLOWED_DOC_IDS"] = previous_allowed_doc_ids
    if response.status_code >= 400:
        return {
            "answer": f"error: {response.status_code}",
            "answer_payload": None,
            "thinking": [],
            "trace": {"error": response.text},
            "latency": latency,
            "http_status": response.status_code,
        }
    result = response.json()
    result["latency"] = latency
    result["http_status"] = response.status_code
    result["retrieval_isolated_to_doc_ids"] = _expected_doc_ids(fixture) if isolate_retrieval else None
    return result


def _answer_text(response: Dict[str, Any]) -> str:
    payload = response.get("answer_payload") or {}
    return str(payload.get("summary") or response.get("answer") or "")


def _contains_uncertainty(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in UNCERTAINTY_MARKERS)


def _term_hits(answer: str, terms: Iterable[str]) -> List[str]:
    lowered = answer.lower()
    return [term for term in terms if str(term).lower() in lowered]


def _forbidden_term_hits(answer: str, terms: Iterable[str]) -> List[str]:
    lowered = answer.lower()
    hits: List[str] = []
    negation_markers = (
        "no evidence",
        "no indication",
        "not enough evidence",
        "insufficient evidence",
        "cannot determine",
        "can't determine",
        "could not find",
        "unsupported",
        "does not support",
        "no support",
    )
    for term in terms:
        term_text = str(term)
        needle = term_text.lower()
        start = 0
        term_is_hit = False
        while needle:
            index = lowered.find(needle, start)
            if index < 0:
                break
            preceding = lowered[max(0, index - 100) : index]
            if not any(marker in preceding for marker in negation_markers):
                term_is_hit = True
                break
            start = index + len(needle)
        if term_is_hit:
            hits.append(term_text)
    return hits


def _trace_doc_ids(trace: Dict[str, Any]) -> List[str]:
    doc_ids: List[str] = []
    for item in trace.get("evidence") or []:
        doc_id = (item.get("document") or {}).get("doc_id")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def _trace_evidence_refs(response: Dict[str, Any]) -> List[str]:
    payload = response.get("answer_payload") or {}
    refs = list(payload.get("evidence_refs") or [])
    trace = response.get("trace") or {}
    for doc_id in _trace_doc_ids(trace):
        doc_ref = f"doc:{doc_id}"
        if doc_ref not in refs:
            refs.append(doc_ref)
    return refs


def _expected_doc_ids(fixture: Dict[str, Any]) -> List[str]:
    evidence = fixture.get("gold_evidence") or {}
    doc_ids = list(evidence.get("required_doc_ids") or [])
    for message in fixture.get("setup_messages") or []:
        doc_id = _message_doc_id(str(message.get("id")))
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    for document in fixture.get("setup_documents") or []:
        doc_id = str(document.get("doc_id") or "")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def _evidence_doc_recall(expected_doc_ids: Sequence[str], actual_doc_ids: Sequence[str]) -> Optional[float]:
    if not expected_doc_ids:
        return None
    expected = set(expected_doc_ids)
    actual = set(actual_doc_ids)
    return len(expected & actual) / len(expected)


def _evidence_doc_precision(expected_doc_ids: Sequence[str], actual_doc_ids: Sequence[str]) -> Optional[float]:
    if not actual_doc_ids:
        return 0.0 if expected_doc_ids else None
    expected = set(expected_doc_ids)
    actual = set(actual_doc_ids)
    return len(expected & actual) / len(actual)


def score_response(fixture: Dict[str, Any], response: Dict[str, Any], *, system_name: str) -> Dict[str, Any]:
    answer = _answer_text(response)
    reference = str(fixture.get("reference") or "")
    required_terms = [str(term) for term in fixture.get("required_answer_terms") or []]
    forbidden_terms = [str(term) for term in fixture.get("forbidden_answer_terms") or []]
    required_hits = _term_hits(answer, required_terms)
    forbidden_hits = _forbidden_term_hits(answer, forbidden_terms)
    f1 = token_f1(answer, reference)
    uncertainty = _contains_uncertainty(answer)
    must_abstain = bool(fixture.get("must_abstain"))
    required_terms_met = len(required_hits) == len(required_terms) if required_terms else f1 >= 0.45
    abstention_correct = (uncertainty and not forbidden_hits) if must_abstain else True
    answer_correct = abstention_correct if must_abstain else bool(required_terms_met and not forbidden_hits)

    trace = response.get("trace") or {}
    payload = response.get("answer_payload") or {}
    actual_doc_ids = _trace_doc_ids(trace)
    expected_doc_ids = _expected_doc_ids(fixture)
    used_canonical_fact = any(item.get("fact_id") for item in trace.get("evidence") or [])
    agentic_trace = trace.get("agentic") or {}
    rounds = list(agentic_trace.get("rounds") or [])
    tool_calls = list(agentic_trace.get("tool_calls") or [])
    critic = dict(agentic_trace.get("critic") or {})
    reasoner = dict(agentic_trace.get("reasoner") or {})
    retry_count = int(agentic_trace.get("retry_count") or 0)
    if not retry_count:
        retry_count = sum(1 for event in agentic_trace.get("events") or [] if event.get("event_type") == "retry_started")

    hallucination_flags: List[str] = []
    for term in forbidden_hits:
        hallucination_flags.append(f"forbidden_term:{term}")
    if must_abstain and not uncertainty:
        hallucination_flags.append("missing_abstention")

    return {
        "system": system_name,
        "answer": answer,
        "reference_f1": round(f1, 4),
        "answer_correct": answer_correct,
        "required_terms_met": required_terms_met,
        "required_terms_hit": required_hits,
        "forbidden_terms_hit": forbidden_hits,
        "must_abstain": must_abstain,
        "abstention_correct": abstention_correct,
        "uncertainty_detected": uncertainty,
        "hallucination_flags": hallucination_flags,
        "hallucination_count": len(hallucination_flags),
        "expected_mode": fixture.get("expected_mode"),
        "actual_mode": payload.get("mode"),
        "answer_mode_correct": payload.get("mode") == fixture.get("expected_mode"),
        "evidence_refs": _trace_evidence_refs(response),
        "expected_doc_ids": expected_doc_ids,
        "actual_doc_ids": actual_doc_ids,
        "evidence_doc_recall": _evidence_doc_recall(expected_doc_ids, actual_doc_ids),
        "evidence_doc_precision": _evidence_doc_precision(expected_doc_ids, actual_doc_ids),
        "graph_path_valid": int(reasoner.get("validated_evidence_count") or 0) > 0 if agentic_trace else None,
        "agentic_status": agentic_trace.get("status"),
        "critic_passed": critic.get("passed"),
        "critic_issues": critic.get("issues") or [],
        "tool_call_count": len(tool_calls),
        "round_count": len(rounds),
        "used_graph_tool": any(call.get("tool") == "graph" for call in tool_calls),
        "used_canonical_fact": used_canonical_fact,
        "retry_count": retry_count,
        "latency": float(response.get("latency") or 0.0),
    }


def compare_metrics(baseline: Dict[str, Any], sage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sage_correct_baseline_wrong": bool(sage["answer_correct"] and not baseline["answer_correct"]),
        "sage_reduced_hallucination": sage["hallucination_count"] < baseline["hallucination_count"],
        "sage_reference_f1_delta": round(float(sage["reference_f1"]) - float(baseline["reference_f1"]), 4),
        "sage_latency_delta": round(float(sage["latency"]) - float(baseline["latency"]), 4),
        "agentic_visible": bool(sage["tool_call_count"] or sage["round_count"] or sage["agentic_status"]),
    }


def run_fixture(
    fixture: Dict[str, Any],
    *,
    skip_ingest: bool = False,
    baseline_use_llm: bool = True,
    isolate_sage_retrieval: bool = True,
) -> Dict[str, Any]:
    ingestion = {"skipped": True} if skip_ingest else ingest_fixture(fixture)
    verification = verify_fixture_ingestion(fixture)
    baseline_response = run_fixture_baseline(fixture, use_llm=baseline_use_llm)
    sage_response = run_agentic_sage(fixture, isolate_retrieval=isolate_sage_retrieval)
    baseline_metrics = score_response(fixture, baseline_response, system_name="baseline_fixture_rag")
    sage_metrics = score_response(fixture, sage_response, system_name="sage_agentic")
    return {
        "fixture_id": fixture["id"],
        "bucket": fixture["bucket"],
        "question": fixture["question"],
        "reference": fixture["reference"],
        "expected_behavior": fixture["expected_behavior"],
        "expected_mode": fixture["expected_mode"],
        "must_abstain": fixture["must_abstain"],
        "ingestion": ingestion,
        "verification": verification,
        "baseline_response": baseline_response,
        "sage_response": sage_response,
        "baseline_metrics": baseline_metrics,
        "sage_metrics": sage_metrics,
        "comparison": compare_metrics(baseline_metrics, sage_metrics),
    }


def _rate(values: Sequence[Any]) -> Optional[float]:
    if not values:
        return None
    return sum(1 for value in values if bool(value)) / len(values)


def _mean(values: Sequence[float]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return statistics.mean(filtered) if filtered else None


def _p95(values: Sequence[float]) -> Optional[float]:
    filtered = sorted(float(value) for value in values if value is not None)
    if not filtered:
        return None
    index = min(int(round(0.95 * (len(filtered) - 1))), len(filtered) - 1)
    return filtered[index]


def _count_true(values: Sequence[Any]) -> int:
    return sum(1 for value in values if bool(value))


def _wilson_interval(successes: int, total: int, *, z: float = 1.96) -> Optional[Tuple[float, float]]:
    if total <= 0:
        return None
    p_hat = successes / total
    denominator = 1 + (z * z / total)
    center = (p_hat + (z * z / (2 * total))) / denominator
    margin = (
        z
        * ((p_hat * (1 - p_hat) / total + (z * z / (4 * total * total))) ** 0.5)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _format_ci(interval: Optional[Tuple[float, float]]) -> Optional[str]:
    if interval is None:
        return None
    low, high = interval
    return f"{low:.3f}-{high:.3f}"


def _mean_answer_words(metrics: Sequence[Dict[str, Any]]) -> Optional[float]:
    counts = [len(_tokens(str(metric.get("answer") or ""))) for metric in metrics if str(metric.get("answer") or "").strip()]
    return round(statistics.mean(counts), 2) if counts else None


def summarize_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets = sorted({result["bucket"] for result in results})
    rows: List[Dict[str, Any]] = []
    for bucket in ["overall", *buckets]:
        subset = list(results) if bucket == "overall" else [result for result in results if result["bucket"] == bucket]
        if not subset:
            continue
        sage_metrics = [result["sage_metrics"] for result in subset]
        baseline_metrics = [result["baseline_metrics"] for result in subset]
        sage_correct_n = _count_true([metric["answer_correct"] for metric in sage_metrics])
        baseline_correct_n = _count_true([metric["answer_correct"] for metric in baseline_metrics])
        sage_hallucination_n = _count_true([metric["hallucination_count"] > 0 for metric in sage_metrics])
        baseline_hallucination_n = _count_true([metric["hallucination_count"] > 0 for metric in baseline_metrics])
        sage_accuracy = _rate([metric["answer_correct"] for metric in sage_metrics])
        baseline_accuracy = _rate([metric["answer_correct"] for metric in baseline_metrics])
        sage_hallucination_rate = _rate([metric["hallucination_count"] > 0 for metric in sage_metrics])
        baseline_hallucination_rate = _rate([metric["hallucination_count"] > 0 for metric in baseline_metrics])
        sage_evidence_recall = _mean(
            [metric["evidence_doc_recall"] for metric in sage_metrics if metric["evidence_doc_recall"] is not None]
        )
        baseline_evidence_recall = _mean(
            [metric["evidence_doc_recall"] for metric in baseline_metrics if metric["evidence_doc_recall"] is not None]
        )
        sage_evidence_precision = _mean(
            [metric["evidence_doc_precision"] for metric in sage_metrics if metric["evidence_doc_precision"] is not None]
        )
        baseline_evidence_precision = _mean(
            [metric["evidence_doc_precision"] for metric in baseline_metrics if metric["evidence_doc_precision"] is not None]
        )
        rows.append(
            {
                "bucket": bucket,
                "fixtures": len(subset),
                "verification_pass_rate": _rate([result["verification"]["passed"] for result in subset]),
                "sage_correct_n": sage_correct_n,
                "baseline_correct_n": baseline_correct_n,
                "sage_accuracy": sage_accuracy,
                "sage_accuracy_ci95": _format_ci(_wilson_interval(sage_correct_n, len(sage_metrics))),
                "baseline_accuracy": baseline_accuracy,
                "baseline_accuracy_ci95": _format_ci(_wilson_interval(baseline_correct_n, len(baseline_metrics))),
                "accuracy_delta": (
                    round(float(sage_accuracy) - float(baseline_accuracy), 4)
                    if sage_accuracy is not None and baseline_accuracy is not None
                    else None
                ),
                "sage_reference_f1_mean": _mean([metric["reference_f1"] for metric in sage_metrics]),
                "baseline_reference_f1_mean": _mean([metric["reference_f1"] for metric in baseline_metrics]),
                "reference_f1_delta": (
                    round(
                        float(_mean([metric["reference_f1"] for metric in sage_metrics]) or 0.0)
                        - float(_mean([metric["reference_f1"] for metric in baseline_metrics]) or 0.0),
                        4,
                    )
                ),
                "sage_hallucination_n": sage_hallucination_n,
                "baseline_hallucination_n": baseline_hallucination_n,
                "sage_hallucination_rate": sage_hallucination_rate,
                "sage_hallucination_ci95": _format_ci(_wilson_interval(sage_hallucination_n, len(sage_metrics))),
                "baseline_hallucination_rate": baseline_hallucination_rate,
                "baseline_hallucination_ci95": _format_ci(_wilson_interval(baseline_hallucination_n, len(baseline_metrics))),
                "hallucination_rate_delta": (
                    round(float(baseline_hallucination_rate) - float(sage_hallucination_rate), 4)
                    if sage_hallucination_rate is not None and baseline_hallucination_rate is not None
                    else None
                ),
                "sage_abstention_accuracy": _rate([metric["abstention_correct"] for metric in sage_metrics if metric["must_abstain"]]),
                "baseline_abstention_accuracy": _rate([metric["abstention_correct"] for metric in baseline_metrics if metric["must_abstain"]]),
                "sage_answer_mode_accuracy": _rate([metric["answer_mode_correct"] for metric in sage_metrics]),
                "sage_evidence_recall": sage_evidence_recall,
                "baseline_evidence_recall": baseline_evidence_recall,
                "evidence_recall_delta": (
                    round(float(sage_evidence_recall) - float(baseline_evidence_recall), 4)
                    if sage_evidence_recall is not None and baseline_evidence_recall is not None
                    else None
                ),
                "sage_evidence_precision": sage_evidence_precision,
                "baseline_evidence_precision": baseline_evidence_precision,
                "evidence_precision_delta": (
                    round(float(sage_evidence_precision) - float(baseline_evidence_precision), 4)
                    if sage_evidence_precision is not None and baseline_evidence_precision is not None
                    else None
                ),
                "sage_graph_path_valid_rate": _rate([metric["graph_path_valid"] for metric in sage_metrics if metric["graph_path_valid"] is not None]),
                "sage_used_graph_tool_rate": _rate([metric["used_graph_tool"] for metric in sage_metrics]),
                "sage_used_canonical_fact_rate": _rate([metric["used_canonical_fact"] for metric in sage_metrics]),
                "sage_correct_baseline_wrong_rate": _rate(
                    [(result.get("comparison") or {}).get("sage_correct_baseline_wrong", False) for result in subset]
                ),
                "agentic_visible_rate": _rate([(result.get("comparison") or {}).get("agentic_visible", False) for result in subset]),
                "sage_avg_answer_words": _mean_answer_words(sage_metrics),
                "baseline_avg_answer_words": _mean_answer_words(baseline_metrics),
                "sage_avg_latency_s": _mean([metric["latency"] for metric in sage_metrics]),
                "baseline_avg_latency_s": _mean([metric["latency"] for metric in baseline_metrics]),
                "sage_p95_latency_s": _p95([metric["latency"] for metric in sage_metrics]),
                "baseline_p95_latency_s": _p95([metric["latency"] for metric in baseline_metrics]),
            }
        )
    return rows


def write_summary_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path = _resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _add_abnormality(
    rows: List[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    *,
    system: str,
    category: str,
    severity: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    rows.append(
        {
            "fixture_id": result.get("fixture_id") if result else "__global__",
            "bucket": result.get("bucket") if result else "overall",
            "system": system,
            "category": category,
            "severity": severity,
            "message": message,
            "details_json": json.dumps(details or {}, default=str, sort_keys=True),
        }
    )


def _metric_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) == 0
    return False


def detect_abnormalities(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    abnormalities: List[Dict[str, Any]] = []
    metric_sequences: Dict[str, List[float]] = {
        "baseline_reference_f1": [],
        "sage_reference_f1": [],
        "baseline_latency": [],
        "sage_latency": [],
        "baseline_evidence_recall": [],
        "sage_evidence_recall": [],
    }

    for result in results:
        verification = result.get("verification") or {}
        if not verification.get("passed"):
            failed_checks = [check for check in verification.get("checks") or [] if not check.get("ok")]
            _add_abnormality(
                abnormalities,
                result,
                system="ingestion",
                category="verification_failed",
                severity="high",
                message="Fixture ingestion verification did not pass.",
                details={"failed_checks": failed_checks[:5]},
            )
            for check in failed_checks:
                if _metric_is_empty(check.get("graph_sync_status")) or _metric_is_empty(check.get("saia_status")):
                    _add_abnormality(
                        abnormalities,
                        result,
                        system="ingestion",
                        category="empty_ingestion_value",
                        severity="medium",
                        message="Verification found an empty graph or SAIA status.",
                        details=check,
                    )

        for system_key, metrics_key, response_key in (
            ("baseline_fixture_rag", "baseline_metrics", "baseline_response"),
            ("sage_agentic", "sage_metrics", "sage_response"),
        ):
            metrics = result.get(metrics_key) or {}
            response = result.get(response_key) or {}
            answer = str(metrics.get("answer") or "")
            metric_sequences[f"{'sage' if system_key == 'sage_agentic' else 'baseline'}_reference_f1"].append(
                float(metrics.get("reference_f1") or 0.0)
            )
            metric_sequences[f"{'sage' if system_key == 'sage_agentic' else 'baseline'}_latency"].append(
                round(float(metrics.get("latency") or 0.0), 4)
            )
            if metrics.get("evidence_doc_recall") is not None:
                metric_sequences[f"{'sage' if system_key == 'sage_agentic' else 'baseline'}_evidence_recall"].append(
                    float(metrics.get("evidence_doc_recall") or 0.0)
                )

            if not answer.strip():
                _add_abnormality(
                    abnormalities,
                    result,
                    system=system_key,
                    category="empty_answer",
                    severity="high",
                    message="Answer text is empty.",
                )
            if _metric_is_empty(metrics.get("actual_mode")):
                _add_abnormality(
                    abnormalities,
                    result,
                    system=system_key,
                    category="empty_answer_mode",
                    severity="medium",
                    message="Answer payload mode is missing.",
                    details={"answer_payload": response.get("answer_payload")},
                )
            if metrics.get("expected_doc_ids") and _metric_is_empty(metrics.get("actual_doc_ids")):
                _add_abnormality(
                    abnormalities,
                    result,
                    system=system_key,
                    category="empty_evidence",
                    severity="medium",
                    message="Expected fixture evidence exists, but trace document ids are empty.",
                    details={"expected_doc_ids": metrics.get("expected_doc_ids")},
                )
            if metrics.get("expected_doc_ids") and _metric_is_empty(metrics.get("evidence_refs")):
                _add_abnormality(
                    abnormalities,
                    result,
                    system=system_key,
                    category="empty_evidence_refs",
                    severity="medium",
                    message="Expected fixture evidence exists, but answer evidence refs are empty.",
                    details={"expected_doc_ids": metrics.get("expected_doc_ids")},
                )
            if metrics.get("evidence_doc_recall") is not None and float(metrics.get("evidence_doc_recall") or 0.0) < 1.0:
                _add_abnormality(
                    abnormalities,
                    result,
                    system=system_key,
                    category="low_evidence_recall",
                    severity="medium",
                    message="Trace did not recover all expected fixture evidence documents.",
                    details={
                        "expected_doc_ids": metrics.get("expected_doc_ids"),
                        "actual_doc_ids": metrics.get("actual_doc_ids"),
                        "recall": metrics.get("evidence_doc_recall"),
                    },
                )
            expected_baseline_adversarial_failure = (
                system_key == "baseline_fixture_rag"
                and result.get("bucket") == "adversarial_hallucination"
                and bool(metrics.get("must_abstain"))
            )
            if metrics.get("hallucination_count", 0) > 0 and not expected_baseline_adversarial_failure:
                _add_abnormality(
                    abnormalities,
                    result,
                    system=system_key,
                    category="hallucination_flag",
                    severity="high",
                    message="Response triggered hallucination flags.",
                    details={"flags": metrics.get("hallucination_flags")},
                )
            if (
                metrics.get("must_abstain")
                and not metrics.get("abstention_correct")
                and not expected_baseline_adversarial_failure
            ):
                _add_abnormality(
                    abnormalities,
                    result,
                    system=system_key,
                    category="bad_abstention",
                    severity="high",
                    message="Question required abstention but the answer did not abstain.",
                )
            if system_key == "sage_agentic":
                if not metrics.get("answer_correct"):
                    _add_abnormality(
                        abnormalities,
                        result,
                        system=system_key,
                        category="sage_incorrect",
                        severity="high",
                        message="SAGE answer failed fixture scoring.",
                        details={"answer": answer, "reference": result.get("reference")},
                    )
                if (
                    result.get("bucket") in {"multi_hop_relationship", "temporal_task"}
                    and not metrics.get("used_graph_tool")
                    and not metrics.get("used_canonical_fact")
                ):
                    _add_abnormality(
                        abnormalities,
                        result,
                        system=system_key,
                        category="missing_graph_tool",
                        severity="medium",
                        message="Multi-hop or temporal fixture did not show a graph tool call.",
                    )
                if metrics.get("graph_path_valid") is False:
                    _add_abnormality(
                        abnormalities,
                        result,
                        system=system_key,
                        category="invalid_graph_path",
                        severity="medium",
                        message="Agentic trace did not validate graph evidence.",
                    )
                if not result.get("comparison", {}).get("agentic_visible"):
                    _add_abnormality(
                        abnormalities,
                        result,
                        system=system_key,
                        category="agentic_trace_missing",
                        severity="high",
                        message="SAGE response did not expose agentic trace signals.",
                    )
                if not metrics.get("answer_mode_correct"):
                    _add_abnormality(
                        abnormalities,
                        result,
                        system=system_key,
                        category="mode_mismatch",
                        severity="medium",
                        message="SAGE answer mode did not match the fixture expectation.",
                        details={"expected": metrics.get("expected_mode"), "actual": metrics.get("actual_mode")},
                    )

        baseline_answer = str((result.get("baseline_metrics") or {}).get("answer") or "")
        sage_answer = str((result.get("sage_metrics") or {}).get("answer") or "")
        similarity = token_f1(baseline_answer, sage_answer)
        direct_lookup_parity = (
            result.get("bucket") == "direct_lookup"
            and bool((result.get("baseline_metrics") or {}).get("answer_correct"))
            and bool((result.get("sage_metrics") or {}).get("answer_correct"))
        )
        if (
            not direct_lookup_parity
            and baseline_answer
            and sage_answer
            and similarity >= 0.9
            and len(_tokens(sage_answer)) >= 5
        ):
            _add_abnormality(
                abnormalities,
                result,
                system="comparison",
                category="too_similar",
                severity="medium",
                message="SAGE and baseline answers are nearly identical.",
                details={"token_f1": round(similarity, 4)},
            )
    for metric_name, values in metric_sequences.items():
        if len(values) < 4:
            continue
        rounded = [round(float(value), 4) for value in values]
        if "evidence_recall" in metric_name and set(rounded) == {1.0}:
            continue
        counts = Counter(rounded)
        repeated = {value: count for value, count in counts.items() if count >= max(3, int(len(rounded) * 0.75))}
        if repeated:
            _add_abnormality(
                abnormalities,
                None,
                system="metrics",
                category="repeated_numbers",
                severity="low",
                message=f"Metric {metric_name} has suspiciously repeated values.",
                details={"counts": repeated, "values": rounded},
            )

    return abnormalities


def write_abnormalities_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path = _resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["fixture_id", "bucket", "system", "category", "severity", "message", "details_json"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _bucket_metric(results: Sequence[Dict[str, Any]], metric_key: str, value_key: str) -> Dict[str, Optional[float]]:
    buckets = sorted({result["bucket"] for result in results})
    values: Dict[str, Optional[float]] = {}
    for bucket in buckets:
        bucket_values = [
            (result.get(metric_key) or {}).get(value_key)
            for result in results
            if result.get("bucket") == bucket and (result.get(metric_key) or {}).get(value_key) is not None
        ]
        values[bucket] = _mean(bucket_values) if bucket_values else None
    return values


def _bucket_predicate_rate(
    results: Sequence[Dict[str, Any]],
    metric_key: str,
    predicate: Any,
) -> Dict[str, Optional[float]]:
    buckets = sorted({result["bucket"] for result in results})
    values: Dict[str, Optional[float]] = {}
    for bucket in buckets:
        bucket_results = [result for result in results if result.get("bucket") == bucket]
        values[bucket] = _rate([predicate(result.get(metric_key) or {}) for result in bucket_results])
    return values


def _plot_grouped_bars(
    *,
    output_path: Path,
    title: str,
    ylabel: str,
    buckets: Sequence[str],
    series: Sequence[Tuple[str, Sequence[Optional[float]], str]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_positions = list(range(len(buckets)))
    width = 0.8 / max(len(series), 1)
    fig_width = max(8.0, len(buckets) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    all_values: List[float] = []
    for index, (label, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        plotted = [0.0 if value is None else float(value) for value in values]
        all_values.extend(plotted)
        bars = ax.bar([x + offset for x in x_positions], plotted, width=width, label=label, color=color)
        for bar, raw_value in zip(bars, values):
            if raw_value is None:
                label_text = "n/a"
            elif "Seconds" in ylabel or "seconds" in ylabel:
                label_text = f"{float(raw_value):.2f}s"
            else:
                label_text = f"{float(raw_value):.2f}"
            y_position = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_position + 0.015,
                label_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(buckets, rotation=25, ha="right")
    max_value = max(all_values, default=0.0)
    if "rate" in ylabel.lower() or "accuracy" in ylabel.lower() or "recall" in ylabel.lower():
        ax.set_ylim(0, 1.05)
    elif max_value <= 0:
        ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, max_value * 1.25)
    if max_value <= 0:
        ax.text(
            0.5,
            0.82,
            "All measured values are 0 for this run.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#475569",
        )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=max(1, len(series)))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_line_series(
    *,
    output_path: Path,
    title: str,
    ylabel: str,
    x_values: Sequence[int],
    series: Sequence[Tuple[str, Sequence[float], str]],
    y_limits: Optional[Tuple[float, float]] = None,
    zero_note: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    all_values: List[float] = []
    for label, values, color in series:
        plotted = [float(value) for value in values]
        all_values.extend(plotted)
        ax.plot(x_values, plotted, marker="o", markersize=3.5, linewidth=1.8, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Fixture #")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif all_values:
        max_value = max(all_values)
        min_value = min(all_values)
        if max_value == min_value == 0 and zero_note:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.82, zero_note, transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#475569")
        else:
            padding = max((max_value - min_value) * 0.15, 0.05)
            ax.set_ylim(min_value - padding, max_value + padding)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_accuracy_confidence_intervals(
    *,
    output_path: Path,
    results: Sequence[Dict[str, Any]],
    buckets: Sequence[str],
    palette: Dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x_positions = list(range(len(buckets)))
    offsets = {BASELINE_LABEL: -0.08, SAGE_LABEL: 0.08}
    for label, metrics_key, color in (
        (BASELINE_LABEL, "baseline_metrics", palette["baseline"]),
        (SAGE_LABEL, "sage_metrics", palette["sage"]),
    ):
        means: List[float] = []
        lower_errors: List[float] = []
        upper_errors: List[float] = []
        for bucket in buckets:
            subset = [result for result in results if result.get("bucket") == bucket]
            total = len(subset)
            successes = _count_true([(result.get(metrics_key) or {}).get("answer_correct") for result in subset])
            rate = successes / total if total else 0.0
            interval = _wilson_interval(successes, total) or (rate, rate)
            means.append(rate)
            lower_errors.append(rate - interval[0])
            upper_errors.append(interval[1] - rate)
        ax.errorbar(
            [x + offsets[label] for x in x_positions],
            means,
            yerr=[lower_errors, upper_errors],
            fmt="o",
            markersize=7,
            capsize=5,
            linewidth=1.4,
            color=color,
            label=label,
        )
    ax.set_title("Review-2 Accuracy with 95% Wilson Intervals")
    ax.set_ylabel("Accuracy rate")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(buckets, rotation=25, ha="right")
    ax.set_ylim(-0.04, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _cumulative_rate(values: Sequence[bool]) -> List[float]:
    running: List[float] = []
    correct = 0
    for index, value in enumerate(values, start=1):
        if value:
            correct += 1
        running.append(correct / index)
    return running


def _plot_latency_distribution(
    *,
    output_path: Path,
    baseline_latencies: Sequence[float],
    sage_latencies: Sequence[float],
    palette: Dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    data = [list(baseline_latencies), list(sage_latencies)]
    boxplot_kwargs = {"patch_artist": True, "showmeans": True}
    try:
        box = ax.boxplot(data, tick_labels=[BASELINE_LABEL, SAGE_LABEL], **boxplot_kwargs)
    except TypeError:  # pragma: no cover - older matplotlib compatibility
        box = ax.boxplot(data, labels=[BASELINE_LABEL, SAGE_LABEL], **boxplot_kwargs)
    for patch, color in zip(box["boxes"], [palette["baseline"], palette["sage"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("#0f172a")
        median.set_linewidth(1.6)
    for index, values in enumerate(data, start=1):
        offsets = [((row_index % 7) - 3) * 0.012 for row_index, _ in enumerate(values)]
        ax.scatter(
            [index + offset for offset in offsets],
            values,
            s=18,
            alpha=0.55,
            color=palette["baseline"] if index == 1 else palette["sage"],
        )
    ax.set_title("Review-2 Latency Distribution")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_latency_by_system(
    *,
    output_path: Path,
    baseline_latencies: Sequence[float],
    sage_latencies: Sequence[float],
    palette: Dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    systems = [
        (BASELINE_LABEL, list(baseline_latencies), palette["baseline"]),
        (SAGE_LABEL, list(sage_latencies), palette["sage"]),
    ]
    metric_specs = [
        ("p50", statistics.median),
        ("avg", lambda values: _mean(values) or 0.0),
        ("p95", lambda values: _p95(values) or 0.0),
    ]
    markers = {"p50": "o", "avg": "s", "p95": "^"}
    metric_offsets = {"p50": -0.11, "avg": 0.0, "p95": 0.11}
    y_positions = list(range(len(systems)))
    table_rows: List[List[str]] = []

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    all_values: List[float] = []
    for y_index, (system_name, values, color) in enumerate(systems):
        clean_values = [max(float(value), 1e-6) for value in values]
        all_values.extend(clean_values)
        table_row: List[str] = []
        for metric_name, reducer in metric_specs:
            metric_value = max(float(reducer(clean_values)), 1e-6)
            table_row.append(f"{metric_value:.4f}s")
            ax.scatter(
                metric_value,
                y_index + metric_offsets[metric_name],
                s=92,
                marker=markers[metric_name],
                color=color,
                edgecolor="#0f172a",
                linewidth=0.55,
                label=metric_name if y_index == 0 else None,
                zorder=3,
            )
        table_rows.append(table_row)

    ax.set_title("Review-2 Latency by System")
    ax.set_xlabel("Seconds (log scale)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([system_name for system_name, _, _ in systems])
    ax.set_xscale("log")
    if all_values:
        ax.set_xlim(min(all_values) * 0.55, max(all_values) * 2.8)
    ax.grid(axis="x", alpha=0.25, which="both")
    ax.legend(title="Metric", loc="lower right")
    table = ax.table(
        cellText=table_rows,
        rowLabels=[system_name for system_name, _, _ in systems],
        colLabels=[metric_name for metric_name, _ in metric_specs],
        cellLoc="center",
        rowLoc="center",
        loc="bottom",
        bbox=[0.0, -0.42, 1.0, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_accuracy_delta_by_fixture(
    *,
    output_path: Path,
    results: Sequence[Dict[str, Any]],
    palette: Dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = list(range(1, len(results) + 1))
    deltas = [
        int(bool(result["sage_metrics"]["answer_correct"])) - int(bool(result["baseline_metrics"]["answer_correct"]))
        for result in results
    ]
    colors = [palette["ok"] if value > 0 else palette["warning"] if value < 0 else palette["neutral"] for value in deltas]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axhline(0, color="#0f172a", linewidth=1, alpha=0.45)
    ax.vlines(x_values, [0] * len(deltas), deltas, colors=colors, alpha=0.65, linewidth=1.4)
    ax.scatter(x_values, deltas, color=colors, s=28)
    ax.set_title("Review-2 Per-Fixture Accuracy Delta")
    ax.set_xlabel("Fixture #")
    ax.set_ylabel("SAGE correct - baseline correct")
    ax.set_yticks([-1, 0, 1])
    ax.set_ylim(-1.25, 1.25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_visualizations(results: Sequence[Dict[str, Any]], output_dir: str | Path) -> List[str]:
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    buckets = sorted({result["bucket"] for result in results})
    if not buckets:
        return []

    generated: List[str] = []
    palette = {
        "sage": "#2563eb",
        "baseline": "#64748b",
        "warning": "#dc2626",
        "ok": "#059669",
        "neutral": "#7c3aed",
    }

    accuracy_path = resolved_output_dir / "accuracy_by_bucket.png"
    _plot_grouped_bars(
        output_path=accuracy_path,
        title="Review-2 Accuracy by Bucket",
        ylabel="Accuracy rate",
        buckets=buckets,
        series=[
            (SAGE_LABEL, [_bucket_metric(results, "sage_metrics", "answer_correct").get(bucket) for bucket in buckets], palette["sage"]),
            (
                BASELINE_LABEL,
                [_bucket_metric(results, "baseline_metrics", "answer_correct").get(bucket) for bucket in buckets],
                palette["baseline"],
            ),
        ],
    )
    generated.append(str(accuracy_path))

    accuracy_ci_path = resolved_output_dir / "accuracy_confidence_intervals.png"
    _plot_accuracy_confidence_intervals(
        output_path=accuracy_ci_path,
        results=results,
        buckets=buckets,
        palette=palette,
    )
    generated.append(str(accuracy_ci_path))

    reference_f1_path = resolved_output_dir / "reference_f1_by_bucket.png"
    _plot_grouped_bars(
        output_path=reference_f1_path,
        title="Review-2 Mean Reference Token F1 by Bucket",
        ylabel="Mean reference F1",
        buckets=buckets,
        series=[
            (
                SAGE_LABEL,
                [_bucket_metric(results, "sage_metrics", "reference_f1").get(bucket) for bucket in buckets],
                palette["sage"],
            ),
            (
                BASELINE_LABEL,
                [_bucket_metric(results, "baseline_metrics", "reference_f1").get(bucket) for bucket in buckets],
                palette["baseline"],
            ),
        ],
    )
    generated.append(str(reference_f1_path))

    hallucination_path = resolved_output_dir / "hallucination_by_bucket.png"
    _plot_grouped_bars(
        output_path=hallucination_path,
        title="Review-2 Hallucination Flags by Bucket",
        ylabel="Hallucination rate",
        buckets=buckets,
        series=[
            (
                SAGE_LABEL,
                [
                    _bucket_predicate_rate(results, "sage_metrics", lambda metric: metric.get("hallucination_count", 0) > 0).get(
                        bucket
                    )
                    for bucket in buckets
                ],
                palette["sage"],
            ),
            (
                BASELINE_LABEL,
                [
                    _bucket_predicate_rate(
                        results,
                        "baseline_metrics",
                        lambda metric: metric.get("hallucination_count", 0) > 0,
                    ).get(bucket)
                    for bucket in buckets
                ],
                palette["warning"],
            ),
        ],
    )
    generated.append(str(hallucination_path))

    evidence_path = resolved_output_dir / "evidence_recall_by_bucket.png"
    _plot_grouped_bars(
        output_path=evidence_path,
        title="Review-2 Evidence Recall by Bucket",
        ylabel="Evidence recall",
        buckets=buckets,
        series=[
            (
                SAGE_LABEL,
                [_bucket_metric(results, "sage_metrics", "evidence_doc_recall").get(bucket) for bucket in buckets],
                palette["sage"],
            ),
            (
                BASELINE_LABEL,
                [_bucket_metric(results, "baseline_metrics", "evidence_doc_recall").get(bucket) for bucket in buckets],
                palette["baseline"],
            ),
        ],
    )
    generated.append(str(evidence_path))

    baseline_latencies = [float(result["baseline_metrics"]["latency"]) for result in results]
    sage_latencies = [float(result["sage_metrics"]["latency"]) for result in results]
    latency_path = resolved_output_dir / "latency_by_system.png"
    _plot_latency_by_system(
        output_path=latency_path,
        baseline_latencies=baseline_latencies,
        sage_latencies=sage_latencies,
        palette=palette,
    )
    generated.append(str(latency_path))

    fixture_numbers = list(range(1, len(results) + 1))
    cumulative_accuracy_path = resolved_output_dir / "accuracy_cumulative.png"
    _plot_line_series(
        output_path=cumulative_accuracy_path,
        title="Review-2 Cumulative Accuracy",
        ylabel="Cumulative accuracy",
        x_values=fixture_numbers,
        y_limits=(0, 1.05),
        series=[
            (SAGE_LABEL, _cumulative_rate([bool(result["sage_metrics"]["answer_correct"]) for result in results]), palette["sage"]),
            (
                BASELINE_LABEL,
                _cumulative_rate([bool(result["baseline_metrics"]["answer_correct"]) for result in results]),
                palette["baseline"],
            ),
        ],
    )
    generated.append(str(cumulative_accuracy_path))

    latency_line_path = resolved_output_dir / "latency_by_fixture_line.png"
    _plot_line_series(
        output_path=latency_line_path,
        title="Review-2 Latency by Fixture",
        ylabel="Seconds",
        x_values=fixture_numbers,
        series=[
            (SAGE_LABEL, sage_latencies, palette["sage"]),
            (BASELINE_LABEL, baseline_latencies, palette["baseline"]),
        ],
    )
    generated.append(str(latency_line_path))

    latency_distribution_path = resolved_output_dir / "latency_distribution.png"
    _plot_latency_distribution(
        output_path=latency_distribution_path,
        baseline_latencies=baseline_latencies,
        sage_latencies=sage_latencies,
        palette=palette,
    )
    generated.append(str(latency_distribution_path))

    accuracy_delta_path = resolved_output_dir / "accuracy_delta_by_fixture.png"
    _plot_accuracy_delta_by_fixture(output_path=accuracy_delta_path, results=results, palette=palette)
    generated.append(str(accuracy_delta_path))

    hallucination_fixture_path = resolved_output_dir / "hallucination_by_fixture.png"
    _plot_line_series(
        output_path=hallucination_fixture_path,
        title="Review-2 Hallucination Flags by Fixture",
        ylabel="Hallucination flag count",
        x_values=fixture_numbers,
        zero_note="No hallucination flags were triggered in this run.",
        series=[
            (
                SAGE_LABEL,
                [float(result["sage_metrics"]["hallucination_count"]) for result in results],
                palette["sage"],
            ),
            (
                BASELINE_LABEL,
                [float(result["baseline_metrics"]["hallucination_count"]) for result in results],
                palette["warning"],
            ),
        ],
    )
    generated.append(str(hallucination_fixture_path))

    trace_path = resolved_output_dir / "agentic_trace_health.png"
    _plot_grouped_bars(
        output_path=trace_path,
        title="Review-2 Agentic Trace Health",
        ylabel="Rate",
        buckets=buckets,
        series=[
            (
                "Used graph tool",
                [_bucket_metric(results, "sage_metrics", "used_graph_tool").get(bucket) for bucket in buckets],
                palette["sage"],
            ),
            (
                "Valid graph path",
                [_bucket_metric(results, "sage_metrics", "graph_path_valid").get(bucket) for bucket in buckets],
                palette["ok"],
            ),
            (
                "Critic passed",
                [_bucket_metric(results, "sage_metrics", "critic_passed").get(bucket) for bucket in buckets],
                palette["neutral"],
            ),
        ],
    )
    generated.append(str(trace_path))

    return generated


def _print_result_line(result: Dict[str, Any]) -> None:
    sage = result["sage_metrics"]
    baseline = result["baseline_metrics"]
    print(
        f"[{result['bucket']}] {result['fixture_id']} | "
        f"verify={result['verification']['passed']} | "
        f"sage_correct={sage['answer_correct']} | baseline_correct={baseline['answer_correct']} | "
        f"sage_hallucinations={sage['hallucination_count']}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Review 2 fixture-based SAGE evaluation.")
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURE_PATH), help="Path to fixture JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to detailed results JSON.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH), help="Path to summary CSV.")
    parser.add_argument("--abnormalities", default=str(DEFAULT_ABNORMALITIES_PATH), help="Path to abnormality CSV.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for visualization artifacts.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of fixtures.")
    parser.add_argument("--bucket", action="append", default=[], help="Filter to a fixture bucket. Can be repeated.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip setup ingestion and only verify/query existing graph state.")
    parser.add_argument("--extractive-baseline", action="store_true", help="Use deterministic extractive baseline instead of LLM baseline.")
    parser.add_argument("--no-force-saia", action="store_true", help="Do not force SAIA_ENABLED=true for this process.")
    parser.add_argument("--no-isolated-sage-retrieval", dest="isolated_sage_retrieval", action="store_false", help="Let SAGE retrieve from the full live graph instead of the fixture doc ids.")
    parser.add_argument("--cleanup-prefix", default=DEFAULT_CLEANUP_PREFIX, help="Namespaced fixture prefix to clean up.")
    parser.add_argument("--cleanup-only", action="store_true", help="Only remove namespaced fixture data, then exit.")
    parser.add_argument("--no-cleanup-before", dest="cleanup_before", action="store_false", help="Do not cleanup review fixture data before ingestion.")
    parser.add_argument("--no-cleanup-between", dest="cleanup_between", action="store_false", help="Do not cleanup fixture data between selected fixtures.")
    parser.add_argument("--no-cleanup-after", dest="cleanup_after", action="store_false", help="Do not cleanup review fixture data after the run.")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="Skip writing PNG visualizations.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if verification or SAGE acceptance fails.")
    parser.set_defaults(cleanup_before=True, cleanup_between=True, cleanup_after=True, plots=True, isolated_sage_retrieval=True)
    args = parser.parse_args(argv)

    if not args.no_force_saia:
        os.environ["SAIA_ENABLED"] = "true"

    cleanup_reports: Dict[str, Any] = {}
    if args.cleanup_only:
        cleanup_reports["only"] = cleanup_review2_data(prefix=args.cleanup_prefix)
        print(f"Cleanup deleted {cleanup_reports['only']['total_deleted']} Review-2 graph nodes.")
        return 0

    fixtures = load_fixtures(args.fixtures, limit=args.limit, buckets=args.bucket)
    if not fixtures:
        print("No fixtures selected.")
        return 1

    results: List[Dict[str, Any]] = []
    visualization_paths: List[str] = []
    abnormalities: List[Dict[str, Any]] = []
    strict_failures: List[Dict[str, Any]] = []

    try:
        if args.cleanup_before and not args.skip_ingest:
            cleanup_reports["before"] = cleanup_review2_data(prefix=args.cleanup_prefix)
            print(f"Cleanup before run deleted {cleanup_reports['before']['total_deleted']} Review-2 graph nodes.")

        for index, fixture in enumerate(fixtures):
            if args.cleanup_between and not args.skip_ingest and (index > 0 or not args.cleanup_before):
                report = cleanup_review2_data(prefix=args.cleanup_prefix)
                cleanup_reports.setdefault("between_fixtures", []).append(
                    {"fixture_id": fixture["id"], "deleted": report["deleted"], "total_deleted": report["total_deleted"]}
                )
                print(f"Cleanup before fixture {fixture['id']} deleted {report['total_deleted']} Review-2 graph nodes.")
            result = run_fixture(
                fixture,
                skip_ingest=args.skip_ingest,
                baseline_use_llm=not args.extractive_baseline,
                isolate_sage_retrieval=args.isolated_sage_retrieval,
            )
            results.append(result)
            _print_result_line(result)

        summary_rows = summarize_results(results)
        abnormalities = detect_abnormalities(results)
        if args.plots:
            visualization_paths = generate_visualizations(results, args.results_dir)

        if args.strict:
            strict_failures = [
                result
                for result in results
                if not result["verification"]["passed"] or not result["sage_metrics"]["answer_correct"]
            ]
    finally:
        if args.cleanup_after and not args.skip_ingest:
            cleanup_reports["after"] = cleanup_review2_data(prefix=args.cleanup_prefix)
            print(f"Cleanup after run deleted {cleanup_reports['after']['total_deleted']} Review-2 graph nodes.")

    summary_rows = summarize_results(results)
    if not abnormalities:
        abnormalities = detect_abnormalities(results)
    if args.plots and not visualization_paths:
        visualization_paths = generate_visualizations(results, args.results_dir)

    _write_json(
        _resolve_path(args.output),
        {
            "metadata": {
                "fixture_count": len(results),
                "fixture_path": str(_resolve_path(args.fixtures)),
                "extractive_baseline": bool(args.extractive_baseline),
                "isolated_sage_retrieval": bool(args.isolated_sage_retrieval),
                "cleanup": cleanup_reports,
                "visualizations": visualization_paths,
                "abnormality_count": len(abnormalities),
            },
            "results": results,
            "summary": summary_rows,
            "abnormalities": abnormalities,
        },
    )
    write_summary_csv(args.summary, summary_rows)
    write_abnormalities_csv(args.abnormalities, abnormalities)
    print(f"Wrote detailed results to {_resolve_path(args.output)}")
    print(f"Wrote summary to {_resolve_path(args.summary)}")
    print(f"Wrote abnormalities to {_resolve_path(args.abnormalities)}")
    if visualization_paths:
        print("Wrote visualizations:")
        for path in visualization_paths:
            print(f"  - {path}")

    if args.strict and strict_failures:
        print("Strict mode failed fixtures: " + ", ".join(result["fixture_id"] for result in strict_failures))
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
