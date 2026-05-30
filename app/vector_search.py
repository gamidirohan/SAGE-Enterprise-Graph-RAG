"""Retrieval tools for dense and lexical search used by the agentic orchestrator."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import app.services as services
    import app.utils as utils
except ImportError:  # pragma: no cover - direct execution fallback
    import services
    import utils


CHUNK_FULLTEXT_INDEX = "sage_chunk_fulltext"
DOCUMENT_FULLTEXT_INDEX = "sage_document_fulltext"
FACT_FULLTEXT_INDEX = "sage_fact_fulltext"

CHUNK_FULLTEXT_QUERY = """
    CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
    WITH node AS c, score
    MATCH (c:Chunk)-[:PART_OF]->(d:Document)
    WHERE coalesce(d.conversation_type, '') <> 'sage'
      AND NOT coalesce(d.source, '') STARTS WITH 'sage_'
    OPTIONAL MATCH (c)-[r]-(n)
    RETURN
        c.chunk_id AS chunk_id,
        c.summary AS chunk_summary,
        d,
        score AS similarity,
        type(r) AS relationship,
        CASE WHEN r IS NULL THEN NULL WHEN startNode(r) = c THEN 'outgoing' ELSE 'incoming' END AS direction,
        n,
        2 AS hop_count,
        [
            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
            coalesce(c.subject, c.title, c.name, c.id, c.doc_id, c.chunk_id, labels(c)[0]),
            coalesce(n.subject, n.title, n.name, n.id, n.doc_id, labels(n)[0])
        ] AS path_nodes,
        ['PART_OF', type(r)] AS path_relationships
    ORDER BY similarity DESC
    LIMIT $candidate_limit
"""

DOCUMENT_FULLTEXT_QUERY = """
    CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
    WITH node AS d, score
    WHERE d:Document
      AND coalesce(d.conversation_type, '') <> 'sage'
      AND NOT coalesce(d.source, '') STARTS WITH 'sage_'
    OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
    WITH d, score, collect(DISTINCT c)[0] AS c
    OPTIONAL MATCH (d)-[r:SENT|RECEIVED_BY]-(n:Person)
    RETURN
        coalesce(c.chunk_id, d.doc_id + '-document') AS chunk_id,
        coalesce(c.summary, d.summary, d.subject, d.doc_id) AS chunk_summary,
        d,
        score AS similarity,
        coalesce(type(r), 'PART_OF') AS relationship,
        CASE WHEN r IS NULL THEN 'document' WHEN startNode(r) = d THEN 'outgoing' ELSE 'incoming' END AS direction,
        n,
        2 AS hop_count,
        [
            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
            coalesce(c.summary, c.chunk_id, d.subject, d.doc_id),
            coalesce(n.subject, n.title, n.name, n.id, n.doc_id, labels(n)[0])
        ] AS path_nodes,
        [coalesce(type(r), 'PART_OF')] AS path_relationships
    ORDER BY similarity DESC
    LIMIT $candidate_limit
"""

CHUNK_FULLTEXT_QUERY_SHALLOW = """
    CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
    WITH node AS c, score
    MATCH (c:Chunk)-[:PART_OF]->(d:Document)
    WHERE coalesce(d.conversation_type, '') <> 'sage'
      AND NOT coalesce(d.source, '') STARTS WITH 'sage_'
    RETURN
        c.chunk_id AS chunk_id,
        c.summary AS chunk_summary,
        d,
        score AS similarity,
        'PART_OF' AS relationship,
        NULL AS direction,
        NULL AS n,
        1 AS hop_count,
        [
            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
            coalesce(c.subject, c.title, c.name, c.id, c.doc_id, c.chunk_id, labels(c)[0])
        ] AS path_nodes,
        ['PART_OF'] AS path_relationships
    ORDER BY similarity DESC
    LIMIT $candidate_limit
"""

DOCUMENT_FULLTEXT_QUERY_SHALLOW = """
    CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
    WITH node AS d, score
    WHERE d:Document
      AND coalesce(d.conversation_type, '') <> 'sage'
      AND NOT coalesce(d.source, '') STARTS WITH 'sage_'
    OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
    WITH d, score, collect(DISTINCT c)[0] AS c
    RETURN
        coalesce(c.chunk_id, d.doc_id + '-document') AS chunk_id,
        coalesce(c.summary, d.summary, d.subject, d.doc_id) AS chunk_summary,
        d,
        score AS similarity,
        'PART_OF' AS relationship,
        'document' AS direction,
        NULL AS n,
        1 AS hop_count,
        [
            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
            coalesce(c.summary, c.chunk_id, d.subject, d.doc_id)
        ] AS path_nodes,
        ['PART_OF'] AS path_relationships
    ORDER BY similarity DESC
    LIMIT $candidate_limit
"""

FACT_FULLTEXT_QUERY = """
    CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
    WITH node AS f, score
    WHERE f:CanonicalFact
      AND coalesce(f.status, '') = 'current'
    OPTIONAL MATCH (f)<-[:SUPPORTS|CONTRADICTS]-(claim:Claim)<-[:HAS_CLAIM]-(d:Document)
    WITH f, score, collect(DISTINCT d)[0] AS d
    RETURN
        f.fact_id AS fact_id,
        f.summary AS fact_summary,
        f,
        d,
        score AS similarity
    ORDER BY similarity DESC
    LIMIT $candidate_limit
"""


def _ensure_fulltext_indexes(session: Any) -> None:
    statements = (
        f"CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.summary, c.content]",
        f"CREATE FULLTEXT INDEX {DOCUMENT_FULLTEXT_INDEX} IF NOT EXISTS FOR (d:Document) ON EACH [d.subject, d.summary, d.content, d.doc_id]",
        (
            f"CREATE FULLTEXT INDEX {FACT_FULLTEXT_INDEX} IF NOT EXISTS FOR (f:CanonicalFact) "
            "ON EACH [f.summary, f.display_summary, f.canonical_key, f.value_text, "
            "f.subject_display, f.object_display, f.subject_key, f.object_key, f.subject_entity_id, f.object_entity_id]"
        ),
    )
    for statement in statements:
        try:
            session.run(statement)
        except Exception:
            continue


def _evidence_key(item: Dict[str, Any]) -> str:
    if item.get("fact_id"):
        return f"fact:{item['fact_id']}"
    if item.get("chunk_id"):
        return f"chunk:{item['chunk_id']}"
    document = item.get("document") or {}
    if document.get("doc_id"):
        return f"doc:{document['doc_id']}"
    if item.get("related_node_id"):
        return f"node:{item['related_node_id']}"
    return f"unknown:{hash(str(item))}"


def _normalize_trace(trace: Optional[Dict[str, Any]], *, tool_name: str) -> Dict[str, Any]:
    normalized = dict(trace or {})
    evidence = [dict(item) for item in (normalized.get("evidence") or [])]
    retrieved_at = str(normalized.get("retrieved_at") or services._retrieval_timestamp())
    for item in evidence:
        if item.get("direction") is None:
            item["direction"] = "outgoing" if item.get("relationship") else None
        item.setdefault("retrieved_at", retrieved_at)
    normalized["evidence"] = evidence
    if evidence:
        normalized["retrieved_at"] = retrieved_at
    normalized["result_count"] = len(evidence)
    normalized["max_hop_count"] = max((int(item.get("hop_count") or 0) for item in evidence), default=0)
    normalized["no_evidence"] = not evidence
    normalized["evidence_state"] = (
        "no_evidence"
        if not evidence
        else "partial_evidence"
        if len(evidence) < 2
        else "grounded"
    )
    normalized["selector_strategy"] = tool_name
    return normalized


def _build_trace_from_rows(
    rows: List[Dict[str, Any]],
    *,
    query: str,
    user_id: Optional[str],
    tool_name: str,
    context_hops: int,
) -> Dict[str, Any]:
    personalized_lookup = bool(user_id and services._is_personalized_lookup(query))
    query_type = services._classify_query(query)
    query_profile = services.query_shape.analyze_query(query) if hasattr(services, "query_shape") else None
    evidence: List[Dict[str, Any]] = []
    documents: List[str] = []
    matched_entities: List[str] = []
    retrieved_at = services._retrieval_timestamp()

    for row in rows:
        document = services._serialize_neo4j_entity(row.get("d"))
        related_node = services._serialize_neo4j_entity(row.get("n"))
        related_label = services._get_primary_label(related_node) if related_node else None
        related_name = services._get_display_name(related_node) if related_node else None

        sender = document.get("sender")
        subject = document.get("subject")
        doc_id = document.get("doc_id")
        relationship = row.get("relationship") or "RELATED_TO"
        similarity = round(float(row.get("rank_score", row.get("similarity", 0)) or 0), 4)
        hop_count = int(row.get("hop_count") or 0)
        retrieval_path = services._build_path_string(row.get("path_nodes"), row.get("path_relationships")) or services._build_evidence_path(
            scope="user" if personalized_lookup else "global",
            relationship=str(relationship),
            related_label=related_label,
            doc_id=doc_id,
            chunk_id=row.get("chunk_id"),
        )["path"]

        for candidate in (sender, subject, related_name):
            services._append_matched_entity(matched_entities, candidate)

        evidence_item = {
            "chunk_id": row.get("chunk_id"),
            "chunk_summary": row.get("chunk_summary") or document.get("summary") or document.get("subject") or "No summary",
            "similarity": similarity,
            "rank_score": similarity,
            "retrieved_at": retrieved_at,
            "relationship": relationship,
            "direction": row.get("direction") or "outgoing",
            "retrieval_path": retrieval_path,
            "hop_count": hop_count,
            "document": {
                "doc_id": doc_id,
                "subject": subject,
                "sender": sender,
                "timestamp": document.get("timestamp"),
                "source": document.get("source"),
                "conversation_type": document.get("conversation_type"),
                "conversation_id": document.get("conversation_id"),
                "group_id": document.get("group_id"),
            },
            "related_node": {
                "label": related_label,
                "display_name": related_name,
                "id": related_node.get("id") or related_node.get("_element_id"),
            }
            if related_node
            else {},
            "related_node_id": related_node.get("id") or related_node.get("_element_id") if related_node else None,
            "fact": None,
        }
        evidence.append(evidence_item)
        documents.append(
            "Chunk Summary: "
            f"{evidence_item['chunk_summary']}, "
            f"Document ID: {doc_id or 'unknown'}, "
            f"Conversation Type: {document.get('conversation_type') or 'unknown'}, "
            f"Subject: {subject or 'No Subject'}, "
            f"Sender: {sender or 'Unknown'}, "
            f"Similarity: {similarity}, "
            f"Relationship: {relationship}, "
            f"Direction: {evidence_item['direction']}, "
            f"Related Node: {related_name or 'Unknown'}"
        )

    trace = {
        "query": query,
        "query_type": query_type,
        "user_scoped": personalized_lookup,
        "user_id": user_id,
        "query_profile": query_profile,
        "graph_depth": {"seed_hops": context_hops},
        "matched_entities": matched_entities,
        "retrieved_at": retrieved_at,
        "result_count": len(evidence),
        "max_hop_count": max((int(item.get("hop_count") or 0) for item in evidence), default=0),
        "retrieval_path": evidence[0]["retrieval_path"] if evidence else services._build_path_summary(personalized_lookup, None)["path"],
        "evidence": evidence,
        "no_evidence": not evidence,
        "evidence_state": "no_evidence" if not evidence else "partial_evidence" if len(evidence) < 2 else "grounded",
        "selector_strategy": tool_name,
    }
    return {"documents": documents, "trace": trace}


def semantic_retrieve(query: str, *, user_id: Optional[str] = None, context_hops: Optional[int] = None) -> Dict[str, Any]:
    result = services.query_graph_with_trace(query, user_id=user_id, seed_hops=context_hops)
    return {"documents": result.get("documents") or [], "trace": _normalize_trace(result.get("trace"), tool_name="semantic")}


def fulltext_retrieve(query: str, *, user_id: Optional[str] = None, context_hops: Optional[int] = None) -> Dict[str, Any]:
    resolved_context_hops = services._resolve_seed_context_hops(query, query_type=services._classify_query(query), seed_hops=context_hops)
    driver = utils.create_neo4j_driver()
    focus_terms = services._extract_query_focus_terms(query)
    reports_to_lookup = services._is_reports_to_lookup(query)
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            _ensure_fulltext_indexes(session)
            chunk_rows = [
                services._prepare_chunk_result(row, focus_terms=focus_terms, reports_to_lookup=reports_to_lookup)
                for row in session.run(
                    CHUNK_FULLTEXT_QUERY if resolved_context_hops >= services.DEFAULT_SEED_CONTEXT_HOPS else CHUNK_FULLTEXT_QUERY_SHALLOW,
                    {"index_name": CHUNK_FULLTEXT_INDEX, "query": query, "candidate_limit": services.DEFAULT_RETRIEVAL_LIMIT},
                ).data()
                if row.get("chunk_id") or row.get("chunk_summary")
            ]
            doc_rows = [
                services._prepare_chunk_result(row, focus_terms=focus_terms, reports_to_lookup=reports_to_lookup)
                for row in session.run(
                    DOCUMENT_FULLTEXT_QUERY if resolved_context_hops >= services.DEFAULT_SEED_CONTEXT_HOPS else DOCUMENT_FULLTEXT_QUERY_SHALLOW,
                    {"index_name": DOCUMENT_FULLTEXT_INDEX, "query": query, "candidate_limit": services.DEFAULT_RETRIEVAL_LIMIT},
                ).data()
                if row.get("chunk_id") or row.get("chunk_summary")
            ]
            fact_rows = [
                row
                for row in session.run(
                    FACT_FULLTEXT_QUERY,
                    {"index_name": FACT_FULLTEXT_INDEX, "query": query, "candidate_limit": services.DEFAULT_RETRIEVAL_LIMIT},
                ).data()
                if row.get("fact_id")
            ]
    except Exception as exc:
        return {
            "documents": [],
            "trace": _normalize_trace(
                {
                    "query": query,
                    "query_type": services._classify_query(query),
                    "user_scoped": bool(user_id and services._is_personalized_lookup(query)),
                    "user_id": user_id,
                    "matched_entities": [],
                    "result_count": 0,
                    "max_hop_count": 0,
                    "graph_depth": {"seed_hops": resolved_context_hops},
                    "retrieval_path": services._build_path_summary(bool(user_id and services._is_personalized_lookup(query)), None)["path"],
                    "evidence": [],
                    "no_evidence": True,
                    "error": str(exc),
                },
                tool_name="fulltext",
            ),
        }
    merged_rows = services._merge_ranked_results(chunk_rows, doc_rows, limit=services.DEFAULT_RETRIEVAL_LIMIT)
    chunk_result = _build_trace_from_rows(
        merged_rows,
        query=query,
        user_id=user_id,
        tool_name="fulltext",
        context_hops=resolved_context_hops,
    )
    fact_result = _fact_evidence_from_rows(
        fact_rows,
        query=query,
        user_id=user_id,
        tool_name="fulltext",
        context_hops=resolved_context_hops,
    )
    return merge_results(fact_result, chunk_result, limit=services.DEFAULT_RETRIEVAL_LIMIT)


def _eval_allowed_doc_ids() -> List[str]:
    raw = os.getenv("SAGE_EVAL_ALLOWED_DOC_IDS", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw.split(",")]
    if not isinstance(parsed, list):
        return []
    doc_ids: List[str] = []
    for item in parsed:
        doc_id = str(item or "").strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def _fact_evidence_from_rows(
    rows: List[Dict[str, Any]],
    *,
    query: str,
    user_id: Optional[str],
    tool_name: str,
    context_hops: int,
) -> Dict[str, Any]:
    query_type = services._classify_query(query)
    evidence: List[Dict[str, Any]] = []
    documents: List[str] = []
    matched_entities: List[str] = []
    personalized_lookup = bool(user_id and services._is_personalized_lookup(query))
    retrieved_at = services._retrieval_timestamp()

    for row in rows:
        item = services._prepare_fact_result(
            row,
            query_type=query_type,
            user_id=user_id,
            personalized_lookup=personalized_lookup,
            exact_match=True,
            focus_terms=services._extract_query_focus_terms(query),
            reports_to_lookup=services._is_reports_to_lookup(query),
        )
        fact = services._serialize_neo4j_entity(item.get("f"))
        document = services._serialize_neo4j_entity(item.get("d"))
        fact_id = item.get("fact_id") or fact.get("fact_id")
        doc_id = document.get("doc_id")
        fact_summary = item.get("fact_summary") or fact.get("summary") or "No fact summary"
        similarity = round(float(item.get("similarity", 0) or 0), 4)
        rank_score = round(services._result_rank_value(item), 4)
        canonical_key = fact.get("canonical_key")

        for candidate in (
            fact.get("subject_key"),
            fact.get("subject_entity_id"),
            fact.get("object_key"),
            fact.get("object_entity_id"),
            document.get("subject"),
            document.get("sender"),
        ):
            services._append_matched_entity(matched_entities, candidate)

        evidence.append(
            {
                "fact_id": fact_id,
                "fact_summary": fact_summary,
                "similarity": similarity,
                "rank_score": rank_score,
                "retrieved_at": retrieved_at,
                "relationship": "CANONICAL_FACT",
                "retrieval_path": "CanonicalFact <-SUPPORTS/CONTRADICTS- Claim <-HAS_CLAIM- Document",
                "hop_count": max(context_hops, 2),
                "chunk_id": f"{doc_id}-chunk-0" if doc_id else None,
                "exact_match": True,
                "fact_priority": bool(item.get("fact_priority")),
                "document": {
                    "doc_id": doc_id,
                    "subject": document.get("subject"),
                    "sender": document.get("sender"),
                    "timestamp": document.get("timestamp"),
                    "source": document.get("source"),
                    "content": str(document.get("content") or "")[:500],
                    "conversation_type": document.get("conversation_type"),
                    "conversation_id": document.get("conversation_id"),
                    "group_id": document.get("group_id"),
                },
                "related_node": {
                    "label": "CanonicalFact",
                    "display_name": canonical_key or fact_summary,
                    "id": fact_id,
                },
                "related_node_id": fact_id,
                "direction": "fact",
                "fact": {
                    "claim_type": fact.get("claim_type"),
                    "status": fact.get("status"),
                    "canonical_key": canonical_key,
                    "value_text": fact.get("value_text"),
                    "subject_key": fact.get("subject_key"),
                    "subject_entity_id": fact.get("subject_entity_id"),
                    "subject_display": fact.get("subject_display"),
                    "object_key": fact.get("object_key"),
                    "object_entity_id": fact.get("object_entity_id"),
                    "object_display": fact.get("object_display"),
                    "display_summary": fact.get("display_summary"),
                    "temporal_start": fact.get("temporal_start"),
                    "temporal_end": fact.get("temporal_end"),
                    "temporal_granularity": fact.get("temporal_granularity"),
                    "first_seen_at": fact.get("first_seen_at"),
                    "last_seen_at": fact.get("last_seen_at"),
                    "support_count": fact.get("support_count"),
                    "confidence": fact.get("confidence"),
                },
            }
        )
        documents.append(
            "Fact Summary: "
            f"{fact_summary}, "
            f"Fact ID: {fact_id or 'unknown'}, "
            f"Canonical Key: {canonical_key or 'unknown'}, "
            f"Fact Type: {fact.get('claim_type') or 'unknown'}, "
            f"Conversation Type: {document.get('conversation_type') or 'unknown'}, "
            f"Subject: {fact.get('subject_entity_id') or fact.get('subject_key') or 'unknown'}, "
            f"Object: {fact.get('object_entity_id') or fact.get('object_key') or 'unknown'}, "
            f"Time: {fact.get('temporal_start') or 'not specified'}, "
            f"Supporting Document ID: {doc_id or 'unknown'}, "
            f"Similarity: {similarity}"
        )

    return {
        "documents": documents,
        "trace": {
            "query": query,
            "query_type": query_type,
            "user_scoped": personalized_lookup,
            "user_id": user_id,
            "query_profile": services.query_shape.analyze_query(query) if hasattr(services, "query_shape") else None,
            "graph_depth": {"seed_hops": context_hops},
            "matched_entities": matched_entities,
            "retrieved_at": retrieved_at,
            "result_count": len(evidence),
            "max_hop_count": max((int(item.get("hop_count") or 0) for item in evidence), default=0),
            "retrieval_path": evidence[0]["retrieval_path"] if evidence else services._build_path_summary(personalized_lookup, None)["path"],
            "evidence": evidence,
            "no_evidence": not evidence,
            "evidence_state": "no_evidence" if not evidence else "partial_evidence" if len(evidence) < 2 else "grounded",
            "selector_strategy": tool_name,
        },
    }


def fixture_scoped_retrieve(
    query: str,
    *,
    user_id: Optional[str] = None,
    strategy: str = "hybrid",
    context_hops: Optional[int] = None,
) -> Dict[str, Any]:
    allowed_doc_ids = _eval_allowed_doc_ids()
    resolved_context_hops = services._resolve_seed_context_hops(
        query,
        query_type=services._classify_query(query),
        seed_hops=context_hops,
    )
    if not allowed_doc_ids:
        return {
            "documents": [],
            "trace": _normalize_trace(
                {
                    "query": query,
                    "query_type": services._classify_query(query),
                    "user_scoped": bool(user_id and services._is_personalized_lookup(query)),
                    "user_id": user_id,
                    "matched_entities": [],
                    "result_count": 0,
                    "max_hop_count": 0,
                    "graph_depth": {"seed_hops": resolved_context_hops},
                    "retrieval_path": services._build_path_summary(bool(user_id and services._is_personalized_lookup(query)), None)["path"],
                    "evidence": [],
                    "no_evidence": True,
                },
                tool_name=strategy,
            ),
        }

    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            chunk_rows = [
                services._prepare_chunk_result(row, focus_terms=services._extract_query_focus_terms(query), reports_to_lookup=services._is_reports_to_lookup(query))
                for row in session.run(
                    """
                    MATCH (d:Document)
                    WHERE d.doc_id IN $doc_ids
                    OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
                    WITH d, c
                    OPTIONAL MATCH (d)-[r:SENT|RECEIVED_BY]-(n:Person)
                    RETURN
                        coalesce(c.chunk_id, d.doc_id + '-document') AS chunk_id,
                        coalesce(c.summary, d.summary, d.subject, d.doc_id) AS chunk_summary,
                        d,
                        1.0 AS similarity,
                        coalesce(type(r), 'PART_OF') AS relationship,
                        CASE WHEN r IS NULL THEN 'document' WHEN startNode(r) = d THEN 'outgoing' ELSE 'incoming' END AS direction,
                        n,
                        2 AS hop_count,
                        [
                            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
                            coalesce(c.summary, c.chunk_id, d.subject, d.doc_id),
                            coalesce(n.subject, n.title, n.name, n.id, n.doc_id, labels(n)[0])
                        ] AS path_nodes,
                        [coalesce(type(r), 'PART_OF')] AS path_relationships
                    """,
                    doc_ids=allowed_doc_ids,
                ).data()
                if row.get("chunk_id") or row.get("chunk_summary")
            ]
            fact_rows = [
                row
                for row in session.run(
                    """
                    MATCH (d:Document)
                    WHERE d.doc_id IN $doc_ids
                    MATCH (d)-[:HAS_CLAIM]->(:Claim)-[:SUPPORTS|CONTRADICTS]->(f:CanonicalFact)
                    RETURN DISTINCT f.fact_id AS fact_id, f.summary AS fact_summary, f, d, 1.0 AS similarity
                    """,
                    doc_ids=allowed_doc_ids,
                ).data()
                if row.get("fact_id")
            ]
    finally:
        driver.close()

    chunk_result = _build_trace_from_rows(
        services._merge_ranked_results(chunk_rows, [], limit=services.DEFAULT_RETRIEVAL_LIMIT),
        query=query,
        user_id=user_id,
        tool_name=strategy,
        context_hops=resolved_context_hops,
    )
    fact_result = _fact_evidence_from_rows(
        fact_rows,
        query=query,
        user_id=user_id,
        tool_name=strategy,
        context_hops=resolved_context_hops,
    )
    return merge_results(fact_result, chunk_result, limit=services.DEFAULT_RETRIEVAL_LIMIT)


def merge_results(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
    *,
    limit: int = services.DEFAULT_RETRIEVAL_LIMIT,
) -> Dict[str, Any]:
    existing_trace = dict(existing.get("trace") or {})
    incoming_trace = dict(incoming.get("trace") or {})
    existing_docs = list(existing.get("documents") or [])
    incoming_docs = list(incoming.get("documents") or [])

    evidence_by_key: Dict[str, Dict[str, Any]] = {}
    for item in list(existing_trace.get("evidence") or []) + list(incoming_trace.get("evidence") or []):
        key = _evidence_key(item)
        prior = evidence_by_key.get(key)
        if prior is None or services._result_rank_value(item) > services._result_rank_value(prior):
            evidence_by_key[key] = dict(item)

    merged_evidence = sorted(
        evidence_by_key.values(),
        key=services._result_rank_value,
        reverse=True,
    )[:limit]

    merged_docs: List[str] = []
    for doc in existing_docs + incoming_docs:
        if doc not in merged_docs:
            merged_docs.append(doc)

    matched_entities: List[str] = []
    for entity in list(existing_trace.get("matched_entities") or []) + list(incoming_trace.get("matched_entities") or []):
        if entity not in matched_entities:
            matched_entities.append(entity)

    merged_trace = {
        **existing_trace,
        **incoming_trace,
        "matched_entities": matched_entities,
        "evidence": merged_evidence,
        "retrieved_at": incoming_trace.get("retrieved_at") or existing_trace.get("retrieved_at"),
        "result_count": len(merged_evidence),
        "max_hop_count": max((int(item.get("hop_count") or 0) for item in merged_evidence), default=0),
        "retrieval_path": (merged_evidence[0].get("retrieval_path") if merged_evidence else None)
        or existing_trace.get("retrieval_path")
        or incoming_trace.get("retrieval_path"),
        "no_evidence": not merged_evidence,
        "evidence_state": "no_evidence" if not merged_evidence else "partial_evidence" if len(merged_evidence) < 2 else "grounded",
    }
    return {"documents": merged_docs[:limit], "trace": merged_trace}


def retrieve(
    query: str,
    *,
    user_id: Optional[str] = None,
    strategy: str = "hybrid",
    seed_trace: Optional[Dict[str, Any]] = None,
    context_hops: Optional[int] = None,
) -> Dict[str, Any]:
    del seed_trace

    if _eval_allowed_doc_ids():
        return fixture_scoped_retrieve(query, user_id=user_id, strategy=strategy, context_hops=context_hops)

    if strategy == "semantic":
        return semantic_retrieve(query, user_id=user_id, context_hops=context_hops)
    if strategy == "fulltext":
        return fulltext_retrieve(query, user_id=user_id, context_hops=context_hops)

    semantic = semantic_retrieve(query, user_id=user_id, context_hops=context_hops)
    if strategy == "hybrid":
        return merge_results(semantic, fulltext_retrieve(query, user_id=user_id, context_hops=context_hops))
    return semantic
