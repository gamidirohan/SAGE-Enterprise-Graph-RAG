"""Retrieval tools for dense and lexical search used by the agentic orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import app.services as services
    import app.utils as utils
except ImportError:  # pragma: no cover - direct execution fallback
    import services
    import utils


CHUNK_FULLTEXT_INDEX = "sage_chunk_fulltext"
DOCUMENT_FULLTEXT_INDEX = "sage_document_fulltext"

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
    LIMIT 5
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
    LIMIT 5
"""


def _ensure_fulltext_indexes(session: Any) -> None:
    statements = (
        f"CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.summary, c.content]",
        f"CREATE FULLTEXT INDEX {DOCUMENT_FULLTEXT_INDEX} IF NOT EXISTS FOR (d:Document) ON EACH [d.subject, d.summary, d.content, d.doc_id]",
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
    for item in evidence:
        if item.get("direction") is None:
            item["direction"] = "outgoing" if item.get("relationship") else None
    normalized["evidence"] = evidence
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
) -> Dict[str, Any]:
    personalized_lookup = bool(user_id and services._contains_first_person(query))
    query_type = services._classify_query(query)
    evidence: List[Dict[str, Any]] = []
    documents: List[str] = []
    matched_entities: List[str] = []

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
        "matched_entities": matched_entities,
        "result_count": len(evidence),
        "max_hop_count": max((int(item.get("hop_count") or 0) for item in evidence), default=0),
        "retrieval_path": evidence[0]["retrieval_path"] if evidence else services._build_path_summary(personalized_lookup, None)["path"],
        "evidence": evidence,
        "no_evidence": not evidence,
        "evidence_state": "no_evidence" if not evidence else "partial_evidence" if len(evidence) < 2 else "grounded",
        "selector_strategy": tool_name,
    }
    return {"documents": documents, "trace": trace}


def semantic_retrieve(query: str, *, user_id: Optional[str] = None) -> Dict[str, Any]:
    result = services.query_graph_with_trace(query, user_id=user_id)
    return {"documents": result.get("documents") or [], "trace": _normalize_trace(result.get("trace"), tool_name="semantic")}


def fulltext_retrieve(query: str, *, user_id: Optional[str] = None) -> Dict[str, Any]:
    driver = utils.create_neo4j_driver()
    focus_terms = services._extract_query_focus_terms(query)
    reports_to_lookup = services._is_reports_to_lookup(query)
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            _ensure_fulltext_indexes(session)
            chunk_rows = [
                services._prepare_chunk_result(row, focus_terms=focus_terms, reports_to_lookup=reports_to_lookup)
                for row in session.run(
                    CHUNK_FULLTEXT_QUERY,
                    {"index_name": CHUNK_FULLTEXT_INDEX, "query": query},
                ).data()
                if row.get("chunk_id") or row.get("chunk_summary")
            ]
            doc_rows = [
                services._prepare_chunk_result(row, focus_terms=focus_terms, reports_to_lookup=reports_to_lookup)
                for row in session.run(
                    DOCUMENT_FULLTEXT_QUERY,
                    {"index_name": DOCUMENT_FULLTEXT_INDEX, "query": query},
                ).data()
                if row.get("chunk_id") or row.get("chunk_summary")
            ]
    except Exception as exc:
        return {
            "documents": [],
            "trace": _normalize_trace(
                {
                    "query": query,
                    "query_type": services._classify_query(query),
                    "user_scoped": bool(user_id and services._contains_first_person(query)),
                    "user_id": user_id,
                    "matched_entities": [],
                    "result_count": 0,
                    "max_hop_count": 0,
                    "retrieval_path": services._build_path_summary(bool(user_id and services._contains_first_person(query)), None)["path"],
                    "evidence": [],
                    "no_evidence": True,
                    "error": str(exc),
                },
                tool_name="fulltext",
            ),
        }
    merged_rows = services._merge_ranked_results(chunk_rows, doc_rows, limit=5)
    return _build_trace_from_rows(merged_rows, query=query, user_id=user_id, tool_name="fulltext")


def merge_results(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
    *,
    limit: int = 6,
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
) -> Dict[str, Any]:
    del seed_trace

    if strategy == "semantic":
        return semantic_retrieve(query, user_id=user_id)
    if strategy == "fulltext":
        return fulltext_retrieve(query, user_id=user_id)

    semantic = semantic_retrieve(query, user_id=user_id)
    if strategy == "hybrid":
        return merge_results(semantic, fulltext_retrieve(query, user_id=user_id))
    return semantic
