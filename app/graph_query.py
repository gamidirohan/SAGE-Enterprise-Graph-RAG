"""Graph-query helpers used by the agentic orchestrator."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import app.services as services
    import app.utils as utils
except ImportError:  # pragma: no cover - direct execution fallback
    import services
    import utils


SCHEMA_SNAPSHOT: Dict[str, List[str]] = {
    "node_types": [
        "User",
        "Person",
        "Message",
        "Conversation",
        "Document",
        "Chunk",
        "Claim",
        "CanonicalFact",
        "SAIARun",
        "Group",
    ],
    "edge_types": [
        "SENT",
        "RECEIVED_BY",
        "PART_OF",
        "IN_CONVERSATION",
        "HAS_EVIDENCE_DOCUMENT",
        "HAS_CLAIM",
        "SUPPORTS",
        "CONTRADICTS",
        "PROCESSED_BY_SAIA",
        "SUPERSEDED_BY",
    ],
}

DEFAULT_EXPAND_HOPS = 1
MAX_EXPAND_HOPS = 3
DEFAULT_GRAPH_EXPAND_SEED_LIMIT = max(1, int(os.getenv("SAGE_GRAPH_EXPAND_SEED_LIMIT", "20")))
DEFAULT_GRAPH_CONTEXT_PER_SEED_LIMIT = max(1, int(os.getenv("SAGE_GRAPH_CONTEXT_PER_SEED_LIMIT", "3")))


def schema_snapshot() -> Dict[str, List[str]]:
    return dict(SCHEMA_SNAPSHOT)


GRAPH_CONTEXT_FROM_CHUNK_QUERY = """
    MATCH (c:Chunk {chunk_id: $chunk_id})-[:PART_OF]->(d:Document)
    OPTIONAL MATCH (d)-[r:SENT|RECEIVED_BY]-(p:Person)
    RETURN
        c.chunk_id AS chunk_id,
        c.summary AS chunk_summary,
        d,
        $base_score AS similarity,
        coalesce(type(r), 'PART_OF') AS relationship,
        CASE WHEN r IS NULL THEN 'document' WHEN startNode(r) = d THEN 'outgoing' ELSE 'incoming' END AS direction,
        p AS n,
        3 AS hop_count,
        [
            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
            coalesce(c.subject, c.title, c.name, c.id, c.doc_id, c.chunk_id, labels(c)[0]),
            coalesce(p.subject, p.title, p.name, p.id, p.doc_id, labels(p)[0])
        ] AS path_nodes,
        [coalesce(type(r), 'PART_OF'), 'PART_OF'] AS path_relationships
    ORDER BY relationship ASC
    LIMIT $context_limit
"""

GRAPH_CONTEXT_FROM_FACT_QUERY = """
    MATCH (f:CanonicalFact {fact_id: $fact_id})
    OPTIONAL MATCH (f)<-[:SUPPORTS|CONTRADICTS]-(c:Claim)<-[:HAS_CLAIM]-(d:Document)
    RETURN
        f.fact_id AS fact_id,
        f.summary AS fact_summary,
        f,
        collect(DISTINCT d)[0] AS d,
        $base_score AS similarity
    LIMIT 1
"""


def _normalize_expand_hops(expand_hops: Optional[int]) -> int:
    try:
        value = int(expand_hops)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = DEFAULT_EXPAND_HOPS
    return max(1, min(value, MAX_EXPAND_HOPS))


def _build_graph_context_from_chunk_query(expand_hops: int) -> str:
    if expand_hops <= DEFAULT_EXPAND_HOPS:
        return GRAPH_CONTEXT_FROM_CHUNK_QUERY
    return f"""
    MATCH (c:Chunk {{chunk_id: $chunk_id}})-[:PART_OF]->(d:Document)
    CALL {{
        WITH c, d
        MATCH p=(d)<-[:PART_OF]-(c)-[*1..{expand_hops}]-(n)
        WHERE all(node IN nodes(p) WHERE single(other IN nodes(p) WHERE other = node))
          AND NOT n:Chunk
          AND NOT n:Document
        RETURN p, n
        ORDER BY length(p) DESC
        LIMIT $context_limit
    }}
    RETURN
        c.chunk_id AS chunk_id,
        c.summary AS chunk_summary,
        d,
        $base_score AS similarity,
        coalesce(type(last(relationships(p))), 'PART_OF') AS relationship,
        CASE WHEN endNode(last(relationships(p))) = n THEN 'outgoing' ELSE 'incoming' END AS direction,
        n AS n,
        length(p) AS hop_count,
        [
            node IN nodes(p) |
            coalesce(node.subject, node.title, node.name, node.id, node.doc_id, node.chunk_id, labels(node)[0])
        ] AS path_nodes,
        [rel IN relationships(p) | type(rel)] AS path_relationships
    ORDER BY hop_count ASC, relationship ASC
    LIMIT $context_limit
    """


def expand_retrieval_context(
    query: str,
    *,
    seed_trace: Optional[Dict[str, Any]],
    user_id: Optional[str] = None,
    expand_hops: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_expand_hops = _normalize_expand_hops(expand_hops)
    evidence = list((seed_trace or {}).get("evidence") or [])
    if not evidence:
        return {
            "documents": [],
            "trace": {
                "query": query,
                "query_type": services._classify_query(query),
                "user_scoped": bool(user_id and services._is_personalized_lookup(query)),
                "user_id": user_id,
                "matched_entities": [],
                "result_count": 0,
                "max_hop_count": 0,
                "graph_depth": {"expand_hops": resolved_expand_hops},
                "retrieval_path": services._build_path_summary(bool(user_id and services._is_personalized_lookup(query)), None)["path"],
                "evidence": [],
                "no_evidence": True,
                "evidence_state": "no_evidence",
                "selector_strategy": "graph",
            },
        }

    driver = utils.create_neo4j_driver()
    ranked_rows: List[Dict[str, Any]] = []
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            for rank, item in enumerate(evidence[:DEFAULT_GRAPH_EXPAND_SEED_LIMIT], start=1):
                base_score = max(float(item.get("rank_score", item.get("similarity", 0.5)) or 0.5) - (rank * 0.01), 0.1)
                if item.get("chunk_id"):
                    chunk_query = _build_graph_context_from_chunk_query(resolved_expand_hops)
                    rows = session.run(
                        chunk_query,
                        chunk_id=item["chunk_id"],
                        base_score=base_score,
                        context_limit=DEFAULT_GRAPH_CONTEXT_PER_SEED_LIMIT,
                    ).data()
                    if not rows and resolved_expand_hops > DEFAULT_EXPAND_HOPS:
                        rows = session.run(
                            GRAPH_CONTEXT_FROM_CHUNK_QUERY,
                            chunk_id=item["chunk_id"],
                            base_score=base_score,
                            context_limit=DEFAULT_GRAPH_CONTEXT_PER_SEED_LIMIT,
                        ).data()
                    ranked_rows.extend(
                        services._prepare_chunk_result(row, focus_terms=[], reports_to_lookup=False)
                        for row in rows
                        if row.get("chunk_id") or row.get("chunk_summary")
                    )
                if item.get("fact_id"):
                    rows = session.run(
                        GRAPH_CONTEXT_FROM_FACT_QUERY,
                        fact_id=item["fact_id"],
                        base_score=base_score,
                    ).data()
                    ranked_rows.extend(
                        services._prepare_fact_result(
                            row,
                            query_type=services._classify_query(query),
                            user_id=user_id,
                            personalized_lookup=bool(user_id and services._is_personalized_lookup(query)),
                            exact_match=True,
                            focus_terms=[],
                            reports_to_lookup=services._is_reports_to_lookup(query),
                        )
                        for row in rows
                        if row.get("fact_id")
                    )
    except Exception as exc:
        return {
            "documents": [],
            "trace": {
                "query": query,
                "query_type": services._classify_query(query),
                "user_scoped": bool(user_id and services._is_personalized_lookup(query)),
                "user_id": user_id,
                "matched_entities": [],
                "result_count": 0,
                "max_hop_count": 0,
                "graph_depth": {"expand_hops": resolved_expand_hops},
                "retrieval_path": services._build_path_summary(bool(user_id and services._is_personalized_lookup(query)), None)["path"],
                "evidence": [],
                "no_evidence": True,
                "evidence_state": "no_evidence",
                "selector_strategy": "graph",
                "error": str(exc),
            },
        }
    finally:
        driver.close()

    merged = services._merge_ranked_results(ranked_rows, [], limit=services.DEFAULT_RETRIEVAL_LIMIT)
    if not merged:
        return {
            "documents": [],
            "trace": {
                "query": query,
                "query_type": services._classify_query(query),
                "user_scoped": bool(user_id and services._is_personalized_lookup(query)),
                "user_id": user_id,
                "matched_entities": [],
                "result_count": 0,
                "max_hop_count": 0,
                "graph_depth": {"expand_hops": resolved_expand_hops},
                "retrieval_path": services._build_path_summary(bool(user_id and services._is_personalized_lookup(query)), None)["path"],
                "evidence": [],
                "no_evidence": True,
                "evidence_state": "no_evidence",
                "selector_strategy": "graph",
            },
        }

    try:
        from app import vector_search
    except ImportError:  # pragma: no cover - direct execution fallback
        import vector_search

    result = vector_search._build_trace_from_rows(
        merged,
        query=query,
        user_id=user_id,
        tool_name="graph",
        context_hops=resolved_expand_hops,
    )
    result["trace"]["selector_strategy"] = "graph"
    result["trace"]["graph_depth"] = {"expand_hops": resolved_expand_hops}
    return result

    # 'validate_trace_paths': G- CT (Graph Constrained Chain of Thought): 
    # Validate evidence has valid graph/fact bindings and sufficient provenance 
    #   for grounding, attribution, and determining if additional retrieval is needed.
    
def validate_trace_paths(trace: Dict[str, Any] | None) -> Dict[str, Any]:
    evidence = list((trace or {}).get("evidence") or [])
    validated = 0
    invalid_refs: List[str] = []
    validated_bindings: List[Dict[str, Any]] = []
    missing_fields: List[Dict[str, Any]] = []

    for index, item in enumerate(evidence):
        if item.get("fact_id") and (item.get("document") or {}).get("doc_id"):
            validated += 1
            validated_bindings.append(
                {
                    "evidence_index": index,
                    "binding_type": "fact_document",
                    "fact_id": item.get("fact_id"),
                    "doc_id": (item.get("document") or {}).get("doc_id"),
                }
            )
            continue
        if item.get("chunk_id") and (item.get("document") or {}).get("doc_id"):
            validated += 1
            validated_bindings.append(
                {
                    "evidence_index": index,
                    "binding_type": "chunk_document",
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": (item.get("document") or {}).get("doc_id"),
                    "relationship": item.get("relationship"),
                    "direction": item.get("direction"),
                    "related_node_id": item.get("related_node_id"),
                }
            )
            continue
        if item.get("related_node_id") and item.get("relationship"):
            validated += 1
            validated_bindings.append(
                {
                    "evidence_index": index,
                    "binding_type": "node_relationship",
                    "related_node_id": item.get("related_node_id"),
                    "relationship": item.get("relationship"),
                    "direction": item.get("direction"),
                }
            )
            continue
        invalid_refs.append(f"evidence[{index}]")
        missing_fields.append(
            {
                "evidence_index": index,
                "missing": [
                    field_name
                    for field_name in ("fact_id", "chunk_id", "related_node_id")
                    if not item.get(field_name)
                ],
            }
        )

    return {
        "valid": validated > 0 or not evidence,
        "validated_evidence_count": validated,
        "validated_bindings": validated_bindings,
        "invalid_refs": invalid_refs,
        "missing_fields": missing_fields,
        "reason": None if not invalid_refs else "Some evidence items were missing provenance bindings.",
    }
