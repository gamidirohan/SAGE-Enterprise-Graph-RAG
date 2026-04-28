"""Graph-query helpers used by the agentic orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List


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


def schema_snapshot() -> Dict[str, List[str]]:
    return dict(SCHEMA_SNAPSHOT)


def validate_trace_paths(trace: Dict[str, Any] | None) -> Dict[str, Any]:
    evidence = list((trace or {}).get("evidence") or [])
    validated = 0
    invalid_refs: List[str] = []

    for index, item in enumerate(evidence):
        if item.get("fact_id") or item.get("chunk_id") or item.get("related_node_id"):
            validated += 1
            continue
        invalid_refs.append(f"evidence[{index}]")

    return {
        "valid": validated > 0 or not evidence,
        "validated_evidence_count": validated,
        "invalid_refs": invalid_refs,
        "reason": None if not invalid_refs else "Some evidence items were missing stable graph identifiers.",
    }
