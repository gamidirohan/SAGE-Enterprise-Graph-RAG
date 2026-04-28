"""Minimal post-retrieval reranking wrapper for the agentic path."""

from __future__ import annotations

from typing import Any, Dict, List


def rerank(documents: List[str], trace: Dict[str, Any] | None) -> Dict[str, Any]:
    evidence = list((trace or {}).get("evidence") or [])
    evidence.sort(key=lambda item: (item.get("rank_score") or item.get("similarity") or 0), reverse=True)
    reranked_trace = dict(trace or {})
    reranked_trace["evidence"] = evidence
    reranked_trace["reranked"] = bool(evidence)
    return {"documents": documents, "trace": reranked_trace}
