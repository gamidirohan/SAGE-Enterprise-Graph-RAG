"""Query-aware post-retrieval reranking for the agentic path."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import app.query_shape as query_shape
except ImportError:  # pragma: no cover - direct execution fallback
    import query_shape


DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEFAULT_RERANK_TOP_K = max(1, int(os.getenv("RERANK_TOP_K", "3")))
MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str = DEFAULT_RERANK_MODEL) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        raise RuntimeError("CrossEncoder is unavailable") from exc

    if CrossEncoder is None:
        return None
    try:
        return CrossEncoder(
            model_name,
            cache_folder=str(MODEL_CACHE_DIR),
            local_files_only=True,
        )
    except Exception:
        return CrossEncoder(
            model_name,
            cache_folder=str(MODEL_CACHE_DIR),
            local_files_only=False,
        )

def _candidate_text(item: Dict[str, Any]) -> str:
    document = item.get("document") or {}
    fact = item.get("fact") or {}
    related_node = item.get("related_node") or {}

    parts = [
        item.get("fact_summary"),
        item.get("chunk_summary"),
        document.get("subject"),
        document.get("sender"),
        related_node.get("display_name"),
        fact.get("canonical_key"),
        fact.get("subject_entity_id"),
        fact.get("subject_key"),
        fact.get("object_entity_id"),
        fact.get("object_key"),
        item.get("relationship"),
    ]
    return " | ".join(str(part).strip() for part in parts if part)


def _format_document(item: Dict[str, Any]) -> str:
    document = item.get("document") or {}
    if item.get("fact_id"):
        fact = item.get("fact") or {}
        return (
            "Fact Summary: "
            f"{item.get('fact_summary') or 'No fact summary'}, "
            f"Fact ID: {item.get('fact_id') or 'unknown'}, "
            f"Canonical Key: {fact.get('canonical_key') or 'unknown'}, "
            f"Fact Type: {fact.get('claim_type') or 'unknown'}, "
            f"Conversation Type: {document.get('conversation_type') or 'unknown'}, "
            f"Subject: {fact.get('subject_entity_id') or fact.get('subject_key') or 'unknown'}, "
            f"Object: {fact.get('object_entity_id') or fact.get('object_key') or 'unknown'}, "
            f"Time: {fact.get('temporal_start') or 'not specified'}, "
            f"Supporting Document ID: {document.get('doc_id') or 'unknown'}, "
            f"Similarity: {item.get('similarity', 0)}"
        )

    related_node = item.get("related_node") or {}
    return (
        "Chunk Summary: "
        f"{item.get('chunk_summary') or 'No summary'}, "
        f"Document ID: {document.get('doc_id') or 'unknown'}, "
        f"Conversation Type: {document.get('conversation_type') or 'unknown'}, "
        f"Subject: {document.get('subject') or 'No Subject'}, "
        f"Sender: {document.get('sender') or 'Unknown'}, "
        f"Similarity: {item.get('similarity', 0)}, "
        f"Relationship: {item.get('relationship') or 'RELATED_TO'}, "
        f"Direction: {item.get('direction') or 'unknown'}, "
        f"Related Node: {related_node.get('display_name') or 'Unknown'}"
    )


def _normalize_text_fingerprint(value: Any) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _evidence_fingerprint(item: Dict[str, Any]) -> str:
    fact = item.get("fact") or {}
    document = item.get("document") or {}
    for candidate in (
        item.get("fact_summary"),
        item.get("chunk_summary"),
        fact.get("canonical_key"),
        fact.get("subject_entity_id"),
        fact.get("subject_key"),
        fact.get("object_entity_id"),
        fact.get("object_key"),
        document.get("subject"),
    ):
        normalized = _normalize_text_fingerprint(candidate)
        if normalized:
            return normalized
    if item.get("fact_id"):
        return f"fact:{item['fact_id']}"
    if item.get("chunk_id"):
        return f"chunk:{item['chunk_id']}"
    if document.get("doc_id"):
        return f"doc:{document['doc_id']}"
    return f"unknown:{hash(str(item))}"

def _cross_encoder_scores(query: str, evidence: List[Dict[str, Any]]) -> Tuple[List[float], str]:
    model = _get_cross_encoder()
    if model is None:
        raise RuntimeError("CrossEncoder is unavailable")

    pairs = [(query, _candidate_text(item)) for item in evidence]
    if not pairs:
        return [], DEFAULT_RERANK_MODEL

    raw_scores = model.predict(pairs)
    scores = [float(score) for score in raw_scores]
    return scores, DEFAULT_RERANK_MODEL


def _score_evidence(query: str, evidence: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scores, model_name = _cross_encoder_scores(query, evidence)
    method = "cross_encoder"

    scored: List[Dict[str, Any]] = []
    for idx, (item, rerank_score) in enumerate(zip(evidence, scores)):
        enriched = dict(item)
        enriched["retrieval_rank"] = idx + 1
        enriched["rerank_score"] = round(float(rerank_score), 6)
        enriched["rank_score"] = round(
            float(rerank_score) if method == "cross_encoder" else float(rerank_score),
            6,
        )
        scored.append(enriched)

    scored.sort(
        key=lambda item: (
            bool(item.get("exact_match")),
            bool(item.get("fact_priority")),
            item.get("rerank_score") or -math.inf,
            item.get("similarity") or item.get("rank_score") or -math.inf,
        ),
        reverse=True,
    )
    for idx, item in enumerate(scored, start=1):
        item["rerank_rank"] = idx

    return scored, {"method": method, "model": model_name}


def _select_diverse_evidence(
    scored: List[Dict[str, Any]],
    *,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], int]:
    selected: List[Dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    distinct_candidates = {_evidence_fingerprint(item) for item in scored}

    for item in scored:
        fingerprint = _evidence_fingerprint(item)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        selected.append(item)
        if len(selected) >= top_k:
            break

    return selected, len(distinct_candidates)


def _unique_documents(items: Iterable[Dict[str, Any]]) -> List[str]:
    documents: List[str] = []
    for item in items:
        formatted = _format_document(item)
        if formatted not in documents:
            documents.append(formatted)
    return documents


def rerank(documents: List[str], trace: Dict[str, Any] | None) -> Dict[str, Any]:
    reranked_trace = dict(trace or {})
    evidence = [dict(item) for item in (reranked_trace.get("evidence") or [])]
    query = str(reranked_trace.get("query") or "").strip()
    query_profile = dict(reranked_trace.get("query_profile") or query_shape.analyze_query(query))
    top_k = DEFAULT_RERANK_TOP_K

    if not evidence:
        reranked_trace["evidence"] = []
        reranked_trace["reranked"] = False
        reranked_trace["query_profile"] = query_profile
        reranked_trace["reranker"] = {
            "enabled": False,
            "method": "none",
            "selected_count": 0,
            "candidate_count": 0,
        }
        return {"documents": list(documents or []), "trace": reranked_trace}

    if not query:
        evidence.sort(key=lambda item: (item.get("rank_score") or item.get("similarity") or 0), reverse=True)
        selected, distinct_candidates = _select_diverse_evidence(
            evidence,
            top_k=min(top_k, len(evidence)),
        )
        reranked_trace["evidence"] = selected
        reranked_trace["query_profile"] = query_profile
        reranked_trace["result_count"] = len(selected)
        reranked_trace["reranked"] = True
        reranked_trace["reranker"] = {
            "enabled": False,
            "method": "score_sort",
            "selected_count": len(selected),
            "candidate_count": len(evidence),
            "distinct_candidate_count": distinct_candidates,
        }
        return {"documents": _unique_documents(selected), "trace": reranked_trace}

    try:
        scored, metadata = _score_evidence(query, evidence)
    except Exception:
        evidence.sort(key=lambda item: (item.get("rank_score") or item.get("similarity") or 0), reverse=True)
        selected, distinct_candidates = _select_diverse_evidence(
            evidence,
            top_k=min(top_k, len(evidence)),
        )
        reranked_trace["evidence"] = selected
        reranked_trace["result_count"] = len(selected)
        reranked_trace["max_hop_count"] = max((int(item.get("hop_count") or 0) for item in selected), default=0)
        reranked_trace["retrieval_path"] = selected[0].get("retrieval_path") if selected else reranked_trace.get("retrieval_path")
        reranked_trace["no_evidence"] = not selected
        reranked_trace["evidence_state"] = "no_evidence" if not selected else "partial_evidence" if len(selected) < 2 else "grounded"
        reranked_trace["reranked"] = False
        reranked_trace["query_profile"] = query_profile
        reranked_trace["reranker"] = {
            "enabled": False,
            "method": "unavailable",
            "model": DEFAULT_RERANK_MODEL,
            "selected_count": len(selected),
            "candidate_count": len(evidence),
            "distinct_candidate_count": distinct_candidates,
        }
        return {"documents": _unique_documents(selected), "trace": reranked_trace}

    selected, distinct_candidates = _select_diverse_evidence(
        scored,
        top_k=min(top_k, len(scored)),
    )

    reranked_trace["evidence"] = selected
    reranked_trace["result_count"] = len(selected)
    reranked_trace["max_hop_count"] = max((int(item.get("hop_count") or 0) for item in selected), default=0)
    reranked_trace["retrieval_path"] = selected[0].get("retrieval_path") if selected else reranked_trace.get("retrieval_path")
    reranked_trace["no_evidence"] = not selected
    reranked_trace["evidence_state"] = "no_evidence" if not selected else "partial_evidence" if len(selected) < 2 else "grounded"
    reranked_trace["reranked"] = True
    reranked_trace["query_profile"] = query_profile
    reranked_trace["reranker"] = {
        "enabled": True,
        "method": metadata["method"],
        "model": metadata["model"],
        "selected_count": len(selected),
        "candidate_count": len(evidence),
        "distinct_candidate_count": distinct_candidates,
    }

    return {"documents": _unique_documents(selected), "trace": reranked_trace}
