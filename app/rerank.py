"""Query-aware post-retrieval reranking for the agentic path."""

from __future__ import annotations

from datetime import datetime, timezone
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


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fact_temporal_sort_key(item: Dict[str, Any], query_type: str) -> Tuple[int, float, float]:
    if query_type not in {"task_commitment_lookup", "schedule_or_timeline"}:
        return (0, float("-inf"), float("-inf"))

    fact = item.get("fact") or {}
    if not item.get("fact_id"):
        return (0, float("-inf"), float("-inf"))

    now = datetime.now(timezone.utc)
    temporal_start = _parse_iso_timestamp(fact.get("temporal_start"))
    last_activity = (
        _parse_iso_timestamp(fact.get("last_seen_at"))
        or _parse_iso_timestamp(fact.get("first_seen_at"))
        or _parse_iso_timestamp((item.get("document") or {}).get("timestamp"))
    )

    object_known = bool(fact.get("object_entity_id") or fact.get("object_key"))
    status = str(fact.get("status") or "").lower()

    bucket = 0
    temporal_seconds = float("-inf")
    last_activity_seconds = float("-inf")

    if temporal_start is not None:
        delta_seconds = (temporal_start - now).total_seconds()
        temporal_seconds = temporal_start.timestamp()
        if delta_seconds >= 0:
            bucket = 4
        elif delta_seconds >= -(14 * 86400):
            bucket = 3
        else:
            bucket = 2
    elif last_activity is not None:
        bucket = 1

    if object_known:
        bucket += 1
    if status == "current":
        bucket += 1
    if last_activity is not None:
        last_activity_seconds = last_activity.timestamp()

    return (bucket, last_activity_seconds, temporal_seconds)


def _sort_key(item: Dict[str, Any], *, query_type: str, score_field: str) -> Tuple[Any, ...]:
    return (
        bool(item.get("exact_match")),
        bool(item.get("fact_priority")),
        *_fact_temporal_sort_key(item, query_type),
        item.get(score_field) if item.get(score_field) is not None else -math.inf,
        item.get("similarity") if item.get("similarity") is not None else item.get("rank_score") or -math.inf,
    )


def _is_reports_to_fact(item: Dict[str, Any]) -> bool:
    return bool(
        item.get("fact_id")
        and str((item.get("fact") or {}).get("claim_type") or "") == "REPORTS_TO"
    )


def _current_fact_value_signature(item: Dict[str, Any]) -> Tuple[str, ...]:
    fact = item.get("fact") or {}
    return (
        str(fact.get("claim_type") or ""),
        str(fact.get("subject_entity_id") or fact.get("subject_key") or ""),
        str(fact.get("object_entity_id") or fact.get("object_key") or ""),
        str(fact.get("temporal_start") or ""),
        str(fact.get("temporal_end") or ""),
        str(fact.get("value_text") or ""),
    )


def _detect_fact_lookup_conflict(
    items: List[Dict[str, Any]],
    *,
    query_type: str,
    query_profile: Dict[str, Any],
) -> Dict[str, Any]:
    if query_profile.get("wants_list_format"):
        return {"ambiguous": False}
    if query_type not in {"task_commitment_lookup", "schedule_or_timeline", "person_lookup"}:
        return {"ambiguous": False}

    current_facts = [
        item
        for item in items
        if item.get("fact_id") and str((item.get("fact") or {}).get("status") or "").lower() == "current"
    ]
    if len(current_facts) < 2:
        return {"ambiguous": False}

    by_canonical_key: Dict[str, List[Dict[str, Any]]] = {}
    for item in current_facts:
        fact = item.get("fact") or {}
        canonical_key = str(fact.get("canonical_key") or "").strip()
        if not canonical_key:
            canonical_key = "|".join(_current_fact_value_signature(item)[:2])
        by_canonical_key.setdefault(canonical_key, []).append(item)

    for canonical_key, candidates in by_canonical_key.items():
        if len(candidates) < 2:
            continue
        value_signatures = {_current_fact_value_signature(item) for item in candidates}
        if len(value_signatures) > 1:
            claim_type = str((candidates[0].get("fact") or {}).get("claim_type") or "")
            return {
                "ambiguous": True,
                "canonical_key": canonical_key,
                "candidate_count": len(candidates),
                "claim_type": claim_type,
                "reason": "conflicting_current_facts",
            }

    return {"ambiguous": False}


def _task_signature(item: Dict[str, Any]) -> str:
    fact = item.get("fact") or {}
    canonical_key = str(fact.get("canonical_key") or "").strip()
    if canonical_key and "::" in canonical_key:
        return canonical_key.rsplit("::", 1)[-1]
    return _normalize_text_fingerprint(item.get("fact_summary"))


def _detect_task_lookup_ambiguity(items: List[Dict[str, Any]], *, query_type: str) -> Dict[str, Any]:
    if query_type != "task_commitment_lookup":
        return {"ambiguous": False}

    task_facts = [
        item
        for item in items
        if item.get("fact_id") and str((item.get("fact") or {}).get("claim_type") or "") == "TASK_ASSIGNMENT"
    ]
    if len(task_facts) < 2:
        return {"ambiguous": False}

    by_signature: Dict[str, List[Dict[str, Any]]] = {}
    for item in task_facts:
        signature = _task_signature(item)
        if not signature:
            continue
        by_signature.setdefault(signature, []).append(item)

    for signature, candidates in by_signature.items():
        if len(candidates) < 2:
            continue
        recipients = {
            str((item.get("fact") or {}).get("object_entity_id") or (item.get("fact") or {}).get("object_key") or "").strip()
            for item in candidates
            if (item.get("fact") or {}).get("object_entity_id") or (item.get("fact") or {}).get("object_key")
        }
        times = {
            str((item.get("fact") or {}).get("temporal_start") or "").strip()
            for item in candidates
            if (item.get("fact") or {}).get("temporal_start")
        }
        if len(recipients) > 1:
            return {
                "ambiguous": True,
                "task_signature": signature,
                "candidate_count": len(candidates),
                "reason": "multiple_recipients",
            }
        if len(times) > 1:
            return {
                "ambiguous": True,
                "task_signature": signature,
                "candidate_count": len(candidates),
                "reason": "multiple_current_times",
            }

    return {"ambiguous": False}


def _apply_lookup_conflicts(
    items: List[Dict[str, Any]],
    *,
    top_k: int,
    query_type: str,
    query_profile: Dict[str, Any],
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    task_ambiguity = _detect_task_lookup_ambiguity(items, query_type=query_type)
    fact_conflict = _detect_fact_lookup_conflict(
        items,
        query_type=query_type,
        query_profile=query_profile,
    )

    for signal in (task_ambiguity, fact_conflict):
        if signal.get("ambiguous"):
            top_k = min(max(top_k, int(signal.get("candidate_count") or 2)), len(items), 3)

    return top_k, task_ambiguity, fact_conflict


def _score_evidence(query: str, evidence: List[Dict[str, Any]], *, query_type: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
        key=lambda item: _sort_key(item, query_type=query_type, score_field="rerank_score"),
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
    query_type = str(reranked_trace.get("query_type") or "").strip()
    query_profile = dict(reranked_trace.get("query_profile") or query_shape.analyze_query(query))
    top_k = DEFAULT_RERANK_TOP_K
    if query_type in {"task_commitment_lookup", "schedule_or_timeline"} and not query_profile.get("wants_list_format"):
        if any(item.get("fact_id") for item in evidence):
            top_k = 1
    if query_type == "person_lookup" and not query_profile.get("wants_list_format"):
        if any(_is_reports_to_fact(item) for item in evidence):
            top_k = 1

    if not evidence:
        reranked_trace["evidence"] = []
        reranked_trace["reranked"] = False
        reranked_trace["query_profile"] = query_profile
        reranked_trace["task_lookup_ambiguity"] = {"ambiguous": False}
        reranked_trace["fact_lookup_conflict"] = {"ambiguous": False}
        reranked_trace["reranker"] = {
            "enabled": False,
            "method": "none",
            "selected_count": 0,
            "candidate_count": 0,
        }
        return {"documents": list(documents or []), "trace": reranked_trace}

    if not query:
        evidence.sort(key=lambda item: _sort_key(item, query_type=query_type, score_field="rank_score"), reverse=True)
        top_k, task_ambiguity, fact_conflict = _apply_lookup_conflicts(
            evidence,
            top_k=top_k,
            query_type=query_type,
            query_profile=query_profile,
        )
        selected, distinct_candidates = _select_diverse_evidence(
            evidence,
            top_k=min(top_k, len(evidence)),
        )
        reranked_trace["evidence"] = selected
        reranked_trace["query_profile"] = query_profile
        reranked_trace["result_count"] = len(selected)
        reranked_trace["reranked"] = True
        reranked_trace["task_lookup_ambiguity"] = task_ambiguity
        reranked_trace["fact_lookup_conflict"] = fact_conflict
        reranked_trace["reranker"] = {
            "enabled": False,
            "method": "score_sort",
            "selected_count": len(selected),
            "candidate_count": len(evidence),
            "distinct_candidate_count": distinct_candidates,
        }
        return {"documents": _unique_documents(selected), "trace": reranked_trace}

    try:
        scored, metadata = _score_evidence(query, evidence, query_type=query_type)
    except Exception:
        evidence.sort(key=lambda item: _sort_key(item, query_type=query_type, score_field="rank_score"), reverse=True)
        top_k, task_ambiguity, fact_conflict = _apply_lookup_conflicts(
            evidence,
            top_k=top_k,
            query_type=query_type,
            query_profile=query_profile,
        )
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
        reranked_trace["task_lookup_ambiguity"] = task_ambiguity
        reranked_trace["fact_lookup_conflict"] = fact_conflict
        reranked_trace["reranker"] = {
            "enabled": False,
            "method": "unavailable",
            "model": DEFAULT_RERANK_MODEL,
            "selected_count": len(selected),
            "candidate_count": len(evidence),
            "distinct_candidate_count": distinct_candidates,
        }
        return {"documents": _unique_documents(selected), "trace": reranked_trace}

    top_k, task_ambiguity, fact_conflict = _apply_lookup_conflicts(
        scored,
        top_k=top_k,
        query_type=query_type,
        query_profile=query_profile,
    )
    selected, distinct_candidates = _select_diverse_evidence(scored, top_k=min(top_k, len(scored)))

    reranked_trace["evidence"] = selected
    reranked_trace["result_count"] = len(selected)
    reranked_trace["max_hop_count"] = max((int(item.get("hop_count") or 0) for item in selected), default=0)
    reranked_trace["retrieval_path"] = selected[0].get("retrieval_path") if selected else reranked_trace.get("retrieval_path")
    reranked_trace["no_evidence"] = not selected
    reranked_trace["evidence_state"] = "no_evidence" if not selected else "partial_evidence" if len(selected) < 2 else "grounded"
    reranked_trace["reranked"] = True
    reranked_trace["query_profile"] = query_profile
    reranked_trace["task_lookup_ambiguity"] = task_ambiguity
    reranked_trace["fact_lookup_conflict"] = fact_conflict
    reranked_trace["reranker"] = {
        "enabled": True,
        "method": metadata["method"],
        "model": metadata["model"],
        "selected_count": len(selected),
        "candidate_count": len(evidence),
        "distinct_candidate_count": distinct_candidates,
    }

    return {"documents": _unique_documents(selected), "trace": reranked_trace}
