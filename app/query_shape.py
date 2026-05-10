"""Lightweight query-shape heuristics shared across agentic stages."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


_MULTI_ITEM_PHRASES = (
    "who all",
    "which all",
    "find all",
    "list all",
    "show all",
    "all of the",
    "all the",
    "full list",
    "various",
    "multiple",
    "different",
    "several",
)
_EXHAUSTIVE_PHRASES = (
    "all",
    "every",
    "each",
    "entire",
    "complete",
    "everything",
    "anything about",
)
_LIST_FORMAT_PHRASES = (
    "list",
    "show",
    "which",
    "who all",
    "what are the",
    "who are the",
)
_BROAD_SYNTHESIS_PHRASES = (
    "compare",
    "detailed summary",
    "detailed overview",
    "explain all",
    "walk me through",
    "everything we know",
)
_FIRST_PERSON_SINGULAR = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
_FIRST_PERSON_PLURAL = re.compile(r"\b(we|us|our|ours|ourselves)\b", re.IGNORECASE)
_PLURAL_INTERROGATIVE_PATTERN = re.compile(
    r"\b(who|what|which)\s+(?:are|were)\s+(?:the\s+)?[a-z][a-z0-9_-]*s\b",
    re.IGNORECASE,
)


def _normalize_query(text: str) -> str:
    return " ".join((text or "").lower().split())


def analyze_query(text: str) -> Dict[str, Any]:
    normalized = _normalize_query(text)
    contains_multi_phrase = any(phrase in normalized for phrase in _MULTI_ITEM_PHRASES)
    contains_broad_synthesis = any(phrase in normalized for phrase in _BROAD_SYNTHESIS_PHRASES)
    contains_plural_interrogative = bool(_PLURAL_INTERROGATIVE_PATTERN.search(normalized))
    contains_list_prompt = any(phrase in normalized for phrase in _LIST_FORMAT_PHRASES)

    expects_multiple_items = contains_multi_phrase or contains_plural_interrogative or contains_broad_synthesis
    requires_broad_coverage = expects_multiple_items or contains_broad_synthesis or any(
        re.search(rf"\b{re.escape(token)}\b", normalized)
        for token in _EXHAUSTIVE_PHRASES
    )
    wants_list_format = expects_multiple_items or contains_list_prompt

    first_person_scope = "none"
    if _FIRST_PERSON_SINGULAR.search(text or ""):
        first_person_scope = "singular"
    elif _FIRST_PERSON_PLURAL.search(text or ""):
        first_person_scope = "plural"

    return {
        "expects_multiple_items": expects_multiple_items,
        "requires_broad_coverage": requires_broad_coverage,
        "wants_list_format": wants_list_format,
        "first_person_scope": first_person_scope,
        "minimum_unique_evidence": 2 if expects_multiple_items else 1,
        "minimum_tool_rounds": 2 if requires_broad_coverage else 1,
    }


def recommend_graph_depth(
    text: str,
    *,
    query_profile: Optional[Dict[str, Any]] = None,
    query_type: Optional[str] = None,
) -> Dict[str, Any]:
    profile = dict(query_profile or analyze_query(text))
    normalized = _normalize_query(text)
    effective_query_type = str(query_type or "").strip().lower()

    if (
        effective_query_type == "compound_lookup"
        or profile.get("requires_broad_coverage")
        or any(
            phrase in normalized
            for phrase in (
                "compare",
                "differences between",
                "detailed summary",
                "detailed overview",
                "walk me through",
                "everything we know",
                "including",
            )
        )
    ):
        return {
            "seed_hops": 1,
            "expand_hops": 3,
            "max_hops": 3,
            "reason": "broad_multi_hop",
        }

    if (
        effective_query_type == "explanation"
        or any(
            token in normalized
            for token in (
                "policy",
                "compliance",
                "compliant",
                "violation",
                "violates",
                "audit",
                "risk",
                "why",
                "because",
                "cause",
                "root cause",
            )
        )
    ):
        return {
            "seed_hops": 1,
            "expand_hops": 3,
            "max_hops": 3,
            "reason": "policy_or_explanatory",
        }

    if (
        effective_query_type in {"task_commitment_lookup", "schedule_or_timeline", "person_lookup", "personal_context"}
        or any(
            token in normalized
            for token in (
                "report to",
                "reports to",
                "manager",
                "owner",
                "ownership",
                "responsible",
                "approved",
                "attended",
                "timeline",
                "meeting",
                "review",
            )
        )
    ):
        return {
            "seed_hops": 1,
            "expand_hops": 2,
            "max_hops": 3,
            "reason": "relationship_or_temporal",
        }

    return {
        "seed_hops": 0,
        "expand_hops": 1,
        "max_hops": 2,
        "reason": "direct_lookup",
    }
