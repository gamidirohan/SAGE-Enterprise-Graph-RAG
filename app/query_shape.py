"""Lightweight query-shape heuristics shared across agentic stages."""

from __future__ import annotations

import re
from typing import Any, Dict


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
    contains_plural_interrogative = bool(_PLURAL_INTERROGATIVE_PATTERN.search(normalized))
    contains_list_prompt = any(phrase in normalized for phrase in _LIST_FORMAT_PHRASES)

    expects_multiple_items = contains_multi_phrase or contains_plural_interrogative
    requires_broad_coverage = expects_multiple_items or any(
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
