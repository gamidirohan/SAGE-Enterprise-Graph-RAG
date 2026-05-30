"""Minimal grounding and policy checks for the agentic answer loop."""

from __future__ import annotations

import re
from typing import Any, Dict, List

try:
    import app.query_shape as query_shape
except ImportError:  # pragma: no cover - direct execution fallback
    import query_shape


_UNCERTAINTY_MARKERS = (
    "don't have any specific information",
    "do not have any specific information",
    "no verified evidence",
    "no evidence",
    "evidence is weak",
    "not enough evidence",
    "i couldn't find",
)
_POLICY_TOKENS = ("policy", "compliance", "compliant", "pii", "gdpr", "approval")
_TEMPORAL_ANSWER_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}:\d{2}\b"
    r"|\b\d{1,2}\s*(?:am|pm)\b"
    r"|\b(?:AM|PM|IST|UTC)\b"
    r"|\b(?:today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_DURATION_PATTERN = re.compile(r"\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\b", re.IGNORECASE)
_DIRECT_LOOKUP_PREFIX = re.compile(r"^\s*(who|whom|what|when|which|where|did|do|does|is|are|was|were|am|can)\b", re.IGNORECASE)
_FOCUS_ACRONYM_PATTERN = re.compile(r"\b[A-Z0-9][A-Z0-9_\-]{1,}\b")
_FOCUS_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
_FOCUS_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b")
_FOCUS_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "for",
    "with",
    "by",
    "on",
    "in",
    "of",
    "and",
    "or",
    "from",
    "about",
    "start",
    "starts",
    "started",
    "starting",
    "work",
    "works",
    "worked",
    "working",
    "project",
    "me",
    "my",
    "you",
    "your",
    "who",
    "whom",
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "am",
    "has",
    "have",
    "had",
    "can",
    "will",
    "would",
    "should",
    "could",
    "tell",
    "show",
    "give",
    "current",
    "currently",
}
_ADDRESS_VALUE_PATTERN = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9][A-Za-z0-9 .'-]*\b(?:street|st\.?|road|rd\.?|avenue|ave\.?|"
    r"drive|dr\.?|lane|ln\.?|way|boulevard|blvd\.?|suite|floor|building)\b"
    r"|\b(?:located|address|headquarters)\b",
    re.IGNORECASE,
)
_ANSWER_SLOT_CONTRACTS = (
    {
        "slot": "temporal_start",
        "query_pattern": re.compile(
            r"^\s*(?:when|by\s+when|from\s+when|what\s+time|what\s+day|what\s+date|which\s+day)\b"
            r"|\b(?:start|starts|started|starting|scheduled|deadline|due)\b",
            re.IGNORECASE,
        ),
        "fact_fields": ("temporal_start",),
        "evidence_pattern": re.compile(
            r"\b(?:starting|starts?|beginning|begins?|from|scheduled\s+for|due|by|on|at)\s+"
            r"(?:today|tomorrow|yesterday|now|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
            r"in\s+\d+\s+(?:day|days|week|weeks)|\d{4}-\d{2}-\d{2}|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
            re.IGNORECASE,
        ),
        "answer_pattern": _TEMPORAL_ANSWER_PATTERN,
    },
    {
        "slot": "temporal_end",
        "query_pattern": re.compile(
            r"\b(?:until|end|ends|ended|ending|finish|finishes|finished|duration|how\s+long)\b",
            re.IGNORECASE,
        ),
        "fact_fields": ("temporal_end",),
        "evidence_pattern": re.compile(
            r"\b(?:until|ending|ends?|through|for\s+\d+\s+(?:day|days|week|weeks|month|months))\b",
            re.IGNORECASE,
        ),
        "answer_pattern": _TEMPORAL_ANSWER_PATTERN,
        "require_evidence_value": True,
    },
)


def _answer_text(answer: str, answer_payload: Dict[str, Any]) -> str:
    payload_text_parts = [answer or "", str((answer_payload or {}).get("summary") or "")]
    payload_text_parts.extend(str(item) for item in ((answer_payload or {}).get("bullets") or []))
    return " ".join(payload_text_parts)


def _evidence_has_field(evidence: List[Dict[str, Any]], field_names: tuple[str, ...]) -> bool:
    return any((item.get("fact") or {}).get(field_name) for item in evidence for field_name in field_names)


def _evidence_text(evidence: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in evidence:
        document = item.get("document") or {}
        for value in (
            item.get("fact_summary"),
            item.get("chunk_summary"),
            document.get("content"),
            document.get("subject"),
        ):
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def _evidence_supports_slot(evidence: List[Dict[str, Any]], contract: Dict[str, Any]) -> bool:
    if _evidence_has_field(evidence, contract["fact_fields"]):
        return True
    evidence_pattern = contract.get("evidence_pattern")
    return bool(evidence_pattern and evidence_pattern.search(_evidence_text(evidence)))


def _evidence_field_values(evidence: List[Dict[str, Any]], field_names: tuple[str, ...]) -> List[str]:
    values: List[str] = []
    for item in evidence:
        fact = item.get("fact") or {}
        for field_name in field_names:
            value = str(fact.get(field_name) or "").strip()
            if value and value not in values:
                values.append(value)
    return values


def _evidence_duration_values(evidence: List[Dict[str, Any]]) -> List[str]:
    values: List[str] = []
    for match in _DURATION_PATTERN.finditer(_evidence_text(evidence)):
        value = " ".join(match.group(0).split())
        if value and value not in values:
            values.append(value)
    return values


def _value_appears_in_answer(value: str, answer_text: str) -> bool:
    if not value:
        return False
    if value in answer_text:
        return True
    date_match = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
    return bool(date_match and date_match.group(1) in answer_text)


def _answer_satisfies_slot(contract: Dict[str, Any], *, evidence: List[Dict[str, Any]], answer_text: str) -> bool:
    if contract.get("require_evidence_value"):
        if any(
            _value_appears_in_answer(value, answer_text)
            for value in _evidence_field_values(evidence, contract["fact_fields"])
        ):
            return True
        if contract.get("slot") == "temporal_end":
            return any(
                _value_appears_in_answer(value, answer_text)
                for value in _evidence_duration_values(evidence)
            )
        return False
    return bool(contract["answer_pattern"].search(answer_text))


def _extract_focus_terms(text: str) -> List[str]:
    focus_terms: List[str] = []
    seen: set[str] = set()

    for pattern in (_FOCUS_ACRONYM_PATTERN, _FOCUS_NAME_PATTERN, _FOCUS_TOKEN_PATTERN):
        for match in pattern.finditer(text or ""):
            term = " ".join(match.group(0).strip().lower().split())
            if len(term) < 2 or term in _FOCUS_STOPWORDS:
                continue
            if len(term.split()) == 1 and term in _FOCUS_STOPWORDS:
                continue
            if term not in seen:
                focus_terms.append(term)
                seen.add(term)
    return focus_terms


def _requires_direct_focus_match(query: str, *, query_type: str, profile: Dict[str, Any]) -> bool:
    if profile.get("wants_list_format") or profile.get("requires_broad_coverage"):
        return False
    if query_type not in {"general_search", "personal_context"}:
        return False
    return bool(_DIRECT_LOOKUP_PREFIX.search(query or ""))


def _minimum_focus_match_threshold(query: str, *, query_type: str, profile: Dict[str, Any], focus_terms: List[str]) -> int:
    if not focus_terms or profile.get("wants_list_format") or profile.get("requires_broad_coverage"):
        return 0
    if query_type in {"task_commitment_lookup", "schedule_or_timeline"}:
        return 2 if len(focus_terms) >= 2 else 0
    if _requires_direct_focus_match(query, query_type=query_type, profile=profile):
        return 1
    return 0


def _evidence_focus_match_score(item: Dict[str, Any], focus_terms: List[str]) -> int:
    if not focus_terms:
        return 0
    document = item.get("document") or {}
    related_node = item.get("related_node") or {}
    fact = item.get("fact") or {}
    parts = [
        item.get("fact_summary"),
        item.get("chunk_summary"),
        document.get("content"),
        document.get("subject"),
        document.get("sender"),
        related_node.get("display_name"),
        related_node.get("id"),
        fact.get("canonical_key"),
        fact.get("claim_type"),
        fact.get("value_text"),
        fact.get("subject_key"),
        fact.get("subject_entity_id"),
        fact.get("subject_display"),
        fact.get("object_key"),
        fact.get("object_entity_id"),
        fact.get("object_display"),
        fact.get("display_summary"),
    ]
    haystack = " ".join(str(part) for part in parts if part).lower()
    return sum(1 for term in focus_terms if term in haystack)


def _query_asks_for_address_value(query: str) -> bool:
    return bool(re.search(r"\b(?:address|location|located|where|hq|headquarters)\b", query or "", re.IGNORECASE))


def _answer_satisfies_address_value(query: str, answer_text: str) -> bool:
    if not _query_asks_for_address_value(query):
        return True
    return bool(_ADDRESS_VALUE_PATTERN.search(answer_text or ""))


def _missing_required_answer_slots(
    *,
    query: str,
    evidence: List[Dict[str, Any]],
    answer: str,
    answer_payload: Dict[str, Any],
) -> List[str]:
    text = _answer_text(answer, answer_payload)
    missing_slots: List[str] = []
    for contract in _ANSWER_SLOT_CONTRACTS:
        if not contract["query_pattern"].search(query or ""):
            continue
        if not _evidence_supports_slot(evidence, contract):
            continue
        if _answer_satisfies_slot(contract, evidence=evidence, answer_text=text):
            continue
        missing_slots.append(str(contract["slot"]))
    return missing_slots


def _query_requires_reports_to(query: str) -> bool:
    text = query or ""
    if re.search(r"\b(?:manager|owner|lead|supervisor)\s+of\s+\w", text, re.IGNORECASE):
        return False
    if re.search(r"\b(?:managed|owned|led|supervised)\s+by\b", text, re.IGNORECASE):
        return False
    return bool(
        re.search(r"\breports?\s+to\b", text, re.IGNORECASE)
        or re.search(r"\b(?:my|your|their|his|her|[A-Z][A-Za-z0-9_\-]+(?:'s|’s))\s+(?:manager|boss|supervisor|lead)\b", text, re.IGNORECASE)
    )


def _evidence_has_claim_type(evidence: List[Dict[str, Any]], claim_type: str) -> bool:
    return any(str((item.get("fact") or {}).get("claim_type") or "") == claim_type for item in evidence)


def _answer_mentions_reporting(answer_text: str) -> bool:
    return bool(re.search(r"\b(?:reports?\s+to|manager|boss|supervisor|lead)\b", answer_text or "", re.IGNORECASE))


def evaluate_answer(
    *,
    query: str,
    answer: str,
    answer_payload: Dict[str, Any],
    trace: Dict[str, Any] | None,
    plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    issues: List[str] = []
    evidence = list((trace or {}).get("evidence") or [])
    evidence_refs = list((answer_payload or {}).get("evidence_refs") or [])
    normalized_answer = (answer or "").lower()
    profile = dict(
        ((trace or {}).get("query_profile") or {})
        or ((plan or {}).get("query_profile") or {})
        or query_shape.analyze_query(query)
    )
    coverage = dict((trace or {}).get("coverage") or {})
    query_type = str((trace or {}).get("query_type") or "")
    uncertainty_answer = any(marker in normalized_answer for marker in _UNCERTAINTY_MARKERS)

    if not evidence and not uncertainty_answer:
        issues.append("missing_grounded_uncertainty")

    if any(token in (query or "").lower() for token in _POLICY_TOKENS) and not evidence_refs:
        issues.append("missing_policy_provenance")

    if profile.get("expects_multiple_items") and int(coverage.get("distinct_evidence_count") or 0) < int(profile.get("minimum_unique_evidence") or 1):
        issues.append("insufficient_answer_coverage")

    for slot in _missing_required_answer_slots(
        query=query,
        evidence=evidence,
        answer=answer,
        answer_payload=answer_payload,
    ):
        issues.append(f"missing_required_answer_slot:{slot}")

    focus_terms = _extract_focus_terms(query)
    minimum_focus_match = _minimum_focus_match_threshold(
        query,
        query_type=query_type,
        profile=profile,
        focus_terms=focus_terms,
    )

    if focus_terms and not uncertainty_answer:
        query_text = (query or "").lower()
        answer_text = _answer_text(answer, answer_payload).lower()
        if _FOCUS_NAME_PATTERN.search(query or "") and not any(term in answer_text for term in focus_terms):
            issues.append("missing_requested_entity_reference")
    if (
        evidence
        and focus_terms
        and minimum_focus_match > 0
        and max((_evidence_focus_match_score(item, focus_terms) for item in evidence), default=0) < minimum_focus_match
        and not uncertainty_answer
    ):
        issues.append("unfocused_evidence_for_direct_lookup")

    if (
        evidence
        and _query_asks_for_address_value(query)
        and not _answer_satisfies_address_value(query, _answer_text(answer, answer_payload))
        and not uncertainty_answer
    ):
        issues.append("missing_required_answer_slot:address_or_location")

    if _query_requires_reports_to(query):
        if not _evidence_has_claim_type(evidence, "REPORTS_TO"):
            if not uncertainty_answer:
                issues.append("missing_required_answer_relation:reports_to")
        elif not _answer_mentions_reporting(_answer_text(answer, answer_payload)):
            issues.append("missing_required_answer_relation:reports_to")

    return {
        "passed": not issues,
        "retryable": bool(issues),
        "issues": issues,
        "grounded_evidence_count": len(evidence),
        "provenance_count": len(evidence_refs),
    }
