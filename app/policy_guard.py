"""Minimal grounding and policy checks for the agentic answer loop."""

from __future__ import annotations

from typing import Any, Dict, List


_UNCERTAINTY_MARKERS = (
    "don't have any specific information",
    "do not have any specific information",
    "evidence is weak",
    "not enough evidence",
    "i couldn't find",
)
_POLICY_TOKENS = ("policy", "compliance", "compliant", "pii", "gdpr", "approval")


def evaluate_answer(
    *,
    query: str,
    answer: str,
    answer_payload: Dict[str, Any],
    trace: Dict[str, Any] | None,
    plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    del plan

    issues: List[str] = []
    evidence = list((trace or {}).get("evidence") or [])
    evidence_refs = list((answer_payload or {}).get("evidence_refs") or [])
    normalized_answer = (answer or "").lower()

    if not evidence and not any(marker in normalized_answer for marker in _UNCERTAINTY_MARKERS):
        issues.append("missing_grounded_uncertainty")

    if any(token in (query or "").lower() for token in _POLICY_TOKENS) and not evidence_refs:
        issues.append("missing_policy_provenance")

    return {
        "passed": not issues,
        "retryable": bool(issues),
        "issues": issues,
        "grounded_evidence_count": len(evidence),
        "provenance_count": len(evidence_refs),
    }
