"""Retrieval strategy selection for agentic query-time orchestration.

This module keeps strategy selection cheap and deterministic by default while
allowing an optional LLM fallback when heuristic confidence is low.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

try:
    import app.utils as utils
except ImportError:  # pragma: no cover - direct execution fallback
    import utils


logger = logging.getLogger(__name__)

ALLOW_LLM_SELECTOR = os.getenv("AGENTIC_RETRIEVAL_SELECTOR_LLM", "true").lower() in {"1", "true", "yes"}

_POLICY_TOKENS = ("policy", "compliance", "compliant", "pii", "gdpr", "approval")
_GRAPH_TOKENS = ("reports to", "manager", "org chart", "who manages", "relationship", "connected", "graph")
_KEYWORD_TOKENS = ("exactly", "subject line", "doc id", "document id", "who said", "quoted")
_FIRST_PERSON_PATTERN = re.compile(r"\b(i|me|my|mine)\b", re.IGNORECASE)


class _SelectorDecision(BaseModel):
    strategy: str = Field(default="hybrid")
    reason: str = Field(default="Heuristic fallback.")


_SELECTOR_PROMPT = ChatPromptTemplate.from_template(
    """
    You are deciding which retrieval strategy should be used for an enterprise
    Graph RAG question. Return JSON only.

    Valid strategies:
    - semantic
    - graph
    - fulltext
    - hybrid

    Use:
    - graph for org-structure / relationship / path questions
    - fulltext for exact wording / identifiers / lexical lookup
    - semantic for broad descriptive questions
    - hybrid when multiple modes are useful

    User id present: {has_user_id}
    Query: {query}
    """
)


def _create_llm_selector():
    if not utils.GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model_name=utils.GROQ_MODEL,
            temperature=0.0,
            groq_api_key=utils.GROQ_API_KEY,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to initialize retrieval selector LLM: %s", exc)
        return None


def _heuristic_strategy(query: str, user_id: Optional[str]) -> Dict[str, Any]:
    normalized = " ".join((query or "").lower().split())
    reasons: List[str] = []

    if any(token in normalized for token in _POLICY_TOKENS):
        reasons.append("Policy/compliance wording benefits from graph plus lexical evidence.")
        return {"strategy": "hybrid", "reasons": reasons, "confidence": 0.9}

    if any(token in normalized for token in _GRAPH_TOKENS):
        reasons.append("Relationship wording suggests graph traversal.")
        return {"strategy": "graph", "reasons": reasons, "confidence": 0.88}

    if any(token in normalized for token in _KEYWORD_TOKENS):
        reasons.append("Exact-match wording suggests lexical retrieval.")
        return {"strategy": "fulltext", "reasons": reasons, "confidence": 0.85}

    if _FIRST_PERSON_PATTERN.search(query or "") and user_id:
        reasons.append("First-person wording with an authenticated user benefits from hybrid retrieval.")
        return {"strategy": "hybrid", "reasons": reasons, "confidence": 0.82}

    reasons.append("Broad question defaults to semantic retrieval.")
    return {"strategy": "semantic", "reasons": reasons, "confidence": 0.7}


def decide_strategy(query: str, *, user_id: Optional[str] = None) -> Dict[str, Any]:
    heuristic = _heuristic_strategy(query, user_id)
    strategy = heuristic["strategy"]
    reasons = list(heuristic["reasons"])
    llm_used = False

    if heuristic["confidence"] < 0.8 and ALLOW_LLM_SELECTOR:
        llm = _create_llm_selector()
        if llm is not None:
            try:
                parser = JsonOutputParser(pydantic_object=_SelectorDecision)
                chain = _SELECTOR_PROMPT | llm | parser
                payload = chain.invoke({"query": query, "has_user_id": bool(user_id)})
                candidate = _SelectorDecision.model_validate(payload)
                if candidate.strategy in {"semantic", "graph", "fulltext", "hybrid"}:
                    strategy = candidate.strategy
                    reasons.append(candidate.reason)
                    llm_used = True
            except Exception as exc:  # pragma: no cover - network/model behavior
                logger.warning("Retrieval selector LLM fallback failed: %s", exc)

    return {
        "strategy": strategy,
        "reasons": reasons,
        "llm_used": llm_used,
        "heuristic_confidence": heuristic["confidence"],
    }
