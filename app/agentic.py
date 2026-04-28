"""Lightweight agentic orchestration scaffold for query-time reasoning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import app.graph_query as graph_query
    import app.policy_guard as policy_guard
    import app.rerank as rerank
    import app.retrieval_selector as retrieval_selector
    import app.services as services
    import app.vector_search as vector_search
except ImportError:  # pragma: no cover - direct execution fallback
    import graph_query
    import policy_guard
    import rerank
    import retrieval_selector
    import services
    import vector_search


def build_plan(query: str, *, user_id: Optional[str] = None) -> Dict[str, Any]:
    selector = retrieval_selector.decide_strategy(query, user_id=user_id)
    return {
        "planner": "GAP",
        "strategy": selector["strategy"],
        "selector": selector,
        "constraints": {
            "allowed_nodes": graph_query.schema_snapshot()["node_types"],
            "allowed_edges": graph_query.schema_snapshot()["edge_types"],
            "max_depth": 3,
        },
        "steps": [
            {"type": "retrieve", "tool": selector["strategy"], "query": query},
            {"type": "rerank", "tool": "score_sort"},
            {"type": "validate_paths", "tool": "graph_path_validator"},
            {"type": "generate", "tool": "groq_generator"},
            {"type": "critic", "tool": "policy_guard"},
        ],
    }


def run_agentic_query(
    query: str,
    *,
    user_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    del history

    plan = build_plan(query, user_id=user_id)

    retrieval_result = vector_search.retrieve(query, user_id=user_id, strategy=plan["strategy"])
    reranked = rerank.rerank(retrieval_result["documents"], retrieval_result.get("trace"))
    reasoning = graph_query.validate_trace_paths(reranked.get("trace"))

    ai_result = services.generate_groq_response(
        query,
        reranked.get("documents") or [],
        user_id=user_id,
        retrieval_trace=reranked.get("trace"),
    )

    trace = {**(reranked.get("trace") or {}), **((ai_result.get("trace") or {}) if isinstance(ai_result, dict) else {})}
    critic = policy_guard.evaluate_answer(
        query=query,
        answer=ai_result.get("answer") or "",
        answer_payload=ai_result.get("answer_payload") or {},
        trace=trace,
        plan=plan,
    )

    trace["agentic"] = {
        "enabled": True,
        "planner": plan,
        "retriever": {
            "documents": len(reranked.get("documents") or []),
            "strategy": plan["strategy"],
            "selector_reasons": plan["selector"]["reasons"],
        },
        "reasoner": reasoning,
        "generator": {
            "answer_mode": (ai_result.get("answer_payload") or {}).get("mode"),
            "reason_code": (ai_result.get("answer_payload") or {}).get("reason_code"),
        },
        "critic": critic,
        "status": "passed" if critic["passed"] else "needs_review",
    }

    thinking = list(ai_result.get("thinking") or [])
    thinking.extend(
        [
            f"Planner selected {plan['strategy']} retrieval.",
            f"Reasoner validated {reasoning['validated_evidence_count']} evidence paths.",
            f"Critic verdict: {'pass' if critic['passed'] else 'review'}",
        ]
    )

    return {
        "answer": ai_result.get("answer"),
        "answer_payload": ai_result.get("answer_payload"),
        "thinking": thinking,
        "trace": trace,
    }
