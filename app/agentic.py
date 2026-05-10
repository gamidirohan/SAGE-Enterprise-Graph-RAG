"""Agentic query orchestration for planner / retriever / reasoner / generator / critic."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import re
import time
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

try:
    import app.graph_query as graph_query
    import app.policy_guard as policy_guard
    import app.query_shape as query_shape
    import app.rerank as rerank
    import app.retrieval_selector as retrieval_selector
    import app.services as services
    import app.vector_search as vector_search
except ImportError:  # pragma: no cover - direct execution fallback
    import graph_query
    import policy_guard
    import query_shape
    import rerank
    import retrieval_selector
    import services
    import vector_search


AGENT_REGISTRY = {
    "planner": "PlannerAgent",
    "retrieval": "RetrievalAgent",
    "reasoner": "ReasonerAgent",
    "generator": "GeneratorAgent",
    "critic": "CriticAgent",
    "saia": "SAIAAgent",
}


AgentEventSink = Callable[[Dict[str, Any]], None]


_AMOUNT_PATTERN = re.compile(
    r"(?P<currency>₹|rs\.?|inr|\$|usd|eur|€)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>lakhs?|lacs?|crores?|k|m|million|billion)?",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"\b(?:q[1-4]\s+\d{4}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
_QUOTED_PATTERN = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,80})[\"“”'‘’]")
_CAPITALIZED_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4}\b")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in event.items() if value is not None}


def _emit_event(
    state: Dict[str, Any],
    event_sink: Optional[AgentEventSink],
    *,
    event_type: str,
    agent: str,
    stage: str,
    status: str,
    message: str,
    tool: Optional[str] = None,
    attempt: Optional[int] = None,
    duration_ms: Optional[int] = None,
    result_count: Optional[int] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    state["_event_counter"] = int(state.get("_event_counter") or 0) + 1
    event = _sanitize_event(
        {
            "event_id": f"{state['run_id']}-{state['_event_counter']}",
            "run_id": state["run_id"],
            "timestamp": _utc_timestamp(),
            "event_type": event_type,
            "agent": agent,
            "stage": stage,
            "status": status,
            "message": message,
            "tool": tool,
            "attempt": attempt,
            "duration_ms": duration_ms,
            "result_count": result_count,
            "error": error,
        }
    )
    state.setdefault("events", []).append(event)
    state.setdefault("route_history", []).append(event)
    if status == "running":
        state["current_agent"] = agent
    if event_sink:
        event_sink(event)
    return event


def _unique_strings(values: List[str], *, limit: int = 12) -> List[str]:
    seen = set()
    unique: List[str] = []
    for value in values:
        normalized = " ".join(str(value).strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
        if len(unique) >= limit:
            break
    return unique


def _extract_entities(query: str) -> List[str]:
    quoted = [match.group(1) for match in _QUOTED_PATTERN.finditer(query)]
    capitalized = [
        match.group(0)
        for match in _CAPITALIZED_ENTITY_PATTERN.finditer(query)
        if match.group(0).lower() not in {"i", "who", "what", "why", "when", "where", "how", "list"}
    ]
    identifiers = re.findall(r"\b[A-Z]{2,}[-_][A-Za-z0-9][A-Za-z0-9_-]*\b", query)
    return _unique_strings([*quoted, *identifiers, *capitalized])


def _extract_constraints(query: str) -> Dict[str, Any]:
    lowered = query.lower()
    constraints: Dict[str, Any] = {
        "temporal": _unique_strings([match.group(0) for match in _DATE_PATTERN.finditer(query)], limit=6),
        "amounts": [],
        "comparators": [],
        "policy_terms": [],
    }
    for match in _AMOUNT_PATTERN.finditer(query):
        value = match.group("value")
        if not value:
            continue
        token = match.group(0).strip()
        if token and any(char.isdigit() for char in token):
            constraints["amounts"].append(token)
    constraints["amounts"] = _unique_strings(constraints["amounts"], limit=6)

    if any(token in lowered for token in ("above", "over", "greater than", "more than", ">")):
        constraints["comparators"].append("greater_than")
    if any(token in lowered for token in ("below", "under", "less than", "<")):
        constraints["comparators"].append("less_than")
    if any(token in lowered for token in ("between", "from", "to", "during")):
        constraints["comparators"].append("range")
    if "policy" in lowered or "compliance" in lowered or "violat" in lowered:
        constraints["policy_terms"] = _unique_strings(
            re.findall(r"\b[\w-]*policy[\w-]*\b|\bcompliance\b|\bviolat\w*\b|\bescalat\w*\b", lowered),
            limit=8,
        )
    return {key: value for key, value in constraints.items() if value}


def _normalize_signature_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _evidence_signature(item: Dict[str, Any]) -> str:
    fact = item.get("fact") or {}
    document = item.get("document") or {}
    parts = [
        item.get("fact_summary"),
        item.get("chunk_summary"),
        fact.get("canonical_key"),
        fact.get("subject_entity_id"),
        fact.get("subject_key"),
        fact.get("object_entity_id"),
        fact.get("object_key"),
        item.get("relationship"),
        document.get("subject"),
    ]
    normalized_parts = [_normalize_signature_text(part) for part in parts if _normalize_signature_text(part)]
    if normalized_parts:
        return " | ".join(normalized_parts[:4])
    if item.get("fact_id"):
        return f"fact:{item['fact_id']}"
    if item.get("chunk_id"):
        return f"chunk:{item['chunk_id']}"
    if document.get("doc_id"):
        return f"doc:{document['doc_id']}"
    return f"unknown:{hash(str(item))}"


def _trace_coverage(
    trace: Dict[str, Any],
    reasoning: Dict[str, Any],
    *,
    plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence = list((trace or {}).get("evidence") or [])
    signatures = {_evidence_signature(item) for item in evidence}
    profile = dict(
        ((trace or {}).get("query_profile") or {})
        or ((plan or {}).get("query_profile") or {})
        or {}
    )
    return {
        "distinct_evidence_count": len(signatures),
        "validated_evidence_count": int(reasoning.get("validated_evidence_count") or 0),
        "has_fact": any(item.get("fact_id") for item in evidence),
        "has_supporting_doc_or_chunk": any(
            item.get("chunk_id") or (item.get("document") or {}).get("doc_id")
            for item in evidence
        ),
        "expects_multiple_items": bool(profile.get("expects_multiple_items")),
        "requires_broad_coverage": bool(profile.get("requires_broad_coverage")),
        "minimum_unique_evidence": int(profile.get("minimum_unique_evidence") or 1),
        "minimum_tool_rounds": int(profile.get("minimum_tool_rounds") or 1),
    }


def _infer_intent(query: str, constraints: Dict[str, Any]) -> str:
    lowered = query.lower()
    if constraints.get("policy_terms") or any(token in lowered for token in ("audit", "compliance", "violation", "violated")):
        return "policy_or_compliance_reasoning"
    if any(token in lowered for token in ("why", "because", "delayed", "blocked", "root cause", "responsible")):
        return "causal_or_responsibility_reasoning"
    if any(token in lowered for token in ("who", "manager", "approved", "approval", "owner")):
        return "entity_relationship_lookup"
    if any(token in lowered for token in ("list", "show", "find all", "which")):
        return "filtered_evidence_search"
    if any(token in lowered for token in ("summarize", "explain", "compare")):
        return "synthesis"
    return "general_graph_rag"


def _required_evidence_for(query: str, intent: str, constraints: Dict[str, Any]) -> List[str]:
    lowered = query.lower()
    required = ["source_document_or_chunk", "stable_evidence_ref"]
    if intent in {"policy_or_compliance_reasoning", "entity_relationship_lookup", "causal_or_responsibility_reasoning"}:
        required.append("graph_or_fact_binding")
    if constraints.get("policy_terms"):
        required.append("policy_reference")
    if constraints.get("temporal") or any(token in lowered for token in ("when", "date", "q1", "q2", "q3", "q4")):
        required.append("timestamp_or_temporal_scope")
    if constraints.get("amounts"):
        required.append("amount_or_numeric_property")
    return _unique_strings(required)


def _risk_flags_for(query: str, constraints: Dict[str, Any]) -> List[str]:
    lowered = query.lower()
    flags = []
    if constraints.get("policy_terms"):
        flags.append("policy_sensitive")
    if any(token in lowered for token in ("only", "final", "all", "every", "violated", "non-compliant")):
        flags.append("requires_exhaustive_or_finality_check")
    if constraints.get("amounts") or constraints.get("temporal"):
        flags.append("requires_structured_filtering")
    if any(token in lowered for token in ("who", "responsible", "manager", "approved")):
        flags.append("requires_identity_grounding")
    return flags


def _route_family_for(query: str, intent: str, query_profile: Dict[str, Any]) -> str:
    lowered = query.lower()
    if intent == "policy_or_compliance_reasoning":
        return "policy_reasoning"
    if intent in {"causal_or_responsibility_reasoning", "entity_relationship_lookup"}:
        return "relationship_lookup"
    if query_profile.get("expects_multiple_items") or query_profile.get("requires_broad_coverage"):
        return "broad_synthesis"
    if any(token in lowered for token in ("when", "date", "timeline", "schedule", "review", "meeting")):
        return "temporal_lookup"
    if intent == "filtered_evidence_search":
        return "filtered_search"
    if intent == "synthesis":
        return "summary_synthesis"
    return "targeted_lookup"


def _orchestration_contract_for(
    query: str,
    *,
    intent: str,
    query_profile: Dict[str, Any],
    selector_strategy: str,
) -> Dict[str, Any]:
    route_family = _route_family_for(query, intent, query_profile)
    return {
        "route_family": route_family,
        "planner_required": True,
        "retriever_required": True,
        "reasoner_required": True,
        "generator_required": True,
        "critic_required": True,
        "tool_owner": {
            "semantic": "retriever",
            "fulltext": "retriever",
            "graph": "retriever",
        },
        "validation_owner": "reasoner",
        "safety_owner": "critic",
        "memory_sources": ["history", "retrieval_trace", "plan"],
        "can_short_circuit": not bool(query_profile.get("expects_multiple_items") or query_profile.get("requires_broad_coverage")),
        "selector_strategy": selector_strategy,
    }


def _tool_plan_for(tool_sequence: List[str]) -> List[Dict[str, str]]:
    purposes = {
        "semantic": "Find conceptually similar chunks and facts.",
        "fulltext": "Run BM25/full-text search for exact names, IDs, dates, policy terms, and amounts.",
        "graph": "Expand graph context and validate node/path provenance.",
    }
    return [{"tool": tool_name, "purpose": purposes.get(tool_name, "Retrieve supporting evidence.")} for tool_name in tool_sequence]


def _invoke_with_supported_kwargs(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(*args, **kwargs)

    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(*args, **supported_kwargs)


def _planned_graph_depth(query: str, *, query_profile: Dict[str, Any], query_type: str) -> Dict[str, Any]:
    depth = query_shape.recommend_graph_depth(
        query,
        query_profile=query_profile,
        query_type=query_type,
    )
    return {
        "seed_hops": int(depth.get("seed_hops") or 0),
        "expand_hops": int(depth.get("expand_hops") or 1),
        "max_hops": int(depth.get("max_hops") or 1),
        "reason": str(depth.get("reason") or "direct_lookup"),
    }


def _active_depth_for_tool(state: Dict[str, Any], tool_name: str) -> Dict[str, int]:
    planned = dict((state.get("graph_depth") or {}) if isinstance(state.get("graph_depth"), dict) else {})
    seed_hops = int(planned.get("seed_hops") or 0)
    expand_hops = int(planned.get("expand_hops") or 1)
    max_hops = max(int(planned.get("max_hops") or expand_hops or 1), 1)

    if tool_name != "graph":
        return {"seed_hops": seed_hops, "expand_hops": expand_hops, "effective_hops": seed_hops}

    prior_graph_calls = sum(1 for call in state.get("tool_calls") or [] if call.get("tool") == "graph")
    effective_expand_hops = min(expand_hops + prior_graph_calls, max_hops)
    return {"seed_hops": seed_hops, "expand_hops": expand_hops, "effective_hops": effective_expand_hops}


def build_plan(query: str, *, user_id: Optional[str] = None) -> Dict[str, Any]:
    selector = retrieval_selector.decide_strategy(query, user_id=user_id)
    entities = _extract_entities(query)
    extracted_constraints = _extract_constraints(query)
    intent = _infer_intent(query, extracted_constraints)
    query_profile = query_shape.analyze_query(query)
    query_type = services._classify_query(query)
    graph_depth = _planned_graph_depth(query, query_profile=query_profile, query_type=query_type)
    risk_flags = _risk_flags_for(query, extracted_constraints)
    orchestration = _orchestration_contract_for(
        query,
        intent=intent,
        query_profile=query_profile,
        selector_strategy=selector["strategy"],
    )
    if query_profile.get("expects_multiple_items"):
        risk_flags.append("requires_answer_diversity")
    # Enforce both semantic and BM25 for exact-term / policy queries to satisfy playbook.
    default_steps = ["semantic", "fulltext", "graph"]
    if selector["strategy"] == "fulltext":
        initial_steps = ["fulltext", "semantic", "graph"]
    elif selector["strategy"] == "semantic":
        initial_steps = ["semantic", "fulltext", "graph"]
    else:
        initial_steps = default_steps
    return {
        "planner": "GAP",
        "intent": intent,
        "entities": entities,
        "extracted_constraints": extracted_constraints,
        "query_profile": query_profile,
        "required_evidence": _required_evidence_for(query, intent, extracted_constraints),
        "risk_flags": risk_flags,
        "strategy": selector["strategy"],
        "selector": selector,
        "graph_depth": graph_depth,
        "orchestration": orchestration,
        "agents": list(AGENT_REGISTRY.values()),
        "constraints": {
            "allowed_nodes": graph_query.schema_snapshot()["node_types"],
            "allowed_edges": graph_query.schema_snapshot()["edge_types"],
            "max_depth": graph_depth["max_hops"],
            "max_rounds": 3,
            "max_retries": 1,
        },
        "tool_sequence": initial_steps,
        "tool_plan": _tool_plan_for(initial_steps),
        "stop_conditions": [
            "grounded_evidence_threshold",
            "one_canonical_fact_plus_supporting_source_for_single_lookup",
            "distinct_evidence_coverage_for_multi_item_queries",
            "minimum_tool_rounds_for_broad_coverage_queries",
            "round_budget_exhausted",
        ],
        "steps": [
            *[
                {"type": "retrieve", "tool": tool_name, "query": query}
                for tool_name in initial_steps
            ],
            {"type": "rerank", "tool": "score_sort"},
            {"type": "validate_paths", "tool": "graph_path_validator"},
            {"type": "generate", "tool": "groq_generator"},
            {"type": "critic", "tool": "policy_guard"},
        ],
    }


def _blank_trace(*, query: str, user_id: Optional[str]) -> Dict[str, Any]:
    return {
        "query": query,
        "query_type": "general_search",
        "user_scoped": bool(user_id),
        "user_id": user_id,
        "graph_depth": None,
        "matched_entities": [],
        "result_count": 0,
        "max_hop_count": 0,
        "retrieval_path": None,
        "evidence": [],
        "no_evidence": True,
        "evidence_state": "no_evidence",
        "query_profile": query_shape.analyze_query(query),
    }


def _initial_state(
    query: str,
    *,
    user_id: Optional[str],
    history: Optional[List[Dict[str, str]]],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": f"agentic-{uuid4().hex[:12]}",
        "query": query,
        "user_id": user_id,
        "history": list(history or []),
        "plan": plan,
        "graph_depth": dict((plan or {}).get("graph_depth") or {}),
        "trace": _blank_trace(query=query, user_id=user_id),
        "documents": [],
        "rounds": [],
        "tool_calls": [],
        "events": [],
        "route_history": [],
        "current_agent": None,
        "retry_count": 0,
        "stop_reason": None,
    }


def _record_tool_call(
    state: Dict[str, Any],
    *,
    tool: str,
    attempt: int,
    status: str,
    result_count: int,
    duration_ms: int,
    depth: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    state["tool_calls"].append(
        {
            "tool": tool,
            "attempt": attempt,
            "query": state["query"],
            "depth": depth,
            "status": status,
            "result_count": result_count,
            "duration_ms": duration_ms,
            "error": error,
        }
    )


def _enough_context(
    trace: Dict[str, Any],
    reasoning: Dict[str, Any],
    *,
    plan: Optional[Dict[str, Any]] = None,
    attempt: int = 1,
) -> bool:
    coverage = _trace_coverage(trace, reasoning, plan=plan)
    if coverage["requires_broad_coverage"] and attempt < coverage["minimum_tool_rounds"]:
        return False
    planned_graph_depth = int((((plan or {}).get("graph_depth") or {}).get("expand_hops")) or 0)
    if (
        coverage["requires_broad_coverage"]
        and planned_graph_depth >= 3
        and "graph" in list((plan or {}).get("tool_sequence") or [])
        and (trace or {}).get("selector_strategy") != "graph"
    ):
        return False
    if coverage["expects_multiple_items"]:
        return (
            coverage["distinct_evidence_count"] >= coverage["minimum_unique_evidence"]
            and coverage["validated_evidence_count"] >= coverage["minimum_unique_evidence"]
        )
    if (trace or {}).get("query_type") in services.FACT_PRIORITY_QUERY_TYPES and not coverage["has_fact"]:
        return False
    if coverage["has_fact"] and coverage["has_supporting_doc_or_chunk"]:
        return True
    evidence_refs = services._derive_evidence_refs(retrieval_trace=trace, limit=8)
    unique_refs = len(set(evidence_refs))
    return unique_refs >= 2 and coverage["validated_evidence_count"] >= 1


def _run_retrieval_round(
    state: Dict[str, Any],
    *,
    tool_name: str,
    attempt: int,
    event_sink: Optional[AgentEventSink] = None,
) -> Dict[str, Any]:
    tool_label = "BM25" if tool_name == "fulltext" else tool_name.title()
    active_depth = _active_depth_for_tool(state, tool_name)
    effective_hops = int(active_depth.get("effective_hops") or 0)
    _emit_event(
        state,
        event_sink,
        event_type="agent_started",
        agent="retriever",
        stage="retrieve",
        status="running",
        message=f"Retriever selected {tool_label} evidence search at depth {effective_hops}.",
        tool=tool_name,
        attempt=attempt,
    )
    _emit_event(
        state,
        event_sink,
        event_type="tool_started",
        agent=tool_name,
        stage="retrieve",
        status="running",
        message=f"{tool_label} tool is gathering candidate evidence.",
        tool=tool_name,
        attempt=attempt,
    )
    started_at = time.perf_counter()
    try:
        if tool_name == "graph":
            result = _invoke_with_supported_kwargs(
                graph_query.expand_retrieval_context,
                state["query"],
                seed_trace=state.get("trace"),
                user_id=state.get("user_id"),
                expand_hops=effective_hops,
            )
        else:
            result = _invoke_with_supported_kwargs(
                vector_search.retrieve,
                state["query"],
                user_id=state.get("user_id"),
                strategy=tool_name,
                seed_trace=state.get("trace"),
                context_hops=effective_hops,
            )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        _record_tool_call(
            state,
            tool=tool_name,
            attempt=attempt,
            status="failed",
            result_count=0,
            duration_ms=duration_ms,
            depth=effective_hops,
            error=str(exc),
        )
        _emit_event(
            state,
            event_sink,
            event_type="tool_finished",
            agent=tool_name,
            stage="retrieve",
            status="failed",
            message=f"{tool_label} tool failed before returning evidence.",
            tool=tool_name,
            attempt=attempt,
            duration_ms=duration_ms,
            result_count=0,
            error=str(exc),
        )
        _emit_event(
            state,
            event_sink,
            event_type="agent_finished",
            agent="retriever",
            stage="retrieve",
            status="failed",
            message="Retriever could not complete this round.",
            tool=tool_name,
            attempt=attempt,
            duration_ms=duration_ms,
            result_count=0,
            error=str(exc),
        )
        raise
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    raw_result_count = len(result.get("documents") or [])
    _emit_event(
        state,
        event_sink,
        event_type="tool_finished",
        agent=tool_name,
        stage="retrieve",
        status="completed",
        message=f"{tool_label} tool returned {raw_result_count} candidate document(s).",
        tool=tool_name,
        attempt=attempt,
        duration_ms=duration_ms,
        result_count=raw_result_count,
    )
    merged = vector_search.merge_results(
        {
            "documents": list(state.get("documents") or []),
            "trace": dict(state.get("trace") or {}),
        },
        result,
        limit=6,
    )
    _emit_event(
        state,
        event_sink,
        event_type="agent_finished",
        agent="retriever",
        stage="retrieve",
        status="completed",
        message=f"Retriever merged evidence from {tool_label}.",
        tool=tool_name,
        attempt=attempt,
        duration_ms=duration_ms,
        result_count=len(merged.get("documents") or []),
    )
    _emit_event(
        state,
        event_sink,
        event_type="agent_started",
        agent="reranker",
        stage="rerank",
        status="running",
        message="Reranker is ordering candidate evidence by relevance.",
        attempt=attempt,
    )
    reranked = rerank.rerank(merged.get("documents") or [], merged.get("trace"))
    _emit_event(
        state,
        event_sink,
        event_type="agent_finished",
        agent="reranker",
        stage="rerank",
        status="completed",
        message=f"Reranker kept {len(reranked.get('documents') or [])} document(s) for reasoning.",
        attempt=attempt,
        result_count=len(reranked.get("documents") or []),
    )
    _emit_event(
        state,
        event_sink,
        event_type="agent_started",
        agent="reasoner",
        stage="validate_paths",
        status="running",
        message="Reasoner is checking graph/fact/document bindings.",
        attempt=attempt,
    )
    reasoning = graph_query.validate_trace_paths(reranked.get("trace"))
    trace_with_profile = dict(reranked.get("trace") or {})
    trace_with_profile["query_profile"] = (
        (state.get("plan") or {}).get("query_profile")
        or trace_with_profile.get("query_profile")
        or query_shape.analyze_query(state.get("query") or "")
    )
    coverage = _trace_coverage(trace_with_profile, reasoning, plan=state.get("plan"))
    trace_with_profile["coverage"] = coverage
    enough_context = _enough_context(trace_with_profile, reasoning, plan=state.get("plan"), attempt=attempt)

    state["documents"] = reranked.get("documents") or []
    state["trace"] = trace_with_profile
    state["trace"]["graph_depth"] = {
        **dict(state.get("graph_depth") or {}),
        "effective_hops": effective_hops,
        "tool": tool_name,
    }
    state["rounds"].append(
        {
            "attempt": attempt,
            "tool": tool_name,
            "depth": effective_hops,
            "result_count": int((state["trace"] or {}).get("result_count") or 0),
            "evidence_ref_count": len(services._derive_evidence_refs(retrieval_trace=state["trace"], limit=8)),
            "distinct_evidence_count": int(((state["trace"] or {}).get("coverage") or {}).get("distinct_evidence_count") or 0),
            "validated_evidence_count": int(reasoning.get("validated_evidence_count") or 0),
            "enough_context": enough_context,
        }
    )
    _record_tool_call(
        state,
        tool=tool_name,
        attempt=attempt,
        status="ok",
        result_count=int((state["trace"] or {}).get("result_count") or 0),
        duration_ms=duration_ms,
        depth=effective_hops,
    )
    evidence_state = (state["trace"] or {}).get("evidence_state") or ("grounded" if enough_context else "partial_evidence")
    _emit_event(
        state,
        event_sink,
        event_type="agent_finished",
        agent="reasoner",
        stage="validate_paths",
        status="completed",
        message=(
            f"Reasoner validated {reasoning.get('validated_evidence_count') or 0} evidence binding(s); "
            f"state is {evidence_state}."
        ),
        attempt=attempt,
        result_count=int(reasoning.get("validated_evidence_count") or 0),
    )
    return reasoning


def _choose_retry_tool(plan: Dict[str, Any], state: Dict[str, Any], critic: Dict[str, Any]) -> str:
    used = [entry.get("tool") for entry in state.get("tool_calls") or []]
    issues = set(critic.get("issues") or [])
    preferred_order = ["graph", "fulltext", "semantic"]
    if "missing_policy_provenance" in issues:
        preferred_order = ["fulltext", "semantic", "graph"]
    if "missing_exact_match" in issues:
        preferred_order = ["fulltext", "semantic", "graph"]
    if "insufficient_answer_coverage" in issues:
        preferred_order = ["fulltext", "graph", "semantic"]
    for tool_name in preferred_order:
        if tool_name not in used:
            return tool_name
    return plan.get("tool_sequence", ["graph"])[-1]


def _legacy_run_agentic_query(
    query: str,
    *,
    user_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    event_sink: Optional[AgentEventSink] = None,
) -> Dict[str, Any]:
    state = _initial_state(query, user_id=user_id, history=history, plan={})
    _emit_event(
        state,
        event_sink,
        event_type="run_started",
        agent="orchestrator",
        stage="start",
        status="running",
        message="SAGE started an agentic reasoning run.",
    )
    _emit_event(
        state,
        event_sink,
        event_type="agent_started",
        agent="planner",
        stage="plan",
        status="running",
        message="Planner is identifying intent, entities, constraints, and evidence needs.",
    )
    try:
        plan = build_plan(query, user_id=user_id)
        state["plan"] = plan
        state["graph_depth"] = dict(plan.get("graph_depth") or {})
        state["trace"]["query_profile"] = plan.get("query_profile") or state["trace"].get("query_profile")
        state["trace"]["graph_depth"] = dict(plan.get("graph_depth") or {})
        _emit_event(
            state,
            event_sink,
            event_type="agent_progress",
            agent="planner",
            stage="plan",
            status="running",
            message=(
                f"Planner inferred {plan.get('intent') or 'general_graph_rag'} and selected "
                f"{', '.join(plan.get('tool_sequence') or [])}."
            ),
        )
        _emit_event(
            state,
            event_sink,
            event_type="agent_finished",
            agent="planner",
            stage="plan",
            status="completed",
            message=f"Planner prepared {len(plan.get('steps') or [])} execution step(s).",
            result_count=len(plan.get("steps") or []),
        )
    except Exception as exc:
        _emit_event(
            state,
            event_sink,
            event_type="run_failed",
            agent="orchestrator",
            stage="plan",
            status="failed",
            message="SAGE could not build an execution plan.",
            error=str(exc),
        )
        raise
    reasoning: Dict[str, Any] = {"valid": True, "validated_evidence_count": 0, "missing_fields": []}

    try:
        for attempt, tool_name in enumerate(plan.get("tool_sequence") or [], start=1):
            reasoning = _run_retrieval_round(state, tool_name=tool_name, attempt=attempt, event_sink=event_sink)
            if state["rounds"][-1]["enough_context"]:
                state["stop_reason"] = "enough_context"
                _emit_event(
                    state,
                    event_sink,
                    event_type="agent_progress",
                    agent="orchestrator",
                    stage="stop_check",
                    status="running",
                    message="SAGE has enough grounded context to draft an answer.",
                    attempt=attempt,
                )
                break

        if not state["stop_reason"]:
            state["stop_reason"] = "round_budget_exhausted"
            _emit_event(
                state,
                event_sink,
                event_type="agent_progress",
                agent="orchestrator",
                stage="stop_check",
                status="running",
                message="SAGE reached the retrieval round budget and will answer with available evidence.",
            )
    except Exception as exc:
        _emit_event(
            state,
            event_sink,
            event_type="run_failed",
            agent="orchestrator",
            stage="retrieve",
            status="failed",
            message="SAGE failed while retrieving or validating evidence.",
            error=str(exc),
        )
        raise

    _emit_event(
        state,
        event_sink,
        event_type="agent_started",
        agent="generator",
        stage="generate",
        status="running",
        message="Generator is drafting an answer from validated evidence.",
    )
    started_at = time.perf_counter()
    ai_result = services.generate_groq_response(
        query,
        state.get("documents") or [],
        user_id=user_id,
        retrieval_trace=state.get("trace"),
    )
    generator_duration_ms = int((time.perf_counter() - started_at) * 1000)
    _emit_event(
        state,
        event_sink,
        event_type="agent_finished",
        agent="generator",
        stage="generate",
        status="completed",
        message="Generator produced a grounded draft answer.",
        duration_ms=generator_duration_ms,
    )
    trace = {
        **(state.get("trace") or {}),
        **((ai_result.get("trace") or {}) if isinstance(ai_result, dict) else {}),
    }
    _emit_event(
        state,
        event_sink,
        event_type="agent_started",
        agent="critic",
        stage="critic",
        status="running",
        message="Critic is checking grounding, citations, and policy-sensitive requirements.",
    )
    critic_started_at = time.perf_counter()
    critic = policy_guard.evaluate_answer(
        query=query,
        answer=ai_result.get("answer") or "",
        answer_payload=ai_result.get("answer_payload") or {},
        trace=trace,
        plan=plan,
    )
    _emit_event(
        state,
        event_sink,
        event_type="agent_finished",
        agent="critic",
        stage="critic",
        status="completed" if critic.get("passed") else "needs_review",
        message="Critic passed the answer." if critic.get("passed") else "Critic requested stronger grounding.",
        duration_ms=int((time.perf_counter() - critic_started_at) * 1000),
        result_count=int(critic.get("grounded_evidence_count") or 0),
    )

    if critic.get("retryable") and not critic.get("passed") and state["retry_count"] < int(plan["constraints"]["max_retries"]):
        retry_tool = _choose_retry_tool(plan, state, critic)
        state["retry_count"] += 1
        _emit_event(
            state,
            event_sink,
            event_type="retry_started",
            agent="orchestrator",
            stage="retry",
            status="running",
            message=f"Critic triggered one retry using {retry_tool}.",
            tool=retry_tool,
            attempt=len(state.get("rounds") or []) + 1,
        )
        reasoning = _run_retrieval_round(
            state,
            tool_name=retry_tool,
            attempt=len(state.get("rounds") or []) + 1,
            event_sink=event_sink,
        )
        state["stop_reason"] = f"critic_retry:{retry_tool}"
        _emit_event(
            state,
            event_sink,
            event_type="agent_started",
            agent="generator",
            stage="generate",
            status="running",
            message="Generator is revising the answer after retry evidence.",
        )
        started_at = time.perf_counter()
        ai_result = services.generate_groq_response(
            query,
            state.get("documents") or [],
            user_id=user_id,
            retrieval_trace=state.get("trace"),
        )
        _emit_event(
            state,
            event_sink,
            event_type="agent_finished",
            agent="generator",
            stage="generate",
            status="completed",
            message="Generator produced a revised grounded answer.",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
        )
        trace = {
            **(state.get("trace") or {}),
            **((ai_result.get("trace") or {}) if isinstance(ai_result, dict) else {}),
        }
        _emit_event(
            state,
            event_sink,
            event_type="agent_started",
            agent="critic",
            stage="critic",
            status="running",
            message="Critic is re-checking the revised answer.",
        )
        critic_started_at = time.perf_counter()
        critic = policy_guard.evaluate_answer(
            query=query,
            answer=ai_result.get("answer") or "",
            answer_payload=ai_result.get("answer_payload") or {},
            trace=trace,
            plan=plan,
        )
        _emit_event(
            state,
            event_sink,
            event_type="agent_finished",
            agent="critic",
            stage="critic",
            status="completed" if critic.get("passed") else "needs_review",
            message="Critic passed the revised answer." if critic.get("passed") else "Critic still found grounding gaps.",
            duration_ms=int((time.perf_counter() - critic_started_at) * 1000),
            result_count=int(critic.get("grounded_evidence_count") or 0),
        )

    trace["agentic"] = {
        "enabled": True,
        "run_id": state["run_id"],
        "planner": plan,
        "rounds": list(state.get("rounds") or []),
        "tool_calls": list(state.get("tool_calls") or []),
        "events": list(state.get("events") or []),
        "route_history": list(state.get("route_history") or []),
        "current_agent": state.get("current_agent"),
        "stop_reason": state.get("stop_reason"),
        "reasoner": reasoning,
        "generator": {
            "answer_mode": (ai_result.get("answer_payload") or {}).get("mode"),
            "reason_code": (ai_result.get("answer_payload") or {}).get("reason_code"),
        },
        "critic": critic,
        "status": "passed" if critic["passed"] else "needs_review",
    }
    _emit_event(
        state,
        event_sink,
        event_type="run_finished",
        agent="orchestrator",
        stage="finish",
        status="completed" if critic.get("passed") else "needs_review",
        message="SAGE finished the agentic reasoning run.",
    )
    trace["agentic"]["events"] = list(state.get("events") or [])
    trace["agentic"]["route_history"] = list(state.get("route_history") or [])
    trace["agentic"]["current_agent"] = None

    thinking = list(ai_result.get("thinking") or [])
    tools_used = ", ".join(entry.get("tool") or "unknown" for entry in state.get("tool_calls") or [])
    thinking.extend(
        [
            f"Planner selected {plan['strategy']} retrieval.",
            f"Retriever used: {tools_used or plan['strategy']}.",
            f"Reasoner validated {reasoning['validated_evidence_count']} evidence bindings.",
            f"Critic verdict: {'pass' if critic['passed'] else 'review'}",
        ]
    )

    return {
        "answer": ai_result.get("answer"),
        "answer_payload": ai_result.get("answer_payload"),
        "thinking": thinking,
        "trace": trace,
    }


def run_agentic_query(
    query: str,
    *,
    user_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    event_sink: Optional[AgentEventSink] = None,
) -> Dict[str, Any]:
    try:
        import app.orchestrator as orchestrator
    except ImportError:  # pragma: no cover - direct execution fallback
        import orchestrator

    return orchestrator.run_agentic_query(
        query,
        user_id=user_id,
        history=history,
        event_sink=event_sink,
    )
