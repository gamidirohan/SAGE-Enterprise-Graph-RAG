"""State-driven orchestration loop for agentic Graph-RAG queries."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

try:
    import app.agentic as agentic
    import app.orchestrator_logging as orchestrator_logging
except ImportError:  # pragma: no cover - direct execution fallback
    import agentic
    import orchestrator_logging


@dataclass
class AgentAction:
    """One orchestrator-selected action."""

    kind: str
    agent: str
    stage: str
    tool: Optional[str] = None
    reason: str = ""


@dataclass
class AgentObservation:
    """Result of an action that mutates the shared run state."""

    action: AgentAction
    status: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorState:
    """Shared state owned by the orchestrator and exposed in answer provenance."""

    query: str
    user_id: Optional[str]
    history: List[Dict[str, Any]]
    event_sink: Optional[agentic.AgentEventSink]
    state: Dict[str, Any]
    reasoning: Dict[str, Any] = field(default_factory=lambda: {"valid": True, "validated_evidence_count": 0, "missing_fields": []})
    ai_result: Dict[str, Any] = field(default_factory=dict)
    critic: Dict[str, Any] = field(default_factory=dict)
    next_tool_index: int = 0
    phase: str = "start"

    def record_step(self, *, action: AgentAction, status: str, summary: str, data: Optional[Dict[str, Any]] = None) -> None:
        steps = self.state.setdefault("completed_steps", [])
        steps.append(
            {
                "index": len(steps) + 1,
                "agent": action.agent,
                "stage": action.stage,
                "action": action.kind,
                "tool": action.tool,
                "status": status,
                "summary": summary,
                "data": data or {},
            }
        )
        self.state["current_step"] = {
            "agent": action.agent,
            "stage": action.stage,
            "action": action.kind,
            "tool": action.tool,
            "status": status,
            "summary": summary,
        }

    def update_context_state(self) -> None:
        trace = self.state.get("trace") or {}
        evidence = list(trace.get("evidence") or [])
        coverage = dict(trace.get("coverage") or {})
        latest_round = (self.state.get("rounds") or [{}])[-1]

        self.state["selected_evidence"] = evidence
        self.state["evidence_pool"] = _merge_evidence_pool(self.state.get("evidence_pool") or [], evidence)
        self.state["validated_bindings"] = list(self.reasoning.get("validated_bindings") or [])
        self.state["coverage_status"] = _coverage_status(trace, self.reasoning, latest_round)
        self.state["open_questions"] = _open_questions(trace, self.reasoning, coverage)
        self.state["running_summary"] = _running_summary(self.state)


def _merge_evidence_pool(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_signature: Dict[str, Dict[str, Any]] = {}
    for item in existing + incoming:
        by_signature[agentic._evidence_signature(item)] = dict(item)
    return list(by_signature.values())


def _coverage_status(trace: Dict[str, Any], reasoning: Dict[str, Any], latest_round: Dict[str, Any]) -> Dict[str, Any]:
    coverage = dict((trace or {}).get("coverage") or {})
    if latest_round.get("enough_context"):
        status = "sufficient"
    elif not (trace or {}).get("evidence"):
        status = "insufficient"
    else:
        status = "partial"

    return {
        "status": status,
        "distinct_evidence_count": int(coverage.get("distinct_evidence_count") or 0),
        "validated_evidence_count": int(reasoning.get("validated_evidence_count") or 0),
        "expects_multiple_items": bool(coverage.get("expects_multiple_items")),
        "minimum_unique_evidence": int(coverage.get("minimum_unique_evidence") or 1),
    }


def _open_questions(trace: Dict[str, Any], reasoning: Dict[str, Any], coverage: Dict[str, Any]) -> List[str]:
    questions: List[str] = []
    if not (trace or {}).get("evidence"):
        questions.append("No grounded evidence has been retrieved yet.")

    minimum = int(coverage.get("minimum_unique_evidence") or 1)
    distinct_count = int(coverage.get("distinct_evidence_count") or 0)
    if coverage.get("expects_multiple_items") and distinct_count < minimum:
        questions.append(f"Need at least {minimum} distinct evidence item(s); found {distinct_count}.")

    if reasoning.get("invalid_refs"):
        questions.append("Some evidence items are missing graph, fact, or document bindings.")
    return questions


def _running_summary(state: Dict[str, Any]) -> str:
    rounds = list(state.get("rounds") or [])
    tool_names = [entry.get("tool") for entry in state.get("tool_calls") or [] if entry.get("tool")]
    evidence_count = len((state.get("trace") or {}).get("evidence") or [])
    validated_count = int(((state.get("coverage_status") or {}).get("validated_evidence_count")) or 0)
    if not rounds:
        return "No retrieval rounds have completed yet."
    return (
        f"Completed {len(rounds)} retrieval round(s) using {', '.join(tool_names) or 'no tools'}; "
        f"selected {evidence_count} evidence item(s), validated {validated_count} binding(s)."
    )


def _initialize_runtime(
    query: str,
    *,
    user_id: Optional[str],
    history: Optional[List[Dict[str, Any]]],
    event_sink: Optional[agentic.AgentEventSink],
) -> OrchestratorState:
    state = agentic._initial_state(query, user_id=user_id, history=history, plan={})
    state.update(
        {
            "current_step": None,
            "completed_steps": [],
            "critic_history": [],
            "evidence_pool": [],
            "selected_evidence": [],
            "validated_bindings": [],
            "open_questions": [],
            "coverage_status": {"status": "not_started"},
            "running_summary": "No retrieval rounds have completed yet.",
        }
    )
    return OrchestratorState(
        query=query,
        user_id=user_id,
        history=list(history or []),
        event_sink=event_sink,
        state=state,
    )


def _emit_start(runtime: OrchestratorState) -> None:
    action = AgentAction(kind="start", agent="orchestrator", stage="start")
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="run_started",
        agent="orchestrator",
        stage="start",
        status="running",
        message="SAGE started an agentic reasoning run.",
    )
    runtime.record_step(action=action, status="completed", summary="Initialized orchestrator state.")


def _run_planner(runtime: OrchestratorState) -> AgentObservation:
    action = AgentAction(kind="plan", agent="planner", stage="plan")
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="agent_started",
        agent="planner",
        stage="plan",
        status="running",
        message="Planner is identifying intent, entities, constraints, and evidence needs.",
    )
    plan = agentic.build_plan(runtime.query, user_id=runtime.user_id)
    runtime.state["plan"] = plan
    runtime.state["graph_depth"] = dict(plan.get("graph_depth") or {})
    runtime.state["orchestration"] = plan.get("orchestration") or {}
    runtime.state["trace"]["query_profile"] = plan.get("query_profile") or runtime.state["trace"].get("query_profile")
    runtime.state["trace"]["graph_depth"] = dict(plan.get("graph_depth") or {})
    
    # Display the orchestration contract in colored logs
    orchestration = runtime.state.get("orchestration") or {}
    if orchestration:
        orchestrator_logging.display_orchestration_contract(orchestration)
    
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="agent_progress",
        agent="planner",
        stage="plan",
        status="running",
        message=(
            f"Planner inferred {plan.get('intent') or 'general_graph_rag'} and selected "
            f"{', '.join(plan.get('tool_sequence') or [])}."
        ),
    )
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="agent_finished",
        agent="planner",
        stage="plan",
        status="completed",
        message=f"Planner prepared {len(plan.get('steps') or [])} execution step(s).",
        result_count=len(plan.get("steps") or []),
    )
    runtime.phase = "retrieve"
    runtime.record_step(
        action=action,
        status="completed",
        summary=f"Prepared plan for {plan.get('intent') or 'general_graph_rag'}.",
        data={"tool_sequence": list(plan.get("tool_sequence") or [])},
    )
    return AgentObservation(action=action, status="completed", message="Planner prepared the execution plan.", data={"plan": plan})


def _next_retrieval_tool(runtime: OrchestratorState) -> Optional[str]:
    tools = list((runtime.state.get("plan") or {}).get("tool_sequence") or [])
    while runtime.next_tool_index < len(tools):
        candidate = tools[runtime.next_tool_index]
        runtime.next_tool_index += 1
        return candidate
    return None


def _run_retrieval(runtime: OrchestratorState, tool_name: str) -> AgentObservation:
    attempt = len(runtime.state.get("rounds") or []) + 1
    action = AgentAction(kind="retrieve", agent="retriever", stage="retrieve", tool=tool_name)
    runtime.reasoning = agentic._run_retrieval_round(
        runtime.state,
        tool_name=tool_name,
        attempt=attempt,
        event_sink=runtime.event_sink,
    )
    runtime.update_context_state()
    latest_round = (runtime.state.get("rounds") or [{}])[-1]
    runtime.record_step(
        action=action,
        status="completed",
        summary=f"Retrieved and validated evidence with {tool_name}.",
        data={
            "enough_context": bool(latest_round.get("enough_context")),
            "coverage_status": dict(runtime.state.get("coverage_status") or {}),
            "open_questions": list(runtime.state.get("open_questions") or []),
        },
    )
    return AgentObservation(
        action=action,
        status="completed",
        message=f"Retrieved evidence with {tool_name}.",
        data={"round": latest_round},
    )


def _mark_stop_if_ready(runtime: OrchestratorState) -> bool:
    latest_round = (runtime.state.get("rounds") or [{}])[-1]
    if latest_round.get("enough_context"):
        runtime.state["stop_reason"] = "enough_context"
        agentic._emit_event(
            runtime.state,
            runtime.event_sink,
            event_type="agent_progress",
            agent="orchestrator",
            stage="stop_check",
            status="running",
            message="SAGE has enough grounded context to draft an answer.",
            attempt=latest_round.get("attempt"),
        )
        runtime.record_step(
            action=AgentAction(kind="stop_check", agent="orchestrator", stage="stop_check"),
            status="completed",
            summary="Coverage is sufficient for generation.",
            data={"coverage_status": dict(runtime.state.get("coverage_status") or {})},
        )
        return True
    return False


def _mark_round_budget(runtime: OrchestratorState) -> None:
    runtime.state["stop_reason"] = "round_budget_exhausted"
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="agent_progress",
        agent="orchestrator",
        stage="stop_check",
        status="running",
        message="SAGE reached the retrieval round budget and will answer with available evidence.",
    )
    runtime.record_step(
        action=AgentAction(kind="stop_check", agent="orchestrator", stage="stop_check"),
        status="completed",
        summary="Retrieval budget exhausted; proceeding with available evidence.",
        data={"open_questions": list(runtime.state.get("open_questions") or [])},
    )


def _run_generator(runtime: OrchestratorState, *, revision: bool = False) -> AgentObservation:
    action = AgentAction(kind="generate", agent="generator", stage="generate")
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="agent_started",
        agent="generator",
        stage="generate",
        status="running",
        message="Generator is revising the answer after retry evidence." if revision else "Generator is drafting an answer from validated evidence.",
    )
    started_at = time.perf_counter()
    runtime.ai_result = agentic.services.generate_groq_response(
        runtime.query,
        runtime.state.get("documents") or [],
        user_id=runtime.user_id,
        retrieval_trace=runtime.state.get("trace"),
        history=runtime.history,
    )
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="agent_finished",
        agent="generator",
        stage="generate",
        status="completed",
        message="Generator produced a revised grounded answer." if revision else "Generator produced a grounded draft answer.",
        duration_ms=int((time.perf_counter() - started_at) * 1000),
    )
    runtime.record_step(
        action=action,
        status="completed",
        summary="Generated answer from current validated context.",
        data={
            "answer_mode": (runtime.ai_result.get("answer_payload") or {}).get("mode"),
            "reason_code": (runtime.ai_result.get("answer_payload") or {}).get("reason_code"),
        },
    )
    return AgentObservation(action=action, status="completed", message="Generated answer.", data={"ai_result": runtime.ai_result})


def _merged_answer_trace(runtime: OrchestratorState) -> Dict[str, Any]:
    return {
        **(runtime.state.get("trace") or {}),
        **((runtime.ai_result.get("trace") or {}) if isinstance(runtime.ai_result, dict) else {}),
    }


def _critic_history_entry(critic: Dict[str, Any], *, revision: bool, attempt: int) -> Dict[str, Any]:
    return {
        "attempt": attempt,
        "revision": revision,
        "passed": bool(critic.get("passed")),
        "retryable": bool(critic.get("retryable")),
        "issues": list(critic.get("issues") or []),
        "grounded_evidence_count": int(critic.get("grounded_evidence_count") or 0),
        "provenance_count": int(critic.get("provenance_count") or 0),
    }


def _run_critic(runtime: OrchestratorState, *, revision: bool = False) -> AgentObservation:
    action = AgentAction(kind="critic", agent="critic", stage="critic")
    trace = _merged_answer_trace(runtime)
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="agent_started",
        agent="critic",
        stage="critic",
        status="running",
        message="Critic is re-checking the revised answer." if revision else "Critic is checking grounding, citations, and policy-sensitive requirements.",
    )
    started_at = time.perf_counter()
    runtime.critic = agentic.policy_guard.evaluate_answer(
        query=runtime.query,
        answer=runtime.ai_result.get("answer") or "",
        answer_payload=runtime.ai_result.get("answer_payload") or {},
        trace=trace,
        plan=runtime.state.get("plan"),
    )
    critic_history = runtime.state.setdefault("critic_history", [])
    critic_history.append(_critic_history_entry(runtime.critic, revision=revision, attempt=len(critic_history) + 1))
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="agent_finished",
        agent="critic",
        stage="critic",
        status="completed" if runtime.critic.get("passed") else "needs_review",
        message=(
            "Critic passed the revised answer."
            if revision and runtime.critic.get("passed")
            else "Critic still found grounding gaps."
            if revision
            else "Critic passed the answer."
            if runtime.critic.get("passed")
            else "Critic requested stronger grounding."
        ),
        duration_ms=int((time.perf_counter() - started_at) * 1000),
        result_count=int(runtime.critic.get("grounded_evidence_count") or 0),
    )
    runtime.record_step(
        action=action,
        status="completed" if runtime.critic.get("passed") else "needs_review",
        summary="Critic passed the answer." if runtime.critic.get("passed") else "Critic requested more evidence.",
        data={
            "revision": revision,
            "retryable": bool(runtime.critic.get("retryable")),
            "issues": list(runtime.critic.get("issues") or []),
            "grounded_evidence_count": int(runtime.critic.get("grounded_evidence_count") or 0),
            "provenance_count": int(runtime.critic.get("provenance_count") or 0),
        },
    )
    return AgentObservation(action=action, status="completed", message="Critic evaluated answer.", data={"critic": runtime.critic})


def _run_critic_retry(runtime: OrchestratorState) -> bool:
    plan = runtime.state.get("plan") or {}
    max_retries = int((plan.get("constraints") or {}).get("max_retries") or 0)
    if not runtime.critic.get("retryable") or runtime.critic.get("passed"):
        return False
    if int(runtime.state.get("retry_count") or 0) >= max_retries:
        return False

    retry_tool = agentic._choose_retry_tool(plan, runtime.state, runtime.critic)
    runtime.state["retry_count"] = int(runtime.state.get("retry_count") or 0) + 1
    runtime.state["retry_attempted"] = True
    runtime.state["retry_tool"] = retry_tool
    retry_attempt = len(runtime.state.get("rounds") or []) + 1
    trace = dict(runtime.state.get("trace") or {})
    trace["critic_feedback"] = {
        "issues": list(runtime.critic.get("issues") or []),
        "answer": runtime.ai_result.get("answer") or "",
        "answer_payload": runtime.ai_result.get("answer_payload") or {},
    }
    runtime.state["trace"] = trace
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="retry_started",
        agent="orchestrator",
        stage="retry",
        status="running",
        message=f"Critic triggered one retry using {retry_tool}.",
        tool=retry_tool,
        attempt=retry_attempt,
    )
    runtime.record_step(
        action=AgentAction(kind="revise_plan", agent="planner", stage="retry", tool=retry_tool),
        status="completed",
        summary=f"Selected {retry_tool} to address critic feedback.",
        data={"critic_issues": list(runtime.critic.get("issues") or [])},
    )
    _run_retrieval(runtime, retry_tool)
    runtime.state["stop_reason"] = f"critic_retry:{retry_tool}"
    _run_generator(runtime, revision=True)
    _run_critic(runtime, revision=True)
    runtime.state["retry_succeeded"] = bool(runtime.critic.get("passed"))
    return True


def _final_trace(runtime: OrchestratorState) -> Dict[str, Any]:
    trace = _merged_answer_trace(runtime)
    retry_attempted = bool(runtime.state.get("retry_attempted"))
    trace["agentic"] = {
        "enabled": True,
        "run_id": runtime.state["run_id"],
        "planner": runtime.state.get("plan") or {},
        "orchestration": dict(runtime.state.get("orchestration") or {}),
        "rounds": list(runtime.state.get("rounds") or []),
        "tool_calls": list(runtime.state.get("tool_calls") or []),
        "events": list(runtime.state.get("events") or []),
        "route_history": list(runtime.state.get("route_history") or []),
        "current_agent": runtime.state.get("current_agent"),
        "current_step": runtime.state.get("current_step"),
        "completed_steps": list(runtime.state.get("completed_steps") or []),
        "coverage_status": dict(runtime.state.get("coverage_status") or {}),
        "open_questions": list(runtime.state.get("open_questions") or []),
        "running_summary": runtime.state.get("running_summary"),
        "evidence_pool_count": len(runtime.state.get("evidence_pool") or []),
        "selected_evidence_count": len(runtime.state.get("selected_evidence") or []),
        "stop_reason": runtime.state.get("stop_reason"),
        "reasoner": runtime.reasoning,
        "generator": {
            "answer_mode": (runtime.ai_result.get("answer_payload") or {}).get("mode"),
            "reason_code": (runtime.ai_result.get("answer_payload") or {}).get("reason_code"),
        },
        "critic": runtime.critic,
        "critic_history": list(runtime.state.get("critic_history") or []),
        "retry_attempted": retry_attempted,
        "retry_tool": runtime.state.get("retry_tool"),
        "retry_succeeded": bool(runtime.state.get("retry_succeeded")) if retry_attempted else None,
        "remaining_critic_issues": [] if runtime.critic.get("passed") else list(runtime.critic.get("issues") or []),
        "status": "passed" if runtime.critic.get("passed") else "needs_review",
    }
    return trace


def _finish(runtime: OrchestratorState, trace: Dict[str, Any]) -> None:
    action = AgentAction(kind="finish", agent="orchestrator", stage="finish")
    agentic._emit_event(
        runtime.state,
        runtime.event_sink,
        event_type="run_finished",
        agent="orchestrator",
        stage="finish",
        status="completed" if runtime.critic.get("passed") else "needs_review",
        message="SAGE finished the agentic reasoning run.",
    )
    trace["agentic"]["events"] = list(runtime.state.get("events") or [])
    trace["agentic"]["route_history"] = list(runtime.state.get("route_history") or [])
    trace["agentic"]["current_agent"] = None
    runtime.record_step(
        action=action,
        status="completed" if runtime.critic.get("passed") else "needs_review",
        summary="Finished agentic reasoning run.",
    )


def _thinking(runtime: OrchestratorState) -> List[str]:
    thinking = list(runtime.ai_result.get("thinking") or [])
    tools_used = ", ".join(entry.get("tool") or "unknown" for entry in runtime.state.get("tool_calls") or [])
    plan = runtime.state.get("plan") or {}
    thinking.extend(
        [
            f"Planner selected {plan.get('strategy')} retrieval.",
            f"Retriever used: {tools_used or plan.get('strategy')}.",
            f"Reasoner validated {runtime.reasoning.get('validated_evidence_count') or 0} evidence bindings.",
            f"Critic verdict: {'pass' if runtime.critic.get('passed') else 'review'}",
        ]
    )
    if runtime.state.get("retry_attempted"):
        retry_tool = runtime.state.get("retry_tool") or "retrieval"
        retry_outcome = "succeeded" if runtime.state.get("retry_succeeded") else "failed"
        thinking.append(f"Critic retry: attempted via {retry_tool}; {retry_outcome}.")
    if not runtime.critic.get("passed"):
        issues = list(runtime.critic.get("issues") or [])
        if issues:
            thinking.append(f"Remaining critic issues: {', '.join(issues[:3])}.")
    return thinking


def run_agentic_query(
    query: str,
    *,
    user_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    event_sink: Optional[agentic.AgentEventSink] = None,
) -> Dict[str, Any]:
    # Use default colored event sink if none provided
    if event_sink is None:
        event_sink = orchestrator_logging.create_default_event_sink()
    
    runtime = _initialize_runtime(query, user_id=user_id, history=history, event_sink=event_sink)
    _emit_start(runtime)

    try:
        _run_planner(runtime)
    except Exception as exc:
        agentic._emit_event(
            runtime.state,
            runtime.event_sink,
            event_type="run_failed",
            agent="orchestrator",
            stage="plan",
            status="failed",
            message="SAGE could not build an execution plan.",
            error=str(exc),
        )
        raise

    try:
        while True:
            tool_name = _next_retrieval_tool(runtime)
            if tool_name is None:
                break
            _run_retrieval(runtime, tool_name)
            if _mark_stop_if_ready(runtime):
                break
        if not runtime.state.get("stop_reason"):
            _mark_round_budget(runtime)
    except Exception as exc:
        agentic._emit_event(
            runtime.state,
            runtime.event_sink,
            event_type="run_failed",
            agent="orchestrator",
            stage="retrieve",
            status="failed",
            message="SAGE failed while retrieving or validating evidence.",
            error=str(exc),
        )
        raise

    _run_generator(runtime)
    _run_critic(runtime)
    _run_critic_retry(runtime)

    trace = _final_trace(runtime)
    _finish(runtime, trace)
    trace["agentic"]["completed_steps"] = list(runtime.state.get("completed_steps") or [])
    trace["agentic"]["current_step"] = runtime.state.get("current_step")

    return {
        "answer": runtime.ai_result.get("answer"),
        "answer_payload": runtime.ai_result.get("answer_payload"),
        "thinking": _thinking(runtime),
        "trace": trace,
    }
