from app import agentic


def test_run_agentic_query_returns_agentic_trace(monkeypatch):
    monkeypatch.setattr(
        agentic.retrieval_selector,
        "decide_strategy",
        lambda _query, user_id=None: {
            "strategy": "semantic",
            "reasons": ["test"],
            "llm_used": False,
            "heuristic_confidence": 0.9,
        },
    )
    monkeypatch.setattr(
        agentic.vector_search,
        "retrieve",
        lambda _query, user_id=None, strategy="hybrid", seed_trace=None: {
            "documents": ["ctx"],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [{"chunk_id": "chunk-1", "rank_score": 0.9}],
                "selector_strategy": strategy,
            },
        },
    )
    monkeypatch.setattr(agentic.rerank, "rerank", lambda documents, trace: {"documents": documents, "trace": trace})
    monkeypatch.setattr(
        agentic.graph_query,
        "expand_retrieval_context",
        lambda _query, seed_trace=None, user_id=None: {"documents": [], "trace": seed_trace or {"evidence": []}},
    )
    monkeypatch.setattr(
        agentic.services,
        "generate_groq_response",
        lambda _query, _documents, user_id=None, retrieval_trace=None: {
            "answer": "answer",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "answer",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-1"],
            },
            "thinking": [],
            "trace": {"user_scoped": bool(user_id), "evidence": retrieval_trace.get("evidence")},
        },
    )

    result = agentic.run_agentic_query("Who is my manager?", user_id="u1")

    assert result["answer"] == "answer"
    assert result["trace"]["agentic"]["enabled"] is True
    assert result["trace"]["agentic"]["planner"]["strategy"] in {"semantic", "graph", "fulltext", "hybrid"}
    assert result["trace"]["agentic"]["critic"]["passed"] is True


def test_run_agentic_query_marks_critic_review_when_answer_is_ungrounded(monkeypatch):
    monkeypatch.setattr(
        agentic.retrieval_selector,
        "decide_strategy",
        lambda _query, user_id=None: {
            "strategy": "semantic",
            "reasons": ["test"],
            "llm_used": False,
            "heuristic_confidence": 0.9,
        },
    )
    monkeypatch.setattr(
        agentic.vector_search,
        "retrieve",
        lambda _query, user_id=None, strategy="hybrid", seed_trace=None: {
            "documents": [],
            "trace": {"query_type": "general_search", "user_scoped": bool(user_id), "evidence": []},
        },
    )
    monkeypatch.setattr(agentic.rerank, "rerank", lambda documents, trace: {"documents": documents, "trace": trace})
    monkeypatch.setattr(
        agentic.graph_query,
        "expand_retrieval_context",
        lambda _query, seed_trace=None, user_id=None: {"documents": [], "trace": seed_trace or {"evidence": []}},
    )
    monkeypatch.setattr(
        agentic.services,
        "generate_groq_response",
        lambda *_args, **_kwargs: {
            "answer": "Bijade is your manager.",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "Bijade is your manager.",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": [],
            },
            "thinking": [],
        },
    )

    result = agentic.run_agentic_query("Who is my manager?", user_id="u1")

    assert result["trace"]["agentic"]["critic"]["passed"] is False
    assert "missing_grounded_uncertainty" in result["trace"]["agentic"]["critic"]["issues"]


def test_run_agentic_query_records_tool_calls_and_stops_on_enough_context(monkeypatch):
    calls = []

    monkeypatch.setattr(
        agentic.retrieval_selector,
        "decide_strategy",
        lambda _query, user_id=None: {
            "strategy": "hybrid",
            "reasons": ["hybrid test"],
            "llm_used": False,
            "heuristic_confidence": 0.95,
        },
    )

    def fake_retrieve(_query, user_id=None, strategy="hybrid", seed_trace=None):
        calls.append(strategy)
        if strategy == "semantic":
            return {
                "documents": ["semantic ctx"],
                "trace": {
                    "query_type": "general_search",
                    "user_scoped": bool(user_id),
                    "evidence": [{"chunk_id": "chunk-1", "rank_score": 0.95, "document": {"doc_id": "doc-1"}}],
                },
            }
        return {
            "documents": ["fulltext ctx"],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [
                    {"chunk_id": "chunk-1", "rank_score": 0.95, "document": {"doc_id": "doc-1"}},
                    {"fact_id": "fact-1", "rank_score": 0.9, "document": {"doc_id": "doc-2"}},
                ],
            },
        }

    monkeypatch.setattr(agentic.vector_search, "retrieve", fake_retrieve)
    monkeypatch.setattr(agentic.graph_query, "expand_retrieval_context", lambda *_args, **_kwargs: {"documents": [], "trace": {"evidence": []}})
    monkeypatch.setattr(agentic.rerank, "rerank", lambda documents, trace: {"documents": documents, "trace": trace})
    monkeypatch.setattr(
        agentic.services,
        "generate_groq_response",
        lambda *_args, **_kwargs: {
            "answer": "answer",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "answer",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-1", "fact:fact-1"],
            },
            "thinking": [],
        },
    )

    result = agentic.run_agentic_query("Who is my manager?", user_id="u1")

    assert calls == ["semantic", "fulltext"]
    assert result["trace"]["agentic"]["stop_reason"] == "enough_context"
    assert len(result["trace"]["agentic"]["tool_calls"]) == 2
    assert result["trace"]["agentic"]["rounds"][-1]["enough_context"] is True


def test_run_agentic_query_uses_single_retry_when_critic_requests_it(monkeypatch):
    calls = []

    monkeypatch.setattr(
        agentic.retrieval_selector,
        "decide_strategy",
        lambda _query, user_id=None: {
            "strategy": "semantic",
            "reasons": ["semantic test"],
            "llm_used": False,
            "heuristic_confidence": 0.95,
        },
    )

    def fake_retrieve(_query, user_id=None, strategy="hybrid", seed_trace=None):
        calls.append(strategy)
        if len(calls) == 1:
            return {"documents": [], "trace": {"query_type": "general_search", "user_scoped": bool(user_id), "evidence": []}}
        return {
            "documents": ["retry ctx"],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [{"chunk_id": "chunk-retry", "rank_score": 0.9, "document": {"doc_id": "doc-retry"}}],
            },
        }

    monkeypatch.setattr(agentic.vector_search, "retrieve", fake_retrieve)
    monkeypatch.setattr(agentic.graph_query, "expand_retrieval_context", lambda *_args, **_kwargs: {"documents": [], "trace": {"evidence": []}})
    monkeypatch.setattr(agentic.rerank, "rerank", lambda documents, trace: {"documents": documents, "trace": trace})

    responses = [
        {
            "answer": "Bijade is your manager.",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "Bijade is your manager.",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": [],
            },
            "thinking": [],
        },
        {
            "answer": "Bijade is your manager.",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "Bijade is your manager.",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-retry"],
            },
            "thinking": [],
        },
    ]
    monkeypatch.setattr(agentic.services, "generate_groq_response", lambda *_args, **_kwargs: responses.pop(0))
    verdicts = [
        {"passed": False, "retryable": True, "issues": ["missing_policy_provenance"], "grounded_evidence_count": 0, "provenance_count": 0},
        {"passed": True, "retryable": False, "issues": [], "grounded_evidence_count": 1, "provenance_count": 1},
    ]
    monkeypatch.setattr(agentic.policy_guard, "evaluate_answer", lambda **_kwargs: verdicts.pop(0))

    result = agentic.run_agentic_query("Who is my manager?", user_id="u1")

    assert len(calls) >= 2
    assert result["trace"]["agentic"]["stop_reason"].startswith("critic_retry:")
    assert result["trace"]["agentic"]["critic"]["passed"] is True


def test_build_plan_includes_generic_intent_and_evidence_contract(monkeypatch):
    monkeypatch.setattr(
        agentic.retrieval_selector,
        "decide_strategy",
        lambda _query, user_id=None: {
            "strategy": "fulltext",
            "reasons": ["exact policy and amount terms"],
            "llm_used": False,
            "heuristic_confidence": 0.92,
        },
    )
    monkeypatch.setattr(
        agentic.graph_query,
        "schema_snapshot",
        lambda: {"node_types": ["Document", "Chunk", "Person", "Policy"], "edge_types": ["MENTIONS", "APPROVED_BY"]},
    )

    plan = agentic.build_plan("List approvals above ₹10 lakhs in Q3 2025 that violated escalation policy", user_id="u1")

    assert plan["intent"] == "policy_or_compliance_reasoning"
    assert "policy_reference" in plan["required_evidence"]
    assert "timestamp_or_temporal_scope" in plan["required_evidence"]
    assert "amount_or_numeric_property" in plan["required_evidence"]
    assert plan["tool_sequence"][0] == "fulltext"
    assert plan["tool_plan"][0]["tool"] == "fulltext"


def test_run_agentic_query_emits_ordered_agent_events(monkeypatch):
    monkeypatch.setattr(
        agentic.retrieval_selector,
        "decide_strategy",
        lambda _query, user_id=None: {
            "strategy": "semantic",
            "reasons": ["test"],
            "llm_used": False,
            "heuristic_confidence": 0.9,
        },
    )
    monkeypatch.setattr(
        agentic.vector_search,
        "retrieve",
        lambda _query, user_id=None, strategy="hybrid", seed_trace=None: {
            "documents": ["ctx"],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [
                    {"chunk_id": "chunk-1", "rank_score": 0.95, "document": {"doc_id": "doc-1"}},
                    {"fact_id": "fact-1", "rank_score": 0.9, "document": {"doc_id": "doc-2"}},
                ],
            },
        },
    )
    monkeypatch.setattr(agentic.rerank, "rerank", lambda documents, trace: {"documents": documents, "trace": trace})
    monkeypatch.setattr(
        agentic.graph_query,
        "expand_retrieval_context",
        lambda _query, seed_trace=None, user_id=None: {"documents": [], "trace": seed_trace or {"evidence": []}},
    )
    monkeypatch.setattr(
        agentic.graph_query,
        "validate_trace_paths",
        lambda _trace: {"valid": True, "validated_evidence_count": 1, "missing_fields": []},
    )
    monkeypatch.setattr(
        agentic.services,
        "generate_groq_response",
        lambda *_args, **_kwargs: {
            "answer": "answer",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "answer",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-1", "fact:fact-1"],
            },
            "thinking": [],
        },
    )
    monkeypatch.setattr(
        agentic.policy_guard,
        "evaluate_answer",
        lambda **_kwargs: {"passed": True, "retryable": False, "issues": [], "grounded_evidence_count": 2, "provenance_count": 2},
    )
    events = []

    result = agentic.run_agentic_query("Explain the approval chain", user_id="u1", event_sink=events.append)

    event_types = [event["event_type"] for event in events]
    assert event_types[0] == "run_started"
    assert "tool_started" in event_types
    assert "tool_finished" in event_types
    assert event_types[-1] == "run_finished"
    assert result["trace"]["agentic"]["events"] == events
    assert result["trace"]["agentic"]["current_agent"] is None


def test_run_agentic_query_requires_distinct_coverage_for_multi_item_questions(monkeypatch):
    calls = []

    monkeypatch.setattr(
        agentic.retrieval_selector,
        "decide_strategy",
        lambda _query, user_id=None: {
            "strategy": "semantic",
            "reasons": ["semantic first"],
            "llm_used": False,
            "heuristic_confidence": 0.91,
        },
    )

    def fake_retrieve(_query, user_id=None, strategy="hybrid", seed_trace=None):
        calls.append(strategy)
        if strategy == "semantic":
            return {
                "documents": ["ctx-1", "ctx-2"],
                "trace": {
                    "query": _query,
                    "query_type": "person_lookup",
                    "user_scoped": bool(user_id),
                    "evidence": [
                        {
                            "chunk_id": "chunk-1",
                            "chunk_summary": "From now on I'm your only Manager Bijade",
                            "rank_score": 0.95,
                            "similarity": 0.95,
                            "document": {"doc_id": "doc-1"},
                        },
                        {
                            "chunk_id": "chunk-2",
                            "chunk_summary": "From now on I'm your only Manager Bijade",
                            "rank_score": 0.94,
                            "similarity": 0.94,
                            "document": {"doc_id": "doc-2"},
                        },
                    ],
                },
            }
        return {
            "documents": ["ctx-3"],
            "trace": {
                "query": _query,
                "query_type": "person_lookup",
                "user_scoped": bool(user_id),
                "evidence": [
                    {
                        "fact_id": "fact-1",
                        "fact_summary": "George Brown reports to Hannah Garcia.",
                        "rank_score": 0.88,
                        "similarity": 0.88,
                        "document": {"doc_id": "doc-3"},
                        "fact": {
                            "canonical_key": "reports_to::7",
                            "subject_entity_id": "George Brown",
                            "object_entity_id": "Hannah Garcia",
                        },
                    }
                ],
            },
        }

    monkeypatch.setattr(agentic.vector_search, "retrieve", fake_retrieve)
    monkeypatch.setattr(agentic.rerank, "rerank", lambda documents, trace: {"documents": documents, "trace": trace})
    monkeypatch.setattr(
        agentic.graph_query,
        "expand_retrieval_context",
        lambda *_args, **_kwargs: {"documents": [], "trace": {"evidence": []}},
    )
    monkeypatch.setattr(
        agentic.graph_query,
        "validate_trace_paths",
        lambda trace: {"valid": True, "validated_evidence_count": len(trace.get("evidence") or []), "missing_fields": []},
    )
    monkeypatch.setattr(
        agentic.services,
        "generate_groq_response",
        lambda *_args, **_kwargs: {
            "answer": "Bijade and Hannah Garcia are managers in the available evidence.",
            "answer_payload": {
                "schema_version": 1,
                "mode": "long",
                "reason_code": "evidence_complexity",
                "summary": "Bijade and Hannah Garcia are managers in the available evidence.",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-1", "fact:fact-1"],
            },
            "thinking": [],
        },
    )
    monkeypatch.setattr(
        agentic.policy_guard,
        "evaluate_answer",
        lambda **_kwargs: {"passed": True, "retryable": False, "issues": [], "grounded_evidence_count": 2, "provenance_count": 2},
    )

    result = agentic.run_agentic_query("Who are the various managers we have?")

    assert calls == ["semantic", "fulltext"]
    assert result["trace"]["agentic"]["rounds"][0]["enough_context"] is False
    assert result["trace"]["agentic"]["rounds"][0]["distinct_evidence_count"] == 1
    assert result["trace"]["agentic"]["rounds"][1]["enough_context"] is True
    assert result["trace"]["agentic"]["rounds"][1]["distinct_evidence_count"] >= 2
