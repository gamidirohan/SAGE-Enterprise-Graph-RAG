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
                "evidence": [
                    {
                        "chunk_id": "chunk-1",
                        "chunk_summary": "Project Alpha overview.",
                        "rank_score": 0.9,
                        "document": {"doc_id": "doc-1", "subject": "Project Alpha"},
                    }
                ],
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
            "answer": "Project Alpha is covered by the retrieved evidence.",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "Project Alpha is covered by the retrieved evidence.",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-1"],
            },
            "thinking": [],
            "trace": {"user_scoped": bool(user_id), "evidence": retrieval_trace.get("evidence")},
        },
    )

    result = agentic.run_agentic_query("What is Project Alpha?", user_id="u1")

    assert result["answer"] == "Project Alpha is covered by the retrieved evidence."
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

    result = agentic.run_agentic_query("What is Project Alpha?", user_id="u1")

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
                    "evidence": [
                        {
                            "chunk_id": "chunk-1",
                            "chunk_summary": "Project Alpha overview.",
                            "rank_score": 0.95,
                            "document": {"doc_id": "doc-1", "subject": "Project Alpha"},
                        }
                    ],
                },
            }
        return {
            "documents": ["fulltext ctx"],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [
                    {
                        "chunk_id": "chunk-1",
                        "chunk_summary": "Project Alpha overview.",
                        "rank_score": 0.95,
                        "document": {"doc_id": "doc-1", "subject": "Project Alpha"},
                    },
                    {
                        "fact_id": "fact-1",
                        "fact_summary": "Project Alpha is an active project.",
                        "rank_score": 0.9,
                        "document": {"doc_id": "doc-2", "subject": "Project Alpha"},
                    },
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
            "answer": "Project Alpha is covered by the retrieved evidence.",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "Project Alpha is covered by the retrieved evidence.",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-1", "fact:fact-1"],
            },
            "thinking": [],
        },
    )

    result = agentic.run_agentic_query("What is Project Alpha?", user_id="u1")

    assert calls == ["semantic", "fulltext"]
    assert result["trace"]["agentic"]["stop_reason"] == "enough_context"
    assert len(result["trace"]["agentic"]["tool_calls"]) == 2
    assert result["trace"]["agentic"]["rounds"][-1]["enough_context"] is True


def test_enough_context_requires_fact_for_fact_priority_lookup():
    trace = {
        "query_type": "schedule_or_timeline",
        "evidence": [
            {"chunk_id": "chunk-review", "rank_score": 0.95, "document": {"doc_id": "doc-review"}},
            {"chunk_id": "chunk-alpha", "rank_score": 0.9, "document": {"doc_id": "doc-alpha"}},
        ],
    }
    reasoning = {"validated_evidence_count": 2}

    assert agentic._enough_context(trace, reasoning) is False

    trace["evidence"].append(
        {"fact_id": "fact-meeting", "rank_score": 1.0, "document": {"doc_id": "doc-meeting"}}
    )

    assert agentic._enough_context(trace, reasoning) is True


def test_enough_context_requires_graph_round_for_deep_broad_query():
    trace = {
        "query_type": "general_search",
        "selector_strategy": "fulltext",
        "evidence": [
            {"chunk_id": "chunk-1", "rank_score": 0.95, "document": {"doc_id": "doc-1"}},
            {"chunk_id": "chunk-2", "rank_score": 0.9, "document": {"doc_id": "doc-2"}},
        ],
        "query_profile": {
            "expects_multiple_items": True,
            "requires_broad_coverage": True,
            "minimum_unique_evidence": 2,
            "minimum_tool_rounds": 2,
        },
    }
    reasoning = {"validated_evidence_count": 2}
    plan = {
        "graph_depth": {"expand_hops": 3},
        "tool_sequence": ["semantic", "fulltext", "graph"],
        "query_profile": trace["query_profile"],
    }

    assert agentic._enough_context(trace, reasoning, plan=plan, attempt=2) is False

    trace["selector_strategy"] = "graph"

    assert agentic._enough_context(trace, reasoning, plan=plan, attempt=3) is True


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
    generator_feedback = []

    def fake_generate(_query, _documents, user_id=None, retrieval_trace=None):
        generator_feedback.append(dict((retrieval_trace or {}).get("critic_feedback") or {}))
        return responses.pop(0)

    monkeypatch.setattr(agentic.services, "generate_groq_response", fake_generate)
    verdicts = [
        {"passed": False, "retryable": True, "issues": ["missing_policy_provenance"], "grounded_evidence_count": 0, "provenance_count": 0},
        {"passed": True, "retryable": False, "issues": [], "grounded_evidence_count": 1, "provenance_count": 1},
    ]
    monkeypatch.setattr(agentic.policy_guard, "evaluate_answer", lambda **_kwargs: verdicts.pop(0))

    result = agentic.run_agentic_query("Who is my manager?", user_id="u1")

    assert len(calls) >= 2
    assert result["trace"]["agentic"]["stop_reason"].startswith("critic_retry:")
    assert result["trace"]["agentic"]["critic"]["passed"] is True
    assert result["trace"]["agentic"]["retry_attempted"] is True
    assert result["trace"]["agentic"]["retry_succeeded"] is True
    assert result["trace"]["agentic"]["remaining_critic_issues"] == []
    assert [entry["passed"] for entry in result["trace"]["agentic"]["critic_history"]] == [False, True]
    assert any("Critic retry:" in item and "succeeded" in item for item in result["thinking"])
    assert generator_feedback[0] == {}
    assert generator_feedback[1]["issues"] == ["missing_policy_provenance"]
    assert generator_feedback[1]["answer"] == "Bijade is your manager."


def test_run_agentic_query_exposes_failed_retry_critic_issues(monkeypatch):
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
        return {
            "documents": ["ctx after retry"] if len(calls) > 1 else [],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [{"chunk_id": "chunk-retry", "rank_score": 0.9, "document": {"doc_id": "doc-retry"}}]
                if len(calls) > 1
                else [],
            },
        }

    monkeypatch.setattr(agentic.vector_search, "retrieve", fake_retrieve)
    monkeypatch.setattr(agentic.graph_query, "expand_retrieval_context", lambda *_args, **_kwargs: {"documents": [], "trace": {"evidence": []}})
    monkeypatch.setattr(agentic.rerank, "rerank", lambda documents, trace: {"documents": documents, "trace": trace})

    responses = [
        {
            "answer": "Charlie is assigned to Project Proton.",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "Charlie is assigned to Project Proton.",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": [],
            },
            "thinking": [],
        },
        {
            "answer": "Charlie is assigned to Project Proton.",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "Charlie is assigned to Project Proton.",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-retry"],
            },
            "thinking": [],
        },
    ]

    monkeypatch.setattr(agentic.services, "generate_groq_response", lambda *_args, **_kwargs: responses.pop(0))
    verdicts = [
        {"passed": False, "retryable": True, "issues": ["missing_required_answer_slot:temporal_end"], "grounded_evidence_count": 0, "provenance_count": 0},
        {"passed": False, "retryable": False, "issues": ["missing_required_answer_slot:temporal_end"], "grounded_evidence_count": 1, "provenance_count": 1},
    ]
    monkeypatch.setattr(agentic.policy_guard, "evaluate_answer", lambda **_kwargs: verdicts.pop(0))

    result = agentic.run_agentic_query("How long is Charlie going to work, and in which project", user_id="u1")
    agentic_trace = result["trace"]["agentic"]

    assert agentic_trace["status"] == "needs_review"
    assert agentic_trace["retry_attempted"] is True
    assert agentic_trace["retry_succeeded"] is False
    assert agentic_trace["remaining_critic_issues"] == ["missing_required_answer_slot:temporal_end"]
    assert [entry["passed"] for entry in agentic_trace["critic_history"]] == [False, False]
    assert agentic_trace["critic_history"][1]["revision"] is True
    assert any("Critic retry:" in item and "failed" in item for item in result["thinking"])
    assert any("missing_required_answer_slot:temporal_end" in item for item in result["thinking"])


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
    assert plan["orchestration"]["planner_required"] is True
    assert plan["orchestration"]["critic_required"] is True
    assert plan["orchestration"]["tool_owner"]["graph"] == "retriever"


def test_build_plan_marks_comparison_queries_as_broad_synthesis(monkeypatch):
    monkeypatch.setattr(
        agentic.retrieval_selector,
        "decide_strategy",
        lambda _query, user_id=None: {
            "strategy": "semantic",
            "reasons": ["comparison"],
            "llm_used": False,
            "heuristic_confidence": 0.9,
        },
    )
    monkeypatch.setattr(
        agentic.graph_query,
        "schema_snapshot",
        lambda: {"node_types": ["Document", "Chunk", "Person"], "edge_types": ["PART_OF", "REPORTS_TO"]},
    )

    plan = agentic.build_plan(
        "Compare Project Beta and Project Gamma based on the chat history.",
        user_id="u1",
    )

    assert plan["orchestration"]["route_family"] == "broad_synthesis"
    assert plan["orchestration"]["planner_required"] is True
    assert plan["orchestration"]["critic_required"] is True
    assert plan["orchestration"]["can_short_circuit"] is False


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
    assert result["trace"]["agentic"]["orchestration"]["planner_required"] is True
    assert result["trace"]["agentic"]["orchestration"]["critic_required"] is True
    assert result["trace"]["agentic"]["completed_steps"]
    assert result["trace"]["agentic"]["coverage_status"]["status"] in {"sufficient", "partial", "insufficient"}
    assert result["trace"]["agentic"]["running_summary"]


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
        lambda *_args, seed_trace=None, **_kwargs: {
            "documents": [],
            "trace": {
                **dict(seed_trace or {}),
                "selector_strategy": "graph",
            },
        },
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
    assert result["trace"]["agentic"]["rounds"][1]["enough_context"] is False
    assert result["trace"]["agentic"]["rounds"][1]["distinct_evidence_count"] >= 2
    assert result["trace"]["agentic"]["rounds"][2]["tool"] == "graph"
    assert result["trace"]["agentic"]["rounds"][2]["depth"] == 3
    assert result["trace"]["agentic"]["coverage_status"]["status"] == "sufficient"
    assert result["trace"]["agentic"]["open_questions"] == []


def test_agent_planner_identifies_policy_reasoning_route(monkeypatch):
    """Verify planner correctly identifies policy_reasoning route family."""
    monkeypatch.setattr(
        agentic.graph_query,
        "schema_snapshot",
        lambda: {"node_types": ["Policy", "Document", "Chunk"], "edge_types": ["ENFORCED_BY"]},
    )
    
    plan = agentic.build_plan(
        "What is our data retention policy and how does it comply with GDPR?",
        user_id="u1",
    )
    
    assert plan["orchestration"]["route_family"] == "policy_reasoning"
    assert plan["orchestration"]["planner_required"] is True
    assert plan["orchestration"]["retriever_required"] is True
    assert plan["orchestration"]["critic_required"] is True
    assert "fulltext" in plan["tool_sequence"]


def test_agent_planner_identifies_relationship_lookup_route(monkeypatch):
    """Verify planner correctly identifies relationship_lookup route family."""
    monkeypatch.setattr(
        agentic.graph_query,
        "schema_snapshot",
        lambda: {"node_types": ["Person", "Department"], "edge_types": ["REPORTS_TO", "MANAGES"]},
    )
    
    plan = agentic.build_plan(
        "Who does John Smith report to?",
        user_id="u1",
    )
    
    assert plan["orchestration"]["route_family"] == "relationship_lookup"
    assert plan["orchestration"]["planner_required"] is True
    assert plan["orchestration"]["retriever_required"] is True
    assert plan["orchestration"]["reasoner_required"] is True


def test_agent_planner_assigns_dynamic_graph_depth_from_query_shape():
    direct_plan = agentic.build_plan("What is the office address for HQ?")
    relationship_plan = agentic.build_plan("Who does Rohan report to?")
    broad_plan = agentic.build_plan(
        "Compare Project Beta and Project Gamma based on the chat history."
    )

    assert direct_plan["graph_depth"]["seed_hops"] == 0
    assert direct_plan["graph_depth"]["expand_hops"] == 1
    assert relationship_plan["graph_depth"]["expand_hops"] == 2
    assert broad_plan["graph_depth"]["expand_hops"] == 3
    assert broad_plan["constraints"]["max_depth"] == 3


def test_agent_retriever_passes_planned_depth_to_tools(monkeypatch):
    calls = []

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

    def fake_retrieve(_query, user_id=None, strategy="hybrid", seed_trace=None, context_hops=None):
        calls.append((strategy, context_hops))
        return {
            "documents": ["ctx"],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [{"chunk_id": "chunk-1", "rank_score": 0.9, "document": {"doc_id": "doc-1"}}],
                "selector_strategy": strategy,
            },
        }

    monkeypatch.setattr(agentic.vector_search, "retrieve", fake_retrieve)
    monkeypatch.setattr(agentic.rerank, "rerank", lambda documents, trace: {"documents": documents, "trace": trace})
    monkeypatch.setattr(
        agentic.graph_query,
        "expand_retrieval_context",
        lambda _query, seed_trace=None, user_id=None, expand_hops=None: {
            "documents": [],
            "trace": {
                **(seed_trace or {"evidence": []}),
                "graph_depth": {"expand_hops": expand_hops},
            },
        },
    )
    monkeypatch.setattr(
        agentic.services,
        "generate_groq_response",
        lambda *_args, **_kwargs: {
            "answer": "test answer",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "test answer",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-1"],
            },
            "thinking": [],
        },
    )
    monkeypatch.setattr(
        agentic.policy_guard,
        "evaluate_answer",
        lambda **_kwargs: {"passed": True, "retryable": False, "issues": [], "grounded_evidence_count": 1, "provenance_count": 1},
    )

    result = agentic.run_agentic_query("What is the office address for HQ?")

    assert ("semantic", 0) in calls
    assert result["trace"]["agentic"]["planner"]["graph_depth"]["seed_hops"] == 0
    assert result["trace"]["agentic"]["tool_calls"][0]["depth"] == 0


def test_agent_graph_depth_escalates_on_repeated_graph_calls():
    state = {
        "graph_depth": {"seed_hops": 1, "expand_hops": 2, "max_hops": 3},
        "tool_calls": [{"tool": "graph"}],
    }

    active = agentic._active_depth_for_tool(state, "graph")

    assert active["effective_hops"] == 3


def test_agent_retriever_executes_tool_sequence(monkeypatch):
    """Verify retriever executes the tool sequence from planner."""
    tool_executions = []
    
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
    
    def track_retrieve(_query, user_id=None, strategy="hybrid", seed_trace=None):
        tool_executions.append(strategy)
        return {
            "documents": ["ctx"],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [{"chunk_id": "chunk-1", "rank_score": 0.9}],
                "selector_strategy": strategy,
            },
        }
    
    monkeypatch.setattr(agentic.vector_search, "retrieve", track_retrieve)
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
            "answer": "test answer",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "test answer",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-1"],
            },
            "thinking": [],
            "trace": {"evidence": [{"chunk_id": "chunk-1"}]},
        },
    )
    monkeypatch.setattr(
        agentic.policy_guard,
        "evaluate_answer",
        lambda **_kwargs: {"passed": True, "retryable": False, "issues": [], "grounded_evidence_count": 1, "provenance_count": 1},
    )
    
    result = agentic.run_agentic_query("Test query", user_id="u1")
    
    # Retriever should have been called with at least semantic strategy
    assert len(tool_executions) > 0
    assert "semantic" in tool_executions
    assert result["trace"]["agentic"]["tool_calls"]


def test_agent_reasoner_validates_evidence_bindings(monkeypatch):
    """Verify reasoner validates evidence bindings correctly."""
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
    
    # Reasoner validates these bindings
    monkeypatch.setattr(
        agentic.graph_query,
        "validate_trace_paths",
        lambda trace: {
            "valid": True,
            "validated_evidence_count": len(trace.get("evidence") or []),
            "missing_fields": []
        },
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
    
    result = agentic.run_agentic_query("Test query", user_id="u1")
    
    # Reasoner should have validated the evidence
    assert result["trace"]["agentic"]["reasoner"]["valid"] is True
    assert result["trace"]["agentic"]["reasoner"]["validated_evidence_count"] == 2


def test_agent_generator_creates_grounded_answer(monkeypatch):
    """Verify generator creates a grounded answer with proper citations."""
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
            "documents": ["Company policy document"],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [{"chunk_id": "chunk-1", "rank_score": 0.9, "document": {"doc_id": "doc-1"}}],
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
            "answer": "The company policy clearly states employees are entitled to 20 days of annual leave.",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "The company policy clearly states employees are entitled to 20 days of annual leave.",
                "bullets": ["20 days annual leave", "Applies to all full-time employees"],
                "explanation": "This is directly stated in the company policy document.",
                "evidence_refs": ["chunk:chunk-1"],
            },
            "thinking": ["Found the leave policy in the documents"],
            "trace": {"user_scoped": bool(user_id), "evidence": retrieval_trace.get("evidence")},
        },
    )
    monkeypatch.setattr(
        agentic.policy_guard,
        "evaluate_answer",
        lambda **_kwargs: {"passed": True, "retryable": False, "issues": [], "grounded_evidence_count": 1, "provenance_count": 1},
    )
    
    result = agentic.run_agentic_query("What is the annual leave policy?", user_id="u1")
    
    # Verify generator created a grounded answer
    assert "annual leave" in result["answer"].lower()
    assert result["trace"]["agentic"]["generator"]["answer_mode"] == "short"
    assert result["trace"]["agentic"]["generator"]["reason_code"] == "direct_lookup"
    assert len(result["thinking"]) > 0


def test_agent_critic_validates_answer_grounding(monkeypatch):
    """Verify critic validates answer grounding and passes well-grounded answers."""
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
                "evidence": [{"chunk_id": "chunk-1", "rank_score": 0.9, "document": {"doc_id": "doc-1"}}],
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
        lambda *_args, **_kwargs: {
            "answer": "test answer",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "test answer",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": ["chunk:chunk-1"],
            },
            "thinking": [],
            "trace": {"evidence": [{"chunk_id": "chunk-1"}]},
        },
    )
    
    # Critic validates and passes
    monkeypatch.setattr(
        agentic.policy_guard,
        "evaluate_answer",
        lambda **_kwargs: {
            "passed": True,
            "retryable": False,
            "issues": [],
            "grounded_evidence_count": 1,
            "provenance_count": 1
        },
    )
    
    result = agentic.run_agentic_query("Test query", user_id="u1")
    
    # Critic should have validated and passed
    assert result["trace"]["agentic"]["critic"]["passed"] is True
    assert result["trace"]["agentic"]["critic"]["grounded_evidence_count"] >= 1
    assert result["trace"]["agentic"]["status"] == "passed"


def test_agent_critic_flags_ungrounded_answer_for_review(monkeypatch):
    """Verify critic flags ungrounded answers for review."""
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
            "answer": "Ungrounded answer without evidence",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "Ungrounded answer",
                "bullets": [],
                "explanation": "ok",
                "evidence_refs": [],  # No evidence references
            },
            "thinking": [],
        },
    )
    
    # Critic flags as ungrounded and not retryable
    monkeypatch.setattr(
        agentic.policy_guard,
        "evaluate_answer",
        lambda **_kwargs: {
            "passed": False,
            "retryable": False,
            "issues": ["Answer lacks sufficient grounding"],
            "grounded_evidence_count": 0,
            "provenance_count": 0
        },
    )
    
    result = agentic.run_agentic_query("Test query", user_id="u1")
    
    # Critic should have flagged as ungrounded
    assert result["trace"]["agentic"]["critic"]["passed"] is False
    assert "grounding" in str(result["trace"]["agentic"]["critic"]["issues"]).lower() or len(result["trace"]["agentic"]["critic"]["issues"]) > 0
    assert result["trace"]["agentic"]["status"] == "needs_review"
