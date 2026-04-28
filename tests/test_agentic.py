from app import agentic


def test_run_agentic_query_returns_agentic_trace(monkeypatch):
    monkeypatch.setattr(
        agentic.vector_search,
        "retrieve",
        lambda _query, user_id=None, strategy="hybrid": {
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
        agentic.vector_search,
        "retrieve",
        lambda _query, user_id=None, strategy="hybrid": {
            "documents": [],
            "trace": {"query_type": "general_search", "user_scoped": bool(user_id), "evidence": []},
        },
    )
    monkeypatch.setattr(agentic.rerank, "rerank", lambda documents, trace: {"documents": documents, "trace": trace})
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
