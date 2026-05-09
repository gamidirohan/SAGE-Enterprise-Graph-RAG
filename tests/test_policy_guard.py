from app import policy_guard


def test_policy_guard_flags_multi_item_queries_with_insufficient_coverage():
    verdict = policy_guard.evaluate_answer(
        query="Who are the various managers we have?",
        answer="Bijade is your manager.",
        answer_payload={"evidence_refs": ["chunk:chunk-1"]},
        trace={
            "query_profile": {
                "expects_multiple_items": True,
                "minimum_unique_evidence": 2,
            },
            "coverage": {
                "distinct_evidence_count": 1,
            },
            "evidence": [{"chunk_id": "chunk-1"}],
        },
        plan={"query_profile": {"expects_multiple_items": True, "minimum_unique_evidence": 2}},
    )

    assert verdict["passed"] is False
    assert "insufficient_answer_coverage" in verdict["issues"]
