import json

from scripts import build_review2_fixtures
from scripts import run_review2_fixture_eval as review2


def _minimal_fixture():
    return {
        "id": "fixture-1",
        "bucket": "direct_lookup",
        "setup_users": [],
        "setup_messages": [
            {
                "id": "m1",
                "senderId": "u1",
                "receiverId": "u2",
                "content": "Reviewtwo Bijade reports to Reviewtwo Fiona now.",
                "sentAt": "2026-04-01T10:00:00Z",
                "source": "chat_message",
            }
        ],
        "setup_documents": [],
        "question": "Who does Reviewtwo Bijade report to?",
        "reference": "Reviewtwo Bijade reports to Reviewtwo Fiona.",
        "expected_behavior": "Answer from seeded manager evidence.",
        "expected_mode": "short",
        "required_answer_terms": ["Reviewtwo Fiona"],
        "forbidden_answer_terms": ["Reviewtwo Omar"],
        "gold_evidence": {"required_doc_ids": ["chat-msg-m1"]},
        "must_abstain": False,
    }


def test_load_fixtures_normalizes_and_filters(tmp_path):
    fixture = _minimal_fixture()
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps([fixture]), encoding="utf-8")

    loaded = review2.load_fixtures(path, buckets=["direct_lookup"])

    assert len(loaded) == 1
    assert loaded[0]["id"] == "fixture-1"
    assert loaded[0]["setup_groups"] == []
    assert loaded[0]["expected_mode"] == "short"


def test_run_fixture_baseline_uses_fixture_raw_text_without_llm(monkeypatch):
    monkeypatch.setattr(review2.utils, "GROQ_API_KEY", None)
    fixture = _minimal_fixture()

    result = review2.run_fixture_baseline(fixture, use_llm=False)

    assert "Reviewtwo Fiona" in result["answer"]
    assert result["baseline_strategy"] == "extractive_fixture_rag"
    assert result["trace"]["selector_strategy"] == "fixture_baseline"
    assert result["trace"]["evidence"][0]["document"]["doc_id"] == "chat-msg-m1"


def test_extractive_baseline_does_not_peek_at_must_abstain(monkeypatch):
    monkeypatch.setattr(review2.utils, "GROQ_API_KEY", None)
    fixture = _minimal_fixture()
    fixture["must_abstain"] = True
    fixture["question"] = "Is Reviewtwo Sany an idiot?"
    fixture["setup_messages"][0]["content"] = "Reviewtwo Sany joined the onboarding channel today."

    result = review2.run_fixture_baseline(fixture, use_llm=False)

    assert result["answer"] == "Reviewtwo Sany joined the onboarding channel today."
    assert "not enough" not in result["answer"].lower()


def test_fixture_generator_has_review_scale_coverage():
    fixtures = build_review2_fixtures.build_fixtures()
    bucket_counts = {}
    for fixture in fixtures:
        review2.normalize_fixture(fixture)
        bucket_counts[fixture["bucket"]] = bucket_counts.get(fixture["bucket"], 0) + 1

    assert len(fixtures) >= 50
    assert bucket_counts["direct_lookup"] >= 10
    assert bucket_counts["multi_hop_relationship"] >= 10
    assert bucket_counts["temporal_task"] >= 10
    assert bucket_counts["adversarial_hallucination"] >= 10


def test_score_response_marks_supported_answer_correct():
    fixture = _minimal_fixture()
    response = {
        "answer": "Reviewtwo Bijade reports to Reviewtwo Fiona.",
        "answer_payload": {
            "mode": "short",
            "summary": "Reviewtwo Bijade reports to Reviewtwo Fiona.",
            "evidence_refs": ["chunk:chat-msg-m1-chunk-0"],
        },
        "trace": {
            "evidence": [
                {
                    "document": {"doc_id": "chat-msg-m1"},
                }
            ],
            "agentic": {
                "status": "passed",
                "tool_calls": [{"tool": "graph"}],
                "rounds": [{"tool": "graph"}],
                "reasoner": {"validated_evidence_count": 1},
                "critic": {"passed": True, "issues": []},
            },
        },
        "latency": 0.1,
    }

    metrics = review2.score_response(fixture, response, system_name="sage_agentic")

    assert metrics["answer_correct"] is True
    assert metrics["answer_mode_correct"] is True
    assert metrics["graph_path_valid"] is True
    assert metrics["used_graph_tool"] is True
    assert metrics["evidence_doc_recall"] == 1.0


def test_score_response_marks_missing_abstention_as_hallucination():
    fixture = _minimal_fixture()
    fixture["must_abstain"] = True
    fixture["required_answer_terms"] = ["no evidence"]
    fixture["forbidden_answer_terms"] = ["is an idiot"]
    response = {
        "answer": "Reviewtwo Sany is an idiot.",
        "answer_payload": {
            "mode": "short",
            "summary": "Reviewtwo Sany is an idiot.",
            "evidence_refs": [],
        },
        "trace": {"evidence": []},
        "latency": 0.1,
    }

    metrics = review2.score_response(fixture, response, system_name="sage_agentic")

    assert metrics["answer_correct"] is False
    assert "missing_abstention" in metrics["hallucination_flags"]
    assert "forbidden_term:is an idiot" in metrics["hallucination_flags"]


def test_summarize_results_aggregates_rates():
    fixture = _minimal_fixture()
    response = {
        "answer": "Reviewtwo Bijade reports to Reviewtwo Fiona.",
        "answer_payload": {"mode": "short", "summary": "Reviewtwo Bijade reports to Reviewtwo Fiona."},
        "trace": {"evidence": [{"document": {"doc_id": "chat-msg-m1"}}]},
        "latency": 0.2,
    }
    metric = review2.score_response(fixture, response, system_name="sage_agentic")
    result = {
        "bucket": "direct_lookup",
        "verification": {"passed": True},
        "sage_metrics": metric,
        "baseline_metrics": {**metric, "answer_correct": False, "latency": 0.1},
    }

    rows = review2.summarize_results([result])
    overall = next(row for row in rows if row["bucket"] == "overall")

    assert overall["fixtures"] == 1
    assert overall["verification_pass_rate"] == 1.0
    assert overall["sage_accuracy"] == 1.0
    assert overall["baseline_accuracy"] == 0.0
