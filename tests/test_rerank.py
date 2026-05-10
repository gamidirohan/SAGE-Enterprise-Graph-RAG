from app import rerank


def test_rerank_disables_semantic_reranking_when_cross_encoder_is_unavailable(monkeypatch):
    monkeypatch.setattr(rerank, "_cross_encoder_scores", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    trace = {
        "query": "How do I update the firmware on a Gen 2 router?",
        "evidence": [
            {
                "chunk_id": "chunk-gen1",
                "chunk_summary": "How to reset a Gen 1 router safely before resale.",
                "similarity": 0.95,
                "rank_score": 0.95,
                "relationship": "PART_OF",
                "direction": "outgoing",
                "document": {"doc_id": "doc-1", "subject": "Gen 1 reset guide", "sender": "ops"},
            },
            {
                "chunk_id": "chunk-gen2",
                "chunk_summary": "Firmware update steps for Gen 2 router devices.",
                "similarity": 0.71,
                "rank_score": 0.71,
                "relationship": "PART_OF",
                "direction": "outgoing",
                "document": {"doc_id": "doc-2", "subject": "Gen 2 firmware", "sender": "ops"},
            },
            {
                "chunk_id": "chunk-wifi",
                "chunk_summary": "General router Wi-Fi troubleshooting checklist.",
                "similarity": 0.69,
                "rank_score": 0.69,
                "relationship": "PART_OF",
                "direction": "outgoing",
                "document": {"doc_id": "doc-3", "subject": "Router troubleshooting", "sender": "ops"},
            },
        ],
    }

    result = rerank.rerank([], trace)

    assert result["trace"]["reranker"]["method"] == "unavailable"
    assert result["trace"]["reranker"]["enabled"] is False
    assert result["trace"]["reranked"] is False
    assert result["trace"]["result_count"] == 3
    assert result["trace"]["evidence"][0]["chunk_id"] == "chunk-gen1"
    assert "Gen 1 reset guide" in result["documents"][0]


def test_rerank_trims_to_top_k_and_preserves_score_sort_without_query():
    trace = {
        "evidence": [
            {"chunk_id": "chunk-1", "chunk_summary": "one", "rank_score": 0.4, "similarity": 0.4, "document": {"doc_id": "doc-1"}},
            {"chunk_id": "chunk-2", "chunk_summary": "two", "rank_score": 0.9, "similarity": 0.9, "document": {"doc_id": "doc-2"}},
            {"chunk_id": "chunk-3", "chunk_summary": "three", "rank_score": 0.8, "similarity": 0.8, "document": {"doc_id": "doc-3"}},
            {"chunk_id": "chunk-4", "chunk_summary": "four", "rank_score": 0.7, "similarity": 0.7, "document": {"doc_id": "doc-4"}},
        ]
    }

    result = rerank.rerank([], trace)

    assert result["trace"]["reranker"]["method"] == "score_sort"
    assert result["trace"]["result_count"] == 3
    assert [item["chunk_id"] for item in result["trace"]["evidence"]] == ["chunk-2", "chunk-3", "chunk-4"]


def test_rerank_collapses_duplicate_evidence_and_keeps_distinct_candidate(monkeypatch):
    trace = {
        "query": "Who are the various managers we have?",
        "evidence": [
            {
                "chunk_id": "chunk-1",
                "chunk_summary": "From now on I'm your only Manager Bijade",
                "similarity": 0.95,
                "rank_score": 0.95,
                "document": {"doc_id": "doc-1", "subject": "chat"},
            },
            {
                "chunk_id": "chunk-2",
                "chunk_summary": "From now on I'm your only Manager Bijade",
                "similarity": 0.94,
                "rank_score": 0.94,
                "document": {"doc_id": "doc-2", "subject": "chat"},
            },
            {
                "fact_id": "fact-1",
                "fact_summary": "George Brown reports to Hannah Garcia.",
                "similarity": 0.8,
                "rank_score": 0.8,
                "document": {"doc_id": "doc-3"},
                "fact": {"canonical_key": "reports_to::7", "subject_entity_id": "George Brown", "object_entity_id": "Hannah Garcia"},
            },
        ],
    }
    monkeypatch.setattr(rerank, "_cross_encoder_scores", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    result = rerank.rerank([], trace)

    assert result["trace"]["result_count"] == 2
    assert result["trace"]["reranker"]["distinct_candidate_count"] == 2
    assert result["trace"]["reranker"]["method"] == "unavailable"
    assert [item.get("chunk_id") or item.get("fact_id") for item in result["trace"]["evidence"]] == ["chunk-1", "fact-1"]


def test_rerank_keeps_exact_fact_match_ahead_of_global_semantic_match(monkeypatch):
    trace = {
        "query": "What did I promise to send and by when?",
        "evidence": [
            {
                "fact_id": "fact-global",
                "fact_summary": "Other user will send Project Alpha budget by 9pm tomorrow.",
                "rank_score": 9.0,
                "document": {"doc_id": "doc-global"},
                "fact": {"claim_type": "TASK_ASSIGNMENT", "canonical_key": "assignment::other"},
            },
            {
                "fact_id": "fact-exact",
                "fact_summary": "Current user will send Project Alpha budget by 6pm tomorrow.",
                "rank_score": 1.0,
                "exact_match": True,
                "document": {"doc_id": "doc-exact"},
                "fact": {"claim_type": "TASK_ASSIGNMENT", "canonical_key": "assignment::current"},
            },
            {
                "chunk_id": "chunk-global",
                "chunk_summary": "Correction: I'll send you the Project Alpha budget by 9pm tomorrow instead.",
                "rank_score": 8.0,
                "document": {"doc_id": "doc-chunk"},
            },
        ],
    }
    monkeypatch.setattr(rerank, "_cross_encoder_scores", lambda *_args, **_kwargs: ([9.0, 1.0, 8.0], "fake"))

    result = rerank.rerank([], trace)

    assert result["trace"]["evidence"][0]["fact_id"] == "fact-exact"


def test_rerank_keeps_fact_priority_ahead_of_chunk(monkeypatch):
    trace = {
        "query": "When am I sending the Project Alpha budget now?",
        "evidence": [
            {
                "chunk_id": "chunk-old",
                "chunk_summary": "I'll send you the Project Alpha budget by 6pm tomorrow.",
                "rank_score": 9.0,
                "document": {"doc_id": "doc-old"},
            },
            {
                "fact_id": "fact-current",
                "fact_summary": "Current user will send Project Alpha budget by 9pm tomorrow.",
                "rank_score": 1.0,
                "fact_priority": True,
                "document": {"doc_id": "doc-current"},
                "fact": {"claim_type": "TASK_ASSIGNMENT", "canonical_key": "assignment::current"},
            },
        ],
    }
    monkeypatch.setattr(rerank, "_cross_encoder_scores", lambda *_args, **_kwargs: ([9.0, 1.0], "fake"))

    result = rerank.rerank([], trace)

    assert result["trace"]["evidence"][0]["fact_id"] == "fact-current"


def test_rerank_prefers_future_task_fact_over_stale_past_fact(monkeypatch):
    trace = {
        "query": "When am I sending the Project Alpha budget now?",
        "query_type": "task_commitment_lookup",
        "evidence": [
            {
                "fact_id": "fact-old",
                "fact_summary": "Current user will send Project Alpha budget to bijade on 2020-04-14T21:00:00+00:00.",
                "rank_score": 9.0,
                "document": {"doc_id": "doc-old", "timestamp": "2020-04-13T10:00:00+00:00"},
                "fact": {
                    "claim_type": "TASK_ASSIGNMENT",
                    "canonical_key": "assignment::old",
                    "status": "current",
                    "subject_entity_id": "currentUser",
                    "object_entity_id": "1774788188804",
                    "temporal_start": "2020-04-14T21:00:00+00:00",
                    "last_seen_at": "2020-04-13T10:00:00+00:00",
                },
            },
            {
                "fact_id": "fact-future",
                "fact_summary": "Current user will send Project Alpha budget on 2099-05-12T20:00:00+00:00.",
                "rank_score": 1.0,
                "document": {"doc_id": "doc-future", "timestamp": "2099-05-10T08:00:00+00:00"},
                "fact": {
                    "claim_type": "TASK_ASSIGNMENT",
                    "canonical_key": "assignment::future",
                    "status": "current",
                    "subject_entity_id": "currentUser",
                    "temporal_start": "2099-05-12T20:00:00+00:00",
                    "last_seen_at": "2099-05-10T08:00:00+00:00",
                },
            },
        ],
    }
    monkeypatch.setattr(rerank, "_cross_encoder_scores", lambda *_args, **_kwargs: ([9.0, 1.0], "fake"))

    result = rerank.rerank([], trace)

    assert len(result["trace"]["evidence"]) == 1
    assert result["trace"]["evidence"][0]["fact_id"] == "fact-future"


def test_rerank_keeps_multiple_current_task_facts_when_recipients_conflict(monkeypatch):
    trace = {
        "query": "When am I sending the Project Alpha budget now?",
        "query_type": "task_commitment_lookup",
        "evidence": [
            {
                "fact_id": "fact-alice",
                "fact_summary": "Test User will send Project Alpha budget to Alice Johnson on 2099-05-12T20:00:00+00:00",
                "rank_score": 9.0,
                "document": {"doc_id": "doc-alice", "timestamp": "2099-05-10T08:00:00+00:00"},
                "fact": {
                    "claim_type": "TASK_ASSIGNMENT",
                    "canonical_key": "assignment::direct:currentUser:1::send-project-alpha-budget",
                    "status": "current",
                    "subject_entity_id": "currentUser",
                    "object_entity_id": "1",
                    "temporal_start": "2099-05-12T20:00:00+00:00",
                    "last_seen_at": "2099-05-10T08:00:00+00:00",
                },
            },
            {
                "fact_id": "fact-bijade",
                "fact_summary": "Test User will send Project Alpha budget to bijade on 2099-05-12T21:00:00+00:00",
                "rank_score": 8.5,
                "document": {"doc_id": "doc-bijade", "timestamp": "2099-05-10T08:30:00+00:00"},
                "fact": {
                    "claim_type": "TASK_ASSIGNMENT",
                    "canonical_key": "assignment::direct:currentUser:1774788188804::send-project-alpha-budget",
                    "status": "current",
                    "subject_entity_id": "currentUser",
                    "object_entity_id": "1774788188804",
                    "temporal_start": "2099-05-12T21:00:00+00:00",
                    "last_seen_at": "2099-05-10T08:30:00+00:00",
                },
            },
        ],
    }
    monkeypatch.setattr(rerank, "_cross_encoder_scores", lambda *_args, **_kwargs: ([9.0, 8.5], "fake"))

    result = rerank.rerank([], trace)

    assert result["trace"]["task_lookup_ambiguity"]["ambiguous"] is True
    assert result["trace"]["task_lookup_ambiguity"]["reason"] == "multiple_recipients"
    assert {item["fact_id"] for item in result["trace"]["evidence"]} == {"fact-alice", "fact-bijade"}
