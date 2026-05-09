from app import rerank


def test_rerank_promotes_query_specific_candidate_with_lexical_fallback(monkeypatch):
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

    assert result["trace"]["reranker"]["method"] == "lexical"
    assert result["trace"]["result_count"] == 3
    assert result["trace"]["evidence"][0]["chunk_id"] == "chunk-gen2"
    assert "Gen 2 firmware" in result["documents"][0]


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
    assert [item.get("chunk_id") or item.get("fact_id") for item in result["trace"]["evidence"]] == ["chunk-1", "fact-1"]
