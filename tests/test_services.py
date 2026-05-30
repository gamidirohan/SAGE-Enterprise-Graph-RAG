from datetime import datetime, timezone

from app import services


def test_extract_structured_data_falls_back_without_groq_key(monkeypatch):
    monkeypatch.setattr(services.utils, "GROQ_API_KEY", None)

    result = services.extract_structured_data("hello world", "doc-1")

    assert result == {
        "doc_id": "doc-1",
        "sender": "Unknown",
        "receivers": [],
        "subject": "No Subject",
        "content": "hello world",
    }


def test_generate_streamlit_response_returns_answer(monkeypatch):
    monkeypatch.setattr(
        services,
        "generate_groq_response",
        lambda _q, _d: {"answer": "final answer", "answer_payload": {"summary": "final answer"}, "thinking": ["step"]},
    )

    result = services.generate_streamlit_response("q", ["ctx"])

    assert result == "final answer"


def test_graph_vector_queries_exclude_sage_documents():
    assert "coalesce(d.conversation_type, '') <> 'sage'" in services.GRAPH_VECTOR_QUERY
    assert "NOT coalesce(d.source, '') STARTS WITH 'sage_'" in services.GRAPH_VECTOR_QUERY
    assert "coalesce(d.conversation_type, '') <> 'sage'" in services.PERSON_GRAPH_VECTOR_QUERY
    assert "NOT coalesce(d.source, '') STARTS WITH 'sage_'" in services.PERSON_GRAPH_VECTOR_QUERY
    assert "WITH person, c, d, pd, c.embedding AS chunk_embedding" in services.PERSON_GRAPH_VECTOR_QUERY
    assert "WITH person, c, d, pd, similarity + recency_weight AS similarity" in services.PERSON_GRAPH_VECTOR_QUERY
    assert "relationships(path)" not in services.GRAPH_VECTOR_QUERY
    assert "relationships(path)" not in services.PERSON_GRAPH_VECTOR_QUERY


def test_classify_query_marks_multi_part_lookup_as_compound():
    assert services._classify_query("What's the new project? When's the orientation? Who all know it so far?") == "compound_lookup"


def test_classify_query_marks_review_questions_as_schedule():
    assert services._classify_query("When is the Project Alpha review?") == "schedule_or_timeline"


def test_classify_query_marks_sending_question_as_task_lookup():
    assert services._classify_query("When am I sending the Project Alpha budget now?") == "task_commitment_lookup"


def test_classify_query_prefers_broad_explanation_over_schedule_keyword():
    assert (
        services._classify_query(
            "Explain all dashboard-related conversations with Elijah, including design review, brand guidelines, and dark mode status."
        )
        == "explanation"
    )


def test_personalized_lookup_ignores_discourse_give_me_prefix():
    assert services._is_personalized_lookup("Give me a detailed summary of everything we know about Project Beta.") is False
    assert services._is_personalized_lookup("Show me what I promised Alice.") is True


def test_build_answer_payload_converts_iso_utc_timestamps_to_ist():
    payload = services._build_answer_payload(
        mode="long",
        reason_code="evidence_complexity",
        summary="Project Alpha review is scheduled for 2026-04-20T10:00:00+00:00.",
        bullets=["Budget goes out at 2026-04-13T18:00:00+00:00."],
        retrieval_trace=None,
    )

    assert payload["summary"] == "Project Alpha review is scheduled for 2026-04-20 03:30 PM IST."
    assert payload["bullets"] == ["Budget goes out at 2026-04-13 11:30 PM IST."]


class _Ctx:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class _Driver:
    def __init__(self, session):
        self._session = session
        self.closed = False

    def session(self, **_kwargs):
        return _Ctx(self._session)

    def close(self):
        self.closed = True


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def run(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return _Result(self.rows)
        return _Result([])


class _Model:
    def encode(self, _text):
        return [0.1, 0.2]


def test_query_graph_with_trace_returns_evidence(monkeypatch):
    rows = [
        {
            "chunk_id": "chunk-1",
            "chunk_summary": "Charlie asked about weekend plans.",
            "d": {"doc_id": "doc-1", "subject": "Weekend", "sender": "3"},
            "similarity": 0.93,
            "relationship": "RECEIVED_BY",
            "n": {"id": "3", "name": "Charlie Davis", "_labels": ["Person"]},
        }
    ]
    session = _Session(rows)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("Do I have any plans this weekend?", user_id="currentUser")

    assert result["trace"]["user_scoped"] is True
    assert result["trace"]["result_count"] == 1
    assert result["trace"]["max_hop_count"] >= 2
    assert result["trace"]["evidence"][0]["document"]["doc_id"] == "doc-1"
    assert "Charlie Davis" in result["trace"]["matched_entities"]
    assert driver.closed is True


def test_query_graph_with_trace_uses_shallow_seed_depth_for_direct_lookup(monkeypatch):
    rows = [
        {
            "chunk_id": "chunk-hq",
            "chunk_summary": "HQ is located at 123 Enterprise Way, Suite 400.",
            "d": {"doc_id": "doc-hq", "subject": "Facilities", "sender": "ops"},
            "similarity": 0.97,
            "relationship": "PART_OF",
            "n": None,
            "hop_count": 1,
            "path_nodes": ["Facilities", "chunk-hq"],
            "path_relationships": ["PART_OF"],
        }
    ]
    session = _Session(rows)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("What is the office address for HQ?")

    assert result["trace"]["graph_depth"]["seed_hops"] == 0
    assert result["trace"]["max_hop_count"] == 1
    assert result["trace"]["evidence"][0]["related_node"] == {}
    assert driver.closed is True


def test_extract_query_focus_terms_keeps_short_acronyms():
    assert "hq" in services._extract_query_focus_terms("What is the office address for HQ?")


def test_extract_query_focus_terms_drops_auxiliary_has_from_task_query():
    focus_terms = services._extract_query_focus_terms("By when does george has to submit the project alpha documents")

    assert "has" not in focus_terms
    assert {"george", "submit", "alpha", "documents"}.issubset(set(focus_terms))


def test_generate_groq_response_abstains_on_unfocused_direct_lookup_evidence(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "general_search",
        "query_profile": {"wants_list_format": False, "requires_broad_coverage": False},
        "evidence": [
            {
                "fact_id": "fact-busy",
                "fact_summary": "Test Alice is busy.",
                "fact": {"claim_type": "STATUS", "display_summary": "Test Alice is busy."},
                "document": {"doc_id": "chat-msg-busy", "content": "Test Alice is busy."},
            }
        ],
    }

    result = services.generate_groq_response(
        "What is the office address for HQ?",
        ["Fact Summary: Test Alice is busy, Fact ID: fact-busy"],
        retrieval_trace=trace,
    )

    assert result["answer"] == "I couldn't find relevant evidence for that lookup."
    assert result["answer_payload"]["evidence_refs"] == ["fact:fact-busy"]


def test_generate_groq_response_abstains_on_weak_focus_task_evidence(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "task_commitment_lookup",
        "query_profile": {"wants_list_format": False, "requires_broad_coverage": False},
        "evidence": [
            {
                "fact_id": "fact-assignment",
                "fact_summary": "Fresh Alice 950 starts working on Project Alpha on 9pm.",
                "fact": {
                    "claim_type": "ASSIGNMENT_STATE",
                    "display_summary": "Fresh Alice 950 starts working on Project Alpha on 9pm.",
                    "temporal_start": "2026-05-14T21:00:00+00:00",
                },
                "document": {"doc_id": "chat-msg-assignment", "content": "Fresh Alice 950 starts working on Project Alpha on 9pm."},
            }
        ],
    }

    result = services.generate_groq_response(
        "By when does george has to submit the project alpha documents",
        ["Fact Summary: Fresh Alice 950 starts working on Project Alpha on 9pm."],
        retrieval_trace=trace,
    )

    assert result["answer"] == "I couldn't find relevant evidence for that lookup."


def test_query_graph_with_trace_does_not_scope_broad_summary_to_authenticated_user(monkeypatch):
    session = _Session([])
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace(
        "Give me a detailed summary of everything we know about Project Beta, including meetings, bugs, preparation work, and follow-up actions.",
        user_id="currentUser",
    )

    assert result["trace"]["user_scoped"] is False
    assert result["trace"]["query_type"] == "general_search"
    assert driver.closed is True


def test_query_graph_with_trace_gives_recent_messages_a_soft_rank_boost(monkeypatch):
    now = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)

    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chunk-old",
                    "chunk_summary": "Project Alpha review happened in the older thread.",
                    "d": {
                        "doc_id": "doc-old",
                        "subject": "Old thread",
                        "sender": "u1",
                        "timestamp": "2026-02-01T09:00:00+00:00",
                    },
                    "similarity": 0.95,
                    "relationship": "PART_OF",
                    "n": {"id": "topic-alpha", "name": "Project Alpha", "_labels": ["Topic"]},
                },
                {
                    "chunk_id": "chunk-recent",
                    "chunk_summary": "Project Alpha review is this Monday at 10am.",
                    "d": {
                        "doc_id": "doc-recent",
                        "subject": "Recent thread",
                        "sender": "u2",
                        "timestamp": "2026-04-18T09:00:00+00:00",
                    },
                    "similarity": 0.81,
                    "relationship": "PART_OF",
                    "n": {"id": "topic-alpha", "name": "Project Alpha", "_labels": ["Topic"]},
                },
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())
    monkeypatch.setattr(services, "_utcnow", lambda: now)

    result = services.query_graph_with_trace("When is the Project Alpha review?")

    assert result["trace"]["evidence"][0]["chunk_id"] == "chunk-recent"
    assert result["trace"]["evidence"][0]["rank_score"] > result["trace"]["evidence"][1]["rank_score"]
    assert driver.closed is True


class _DispatchSession:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def run(self, query, **params):
        self.calls.append({"query": query, "params": params})
        return _Result(self.handler(query, params))


def test_query_graph_with_trace_merges_canonical_fact_results(monkeypatch):
    def handler(query, _params):
        if "f.claim_type IN $claim_types" in query:
            return [
                {
                    "fact_id": "fact-1",
                    "fact_summary": "u1 will send report to u2 on 2026-04-02",
                    "f": {
                        "fact_id": "fact-1",
                        "canonical_key": "assignment::direct:u1:u2::send-report",
                        "claim_type": "TASK_ASSIGNMENT",
                        "status": "current",
                        "subject_key": "u1",
                        "subject_entity_id": "u1",
                        "object_key": "u2",
                        "object_entity_id": "u2",
                        "temporal_start": "2026-04-02",
                        "temporal_granularity": "date",
                    },
                    "d": {"doc_id": "chat-msg-m1", "subject": "Chat message m1", "sender": "u1", "source": "chat_message"},
                    "similarity": 1.0,
                }
            ]
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chunk-1",
                    "chunk_summary": "Legacy chunk evidence.",
                    "d": {"doc_id": "doc-1", "subject": "Legacy", "sender": "u1"},
                    "similarity": 0.81,
                    "relationship": "PART_OF",
                    "n": {"id": "u1", "name": "Alice", "_labels": ["Person"]},
                }
            ]
        if "MATCH (f:CanonicalFact)" in query:
            return [
                {
                    "fact_id": "fact-1",
                    "fact_summary": "u1 will send report for u2 on 2026-04-02",
                    "f": {
                        "fact_id": "fact-1",
                        "canonical_key": "assignment::direct:u1:u2::send-report",
                        "claim_type": "TASK_ASSIGNMENT",
                        "status": "current",
                        "subject_key": "u1",
                        "subject_entity_id": "u1",
                        "object_key": "u2",
                        "object_entity_id": "u2",
                        "temporal_start": "2026-04-02",
                        "temporal_granularity": "date",
                    },
                    "d": {"doc_id": "chat-msg-m1", "subject": "Chat message m1", "sender": "u1", "source": "chat_message"},
                    "similarity": 0.62,
                }
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("What am I supposed to send tomorrow?", user_id="u1")

    assert result["trace"]["query_type"] == "task_commitment_lookup"
    assert result["trace"]["result_count"] == 1
    assert result["trace"]["evidence"][0]["fact_id"] == "fact-1"
    assert result["trace"]["evidence"][0]["relationship"] == "CANONICAL_FACT"
    assert result["trace"]["evidence"][0]["fact"]["claim_type"] == "TASK_ASSIGNMENT"
    assert result["documents"][0].startswith("Fact Summary: u1 will send report to u2 on 2026-04-02")
    assert driver.closed is True


def test_query_graph_with_trace_prioritizes_exact_task_fact_over_higher_similarity_chunk(monkeypatch):
    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chunk-1",
                    "chunk_summary": "Weekend plans and casual chat.",
                    "d": {"doc_id": "doc-1", "subject": "Weekend", "sender": "u3"},
                    "similarity": 0.92,
                    "relationship": "PART_OF",
                    "n": {"id": "u3", "name": "Charlie", "_labels": ["Person"]},
                }
            ]
        if "f.claim_type IN $claim_types" in query:
            return [
                {
                    "fact_id": "fact-commitment",
                    "fact_summary": "u1 will share report to u2 on 2026-04-02T20:00:00Z",
                    "f": {
                        "fact_id": "fact-commitment",
                        "canonical_key": "assignment::direct:u1:u2::share-report",
                        "claim_type": "TASK_ASSIGNMENT",
                        "status": "current",
                        "subject_key": "u1",
                        "subject_entity_id": "u1",
                        "object_key": "u2",
                        "object_entity_id": "u2",
                        "temporal_start": "2026-04-02T20:00:00Z",
                        "temporal_granularity": "datetime",
                    },
                    "d": {"doc_id": "chat-msg-m9", "subject": "Chat message m9", "sender": "u1", "source": "chat_message"},
                    "similarity": 1.0,
                }
            ]
        if "MATCH (f:CanonicalFact)" in query:
            return []
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("What did I promise to send and by when?", user_id="u1")

    assert result["trace"]["query_type"] == "task_commitment_lookup"
    assert result["trace"]["result_count"] == 1
    assert result["trace"]["evidence"][0]["fact_id"] == "fact-commitment"
    assert result["trace"]["evidence"][0]["fact_priority"] is True


def test_query_graph_with_trace_rejects_unfocused_task_facts_when_query_has_exact_entities(monkeypatch):
    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chunk-proton",
                    "chunk_summary": "Charlie will work on Project Proton starting tomorrow.",
                    "d": {"doc_id": "chat-msg-proton", "subject": "Chat message", "sender": "u1"},
                    "similarity": 0.82,
                    "relationship": "PART_OF",
                    "n": {"id": "charlie-id", "name": "Charlie", "_labels": ["Person"]},
                }
            ]
        if "MATCH (f:CanonicalFact)" in query:
            return [
                {
                    "fact_id": "fact-alpha",
                    "fact_summary": "Alice Johnson will send Project Alpha budget on 2026-05-12T20:00:00+00:00",
                    "f": {
                        "fact_id": "fact-alpha",
                        "canonical_key": "assignment::direct:1:3::send-project-alpha-budget",
                        "claim_type": "TASK_ASSIGNMENT",
                        "status": "current",
                        "subject_key": "1",
                        "subject_entity_id": "1",
                        "object_key": "3",
                        "object_entity_id": "3",
                        "temporal_start": "2026-05-12T20:00:00+00:00",
                    },
                    "d": {"doc_id": "chat-msg-alpha", "subject": "Chat message", "sender": "1", "source": "chat_message"},
                    "similarity": 1.0,
                }
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("From when will Charlie start working on Project Proton?")

    assert result["trace"]["query_type"] == "task_commitment_lookup"
    assert result["trace"]["result_count"] == 1
    assert result["trace"]["evidence"][0]["chunk_id"] == "chunk-proton"
    assert all(item.get("fact_id") != "fact-alpha" for item in result["trace"]["evidence"])


def test_query_graph_with_trace_prioritizes_meeting_fact_for_schedule_lookup(monkeypatch):
    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chunk-noisy",
                    "chunk_summary": "Upcoming quarterly review for Project Alpha requires latest figures.",
                    "d": {"doc_id": "chat-msg-noisy", "subject": "Chat message", "sender": "u3"},
                    "similarity": 0.95,
                    "relationship": "PART_OF",
                    "n": {"id": "u3", "name": "Noisy User", "_labels": ["Person"]},
                }
            ]
        if "MATCH (f:CanonicalFact)" in query:
            return [
                {
                    "fact_id": "fact-meeting",
                    "fact_summary": "Project Alpha review scheduled for 2026-05-11T10:00:00+00:00",
                    "f": {
                        "fact_id": "fact-meeting",
                        "canonical_key": "meeting::group-alpha::project-alpha-review",
                        "claim_type": "MEETING_EVENT",
                        "status": "current",
                        "subject_key": "group-alpha",
                        "temporal_start": "2026-05-11T10:00:00+00:00",
                        "temporal_granularity": "datetime",
                    },
                    "d": {"doc_id": "chat-msg-meeting", "subject": "Chat message", "sender": "u1", "source": "chat_message"},
                    "similarity": 0.5,
                }
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("When is the Project Alpha review?")

    assert result["trace"]["query_type"] == "schedule_or_timeline"
    assert result["trace"]["evidence"][0]["fact_id"] == "fact-meeting"
    assert result["trace"]["evidence"][0]["fact"]["claim_type"] == "MEETING_EVENT"
    assert result["documents"][0].startswith("Fact Summary: Project Alpha review scheduled")


def test_query_graph_with_trace_filters_person_lookup_to_focus_entity(monkeypatch):
    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chunk-george-new",
                    "chunk_summary": "George Brown now reports to Vinitha.",
                    "d": {"doc_id": "chat-msg-george-new", "subject": "Chat message", "sender": "1"},
                    "similarity": 0.7387,
                    "relationship": "PART_OF",
                    "n": {"id": "7", "name": "George Brown", "_labels": ["Person"]},
                },
                {
                    "chunk_id": "chunk-charlie",
                    "chunk_summary": "There is no information available about Charlie's current activities.",
                    "d": {"doc_id": "chat-msg-charlie", "subject": "Chat message", "sender": "1"},
                    "similarity": 0.94,
                    "relationship": "PART_OF",
                    "n": {"id": "3", "name": "Charlie Davis", "_labels": ["Person"]},
                },
                {
                    "chunk_id": "chunk-george-old",
                    "chunk_summary": "George Brown reports to Bob Smith.",
                    "d": {"doc_id": "chat-msg-george-old", "subject": "Chat message", "sender": "1"},
                    "similarity": 0.7194,
                    "relationship": "PART_OF",
                    "n": {"id": "7", "name": "George Brown", "_labels": ["Person"]},
                },
            ]
        if "MATCH (f:CanonicalFact)" in query:
            return [
                {
                    "fact_id": "fact-share-report",
                    "fact_summary": "u1 will share report to u2 on 2026-04-02T20:00:00Z",
                    "f": {
                        "fact_id": "fact-share-report",
                        "canonical_key": "assignment::direct:u1:u2::share-report",
                        "claim_type": "TASK_ASSIGNMENT",
                        "status": "current",
                        "subject_key": "u1",
                        "subject_entity_id": "u1",
                        "object_key": "u2",
                        "object_entity_id": "u2",
                        "temporal_start": "2026-04-02T20:00:00Z",
                        "temporal_granularity": "datetime",
                    },
                    "d": {"doc_id": "chat-msg-share-report", "subject": "Chat message", "sender": "u1", "source": "chat_message"},
                    "similarity": 0.84,
                }
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("Whom does George Brown now report to?")

    assert result["trace"]["query_type"] == "person_lookup"
    assert result["trace"]["result_count"] == 2
    assert all("George Brown" in (item.get("chunk_summary") or item.get("fact_summary") or "") for item in result["trace"]["evidence"])
    assert all("Charlie" not in document for document in result["documents"])
    assert all("share report" not in document for document in result["documents"])
    assert driver.closed is True


def test_query_graph_with_trace_prioritizes_reports_to_fact_over_older_chunk(monkeypatch):
    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chunk-rohan-old",
                    "chunk_summary": "Rohan reports to Hrithik.",
                    "d": {"doc_id": "chat-msg-rohan-old", "subject": "Chat message", "sender": "1"},
                    "similarity": 0.97,
                    "relationship": "PART_OF",
                    "n": {"id": "7", "name": "Rohan", "_labels": ["Person"]},
                }
            ]
        if "MATCH (f:CanonicalFact)" in query:
            return [
                {
                    "fact_id": "fact-rohan-current",
                    "fact_summary": "Rohan reports to Anil Fresh.",
                    "f": {
                        "fact_id": "fact-rohan-current",
                        "canonical_key": "reports_to::rohan-id",
                        "claim_type": "REPORTS_TO",
                        "status": "current",
                        "subject_key": "rohan-id",
                        "subject_entity_id": "rohan-id",
                        "object_key": "anil-id",
                        "object_entity_id": "anil-id",
                        "last_seen_at": "2026-05-10T10:00:00+00:00",
                    },
                    "d": {"doc_id": "chat-msg-rohan-current", "subject": "Chat message", "sender": "1", "source": "chat_message"},
                    "similarity": 0.42,
                }
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("Who does Rohan report to?")

    assert result["trace"]["query_type"] == "person_lookup"
    assert result["trace"]["evidence"][0]["fact_id"] == "fact-rohan-current"
    assert driver.closed is True


def test_query_graph_with_trace_prioritizes_reports_to_fact_for_manager_wording(monkeypatch):
    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return []
        if "MATCH (f:CanonicalFact)" in query:
            return [
                {
                    "fact_id": "fact-assignment",
                    "fact_summary": "Charlie is assigned to Project Proton.",
                    "f": {
                        "fact_id": "fact-assignment",
                        "canonical_key": "assignment_state::charlie::project-proton",
                        "claim_type": "ASSIGNMENT_STATE",
                        "status": "current",
                        "subject_key": "charlie",
                        "subject_entity_id": "charlie",
                        "object_key": "project-proton",
                        "object_entity_id": "project-proton",
                    },
                    "d": {"doc_id": "chat-msg-assignment", "subject": "Chat message", "sender": "u1", "source": "chat_message"},
                    "similarity": 0.95,
                },
                {
                    "fact_id": "fact-manager",
                    "fact_summary": "Charlie reports to Alice Manager.",
                    "f": {
                        "fact_id": "fact-manager",
                        "canonical_key": "reports_to::charlie",
                        "claim_type": "REPORTS_TO",
                        "status": "current",
                        "subject_key": "charlie",
                        "subject_entity_id": "charlie",
                        "object_key": "alice-manager",
                        "object_entity_id": "alice-manager",
                        "display_summary": "Charlie reports to Alice Manager.",
                    },
                    "d": {"doc_id": "chat-msg-manager", "subject": "Chat message", "sender": "u1", "source": "chat_message"},
                    "similarity": 0.3,
                },
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("Who is Charlie's Manager now?")

    assert result["trace"]["query_type"] == "person_lookup"
    assert result["trace"]["evidence"][0]["fact_id"] == "fact-manager"
    assert result["trace"]["evidence"][0]["fact"]["claim_type"] == "REPORTS_TO"


def test_query_graph_with_trace_filters_group_request_lookup_to_object_terms(monkeypatch):
    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chunk-deck",
                    "chunk_summary": "Can you send the deck?",
                    "d": {
                        "doc_id": "chat-msg-deck",
                        "subject": "Chat message",
                        "sender": "1",
                        "conversation_type": "group",
                        "group_id": "g1",
                    },
                    "similarity": 0.5828,
                    "relationship": "PART_OF",
                    "n": {"id": "chat-msg-deck", "_labels": ["Node"]},
                },
                {
                    "chunk_id": "chunk-ui",
                    "chunk_summary": '{ "summary": "Elijah is asked to review the UI designs for the new dashboard." }',
                    "d": {"doc_id": "chat-msg-ui", "subject": "Chat message", "sender": "2", "conversation_type": "direct"},
                    "similarity": 0.2597,
                    "relationship": "PART_OF",
                    "n": {"id": "chat-msg-ui", "_labels": ["Node"]},
                },
                {
                    "chunk_id": "chunk-leave",
                    "chunk_summary": "I'm going on a leave on 19th Dec",
                    "d": {"doc_id": "chat-msg-leave", "subject": "Chat message", "sender": "8", "conversation_type": "direct"},
                    "similarity": 0.2351,
                    "relationship": "PART_OF",
                    "n": {"id": "chat-msg-leave", "_labels": ["Node"]},
                },
            ]
        if "MATCH (f:CanonicalFact)" in query:
            return [
                {
                    "fact_id": "fact-budget",
                    "fact_summary": "Alice Johnson will send Project Alpha budget to bijade on 2026-04-14T21:00:00+00:00",
                    "f": {
                        "fact_id": "fact-budget",
                        "canonical_key": "assignment::direct:1:1774788188804::send-project-alpha-budget",
                        "claim_type": "TASK_ASSIGNMENT",
                        "status": "current",
                        "subject_key": "1",
                        "subject_entity_id": "1",
                        "object_key": "1774788188804",
                        "object_entity_id": "1774788188804",
                        "temporal_start": "2026-04-14T21:00:00+00:00",
                        "temporal_granularity": "datetime",
                    },
                    "d": {"doc_id": "chat-msg-budget", "subject": "Chat message", "sender": "1", "source": "chat_message", "conversation_type": "direct"},
                    "similarity": 0.1863,
                }
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("Who was asked to send the deck?")

    assert result["trace"]["query_type"] == "person_lookup"
    assert result["trace"]["result_count"] == 1
    assert result["trace"]["evidence"][0]["chunk_summary"] == "Can you send the deck?"
    assert result["trace"]["evidence"][0]["document"]["conversation_type"] == "group"
    assert result["documents"] == [
        "Chunk Summary: Can you send the deck?, Document ID: chat-msg-deck, Conversation Type: group, Subject: Chat message, Sender: 1, Similarity: 0.5828, Relationship: PART_OF, Related Node: chat-msg-deck"
    ]
    assert driver.closed is True


def test_query_graph_with_trace_compound_lookup_keeps_mixed_evidence_and_filters_noisy_entities(monkeypatch):
    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chat-msg-m1-chunk-0",
                    "chunk_summary": "We have a Project Alpha review next Monday at 10am.",
                    "d": {"doc_id": "chat-msg-m1", "subject": "Chat message m1", "sender": "currentUser"},
                    "similarity": 0.61,
                    "rank_score": 0.74,
                    "relationship": "PART_OF",
                    "n": {"id": "group1", "name": "Project Alpha", "_labels": ["Topic"]},
                }
            ]
        if "MATCH (f:CanonicalFact)" in query:
            return [
                {
                    "fact_id": "fact-1",
                    "fact_summary": "Project Alpha review is scheduled for 2026-04-20T10:00:00Z",
                    "f": {
                        "fact_id": "fact-1",
                        "canonical_key": "meeting::group1::project-alpha-review",
                        "claim_type": "MEETING_EVENT",
                        "status": "current",
                        "subject_key": "group1",
                        "subject_entity_id": "group1",
                        "object_key": None,
                        "object_entity_id": None,
                        "temporal_start": "2026-04-20T10:00:00Z",
                        "temporal_granularity": "datetime",
                    },
                    "d": {"doc_id": "chat-msg-m1", "subject": "Chat message m1", "sender": "currentUser", "source": "chat_message"},
                    "similarity": 0.58,
                }
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("What's the new project? When's the orientation? Who all know it so far")

    assert result["trace"]["query_type"] == "compound_lookup"
    assert result["trace"]["result_count"] == 2
    assert any(item.get("fact_id") == "fact-1" for item in result["trace"]["evidence"])
    assert any(item.get("chunk_id") == "chat-msg-m1-chunk-0" for item in result["trace"]["evidence"])
    assert "Project Alpha" in result["trace"]["matched_entities"]
    assert "currentUser" not in result["trace"]["matched_entities"]
    assert "Chat message m1" not in result["trace"]["matched_entities"]
    assert "group1" not in result["trace"]["matched_entities"]
    assert driver.closed is True


def test_query_graph_with_trace_compound_lookup_prefers_focus_matched_chunk_over_unrelated_fact(monkeypatch):
    def handler(query, _params):
        if "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query:
            return [
                {
                    "chunk_id": "chat-msg-investor-chunk-0",
                    "chunk_summary": "Need to get on an investors call with Antler on Monday (11-May-26).",
                    "d": {"doc_id": "chat-msg-investor", "subject": "Chat message", "sender": "currentUser", "conversation_type": "group"},
                    "similarity": 0.66,
                    "rank_score": 0.66,
                    "relationship": "PART_OF",
                    "n": {"id": "g-940-saia", "_labels": ["Group"]},
                }
            ]
        if "MATCH (f:CanonicalFact)" in query:
            return [
                {
                    "fact_id": "fact-review",
                    "fact_summary": "Project Alpha review is scheduled for 2026-05-11T10:00:00Z",
                    "f": {
                        "fact_id": "fact-review",
                        "canonical_key": "meeting::group-alpha::project-alpha-review",
                        "claim_type": "MEETING_EVENT",
                        "status": "current",
                        "subject_key": "group-alpha",
                        "subject_entity_id": "group-alpha",
                        "temporal_start": "2026-05-11T10:00:00Z",
                        "temporal_granularity": "datetime",
                    },
                    "d": {"doc_id": "chat-msg-review", "subject": "Chat message", "sender": "u1", "source": "chat_message"},
                    "similarity": 0.82,
                }
            ]
        return []

    session = _DispatchSession(handler)
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "get_cached_embedding_model", lambda: _Model())

    result = services.query_graph_with_trace("When do we have an investor call and with whom is it?")

    assert result["trace"]["query_type"] == "compound_lookup"
    assert result["trace"]["evidence"][0]["chunk_id"] == "chat-msg-investor-chunk-0"
    assert all(item.get("fact_id") != "fact-review" for item in result["trace"]["evidence"])
    assert driver.closed is True


def test_generate_groq_response_builds_fact_first_context(monkeypatch):
    captured = {}

    class FakeChain:
        def __or__(self, _other):
            return self

        def invoke(self, payload):
            captured.update(payload)
            return '{"summary":"You promised to send the report by 2026-04-02T20:00:00Z.","bullets":["Recipient: u2","Time: 2026-04-02T20:00:00Z"]}'

    monkeypatch.setattr(services, "CHAT_PROMPT", FakeChain())
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: object())
    monkeypatch.setattr(services, "StrOutputParser", lambda: object())

    trace = {
        "query_type": "task_commitment_lookup",
        "query_profile": {"requires_broad_coverage": True, "expects_multiple_items": False, "wants_list_format": False},
        "evidence": [
            {
                "fact_id": "fact-1",
                "fact_summary": "u1 will share report to u2 on 2026-04-02T20:00:00Z",
                "similarity": 1.0,
                "document": {"doc_id": "chat-msg-m1"},
                "related_node": {"display_name": "assignment::direct:u1:u2::share-report"},
                "fact": {
                    "claim_type": "TASK_ASSIGNMENT",
                    "status": "current",
                    "canonical_key": "assignment::direct:u1:u2::share-report",
                    "subject_key": "u1",
                    "subject_entity_id": "u1",
                    "object_key": "u2",
                    "object_entity_id": "u2",
                    "temporal_start": "2026-04-02T20:00:00Z",
                    "temporal_granularity": "datetime",
                },
            },
            {
                "chunk_id": "chunk-1",
                "chunk_summary": "Weekend plans and casual chat.",
                "similarity": 0.92,
                "relationship": "PART_OF",
                "document": {"doc_id": "doc-1", "subject": "Weekend", "sender": "u3"},
                "related_node": {"display_name": "Charlie"},
            },
        ],
    }

    result = services.generate_groq_response(
        "What did I promise to send and by when?",
        ["Chunk Summary: Weekend plans and casual chat., Document ID: doc-1, Subject: Weekend, Sender: u3, Similarity: 0.92, Relationship: PART_OF, Related Node: Charlie"],
        user_id="u1",
        retrieval_trace=trace,
    )

    assert result["answer"] == "You promised to send the report by 2026-04-03 01:30 AM IST."
    assert result["answer_payload"]["mode"] == "long"
    assert result["answer_payload"]["reason_code"] == "evidence_complexity"
    assert result["answer_payload"]["evidence_refs"] == ["fact:fact-1", "chunk:chunk-1"]
    assert result["answer_payload"]["bullets"] == ["Recipient: u2", "Time: 2026-04-03 01:30 AM IST"]
    assert captured["user_context"].endswith("Query classification: task_commitment_lookup.")
    assert captured["retrieval_guidance"].startswith("This is a task or commitment lookup.")
    assert captured["answer_mode"] == "long"
    assert captured["context"].split("\n\n")[0].startswith("Canonical facts")


def test_generate_groq_response_uses_people_chat_window_when_available(monkeypatch):
    captured = {}

    class FakeChain:
        def __or__(self, _other):
            return self

        def invoke(self, payload):
            captured.update(payload)
            return '{"summary":"Alice Johnson was appointed as Project Manager.","bullets":[]}'

    monkeypatch.setattr(services, "CHAT_PROMPT", FakeChain())
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: object())
    monkeypatch.setattr(services, "StrOutputParser", lambda: object())
    monkeypatch.setattr(
        services,
        "_build_guarded_abstention_answer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("people-chat-backed SAGE answers should use the main generator directly")),
    )
    monkeypatch.setattr(
        services,
        "_build_graph_conversation_window",
        lambda *_args, **_kwargs: [
            "- 2026-05-12 01:30 AM IST | Bob Smith -> Alice Johnson | Alice Johnson was appointed as Project Manager."
        ],
    )

    trace = {
        "query_type": "person_lookup",
        "query_profile": {"wants_list_format": False},
        "evidence": [
            {
                "fact_id": "fact-1",
                "fact_summary": "Alice Johnson reports to Bob Smith.",
                "fact": {
                    "claim_type": "REPORTS_TO",
                    "status": "current",
                    "canonical_key": "reports_to::alice-id",
                    "display_summary": "Alice Johnson reports to Bob Smith.",
                },
                "document": {"doc_id": "chat-msg-m1"},
            }
        ],
    }

    result = services.generate_groq_response(
        "What is Alice Johnson appointed as?",
        [],
        retrieval_trace=trace,
        history=[{"content": "placeholder"}],
    )

    assert result["answer"] == "Alice Johnson was appointed as Project Manager."
    assert "Recent people chat evidence" in captured["conversation_window"]
    assert captured["context"].startswith("Canonical facts")
    assert "additional chat context" in captured["retrieval_guidance"]


def test_build_conversation_window_ignores_sage_history_when_people_chat_is_missing(monkeypatch):
    monkeypatch.setattr(services, "_build_graph_conversation_window", lambda *_args, **_kwargs: [])

    window = services._build_conversation_window(
        "What is Alice Johnson appointed as?",
        history=[{"content": "This is a SAGE-only thread message."}],
        retrieval_trace={"evidence": []},
    )

    assert window == "No recent people chat window was available."


def test_generate_groq_response_uses_fact_summary_for_reports_to_lookup(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "person_lookup",
        "query_profile": {"wants_list_format": False},
        "evidence": [
            {
                "fact_id": "fact-1",
                "fact_summary": "Rohan reports to Anil Fresh.",
                "fact": {
                    "claim_type": "REPORTS_TO",
                    "status": "current",
                    "canonical_key": "reports_to::rohan-id",
                    "display_summary": "Rohan reports to Anil Fresh.",
                },
                "document": {"doc_id": "chat-msg-m1"},
            },
            {
                "chunk_id": "chunk-1",
                "chunk_summary": "Rohan reports to Hrithik.",
                "document": {"doc_id": "chat-msg-m2"},
            },
        ],
    }

    result = services.generate_groq_response(
        "Who does Rohan report to?",
        [],
        retrieval_trace=trace,
    )

    assert result["answer"] == "Rohan reports to Anil Fresh."
    assert result["answer_payload"]["mode"] == "short"
    assert result["answer_payload"]["reason_code"] == "direct_lookup"


def test_generate_groq_response_uses_reports_to_fact_for_manager_lookup(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "person_lookup",
        "query_profile": {"wants_list_format": False},
        "evidence": [
            {
                "fact_id": "fact-assignment",
                "fact_summary": "Charlie is assigned to Project Proton.",
                "fact": {
                    "claim_type": "ASSIGNMENT_STATE",
                    "status": "current",
                    "display_summary": "Charlie is assigned to Project Proton.",
                },
                "document": {"doc_id": "chat-msg-assignment"},
            },
            {
                "fact_id": "fact-manager",
                "fact_summary": "Charlie reports to Alice Manager.",
                "fact": {
                    "claim_type": "REPORTS_TO",
                    "status": "current",
                    "canonical_key": "reports_to::charlie",
                    "display_summary": "Charlie reports to Alice Manager.",
                },
                "document": {"doc_id": "chat-msg-manager"},
            },
        ],
    }

    result = services.generate_groq_response(
        "Who is Charlie's Manager now?",
        [],
        retrieval_trace=trace,
    )

    assert result["answer"] == "Charlie reports to Alice Manager."
    assert "assigned" not in result["answer"].lower()


def test_generate_groq_response_refuses_assignment_fact_for_manager_lookup(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "person_lookup",
        "query_profile": {"wants_list_format": False},
        "evidence": [
            {
                "fact_id": "fact-assignment",
                "fact_summary": "Charlie is assigned to Project Proton.",
                "fact": {
                    "claim_type": "ASSIGNMENT_STATE",
                    "status": "current",
                    "display_summary": "Charlie is assigned to Project Proton.",
                },
                "document": {"doc_id": "chat-msg-assignment"},
            },
        ],
    }

    result = services.generate_groq_response(
        "Who is Charlie's Manager now?",
        [],
        retrieval_trace=trace,
    )

    assert result["answer"] == "I couldn't find current manager or reporting evidence for that lookup."
    assert "assigned" not in result["answer"].lower()


def test_generate_groq_response_answers_historical_manager_of_project_from_role_evidence(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "person_lookup",
        "query_profile": {"wants_list_format": False},
        "evidence": [
            {
                "chunk_id": "chunk-role",
                "chunk_summary": "Bob Smith was the manager of Project Alpha in 2022.",
                "document": {
                    "doc_id": "chat-msg-role",
                    "content": "Bob Smith was the manager of Project Alpha in 2022.",
                },
            },
            {
                "fact_id": "fact-budget",
                "fact_summary": "Alice Johnson will send Project Alpha budget on 2026-05-12T20:00:00+00:00",
                "fact": {
                    "claim_type": "TASK_ASSIGNMENT",
                    "status": "current",
                    "display_summary": "Alice Johnson will send Project Alpha budget on 2026-05-12T20:00:00+00:00",
                },
                "document": {"doc_id": "chat-msg-budget"},
            },
        ],
    }

    result = services.generate_groq_response(
        "Who was the manager of Project Alpha in 2022?",
        [],
        retrieval_trace=trace,
    )

    assert result["answer"] == "Bob Smith was the manager of Project Alpha in 2022."
    assert "current manager or reporting evidence" not in result["answer"]


def test_generate_groq_response_surfaces_person_lookup_fact_conflict_without_llm(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "person_lookup",
        "query_profile": {"wants_list_format": False},
        "fact_lookup_conflict": {"ambiguous": True, "claim_type": "REPORTS_TO"},
        "evidence": [
            {
                "fact_id": "fact-anil",
                "fact_summary": "Rohan reports to Anil Fresh.",
                "fact": {
                    "claim_type": "REPORTS_TO",
                    "status": "current",
                    "canonical_key": "reports_to::rohan-id",
                    "display_summary": "Rohan reports to Anil Fresh.",
                },
                "document": {"doc_id": "chat-msg-anil"},
            },
            {
                "fact_id": "fact-hrithik",
                "fact_summary": "Rohan reports to Hrithik.",
                "fact": {
                    "claim_type": "REPORTS_TO",
                    "status": "current",
                    "canonical_key": "reports_to::rohan-id",
                    "display_summary": "Rohan reports to Hrithik.",
                },
                "document": {"doc_id": "chat-msg-hrithik"},
            },
        ],
    }

    result = services.generate_groq_response(
        "Who does Rohan report to?",
        [],
        retrieval_trace=trace,
    )

    assert result["answer"] == "I found conflicting current reporting relationships for that lookup, so I can't collapse them to one safely."
    assert result["answer_payload"]["bullets"] == [
        "Rohan reports to Anil Fresh.",
        "Rohan reports to Hrithik.",
    ]


def test_build_response_context_hides_internal_metadata_but_keeps_group_signal():
    context = services._build_response_context(
        [],
        retrieval_trace={
            "retrieved_at": "2026-05-12T10:15:00Z",
            "evidence": [
                {
                    "chunk_id": "chunk-investor",
                    "chunk_summary": "Need to get on an investors call with Antler on Monday (11-May-26).",
                    "document": {
                        "doc_id": "chat-msg-investor",
                        "subject": "g-940-saia",
                        "sender": "9501",
                        "conversation_type": "group",
                        "timestamp": "2026-05-11T09:00:00Z",
                    },
                    "retrieved_at": "2026-05-12T10:15:00Z",
                    "related_node": {"display_name": "g-940-saia"},
                },
                {
                    "fact_id": "fact-1",
                    "fact_summary": "Project Alpha review is scheduled for 2026-05-11T10:00:00Z",
                    "fact": {
                        "claim_type": "MEETING_EVENT",
                        "status": "current",
                        "canonical_key": "meeting::g-940-saia::review-next-monday-at-10am",
                        "subject_entity_id": "g-940-saia",
                        "temporal_start": "2026-05-11T10:00:00Z",
                        "temporal_granularity": "datetime",
                        "last_seen_at": "2026-05-11T09:10:00Z",
                    },
                    "document": {"doc_id": "chat-msg-review", "conversation_type": "group", "timestamp": "2026-05-11T09:05:00Z"},
                    "retrieved_at": "2026-05-12T10:15:00Z",
                },
            ]
        },
    )

    assert "Conversation Type: group" in context
    assert "g-940-saia" not in context
    assert "9501" not in context
    assert "chat-msg-investor" not in context
    assert "canonical_key" not in context.lower()
    assert "Message Time: 2026-05-11 02:30 PM IST" in context
    assert "Source Message Time: 2026-05-11 02:35 PM IST" in context
    assert "Retrieved At: 2026-05-12 03:45 PM IST" in context


def test_generate_groq_response_uses_fact_summary_for_task_when_lookup(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "task_commitment_lookup",
        "evidence": [
            {
                "fact_id": "fact-1",
                "fact_summary": "Test User will send Project Alpha budget to Alice Johnson on 2026-05-10T15:30:00+00:00",
                "fact": {
                    "claim_type": "TASK_ASSIGNMENT",
                    "status": "current",
                    "canonical_key": "assignment::direct:currentUser:1::send-project-alpha-budget",
                    "display_summary": "Test User will send Project Alpha budget to Alice Johnson on 2026-05-10T15:30:00+00:00",
                    "temporal_start": "2026-05-10T15:30:00+00:00",
                },
                "document": {"doc_id": "chat-msg-m1"},
            }
        ],
    }

    result = services.generate_groq_response(
        "When am I sending the Project Alpha budget now?",
        ["Fact Summary: Test User will send Project Alpha budget to Alice Johnson on 2026-05-10T15:30:00+00:00"],
        user_id="currentUser",
        retrieval_trace=trace,
    )

    assert result["answer"] == "Test User will send Project Alpha budget to Alice Johnson on 2026-05-10 09:00 PM IST."
    assert result["answer_payload"]["mode"] == "short"
    assert result["answer_payload"]["reason_code"] == "direct_lookup"


def test_generate_groq_response_uses_assignment_start_for_from_when_lookup(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "task_commitment_lookup",
        "evidence": [
            {
                "fact_id": "fact-proton",
                "fact_summary": "Charlie is assigned to Project Proton.",
                "fact": {
                    "claim_type": "ASSIGNMENT_STATE",
                    "status": "current",
                    "canonical_key": "assignment_state::charlie-id::project-proton",
                    "display_summary": "Charlie is assigned to Project Proton.",
                    "subject_display": "Charlie",
                    "object_display": "Project Proton",
                    "temporal_start": "2026-05-12",
                    "temporal_end": "2026-07-12",
                },
                "document": {"doc_id": "chat-msg-proton"},
            }
        ],
    }

    result = services.generate_groq_response(
        "From when will Charlie start working on Project Proton?",
        [],
        retrieval_trace=trace,
    )

    assert result["answer"] == "Charlie starts working on Project Proton on 2026-05-12."
    assert result["answer_payload"]["mode"] == "short"
    assert result["answer_payload"]["reason_code"] == "direct_lookup"


def test_generate_groq_response_uses_source_text_start_when_fact_temporal_is_missing(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "task_commitment_lookup",
        "evidence": [
            {
                "fact_id": "fact-proton",
                "fact_summary": "Charlie is assigned to Project Proton.",
                "fact": {
                    "claim_type": "ASSIGNMENT_STATE",
                    "status": "current",
                    "canonical_key": "assignment_state::charlie::project-proton",
                    "display_summary": "Charlie is assigned to Project Proton.",
                    "subject_display": "Charlie",
                    "object_display": "Project Proton",
                },
                "document": {
                    "doc_id": "chat-msg-proton",
                    "content": "Hi Charlie, I'll be you manager for project proton, and you'll work on it for 2 months starting tomorrow",
                },
            }
        ],
    }

    result = services.generate_groq_response(
        "From when will Charlie start working on Project Proton?",
        [],
        retrieval_trace=trace,
    )

    assert result["answer"] == "Charlie starts working on Project Proton starting tomorrow."
    assert result["answer_payload"]["mode"] == "short"
    assert result["answer_payload"]["reason_code"] == "direct_lookup"


def test_generate_groq_response_answers_assignment_duration_project_without_manager(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "task_commitment_lookup",
        "query_profile": {"expects_multiple_items": False, "requires_broad_coverage": False, "wants_list_format": False},
        "evidence": [
            {
                "fact_id": "fact-manager",
                "fact_summary": "Charlie reports to Elijah Parker.",
                "fact": {
                    "claim_type": "REPORTS_TO",
                    "status": "current",
                    "display_summary": "Charlie reports to Elijah Parker.",
                    "subject_display": "Charlie",
                    "object_display": "Elijah Parker",
                },
                "document": {"doc_id": "chat-msg-other"},
            },
            {
                "fact_id": "fact-proton",
                "fact_summary": "Charlie is assigned to Project Proton.",
                "fact": {
                    "claim_type": "ASSIGNMENT_STATE",
                    "status": "current",
                    "canonical_key": "assignment_state::charlie::project-proton",
                    "display_summary": "Charlie is assigned to Project Proton.",
                    "subject_display": "Charlie",
                    "object_display": "Project Proton",
                    "temporal_start": "2026-05-12",
                    "temporal_end": "2026-07-12",
                },
                "document": {
                    "doc_id": "chat-msg-proton",
                    "content": "Hi Charlie, I'll be you manager for project proton, and you'll work on it for 2 months starting tomorrow",
                },
            },
        ],
    }

    result = services.generate_groq_response(
        "How long is Charlie going to work, and in which project",
        [],
        retrieval_trace=trace,
    )

    assert result["answer"] == "Charlie will work on Project Proton for 2 months."
    assert result["answer_payload"]["bullets"] == []
    assert "manager" not in result["answer"].lower()


def test_generate_groq_response_surfaces_task_lookup_ambiguity_without_llm(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "task_commitment_lookup",
        "task_lookup_ambiguity": {"ambiguous": True, "reason": "multiple_recipients"},
        "evidence": [
            {
                "fact_id": "fact-alice",
                "fact_summary": "Test User will send Project Alpha budget to Alice Johnson on 2026-05-10T15:30:00+00:00",
                "fact": {
                    "claim_type": "TASK_ASSIGNMENT",
                    "status": "current",
                    "canonical_key": "assignment::direct:currentUser:1::send-project-alpha-budget",
                    "display_summary": "Test User will send Project Alpha budget to Alice Johnson on 2026-05-10T15:30:00+00:00",
                    "temporal_start": "2026-05-10T15:30:00+00:00",
                },
                "document": {"doc_id": "chat-msg-a"},
            },
            {
                "fact_id": "fact-bijade",
                "fact_summary": "Test User will send Project Alpha budget to bijade on 2026-05-10T16:30:00+00:00",
                "fact": {
                    "claim_type": "TASK_ASSIGNMENT",
                    "status": "current",
                    "canonical_key": "assignment::direct:currentUser:1774788188804::send-project-alpha-budget",
                    "display_summary": "Test User will send Project Alpha budget to bijade on 2026-05-10T16:30:00+00:00",
                    "temporal_start": "2026-05-10T16:30:00+00:00",
                },
                "document": {"doc_id": "chat-msg-b"},
            },
        ],
    }

    result = services.generate_groq_response(
        "When am I sending the Project Alpha budget now?",
        [],
        user_id="currentUser",
        retrieval_trace=trace,
    )

    assert result["answer"] == "I found multiple current commitments that match that request, so I can't collapse them to one safely."
    assert result["answer_payload"]["bullets"] == [
        "Test User will send Project Alpha budget to Alice Johnson on 2026-05-10 09:00 PM IST.",
        "Test User will send Project Alpha budget to bijade on 2026-05-10 10:00 PM IST.",
    ]


def test_generate_groq_response_surfaces_schedule_fact_conflict_without_llm(monkeypatch):
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))

    trace = {
        "query_type": "schedule_or_timeline",
        "query_profile": {"wants_list_format": False},
        "fact_lookup_conflict": {"ambiguous": True, "claim_type": "MEETING_EVENT"},
        "evidence": [
            {
                "fact_id": "fact-1",
                "fact_summary": "Project Alpha review is scheduled for 2026-05-11T10:00:00+00:00",
                "fact": {
                    "claim_type": "MEETING_EVENT",
                    "status": "current",
                    "canonical_key": "meeting::group-alpha::project-alpha-review",
                    "display_summary": "Project Alpha review is scheduled for 2026-05-11T10:00:00+00:00",
                    "temporal_start": "2026-05-11T10:00:00+00:00",
                },
                "document": {"doc_id": "chat-msg-a"},
            },
            {
                "fact_id": "fact-2",
                "fact_summary": "Project Alpha review is scheduled for 2026-05-11T11:00:00+00:00",
                "fact": {
                    "claim_type": "MEETING_EVENT",
                    "status": "current",
                    "canonical_key": "meeting::group-alpha::project-alpha-review",
                    "display_summary": "Project Alpha review is scheduled for 2026-05-11T11:00:00+00:00",
                    "temporal_start": "2026-05-11T11:00:00+00:00",
                },
                "document": {"doc_id": "chat-msg-b"},
            },
        ],
    }

    result = services.generate_groq_response(
        "When is the Project Alpha review?",
        [],
        retrieval_trace=trace,
    )

    assert result["answer"] == "I found conflicting current schedule evidence for that lookup, so I can't collapse it to one safely."
    assert result["answer_payload"]["bullets"] == [
        "Project Alpha review is scheduled for 2026-05-11 03:30 PM IST.",
        "Project Alpha review is scheduled for 2026-05-11 04:30 PM IST.",
    ]


def test_build_response_context_marks_reports_to_object_as_manager():
    context = services._build_response_context(
        [],
        retrieval_trace={
            "evidence": [
                {
                    "fact_id": "fact-1",
                    "fact_summary": "George Brown reports to Hannah Garcia",
                    "fact": {
                        "claim_type": "REPORTS_TO",
                        "status": "current",
                        "subject_entity_id": "George Brown",
                        "object_entity_id": "Hannah Garcia",
                    },
                    "document": {"doc_id": "doc-1"},
                    "similarity": 0.9,
                }
            ]
        },
    )

    assert "Relationship Semantics: subject/person reports to object/manager" in context


def test_generate_groq_response_includes_group_ambiguity_guidance(monkeypatch):
    captured = {}

    class FakeChain:
        def __or__(self, _other):
            return self

        def invoke(self, payload):
            captured.update(payload)
            return '{"summary":"It is not clear who was asked to send the deck.","bullets":["The request appears in a group conversation without a single resolved target."]}'

    monkeypatch.setattr(services, "CHAT_PROMPT", FakeChain())
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: object())
    monkeypatch.setattr(services, "StrOutputParser", lambda: object())

    trace = {
        "query_type": "person_lookup",
        "evidence": [
            {
                "chunk_id": "chunk-deck",
                "chunk_summary": "Can you send the deck?",
                "similarity": 0.5828,
                "relationship": "PART_OF",
                "document": {"doc_id": "chat-msg-deck", "subject": "Chat message", "sender": "1", "conversation_type": "group"},
                "related_node": {"display_name": "chat-msg-deck"},
            }
        ],
    }

    result = services.generate_groq_response(
        "Who was asked to send the deck?",
        ["Chunk Summary: Can you send the deck?, Document ID: chat-msg-deck, Conversation Type: group, Subject: Chat message, Sender: 1, Similarity: 0.5828, Relationship: PART_OF, Related Node: chat-msg-deck"],
        retrieval_trace=trace,
    )

    assert result["answer"] == "It is not clear who was asked to send the deck."
    assert captured["answer_mode"] == "short"
    assert "target is ambiguous" in captured["retrieval_guidance"]
    assert "Conversation Type: group" in captured["context"]


def test_generate_groq_response_uses_explicit_long_mode(monkeypatch):
    class FakeChain:
        def __or__(self, _other):
            return self

        def invoke(self, _payload):
            return '{"summary":"Project Beta has multiple related updates.","bullets":["A client review is scheduled for tomorrow at 3pm.","Critical bugs were fixed and the test environment was updated."]}'

    monkeypatch.setattr(services, "CHAT_PROMPT", FakeChain())
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: object())
    monkeypatch.setattr(services, "StrOutputParser", lambda: object())

    trace = {
        "query_type": "general_search",
        "result_count": 2,
        "max_hop_count": 1,
        "evidence": [
            {"chunk_id": "chunk-1", "chunk_summary": "Project Beta review prep."},
            {"chunk_id": "chunk-2", "chunk_summary": "Project Beta bug fixes."},
        ],
    }

    result = services.generate_groq_response(
        "Give me a detailed summary of everything we know about Project Beta.",
        ["Chunk Summary: Project Beta review prep.", "Chunk Summary: Project Beta bug fixes."],
        retrieval_trace=trace,
    )

    assert result["answer_payload"]["mode"] == "long"
    assert result["answer_payload"]["reason_code"] == "explicit_long"
    assert len(result["answer_payload"]["bullets"]) == 2


def test_select_answer_mode_uses_complexity_as_tie_breaker():
    mode, reason_code = services._select_answer_mode(
        "Project Phoenix",
        retrieval_trace={
            "query_type": "general_search",
            "result_count": 4,
            "max_hop_count": 2,
        },
    )

    assert mode == "long"
    assert reason_code == "evidence_complexity"


def test_select_answer_mode_treats_whom_lookup_as_direct_lookup():
    mode, reason_code = services._select_answer_mode(
        "Whom does George Brown now report to?",
        retrieval_trace={
            "query_type": "person_lookup",
            "result_count": 5,
            "max_hop_count": 2,
        },
    )

    assert mode == "short"
    assert reason_code == "direct_lookup"


def test_generate_groq_response_uses_boring_fallback_on_invalid_json(monkeypatch):
    class FakeChain:
        def __or__(self, _other):
            return self

        def invoke(self, _payload):
            return "not-json"

    monkeypatch.setattr(services, "CHAT_PROMPT", FakeChain())
    monkeypatch.setattr(services, "_create_groq_client", lambda **_kwargs: object())
    monkeypatch.setattr(services, "StrOutputParser", lambda: object())

    trace = {
        "query_type": "general_search",
        "result_count": 3,
        "max_hop_count": 2,
        "evidence": [
            {"fact_id": "fact-1"},
            {"chunk_id": "chunk-1"},
        ],
    }

    result = services.generate_groq_response(
        "Project Phoenix",
        ["Chunk Summary: Project Phoenix work."],
        retrieval_trace=trace,
    )

    assert result["answer_payload"]["schema_version"] == 1
    assert result["answer_payload"]["mode"] == "short"
    assert result["answer_payload"]["reason_code"] == "fallback_invalid_json"
    assert result["answer_payload"]["summary"]
    assert result["answer_payload"]["bullets"] == []
    assert result["answer_payload"]["evidence_refs"] == ["fact:fact-1", "chunk:chunk-1"]
