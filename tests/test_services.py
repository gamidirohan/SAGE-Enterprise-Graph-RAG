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
    assert "relationships(path)" not in services.GRAPH_VECTOR_QUERY
    assert "relationships(path)" not in services.PERSON_GRAPH_VECTOR_QUERY


def test_classify_query_marks_multi_part_lookup_as_compound():
    assert services._classify_query("What's the new project? When's the orientation? Who all know it so far?") == "compound_lookup"


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
    assert result["trace"]["result_count"] == 2
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
    assert result["trace"]["evidence"][0]["fact_id"] == "fact-commitment"
    assert result["trace"]["evidence"][0]["rank_score"] > result["trace"]["evidence"][1]["rank_score"]


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
    assert result["answer_payload"]["mode"] == "short"
    assert result["answer_payload"]["reason_code"] == "direct_lookup"
    assert result["answer_payload"]["evidence_refs"] == ["fact:fact-1", "chunk:chunk-1"]
    assert result["answer_payload"]["bullets"] == ["Recipient: u2", "Time: 2026-04-03 01:30 AM IST"]
    assert captured["user_context"].endswith("Query classification: task_commitment_lookup.")
    assert captured["retrieval_guidance"].startswith("This is a task or commitment lookup.")
    assert captured["answer_mode"] == "short"
    assert captured["context"].split("\n\n")[0].startswith("Canonical facts")


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
