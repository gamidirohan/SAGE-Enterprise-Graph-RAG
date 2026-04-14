from app import chat_store


def test_sync_message_to_graph_triggers_saia_after_store(monkeypatch):
    captured_store = {}
    captured_saia = {}

    monkeypatch.setattr(chat_store, "_conversation_member_ids", lambda _session, _conversation_id: ["u1", "u2"])
    monkeypatch.setattr(chat_store.services, "store_in_neo4j", lambda payload: captured_store.update(payload) or True)
    monkeypatch.setattr(chat_store.saia, "process_chat_message", lambda **kwargs: captured_saia.update(kwargs) or {"status": "completed"})

    result = chat_store._sync_message_to_graph(
        session=object(),
        conversation={"id": "direct:u1:u2", "type": "direct", "group_id": None},
        message={
            "senderId": "u1",
            "receiverId": "u2",
            "content": "I'll send you the report tomorrow",
            "sentAt": "2026-04-01T10:00:00Z",
            "source": "chat_message",
            "trace": {"query_type": "general_search"},
            "isAiResponse": False,
            "syncToGraph": True,
        },
        message_id="m1",
    )

    assert result == chat_store.GRAPH_SYNC_READY
    assert captured_store["doc_id"] == "chat-msg-m1"
    assert captured_store["origin_message_id"] == "m1"
    assert captured_saia["message_id"] == "m1"
    assert captured_saia["conversation_id"] == "direct:u1:u2"
    assert captured_saia["receiver_ids"] == ["u2"]


def test_sync_message_to_graph_keeps_message_ready_when_saia_fails(monkeypatch):
    monkeypatch.setattr(chat_store, "_conversation_member_ids", lambda _session, _conversation_id: ["u1", "u2"])
    monkeypatch.setattr(chat_store.services, "store_in_neo4j", lambda _payload: True)

    def fail_saia(**_kwargs):
        raise RuntimeError("saia blew up")

    monkeypatch.setattr(chat_store.saia, "process_chat_message", fail_saia)

    result = chat_store._sync_message_to_graph(
        session=object(),
        conversation={"id": "direct:u1:u2", "type": "direct", "group_id": None},
        message={
            "senderId": "u1",
            "receiverId": "u2",
            "content": "hello there",
            "sentAt": "2026-04-01T10:00:00Z",
            "source": "chat_message",
            "trace": None,
            "isAiResponse": False,
            "syncToGraph": True,
        },
        message_id="m1",
    )

    assert result == chat_store.GRAPH_SYNC_READY


def test_sync_message_to_graph_skips_sage_conversation(monkeypatch):
    captured = {"store_called": False, "saia_called": False}

    monkeypatch.setattr(chat_store, "_conversation_member_ids", lambda _session, _conversation_id: ["u1", "sage"])
    monkeypatch.setattr(chat_store.services, "store_in_neo4j", lambda _payload: captured.update({"store_called": True}) or True)
    monkeypatch.setattr(chat_store.saia, "process_chat_message", lambda **_kwargs: captured.update({"saia_called": True}) or {"status": "completed"})

    result = chat_store._sync_message_to_graph(
        session=object(),
        conversation={"id": "sage:u1", "type": "sage", "group_id": None},
        message={
            "senderId": "u1",
            "receiverId": "sage",
            "content": "What did I promise to send?",
            "sentAt": "2026-04-01T10:00:00Z",
            "source": "chat_message",
            "trace": None,
            "isAiResponse": False,
            "syncToGraph": True,
        },
        message_id="sage-m1",
    )

    assert result == chat_store.GRAPH_SYNC_SKIPPED
    assert captured["store_called"] is False
    assert captured["saia_called"] is False


def test_create_message_persists_and_returns_answer_payload(monkeypatch):
    class _Result:
        def data(self):
            return []

    class _Session:
        def __init__(self):
            self.calls = []

        def run(self, query, **params):
            self.calls.append({"query": query, "params": params})
            return _Result()

    session = _Session()

    monkeypatch.setattr(
        chat_store,
        "_conversation_record",
        lambda _session, _conversation_id, _user_id: {
            "id": "sage:u1",
            "type": "sage",
            "group_id": None,
        },
    )
    monkeypatch.setattr(chat_store, "_conversation_member_ids", lambda _session, _conversation_id: ["u1", "sage"])

    result = chat_store.create_message(
        session,
        authenticated_user_id="u1",
        conversation_id="sage:u1",
        payload={
            "id": "m-answer",
            "senderId": "sage",
            "receiverId": "u1",
            "content": "Project Beta has multiple related updates.",
            "sentAt": "2026-04-01T10:00:00Z",
            "source": "sage_response",
            "thinking": ["legacy internal note"],
            "answerPayload": {
                "schema_version": 1,
                "mode": "long",
                "reason_code": "explicit_long",
                "summary": "Project Beta has multiple related updates.",
                "bullets": ["A client review is scheduled for tomorrow at 3pm."],
                "explanation": "SAGE expanded this answer because your question explicitly asked for more detail.",
                "evidence_refs": ["chunk:chunk-1"],
            },
            "role": "assistant",
            "isAiResponse": True,
            "syncToGraph": False,
        },
    )

    write_params = next(call["params"] for call in session.calls if "MERGE (m:Message {id: $message_id})" in call["query"])

    assert '"mode": "long"' in write_params["answer_payload_json"]
    assert result["message"]["answerPayload"]["schema_version"] == 1
    assert result["message"]["answerPayload"]["summary"] == "Project Beta has multiple related updates."
    assert result["message"]["content"] == "Project Beta has multiple related updates."
