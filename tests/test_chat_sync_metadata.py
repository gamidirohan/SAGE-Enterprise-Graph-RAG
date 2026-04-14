from app import backend, services


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

    def close(self):
        self.closed = True


class _Session:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append({"query": query, "params": params})
        return type("_Result", (), {"data": lambda self: []})()


def test_store_chat_message_in_neo4j_forwards_full_metadata(monkeypatch):
    captured = {}
    saia_calls = {}

    def fake_store(payload):
        captured.update(payload)
        return True

    monkeypatch.setattr(backend, "store_in_neo4j", fake_store)
    monkeypatch.setattr(backend.saia, "process_chat_message", lambda **kwargs: saia_calls.update(kwargs) or {"status": "completed"})

    message = backend.ChatMessageSyncItem(
        id="m1",
        senderId="u1",
        receiverId="u2",
        content="hello there",
        timestamp="2026-04-01T10:00:00Z",
        source="chat_message",
        conversationId="direct:u1:u2",
        conversationType="direct",
        attachment={
            "id": "file-1",
            "name": "notes.pdf",
            "type": "application/pdf",
            "url": "/uploads/notes.pdf",
        },
        trace={"query_type": "general_search"},
    )

    assert backend.store_chat_message_in_neo4j(message) is True
    assert captured["doc_id"] == "chat-msg-m1"
    assert captured["sender"] == "u1"
    assert captured["receivers"] == ["u2"]
    assert captured["subject"] == "Chat message m1"
    assert captured["content"] == "hello there"
    assert captured["timestamp"] == "2026-04-01T10:00:00Z"
    assert captured["source"] == "chat_message"
    assert captured["conversation_type"] == "direct"
    assert captured["conversation_id"] == "direct:u1:u2"
    assert captured["attachment_name"] == "notes.pdf"
    assert captured["attachment_type"] == "application/pdf"
    assert captured["attachment_url"] == "/uploads/notes.pdf"
    assert captured["origin_message_id"] == "m1"
    assert captured["linked_message_id"] is None
    assert captured["graph_sync_status"] == backend.chat_store.GRAPH_SYNC_READY
    assert captured["trace_json"] == '{"query_type": "general_search"}'
    assert saia_calls["message_id"] == "m1"
    assert saia_calls["conversation_id"] == "direct:u1:u2"
    assert saia_calls["receiver_ids"] == ["u2"]


def test_store_chat_message_in_neo4j_skips_sage_messages(monkeypatch):
    captured = {"store_called": False, "saia_called": False}

    monkeypatch.setattr(backend, "store_in_neo4j", lambda _payload: captured.update({"store_called": True}) or True)
    monkeypatch.setattr(backend.saia, "process_chat_message", lambda **_kwargs: captured.update({"saia_called": True}) or {"status": "completed"})

    message = backend.ChatMessageSyncItem(
        id="sage-m1",
        senderId="1",
        receiverId="sage",
        content="What did I promise to send?",
        timestamp="2026-04-01T10:00:00Z",
        source="chat_message",
        conversationId="sage:1",
        conversationType="sage",
    )

    assert backend.store_chat_message_in_neo4j(message) is False
    assert captured["store_called"] is False
    assert captured["saia_called"] is False


def test_store_in_neo4j_persists_timestamp_and_source(monkeypatch):
    session = _Session()
    driver = _Driver(session)

    monkeypatch.setattr(services.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(services.utils, "open_neo4j_session", lambda _driver, _database: _Ctx(session))
    monkeypatch.setattr(services.utils, "generate_embedding", lambda _text: [0.1, 0.2])
    monkeypatch.setattr(services.utils, "chunk_document", lambda _text, **_kwargs: [])
    monkeypatch.setattr(services.utils, "GROQ_API_KEY", None)

    stored = services.store_in_neo4j(
        {
            "doc_id": "chat-msg-m1",
            "sender": "u1",
            "receivers": ["u2"],
            "subject": "Chat message m1",
            "content": "hello there",
            "timestamp": "2026-04-01T10:00:00Z",
            "source": "chat_message",
            "conversation_type": "direct",
            "conversation_id": "direct:u1:u2",
            "attachment_name": "notes.pdf",
            "attachment_type": "application/pdf",
            "attachment_url": "/uploads/notes.pdf",
            "origin_message_id": "m1",
            "linked_message_id": None,
            "trace_json": '{"query_type":"general_search"}',
            "graph_sync_status": "ready",
        }
    )

    assert stored is True
    assert driver.closed is True
    assert session.calls

    document_write = next(call["params"] for call in session.calls if "MERGE (d:Document {doc_id: $doc_id})" in call["query"])
    assert document_write["doc_id"] == "chat-msg-m1"
    assert document_write["sender"] == "u1"
    assert document_write["subject"] == "Chat message m1"
    assert document_write["content"] == "hello there"
    assert document_write["timestamp"] == "2026-04-01T10:00:00Z"
    assert document_write["source"] == "chat_message"
    assert document_write["conversation_type"] == "direct"
    assert document_write["conversation_id"] == "direct:u1:u2"
    assert document_write["attachment_name"] == "notes.pdf"
    assert document_write["attachment_type"] == "application/pdf"
    assert document_write["attachment_url"] == "/uploads/notes.pdf"
    assert document_write["origin_message_id"] == "m1"
    assert document_write["linked_message_id"] is None
    assert document_write["trace_json"] == '{"query_type":"general_search"}'
    assert document_write["graph_sync_status"] == "ready"
    assert any("HAS_EVIDENCE_DOCUMENT" in call["query"] for call in session.calls)
