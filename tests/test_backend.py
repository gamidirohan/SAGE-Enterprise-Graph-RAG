import asyncio
import io
import zipfile

from fastapi import BackgroundTasks, HTTPException
from neo4j.exceptions import ServiceUnavailable
import pytest
from starlette.datastructures import UploadFile

from app import backend


class _Ctx:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class _Driver:
    def __init__(self, session=None):
        self._session = session
        self.closed = False
        self.kwargs = None

    def session(self, **kwargs):
        self.kwargs = kwargs
        return _Ctx(self._session)

    def close(self):
        self.closed = True


class _Result:
    def __init__(self, rows=None):
        self._rows = rows or []

    def data(self):
        return self._rows


class _Session:
    def __init__(self, rows=None, should_fail=False, handler=None):
        self.rows = rows or []
        self.should_fail = should_fail
        self.handler = handler

    def run(self, *args, **kwargs):
        if self.should_fail:
            raise RuntimeError("db error")
        if self.handler:
            return _Result(self.handler(*args, **kwargs))
        return _Result(self.rows)


class _Model:
    def encode(self, _text):
        return [0.1, 0.2]


def test_open_neo4j_session_uses_database(monkeypatch):
    driver = _Driver()
    backend.utils.open_neo4j_session(driver, "neo4j_db")
    assert driver.kwargs == {"database": "neo4j_db"}


def test_open_neo4j_session_without_database(monkeypatch):
    driver = _Driver()
    backend.utils.open_neo4j_session(driver, None)
    assert driver.kwargs == {}


def test_generate_doc_id_is_deterministic():
    a = backend.utils.generate_doc_id("sample")
    b = backend.utils.generate_doc_id("sample")
    assert a == b
    assert len(a) == 64


def test_query_graph_success(monkeypatch):
    rows = [
        {
            "chunk_summary": "important chunk",
            "d": {"doc_id": "d1"},
            "similarity": 0.99,
            "relationship": "PART_OF",
            "n": {"id": "n1"},
        }
    ]
    session = _Session(rows=rows)
    driver = _Driver(session=session)
    monkeypatch.setattr(backend.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(backend.utils, "get_cached_embedding_model", lambda: _Model())

    result = backend.query_graph("what is this?")
    assert len(result) == 1
    assert "important chunk" in result[0]
    assert driver.closed is True


def test_query_graph_failure_returns_fallback(monkeypatch):
    session = _Session(should_fail=True)
    driver = _Driver(session=session)
    monkeypatch.setattr(backend.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(backend.utils, "get_cached_embedding_model", lambda: _Model())

    result = backend.query_graph("what is this?")
    assert len(result) == 1
    assert "technical issue" in result[0]
    assert driver.closed is True


def test_generate_groq_response_with_no_documents():
    result = backend.generate_groq_response("q", [])
    assert "knowledge base" in result["answer"]
    assert result["answer_payload"]["schema_version"] == 1
    assert result["answer_payload"]["summary"] == result["answer"]
    assert result["thinking"] == []


def test_generate_groq_response_returns_structured_answer_payload(monkeypatch):
    class FakeChain:
        def __or__(self, _other):
            return self

        def invoke(self, _payload):
            return '{"summary":"Final answer","bullets":["Detail 1","Detail 2"]}'

    monkeypatch.setattr(backend.services, "CHAT_PROMPT", FakeChain())
    monkeypatch.setattr(backend.services, "_create_groq_client", lambda **_kwargs: object())
    monkeypatch.setattr(backend.services, "StrOutputParser", lambda: object())

    result = backend.generate_groq_response("q", ["Chunk Summary: ctx, Document: d"])
    assert result["answer"] == "Final answer"
    assert result["answer_payload"]["schema_version"] == 1
    assert result["answer_payload"]["summary"] == "Final answer"
    assert result["answer_payload"]["bullets"] == ["Detail 1", "Detail 2"]
    assert result["thinking"] == []


def test_chat_endpoint_uses_query_and_generator(monkeypatch):
    monkeypatch.setattr(
        backend,
        "query_graph_with_trace",
        lambda _m, user_id=None: {
            "documents": ["ctx"],
            "trace": {"query_type": "general_search", "user_scoped": bool(user_id), "evidence": []},
        },
    )
    captured = {}
    monkeypatch.setattr(
        backend,
        "generate_groq_response",
        lambda _q, _d, user_id=None, retrieval_trace=None: captured.update({"retrieval_trace": retrieval_trace}) or {
            "answer": "ok",
            "answer_payload": {
                "schema_version": 1,
                "mode": "short",
                "reason_code": "direct_lookup",
                "summary": "ok",
                "bullets": [],
                "explanation": "short lookup",
                "evidence_refs": [],
            },
            "thinking": [],
            "trace": {"user_scoped": bool(user_id)},
        },
    )

    request = backend.ChatRequest(message="hi", history=[])
    result = asyncio.run(backend.chat_endpoint(request))
    assert result["answer"] == "ok"
    assert result["answer_payload"]["summary"] == "ok"
    assert result["trace"]["query_type"] == "general_search"
    assert captured["retrieval_trace"]["query_type"] == "general_search"


def test_chat_endpoint_builds_fallback_answer_payload_with_evidence_refs(monkeypatch):
    monkeypatch.setattr(
        backend,
        "query_graph_with_trace",
        lambda _m, user_id=None: {
            "documents": ["ctx"],
            "trace": {
                "query_type": "general_search",
                "user_scoped": bool(user_id),
                "evidence": [{"fact_id": "fact-1"}, {"chunk_id": "chunk-1"}],
            },
        },
    )
    monkeypatch.setattr(
        backend,
        "generate_groq_response",
        lambda _q, _d, user_id=None, retrieval_trace=None: {
            "answer": "Fallback answer",
            "thinking": [],
        },
    )

    request = backend.ChatRequest(message="hi", history=[])
    result = asyncio.run(backend.chat_endpoint(request))

    assert result["answer"] == "Fallback answer"
    assert result["answer_payload"]["schema_version"] == 1
    assert result["answer_payload"]["reason_code"] == "fallback_invalid_json"
    assert result["answer_payload"]["evidence_refs"] == ["fact:fact-1", "chunk:chunk-1"]


def test_health_check_endpoint():
    result = asyncio.run(backend.health_check())
    assert result == {"status": "ok"}


def test_bootstrap_endpoint_returns_503_when_graph_is_unavailable(monkeypatch):
    def fail_bootstrap(*_args, **_kwargs):
        raise ServiceUnavailable("Unable to retrieve routing information")

    monkeypatch.setattr(backend.chat_store, "bootstrap_seed_data", fail_bootstrap)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(backend.bootstrap_endpoint(backend.BootstrapRequest(users=[], groups=[], messages=[])))

    assert exc.value.status_code == 503
    assert exc.value.detail == "Graph database is unavailable while bootstrapping seed data. Check Neo4j connectivity and try again."


def test_sync_user_endpoint(monkeypatch):
    monkeypatch.setattr(backend, "upsert_user_in_neo4j", lambda _req: True)
    request = backend.UserSyncRequest(id="u1", name="Test User", email="u@example.com", team=[])
    result = asyncio.run(backend.sync_user_endpoint(request))
    assert result["success"] is True
    assert result["user_id"] == "u1"


def test_sync_messages_endpoint(monkeypatch):
    monkeypatch.setattr(backend, "store_chat_message_in_neo4j", lambda _item: True)
    request = backend.ChatMessageSyncRequest(
        messages=[
            backend.ChatMessageSyncItem(
                id="m1",
                senderId="u1",
                receiverId="u2",
                content="hello",
                timestamp="2026-03-29T00:00:00Z",
            )
        ]
    )
    result = asyncio.run(backend.sync_messages_endpoint(request))
    assert result["success"] is True
    assert result["ingested"] == 1
    assert result["failed"] == 0


def test_process_document_unsupported_extension_raises_http_exception():
    file = UploadFile(filename="bad.csv", file=io.BytesIO(b"hello"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(backend.process_document(BackgroundTasks(), file))
    assert exc.value.status_code == 400


def test_process_document_supports_docx(monkeypatch):
    document_bytes = io.BytesIO()
    with zipfile.ZipFile(document_bytes, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>meeting notes</w:t></w:r></w:p></w:body></w:document>"
            ),
        )
    document_bytes.seek(0)

    file = UploadFile(filename="notes.docx", file=document_bytes)
    monkeypatch.setattr(backend.utils, "generate_doc_id", lambda _text: "doc-123")
    monkeypatch.setattr(
        backend.services,
        "extract_structured_data",
        lambda _text, doc_id: {
            "doc_id": doc_id,
            "sender": "u1",
            "receivers": ["u2"],
            "subject": "Meeting Notes",
            "content": "meeting notes",
        },
    )
    monkeypatch.setattr(backend, "store_in_neo4j", lambda _data: True)

    result = asyncio.run(backend.process_document(BackgroundTasks(), file))

    assert result["success"] is True
    assert result["doc_id"] == "doc-123"


def test_process_document_returns_success_after_storage(monkeypatch):
    file = UploadFile(filename="notes.txt", file=io.BytesIO(b"meeting notes"))
    monkeypatch.setattr(backend.utils, "generate_doc_id", lambda _text: "doc-123")
    monkeypatch.setattr(
        backend.services,
        "extract_structured_data",
        lambda _text, doc_id: {
            "doc_id": doc_id,
            "sender": "u1",
            "receivers": ["u2"],
            "subject": "Meeting Notes",
            "content": "meeting notes",
        },
    )
    monkeypatch.setattr(backend, "store_in_neo4j", lambda _data: True)

    result = asyncio.run(backend.process_document(BackgroundTasks(), file))

    assert result["success"] is True
    assert result["doc_id"] == "doc-123"
    assert result["message"] == "Document processed and stored in the graph database."


def test_process_document_triggers_saia_for_message_attachments(monkeypatch):
    file = UploadFile(filename="notes.txt", file=io.BytesIO(b"meeting notes"))
    captured = {}

    monkeypatch.setattr(backend.utils, "generate_doc_id", lambda _text: "doc-123")
    monkeypatch.setattr(
        backend.services,
        "extract_structured_data",
        lambda _text, doc_id: {
            "doc_id": doc_id,
            "sender": "u1",
            "receivers": ["u2"],
            "subject": "Meeting Notes",
            "content": "meeting notes",
        },
    )
    monkeypatch.setattr(backend, "store_in_neo4j", lambda _data: True)
    monkeypatch.setattr(backend.saia, "process_message_attachment", lambda **kwargs: captured.update(kwargs) or {"status": "completed"})

    result = asyncio.run(
        backend.process_document(
            BackgroundTasks(),
            file,
            sender_id="u1",
            receiver_id="u2",
            conversation_id="direct:u1:u2",
            conversation_type="direct",
            sent_at="2026-04-01T10:00:00Z",
            linked_message_id="m1",
            attachment_name="notes.txt",
        )
    )

    assert result["success"] is True
    assert result["doc_id"] == "message-attachment-m1"
    assert captured["doc_id"] == "message-attachment-m1"
    assert captured["linked_message_id"] == "m1"
    assert captured["conversation_id"] == "direct:u1:u2"
    assert captured["receiver_ids"] == ["u2"]


def test_process_document_raises_if_storage_fails(monkeypatch):
    file = UploadFile(filename="notes.txt", file=io.BytesIO(b"meeting notes"))
    monkeypatch.setattr(backend.utils, "generate_doc_id", lambda _text: "doc-123")
    monkeypatch.setattr(
        backend.services,
        "extract_structured_data",
        lambda _text, doc_id: {
            "doc_id": doc_id,
            "sender": "u1",
            "receivers": ["u2"],
            "subject": "Meeting Notes",
            "content": "meeting notes",
        },
    )
    monkeypatch.setattr(backend, "store_in_neo4j", lambda _data: False)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(backend.process_document(BackgroundTasks(), file))

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to store document in the graph database."


def test_get_message_saia_insight_endpoint_returns_inspection_payload(monkeypatch):
    def handler(query, **params):
        if "PARTICIPATES_IN" in query and "IN_CONVERSATION" in query:
            return [{"id": "m1"}]
        return []

    session = _Session(handler=handler)
    driver = _Driver(session=session)
    monkeypatch.setattr(backend.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(backend.utils, "open_neo4j_session", lambda _driver, _database: _Ctx(session))
    monkeypatch.setattr(
        backend.saia,
        "collect_message_insight",
        lambda _session, message_id: {"message_id": message_id, "summary": {"claim_count": 1}},
    )

    result = asyncio.run(backend.get_message_saia_insight_endpoint("m1", x_user_id="u1"))

    assert result["message_id"] == "m1"
    assert result["summary"]["claim_count"] == 1
    assert driver.closed is True


def test_get_message_saia_insight_endpoint_denies_non_participant(monkeypatch):
    session = _Session(rows=[])
    driver = _Driver(session=session)
    monkeypatch.setattr(backend.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(backend.utils, "open_neo4j_session", lambda _driver, _database: _Ctx(session))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(backend.get_message_saia_insight_endpoint("m1", x_user_id="u9"))

    assert exc.value.status_code == 403
    assert exc.value.detail == "Message not found or access denied"
    assert driver.closed is True
