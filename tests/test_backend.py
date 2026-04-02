import asyncio
import io
import zipfile

from fastapi import BackgroundTasks, HTTPException
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
    def __init__(self, rows=None, should_fail=False):
        self.rows = rows or []
        self.should_fail = should_fail

    def run(self, *_args, **_kwargs):
        if self.should_fail:
            raise RuntimeError("db error")
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
    assert result["thinking"] == []


def test_generate_groq_response_parses_thinking(monkeypatch):
    class FakeChain:
        def __or__(self, _other):
            return self

        def invoke(self, _payload):
            return "<think>step 1</think>Final answer"

    monkeypatch.setattr(backend.services, "CHAT_PROMPT", FakeChain())
    monkeypatch.setattr(backend.services, "_create_groq_client", lambda **_kwargs: object())
    monkeypatch.setattr(backend.services, "StrOutputParser", lambda: object())

    result = backend.generate_groq_response("q", ["Chunk Summary: ctx, Document: d"])
    assert result["answer"] == "Final answer"
    assert result["thinking"] == ["step 1"]


def test_chat_endpoint_uses_query_and_generator(monkeypatch):
    monkeypatch.setattr(
        backend,
        "query_graph_with_trace",
        lambda _m, user_id=None: {
            "documents": ["ctx"],
            "trace": {"query_type": "general_search", "user_scoped": bool(user_id), "evidence": []},
        },
    )
    monkeypatch.setattr(
        backend,
        "generate_groq_response",
        lambda _q, _d, user_id=None: {"answer": "ok", "thinking": [], "trace": {"user_scoped": bool(user_id)}},
    )

    request = backend.ChatRequest(message="hi", history=[])
    result = asyncio.run(backend.chat_endpoint(request))
    assert result["answer"] == "ok"
    assert result["trace"]["query_type"] == "general_search"


def test_health_check_endpoint():
    result = asyncio.run(backend.health_check())
    assert result == {"status": "ok"}


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
