import asyncio
import io

from fastapi import BackgroundTasks, HTTPException
import pytest
from starlette.datastructures import UploadFile

import backend


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


def test_get_neo4j_session_uses_database(monkeypatch):
    driver = _Driver()
    monkeypatch.setattr(backend, "NEO4J_DATABASE", "neo4j_db")
    backend.get_neo4j_session(driver)
    assert driver.kwargs == {"database": "neo4j_db"}


def test_get_neo4j_session_without_database(monkeypatch):
    driver = _Driver()
    monkeypatch.setattr(backend, "NEO4J_DATABASE", None)
    backend.get_neo4j_session(driver)
    assert driver.kwargs == {}


def test_generate_doc_id_is_deterministic():
    a = backend.generate_doc_id("sample")
    b = backend.generate_doc_id("sample")
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
    monkeypatch.setattr(backend, "get_neo4j_driver", lambda: driver)
    monkeypatch.setattr(backend, "get_embedding_model", lambda: _Model())

    result = backend.query_graph("what is this?")
    assert len(result) == 1
    assert "important chunk" in result[0]
    assert driver.closed is True


def test_query_graph_failure_returns_fallback(monkeypatch):
    session = _Session(should_fail=True)
    driver = _Driver(session=session)
    monkeypatch.setattr(backend, "get_neo4j_driver", lambda: driver)
    monkeypatch.setattr(backend, "get_embedding_model", lambda: _Model())

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

    class FakePrompt:
        @staticmethod
        def from_template(_template):
            return FakeChain()

    monkeypatch.setattr(backend, "ChatPromptTemplate", FakePrompt)
    monkeypatch.setattr(backend, "ChatGroq", lambda **_kwargs: object())
    monkeypatch.setattr(backend, "StrOutputParser", lambda: object())

    result = backend.generate_groq_response("q", ["Chunk Summary: ctx, Document: d"])
    assert result["answer"] == "Final answer"
    assert result["thinking"] == ["step 1"]


def test_chat_endpoint_uses_query_and_generator(monkeypatch):
    monkeypatch.setattr(backend, "query_graph", lambda _m: ["ctx"])
    monkeypatch.setattr(backend, "generate_groq_response", lambda _q, _d: {"answer": "ok", "thinking": []})

    request = backend.ChatRequest(message="hi", history=[])
    result = asyncio.run(backend.chat_endpoint(request))
    assert result["answer"] == "ok"


def test_health_check_endpoint():
    result = asyncio.run(backend.health_check())
    assert result == {"status": "ok"}


def test_process_document_unsupported_extension_raises_http_exception():
    file = UploadFile(filename="bad.docx", file=io.BytesIO(b"hello"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(backend.process_document(BackgroundTasks(), file))
    assert exc.value.status_code == 500
