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
        lambda _q, _d: {"answer": "final answer", "thinking": ["step"]},
    )

    result = services.generate_streamlit_response("q", ["ctx"])

    assert result == "final answer"


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
