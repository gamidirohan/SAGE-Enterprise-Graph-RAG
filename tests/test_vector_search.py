import app.vector_search as vector_search


class _FakeResult:
    def data(self):
        return []


class _FakeSession:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def run(self, cypher, parameters=None, **kwargs):
        if "query" in kwargs:
            raise AssertionError("Cypher parameter named query must be passed via parameters dict")
        self.calls.append((cypher, parameters, kwargs))
        return _FakeResult()


class _FakeDriver:
    def close(self):
        pass


def test_fulltext_retrieve_passes_query_parameter_without_driver_keyword_collision(monkeypatch):
    session = _FakeSession()

    monkeypatch.setattr(vector_search.utils, "create_neo4j_driver", lambda: _FakeDriver())
    monkeypatch.setattr(vector_search.utils, "open_neo4j_session", lambda *_args: session)

    result = vector_search.fulltext_retrieve("What did I promise to send?", user_id="currentUser")

    assert result["trace"]["result_count"] == 0
    assert "error" not in result["trace"]
    fulltext_calls = [call for call in session.calls if "db.index.fulltext.queryNodes" in call[0]]
    assert len(fulltext_calls) == 2
    assert all(call[1]["query"] == "What did I promise to send?" for call in fulltext_calls)
