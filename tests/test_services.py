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
