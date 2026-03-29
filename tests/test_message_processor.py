from pathlib import Path

from app import message_processor


def test_extract_message_data_parses_multiline_message(tmp_path):
    message_file = tmp_path / "msg.txt"
    message_file.write_text(
        "Sender ID: EMP001\n"
        "Receiver ID: EMP002\n"
        "Message: First line\n"
        "Second line\n"
        "Sent Time: 2026-03-15 09:00\n",
        encoding="utf-8",
    )

    data = message_processor.extract_message_data(str(message_file))
    assert data["sender"] == "EMP001"
    assert data["receivers"] == ["EMP002"]
    assert "First line\nSecond line" == data["content"]
    assert data["sent_time"] == "2026-03-15 09:00"


def test_extract_message_data_returns_none_when_required_fields_missing(tmp_path):
    message_file = tmp_path / "bad.txt"
    message_file.write_text("Sender ID: EMP001\nMessage: only sender\n", encoding="utf-8")
    assert message_processor.extract_message_data(str(message_file)) is None


def test_process_message_files_returns_empty_for_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing"
    result = message_processor.process_message_files(str(missing_dir))
    assert result == []


def test_process_message_files_processes_txt_files_only(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("z", encoding="utf-8")

    monkeypatch.setattr(
        message_processor,
        "extract_message_data",
        lambda p: {"doc_id": Path(p).name, "sender": "EMP001", "receivers": ["EMP002"], "subject": "s", "content": "c"},
    )
    monkeypatch.setattr(message_processor, "store_in_neo4j", lambda _d: True)

    result = message_processor.process_message_files(str(tmp_path))
    assert len(result) == 2


def test_get_session_uses_database(monkeypatch):
    class Driver:
        def __init__(self):
            self.kwargs = None

        def session(self, **kwargs):
            self.kwargs = kwargs
            return object()

    driver = Driver()
    monkeypatch.setattr(message_processor, "NEO4J_DATABASE", "neo_db")
    message_processor.get_session(driver)
    assert driver.kwargs == {"database": "neo_db"}


def test_store_in_neo4j_returns_false_on_session_error(monkeypatch):
    class Driver:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FailingCtx:
        def __enter__(self):
            raise RuntimeError("session failed")

        def __exit__(self, exc_type, exc, tb):
            return False

    driver = Driver()
    monkeypatch.setattr(message_processor, "get_neo4j_driver", lambda: driver)
    monkeypatch.setattr(message_processor, "get_session", lambda _d: FailingCtx())
    monkeypatch.setattr(message_processor.utils, "GROQ_API_KEY", None)

    ok = message_processor.store_in_neo4j(
        {"doc_id": "d1", "sender": "EMP001", "receivers": ["EMP002"], "subject": "s", "content": "c"}
    )
    assert ok is False
    assert driver.closed is True
