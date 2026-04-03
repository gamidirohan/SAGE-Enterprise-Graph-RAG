from pathlib import Path

from app import document_ingestion


def test_list_document_files_excludes_mapping_files(tmp_path):
    (tmp_path / "Document 1.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "Document 2.pdf").write_text("fake-pdf", encoding="utf-8")
    (tmp_path / "ID Mappings.txt").write_text("IDs:\nEMP001: Alice (PM)\n", encoding="utf-8")
    (tmp_path / "diagram.png").write_text("ignore", encoding="utf-8")

    result = document_ingestion.list_document_files(tmp_path)

    assert [path.name for path in result] == ["Document 1.txt", "Document 2.pdf"]


def test_ingest_document_file_skips_existing_duplicate(monkeypatch, tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("meeting notes", encoding="utf-8")

    stored_payloads = []
    extraction_calls = []

    monkeypatch.setattr(
        document_ingestion.services,
        "extract_structured_data",
        lambda content, doc_id: extraction_calls.append((content, doc_id)) or {
            "doc_id": doc_id,
            "sender": "u1",
            "receivers": ["u2"],
            "subject": "Meeting Notes",
            "content": content,
        },
    )
    monkeypatch.setattr(document_ingestion, "document_exists", lambda _doc_id: True)
    monkeypatch.setattr(
        document_ingestion.services,
        "store_in_neo4j",
        lambda payload: stored_payloads.append(payload) or True,
    )

    result = document_ingestion.ingest_document_file(file_path, skip_existing=True)

    assert result["status"] == "skipped_duplicate"
    assert result["stored"] is False
    assert extraction_calls == []
    assert stored_payloads == []


def test_ingest_document_file_reprocesses_when_requested(monkeypatch, tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("meeting notes", encoding="utf-8")

    stored_payloads = []

    monkeypatch.setattr(
        document_ingestion.services,
        "extract_structured_data",
        lambda content, doc_id: {
            "doc_id": doc_id,
            "sender": "u1",
            "receivers": ["u2"],
            "subject": "Meeting Notes",
            "content": content,
        },
    )
    monkeypatch.setattr(document_ingestion, "document_exists", lambda _doc_id: True)
    monkeypatch.setattr(
        document_ingestion.services,
        "store_in_neo4j",
        lambda payload: stored_payloads.append(payload) or True,
    )

    result = document_ingestion.ingest_document_file(file_path, skip_existing=False)

    assert result["status"] == "stored"
    assert result["stored"] is True
    assert len(stored_payloads) == 1
    assert stored_payloads[0]["subject"] == "Meeting Notes"


def test_ingest_document_directory_counts_results_and_mappings(monkeypatch, tmp_path):
    (tmp_path / "ID Mappings.txt").write_text(
        "IDs:\nEMP001: Alice Johnson (Project Manager)\nEMP002: Bob Smith (Engineer)\n",
        encoding="utf-8",
    )
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    captured_mappings = []

    monkeypatch.setattr(
        document_ingestion,
        "upsert_id_mappings_to_neo4j",
        lambda mappings: captured_mappings.extend(mappings) or True,
    )
    monkeypatch.setattr(
        document_ingestion,
        "ingest_document_file",
        lambda path, **_kwargs: {
            "file": str(path),
            "doc_id": Path(path).stem,
            "status": "stored" if Path(path).name == "a.txt" else "skipped_duplicate",
            "stored": Path(path).name == "a.txt",
            "subject": Path(path).name,
        },
    )

    summary = document_ingestion.ingest_document_directory(tmp_path)

    assert summary["exists"] is True
    assert summary["mapping_files_processed"] == 1
    assert summary["mapping_entries_upserted"] == 2
    assert summary["document_files_seen"] == 2
    assert summary["stored"] == 1
    assert summary["skipped_duplicates"] == 1
    assert summary["failed"] == 0
    assert [item["id"] for item in captured_mappings] == ["EMP001", "EMP002"]
