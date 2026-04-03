"""Helpers for single-file and batch document ingestion into Neo4j."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import app.services as services
    import app.utils as utils
except ImportError:
    import services
    import utils


logger = logging.getLogger(__name__)

SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAPPING_FILE_PATTERN = re.compile(r"^(EMP\d+)\s*:\s*(.*?)\s*\((.*?)\)\s*$")


def resolve_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return utils.ROOT_DIR / candidate


def default_document_directory() -> Path:
    documents_ui = utils.ROOT_DIR / "data" / "documents_ui"
    if documents_ui.exists():
        return documents_ui
    return utils.ROOT_DIR / "data" / "documents"


def is_mapping_file(file_path: Union[str, Path]) -> bool:
    return "id mapping" in Path(file_path).name.lower()


def list_document_files(directory: Union[str, Path]) -> List[Path]:
    directory_path = resolve_path(directory)
    if not directory_path.exists():
        return []
    return sorted(
        [
            path
            for path in directory_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
            and not is_mapping_file(path)
        ],
        key=lambda path: path.name.lower(),
    )


def list_mapping_files(directory: Union[str, Path]) -> List[Path]:
    directory_path = resolve_path(directory)
    if not directory_path.exists():
        return []
    return sorted(
        [
            path
            for path in directory_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt" and is_mapping_file(path)
        ],
        key=lambda path: path.name.lower(),
    )


def extract_document_text(file_path: Union[str, Path]) -> str:
    path = resolve_path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return utils.extract_text_from_pdf(str(path))
    if suffix == ".docx":
        return utils.extract_text_from_docx(str(path))
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported document extension: {path.suffix}")


def load_document_identity(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = resolve_path(file_path)
    content = extract_document_text(path)
    return {
        "path": path,
        "content": content,
        "doc_id": utils.generate_doc_id(content),
    }


def extract_id_mappings(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    mappings: List[Dict[str, str]] = []
    path = resolve_path(file_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.lower().startswith("ids"):
                    continue
                match = MAPPING_FILE_PATTERN.match(line)
                if match:
                    emp_id, name, role = match.groups()
                    mappings.append({"id": emp_id, "name": name, "role": role})
    except Exception as exc:
        logger.error("Error reading ID mappings from %s: %s", path, exc)
    return mappings


def upsert_id_mappings_to_neo4j(mappings: List[Dict[str, str]]) -> bool:
    if not mappings:
        return False

    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            for row in mappings:
                session.run(
                    """
                    MERGE (p:Person {id: $id})
                    SET p.name = $name, p.role = $role
                    """,
                    id=row["id"],
                    name=row["name"],
                    role=row["role"],
                )
        return True
    except Exception as exc:
        logger.error("Error upserting ID mappings to Neo4j: %s", exc)
        return False
    finally:
        driver.close()


def build_document_payload(
    file_path: Union[str, Path],
    *,
    content: Optional[str] = None,
    doc_id: Optional[str] = None,
    source: str = "document_file",
) -> Dict[str, Any]:
    path = resolve_path(file_path)
    actual_content = content if content is not None else extract_document_text(path)
    actual_doc_id = doc_id or utils.generate_doc_id(actual_content)
    structured = services.extract_structured_data(actual_content, actual_doc_id)

    receivers = structured.get("receivers") or []
    return {
        "doc_id": actual_doc_id,
        "sender": structured.get("sender") or "Unknown",
        "receivers": [str(receiver) for receiver in receivers if str(receiver).strip()],
        "subject": structured.get("subject") or path.name,
        "content": structured.get("content") or actual_content,
        "timestamp": None,
        "source": source,
        "conversation_type": None,
        "conversation_id": None,
        "group_id": None,
        "attachment_name": path.name,
        "attachment_type": None,
        "attachment_url": None,
        "trace_json": None,
        "graph_sync_status": None,
    }


def document_exists(doc_id: str) -> bool:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            rows = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN d.doc_id AS doc_id
                LIMIT 1
                """,
                doc_id=doc_id,
            ).data()
            return bool(rows)
    except Exception as exc:
        logger.error("Failed to check existing document %s: %s", doc_id, exc)
        return False
    finally:
        driver.close()


def ingest_document_file(
    file_path: Union[str, Path],
    *,
    skip_existing: bool = True,
    source: str = "document_file",
) -> Dict[str, Any]:
    identity = load_document_identity(file_path)
    path = identity["path"]
    doc_id = identity["doc_id"]

    already_exists = document_exists(doc_id)
    if already_exists and skip_existing:
        return {
            "file": str(path),
            "doc_id": doc_id,
            "status": "skipped_duplicate",
            "stored": False,
            "subject": path.name,
        }

    payload = build_document_payload(
        path,
        content=identity["content"],
        doc_id=doc_id,
        source=source,
    )
    stored = services.store_in_neo4j(payload)
    return {
        "file": str(path),
        "doc_id": doc_id,
        "status": "stored" if stored else "failed",
        "stored": stored,
        "subject": payload["subject"],
    }


def ingest_document_directory(
    directory: Union[str, Path],
    *,
    skip_existing: bool = True,
    source: str = "document_file",
) -> Dict[str, Any]:
    directory_path = resolve_path(directory)
    if not directory_path.exists():
        return {
            "directory": str(directory_path),
            "exists": False,
            "mapping_files_processed": 0,
            "mapping_entries_upserted": 0,
            "document_files_seen": 0,
            "stored": 0,
            "skipped_duplicates": 0,
            "failed": 0,
            "results": [],
        }

    mapping_files_processed = 0
    mapping_entries_upserted = 0
    for mapping_file in list_mapping_files(directory_path):
        mappings = extract_id_mappings(mapping_file)
        if mappings and upsert_id_mappings_to_neo4j(mappings):
            mapping_files_processed += 1
            mapping_entries_upserted += len(mappings)

    results = [
        ingest_document_file(path, skip_existing=skip_existing, source=source)
        for path in list_document_files(directory_path)
    ]

    stored = sum(1 for item in results if item["status"] == "stored")
    skipped_duplicates = sum(1 for item in results if item["status"] == "skipped_duplicate")
    failed = sum(1 for item in results if item["status"] == "failed")

    return {
        "directory": str(directory_path),
        "exists": True,
        "mapping_files_processed": mapping_files_processed,
        "mapping_entries_upserted": mapping_entries_upserted,
        "document_files_seen": len(results),
        "stored": stored,
        "skipped_duplicates": skipped_duplicates,
        "failed": failed,
        "results": results,
    }
