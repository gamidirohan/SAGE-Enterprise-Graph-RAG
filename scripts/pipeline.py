"""Single Streamlit entrypoint for SAGE.

This app handles chat, document processing, message ingestion, and Neo4j
debugging in one place so the Streamlit workflow stays simple.
"""

from pathlib import Path
import logging
import sys
from typing import Any, Dict, List, Optional

import streamlit as st
from fastapi import HTTPException
from langchain_groq import ChatGroq

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import app.document_ingestion as document_ingestion
    import app.services as services
    import app.utils as utils
except ImportError:
    import document_ingestion
    import services
    import utils


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ROOT_DIR = utils.ROOT_DIR
NEO4J_DATABASE = utils.NEO4J_DATABASE
DATA_DIR = ROOT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def get_neo4j_driver():
    return utils.create_neo4j_driver()


def get_session(driver):
    return utils.open_neo4j_session(driver, NEO4J_DATABASE)


def summarize_text_fallback(text: str, max_len: int = 600) -> str:
    clean = " ".join(text.split())
    return clean[:max_len] if len(clean) > max_len else clean


def summarize_with_optional_llm(llm, text: str) -> str:
    if not llm:
        return summarize_text_fallback(text)
    try:
        return llm.invoke(f"Summarize this content, include the word json in the summary: {text}").content
    except Exception as exc:
        logger.warning(f"Groq summary failed, using fallback summary: {exc}")
        return summarize_text_fallback(text)


def extract_message_data(file_path: str) -> Dict[str, Any] | None:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.strip().split("\n")
        sender_id = None
        receiver_ids: List[str] = []
        subject = None
        message_text = None
        sent_time = None
        attachment_name = None

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("Sender ID:"):
                sender_id = stripped.replace("Sender ID:", "").strip()
            elif stripped.startswith("Sender:"):
                sender_id = stripped.replace("Sender:", "").strip()
            elif stripped.startswith("Receiver ID:"):
                receiver_ids = [stripped.replace("Receiver ID:", "").strip()]
            elif stripped.startswith("Receiver:"):
                receivers_raw = stripped.replace("Receiver:", "").strip()
                receiver_ids = [r.strip() for r in receivers_raw.split(",") if r.strip()]
            elif stripped.startswith("Subject:"):
                subject = stripped.replace("Subject:", "").strip()
            elif stripped.startswith("Message:"):
                message_text = stripped.replace("Message:", "").strip()
                next_line_index = idx + 1
                while next_line_index < len(lines) and not lines[next_line_index].startswith("Sent Time:"):
                    message_text += "\n" + lines[next_line_index]
                    next_line_index += 1
            elif stripped.startswith("Sent Time:"):
                sent_time = stripped.replace("Sent Time:", "").strip()
            elif stripped.startswith("Attachment:"):
                attachment_name = stripped.replace("Attachment:", "").strip()

        if not message_text and subject:
            subject_index = None
            for idx, line in enumerate(lines):
                if line.strip().startswith("Subject:"):
                    subject_index = idx
                    break
            if subject_index is not None:
                body_lines = lines[subject_index + 1 :]
                while body_lines and not body_lines[0].strip():
                    body_lines.pop(0)
                message_text = "\n".join(body_lines).strip()

        if not sender_id or not receiver_ids or not message_text:
            logger.error(f"Missing required fields in {file_path}")
            return None

        doc_id = utils.generate_doc_id(content)
        if not subject:
            words = message_text.split()
            subject = " ".join(words[: min(5, len(words))]) + "..."

        return {
            "doc_id": doc_id,
            "sender": sender_id,
            "receivers": receiver_ids,
            "subject": subject,
            "content": message_text,
            "timestamp": sent_time,
            "attachment_name": attachment_name,
            "source": "message_file",
        }
    except Exception as exc:
        logger.error(f"Error extracting data from {file_path}: {exc}")
        return None


def store_in_neo4j(data: Dict[str, Any]) -> bool:
    driver = get_neo4j_driver()
    llm = None
    if utils.GROQ_API_KEY:
        try:
            llm = ChatGroq(
                model_name=utils.GROQ_MODEL,
                temperature=0.0,
                model_kwargs={"response_format": {"type": "json_object"}},
                groq_api_key=utils.GROQ_API_KEY,
            )
        except Exception as exc:
            logger.warning(f"Failed to initialize Groq client, using fallback summaries: {exc}")
    else:
        logger.warning("GROQ_API_KEY not found. Falling back to local summaries for ingestion.")

    try:
        with get_session(driver) as session:
            document_summary = summarize_with_optional_llm(llm, data["content"])
            embedding = utils.generate_embedding(document_summary[:5000])
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.sender = $sender,
                    d.subject = $subject,
                    d.content = $content,
                    d.embedding = $embedding,
                    d.summary = $summary,
                    d.timestamp = $timestamp,
                    d.source = $source,
                    d.conversation_type = $conversation_type,
                    d.conversation_id = $conversation_id,
                    d.group_id = $group_id,
                    d.attachment_name = $attachment_name,
                    d.attachment_type = $attachment_type,
                    d.attachment_url = $attachment_url,
                    d.trace_json = $trace_json,
                    d.graph_sync_status = $graph_sync_status
                """,
                doc_id=data["doc_id"],
                sender=data["sender"],
                subject=data["subject"],
                content=data["content"],
                embedding=embedding,
                summary=document_summary,
                timestamp=data.get("timestamp"),
                source=data.get("source"),
                conversation_type=data.get("conversation_type"),
                conversation_id=data.get("conversation_id"),
                group_id=data.get("group_id"),
                attachment_name=data.get("attachment_name"),
                attachment_type=data.get("attachment_type"),
                attachment_url=data.get("attachment_url"),
                trace_json=data.get("trace_json"),
                graph_sync_status=data.get("graph_sync_status"),
            )

            chunks = utils.chunk_document(data["content"], max_chunk_words=250, overlap_sentences=2)
            for i, chunk in enumerate(chunks):
                chunk_summary = summarize_with_optional_llm(llm, chunk)
                chunk_embedding = utils.generate_embedding(chunk_summary)
                session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.content = $content, c.embedding = $embedding, c.summary = $summary
                    MERGE (d:Document {doc_id: $doc_id})
                    MERGE (c)-[:PART_OF]->(d)
                    """,
                    chunk_id=f"{data['doc_id']}-chunk-{i}",
                    content=chunk,
                    embedding=chunk_embedding,
                    summary=chunk_summary,
                    doc_id=data["doc_id"],
                )

            session.run(
                """
                MERGE (s:Person {id: $sender_id})
                MERGE (d:Document {doc_id: $doc_id})
                MERGE (s)-[:SENT]->(d)
                """,
                sender_id=data["sender"],
                doc_id=data["doc_id"],
            )
            for receiver in data["receivers"]:
                session.run(
                    """
                    MERGE (r:Person {id: $receiver_id})
                    MERGE (d:Document {doc_id: $doc_id})
                    MERGE (d)-[:RECEIVED_BY]->(r)
                    """,
                    receiver_id=receiver,
                    doc_id=data["doc_id"],
                )
        return True
    except Exception as exc:
        logger.error(f"Error storing document in Neo4j: {exc}")
        return False
    finally:
        driver.close()


def process_message_files(directory: str) -> List[Dict[str, Any]]:
    processed_data: List[Dict[str, Any]] = []
    directory_path = Path(directory)
    if not directory_path.is_absolute():
        directory_path = ROOT_DIR / directory_path
    if not directory_path.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return processed_data

    file_paths = [str(p) for p in directory_path.iterdir() if p.suffix.lower() == ".txt" and p.is_file()]
    for file_path in file_paths:
        file_name = Path(file_path).name.lower()
        if "id mapping" in file_name:
            mappings = document_ingestion.extract_id_mappings(file_path)
            if mappings and document_ingestion.upsert_id_mappings_to_neo4j(mappings):
                logger.info(f"Successfully ingested {len(mappings)} ID mappings from {file_path}")
            continue
        data = extract_message_data(file_path)
        if data and store_in_neo4j(data):
            processed_data.append(data)
    return processed_data


def extract_structured_data(document_text: str, doc_id: str):
    return services.extract_structured_data(document_text, doc_id)


def render_chat_tab():
    st.subheader("Chat")
    question = st.text_input("Ask a question")
    if st.button("Ask") and question:
        graph_results = services.query_graph(question)
        response = services.generate_groq_response(question, graph_results)
        st.write(response["answer"])


def render_document_tab():
    st.subheader("Document Ingestion")
    files_dir = document_ingestion.default_document_directory()
    document_files = document_ingestion.list_document_files(files_dir)
    file_names = [path.name for path in document_files]
    if not file_names:
        st.warning(f"No supported documents found in directory: {files_dir}")
        return

    selected_file = st.selectbox("Select a file", file_names)
    file_path = files_dir / selected_file

    if st.button("Preview Selected Document"):
        payload = document_ingestion.build_document_payload(file_path)
        st.json(
            {
                "doc_id": payload["doc_id"],
                "sender": payload["sender"],
                "receivers": payload["receivers"],
                "subject": payload["subject"],
                "source": payload["source"],
            }
        )

    if st.button("Ingest Selected Document"):
        result = document_ingestion.ingest_document_file(file_path, skip_existing=True)
        if result["status"] == "stored":
            st.success(f"Stored {result['subject']} in Neo4j.")
        elif result["status"] == "skipped_duplicate":
            st.info(f"Skipped duplicate document {result['subject']} based on content hash.")
        else:
            st.error(f"Failed to ingest {result['subject']}.")

    st.divider()
    st.subheader("Batch Document Ingestion")
    batch_directory = st.text_input("Document directory", value=str(files_dir), key="document_batch_directory")
    skip_existing = st.checkbox("Skip duplicates by content hash", value=True)
    if st.button("Batch Ingest Documents"):
        summary = document_ingestion.ingest_document_directory(batch_directory, skip_existing=skip_existing)
        if not summary["exists"]:
            st.error(f"Directory does not exist: {summary['directory']}")
            return

        st.success(
            "Stored "
            f"{summary['stored']} documents, skipped {summary['skipped_duplicates']} duplicates, "
            f"failed {summary['failed']}."
        )
        if summary["mapping_entries_upserted"]:
            st.info(
                "Upserted "
                f"{summary['mapping_entries_upserted']} people from {summary['mapping_files_processed']} mapping file(s)."
            )
        if summary["results"]:
            st.json(summary["results"])


def render_message_tab():
    st.subheader("Batch Message Ingestion")
    directory = st.text_input("Directory", value=str(UPLOADS_DIR))
    if st.button("Process Messages"):
        processed = process_message_files(directory)
        st.success(f"Processed {len(processed)} files")


def render_debug_tab():
    st.subheader("Debug Graph")
    if st.button("Refresh Debug"):
        driver = utils.create_neo4j_driver()
        try:
            with utils.open_neo4j_session(driver, NEO4J_DATABASE) as session:
                st.json(session.run("MATCH (n) RETURN count(n) AS nodes").data())
        finally:
            driver.close()


st.set_page_config(page_title="SAGE", layout="wide")
st.title("SAGE")
tabs = st.tabs(["Chat", "Documents", "Messages", "Debug"])
with tabs[0]:
    render_chat_tab()
with tabs[1]:
    render_document_tab()
with tabs[2]:
    render_message_tab()
with tabs[3]:
    render_debug_tab()
