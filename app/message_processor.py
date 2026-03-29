"""
Message Processor for SAGE Enterprise Graph RAG

This script:
1. Processes message files from the data/uploads directory
2. Extracts structured data from each message
3. Pushes the data to the Neo4j database
4. Generates question-answer pairs for evaluation
"""

import logging
import re
from typing import List, Dict, Any
from pathlib import Path

from langchain_groq import ChatGroq

try:
    import app.utils as utils
except ImportError:
    import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Keep ingestion logs readable by suppressing verbose dependency HTTP/debug logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

ROOT_DIR = utils.ROOT_DIR
NEO4J_DATABASE = utils.NEO4J_DATABASE
DATA_DIR = utils.ROOT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"


def get_neo4j_driver():
    return utils.create_neo4j_driver()

def get_session(driver):
    return utils.open_neo4j_session(driver, NEO4J_DATABASE)

def summarize_text_fallback(text: str, max_len: int = 600) -> str:
    """Create a lightweight summary when LLM is unavailable."""
    clean = " ".join(text.split())
    return clean[:max_len] if len(clean) > max_len else clean

def summarize_with_optional_llm(llm, text: str) -> str:
    """Use Groq summary when available; fallback to local summary on any error."""
    if not llm:
        return summarize_text_fallback(text)
    try:
        return llm.invoke(f"Summarize this content, include the word json in the summary: {text}").content
    except Exception as exc:
        logger.warning(f"Groq summary failed, using fallback summary: {exc}")
        return summarize_text_fallback(text)

def extract_id_mappings(file_path: str) -> List[Dict[str, str]]:
    """Extract employee ID to name/role mappings from a text file."""
    mappings: List[Dict[str, str]] = []
    pattern = re.compile(r"^(EMP\d+)\s*:\s*(.*?)\s*\((.*?)\)\s*$")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.lower().startswith("ids"):
                    continue
                match = pattern.match(line)
                if not match:
                    continue
                emp_id, name, role = match.groups()
                mappings.append({"id": emp_id, "name": name, "role": role})
    except Exception as exc:
        logger.error(f"Error reading ID mappings from {file_path}: {exc}")
        return []
    return mappings

def upsert_id_mappings_to_neo4j(mappings: List[Dict[str, str]]) -> bool:
    """Upsert employee metadata to Person nodes in Neo4j."""
    if not mappings:
        return False

    driver = get_neo4j_driver()
    try:
        with get_session(driver) as session:
            for row in mappings:
                session.run(
                    """
                    MERGE (p:Person {id: $id})
                    SET p.name = $name,
                        p.role = $role
                    """,
                    id=row["id"],
                    name=row["name"],
                    role=row["role"],
                )
        return True
    except Exception as exc:
        logger.error(f"Error upserting ID mappings to Neo4j: {exc}")
        return False
    finally:
        driver.close()

# Function to extract structured data from a message file
def extract_message_data(file_path: str) -> Dict[str, Any]:
    """Extract structured data from a message file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Support both legacy message files and documents_ui mail-style files.
        lines = content.strip().split('\n')

        # Parse fields from either format.
        sender_id = None
        receiver_ids: List[str] = []
        subject = None
        message_text = None
        sent_time = None

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('Sender ID:'):
                sender_id = stripped.replace('Sender ID:', '').strip()
            elif stripped.startswith('Sender:'):
                sender_id = stripped.replace('Sender:', '').strip()
            elif stripped.startswith('Receiver ID:'):
                receiver_ids = [stripped.replace('Receiver ID:', '').strip()]
            elif stripped.startswith('Receiver:'):
                receivers_raw = stripped.replace('Receiver:', '').strip()
                receiver_ids = [r.strip() for r in receivers_raw.split(',') if r.strip()]
            elif stripped.startswith('Subject:'):
                subject = stripped.replace('Subject:', '').strip()
            elif stripped.startswith('Message:'):
                message_text = stripped.replace('Message:', '').strip()
                next_line_index = idx + 1
                while next_line_index < len(lines) and not lines[next_line_index].startswith('Sent Time:'):
                    message_text += '\n' + lines[next_line_index]
                    next_line_index += 1
            elif stripped.startswith('Sent Time:'):
                sent_time = stripped.replace('Sent Time:', '').strip()

        # For mail-style docs, treat body after Subject as message content.
        if not message_text and subject:
            subject_index = None
            for idx, line in enumerate(lines):
                if line.strip().startswith('Subject:'):
                    subject_index = idx
                    break

            if subject_index is not None:
                body_lines = lines[subject_index + 1:]
                while body_lines and not body_lines[0].strip():
                    body_lines.pop(0)
                message_text = '\n'.join(body_lines).strip()

        # Validate required fields
        if not sender_id or not receiver_ids or not message_text:
            logger.error(f"Missing required fields in {file_path}")
            return None

        # Generate document ID
        doc_id = utils.generate_doc_id(content)

        # Create a fallback subject from the first few words if subject is missing.
        if not subject:
            words = message_text.split()
            subject = ' '.join(words[:min(5, len(words))]) + '...'

        return {
            "doc_id": doc_id,
            "sender": sender_id,
            "receivers": receiver_ids,
            "subject": subject,
            "content": message_text,
            "sent_time": sent_time
        }
    except Exception as e:
        logger.error(f"Error extracting data from {file_path}: {str(e)}")
        return None

# Function to store document in Neo4j
def store_in_neo4j(data: Dict[str, Any]) -> bool:
    """Store document and its chunks in Neo4j."""
    driver = get_neo4j_driver()
    llm = None
    if utils.GROQ_API_KEY:
        try:
            llm = ChatGroq(
                model_name=utils.GROQ_MODEL,
                temperature=0.0,
                model_kwargs={"response_format": {"type": "json_object"}},
                groq_api_key=utils.GROQ_API_KEY
            )
        except Exception as exc:
            logger.warning(f"Failed to initialize Groq client, using fallback summaries: {exc}")
            llm = None
    else:
        logger.warning("GROQ_API_KEY not found. Falling back to local summaries for ingestion.")

    try:
        with get_session(driver) as session:
            # Always upsert document/chunks/relationships to repair partial ingestions.

            # Create Document node
            document_summary = summarize_with_optional_llm(llm, data["content"])
            embedding = utils.generate_embedding(document_summary[:5000])
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.sender = $sender, d.subject = $subject, d.content = $content, d.embedding = $embedding, d.summary = $summary
                """,
                doc_id=data["doc_id"],
                sender=data["sender"],
                subject=data["subject"],
                content=data["content"],
                embedding=embedding,
                summary=document_summary
            )

            # Chunk and create Chunk nodes
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
                    doc_id=data["doc_id"]
                )

            # Create Sender and Receiver nodes and relationships
            session.run(
                """
                MERGE (s:Person {id: $sender_id})
                MERGE (d:Document {doc_id: $doc_id})
                MERGE (s)-[:SENT]->(d)
                """,
                sender_id=data["sender"],
                doc_id=data["doc_id"]
            )

            for receiver in data["receivers"]:
                session.run(
                    """
                    MERGE (r:Person {id: $receiver_id})
                    MERGE (d:Document {doc_id: $doc_id})
                    MERGE (d)-[:RECEIVED_BY]->(r)
                    """,
                    receiver_id=receiver,
                    doc_id=data["doc_id"]
                )

            logger.info(f"Successfully stored document {data['doc_id']} in Neo4j.")
            # Trigger SAIA to adjust existing knowledge given this new document
            # try:
            #     import under_development.saia as _saia
            #     _saia.trigger_saia(data["doc_id"], data["content"])
            # except Exception as e:
            #     logger.error(f"SAIA trigger error for {data['doc_id']}: {str(e)}")
            return True
    except Exception as e:
        logger.error(f"Error storing document in Neo4j: {str(e)}")
        return False
    finally:
        driver.close()

# Function to process all message files in a directory
def process_message_files(directory: str) -> List[Dict[str, Any]]:
    """Process all message files in the specified directory."""
    processed_data = []
    directory_path = Path(directory)
    if not directory_path.is_absolute():
        directory_path = ROOT_DIR / directory_path

    # Get all text files in the directory
    if not directory_path.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return processed_data

    file_paths = [str(p) for p in directory_path.iterdir() if p.suffix.lower() == ".txt" and p.is_file()]

    logger.info(f"Found {len(file_paths)} message files in {directory_path}")

    # Process files in batches to avoid overwhelming the database
    batch_size = 10
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")

        for file_path in batch:
            logger.info(f"Processing file: {file_path}")

            file_name = Path(file_path).name.lower()
            if "id mapping" in file_name:
                mappings = extract_id_mappings(file_path)
                if mappings and upsert_id_mappings_to_neo4j(mappings):
                    logger.info(f"Successfully ingested {len(mappings)} ID mappings from {file_path}")
                else:
                    logger.error(f"Failed to ingest ID mappings from {file_path}")
                continue

            # Extract data from file
            data = extract_message_data(file_path)
            if data:
                # Store in Neo4j
                success = store_in_neo4j(data)
                if success:
                    processed_data.append(data)
                    logger.info(f"Successfully processed {file_path}")
                else:
                    logger.error(f"Failed to store data from {file_path} in Neo4j")
            else:
                logger.error(f"Failed to extract data from {file_path}")

    return processed_data

# Main function
def main():
    """Main function to process messages."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process messages and generate QA pairs")
    parser.add_argument("--directory", type=str, default=str(UPLOADS_DIR), help="Directory containing message files")
    parser.add_argument("--skip-processing", action="store_true", help="Skip message processing")
    args = parser.parse_args()

    # Process message files
    if not args.skip_processing:
        logger.info(f"Processing message files from {args.directory}...")
        processed_data = process_message_files(args.directory)
        logger.info(f"Processed {len(processed_data)} message files.")
    else:
        logger.info("Skipping message processing.")

if __name__ == "__main__":
    main()
