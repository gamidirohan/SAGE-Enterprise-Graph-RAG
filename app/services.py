"""Business logic and LLM services for SAGE.

Use this file for prompt templates, document extraction, graph retrieval,
chat response generation, and other domain-level application behavior.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

try:
    import app.utils as utils
except ImportError:
    import utils


logger = logging.getLogger(__name__)

GRAPH_VECTOR_QUERY = """
    MATCH (c:Chunk)-[:PART_OF]->(d:Document)
    WHERE c.embedding IS NOT NULL
    WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
    WITH c, d, gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT 3
    MATCH (c)-[r]-(n)
    RETURN c.chunk_id AS chunk_id, c.summary AS chunk_summary, d, similarity, type(r) as relationship, n
"""

PERSON_GRAPH_VECTOR_QUERY = """
    MATCH (p:Person {id: $user_id})
    MATCH (p)-[:SENT|RECEIVED_BY]-(d:Document)<-[:PART_OF]-(c:Chunk)
    WHERE c.embedding IS NOT NULL
    WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
    WITH c, d, gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT 3
    MATCH (c)-[r]-(n)
    RETURN c.chunk_id AS chunk_id, c.summary AS chunk_summary, d, similarity, type(r) as relationship, n
"""

FIRST_PERSON_PATTERN = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)

DOCUMENT_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string"},
        "sender": {"type": "string"},
        "receivers": {"type": "array", "items": {"type": "string"}},
        "subject": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["doc_id", "sender", "receivers", "subject", "content"],
}

DOCUMENT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an advanced document intelligence system. Extract Sender, Receivers, Subject and content from the following document.

    Instructions:
    1. Extract the Sender ID.
    2. Extract the Receiver IDs as an array.
    3. Extract the Subject.
    4. Extract the main Content.

    Output format (JSON only):
    {{
        "doc_id": "<hashed_document_id>",
        "sender": "<sender_id>",
        "receivers": ["<receiver_id1>", "<receiver_id2>"],
        "subject": "<subject>",
        "content": "<content>"
    }}

    Input document:
    {input}
    """
)

CHAT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are SAGE, an enterprise Graph-RAG assistant.
    SAGE proves answers instead of giving conversational fluff.
    Your tone is precise, calm, professional, and evidence-driven.
    Do not sound casual, overly enthusiastic, or vague.

    Answering style:
    - Lead with the direct answer first
    - Then show the supporting evidence in a structured way
    - Use short sections such as `Answer`, `Findings`, and `Evidence and Provenance` when useful
    - If the user asks for a list, violations, approvals, delays, responsibility, policy checks, or audit-style output, prefer a numbered findings format
    - Mention names, dates, document IDs, senders, timestamps, relationships, and policy references explicitly when they are present in the context
    - Do not replace a known person with vague phrases like "someone" or "a person"
    - If evidence is incomplete, weak, or missing, say that clearly instead of overstating confidence
    - Separate confirmed facts from inference
    - Do not invent graph paths, document IDs, policy IDs, timestamps, Cypher queries, approvals, or reasoning steps that are not supported by the provided context

    Preferred response shape:
    - For direct managerial questions, start with `Answer:` followed by 1 concise paragraph
    - Then add `Evidence and Provenance:` with flat bullet points summarizing the strongest support
    - For audit/compliance or multi-result queries, start with `Findings:` and enumerate the results
    - After the findings, add `Evidence and Provenance:` with the supporting references
    - If a requested provenance item is not available from the retrieved context, say `Not available in retrieved evidence`

    Writing standard:
    - Be concise, but make the reasoning traceable
    - Prefer operational clarity over conversational warmth
    - Sound like a system built for managers, auditors, and enterprise review
    - Avoid filler language, hedging that adds no value, or generic assistant phrases

    Here is the user's question: {query}

    Identity context:
    {user_context}

    Here is the relevant context from the documents (keep in mind you get limited context and that's what you should work with):
    {context}

    Respond to the user's question in that enterprise, evidence-driven style using only the context provided:
    """
)


def _create_groq_client(*, temperature: float, require_json: bool = False):
    kwargs: Dict[str, Any] = {
        "model_name": utils.GROQ_MODEL,
        "temperature": temperature,
        "groq_api_key": utils.GROQ_API_KEY,
    }
    if require_json:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatGroq(**kwargs)


def _extract_context_parts(documents: List[str]) -> List[str]:
    context_parts: List[str] = []
    for item in documents:
        try:
            context_parts.append(item.split("Chunk Summary: ")[1].split(", Document: ")[0])
        except (IndexError, AttributeError):
            context_parts.append(str(item))
    return context_parts


def _contains_first_person(text: str) -> bool:
    return bool(FIRST_PERSON_PATTERN.search(text))


def _classify_query(text: str) -> str:
    lowered = text.lower()
    if _contains_first_person(text):
        return "personal_context"
    if any(token in lowered for token in ("weekend", "today", "tomorrow", "schedule", "meeting", "plan")):
        return "schedule_or_timeline"
    if any(token in lowered for token in ("why", "reason", "cause", "delayed")):
        return "explanation"
    if any(token in lowered for token in ("who", "whose", "person", "people")):
        return "person_lookup"
    return "general_search"


def _serialize_neo4j_entity(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    serialized: Dict[str, Any]
    if isinstance(value, dict):
        serialized = dict(value)
    elif hasattr(value, "items"):
        serialized = dict(value.items())
    else:
        try:
            serialized = dict(value)
        except Exception:
            serialized = {"value": str(value)}

    labels = list(getattr(value, "labels", []))
    if labels:
        serialized["_labels"] = labels

    element_id = getattr(value, "element_id", None)
    if element_id:
        serialized["_element_id"] = element_id

    return serialized


def _get_primary_label(entity: Dict[str, Any]) -> str:
    labels = entity.get("_labels") or []
    if labels:
        return str(labels[0])
    return "Node"


def _get_display_name(entity: Dict[str, Any]) -> str:
    for key in ("name", "id", "doc_id", "title", "subject", "value"):
        value = entity.get(key)
        if value:
            return str(value)
    return _get_primary_label(entity)


def _build_path_summary(user_scoped: bool, related_label: Optional[str]) -> Dict[str, Any]:
    nodes = ["Person", "Document", "Chunk"] if user_scoped else ["Document", "Chunk"]
    if related_label:
        nodes.append(related_label)
    return {
        "nodes": nodes,
        "path": " -> ".join(nodes),
        "hop_count": max(len(nodes) - 1, 0),
    }


def _merge_ranked_results(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    by_chunk: Dict[str, Dict[str, Any]] = {}
    for row in primary + secondary:
        chunk_id = str(row.get("chunk_id", ""))
        existing = by_chunk.get(chunk_id)
        if existing is None or row.get("similarity", 0) > existing.get("similarity", 0):
            by_chunk[chunk_id] = row
    merged = list(by_chunk.values())
    merged.sort(key=lambda item: item.get("similarity", 0), reverse=True)
    return merged[:limit]


def extract_structured_data(document_text: str, doc_id: str) -> Dict[str, Any]:
    if not utils.GROQ_API_KEY:
        return {
            "doc_id": doc_id,
            "sender": "Unknown",
            "receivers": [],
            "subject": "No Subject",
            "content": document_text,
        }

    llm = _create_groq_client(temperature=0.0, require_json=True)
    parser = JsonOutputParser(pydantic_object=DOCUMENT_EXTRACTION_SCHEMA)
    chain = DOCUMENT_EXTRACTION_PROMPT | llm | parser
    structured_data = chain.invoke({"input": document_text})

    structured_data["doc_id"] = doc_id
    structured_data["sender"] = structured_data.get("sender") or "Unknown"
    structured_data["receivers"] = structured_data.get("receivers") or []
    structured_data["subject"] = structured_data.get("subject") or "No Subject"
    structured_data["content"] = structured_data.get("content") or document_text
    return structured_data


def query_graph_with_trace(user_input: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    driver = None
    personalized_lookup = bool(user_id and _contains_first_person(user_input))
    query_type = _classify_query(user_input)

    try:
        driver = utils.create_neo4j_driver()
        model = utils.get_cached_embedding_model()
        query_text = user_input if not personalized_lookup else f"{user_input}\nAuthenticated user id: {user_id}"
        query_embedding = np.array(model.encode(query_text), dtype=np.float32)

        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            global_results = session.run(
                GRAPH_VECTOR_QUERY,
                query_embedding=query_embedding.tolist(),
            ).data()
            person_results: List[Dict[str, Any]] = []
            if personalized_lookup:
                person_results = session.run(
                    PERSON_GRAPH_VECTOR_QUERY,
                    user_id=user_id,
                    query_embedding=query_embedding.tolist(),
                ).data()

        vector_results = _merge_ranked_results(person_results, global_results, limit=5)
        evidence: List[Dict[str, Any]] = []
        documents: List[str] = []
        matched_entities: List[str] = []

        for item in vector_results:
            document = _serialize_neo4j_entity(item.get("d"))
            related_node = _serialize_neo4j_entity(item.get("n"))
            related_label = _get_primary_label(related_node) if related_node else None
            related_name = _get_display_name(related_node) if related_node else None
            path_summary = _build_path_summary(personalized_lookup, related_label)

            sender = document.get("sender")
            subject = document.get("subject")
            doc_id = document.get("doc_id")
            similarity = round(float(item.get("similarity", 0) or 0), 4)
            relationship = item.get("relationship") or "RELATED_TO"

            for candidate in (sender, subject, related_name):
                if candidate and candidate not in matched_entities:
                    matched_entities.append(str(candidate))

            evidence_item = {
                "chunk_id": item.get("chunk_id"),
                "chunk_summary": item.get("chunk_summary", "No summary"),
                "similarity": similarity,
                "relationship": relationship,
                "retrieval_path": path_summary["path"],
                "hop_count": path_summary["hop_count"],
                "document": {
                    "doc_id": doc_id,
                    "subject": subject,
                    "sender": sender,
                    "timestamp": document.get("timestamp"),
                    "source": document.get("source"),
                },
                "related_node": {
                    "label": related_label,
                    "display_name": related_name,
                    "id": related_node.get("id"),
                } if related_node else {},
            }
            evidence.append(evidence_item)

            documents.append(
                "Chunk Summary: "
                f"{evidence_item['chunk_summary']}, "
                f"Document ID: {doc_id or 'unknown'}, "
                f"Subject: {subject or 'No Subject'}, "
                f"Sender: {sender or 'Unknown'}, "
                f"Similarity: {similarity}, "
                f"Relationship: {relationship}, "
                f"Related Node: {related_name or 'Unknown'}"
            )

        if not documents:
            documents = [
                "I don't seem to have any relevant information about that in my knowledge base. Let me know if you'd like to ask about something else!"
            ]

        trace = {
            "query": user_input,
            "query_type": query_type,
            "user_scoped": personalized_lookup,
            "user_id": user_id,
            "matched_entities": matched_entities,
            "result_count": len(evidence),
            "max_hop_count": max((item["hop_count"] for item in evidence), default=0),
            "retrieval_path": evidence[0]["retrieval_path"] if evidence else _build_path_summary(personalized_lookup, None)["path"],
            "evidence": evidence,
        }
        return {"documents": documents, "trace": trace}
    except Exception as exc:
        logger.error(f"Vector search failed: {exc}")
        return {
            "documents": [
                "I encountered a technical issue while searching for information. I'd be happy to try again if you rephrase your question!"
            ],
            "trace": {
                "query": user_input,
                "query_type": "error",
                "user_scoped": personalized_lookup,
                "user_id": user_id,
                "matched_entities": [],
                "result_count": 0,
                "max_hop_count": 0,
                "retrieval_path": _build_path_summary(personalized_lookup, None)["path"],
                "evidence": [],
                "error": str(exc),
            },
        }
    finally:
        if driver:
            driver.close()


def query_graph(user_input: str, user_id: Optional[str] = None) -> List[str]:
    return query_graph_with_trace(user_input, user_id=user_id)["documents"]


def generate_groq_response(query: str, documents: List[str], user_id: Optional[str] = None) -> Dict[str, List[str] | str]:
    if not documents:
        return {
            "answer": "I've searched through my knowledge base, but I don't have any specific information about that topic yet. Would you like to ask about something else or perhaps upload a document with this information?",
            "thinking": [],
        }

    try:
        context = "\n\n".join(_extract_context_parts(documents))
        user_context = "No authenticated user context was provided."
        if user_id:
            user_context = f"Authenticated user id: {user_id}."
            if _contains_first_person(query):
                user_context += " Treat first-person references (I/me/my) as this user unless the query says otherwise."
        llm = _create_groq_client(temperature=0.3)
        chain = CHAT_PROMPT | llm | StrOutputParser()
        response = chain.invoke({"query": query, "context": context, "user_context": user_context})

        think_parts = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
        answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if not answer:
            answer = (
                "I'm sorry, but I couldn't find a specific answer to your question in the documents I have access to. "
                "Could you try rephrasing your question or asking about something else? I'm here to help!"
            )

        return {"answer": answer, "thinking": think_parts if think_parts else []}
    except Exception as exc:
        logger.error(f"Groq API error: {exc}")
        return {
            "answer": "I'm sorry, but I seem to be having a bit of trouble processing your question right now. Could we try again in a moment? If the issue persists, you might want to try rephrasing your question. I'm eager to help once we get past this hiccup!",
            "thinking": [f"Error: {exc}"],
        }


def generate_streamlit_response(query: str, documents: List[str]) -> str:
    if not documents:
        return "No relevant information found."
    return generate_groq_response(query, documents)["answer"]


def _summarize_text_fallback(text: str, max_len: int = 600) -> str:
    clean = " ".join(text.split())
    return clean[:max_len] if len(clean) > max_len else clean


def _summarize_with_optional_llm(llm, text: str) -> str:
    if not llm:
        return _summarize_text_fallback(text)
    try:
        return llm.invoke(f"Summarize this content, include the word json in the summary: {text}").content
    except Exception as exc:
        logger.warning(f"Groq summary failed, using fallback summary: {exc}")
        return _summarize_text_fallback(text)


_SHORT_CONTENT_CHAR_LIMIT = 500
_SHORT_CONTENT_WORD_LIMIT = 200


def _document_exists(session, doc_id: str) -> bool:
    """Fast check: does a Document node with this doc_id already exist?"""
    rows = session.run(
        "MATCH (d:Document {doc_id: $doc_id}) RETURN d.doc_id AS id LIMIT 1",
        doc_id=doc_id,
    ).data()
    return bool(rows)


def _smart_summarize(llm, content: str) -> str:
    """Skip expensive LLM summarization for short content (e.g. chat messages).

    For text under _SHORT_CONTENT_CHAR_LIMIT characters, the content itself
    is a perfectly adequate summary. LLM summarization is reserved for longer
    documents where compression actually adds value.
    """
    if len(content) <= _SHORT_CONTENT_CHAR_LIMIT:
        return _summarize_text_fallback(content)
    return _summarize_with_optional_llm(llm, content)


def store_in_neo4j(data: Dict[str, Any]) -> bool:
    driver = utils.create_neo4j_driver()

    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            # ── Optimization 1: skip entirely if document already ingested ──
            if _document_exists(session, data["doc_id"]):
                logger.debug("Document %s already exists, skipping ingestion.", data["doc_id"])
                return True

            # ── Lazy LLM init: only create when we actually need it ──
            llm = None
            content = data["content"]
            needs_llm = len(content) > _SHORT_CONTENT_CHAR_LIMIT
            if needs_llm and utils.GROQ_API_KEY:
                try:
                    llm = _create_groq_client(temperature=0.0, require_json=True)
                except Exception as exc:
                    logger.warning(f"Failed to initialize Groq client, using fallback summaries: {exc}")
            elif needs_llm:
                logger.warning("GROQ_API_KEY not found. Falling back to local summaries for ingestion.")

            # ── Optimization 2: smart summarization ──
            document_summary = _smart_summarize(llm, content)
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
                content=content,
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

            # ── Optimization 3: skip chunking for short content ──
            word_count = len(content.split())
            if word_count <= _SHORT_CONTENT_WORD_LIMIT:
                # Short content: store as a single chunk, no splitting needed
                chunk_embedding = utils.generate_embedding(document_summary)
                session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.content = $content, c.embedding = $embedding, c.summary = $summary
                    MERGE (d:Document {doc_id: $doc_id})
                    MERGE (c)-[:PART_OF]->(d)
                    """,
                    chunk_id=f"{data['doc_id']}-chunk-0",
                    content=content,
                    embedding=chunk_embedding,
                    summary=document_summary,
                    doc_id=data["doc_id"],
                )
            else:
                # Long content: full chunking pipeline with LLM summaries
                chunks = utils.chunk_document(content, max_chunk_words=250, overlap_sentences=2)
                for i, chunk in enumerate(chunks):
                    chunk_summary = _smart_summarize(llm, chunk)
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
