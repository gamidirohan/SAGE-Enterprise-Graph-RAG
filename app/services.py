"""Shared application services for SAGE.

This file centralizes reusable document-extraction, graph-retrieval, and LLM
response logic so the Streamlit and FastAPI entrypoints stay smaller and cleaner.
"""

import logging
import re
from typing import Any, Dict, List

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
    RETURN c.summary AS chunk_summary, d, similarity, type(r) as relationship, n
"""

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
    {
        "doc_id": "<hashed_document_id>",
        "sender": "<sender_id>",
        "receivers": ["<receiver_id1>", "<receiver_id2>"],
        "subject": "<subject>",
        "content": "<content>"
    }

    Input document:
    {input}
    """
)

CHAT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are SAGE, an intelligent and friendly AI assistant specialized in retrieving and explaining information from documents.
    Your tone is helpful, conversational, and slightly enthusiastic. You refer to yourself as "I" and address the user directly.

    When answering questions:
    - Be concise but thorough
    - Use a friendly, conversational tone
    - If you're not sure about something, be honest about it
    - Occasionally use phrases like "I found" or "According to the documents" to emphasize your retrieval capabilities
    - Format your responses in a readable way with paragraphs and bullet points when appropriate

    Here is the user's question: {query}

    Here is the relevant context from the documents (keep in mind you get limited context and that's what you should work with):
    {context}

    Respond to the user's question in a helpful, conversational way based on the context provided:
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


def query_graph(user_input: str) -> List[str]:
    driver = None
    try:
        driver = utils.create_neo4j_driver()
        model = utils.get_cached_embedding_model()
        query_embedding = np.array(model.encode(user_input), dtype=np.float32)

        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            vector_results = session.run(
                GRAPH_VECTOR_QUERY,
                query_embedding=query_embedding.tolist(),
            ).data()

        if not vector_results:
            return [
                "I don't seem to have any relevant information about that in my knowledge base. Let me know if you'd like to ask about something else!"
            ]

        return [
            f"Chunk Summary: {item.get('chunk_summary', 'No summary')}, "
            f"Document: {item.get('d', {})}, "
            f"Similarity: {item.get('similarity', 0)}, "
            f"Relationship: {item.get('relationship', 'unknown')}, "
            f"Related Node: {item.get('n', {})}"
            for item in vector_results
        ]
    except Exception as exc:
        logger.error(f"Vector search failed: {exc}")
        return [
            "I encountered a technical issue while searching for information. I'd be happy to try again if you rephrase your question!"
        ]
    finally:
        if driver:
            driver.close()


def generate_groq_response(query: str, documents: List[str]) -> Dict[str, List[str] | str]:
    if not documents:
        return {
            "answer": "I've searched through my knowledge base, but I don't have any specific information about that topic yet. Would you like to ask about something else or perhaps upload a document with this information?",
            "thinking": [],
        }

    try:
        context = "\n\n".join(_extract_context_parts(documents))
        llm = _create_groq_client(temperature=0.3)
        chain = CHAT_PROMPT | llm | StrOutputParser()
        response = chain.invoke({"query": query, "context": context})

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
