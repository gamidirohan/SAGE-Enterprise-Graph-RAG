"""Business logic and LLM services for SAGE.

Use this file for prompt templates, document extraction, graph retrieval,
chat response generation, and other domain-level application behavior.
"""

import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

try:
    import app.utils as utils
except ImportError:
    import utils


logger = logging.getLogger(__name__)

IST_TIMEZONE = ZoneInfo("Asia/Kolkata")
ISO_OFFSET_TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b"
)
RECENCY_BOOST_MAX = 0.18
RECENCY_DECAY_DAYS = 21.0

GRAPH_VECTOR_QUERY = """
    MATCH (c:Chunk)-[:PART_OF]->(d:Document)
    WHERE c.embedding IS NOT NULL
      AND coalesce(d.conversation_type, '') <> 'sage'
      AND NOT coalesce(d.source, '') STARTS WITH 'sage_'
    WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
    WITH c, d, gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT 3
    MATCH (c)-[r]-(n)
    WITH c, d, similarity, r, n
    RETURN
        c.chunk_id AS chunk_id,
        c.summary AS chunk_summary,
        d,
        similarity,
        type(r) AS relationship,
        n,
        2 AS hop_count,
        [
            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
            coalesce(c.subject, c.title, c.name, c.id, c.doc_id, c.chunk_id, labels(c)[0]),
            coalesce(n.subject, n.title, n.name, n.id, n.doc_id, labels(n)[0])
        ] AS path_nodes,
        ['PART_OF', type(r)] AS path_relationships
"""

PERSON_GRAPH_VECTOR_QUERY = """
    MATCH (person:Person {id: $user_id})
    MATCH (person)-[pd:SENT|RECEIVED_BY]-(d:Document)<-[:PART_OF]-(c:Chunk)
    WHERE c.embedding IS NOT NULL
      AND coalesce(d.conversation_type, '') <> 'sage'
      AND NOT coalesce(d.source, '') STARTS WITH 'sage_'
    WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
    WITH c, d, gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT 3
    MATCH (c)-[r]-(n)
    WITH person, pd, c, d, similarity, r, n
    RETURN
        c.chunk_id AS chunk_id,
        c.summary AS chunk_summary,
        d,
        similarity,
        type(r) AS relationship,
        n,
        3 AS hop_count,
        [
            coalesce(person.subject, person.title, person.name, person.id, person.doc_id, labels(person)[0]),
            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
            coalesce(c.subject, c.title, c.name, c.id, c.doc_id, c.chunk_id, labels(c)[0]),
            coalesce(n.subject, n.title, n.name, n.id, n.doc_id, labels(n)[0])
        ] AS path_nodes,
        [type(pd), 'PART_OF', type(r)] AS path_relationships
"""

FACT_VECTOR_QUERY = """
    MATCH (f:CanonicalFact)
    WHERE f.status = 'current' AND f.embedding IS NOT NULL
    WITH f, f.embedding AS fact_embedding, $query_embedding AS query_embedding
    WITH f, gds.similarity.cosine(fact_embedding, query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT 3
    OPTIONAL MATCH (f)<-[:SUPPORTS]-(claim:Claim)<-[:HAS_CLAIM]-(d:Document)
    WITH f, similarity, collect(DISTINCT d)[0] AS d
    RETURN f.fact_id AS fact_id, f.summary AS fact_summary, f, d, similarity
"""

PERSON_FACT_VECTOR_QUERY = """
    MATCH (p:Person {id: $user_id})-[:HAS_FACT]-(f:CanonicalFact)
    WHERE f.status = 'current' AND f.embedding IS NOT NULL
    WITH f, f.embedding AS fact_embedding, $query_embedding AS query_embedding
    WITH f, gds.similarity.cosine(fact_embedding, query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT 3
    OPTIONAL MATCH (f)<-[:SUPPORTS]-(claim:Claim)<-[:HAS_CLAIM]-(d:Document)
    WITH f, similarity, collect(DISTINCT d)[0] AS d
    RETURN f.fact_id AS fact_id, f.summary AS fact_summary, f, d, similarity
"""

PERSON_TASK_FACT_QUERY = """
    MATCH (f:CanonicalFact)
    WHERE f.status = 'current'
      AND f.claim_type IN $claim_types
      AND (
        f.subject_entity_id = $user_id
        OR f.subject_key = $user_id
        OR f.object_entity_id = $user_id
        OR f.object_key = $user_id
      )
    OPTIONAL MATCH (f)<-[:SUPPORTS]-(claim:Claim)<-[:HAS_CLAIM]-(d:Document)
    WITH f, collect(DISTINCT d)[0] AS d
    RETURN f.fact_id AS fact_id, f.summary AS fact_summary, f, d, 1.0 AS similarity
    ORDER BY coalesce(f.last_seen_at, f.first_seen_at, '') DESC
    LIMIT 5
"""

FACT_VECTOR_QUERY = """
    MATCH (f:CanonicalFact)
    WHERE f.status = 'current' AND f.embedding IS NOT NULL
    WITH f, f.embedding AS fact_embedding, $query_embedding AS query_embedding
    WITH f, gds.similarity.cosine(fact_embedding, query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT 3
    OPTIONAL MATCH (f)<-[:SUPPORTS]-(claim:Claim)<-[:HAS_CLAIM]-(d:Document)
    WITH f, similarity, collect(DISTINCT d)[0] AS d
    RETURN f.fact_id AS fact_id, f.summary AS fact_summary, f, d, similarity
"""

PERSON_FACT_VECTOR_QUERY = """
    MATCH (p:Person {id: $user_id})-[:HAS_FACT]-(f:CanonicalFact)
    WHERE f.status = 'current' AND f.embedding IS NOT NULL
    WITH f, f.embedding AS fact_embedding, $query_embedding AS query_embedding
    WITH f, gds.similarity.cosine(fact_embedding, query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT 3
    OPTIONAL MATCH (f)<-[:SUPPORTS]-(claim:Claim)<-[:HAS_CLAIM]-(d:Document)
    WITH f, similarity, collect(DISTINCT d)[0] AS d
    RETURN f.fact_id AS fact_id, f.summary AS fact_summary, f, d, similarity
"""

PERSON_TASK_FACT_QUERY = """
    MATCH (f:CanonicalFact)
    WHERE f.status = 'current'
      AND f.claim_type IN $claim_types
      AND (
        f.subject_entity_id = $user_id
        OR f.subject_key = $user_id
        OR f.object_entity_id = $user_id
        OR f.object_key = $user_id
      )
    OPTIONAL MATCH (f)<-[:SUPPORTS]-(claim:Claim)<-[:HAS_CLAIM]-(d:Document)
    WITH f, collect(DISTINCT d)[0] AS d
    RETURN f.fact_id AS fact_id, f.summary AS fact_summary, f, d, 1.0 AS similarity
    ORDER BY coalesce(f.last_seen_at, f.first_seen_at, '') DESC
    LIMIT 5
"""

FIRST_PERSON_PATTERN = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
TASK_LOOKUP_PATTERN = re.compile(
    r"\b("
    r"promise|promised|commit|committed|commitment|agreed|supposed to|meant to|"
    r"assigned|assignment|working on|responsible for|deadline|due|by when|"
    r"send|share|deliver|submit|upload|provide|finish|complete"
    r")\b",
    re.IGNORECASE,
)
TASK_LIKE_FACT_TYPES = {"TASK_ASSIGNMENT", "ASSIGNMENT_STATE", "MEETING_EVENT"}
FACT_PRIORITY_QUERY_TYPES = {"task_commitment_lookup"}
ANSWER_PAYLOAD_SCHEMA_VERSION = 1
ANSWER_MODE_SHORT = "short"
ANSWER_MODE_LONG = "long"
REASON_CODE_EXPLICIT_SHORT = "explicit_short"
REASON_CODE_EXPLICIT_LONG = "explicit_long"
REASON_CODE_DIRECT_LOOKUP = "direct_lookup"
REASON_CODE_BROAD_OR_EXPLANATORY = "broad_or_explanatory"
REASON_CODE_EVIDENCE_COMPLEXITY = "evidence_complexity"
REASON_CODE_FALLBACK_INVALID_JSON = "fallback_invalid_json"

SHORT_OVERRIDE_PHRASES = (
    "brief",
    "short",
    "quick answer",
    "one line",
    "tl;dr",
)
LONG_OVERRIDE_PHRASES = (
    "detailed",
    "explain",
    "walk me through",
    "summarize",
    "summary",
    "compare",
    "audit",
    "anything about",
    "everything",
    "provenance",
    "all mentions",
    "overview",
)
BROAD_SCOPE_PHRASES = (
    "anything about",
    "everything",
    "all mentions",
    "overview",
    "walk me through",
    "all dashboard-related conversations",
    "everything we know",
)
DIRECT_LOOKUP_PREFIX = re.compile(r"^\s*(who|whom|what|when|which|did|do|does|is|are|was|were|am|can)\b", re.IGNORECASE)
QUERY_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
QUERY_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
QUERY_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b")
COMPOUND_LOOKUP_PATTERN = re.compile(r"\b(what|when|who|whom|which)\b", re.IGNORECASE)
QUERY_FOCUS_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "for",
    "with",
    "by",
    "on",
    "in",
    "of",
    "and",
    "or",
    "from",
    "into",
    "about",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "was",
    "were",
    "be",
    "been",
    "being",
    "now",
    "that",
    "this",
    "these",
    "those",
    "who",
    "whom",
    "what",
    "when",
    "where",
    "why",
    "how",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "am",
    "can",
    "will",
    "would",
    "should",
    "could",
    "tell",
    "show",
    "give",
    "anything",
    "everything",
    "asked",
    "ask",
    "asking",
    "request",
    "requested",
    "requesting",
    "send",
    "sending",
    "share",
    "sharing",
    "provide",
    "providing",
    "deliver",
    "delivering",
    "review",
    "reviewing",
    "report",
    "reports",
    "reporting",
    "current",
    "currently",
}

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
    Return JSON only with this exact shape:
    {{
      "summary": "string",
      "bullets": ["string"]
    }}

    Visible answer contract:
    - `summary` is always required, non-empty, and contains only user-facing chat text
    - `bullets` contains only extra user-facing detail points; use an empty array if no extra detail is needed
    - Do not emit markdown headings like `Answer:` or `Evidence and Provenance:`
    - Do not emit JSON code fences, metadata labels, raw trace fields, document IDs, fact IDs, or reasoning notes
    - Do not mention the answer mode, explanation policy, or why the answer is short or long
    - Do not invent graph paths, document IDs, policy IDs, timestamps, approvals, or reasoning steps that are not supported by the provided context
    - Treat canonical facts as higher-trust evidence than chunk summaries when both are present
    - If evidence is incomplete or weak, say that clearly in the visible answer instead of overstating confidence

    Answer mode:
    - Requested answer mode: {answer_mode}
    - If mode is `short`, keep the answer compact and only add bullets if they materially help
    - If mode is `long`, keep one clear summary and add concise bullets when extra detail helps
    - Long mode means more detail, not less structure

    Here is the user's question: {query}

    Identity context:
    {user_context}

    Retrieval guidance:
    {retrieval_guidance}

    Here is the relevant context from the documents (keep in mind you get limited context and that's what you should work with):
    {context}

    Respond to the user's question using only the provided context and return JSON only:
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
            if "Chunk Summary: " in item:
                context_parts.append(item.split("Chunk Summary: ", 1)[1].split(", Document ID: ", 1)[0])
            elif "Fact Summary: " in item:
                context_parts.append(item.split("Fact Summary: ", 1)[1].split(", Fact ID: ", 1)[0])
            else:
                context_parts.append(str(item))
        except (IndexError, AttributeError):
            context_parts.append(str(item))
    return context_parts


def _contains_first_person(text: str) -> bool:
    return bool(FIRST_PERSON_PATTERN.search(text))


def _normalize_query_text(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _normalize_query_text(text)
    return any(phrase in normalized for phrase in phrases)


def _looks_like_task_lookup(text: str) -> bool:
    lowered = text.lower()
    if not TASK_LOOKUP_PATTERN.search(text):
        return False
    if any(token in lowered for token in ("promise", "promised", "supposed to", "assigned", "assignment", "working on", "responsible for", "deadline", "due", "by when")):
        return True
    return _contains_first_person(text) and any(token in lowered for token in ("what", "which", "when", "am i", "did i", "do i", "have i"))


def _looks_like_compound_lookup(text: str) -> bool:
    normalized = _normalize_query_text(text)
    if normalized.count("?") > 1:
        return True
    interrogatives = {match.group(1).lower() for match in COMPOUND_LOOKUP_PATTERN.finditer(text or "")}
    if len(interrogatives) >= 2:
        return True
    return "who all" in normalized and any(token in interrogatives for token in {"what", "when", "which"})


def _classify_query(text: str) -> str:
    lowered = text.lower()
    if _looks_like_task_lookup(text):
        return "task_commitment_lookup"
    if _looks_like_compound_lookup(text):
        return "compound_lookup"
    if _contains_first_person(text):
        return "personal_context"
    if any(token in lowered for token in ("weekend", "today", "tomorrow", "schedule", "meeting", "plan")):
        return "schedule_or_timeline"
    if any(token in lowered for token in ("why", "reason", "cause", "delayed")):
        return "explanation"
    if any(token in lowered for token in ("who", "whose", "person", "people")):
        return "person_lookup"
    return "general_search"


def _looks_like_broad_or_explanatory_request(text: str, query_type: Optional[str]) -> bool:
    if query_type == "compound_lookup":
        return True
    if query_type == "explanation":
        return True
    normalized = _normalize_query_text(text)
    if _contains_phrase(normalized, LONG_OVERRIDE_PHRASES):
        return True
    return _contains_phrase(normalized, BROAD_SCOPE_PHRASES)


def _looks_like_direct_lookup_request(text: str, query_type: Optional[str]) -> bool:
    if query_type == "compound_lookup":
        return False
    if query_type in FACT_PRIORITY_QUERY_TYPES:
        return True
    if query_type in {"schedule_or_timeline", "person_lookup"} and DIRECT_LOOKUP_PREFIX.search(text):
        return True
    if DIRECT_LOOKUP_PREFIX.search(text) and not _looks_like_broad_or_explanatory_request(text, query_type):
        return True
    return False


def _select_answer_mode(query: str, retrieval_trace: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
    query_type = (retrieval_trace or {}).get("query_type")
    result_count = int((retrieval_trace or {}).get("result_count") or 0)
    max_hop_count = int((retrieval_trace or {}).get("max_hop_count") or 0)

    # TODO(agentic): Future planner/critic flows can replace this selector, but they must keep the
    # stable answer payload contract and return the same mode/reason_code semantics to the UI.
    if _contains_phrase(query, SHORT_OVERRIDE_PHRASES):
        return ANSWER_MODE_SHORT, REASON_CODE_EXPLICIT_SHORT
    if _contains_phrase(query, LONG_OVERRIDE_PHRASES):
        return ANSWER_MODE_LONG, REASON_CODE_EXPLICIT_LONG
    if query_type == "compound_lookup":
        return ANSWER_MODE_LONG, REASON_CODE_EVIDENCE_COMPLEXITY
    if _looks_like_broad_or_explanatory_request(query, query_type):
        return ANSWER_MODE_LONG, REASON_CODE_BROAD_OR_EXPLANATORY
    if _looks_like_direct_lookup_request(query, query_type):
        return ANSWER_MODE_SHORT, REASON_CODE_DIRECT_LOOKUP
    if result_count > 2 or max_hop_count > 1:
        return ANSWER_MODE_LONG, REASON_CODE_EVIDENCE_COMPLEXITY
    return ANSWER_MODE_SHORT, REASON_CODE_DIRECT_LOOKUP


def _build_answer_explanation(mode: str, reason_code: str) -> str:
    if reason_code == REASON_CODE_EXPLICIT_SHORT:
        return "SAGE kept this answer short because your question explicitly asked for brevity."
    if reason_code == REASON_CODE_EXPLICIT_LONG:
        return "SAGE expanded this answer because your question explicitly asked for more detail."
    if reason_code == REASON_CODE_BROAD_OR_EXPLANATORY:
        return "SAGE used a longer answer because this question asks for explanation, summary, comparison, or broad coverage."
    if reason_code == REASON_CODE_EVIDENCE_COMPLEXITY:
        return "SAGE used a longer answer because the retrieved evidence spans multiple items or hops."
    if reason_code == REASON_CODE_FALLBACK_INVALID_JSON:
        return "SAGE returned a safe short answer because the detailed response could not be formatted reliably."
    if mode == ANSWER_MODE_SHORT:
        return "SAGE kept this answer short because the question looks like a narrow lookup with a direct answer."
    return "SAGE used a longer answer because extra detail helps explain the available evidence."


def _derive_evidence_refs(retrieval_trace: Optional[Dict[str, Any]] = None, limit: int = 3) -> List[str]:
    refs: List[str] = []
    for item in (retrieval_trace or {}).get("evidence") or []:
        ref = None
        if item.get("fact_id"):
            ref = f"fact:{item['fact_id']}"
        elif item.get("chunk_id"):
            ref = f"chunk:{item['chunk_id']}"
        if ref and ref not in refs:
            refs.append(ref)
        if len(refs) >= limit:
            break
    return refs


def _normalize_summary_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _normalize_bullets(values: Any) -> List[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    bullets = [_normalize_summary_text(value) for value in values]
    return [value for value in bullets if value]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp_as_ist(value: str) -> str:
    parsed = _parse_iso_timestamp(value)
    if parsed is None:
        return value
    localized = parsed.astimezone(IST_TIMEZONE)
    return localized.strftime("%Y-%m-%d %I:%M %p IST")


def _convert_iso_timestamps_to_ist_text(text: str) -> str:
    if not text:
        return text
    return ISO_OFFSET_TIMESTAMP_PATTERN.sub(lambda match: _format_timestamp_as_ist(match.group(0)), text)


def _extract_row_recency_timestamp(row: Dict[str, Any]) -> Optional[datetime]:
    document = _serialize_neo4j_entity(row.get("d"))
    fact = _serialize_neo4j_entity(row.get("f"))
    for candidate in (
        document.get("timestamp"),
        fact.get("last_seen_at"),
        fact.get("first_seen_at"),
    ):
        parsed = _parse_iso_timestamp(candidate)
        if parsed is not None:
            return parsed
    return None


def _compute_recency_rank_boost(row: Dict[str, Any]) -> float:
    timestamp = _extract_row_recency_timestamp(row)
    if timestamp is None:
        return 0.0
    age_days = max((_utcnow() - timestamp).total_seconds() / 86400.0, 0.0)
    return RECENCY_BOOST_MAX * math.exp(-age_days / RECENCY_DECAY_DAYS)


def _build_fallback_summary(query: str, documents: List[str], retrieval_trace: Optional[Dict[str, Any]] = None) -> str:
    if not documents or not (retrieval_trace or {}).get("evidence"):
        return (
            "I couldn't find enough reliable information in the available evidence to answer that confidently."
        )
    if _looks_like_broad_or_explanatory_request(query, (retrieval_trace or {}).get("query_type")):
        return "I found relevant evidence, but I could not format a detailed answer reliably. Please use the evidence panel for more detail."
    return "I found relevant evidence, but I could not format the final answer reliably."


def _build_answer_payload(
    *,
    mode: str,
    reason_code: str,
    summary: str,
    bullets: List[str],
    retrieval_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_summary = _convert_iso_timestamps_to_ist_text(_normalize_summary_text(summary))
    if not normalized_summary:
        normalized_summary = "I couldn't produce a readable answer from the available evidence."

    # TODO(agentic): Future planner/generator pipelines can replace this payload producer, but they
    # must keep answer_payload stable so the UI remains decoupled from backend execution changes.
    return {
        "schema_version": ANSWER_PAYLOAD_SCHEMA_VERSION,
        "mode": mode,
        "reason_code": reason_code,
        "summary": normalized_summary,
        "bullets": [_convert_iso_timestamps_to_ist_text(value) for value in _normalize_bullets(bullets)],
        "explanation": _build_answer_explanation(mode, reason_code),
        "evidence_refs": _derive_evidence_refs(retrieval_trace=retrieval_trace),
    }


def _parse_answer_response(raw_response: str, *, mode: str) -> Dict[str, Any]:
    payload = json.loads(raw_response)
    if not isinstance(payload, dict):
        raise ValueError("Structured answer response must be a JSON object")

    summary = _normalize_summary_text(payload.get("summary"))
    if not summary:
        raise ValueError("Structured answer response must include a non-empty summary")

    return {
        "summary": summary,
        "bullets": _normalize_bullets(payload.get("bullets")),
        "thinking": payload.get("thinking") if isinstance(payload.get("thinking"), list) else [],
    }


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


def _build_fact_path_summary(user_scoped: bool) -> Dict[str, Any]:
    nodes = ["Person", "CanonicalFact", "Document"] if user_scoped else ["CanonicalFact", "Document"]
    return {
        "nodes": nodes,
        "path": " -> ".join(nodes),
        "hop_count": max(len(nodes) - 1, 0),
    }


def _result_rank_value(item: Dict[str, Any]) -> float:
    return float(item.get("rank_score", item.get("similarity", 0)) or 0)


def _extract_query_focus_terms(query: str) -> List[str]:
    focus_terms: List[str] = []
    seen: set[str] = set()

    for match in QUERY_EMAIL_PATTERN.finditer(query or ""):
        term = match.group(0).strip().lower()
        if term and term not in seen:
            focus_terms.append(term)
            seen.add(term)

    for match in QUERY_NAME_PATTERN.finditer(query or ""):
        raw = match.group(0).strip()
        if not raw:
            continue
        term = raw.lower()
        if term in QUERY_FOCUS_STOPWORDS:
            continue
        if len(raw.split()) == 1 and raw.lower() in QUERY_FOCUS_STOPWORDS:
            continue
        if term not in seen:
            focus_terms.append(term)
            seen.add(term)

    for match in QUERY_TOKEN_PATTERN.finditer(query or ""):
        term = match.group(0).strip().lower()
        if len(term) < 3 or term in QUERY_FOCUS_STOPWORDS:
            continue
        if term not in seen:
            focus_terms.append(term)
            seen.add(term)

    return focus_terms


def _is_displayable_trace_entity(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False

    lowered = text.lower()
    if lowered in {"currentuser", "unknown", "node", "group", "sage"}:
        return False
    if lowered.startswith("chat message "):
        return False
    if lowered.startswith("chat-msg-") or "-chunk-" in lowered:
        return False
    if lowered.startswith("assignment::") or lowered.startswith("meeting::") or lowered.startswith("reports_to::"):
        return False
    if lowered.startswith("direct:") or lowered.startswith("group") or lowered.startswith("message-attachment-"):
        return False
    if re.fullmatch(r"[0-9]+", text):
        return False
    if re.fullmatch(r"[0-9a-f]{32,64}", lowered):
        return False
    if not re.search(r"[a-zA-Z]", text):
        return False
    return True


def _append_matched_entity(matched_entities: List[str], candidate: Any) -> None:
    text = str(candidate or "").strip()
    if not _is_displayable_trace_entity(text):
        return
    if text not in matched_entities:
        matched_entities.append(text)


def _is_reports_to_lookup(query: str) -> bool:
    normalized = (query or "").lower()
    return "report to" in normalized or "reports to" in normalized


def _collect_row_search_text(row: Dict[str, Any]) -> str:
    fields: List[str] = []
    for value in (
        row.get("chunk_summary"),
        row.get("fact_summary"),
        _serialize_neo4j_entity(row.get("d")).get("subject"),
        _serialize_neo4j_entity(row.get("d")).get("sender"),
        _serialize_neo4j_entity(row.get("d")).get("doc_id"),
        _get_display_name(_serialize_neo4j_entity(row.get("n"))) if row.get("n") else None,
        _serialize_neo4j_entity(row.get("n")).get("id") if row.get("n") else None,
        _serialize_neo4j_entity(row.get("f")).get("canonical_key") if row.get("f") else None,
        _serialize_neo4j_entity(row.get("f")).get("subject_key") if row.get("f") else None,
        _serialize_neo4j_entity(row.get("f")).get("subject_entity_id") if row.get("f") else None,
        _serialize_neo4j_entity(row.get("f")).get("object_key") if row.get("f") else None,
        _serialize_neo4j_entity(row.get("f")).get("object_entity_id") if row.get("f") else None,
        _serialize_neo4j_entity(row.get("f")).get("claim_type") if row.get("f") else None,
    ):
        if value:
            fields.append(str(value))
    return " ".join(fields).lower()


def _focus_match_score(row: Dict[str, Any], focus_terms: List[str]) -> int:
    if not focus_terms:
        return 0
    haystack = _collect_row_search_text(row)
    return sum(1 for term in focus_terms if term in haystack)


def _build_evidence_path(*, scope: str, relationship: str, related_label: Optional[str], doc_id: Optional[str], chunk_id: Optional[str]) -> Dict[str, Any]:
    """Build a traceable, relationship-based path label and hop count.

    This is not a shortest-path computation; it's a concrete chain that mirrors the
    retrieval pattern used by the vector query results.
    """

    parts: List[str] = []
    hops = 0

    if scope == "user":
        parts.append("Person")
        parts.append("-(SENT|RECEIVED_BY)-")
        parts.append(f"Document({doc_id or 'unknown'})")
        hops += 1
    else:
        parts.append(f"Document({doc_id or 'unknown'})")

    parts.append("<-PART_OF-")
    parts.append(f"Chunk({chunk_id or 'unknown'})")
    hops += 1

    if related_label:
        parts.append(f"-{relationship}-")
        parts.append(related_label)
        hops += 1

    return {"path": " ".join(parts), "hop_count": hops}


def _build_path_string(path_nodes: Any, path_relationships: Any) -> Optional[str]:
    try:
        nodes_list = [str(x) for x in (path_nodes or []) if x is not None]
        rels_list = [str(x) for x in (path_relationships or []) if x is not None]
    except Exception:
        return None

    if not nodes_list:
        return None

    # If relationships align with nodes (n rels, n+1 nodes), interleave for readability.
    if rels_list and len(nodes_list) == len(rels_list) + 1:
        parts: List[str] = [nodes_list[0]]
        for rel, node in zip(rels_list, nodes_list[1:]):
            parts.append(f"-{rel}-")
            parts.append(node)
        return " ".join(parts)

    return " -> ".join(nodes_list)


def _merge_ranked_results(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    by_identifier: Dict[str, Dict[str, Any]] = {}
    for row in primary + secondary:
        identifier = str(
            row.get("fact_id")
            or row.get("chunk_id")
            or row.get("canonical_key")
            or row.get("fact_summary")
            or ""
        )
        existing = by_identifier.get(identifier)
        if existing is None or _result_rank_value(row) > _result_rank_value(existing):
            by_identifier[identifier] = row
    merged = list(by_identifier.values())
    merged.sort(key=_result_rank_value, reverse=True)
    return merged[:limit]


def _identity_matches(candidate: Optional[str], user_id: Optional[str]) -> bool:
    return bool(candidate and user_id and str(candidate).strip().lower() == str(user_id).strip().lower())


def _prepare_chunk_result(
    row: Dict[str, Any],
    *,
    focus_terms: Optional[List[str]] = None,
    reports_to_lookup: bool = False,
) -> Dict[str, Any]:
    ranked = dict(row)
    focus_score = _focus_match_score(row, list(focus_terms or []))
    recency_boost = _compute_recency_rank_boost(row)
    rank_score = float(row.get("similarity", 0) or 0)
    if focus_score:
        rank_score += 0.35 * focus_score
    if reports_to_lookup and "reports to" in str(row.get("chunk_summary") or "").lower():
        rank_score += 0.3
    rank_score += recency_boost
    ranked["focus_match_score"] = focus_score
    ranked["recency_boost"] = recency_boost
    ranked["rank_score"] = rank_score
    return ranked


def _prepare_fact_result(
    row: Dict[str, Any],
    *,
    query_type: str,
    user_id: Optional[str],
    personalized_lookup: bool,
    exact_match: bool = False,
    focus_terms: Optional[List[str]] = None,
    reports_to_lookup: bool = False,
) -> Dict[str, Any]:
    ranked = dict(row)
    fact = _serialize_neo4j_entity(row.get("f"))
    similarity = float(row.get("similarity", 0) or 0)
    recency_boost = _compute_recency_rank_boost(row)
    rank_score = similarity
    focus_score = _focus_match_score(row, list(focus_terms or []))

    if fact.get("status") == "current":
        rank_score += 0.05
    if exact_match:
        rank_score += 0.75
    if query_type in FACT_PRIORITY_QUERY_TYPES and fact.get("claim_type") in TASK_LIKE_FACT_TYPES:
        rank_score += 0.35
    if personalized_lookup:
        subject_candidate = fact.get("subject_entity_id") or fact.get("subject_key")
        object_candidate = fact.get("object_entity_id") or fact.get("object_key")
        if _identity_matches(subject_candidate, user_id):
            rank_score += 0.25
        elif _identity_matches(object_candidate, user_id):
            rank_score += 0.1
    if focus_score:
        rank_score += 0.35 * focus_score
    if reports_to_lookup and fact.get("claim_type") == "REPORTS_TO":
        rank_score += 0.4
    rank_score += recency_boost

    ranked["focus_match_score"] = focus_score
    ranked["recency_boost"] = recency_boost
    ranked["rank_score"] = rank_score
    return ranked


def _combine_ranked_results(
    vector_results: List[Dict[str, Any]],
    fact_results: List[Dict[str, Any]],
    *,
    query_type: str,
    focus_terms: Optional[List[str]] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    if query_type in FACT_PRIORITY_QUERY_TYPES and fact_results:
        combined = fact_results[:limit]
        remaining = max(limit - len(combined), 0)
        if remaining:
            combined.extend(vector_results[:remaining])
        return combined[:limit]

    combined = vector_results + fact_results
    if query_type == "person_lookup" and focus_terms:
        focused = [item for item in combined if int(item.get("focus_match_score") or 0) > 0]
        if focused:
            combined = focused
    combined.sort(key=_result_rank_value, reverse=True)
    return combined[:limit]


def _build_response_context(documents: List[str], retrieval_trace: Optional[Dict[str, Any]] = None) -> str:
    evidence = list((retrieval_trace or {}).get("evidence") or [])
    if not evidence:
        return "\n\n".join(_extract_context_parts(documents))

    fact_lines: List[str] = []
    chunk_lines: List[str] = []
    other_lines: List[str] = []

    for item in evidence:
        if item.get("fact_id"):
            fact = item.get("fact") or {}
            document = item.get("document") or {}
            fact_lines.append(
                "- "
                f"Summary: {item.get('fact_summary') or 'No fact summary'} | "
                f"Type: {fact.get('claim_type') or 'unknown'} | "
                f"Status: {fact.get('status') or 'unknown'} | "
                f"Conversation Type: {document.get('conversation_type') or 'unknown'} | "
                f"Subject: {fact.get('subject_entity_id') or fact.get('subject_key') or 'unknown'} | "
                f"Object: {fact.get('object_entity_id') or fact.get('object_key') or 'unknown'} | "
                f"Time: {fact.get('temporal_start') or 'not specified'} ({fact.get('temporal_granularity') or 'unresolved'}) | "
                f"Canonical Key: {fact.get('canonical_key') or item.get('related_node', {}).get('display_name') or 'unknown'} | "
                f"Supporting Document ID: {document.get('doc_id') or 'unknown'} | "
                f"Similarity: {item.get('similarity', 0)}"
            )
            continue

        if item.get("chunk_id"):
            document = item.get("document") or {}
            related_node = item.get("related_node") or {}
            chunk_lines.append(
                "- "
                f"Summary: {item.get('chunk_summary') or 'No summary'} | "
                f"Document ID: {document.get('doc_id') or 'unknown'} | "
                f"Conversation Type: {document.get('conversation_type') or 'unknown'} | "
                f"Subject: {document.get('subject') or 'No Subject'} | "
                f"Sender: {document.get('sender') or 'Unknown'} | "
                f"Relationship: {item.get('relationship') or 'RELATED_TO'} | "
                f"Related Node: {related_node.get('display_name') or 'Unknown'} | "
                f"Similarity: {item.get('similarity', 0)}"
            )
            continue

        other_lines.append(str(item))

    sections: List[str] = []
    if fact_lines:
        sections.append("Canonical facts (highest-trust graph evidence):\n" + "\n".join(fact_lines))
    if chunk_lines:
        sections.append("Supporting document and chunk evidence:\n" + "\n".join(chunk_lines))
    if other_lines:
        sections.append("Additional evidence:\n" + "\n".join(f"- {line}" for line in other_lines))
    return "\n\n".join(sections) if sections else "\n\n".join(_extract_context_parts(documents))


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
    focus_terms = _extract_query_focus_terms(user_input)
    reports_to_lookup = _is_reports_to_lookup(user_input)

    try:
        driver = utils.create_neo4j_driver()
        model = utils.get_cached_embedding_model()
        query_text = user_input if not personalized_lookup else f"{user_input}\nAuthenticated user id: {user_id}"
        query_embedding = np.array(model.encode(query_text), dtype=np.float32)

        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            global_results = [
                _prepare_chunk_result(row, focus_terms=focus_terms, reports_to_lookup=reports_to_lookup)
                for row in session.run(
                GRAPH_VECTOR_QUERY,
                query_embedding=query_embedding.tolist(),
                ).data()
                if row.get("chunk_id") or row.get("chunk_summary")
            ]
            person_results: List[Dict[str, Any]] = []
            if personalized_lookup:
                person_results = [
                    _prepare_chunk_result(row, focus_terms=focus_terms, reports_to_lookup=reports_to_lookup)
                    for row in session.run(
                        PERSON_GRAPH_VECTOR_QUERY,
                        user_id=user_id,
                        query_embedding=query_embedding.tolist(),
                    ).data()
                    if row.get("chunk_id") or row.get("chunk_summary")
                ]
            global_fact_results = [
                _prepare_fact_result(
                        row,
                        query_type=query_type,
                        user_id=user_id,
                        personalized_lookup=personalized_lookup,
                        focus_terms=focus_terms,
                        reports_to_lookup=reports_to_lookup,
                    )
                    for row in session.run(
                        FACT_VECTOR_QUERY,
                        query_embedding=query_embedding.tolist(),
                ).data()
                if row.get("fact_id")
            ]
            person_fact_results: List[Dict[str, Any]] = []
            if personalized_lookup:
                person_fact_results = [
                    _prepare_fact_result(
                        row,
                        query_type=query_type,
                        user_id=user_id,
                        personalized_lookup=personalized_lookup,
                        focus_terms=focus_terms,
                        reports_to_lookup=reports_to_lookup,
                    )
                    for row in session.run(
                        PERSON_FACT_VECTOR_QUERY,
                        user_id=user_id,
                        query_embedding=query_embedding.tolist(),
                    ).data()
                    if row.get("fact_id")
                ]
            exact_task_fact_results: List[Dict[str, Any]] = []
            if personalized_lookup and query_type in FACT_PRIORITY_QUERY_TYPES:
                exact_task_fact_results = [
                    _prepare_fact_result(
                        row,
                        query_type=query_type,
                        user_id=user_id,
                        personalized_lookup=personalized_lookup,
                        exact_match=True,
                        focus_terms=focus_terms,
                        reports_to_lookup=reports_to_lookup,
                    )
                    for row in session.run(
                        PERSON_TASK_FACT_QUERY,
                        user_id=user_id,
                        claim_types=sorted(TASK_LIKE_FACT_TYPES),
                    ).data()
                    if row.get("fact_id")
                ]

        vector_results = _merge_ranked_results(person_results, global_results, limit=5)
        fact_results = _merge_ranked_results(exact_task_fact_results + person_fact_results, global_fact_results, limit=5)
        combined_results = _combine_ranked_results(
            vector_results,
            fact_results,
            query_type=query_type,
            focus_terms=focus_terms,
            limit=5,
        )
        evidence: List[Dict[str, Any]] = []
        documents: List[str] = []
        matched_entities: List[str] = []

        for item in combined_results:
            if item.get("fact_id"):
                supporting_document = _serialize_neo4j_entity(item.get("d"))
                fact = _serialize_neo4j_entity(item.get("f"))
                path_summary = _build_fact_path_summary(personalized_lookup)
                fact_summary = item.get("fact_summary") or fact.get("summary") or "No fact summary"
                fact_id = item.get("fact_id") or fact.get("fact_id")
                canonical_key = fact.get("canonical_key")
                similarity = round(float(item.get("similarity", 0) or 0), 4)
                rank_score = round(_result_rank_value(item), 4)

                for candidate in (
                    fact.get("subject_key"),
                    fact.get("subject_entity_id"),
                    fact.get("object_key"),
                    fact.get("object_entity_id"),
                    supporting_document.get("subject"),
                    supporting_document.get("sender"),
                    item.get("related_node", {}).get("display_name") if isinstance(item.get("related_node"), dict) else None,
                ):
                    _append_matched_entity(matched_entities, candidate)

                evidence_item = {
                    "fact_id": fact_id,
                    "fact_summary": fact_summary,
                    "similarity": similarity,
                    "rank_score": rank_score,
                    "relationship": "CANONICAL_FACT",
                    "retrieval_path": path_summary["path"],
                    "hop_count": path_summary["hop_count"],
                    "document": {
                        "doc_id": supporting_document.get("doc_id"),
                        "subject": supporting_document.get("subject"),
                        "sender": supporting_document.get("sender"),
                        "timestamp": supporting_document.get("timestamp"),
                        "source": supporting_document.get("source"),
                        "conversation_type": supporting_document.get("conversation_type"),
                        "conversation_id": supporting_document.get("conversation_id"),
                        "group_id": supporting_document.get("group_id"),
                    },
                    "related_node": {
                        "label": "CanonicalFact",
                        "display_name": canonical_key or fact_summary,
                        "id": fact_id,
                    },
                    "fact": {
                        "claim_type": fact.get("claim_type"),
                        "status": fact.get("status"),
                        "canonical_key": canonical_key,
                        "subject_key": fact.get("subject_key"),
                        "subject_entity_id": fact.get("subject_entity_id"),
                        "object_key": fact.get("object_key"),
                        "object_entity_id": fact.get("object_entity_id"),
                        "temporal_start": fact.get("temporal_start"),
                        "temporal_end": fact.get("temporal_end"),
                        "temporal_granularity": fact.get("temporal_granularity"),
                        "support_count": fact.get("support_count"),
                        "confidence": fact.get("confidence"),
                    },
                }
                evidence.append(evidence_item)
                documents.append(
                    "Fact Summary: "
                    f"{fact_summary}, "
                    f"Fact ID: {fact_id or 'unknown'}, "
                    f"Canonical Key: {canonical_key or 'unknown'}, "
                    f"Fact Type: {fact.get('claim_type') or 'unknown'}, "
                    f"Conversation Type: {supporting_document.get('conversation_type') or 'unknown'}, "
                    f"Subject: {fact.get('subject_entity_id') or fact.get('subject_key') or 'unknown'}, "
                    f"Object: {fact.get('object_entity_id') or fact.get('object_key') or 'unknown'}, "
                    f"Time: {fact.get('temporal_start') or 'not specified'}, "
                    f"Supporting Document ID: {supporting_document.get('doc_id') or 'unknown'}, "
                    f"Similarity: {similarity}"
                )
                continue

            document = _serialize_neo4j_entity(item.get("d"))
            related_node = _serialize_neo4j_entity(item.get("n"))
            related_label = _get_primary_label(related_node) if related_node else None
            related_name = _get_display_name(related_node) if related_node else None

            sender = document.get("sender")
            subject = document.get("subject")
            doc_id = document.get("doc_id")
            similarity = round(float(item.get("similarity", 0) or 0), 4)
            rank_score = round(_result_rank_value(item), 4)
            relationship = item.get("relationship") or "RELATED_TO"

            hop_count_value = item.get("hop_count")
            computed_hop_count: Optional[int]
            try:
                computed_hop_count = int(hop_count_value) if hop_count_value is not None else None
            except (TypeError, ValueError):
                computed_hop_count = None

            computed_path = _build_path_string(item.get("path_nodes"), item.get("path_relationships"))

            if computed_hop_count is None or not computed_path:
                scope = str(item.get("_scope") or ("user" if personalized_lookup else "global"))
                fallback = _build_evidence_path(
                    scope=scope,
                    relationship=str(relationship),
                    related_label=related_label,
                    doc_id=doc_id,
                    chunk_id=item.get("chunk_id"),
                )
                computed_hop_count = computed_hop_count if computed_hop_count is not None else int(fallback["hop_count"])
                computed_path = computed_path or str(fallback["path"])

            for candidate in (sender, subject, related_name):
                _append_matched_entity(matched_entities, candidate)

            evidence_item = {
                "chunk_id": item.get("chunk_id"),
                "chunk_summary": item.get("chunk_summary", "No summary"),
                "similarity": similarity,
                "rank_score": rank_score,
                "relationship": relationship,
                "retrieval_path": computed_path,
                "hop_count": computed_hop_count,
                "document": {
                    "doc_id": doc_id,
                    "subject": subject,
                    "sender": sender,
                    "timestamp": document.get("timestamp"),
                    "source": document.get("source"),
                    "conversation_type": document.get("conversation_type"),
                    "conversation_id": document.get("conversation_id"),
                    "group_id": document.get("group_id"),
                },
                "related_node": {
                    "label": related_label,
                    "display_name": related_name,
                    "id": related_node.get("id") or related_node.get("_element_id"),
                } if related_node else {},
            }
            evidence.append(evidence_item)

            documents.append(
                "Chunk Summary: "
                f"{evidence_item['chunk_summary']}, "
                f"Document ID: {doc_id or 'unknown'}, "
                f"Conversation Type: {document.get('conversation_type') or 'unknown'}, "
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


def generate_groq_response(
    query: str,
    documents: List[str],
    user_id: Optional[str] = None,
    retrieval_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mode, reason_code = _select_answer_mode(query, retrieval_trace=retrieval_trace)
    if not documents:
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary=(
                "I've searched through my knowledge base, but I don't have any specific information about that topic yet. "
                "Would you like to ask about something else or perhaps upload a document with this information?"
            ),
            bullets=[],
            retrieval_trace=retrieval_trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    try:
        context = _build_response_context(documents, retrieval_trace=retrieval_trace)
        user_context = "No authenticated user context was provided."
        if user_id:
            user_context = f"Authenticated user id: {user_id}."
            if _contains_first_person(query):
                user_context += " Treat first-person references (I/me/my) as this user unless the query says otherwise."
        retrieval_guidance = "Use only the retrieved evidence. If evidence is weak or missing, say so clearly."
        query_type = (retrieval_trace or {}).get("query_type")
        if query_type:
            user_context += f" Query classification: {query_type}."
        if query_type in FACT_PRIORITY_QUERY_TYPES:
            retrieval_guidance = (
                "This is a task or commitment lookup. Prioritize current CanonicalFact evidence over chunk summaries. "
                "Use chunk evidence only to support provenance, add timestamps, or clarify ambiguity."
            )
        elif (retrieval_trace or {}).get("evidence"):
            retrieval_guidance = (
                "Canonical facts are the highest-trust evidence layer. If a canonical fact conflicts with a chunk summary, trust the canonical fact and mention the discrepancy."
            )
        if any(
            (item.get("document") or {}).get("conversation_type") == "group"
            for item in ((retrieval_trace or {}).get("evidence") or [])
        ):
            retrieval_guidance += (
                " If a request or instruction comes from a group conversation without a resolved target person, "
                "say that the target is ambiguous instead of assigning it to one person."
            )
        llm = _create_groq_client(temperature=0.3, require_json=True)
        chain = CHAT_PROMPT | llm | StrOutputParser()
        response = chain.invoke(
            {
                "query": query,
                "context": context,
                "user_context": user_context,
                "retrieval_guidance": retrieval_guidance,
                "answer_mode": mode,
            }
        )
        parsed = _parse_answer_response(response, mode=mode)
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary=parsed["summary"],
            bullets=parsed["bullets"],
            retrieval_trace=retrieval_trace,
        )

        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [str(item) for item in parsed.get("thinking") or [] if str(item).strip()],
        }
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Structured answer parsing failed: %s", exc)
        answer_payload = _build_answer_payload(
            mode=ANSWER_MODE_SHORT,
            reason_code=REASON_CODE_FALLBACK_INVALID_JSON,
            summary=_build_fallback_summary(query, documents, retrieval_trace=retrieval_trace),
            bullets=[],
            retrieval_trace=retrieval_trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }
    except Exception as exc:
        logger.error(f"Groq API error: {exc}")
        answer_payload = _build_answer_payload(
            mode=ANSWER_MODE_SHORT,
            reason_code=REASON_CODE_FALLBACK_INVALID_JSON,
            summary=(
                "I'm sorry, but I seem to be having trouble processing that request right now. "
                "Please try again in a moment."
            ),
            bullets=[],
            retrieval_trace=retrieval_trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
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
                    d.origin_message_id = $origin_message_id,
                    d.linked_message_id = $linked_message_id,
                    d.trace_json = $trace_json,
                    d.graph_sync_status = $graph_sync_status,
                    d.saia_status = coalesce(d.saia_status, null),
                    d.saia_processed_at = coalesce(d.saia_processed_at, null),
                    d.saia_error = coalesce(d.saia_error, null)
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
                origin_message_id=data.get("origin_message_id"),
                linked_message_id=data.get("linked_message_id"),
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
            if data.get("origin_message_id"):
                session.run(
                    """
                    MATCH (m:Message {id: $message_id})
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (m)-[:HAS_EVIDENCE_DOCUMENT]->(d)
                    """,
                    message_id=data["origin_message_id"],
                    doc_id=data["doc_id"],
                )
        return True
    except Exception as exc:
        logger.error(f"Error storing document in Neo4j: {exc}")
        return False
    finally:
        driver.close()
