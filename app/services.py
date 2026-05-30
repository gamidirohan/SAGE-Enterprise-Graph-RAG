"""Business logic and LLM services for SAGE.

Use this file for prompt templates, document extraction, graph retrieval,
chat response generation, and other domain-level application behavior.
"""

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional during test collection
    ChatGroq = None

try:
    import app.query_shape as query_shape
    import app.utils as utils
except ImportError:
    import query_shape
    import utils


logger = logging.getLogger(__name__)

IST_TIMEZONE = ZoneInfo("Asia/Kolkata")
ISO_OFFSET_TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b"
)
RECENCY_BOOST_MAX = 0.18
RECENCY_DECAY_DAYS = 21.0
DEFAULT_SEED_CONTEXT_HOPS = 1
MAX_SEED_CONTEXT_HOPS = 1
DEFAULT_RETRIEVAL_LIMIT = max(1, int(os.getenv("SAGE_RETRIEVAL_LIMIT", "20")))

GRAPH_VECTOR_QUERY = """
    MATCH (c:Chunk)-[:PART_OF]->(d:Document)
    WHERE c.embedding IS NOT NULL
      AND coalesce(d.conversation_type, '') <> 'sage'
      AND NOT coalesce(d.source, '') STARTS WITH 'sage_'
    WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
    WITH c, d,
         gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity,
         c.timestamp AS chunk_ts,
         d.timestamp AS doc_ts
    WITH c, d, similarity,
         coalesce(chunk_ts, doc_ts) AS recency_ts
    WITH c, d, similarity,
         recency_ts,
         CASE
             WHEN recency_ts IS NULL THEN 0.0
             ELSE exp(-1.0 * duration.inDays(datetime(recency_ts), datetime()).days / $recency_decay_days) * $recency_boost_max
         END AS recency_weight
    WITH c, d, similarity + recency_weight AS similarity
    ORDER BY similarity DESC
    LIMIT $candidate_limit
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
    WITH person, c, d, pd, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
    WITH person, c, d, pd,
         gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity,
         c.timestamp AS chunk_ts,
         d.timestamp AS doc_ts
    WITH person, c, d, pd, similarity,
         coalesce(chunk_ts, doc_ts) AS recency_ts
    WITH person, c, d, pd, similarity,
         recency_ts,
         CASE
             WHEN recency_ts IS NULL THEN 0.0
             ELSE exp(-1.0 * duration.inDays(datetime(recency_ts), datetime()).days / $recency_decay_days) * $recency_boost_max
         END AS recency_weight
    WITH person, c, d, pd, similarity + recency_weight AS similarity
    ORDER BY similarity DESC
    LIMIT $candidate_limit
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

GRAPH_VECTOR_QUERY_SHALLOW = """
    MATCH (c:Chunk)-[:PART_OF]->(d:Document)
    WHERE c.embedding IS NOT NULL
      AND coalesce(d.conversation_type, '') <> 'sage'
      AND NOT coalesce(d.source, '') STARTS WITH 'sage_'
    WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
    WITH c, d,
         gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity,
         c.timestamp AS chunk_ts,
         d.timestamp AS doc_ts
    WITH c, d, similarity,
         coalesce(chunk_ts, doc_ts) AS recency_ts
    WITH c, d, similarity,
         recency_ts,
         CASE
             WHEN recency_ts IS NULL THEN 0.0
             ELSE exp(-1.0 * duration.inDays(datetime(recency_ts), datetime()).days / $recency_decay_days) * $recency_boost_max
         END AS recency_weight
    WITH c, d, similarity + recency_weight AS similarity
    ORDER BY similarity DESC
    LIMIT $candidate_limit
    RETURN
        c.chunk_id AS chunk_id,
        c.summary AS chunk_summary,
        d,
        similarity,
        'PART_OF' AS relationship,
        NULL AS n,
        1 AS hop_count,
        [
            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
            coalesce(c.subject, c.title, c.name, c.id, c.doc_id, c.chunk_id, labels(c)[0])
        ] AS path_nodes,
        ['PART_OF'] AS path_relationships
"""

PERSON_GRAPH_VECTOR_QUERY_SHALLOW = """
    MATCH (person:Person {id: $user_id})
    MATCH (person)-[pd:SENT|RECEIVED_BY]-(d:Document)<-[:PART_OF]-(c:Chunk)
    WHERE c.embedding IS NOT NULL
      AND coalesce(d.conversation_type, '') <> 'sage'
      AND NOT coalesce(d.source, '') STARTS WITH 'sage_'
    WITH person, c, d, pd, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
    WITH person, c, d, pd,
         gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity,
         c.timestamp AS chunk_ts,
         d.timestamp AS doc_ts
    WITH person, c, d, pd, similarity,
         coalesce(chunk_ts, doc_ts) AS recency_ts
    WITH person, c, d, pd, similarity,
         recency_ts,
         CASE
             WHEN recency_ts IS NULL THEN 0.0
             ELSE exp(-1.0 * duration.inDays(datetime(recency_ts), datetime()).days / $recency_decay_days) * $recency_boost_max
         END AS recency_weight
    WITH person, c, d, pd, similarity + recency_weight AS similarity
    ORDER BY similarity DESC
    LIMIT $candidate_limit
    RETURN
        c.chunk_id AS chunk_id,
        c.summary AS chunk_summary,
        d,
        similarity,
        'PART_OF' AS relationship,
        NULL AS n,
        2 AS hop_count,
        [
            coalesce(person.subject, person.title, person.name, person.id, person.doc_id, labels(person)[0]),
            coalesce(d.subject, d.title, d.name, d.id, d.doc_id, labels(d)[0]),
            coalesce(c.subject, c.title, c.name, c.id, c.doc_id, c.chunk_id, labels(c)[0])
        ] AS path_nodes,
        [type(pd), 'PART_OF'] AS path_relationships
"""

FACT_VECTOR_QUERY = """
    MATCH (f:CanonicalFact)
    WHERE f.status = 'current' AND f.embedding IS NOT NULL
    WITH f, f.embedding AS fact_embedding, $query_embedding AS query_embedding
    WITH f,
         gds.similarity.cosine(fact_embedding, query_embedding) AS similarity,
         coalesce(f.last_seen_at, f.first_seen_at) AS recency_ts
    WITH f, similarity,
         CASE
             WHEN recency_ts IS NULL THEN 0.0
             ELSE exp(-1.0 * duration.inDays(datetime(recency_ts), datetime()).days / $recency_decay_days) * $recency_boost_max
         END AS recency_weight
    WITH f, similarity + recency_weight AS similarity
    ORDER BY similarity DESC
    LIMIT $candidate_limit
    OPTIONAL MATCH (f)<-[:SUPPORTS]-(claim:Claim)<-[:HAS_CLAIM]-(d:Document)
    WITH f, similarity, collect(DISTINCT d)[0] AS d
    RETURN f.fact_id AS fact_id, f.summary AS fact_summary, f, d, similarity
"""

PERSON_FACT_VECTOR_QUERY = """
    MATCH (p:Person {id: $user_id})-[:HAS_FACT]-(f:CanonicalFact)
    WHERE f.status = 'current' AND f.embedding IS NOT NULL
    WITH f, f.embedding AS fact_embedding, $query_embedding AS query_embedding
    WITH f,
         gds.similarity.cosine(fact_embedding, query_embedding) AS similarity,
         coalesce(f.last_seen_at, f.first_seen_at) AS recency_ts
    WITH f, similarity,
         CASE
             WHEN recency_ts IS NULL THEN 0.0
             ELSE exp(-1.0 * duration.inDays(datetime(recency_ts), datetime()).days / $recency_decay_days) * $recency_boost_max
         END AS recency_weight
    WITH f, similarity + recency_weight AS similarity
    ORDER BY similarity DESC
    LIMIT $candidate_limit
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
    LIMIT $candidate_limit
"""

FIRST_PERSON_PATTERN = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
DISCOURSE_FIRST_PERSON_PREFIX = re.compile(
    r"^\s*(?:can\s+you\s+)?(?:please\s+)?(?:give|show|tell|walk|help)\s+me(?:\s+through)?\b[\s,:-]*",
    re.IGNORECASE,
)
TASK_LOOKUP_PATTERN = re.compile(
    r"\b("
    r"promise|promised|commit|committed|commitment|agreed|supposed to|meant to|"
    r"assigned|assignment|working on|work on|responsible for|deadline|due|by when|what time|"
    r"send|sending|share|sharing|deliver|delivering|submit|submitting|upload|uploading|provide|providing|finish|complete"
    r")\b",
    re.IGNORECASE,
)
TASK_LIKE_FACT_TYPES = {"TASK_ASSIGNMENT", "ASSIGNMENT_STATE", "MEETING_EVENT"}
FACT_PRIORITY_QUERY_TYPES = {"task_commitment_lookup", "schedule_or_timeline"}
ANSWER_PAYLOAD_SCHEMA_VERSION = 1
ANSWER_MODE_SHORT = "short"
ANSWER_MODE_LONG = "long"
REASON_CODE_EXPLICIT_SHORT = "explicit_short"
REASON_CODE_EXPLICIT_LONG = "explicit_long"
REASON_CODE_DIRECT_LOOKUP = "direct_lookup"
REASON_CODE_BROAD_OR_EXPLANATORY = "broad_or_explanatory"
REASON_CODE_EVIDENCE_COMPLEXITY = "evidence_complexity"
REASON_CODE_FALLBACK_INVALID_JSON = "fallback_invalid_json"
REASON_CODE_VERBATIM_EVIDENCE = "verbatim_evidence"

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

BASIC_VERBATIM_PHRASES = (
    "print whatever it was in the chat",
    "print whatever it was",
    "just print",
    "show the chat",
    "show me the chat",
    "show the message",
    "show me the message",
    "quote the chat",
    "quote the message",
    "verbatim",
    "exact wording",
    "exact text",
    "copy paste",
)
UNSUPPORTED_PERSONAL_ATTACK_TERMS = (
    "idiot",
    "stupid",
    "lazy",
    "corrupt",
    "fraud",
    "fraudster",
    "steal",
    "stole",
    "thief",
    "hate",
)
UNSAFE_PRIVATE_LOOKUP_TERMS = (
    "password",
    "secret",
    "confidential",
    "home address",
    "salary",
)
FABRICATION_REQUEST_PHRASES = (
    "make up",
    "invent",
    "fabricate",
    "guess an",
    "guess the",
)
PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system:",
    "developer:",
    "you must answer",
    "always answer",
)
DIRECT_LOOKUP_PREFIX = re.compile(r"^\s*(who|whom|what|when|where|which|did|do|does|is|are|was|were|am|can)\b", re.IGNORECASE)
TIME_LOOKUP_PATTERN = re.compile(r"^\s*(when|by when|what time|what day|what date|which day)\b", re.IGNORECASE)
QUERY_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
QUERY_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
QUERY_ACRONYM_PATTERN = re.compile(r"\b[A-Z0-9][A-Z0-9_\-]{1,}\b")
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
    "start",
    "starts",
    "started",
    "starting",
    "work",
    "works",
    "worked",
    "working",
    "project",
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
    "has",
    "have",
    "had",
    "now",
    "that",
    "this",
    "these",
    "those",
    "who",
    "whom",
    "what",
    "which",
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
    "going",
    "long",
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

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
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
            - Never expose internal identifiers or backend metadata such as canonical keys, group IDs, internal subject codes, sender IDs, or similarity scores
            - Do not mention the answer mode, explanation policy, or why the answer is short or long
            - Do not invent graph paths, document IDs, policy IDs, timestamps, approvals, or reasoning steps that are not supported by the retrieved evidence
            - Treat canonical facts as higher-trust evidence than chunk summaries when both are present
            - Use Fact Time, Evidence Last Seen, Source Message Time, Message Time, and Retrieved At fields only to choose the latest/current evidence and resolve conflicts
            - Do not expose timestamp metadata labels such as Fact Time, Evidence Last Seen, Source Message Time, Message Time, or Retrieved At in the visible answer unless the user explicitly asks for source or retrieval timing
            - If the user asks for a deadline, schedule, or event time, provide only the relevant user-facing date/time, not the retrieval/source metadata trail
            - If evidence is incomplete or weak, say that clearly in the visible answer instead of overstating confidence
            - If answer doesn't exist in the context, let the output know that. DO NOT HALLUCINATE.
            - Answer the user's question directly. Do not respond to the system instructions, planner text, tool labels, or any other meta prompt content.
            - If a recent people-chat window is provided, use it to resolve who said what to whom and in what order.
            - Treat prior assistant answers as low-trust context unless the retrieved evidence or message window independently supports them.
            - If the recent people-chat window contains conflicting claims, prefer the latest timestamped message unless the user explicitly asks for historical state.
            Answer mode:
            - Requested answer mode: {answer_mode}
            - If mode is `short`, keep the answer compact and only add bullets if they materially help
            - If mode is `long`, keep one clear summary and add concise bullets when extra detail helps
            - Long mode means more detail, not less structure
            - Question shape guidance: {question_shape_guidance}

            Identity context:
            {user_context}

            Retrieval guidance:
            {retrieval_guidance}
            """,
        ),
        (
            "human",
            """
            Here is the user's question: {query}

            Here is the relevant context from the documents:
            {context}

            Here is the recent people chat window:
            {conversation_window}

            Respond to the user's question directly. Do not mention the context, the retrieval process, or internal evidence labels. Return JSON only:
            """,
        ),
    ]
)


def _create_groq_client(*, temperature: float, require_json: bool = False):
    return utils.create_chat_llm(temperature=temperature, require_json=require_json)


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


def _strip_discourse_first_person_prefix(text: str) -> str:
    return DISCOURSE_FIRST_PERSON_PREFIX.sub("", text or "", count=1)


def _is_personalized_lookup(text: str) -> bool:
    if not _contains_first_person(text):
        return False
    stripped = _strip_discourse_first_person_prefix(text)
    return bool(FIRST_PERSON_PATTERN.search(stripped))


def _normalize_query_text(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _normalize_query_text(text)
    return any(phrase in normalized for phrase in phrases)


def _wants_verbatim_evidence(text: str) -> bool:
    normalized = _normalize_query_text(text)
    return _contains_phrase(normalized, BASIC_VERBATIM_PHRASES)


def _documents_context_text(documents: List[str], retrieval_trace: Optional[Dict[str, Any]] = None) -> str:
    parts = _extract_context_parts(documents)
    for item in (retrieval_trace or {}).get("evidence") or []:
        document = item.get("document") or {}
        for field in ("subject", "content"):
            value = document.get(field)
            if value:
                parts.append(str(value))
        if item.get("fact_summary"):
            parts.append(str(item.get("fact_summary")))
    return "\n".join(parts)


def _has_canonical_fact_evidence(retrieval_trace: Optional[Dict[str, Any]] = None) -> bool:
    return any(item.get("fact_id") for item in (retrieval_trace or {}).get("evidence") or [])


def _build_guarded_abstention_answer(
    query: str,
    documents: List[str],
    *,
    mode: str,
    reason_code: str,
    retrieval_trace: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    normalized_query = _normalize_query_text(query)
    context = _normalize_query_text(_documents_context_text(documents, retrieval_trace))
    has_fact = _has_canonical_fact_evidence(retrieval_trace)
    trace = dict(retrieval_trace or {})
    evidence = [item for item in (trace.get("evidence") or []) if isinstance(item, dict)]
    query_type = str(trace.get("query_type") or _classify_query(query))
    query_profile = dict(trace.get("query_profile") or query_shape.analyze_query(query))
    focus_terms = _extract_query_focus_terms(query)
    minimum_focus_match = _minimum_focus_match_threshold(
        query,
        query_type=query_type,
        query_profile=query_profile,
        focus_terms=focus_terms,
    )

    reason: Optional[str] = None
    if (
        evidence
        and focus_terms
        and minimum_focus_match > 0
        and max((_evidence_focus_match_score(item, focus_terms) for item in evidence), default=0) < minimum_focus_match
    ):
        reason = "I couldn't find relevant evidence for that lookup."
    elif any(term in normalized_query for term in UNSUPPORTED_PERSONAL_ATTACK_TERMS):
        reason = "There is no evidence in the retrieved information to support that claim."
    elif any(phrase in normalized_query for phrase in FABRICATION_REQUEST_PHRASES):
        reason = "I cannot make up an answer. There is no evidence in the retrieved information that supports the requested value."
    elif any(term in normalized_query for term in UNSAFE_PRIVATE_LOOKUP_TERMS) and not has_fact:
        reason = "There is no evidence in the retrieved information that supports sharing that private or confidential detail."
    elif any(marker in context for marker in PROMPT_INJECTION_MARKERS) and not has_fact:
        reason = "The retrieved text contains instruction-like content, but there is no verified evidence that supports the requested claim."
    elif (
        not has_fact
        and any(
            term in normalized_query
            for term in (
                "procurement",
                "approval",
                "approvals",
                "violat",
                "vendor",
                "vendors",
                "manager",
                "manages",
                "meeting",
                "event",
            )
        )
        and any(marker in context for marker in ("no ", "not ", "not mentioned", "not stored", "not part of", "only covers", "only cover"))
    ):
        reason = "There is no evidence in the retrieved information that supports a specific answer to that question."

    if reason is None:
        return None

    answer_payload = _build_answer_payload(
        mode=mode,
        reason_code=reason_code,
        summary=reason,
        bullets=[],
        retrieval_trace=retrieval_trace,
    )
    return {
        "answer": answer_payload["summary"],
        "answer_payload": answer_payload,
        "thinking": [],
    }


def _extract_verbatim_chat_excerpts(retrieval_trace: Optional[Dict[str, Any]] = None, limit: int = 5) -> List[str]:
    excerpts: List[str] = []
    seen: set[str] = set()
    for item in (retrieval_trace or {}).get("evidence") or []:
        document = item.get("document") or {}
        content = document.get("content")
        if not content:
            continue
        text = " ".join(str(content).split()).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        excerpts.append(text)
        if len(excerpts) >= limit:
            break
    return excerpts


def _looks_like_task_lookup(text: str) -> bool:
    lowered = text.lower()
    if _query_asks_for_assignment_details(text):
        return True
    if not TASK_LOOKUP_PATTERN.search(text):
        return False
    if any(token in lowered for token in ("promise", "promised", "supposed to", "assigned", "assignment", "working on", "responsible for", "deadline", "due", "by when")):
        return True
    return _is_personalized_lookup(text) and any(token in lowered for token in ("what", "which", "when", "am i", "did i", "do i", "have i"))


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
    query_profile = query_shape.analyze_query(text)
    if _looks_like_task_lookup(text):
        return "task_commitment_lookup"
    if _looks_like_compound_lookup(text):
        return "compound_lookup"
    if query_profile.get("requires_broad_coverage"):
        if any(token in lowered for token in ("explain", "why", "reason", "because", "cause", "walk me through")):
            return "explanation"
        return "general_search"
    if _is_personalized_lookup(text):
        return "personal_context"
    if any(token in lowered for token in ("weekend", "today", "tomorrow", "schedule", "meeting", "plan", "review", "when")):
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


def _resolve_seed_context_hops(
    query: str,
    *,
    query_profile: Optional[Dict[str, Any]] = None,
    query_type: Optional[str] = None,
    seed_hops: Optional[int] = None,
) -> int:
    if seed_hops is None:
        seed_hops = query_shape.recommend_graph_depth(
            query,
            query_profile=query_profile,
            query_type=query_type,
        ).get("seed_hops")
    try:
        value = int(seed_hops)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = DEFAULT_SEED_CONTEXT_HOPS
    return max(0, min(value, MAX_SEED_CONTEXT_HOPS))


def _select_answer_mode(query: str, retrieval_trace: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
    query_type = (retrieval_trace or {}).get("query_type")
    result_count = int((retrieval_trace or {}).get("result_count") or 0)
    max_hop_count = int((retrieval_trace or {}).get("max_hop_count") or 0)
    query_profile = dict((retrieval_trace or {}).get("query_profile") or query_shape.analyze_query(query))

    # TODO(agentic): Future planner/critic flows can replace this selector, but they must keep the
    # stable answer payload contract and return the same mode/reason_code semantics to the UI.
    if _contains_phrase(query, SHORT_OVERRIDE_PHRASES):
        return ANSWER_MODE_SHORT, REASON_CODE_EXPLICIT_SHORT
    if _contains_phrase(query, LONG_OVERRIDE_PHRASES):
        return ANSWER_MODE_LONG, REASON_CODE_EXPLICIT_LONG
    if query_type == "compound_lookup":
        return ANSWER_MODE_LONG, REASON_CODE_EVIDENCE_COMPLEXITY
    if query_profile.get("wants_list_format") or query_profile.get("requires_broad_coverage"):
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
    if reason_code == REASON_CODE_VERBATIM_EVIDENCE:
        return "SAGE returned verbatim chat evidence because you asked to see the exact wording."
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
        elif (item.get("document") or {}).get("doc_id"):
            ref = f"doc:{(item.get('document') or {}).get('doc_id')}"
        elif (item.get("related_node") or {}).get("id"):
            ref = f"node:{(item.get('related_node') or {}).get('id')}"
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


def _ensure_sentence(text: str) -> str:
    normalized = _normalize_summary_text(text)
    if not normalized:
        return ""
    if normalized[-1] in ".!?":
        return normalized
    return f"{normalized}."


def _query_asks_for_time(query: str) -> bool:
    normalized = _normalize_query_text(query or "")
    return bool(
        TIME_LOOKUP_PATTERN.search(query or "")
        or normalized.startswith("from when")
        or normalized.startswith("starting when")
        or re.search(r"\bwhen\b.*\bstart(?:s|ed|ing)?\b", normalized)
        or re.search(r"\bstart(?:s|ed|ing)?\b", normalized)
    )


def _query_asks_for_duration(query: str) -> bool:
    normalized = _normalize_query_text(query or "")
    return bool(
        re.search(r"\bhow\s+long\b", normalized)
        or re.search(r"\b(?:duration|until|ending|ends?|finish(?:es|ed)?|through)\b", normalized)
    )


def _query_asks_for_assignment_target(query: str) -> bool:
    normalized = _normalize_query_text(query or "")
    return bool(
        re.search(r"\b(?:which|what)\s+project\b", normalized)
        or re.search(r"\bin\s+which\s+project\b", normalized)
        or re.search(r"\b(?:assigned|assignment|work|working)\b.*\b(?:which|what)\s+project\b", normalized)
    )


def _query_asks_for_assignment_details(query: str) -> bool:
    normalized = _normalize_query_text(query or "")
    assignmentish = bool(re.search(r"\b(?:assigned|assignment|work|working|responsible)\b", normalized))
    return assignmentish and (
        _query_asks_for_duration(query)
        or _query_asks_for_assignment_target(query)
        or _query_asks_for_time(query)
    )


def _query_asks_for_manager(query: str) -> bool:
    normalized = _normalize_query_text(query or "")
    return bool(re.search(r"\b(?:manager|boss|supervisor|lead|reports?\s+to)\b", normalized))


def _is_object_role_lookup(query: str) -> bool:
    normalized = _normalize_query_text(query or "")
    return bool(
        re.search(r"\b(?:manager|owner|lead|supervisor)\s+of\s+\w", normalized)
        or re.search(r"\b(?:managed|owned|led|supervised)\s+by\b", normalized)
    )


def _fact_visible_summary(item: Dict[str, Any]) -> str:
    fact = item.get("fact") or {}
    return _ensure_sentence(
        str(
            fact.get("display_summary")
            or item.get("fact_summary")
            or ""
        )
    )


def _summary_mentions_temporal(summary: str, temporal_start: str) -> bool:
    if not summary or not temporal_start:
        return False
    if temporal_start in summary:
        return True
    return bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b|\b(?:AM|PM|IST|UTC)\b", summary))


def _evidence_temporal_value(item: Dict[str, Any], *, slot: str) -> str:
    fact = item.get("fact") or {}
    structured = str(fact.get(slot) or "").strip()
    if structured:
        return structured

    text_parts: List[str] = []
    document = item.get("document") or {}
    for value in (
        document.get("content"),
        item.get("fact_summary"),
        item.get("chunk_summary"),
    ):
        if value:
            text_parts.append(str(value))
    evidence_text = " ".join(text_parts)
    if not evidence_text:
        return ""

    if slot == "temporal_start":
        match = re.search(
            r"\b(?:starting|starts?|beginning|begins?|from|scheduled\s+for|due|by|on|at)\s+"
            r"(?P<value>today|tomorrow|yesterday|now|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
            r"in\s+\d+\s+(?:day|days|week|weeks)|\d{4}-\d{2}-\d{2}|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
            evidence_text,
            flags=re.IGNORECASE,
        )
        return _normalize_summary_text(match.group("value")) if match else ""

    if slot == "temporal_end":
        match = re.search(
            r"\b(?:until|ending|ends?|through)\s+"
            r"(?P<value>today|tomorrow|yesterday|now|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
            r"in\s+\d+\s+(?:day|days|week|weeks)|\d{4}-\d{2}-\d{2})\b",
            evidence_text,
            flags=re.IGNORECASE,
        )
        if match:
            return _normalize_summary_text(match.group("value"))
        duration_match = re.search(
            r"\b(?P<value>for\s+\d+\s+(?:day|days|week|weeks|month|months))\b",
            evidence_text,
            flags=re.IGNORECASE,
        )
        return _normalize_summary_text(duration_match.group("value")) if duration_match else ""
    return ""


def _evidence_duration_value(item: Dict[str, Any]) -> str:
    text_parts: List[str] = []
    document = item.get("document") or {}
    for value in (
        document.get("content"),
        item.get("fact_summary"),
        item.get("chunk_summary"),
        (item.get("fact") or {}).get("display_summary"),
    ):
        if value:
            text_parts.append(str(value))
    match = re.search(
        r"\bfor\s+(?P<value>\d+\s+(?:day|days|week|weeks|month|months|year|years))\b",
        " ".join(text_parts),
        flags=re.IGNORECASE,
    )
    return _normalize_summary_text(match.group("value")) if match else ""


def _format_assignment_start_phrase(value: str) -> str:
    if not value:
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}", value) or re.match(r"^\d{1,2}(?::\d{2})?\s*(?:am|pm)$", value, re.IGNORECASE):
        return f"on {value}"
    return f"starting {value}"


def _assignment_subject_object_labels(item: Dict[str, Any]) -> tuple[str, str]:
    fact = item.get("fact") or {}
    visible_summary = _fact_visible_summary(item)
    subject = _context_entity_label(fact.get("subject_display"))
    obj = _context_entity_label(fact.get("object_display"))
    if not subject or not obj:
        match = re.search(
            r"(?P<subject>.+?)\s+is\s+(?:assigned\s+to|working\s+on)\s+(?P<object>.+?)(?:\s+starting\b|\s+until\b|\.|$)",
            visible_summary,
            re.IGNORECASE,
        )
        if match:
            subject = subject or _normalize_summary_text(match.group("subject"))
            obj = obj or _normalize_summary_text(match.group("object"))

    subject = subject or _context_entity_label(fact.get("subject_entity_id") or fact.get("subject_key")) or "The person"
    obj = obj or _context_entity_label(fact.get("object_entity_id") or fact.get("object_key")) or "that assignment"
    return subject, obj


def _assignment_time_visible_summary(item: Dict[str, Any], temporal_start: str) -> str:
    if not temporal_start:
        return ""
    fact = item.get("fact") or {}
    if str(fact.get("claim_type") or "") != "ASSIGNMENT_STATE":
        return ""

    subject, obj = _assignment_subject_object_labels(item)
    return _ensure_sentence(f"{subject} starts working on {obj} {_format_assignment_start_phrase(temporal_start)}")


def _assignment_requested_slots_answer(
    query: str,
    item: Dict[str, Any],
    *,
    mode: str,
    reason_code: str,
    retrieval_trace: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    fact = item.get("fact") or {}
    if str(fact.get("claim_type") or "") != "ASSIGNMENT_STATE":
        return None
    if _query_asks_for_manager(query):
        return None

    asks_duration = _query_asks_for_duration(query)
    asks_target = _query_asks_for_assignment_target(query)
    asks_start = _query_asks_for_time(query)
    if not (asks_duration or asks_target or asks_start):
        return None

    subject, obj = _assignment_subject_object_labels(item)
    temporal_start = _evidence_temporal_value(item, slot="temporal_start")
    temporal_end = _evidence_temporal_value(item, slot="temporal_end")
    duration = _evidence_duration_value(item)

    if asks_duration:
        if duration:
            summary = f"{subject} will work on {obj} for {duration}."
        elif temporal_start and temporal_end:
            summary = f"{subject} will work on {obj} from {temporal_start} until {temporal_end}."
        elif temporal_end:
            summary = f"{subject} will work on {obj} until {temporal_end}."
        else:
            summary = f"{subject} is assigned to {obj}, but I could not find the assignment duration."
    elif asks_start:
        summary = _assignment_time_visible_summary(item, temporal_start) or (
            f"{subject} is assigned to {obj}, but I could not find when the assignment starts."
        )
    elif asks_target:
        summary = f"{subject} is assigned to {obj}."
    else:
        return None

    answer_payload = _build_answer_payload(
        mode=mode,
        reason_code=reason_code,
        summary=summary,
        bullets=[],
        retrieval_trace=retrieval_trace,
    )
    return {
        "answer": answer_payload["summary"],
        "answer_payload": answer_payload,
        "thinking": [],
    }


def _fact_conflict_summary(query_type: str, claim_type: str) -> str:
    if claim_type == "REPORTS_TO" or query_type == "person_lookup":
        return "I found conflicting current reporting relationships for that lookup, so I can't collapse them to one safely."
    if query_type == "schedule_or_timeline":
        return "I found conflicting current schedule evidence for that lookup, so I can't collapse it to one safely."
    return "I found conflicting current evidence for that lookup, so I can't collapse it to one safely."


def _object_role_evidence_text(item: Dict[str, Any]) -> str:
    document = item.get("document") or {}
    parts = [
        item.get("fact_summary"),
        item.get("chunk_summary"),
        (item.get("fact") or {}).get("display_summary"),
        document.get("content"),
        document.get("subject"),
    ]
    return _normalize_summary_text(" ".join(str(part) for part in parts if part))


def _extract_object_role_fact(text: str) -> Optional[Dict[str, str]]:
    normalized = _normalize_summary_text(text)
    if not normalized:
        return None

    active_pattern = re.compile(
        r"\b(?P<person>[A-Z][A-Za-z0-9_\-]*(?:\s+[A-Z][A-Za-z0-9_\-]*){0,4})\s+"
        r"(?:is|was|has\s+been|served\s+as)\s+(?:the\s+)?"
        r"(?P<role>manager|owner|lead|supervisor)\s+of\s+"
        r"(?P<object>[A-Z][A-Za-z0-9_\-]*(?:\s+(?!in\b|during\b|for\b)[A-Z][A-Za-z0-9_\-]*){0,5})"
        r"(?:\s+(?:in|during|for)\s+(?P<time>\d{4}|Q[1-4]\s+\d{4}))?",
        re.IGNORECASE,
    )
    passive_pattern = re.compile(
        r"\b(?P<object>[A-Z][A-Za-z0-9_\-]*(?:\s+(?!in\b|during\b|for\b)[A-Z][A-Za-z0-9_\-]*){0,5})\s+"
        r"(?:is|was|has\s+been)\s+(?P<verb>managed|owned|led|supervised)\s+by\s+"
        r"(?P<person>[A-Z][A-Za-z0-9_\-]*(?:\s+[A-Z][A-Za-z0-9_\-]*){0,4})"
        r"(?:\s+(?:in|during|for)\s+(?P<time>\d{4}|Q[1-4]\s+\d{4}))?",
        re.IGNORECASE,
    )

    match = active_pattern.search(normalized)
    if match:
        return {
            "person": _normalize_summary_text(match.group("person")),
            "role": _normalize_summary_text(match.group("role")).lower(),
            "object": _normalize_summary_text(match.group("object").rstrip(".?!")),
            "time": _normalize_summary_text(match.group("time") or ""),
        }

    match = passive_pattern.search(normalized)
    if match:
        verb_to_role = {
            "managed": "manager",
            "owned": "owner",
            "led": "lead",
            "supervised": "supervisor",
        }
        return {
            "person": _normalize_summary_text(match.group("person").rstrip(".?!")),
            "role": verb_to_role.get(_normalize_summary_text(match.group("verb")).lower(), "manager"),
            "object": _normalize_summary_text(match.group("object")),
            "time": _normalize_summary_text(match.group("time") or ""),
        }
    return None


def _object_role_matches_query(role_fact: Dict[str, str], query: str) -> bool:
    normalized_query = _normalize_query_text(query)
    role = role_fact.get("role") or ""
    obj = _normalize_query_text(role_fact.get("object") or "")
    time_value = _normalize_query_text(role_fact.get("time") or "")
    if role and role not in normalized_query:
        return False
    object_terms = [token for token in obj.split() if token not in QUERY_FOCUS_STOPWORDS]
    if object_terms and not all(token in normalized_query for token in object_terms):
        return False
    if time_value and time_value not in normalized_query:
        return False
    return True


def _build_object_role_lookup_answer(
    query: str,
    evidence: List[Dict[str, Any]],
    *,
    mode: str,
    reason_code: str,
    retrieval_trace: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not _is_object_role_lookup(query):
        return None

    for item in evidence:
        role_fact = _extract_object_role_fact(_object_role_evidence_text(item))
        if not role_fact or not _object_role_matches_query(role_fact, query):
            continue
        time_phrase = f" in {role_fact['time']}" if role_fact.get("time") else ""
        summary = (
            f"{role_fact['person']} was the {role_fact['role']} of "
            f"{role_fact['object']}{time_phrase}."
        )
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary=summary,
            bullets=[],
            retrieval_trace=retrieval_trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    return None


def _reports_to_labels(item: Dict[str, Any]) -> tuple[str, str]:
    fact = item.get("fact") or {}
    summary = str(item.get("fact_summary") or fact.get("display_summary") or fact.get("summary") or "")
    match = re.search(r"(.+?)\s+reports\s+to\s+(.+?)(?:\.|$)", summary, re.IGNORECASE)
    if match:
        return _normalize_summary_text(match.group(1)), _normalize_summary_text(match.group(2))
    return (
        _context_entity_label(fact.get("subject_display") or fact.get("subject_entity_id") or fact.get("subject_key")),
        _context_entity_label(fact.get("object_display") or fact.get("object_entity_id") or fact.get("object_key")),
    )


def _build_reports_to_chain_answer(
    query: str,
    fact_items: List[Dict[str, Any]],
    *,
    mode: str,
    reason_code: str,
    retrieval_trace: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    query_profile = dict((retrieval_trace or {}).get("query_profile") or {})
    if not query_profile.get("requires_multi_hop"):
        return None

    reports_to_items = [
        item
        for item in fact_items
        if str(((item.get("fact") or {}).get("claim_type")) or "") == "REPORTS_TO"
        and str(((item.get("fact") or {}).get("status")) or "current") == "current"
    ]
    if len(reports_to_items) < 2:
        return None

    manager_by_subject: Dict[str, str] = {}
    labels_by_entity: Dict[str, str] = {}
    for item in reports_to_items:
        fact = item.get("fact") or {}
        subject = str(fact.get("subject_entity_id") or fact.get("subject_key") or "").strip()
        manager = str(fact.get("object_entity_id") or fact.get("object_key") or "").strip()
        if not subject or not manager:
            continue
        subject_label, manager_label = _reports_to_labels(item)
        manager_by_subject[subject] = manager
        labels_by_entity[subject] = subject_label or subject
        labels_by_entity[manager] = manager_label or manager

    if len(manager_by_subject) < 2:
        return None

    lowered_query = query.lower()
    start_subject = None
    for subject, label in labels_by_entity.items():
        if subject in manager_by_subject and label and label.lower() in lowered_query:
            start_subject = subject
            break
    if start_subject is None:
        object_ids = set(manager_by_subject.values())
        start_subject = next((subject for subject in manager_by_subject if subject not in object_ids), None)
    if start_subject is None:
        return None

    first_manager = manager_by_subject.get(start_subject)
    second_manager = manager_by_subject.get(first_manager or "")
    if not first_manager or not second_manager:
        return None

    summary = (
        f"{labels_by_entity.get(start_subject, start_subject)} reports to "
        f"{labels_by_entity.get(first_manager, first_manager)}, who reports to "
        f"{labels_by_entity.get(second_manager, second_manager)}."
    )
    answer_payload = _build_answer_payload(
        mode=mode,
        reason_code=reason_code,
        summary=summary,
        bullets=[],
        retrieval_trace=retrieval_trace,
    )
    return {
        "answer": answer_payload["summary"],
        "answer_payload": answer_payload,
        "thinking": [],
    }


def _build_fact_backed_answer(
    query: str,
    *,
    mode: str,
    reason_code: str,
    retrieval_trace: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    trace = dict(retrieval_trace or {})
    query_type = str(trace.get("query_type") or "").strip()
    query_profile = dict(trace.get("query_profile") or {})
    evidence = [dict(item) for item in (trace.get("evidence") or []) if isinstance(item, dict)]
    fact_items = [item for item in evidence if item.get("fact_id")]
    reports_to_items = [
        item
        for item in fact_items
        if str(((item.get("fact") or {}).get("claim_type")) or "") == "REPORTS_TO"
        and str(((item.get("fact") or {}).get("status")) or "current") == "current"
    ]
    assignment_items = [
        item
        for item in fact_items
        if str(((item.get("fact") or {}).get("claim_type")) or "") == "ASSIGNMENT_STATE"
        and str(((item.get("fact") or {}).get("status")) or "current") == "current"
    ]
    assignment_slot_lookup = bool(assignment_items and _query_asks_for_assignment_details(query) and not _query_asks_for_manager(query))
    direct_fact_lookup = bool(
        _looks_like_direct_lookup_request(query, query_type)
        and not query_profile.get("requires_broad_coverage")
        and not query_profile.get("expects_multiple_items")
    )
    chain_fact_lookup = bool(query_profile.get("requires_multi_hop"))

    if (
        query_type not in FACT_PRIORITY_QUERY_TYPES
        and query_type != "person_lookup"
        and not direct_fact_lookup
        and not chain_fact_lookup
        and not assignment_slot_lookup
        and not _is_object_role_lookup(query)
    ):
        return None
    object_role_answer = _build_object_role_lookup_answer(
        query,
        evidence,
        mode=mode,
        reason_code=reason_code,
        retrieval_trace=trace,
    )
    if object_role_answer is not None:
        return object_role_answer
    if not fact_items:
        return None

    fact_conflict = dict(trace.get("fact_lookup_conflict") or {})
    ambiguity = dict(trace.get("task_lookup_ambiguity") or {})
    if fact_conflict.get("ambiguous"):
        summaries = [_fact_visible_summary(item) for item in fact_items[:3]]
        summaries = [summary for summary in summaries if summary]
        if not summaries:
            summaries = ["I found multiple current facts that conflict for this lookup."]
        claim_type = str(
            fact_conflict.get("claim_type")
            or ((fact_items[0].get("fact") or {}).get("claim_type"))
            or ""
        )
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary=_fact_conflict_summary(query_type, claim_type),
            bullets=summaries,
            retrieval_trace=trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    if query_type == "person_lookup" and _is_reports_to_lookup(query):
        summary = ""
        if reports_to_items:
            summary = _fact_visible_summary(reports_to_items[0])
        if not summary:
            summary = "I couldn't find current manager or reporting evidence for that lookup."
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary=summary,
            bullets=[],
            retrieval_trace=trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    if query_type == "task_commitment_lookup" and ambiguity.get("ambiguous"):
        summaries = [_fact_visible_summary(item) for item in fact_items[:3]]
        summaries = [summary for summary in summaries if summary]
        if not summaries:
            summaries = ["I found multiple current commitments that match that request."]
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary="I found multiple current commitments that match that request, so I can't collapse them to one safely.",
            bullets=summaries,
            retrieval_trace=trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    top_fact = fact_items[0]
    visible_summary = _fact_visible_summary(top_fact)
    temporal_start = _evidence_temporal_value(top_fact, slot="temporal_start")

    if assignment_slot_lookup:
        assignment_answer = _assignment_requested_slots_answer(
            query,
            assignment_items[0],
            mode=mode,
            reason_code=reason_code,
            retrieval_trace=trace,
        )
        if assignment_answer is not None:
            return assignment_answer

    if query_type == "task_commitment_lookup" and _query_asks_for_time(query):
        if temporal_start and not _summary_mentions_temporal(visible_summary, temporal_start):
            time_summary = _assignment_time_visible_summary(top_fact, temporal_start)
            if time_summary:
                summary = time_summary
            else:
                summary = f"The current recorded time for that commitment is {temporal_start}."
        else:
            summary = visible_summary
        summary = (
            summary
            or (f"The current recorded time for that commitment is {temporal_start}." if temporal_start else "")
            or "I found a current matching commitment, but it does not include a scheduled time."
        )
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary=summary,
            bullets=[],
            retrieval_trace=trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    if query_type == "schedule_or_timeline" and _query_asks_for_time(query):
        summary = (
            f"The scheduled time is {temporal_start}."
            if temporal_start
            else visible_summary
            or "I found a matching schedule fact, but it does not include a time."
        )
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary=summary,
            bullets=[],
            retrieval_trace=trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    chain_answer = _build_reports_to_chain_answer(
        query,
        fact_items,
        mode=mode,
        reason_code=reason_code,
        retrieval_trace=trace,
    )
    if chain_answer is not None:
        return chain_answer

    if (
        query_type == "person_lookup"
        and not query_profile.get("wants_list_format")
        and str(((top_fact.get("fact") or {}).get("claim_type")) or "") == "REPORTS_TO"
    ):
        summary = visible_summary or "I found a current reporting relationship, but I could not render it clearly."
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary=summary,
            bullets=[],
            retrieval_trace=trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    if direct_fact_lookup and visible_summary:
        answer_payload = _build_answer_payload(
            mode=mode,
            reason_code=reason_code,
            summary=visible_summary,
            bullets=[],
            retrieval_trace=trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _retrieval_timestamp() -> str:
    return _utcnow().isoformat().replace("+00:00", "Z")


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


def _format_context_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _format_timestamp_as_ist(text)


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

    for match in QUERY_ACRONYM_PATTERN.finditer(query or ""):
        term = match.group(0).strip().lower()
        if len(term) < 2 or term in QUERY_FOCUS_STOPWORDS:
            continue
        if term not in seen:
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


def _collect_evidence_search_text(item: Dict[str, Any]) -> str:
    document = item.get("document") or {}
    related_node = item.get("related_node") or {}
    fact = item.get("fact") or {}
    fields: List[Any] = [
        item.get("fact_summary"),
        item.get("chunk_summary"),
        document.get("content"),
        document.get("subject"),
        document.get("sender"),
        related_node.get("display_name"),
        related_node.get("id"),
        fact.get("canonical_key"),
        fact.get("claim_type"),
        fact.get("value_text"),
        fact.get("subject_key"),
        fact.get("subject_entity_id"),
        fact.get("subject_display"),
        fact.get("object_key"),
        fact.get("object_entity_id"),
        fact.get("object_display"),
        fact.get("display_summary"),
    ]
    return " ".join(str(value) for value in fields if value).lower()


def _evidence_focus_match_score(item: Dict[str, Any], focus_terms: List[str]) -> int:
    if not focus_terms:
        return 0
    haystack = _collect_evidence_search_text(item)
    return sum(1 for term in focus_terms if term.lower() in haystack)


def _requires_direct_focus_match(
    query: str,
    *,
    query_type: Optional[str],
    query_profile: Optional[Dict[str, Any]] = None,
) -> bool:
    profile = dict(query_profile or query_shape.analyze_query(query))
    if profile.get("wants_list_format") or profile.get("requires_broad_coverage"):
        return False
    if query_type not in {"general_search", "personal_context"}:
        return False
    return _looks_like_direct_lookup_request(query, query_type)


def _minimum_focus_match_threshold(
    query: str,
    *,
    query_type: Optional[str],
    query_profile: Optional[Dict[str, Any]] = None,
    focus_terms: Optional[List[str]] = None,
) -> int:
    profile = dict(query_profile or query_shape.analyze_query(query))
    terms = list(focus_terms or [])
    effective_query_type = str(query_type or "").strip()

    if not terms or profile.get("wants_list_format") or profile.get("requires_broad_coverage"):
        return 0
    if effective_query_type in {"task_commitment_lookup", "schedule_or_timeline"}:
        return 2 if len(terms) >= 2 else 0
    if _requires_direct_focus_match(query, query_type=effective_query_type, query_profile=profile):
        return 1
    return 0


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
    if re.fullmatch(r"g-[a-z0-9-]+", lowered):
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


def _context_entity_label(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not _is_displayable_trace_entity(text):
        return None
    return text


def _join_context_parts(*parts: Optional[str]) -> str:
    values = [_normalize_summary_text(part) for part in parts if _normalize_summary_text(part)]
    return " | ".join(values)


def _is_reports_to_lookup(query: str) -> bool:
    normalized = (query or "").lower()
    if _is_object_role_lookup(query):
        return False
    return bool(
        re.search(r"\breports?\s+to\b", normalized)
        or re.search(r"\b(?:my|your|their|his|her|[A-Z][A-Za-z0-9_\-]+(?:'s|’s))\s+(?:manager|boss|supervisor|lead)\b", query or "", re.IGNORECASE)
    )


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
    if reports_to_lookup and _is_reports_to_lookup(str(row.get("chunk_summary") or "")):
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
    claim_type = str(fact.get("claim_type") or "")
    similarity = float(row.get("similarity", 0) or 0)
    recency_boost = _compute_recency_rank_boost(row)
    rank_score = similarity
    focus_score = _focus_match_score(row, list(focus_terms or []))
    fact_priority = bool(query_type in FACT_PRIORITY_QUERY_TYPES and claim_type in TASK_LIKE_FACT_TYPES)

    if fact.get("status") == "current":
        rank_score += 0.05
    if exact_match:
        rank_score += 0.75
    if fact_priority:
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
    if reports_to_lookup and claim_type == "REPORTS_TO":
        rank_score += 0.4
        if query_type == "person_lookup":
            fact_priority = True
            rank_score += 0.25
    rank_score += recency_boost

    ranked["exact_match"] = bool(exact_match)
    ranked["fact_priority"] = fact_priority
    ranked["focus_match_score"] = focus_score
    ranked["recency_boost"] = recency_boost
    ranked["rank_score"] = rank_score
    return ranked


def _fact_result_matches_user(item: Dict[str, Any], user_id: Optional[str]) -> bool:
    if not user_id:
        return False
    fact = _serialize_neo4j_entity(item.get("f")) if item.get("f") is not None else dict(item.get("fact") or {})
    subject_candidate = fact.get("subject_entity_id") or fact.get("subject_key")
    object_candidate = fact.get("object_entity_id") or fact.get("object_key")
    return _identity_matches(subject_candidate, user_id) or _identity_matches(object_candidate, user_id)


def _combine_ranked_results(
    vector_results: List[Dict[str, Any]],
    fact_results: List[Dict[str, Any]],
    *,
    query_type: str,
    focus_terms: Optional[List[str]] = None,
    reports_to_lookup: bool = False,
    minimum_focus_match: int = 0,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    if query_type in FACT_PRIORITY_QUERY_TYPES and fact_results:
        if focus_terms:
            focused_facts = [item for item in fact_results if int(item.get("focus_match_score") or 0) > 0]
            focused_vectors = [item for item in vector_results if int(item.get("focus_match_score") or 0) > 0]
            if focused_facts:
                fact_results = focused_facts
                vector_results = focused_vectors or vector_results
            elif focused_vectors:
                return focused_vectors[:limit]
        combined = fact_results[:limit]
        remaining = max(limit - len(combined), 0)
        if remaining:
            combined.extend(vector_results[:remaining])
        return combined[:limit]
    if (
        query_type == "person_lookup"
        and reports_to_lookup
        and any(str((item.get("f") or item.get("fact") or {}).get("claim_type") or "") == "REPORTS_TO" for item in fact_results)
    ):
        combined = [
            item
            for item in fact_results
            if str((item.get("f") or item.get("fact") or {}).get("claim_type") or "") == "REPORTS_TO"
        ][:limit]
        remaining = max(limit - len(combined), 0)
        if remaining:
            combined.extend(vector_results[:remaining])
        return combined[:limit]

    combined = vector_results + fact_results
    if focus_terms and minimum_focus_match > 0:
        combined = [item for item in combined if int(item.get("focus_match_score") or 0) >= minimum_focus_match]
        if not combined:
            return []
        if len(focus_terms) >= 2:
            strongest_focus = max(int(item.get("focus_match_score") or 0) for item in combined)
            combined = [item for item in combined if int(item.get("focus_match_score") or 0) == strongest_focus]
        combined.sort(key=_result_rank_value, reverse=True)
        return combined[:limit]
    if query_type == "person_lookup" and focus_terms:
        focused = [item for item in combined if int(item.get("focus_match_score") or 0) > 0]
        if focused:
            combined = focused
    elif query_type in {"compound_lookup", "general_search", "explanation", "schedule_or_timeline"} and focus_terms:
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
            relationship_semantics = None
            if fact.get("claim_type") == "REPORTS_TO":
                relationship_semantics = "Relationship Semantics: subject/person reports to object/manager"
            subject_label = _context_entity_label(fact.get("subject_display") or fact.get("subject_entity_id") or fact.get("subject_key"))
            object_label = _context_entity_label(fact.get("object_display") or fact.get("object_entity_id") or fact.get("object_key"))
            conversation_type = _normalize_summary_text(document.get("conversation_type") or "")
            time_text = (
                f"Fact Time: {fact.get('temporal_start')} ({fact.get('temporal_granularity') or 'unresolved'})"
                if fact.get("temporal_start")
                else None
            )
            last_seen_text = _format_context_timestamp(fact.get("last_seen_at") or fact.get("first_seen_at"))
            source_message_time = _format_context_timestamp(document.get("timestamp"))
            retrieved_at = _format_context_timestamp(item.get("retrieved_at") or (retrieval_trace or {}).get("retrieved_at"))
            fact_lines.append(
                "- "
                + _join_context_parts(
                    f"Summary: {item.get('fact_summary') or 'No fact summary'}",
                    f"Type: {fact.get('claim_type') or 'unknown'}",
                    f"Status: {fact.get('status') or 'unknown'}",
                    f"Conversation Type: {conversation_type}" if conversation_type else None,
                    f"Subject: {subject_label}" if subject_label else None,
                    f"Object: {object_label}" if object_label else None,
                    time_text,
                    f"Evidence Last Seen: {last_seen_text}" if last_seen_text else None,
                    f"Source Message Time: {source_message_time}" if source_message_time else None,
                    f"Retrieved At: {retrieved_at}" if retrieved_at else None,
                    relationship_semantics,
                )
            )
            continue

        if item.get("chunk_id"):
            document = item.get("document") or {}
            related_node = item.get("related_node") or {}
            conversation_type = _normalize_summary_text(document.get("conversation_type") or "")
            subject_label = _context_entity_label(document.get("subject"))
            sender_label = _context_entity_label(document.get("sender"))
            related_label = _context_entity_label(related_node.get("display_name"))
            message_time = _format_context_timestamp(document.get("timestamp"))
            retrieved_at = _format_context_timestamp(item.get("retrieved_at") or (retrieval_trace or {}).get("retrieved_at"))
            chunk_lines.append(
                "- "
                + _join_context_parts(
                    f"Summary: {item.get('chunk_summary') or 'No summary'}",
                    f"Conversation Type: {conversation_type}" if conversation_type else None,
                    f"Subject: {subject_label}" if subject_label else None,
                    f"Sender: {sender_label}" if sender_label else None,
                    f"Message Time: {message_time}" if message_time else None,
                    f"Retrieved At: {retrieved_at}" if retrieved_at else None,
                    f"Related Node: {related_label}" if related_label else None,
                )
            )
            continue

        other_lines.append(str(item))

    sections: List[str] = []
    if fact_lines:
        sections.append("Canonical facts:\n" + "\n".join(fact_lines))
    if chunk_lines:
        sections.append("Supporting message and document evidence:\n" + "\n".join(chunk_lines))
    if other_lines:
        sections.append("Additional evidence:\n" + "\n".join(f"- {line}" for line in other_lines))
    return "\n\n".join(sections) if sections else "\n\n".join(_extract_context_parts(documents))


def extract_structured_data(document_text: str, doc_id: str) -> Dict[str, Any]:
    if not utils.chat_llm_configured():
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


def query_graph_with_trace(
    user_input: str,
    user_id: Optional[str] = None,
    *,
    seed_hops: Optional[int] = None,
) -> Dict[str, Any]:
    driver = None
    personalized_lookup = bool(user_id and _is_personalized_lookup(user_input))
    query_type = _classify_query(user_input)
    query_profile = query_shape.analyze_query(user_input)
    resolved_seed_hops = _resolve_seed_context_hops(
        user_input,
        query_profile=query_profile,
        query_type=query_type,
        seed_hops=seed_hops,
    )
    graph_depth = query_shape.recommend_graph_depth(
        user_input,
        query_profile=query_profile,
        query_type=query_type,
    )
    graph_depth["seed_hops"] = resolved_seed_hops
    focus_terms = _extract_query_focus_terms(user_input)
    reports_to_lookup = _is_reports_to_lookup(user_input)
    retrieved_at = _retrieval_timestamp()

    try:
        driver = utils.create_neo4j_driver()
        model = utils.get_cached_embedding_model()
        query_text = user_input if not personalized_lookup else f"{user_input}\nAuthenticated user id: {user_id}"
        query_embedding = np.array(model.encode(query_text), dtype=np.float32)

        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            global_results = [
                _prepare_chunk_result(row, focus_terms=focus_terms, reports_to_lookup=reports_to_lookup)
                for row in session.run(
                    GRAPH_VECTOR_QUERY if resolved_seed_hops >= DEFAULT_SEED_CONTEXT_HOPS else GRAPH_VECTOR_QUERY_SHALLOW,
                    query_embedding=query_embedding.tolist(),
                    recency_decay_days=RECENCY_DECAY_DAYS,
                    recency_boost_max=RECENCY_BOOST_MAX,
                    candidate_limit=DEFAULT_RETRIEVAL_LIMIT,
                ).data()
                if row.get("chunk_id") or row.get("chunk_summary")
            ]
            person_results: List[Dict[str, Any]] = []
            if personalized_lookup:
                person_results = [
                    _prepare_chunk_result(row, focus_terms=focus_terms, reports_to_lookup=reports_to_lookup)
                    for row in session.run(
                        PERSON_GRAPH_VECTOR_QUERY if resolved_seed_hops >= DEFAULT_SEED_CONTEXT_HOPS else PERSON_GRAPH_VECTOR_QUERY_SHALLOW,
                        user_id=user_id,
                        query_embedding=query_embedding.tolist(),
                        recency_decay_days=RECENCY_DECAY_DAYS,
                        recency_boost_max=RECENCY_BOOST_MAX,
                        candidate_limit=DEFAULT_RETRIEVAL_LIMIT,
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
                        recency_decay_days=RECENCY_DECAY_DAYS,
                        recency_boost_max=RECENCY_BOOST_MAX,
                        candidate_limit=DEFAULT_RETRIEVAL_LIMIT,
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
                        recency_decay_days=RECENCY_DECAY_DAYS,
                        recency_boost_max=RECENCY_BOOST_MAX,
                        candidate_limit=DEFAULT_RETRIEVAL_LIMIT,
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
                        focus_terms=focus_terms,
                        reports_to_lookup=reports_to_lookup,
                    )
                    for row in session.run(
                        PERSON_TASK_FACT_QUERY,
                        user_id=user_id,
                        claim_types=sorted(TASK_LIKE_FACT_TYPES),
                        candidate_limit=DEFAULT_RETRIEVAL_LIMIT,
                    ).data()
                    if row.get("fact_id")
                ]

        if personalized_lookup and query_type in FACT_PRIORITY_QUERY_TYPES:
            vector_results = _merge_ranked_results(person_results, [], limit=DEFAULT_RETRIEVAL_LIMIT)
            fact_results = _merge_ranked_results(exact_task_fact_results + person_fact_results, [], limit=DEFAULT_RETRIEVAL_LIMIT)
            fact_results = [item for item in fact_results if _fact_result_matches_user(item, user_id)]
        else:
            vector_results = _merge_ranked_results(person_results, global_results, limit=DEFAULT_RETRIEVAL_LIMIT)
            fact_results = _merge_ranked_results(exact_task_fact_results + person_fact_results, global_fact_results, limit=DEFAULT_RETRIEVAL_LIMIT)
        combined_results = _combine_ranked_results(
            vector_results,
            fact_results,
            query_type=query_type,
            focus_terms=focus_terms,
            reports_to_lookup=reports_to_lookup,
            minimum_focus_match=_minimum_focus_match_threshold(
                user_input,
                query_type=query_type,
                query_profile=query_profile,
                focus_terms=focus_terms,
            ),
            limit=DEFAULT_RETRIEVAL_LIMIT,
        )
        evidence: List[Dict[str, Any]] = []
        documents: List[str] = []
        matched_entities: List[str] = []

        for item in combined_results:
            if item.get("fact_id"):
                if personalized_lookup and query_type in FACT_PRIORITY_QUERY_TYPES and not _fact_result_matches_user(item, user_id):
                    continue
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
                    "retrieved_at": retrieved_at,
                    "relationship": "CANONICAL_FACT",
                    "retrieval_path": path_summary["path"],
                    "hop_count": path_summary["hop_count"],
                    # UI debug tooling expects a chunk_id to visualize a Neo4j subgraph.
                    # Canonical-fact evidence is anchored to a supporting Document, which
                    # is always stored with at least one chunk ("<doc_id>-chunk-0").
                    "chunk_id": f"{supporting_document.get('doc_id')}-chunk-0" if supporting_document.get("doc_id") else None,
                    "exact_match": bool(item.get("exact_match")),
                    "fact_priority": bool(item.get("fact_priority")),
                    "document": {
                        "doc_id": supporting_document.get("doc_id"),
                        "subject": supporting_document.get("subject"),
                        "sender": supporting_document.get("sender"),
                        "timestamp": supporting_document.get("timestamp"),
                        "source": supporting_document.get("source"),
                        "content": _preview_trace_content(supporting_document.get("content")),
                        "conversation_type": supporting_document.get("conversation_type"),
                        "conversation_id": supporting_document.get("conversation_id"),
                        "group_id": supporting_document.get("group_id"),
                    },
                    "related_node": {
                        "label": "CanonicalFact",
                        "display_name": canonical_key or fact_summary,
                        "id": fact_id,
                    },
                    "related_node_id": fact_id,
                    "direction": "fact",
                    "fact": {
                        "claim_type": fact.get("claim_type"),
                        "status": fact.get("status"),
                        "canonical_key": canonical_key,
                        "value_text": fact.get("value_text"),
                        "subject_key": fact.get("subject_key"),
                        "subject_entity_id": fact.get("subject_entity_id"),
                        "subject_display": fact.get("subject_display"),
                        "object_key": fact.get("object_key"),
                        "object_entity_id": fact.get("object_entity_id"),
                        "object_display": fact.get("object_display"),
                        "display_summary": fact.get("display_summary"),
                        "temporal_start": fact.get("temporal_start"),
                        "temporal_end": fact.get("temporal_end"),
                        "temporal_granularity": fact.get("temporal_granularity"),
                        "first_seen_at": fact.get("first_seen_at"),
                        "last_seen_at": fact.get("last_seen_at"),
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
                "retrieved_at": retrieved_at,
                "relationship": relationship,
                "retrieval_path": computed_path,
                "hop_count": computed_hop_count,
                "document": {
                    "doc_id": doc_id,
                    "subject": subject,
                    "sender": sender,
                    "timestamp": document.get("timestamp"),
                    "source": document.get("source"),
                    "content": _preview_trace_content(document.get("content")),
                    "conversation_type": document.get("conversation_type"),
                    "conversation_id": document.get("conversation_id"),
                    "group_id": document.get("group_id"),
                },
                "related_node": {
                    "label": related_label,
                    "display_name": related_name,
                    "id": related_node.get("id") or related_node.get("_element_id"),
                }
                if related_node
                else {},
                "related_node_id": (related_node.get("id") or related_node.get("_element_id")) if related_node else None,
                "direction": "OUT",
                "fact": None,
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
            "query_profile": query_profile,
            "graph_depth": graph_depth,
            "matched_entities": matched_entities,
            "retrieved_at": retrieved_at,
            "result_count": len(evidence),
            "max_hop_count": max((item["hop_count"] for item in evidence), default=0),
            "retrieval_path": evidence[0]["retrieval_path"] if evidence else _build_path_summary(personalized_lookup, None)["path"],
            "evidence": evidence,
            "no_evidence": not evidence,
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
                "query_profile": query_profile,
                "graph_depth": graph_depth,
                "matched_entities": [],
                "retrieved_at": retrieved_at,
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
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    mode, reason_code = _select_answer_mode(query, retrieval_trace=retrieval_trace)
    no_conversation_window = "No recent people chat window was available."
    conversation_window = _build_conversation_window(query, history=history, retrieval_trace=retrieval_trace)
    use_people_chat_context = conversation_window != no_conversation_window

    if not use_people_chat_context:
        guarded_answer = _build_guarded_abstention_answer(
            query,
            documents,
            mode=mode,
            reason_code=reason_code,
            retrieval_trace=retrieval_trace,
        )
        if guarded_answer is not None:
            return guarded_answer
        fact_backed_answer = _build_fact_backed_answer(
            query,
            mode=mode,
            reason_code=reason_code,
            retrieval_trace=retrieval_trace,
        )
        if fact_backed_answer is not None:
            return fact_backed_answer

    if not documents and conversation_window == no_conversation_window:
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

    if not use_people_chat_context and _wants_verbatim_evidence(query):
        excerpts = _extract_verbatim_chat_excerpts(retrieval_trace)
        if excerpts:
            summary = excerpts[0]
            bullets = excerpts[1:]
        else:
            summary = "I couldn't find any retrieved chat text to quote verbatim."
            bullets = []
        answer_payload = _build_answer_payload(
            mode=ANSWER_MODE_SHORT,
            reason_code=REASON_CODE_VERBATIM_EVIDENCE,
            summary=summary,
            bullets=bullets,
            retrieval_trace=retrieval_trace,
        )
        return {
            "answer": answer_payload["summary"],
            "answer_payload": answer_payload,
            "thinking": [],
        }

    try:
        context = _build_response_context(documents, retrieval_trace=retrieval_trace)
        query_profile = dict((retrieval_trace or {}).get("query_profile") or query_shape.analyze_query(query))
        user_context = "No authenticated user context was provided."
        if user_id:
            user_context = f"Authenticated user id: {user_id}."
            if _is_personalized_lookup(query):
                user_context += " Treat first-person references (I/me/my) as this user unless the query says otherwise."
        retrieval_guidance = "Use only the retrieved evidence. If evidence is weak or missing, say so clearly."
        question_shape_guidance = "Answer directly from the available evidence."
        query_type = (retrieval_trace or {}).get("query_type")
        if query_type:
            user_context += f" Query classification: {query_type}."
        if query_type in FACT_PRIORITY_QUERY_TYPES:
            retrieval_guidance = (
                "This is a task or commitment lookup. Prioritize current CanonicalFact evidence over chunk summaries. "
                "Prefer the most temporally relevant active fact over stale past facts, and use chunk evidence only to support provenance or clarify ambiguity. "
                "Use timestamp metadata silently for recency decisions; do not expose retrieval/source timestamp metadata in the visible answer unless the user asks for it. "
                "Do not list older alternatives unless the user asked for history or the top evidence is genuinely conflicting."
            )
        elif (retrieval_trace or {}).get("evidence"):
            retrieval_guidance = (
                "Canonical facts are the highest-trust evidence layer. If a canonical fact conflicts with a chunk summary, trust the canonical fact and mention the discrepancy."
            )
        if use_people_chat_context:
            retrieval_guidance += (
                " Use the recent people chat window as additional chat context for speaker, recipient, and recency resolution, "
                "while still grounding the answer in the retrieved SAGE evidence and the user's actual question."
            )
        if query_profile.get("wants_list_format"):
            question_shape_guidance = (
                "The user is asking for multiple items. Keep the answer set-oriented, surface distinct supported items, "
                "and do not rewrite the question into a single-person lookup unless the evidence only supports one item."
            )
            retrieval_guidance += (
                " Prefer distinct evidence items over near-duplicate restatements when summarizing the answer."
            )
        if any(
            (item.get("document") or {}).get("conversation_type") == "group"
            for item in ((retrieval_trace or {}).get("evidence") or [])
        ):
            retrieval_guidance += (
                " If a request or instruction comes from a group conversation without a resolved target person, "
                "say that the target is ambiguous instead of assigning it to one person."
            )
        critic_feedback = dict((retrieval_trace or {}).get("critic_feedback") or {})
        critic_issues = [str(issue) for issue in critic_feedback.get("issues") or [] if str(issue).strip()]
        if critic_issues:
            retrieval_guidance += (
                " The previous answer failed critic review for these issue(s): "
                + ", ".join(critic_issues)
                + ". Revise the answer so it directly satisfies the original question's requested answer slots using only the evidence."
            )
        llm = _create_groq_client(temperature=0.3, require_json=True)
        chain = CHAT_PROMPT | llm | StrOutputParser()
        response = chain.invoke(
            {
                "query": query,
                "context": context,
                "conversation_window": conversation_window,
                "user_context": user_context,
                "retrieval_guidance": retrieval_guidance,
                "answer_mode": mode,
                "question_shape_guidance": question_shape_guidance,
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
        logger.error(f"LLM API error: {exc}")
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
        logger.warning(f"LLM summary failed, using fallback summary: {exc}")
        return _summarize_text_fallback(text)


_SHORT_CONTENT_CHAR_LIMIT = 500
_SHORT_CONTENT_WORD_LIMIT = 200

_TRACE_EVIDENCE_CONTENT_LIMIT = 800
_CONVERSATION_WINDOW_LIMIT = 20
_CONVERSATION_WINDOW_CONTENT_LIMIT = 500


def _preview_trace_content(value: Any, limit: int = _TRACE_EVIDENCE_CONTENT_LIMIT) -> Optional[str]:
    """Return a safe-to-display content preview for trace evidence.

    Chat messages are typically short and will pass through unchanged.
    Uploaded documents can be very large, so we cap the preview to keep
    UI payloads reasonable.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _history_field(entry: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry and entry.get(key) is not None:
            return entry.get(key)
    return None


def _window_preview_text(value: Any, *, limit: int = _CONVERSATION_WINDOW_CONTENT_LIMIT) -> str:
    text = _convert_iso_timestamps_to_ist_text(str(value or "").strip())
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _window_timestamp_label(value: Any) -> str:
    formatted = _format_context_timestamp(value)
    return formatted or "Time unknown"


def _history_sender_label(entry: Dict[str, Any]) -> str:
    sender = _history_field(entry, "sender_name", "senderName", "sender", "sender_id", "senderId")
    if sender:
        return _normalize_summary_text(sender) or "Unknown sender"
    role = str(_history_field(entry, "role") or "").strip().lower()
    if role == "assistant":
        return "SAGE assistant"
    if role == "user":
        return "User"
    return "Unknown sender"


def _history_recipient_label(entry: Dict[str, Any]) -> str:
    direct_recipient = _history_field(
        entry,
        "recipient_name",
        "recipientName",
        "receiver_name",
        "receiverName",
        "recipient",
        "receiver",
    )
    if direct_recipient:
        return _normalize_summary_text(direct_recipient) or "Unspecified recipient"

    receiver_ids = _history_field(entry, "receiver_ids", "receiverIds", "recipient_ids", "recipientIds")
    if isinstance(receiver_ids, list):
        labels = [_normalize_summary_text(value) for value in receiver_ids if _normalize_summary_text(value)]
        if labels:
            return ", ".join(labels[:4])

    group_name = _history_field(entry, "group_name", "groupName")
    if group_name:
        return _normalize_summary_text(group_name) or "Group chat"

    conversation_type = str(_history_field(entry, "conversation_type", "conversationType") or "").strip().lower()
    role = str(_history_field(entry, "role") or "").strip().lower()
    if conversation_type == "sage":
        return "User" if role == "assistant" else "SAGE assistant"
    if conversation_type == "group":
        return "Group chat"
    return "Unspecified recipient"


def _build_history_conversation_window(history: Optional[List[Dict[str, Any]]], *, limit: int = _CONVERSATION_WINDOW_LIMIT) -> List[str]:
    lines: List[str] = []
    for entry in list(history or [])[-limit:]:
        if not isinstance(entry, dict):
            continue
        content = _window_preview_text(_history_field(entry, "content", "message", "text"))
        if not content:
            continue
        timestamp = _window_timestamp_label(_history_field(entry, "sent_at", "sentAt", "timestamp"))
        sender = _history_sender_label(entry)
        recipient = _history_recipient_label(entry)
        lines.append(f"- {timestamp} | {sender} -> {recipient} | {content}")
    return lines


def _candidate_window_doc_ids(retrieval_trace: Optional[Dict[str, Any]]) -> List[str]:
    doc_ids: List[str] = []
    for item in (retrieval_trace or {}).get("evidence") or []:
        doc_id = (item.get("document") or {}).get("doc_id")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(str(doc_id))
        if len(doc_ids) >= _CONVERSATION_WINDOW_LIMIT:
            break
    return doc_ids


def _candidate_window_conversation_ids(retrieval_trace: Optional[Dict[str, Any]]) -> List[str]:
    conversation_ids: List[str] = []
    for item in (retrieval_trace or {}).get("evidence") or []:
        conversation_id = (item.get("document") or {}).get("conversation_id")
        if conversation_id and conversation_id not in conversation_ids:
            conversation_ids.append(str(conversation_id))
    return conversation_ids


def _fetch_recent_graph_messages(
    query: str,
    *,
    retrieval_trace: Optional[Dict[str, Any]] = None,
    limit: int = _CONVERSATION_WINDOW_LIMIT,
) -> List[Dict[str, Any]]:
    candidate_doc_ids = _candidate_window_doc_ids(retrieval_trace)
    conversation_ids = _candidate_window_conversation_ids(retrieval_trace)
    terms = _extract_query_focus_terms(query)
    use_term_match = not candidate_doc_ids and not conversation_ids and bool(terms)
    if not candidate_doc_ids and not conversation_ids and not use_term_match:
        return []

    driver = None
    try:
        driver = utils.create_neo4j_driver()
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            rows = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (sender:Person {id: d.sender})
                OPTIONAL MATCH (d)-[:RECEIVED_BY]->(receiver:Person)
                WITH d,
                     coalesce(sender.name, sender.id, d.sender) AS sender_name,
                     [name IN collect(DISTINCT coalesce(receiver.name, receiver.id))
                        WHERE name IS NOT NULL AND trim(name) <> ''] AS receiver_names
                WHERE coalesce(d.conversation_type, '') <> 'sage'
                  AND (
                    (size($candidate_doc_ids) > 0 AND d.doc_id IN $candidate_doc_ids)
                    OR (size($conversation_ids) > 0 AND d.conversation_id IN $conversation_ids)
                    OR (
                        $use_term_match
                        AND any(term IN $terms WHERE
                            toLower(coalesce(d.content, '')) CONTAINS term
                            OR toLower(coalesce(d.summary, '')) CONTAINS term
                            OR toLower(coalesce(d.subject, '')) CONTAINS term
                            OR toLower(coalesce(sender_name, '')) CONTAINS term
                            OR any(name IN receiver_names WHERE toLower(name) CONTAINS term)
                        )
                    )
                  )
                RETURN d.doc_id AS doc_id,
                       d.content AS content,
                       d.timestamp AS timestamp,
                       d.conversation_type AS conversation_type,
                       d.conversation_id AS conversation_id,
                       d.group_id AS group_id,
                       d.subject AS subject,
                       sender_name AS sender_name,
                       receiver_names AS receiver_names
                ORDER BY coalesce(d.timestamp, '') DESC
                LIMIT $limit
                """,
                candidate_doc_ids=candidate_doc_ids,
                conversation_ids=conversation_ids,
                terms=terms,
                use_term_match=use_term_match,
                limit=max(1, int(limit)),
            ).data()
            return [dict(row) for row in rows]
    except Exception as exc:
        logger.warning("Failed to fetch recent graph messages for transcript window: %s", exc)
        return []
    finally:
        if driver:
            driver.close()


def _build_graph_conversation_window(
    query: str,
    *,
    retrieval_trace: Optional[Dict[str, Any]] = None,
    limit: int = _CONVERSATION_WINDOW_LIMIT,
) -> List[str]:
    rows = _fetch_recent_graph_messages(query, retrieval_trace=retrieval_trace, limit=limit)
    if not rows:
        return []

    lines: List[str] = []
    for row in reversed(rows):
        content = _window_preview_text(row.get("content") or row.get("subject"))
        if not content:
            continue
        sender = _normalize_summary_text(row.get("sender_name")) or "Unknown sender"
        receiver_names = [
            _normalize_summary_text(value)
            for value in (row.get("receiver_names") or [])
            if _normalize_summary_text(value)
        ]
        recipient = ", ".join(receiver_names[:4]) if receiver_names else (
            "Group chat" if str(row.get("conversation_type") or "").strip().lower() == "group" else "Unspecified recipient"
        )
        timestamp = _window_timestamp_label(row.get("timestamp"))
        lines.append(f"- {timestamp} | {sender} -> {recipient} | {content}")
    return lines


def _build_conversation_window(
    query: str,
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    retrieval_trace: Optional[Dict[str, Any]] = None,
    limit: int = _CONVERSATION_WINDOW_LIMIT,
) -> str:
    graph_lines = _build_graph_conversation_window(query, retrieval_trace=retrieval_trace, limit=limit)
    if graph_lines:
        return "Recent people chat evidence (oldest to newest):\n" + "\n".join(graph_lines)

    return "No recent people chat window was available."


def _document_exists(session, doc_id: str) -> bool:
    """Fast check: does a Document node with this doc_id already exist?"""
    rows = session.run(
        "MATCH (d:Document {doc_id: $doc_id}) RETURN d.doc_id AS id LIMIT 1",
        doc_id=doc_id,
    ).data()
    return bool(rows)


def _chunk_exists(session, chunk_id: str) -> bool:
    rows = session.run(
        "MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN c.chunk_id AS id LIMIT 1",
        chunk_id=chunk_id,
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
            if needs_llm and utils.chat_llm_configured():
                try:
                    llm = _create_groq_client(temperature=0.0, require_json=True)
                except Exception as exc:
                    logger.warning(f"Failed to initialize LLM client, using fallback summaries: {exc}")
            elif needs_llm:
                logger.warning("No chat LLM is configured. Falling back to local summaries for ingestion.")

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
                    d.source_normalized = $source_normalized,
                    d.source_version = $source_version,
                    d.schema_version = $schema_version,
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
                source_normalized=data.get("source_normalized") or data.get("source"),
                source_version=data.get("source_version") or 1,
                schema_version=data.get("schema_version") or 1,
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
                chunk_id = f"{data['doc_id']}-chunk-0"
                if not _chunk_exists(session, chunk_id):
                    session.run(
                        """
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.content = $content,
                            c.embedding = $embedding,
                            c.summary = $summary,
                            c.timestamp = $timestamp
                        MERGE (d:Document {doc_id: $doc_id})
                        MERGE (c)-[:PART_OF]->(d)
                        """,
                        chunk_id=chunk_id,
                        content=content,
                        embedding=chunk_embedding,
                        summary=document_summary,
                        timestamp=data.get("timestamp"),
                        doc_id=data["doc_id"],
                    )
            else:
                # Long content: full chunking pipeline with LLM summaries
                chunks = utils.chunk_document(content, max_chunk_words=250, overlap_sentences=2)
                for i, chunk in enumerate(chunks):
                    chunk_summary = _smart_summarize(llm, chunk)
                    chunk_embedding = utils.generate_embedding(chunk_summary)
                    chunk_id = f"{data['doc_id']}-chunk-{i}"
                    if _chunk_exists(session, chunk_id):
                        continue
                    session.run(
                        """
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.content = $content,
                            c.embedding = $embedding,
                            c.summary = $summary,
                            c.timestamp = $timestamp
                        MERGE (d:Document {doc_id: $doc_id})
                        MERGE (c)-[:PART_OF]->(d)
                        """,
                        chunk_id=chunk_id,
                        content=chunk,
                        embedding=chunk_embedding,
                        summary=chunk_summary,
                        timestamp=data.get("timestamp"),
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
