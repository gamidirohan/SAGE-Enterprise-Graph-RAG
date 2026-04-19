"""SAIA support for chat-centric claim extraction and canonical fact maintenance.

This module keeps raw chat messages/documents immutable and writes a separate
claim + canonical-fact layer on top of the evidence graph.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import app.utils as utils
except ImportError:  # pragma: no cover - import fallback for direct execution
    import utils


logger = logging.getLogger(__name__)

_ALLOWED_SOURCES = {"chat_message", "message_attachment"}
_NON_CANONICAL_CLAIM_TYPES = {"REQUEST"}

TIMEZONE_DEFAULT = os.getenv("SAIA_DEFAULT_TIMEZONE", "UTC")
CONTEXT_WINDOW_MESSAGES = int(os.getenv("SAIA_CONTEXT_WINDOW_MESSAGES", "3"))
MIN_GRAPH_WORTHY_CONFIDENCE = float(os.getenv("SAIA_MIN_GRAPH_WORTHY_CONFIDENCE", "0.70"))
MIN_CANONICAL_CONFIDENCE = float(os.getenv("SAIA_MIN_CANONICAL_CONFIDENCE", "0.80"))
ALLOW_AI_AUTHORED_EVIDENCE = os.getenv("SAIA_ALLOW_AI_AUTHORED_EVIDENCE", "false").lower() in {"1", "true", "yes"}

FIRST_PERSON_TOKENS = {"i", "me", "my", "mine", "myself"}
SECOND_PERSON_TOKENS = {"you", "your", "yours", "yourself", "yourselves"}
FIRST_PERSON_PLURAL_TOKENS = {"we", "our", "ours", "us", "ourselves"}
NEUTRAL_ANAPHORA_TOKENS = {"it", "its", "itself", "this", "that", "these", "those", "they", "them", "their", "theirs"}
CONTEXTUAL_CONTINUATION_PREFIXES = {"for", "regarding", "about", "re", "under", "within", "on", "by", "at"}

COMMITMENT_VERB_MAP = {
    "send": "send",
    "sending": "send",
    "share": "share",
    "sharing": "share",
    "review": "review",
    "reviewing": "review",
    "prepare": "prepare",
    "preparing": "prepare",
    "update": "update",
    "updating": "update",
    "schedule": "schedule",
    "scheduling": "schedule",
    "deliver": "deliver",
    "delivering": "deliver",
    "discuss": "discuss",
    "discussing": "discuss",
    "confirm": "confirm",
    "confirming": "confirm",
    "provide": "provide",
    "providing": "provide",
}
COMMITMENT_VERB_PATTERN = "|".join(sorted((re.escape(verb) for verb in COMMITMENT_VERB_MAP), key=len, reverse=True))

ENTITY_TOKEN_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_\-]*"
ENTITY_PHRASE_PATTERN = rf"{ENTITY_TOKEN_PATTERN}(?:\s+{ENTITY_TOKEN_PATTERN}){{0,5}}"
OPTIONAL_SCOPED_ENTITY_PATTERN = rf"(?:the\s+|our\s+|my\s+)?{ENTITY_PHRASE_PATTERN}"

TIME_WORD_PATTERN = re.compile(
    r"\b(today|tomorrow|yesterday|now|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|in\s+\d+\s+(?:day|days|week|weeks)|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
CLOCK_PATTERN = re.compile(r"\b(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)\b", re.IGNORECASE)
AT_CLOCK_PATTERN = re.compile(r"\bat\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)?\b", re.IGNORECASE)
REQUEST_PATTERN = re.compile(r"^\s*(?:can|could|would|will)\s+you\s+(?P<action>[^?.!]+)\??$", re.IGNORECASE)
REPORTS_TO_PATTERN = re.compile(
    r"\b(?P<subject>[A-Z][A-Za-z0-9_\-]*(?:\s+[A-Z][A-Za-z0-9_\-]*)*|EMP\d{3})\s+(?:now\s+)?reports\s+to\s+(?P<object>[A-Z][A-Za-z0-9_\-]*(?:\s+[A-Z][A-Za-z0-9_\-]*)*|EMP\d{3})\b",
    re.IGNORECASE,
)
APPROVAL_PATTERN = re.compile(
    rf"\b(?P<subject>{OPTIONAL_SCOPED_ENTITY_PATTERN})\s+(?P<verb>approved|approves|authorized|authorised)\s+(?P<object>[^.?!]+)",
    re.IGNORECASE,
)
PASSIVE_APPROVAL_PATTERN = re.compile(
    rf"\b(?P<object>{OPTIONAL_SCOPED_ENTITY_PATTERN})\s+is\s+(?P<status>approved|authorized|authorised)\b",
    re.IGNORECASE,
)
STATUS_PATTERN = re.compile(
    rf"\b(?P<subject>{OPTIONAL_SCOPED_ENTITY_PATTERN})\s+is\s+(?P<status>on\s+track|delayed|blocked|complete|completed|pending|down|offline|degraded|up|unavailable)\b",
    re.IGNORECASE,
)
MEETING_PATTERN = re.compile(
    r"\b(?P<event>(?:[A-Za-z]+\s+){0,3}(?:meeting|call|discussion|review))\b",
    re.IGNORECASE,
)
MEET_VERB_PATTERN = re.compile(
    r"\b(?:let'?s|let us|can we|could we|should we|shall we|please)?\s*meet(?:\s+with\s+(?P<counterparty>[^?.!,]+))?\b",
    re.IGNORECASE,
)
ASSIGNMENT_START_PATTERN = re.compile(
    rf"\b(?P<subject>{ENTITY_PHRASE_PATTERN})\s+is\s+(?:currently\s+)?(?:assigned\s+to|working\s+on)\s+(?P<object>[^.?!]+)\b",
    re.IGNORECASE,
)
ASSIGNMENT_END_PATTERN = re.compile(
    rf"\b(?P<subject>{ENTITY_PHRASE_PATTERN})\s+is\s+no\s+longer\s+(?:assigned\s+to|working\s+on)\s+(?P<object>[^.?!]+)\b",
    re.IGNORECASE,
)
FIRST_PERSON_COMMITMENT_PATTERN = re.compile(
    rf"\bI(?:'ll|\s+will)(?:\s+be)?\s+(?P<verb>{COMMITMENT_VERB_PATTERN})\s+(?P<body>[^?.!]+)",
    re.IGNORECASE,
)
NAMED_COMMITMENT_PATTERN = re.compile(
    rf"\b(?P<subject>[A-Z][A-Za-z0-9_\-]*(?:\s+[A-Z][A-Za-z0-9_\-]*)*|EMP\d{{3}})\s+(?:will|should|must)(?:\s+be)?\s+(?P<verb>{COMMITMENT_VERB_PATTERN})\s+(?P<body>[^?.!]+)",
    re.IGNORECASE,
)


@dataclass
class GroundingContext:
    source_kind: str
    source_doc_id: str
    source_message_id: Optional[str]
    linked_message_id: Optional[str]
    sender_id: str
    receiver_ids: List[str]
    conversation_id: Optional[str]
    conversation_type: Optional[str]
    group_id: Optional[str]
    sent_at: str
    timezone: str = TIMEZONE_DEFAULT
    attachment_name: Optional[str] = None
    source: str = "chat_message"
    is_ai_response: bool = False
    trace: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None

    @property
    def scope_type(self) -> str:
        if self.conversation_type == "group" and self.group_id:
            return "group"
        if self.conversation_id:
            return "conversation"
        if self.source_message_id:
            return "message"
        return "document"

    @property
    def scope_id(self) -> str:
        if self.scope_type == "group" and self.group_id:
            return self.group_id
        if self.scope_type == "conversation" and self.conversation_id:
            return self.conversation_id
        if self.scope_type == "message" and self.source_message_id:
            return self.source_message_id
        return self.source_doc_id


@dataclass
class Resolution:
    raw: str
    key: Optional[str]
    entity_id: Optional[str]
    entity_type: Optional[str]
    status: str
    display_name: Optional[str] = None


WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def is_enabled() -> bool:
    return os.getenv("SAIA_ENABLED", "false").lower() in {"1", "true", "yes"}


def process_chat_message(
    *,
    message_id: str,
    sender_id: str,
    receiver_ids: Sequence[str],
    conversation_id: Optional[str],
    conversation_type: Optional[str],
    group_id: Optional[str],
    sent_at: str,
    content: str,
    source: str = "chat_message",
    trace: Optional[Dict[str, Any]] = None,
    is_ai_response: bool = False,
    attachment_name: Optional[str] = None,
) -> Dict[str, Any]:
    context = GroundingContext(
        source_kind="chat_message",
        source_doc_id=f"chat-msg-{message_id}",
        source_message_id=message_id,
        linked_message_id=None,
        sender_id=sender_id,
        receiver_ids=list(receiver_ids),
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        group_id=group_id,
        sent_at=sent_at,
        source=source,
        is_ai_response=is_ai_response,
        trace=trace,
        attachment_name=attachment_name,
    )
    return process_text(content, context)


def process_message_attachment(
    *,
    doc_id: str,
    linked_message_id: str,
    sender_id: str,
    receiver_ids: Sequence[str],
    conversation_id: Optional[str],
    conversation_type: Optional[str],
    group_id: Optional[str],
    sent_at: str,
    content: str,
    source: str = "message_attachment",
    attachment_name: Optional[str] = None,
) -> Dict[str, Any]:
    context = GroundingContext(
        source_kind="message_attachment",
        source_doc_id=doc_id,
        source_message_id=linked_message_id,
        linked_message_id=linked_message_id,
        sender_id=sender_id,
        receiver_ids=list(receiver_ids),
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        group_id=group_id,
        sent_at=sent_at,
        source=source,
        attachment_name=attachment_name,
    )
    return process_text(content, context)


def process_text(content: str, context: GroundingContext) -> Dict[str, Any]:
    if not is_enabled():
        return {"status": "disabled", "claims_extracted": 0, "claims_canonicalized": 0, "conflicts_found": 0}

    cleaned_content = _prepare_text(content)
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            if not _is_source_eligible(context, cleaned_content):
                result = _build_result("skipped", [], 0, 0)
                _finalize_run(session, context, result, reason="source_ineligible")
                return result

            claims = extract_claims_from_text(cleaned_content, context, session=session)
            if not claims:
                result = _build_result("skipped", [], 0, 0)
                _finalize_run(session, context, result, reason="no_claims")
                return result

            canonicalized = 0
            conflicts = 0
            stored_claims = 0
            run_id = _make_run_id(context)

            _link_message_to_document(session, context)
            for claim in claims:
                stored_claims += 1
                claim["claim_id"] = _make_claim_id(context, claim)
                claim["canonical_key"] = _build_canonical_key(claim)
                claim["promotion_status"] = claim.get("promotion_status") or "pending"
                claim["mutation_action"] = claim.get("mutation_action") or "awaiting_decision"
                _persist_claim(session, context, claim)
                if not should_promote_claim(claim):
                    claim["mutation_action"] = "not_promoted"
                    _mark_claim_promotion(
                        session,
                        claim["claim_id"],
                        claim["promotion_status"],
                        claim["mutation_action"],
                    )
                    continue
                action, existing_fact_id = choose_mutation_action(session, claim)
                if action == "confirm_existing_fact" and existing_fact_id:
                    _link_claim_to_fact(session, claim["claim_id"], existing_fact_id, relation_type="SUPPORTS")
                    _touch_existing_fact(session, existing_fact_id)
                    claim["promotion_status"] = "confirmed"
                    claim["mutation_action"] = action
                    _mark_claim_promotion(
                        session,
                        claim["claim_id"],
                        claim["promotion_status"],
                        claim["mutation_action"],
                    )
                    canonicalized += 1
                elif action == "insert_new_fact":
                    fact_id = _persist_fact(session, claim)
                    _link_claim_to_fact(session, claim["claim_id"], fact_id, relation_type="SUPPORTS")
                    claim["promotion_status"] = "promoted"
                    claim["mutation_action"] = action
                    _mark_claim_promotion(
                        session,
                        claim["claim_id"],
                        claim["promotion_status"],
                        claim["mutation_action"],
                    )
                    canonicalized += 1
                elif action == "supersede_current_fact" and existing_fact_id:
                    fact_id = _persist_fact(session, claim)
                    _supersede_existing_fact(session, existing_fact_id, fact_id)
                    _link_claim_to_fact(session, claim["claim_id"], fact_id, relation_type="SUPPORTS")
                    _link_claim_to_fact(session, claim["claim_id"], existing_fact_id, relation_type="CONTRADICTS")
                    conflicts += 1
                    claim["promotion_status"] = "promoted"
                    claim["mutation_action"] = action
                    _mark_claim_promotion(
                        session,
                        claim["claim_id"],
                        claim["promotion_status"],
                        claim["mutation_action"],
                    )
                    canonicalized += 1
                else:
                    claim["promotion_status"] = "pending_review"
                    claim["mutation_action"] = "pending_review"
                    if existing_fact_id:
                        _link_claim_to_fact(session, claim["claim_id"], existing_fact_id, relation_type="CONTRADICTS")
                        conflicts += 1
                    _mark_claim_promotion(
                        session,
                        claim["claim_id"],
                        claim["promotion_status"],
                        claim["mutation_action"],
                    )

            result = _build_result("completed", claims, canonicalized, conflicts)
            _persist_run(session, run_id, context, result)
            _mark_source_status(session, context, "completed", None)
            return result
    except Exception as exc:
        logger.exception("SAIA processing failed for %s", context.source_doc_id)
        try:
            with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
                _mark_source_status(session, context, "failed", str(exc))
                result = {
                    "status": "failed",
                    "claims_extracted": 0,
                    "claims_canonicalized": 0,
                    "conflicts_found": 0,
                    "reason": str(exc),
                }
                _persist_run(session, _make_run_id(context), context, result)
        except Exception:  # pragma: no cover - best effort
            logger.exception("Failed to record SAIA failure for %s", context.source_doc_id)
        return {"status": "failed", "claims_extracted": 0, "claims_canonicalized": 0, "conflicts_found": 0, "reason": str(exc)}
    finally:
        driver.close()


def extract_claims_from_text(text: str, context: GroundingContext, session: Any = None) -> List[Dict[str, Any]]:
    sentences = _split_claim_spans(text)
    claims: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for sentence in sentences:
        for extractor in (
            _extract_request_claims,
            _extract_reports_to_claims,
            _extract_approval_claims,
            _extract_status_claims,
            _extract_assignment_claims,
            _extract_meeting_claims,
            _extract_commitment_claims,
        ):
            for claim in extractor(sentence, context, session=session):
                dedupe_key = _claim_dedupe_key(claim)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                claims.append(claim)
    return claims


def should_promote_claim(claim: Dict[str, Any]) -> bool:
    if claim.get("claim_type") in _NON_CANONICAL_CLAIM_TYPES:
        return False
    if not claim.get("graph_worthy"):
        return False
    if claim.get("resolution_status") != "resolved":
        return False
    if float(claim.get("canonical_confidence") or 0.0) < MIN_CANONICAL_CONFIDENCE:
        return False
    return True


def choose_mutation_action(session: Any, claim: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    existing = _load_current_facts(session, claim["canonical_key"])
    if not existing:
        return "insert_new_fact", None
    for fact in existing:
        if _facts_match(fact, claim):
            return "confirm_existing_fact", fact.get("fact_id")
    if len(existing) == 1 and float(claim.get("canonical_confidence") or 0.0) >= MIN_CANONICAL_CONFIDENCE:
        return "supersede_current_fact", existing[0].get("fact_id")
    return "pending_review", existing[0].get("fact_id") if existing else None


def normalize_temporal_reference(text: str, anchor_iso: str, timezone_name: str = TIMEZONE_DEFAULT) -> Dict[str, Optional[str]]:
    anchor = _parse_iso_datetime(anchor_iso)
    lowered = text.lower().strip()
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", lowered)
    time_match = CLOCK_PATTERN.search(lowered) or AT_CLOCK_PATTERN.search(lowered)

    target_date: Optional[date] = None
    granularity = "unresolved"
    if "now" in lowered:
        return {
            "temporal_start": anchor.isoformat(),
            "temporal_end": None,
            "temporal_granularity": "datetime",
            "timezone": timezone_name,
        }
    if date_match:
        target_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
        granularity = "date"
    elif "today" in lowered:
        target_date = anchor.date()
        granularity = "date"
    elif "tomorrow" in lowered:
        target_date = anchor.date() + timedelta(days=1)
        granularity = "date"
    elif "yesterday" in lowered:
        target_date = anchor.date() - timedelta(days=1)
        granularity = "date"
    else:
        next_weekday_match = re.search(
            r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            lowered,
        )
        if next_weekday_match:
            target_date = _next_weekday(anchor.date(), WEEKDAYS[next_weekday_match.group(1)])
            granularity = "date"
        else:
            in_match = re.search(r"\bin\s+(\d+)\s+(day|days|week|weeks)\b", lowered)
            if in_match:
                value = int(in_match.group(1))
                unit = in_match.group(2)
                delta_days = value * 7 if unit.startswith("week") else value
                target_date = anchor.date() + timedelta(days=delta_days)
                granularity = "date"

    if target_date is None:
        return {
            "temporal_start": None,
            "temporal_end": None,
            "temporal_granularity": "unresolved",
            "timezone": timezone_name,
        }

    if time_match:
        hour = int(time_match.group("hour"))
        minute = int(time_match.group("minute") or 0)
        ampm = (time_match.group("ampm") or "").lower()
        if hour > 23:
            return {
                "temporal_start": target_date.isoformat(),
                "temporal_end": None,
                "temporal_granularity": granularity,
                "timezone": timezone_name,
            }
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        dt = datetime.combine(target_date, time(hour=hour, minute=minute, tzinfo=timezone.utc))
        return {
            "temporal_start": dt.isoformat(),
            "temporal_end": None,
            "temporal_granularity": "datetime",
            "timezone": timezone_name,
        }

    return {
        "temporal_start": target_date.isoformat(),
        "temporal_end": None,
        "temporal_granularity": granularity,
        "timezone": timezone_name,
    }


def _prepare_text(text: str) -> str:
    normalized = text.replace("’", "'").replace("“", '"').replace("”", '"')
    normalized = re.sub(r"(?m)^>.*$", "", normalized)
    normalized = re.sub(r"(?im)^on .+ wrote:$", "", normalized)
    return " ".join(normalized.split())


def _is_source_eligible(context: GroundingContext, text: str) -> bool:
    if not text.strip():
        return False
    if context.source not in _ALLOWED_SOURCES and context.source_kind not in _ALLOWED_SOURCES:
        return False
    if context.is_ai_response and not ALLOW_AI_AUTHORED_EVIDENCE:
        return False
    if context.sender_id.lower() == "sage" and not ALLOW_AI_AUTHORED_EVIDENCE:
        return False
    return True


def _extract_request_claims(sentence: str, context: GroundingContext, session: Any = None) -> List[Dict[str, Any]]:
    match = REQUEST_PATTERN.match(sentence)
    if not match:
        return []
    object_resolution = _resolve_reference("you", context, session=session, allow_pronouns=True)
    action_text = _normalize_whitespace(match.group("action"))
    claim = _base_claim(
        context,
        sentence,
        claim_type="REQUEST",
        predicate="REQUEST_ACTION",
        subject_resolution=_resolved_sender(context, session=session),
        object_resolution=object_resolution,
        value_text=action_text,
        graph_worthy=False,
        extraction_confidence=0.75,
        canonical_confidence=0.0,
        normalized_text=_normalize_request_text(context, object_resolution, action_text),
    )
    claim["promotion_status"] = "skipped_noncanonical"
    return [claim]


def _extract_reports_to_claims(sentence: str, context: GroundingContext, session: Any = None) -> List[Dict[str, Any]]:
    relation_match = re.search(r"\breports\s+to\b", sentence, re.IGNORECASE)
    if not relation_match:
        return []

    left = sentence[: relation_match.start()].strip(" ,.")
    right = sentence[relation_match.end() :].strip(" ,.")
    left = re.sub(r"\bnow\b$", "", left, flags=re.IGNORECASE).strip(" ,.")
    if not left or not right:
        return []

    subject = _resolve_reference(left, context, session=session, allow_pronouns=False)
    obj = _resolve_reference(right, context, session=session, allow_pronouns=False)
    normalized_text = f"{_resolution_label(subject, fallback=left)} reports to {_resolution_label(obj, fallback=right)}"
    return [
        _base_claim(
            context,
            sentence.strip(),
            claim_type="REPORTS_TO",
            predicate="REPORTS_TO",
            subject_resolution=subject,
            object_resolution=obj,
            value_text=None,
            graph_worthy=True,
            extraction_confidence=0.96,
            canonical_confidence=0.96 if subject.entity_id and obj.entity_id else 0.55,
            normalized_text=normalized_text,
        )
    ]


def _extract_approval_claims(sentence: str, context: GroundingContext, session: Any = None) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for match in APPROVAL_PATTERN.finditer(sentence):
        approver = _resolve_reference(match.group("subject"), context, session=session, allow_pronouns=False)
        approved_target = _normalize_whitespace(match.group("object").rstrip(".?!"))
        target = _resolve_reference(approved_target, context, session=session, allow_pronouns=False)
        target_key = target.key or _slugify(approved_target)
        normalized_text = f"{_resolution_label(target, fallback=target_key)} is approved"
        if approver.key:
            normalized_text += f" by {_resolution_label(approver, fallback=approver.key)}"
        claims.append(
            _base_claim(
                context,
                match.group(0),
                claim_type="APPROVAL_STATE",
                predicate="APPROVED",
                subject_resolution=target,
                object_resolution=approver,
                value_text="approved",
                graph_worthy=True,
                extraction_confidence=0.92,
                canonical_confidence=0.88 if target.key else 0.55,
                normalized_text=normalized_text,
            )
        )
        claims[-1]["payload_json"] = json.dumps(
            {
                "approval_target": target_key,
                "approval_state": "approved",
                "approver_id": approver.entity_id,
                "approver_key": approver.key,
            },
            sort_keys=True,
        )
    for match in PASSIVE_APPROVAL_PATTERN.finditer(sentence):
        approved_target = _normalize_whitespace(match.group("object").rstrip(".?!"))
        target = _resolve_reference(approved_target, context, session=session, allow_pronouns=False)
        target_key = target.key or _slugify(approved_target)
        claim = _base_claim(
            context,
            match.group(0),
            claim_type="APPROVAL_STATE",
            predicate="APPROVED",
            subject_resolution=target,
            object_resolution=None,
            value_text="approved",
            graph_worthy=True,
            extraction_confidence=0.9,
            canonical_confidence=0.86 if target.key else 0.55,
            normalized_text=f"{_resolution_label(target, fallback=target_key)} is approved",
        )
        claim["payload_json"] = json.dumps(
            {
                "approval_target": target_key,
                "approval_state": "approved",
            },
            sort_keys=True,
        )
        claims.append(claim)
    return claims


def _extract_status_claims(sentence: str, context: GroundingContext, session: Any = None) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for match in STATUS_PATTERN.finditer(sentence):
        subject = _resolve_reference(match.group("subject"), context, session=session, allow_pronouns=False)
        status_value = _normalize_whitespace(match.group("status").lower())
        normalized_text = f"{_resolution_label(subject, fallback=_slugify(match.group('subject')))} is {status_value}"
        claims.append(
            _base_claim(
                context,
                match.group(0),
                claim_type="STATUS_UPDATE",
                predicate="STATUS",
                subject_resolution=subject,
                object_resolution=None,
                value_text=status_value,
                graph_worthy=True,
                extraction_confidence=0.86,
                canonical_confidence=0.82 if subject.key else 0.6,
                normalized_text=normalized_text,
            )
        )
    return claims


def _extract_assignment_claims(sentence: str, context: GroundingContext, session: Any = None) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    patterns = (
        (ASSIGNMENT_END_PATTERN, "inactive", "is no longer assigned to"),
        (ASSIGNMENT_START_PATTERN, "active", "is assigned to"),
    )
    for pattern, state, template in patterns:
        for match in pattern.finditer(sentence):
            subject = _resolve_reference(match.group("subject"), context, session=session, allow_pronouns=False)
            assignment_target = _normalize_whitespace(match.group("object").rstrip(".?!"))
            target_resolution = _resolve_reference(assignment_target, context, session=session, allow_pronouns=False)
            normalized_target = target_resolution.key or _slugify(assignment_target)
            normalized_text = (
                f"{_resolution_label(subject, fallback=_slugify(match.group('subject')))} "
                f"{template} {_resolution_label(target_resolution, fallback=normalized_target)}"
            )
            claim = _base_claim(
                context,
                match.group(0),
                claim_type="ASSIGNMENT_STATE",
                predicate="ASSIGNED_TO",
                subject_resolution=subject,
                object_resolution=target_resolution,
                value_text=state,
                graph_worthy=True,
                extraction_confidence=0.9 if state == "inactive" else 0.88,
                canonical_confidence=0.86 if subject.key and target_resolution.key else 0.6,
                normalized_text=normalized_text,
            )
            claim["payload_json"] = json.dumps(
                {
                    "assignment_state": state,
                    "assignment_target": normalized_target,
                },
                sort_keys=True,
            )
            claims.append(claim)
    return claims


def _extract_meeting_claims(sentence: str, context: GroundingContext, session: Any = None) -> List[Dict[str, Any]]:
    event_phrase: Optional[str] = None
    match = MEETING_PATTERN.search(sentence)
    if match:
        event_phrase = _canonicalize_event_phrase(match.group("event"))
    else:
        meet_match = MEET_VERB_PATTERN.search(sentence)
        if meet_match:
            counterparty = _normalize_whitespace(meet_match.group("counterparty") or "")
            event_phrase = "meeting"
            if counterparty:
                event_phrase = f"meeting with {counterparty}"
    if not event_phrase:
        return []
    temporal = normalize_temporal_reference(sentence, context.sent_at, context.timezone)
    subject_raw = None
    if re.search(r"\bwe\b", sentence, re.IGNORECASE) or re.search(r"\blet'?s\b", sentence, re.IGNORECASE):
        subject_raw = "we"
    subject = _group_or_scope_subject(context, session=session, raw=subject_raw)
    normalized_text = event_phrase
    if temporal.get("temporal_start"):
        normalized_text = f"{event_phrase} scheduled for {temporal['temporal_start']}"
    claim = _base_claim(
        context,
        match.group(0) if match else sentence.strip(),
        claim_type="MEETING_EVENT",
        predicate="SCHEDULED_FOR",
        subject_resolution=subject,
        object_resolution=None,
        value_text=event_phrase,
        graph_worthy=True,
        extraction_confidence=0.84,
        canonical_confidence=0.84 if temporal.get("temporal_start") else 0.6,
        normalized_text=normalized_text,
        temporal=temporal,
    )
    claim["payload_json"] = json.dumps({"event_signature": _slugify(event_phrase)}, sort_keys=True)
    return [claim]


def _extract_commitment_claims(sentence: str, context: GroundingContext, session: Any = None) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for match in FIRST_PERSON_COMMITMENT_PATTERN.finditer(sentence):
        claims.extend(
            _commitment_from_match(
                match,
                sentence,
                context,
                session=session,
                subject_resolution=_resolved_sender(context, session=session, raw="I"),
            )
        )
    for match in NAMED_COMMITMENT_PATTERN.finditer(sentence):
        subject = _resolve_reference(match.group("subject"), context, session=session, allow_pronouns=False)
        claims.extend(_commitment_from_match(match, sentence, context, session=session, subject_resolution=subject))
    return claims


def _commitment_from_match(match: re.Match[str], sentence: str, context: GroundingContext, session: Any = None, subject_resolution: Optional[Resolution] = None) -> List[Dict[str, Any]]:
    subject_resolution = subject_resolution or _resolved_sender(context, session=session)
    verb = _normalize_commitment_verb(match.group("verb"))
    body = _normalize_whitespace(match.group("body"))
    body, context_fragments = _extract_commitment_context_fragments(body)
    temporal = normalize_temporal_reference(body, context.sent_at, context.timezone)
    recipient_resolution, recipient_relation = _resolve_commitment_recipient(body, context, session=session)
    item_text = _strip_temporal_tokens(body)
    item_text = _strip_recipient_tokens(
        item_text,
        recipient_raw=recipient_resolution.raw,
        recipient_relation=recipient_relation,
    )
    item_text = _clean_commitment_item_text(item_text)
    item_text = re.sub(r"^(?:the|a|an)\s+", "", item_text, flags=re.IGNORECASE)
    item_text = _normalize_whitespace(item_text)
    if context_fragments:
        item_text = _normalize_whitespace(" ".join([item_text, *context_fragments]))
    if not item_text:
        item_text = verb
    task_signature = _slugify(f"{verb} {item_text}")
    normalized_text = f"{_resolution_label(subject_resolution, fallback=context.sender_id)} will {verb} {item_text}"
    if recipient_resolution.key:
        relation = recipient_relation or _default_commitment_recipient_relation(verb)
        recipient_text = _resolution_label(recipient_resolution, fallback=recipient_resolution.key)
        normalized_text += f" {relation} {recipient_text}" if relation else f" {recipient_text}"
    if temporal.get("temporal_start"):
        normalized_text += f" on {temporal['temporal_start']}"
    claim = _base_claim(
        context,
        match.group(0),
        claim_type="TASK_ASSIGNMENT",
        predicate="TASK_COMMITMENT",
        subject_resolution=subject_resolution,
        object_resolution=recipient_resolution,
        value_text=f"{verb} {item_text}",
        graph_worthy=True,
        extraction_confidence=0.88,
        canonical_confidence=0.88 if subject_resolution.entity_id and temporal.get("temporal_start") else 0.72,
        normalized_text=normalized_text,
        temporal=temporal,
    )
    claim["payload_json"] = json.dumps(
        {
            "task_signature": task_signature,
            "verb": verb,
            "item": item_text,
            "recipient_id": recipient_resolution.entity_id,
            "recipient_key": recipient_resolution.key,
            "recipient_relation": recipient_relation or _default_commitment_recipient_relation(verb),
            "context_fragments": list(context_fragments),
        },
        sort_keys=True,
    )
    return [claim]


def _base_claim(
    context: GroundingContext,
    source_span_text: str,
    *,
    claim_type: str,
    predicate: str,
    subject_resolution: Resolution,
    object_resolution: Optional[Resolution],
    value_text: Optional[str],
    graph_worthy: bool,
    extraction_confidence: float,
    canonical_confidence: float,
    normalized_text: str,
    temporal: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    temporal = temporal or normalize_temporal_reference(source_span_text, context.sent_at, context.timezone)
    grounding = _build_grounding_payload(
        context,
        source_span_text=source_span_text,
        subject_resolution=subject_resolution,
        object_resolution=object_resolution,
        temporal=temporal,
    )
    resolution_status = "resolved"
    if subject_resolution.status != "resolved":
        resolution_status = subject_resolution.status
    if object_resolution and object_resolution.status != "resolved":
        resolution_status = object_resolution.status if resolution_status == "resolved" else "partial"
    if not graph_worthy and resolution_status == "resolved":
        resolution_status = "resolved"
    return {
        "claim_type": claim_type,
        "predicate": predicate,
        "subject_raw": subject_resolution.raw,
        "subject_key": subject_resolution.key,
        "subject_entity_id": subject_resolution.entity_id,
        "subject_entity_type": subject_resolution.entity_type,
        "object_raw": object_resolution.raw if object_resolution else None,
        "object_key": object_resolution.key if object_resolution else None,
        "object_entity_id": object_resolution.entity_id if object_resolution else None,
        "object_entity_type": object_resolution.entity_type if object_resolution else None,
        "grounding_json": json.dumps(grounding, sort_keys=True),
        "value_text": value_text,
        "payload_json": json.dumps({}, sort_keys=True),
        "scope_type": context.scope_type,
        "scope_id": context.scope_id,
        "temporal_start": temporal.get("temporal_start"),
        "temporal_end": temporal.get("temporal_end"),
        "temporal_granularity": temporal.get("temporal_granularity"),
        "timezone": temporal.get("timezone") or context.timezone,
        "normalized_text": normalized_text,
        "source_span_text": source_span_text,
        "graph_worthy": graph_worthy,
        "resolution_status": resolution_status,
        "promotion_status": "pending",
        "mutation_action": "awaiting_decision",
        "extraction_confidence": extraction_confidence,
        "canonical_confidence": canonical_confidence,
        "created_at": _utcnow_iso(),
    }


def _resolve_reference(raw: str, context: GroundingContext, session: Any = None, *, allow_pronouns: bool) -> Resolution:
    token = _normalize_whitespace(raw)
    canonical_token = re.sub(r"^(?:the|a|an)\s+", "", token, flags=re.IGNORECASE).strip() or token
    lowered = canonical_token.lower()
    if allow_pronouns and context.source_kind in {"chat_message", "message_attachment"}:
        # TODO(agentic): Neutral anaphora such as "it", "this", "that", and "they"
        # should be resolved by a future planner/coreference agent, not this lexical resolver.
        if lowered in NEUTRAL_ANAPHORA_TOKENS:
            return Resolution(raw=raw, key=None, entity_id=None, entity_type=None, status="unresolved", display_name=None)
        if lowered in FIRST_PERSON_TOKENS:
            return _resolved_sender(context, session=session, raw=raw)
        if lowered in SECOND_PERSON_TOKENS:
            if context.conversation_type == "direct" and len(context.receiver_ids) == 1:
                return _resolve_person_identity(context.receiver_ids[0], session=session, raw=raw)
            return Resolution(raw=raw, key=None, entity_id=None, entity_type=None, status="unresolved", display_name=None)
        if lowered in FIRST_PERSON_PLURAL_TOKENS:
            if context.conversation_type == "group" and context.group_id:
                return Resolution(
                    raw=raw,
                    key=context.group_id,
                    entity_id=context.group_id,
                    entity_type="Group",
                    status="resolved",
                    display_name=_lookup_entity_display_name(session, context.group_id),
                )
            return Resolution(raw=raw, key=None, entity_id=None, entity_type=None, status="unresolved", display_name=None)

    if re.fullmatch(r"EMP\d{3}", canonical_token, flags=re.IGNORECASE):
        canonical = canonical_token.upper()
        return Resolution(
            raw=raw,
            key=canonical,
            entity_id=canonical,
            entity_type="Person",
            status="resolved",
            display_name=_lookup_entity_display_name(session, canonical),
        )

    if canonical_token == context.sender_id:
        return _resolved_sender(context, session=session, raw=raw)
    if canonical_token in context.receiver_ids:
        return _resolve_person_identity(canonical_token, session=session, raw=raw)
    if context.group_id and canonical_token == context.group_id:
        return Resolution(
            raw=raw,
            key=canonical_token,
            entity_id=canonical_token,
            entity_type="Group",
            status="resolved",
            display_name=_lookup_entity_display_name(session, canonical_token),
        )

    if session is not None:
        row = _lookup_person_records(session, canonical_token)
        preferred = _select_preferred_person_record(row, canonical_token)
        if preferred is not None:
            labels = preferred.get("labels") or []
            entity_type = labels[0] if labels else "Person"
            entity_id = preferred.get("id")
            return Resolution(
                raw=raw,
                key=entity_id,
                entity_id=entity_id,
                entity_type=entity_type,
                status="resolved",
                display_name=preferred.get("display_name") or entity_id,
            )
        if len(row) > 1:
            return Resolution(raw=raw, key=None, entity_id=None, entity_type=None, status="ambiguous", display_name=None)

    slug = _slugify(canonical_token)
    return Resolution(
        raw=raw,
        key=slug,
        entity_id=None,
        entity_type=None,
        status="resolved",
        display_name=_humanize_entity_label(slug),
    )


def _resolve_commitment_recipient(body: str, context: GroundingContext, session: Any = None) -> Tuple[Resolution, Optional[str]]:
    body_lower = body.lower()
    if re.match(r"^\s*you\b", body_lower):
        return _resolve_reference("you", context, session=session, allow_pronouns=True), None
    pronoun_match = re.search(r"\b(?P<relation>to|with|for)\s+you\b", body_lower)
    if pronoun_match:
        return _resolve_reference("you", context, session=session, allow_pronouns=True), pronoun_match.group("relation").lower()
    leading_name = re.match(
        r"^\s*(?P<name>[A-Z][A-Za-z0-9_\-]*(?:\s+[A-Z][A-Za-z0-9_\-]*)*|EMP\d{3})\b",
        body,
    )
    if leading_name:
        return _resolve_reference(leading_name.group("name"), context, session=session, allow_pronouns=False), None
    direct_name = re.search(
        r"\b(?P<relation>to|with|for)\s+(?P<name>[A-Z][A-Za-z0-9_\-]*(?:\s+[A-Z][A-Za-z0-9_\-]*)*|EMP\d{3})\b",
        body,
    )
    if direct_name:
        return (
            _resolve_reference(direct_name.group("name"), context, session=session, allow_pronouns=False),
            direct_name.group("relation").lower(),
        )
    return Resolution(raw="", key=None, entity_id=None, entity_type=None, status="resolved", display_name=None), None


def _resolved_sender(context: GroundingContext, session: Any = None, raw: Optional[str] = None) -> Resolution:
    resolved = _resolve_person_identity(context.sender_id, session=session, raw=raw or context.sender_id)
    if resolved.entity_id or resolved.key != context.sender_id:
        return resolved
    return Resolution(
        raw=raw or context.sender_id,
        key=context.sender_id,
        entity_id=context.sender_id,
        entity_type="Person",
        status="resolved",
        display_name=_lookup_entity_display_name(session, context.sender_id),
    )


def _group_or_scope_subject(context: GroundingContext, session: Any = None, raw: Optional[str] = None) -> Resolution:
    if context.conversation_type == "group" and context.group_id:
        return Resolution(
            raw=raw or context.group_id,
            key=context.group_id,
            entity_id=context.group_id,
            entity_type="Group",
            status="resolved",
            display_name=_lookup_entity_display_name(session, context.group_id),
        )
    if context.sender_id:
        return _resolved_sender(context, session=session, raw=raw)
    return Resolution(
        raw=raw or context.scope_id,
        key=context.scope_id,
        entity_id=None,
        entity_type=None,
        status="resolved",
        display_name=_humanize_entity_label(context.scope_id),
    )


def _build_canonical_key(claim: Dict[str, Any]) -> str:
    claim_type = claim["claim_type"]
    subject_key = claim.get("subject_entity_id") or claim.get("subject_key") or "unknown"
    object_key = claim.get("object_entity_id") or claim.get("object_key") or "unknown"
    if claim_type == "REPORTS_TO":
        return f"reports_to::{subject_key}"
    if claim_type == "APPROVAL_STATE":
        payload = _load_payload(claim)
        target = payload.get("approval_target") or subject_key or object_key or _slugify(claim.get("value_text") or "approval")
        return f"approval::{target}"
    if claim_type == "TASK_ASSIGNMENT":
        payload = _load_payload(claim)
        task_signature = payload.get("task_signature") or _slugify(claim.get("value_text") or "task")
        return f"assignment::{claim.get('scope_id')}::{task_signature}"
    if claim_type == "ASSIGNMENT_STATE":
        return f"assignment_state::{subject_key}::{object_key}"
    if claim_type == "MEETING_EVENT":
        payload = _load_payload(claim)
        event_signature = payload.get("event_signature") or _slugify(claim.get("value_text") or "meeting")
        return f"meeting::{claim.get('scope_id')}::{event_signature}"
    if claim_type == "STATUS_UPDATE":
        return f"status::{subject_key}::general"
    return f"claim::{claim_type.lower()}::{subject_key}::{object_key}"


def _claim_dedupe_key(claim: Dict[str, Any]) -> str:
    parts = [
        claim["claim_type"],
        claim.get("predicate") or "",
        claim.get("subject_key") or claim.get("subject_raw") or "",
        claim.get("object_key") or claim.get("object_raw") or "",
        claim.get("value_text") or "",
        claim.get("temporal_start") or "",
        claim.get("normalized_text") or "",
    ]
    return "|".join(parts)


def _make_claim_id(context: GroundingContext, claim: Dict[str, Any]) -> str:
    basis = "::".join(
        [
            context.source_doc_id,
            claim.get("source_span_text") or "",
            claim.get("claim_type") or "",
            claim.get("predicate") or "",
            claim.get("subject_key") or "",
            claim.get("object_key") or "",
            claim.get("value_text") or "",
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _make_run_id(context: GroundingContext) -> str:
    seed = f"{context.source_doc_id}:{context.source_message_id}:{_utcnow_iso()}"
    return f"saia-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _persist_claim(session: Any, context: GroundingContext, claim: Dict[str, Any]) -> None:
    session.run(
        """
        MERGE (c:Claim {claim_id: $claim_id})
        SET c.claim_type = $claim_type,
            c.predicate = $predicate,
            c.subject_raw = $subject_raw,
            c.subject_key = $subject_key,
            c.subject_entity_id = $subject_entity_id,
            c.subject_entity_type = $subject_entity_type,
            c.object_raw = $object_raw,
            c.object_key = $object_key,
            c.object_entity_id = $object_entity_id,
            c.object_entity_type = $object_entity_type,
            c.value_text = $value_text,
            c.grounding_json = $grounding_json,
            c.payload_json = $payload_json,
            c.scope_type = $scope_type,
            c.scope_id = $scope_id,
            c.temporal_start = $temporal_start,
            c.temporal_end = $temporal_end,
            c.temporal_granularity = $temporal_granularity,
            c.timezone = $timezone,
            c.normalized_text = $normalized_text,
            c.source_span_text = $source_span_text,
            c.graph_worthy = $graph_worthy,
            c.resolution_status = $resolution_status,
            c.promotion_status = $promotion_status,
            c.mutation_action = $mutation_action,
            c.extraction_confidence = $extraction_confidence,
            c.canonical_confidence = $canonical_confidence,
            c.canonical_key = $canonical_key,
            c.created_at = $created_at
        """,
        **claim,
    )
    session.run(
        """
        MATCH (d:Document {doc_id: $doc_id})
        MATCH (c:Claim {claim_id: $claim_id})
        MERGE (d)-[:HAS_CLAIM]->(c)
        """,
        doc_id=context.source_doc_id,
        claim_id=claim["claim_id"],
    )


def _mark_claim_promotion(session: Any, claim_id: str, promotion_status: str, mutation_action: Optional[str] = None) -> None:
    session.run(
        """
        MATCH (c:Claim {claim_id: $claim_id})
        SET c.promotion_status = $promotion_status,
            c.mutation_action = $mutation_action
        """,
        claim_id=claim_id,
        promotion_status=promotion_status,
        mutation_action=mutation_action,
    )


def _persist_fact(session: Any, claim: Dict[str, Any]) -> str:
    fact_id = hashlib.sha256(f"fact::{claim['claim_id']}::{claim['canonical_key']}".encode("utf-8")).hexdigest()
    summary = claim.get("normalized_text") or claim.get("value_text") or claim.get("predicate")
    embedding = utils.generate_embedding(summary[:5000])
    session.run(
        """
        MERGE (f:CanonicalFact {fact_id: $fact_id})
        SET f.canonical_key = $canonical_key,
            f.claim_type = $claim_type,
            f.predicate = $predicate,
            f.subject_key = $subject_key,
            f.subject_entity_id = $subject_entity_id,
            f.object_key = $object_key,
            f.object_entity_id = $object_entity_id,
            f.value_text = $value_text,
            f.payload_json = $payload_json,
            f.summary = $summary,
            f.embedding = $embedding,
            f.scope_type = $scope_type,
            f.scope_id = $scope_id,
            f.temporal_start = $temporal_start,
            f.temporal_end = $temporal_end,
            f.temporal_granularity = $temporal_granularity,
            f.timezone = $timezone,
            f.status = 'current',
            f.confidence = $confidence,
            f.first_seen_at = $seen_at,
            f.last_seen_at = $seen_at,
            f.support_count = coalesce(f.support_count, 0) + 1
        """,
        fact_id=fact_id,
        canonical_key=claim["canonical_key"],
        claim_type=claim["claim_type"],
        predicate=claim["predicate"],
        subject_key=claim.get("subject_key"),
        subject_entity_id=claim.get("subject_entity_id"),
        object_key=claim.get("object_key"),
        object_entity_id=claim.get("object_entity_id"),
        value_text=claim.get("value_text"),
        payload_json=claim.get("payload_json"),
        summary=summary,
        embedding=embedding,
        scope_type=claim.get("scope_type"),
        scope_id=claim.get("scope_id"),
        temporal_start=claim.get("temporal_start"),
        temporal_end=claim.get("temporal_end"),
        temporal_granularity=claim.get("temporal_granularity"),
        timezone=claim.get("timezone"),
        confidence=claim.get("canonical_confidence"),
        seen_at=_utcnow_iso(),
    )
    if claim.get("subject_entity_id"):
        if claim.get("subject_entity_type") == "Group":
            session.run(
                """
                MERGE (g:Group {id: $subject_id})
                MERGE (f:CanonicalFact {fact_id: $fact_id})
                MERGE (g)-[:HAS_FACT]->(f)
                """,
                subject_id=claim["subject_entity_id"],
                fact_id=fact_id,
            )
        else:
            session.run(
                """
                MERGE (p:Person {id: $subject_id})
                MERGE (f:CanonicalFact {fact_id: $fact_id})
                MERGE (p)-[:HAS_FACT]->(f)
                """,
                subject_id=claim["subject_entity_id"],
                fact_id=fact_id,
            )
    if claim.get("object_entity_id") and claim.get("object_entity_type") != "Group":
        session.run(
            """
            MERGE (p:Person {id: $object_id})
            MERGE (f:CanonicalFact {fact_id: $fact_id})
            MERGE (f)-[:OBJECT_ENTITY]->(p)
            """,
            object_id=claim["object_entity_id"],
            fact_id=fact_id,
        )
    return fact_id


def _link_claim_to_fact(session: Any, claim_id: str, fact_id: str, *, relation_type: str) -> None:
    query = (
        f"MATCH (c:Claim {{claim_id: $claim_id}}) MATCH (f:CanonicalFact {{fact_id: $fact_id}}) MERGE (c)-[:{relation_type}]->(f)"
    )
    session.run(query, claim_id=claim_id, fact_id=fact_id)


def _touch_existing_fact(session: Any, fact_id: str) -> None:
    session.run(
        "MATCH (f:CanonicalFact {fact_id: $fact_id}) SET f.last_seen_at = $seen_at, f.support_count = coalesce(f.support_count, 0) + 1",
        fact_id=fact_id,
        seen_at=_utcnow_iso(),
    )


def _supersede_existing_fact(session: Any, existing_fact_id: str, replacement_fact_id: str) -> None:
    now = _utcnow_iso()
    session.run(
        """
        MATCH (old:CanonicalFact {fact_id: $existing_fact_id})
        MATCH (replacement:CanonicalFact {fact_id: $replacement_fact_id})
        SET old.status = 'superseded',
            old.superseded_at = $superseded_at,
            old.superseded_by_fact_id = $replacement_fact_id
        MERGE (old)-[:SUPERSEDED_BY]->(replacement)
        """,
        existing_fact_id=existing_fact_id,
        replacement_fact_id=replacement_fact_id,
        superseded_at=now,
    )


def _load_current_facts(session: Any, canonical_key: str) -> List[Dict[str, Any]]:
    rows = session.run(
        "MATCH (f:CanonicalFact {canonical_key: $canonical_key, status: 'current'}) RETURN f",
        canonical_key=canonical_key,
    ).data()
    return [_serialize_node(row.get("f")) for row in rows]


def _facts_match(fact: Dict[str, Any], claim: Dict[str, Any]) -> bool:
    if claim.get("claim_type") == "APPROVAL_STATE":
        return (
            (fact.get("canonical_key") or "") == (claim.get("canonical_key") or "")
            and (fact.get("value_text") or "") == (claim.get("value_text") or "")
        )
    comparable_keys = (
        "claim_type",
        "predicate",
        "subject_entity_id",
        "subject_key",
        "object_entity_id",
        "object_key",
        "value_text",
        "temporal_start",
        "temporal_granularity",
    )
    return all((fact.get(key) or "") == (claim.get(key) or "") for key in comparable_keys)


def collect_message_insight(session: Any, message_id: str) -> Dict[str, Any]:
    chat_doc_id = f"chat-msg-{message_id}"
    message_rows = session.run(
        """
        MATCH (m:Message {id: $message_id})
        RETURN m
        LIMIT 1
        """,
        message_id=message_id,
    ).data()
    if not message_rows:
        raise ValueError("Message not found")

    message = _serialize_node(message_rows[0].get("m"))

    document_rows = session.run(
        """
        MATCH (m:Message {id: $message_id})-[:HAS_EVIDENCE_DOCUMENT]->(d:Document)
        WHERE d.doc_id = $chat_doc_id
           OR d.origin_message_id = $message_id
           OR d.linked_message_id = $message_id
        RETURN DISTINCT d
        ORDER BY d.timestamp ASC
        """,
        message_id=message_id,
        chat_doc_id=chat_doc_id,
    ).data()
    source_documents = [_serialize_node(row.get("d")) for row in document_rows if row.get("d") is not None]

    claim_rows = session.run(
        """
        MATCH (m:Message {id: $message_id})-[:HAS_EVIDENCE_DOCUMENT]->(d:Document)-[:HAS_CLAIM]->(c:Claim)
        WHERE d.doc_id = $chat_doc_id
           OR d.origin_message_id = $message_id
           OR d.linked_message_id = $message_id
        RETURN DISTINCT d.doc_id AS source_doc_id, c
        ORDER BY c.created_at ASC
        """,
        message_id=message_id,
        chat_doc_id=chat_doc_id,
    ).data()
    claims: List[Dict[str, Any]] = []
    claims_by_id: Dict[str, Dict[str, Any]] = {}
    for row in claim_rows:
        claim = _serialize_node(row.get("c"))
        if not claim:
            continue
        claim["source_doc_id"] = row.get("source_doc_id")
        claim["facts"] = []
        claim["grounding"] = _load_json_blob(claim.get("grounding_json"))
        claims.append(claim)
        if claim.get("claim_id"):
            claims_by_id[claim["claim_id"]] = claim

    fact_link_rows = session.run(
        """
        MATCH (m:Message {id: $message_id})-[:HAS_EVIDENCE_DOCUMENT]->(d:Document)-[:HAS_CLAIM]->(c:Claim)-[rel:SUPPORTS|CONTRADICTS]->(f:CanonicalFact)
        WHERE d.doc_id = $chat_doc_id
           OR d.origin_message_id = $message_id
           OR d.linked_message_id = $message_id
        RETURN DISTINCT c.claim_id AS claim_id, type(rel) AS relation_type, f
        """,
        message_id=message_id,
        chat_doc_id=chat_doc_id,
    ).data()
    canonical_facts: List[Dict[str, Any]] = []
    canonical_facts_by_id: Dict[str, Dict[str, Any]] = {}
    for row in fact_link_rows:
        claim = claims_by_id.get(row.get("claim_id"))
        fact = _serialize_node(row.get("f"))
        if not claim or not fact:
            continue
        fact_id = fact.get("fact_id")
        fact_link = {
            "relation_type": row.get("relation_type"),
            "fact_id": fact_id,
            "canonical_key": fact.get("canonical_key"),
            "summary": fact.get("summary"),
            "status": fact.get("status"),
            "support_count": fact.get("support_count"),
            "superseded_by_fact_id": fact.get("superseded_by_fact_id"),
            "superseded_at": fact.get("superseded_at"),
        }
        claim["facts"].append(fact_link)
        if fact_id and fact_id not in canonical_facts_by_id:
            canonical_facts_by_id[fact_id] = fact
            canonical_facts.append(fact)

    replacement_rows = session.run(
        """
        MATCH (m:Message {id: $message_id})-[:HAS_EVIDENCE_DOCUMENT]->(d:Document)-[:HAS_CLAIM]->(c:Claim)-[:SUPPORTS]->(replacement:CanonicalFact)
        WHERE d.doc_id = $chat_doc_id
           OR d.origin_message_id = $message_id
           OR d.linked_message_id = $message_id
        MATCH (previous:CanonicalFact)-[:SUPERSEDED_BY]->(replacement)
        RETURN DISTINCT c.claim_id AS claim_id, previous, replacement
        """,
        message_id=message_id,
        chat_doc_id=chat_doc_id,
    ).data()
    replacements: List[Dict[str, Any]] = []
    for row in replacement_rows:
        previous = _serialize_node(row.get("previous"))
        replacement = _serialize_node(row.get("replacement"))
        if not previous or not replacement:
            continue
        replacements.append(
            {
                "claim_id": row.get("claim_id"),
                "previous_fact_id": previous.get("fact_id"),
                "previous_summary": previous.get("summary"),
                "previous_status": previous.get("status"),
                "replacement_fact_id": replacement.get("fact_id"),
                "replacement_summary": replacement.get("summary"),
                "canonical_key": replacement.get("canonical_key"),
                "superseded_at": previous.get("superseded_at"),
                "_previous_fact": previous,
                "_replacement_fact": replacement,
            }
        )

    run_rows = session.run(
        """
        MATCH (m:Message {id: $message_id})-[:HAS_EVIDENCE_DOCUMENT]->(d:Document)-[:PROCESSED_BY_SAIA]->(r:SAIARun)
        WHERE d.doc_id = $chat_doc_id
           OR d.origin_message_id = $message_id
           OR d.linked_message_id = $message_id
        RETURN DISTINCT d.doc_id AS source_doc_id, r
        ORDER BY r.processed_at ASC
        """,
        message_id=message_id,
        chat_doc_id=chat_doc_id,
    ).data()
    runs: List[Dict[str, Any]] = []
    for row in run_rows:
        run = _serialize_node(row.get("r"))
        if not run:
            continue
        run["source_doc_id"] = row.get("source_doc_id")
        if run.get("errors_json"):
            try:
                run["errors"] = json.loads(run["errors_json"])
            except json.JSONDecodeError:
                run["errors"] = {"raw": run["errors_json"]}
        runs.append(run)

    preview_claims: List[Dict[str, Any]] = []
    if not claims:
        preview_claims = _preview_message_claims(session, message_id, message)

    entity_display_names = _load_entity_display_names(
        session,
        _collect_entity_ids(claims, preview_claims, canonical_facts, replacements),
    )
    for claim in claims:
        _decorate_claim_for_insight(claim, entity_display_names)
    for claim in preview_claims:
        _decorate_claim_for_insight(claim, entity_display_names)
    for fact in canonical_facts:
        _decorate_fact_for_insight(fact, entity_display_names)
    fact_display_map = {
        fact.get("fact_id"): fact
        for fact in canonical_facts
        if fact.get("fact_id")
    }
    for claim in claims:
        for fact_link in claim.get("facts") or []:
            decorated_fact = fact_display_map.get(fact_link.get("fact_id"))
            if decorated_fact:
                fact_link["summary"] = decorated_fact.get("display_summary") or fact_link.get("summary")
                fact_link["subject_display"] = decorated_fact.get("subject_display")
                fact_link["object_display"] = decorated_fact.get("object_display")
    for replacement in replacements:
        previous_fact = replacement.pop("_previous_fact", None)
        replacement_fact = replacement.pop("_replacement_fact", None)
        if previous_fact:
            replacement["previous_display_summary"] = _render_record_display_text(previous_fact, entity_display_names)
        if replacement_fact:
            replacement["replacement_display_summary"] = _render_record_display_text(replacement_fact, entity_display_names)

    return {
        "message_id": message_id,
        "message_source": message.get("source"),
        "saia_status": message.get("saia_status") or ("disabled" if not is_enabled() else None),
        "saia_processed_at": message.get("saia_processed_at"),
        "saia_error": message.get("saia_error")
        or ("SAIA processing is disabled in backend configuration." if not is_enabled() else None),
        "source_documents": source_documents,
        "runs": runs,
        "claims": claims,
        "preview_claims": preview_claims,
        "canonical_facts": canonical_facts,
        "replacements": replacements,
        "summary": {
            "document_count": len(source_documents),
            "run_count": len(runs),
            "claim_count": len(claims),
            "preview_claim_count": len(preview_claims),
            "canonical_fact_count": len(canonical_facts),
            "replacement_count": len(replacements),
        },
    }


def _persist_run(session: Any, run_id: str, context: GroundingContext, result: Dict[str, Any]) -> None:
    session.run(
        """
        MERGE (r:SAIARun {run_id: $run_id})
        SET r.source_doc_id = $source_doc_id,
            r.source_message_id = $source_message_id,
            r.source_kind = $source_kind,
            r.status = $status,
            r.processed_at = $processed_at,
            r.claims_extracted = $claims_extracted,
            r.claims_canonicalized = $claims_canonicalized,
            r.conflicts_found = $conflicts_found,
            r.errors_json = $errors_json
        """,
        run_id=run_id,
        source_doc_id=context.source_doc_id,
        source_message_id=context.source_message_id,
        source_kind=context.source_kind,
        status=result.get("status"),
        processed_at=_utcnow_iso(),
        claims_extracted=result.get("claims_extracted", 0),
        claims_canonicalized=result.get("claims_canonicalized", 0),
        conflicts_found=result.get("conflicts_found", 0),
        errors_json=json.dumps({"reason": result.get("reason")}) if result.get("reason") else None,
    )
    session.run(
        """
        MATCH (d:Document {doc_id: $doc_id})
        MATCH (r:SAIARun {run_id: $run_id})
        MERGE (d)-[:PROCESSED_BY_SAIA]->(r)
        """,
        doc_id=context.source_doc_id,
        run_id=run_id,
    )


def _finalize_run(session: Any, context: GroundingContext, result: Dict[str, Any], *, reason: Optional[str]) -> None:
    result["reason"] = reason
    _persist_run(session, _make_run_id(context), context, result)
    _mark_source_status(session, context, result["status"], reason)


def _mark_source_status(session: Any, context: GroundingContext, status: str, error: Optional[str]) -> None:
    processed_at = _utcnow_iso()
    session.run(
        """
        MATCH (d:Document {doc_id: $doc_id})
        SET d.saia_status = $status,
            d.saia_processed_at = $processed_at,
            d.saia_error = $error
        """,
        doc_id=context.source_doc_id,
        status=status,
        processed_at=processed_at,
        error=error,
    )
    if context.source_message_id:
        session.run(
            """
            MATCH (m:Message {id: $message_id})
            SET m.saia_status = $status,
                m.saia_processed_at = $processed_at,
                m.saia_error = $error
            """,
            message_id=context.source_message_id,
            status=status,
            processed_at=processed_at,
            error=error,
        )


def _link_message_to_document(session: Any, context: GroundingContext) -> None:
    if not context.source_message_id:
        return
    session.run(
        """
        MATCH (m:Message {id: $message_id})
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (m)-[:HAS_EVIDENCE_DOCUMENT]->(d)
        """,
        message_id=context.source_message_id,
        doc_id=context.source_doc_id,
    )


def _build_result(status: str, claims: Sequence[Dict[str, Any]], canonicalized: int, conflicts: int) -> Dict[str, Any]:
    return {
        "status": status,
        "claims_extracted": len(claims),
        "claims_canonicalized": canonicalized,
        "conflicts_found": conflicts,
    }


def _normalize_request_text(context: GroundingContext, object_resolution: Resolution, action_text: str) -> str:
    target = object_resolution.key or object_resolution.raw or "recipient"
    return f"{context.sender_id} requested {target} to {action_text}"


def _split_claim_spans(text: str) -> List[str]:
    fragments = [fragment.strip() for fragment in re.split(r"(?<=[.!?])\s+", text) if fragment.strip()]
    spans: List[str] = []
    for fragment in fragments:
        cleaned = fragment.strip().rstrip(".!?").strip()
        if not cleaned:
            continue
        if spans and _is_contextual_continuation(cleaned):
            spans[-1] = f"{spans[-1]}; {cleaned}"
            continue
        spans.append(cleaned)
    return spans


def _is_contextual_continuation(fragment: str) -> bool:
    tokens = _normalize_whitespace(fragment).split()
    if len(tokens) < 3:
        return False
    if tokens[0].lower() not in CONTEXTUAL_CONTINUATION_PREFIXES:
        return False
    lowered = " ".join(tokens).lower()
    if re.search(r"\b(?:i|you|we|they|he|she|it|can|could|would|will|should|must|reports|approved|assigned|meeting|meet)\b", lowered):
        return False
    return True


def _extract_commitment_context_fragments(body: str) -> Tuple[str, List[str]]:
    parts = [_normalize_whitespace(part) for part in re.split(r"\s*;\s*", body) if _normalize_whitespace(part)]
    if not parts:
        return body, []

    main_body = parts[0]
    context_fragments: List[str] = []
    for fragment in parts[1:]:
        qualifier_match = re.match(
            r"^(?P<prefix>for|regarding|about|re|under|within|on)\s+(?P<context>[A-Za-z0-9][^;,.!?]*)$",
            fragment,
            flags=re.IGNORECASE,
        )
        if qualifier_match:
            prefix = qualifier_match.group("prefix").lower()
            context = _normalize_whitespace(qualifier_match.group("context"))
            if context:
                context_fragments.append(f"{prefix} {context}")
            continue
        main_body = _normalize_whitespace(f"{main_body} {fragment}")

    return main_body, context_fragments


def _canonicalize_event_phrase(text: str) -> str:
    tokens = _normalize_whitespace(text).split()
    filler_tokens = {"we", "i", "they", "have", "has", "had", "a", "an", "the"}
    while len(tokens) > 1 and tokens[0].lower() in filler_tokens:
        tokens = tokens[1:]
    return " ".join(tokens) or "meeting"


def _load_payload(claim: Dict[str, Any]) -> Dict[str, Any]:
    return _load_json_blob(claim.get("payload_json"))


def _load_json_blob(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}


def _preview_message_claims(session: Any, message_id: str, message: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = _prepare_text(message.get("content") or "")
    if not content:
        return []

    receiver_ids = []
    if message.get("receiver_id"):
        receiver_ids.append(str(message["receiver_id"]))
    if not receiver_ids:
        receiver_rows = session.run(
            """
            MATCH (d:Document {doc_id: $doc_id})-[:RECEIVED_BY]->(r:Person)
            RETURN DISTINCT r.id AS id
            ORDER BY r.id ASC
            """,
            doc_id=f"chat-msg-{message_id}",
        ).data()
        receiver_ids = [str(row.get("id")) for row in receiver_rows if row.get("id")]

    context = GroundingContext(
        source_kind="chat_message",
        source_doc_id=f"chat-msg-{message_id}",
        source_message_id=message_id,
        linked_message_id=None,
        sender_id=str(message.get("sender_id") or ""),
        receiver_ids=receiver_ids,
        conversation_id=message.get("conversation_id"),
        conversation_type=message.get("conversation_type"),
        group_id=message.get("group_id"),
        sent_at=message.get("sent_at") or _utcnow_iso(),
        source=message.get("source") or "chat_message",
        is_ai_response=bool(message.get("is_ai_response")),
        attachment_name=message.get("attachment_name"),
    )
    preview_claims = extract_claims_from_text(content, context, session=session)
    for claim in preview_claims:
        claim["facts"] = []
        claim["preview_only"] = True
        claim["source_doc_id"] = context.source_doc_id
        claim["grounding"] = _load_json_blob(claim.get("grounding_json"))
    return preview_claims


def _strip_temporal_tokens(text: str) -> str:
    without_dates = re.sub(
        r"\b(?:by|before|after|at|on|around)\s+(?=(?:today|tomorrow|yesterday|now|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|in\s+\d+\s+(?:day|days|week|weeks)|\d{4}-\d{2}-\d{2})\b)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    without_dates = re.sub(
        r"\b(?:by|before|after|at|on|around)\s+(?=\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)",
        "",
        without_dates,
        flags=re.IGNORECASE,
    )
    without_dates = TIME_WORD_PATTERN.sub("", without_dates)
    without_dates = CLOCK_PATTERN.sub("", without_dates)
    without_dates = AT_CLOCK_PATTERN.sub("", without_dates)
    return _normalize_whitespace(without_dates)


def _strip_recipient_tokens(
    text: str,
    *,
    recipient_raw: Optional[str] = None,
    recipient_relation: Optional[str] = None,
) -> str:
    stripped = text
    if recipient_raw:
        normalized_raw = _normalize_whitespace(recipient_raw)
        stripped = re.sub(
            rf"^\s*{re.escape(normalized_raw)}\b",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        if recipient_relation:
            stripped = re.sub(
                rf"\b{re.escape(recipient_relation)}\s+{re.escape(normalized_raw)}\b",
                "",
                stripped,
                flags=re.IGNORECASE,
            )
        stripped = re.sub(
            rf"\b(?:to|with|for)\s+{re.escape(normalized_raw)}\b",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
    else:
        stripped = re.sub(r"^\s*you\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\b(?:to|with|for)\s+you\b", "", stripped, flags=re.IGNORECASE)
    return _normalize_whitespace(stripped)


def _clean_commitment_item_text(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    cleaned = re.sub(
        r"\b(?:by|at|on|before|after|around)\s+(for|regarding|about|re|under|within|on)\b",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )
    while cleaned:
        updated = re.sub(r"(?:,\s*)?\binstead\b\s*$", "", cleaned, flags=re.IGNORECASE)
        updated = re.sub(r"\b(?:by|before|after|at|on|to|for|with|around)\b\s*$", "", updated, flags=re.IGNORECASE)
        updated = re.sub(r"^\s*(?:to|for|with)\b", "", updated, flags=re.IGNORECASE)
        updated = _normalize_whitespace(updated)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = re.sub(r"\b(For|Regarding|About|Re|Under|Within|On)\b", lambda match: match.group(0).lower(), cleaned)
    return cleaned


def _normalize_commitment_verb(raw: str) -> str:
    return COMMITMENT_VERB_MAP.get((raw or "").lower(), (raw or "").lower())


def _default_commitment_recipient_relation(verb: str) -> Optional[str]:
    if verb in {"send", "share", "provide", "deliver"}:
        return "to"
    if verb in {"review", "discuss", "confirm"}:
        return "with"
    return None


def _resolution_label(resolution: Optional[Resolution], *, fallback: Optional[str] = None) -> str:
    if resolution is None:
        return fallback or "unknown"
    if resolution.display_name:
        return resolution.display_name
    if resolution.key:
        return _humanize_entity_label(resolution.key) or resolution.key
    if fallback:
        return _humanize_entity_label(fallback) or fallback
    return _humanize_entity_label(resolution.raw) or resolution.raw


def _humanize_entity_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    text = str(value).strip()
    if not text:
        return text
    if "@" in text or re.fullmatch(r"[A-Za-z]*\d+[A-Za-z0-9_\-]*", text):
        return text
    if "-" in text or "_" in text:
        parts = [part for part in re.split(r"[-_]+", text) if part]
        return " ".join(part.capitalize() if part.islower() else part for part in parts)
    if text.islower():
        return " ".join(token.capitalize() for token in text.split())
    return text


def _lookup_person_records(session: Any, value: Optional[str]) -> List[Dict[str, Any]]:
    if session is None or not value:
        return []
    return session.run(
        """
        MATCH (p:Person)
        WHERE toLower(p.id) = toLower($value)
           OR toLower(coalesce(p.email, '')) = toLower($value)
           OR toLower(coalesce(p.name, '')) = toLower($value)
        RETURN p.id AS id,
               labels(p) AS labels,
               p.name AS name,
               p.email AS email,
               coalesce(p.name, p.email, p.id) AS display_name
        LIMIT 5
        """,
        value=value,
    ).data()


def _select_preferred_person_record(rows: List[Dict[str, Any]], value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]

    normalized_value = _normalize_whitespace(value or "").lower()
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for row in rows:
        labels = {str(label) for label in (row.get("labels") or [])}
        entity_id = str(row.get("id") or "")
        name = str(row.get("name") or "")
        email = str(row.get("email") or "")
        display_name = str(row.get("display_name") or "")

        score = 0
        if entity_id.lower() == normalized_value:
            score += 6
        if email.lower() == normalized_value:
            score += 5
        if name.lower() == normalized_value or display_name.lower() == normalized_value:
            score += 4
        if "User" in labels:
            score += 3
        if email:
            score += 1
        if name:
            score += 1
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    if best_score <= 0:
        return None

    best_rows = [row for score, row in scored if score == best_score]
    if len(best_rows) != 1:
        return None
    return best_rows[0]


def _resolve_person_identity(value: Optional[str], session: Any = None, raw: Optional[str] = None) -> Resolution:
    normalized_value = _normalize_whitespace(value or "")
    raw_value = raw or normalized_value
    if session is not None:
        rows = _lookup_person_records(session, normalized_value)
        preferred = _select_preferred_person_record(rows, normalized_value)
        if preferred is not None:
            labels = preferred.get("labels") or []
            entity_type = labels[0] if labels else "Person"
            entity_id = preferred.get("id") or normalized_value
            return Resolution(
                raw=raw_value,
                key=entity_id,
                entity_id=entity_id,
                entity_type=entity_type,
                status="resolved",
                display_name=preferred.get("display_name") or entity_id,
            )
    return Resolution(
        raw=raw_value,
        key=normalized_value or None,
        entity_id=normalized_value or None,
        entity_type="Person" if normalized_value else None,
        status="resolved" if normalized_value else "unresolved",
        display_name=_lookup_entity_display_name(session, normalized_value) if normalized_value else None,
    )


def _lookup_entity_display_name(session: Any, entity_id: Optional[str]) -> Optional[str]:
    if session is None or not entity_id:
        return None
    rows = session.run(
        """
        OPTIONAL MATCH (u:User {id: $entity_id})
        OPTIONAL MATCH (g:Group {id: $entity_id})
        OPTIONAL MATCH (p:Person {id: $entity_id})
        OPTIONAL MATCH (u_email:User)
        WHERE toLower(coalesce(u_email.email, '')) = toLower($entity_id)
        OPTIONAL MATCH (p_email:Person)
        WHERE toLower(coalesce(p_email.email, '')) = toLower($entity_id)
        RETURN coalesce(u.name, g.name, p.name, u_email.name, p_email.name, $entity_id) AS display_name
        LIMIT 1
        """,
        entity_id=entity_id,
    ).data()
    if not rows:
        return None
    return rows[0].get("display_name") or entity_id


def _load_entity_display_names(session: Any, entity_ids: Sequence[str]) -> Dict[str, str]:
    if session is None or not entity_ids:
        return {}
    rows = session.run(
        """
        UNWIND $entity_ids AS entity_id
        OPTIONAL MATCH (u:User {id: entity_id})
        OPTIONAL MATCH (g:Group {id: entity_id})
        OPTIONAL MATCH (p:Person {id: entity_id})
        OPTIONAL MATCH (u_email:User)
        WHERE toLower(coalesce(u_email.email, '')) = toLower(entity_id)
        OPTIONAL MATCH (p_email:Person)
        WHERE toLower(coalesce(p_email.email, '')) = toLower(entity_id)
        RETURN entity_id AS entity_id, coalesce(u.name, g.name, p.name, u_email.name, p_email.name, entity_id) AS display_name
        """,
        entity_ids=list(dict.fromkeys(entity_ids)),
    ).data()
    return {
        str(row.get("entity_id")): str(row.get("display_name") or row.get("entity_id"))
        for row in rows
        if row.get("entity_id")
    }


def _collect_entity_ids(*collections: Sequence[Dict[str, Any]]) -> List[str]:
    entity_ids: set[str] = set()
    for collection in collections:
        for record in collection or []:
            for key in ("subject_entity_id", "object_entity_id"):
                value = record.get(key)
                if value:
                    entity_ids.add(str(value))
            for raw_key in ("_previous_fact", "_replacement_fact"):
                nested = record.get(raw_key)
                if isinstance(nested, dict):
                    for key in ("subject_entity_id", "object_entity_id"):
                        value = nested.get(key)
                        if value:
                            entity_ids.add(str(value))
    return sorted(entity_ids)


def _display_label_from_fields(
    *,
    entity_id: Optional[str],
    key: Optional[str],
    raw: Optional[str],
    display_names: Dict[str, str],
) -> Optional[str]:
    if entity_id and display_names.get(entity_id):
        return display_names[entity_id]
    if key and display_names.get(key):
        return display_names[key]
    if key:
        return _humanize_entity_label(key)
    if raw:
        return _humanize_entity_label(raw)
    return None


def _decorate_grounding(grounding: Dict[str, Any], display_names: Dict[str, str]) -> Dict[str, Any]:
    references = grounding.get("references") or []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        display_name = _display_label_from_fields(
            entity_id=reference.get("entity_id"),
            key=reference.get("resolved_key"),
            raw=reference.get("raw"),
            display_names=display_names,
        )
        if display_name:
            reference["display_name"] = display_name
    grounding["references"] = references
    return grounding


def _decorate_claim_for_insight(claim: Dict[str, Any], display_names: Dict[str, str]) -> None:
    grounding = _decorate_grounding(claim.get("grounding") or {}, display_names)
    claim["grounding"] = grounding
    claim["subject_display"] = _display_label_from_fields(
        entity_id=claim.get("subject_entity_id"),
        key=claim.get("subject_key"),
        raw=claim.get("subject_raw"),
        display_names=display_names,
    )
    claim["object_display"] = _display_label_from_fields(
        entity_id=claim.get("object_entity_id"),
        key=claim.get("object_key"),
        raw=claim.get("object_raw"),
        display_names=display_names,
    )
    claim["display_text"] = _render_record_display_text(claim, display_names)


def _decorate_fact_for_insight(fact: Dict[str, Any], display_names: Dict[str, str]) -> None:
    fact["subject_display"] = _display_label_from_fields(
        entity_id=fact.get("subject_entity_id"),
        key=fact.get("subject_key"),
        raw=fact.get("subject_raw"),
        display_names=display_names,
    )
    fact["object_display"] = _display_label_from_fields(
        entity_id=fact.get("object_entity_id"),
        key=fact.get("object_key"),
        raw=fact.get("object_raw"),
        display_names=display_names,
    )
    fact["display_summary"] = _render_record_display_text(fact, display_names)


def _render_record_display_text(record: Dict[str, Any], display_names: Dict[str, str]) -> str:
    claim_type = record.get("claim_type")
    payload = _load_payload(record)
    subject = _display_label_from_fields(
        entity_id=record.get("subject_entity_id"),
        key=record.get("subject_key"),
        raw=record.get("subject_raw"),
        display_names=display_names,
    ) or "Unknown"
    obj = _display_label_from_fields(
        entity_id=record.get("object_entity_id"),
        key=record.get("object_key"),
        raw=record.get("object_raw"),
        display_names=display_names,
    )
    value_text = record.get("value_text")
    temporal_start = record.get("temporal_start")

    if claim_type == "TASK_ASSIGNMENT":
        text = f"{subject} will {value_text or payload.get('verb') or 'do'}"
        if obj:
            relation = payload.get("recipient_relation") or _default_commitment_recipient_relation(payload.get("verb") or "")
            text += f" {relation} {obj}" if relation else f" {obj}"
        if temporal_start:
            text += f" on {temporal_start}"
        return _normalize_whitespace(text)
    if claim_type == "REPORTS_TO":
        return _normalize_whitespace(f"{subject} reports to {obj or 'Unknown'}")
    if claim_type == "APPROVAL_STATE":
        text = f"{subject} is approved"
        if obj:
            text += f" by {obj}"
        return text
    if claim_type == "STATUS_UPDATE":
        return _normalize_whitespace(f"{subject} is {value_text or 'updated'}")
    if claim_type == "ASSIGNMENT_STATE":
        relation_text = "is no longer assigned to" if value_text == "inactive" else "is assigned to"
        return _normalize_whitespace(f"{subject} {relation_text} {obj or _humanize_entity_label(payload.get('assignment_target')) or 'Unknown'}")
    if claim_type == "MEETING_EVENT":
        text = value_text or "meeting"
        if temporal_start:
            text += f" scheduled for {temporal_start}"
        return text
    if claim_type == "REQUEST":
        text = f"{subject} requested"
        if obj:
            text += f" {obj}"
        if value_text:
            text += f" to {value_text}"
        return _normalize_whitespace(text)
    return record.get("normalized_text") or record.get("summary") or value_text or "Unknown"


def _build_grounding_payload(
    context: GroundingContext,
    *,
    source_span_text: str,
    subject_resolution: Resolution,
    object_resolution: Optional[Resolution],
    temporal: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    references = [
        {
            "role": "subject",
            "raw": subject_resolution.raw,
            "resolved_key": subject_resolution.key,
            "entity_id": subject_resolution.entity_id,
            "entity_type": subject_resolution.entity_type,
            "status": subject_resolution.status,
            "display_name": subject_resolution.display_name,
        }
    ]
    if object_resolution is not None:
        references.append(
            {
                "role": "object",
                "raw": object_resolution.raw,
                "resolved_key": object_resolution.key,
                "entity_id": object_resolution.entity_id,
                "entity_type": object_resolution.entity_type,
                "status": object_resolution.status,
                "display_name": object_resolution.display_name,
            }
        )
    return {
        "source_kind": context.source_kind,
        "conversation_type": context.conversation_type,
        "scope_type": context.scope_type,
        "scope_id": context.scope_id,
        "anchor_sent_at": context.sent_at,
        "sender_id": context.sender_id,
        "receiver_ids": list(context.receiver_ids),
        "group_id": context.group_id,
        "references": references,
        "temporal_expressions": _extract_temporal_expressions(source_span_text),
        "temporal_start": temporal.get("temporal_start"),
        "temporal_end": temporal.get("temporal_end"),
        "temporal_granularity": temporal.get("temporal_granularity"),
        "timezone": temporal.get("timezone") or context.timezone,
    }


def _extract_temporal_expressions(text: str) -> List[str]:
    matches: List[Tuple[int, int, str]] = []
    for pattern in (TIME_WORD_PATTERN, AT_CLOCK_PATTERN, CLOCK_PATTERN):
        for match in pattern.finditer(text or ""):
            matches.append((match.start(), match.end(), match.group(0)))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    expressions: List[str] = []
    last_end = -1
    seen: set[str] = set()
    for start, end, raw in matches:
        key = raw.strip().lower()
        if start < last_end or key in seen:
            continue
        expressions.append(raw.strip())
        seen.add(key)
        last_end = end
    return expressions


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower())
    return value.strip("-") or "unknown"


def _parse_iso_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _next_weekday(start: date, weekday: int) -> date:
    delta = (weekday - start.weekday()) % 7
    delta = 7 if delta == 0 else delta
    return start + timedelta(days=delta)


def _serialize_node(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    try:
        return dict(value)
    except Exception:
        return {"value": str(value)}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
