"""Canonical Neo4j-backed chat, auth, and group storage for SAGE."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

try:
    import bcrypt
except ImportError:  # pragma: no cover - exercised in dependency-light test environments
    bcrypt = None

try:
    import app.saia as saia
    import app.services as services
    import app.utils as utils
except ImportError:
    import saia
    import services
    import utils


logger = logging.getLogger(__name__)

SAGE_USER_ID = "sage"
GRAPH_SYNC_READY = "ready"
GRAPH_SYNC_FAILED = "failed"
GRAPH_SYNC_SKIPPED = "skipped"
GRAPH_SYNC_PENDING = "pending"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_direct_conversation_id(left: str, right: str) -> str:
    first, second = sorted([str(left), str(right)])
    return f"direct:{first}:{second}"


def stable_sage_conversation_id(user_id: str) -> str:
    return f"sage:{user_id}"


def stable_group_conversation_id(group_id: str) -> str:
    return f"group:{group_id}"


def hash_password(password: str) -> str:
    if bcrypt is not None:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    ).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith("pbkdf2_sha256$"):
        try:
            _scheme, salt, digest = password_hash.split("$", 2)
        except ValueError:
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            200_000,
        ).hex()
        return hmac.compare_digest(candidate, digest)

    if bcrypt is None:
        logger.warning("bcrypt is unavailable; cannot verify legacy bcrypt password hashes.")
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def _rows(result: Any) -> List[Dict[str, Any]]:
    try:
        return result.data()
    except Exception:
        return []


def _first(rows: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in rows:
        return row
    return None


def _serialize_attachment(attachment: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not attachment:
        return None
    return {
        "id": attachment.get("id"),
        "name": attachment.get("name"),
        "type": attachment.get("type"),
        "size": attachment.get("size"),
        "url": attachment.get("url"),
    }


def _deserialize_json_object(raw: Any) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _deserialize_message(record: Dict[str, Any]) -> Dict[str, Any]:
    attachment_name = record.get("attachment_name")
    attachment = None
    if attachment_name:
        attachment = {
            "id": record.get("attachment_id"),
            "name": attachment_name,
            "type": record.get("attachment_type"),
            "size": record.get("attachment_size"),
            "url": record.get("attachment_url"),
        }

    trace = _deserialize_json_object(record.get("trace_json"))
    answer_payload = _deserialize_json_object(record.get("answer_payload_json"))

    thinking = record.get("thinking") or []
    return {
        "id": record.get("id"),
        "conversationId": record.get("conversation_id"),
        "conversationType": record.get("conversation_type"),
        "senderId": record.get("sender_id"),
        "receiverId": record.get("receiver_id"),
        "groupId": record.get("group_id"),
        "content": record.get("content") or "",
        "sentAt": record.get("sent_at"),
        "readByCurrentUser": bool(record.get("read_by_current_user")),
        "source": record.get("source") or "chat_message",
        "graphSyncStatus": record.get("graph_sync_status") or GRAPH_SYNC_SKIPPED,
        "role": record.get("role"),
        "isAiResponse": bool(record.get("is_ai_response")),
        "thinking": list(thinking),
        "answerPayload": answer_payload,
        "trace": trace,
        "attachment": attachment,
    }


def _serialize_user(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record.get("id"),
        "name": record.get("name"),
        "email": record.get("email"),
        "avatar": record.get("avatar"),
        "team": record.get("team") or [],
        "isBot": bool(record.get("is_bot")),
        "isPinned": bool(record.get("is_pinned")),
    }


def _ensure_user_node(session: Any, user: Dict[str, Any]) -> None:
    password_hash = user.get("passwordHash")
    raw_password = user.get("password")
    if not password_hash and raw_password:
        password_hash = hash_password(raw_password)

    session.run(
        """
        MERGE (u:User:Person {id: $id})
        SET u.name = $name,
            u.email = $email,
            u.avatar = $avatar,
            u.team = $team,
            u.is_bot = $is_bot,
            u.is_pinned = $is_pinned,
            u.password_hash = COALESCE($password_hash, u.password_hash),
            u.updated_at = $updated_at
        """,
        id=user["id"],
        name=user.get("name") or user["id"],
        email=(user.get("email") or "").lower() or None,
        avatar=user.get("avatar"),
        team=user.get("team") or [],
        is_bot=bool(user.get("isBot")),
        is_pinned=bool(user.get("isPinned")),
        password_hash=password_hash,
        updated_at=utcnow_iso(),
    )


def _ensure_group_node(session: Any, group: Dict[str, Any]) -> None:
    session.run(
        """
        MERGE (g:Group {id: $id})
        SET g.name = $name,
            g.description = $description,
            g.avatar = $avatar,
            g.updated_at = $updated_at
        """,
        id=group["id"],
        name=group.get("name") or group["id"],
        description=group.get("description"),
        avatar=group.get("avatar"),
        updated_at=utcnow_iso(),
    )
    for member_id in group.get("memberIds") or []:
        session.run(
            """
            MATCH (u:User {id: $user_id})
            MATCH (g:Group {id: $group_id})
            MERGE (u)-[:MEMBER_OF]->(g)
            """,
            user_id=str(member_id),
            group_id=group["id"],
        )


def _ensure_direct_conversation(session: Any, left: str, right: str) -> str:
    conversation_id = stable_direct_conversation_id(left, right)
    session.run(
        """
        MATCH (first:User {id: $left_id})
        MATCH (second:User {id: $right_id})
        MERGE (c:Conversation {id: $conversation_id})
        SET c.type = 'direct',
            c.group_id = null,
            c.title = null,
            c.updated_at = COALESCE(c.updated_at, $updated_at)
        MERGE (first)-[:PARTICIPATES_IN]->(c)
        MERGE (second)-[:PARTICIPATES_IN]->(c)
        """,
        left_id=left,
        right_id=right,
        conversation_id=conversation_id,
        updated_at=utcnow_iso(),
    )
    return conversation_id


def _ensure_sage_conversation(session: Any, user_id: str) -> str:
    conversation_id = stable_sage_conversation_id(user_id)
    session.run(
        """
        MATCH (user:User {id: $user_id})
        MATCH (sage:User {id: $sage_id})
        MERGE (c:Conversation {id: $conversation_id})
        SET c.type = 'sage',
            c.group_id = null,
            c.title = 'SAGE',
            c.updated_at = COALESCE(c.updated_at, $updated_at)
        MERGE (user)-[:PARTICIPATES_IN]->(c)
        MERGE (sage)-[:PARTICIPATES_IN]->(c)
        """,
        user_id=user_id,
        sage_id=SAGE_USER_ID,
        conversation_id=conversation_id,
        updated_at=utcnow_iso(),
    )
    return conversation_id


def _ensure_group_conversation(session: Any, group_id: str) -> str:
    conversation_id = stable_group_conversation_id(group_id)
    session.run(
        """
        MATCH (g:Group {id: $group_id})
        MERGE (c:Conversation {id: $conversation_id})
        SET c.type = 'group',
            c.group_id = $group_id,
            c.title = g.name,
            c.updated_at = COALESCE(c.updated_at, $updated_at)
        MERGE (c)-[:ABOUT_GROUP]->(g)
        WITH c, g
        MATCH (member:User)-[:MEMBER_OF]->(g)
        MERGE (member)-[:PARTICIPATES_IN]->(c)
        """,
        group_id=group_id,
        conversation_id=conversation_id,
        updated_at=utcnow_iso(),
    )
    return conversation_id


def _ensure_default_sage_user(session: Any) -> None:
    _ensure_user_node(
        session,
        {
            "id": SAGE_USER_ID,
            "name": "SAGE - ASK ANYTHING",
            "email": "sage@system.ai",
            "avatar": "SA",
            "password": "systempassword",
            "team": [],
            "isBot": True,
            "isPinned": True,
        },
    )


def _message_exists(session: Any, message_id: str) -> bool:
    rows = _rows(
        session.run(
            """
            MATCH (m:Message {id: $message_id})
            RETURN m.id AS id
            LIMIT 1
            """,
            message_id=message_id,
        )
    )
    return bool(rows)


def _conversation_record(session: Any, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    rows = _rows(
        session.run(
            """
            MATCH (user:User {id: $user_id})-[:PARTICIPATES_IN]->(c:Conversation {id: $conversation_id})
            OPTIONAL MATCH (c)-[:ABOUT_GROUP]->(g:Group)
            OPTIONAL MATCH (c)<-[:PARTICIPATES_IN]-(participant:User)
            RETURN c.id AS id,
                   c.type AS type,
                   c.group_id AS group_id,
                   c.title AS title,
                   g.name AS group_name,
                   collect(DISTINCT participant.id) AS participant_ids
            LIMIT 1
            """,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    )
    return _first(rows)


def _conversation_member_ids(session: Any, conversation_id: str) -> List[str]:
    rows = _rows(
        session.run(
            """
            MATCH (:User)-[:PARTICIPATES_IN]->(c:Conversation {id: $conversation_id})<-[:PARTICIPATES_IN]-(participant:User)
            RETURN collect(DISTINCT participant.id) AS participant_ids
            """,
            conversation_id=conversation_id,
        )
    )
    record = _first(rows) or {}
    return list(record.get("participant_ids") or [])


def _build_document_payload(
    message_id: str,
    message: Dict[str, Any],
    conversation: Dict[str, Any],
    receiver_ids: List[str],
) -> Dict[str, Any]:
    attachment = message.get("attachment") or {}
    trace = message.get("trace")
    return {
        "doc_id": f"chat-msg-{message_id}",
        "sender": message["senderId"],
        "receivers": receiver_ids,
        "subject": f"Chat message {message_id}",
        "content": message.get("content") or "",
        "timestamp": message.get("sentAt"),
        "source": message.get("source") or "chat_message",
        "conversation_type": conversation.get("type"),
        "conversation_id": conversation.get("id"),
        "group_id": conversation.get("group_id"),
        "attachment_name": attachment.get("name"),
        "attachment_type": attachment.get("type"),
        "attachment_url": attachment.get("url"),
        "origin_message_id": message_id,
        "linked_message_id": None,
        "trace_json": json.dumps(trace) if trace else None,
        "graph_sync_status": message.get("graphSyncStatus") or GRAPH_SYNC_READY,
    }


def _is_graph_ineligible_message(
    *,
    conversation_type: Optional[str],
    sender_id: Optional[str],
    receiver_id: Optional[str],
    source: Optional[str],
    is_ai_response: bool,
) -> bool:
    normalized_source = (source or "").strip().lower()
    normalized_conversation_type = (conversation_type or "").strip().lower()
    if normalized_conversation_type == "sage":
        return True
    if sender_id == SAGE_USER_ID or receiver_id == SAGE_USER_ID:
        return True
    if normalized_source.startswith("sage_"):
        return True
    return bool(is_ai_response and sender_id == SAGE_USER_ID)


def _sync_message_to_graph(
    session: Any,
    conversation: Dict[str, Any],
    message: Dict[str, Any],
    message_id: str,
) -> str:
    if not (message.get("content") or "").strip():
        return GRAPH_SYNC_SKIPPED
    if not message.get("syncToGraph", True):
        return GRAPH_SYNC_SKIPPED
    if _is_graph_ineligible_message(
        conversation_type=conversation.get("type"),
        sender_id=message.get("senderId"),
        receiver_id=message.get("receiverId"),
        source=message.get("source"),
        is_ai_response=bool(message.get("isAiResponse")),
    ):
        return GRAPH_SYNC_SKIPPED

    participant_ids = _conversation_member_ids(session, conversation["id"])
    receiver_ids = [
        participant_id
        for participant_id in participant_ids
        if participant_id != message["senderId"]
    ]
    if conversation.get("type") == "sage" and not receiver_ids:
        receiver_ids = [message.get("receiverId") or message["senderId"]]

    try:
        payload = _build_document_payload(message_id, message, conversation, receiver_ids)
        ok = services.store_in_neo4j(payload)
        if ok:
            try:
                saia.process_chat_message(
                    message_id=message_id,
                    sender_id=message["senderId"],
                    receiver_ids=receiver_ids,
                    conversation_id=conversation.get("id"),
                    conversation_type=conversation.get("type"),
                    group_id=conversation.get("group_id"),
                    sent_at=message.get("sentAt"),
                    content=message.get("content") or "",
                    source=message.get("source") or "chat_message",
                    trace=message.get("trace"),
                    is_ai_response=bool(message.get("isAiResponse")),
                    attachment_name=(message.get("attachment") or {}).get("name"),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("SAIA processing failed for message %s: %s", message_id, exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Graph sync failed for message %s: %s", message_id, exc)
        ok = False
    return GRAPH_SYNC_READY if ok else GRAPH_SYNC_FAILED


def create_message(
    session: Any,
    authenticated_user_id: str,
    conversation_id: str,
    payload: Dict[str, Any],
    *,
    mark_recipient_read: bool = False,
) -> Dict[str, Any]:
    conversation = _conversation_record(session, conversation_id, authenticated_user_id)
    if not conversation:
        raise ValueError("Conversation not found or access denied")

    sender_id = str(payload.get("senderId") or authenticated_user_id)
    is_sage_reply = conversation.get("type") == "sage" and sender_id == SAGE_USER_ID
    if sender_id != authenticated_user_id and authenticated_user_id != SAGE_USER_ID and not is_sage_reply:
        raise ValueError("Sender does not match authenticated user")

    message_id = str(payload.get("id") or uuid4())
    attachment = _serialize_attachment(payload.get("attachment"))
    sent_at = payload.get("sentAt") or utcnow_iso()
    content = payload.get("content") or ""
    source = payload.get("source") or "chat_message"
    thinking = payload.get("thinking") or []
    answer_payload = payload.get("answerPayload")
    trace = payload.get("trace")
    role = payload.get("role") or ("assistant" if payload.get("isAiResponse") else "user")
    receiver_id = payload.get("receiverId")
    group_id = payload.get("groupId") or conversation.get("group_id")

    should_sync_to_graph = (
        bool((content or "").strip())
        and payload.get("syncToGraph", True)
        and not _is_graph_ineligible_message(
            conversation_type=conversation.get("type"),
            sender_id=sender_id,
            receiver_id=receiver_id,
            source=source,
            is_ai_response=bool(payload.get("isAiResponse")),
        )
    )
    graph_sync_status = GRAPH_SYNC_PENDING if should_sync_to_graph else GRAPH_SYNC_SKIPPED

    session.run(
        """
        MATCH (c:Conversation {id: $conversation_id})
        MATCH (sender:User {id: $sender_id})
        MERGE (m:Message {id: $message_id})
        SET m.conversation_id = $conversation_id,
            m.conversation_type = $conversation_type,
            m.sender_id = $sender_id,
            m.receiver_id = $receiver_id,
            m.group_id = $group_id,
            m.content = $content,
            m.sent_at = $sent_at,
            m.source = $source,
            m.graph_sync_status = $graph_sync_status,
            m.role = $role,
            m.is_ai_response = $is_ai_response,
            m.thinking = $thinking,
            m.answer_payload_json = $answer_payload_json,
            m.trace_json = $trace_json,
            m.attachment_id = $attachment_id,
            m.attachment_name = $attachment_name,
            m.attachment_type = $attachment_type,
            m.attachment_size = $attachment_size,
            m.attachment_url = $attachment_url
        MERGE (m)-[:IN_CONVERSATION]->(c)
        MERGE (m)-[:SENT_BY]->(sender)
        SET c.updated_at = $sent_at
        """,
        conversation_id=conversation_id,
        message_id=message_id,
        conversation_type=conversation.get("type"),
        sender_id=sender_id,
        receiver_id=receiver_id,
        group_id=group_id,
        content=content,
        sent_at=sent_at,
        source=source,
        graph_sync_status=graph_sync_status,
        role=role,
        is_ai_response=bool(payload.get("isAiResponse")),
        thinking=list(thinking),
        answer_payload_json=json.dumps(answer_payload) if answer_payload else None,
        trace_json=json.dumps(trace) if trace else None,
        attachment_id=attachment.get("id") if attachment else None,
        attachment_name=attachment.get("name") if attachment else None,
        attachment_type=attachment.get("type") if attachment else None,
        attachment_size=attachment.get("size") if attachment else None,
        attachment_url=attachment.get("url") if attachment else None,
    )

    if should_sync_to_graph:
        graph_sync_status = _sync_message_to_graph(
            session,
            conversation,
            {
                "senderId": sender_id,
                "receiverId": receiver_id,
                "content": content,
                "sentAt": sent_at,
                "source": source,
                "attachment": attachment,
                "trace": trace,
                "isAiResponse": bool(payload.get("isAiResponse")),
                "graphSyncStatus": payload.get("graphSyncStatus"),
                "syncToGraph": payload.get("syncToGraph", True),
            },
            message_id,
        )
        session.run(
            """
            MATCH (m:Message {id: $message_id})
            SET m.graph_sync_status = $graph_sync_status
            """,
            message_id=message_id,
            graph_sync_status=graph_sync_status,
        )

    session.run(
        """
        MATCH (u:User {id: $user_id})
        MATCH (m:Message {id: $message_id})
        MERGE (u)-[r:HAS_READ]->(m)
        SET r.at = $read_at
        """,
        user_id=sender_id,
        message_id=message_id,
        read_at=sent_at,
    )

    if mark_recipient_read:
        for participant_id in _conversation_member_ids(session, conversation_id):
            if participant_id == sender_id:
                continue
            session.run(
                """
                MATCH (u:User {id: $user_id})
                MATCH (m:Message {id: $message_id})
                MERGE (u)-[r:HAS_READ]->(m)
                SET r.at = $read_at
                """,
                user_id=participant_id,
                message_id=message_id,
                read_at=sent_at,
            )

    message_record = {
        "id": message_id,
        "conversation_id": conversation_id,
        "conversation_type": conversation.get("type"),
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "group_id": group_id,
        "content": content,
        "sent_at": sent_at,
        "source": source,
        "graph_sync_status": graph_sync_status,
        "role": role,
        "is_ai_response": bool(payload.get("isAiResponse")),
        "thinking": list(thinking),
        "answer_payload_json": json.dumps(answer_payload) if answer_payload else None,
        "trace_json": json.dumps(trace) if trace else None,
        "attachment_id": attachment.get("id") if attachment else None,
        "attachment_name": attachment.get("name") if attachment else None,
        "attachment_type": attachment.get("type") if attachment else None,
        "attachment_size": attachment.get("size") if attachment else None,
        "attachment_url": attachment.get("url") if attachment else None,
        "read_by_current_user": True,
    }
    notify_user_ids = [
        participant_id
        for participant_id in _conversation_member_ids(session, conversation_id)
        if participant_id != sender_id
    ]
    return {
        "message": _deserialize_message(message_record),
        "notifyUserIds": notify_user_ids,
    }


def create_message_for_conversation(
    authenticated_user_id: str,
    conversation_id: str,
    payload: Dict[str, Any],
    *,
    mark_recipient_read: bool = False,
) -> Dict[str, Any]:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            return create_message(
                session,
                authenticated_user_id=authenticated_user_id,
                conversation_id=conversation_id,
                payload=payload,
                mark_recipient_read=mark_recipient_read,
            )
    finally:
        driver.close()


def bootstrap_seed_data(
    users: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
) -> Dict[str, int]:
    driver = utils.create_neo4j_driver()
    imported_users = 0
    imported_groups = 0
    imported_messages = 0

    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            _ensure_default_sage_user(session)
            known_user_ids = {SAGE_USER_ID}

            for user in users:
                user_id = str(user.get("id") or "").strip()
                if not user_id:
                    continue
                _ensure_user_node(session, user)
                known_user_ids.add(user_id)
                imported_users += 1

            group_ids = {str(group.get("id")) for group in groups if group.get("id")}
            for group in groups:
                _ensure_group_node(session, group)
                _ensure_group_conversation(session, group["id"])
                imported_groups += 1

            # Messages can reference legacy users that do not exist in users.json.
            # Create placeholder user nodes so bootstrap never fails on those rows.
            for message in messages:
                sender_id = str(message.get("senderId") or "").strip()
                receiver_id = str(message.get("receiverId") or "").strip()

                for participant_id in (sender_id, receiver_id):
                    if (
                        not participant_id
                        or participant_id in known_user_ids
                        or participant_id in group_ids
                        or participant_id == SAGE_USER_ID
                    ):
                        continue

                    _ensure_user_node(
                        session,
                        {
                            "id": participant_id,
                            "name": f"User {participant_id}",
                            "email": None,
                            "avatar": None,
                            "team": [],
                            "isBot": participant_id == "ai",
                            "isPinned": False,
                        },
                    )
                    known_user_ids.add(participant_id)
                    imported_users += 1

            user_ids = [
                str(item.get("id"))
                for item in users
                if item.get("id") and item.get("id") != SAGE_USER_ID
            ]
            user_ids.extend(
                user_id
                for user_id in known_user_ids
                if user_id not in {SAGE_USER_ID} and user_id not in user_ids
            )

            for user_id in user_ids:
                _ensure_sage_conversation(session, user_id)
                for other_id in user_ids:
                    if other_id == user_id or other_id == SAGE_USER_ID:
                        continue
                    _ensure_direct_conversation(session, user_id, other_id)

            for message in messages:
                message_id = str(message.get("id") or uuid4())
                if _message_exists(session, message_id):
                    continue

                sender_id = str(message.get("senderId") or "").strip()
                receiver_id = str(message.get("receiverId") or "").strip()
                if not sender_id:
                    logger.warning("Skipping seed message %s: missing senderId", message_id)
                    continue

                if receiver_id in group_ids:
                    conversation_id = _ensure_group_conversation(session, receiver_id)
                    conversation_type = "group"
                    group_id = receiver_id
                elif sender_id == SAGE_USER_ID or receiver_id == SAGE_USER_ID:
                    owner_id = receiver_id if sender_id == SAGE_USER_ID else sender_id
                    if not owner_id:
                        logger.warning("Skipping seed message %s: missing owner for sage conversation", message_id)
                        continue
                    conversation_id = _ensure_sage_conversation(session, owner_id)
                    conversation_type = "sage"
                    group_id = None
                else:
                    if not receiver_id:
                        logger.warning("Skipping seed message %s: missing receiverId", message_id)
                        continue
                    conversation_id = _ensure_direct_conversation(session, sender_id, receiver_id)
                    conversation_type = "direct"
                    group_id = None

                try:
                    create_message(
                        session,
                        authenticated_user_id=sender_id,
                        conversation_id=conversation_id,
                        payload={
                            "id": message_id,
                            "senderId": sender_id,
                            "receiverId": receiver_id or None,
                            "groupId": group_id,
                            "content": message.get("content") or "",
                            "sentAt": message.get("sentAt") or message.get("timestamp") or utcnow_iso(),
                            "source": message.get("source") or "seed_message",
                            "attachment": message.get("attachment"),
                            "trace": message.get("trace"),
                            "thinking": message.get("thinking") or [],
                            "answerPayload": message.get("answerPayload"),
                            "role": message.get("role"),
                            "isAiResponse": bool(message.get("isAiResponse")),
                            "syncToGraph": not bool(message.get("skipGraphSync")),
                            "conversationType": conversation_type,
                        },
                        mark_recipient_read=bool(message.get("read")),
                    )
                    imported_messages += 1
                except ValueError as exc:
                    logger.warning("Skipping seed message %s: %s", message_id, exc)

        return {"users": imported_users, "groups": imported_groups, "messages": imported_messages}
    finally:
        driver.close()


def register_user(payload: Dict[str, Any]) -> Dict[str, Any]:
    driver = utils.create_neo4j_driver()
    email = (payload.get("email") or "").strip().lower()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            existing = _first(
                _rows(
                    session.run(
                        """
                        MATCH (u:User {email: $email})
                        RETURN u.id AS id
                        LIMIT 1
                        """,
                        email=email,
                    )
                )
            )
            if existing:
                raise ValueError("Email is already registered")

            user_id = str(payload.get("id") or int(datetime.now(timezone.utc).timestamp() * 1000))
            user = {
                "id": user_id,
                "name": payload.get("name") or user_id,
                "email": email,
                "avatar": payload.get("avatar") or None,
                "team": payload.get("team") or [],
                "password": payload.get("password") or "",
                "isBot": False,
                "isPinned": False,
            }
            _ensure_user_node(session, user)
            _ensure_sage_conversation(session, user_id)
            return {**user, "password": None}
    finally:
        driver.close()


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            record = _first(
                _rows(
                    session.run(
                        """
                        MATCH (u:User {email: $email})
                        RETURN u.id AS id,
                               u.name AS name,
                               u.email AS email,
                               u.avatar AS avatar,
                               u.team AS team,
                               u.password_hash AS password_hash,
                               u.is_bot AS is_bot,
                               u.is_pinned AS is_pinned
                        LIMIT 1
                        """,
                        email=email.strip().lower(),
                    )
                )
            )
            if not record:
                return None
            if not verify_password(password, record.get("password_hash") or ""):
                return None
            return _serialize_user(record)
    finally:
        driver.close()


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            record = _first(
                _rows(
                    session.run(
                        """
                        MATCH (u:User {id: $user_id})
                        RETURN u.id AS id,
                               u.name AS name,
                               u.email AS email,
                               u.avatar AS avatar,
                               u.team AS team,
                               u.is_bot AS is_bot,
                               u.is_pinned AS is_pinned
                        LIMIT 1
                        """,
                        user_id=user_id,
                    )
                )
            )
            return _serialize_user(record) if record else None
    finally:
        driver.close()


def update_user_profile(user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            record = _first(
                _rows(
                    session.run(
                        """
                        MATCH (u:User {id: $user_id})
                        SET u.name = $name,
                            u.email = $email,
                            u.avatar = $avatar,
                            u.updated_at = $updated_at
                        RETURN u.id AS id,
                               u.name AS name,
                               u.email AS email,
                               u.avatar AS avatar,
                               u.team AS team,
                               u.is_bot AS is_bot,
                               u.is_pinned AS is_pinned
                        """,
                        user_id=user_id,
                        name=payload.get("name"),
                        email=(payload.get("email") or "").strip().lower(),
                        avatar=payload.get("avatar") or None,
                        updated_at=utcnow_iso(),
                    )
                )
            )
            if not record:
                raise ValueError("User not found")
            return _serialize_user(record)
    finally:
        driver.close()


def list_groups_for_user(user_id: str) -> List[Dict[str, Any]]:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            rows = _rows(
                session.run(
                    """
                    MATCH (:User {id: $user_id})-[:MEMBER_OF]->(g:Group)
                    OPTIONAL MATCH (member:User)-[:MEMBER_OF]->(g)
                    RETURN g.id AS id,
                           g.name AS name,
                           g.description AS description,
                           g.avatar AS avatar,
                           collect(DISTINCT member.id) AS member_ids
                    ORDER BY g.name ASC
                    """,
                    user_id=user_id,
                )
            )
            return [
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "description": row.get("description"),
                    "avatar": row.get("avatar"),
                    "memberIds": list(row.get("member_ids") or []),
                }
                for row in rows
            ]
    finally:
        driver.close()


def ensure_user_conversations(user_id: str) -> None:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            _ensure_sage_conversation(session, user_id)
            rows = _rows(
                session.run(
                    """
                    MATCH (u:User)
                    WHERE u.id <> $user_id AND coalesce(u.is_bot, false) = false
                    RETURN u.id AS id
                    ORDER BY u.name ASC
                    """,
                    user_id=user_id,
                )
            )
            for row in rows:
                other_id = row.get("id")
                if other_id:
                    _ensure_direct_conversation(session, user_id, other_id)

            group_rows = _rows(
                session.run(
                    """
                    MATCH (:User {id: $user_id})-[:MEMBER_OF]->(g:Group)
                    RETURN g.id AS id
                    """,
                    user_id=user_id,
                )
            )
            for row in group_rows:
                if row.get("id"):
                    _ensure_group_conversation(session, row["id"])
    finally:
        driver.close()


def list_conversation_summaries(user_id: str) -> List[Dict[str, Any]]:
    ensure_user_conversations(user_id)
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            rows = _rows(
                session.run(
                    """
                    MATCH (current:User {id: $user_id})-[:PARTICIPATES_IN]->(c:Conversation)
                    OPTIONAL MATCH (c)-[:ABOUT_GROUP]->(g:Group)
                    OPTIONAL MATCH (c)<-[:PARTICIPATES_IN]-(participant:User)
                    WITH current, c, g, collect(DISTINCT participant) AS participants
                    OPTIONAL MATCH (c)<-[:IN_CONVERSATION]-(last_message:Message)
                    WITH current, c, g, participants, last_message
                    ORDER BY last_message.sent_at DESC
                    WITH current,
                         c,
                         g,
                         participants,
                         collect(last_message)[0] AS last_message,
                         c.updated_at AS updated_at,
                         c.title AS conversation_title
                    OPTIONAL MATCH (c)<-[:IN_CONVERSATION]-(unread:Message)
                    WHERE unread.sender_id <> $user_id AND NOT (current)-[:HAS_READ]->(unread)
                    WITH c,
                         g,
                         participants,
                         last_message,
                         updated_at,
                         conversation_title,
                         count(DISTINCT unread) AS unread_count
                    RETURN c.id AS id,
                           c.type AS type,
                           c.title AS title,
                           c.group_id AS group_id,
                           g.name AS group_name,
                           g.avatar AS group_avatar,
                           [participant IN participants | {
                               id: participant.id,
                               name: participant.name,
                               avatar: participant.avatar,
                               is_bot: participant.is_bot,
                               is_pinned: participant.is_pinned
                           }] AS participants,
                           unread_count AS unread_count,
                           last_message.id AS last_message_id,
                           last_message.content AS last_message_content,
                           last_message.sent_at AS last_message_sent_at,
                           last_message.sender_id AS last_message_sender_id,
                           last_message.attachment_name AS last_message_attachment_name,
                           last_message.graph_sync_status AS last_message_graph_sync_status,
                           updated_at AS updated_at,
                           conversation_title AS conversation_title
                    ORDER BY coalesce(last_message_sent_at, updated_at) DESC, conversation_title ASC
                    """,
                    user_id=user_id,
                )
            )

            summaries: List[Dict[str, Any]] = []
            for row in rows:
                participants = [item for item in row.get("participants") or [] if item.get("id") != user_id]
                other_user = participants[0] if participants else None
                title = row.get("title") or row.get("group_name")
                avatar = row.get("group_avatar")
                if row.get("type") in {"direct", "sage"} and other_user:
                    title = other_user.get("name") or title
                    avatar = other_user.get("avatar")

                summaries.append(
                    {
                        "id": row.get("id"),
                        "type": row.get("type"),
                        "title": title,
                        "avatar": avatar,
                        "unreadCount": int(row.get("unread_count") or 0),
                        "groupId": row.get("group_id"),
                        "participants": participants,
                        "participantIds": [item.get("id") for item in participants if item.get("id")],
                        "otherUser": other_user,
                        "lastMessage": {
                            "id": row.get("last_message_id"),
                            "content": row.get("last_message_content"),
                            "sentAt": row.get("last_message_sent_at"),
                            "senderId": row.get("last_message_sender_id"),
                            "attachmentName": row.get("last_message_attachment_name"),
                            "graphSyncStatus": row.get("last_message_graph_sync_status"),
                        }
                        if row.get("last_message_id")
                        else None,
                    }
                )
            return summaries
    finally:
        driver.close()


def get_conversation_messages(user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            conversation = _conversation_record(session, conversation_id, user_id)
            if not conversation:
                raise ValueError("Conversation not found or access denied")

            rows = _rows(
                session.run(
                    """
                    MATCH (current:User {id: $user_id})-[:PARTICIPATES_IN]->(:Conversation {id: $conversation_id})
                    MATCH (m:Message {conversation_id: $conversation_id})
                    OPTIONAL MATCH (current)-[:HAS_READ]->(m)
                    RETURN m.id AS id,
                           m.conversation_id AS conversation_id,
                           m.conversation_type AS conversation_type,
                           m.sender_id AS sender_id,
                           m.receiver_id AS receiver_id,
                           m.group_id AS group_id,
                           m.content AS content,
                           m.sent_at AS sent_at,
                           m.source AS source,
                           m.graph_sync_status AS graph_sync_status,
                           m.role AS role,
                           m.is_ai_response AS is_ai_response,
                           m.thinking AS thinking,
                           m.answer_payload_json AS answer_payload_json,
                           m.trace_json AS trace_json,
                           m.attachment_id AS attachment_id,
                           m.attachment_name AS attachment_name,
                           m.attachment_type AS attachment_type,
                           m.attachment_size AS attachment_size,
                           m.attachment_url AS attachment_url,
                           COUNT(*) > 0 AS read_by_current_user
                    ORDER BY m.sent_at ASC
                    """,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            )
            return [_deserialize_message(row) for row in rows]
    finally:
        driver.close()


def mark_message_read(user_id: str, message_id: str) -> Dict[str, Any]:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            record = _first(
                _rows(
                    session.run(
                        """
                        MATCH (u:User {id: $user_id})-[:PARTICIPATES_IN]->(c:Conversation)<-[:IN_CONVERSATION]-(m:Message {id: $message_id})
                        MERGE (u)-[r:HAS_READ]->(m)
                        SET r.at = $read_at
                        RETURN m.id AS id, m.sender_id AS sender_id, c.id AS conversation_id
                        LIMIT 1
                        """,
                        user_id=user_id,
                        message_id=message_id,
                        read_at=utcnow_iso(),
                    )
                )
            )
            if not record:
                raise ValueError("Message not found or access denied")

            notify_user_ids: List[str] = []
            sender_id = record.get("sender_id")
            if sender_id and sender_id != user_id:
                notify_user_ids.append(sender_id)
            return {
                "messageId": record.get("id"),
                "conversationId": record.get("conversation_id"),
                "notifyUserIds": notify_user_ids,
            }
    finally:
        driver.close()
