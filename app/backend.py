"""FastAPI backend for the SAGE app."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import shutil
import sys
from typing import Any, Dict, List, Literal, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from pydantic import BaseModel, Field

try:
    import app.chat_store as chat_store
    import app.saia as saia
    import app.services as services
    import app.utils as utils
except ImportError:
    import chat_store
    import saia
    import services
    import utils


logger = logging.getLogger(__name__)

app = FastAPI(title="SAGE Enterprise Graph RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = utils.ROOT_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

query_graph = services.query_graph
query_graph_with_trace = services.query_graph_with_trace
generate_groq_response = services.generate_groq_response
store_in_neo4j = services.store_in_neo4j


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    user_id: Optional[str] = None


class AnswerPayload(BaseModel):
    schema_version: Literal[1] = 1
    mode: Literal["short", "long"]
    reason_code: Literal[
        "explicit_short",
        "explicit_long",
        "direct_lookup",
        "broad_or_explanatory",
        "evidence_complexity",
        "fallback_invalid_json",
    ]
    summary: str
    bullets: List[str] = Field(default_factory=list)
    explanation: str
    evidence_refs: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    answer_payload: AnswerPayload
    thinking: List[str] = Field(default_factory=list)
    trace: Dict[str, Any] = Field(default_factory=dict)


class DocumentProcessResponse(BaseModel):
    doc_id: str
    sender: str
    receivers: List[str]
    subject: str
    success: bool
    message: str


class GraphDebugResponse(BaseModel):
    node_counts: List[Dict[str, Any]]
    rel_counts: List[Dict[str, Any]]
    sample_docs: List[Dict[str, Any]]
    connectivity: List[Dict[str, Any]]
    entity_doc_connections: List[Dict[str, Any]]


class GraphPathNode(BaseModel):
    element_id: str
    labels: List[str] = Field(default_factory=list)
    display_name: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphPathRelationship(BaseModel):
    type: str
    source: str
    target: str


class GraphRetrievalPathResponse(BaseModel):
    hop_count: int
    nodes: List[GraphPathNode] = Field(default_factory=list)
    relationships: List[GraphPathRelationship] = Field(default_factory=list)


class GraphSubgraphResponse(BaseModel):
    depth: int
    nodes: List[GraphPathNode] = Field(default_factory=list)
    relationships: List[GraphPathRelationship] = Field(default_factory=list)


class UserSyncRequest(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    avatar: Optional[str] = None
    team: Optional[List[str]] = Field(default_factory=list)
    isBot: bool = False
    isPinned: bool = False


class UserSyncResponse(BaseModel):
    success: bool
    user_id: str
    message: str


class ChatMessageSyncItem(BaseModel):
    id: str
    senderId: str
    receiverId: str
    content: str
    timestamp: str
    source: Optional[str] = "chat_message"
    attachment: Optional[Dict[str, Any]] = None
    trace: Optional[Dict[str, Any]] = None
    thinking: Optional[List[str]] = Field(default_factory=list)
    answerPayload: Optional[AnswerPayload] = None
    conversationId: Optional[str] = None
    conversationType: Optional[str] = None
    groupId: Optional[str] = None
    syncToGraph: bool = True


class ChatMessageSyncRequest(BaseModel):
    messages: List[ChatMessageSyncItem]


class ChatMessageSyncResponse(BaseModel):
    success: bool
    ingested: int
    failed: int
    message: str


class GroupSeedItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    avatar: Optional[str] = None
    memberIds: List[str] = Field(default_factory=list)


class BootstrapRequest(BaseModel):
    users: List[Dict[str, Any]] = Field(default_factory=list)
    groups: List[GroupSeedItem] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class AuthRegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    avatar: Optional[str] = None
    team: List[str] = Field(default_factory=list)


class AuthLoginRequest(BaseModel):
    email: str
    password: str


class AuthSessionRequest(BaseModel):
    user_id: str


class ProfileUpdateRequest(BaseModel):
    name: str
    email: str
    avatar: Optional[str] = None


class ConversationMessageRequest(BaseModel):
    id: Optional[str] = None
    senderId: Optional[str] = None
    receiverId: Optional[str] = None
    groupId: Optional[str] = None
    content: str = ""
    sentAt: Optional[str] = None
    source: Optional[str] = "chat_message"
    attachment: Optional[Dict[str, Any]] = None
    trace: Optional[Dict[str, Any]] = None
    thinking: List[str] = Field(default_factory=list)
    answerPayload: Optional[AnswerPayload] = None
    role: Optional[str] = None
    isAiResponse: bool = False
    syncToGraph: bool = True


def _require_user_id(x_user_id: Optional[str]) -> str:
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing authenticated user id")
    return user_id


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raise_graph_unavailable(operation: str, exc: Exception) -> None:
    logger.error("%s failed because Neo4j is unavailable: %s", operation, exc)
    raise HTTPException(
        status_code=503,
        detail=f"Graph database is unavailable while {operation}. Check Neo4j connectivity and try again.",
    ) from exc


def _extract_document_text(file_path: Path, filename: str) -> str:
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        return utils.extract_text_from_pdf(str(file_path))
    if lowered.endswith(".txt"):
        return file_path.read_text(encoding="utf-8")
    if lowered.endswith(".docx"):
        try:
            return utils.extract_text_from_docx(str(file_path))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid DOCX file: {exc}") from exc
    raise HTTPException(
        status_code=400,
        detail="Unsupported file format. Only PDF, TXT, and DOCX are supported.",
    )


def _normalize_optional_text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def _parse_receiver_ids(receiver_id: Optional[str], receiver_ids_json: Optional[str]) -> List[str]:
    if isinstance(receiver_ids_json, str) and receiver_ids_json:
        try:
            parsed = json.loads(receiver_ids_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid receiver_ids_json: {exc}") from exc
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    if isinstance(receiver_id, str) and receiver_id:
        return [receiver_id]
    return []


def _is_sage_graph_ineligible_message(
    *,
    sender_id: Optional[str],
    receiver_id: Optional[str],
    conversation_type: Optional[str],
    source: Optional[str],
    is_ai_response: bool,
) -> bool:
    normalized_source = (source or "").strip().lower()
    normalized_conversation_type = (conversation_type or "").strip().lower()
    if normalized_conversation_type == "sage":
        return True
    if sender_id == chat_store.SAGE_USER_ID or receiver_id == chat_store.SAGE_USER_ID:
        return True
    if normalized_source.startswith("sage_"):
        return True
    return bool(is_ai_response and sender_id == chat_store.SAGE_USER_ID)


def _derive_conversation_id(
    *,
    sender_id: Optional[str],
    receiver_id: Optional[str],
    conversation_id: Optional[str],
    conversation_type: Optional[str],
    group_id: Optional[str],
) -> Optional[str]:
    if conversation_id:
        return conversation_id
    if conversation_type == "group" and group_id:
        return chat_store.stable_group_conversation_id(group_id)
    if conversation_type == "sage":
        owner_id = receiver_id if sender_id == chat_store.SAGE_USER_ID else sender_id
        return chat_store.stable_sage_conversation_id(owner_id) if owner_id else None
    if sender_id and receiver_id:
        return chat_store.stable_direct_conversation_id(sender_id, receiver_id)
    return None


def _build_document_payload(
    *,
    doc_id: str,
    content: str,
    filename: str,
    sender_id: Optional[str],
    receiver_id: Optional[str],
    receiver_ids_json: Optional[str],
    group_id: Optional[str],
    conversation_id: Optional[str],
    conversation_type: Optional[str],
    source: Optional[str],
    sent_at: Optional[str],
    attachment_name: Optional[str],
    attachment_type: Optional[str],
    attachment_url: Optional[str],
    linked_message_id: Optional[str],
) -> Dict[str, Any]:
    structured = services.extract_structured_data(content, doc_id)
    normalized_sender_id = _normalize_optional_text(sender_id)
    normalized_receiver_id = _normalize_optional_text(receiver_id)
    normalized_receiver_ids_json = _normalize_optional_text(receiver_ids_json)
    normalized_group_id = _normalize_optional_text(group_id)
    normalized_conversation_id = _normalize_optional_text(conversation_id)
    normalized_conversation_type = _normalize_optional_text(conversation_type)
    normalized_source = _normalize_optional_text(source)
    normalized_sent_at = _normalize_optional_text(sent_at)
    normalized_attachment_name = _normalize_optional_text(attachment_name)
    normalized_attachment_type = _normalize_optional_text(attachment_type)
    normalized_attachment_url = _normalize_optional_text(attachment_url)
    normalized_linked_message_id = _normalize_optional_text(linked_message_id)

    receivers = _parse_receiver_ids(normalized_receiver_id, normalized_receiver_ids_json) or structured.get("receivers") or []
    subject = normalized_attachment_name or filename or structured.get("subject") or "Uploaded document"
    sender = normalized_sender_id or structured.get("sender") or "Unknown"
    actual_source = normalized_source or ("message_attachment" if normalized_linked_message_id else "document_upload")
    actual_conversation_id = _derive_conversation_id(
        sender_id=sender,
        receiver_id=normalized_receiver_id,
        conversation_id=normalized_conversation_id,
        conversation_type=normalized_conversation_type,
        group_id=normalized_group_id,
    )
    return {
        "doc_id": doc_id,
        "sender": sender,
        "receivers": receivers,
        "subject": subject,
        "content": content,
        "timestamp": normalized_sent_at or _utcnow_iso(),
        "source": actual_source,
        "conversation_type": normalized_conversation_type,
        "conversation_id": actual_conversation_id,
        "group_id": normalized_group_id,
        "attachment_name": normalized_attachment_name or filename,
        "attachment_type": normalized_attachment_type,
        "attachment_url": normalized_attachment_url,
        "origin_message_id": normalized_linked_message_id,
        "linked_message_id": normalized_linked_message_id,
        "trace_json": None,
        "graph_sync_status": chat_store.GRAPH_SYNC_READY,
    }


def upsert_user_in_neo4j(user_data: UserSyncRequest) -> bool:
    try:
        chat_store.bootstrap_seed_data([user_data.model_dump()], [], [])
        return True
    except Exception as exc:
        logger.error("Failed to sync user %s to Neo4j: %s", user_data.id, exc)
        return False


def store_chat_message_in_neo4j(message: ChatMessageSyncItem) -> bool:
    content = (message.content or "").strip()
    if not content or not message.syncToGraph:
        return False
    if _is_sage_graph_ineligible_message(
        sender_id=message.senderId,
        receiver_id=message.receiverId,
        conversation_type=message.conversationType,
        source=message.source,
        is_ai_response=False,
    ):
        return False
    try:
        conversation_id = _derive_conversation_id(
            sender_id=message.senderId,
            receiver_id=message.receiverId,
            conversation_id=message.conversationId,
            conversation_type=message.conversationType,
            group_id=message.groupId,
        )
        payload = {
            "doc_id": f"chat-msg-{message.id}",
            "sender": message.senderId,
            "receivers": [message.receiverId] if message.receiverId else [],
            "subject": f"Chat message {message.id}",
            "content": content,
            "timestamp": message.timestamp,
            "source": message.source or "chat_message",
            "conversation_type": message.conversationType,
            "conversation_id": conversation_id,
            "group_id": message.groupId,
            "attachment_name": (message.attachment or {}).get("name"),
            "attachment_type": (message.attachment or {}).get("type"),
            "attachment_url": (message.attachment or {}).get("url"),
            "origin_message_id": message.id,
            "linked_message_id": None,
            "trace_json": json.dumps(message.trace) if message.trace else None,
            "graph_sync_status": chat_store.GRAPH_SYNC_READY,
        }
        stored = store_in_neo4j(payload)
        if stored:
            try:
                saia.process_chat_message(
                    message_id=message.id,
                    sender_id=message.senderId,
                    receiver_ids=payload["receivers"],
                    conversation_id=conversation_id,
                    conversation_type=message.conversationType,
                    group_id=message.groupId,
                    sent_at=message.timestamp,
                    content=content,
                    source=message.source or "chat_message",
                    trace=message.trace,
                    is_ai_response=False,
                    attachment_name=(message.attachment or {}).get("name"),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("SAIA processing failed for synced message %s: %s", message.id, exc)
        return stored
    except Exception as exc:
        logger.error("Failed to sync message %s: %s", message.id, exc)
        return False


@app.post("/api/bootstrap")
async def bootstrap_endpoint(request: BootstrapRequest):
    try:
        result = chat_store.bootstrap_seed_data(
            users=request.users,
            groups=[group.model_dump() for group in request.groups],
            messages=request.messages,
        )
    except (ServiceUnavailable, SessionExpired) as exc:
        _raise_graph_unavailable("bootstrapping seed data", exc)
    return {"success": True, **result}


@app.post("/api/auth/register")
async def auth_register_endpoint(request: AuthRegisterRequest):
    try:
        user = chat_store.register_user(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return user


@app.post("/api/auth/login")
async def auth_login_endpoint(request: AuthLoginRequest):
    user = chat_store.authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return user


@app.post("/api/auth/session")
async def auth_session_endpoint(request: AuthSessionRequest):
    user = chat_store.get_user_by_id(request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.put("/api/profile")
async def update_profile_endpoint(
    request: ProfileUpdateRequest,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = _require_user_id(x_user_id)
    try:
        return chat_store.update_user_profile(user_id, request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/groups")
async def list_groups_endpoint(x_user_id: Optional[str] = Header(default=None)):
    user_id = _require_user_id(x_user_id)
    return {"groups": chat_store.list_groups_for_user(user_id)}


@app.get("/api/conversations")
async def list_conversations_endpoint(x_user_id: Optional[str] = Header(default=None)):
    user_id = _require_user_id(x_user_id)
    return {"conversations": chat_store.list_conversation_summaries(user_id)}


@app.get("/api/conversations/{conversation_id}/messages")
async def list_conversation_messages_endpoint(
    conversation_id: str,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = _require_user_id(x_user_id)
    try:
        return {"messages": chat_store.get_conversation_messages(user_id, conversation_id)}
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/api/conversations/{conversation_id}/messages")
async def create_conversation_message_endpoint(
    conversation_id: str,
    request: ConversationMessageRequest,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = _require_user_id(x_user_id)
    try:
        return chat_store.create_message_for_conversation(
            authenticated_user_id=user_id,
            conversation_id=conversation_id,
            payload=request.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/api/messages/{message_id}/read")
async def mark_message_read_endpoint(
    message_id: str,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = _require_user_id(x_user_id)
    try:
        return {"success": True, **chat_store.mark_message_read(user_id, message_id)}
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/api/messages/{message_id}/saia")
async def get_message_saia_insight_endpoint(
    message_id: str,
    x_user_id: Optional[str] = Header(default=None),
):
    user_id = _require_user_id(x_user_id)
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            access_rows = session.run(
                """
                MATCH (u:User {id: $user_id})-[:PARTICIPATES_IN]->(:Conversation)<-[:IN_CONVERSATION]-(m:Message {id: $message_id})
                RETURN m.id AS id
                LIMIT 1
                """,
                user_id=user_id,
                message_id=message_id,
            ).data()
            if not access_rows:
                raise HTTPException(status_code=403, detail="Message not found or access denied")
            return saia.collect_message_insight(session, message_id)
    finally:
        driver.close()


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    graph_result = query_graph_with_trace(message, user_id=request.user_id)
    ai_result = generate_groq_response(
        message,
        graph_result.get("documents") or [],
        user_id=request.user_id,
        retrieval_trace=graph_result.get("trace"),
    )
    trace = {**(graph_result.get("trace") or {}), **((ai_result.get("trace") or {}) if isinstance(ai_result, dict) else {})}
    answer_payload = ai_result.get("answer_payload") or {
        "schema_version": 1,
        "mode": "short",
        "reason_code": "fallback_invalid_json",
        "summary": ai_result.get("answer", "") or "I couldn't produce a readable answer from the available evidence.",
        "bullets": [],
        "explanation": "SAGE returned a safe short answer because the detailed response could not be formatted reliably.",
        "evidence_refs": services._derive_evidence_refs(retrieval_trace=trace),
    }

    return {
        "answer": answer_payload.get("summary", ai_result.get("answer", "")),
        "answer_payload": answer_payload,
        "thinking": ai_result.get("thinking") or [],
        "trace": trace,
    }


@app.post("/api/process-document", response_model=DocumentProcessResponse)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sender_id: Optional[str] = Form(default=None),
    receiver_id: Optional[str] = Form(default=None),
    receiver_ids_json: Optional[str] = Form(default=None),
    group_id: Optional[str] = Form(default=None),
    conversation_id: Optional[str] = Form(default=None),
    conversation_type: Optional[str] = Form(default=None),
    source: Optional[str] = Form(default=None),
    sent_at: Optional[str] = Form(default=None),
    linked_message_id: Optional[str] = Form(default=None),
    attachment_name: Optional[str] = Form(default=None),
    attachment_type: Optional[str] = Form(default=None),
    attachment_url: Optional[str] = Form(default=None),
):
    del background_tasks

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    stored_path = UPLOAD_DIR / f"{_utcnow_iso().replace(':', '-')}-{file.filename}"
    try:
        with stored_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)

        content = _extract_document_text(stored_path, file.filename)
        normalized_linked_message_id = _normalize_optional_text(linked_message_id)
        doc_id = (
            f"message-attachment-{normalized_linked_message_id}"
            if normalized_linked_message_id
            else utils.generate_doc_id(content)
        )
        payload = _build_document_payload(
            doc_id=doc_id,
            content=content,
            filename=file.filename,
            sender_id=sender_id,
            receiver_id=receiver_id,
            receiver_ids_json=receiver_ids_json,
            group_id=group_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            source=source,
            sent_at=sent_at,
            attachment_name=attachment_name,
            attachment_type=attachment_type or file.content_type,
            attachment_url=attachment_url,
            linked_message_id=normalized_linked_message_id,
        )

        if not store_in_neo4j(payload):
            raise HTTPException(status_code=500, detail="Failed to store document in the graph database.")

        if payload["source"] == "message_attachment" and normalized_linked_message_id:
            try:
                saia.process_message_attachment(
                    doc_id=payload["doc_id"],
                    linked_message_id=normalized_linked_message_id,
                    sender_id=payload["sender"],
                    receiver_ids=payload["receivers"],
                    conversation_id=payload["conversation_id"],
                    conversation_type=payload["conversation_type"],
                    group_id=payload["group_id"],
                    sent_at=payload["timestamp"],
                    content=payload["content"],
                    source=payload["source"],
                    attachment_name=payload["attachment_name"],
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("SAIA processing failed for attachment %s: %s", payload["doc_id"], exc)

        return {
            "doc_id": payload["doc_id"],
            "sender": payload["sender"],
            "receivers": payload["receivers"],
            "subject": payload["subject"],
            "success": True,
            "message": "Document processed and stored in the graph database.",
        }
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        if stored_path.exists():
            stored_path.unlink()


@app.get("/api/debug-graph", response_model=GraphDebugResponse)
async def debug_graph_endpoint(summary_only: bool = False):
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            node_counts = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] AS Label, count(*) AS Count
                ORDER BY Count DESC
                """
            ).data()
            rel_counts = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS RelationType, count(*) AS Count
                ORDER BY Count DESC
                """
            ).data()
            if summary_only:
                return {
                    "node_counts": node_counts,
                    "rel_counts": rel_counts,
                    "sample_docs": [],
                    "connectivity": [],
                    "entity_doc_connections": [],
                }
            sample_docs = session.run(
                """
                MATCH (d:Document)
                RETURN d.doc_id AS DocID,
                       d.subject AS Subject,
                       d.sender AS Sender,
                       d.timestamp AS Timestamp,
                       d.source AS Source
                ORDER BY d.timestamp DESC
                LIMIT 5
                """
            ).data()
            connectivity = session.run(
                """
                MATCH (n)
                WHERE NOT (n)--()
                RETURN labels(n)[0] AS IsolatedNodeType, count(*) AS Count
                ORDER BY Count DESC
                """
            ).data()
            entity_doc_connections = session.run(
                """
                MATCH (p:Person)
                RETURN p.id AS Person,
                       p.name AS Name,
                       COUNT { (p)--() } AS ConnectionCount
                ORDER BY ConnectionCount DESC
                LIMIT 10
                """
            ).data()

        return {
            "node_counts": node_counts,
            "rel_counts": rel_counts,
            "sample_docs": sample_docs,
            "connectivity": connectivity,
            "entity_doc_connections": entity_doc_connections,
        }
    except Exception as exc:
        logger.error("Graph debug failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to inspect graph: {exc}") from exc
    finally:
        driver.close()


def _node_to_path_node(node: Any) -> GraphPathNode:
    labels = list(getattr(node, "labels", []))
    element_id = getattr(node, "element_id", None) or getattr(node, "id", None) or str(node)

    props: Dict[str, Any]
    try:
        props = dict(node.items())
    except Exception:
        try:
            props = dict(node)
        except Exception:
            props = {}

    display_name = None
    for key in ("subject", "title", "name", "doc_id", "chunk_id", "id", "value"):
        value = props.get(key)
        if value:
            display_name = str(value)
            break
    if not display_name:
        display_name = str(labels[0]) if labels else "Node"

    return GraphPathNode(
        element_id=str(element_id),
        labels=[str(label) for label in labels],
        display_name=display_name,
        properties={str(k): v for k, v in props.items()},
    )


@app.get("/api/debug-retrieval-path", response_model=GraphRetrievalPathResponse)
async def debug_retrieval_path_endpoint(
    chunk_id: str,
    user_id: Optional[str] = None,
    related_node_id: Optional[str] = None,
    relationship: Optional[str] = None,
):
    """Return the concrete Neo4j hop path for a retrieved evidence chunk.

    The path mirrors the retrieval pattern used by the vector query:
    - Global: (Document)<-[:PART_OF]-(Chunk)-[r]-(n)
    - Personalized: (Person)-[:SENT|RECEIVED_BY]-(Document)<-[:PART_OF]-(Chunk)-[r]-(n)
    """

    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            rel_filter = ""
            base_params: Dict[str, Any] = {"chunk_id": chunk_id}

            if relationship:
                rel_filter = ":" + relationship

            related_where_clause = ""
            if related_node_id:
                base_params["related_node_id"] = related_node_id
                related_where_clause = "WHERE n IS NULL OR elementId(n) = $related_node_id OR n.id = $related_node_id"

            user_query = f"""
            MATCH (c:Chunk {{chunk_id: $chunk_id}})-[:PART_OF]->(d:Document)
            MATCH (p:Person {{id: $user_id}})-[pd:SENT|RECEIVED_BY]-(d)
            OPTIONAL MATCH (c)-[r{rel_filter}]-(n)
            {related_where_clause}
            WITH p, pd, d, c, r, n
            RETURN p AS p, type(pd) AS pd_type, d AS d, c AS c, type(r) AS r_type, n AS n
            LIMIT 1
            """

            global_query = f"""
            MATCH (c:Chunk {{chunk_id: $chunk_id}})-[:PART_OF]->(d:Document)
            OPTIONAL MATCH (c)-[r{rel_filter}]-(n)
            {related_where_clause}
            WITH d, c, r, n
            RETURN d AS d, c AS c, type(r) AS r_type, n AS n
            LIMIT 1
            """

            rows: List[Dict[str, Any]] = []
            if user_id:
                params = dict(base_params)
                params["user_id"] = user_id
                rows = session.run(user_query, **params).data()

                # Fallback: evidence can be global even when the query is user-scoped.
                # In that case, the (Person)-[:SENT|RECEIVED_BY]-(Document) hop won't exist.
                if not rows:
                    rows = session.run(global_query, **base_params).data()
            else:
                rows = session.run(global_query, **base_params).data()

            if not rows:
                raise HTTPException(status_code=404, detail="No retrieval path found for the provided chunk_id")

            row = rows[0]

            nodes: List[GraphPathNode] = []
            relationships_out: List[GraphPathRelationship] = []

            # Core nodes
            document_node = _node_to_path_node(row["d"])
            chunk_node = _node_to_path_node(row["c"])

            # Optional nodes
            person_node = _node_to_path_node(row["p"]) if "p" in row and row.get("p") is not None else None
            related_node = _node_to_path_node(row["n"]) if row.get("n") is not None else None

            if person_node:
                nodes.append(person_node)
            nodes.extend([document_node, chunk_node])
            if related_node:
                nodes.append(related_node)

            # Relationships as a simple chain
            if person_node:
                relationships_out.append(
                    GraphPathRelationship(
                        type=str(row.get("pd_type") or "SENT_OR_RECEIVED"),
                        source=person_node.element_id,
                        target=document_node.element_id,
                    )
                )

            relationships_out.append(
                GraphPathRelationship(
                    type="PART_OF",
                    source=chunk_node.element_id,
                    target=document_node.element_id,
                )
            )

            if related_node and row.get("r_type"):
                relationships_out.append(
                    GraphPathRelationship(
                        type=str(row.get("r_type") or "RELATED_TO"),
                        source=chunk_node.element_id,
                        target=related_node.element_id,
                    )
                )

            return GraphRetrievalPathResponse(
                hop_count=len(relationships_out),
                nodes=nodes,
                relationships=relationships_out,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Debug retrieval path failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to inspect retrieval path: {exc}") from exc
    finally:
        driver.close()


@app.get("/api/debug-subgraph", response_model=GraphSubgraphResponse)
async def debug_subgraph_endpoint(
    chunk_id: str,
    depth: int = 2,
    node_limit: int = 80,
    rel_limit: int = 120,
):
    """Return a neighborhood subgraph around a chunk for visualization.

    This is meant for UI debugging: show the *actual* Neo4j nodes/relationships
    near the Top-1 evidence chunk, up to `depth` hops.
    """

    safe_depth = max(1, min(int(depth), 4))
    safe_node_limit = max(10, min(int(node_limit), 250))
    safe_rel_limit = max(10, min(int(rel_limit), 400))

    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            nodes_query = f"""
            MATCH (c:Chunk {{chunk_id: $chunk_id}})
            MATCH p=(c)-[*0..{safe_depth}]-(m)
            UNWIND nodes(p) AS n
            WITH DISTINCT n
            RETURN elementId(n) AS element_id,
                   labels(n) AS labels,
                   properties(n) AS props
            LIMIT $node_limit
            """

            rels_query = f"""
            MATCH (c:Chunk {{chunk_id: $chunk_id}})
            MATCH p=(c)-[*1..{safe_depth}]-(m)
            UNWIND relationships(p) AS r
            WITH DISTINCT r
            RETURN elementId(startNode(r)) AS source,
                   elementId(endNode(r)) AS target,
                   type(r) AS type
            LIMIT $rel_limit
            """

            node_rows = session.run(
                nodes_query,
                chunk_id=chunk_id,
                node_limit=safe_node_limit,
            ).data()
            rel_rows = session.run(
                rels_query,
                chunk_id=chunk_id,
                rel_limit=safe_rel_limit,
            ).data()

        nodes_out: List[GraphPathNode] = []
        for row in node_rows:
            labels = [str(label) for label in (row.get("labels") or [])]
            element_id = str(row.get("element_id") or "")
            props = row.get("props") or {}
            if not isinstance(props, dict):
                try:
                    props = dict(props)
                except Exception:
                    props = {}

            display_name = None
            for key in ("subject", "title", "name", "doc_id", "chunk_id", "id", "value"):
                value = props.get(key)
                if value:
                    display_name = str(value)
                    break
            if not display_name:
                display_name = labels[0] if labels else "Node"

            nodes_out.append(
                GraphPathNode(
                    element_id=element_id,
                    labels=labels,
                    display_name=display_name,
                    properties={str(k): v for k, v in props.items()},
                )
            )

        rels_out = [
            GraphPathRelationship(
                type=str(row.get("type") or "RELATED_TO"),
                source=str(row.get("source") or ""),
                target=str(row.get("target") or ""),
            )
            for row in rel_rows
            if row.get("source") and row.get("target")
        ]

        if not nodes_out:
            raise HTTPException(status_code=404, detail="No subgraph found for the provided chunk_id")

        return GraphSubgraphResponse(
            depth=safe_depth,
            nodes=nodes_out,
            relationships=rels_out,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Debug subgraph failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to load subgraph: {exc}") from exc
    finally:
        driver.close()


@app.post("/api/sync-user", response_model=UserSyncResponse)
async def sync_user_endpoint(request: UserSyncRequest):
    if not upsert_user_in_neo4j(request):
        raise HTTPException(status_code=500, detail="Failed to sync user to the graph database.")
    return {
        "success": True,
        "user_id": request.id,
        "message": "User synced to the graph database.",
    }


@app.post("/api/sync-messages", response_model=ChatMessageSyncResponse)
async def sync_messages_endpoint(request: ChatMessageSyncRequest):
    ingested = 0
    failed = 0

    for item in request.messages:
        if store_chat_message_in_neo4j(item):
            ingested += 1
        else:
            failed += 1

    return {
        "success": failed == 0,
        "ingested": ingested,
        "failed": failed,
        "message": "Messages synced to the graph database." if failed == 0 else "Some messages failed to sync.",
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
