"""FastAPI backend for the SAGE app."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import shutil
import sys
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import app.chat_store as chat_store
    import app.services as services
    import app.utils as utils
except ImportError:
    import chat_store
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


class ChatResponse(BaseModel):
    answer: str
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
    return {
        "doc_id": doc_id,
        "sender": sender,
        "receivers": receivers,
        "subject": subject,
        "content": content,
        "timestamp": normalized_sent_at or _utcnow_iso(),
        "source": actual_source,
        "conversation_type": normalized_conversation_type,
        "conversation_id": normalized_conversation_id,
        "group_id": normalized_group_id,
        "attachment_name": normalized_attachment_name or filename,
        "attachment_type": normalized_attachment_type,
        "attachment_url": normalized_attachment_url,
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
    try:
        payload = {
            "doc_id": f"chat-msg-{message.id}",
            "sender": message.senderId,
            "receivers": [message.receiverId] if message.receiverId else [],
            "subject": f"Chat message {message.id}",
            "content": content,
            "timestamp": message.timestamp,
            "source": message.source or "chat_message",
            "conversation_type": message.conversationType,
            "group_id": message.groupId,
            "attachment_name": (message.attachment or {}).get("name"),
            "attachment_type": (message.attachment or {}).get("type"),
            "attachment_url": (message.attachment or {}).get("url"),
            "trace_json": json.dumps(message.trace) if message.trace else None,
            "graph_sync_status": chat_store.GRAPH_SYNC_READY,
        }
        return store_in_neo4j(payload)
    except Exception as exc:
        logger.error("Failed to sync message %s: %s", message.id, exc)
        return False


@app.post("/api/bootstrap")
async def bootstrap_endpoint(request: BootstrapRequest):
    result = chat_store.bootstrap_seed_data(
        users=request.users,
        groups=[group.model_dump() for group in request.groups],
        messages=request.messages,
    )
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


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    graph_result = query_graph_with_trace(message, user_id=request.user_id)
    ai_result = generate_groq_response(message, graph_result.get("documents") or [], user_id=request.user_id)
    trace = {**(graph_result.get("trace") or {}), **((ai_result.get("trace") or {}) if isinstance(ai_result, dict) else {})}

    return {
        "answer": ai_result.get("answer", ""),
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
async def debug_graph_endpoint():
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
