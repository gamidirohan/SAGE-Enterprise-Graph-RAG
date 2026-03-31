"""FastAPI backend for the SAGE app.

This file exposes the HTTP API for chat, document ingestion, graph debugging,
and health checks while delegating shared business logic to app services.
"""

from pathlib import Path
import sys
import logging
import shutil
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import app.services as services
    import app.utils as utils
except ImportError:
    import services
    import utils


logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="SAGE Enterprise Graph RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = utils.ROOT_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

query_graph = services.query_graph
query_graph_with_trace = services.query_graph_with_trace
generate_groq_response = services.generate_groq_response
store_in_neo4j = services.store_in_neo4j

# Pydantic models for request/response validation
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


class ChatMessageSyncRequest(BaseModel):
    messages: List[ChatMessageSyncItem]


class ChatMessageSyncResponse(BaseModel):
    success: bool
    ingested: int
    failed: int
    message: str

def upsert_user_in_neo4j(user_data: UserSyncRequest) -> bool:
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            session.run(
                """
                MERGE (p:Person {id: $id})
                SET p.name = $name,
                    p.email = $email,
                    p.avatar = $avatar,
                    p.team = $team
                """,
                id=user_data.id,
                name=user_data.name,
                email=user_data.email,
                avatar=user_data.avatar,
                team=user_data.team or [],
            )
        return True
    except Exception as exc:
        logger.error(f"Failed to sync user {user_data.id} to Neo4j: {exc}")
        return False
    finally:
        driver.close()


def store_chat_message_in_neo4j(message: ChatMessageSyncItem) -> bool:
    content = (message.content or "").strip()
    if not content:
        return False

    try:
        return store_in_neo4j(
            {
                "doc_id": f"chat-msg-{message.id}",
                "sender": message.senderId,
                "receivers": [message.receiverId],
                "subject": f"Chat message {message.id}",
                "content": content,
            }
        )
    except Exception as exc:
        logger.error(f"Failed to sync message {message.id}: {exc}")
        return False

# API Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message and return a response using graph-based RAG.
    """
    try:
        user_id = request.user_id.strip() if request.user_id else None

        retrieval = query_graph_with_trace(request.message, user_id=user_id)
        graph_results = retrieval.get("documents", [])
        response = generate_groq_response(request.message, graph_results, user_id=user_id)
        response["trace"] = retrieval.get("trace", {})
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        # Return a graceful error response instead of raising an exception
        return {
            "answer": "I apologize, but I ran into a small issue while trying to answer your question. Would you mind trying again? I'm here and ready to assist you as soon as we can get past this technical glitch!",
            "thinking": [f"Error: {str(e)}"],
            "trace": {},
        }

@app.post("/api/process-document", response_model=DocumentProcessResponse)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Process an uploaded document and store it in the graph database.
    """
    try:
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from the document
        if file.filename.endswith('.pdf'):
            document_text = utils.extract_text_from_pdf(str(file_path))
        elif file.filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and TXT are supported.")

        # Generate document ID
        doc_id = utils.generate_doc_id(document_text)
        structured_data = services.extract_structured_data(document_text, doc_id)

        # Store in Neo4j synchronously so the caller only gets success after
        # the document is actually queryable.
        stored = store_in_neo4j(structured_data)
        if not stored:
            raise HTTPException(status_code=500, detail="Failed to store document in the graph database.")

        return {
            "doc_id": doc_id,
            "sender": structured_data["sender"],
            "receivers": structured_data["receivers"],
            "subject": structured_data["subject"],
            "success": True,
            "message": "Document processed and stored in the graph database."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Close the file
        file.file.close()

@app.get("/api/debug-graph", response_model=GraphDebugResponse)
async def debug_graph():
    """
    Analyze and return information about the graph structure.
    """
    driver = utils.create_neo4j_driver()
    try:
        with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS Label, count(*) AS Count
                ORDER BY Count DESC
            """).data()

            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS RelationType, count(*) AS Count
                ORDER BY Count DESC
            """).data()

            sample_docs = session.run("""
                MATCH (d:Document)
                RETURN d.doc_id AS DocID, d.subject AS Title, d.sender AS Sender
                LIMIT 5
            """).data()

            connectivity = session.run("""
                MATCH (n)
                WHERE NOT (n)--()
                RETURN labels(n)[0] AS IsolatedNodeType, count(*) AS Count
                ORDER BY Count DESC
            """).data()

            entity_doc_connections = session.run("""
                MATCH (p:Person)
                RETURN p.id AS Entity,
                    'Person' AS Type,
                    COUNT { (p)--() } AS ConnectionCount
                ORDER BY ConnectionCount DESC
                LIMIT 10
            """).data()

        return {
            "node_counts": node_counts,
            "rel_counts": rel_counts,
            "sample_docs": sample_docs,
            "connectivity": connectivity,
            "entity_doc_connections": entity_doc_connections
        }

    except Exception as e:
        logger.error(f"Error analyzing graph structure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing graph structure: {str(e)}")
    finally:
        driver.close()

@app.post("/api/sync-user", response_model=UserSyncResponse)
async def sync_user_endpoint(request: UserSyncRequest):
    ok = upsert_user_in_neo4j(request)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to sync user {request.id}")
    return {"success": True, "user_id": request.id, "message": "User synced to graph database."}

@app.post("/api/sync-messages", response_model=ChatMessageSyncResponse)
async def sync_messages_endpoint(request: ChatMessageSyncRequest):
    ingested = 0
    failed = 0
    for item in request.messages:
        if store_chat_message_in_neo4j(item):
            ingested += 1
        else:
            failed += 1

    success = failed == 0
    return {
        "success": success,
        "ingested": ingested,
        "failed": failed,
        "message": f"Processed {len(request.messages)} messages. Ingested: {ingested}, Failed: {failed}.",
    }

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Run the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend:app", host="0.0.0.0", port=8000, reload=True)
