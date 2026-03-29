"""FastAPI backend for the SAGE app.

This file exposes the HTTP API for chat, document ingestion, graph debugging,
and health checks while delegating shared business logic to app services.
"""

import logging
import shutil
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import app.services as services
    import app.utils as utils
    from app.message_processor import store_in_neo4j
except ImportError:
    import services
    import utils
    from message_processor import store_in_neo4j


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
generate_groq_response = services.generate_groq_response

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    answer: str
    thinking: List[str] = Field(default_factory=list)

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

# API Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message and return a response using graph-based RAG.
    """
    try:
        # Query the graph for relevant information
        graph_results = query_graph(request.message)

        # Generate response using Groq
        response = generate_groq_response(request.message, graph_results)

        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        # Return a graceful error response instead of raising an exception
        return {
            "answer": "I apologize, but I ran into a small issue while trying to answer your question. Would you mind trying again? I'm here and ready to assist you as soon as we can get past this technical glitch!",
            "thinking": [f"Error: {str(e)}"]
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

        # Store in Neo4j (run in background to avoid blocking)
        background_tasks.add_task(store_in_neo4j, structured_data)

        return {
            "doc_id": doc_id,
            "sender": structured_data["sender"],
            "receivers": structured_data["receivers"],
            "subject": structured_data["subject"],
            "success": True,
            "message": "Document processing started. Data will be stored in the graph database."
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

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Run the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend:app", host="0.0.0.0", port=8000, reload=True)
