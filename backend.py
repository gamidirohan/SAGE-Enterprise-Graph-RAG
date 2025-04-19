from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import hashlib
import numpy as np
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import shutil
from pathlib import Path
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize Neo4j connection
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize embedding model (cached)
_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return _embedding_model

# Function to generate embeddings
def generate_embedding(text):
    model = get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Function to generate document ID
def generate_doc_id(content: str) -> str:
    """Generate a unique document ID based on the hashed content."""
    return hashlib.sha256(content.encode()).hexdigest()

# Function to chunk document
def chunk_document(document_text, max_chunk_words=250, overlap_sentences=2):
    """
    Chunks a document into semantically meaningful units.

    Args:
        document_text (str): The document text.
        max_chunk_words (int): Maximum number of words per chunk.
        overlap_sentences (int): Number of sentences to overlap between chunks.

    Returns:
        list: A list of chunks.
    """
    paragraphs = document_text.split("\n\n")
    chunks = []
    previous_sentences = []

    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        current_chunk = []

        for sentence in sentences:
            current_chunk.append(sentence)

            if len(word_tokenize(" ".join(current_chunk))) > max_chunk_words:
                chunk_text = " ".join(previous_sentences + current_chunk[:-1])
                chunks.append(chunk_text)

                previous_sentences = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else current_chunk
                current_chunk = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else []

        if current_chunk:
            chunk_text = " ".join(previous_sentences + current_chunk)
            chunks.append(chunk_text)
            previous_sentences = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else current_chunk

    return chunks

# Function to store document in Neo4j
def store_in_neo4j(data, document_text):
    driver = get_neo4j_driver()
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b", 
        temperature=0.0, 
        model_kwargs={"response_format": {"type": "json_object"}},
        groq_api_key=GROQ_API_KEY
    )
    
    try:
        with driver.session() as session:
            # Create Document node
            document_summary = llm.invoke(f"Summarize this document, include the word json in the summary: {document_text}").content
            embedding = generate_embedding(document_summary[:5000])
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.sender = $sender, d.subject = $subject, d.content = $content, d.embedding = $embedding, d.summary = $summary
                """,
                doc_id=data["doc_id"],
                sender=data["sender"],
                subject=data["subject"],
                content=data["content"],
                embedding=embedding,
                summary=document_summary
            )

            # Chunk and create Chunk nodes
            chunks = chunk_document(document_text)
            for i, chunk in enumerate(chunks):
                chunk_summary = llm.invoke(f"Summarize this chunk, include the word json in the summary: {chunk}").content
                chunk_embedding = generate_embedding(chunk_summary)
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
                    doc_id=data["doc_id"]
                )

            # Create Sender and Receiver nodes and relationships
            session.run(
                """
                MERGE (s:Person {id: $sender_id})
                MERGE (d:Document {doc_id: $doc_id})
                MERGE (s)-[:SENT]->(d)
                """,
                sender_id=data["sender"],
                doc_id=data["doc_id"]
            )
            for receiver in data["receivers"]:
                session.run(
                    """
                    MERGE (r:Person {id: $receiver_id})
                    MERGE (d:Document {doc_id: $doc_id})
                    MERGE (d)-[:RECEIVED_BY]->(r)
                    """,
                    receiver_id=receiver,
                    doc_id=data["doc_id"]
                )
            
            return True
    except Exception as e:
        logger.error(f"Error storing in Neo4j: {str(e)}")
        raise
    finally:
        driver.close()

# Function to query Neo4j for related data
def query_graph(user_input):
    driver = get_neo4j_driver()
    model = get_embedding_model()
    query_embedding = model.encode(user_input)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    results = []
    try:
        with driver.session() as session:
            vector_query = """
                MATCH (c:Chunk)-[:PART_OF]->(d:Document)
                WHERE c.embedding IS NOT NULL
                WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
                WITH c, d, gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity
                ORDER BY similarity DESC
                LIMIT 3
                MATCH (c)-[r]-(n)
                RETURN c.summary AS chunk_summary, d, similarity, type(r) as relationship, n
            """
            vector_results = session.run(vector_query, query_embedding=query_embedding.tolist()).data()

            if vector_results:
                results = [
                    f"Chunk Summary: {item['chunk_summary']}, Document: {item['d']}, Similarity: {item['similarity']}, Relationship: {item['relationship']}, Related Node: {item['n']}"
                    for item in vector_results
                ]
            else:
                results = ["No relevant data found in the graph."]

    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        raise
    finally:
        driver.close()
        
    return results

# Function to generate response using Groq
def generate_groq_response(query, documents):
    if not documents:
        return "No relevant information found."
    
    context = "\n\n".join([item.split('Chunk Summary: ')[1].split(', Document: ')[0] for item in documents])
    prompt_template = ChatPromptTemplate.from_template(
        "Answer the following question based on the provided context:\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"
    )
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b", 
        temperature=0.0, 
        groq_api_key=GROQ_API_KEY
    )
    chain = prompt_template | llm | StrOutputParser()

    try:
        response = chain.invoke({"query": query, "context": context})
        
        # Process <think> tags
        think_parts = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
        answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
        return {
            "answer": answer,
            "thinking": think_parts if think_parts else []
        }
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM processing error: {str(e)}")

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    answer: str
    thinking: List[str] = []

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
        raise HTTPException(status_code=500, detail=str(e))

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
            document_text = extract_text_from_pdf(str(file_path))
        elif file.filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and TXT are supported.")
        
        # Generate document ID
        doc_id = generate_doc_id(document_text)
        
        # Initialize LLM for entity extraction
        llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}},
            groq_api_key=GROQ_API_KEY
        )
        
        # Define entity extraction schema
        schema = {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "sender": {"type": "string"},
                "receivers": {"type": "array", "items": {"type": "string"}},
                "subject": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["doc_id", "sender", "receivers", "subject", "content"]
        }
        
        # JSON Output Parser
        parser = JsonOutputParser(pydantic_object=schema)
        
        # Define LLM Prompt
        prompt = ChatPromptTemplate.from_template(
            """
            You are an advanced document intelligence system. Extract Sender, Receivers, Subject and content from the following document.

            **Instructions:**
            1. Extract the Sender ID.
            2. Extract the Receiver IDs as an array.
            3. Extract the Subject.
            4. Extract the main Content.

            **Output Format:**
            Provide the structured data in JSON format following this schema:
            ```json
            {{
                "doc_id": "<hashed_document_id>",
                "sender": "<sender_id>",
                "receivers": ["<receiver_id1>", "<receiver_id2>", ...],
                "subject": "<subject>",
                "content": "<content>"
            }}
            ```

            **Input Document:**
            ```
            {input}
            ```

            Remember to output ONLY valid JSON that follows the schema exactly.
            """
        )
        
        # Create processing pipeline
        chain = prompt | llm | parser
        
        # Extract structured entities & relationships
        structured_data = chain.invoke({"input": document_text})
        structured_data["doc_id"] = doc_id  # Ensure we use our generated doc_id
        
        # Store in Neo4j (run in background to avoid blocking)
        background_tasks.add_task(store_in_neo4j, structured_data, document_text)
        
        return {
            "doc_id": doc_id,
            "sender": structured_data["sender"],
            "receivers": structured_data["receivers"],
            "subject": structured_data["subject"],
            "success": True,
            "message": "Document processing started. Data will be stored in the graph database."
        }
        
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
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
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
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
