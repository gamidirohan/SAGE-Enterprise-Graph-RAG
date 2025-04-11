import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import hashlib
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK data
nltk.download('punkt', quiet=True)

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Neo4j connection
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize embedding model
def get_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

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

# Function to query Neo4j for related data
def query_graph(user_input):
    driver = get_neo4j_driver()
    model = get_embedding_model()
    query_embedding = model.encode(user_input)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    results = []
    with driver.session() as session:
        try:
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
            results = [f"Vector search failed: {str(e)}"]

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
    llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.0, groq_api_key=GROQ_API_KEY)
    chain = prompt_template | llm | StrOutputParser()

    try:
        response = chain.invoke({"query": query, "context": context})
        
        # Process <think> tags
        think_parts = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
        answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
        return answer, think_parts
    except Exception as e:
        return f"Groq API error: {str(e)}", []

# Function to store message in Neo4j
def store_message_in_neo4j(message_content, sender_id, chat_id):
    driver = get_neo4j_driver()
    
    # Generate a unique message ID
    message_id = hashlib.sha256(f"{sender_id}:{chat_id}:{message_content}:{str(os.urandom(8))}".encode()).hexdigest()
    
    # Generate embedding for the message
    embedding = generate_embedding(message_content)
    
    with driver.session() as session:
        try:
            # Create Message node
            session.run(
                """
                CREATE (m:Message {message_id: $message_id})
                SET m.content = $content, m.embedding = $embedding, m.sender_id = $sender_id, m.chat_id = $chat_id
                """,
                message_id=message_id,
                content=message_content,
                embedding=embedding,
                sender_id=sender_id,
                chat_id=chat_id
            )
            
            # Create relationship to sender
            session.run(
                """
                MATCH (m:Message {message_id: $message_id})
                MERGE (p:Person {id: $sender_id})
                CREATE (p)-[:SENT]->(m)
                """,
                message_id=message_id,
                sender_id=sender_id
            )
            
            return True
        except Exception as e:
            print(f"Error storing message in Neo4j: {e}")
            return False
        finally:
            driver.close()

# Function to process and store document in Neo4j
def process_document(file_path, filename, uploader_id):
    # Extract text from document
    if filename.endswith('.pdf'):
        document_text = extract_text_from_pdf(file_path)
    elif filename.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
    else:
        return None, "Unsupported file type"
    
    # Generate a unique doc_id
    doc_id = hashlib.sha256(document_text.encode()).hexdigest()
    
    # Initialize LLM
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.0
    )
    
    # Store in Neo4j
    driver = get_neo4j_driver()
    with driver.session() as session:
        try:
            # Create Document node
            document_summary = llm.invoke(f"Summarize this document in 2-3 sentences: {document_text[:5000]}").content
            embedding = generate_embedding(document_summary)
            
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.filename = $filename, d.uploader_id = $uploader_id, 
                    d.content = $content, d.embedding = $embedding, d.summary = $summary
                """,
                doc_id=doc_id,
                filename=filename,
                uploader_id=uploader_id,
                content=document_text[:1000],  # Store first 1000 chars of content
                embedding=embedding,
                summary=document_summary
            )
            
            # Chunk the document
            chunks = chunk_document(document_text)
            for i, chunk in enumerate(chunks):
                chunk_summary = llm.invoke(f"Summarize this chunk in 1-2 sentences: {chunk}").content
                chunk_embedding = generate_embedding(chunk_summary)
                
                session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.content = $content, c.embedding = $embedding, c.summary = $summary
                    MERGE (d:Document {doc_id: $doc_id})
                    MERGE (c)-[:PART_OF]->(d)
                    """,
                    chunk_id=f"{doc_id}-chunk-{i}",
                    content=chunk,
                    embedding=chunk_embedding,
                    summary=chunk_summary,
                    doc_id=doc_id
                )
            
            # Create relationship to uploader
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (p:Person {id: $uploader_id})
                MERGE (p)-[:UPLOADED]->(d)
                """,
                doc_id=doc_id,
                uploader_id=uploader_id
            )
            
            return doc_id, "Document processed successfully"
        except Exception as e:
            return None, f"Error processing document: {str(e)}"
        finally:
            driver.close()

# Function to chunk document
def chunk_document(document_text, max_chunk_words=250, overlap_sentences=2):
    """
    Chunks a document into semantically meaningful units.
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
