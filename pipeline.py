import streamlit as st
import os
import hashlib
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Load Neo4j credentials from .env
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j connection
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

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

# Streamlit UI Setup
st.set_page_config(page_title="Document Processor", layout="wide")
st.title("Document Processor")

# File selection from directory
files_dir = "files1"  # Directory containing files
file_names = [f for f in os.listdir(files_dir) if f.endswith(('.txt', '.pdf'))]

if not file_names:
    st.warning(f"No .txt or .pdf files found in directory: {files_dir}")
else:
    selected_file = st.selectbox("Select a file:", file_names)
    file_path = os.path.join(files_dir, selected_file)

    if st.button("Process Document"):
        if selected_file:
            try:
                if selected_file.endswith('.pdf'):
                    document_text = extract_text_from_pdf(file_path)
                elif selected_file.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        document_text = f.read()

                def generate_doc_id(content: str) -> str:
                    """Generate a unique document ID based on the hashed content."""
                    return hashlib.sha256(content.encode()).hexdigest()

                # Generate a unique doc_id using hashed content
                doc_id = generate_doc_id(document_text)

                # Initialize LLM
                llm = ChatGroq(
                    model_name="deepseek-r1-distill-llama-70b",
                    temperature=0.0,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )

                # Define simplified entity extraction schema
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
                with st.spinner("Processing with LLM..."):
                    structured_data = chain.invoke({"input": document_text})
                    st.json(structured_data)

                # Store in Neo4j
                def store_in_neo4j(data):
                    driver = get_neo4j_driver()
                    with driver.session() as session:
                        # Create Document node
                        embedding = generate_embedding(data["content"][:5000])
                        print("Generated Embedding:", embedding)
                        print("Embedding Data Type:", type(embedding))
                        print("Embedding First Element Data Type:", type(embedding[0]))
                        session.run(
                            """
                            MERGE (d:Document {doc_id: $doc_id})
                            SET d.sender = $sender, d.subject = $subject, d.content = $content, d.embedding = $embedding
                            """,
                            doc_id=data["doc_id"],
                            sender=data["sender"],
                            subject=data["subject"],
                            content=data["content"],
                            embedding=embedding
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
                    driver.close()
                    return True

                if store_in_neo4j(structured_data):
                    st.success("📊 Data successfully stored in Neo4j!")
            except Exception as e:
                st.error(f"Error processing document: {e}")
        else:
            st.warning("Please select a file.")