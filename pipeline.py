import streamlit as st
import os
import hashlib
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from PyPDF2 import PdfReader

# Add after other imports
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Neo4j credentials from .env
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j connection
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Streamlit UI Setup
st.set_page_config(page_title="PDF Chatbot with Knowledge Graph", layout="wide")
st.title("📄🔗 PDF Chatbot + Knowledge Graph in Neo4j")

# Load PDF files from 'files' directory
pdf_files = [f for f in os.listdir("files") if f.endswith(".pdf")]
selected_file = st.selectbox("📂 Select a PDF file:", pdf_files)

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to generate embeddings
def generate_embedding(text):
    model = get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()

if st.button("Process PDF"):
    if selected_file:
        file_path = os.path.join("files", selected_file)

        def generate_doc_id(content: str) -> str:
            """Generate a unique document ID based on the hashed content."""
            return hashlib.sha256(content.encode()).hexdigest()

        # Extract text from PDF
        def extract_text_from_pdf(file_path):
            with open(file_path, "rb") as f:
                pdf_reader = PdfReader(f)
                text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            return text

        text_data = extract_text_from_pdf(file_path)
        st.success(f"Extracted text from {selected_file}")

        # Generate a unique doc_id using hashed content
        doc_id = generate_doc_id(text_data)

        # Initialize LLM
        llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        # Define detailed entity extraction schema
        schema = {
            "type": "object",
            "properties": {
                "doc_id": { "type": "string" },
                "doc_type": {
                "type": "string",
                "enum": [
                    "email",
                    "report",
                    "contract",
                    "document",
                    "other"
                ]
                },
                "title": { "type": "string" },
                "content": { "type": "string" },
                "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "name": { "type": "string" },
                    "type": { "type": "string" }
                    }
                }
                },
                "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "source": { "type": "string" },
                    "target": { "type": "string" },
                    "relation": { "type": "string" }
                    }
                }
                }
            },
            "required": [
                "doc_id",
                "doc_type",
                "entities",
                "relationships"
            ]
        }

        # JSON Output Parser
        parser = JsonOutputParser(pydantic_object=schema)

        # Define LLM Prompt
        prompt = ChatPromptTemplate.from_template(
            """
            You are an advanced document intelligence system designed to extract structured information from enterprise documents.  
            Your goal is to convert unstructured text into a well-defined knowledge graph representation in JSON format.

            **Instructions:**
            1️⃣ **Extract Entities**: Identify key entities and classify them into **relevant types**:
            - **Documents**: title, document type, author, version, date, content summary  
            - **People**: names, roles, organizations, emails, phone numbers  
            - **Projects**: project names, IDs, stakeholders  
            - **Financials**: invoices, vendors, amounts, billing dates  
            - **Agile/SDLC Data**: sprint details, requirements, testing notes  

            2️⃣ **Extract Relationships**:  
            - Define meaningful **connections** between entities using **precise relationship types**.  
            - Example: `"source": "Project", "target": "Document", "type": "specifies"`  

            3️⃣ **Preserve Document Structure**:  
            - Extract **hierarchical sections** and subsections.  
            - Maintain **traceability links** between sections.  

            4️⃣ **Ensure Accuracy**:  
            - Avoid hallucinating information.  
            - Retain original terminology used in the document.  
            - Include confidence scores for each entity extraction.  

            **Output Format:**  
            Provide the structured data in **JSON format** following this schema:  
            ```json
            {{
                "doc_id": "<hashed_document_id>",
                "title": "<document_title>",
                "doc_type": "<document_type>",
                "entities": [
                    {{"name": "<entity_name>", "type": "<entity_type>", "confidence": <score>}}
                ],
                "relationships": [
                    {{"source": "<entity_1>", "target": "<entity_2>", "type": "<relationship_type>"}}
                ],
                "document_structure": [
                    {{"title": "<section_title>", "content": "<section_content>", "subsections": [...]}}
                ]
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
            structured_data = chain.invoke({"input": text_data})
            st.json(structured_data)

        # Function to clean data (remove None or empty string values)
        def clean_properties(props):
            return {k: v for k, v in props.items() if v not in [None, ""]}

        # Store in Neo4j
        def store_in_neo4j(data):
            driver = get_neo4j_driver()
            # Use content if available; otherwise, fall back to document_info.document_type if exists
            doc_id = generate_doc_id(data.get("content", data.get("document_info", {}).get("document_type", "")))

            with driver.session() as session:
                # Prepare Document properties dynamically
                doc_props = {
                    "title": data.get("title", "Untitled"),
                    "doc_type": data.get("doc_type", "other"),
                    "subject": data.get("subject", ""),
                    "creation_date": data.get("creation_date", "1970-01-01T00:00:00Z"),
                    "modified_date": data.get("modified_date", "1970-01-01T00:00:00Z"),
                    "content": data.get("content", ""),
                    "summary": data.get("summary", ""),
                    "version": data.get("version", "1.0")
                }
                doc_props = clean_properties(doc_props)
                set_query = ", ".join([f"d.{k} = ${k}" for k in doc_props.keys()])
                query = f"""
                    MERGE (d:Document {{doc_id: $doc_id}})
                    SET {set_query}
                """
                params = {"doc_id": doc_id}
                params.update(doc_props)

                # Generate embeddings for document content
                content_for_embedding = data.get("title", "") + " " + data.get("content", "") + " " + data.get("summary", "")
                content_embedding = generate_embedding(content_for_embedding[:5000])  # Limit text length

                session.run(
                    """
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.title = $title, 
                        d.doc_type = $doc_type,
                        d.embedding = $embedding
                    """,
                    doc_id=doc_id, 
                    title=data["title"], 
                    doc_type=data["doc_type"],
                    embedding=content_embedding
                )

                # Store Entities (Global Entity Linking)
                for entity in data.get("entities", []):
                    entity_embedding = generate_embedding(entity["name"])
                    session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type,
                            e.embedding = $embedding
                        MERGE (e)-[:MENTIONED_IN]->(d)
                    """, name=entity["name"], type=entity["type"], doc_id=doc_id, embedding=entity_embedding)
                
                # Store Relationships
                for rel in data.get("relationships", []):
                    session.run("""
                        MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                        MERGE (a)-[:RELATION {type: $relation}]->(b)
                    """, source=rel["source"], target=rel["target"], relation=rel["type"])
                
                # Infer Cross-Document Relationships
                session.run("""
                    MATCH (p:Entity)-[:MENTIONED_IN]->(d1:Document), (p)-[:MENTIONED_IN]->(d2:Document)
                    WHERE d1 <> d2
                    MERGE (d1)-[:RELATED]->(d2)
                """)
                
                # Group Documents by Type
                session.run("""
                    MERGE (g:DocumentGroup {name: $doc_type})
                    WITH g
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[:PART_OF]->(g)
                """, doc_type=data.get("doc_type", "Other"), doc_id=doc_id)

                # Store Attachments
                for attachment in data.get("attachments", []):
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (a:Attachment {filename: $filename})
                        SET a.filetype = $filetype, a.filesize = $filesize, a.url = $url
                        MERGE (d)-[:HAS_ATTACHMENT]->(a)
                    """, 
                        doc_id=doc_id,
                        filename=attachment.get("filename", "Unknown"),
                        filetype=attachment.get("filetype", ""),
                        filesize=attachment.get("filesize", 0),
                        url=attachment.get("url", "")
                    )

                # Store Agile Details if available
                if "agile_details" in data:
                    agile = data["agile_details"]
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (a:AgileDetails {sprint_number: $sprint_number})
                        SET a.sprint_goal = $sprint_goal, 
                            a.retrospective_notes = $retrospective_notes
                        MERGE (d)-[:HAS_AGILE_DETAILS]->(a)
                    """, 
                        doc_id=doc_id,
                        sprint_number=agile.get("sprint_number", 0),
                        sprint_goal=agile.get("sprint_goal", ""),
                        retrospective_notes=agile.get("retrospective_notes", "")
                    )

                # Store SDLC Details if available
                if "sdlc_details" in data:
                    sdlc = data["sdlc_details"]
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (s:SDLCDetails {doc_id: $doc_id})
                        SET s.requirements = $requirements, 
                            s.design_decisions = $design_decisions, 
                            s.implementation_notes = $implementation_notes, 
                            s.testing_results = $testing_results,
                            s.deployment_notes = $deployment_notes, 
                            s.maintenance_notes = $maintenance_notes
                        MERGE (d)-[:HAS_SDLC_DETAILS]->(s)
                    """, 
                        doc_id=doc_id,
                        requirements=sdlc.get("requirements", ""),
                        design_decisions=sdlc.get("design_decisions", ""),
                        implementation_notes=sdlc.get("implementation_notes", ""),
                        testing_results=sdlc.get("testing_results", ""),
                        deployment_notes=sdlc.get("deployment_notes", ""),
                        maintenance_notes=sdlc.get("maintenance_notes", "")
                    )

                # Store Bill Details if available
                if "bill_details" in data:
                    bill = data["bill_details"]
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (b:Bill {invoice_number: $invoice_number})
                        SET b.vendor = $vendor, 
                            b.billing_date = $billing_date, 
                            b.due_date = $due_date,
                            b.total_amount = $total_amount, 
                            b.currency = $currency
                        MERGE (d)-[:HAS_BILL]->(b)
                    """, 
                        doc_id=doc_id,
                        invoice_number=bill.get("invoice_number", "Unknown"),
                        vendor=bill.get("vendor", ""),
                        billing_date=bill.get("billing_date"),
                        due_date=bill.get("due_date"),
                        total_amount=bill.get("total_amount", 0.0),
                        currency=bill.get("currency", "USD")
                    )

                # Store Meeting Details if available
                if "meeting_details" in data:
                    meeting = data["meeting_details"]
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (m:Meeting {meeting_date: $meeting_date})
                        SET m.location = $location, 
                            m.agenda = $agenda, 
                            m.decisions = $decisions,
                            m.minutes = $minutes
                        MERGE (d)-[:HAS_MEETING]->(m)
                    """, 
                        doc_id=doc_id,
                        meeting_date=meeting.get("meeting_date"),
                        location=meeting.get("location", ""),
                        agenda=meeting.get("agenda", ""),
                        decisions=meeting.get("decisions", ""),
                        minutes=meeting.get("minutes", "")
                    )

                # Store Project Details if available
                if "project_details" in data:
                    project = data["project_details"]
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (p:Project {project_id: $project_id})
                        SET p.project_name = $project_name, 
                            p.start_date = $start_date, 
                            p.end_date = $end_date, 
                            p.status = $status, 
                            p.description = $description
                        MERGE (d)-[:RELATED_TO_PROJECT]->(p)
                    """, 
                        doc_id=doc_id,
                        project_id=project.get("project_id", "Unknown"),
                        project_name=project.get("project_name", ""),
                        start_date=project.get("start_date"),
                        end_date=project.get("end_date"),
                        status=project.get("status", ""),
                        description=project.get("description", "")
                    )

                # Store Signatures if available
                if "signature" in data:
                    signature = data["signature"]
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (s:Signature {signed_date: $signed_date})
                        SET s.signer_name = $signer_name, 
                            s.signer_email = $signer_email, 
                            s.signature_image = $signature_image
                        MERGE (d)-[:SIGNED_BY]->(s)
                    """, 
                        doc_id=doc_id,
                        signed_date=signature.get("signed_date"),
                        signer_name=signature.get("signer", {}).get("name", "Unknown"),
                        signer_email=signature.get("signer", {}).get("email", ""),
                        signature_image=signature.get("signature_image", "")
                    )

                # Re-query the document node to confirm insertion
                result = session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d.doc_id AS doc_id, d.title AS title, d.doc_type AS doc_type, d.creation_date AS creation_date
                """, doc_id=doc_id)
                stored_document = result.single()

            driver.close()

            if stored_document:
                return {
                    "status": "Success",
                    "message": "Document stored successfully",
                    "document": dict(stored_document)
                }
            else:
                return {
                    "status": "Error",
                    "message": "Failed to retrieve document after insertion."
                }

        print("Extracted Relationships:", structured_data.get("relationships", []))

        if store_in_neo4j(structured_data):
            st.success("📊 Data successfully stored in Neo4j!")

    else:
        st.warning("Please select a PDF file to process.")
