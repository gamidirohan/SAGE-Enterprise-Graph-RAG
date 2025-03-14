import streamlit as st
import os
import hashlib
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader

# Add these imports at the top
import numpy as np
from sentence_transformers import SentenceTransformer

# Add this function after the get_neo4j_driver function
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
FILES_DIR = "files"  # Directory containing PDFs

# Initialize Neo4j connection
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize LLM
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.0)

# Streamlit UI
st.set_page_config(page_title="SAGE: Graph-Based Chat", layout="wide")
st.title("📄🔗 SAGE: Graph + LLM Query Interface")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False  # Track if LLM is running

# Display existing chat history at the top
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    if "<think>" in bot_msg and "</think>" in bot_msg:
        bot_msg = bot_msg.replace("<think>", "🔍 **Chain of Thought**\n\n```")
        bot_msg = bot_msg.replace("</think>", "```\n")
        st.chat_message("assistant").markdown(bot_msg)
    else:
        st.chat_message("assistant").write(bot_msg)

# Add a Debug Graph Structure section
if st.checkbox("Debug Graph Structure"):
    with st.spinner("Analyzing graph structure..."):
        driver = get_neo4j_driver()
        try:
            with driver.session() as session:
                # Count nodes by label
                node_counts = session.run("""
                    MATCH (n) 
                    RETURN labels(n)[0] AS Label, count(*) AS Count
                    ORDER BY Count DESC
                """).data()
                
                # Count relationships by type
                rel_counts = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) AS RelationType, count(*) AS Count
                    ORDER BY Count DESC
                """).data()
                
                # Sample some documents
                sample_docs = session.run("""
                    MATCH (d:Document)
                    RETURN d.doc_id AS DocID, d.title AS Title, d.doc_type AS Type
                    LIMIT 5
                """).data()
                
                # Check connectivity
                connectivity = session.run("""
                    MATCH (n)
                    WHERE NOT (n)--()
                    RETURN labels(n)[0] AS IsolatedNodeType, count(*) AS Count
                    ORDER BY Count DESC
                """).data()
                
                # Check entity to document connections
                entity_doc_connections = session.run("""
                    MATCH (e:Entity)
                    RETURN e.name AS Entity, 
                           e.type AS Type,
                           size((e)--()) AS ConnectionCount
                    ORDER BY ConnectionCount DESC
                    LIMIT 10
                """).data()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Node Distribution")
                st.table(node_counts)
                
                st.subheader("Sample Documents")
                st.table(sample_docs)
                
            with col2:
                st.subheader("Relationship Distribution")
                st.table(rel_counts)
                
                st.subheader("Isolated Nodes")
                if connectivity:
                    st.table(connectivity)
                    st.warning(f"Found {sum(item['Count'] for item in connectivity)} isolated nodes!")
                else:
                    st.success("No isolated nodes found - good connectivity!")
            
            st.subheader("Top Connected Entities")
            st.table(entity_doc_connections)
            
            # Visualization option
            if st.button("Generate Graph Visualization"):
                viz_data = session.run("""
                    MATCH path = (e:Entity)-[r]-(d:Document)
                    RETURN path
                    LIMIT 25
                """).data()
                
                # Generate a basic visualization (this is a placeholder - you'd need to implement actual visualization)
                st.info("Graph visualization would appear here. Consider using Pyvis, NetworkX, or Neo4j Bloom.")
                
        except Exception as e:
            st.error(f"Error analyzing graph structure: {str(e)}")
        finally:
            driver.close()

# User selects query mode
query_mode = st.radio("Select Query Mode:", ["LLM Only", "Graph + PDF + LLM"])

# Function to find the PDF file by matching hash
def find_pdf_by_hash(doc_id):
    for filename in os.listdir(FILES_DIR):
        file_path = os.path.join(FILES_DIR, filename)
        with open(file_path, "rb") as f:
            text_data = extract_text_from_pdf(file_path)  # Hash extracted text, not raw bytes
            file_hash = hashlib.sha256(text_data.encode()).hexdigest()
            if file_hash == doc_id:
                return file_path
    return None

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

    results = []
    error_message = ""

    with driver.session() as session:
        try:
            # Vector Search using GDS
            st.write(f"Attempting vector search with embedding: {query_embedding}")

            vector_query = """
                    MATCH (d:Document)
                    WHERE d.embedding IS NOT NULL
                    WITH d, gds.similarity.cosine(d.embedding, $query_embedding) AS score
                    WHERE score > 0.7
                    RETURN d.doc_id AS doc_id, 
                           d.title AS document_title, 
                           d.doc_type AS document_type, 
                           score
                    ORDER BY score DESC
                    LIMIT 5
                """
            vector_results = session.run(vector_query, 
                                       query_embedding=query_embedding.tolist()).data()

            if vector_results:
                st.write("Vector search successful")
                results = vector_results
                driver.close()
                return results

        except Exception as e:
            st.error(f"Vector search failed: {str(e)}")
            st.write("Falling back to keyword search")

        # Fallback to keyword search
        st.write("Attempting keyword search")
        keyword_query = """
            MATCH (d:Document)
            WHERE toLower(d.title) CONTAINS toLower($user_input)
            OR toLower(d.content) CONTAINS toLower($user_input)
            RETURN d.doc_id AS doc_id, 
                   d.title AS document_title, 
                   d.doc_type AS document_type
            LIMIT 5
        """
        try:
            keyword_results = session.run(keyword_query, 
                                       user_input=user_input).data()
            if keyword_results:
                st.write("Keyword search successful")
                results = keyword_results
        except Exception as e:
            st.error(f"Keyword search failed: {str(e)}")
            st.write("Falling back to entity search")

        # If still no results, try entity-based search
        if not results:
            st.write("Attempting entity search")
            # In graph_rag.py, update the entity_query:
            entity_query = """
                MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                WHERE toLower(e.name) CONTAINS toLower($user_input)
                RETURN DISTINCT d.doc_id AS doc_id, 
                            d.title AS document_title, 
                            d.doc_type AS document_type
                LIMIT 5
            """
            try:
                entity_results = session.run(entity_query, 
                                           user_input=user_input).data()
                if entity_results:
                    st.write("Entity search successful")
                    results = entity_results
                else:
                    st.warning("No results found in entity search")
            except Exception as e:
                st.error(f"Entity search failed: {str(e)}")
                results = []

    driver.close()
    return results

# Custom chat input with dynamic button
def on_enter():
    if st.session_state.user_input:
        st.session_state.temp_input = st.session_state.user_input
        st.session_state.user_input = ""
        st.session_state.processing = True

col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Ask a question...", key="user_input", on_change=on_enter)
with col2:
    st.write("")  # Add vertical space to align
    button_label = "➤" if not st.session_state.processing else "⏳"
    send_clicked = st.button(button_label)

# Process Query if user clicks send
if send_clicked and user_input:
    st.session_state.temp_input = user_input
    st.session_state.processing = True
    st.rerun()

if st.session_state.processing:
    with st.spinner("Processing..."):
        try:
            user_input = st.session_state.get("temp_input", "")
            if query_mode == "LLM Only":
                response = llm.invoke(user_input)
                answer = response.content  # ✅ FIXED: Accessing AIMessage content
            else:
                graph_results = query_graph(user_input)

                if graph_results:
                    doc_id = graph_results[0]["doc_id"]
                    pdf_path = find_pdf_by_hash(doc_id)
                    pdf_text = extract_text_from_pdf(pdf_path) if pdf_path else ""

                    # Combine Graph Data + PDF Text + User Query
                    combined_input = f"Graph Data: {graph_results}\n\nPDF Content: {pdf_text}\n\nUser Query: {user_input}"
                    response = llm.invoke(combined_input)
                    answer = response.content  # ✅ FIXED: Accessing AIMessage content
                else:
                    answer = "No relevant data found in the graph."

            # Store in chat history
            st.session_state.chat_history.append((user_input, answer))

        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"
            st.session_state.chat_history.append((user_input, answer))

        finally:
            st.session_state.processing = False  # Reset processing state
            st.rerun()  # Rerun UI to update button