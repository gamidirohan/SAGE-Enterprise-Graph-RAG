import streamlit as st
import os
import hashlib
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
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
    return SentenceTransformer('all-mpnet-base-v2')

# Streamlit UI
st.set_page_config(page_title="SAGE: Graph-Based Chat", layout="wide")
st.title("📄🔗 SAGE: Graph Query Interface (No LLM)")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# Display existing chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)

# Debug Graph Structure section
if st.checkbox("Debug Graph Structure"):
    with st.spinner("Analyzing graph structure..."):
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
                    RETURN d.doc_id AS DocID, d.title AS Title, d.doc_type AS Type
                    LIMIT 5
                """).data()

                connectivity = session.run("""
                    MATCH (n)
                    WHERE NOT (n)--()
                    RETURN labels(n)[0] AS IsolatedNodeType, count(*) AS Count
                    ORDER BY Count DESC
                """).data()

                entity_doc_connections = session.run("""
                    MATCH (e:Entity)
                    RETURN e.name AS Entity, 
                        e.type AS Type,
                        COUNT { (e)--() } AS ConnectionCount
                    ORDER BY ConnectionCount DESC
                    LIMIT 10
                """).data()

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

        except Exception as e:
            st.error(f"Error analyzing graph structure: {str(e)}")
        finally:
            driver.close()

# Function to query Neo4j for related data (No LLM)
def query_graph(user_input):
    driver = get_neo4j_driver()
    model = get_embedding_model()
    query_embedding = model.encode(user_input)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    results = []
    with driver.session() as session:
        try:
            vector_query = """
                MATCH (d:Document)
                WHERE d.embedding IS NOT NULL
                WITH d, d.embedding AS doc_embedding, $query_embedding AS query_embedding
                WITH d, gds.similarity.cosine(doc_embedding, query_embedding) AS similarity
                ORDER BY similarity DESC
                LIMIT 5
                RETURN d.doc_id AS doc_id, d.content AS content, similarity
            """
            vector_results = session.run(vector_query, query_embedding=query_embedding.tolist()).data()
            if vector_results:
                results = vector_results

        except Exception as e:
            st.error(f"Vector search failed: {str(e)}")

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
    st.write("")
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
            graph_results = query_graph(user_input)

            if graph_results:
                st.write("Results:")
                st.write(graph_results)
                answer = str(graph_results)
            else:
                answer = "No relevant data found in the graph."

            st.session_state.chat_history.append((user_input, answer))

        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"
            st.session_state.chat_history.append((user_input, answer))

        finally:
            st.session_state.processing = False
            st.rerun()