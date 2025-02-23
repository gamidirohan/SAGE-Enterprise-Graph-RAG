import streamlit as st
import os
import hashlib
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader

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

# User selects query mode
query_mode = st.radio("Select Query Mode:", ["LLM Only", "Graph + PDF + LLM"])

# Function to find the PDF file by matching hash
def find_pdf_by_hash(doc_id):
    for filename in os.listdir(FILES_DIR):
        file_path = os.path.join(FILES_DIR, filename)
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
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
    with driver.session() as session:
        query = f"""
        MATCH (d:Document)-[r]->(e)
        WHERE d.content CONTAINS '{user_input}'
        RETURN d.doc_id AS doc_id, d.title AS title, e.name AS entity, TYPE(r) AS relation
        LIMIT 5
        """
        results = session.run(query).data()
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