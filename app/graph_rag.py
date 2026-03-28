import streamlit as st
import numpy as np
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    import app.utils as utils
except ImportError:
    import utils

NEO4J_DATABASE = utils.NEO4J_DATABASE

# Streamlit UI
st.set_page_config(page_title="SAGE: Graph-Based Chat", layout="wide")
st.title("📄🔗 SAGE: Graph Query Interface (LangChain Groq DeepSeek)")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# Display chat history in a scrollable frame or box with fixed height
st.subheader("Chat History")
chat_history_box = st.empty()
with chat_history_box.container():
    chat_history_style = """
    <style>
    .chat-history {
        height: 600px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid grey;
        border-radius: 5px;
    }
    </style>
    """
    st.markdown(chat_history_style, unsafe_allow_html=True)
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for user_msg, bot_msg in st.session_state.chat_history:
        st.markdown(f"**User:** {user_msg}")
        st.markdown(f"**Assistant:** {bot_msg}")
        st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)

# Debug Graph Structure section
if st.checkbox("Debug Graph Structure"):
    with st.spinner("Analyzing graph structure..."):
        driver = utils.create_neo4j_driver()
        try:
            with utils.open_neo4j_session(driver, NEO4J_DATABASE) as session:
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
                    RETURN d.doc_id AS DocID, d.subject AS Subject, d.sender AS Sender
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
                    RETURN p.id AS Person,
                        p.name AS Name,
                        p.role AS Role,
                        COUNT { (p)--() } AS ConnectionCount
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

            st.subheader("Top Connected People")
            st.table(entity_doc_connections)

        except Exception as e:
            st.error(f"Error analyzing graph structure: {str(e)}")
        finally:
            driver.close()

# Function to query Neo4j for related data (No LLM)
def query_graph(user_input):
    driver = utils.create_neo4j_driver()
    model = utils.get_cached_embedding_model()
    query_embedding = model.encode(user_input)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    results = []
    with utils.open_neo4j_session(driver, NEO4J_DATABASE) as session:
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
            st.error(f"Vector search failed: {str(e)}")

    driver.close()
    return results

def generate_groq_response(query, documents):
    if not documents:
        return "No relevant information found."
    context_parts = []
    for item in documents:
        try:
            context_parts.append(item.split("Chunk Summary: ")[1].split(", Document: ")[0])
        except Exception:
            context_parts.append(str(item))
    context = "\n\n".join(context_parts)

    prompt_template = ChatPromptTemplate.from_template(
        "Answer the following question based on the provided context:\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"
    )
    llm = ChatGroq(model_name=utils.GROQ_MODEL, temperature=0.0, groq_api_key=utils.GROQ_API_KEY)
    chain = prompt_template | llm | StrOutputParser()

    try:
        return chain.invoke({"query": query, "context": context})
    except Exception as e:
        return f"Groq API error: {str(e)}"

def on_enter():
    if st.session_state.user_input:
        st.session_state.temp_input = st.session_state.user_input
        st.session_state.user_input = ""
        st.session_state.processing = True

# Chat input at the bottom
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
            groq_response = generate_groq_response(user_input, graph_results)

            # Process <think> tags
            think_parts = re.findall(r"<think>(.*?)</think>", groq_response, re.DOTALL)
            answer = re.sub(r"<think>.*?</think>", "", groq_response, flags=re.DOTALL).strip()

            if think_parts:
                for think_part in think_parts:
                    st.markdown(f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">{think_part}</div>', unsafe_allow_html=True)

            st.session_state.chat_history.append((user_input, answer))

        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"
            st.session_state.chat_history.append((user_input, answer))

        finally:
            st.session_state.processing = False
            st.rerun()
