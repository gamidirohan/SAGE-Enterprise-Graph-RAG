"""Streamlit chat interface for SAGE.

This file renders the interactive graph-aware chat UI and connects user input
to the shared retrieval and response-generation services.
"""

import re

import streamlit as st

try:
    import app.services as services
    import app.utils as utils
except ImportError:
    import services
    import utils


st.set_page_config(page_title="SAGE: Graph-Based Chat", layout="wide")
st.title("SAGE: Graph Query Interface")

query_graph = services.query_graph


def generate_groq_response(query, documents):
    return services.generate_streamlit_response(query, documents)


def render_chat_history():
    st.subheader("Chat History")
    chat_history_box = st.empty()
    with chat_history_box.container():
        st.markdown(
            """
            <style>
            .chat-history {
                height: 600px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid grey;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for user_msg, bot_msg in st.session_state.chat_history:
            st.markdown(f"**User:** {user_msg}")
            st.markdown(f"**Assistant:** {bot_msg}")
            st.markdown("---")
        st.markdown("</div>", unsafe_allow_html=True)


def render_graph_debug():
    if not st.checkbox("Debug Graph Structure"):
        return

    with st.spinner("Analyzing graph structure..."):
        driver = utils.create_neo4j_driver()
        try:
            with utils.open_neo4j_session(driver, utils.NEO4J_DATABASE) as session:
                node_counts = session.run(
                    """
                    MATCH (n)
                    RETURN labels(n)[0] AS Label, count(*) AS Count
                    ORDER BY Count DESC
                    """
                ).data()

                rel_counts = session.run(
                    """
                    MATCH ()-[r]->()
                    RETURN type(r) AS RelationType, count(*) AS Count
                    ORDER BY Count DESC
                    """
                ).data()

                sample_docs = session.run(
                    """
                    MATCH (d:Document)
                    RETURN d.doc_id AS DocID, d.subject AS Subject, d.sender AS Sender
                    LIMIT 5
                    """
                ).data()

                connectivity = session.run(
                    """
                    MATCH (n)
                    WHERE NOT (n)--()
                    RETURN labels(n)[0] AS IsolatedNodeType, count(*) AS Count
                    ORDER BY Count DESC
                    """
                ).data()

                entity_doc_connections = session.run(
                    """
                    MATCH (p:Person)
                    RETURN p.id AS Person,
                        p.name AS Name,
                        p.role AS Role,
                        COUNT { (p)--() } AS ConnectionCount
                    ORDER BY ConnectionCount DESC
                    LIMIT 10
                    """
                ).data()

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
        except Exception as exc:
            st.error(f"Error analyzing graph structure: {exc}")
        finally:
            driver.close()


def on_enter():
    if st.session_state.user_input:
        st.session_state.temp_input = st.session_state.user_input
        st.session_state.user_input = ""
        st.session_state.processing = True


def process_pending_message():
    if not st.session_state.processing:
        return

    with st.spinner("Processing..."):
        try:
            user_input = st.session_state.get("temp_input", "")
            graph_results = query_graph(user_input)
            groq_response = generate_groq_response(user_input, graph_results)

            think_parts = re.findall(r"<think>(.*?)</think>", groq_response, re.DOTALL)
            answer = re.sub(r"<think>.*?</think>", "", groq_response, flags=re.DOTALL).strip()

            if think_parts:
                for think_part in think_parts:
                    st.markdown(
                        f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">{think_part}</div>',
                        unsafe_allow_html=True,
                    )

            st.session_state.chat_history.append((user_input, answer))
        except Exception as exc:
            st.session_state.chat_history.append((user_input, f"Error: {exc}"))
        finally:
            st.session_state.processing = False
            st.rerun()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

render_chat_history()
render_graph_debug()

col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Ask a question...", key="user_input", on_change=on_enter)
with col2:
    st.write("")
    send_clicked = st.button("Send" if not st.session_state.processing else "Processing")

if send_clicked and user_input:
    st.session_state.temp_input = user_input
    st.session_state.processing = True
    st.rerun()

process_pending_message()
