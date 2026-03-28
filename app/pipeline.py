import streamlit as st
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

try:
    import app.utils as utils
except ImportError:
    import utils

try:
    from app.message_processor import store_in_neo4j
except ImportError:
    from message_processor import store_in_neo4j


def extract_structured_data(document_text: str, doc_id: str):
    if not utils.GROQ_API_KEY:
        # Lightweight fallback keeps pipeline usable without external LLM calls.
        return {
            "doc_id": doc_id,
            "sender": "Unknown",
            "receivers": [],
            "subject": "No Subject",
            "content": document_text,
        }

    llm = ChatGroq(
        model_name=utils.GROQ_MODEL,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
        groq_api_key=utils.GROQ_API_KEY,
    )

    schema = {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string"},
            "sender": {"type": "string"},
            "receivers": {"type": "array", "items": {"type": "string"}},
            "subject": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["doc_id", "sender", "receivers", "subject", "content"],
    }

    parser = JsonOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_template(
        """
        You are an advanced document intelligence system. Extract Sender, Receivers, Subject and content from the following document.

        Instructions:
        1. Extract the Sender ID.
        2. Extract the Receiver IDs as an array.
        3. Extract the Subject.
        4. Extract the main Content.

        Output format (JSON only):
        {
            "doc_id": "<hashed_document_id>",
            "sender": "<sender_id>",
            "receivers": ["<receiver_id1>", "<receiver_id2>"],
            "subject": "<subject>",
            "content": "<content>"
        }

        Input document:
        {input}
        """
    )
    chain = prompt | llm | parser
    structured_data = chain.invoke({"input": document_text})

    structured_data["doc_id"] = doc_id
    structured_data["sender"] = structured_data.get("sender") or "Unknown"
    structured_data["receivers"] = structured_data.get("receivers") or []
    structured_data["subject"] = structured_data.get("subject") or "No Subject"
    structured_data["content"] = structured_data.get("content") or document_text
    return structured_data


st.set_page_config(page_title="Document Processor", layout="wide")
st.title("Document Processor")

data_dir = utils.ROOT_DIR / "data"
docs_ui_dir = data_dir / "documents_ui"
docs_dir = data_dir / "documents"
files_dir = docs_ui_dir if docs_ui_dir.exists() else docs_dir

file_names = []
if files_dir.exists():
    file_names = [p.name for p in files_dir.iterdir() if p.suffix.lower() in (".txt", ".pdf")]

if not file_names:
    st.warning(f"No .txt or .pdf files found in directory: {files_dir}")
else:
    selected_file = st.selectbox("Select a file:", file_names)
    file_path = files_dir / selected_file

    if st.button("Process Document"):
        try:
            if selected_file.endswith(".pdf"):
                document_text = utils.extract_text_from_pdf(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    document_text = f.read()

            doc_id = utils.generate_doc_id(document_text)

            with st.spinner("Extracting structured data..."):
                structured_data = extract_structured_data(document_text, doc_id)
                st.json(structured_data)

            with st.spinner("Storing in Neo4j..."):
                if store_in_neo4j(structured_data):
                    st.success("Data successfully stored in Neo4j")
                else:
                    st.error("Failed to store data in Neo4j")
        except Exception as e:
            st.error(f"Error processing document: {e}")
