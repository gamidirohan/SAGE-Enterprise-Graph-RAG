"""Streamlit document-processing interface for SAGE.

This file lets a user pick local documents, extract structured metadata from
them, and send the processed result into the Neo4j-backed graph pipeline.
"""

from pathlib import Path

import streamlit as st

try:
    import app.services as services
    import app.utils as utils
    from app.message_processor import store_in_neo4j
except ImportError:
    import services
    import utils
    from message_processor import store_in_neo4j


extract_structured_data = services.extract_structured_data


def _get_documents_directory() -> Path:
    data_dir = utils.ROOT_DIR / "data"
    docs_ui_dir = data_dir / "documents_ui"
    docs_dir = data_dir / "documents"
    return docs_ui_dir if docs_ui_dir.exists() else docs_dir


def _list_supported_files(files_dir: Path):
    if not files_dir.exists():
        return []
    return [path.name for path in files_dir.iterdir() if path.suffix.lower() in (".txt", ".pdf")]


st.set_page_config(page_title="Document Processor", layout="wide")
st.title("Document Processor")

files_dir = _get_documents_directory()
file_names = _list_supported_files(files_dir)

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
                document_text = file_path.read_text(encoding="utf-8")

            doc_id = utils.generate_doc_id(document_text)

            with st.spinner("Extracting structured data..."):
                structured_data = extract_structured_data(document_text, doc_id)
                st.json(structured_data)

            with st.spinner("Storing in Neo4j..."):
                if store_in_neo4j(structured_data):
                    st.success("Data successfully stored in Neo4j")
                else:
                    st.error("Failed to store data in Neo4j")
        except Exception as exc:
            st.error(f"Error processing document: {exc}")
