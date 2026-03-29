"""Shared utility helpers and configuration for SAGE.

This file loads environment settings, creates Neo4j/model helpers, and provides
common text-processing functions used throughout the app package.
"""

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Keep model download/loading output concise in app logs.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    # Transformers logging controls are optional at runtime.
    pass


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_Password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
MODEL_CACHE_DIR = ROOT_DIR / ".cache" / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def create_neo4j_driver(uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
    return GraphDatabase.driver(uri, auth=(user, password))


def open_neo4j_session(driver, database: str = NEO4J_DATABASE):
    if database:
        return driver.session(database=database)
    return driver.session()


@lru_cache(maxsize=1)
def get_cached_embedding_model(model_name: str = EMBEDDING_MODEL):
    # Prefer local cache first; if files are incomplete/missing, fall back to
    # one network fetch into cache and reuse it on subsequent runs.
    try:
        return SentenceTransformer(
            model_name,
            cache_folder=str(MODEL_CACHE_DIR),
            local_files_only=True,
        )
    except Exception:
        return SentenceTransformer(
            model_name,
            cache_folder=str(MODEL_CACHE_DIR),
            local_files_only=False,
        )


def generate_embedding(text: str, model_name: str = EMBEDDING_MODEL):
    model = get_cached_embedding_model(model_name=model_name)
    return model.encode(text).tolist()


def generate_doc_id(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def extract_text_from_pdf(file_path) -> str:
    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])


def chunk_document(text: str, max_chunk_words: int = 250, overlap_sentences: int = 2) -> List[str]:
    """Split text into sentence chunks using a word budget and sentence overlap.

    This implementation is NLTK-free to avoid runtime tokenizer downloads while
    keeping chunk continuity via configurable sentence overlap.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    previous_sentences: List[str] = []
    current_chunk: List[str] = []

    for sentence in sentences:
        current_chunk.append(sentence)
        token_count = len(" ".join(current_chunk).split())

        if token_count > max_chunk_words:
            chunk_text = " ".join(previous_sentences + current_chunk[:-1]).strip()
            if chunk_text:
                chunks.append(chunk_text)

            previous_sentences = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else current_chunk
            current_chunk = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else []

    if current_chunk:
        chunk_text = " ".join(previous_sentences + current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks
