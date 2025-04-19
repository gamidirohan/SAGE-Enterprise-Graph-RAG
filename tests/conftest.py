import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add the parent directory to sys.path to import the backend module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after adding to sys.path
from backend import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)

@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver"""
    with patch('backend.get_neo4j_driver') as mock:
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock.return_value = mock_driver
        yield mock_driver

@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model"""
    with patch('backend.get_embedding_model') as mock:
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock.return_value = mock_model
        yield mock_model

@pytest.fixture
def mock_llm():
    """Create a mock LLM"""
    with patch('backend.ChatGroq') as mock:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "This is a mock response"
        mock.return_value = mock_llm
        yield mock_llm

@pytest.fixture
def sample_document_text():
    """Sample document text for testing"""
    return """
    From: Person123
    To: Person456, Person789
    Subject: Test Document
    
    This is a test document content.
    It has multiple lines.
    """

@pytest.fixture
def sample_document_data():
    """Sample structured document data for testing"""
    return {
        "doc_id": "test123",
        "sender": "Person123",
        "receivers": ["Person456", "Person789"],
        "subject": "Test Document",
        "content": "This is a test document content.\nIt has multiple lines."
    }

@pytest.fixture
def sample_graph_results():
    """Sample graph query results for testing"""
    return [
        "Chunk Summary: Person123 sent a budget report on March 15, Document: {'doc_id': 'abc123'}, Similarity: 0.85, Relationship: SENT, Related Node: {'id': 'Person456'}",
        "Chunk Summary: Person123 also sent meeting notes on April 2, Document: {'doc_id': 'def456'}, Similarity: 0.75, Relationship: SENT, Related Node: {'id': 'Person789'}"
    ]
