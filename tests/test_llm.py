import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import the backend module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import generate_groq_response

# Mock data
MOCK_QUERY = "What documents were sent by Person123?"

MOCK_DOCUMENTS = [
    "Chunk Summary: Person123 sent a budget report on March 15, Document: {'doc_id': 'abc123'}, Similarity: 0.85, Relationship: SENT, Related Node: {'id': 'Person456'}",
    "Chunk Summary: Person123 also sent meeting notes on April 2, Document: {'doc_id': 'def456'}, Similarity: 0.75, Relationship: SENT, Related Node: {'id': 'Person789'}"
]

MOCK_LLM_RESPONSE = """
<think>
Let me analyze the provided context to answer the question about documents sent by Person123.
From the context, I can see that Person123 sent two documents:
1. A budget report on March 15
2. Meeting notes on April 2
</think>

Based on the information provided, Person123 sent two documents: a budget report on March 15 and meeting notes on April 2.
"""

EXPECTED_PARSED_RESPONSE = {
    "answer": "Based on the information provided, Person123 sent two documents: a budget report on March 15 and meeting notes on April 2.",
    "thinking": [
        "\nLet me analyze the provided context to answer the question about documents sent by Person123.\nFrom the context, I can see that Person123 sent two documents:\n1. A budget report on March 15\n2. Meeting notes on April 2\n"
    ]
}

@patch('backend.ChatGroq')
def test_generate_groq_response(mock_chatgroq):
    """Test generating a response using Groq LLM"""
    # Setup mock
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MOCK_LLM_RESPONSE
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MOCK_LLM_RESPONSE

    # Mock the chain creation
    mock_chatgroq.return_value = mock_llm

    # Patch the chain creation
    with patch('backend.ChatPromptTemplate.from_template') as mock_prompt:
        with patch('backend.StrOutputParser') as mock_parser:
            mock_prompt.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            # Mock the pipe operator
            mock_prompt.return_value.__or__.return_value = MagicMock()
            mock_prompt.return_value.__or__.return_value.__or__.return_value = mock_chain

            # Call the function
            response = generate_groq_response(MOCK_QUERY, MOCK_DOCUMENTS)

    # Verify the response
    assert "answer" in response
    assert "thinking" in response
    assert len(response["thinking"]) == len(EXPECTED_PARSED_RESPONSE["thinking"])
    assert response["answer"].strip() == EXPECTED_PARSED_RESPONSE["answer"].strip()

@patch('backend.ChatGroq')
def test_generate_groq_response_no_documents(mock_chatgroq):
    """Test generating a response with no documents"""
    # Call the function with empty documents
    response = generate_groq_response(MOCK_QUERY, [])

    # Verify the response
    assert response["answer"] == "No relevant information found."
    assert response["thinking"] == []

@patch('backend.ChatGroq')
def test_generate_groq_response_exception(mock_chatgroq):
    """Test error handling in response generation"""
    # Setup mock to raise an exception
    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("Test error")

    # Mock the chain creation
    mock_chatgroq.return_value = mock_llm

    # Patch the chain creation
    with patch('backend.ChatPromptTemplate.from_template') as mock_prompt:
        with patch('backend.StrOutputParser') as mock_parser:
            mock_prompt.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            # Mock the pipe operator
            mock_prompt.return_value.__or__.return_value = MagicMock()
            mock_prompt.return_value.__or__.return_value.__or__.return_value = mock_chain

            # Call the function - it should handle the exception gracefully
            response = generate_groq_response(MOCK_QUERY, MOCK_DOCUMENTS)

            # Verify the response contains an error message
            assert "error" in response["thinking"][0].lower()
            assert "encountered an error" in response["answer"].lower()

def test_context_extraction():
    """Test the extraction of context from document chunks"""
    # Create a function to simulate the context extraction logic
    def extract_context(documents):
        return "\n\n".join([item.split('Chunk Summary: ')[1].split(', Document: ')[0] for item in documents])

    # Test with our mock documents
    context = extract_context(MOCK_DOCUMENTS)

    # Verify the extracted context
    assert "Person123 sent a budget report on March 15" in context
    assert "Person123 also sent meeting notes on April 2" in context
