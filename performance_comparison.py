"""
Enhanced Performance Comparison: SAGE (Graph-based RAG) vs Traditional RAG

This script compares the performance of SAGE's graph-based RAG approach with a traditional
vector-based RAG implementation using various metrics:
- ROUGE metrics
- BERT Score (if available)
- LLM-based evaluation (using various models to judge answer quality)

The script also compares different:
- LLM models (deepseek, gemma2, llama3, etc.)
- Embedding models (sentence-transformers, etc.)

Usage:
    python performance_comparison.py [--queries QUERIES_FILE] [--output OUTPUT_FILE] [--models MODEL1,MODEL2,...]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional, Union
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("BERT Score not available. Will use alternative metrics.")
from rouge import Rouge
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = "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" or os.getenv("GROQ_API_KEY")

# Available models
AVAILABLE_LLM_MODELS = {
    "gemma2-9b-it": {
        "provider": "groq",
        "name": "gemma2-9b-it",
        "description": "Gemma 2 9B Instruct"
    },
    "llama-guard-3-8b": {
        "provider": "groq",
        "name": "llama-guard-3-8b",
        "description": "Llama Guard 3 8B"
    },
    "mistral-saba-24b": {
        "provider": "groq",
        "name": "mistral-saba-24b",
        "description": "Mistral Saba 24B"
    },
    "llama3-8b-8192": {
        "provider": "groq",
        "name": "llama3-8b-8192",
        "description": "Llama 3 8B"
    },
    "compound-beta-mini": {
        "provider": "groq",
        "name": "compound-beta-mini",
        "description": "Compound Beta Mini"
    },
    "deepseek-r1-distill-llama-70b": {
        "provider": "groq",
        "name": "deepseek-r1-distill-llama-70b",
        "description": "DeepSeek R1 Distill Llama 70B"
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "name": "llama-3.3-70b-versatile",
        "description": "Llama 3.3 70B Versatile"
    },
    "llama3-70b-8192": {
        "provider": "groq",
        "name": "llama3-70b-8192",
        "description": "Llama 3 70B"
    },
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "name": "llama-3.1-8b-instant",
        "description": "Llama 3.1 8B Instant"
    }
}

# Available embedding models
AVAILABLE_EMBEDDING_MODELS = {
    "all-mpnet-base-v2": {
        "provider": "sentence-transformers",
        "name": "all-mpnet-base-v2",
        "description": "MPNet Base v2 (Default)"
    },
    "all-MiniLM-L6-v2": {
        "provider": "sentence-transformers",
        "name": "all-MiniLM-L6-v2",
        "description": "MiniLM L6 v2 (Faster, smaller)"
    },
    "multi-qa-mpnet-base-dot-v1": {
        "provider": "sentence-transformers",
        "name": "multi-qa-mpnet-base-dot-v1",
        "description": "MPNet Base optimized for question-answering"
    },
    "all-distilroberta-v1": {
        "provider": "sentence-transformers",
        "name": "all-distilroberta-v1",
        "description": "DistilRoBERTa v1 (Good balance of speed and quality)"
    },
    "paraphrase-multilingual-mpnet-base-v2": {
        "provider": "sentence-transformers",
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "description": "Multilingual MPNet Base v2 (Good for multiple languages)"
    }
}

# Default test queries if none provided
DEFAULT_TEST_QUERIES = [
    "What are the main responsibilities of Alice Johnson?",
    "Summarize the key points from the latest meeting notes",
    "What projects is Bob Smith working on?",
    "What are the deadlines for the current sprint?",
    "Who is responsible for the data analysis task?",
]

# Try to load QA pairs if available
QA_PAIRS_FILE = "qa_pairs.json"
try:
    if os.path.exists(QA_PAIRS_FILE):
        with open(QA_PAIRS_FILE, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        DEFAULT_TEST_QUERIES = [pair["question"] for pair in qa_pairs[:10]]
        logger.info(f"Loaded {len(DEFAULT_TEST_QUERIES)} test queries from {QA_PAIRS_FILE}")
except Exception as e:
    logger.warning(f"Could not load QA pairs from {QA_PAIRS_FILE}: {str(e)}")

class RAGSystem:
    """Base class for RAG systems"""

    def __init__(self, name: str, llm_model: str = "deepseek-r1-distill-llama-70b", embedding_model: str = "all-mpnet-base-v2"):
        self.name = name
        self.llm_model_name = llm_model
        self.embedding_model_name = embedding_model

        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model(embedding_model)

        # Initialize LLM
        self.llm = self._initialize_llm(llm_model)

    def _initialize_embedding_model(self, model_name: str):
        """Initialize the embedding model"""
        if model_name not in AVAILABLE_EMBEDDING_MODELS:
            logger.warning(f"Embedding model {model_name} not found. Using default model.")
            model_name = "all-mpnet-base-v2"

        logger.info(f"Initializing embedding model: {model_name}")
        return SentenceTransformer(f"sentence-transformers/{model_name}")

    def _initialize_llm(self, model_name: str):
        """Initialize the LLM"""
        if model_name not in AVAILABLE_LLM_MODELS:
            logger.warning(f"LLM model {model_name} not found. Using default model.")
            model_name = "deepseek-r1-distill-llama-70b"

        logger.info(f"Initializing LLM: {model_name}")
        return ChatGroq(
            model_name=model_name,
            temperature=0.0,
            groq_api_key=GROQ_API_KEY
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Query the system with a question and return the answer with metadata"""
        raise NotImplementedError("Subclasses must implement this method")


class SAGEGraphRAG(RAGSystem):
    """SAGE Graph-based RAG implementation"""

    def __init__(self, llm_model: str = "deepseek-r1-distill-llama-70b", embedding_model: str = "all-mpnet-base-v2"):
        super().__init__(f"SAGE Graph RAG ({llm_model}, {embedding_model})", llm_model, embedding_model)
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Query the SAGE system using graph-based RAG"""
        start_time = time.time()

        # Get relevant documents from graph
        graph_results = self._query_graph(question)

        # Generate response using retrieved context
        response = self._generate_response(question, graph_results)

        end_time = time.time()

        return {
            "question": question,
            "answer": response["answer"],
            "thinking": response.get("thinking", []),
            "context": graph_results,
            "latency": end_time - start_time
        }

    def _query_graph(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Enhanced query for Neo4j graph that leverages relationships more effectively
        and implements multi-hop traversal to find related information
        """
        try:
            query_embedding = self.embedding_model.encode(user_input)
            query_embedding = np.array(query_embedding, dtype=np.float32)

            results = []
            with self.driver.session() as session:
                # Step 1: Find initial chunks based on vector similarity
                initial_query = """
                    MATCH (c:Chunk)-[:PART_OF]->(d:Document)
                    WHERE c.embedding IS NOT NULL
                    WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
                    WITH c, d, gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity
                    WHERE similarity > 0.6
                    ORDER BY similarity DESC
                    LIMIT 3
                    RETURN c.chunk_id AS chunk_id, c.summary AS chunk_summary, c.content AS chunk_content,
                           d.doc_id AS doc_id, d.subject AS doc_subject, d.sender AS doc_sender,
                           similarity
                """
                initial_results = session.run(initial_query, query_embedding=query_embedding.tolist()).data()

                if not initial_results:
                    # Fallback to a lower similarity threshold if no results
                    initial_query = """
                        MATCH (c:Chunk)-[:PART_OF]->(d:Document)
                        WHERE c.embedding IS NOT NULL
                        WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
                        WITH c, d, gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity
                        ORDER BY similarity DESC
                        LIMIT 3
                        RETURN c.chunk_id AS chunk_id, c.summary AS chunk_summary, c.content AS chunk_content,
                               d.doc_id AS doc_id, d.subject AS doc_subject, d.sender AS doc_sender,
                               similarity
                    """
                    initial_results = session.run(initial_query, query_embedding=query_embedding.tolist()).data()

                # Step 2: For each initial chunk, find related nodes up to 2 hops away
                for item in initial_results:
                    chunk_id = item.get('chunk_id')

                    # Add the initial chunk to results
                    chunk_info = {
                        "type": "chunk",
                        "chunk_id": chunk_id,
                        "summary": item.get('chunk_summary', ''),
                        "content": item.get('chunk_content', ''),
                        "doc_id": item.get('doc_id', ''),
                        "doc_subject": item.get('doc_subject', ''),
                        "doc_sender": item.get('doc_sender', ''),
                        "similarity": item.get('similarity', 0),
                        "relationships": []
                    }

                    # Find related nodes and their relationships
                    relationship_query = """
                        MATCH (c:Chunk {chunk_id: $chunk_id})-[r1]-(n1)
                        OPTIONAL MATCH (n1)-[r2]-(n2)
                        WHERE n2 <> c AND NOT (n2:Chunk) OR (n2:Chunk AND n2.chunk_id <> $chunk_id)
                        RETURN
                            c.chunk_id AS source_id,
                            labels(n1) AS n1_labels,
                            properties(n1) AS n1_props,
                            type(r1) AS r1_type,
                            CASE WHEN n2 IS NOT NULL THEN labels(n2) ELSE [] END AS n2_labels,
                            CASE WHEN n2 IS NOT NULL THEN properties(n2) ELSE {} END AS n2_props,
                            CASE WHEN r2 IS NOT NULL THEN type(r2) ELSE '' END AS r2_type
                    """
                    relationship_results = session.run(relationship_query, chunk_id=chunk_id).data()

                    # Process relationship results
                    for rel in relationship_results:
                        n1_type = rel.get('n1_labels', ['Unknown'])[0]
                        n1_props = rel.get('n1_props', {})
                        r1_type = rel.get('r1_type', 'unknown')

                        relationship_info = {
                            "node_type": n1_type,
                            "node_properties": self._clean_properties(n1_props),
                            "relationship_type": r1_type
                        }

                        # Add second-hop information if available
                        if rel.get('n2_labels') and rel.get('n2_props'):
                            n2_type = rel.get('n2_labels', ['Unknown'])[0]
                            n2_props = rel.get('n2_props', {})
                            r2_type = rel.get('r2_type', 'unknown')

                            relationship_info["second_hop"] = {
                                "node_type": n2_type,
                                "node_properties": self._clean_properties(n2_props),
                                "relationship_type": r2_type
                            }

                        chunk_info["relationships"].append(relationship_info)

                    results.append(chunk_info)

                # Step 3: Find any Person nodes that might be relevant to the query
                person_query = """
                    MATCH (p:Person)
                    WHERE p.id IS NOT NULL
                    WITH p, $query_text AS query_text
                    WHERE query_text CONTAINS p.id
                    RETURN p.id AS person_id,
                           [(p)-[r]-(n) | {type: type(r), node_labels: labels(n), node_props: properties(n)}] AS connections
                    LIMIT 3
                """
                person_results = session.run(person_query, query_text=user_input).data()

                # Add person nodes to results if found
                for person in person_results:
                    person_info = {
                        "type": "person",
                        "person_id": person.get('person_id', ''),
                        "connections": []
                    }

                    for conn in person.get('connections', []):
                        connection_info = {
                            "relationship_type": conn.get('type', 'unknown'),
                            "node_type": conn.get('node_labels', ['Unknown'])[0],
                            "node_properties": self._clean_properties(conn.get('node_props', {}))
                        }
                        person_info["connections"].append(connection_info)

                    results.append(person_info)

                # Step 4: Find semantically similar entities based on the query
                entity_query = """
                    MATCH (e)
                    WHERE (e:Person OR e:Organization OR e:Project OR e:Topic) AND e.name IS NOT NULL
                    WITH e, toLower(e.name) AS entity_name, toLower($query_text) AS query_text
                    WHERE query_text CONTAINS entity_name OR entity_name CONTAINS query_text
                    RETURN labels(e) AS entity_type, properties(e) AS entity_props,
                           [(e)-[r]-(n) | {type: type(r), node_labels: labels(n), node_props: properties(n)}] AS connections
                    LIMIT 5
                """
                entity_results = session.run(entity_query, query_text=user_input).data()

                # Add entity nodes to results if found
                for entity in entity_results:
                    entity_type = entity.get('entity_type', ['Unknown'])[0]
                    entity_props = entity.get('entity_props', {})

                    entity_info = {
                        "type": "entity",
                        "entity_type": entity_type,
                        "entity_properties": self._clean_properties(entity_props),
                        "connections": []
                    }

                    for conn in entity.get('connections', []):
                        connection_info = {
                            "relationship_type": conn.get('type', 'unknown'),
                            "node_type": conn.get('node_labels', ['Unknown'])[0],
                            "node_properties": self._clean_properties(conn.get('node_props', {}))
                        }
                        entity_info["connections"].append(connection_info)

                    results.append(entity_info)

                if not results:
                    # Create a placeholder result if nothing found
                    results = [{
                        "type": "no_results",
                        "message": "No relevant data found in the graph."
                    }]

            return results
        except Exception as e:
            logger.error(f"Graph query failed: {str(e)}")
            return [{
                "type": "error",
                "message": f"Error: {str(e)}"
            }]

    def _clean_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Remove embedding and other large properties from node properties"""
        cleaned = {}
        for k, v in props.items():
            if k != 'embedding' and not isinstance(v, list) and len(str(v)) < 1000:
                cleaned[k] = v
        return cleaned

    def _generate_response(self, query: str, graph_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate response using Groq with enhanced graph-aware prompting
        and chain-of-thought reasoning
        """
        if not graph_results:
            return {
                "answer": "I don't have enough information to answer that question.",
                "thinking": []
            }

        # Check for error or no results
        if len(graph_results) == 1 and (
            graph_results[0].get("type") == "error" or
            graph_results[0].get("type") == "no_results"
        ):
            return {
                "answer": "I don't have enough information to answer that question.",
                "thinking": []
            }

        try:
            # Format the context in a structured way that highlights relationships
            context_parts = []

            # Process chunk results - optimized for token efficiency
            chunk_contexts = []
            for item in graph_results:
                if item.get("type") == "chunk":
                    # Only include essential information
                    chunk_context = f"CONTENT: {item.get('content', '')}\n"

                    # Add only the most important relationships
                    if item.get("relationships"):
                        important_rels = []
                        for rel in item.get("relationships", [])[:2]:  # Limit to 2 most important relationships
                            rel_type = rel.get("relationship_type", "")
                            node_type = rel.get("node_type", "")

                            # Only include key properties
                            node_props = rel.get("node_properties", {})
                            key_props = {}
                            for k, v in node_props.items():
                                if k in ['name', 'id', 'title', 'role']:
                                    key_props[k] = v

                            # Format node properties concisely
                            if key_props:
                                props_str = ", ".join([f"{k}: {v}" for k, v in key_props.items()])
                                important_rels.append(f"- {rel_type} to {node_type} ({props_str})")
                            else:
                                important_rels.append(f"- {rel_type} to {node_type}")

                        if important_rels:
                            chunk_context += "RELATED TO:\n" + "\n".join(important_rels)

                    chunk_contexts.append(chunk_context)

            if chunk_contexts:
                context_parts.append("## RELEVANT INFORMATION\n" + "\n---\n".join(chunk_contexts))

            # Process person results - optimized for token efficiency
            person_contexts = []
            for item in graph_results:
                if item.get("type") == "person":
                    person_id = item.get("person_id", "")
                    person_context = f"PERSON: {person_id}\n"

                    # Add only key connections
                    if item.get("connections"):
                        key_connections = []
                        for conn in item.get("connections", [])[:2]:  # Limit to 2 most important connections
                            rel_type = conn.get("relationship_type", "")
                            node_type = conn.get("node_type", "")

                            # Only include essential properties
                            node_props = conn.get("node_properties", {})
                            key_props = {}
                            for k, v in node_props.items():
                                if k in ['name', 'id', 'title', 'role', 'subject']:
                                    key_props[k] = v

                            if key_props:
                                props_str = ", ".join([f"{k}: {v}" for k, v in key_props.items()])
                                key_connections.append(f"- {rel_type} to {node_type} ({props_str})")
                            else:
                                key_connections.append(f"- {rel_type} to {node_type}")

                        if key_connections:
                            person_context += "CONNECTED TO:\n" + "\n".join(key_connections)

                    person_contexts.append(person_context)

            if person_contexts:
                context_parts.append("## PEOPLE\n" + "\n---\n".join(person_contexts))

            # Process entity results - optimized for token efficiency
            entity_contexts = []
            for item in graph_results:
                if item.get("type") == "entity":
                    entity_type = item.get("entity_type", "")

                    # Only include essential properties
                    entity_props = item.get("entity_properties", {})
                    key_props = {}
                    for k, v in entity_props.items():
                        if k in ['name', 'id', 'title', 'description']:
                            key_props[k] = v

                    # Format entity properties concisely
                    if key_props:
                        props_str = ", ".join([f"{k}: {v}" for k, v in key_props.items()])
                        entity_context = f"{entity_type}: {props_str}\n"
                    else:
                        entity_context = f"{entity_type}\n"

                    # Add only key connections
                    if item.get("connections"):
                        key_connections = []
                        for conn in item.get("connections", [])[:1]:  # Limit to 1 most important connection
                            rel_type = conn.get("relationship_type", "")
                            node_type = conn.get("node_type", "")
                            key_connections.append(f"- {rel_type} to {node_type}")

                        if key_connections:
                            entity_context += "CONNECTED TO:\n" + "\n".join(key_connections)

                    entity_contexts.append(entity_context)

            if entity_contexts:
                context_parts.append("## ENTITIES\n" + "\n---\n".join(entity_contexts))

            # Join all context parts
            context = "\n\n".join(context_parts)

            # Create optimized prompt template with graph-aware reasoning
            prompt_template = ChatPromptTemplate.from_template(
                """
                You are a graph-aware assistant that answers questions based on the provided context from a knowledge graph.

                <context>
                {context}
                </context>

                User question: {query}

                Use the graph relationships in the context to find connections between entities that help answer the question.
                Pay attention to how different pieces of information relate to each other through the graph structure.

                If the context doesn't contain the information needed to answer the question,
                say that you don't have enough information rather than making up an answer.

                Answer:
                """
            )

            # Simplified thinking prompt
            thinking_prompt = f"""
            Analyzing question: "{query}"
            Looking for relevant entities and their relationships in the knowledge graph.
            """

            # Create chain for final response
            chain = prompt_template | self.llm | StrOutputParser()

            # Invoke chain
            response = chain.invoke({
                "query": query,
                "context": context,
                "thinking": thinking_prompt
            })

            return {
                "answer": response,
                "thinking": [thinking_prompt],
                "structured_context": context
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "thinking": []
            }

    def close(self):
        """Close the Neo4j driver"""
        if self.driver:
            self.driver.close()


class TraditionalRAG(RAGSystem):
    """Traditional vector-based RAG implementation"""

    def __init__(self, llm_model: str = "deepseek-r1-distill-llama-70b", embedding_model: str = "all-mpnet-base-v2"):
        super().__init__(f"Traditional RAG ({llm_model}, {embedding_model})", llm_model, embedding_model)
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.documents = self._load_documents()

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from Neo4j but without using graph relationships"""
        documents = []

        try:
            with self.driver.session() as session:
                # Get all chunks with their embeddings
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.embedding IS NOT NULL
                    RETURN c.chunk_id AS id, c.content AS content, c.summary AS summary, c.embedding AS embedding
                """).data()

                for doc in result:
                    documents.append({
                        "id": doc["id"],
                        "content": doc["content"],
                        "summary": doc["summary"],
                        "embedding": doc["embedding"]
                    })

            logger.info(f"Loaded {len(documents)} documents for traditional RAG")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return []

    def query(self, question: str) -> Dict[str, Any]:
        """Query using traditional RAG approach"""
        start_time = time.time()

        # Get relevant documents using vector similarity
        relevant_docs = self._get_relevant_documents(question)

        # Generate response using retrieved context
        response = self._generate_response(question, relevant_docs)

        end_time = time.time()

        return {
            "question": question,
            "answer": response["answer"],
            "thinking": response.get("thinking", []),
            "context": [doc["summary"] for doc in relevant_docs],
            "latency": end_time - start_time
        }

    def _get_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant documents using vector similarity with basic retrieval"""
        if not self.documents:
            return []

        query_embedding = self.embedding_model.encode(query)

        # Calculate similarity scores
        similarities = []
        for doc in self.documents:
            if doc["embedding"]:
                similarity = np.dot(query_embedding, doc["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc["embedding"])
                )
                similarities.append((doc, similarity))

        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top documents
        top_docs = [doc for doc, _ in similarities[:top_k]]

        # Add basic document metadata from Neo4j
        with self.driver.session() as session:
            for doc in top_docs:
                chunk_id = doc.get("id")
                if chunk_id:
                    # Get document metadata (minimal)
                    metadata_query = """
                        MATCH (c:Chunk {chunk_id: $chunk_id})-[:PART_OF]->(d:Document)
                        RETURN d.subject AS subject
                    """
                    metadata_result = session.run(metadata_query, chunk_id=chunk_id).data()

                    if metadata_result:
                        doc["subject"] = metadata_result[0].get("subject", "")

        return top_docs

    def _generate_response(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using Groq with basic context formatting"""
        if not documents:
            return {
                "answer": "I don't have enough information to answer that question.",
                "thinking": []
            }

        try:
            # Format context simply
            context_parts = []
            for doc in documents:
                context_part = f"CONTENT: {doc.get('content', '')}\n"
                context_part += f"SUMMARY: {doc.get('summary', '')}\n"
                context_parts.append(context_part)

            context = "\n---\n".join(context_parts)

            # Create simple prompt template
            prompt_template = ChatPromptTemplate.from_template(
                """
                You are a helpful assistant that answers questions based on the provided context.

                <context>
                {context}
                </context>

                User question: {query}

                If the context doesn't contain the information needed to answer the question,
                say that you don't have enough information rather than making up an answer.

                Answer:
                """
            )

            # Create chain
            chain = prompt_template | self.llm | StrOutputParser()

            # Invoke chain
            response = chain.invoke({"query": query, "context": context})

            return {
                "answer": response,
                "thinking": []
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "thinking": []
            }

    def close(self):
        """Close the Neo4j driver"""
        if self.driver:
            self.driver.close()


class PerformanceEvaluator:
    """Evaluates performance of RAG systems"""

    def __init__(self, llm_model: str = "deepseek-r1-distill-llama-70b"):
        self.rouge = Rouge()
        self.llm_model_name = llm_model
        self.llm = self._initialize_llm(llm_model)

    def _initialize_llm(self, model_name: str):
        """Initialize the LLM"""
        if model_name not in AVAILABLE_LLM_MODELS:
            logger.warning(f"LLM model {model_name} not found for evaluation. Using default model.")
            model_name = "deepseek-r1-distill-llama-70b"

        logger.info(f"Initializing evaluation LLM: {model_name}")
        return ChatGroq(
            model_name=model_name,
            temperature=0.0,
            groq_api_key=GROQ_API_KEY
        )

    def evaluate_similarity(self, answer1: str, answer2: str, reference: str = None) -> Dict[str, float]:
        """
        Calculate similarity between two answers using a simple approach.
        If reference is provided, calculate similarity against the reference.
        """
        # We'll use ROUGE-L as a substitute for BERT Score
        if reference:
            # Calculate against reference
            scores1 = self.rouge.get_scores(answer1, reference)[0]
            scores2 = self.rouge.get_scores(answer2, reference)[0]

            return {
                "system1_precision": scores1["rouge-l"]["p"],
                "system1_recall": scores1["rouge-l"]["r"],
                "system1_f1": scores1["rouge-l"]["f"],
                "system2_precision": scores2["rouge-l"]["p"],
                "system2_recall": scores2["rouge-l"]["r"],
                "system2_f1": scores2["rouge-l"]["f"]
            }
        else:
            # Compare the two answers directly
            scores = self.rouge.get_scores(answer1, answer2)[0]

            return {
                "precision": scores["rouge-l"]["p"],
                "recall": scores["rouge-l"]["r"],
                "f1": scores["rouge-l"]["f"]
            }

    def evaluate_rouge(self, answer1: str, answer2: str, reference: str = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE metrics between two answers.
        If reference is provided, calculate scores against the reference.
        """
        if reference:
            # Calculate against reference
            scores1 = self.rouge.get_scores(answer1, reference)[0]
            scores2 = self.rouge.get_scores(answer2, reference)[0]

            return {
                "system1": {
                    "rouge-1": scores1["rouge-1"],
                    "rouge-2": scores1["rouge-2"],
                    "rouge-l": scores1["rouge-l"]
                },
                "system2": {
                    "rouge-1": scores2["rouge-1"],
                    "rouge-2": scores2["rouge-2"],
                    "rouge-l": scores2["rouge-l"]
                }
            }
        else:
            # Compare the two answers directly
            scores = self.rouge.get_scores(answer1, answer2)[0]

            return {
                "rouge-1": scores["rouge-1"],
                "rouge-2": scores["rouge-2"],
                "rouge-l": scores["rouge-l"]
            }

    def evaluate_with_llm(self, question: str, answer1: str, answer2: str) -> Dict[str, Any]:
        """Use Groq API to evaluate which answer is better"""
        try:
            prompt_template = ChatPromptTemplate.from_template(
                """
                You are an expert evaluator of question-answering systems.
                You will be given a question and two different answers from two different systems.

                Question: {question}

                System 1 Answer (SAGE Graph RAG): {answer1}

                System 2 Answer (Traditional RAG): {answer2}

                Please evaluate both answers based on:
                1. Relevance to the question
                2. Accuracy of information
                3. Completeness of the answer
                4. Clarity and coherence

                Return your evaluation as a JSON object with the following structure:
                {{
                    "system1_score": <score from 1-10>,
                    "system2_score": <score from 1-10>,
                    "better_system": <"system1", "system2", or "tie">,
                    "reasoning": <your explanation for the scores>
                }}
                """
            )

            # Set up JSON output parser
            parser = JsonOutputParser()

            # Create chain
            chain = prompt_template | self.llm | parser

            # Invoke chain
            result = chain.invoke({
                "question": question,
                "answer1": answer1,
                "answer2": answer2
            })

            return result
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            return {
                "system1_score": 0,
                "system2_score": 0,
                "better_system": "error",
                "reasoning": f"Error: {str(e)}"
            }


def run_comparison(
    queries: List[str],
    output_file: str = None,
    llm_models: List[str] = None,
    embedding_models: List[str] = None,
    evaluation_model: str = "deepseek-r1-distill-llama-70b"
):
    """
    Run comparison between SAGE and Traditional RAG with different models

    Args:
        queries: List of queries to test
        output_file: Path to save results
        llm_models: List of LLM models to test
        embedding_models: List of embedding models to test
        evaluation_model: Model to use for evaluation
    """
    # Use default models if none specified
    if llm_models is None:
        llm_models = ["deepseek-r1-distill-llama-70b"]

    if embedding_models is None:
        embedding_models = ["all-mpnet-base-v2"]

    # Validate models
    valid_llm_models = [model for model in llm_models if model in AVAILABLE_LLM_MODELS]
    if not valid_llm_models:
        logger.warning("No valid LLM models specified. Using default model.")
        valid_llm_models = ["deepseek-r1-distill-llama-70b"]

    valid_embedding_models = [model for model in embedding_models if model in AVAILABLE_EMBEDDING_MODELS]
    if not valid_embedding_models:
        logger.warning("No valid embedding models specified. Using default model.")
        valid_embedding_models = ["all-mpnet-base-v2"]

    # Initialize evaluator
    evaluator = PerformanceEvaluator(evaluation_model)

    all_results = []

    # Run comparison for each combination of models
    for llm_model in valid_llm_models:
        for embedding_model in valid_embedding_models:
            logger.info(f"Testing with LLM: {llm_model}, Embedding: {embedding_model}")

            # Initialize systems
            sage_rag = SAGEGraphRAG(llm_model, embedding_model)
            traditional_rag = TraditionalRAG(llm_model, embedding_model)

            model_results = []

            try:
                for i, query in enumerate(queries):
                    logger.info(f"Processing query {i+1}/{len(queries)}: {query}")

                    # Get responses from both systems
                    sage_response = sage_rag.query(query)
                    traditional_response = traditional_rag.query(query)

                    # Evaluate with metrics
                    similarity_scores = evaluator.evaluate_similarity(
                        sage_response["answer"],
                        traditional_response["answer"]
                    )

                    rouge_scores = evaluator.evaluate_rouge(
                        sage_response["answer"],
                        traditional_response["answer"]
                    )

                    llm_evaluation = evaluator.evaluate_with_llm(
                        query,
                        sage_response["answer"],
                        traditional_response["answer"]
                    )

                    # Compile results
                    result = {
                        "query": query,
                        "llm_model": llm_model,
                        "embedding_model": embedding_model,
                        "sage_response": sage_response,
                        "traditional_response": traditional_response,
                        "similarity_scores": similarity_scores,
                        "rouge_scores": rouge_scores,
                        "llm_evaluation": llm_evaluation
                    }

                    model_results.append(result)

                    # Print summary for this query
                    print(f"\n--- Query {i+1} with {llm_model}, {embedding_model}: {query} ---")
                    print(f"SAGE Answer: {sage_response['answer'][:100]}...")
                    print(f"Traditional RAG Answer: {traditional_response['answer'][:100]}...")
                    print(f"Similarity F1: {similarity_scores.get('f1', 0):.4f}")
                    print(f"LLM Evaluation: SAGE={llm_evaluation.get('system1_score', 0)}, Traditional={llm_evaluation.get('system2_score', 0)}")
                    print(f"Better System: {llm_evaluation.get('better_system', 'unknown')}")
                    print(f"Latency: SAGE={sage_response['latency']:.2f}s, Traditional={traditional_response['latency']:.2f}s")

                # Add model results to all results
                all_results.extend(model_results)

                # Generate summary statistics for this model combination
                print(f"\n=== Summary for {llm_model}, {embedding_model} ===")
                generate_summary(model_results, f"{llm_model}_{embedding_model}")

            finally:
                # Clean up
                sage_rag.close()
                traditional_rag.close()

    # Save all results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"All results saved to {output_file}")

    # Generate overall summary statistics
    generate_summary(all_results, "overall")


def generate_summary(results: List[Dict[str, Any]], prefix: str = ""):
    """
    Generate and display summary statistics

    Args:
        results: List of results to summarize
        prefix: Prefix for output files (e.g., model name)
    """
    if not results:
        logger.warning("No results to summarize")
        return

    # Extract metrics
    similarity_f1_scores = [r["similarity_scores"].get("f1", 0) for r in results]
    sage_scores = [r["llm_evaluation"].get("system1_score", 0) for r in results]
    trad_scores = [r["llm_evaluation"].get("system2_score", 0) for r in results]

    better_system_counts = {
        "system1": sum(1 for r in results if r["llm_evaluation"].get("better_system") == "system1"),
        "system2": sum(1 for r in results if r["llm_evaluation"].get("better_system") == "system2"),
        "tie": sum(1 for r in results if r["llm_evaluation"].get("better_system") == "tie"),
        "error": sum(1 for r in results if r["llm_evaluation"].get("better_system") == "error")
    }

    sage_latencies = [r["sage_response"]["latency"] for r in results]
    trad_latencies = [r["traditional_response"]["latency"] for r in results]

    # Get model information if available
    if "llm_model" in results[0] and "embedding_model" in results[0]:
        llm_model = results[0]["llm_model"]
        embedding_model = results[0]["embedding_model"]
        model_info = f" ({llm_model}, {embedding_model})"
    else:
        model_info = ""

    # Print summary
    print(f"\n=== PERFORMANCE COMPARISON SUMMARY{model_info} ===")
    print(f"Number of queries: {len(results)}")
    print("\nAverage Scores:")
    print(f"  SAGE Graph RAG: {np.mean(sage_scores):.2f}/10")
    print(f"  Traditional RAG: {np.mean(trad_scores):.2f}/10")

    print("\nBetter System Counts:")
    print(f"  SAGE Graph RAG better: {better_system_counts['system1']}")
    print(f"  Traditional RAG better: {better_system_counts['system2']}")
    print(f"  Tie: {better_system_counts['tie']}")
    if better_system_counts['error'] > 0:
        print(f"  Error: {better_system_counts['error']}")

    print("\nAverage Latency:")
    print(f"  SAGE Graph RAG: {np.mean(sage_latencies):.2f}s")
    print(f"  Traditional RAG: {np.mean(trad_latencies):.2f}s")

    print("\nSimilarity Score (between answers):")
    print(f"  Average F1: {np.mean(similarity_f1_scores):.4f}")

    # Create visualizations
    create_visualizations(results, prefix)


def create_visualizations(results: List[Dict[str, Any]], prefix: str = ""):
    """
    Create visualizations of the results

    Args:
        results: List of results to visualize
        prefix: Prefix for output files (e.g., model name)
    """
    if not results:
        logger.warning("No results to visualize")
        return

    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Extract data
    queries = [r["query"] for r in results]
    sage_scores = [r["llm_evaluation"].get("system1_score", 0) for r in results]
    trad_scores = [r["llm_evaluation"].get("system2_score", 0) for r in results]
    sage_latencies = [r["sage_response"]["latency"] for r in results]
    trad_latencies = [r["traditional_response"]["latency"] for r in results]
    similarity_f1_scores = [r["similarity_scores"].get("f1", 0) for r in results]

    # Get model information if available
    model_info = ""
    if results and "llm_model" in results[0] and "embedding_model" in results[0]:
        llm_model = results[0]["llm_model"]
        embedding_model = results[0]["embedding_model"]
        model_info = f"_{llm_model}_{embedding_model}"

    # Create DataFrame
    df = pd.DataFrame({
        'Query': queries,
        'SAGE Score': sage_scores,
        'Traditional Score': trad_scores,
        'SAGE Latency': sage_latencies,
        'Traditional Latency': trad_latencies,
        'Similarity F1': similarity_f1_scores
    })

    # Save DataFrame to CSV
    csv_filename = f"results/performance_data{model_info if prefix == '' else '_' + prefix}.csv"
    df.to_csv(csv_filename, index=False)
    logger.info(f"Performance data saved to {csv_filename}")

    # 1. Basic bar chart for scores
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot scores
    x = np.arange(len(queries))
    width = 0.35

    ax1.bar(x - width/2, sage_scores, width, label='SAGE Graph RAG')
    ax1.bar(x + width/2, trad_scores, width, label='Traditional RAG')

    ax1.set_ylabel('Score (out of 10)')
    ax1.set_title(f'Quality Scores by Query{" - " + prefix if prefix else ""}')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Q{i+1}" for i in range(len(queries))], rotation=45)
    ax1.legend()

    # Plot latencies
    ax2.bar(x - width/2, sage_latencies, width, label='SAGE Graph RAG')
    ax2.bar(x + width/2, trad_latencies, width, label='Traditional RAG')

    ax2.set_ylabel('Latency (seconds)')
    ax2.set_title(f'Response Latency by Query{" - " + prefix if prefix else ""}')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Q{i+1}" for i in range(len(queries))], rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"results/performance_comparison{model_info if prefix == '' else '_' + prefix}.png")
    plt.close()

    # 2. Create a scatter plot of score vs latency
    plt.figure(figsize=(10, 8))
    plt.scatter(sage_latencies, sage_scores, label='SAGE Graph RAG', alpha=0.7, s=100)
    plt.scatter(trad_latencies, trad_scores, label='Traditional RAG', alpha=0.7, s=100)

    # Add trend lines
    sage_z = np.polyfit(sage_latencies, sage_scores, 1)
    sage_p = np.poly1d(sage_z)
    plt.plot(sorted(sage_latencies), sage_p(sorted(sage_latencies)), "r--", alpha=0.5)

    trad_z = np.polyfit(trad_latencies, trad_scores, 1)
    trad_p = np.poly1d(trad_z)
    plt.plot(sorted(trad_latencies), trad_p(sorted(trad_latencies)), "b--", alpha=0.5)

    plt.xlabel('Latency (seconds)')
    plt.ylabel('Score (out of 10)')
    plt.title(f'Score vs Latency{" - " + prefix if prefix else ""}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results/score_vs_latency{model_info if prefix == '' else '_' + prefix}.png")
    plt.close()

    # 3. Create a histogram of score differences
    score_diffs = [sage - trad for sage, trad in zip(sage_scores, trad_scores)]
    plt.figure(figsize=(10, 6))
    plt.hist(score_diffs, bins=10, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('SAGE Score - Traditional Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Score Differences{" - " + prefix if prefix else ""}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results/score_differences{model_info if prefix == '' else '_' + prefix}.png")
    plt.close()

    # 4. Create a heatmap of scores by query
    if len(queries) >= 5:  # Only create heatmap if we have enough queries
        plt.figure(figsize=(12, 8))
        heatmap_data = pd.DataFrame({
            'SAGE': sage_scores,
            'Traditional': trad_scores
        }, index=[f"Q{i+1}" for i in range(len(queries))])

        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=10)
        plt.title(f'Score Heatmap by Query{" - " + prefix if prefix else ""}')
        plt.tight_layout()
        plt.savefig(f"results/score_heatmap{model_info if prefix == '' else '_' + prefix}.png")
        plt.close()

    logger.info(f"Visualizations saved to results/ directory with prefix {prefix if prefix else 'performance_comparison'}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare SAGE Graph RAG with Traditional RAG')
    parser.add_argument('--queries', type=str, help='Path to JSON file with test queries')
    parser.add_argument('--output', type=str, help='Path to output JSON file for results')
    parser.add_argument('--llm-models', type=str, help='Comma-separated list of LLM models to test')
    parser.add_argument('--embedding-models', type=str, help='Comma-separated list of embedding models to test')
    parser.add_argument('--evaluation-model', type=str, default="deepseek-r1-distill-llama-70b",
                        help='Model to use for evaluation')
    parser.add_argument('--process-messages', action='store_true',
                        help='Process messages from uploads folder before running comparison')
    parser.add_argument('--generate-qa', action='store_true',
                        help='Generate new QA pairs before running comparison')
    args = parser.parse_args()

    # Process messages if requested
    if args.process_messages:
        logger.info("Processing messages from uploads folder...")
        try:
            from message_processor import process_message_files
            processed_data = process_message_files("uploads")
            logger.info(f"Processed {len(processed_data)} message files.")
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}")

    # Generate QA pairs if requested
    if args.generate_qa:
        logger.info("Generating question-answer pairs...")
        try:
            from message_processor import generate_qa_pairs, save_qa_pairs
            qa_pairs = generate_qa_pairs(num_pairs=30)
            save_qa_pairs(qa_pairs, "qa_pairs.json")
            logger.info(f"Generated {len(qa_pairs)} question-answer pairs.")
        except Exception as e:
            logger.error(f"Error generating QA pairs: {str(e)}")

    # Load queries
    if args.queries:
        try:
            with open(args.queries, 'r') as f:
                queries = json.load(f)
                if isinstance(queries, list) and all(isinstance(q, dict) for q in queries):
                    # If the queries file contains QA pairs
                    queries = [q["question"] for q in queries]
        except Exception as e:
            logger.error(f"Error loading queries file: {str(e)}")
            queries = DEFAULT_TEST_QUERIES
    else:
        queries = DEFAULT_TEST_QUERIES

    # Parse models
    llm_models = None
    if args.llm_models:
        llm_models = [model.strip() for model in args.llm_models.split(',')]
        logger.info(f"Testing with LLM models: {llm_models}")

    embedding_models = None
    if args.embedding_models:
        embedding_models = [model.strip() for model in args.embedding_models.split(',')]
        logger.info(f"Testing with embedding models: {embedding_models}")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Run comparison
    run_comparison(
        queries=queries,
        output_file=args.output,
        llm_models=llm_models,
        embedding_models=embedding_models,
        evaluation_model=args.evaluation_model
    )


if __name__ == "__main__":
    main()
