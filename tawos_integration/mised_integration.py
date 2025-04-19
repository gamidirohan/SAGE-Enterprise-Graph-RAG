"""
Script to integrate the MISeD dataset with SAGE's graph RAG system.

This script:
1. Processes the MISeD dataset (meeting transcripts with dialog turns)
2. Extracts entities and relationships
3. Creates a knowledge graph in Neo4j
4. Connects it with the existing SAGE knowledge graph
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_mised_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load the MISeD dataset from JSONL files.

    Args:
        data_path: Path to the MISeD dataset directory

    Returns:
        List of meeting transcripts with dialog turns
    """
    logger.info(f"Loading MISeD data from {data_path}")

    data_path = Path(data_path)
    all_data = []

    # Load data from train, validation, and test files
    for file_name in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        file_path = data_path / file_name
        if file_path.exists():
            logger.info(f"Loading data from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        all_data.append(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Error decoding JSON line in {file_path}")

    logger.info(f"Loaded {len(all_data)} meeting transcripts")
    return all_data

def extract_entities_from_transcript(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract entities from a meeting transcript.

    Args:
        transcript: Meeting transcript with dialog turns

    Returns:
        List of extracted entities
    """
    entities = []

    # Extract speakers as entities
    speakers = set()
    for segment in transcript.get("meeting", {}).get("transcriptSegments", []):
        speaker_name = segment.get("speakerName", "")
        if speaker_name and speaker_name not in speakers:
            speakers.add(speaker_name)
            entities.append({
                "id": f"speaker_{speaker_name.replace(' ', '_')}",
                "name": speaker_name,
                "type": "person",
                "source": "MISeD",
                "metadata": {
                    "role": "speaker",
                    "meeting_id": transcript.get("meeting", {}).get("meetingId", "")
                }
            })

    # Extract topics and concepts from dialog turns
    dialog_turns = transcript.get("dialog", {}).get("dialogTurns", [])
    for turn in dialog_turns:
        query = turn.get("query", "")
        response = turn.get("response", "")

        # Extract topics from query
        topics = extract_topics_from_text(query)
        for topic in topics:
            entity_id = f"topic_{topic.replace(' ', '_')}"
            # Check if entity already exists
            if not any(e["id"] == entity_id for e in entities):
                entities.append({
                    "id": entity_id,
                    "name": topic,
                    "type": "topic",
                    "source": "MISeD",
                    "metadata": {
                        "meeting_id": transcript.get("meetingId", "")
                    }
                })

        # Extract concepts from response
        concepts = extract_concepts_from_text(response)
        for concept in concepts:
            entity_id = f"concept_{concept.replace(' ', '_')}"
            # Check if entity already exists
            if not any(e["id"] == entity_id for e in entities):
                entities.append({
                    "id": entity_id,
                    "name": concept,
                    "type": "concept",
                    "source": "MISeD",
                    "metadata": {
                        "meeting_id": transcript.get("meetingId", "")
                    }
                })

    return entities

def extract_topics_from_text(text: str) -> List[str]:
    """
    Extract topics from text using simple NLP techniques.

    Args:
        text: Text to extract topics from

    Returns:
        List of extracted topics
    """
    # Simple approach: extract noun phrases
    # In a real implementation, you would use a more sophisticated NLP approach
    topics = []

    # Extract words that start with capital letters (potential topics)
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    topics.extend(words)

    # Extract technical terms
    tech_terms = [
        "microphone", "lapel", "digits", "recognition", "adaptation",
        "alignment", "forced alignment", "HTK", "SRI", "system",
        "TI-digits", "Switchboard", "speaker", "noise", "acoustics"
    ]

    for term in tech_terms:
        if term.lower() in text.lower() and term not in topics:
            topics.append(term)

    return topics

def extract_concepts_from_text(text: str) -> List[str]:
    """
    Extract concepts from text using simple NLP techniques.

    Args:
        text: Text to extract concepts from

    Returns:
        List of extracted concepts
    """
    # Simple approach: extract key phrases
    # In a real implementation, you would use a more sophisticated NLP approach
    concepts = []

    # Extract bullet points from markdown lists
    bullet_points = re.findall(r'\* (.*?)(?=\n\*|\n\n|$)', text)
    for point in bullet_points:
        # Extract the main concept from each bullet point
        # Remove leading/trailing punctuation and whitespace
        point = point.strip('.,: \t\n')
        if point and len(point) > 10:  # Minimum length to be considered a concept
            concepts.append(point)

    return concepts

def extract_relationships_from_transcript(transcript: Dict[str, Any], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract relationships from a meeting transcript.

    Args:
        transcript: Meeting transcript with dialog turns
        entities: List of extracted entities

    Returns:
        List of extracted relationships
    """
    relationships = []

    # Create a mapping of entity IDs to entities for easier lookup
    entity_map = {entity["id"]: entity for entity in entities}

    # Extract relationships between speakers and topics
    dialog_turns = transcript.get("dialog", {}).get("dialogTurns", [])
    for turn in dialog_turns:
        query = turn.get("query", "")
        response = turn.get("response", "")

        # Extract topics from query
        topics = extract_topics_from_text(query)

        # Extract speakers mentioned in the query
        speakers = []
        for entity in entities:
            if entity["type"] == "person" and entity["name"] in query:
                speakers.append(entity["id"])

        # Create relationships between speakers and topics
        for speaker_id in speakers:
            for topic in topics:
                topic_id = f"topic_{topic.replace(' ', '_')}"
                if topic_id in entity_map:
                    relationship_id = f"{speaker_id}_discusses_{topic_id}"
                    # Check if relationship already exists
                    if not any(r["id"] == relationship_id for r in relationships):
                        relationships.append({
                            "id": relationship_id,
                            "source_id": speaker_id,
                            "target_id": topic_id,
                            "type": "DISCUSSES",
                            "source": "MISeD",
                            "metadata": {
                                "meeting_id": transcript.get("meetingId", ""),
                                "confidence": 0.8
                            }
                        })

        # Extract concepts from response
        concepts = extract_concepts_from_text(response)

        # Create relationships between topics and concepts
        for topic in topics:
            topic_id = f"topic_{topic.replace(' ', '_')}"
            if topic_id in entity_map:
                for concept in concepts:
                    concept_id = f"concept_{concept.replace(' ', '_')}"
                    if concept_id in entity_map:
                        relationship_id = f"{topic_id}_relates_to_{concept_id}"
                        # Check if relationship already exists
                        if not any(r["id"] == relationship_id for r in relationships):
                            relationships.append({
                                "id": relationship_id,
                                "source_id": topic_id,
                                "target_id": concept_id,
                                "type": "RELATES_TO",
                                "source": "MISeD",
                                "metadata": {
                                    "meeting_id": transcript.get("meetingId", ""),
                                    "confidence": 0.7
                                }
                            })

    # Create a "PARTICIPATED_IN" relationship between speakers and the meeting
    meeting_id = transcript.get("meeting", {}).get("meetingId", "")
    if meeting_id:
        meeting_entity_id = f"meeting_{meeting_id}"

        # Add the meeting as an entity if it doesn't exist
        if meeting_entity_id not in entity_map:
            meeting_entity = {
                "id": meeting_entity_id,
                "name": f"Meeting {meeting_id}",
                "type": "meeting",
                "source": "MISeD",
                "metadata": {
                    "meeting_id": meeting_id
                }
            }
            entities.append(meeting_entity)
            entity_map[meeting_entity_id] = meeting_entity

        # Create relationships between speakers and the meeting
        for entity in entities:
            if entity["type"] == "person":
                relationship_id = f"{entity['id']}_participated_in_{meeting_entity_id}"
                # Check if relationship already exists
                if not any(r["id"] == relationship_id for r in relationships):
                    relationships.append({
                        "id": relationship_id,
                        "source_id": entity["id"],
                        "target_id": meeting_entity_id,
                        "type": "PARTICIPATED_IN",
                        "source": "MISeD",
                        "metadata": {
                            "meeting_id": meeting_id,
                            "confidence": 1.0
                        }
                    })

    return relationships

def create_document_from_transcript(transcript: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a document from a meeting transcript.

    Args:
        transcript: Meeting transcript with dialog turns

    Returns:
        Document in the format expected by pipeline.py
    """
    # Extract meeting ID and dialog ID
    meeting_id = transcript.get("meeting", {}).get("meetingId", "")
    dialog_id = transcript.get("dialogId", "")

    # Extract transcript segments
    segments = transcript.get("meeting", {}).get("transcriptSegments", [])

    # Create content from transcript segments
    content = ""
    for segment in segments:
        speaker = segment.get("speakerName", "")
        text = segment.get("text", "")
        content += f"{speaker}: {text}\n"

    # Create document
    document = {
        "doc_id": dialog_id,
        "sender": "MISeD",
        "receivers": ["SAGE"],
        "subject": f"Meeting Transcript: {meeting_id}",
        "content": content,
        "metadata": {
            "source": "MISeD",
            "meeting_id": meeting_id,
            "dialog_id": dialog_id
        }
    }

    return document

def process_mised_data(data_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Process the MISeD dataset and extract entities and relationships.

    Args:
        data_path: Path to the MISeD dataset directory
        output_dir: Directory to save the processed data

    Returns:
        Dictionary containing processed data
    """
    logger.info(f"Processing MISeD data from {data_path}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MISeD data
    transcripts = load_mised_data(data_path)

    # Process each transcript
    all_entities = []
    all_relationships = []
    all_documents = []

    for transcript in tqdm(transcripts, desc="Processing transcripts"):
        # Extract entities
        entities = extract_entities_from_transcript(transcript)
        all_entities.extend(entities)

        # Extract relationships
        relationships = extract_relationships_from_transcript(transcript, entities)
        all_relationships.extend(relationships)

        # Create document
        document = create_document_from_transcript(transcript)
        all_documents.append(document)

    # Remove duplicate entities and relationships
    unique_entities = []
    entity_ids = set()
    for entity in all_entities:
        if entity["id"] not in entity_ids:
            unique_entities.append(entity)
            entity_ids.add(entity["id"])

    unique_relationships = []
    relationship_ids = set()
    for relationship in all_relationships:
        if relationship["id"] not in relationship_ids:
            unique_relationships.append(relationship)
            relationship_ids.add(relationship["id"])

    # Create processed data
    processed_data = {
        "documents": all_documents,
        "entities": unique_entities,
        "relationships": unique_relationships
    }

    # Save processed data
    with open(output_dir / "mised_processed.json", "w") as f:
        json.dump(processed_data, f, indent=2)

    logger.info(f"Saved processed data to {output_dir / 'mised_processed.json'}")
    logger.info(f"Processed {len(all_documents)} documents, {len(unique_entities)} entities, and {len(unique_relationships)} relationships")

    return processed_data

def create_neo4j_import_files(processed_data: Dict[str, Any], output_dir: str) -> None:
    """
    Create Neo4j import files from processed data.

    Args:
        processed_data: Dictionary containing processed data
        output_dir: Directory to save the Neo4j import files
    """
    logger.info(f"Creating Neo4j import files in {output_dir}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create document nodes CSV
    with open(output_dir / "document_nodes.csv", "w", newline='', encoding='utf-8') as f:
        f.write("id,subject,content,source\n")
        for doc in processed_data["documents"]:
            # Escape quotes in content and subject
            content = doc.get("content", "").replace('"', '""')
            subject = doc.get("subject", "").replace('"', '""')

            # Write CSV row
            f.write(f"{doc.get('doc_id', '')},\"{subject}\",\"{content}\",\"MISeD\"\n")

    # Create entity nodes CSV
    with open(output_dir / "entity_nodes.csv", "w", newline='', encoding='utf-8') as f:
        f.write("id,name,type,source\n")
        for entity in processed_data["entities"]:
            # Escape quotes in name
            name = entity.get("name", "").replace('"', '""')

            # Write CSV row
            f.write(f"{entity.get('id', '')},\"{name}\",\"{entity.get('type', '')}\",\"MISeD\"\n")

    # Create relationship edges CSV
    with open(output_dir / "relationships.csv", "w", newline='', encoding='utf-8') as f:
        f.write("start_id,end_id,type,source,confidence\n")
        for rel in processed_data["relationships"]:
            # Write CSV row
            f.write(f"{rel.get('source_id', '')},{rel.get('target_id', '')},{rel.get('type', '')},MISeD,{rel.get('metadata', {}).get('confidence', 0.5)}\n")

    # Create Cypher script for importing the data
    with open(output_dir / "import.cypher", "w", encoding='utf-8') as f:
        f.write("// Cypher script for importing MISeD data into Neo4j\n\n")

        # Get absolute paths for CSV files
        doc_path = output_dir / "document_nodes.csv"
        entity_path = output_dir / "entity_nodes.csv"
        rel_path = output_dir / "relationships.csv"

        # Convert to forward slashes for Neo4j
        doc_path_str = str(doc_path.resolve()).replace("\\", "/")
        entity_path_str = str(entity_path.resolve()).replace("\\", "/")
        rel_path_str = str(rel_path.resolve()).replace("\\", "/")

        # Load document nodes
        f.write("// Load document nodes\n")
        f.write(f"LOAD CSV WITH HEADERS FROM 'file:///{doc_path_str}' AS row\n")
        f.write("CREATE (d:Document {\n")
        f.write("  id: row.id,\n")
        f.write("  subject: row.subject,\n")
        f.write("  content: row.content,\n")
        f.write("  source: row.source\n")
        f.write("});\n\n")

        # Load entity nodes
        f.write("// Load entity nodes\n")
        f.write(f"LOAD CSV WITH HEADERS FROM 'file:///{entity_path_str}' AS row\n")
        f.write("CREATE (e:Entity {\n")
        f.write("  id: row.id,\n")
        f.write("  name: row.name,\n")
        f.write("  type: row.type,\n")
        f.write("  source: row.source\n")
        f.write("});\n\n")

        # Create indexes
        f.write("// Create indexes\n")
        f.write("CREATE INDEX document_id_index FOR (d:Document) ON (d.id);\n")
        f.write("CREATE INDEX entity_id_index FOR (e:Entity) ON (e.id);\n\n")

        # Load relationships
        f.write("// Load relationships\n")
        f.write(f"LOAD CSV WITH HEADERS FROM 'file:///{rel_path_str}' AS row\n")
        f.write("MATCH (source:Entity {id: row.start_id})\n")
        f.write("MATCH (target:Entity {id: row.end_id})\n")
        f.write("CREATE (source)-[r:RELATES_TO {\n")
        f.write("  type: row.type,\n")
        f.write("  source: row.source,\n")
        f.write("  confidence: toFloat(row.confidence)\n")
        f.write("  }]->(target);\n\n")

        # Create document-entity relationships
        f.write("// Create document-entity relationships\n")
        f.write("MATCH (d:Document {source: 'MISeD'})\n")
        f.write("MATCH (e:Entity {source: 'MISeD'})\n")
        f.write("WHERE d.content CONTAINS e.name\n")
        f.write("CREATE (d)-[r:MENTIONS {source: 'MISeD'}]->(e);\n")

    logger.info(f"Created Neo4j import files in {output_dir}")

def load_into_neo4j(neo4j_dir: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
    """
    Load the processed data into Neo4j.

    Args:
        neo4j_dir: Directory containing Neo4j import files
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
    """
    logger.info(f"Loading data into Neo4j at {neo4j_uri}")

    try:
        from neo4j import GraphDatabase
        import json

        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Load the processed data
        output_dir = Path(neo4j_dir).parent / "processed"
        with open(output_dir / "mised_processed.json", "r", encoding="utf-8") as f:
            processed_data = json.load(f)

        # Load data directly using the Neo4j Python driver
        with driver.session() as session:
            # Create document nodes
            logger.info("Creating document nodes")
            for i, doc in enumerate(processed_data.get("documents", [])):
                if i % 50 == 0:
                    logger.info(f"Creating document {i+1}/{len(processed_data.get('documents', []))}")
                try:
                    session.run("""
                    CREATE (d:Document {
                        id: $id,
                        subject: $subject,
                        content: $content,
                        source: 'MISeD'
                    })
                    """, {
                        "id": doc.get("doc_id", ""),
                        "subject": doc.get("subject", ""),
                        "content": doc.get("content", "")
                    })
                except Exception as e:
                    logger.error(f"Error creating document node: {e}")

            # Create entity nodes
            logger.info("Creating entity nodes")
            for i, entity in enumerate(processed_data.get("entities", [])):
                if i % 100 == 0:
                    logger.info(f"Creating entity {i+1}/{len(processed_data.get('entities', []))}")
                try:
                    session.run("""
                    CREATE (e:Entity {
                        id: $id,
                        name: $name,
                        type: $type,
                        source: 'MISeD'
                    })
                    """, {
                        "id": entity.get("id", ""),
                        "name": entity.get("name", ""),
                        "type": entity.get("type", "")
                    })
                except Exception as e:
                    logger.error(f"Error creating entity node: {e}")

            # Create indexes
            logger.info("Creating indexes")
            try:
                session.run("CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)")
                session.run("CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            except Exception as e:
                logger.error(f"Error creating indexes: {e}")

            # Create relationships
            logger.info("Creating relationships")
            for i, rel in enumerate(processed_data.get("relationships", [])):
                if i % 100 == 0:
                    logger.info(f"Creating relationship {i+1}/{len(processed_data.get('relationships', []))}")
                try:
                    session.run("""
                    MATCH (source:Entity {id: $source_id})
                    MATCH (target:Entity {id: $target_id})
                    CREATE (source)-[r:RELATES_TO {
                        type: $type,
                        source: 'MISeD',
                        confidence: $confidence
                    }]->(target)
                    """, {
                        "source_id": rel.get("source_id", ""),
                        "target_id": rel.get("target_id", ""),
                        "type": rel.get("type", ""),
                        "confidence": rel.get("metadata", {}).get("confidence", 0.5)
                    })
                except Exception as e:
                    logger.error(f"Error creating relationship: {e}")

            # Create document-entity relationships
            logger.info("Creating document-entity relationships")
            try:
                session.run("""
                MATCH (d:Document {source: 'MISeD'})
                MATCH (e:Entity {source: 'MISeD'})
                WHERE d.content CONTAINS e.name AND size(e.name) > 3
                CREATE (d)-[r:MENTIONS {source: 'MISeD'}]->(e)
                """)
            except Exception as e:
                logger.error(f"Error creating document-entity relationships: {e}")

        # Close the connection
        driver.close()

        logger.info("Successfully loaded data into Neo4j")

    except ImportError:
        logger.error("Neo4j Python driver not installed. Install with 'pip install neo4j'")
    except Exception as e:
        logger.error(f"Error loading data into Neo4j: {e}")

def connect_to_sage(neo4j_uri: str, neo4j_user: str, neo4j_password: str, config_path: str = None) -> None:
    """
    Connect the MISeD knowledge graph to SAGE's existing knowledge graph.

    Args:
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        config_path: Path to the SAGE configuration file
    """
    logger.info("Connecting MISeD graph to SAGE graph")

    try:
        from neo4j import GraphDatabase

        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Create connections between MISeD and SAGE entities
        with driver.session() as session:
            # Find potential connections based on entity names
            result = session.run("""
                MATCH (m:Entity)
                WHERE m.source = 'MISeD'
                WITH m
                MATCH (s:Entity)
                WHERE s.name = m.name AND NOT s.source = 'MISeD'
                RETURN m.id AS mised_id, m.name AS name, m.type AS mised_type,
                       s.id AS sage_id, s.name AS sage_name, s.type AS sage_type
                LIMIT 100
            """)

            connections = []
            for record in result:
                connections.append({
                    'mised_id': record['mised_id'],
                    'sage_id': record['sage_id'],
                    'name': record['name'],
                    'mised_type': record['mised_type'],
                    'sage_type': record['sage_type'],
                    'confidence': 0.9  # High confidence for exact name matches
                })

            logger.info(f"Found {len(connections)} potential connections between MISeD and SAGE entities")

            # Create SAME_AS relationships between matching entities
            for connection in connections:
                session.run("""
                    MATCH (m:Entity {id: $mised_id})
                    MATCH (s:Entity {id: $sage_id})
                    MERGE (m)-[r:SAME_AS {confidence: $confidence}]->(s)
                    RETURN count(r) AS count
                """, mised_id=connection['mised_id'], sage_id=connection['sage_id'], confidence=connection['confidence'])

            logger.info(f"Created {len(connections)} connections between MISeD and SAGE entities")

        # Close the connection
        driver.close()

        # Update SAGE configuration if provided
        if config_path:
            update_sage_config(config_path)

        logger.info("MISeD to SAGE connection process complete!")

    except ImportError:
        logger.error("Neo4j Python driver not installed. Install with 'pip install neo4j'")
    except Exception as e:
        logger.error(f"Error connecting to SAGE: {e}")

def update_sage_config(config_path: str) -> bool:
    """
    Update SAGE configuration to include MISeD data in queries.

    Args:
        config_path: Path to the SAGE configuration file

    Returns:
        True if successful, False otherwise
    """
    try:
        config_file = Path(config_path)

        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return False

        # Load the configuration
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Update the configuration to include MISeD data
        if 'graph_rag' not in config:
            config['graph_rag'] = {}

        if 'data_sources' not in config['graph_rag']:
            config['graph_rag']['data_sources'] = []

        # Check if MISeD is already in the data sources
        mised_exists = any(source.get('name') == 'MISeD' for source in config['graph_rag']['data_sources'])

        if not mised_exists:
            # Add MISeD as a data source
            config['graph_rag']['data_sources'].append({
                'name': 'MISeD',
                'type': 'neo4j',
                'enabled': True,
                'weight': 1.0,
                'description': 'Meeting Intelligence and Summarization Evaluation Dataset'
            })

        # Update the Neo4j query to include MISeD entities
        if 'neo4j' not in config:
            config['neo4j'] = {}

        if 'queries' not in config['neo4j']:
            config['neo4j']['queries'] = {}

        # Update the entity query to include MISeD entities
        entity_query = config['neo4j'].get('queries', {}).get('entity_query', '')
        if 'MISeD' not in entity_query:
            # Add MISeD to the entity query
            if entity_query:
                # Append to existing query
                entity_query += " UNION MATCH (e:Entity) WHERE e.source = 'MISeD' RETURN e"
            else:
                # Create new query
                entity_query = "MATCH (e:Entity) WHERE e.source = 'MISeD' RETURN e"

            config['neo4j']['queries']['entity_query'] = entity_query

        # Save the updated configuration
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Updated SAGE configuration: {config_path}")
        return True

    except Exception as e:
        logger.error(f"Error updating SAGE configuration: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Integrate MISeD dataset with SAGE")
    parser.add_argument("--data-path", type=str, default="C:/Users/rm140/OneDrive/Desktop/2024.findings-emnlp.106.data/MISeD-main/mised", help="Path to the MISeD dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save the processed data")
    parser.add_argument("--neo4j-dir", type=str, default="data/neo4j", help="Directory to save the Neo4j import files")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j database URI")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password")
    parser.add_argument("--config-path", type=str, default="config/sage_config.json", help="Path to the SAGE configuration file")
    parser.add_argument("--load-to-neo4j", action="store_true", help="Load the data into Neo4j")
    parser.add_argument("--connect-to-sage", action="store_true", help="Connect the MISeD graph to SAGE's graph")

    args = parser.parse_args()

    # Process MISeD data
    processed_data = process_mised_data(args.data_path, args.output_dir)

    # Create Neo4j import files
    create_neo4j_import_files(processed_data, args.neo4j_dir)

    # Load data into Neo4j if requested
    if args.load_to_neo4j:
        load_into_neo4j(args.neo4j_dir, args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    # Connect to SAGE if requested
    if args.connect_to_sage:
        connect_to_sage(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.config_path)

if __name__ == "__main__":
    main()
