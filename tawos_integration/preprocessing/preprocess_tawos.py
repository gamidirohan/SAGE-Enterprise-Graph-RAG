"""
TAWOS Dataset Preprocessing for SAGE Integration

This script preprocesses the TAWOS dataset to make it compatible with SAGE's
graph-based RAG system. It handles downloading, extracting, and transforming
the data into a format suitable for training and evaluation.
"""

import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TAWOS_GITHUB_URL = "https://github.com/microsoft/TAWOS"
TAWOS_DATA_URL = "https://github.com/microsoft/TAWOS/raw/main/data/"
OUTPUT_DIR = Path("../data/processed")

def locate_tawos_sql_file(input_path: str = None) -> Path:
    """
    Locate the TAWOS SQL file in the specified path or search for it.

    Args:
        input_path: Path to the TAWOS SQL file or directory containing it

    Returns:
        Path to the TAWOS SQL file
    """
    if input_path:
        sql_path = Path(input_path)
        if sql_path.is_file() and sql_path.suffix.lower() == '.sql':
            logger.info(f"Found TAWOS SQL file at {sql_path}")
            return sql_path
        elif sql_path.is_dir():
            # Search for SQL files in the directory
            sql_files = list(sql_path.glob("**/*.sql"))
            if sql_files:
                logger.info(f"Found TAWOS SQL file at {sql_files[0]}")
                return sql_files[0]

    # Default location
    default_path = Path("D:/College/Sem_6/NLP/Project/SAGE-Enterprise-Graph-RAG/files/21308124/TAWOS.sql")
    if default_path.exists():
        logger.info(f"Using default TAWOS SQL file at {default_path}")
        return default_path

    raise FileNotFoundError("Could not locate TAWOS SQL file. Please specify the correct path.")

def parse_sql_file(sql_file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse the TAWOS SQL file and extract table data.

    Args:
        sql_file_path: Path to the TAWOS SQL file

    Returns:
        Dictionary containing extracted table data
    """
    logger.info(f"Parsing SQL file: {sql_file_path}")

    # Read the SQL file content
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_content = f.read()

    # Extract table creation statements and data insertion statements
    tables = {}

    # Find all CREATE TABLE statements - useful for schema analysis
    # but not used in this simplified implementation
    # create_table_pattern = re.compile(r'CREATE TABLE [^(]*\(([^;]*)\);', re.DOTALL)
    # create_table_matches = create_table_pattern.findall(sql_content)

    # Find all INSERT INTO statements
    insert_pattern = re.compile(r'INSERT INTO `([^`]+)`[^(]*\(([^)]*)\) VALUES\s*([^;]*);', re.DOTALL)
    insert_matches = insert_pattern.findall(sql_content)

    # Process INSERT statements to extract data
    for table_name, columns_str, values_str in insert_matches:
        if table_name not in tables:
            tables[table_name] = []

        # Parse column names
        columns = [col.strip('`') for col in columns_str.split(',')]

        # Parse values
        # This is a simplified approach - in a real implementation, you would need
        # more robust parsing to handle complex SQL value formats
        values_pattern = re.compile(r'\(([^)]*)\)', re.DOTALL)
        values_matches = values_pattern.findall(values_str)

        for value_str in values_matches:
            # Split by comma, but respect quoted strings
            values = []
            in_quote = False
            current_value = ""

            for char in value_str:
                if char == "'" and not in_quote:
                    in_quote = True
                    current_value += char
                elif char == "'" and in_quote:
                    in_quote = False
                    current_value += char
                elif char == ',' and not in_quote:
                    values.append(current_value.strip())
                    current_value = ""
                else:
                    current_value += char

            if current_value:
                values.append(current_value.strip())

            # Create a dictionary for this row
            row_dict = {}
            for i, col in enumerate(columns):
                if i < len(values):
                    # Remove quotes from string values
                    value = values[i]
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    row_dict[col] = value

            tables[table_name].append(row_dict)

    logger.info(f"Extracted data from {len(tables)} tables")
    return tables

def preprocess_tawos_for_graph_rag(
    sql_file_path: Path,
    output_dir: Path,
    split_ratio: Dict[str, float] = {"train": 0.7, "dev": 0.15, "test": 0.15}
) -> None:
    """
    Preprocess TAWOS dataset for integration with SAGE's graph RAG system.

    Args:
        sql_file_path: Path to the TAWOS SQL file
        output_dir: Directory to save the processed dataset
        split_ratio: Ratio for splitting data into train/dev/test sets
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preprocessing TAWOS dataset from SQL file")

    # Parse the SQL file
    tables = parse_sql_file(sql_file_path)

    # Process the extracted data
    documents = []
    entities = []
    relationships = []

    # Process tables to extract documents, entities, and relationships
    # This is a simplified example - in a real implementation, you would need
    # to understand the TAWOS schema and extract relevant information

    # Example: Extract documents from a 'documents' table
    if 'documents' in tables:
        for doc in tables['documents']:
            documents.append({
                'id': doc.get('id', ''),
                'title': doc.get('title', ''),
                'content': doc.get('content', ''),
                'metadata': {
                    'source': 'TAWOS',
                    'type': doc.get('type', ''),
                    'created_at': doc.get('created_at', '')
                }
            })

    # Example: Extract entities from an 'entities' table
    if 'entities' in tables:
        for entity in tables['entities']:
            entities.append({
                'id': entity.get('id', ''),
                'name': entity.get('name', ''),
                'type': entity.get('type', ''),
                'metadata': {
                    'source': 'TAWOS',
                    'document_id': entity.get('document_id', ''),
                    'created_at': entity.get('created_at', '')
                }
            })

    # Example: Extract relationships from a 'relationships' table
    if 'relationships' in tables:
        for rel in tables['relationships']:
            relationships.append({
                'id': rel.get('id', ''),
                'source_id': rel.get('source_id', ''),
                'target_id': rel.get('target_id', ''),
                'type': rel.get('type', ''),
                'metadata': {
                    'source': 'TAWOS',
                    'document_id': rel.get('document_id', ''),
                    'created_at': rel.get('created_at', '')
                }
            })

    # Create processed data for each split
    # No need to store the combined data since we're splitting it

    # Split the data into train/dev/test sets
    for split, ratio in split_ratio.items():
        # Simple random sampling for demonstration
        # In a real implementation, you might want to use stratified sampling
        # or other techniques to ensure balanced splits
        split_documents = documents[:int(len(documents) * ratio)]
        split_entities = entities[:int(len(entities) * ratio)]
        split_relationships = relationships[:int(len(relationships) * ratio)]

        split_data = {
            "documents": split_documents,
            "entities": split_entities,
            "relationships": split_relationships
        }

        # Save processed data
        with open(output_dir / f"{split}_processed.json", "w") as f:
            json.dump(split_data, f, indent=2)

        logger.info(f"Saved processed {split} data to {output_dir / f'{split}_processed.json'}")

def convert_to_neo4j_format(input_dir: Path, output_path: Path) -> None:
    """
    Convert processed TAWOS data to Neo4j-compatible format for graph integration.

    Args:
        input_dir: Directory containing the processed TAWOS data
        output_path: Path to save the Neo4j-compatible data
    """
    logger.info(f"Converting processed data to Neo4j format")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load processed data from all splits
    all_documents = []
    all_entities = []
    all_relationships = []

    for split in ["train", "dev", "test"]:
        file_path = input_dir / f"{split}_processed.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
                all_documents.extend(data.get("documents", []))
                all_entities.extend(data.get("entities", []))
                all_relationships.extend(data.get("relationships", []))

    # Create CSV files for Neo4j import

    # Document nodes
    with open(output_path / "document_nodes.csv", "w", newline='') as f:
        f.write("id,title,content,type,source,created_at\n")
        for doc in all_documents:
            # Escape quotes in content
            content = doc.get('content', '').replace('"', '""')
            title = doc.get('title', '').replace('"', '""')

            # Write CSV row
            f.write(f"{doc.get('id', '')},\"{title}\",\"{content}\",\"{doc.get('metadata', {}).get('type', '')}\",\"{doc.get('metadata', {}).get('source', '')}\",\"{doc.get('metadata', {}).get('created_at', '')}\"\n")

    # Entity nodes
    with open(output_path / "entity_nodes.csv", "w", newline='') as f:
        f.write("id,name,type,document_id,source,created_at\n")
        for entity in all_entities:
            # Escape quotes in name
            name = entity.get('name', '').replace('"', '""')

            # Write CSV row
            f.write(f"{entity.get('id', '')},\"{name}\",\"{entity.get('type', '')}\",\"{entity.get('metadata', {}).get('document_id', '')}\",\"{entity.get('metadata', {}).get('source', '')}\",\"{entity.get('metadata', {}).get('created_at', '')}\"\n")

    # Relationship edges
    with open(output_path / "relationships.csv", "w", newline='') as f:
        f.write("start_id,end_id,type,document_id,source,created_at\n")
        for rel in all_relationships:
            # Write CSV row
            f.write(f"{rel.get('source_id', '')},{rel.get('target_id', '')},{rel.get('type', '')},{rel.get('metadata', {}).get('document_id', '')},{rel.get('metadata', {}).get('source', '')},{rel.get('metadata', {}).get('created_at', '')}\n")

    # Create Cypher script for importing the data
    with open(output_path / "import.cypher", "w") as f:
        f.write("// Cypher script for importing TAWOS data into Neo4j\n\n")

        # Load document nodes
        f.write("// Load document nodes\n")
        f.write("LOAD CSV WITH HEADERS FROM 'file:///document_nodes.csv' AS row\n")
        f.write("CREATE (d:Document {\n")
        f.write("  id: row.id,\n")
        f.write("  title: row.title,\n")
        f.write("  content: row.content,\n")
        f.write("  type: row.type,\n")
        f.write("  source: row.source,\n")
        f.write("  created_at: row.created_at\n")
        f.write("});\n\n")

        # Load entity nodes
        f.write("// Load entity nodes\n")
        f.write("LOAD CSV WITH HEADERS FROM 'file:///entity_nodes.csv' AS row\n")
        f.write("CREATE (e:Entity {\n")
        f.write("  id: row.id,\n")
        f.write("  name: row.name,\n")
        f.write("  type: row.type,\n")
        f.write("  document_id: row.document_id,\n")
        f.write("  source: row.source,\n")
        f.write("  created_at: row.created_at\n")
        f.write("});\n\n")

        # Create indexes
        f.write("// Create indexes\n")
        f.write("CREATE INDEX document_id_index FOR (d:Document) ON (d.id);\n")
        f.write("CREATE INDEX entity_id_index FOR (e:Entity) ON (e.id);\n\n")

        # Load relationships
        f.write("// Load relationships\n")
        f.write("LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row\n")
        f.write("MATCH (source) WHERE source.id = row.start_id\n")
        f.write("MATCH (target) WHERE target.id = row.end_id\n")
        f.write("CREATE (source)-[r:RELATES_TO {\n")
        f.write("  type: row.type,\n")
        f.write("  document_id: row.document_id,\n")
        f.write("  source: row.source,\n")
        f.write("  created_at: row.created_at\n")
        f.write("  }]->(target);\n")

    logger.info(f"Saved Neo4j-compatible data to {output_path}")
    logger.info(f"Created {len(all_documents)} document nodes, {len(all_entities)} entity nodes, and {len(all_relationships)} relationships")

def main():
    parser = argparse.ArgumentParser(description="Preprocess TAWOS dataset for SAGE integration")
    parser.add_argument("--sql-file", type=str, help="Path to the TAWOS SQL file")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the TAWOS dataset")
    parser.add_argument("--convert", action="store_true", help="Convert to Neo4j format")
    parser.add_argument("--output-dir", type=str, default="../data/processed", help="Output directory for processed data")
    parser.add_argument("--neo4j-dir", type=str, default="../data/neo4j", help="Output directory for Neo4j data")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    neo4j_dir = Path(args.neo4j_dir)

    # Locate the SQL file
    try:
        sql_file_path = locate_tawos_sql_file(args.sql_file)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return

    if args.preprocess:
        preprocess_tawos_for_graph_rag(sql_file_path, output_dir)

    if args.convert:
        convert_to_neo4j_format(output_dir, neo4j_dir)

    logger.info("TAWOS preprocessing complete!")

if __name__ == "__main__":
    main()
