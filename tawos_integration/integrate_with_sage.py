"""
Script to integrate the TAWOS dataset with SAGE's graph RAG system.

This script:
1. Preprocesses the TAWOS dataset
2. Converts it to Neo4j format
3. Loads it into the Neo4j database
4. Connects it with the existing SAGE knowledge graph
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.preprocess_tawos import (
    locate_tawos_sql_file,
    preprocess_tawos_for_graph_rag,
    convert_to_neo4j_format
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def integrate_with_sage(
    sql_file_path: str = None,
    output_dir: str = "data/processed",
    neo4j_dir: str = "data/neo4j",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    load_to_neo4j: bool = False
):
    """
    Integrate the TAWOS dataset with SAGE's graph RAG system.
    
    Args:
        sql_file_path: Path to the TAWOS SQL file
        output_dir: Directory to save the processed data
        neo4j_dir: Directory to save the Neo4j-compatible data
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        load_to_neo4j: Whether to load the data into Neo4j
    """
    logger.info("Starting TAWOS integration with SAGE")
    
    # Step 1: Locate the SQL file
    try:
        sql_path = locate_tawos_sql_file(sql_file_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return
    
    # Create output directories
    output_path = Path(output_dir)
    neo4j_path = Path(neo4j_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    neo4j_path.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Preprocess the TAWOS dataset
    logger.info("Preprocessing TAWOS dataset")
    preprocess_tawos_for_graph_rag(sql_path, output_path)
    
    # Step 3: Convert to Neo4j format
    logger.info("Converting to Neo4j format")
    convert_to_neo4j_format(output_path, neo4j_path)
    
    # Step 4: Load into Neo4j (if requested)
    if load_to_neo4j:
        try:
            from neo4j import GraphDatabase
            
            logger.info(f"Connecting to Neo4j at {neo4j_uri}")
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            
            # Read the Cypher script
            with open(neo4j_path / "import.cypher", "r") as f:
                cypher_script = f.read()
            
            # Split the script into individual statements
            statements = [stmt.strip() for stmt in cypher_script.split(";") if stmt.strip()]
            
            # Execute each statement
            with driver.session() as session:
                for i, statement in enumerate(statements):
                    logger.info(f"Executing statement {i+1}/{len(statements)}")
                    session.run(statement)
            
            logger.info("Successfully loaded TAWOS data into Neo4j")
            driver.close()
            
        except ImportError:
            logger.error("Neo4j Python driver not installed. Install with 'pip install neo4j'")
        except Exception as e:
            logger.error(f"Error loading data into Neo4j: {e}")
    
    logger.info("TAWOS integration complete!")
    
    # Return paths to the generated files
    return {
        "processed_data": output_path,
        "neo4j_data": neo4j_path
    }

def main():
    parser = argparse.ArgumentParser(description="Integrate TAWOS dataset with SAGE")
    parser.add_argument("--sql-file", type=str, help="Path to the TAWOS SQL file")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save the processed data")
    parser.add_argument("--neo4j-dir", type=str, default="data/neo4j", help="Directory to save the Neo4j-compatible data")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j database URI")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password")
    parser.add_argument("--load-to-neo4j", action="store_true", help="Load the data into Neo4j")
    
    args = parser.parse_args()
    
    integrate_with_sage(
        args.sql_file,
        args.output_dir,
        args.neo4j_dir,
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.load_to_neo4j
    )

if __name__ == "__main__":
    main()
