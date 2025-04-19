"""
Script to connect the TAWOS knowledge graph with SAGE's existing knowledge graph.

This script:
1. Identifies connection points between TAWOS and SAGE graphs
2. Creates relationships between the two graphs
3. Updates the SAGE configuration to include TAWOS data in queries
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def identify_connection_points(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str
) -> List[Dict[str, Any]]:
    """
    Identify potential connection points between TAWOS and SAGE graphs.
    
    Args:
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        
    Returns:
        List of potential connection points
    """
    try:
        from neo4j import GraphDatabase
        
        logger.info(f"Connecting to Neo4j at {neo4j_uri}")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        connection_points = []
        
        # Find potential connections based on entity names
        with driver.session() as session:
            # Find TAWOS entities that match SAGE entities by name
            result = session.run("""
                MATCH (t:Entity)
                WHERE t.source = 'TAWOS'
                WITH t
                MATCH (s)
                WHERE s.name = t.name AND NOT s.source = 'TAWOS'
                RETURN t.id AS tawos_id, t.name AS name, t.type AS tawos_type,
                       id(s) AS sage_id, labels(s) AS sage_labels
                LIMIT 100
            """)
            
            for record in result:
                connection_points.append({
                    'tawos_id': record['tawos_id'],
                    'sage_id': record['sage_id'],
                    'name': record['name'],
                    'tawos_type': record['tawos_type'],
                    'sage_labels': record['sage_labels'],
                    'confidence': 0.8  # High confidence for exact name matches
                })
            
            # Find TAWOS entities that partially match SAGE entities
            result = session.run("""
                MATCH (t:Entity)
                WHERE t.source = 'TAWOS'
                WITH t
                MATCH (s)
                WHERE s.name CONTAINS t.name OR t.name CONTAINS s.name
                AND NOT s.name = t.name AND NOT s.source = 'TAWOS'
                RETURN t.id AS tawos_id, t.name AS tawos_name, t.type AS tawos_type,
                       id(s) AS sage_id, s.name AS sage_name, labels(s) AS sage_labels
                LIMIT 100
            """)
            
            for record in result:
                connection_points.append({
                    'tawos_id': record['tawos_id'],
                    'sage_id': record['sage_id'],
                    'tawos_name': record['tawos_name'],
                    'sage_name': record['sage_name'],
                    'tawos_type': record['tawos_type'],
                    'sage_labels': record['sage_labels'],
                    'confidence': 0.5  # Medium confidence for partial matches
                })
        
        driver.close()
        logger.info(f"Found {len(connection_points)} potential connection points")
        return connection_points
        
    except ImportError:
        logger.error("Neo4j Python driver not installed. Install with 'pip install neo4j'")
        return []
    except Exception as e:
        logger.error(f"Error identifying connection points: {e}")
        return []

def create_connections(
    connection_points: List[Dict[str, Any]],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    confidence_threshold: float = 0.7
) -> int:
    """
    Create relationships between TAWOS and SAGE entities.
    
    Args:
        connection_points: List of potential connection points
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        confidence_threshold: Minimum confidence score to create a connection
        
    Returns:
        Number of connections created
    """
    try:
        from neo4j import GraphDatabase
        
        logger.info(f"Connecting to Neo4j at {neo4j_uri}")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Filter connection points by confidence
        filtered_points = [p for p in connection_points if p.get('confidence', 0) >= confidence_threshold]
        logger.info(f"Creating {len(filtered_points)} connections with confidence >= {confidence_threshold}")
        
        connections_created = 0
        
        with driver.session() as session:
            for point in filtered_points:
                # Create a SAME_AS relationship between the entities
                result = session.run("""
                    MATCH (t:Entity {id: $tawos_id})
                    MATCH (s) WHERE id(s) = $sage_id
                    MERGE (t)-[r:SAME_AS {confidence: $confidence}]->(s)
                    RETURN count(r) AS count
                """, tawos_id=point['tawos_id'], sage_id=point['sage_id'], confidence=point['confidence'])
                
                connections_created += result.single()['count']
        
        driver.close()
        logger.info(f"Created {connections_created} connections between TAWOS and SAGE entities")
        return connections_created
        
    except ImportError:
        logger.error("Neo4j Python driver not installed. Install with 'pip install neo4j'")
        return 0
    except Exception as e:
        logger.error(f"Error creating connections: {e}")
        return 0

def update_sage_config(config_path: str) -> bool:
    """
    Update SAGE configuration to include TAWOS data in queries.
    
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
        
        # Update the configuration to include TAWOS data
        if 'graph_rag' not in config:
            config['graph_rag'] = {}
        
        if 'data_sources' not in config['graph_rag']:
            config['graph_rag']['data_sources'] = []
        
        # Check if TAWOS is already in the data sources
        tawos_exists = any(source.get('name') == 'TAWOS' for source in config['graph_rag']['data_sources'])
        
        if not tawos_exists:
            # Add TAWOS as a data source
            config['graph_rag']['data_sources'].append({
                'name': 'TAWOS',
                'type': 'neo4j',
                'enabled': True,
                'weight': 1.0,
                'description': 'Tabular And Web Object Segmentation dataset'
            })
        
        # Update the Neo4j query to include TAWOS entities
        if 'neo4j' not in config:
            config['neo4j'] = {}
        
        if 'queries' not in config['neo4j']:
            config['neo4j']['queries'] = {}
        
        # Update the entity query to include TAWOS entities
        entity_query = config['neo4j'].get('queries', {}).get('entity_query', '')
        if 'TAWOS' not in entity_query:
            # Add TAWOS to the entity query
            if entity_query:
                # Append to existing query
                entity_query += " UNION MATCH (e:Entity) WHERE e.source = 'TAWOS' RETURN e"
            else:
                # Create new query
                entity_query = "MATCH (e:Entity) WHERE e.source = 'TAWOS' RETURN e"
            
            config['neo4j']['queries']['entity_query'] = entity_query
        
        # Save the updated configuration
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated SAGE configuration: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating SAGE configuration: {e}")
        return False

def connect_to_sage(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    config_path: str,
    confidence_threshold: float = 0.7,
    auto_connect: bool = False
):
    """
    Connect the TAWOS knowledge graph with SAGE's existing knowledge graph.
    
    Args:
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        config_path: Path to the SAGE configuration file
        confidence_threshold: Minimum confidence score to create a connection
        auto_connect: Whether to automatically create connections
    """
    logger.info("Starting connection of TAWOS graph to SAGE graph")
    
    # Step 1: Identify connection points
    connection_points = identify_connection_points(neo4j_uri, neo4j_user, neo4j_password)
    
    if not connection_points:
        logger.warning("No connection points found between TAWOS and SAGE graphs")
        return
    
    # Save connection points to a file for review
    connections_file = Path("data/connection_points.json")
    connections_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(connections_file, 'w') as f:
        json.dump(connection_points, f, indent=2)
    
    logger.info(f"Saved {len(connection_points)} connection points to {connections_file}")
    
    # Step 2: Create connections (if auto_connect is enabled)
    if auto_connect:
        connections_created = create_connections(
            connection_points,
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            confidence_threshold
        )
        
        if connections_created == 0:
            logger.warning("No connections created between TAWOS and SAGE graphs")
    else:
        logger.info("Auto-connect is disabled. Review connection points and create connections manually")
    
    # Step 3: Update SAGE configuration
    if config_path:
        success = update_sage_config(config_path)
        if success:
            logger.info("SAGE configuration updated to include TAWOS data")
        else:
            logger.warning("Failed to update SAGE configuration")
    
    logger.info("TAWOS to SAGE connection process complete!")

def main():
    parser = argparse.ArgumentParser(description="Connect TAWOS graph to SAGE graph")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j database URI")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password")
    parser.add_argument("--config-path", type=str, default="config/sage_config.json", help="Path to SAGE configuration file")
    parser.add_argument("--confidence", type=float, default=0.7, help="Minimum confidence score to create a connection")
    parser.add_argument("--auto-connect", action="store_true", help="Automatically create connections")
    
    args = parser.parse_args()
    
    connect_to_sage(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.config_path,
        args.confidence,
        args.auto_connect
    )

if __name__ == "__main__":
    main()
