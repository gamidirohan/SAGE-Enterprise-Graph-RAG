"""
Script to check the Neo4j database for MISeD data.
"""

from neo4j import GraphDatabase

def check_neo4j_data():
    """Check the Neo4j database for MISeD data."""
    # Connect to Neo4j
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    
    # Check document count
    with driver.session() as session:
        result = session.run('MATCH (n:Document {source: "MISeD"}) RETURN count(n) as count')
        print(f'Number of MISeD documents: {result.single()["count"]}')
    
    # Check entity count
    with driver.session() as session:
        result = session.run('MATCH (n:Entity {source: "MISeD"}) RETURN count(n) as count')
        print(f'Number of MISeD entities: {result.single()["count"]}')
    
    # Check relationship count
    with driver.session() as session:
        result = session.run('MATCH (s)-[r:RELATES_TO]->(t) WHERE r.source = "MISeD" RETURN count(r) as count')
        print(f'Number of MISeD relationships: {result.single()["count"]}')
    
    # Check MENTIONS relationship count
    with driver.session() as session:
        result = session.run('MATCH (d:Document)-[r:MENTIONS]->(e:Entity) WHERE r.source = "MISeD" RETURN count(r) as count')
        print(f'Number of MENTIONS relationships: {result.single()["count"]}')
    
    # Close the connection
    driver.close()

if __name__ == "__main__":
    check_neo4j_data()
