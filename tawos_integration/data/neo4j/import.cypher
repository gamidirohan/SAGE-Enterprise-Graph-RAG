// Cypher script for importing MISeD data into Neo4j

// Load document nodes
LOAD CSV WITH HEADERS FROM 'file:///D:/College/Sem_6/NLP/Project/SAGE-Enterprise-Graph-RAG/tawos_integration/data/neo4j/document_nodes.csv' AS row
CREATE (d:Document {
  id: row.id,
  subject: row.subject,
  content: row.content,
  source: row.source
});

// Load entity nodes
LOAD CSV WITH HEADERS FROM 'file:///D:/College/Sem_6/NLP/Project/SAGE-Enterprise-Graph-RAG/tawos_integration/data/neo4j/entity_nodes.csv' AS row
CREATE (e:Entity {
  id: row.id,
  name: row.name,
  type: row.type,
  source: row.source
});

// Create indexes
CREATE INDEX document_id_index FOR (d:Document) ON (d.id);
CREATE INDEX entity_id_index FOR (e:Entity) ON (e.id);

// Load relationships
LOAD CSV WITH HEADERS FROM 'file:///D:/College/Sem_6/NLP/Project/SAGE-Enterprise-Graph-RAG/tawos_integration/data/neo4j/relationships.csv' AS row
MATCH (source:Entity {id: row.start_id})
MATCH (target:Entity {id: row.end_id})
CREATE (source)-[r:RELATES_TO {
  type: row.type,
  source: row.source,
  confidence: toFloat(row.confidence)
  }]->(target);

// Create document-entity relationships
MATCH (d:Document {source: 'MISeD'})
MATCH (e:Entity {source: 'MISeD'})
WHERE d.content CONTAINS e.name
CREATE (d)-[r:MENTIONS {source: 'MISeD'}]->(e);
