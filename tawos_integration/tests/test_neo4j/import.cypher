// Cypher script for importing TAWOS data into Neo4j

// Load document nodes
LOAD CSV WITH HEADERS FROM 'file:///document_nodes.csv' AS row
CREATE (d:Document {
  id: row.id,
  title: row.title,
  content: row.content,
  type: row.type,
  source: row.source,
  created_at: row.created_at
});

// Load entity nodes
LOAD CSV WITH HEADERS FROM 'file:///entity_nodes.csv' AS row
CREATE (e:Entity {
  id: row.id,
  name: row.name,
  type: row.type,
  document_id: row.document_id,
  source: row.source,
  created_at: row.created_at
});

// Create indexes
CREATE INDEX document_id_index FOR (d:Document) ON (d.id);
CREATE INDEX entity_id_index FOR (e:Entity) ON (e.id);

// Load relationships
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (source) WHERE source.id = row.start_id
MATCH (target) WHERE target.id = row.end_id
CREATE (source)-[r:RELATES_TO {
  type: row.type,
  document_id: row.document_id,
  source: row.source,
  created_at: row.created_at
  }]->(target);
