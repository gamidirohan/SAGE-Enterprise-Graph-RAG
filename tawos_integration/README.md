# TAWOS Dataset Integration for SAGE

This directory contains code and resources for integrating the TAWOS (Tabular And Web Object Segmentation) dataset with the SAGE Enterprise Graph RAG system.

## Purpose

The TAWOS dataset is integrated with SAGE for:
1. Training and evaluating NLP components in SAGE
2. Benchmarking performance against state-of-the-art systems
3. Data augmentation to improve model performance on enterprise documents
4. Enhancing the knowledge graph with additional entities and relationships

## Directory Structure

- `preprocessing/`: Scripts for preprocessing TAWOS data
- `evaluation/`: Scripts for evaluating model performance using TAWOS
- `tests/`: Test scripts to verify the integration functionality
- `data/`: Processed TAWOS data (created during preprocessing)

## Requirements

```
pip install -r requirements.txt
```

## Quick Start

### 1. Integrate TAWOS with SAGE

```bash
python integrate_with_sage.py --sql-file "path/to/TAWOS.sql" --output-dir "data/processed" --neo4j-dir "data/neo4j"
```

This will:
- Preprocess the TAWOS SQL data
- Convert it to a format compatible with SAGE's graph RAG system
- Generate Neo4j-compatible files for import

### 2. Load TAWOS Data into Neo4j

```bash
python integrate_with_sage.py --sql-file "path/to/TAWOS.sql" --load-to-neo4j --neo4j-uri "bolt://localhost:7687" --neo4j-user "neo4j" --neo4j-password "your_password"
```

### 3. Connect TAWOS Graph to SAGE Graph

```bash
python connect_to_sage.py --neo4j-uri "bolt://localhost:7687" --neo4j-user "neo4j" --neo4j-password "your_password" --config-path "path/to/sage_config.json" --auto-connect
```

This will:
- Identify connection points between TAWOS and SAGE entities
- Create relationships between the two graphs
- Update the SAGE configuration to include TAWOS data in queries

## Manual Integration

### 1. Preprocess TAWOS Data

```bash
python -m preprocessing.preprocess_tawos --sql-file "path/to/TAWOS.sql" --preprocess --output-dir "data/processed"
```

### 2. Convert to Neo4j Format

```bash
python -m preprocessing.preprocess_tawos --sql-file "path/to/TAWOS.sql" --convert --output-dir "data/processed" --neo4j-dir "data/neo4j"
```

### 3. Import into Neo4j

Use the generated Cypher script in `data/neo4j/import.cypher` to import the data into Neo4j.

## Testing

Run the tests to verify the integration functionality:

```bash
python -m tests.test_preprocessing
```

## Integration Details

### Data Preprocessing

The TAWOS SQL data is processed to extract:
- Documents: Text content from the dataset
- Entities: Named entities, concepts, and keywords
- Relationships: Connections between entities

### Neo4j Integration

The processed data is converted to Neo4j-compatible format:
- Document nodes: Represent text documents
- Entity nodes: Represent entities extracted from documents
- Relationships: Connect entities to each other and to documents

### SAGE Integration

The TAWOS knowledge graph is connected to the SAGE knowledge graph by:
1. Identifying matching entities between the two graphs
2. Creating SAME_AS relationships between matching entities
3. Updating the SAGE configuration to include TAWOS data in queries

## Troubleshooting

### Permission Issues

If you encounter permission issues with the SQL file, make sure you have read access to the file.

### Neo4j Connection Issues

If you have trouble connecting to Neo4j, check that:
- Neo4j is running
- The URI, username, and password are correct
- The Neo4j Python driver is installed (`pip install neo4j`)

### Data Quality Issues

If the processed data doesn't look right, check that:
- The SQL file has the expected structure
- The preprocessing script is correctly parsing the SQL file
- The Neo4j import script is correctly formatting the data
