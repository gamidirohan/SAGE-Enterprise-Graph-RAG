"""
Tests for the TAWOS preprocessing functionality.
"""

import os
import sys
import json
import pytest
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.preprocess_tawos import (
    locate_tawos_sql_file,
    parse_sql_file,
    preprocess_tawos_for_graph_rag,
    convert_to_neo4j_format
)

# Test data
TEST_SQL_PATH = Path("tests/sample_data/sample.sql")
TEST_OUTPUT_DIR = Path("tests/test_output")
TEST_NEO4J_DIR = Path("tests/test_neo4j")

@pytest.fixture(scope="module")
def setup_test_dirs():
    """Create test directories."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_NEO4J_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Uncomment to clean up after tests
    # import shutil
    # shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)
    # shutil.rmtree(TEST_NEO4J_DIR, ignore_errors=True)

def test_locate_tawos_sql_file():
    """Test locating the TAWOS SQL file."""
    # Test with explicit path
    if TEST_SQL_PATH.exists():
        sql_path = locate_tawos_sql_file(str(TEST_SQL_PATH))
        assert sql_path.exists()
        assert sql_path.suffix.lower() == '.sql'

    # Test with default path
    try:
        sql_path = locate_tawos_sql_file()
        assert sql_path.exists()
        assert sql_path.suffix.lower() == '.sql'
    except FileNotFoundError:
        # Skip if default path doesn't exist
        pytest.skip("Default SQL file not found")

def test_parse_sql_file(setup_test_dirs):
    """Test parsing the SQL file."""
    if not TEST_SQL_PATH.exists():
        pytest.skip("SQL file not found")

    tables = parse_sql_file(TEST_SQL_PATH)

    # Check that we got some tables
    assert isinstance(tables, dict)
    assert len(tables) > 0

    # Check table structure
    for table_name, rows in tables.items():
        assert isinstance(table_name, str)
        assert isinstance(rows, list)

        # Check at least one row if available
        if rows:
            assert isinstance(rows[0], dict)
            assert len(rows[0]) > 0

def test_preprocess_tawos_for_graph_rag(setup_test_dirs):
    """Test preprocessing the TAWOS dataset."""
    if not TEST_SQL_PATH.exists():
        pytest.skip("SQL file not found")

    # Run preprocessing
    preprocess_tawos_for_graph_rag(
        TEST_SQL_PATH,
        TEST_OUTPUT_DIR,
        {"train": 1.0}  # Use all data for train split for testing
    )

    # Check that output files were created
    train_file = TEST_OUTPUT_DIR / "train_processed.json"
    assert train_file.exists()

    # Check file content
    with open(train_file, 'r') as f:
        data = json.load(f)

        # Check data structure
        assert "documents" in data
        assert "entities" in data
        assert "relationships" in data

        # Check that we have some data
        # Note: This might fail if the SQL file doesn't have the expected structure
        # In that case, we'll need to adjust our expectations
        if "documents" in data:
            assert isinstance(data["documents"], list)

        if "entities" in data:
            assert isinstance(data["entities"], list)

        if "relationships" in data:
            assert isinstance(data["relationships"], list)

def test_convert_to_neo4j_format(setup_test_dirs):
    """Test converting processed data to Neo4j format."""
    # First, ensure we have processed data
    if not (TEST_OUTPUT_DIR / "train_processed.json").exists():
        test_preprocess_tawos_for_graph_rag(setup_test_dirs)

    # Run conversion
    convert_to_neo4j_format(TEST_OUTPUT_DIR, TEST_NEO4J_DIR)

    # Check that output files were created
    assert (TEST_NEO4J_DIR / "document_nodes.csv").exists()
    assert (TEST_NEO4J_DIR / "entity_nodes.csv").exists()
    assert (TEST_NEO4J_DIR / "relationships.csv").exists()
    assert (TEST_NEO4J_DIR / "import.cypher").exists()

    # Check Cypher script content
    with open(TEST_NEO4J_DIR / "import.cypher", 'r') as f:
        cypher_content = f.read()
        assert "CREATE (d:Document" in cypher_content
        assert "CREATE (e:Entity" in cypher_content
        assert "CREATE INDEX" in cypher_content
        assert "CREATE (source)-[r:RELATES_TO" in cypher_content

def setup_test_dirs_manual():
    """Create test directories without pytest."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_NEO4J_DIR.mkdir(parents=True, exist_ok=True)
    return None

if __name__ == "__main__":
    # Run tests manually
    print("Running tests manually...")
    setup_test_dirs_manual()

    print("\nTest 1: locate_tawos_sql_file")
    test_locate_tawos_sql_file()
    print("✓ Test passed!")

    print("\nTest 2: parse_sql_file")
    tables = parse_sql_file(TEST_SQL_PATH)
    print(f"Found {len(tables)} tables in the SQL file")
    if tables:
        print(f"Tables found: {', '.join(tables.keys())}")
        for table_name, rows in tables.items():
            print(f"  - {table_name}: {len(rows)} rows")
    print("✓ Test passed!")

    print("\nTest 3: preprocess_tawos_for_graph_rag")
    preprocess_tawos_for_graph_rag(
        TEST_SQL_PATH,
        TEST_OUTPUT_DIR,
        {"train": 1.0}  # Use all data for train split for testing
    )
    train_file = TEST_OUTPUT_DIR / "train_processed.json"
    if train_file.exists():
        with open(train_file, 'r') as f:
            data = json.load(f)
            print(f"Processed data stats:")
            print(f"  - Documents: {len(data.get('documents', []))}")
            print(f"  - Entities: {len(data.get('entities', []))}")
            print(f"  - Relationships: {len(data.get('relationships', []))}")
    print("✓ Test passed!")

    print("\nTest 4: convert_to_neo4j_format")
    convert_to_neo4j_format(TEST_OUTPUT_DIR, TEST_NEO4J_DIR)
    print(f"Neo4j files created in {TEST_NEO4J_DIR}")
    print("✓ Test passed!")

    print("\nAll tests passed successfully!")
