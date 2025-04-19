"""
Test script for MISeD integration.
"""

import os
import sys
import json
import unittest
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mised_integration import (
    load_mised_data,
    extract_entities_from_transcript,
    extract_relationships_from_transcript,
    create_document_from_transcript,
    process_mised_data,
    create_neo4j_import_files
)

class TestMISeDIntegration(unittest.TestCase):
    """Test cases for MISeD integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent / "sample_data"
        self.test_output_dir = Path(__file__).parent / "test_output"
        self.test_neo4j_dir = Path(__file__).parent / "test_neo4j"

        # Create test directories if they don't exist
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        self.test_neo4j_dir.mkdir(parents=True, exist_ok=True)

        # Create a sample MISeD transcript
        self.sample_transcript = {
            "dialogId": "test_dialog_id",
            "meeting": {
                "meetingId": "test_meeting_id",
                "transcriptSegments": [
                    {"text": "Hello, everyone.", "speakerName": "Speaker A"},
                    {"text": "Today we'll discuss the project.", "speakerName": "Speaker A"},
                    {"text": "I have some concerns about the timeline.", "speakerName": "Speaker B"},
                    {"text": "Let's review the requirements.", "speakerName": "Speaker C"}
                ]
            },
            "dialog": {
                "dialogTurns": [
                    {
                        "query": "What did Speaker A discuss?",
                        "response": "Speaker A introduced the meeting and mentioned that they would discuss the project.",
                        "responseAttribution": {"indexRanges": [{"startIndex": 0, "endIndex": 1}]},
                        "queryMetadata": {"queryType": "QUERY_TYPE_SPECIFIC"}
                    },
                    {
                        "query": "What concerns did Speaker B raise?",
                        "response": "Speaker B expressed concerns about the project timeline.",
                        "responseAttribution": {"indexRanges": [{"startIndex": 2, "endIndex": 2}]},
                        "queryMetadata": {"queryType": "QUERY_TYPE_SPECIFIC"}
                    }
                ]
            }
        }

        # Save the sample transcript to a JSONL file
        sample_file = self.test_data_dir / "sample.jsonl"
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.sample_transcript))

        # Also save as train.jsonl to match the expected file pattern
        train_file = self.test_data_dir / "train.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.sample_transcript))

    def test_load_mised_data(self):
        """Test loading MISeD data."""
        # Test with the sample data
        data = load_mised_data(self.test_data_dir)

        # Check that data was loaded
        self.assertGreater(len(data), 0)
        self.assertEqual(data[0]["dialogId"], self.sample_transcript["dialogId"])

    def test_extract_entities_from_transcript(self):
        """Test extracting entities from a transcript."""
        # Extract entities from the sample transcript
        entities = extract_entities_from_transcript(self.sample_transcript)

        # Check that entities were extracted
        self.assertGreater(len(entities), 0)

        # Check that speakers were extracted as entities
        speaker_entities = [e for e in entities if e["type"] == "person"]
        self.assertEqual(len(speaker_entities), 3)

        # Check that topics were extracted
        topic_entities = [e for e in entities if e["type"] == "topic"]
        self.assertGreater(len(topic_entities), 0)

    def test_extract_relationships_from_transcript(self):
        """Test extracting relationships from a transcript."""
        # Extract entities first
        entities = extract_entities_from_transcript(self.sample_transcript)

        # Extract relationships
        relationships = extract_relationships_from_transcript(self.sample_transcript, entities)

        # Check that relationships were extracted
        self.assertGreater(len(relationships), 0)

        # Check that there are PARTICIPATED_IN relationships
        participated_relationships = [r for r in relationships if r["type"] == "PARTICIPATED_IN"]
        self.assertGreater(len(participated_relationships), 0)

    def test_create_document_from_transcript(self):
        """Test creating a document from a transcript."""
        # Create a document from the sample transcript
        document = create_document_from_transcript(self.sample_transcript)

        # Check that the document was created correctly
        self.assertEqual(document["doc_id"], self.sample_transcript["dialogId"])
        self.assertEqual(document["subject"], f"Meeting Transcript: {self.sample_transcript['meeting']['meetingId']}")
        self.assertIn("Speaker A: Hello, everyone.", document["content"])

    def test_process_mised_data(self):
        """Test processing MISeD data."""
        # Process the sample data
        processed_data = process_mised_data(self.test_data_dir, self.test_output_dir)

        # Check that data was processed
        self.assertIn("documents", processed_data)
        self.assertIn("entities", processed_data)
        self.assertIn("relationships", processed_data)

        # Check that the output file was created
        output_file = self.test_output_dir / "mised_processed.json"
        self.assertTrue(output_file.exists())

    def test_create_neo4j_import_files(self):
        """Test creating Neo4j import files."""
        # Process the sample data first
        processed_data = process_mised_data(self.test_data_dir, self.test_output_dir)

        # Create Neo4j import files
        create_neo4j_import_files(processed_data, self.test_neo4j_dir)

        # Check that the import files were created
        self.assertTrue((self.test_neo4j_dir / "document_nodes.csv").exists())
        self.assertTrue((self.test_neo4j_dir / "entity_nodes.csv").exists())
        self.assertTrue((self.test_neo4j_dir / "relationships.csv").exists())
        self.assertTrue((self.test_neo4j_dir / "import.cypher").exists())

if __name__ == "__main__":
    # Create the sample_data directory if it doesn't exist
    sample_data_dir = Path(__file__).parent / "sample_data"
    sample_data_dir.mkdir(parents=True, exist_ok=True)

    # Run the tests
    unittest.main()
