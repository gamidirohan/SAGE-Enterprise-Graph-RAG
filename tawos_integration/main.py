"""
Main Script for TAWOS Dataset Integration with SAGE

This script orchestrates the entire process of integrating the TAWOS dataset
with SAGE's graph-based RAG system, including downloading, preprocessing,
graph creation, and evaluation.
"""

import os
import argparse
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_preprocessing(args):
    """Run the preprocessing pipeline."""
    logger.info("Starting preprocessing pipeline")
    
    cmd = [
        "python", "preprocessing/preprocess_tawos.py",
        "--download" if args.download else "",
        "--preprocess",
        "--convert" if args.convert_neo4j else "",
        f"--input-dir={args.input_dir}",
        f"--output-dir={args.output_dir}",
        f"--neo4j-dir={args.neo4j_dir}",
        f"--splits={args.splits}"
    ]
    
    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    logger.info("Preprocessing complete")

def run_graph_integration(args):
    """Run the graph integration pipeline."""
    logger.info("Starting graph integration pipeline")
    
    cmd = [
        "python", "graph_integration.py",
        f"--data-dir={args.output_dir}",
        f"--neo4j-uri={args.neo4j_uri}",
        f"--neo4j-user={args.neo4j_user}",
        f"--neo4j-password={args.neo4j_password}"
    ]
    
    if args.mapping_file:
        cmd.append(f"--mapping-file={args.mapping_file}")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    logger.info("Graph integration complete")

def run_evaluation(args):
    """Run the evaluation pipeline."""
    logger.info("Starting evaluation pipeline")
    
    # Assuming we have test data and predictions
    test_data_path = Path(args.output_dir) / "test_processed.json"
    predictions_path = Path(args.output_dir) / "predictions.json"
    
    if not test_data_path.exists():
        logger.warning(f"Test data not found at {test_data_path}. Skipping evaluation.")
        return
    
    if not predictions_path.exists():
        logger.warning(f"Predictions not found at {predictions_path}. Skipping evaluation.")
        return
    
    cmd = [
        "python", "evaluation/evaluate_models.py",
        f"--test-data={test_data_path}",
        f"--predictions={predictions_path}",
        f"--output-dir={args.results_dir}",
        f"--tasks={args.tasks}"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    logger.info("Evaluation complete")

def main():
    parser = argparse.ArgumentParser(description="Integrate TAWOS dataset with SAGE")
    
    # General arguments
    parser.add_argument("--download", action="store_true", help="Download the TAWOS dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the TAWOS dataset")
    parser.add_argument("--integrate", action="store_true", help="Integrate with graph database")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model performance")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    # Directory arguments
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Input directory for raw data")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--neo4j-dir", type=str, default="data/neo4j", help="Output directory for Neo4j data")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory for evaluation results")
    
    # Preprocessing arguments
    parser.add_argument("--splits", type=str, default="train,dev,test", help="Data splits to process")
    parser.add_argument("--convert-neo4j", action="store_true", help="Convert to Neo4j format")
    
    # Graph integration arguments
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j database URI")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password")
    parser.add_argument("--mapping-file", type=str, help="Path to mapping file for connecting to enterprise graph")
    
    # Evaluation arguments
    parser.add_argument("--tasks", type=str, default="ner,re,dc", help="Tasks to evaluate")
    
    args = parser.parse_args()
    
    # Create directories
    for dir_path in [args.input_dir, args.output_dir, args.neo4j_dir, args.results_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Run the requested steps
    if args.all or args.preprocess:
        run_preprocessing(args)
    
    if args.all or args.integrate:
        run_graph_integration(args)
    
    if args.all or args.evaluate:
        run_evaluation(args)
    
    logger.info("TAWOS integration process complete!")

if __name__ == "__main__":
    main()
