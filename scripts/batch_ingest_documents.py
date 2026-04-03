"""Batch ingest document files into Neo4j."""

from pathlib import Path
import argparse
import sys


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import document_ingestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch ingest supported documents into Neo4j.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(document_ingestion.default_document_directory()),
        help="Directory containing TXT, PDF, or DOCX documents.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Recompute and upsert documents even when the same content hash already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = document_ingestion.ingest_document_directory(
        args.directory,
        skip_existing=not args.reprocess,
    )

    if not summary["exists"]:
        print(f"Directory does not exist: {summary['directory']}")
        return 1

    print(f"Directory: {summary['directory']}")
    print(f"Document files seen: {summary['document_files_seen']}")
    print(f"Stored: {summary['stored']}")
    print(f"Skipped duplicates: {summary['skipped_duplicates']}")
    print(f"Failed: {summary['failed']}")
    print(f"Mapping files processed: {summary['mapping_files_processed']}")
    print(f"Mapping entries upserted: {summary['mapping_entries_upserted']}")
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
