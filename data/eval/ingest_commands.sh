#!/usr/bin/env bash
# Template ingestion commands (edit host/endpoint/cli as needed)
# 1) Ingest users (via curl to API)
# Replace HOST with your backend host (e.g., http://localhost:8000)
HOST="http://localhost:8000"

# Ingest users (bulk)
curl -X POST "$HOST/api/ingest/users/bulk" \
  -H "Content-Type: application/json" \
  -d @data/eval/ingest_users_payloads.json

# Ingest documents (bulk)
curl -X POST "$HOST/api/ingest/documents/bulk" \
  -H "Content-Type: application/json" \
  -d @data/eval/ingest_payloads.json

# Bruno CLI template (if you have bruno installed): adjust flags to your CLI
# Example: bruno ingest users --file data/eval/ingest_users_payloads.json
# Example: bruno ingest docs --file data/eval/ingest_payloads.json

# Per-document ingestion using curl loop (if your API accepts single docs):
# jq is required for this loop; installs: apt-get install -y jq
# for i in $(jq -c '.[]' data/eval/ingest_payloads.json); do
#   echo "$i" | curl -X POST "$HOST/api/ingest/document" -H "Content-Type: application/json" -d @-
# done

# Run the pilot harness in mock mode (safe, offline):
python scripts/run_pilot_harness.py --mock --output results/pilot_results.json --limit 30

# Run the full pilot (requires env vars: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY):
# NEO4J_URI=bolt://localhost:7687 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=pass GROQ_API_KEY=xxx python scripts/run_pilot_harness.py --output results/pilot_results.json
