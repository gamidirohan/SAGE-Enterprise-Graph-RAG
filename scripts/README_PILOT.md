# Pilot Harness README

Requirements
- Python 3.9+
- Environment variables required by the harness when not using `--mock`:
  - `NEO4J_URI` (e.g. bolt://localhost:7687)
  - `NEO4J_USERNAME` or `NEO4J_USER`
  - `NEO4J_PASSWORD` or `NEO4J_Password`
  - `NEO4J_DATABASE` (optional)
  - `GROQ_API_KEY` (if using Groq-backed LLMs)

Recommended setup from the repository root:
```bash
uv sync
```

Quick commands

- Run mock smoke test (deterministic, no DB or LLM calls):
```bash
python scripts/run_pilot_harness.py --mock --output results/pilot_results.json --limit 5
```

- Run full pilot using the default pilot config (reads config/pilot_config.json):
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=secret
export GROQ_API_KEY=your_api_key_here
python scripts/run_pilot_harness.py --output results/pilot_results.json
```

- Provide custom QA pairs file:
```bash
python scripts/run_pilot_harness.py --queries data/eval/qa_pairs.json --output results/pilot_results.json
```

Review 2 fixture-based harness

This harness is the reproducible review/demo path. Each fixture first ingests
the exact users, messages, and documents needed for a question, verifies graph
and SAIA artifacts, then compares baseline fixture-only RAG with live agentic
SAGE through `/api/chat`.

- Run a small smoke pass:
```bash
source .venv/bin/activate
uv run python scripts/run_review2_fixture_eval.py --limit 2 --extractive-baseline
```

- Run all Review 2 fixtures:
```bash
source .venv/bin/activate
uv run python scripts/run_review2_fixture_eval.py --strict --extractive-baseline
```

- Run one bucket:
```bash
source .venv/bin/activate
uv run python scripts/run_review2_fixture_eval.py --bucket multi_hop_relationship
```

- Remove only Review-2 temporary fixture graph data:
```bash
source .venv/bin/activate
uv run python scripts/run_review2_fixture_eval.py --cleanup-only
```

Outputs:
- `../results/results1/review2_fixture_results.json`
- `../results/results1/review2_fixture_summary.csv`
- `../results/results1/review2_abnormalities.csv`
- `../results/results1/*.png`

The Review-2 runner cleans namespaced fixture data before the run, between
fixtures, and after the run by default. Use `--no-cleanup-after` only when you
intentionally want to inspect the temporary `review2-*` graph nodes in Neo4j
after execution.

During Review-2 evaluation, SAGE retrieval is also scoped to the current
fixture's document ids by default through `SAGE_EVAL_ALLOWED_DOC_IDS`. Use
`--no-isolated-sage-retrieval` only when you want to measure behavior against
the full live graph, including unrelated historical data.

Notes
- The harness will try `data/eval/qa_pairs.json` first (if present), otherwise it falls back to built-in defaults from `scripts/performance_comparison.py`.
- For reproducibility, use `--mock` during development and for CI smoke checks.
- Output: `results/pilot_results.json` plus optional CSV/plots created by the visualization helper.
