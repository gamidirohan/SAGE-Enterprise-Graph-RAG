# Pilot Harness README

Requirements
- Python 3.9+
- Environment variables required by the harness when not using `--mock`:
  - `NEO4J_URI` (e.g. bolt://localhost:7687)
  - `NEO4J_USERNAME` or `NEO4J_USER`
  - `NEO4J_PASSWORD` or `NEO4J_Password`
  - `NEO4J_DATABASE` (optional)
  - `GROQ_API_KEY` (if using Groq-backed LLMs)

Recommended pip install (install from repository root):
```bash
python -m pip install -r requirements.txt
# If you don't have a requirements file, these packages are required:
python -m pip install neo4j sentence-transformers rouge-metric matplotlib pandas seaborn tqdm python-dotenv
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

Notes
- The harness will try `data/eval/qa_pairs.json` first (if present), otherwise it falls back to built-in defaults from `scripts/performance_comparison.py`.
- For reproducibility, use `--mock` during development and for CI smoke checks.
- Output: `results/pilot_results.json` plus optional CSV/plots created by the visualization helper.
