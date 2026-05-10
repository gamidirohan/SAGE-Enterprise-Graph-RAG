"""
Pilot evaluation harness: run_pilot_harness.py

Features:
- Load QA pairs from `data/eval/qa_pairs.json` (fallback to defaults in scripts/performance_comparison.py).
- Import SAGEGraphRAG, TraditionalRAG, PerformanceEvaluator from scripts/performance_comparison.py
- Run each query through both systems, capture `answer`, `thinking`, `context`, `latency`.
- Compute similarity/rouge/llm eval via PerformanceEvaluator.
- Grounding checks for `gold_evidence.cypher_paths` (runs Neo4j queries and records existence).
- Persist results to `results/pilot_results.json` and call `generate_summary()` / `create_visualizations()` if present.
- Supports `--mock` mode: deterministic placeholders for offline smoke tests.

Usage examples:
    python scripts/run_pilot_harness.py --output results/pilot_results.json --mock --limit 10

"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def safe_import_performance_module():
    try:
        from scripts.performance_comparison import (
            SAGEGraphRAG,
            TraditionalRAG,
            PerformanceEvaluator,
            DEFAULT_TEST_QUERIES,
            generate_summary,
            create_visualizations,
            get_neo4j_driver,
        )

        return {
            "SAGEGraphRAG": SAGEGraphRAG,
            "TraditionalRAG": TraditionalRAG,
            "PerformanceEvaluator": PerformanceEvaluator,
            "DEFAULT_TEST_QUERIES": DEFAULT_TEST_QUERIES,
            "generate_summary": generate_summary,
            "create_visualizations": create_visualizations,
            "get_neo4j_driver": get_neo4j_driver,
        }
    except Exception as e:
        logger.error("Could not import scripts.performance_comparison: %s", e)
        raise


class MockRAG:
    def __init__(self, name: str, llm_model: str = "mock-llm", embedding_model: str = "mock-emb"):
        self.name = name
        self.llm_model_name = llm_model
        self.embedding_model_name = embedding_model

    def query(self, question: str) -> Dict[str, Any]:
        start = time.time()
        # deterministic placeholder outputs
        answer = f"MOCK ANSWER for: {question[:80]}"
        thinking = ["mock_step_1", "mock_step_2"]
        context = [{"type": "mock", "note": "no real context in mock mode"}]
        latency = max(0.001, time.time() - start)
        return {"question": question, "answer": answer, "thinking": thinking, "context": context, "latency": latency}


def load_queries(queries_path: Optional[str], fallback_queries: List[str]) -> List[Dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    # primary: explicit queries_path if provided
    if queries_path:
        p = Path(queries_path)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # normalize to list of dicts
                if isinstance(data, list):
                    qa_list = []
                    for item in data:
                        if isinstance(item, str):
                            qa_list.append({"question": item})
                        elif isinstance(item, dict):
                            qa_list.append(item)
                    return qa_list
            except Exception as e:
                logger.warning("Failed to load queries from %s: %s. Falling back.", p, e)

    # fallback: data/eval/qa_pairs.json in repo
    fallback_path = root / "data" / "eval" / "qa_pairs.json"
    try:
        if fallback_path.exists():
            with open(fallback_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                qa_list = []
                for item in data:
                    if isinstance(item, str):
                        qa_list.append({"question": item})
                    elif isinstance(item, dict):
                        qa_list.append(item)
                if qa_list:
                    return qa_list
    except Exception as e:
        logger.warning("Failed to load fallback qa_pairs.json: %s", e)

    # final fallback: DEFAULT_TEST_QUERIES list
    qa_list = [{"question": q} for q in fallback_queries]
    return qa_list


def run_grounding_checks(driver_getter, cypher_paths: List[str]):
    results = []
    if driver_getter is None:
        for p in cypher_paths:
            results.append({"cypher": p, "exists": False, "error": "no_neo4j_driver"})
        return results

    try:
        driver = driver_getter()
    except Exception as e:
        logger.warning("Could not create Neo4j driver for grounding checks: %s", e)
        for p in cypher_paths:
            results.append({"cypher": p, "exists": False, "error": str(e)})
        return results

    try:
        session = driver.session()
    except Exception as e:
        logger.warning("Could not start Neo4j session: %s", e)
        for p in cypher_paths:
            results.append({"cypher": p, "exists": False, "error": str(e)})
        try:
            driver.close()
        except Exception:
            pass
        return results

    for p in cypher_paths:
        try:
            recs = session.run(p).data()
            exists = bool(recs)
            sample = recs[0] if recs else None
            results.append({"cypher": p, "exists": exists, "sample": sample})
        except Exception as e:
            results.append({"cypher": p, "exists": False, "error": str(e)})

    try:
        session.close()
    except Exception:
        pass
    try:
        driver.close()
    except Exception:
        pass

    return results


def main(argv: Optional[List[str]] = None):
    pm = safe_import_performance_module()
    SAGEGraphRAG = pm["SAGEGraphRAG"]
    TraditionalRAG = pm["TraditionalRAG"]
    PerformanceEvaluator = pm["PerformanceEvaluator"]
    DEFAULT_TEST_QUERIES = pm["DEFAULT_TEST_QUERIES"]
    generate_summary = pm.get("generate_summary")
    create_visualizations = pm.get("create_visualizations")
    get_neo4j_driver = pm.get("get_neo4j_driver")

    parser = argparse.ArgumentParser(description="Run a 30-query pilot comparing SAGEGraphRAG and TraditionalRAG")
    parser.add_argument("--queries", type=str, help="Path to QA pairs JSON (list of objects or questions)")
    parser.add_argument("--output", type=str, default="results/pilot_results.json", help="Path for results JSON")
    parser.add_argument("--models", type=str, help="Comma-separated LLM model(s) to test")
    parser.add_argument("--embeddings", type=str, help="Comma-separated embedding model(s) to test")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode with deterministic outputs (no LLM/DB calls)")
    parser.add_argument("--limit", type=int, default=30, help="Limit number of queries (default 30)")
    args = parser.parse_args(argv)

    # Load queries
    qa_pairs = load_queries(args.queries, DEFAULT_TEST_QUERIES)
    if not qa_pairs:
        logger.error("No QA pairs available to run the pilot")
        sys.exit(1)

    # Limit to specified number
    qa_pairs = qa_pairs[: args.limit]

    # Parse models and embeddings
    llm_models = []
    if args.models:
        llm_models = [m.strip() for m in args.models.split(",") if m.strip()]
    embedding_models = []
    if args.embeddings:
        embedding_models = [m.strip() for m in args.embeddings.split(",") if m.strip()]

    # Default to pilot config if none provided
    pilot_config_path = Path(__file__).resolve().parents[1] / "config" / "pilot_config.json"
    if (not llm_models or not embedding_models) and pilot_config_path.exists():
        try:
            with open(pilot_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not llm_models:
                llm_models = cfg.get("llms", [])
            if not embedding_models:
                embedding_models = cfg.get("embeddings", [])
        except Exception as e:
            logger.warning("Could not load pilot_config.json: %s", e)

    if not llm_models:
        llm_models = ["llama3-8b-8192"]
    if not embedding_models:
        embedding_models = ["all-mpnet-base-v2"]

    # Prepare output path
    out_path = Path(args.output)
    root = Path(__file__).resolve().parents[1]
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for llm in llm_models:
        for emb in embedding_models:
            logger.info("Running pilot with LLM=%s EMB=%s", llm, emb)

            # Instantiate systems (or mocks)
            if args.mock:
                sage = MockRAG("SAGEGraphRAG-mock", llm, emb)
                trad = MockRAG("TraditionalRAG-mock", llm, emb)
                # In mock mode avoid instantiating heavyweight evaluators or LLM clients
                evaluator = None
                neo_driver_getter = None
            else:
                try:
                    sage = SAGEGraphRAG(llm, emb)
                    trad = TraditionalRAG(llm, emb)
                except Exception as e:
                    logger.error("Failed to initialize RAG systems: %s", e)
                    raise

                evaluator = PerformanceEvaluator(llm_model=llm)
                neo_driver_getter = get_neo4j_driver if callable(get_neo4j_driver) else None

            # For each QA pair
            for idx, qa in enumerate(qa_pairs, start=1):
                question = qa.get("question") if isinstance(qa, dict) else str(qa)
                reference = qa.get("reference") if isinstance(qa, dict) else None
                gold = qa.get("gold_evidence") if isinstance(qa, dict) else None

                logger.info("Processing query %d/%d: %s", idx, len(qa_pairs), question[:140])

                try:
                    sage_resp = sage.query(question)
                except Exception as e:
                    logger.warning("SAGE query failed: %s", e)
                    sage_resp = {"question": question, "answer": f"error: {e}", "thinking": [], "context": [], "latency": 0}

                try:
                    trad_resp = trad.query(question)
                except Exception as e:
                    logger.warning("Traditional query failed: %s", e)
                    trad_resp = {"question": question, "answer": f"error: {e}", "thinking": [], "context": [], "latency": 0}

                # Evaluate
                similarity = None
                rouge = None
                llm_eval = None
                try:
                    if evaluator:
                        # If reference present, compare both to reference, else compare to each other
                        if reference:
                            similarity = evaluator.evaluate_similarity(sage_resp.get("answer", ""), trad_resp.get("answer", ""), reference=reference)
                            rouge = evaluator.evaluate_rouge(sage_resp.get("answer", ""), trad_resp.get("answer", ""), reference=reference)
                        else:
                            similarity = evaluator.evaluate_similarity(sage_resp.get("answer", ""), trad_resp.get("answer", ""), reference=None)
                            rouge = evaluator.evaluate_rouge(sage_resp.get("answer", ""), trad_resp.get("answer", ""), reference=None)

                        # LLM-based comparison (may be expensive) — wrap in try
                        try:
                            llm_eval = evaluator.evaluate_with_llm(question, sage_resp.get("answer", ""), trad_resp.get("answer", ""))
                        except Exception as e:
                            logger.warning("LLM-based evaluation failed: %s", e)
                            llm_eval = {"error": str(e)}
                except Exception as e:
                    logger.warning("Evaluation failed: %s", e)

                # Grounding checks
                grounding = None
                if gold and isinstance(gold, dict) and "cypher_paths" in gold:
                    try:
                        grounding = run_grounding_checks(neo_driver_getter, gold.get("cypher_paths", []))
                    except Exception as e:
                        grounding = [{"error": str(e)}]

                record = {
                    "query_index": idx,
                    "query": question,
                    "reference": reference,
                    "gold_evidence": gold,
                    "llm_model": llm,
                    "embedding_model": emb,
                    "sage_response": sage_resp,
                    "traditional_response": trad_resp,
                    "similarity": similarity,
                    "rouge": rouge,
                    "llm_evaluation": llm_eval,
                    "grounding_checks": grounding,
                }

                all_results.append(record)

            # cleanup (if real instances have close())
            if not args.mock:
                try:
                    sage.close()
                except Exception:
                    pass
                try:
                    trad.close()
                except Exception:
                    pass

    # Persist results
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Saved pilot results to %s", out_path)
    except Exception as e:
        logger.error("Failed to save results: %s", e)

    # Invoke generate_summary/create_visualizations if available
    try:
        if generate_summary:
            logger.info("Generating summary via performance_comparison.generate_summary()")
            try:
                generate_summary(all_results, prefix="pilot")
            except Exception as e:
                logger.warning("generate_summary() failed: %s", e)
    except Exception:
        pass

    try:
        if create_visualizations:
            logger.info("Creating visualizations via performance_comparison.create_visualizations()")
            try:
                create_visualizations(all_results, prefix="pilot")
            except Exception as e:
                logger.warning("create_visualizations() failed: %s", e)
    except Exception:
        pass

    logger.info("Pilot run complete. Results saved to %s", out_path)


if __name__ == "__main__":
    main()
