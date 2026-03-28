"""SAIA - Self-Adjustment on Information Addition

Lightweight implementation for change detection, impact radius, incremental re-embedding,
conflict resolution and plan invalidation. This is an initial implementation with
heuristic fact extraction and Neo4j integration. Replace heuristics with LLM-based
extractors and richer conflict resolution strategies as needed.
"""
import os
import re
import json
import logging
from typing import List, Dict, Any, Set
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"

load_dotenv(ROOT_DIR / ".env")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_Password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

logger = logging.getLogger(__name__)

# Simple helper to get driver (keeps parity with other modules)
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_session(driver):
    if NEO4J_DATABASE:
        return driver.session(database=NEO4J_DATABASE)
    return driver.session()

# Reuse existing embedding model if available; fallback to local instantiation
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return _embedding_model

# Heuristic fact extractor (quick, robust for current dataset)
def extract_facts(text: str) -> Dict[str, Any]:
    """Extract simple facts: employee IDs, mentions of approvals, reporting relationships, policies, dates."""
    facts = {
        "employees": set(),
        "relationships": [],  # list of tuples (subj, rel, obj)
        "policies": set(),
        "dates": set(),
    }

    # EMP IDs (EMP001 etc.)
    for m in re.finditer(r"EMP\d{3}", text):
        facts["employees"].add(m.group(0))

    # Simple patterns for "reports to" and "approved"
    for m in re.finditer(r"(EMP\d{3})\s+(reports to|reported to)\s+(EMP\d{3})", text, flags=re.IGNORECASE):
        facts["relationships"].append({"subject": m.group(1), "type": "REPORTS_TO", "object": m.group(3)})

    for m in re.finditer(r"(EMP\d{3})\s+(approved|approves|authorized)\s+([A-Za-z0-9_\-\s]+)", text, flags=re.IGNORECASE):
        facts["relationships"].append({"subject": m.group(1), "type": "APPROVES", "object": m.group(3).strip()})

    # Very small date extractor (YYYY-MM-DD or Month Year)
    for m in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", text):
        facts["dates"].add(m.group(1))

    # Policies (heuristic: 'Policy' followed by word/number)
    for m in re.finditer(r"Policy\s+([A-Za-z0-9_\-]+)", text, flags=re.IGNORECASE):
        facts["policies"].add(m.group(1))

    # Normalize sets to lists for JSON friendliness
    facts["employees"] = list(facts["employees"])
    facts["policies"] = list(facts["policies"])
    facts["dates"] = list(facts["dates"])

    logger.debug(f"Extracted facts: {facts}")
    return facts

# Query existing graph for overlapping facts
def query_existing_facts(driver, facts: Dict[str, Any]) -> Dict[str, Any]:
    existing = {"employees": set(), "relationships": [], "policies": set(), "dates": set()}
    with get_session(driver) as session:
        # Employees
        if facts.get("employees"):
            res = session.run(
                "MATCH (p:Person) WHERE p.id IN $ids RETURN p.id AS id",
                ids=facts["employees"]
            ).data()
            existing["employees"] = set([r["id"] for r in res])

        # Relationships: check REPORTS_TO edges among identified employees
        if facts.get("relationships"):
            for rel in facts["relationships"]:
                if rel["type"] == "REPORTS_TO" and rel["subject"] and rel["object"]:
                    r = session.run(
                        "MATCH (a:Person {id:$a})-[r:REPORTS_TO]->(b:Person {id:$b}) RETURN r",
                        a=rel["subject"], b=rel["object"]
                    ).single()
                    if r:
                        existing["relationships"].append({"subject": rel["subject"], "type": "REPORTS_TO", "object": rel["object"]})

        # Policies and dates - we keep as placeholders (could be stored on Policy nodes)
        # No-op for now

    # Normalize
    existing["employees"] = list(existing["employees"])
    logger.debug(f"Existing facts found: {existing}")
    return existing

# Compute diff between new facts and existing facts
def compute_diff(new_facts: Dict[str, Any], existing_facts: Dict[str, Any]) -> Dict[str, List[Any]]:
    diff = {"added": [], "modified": [], "contradicted": [], "confirmed": []}

    # Employees: added vs confirmed
    for emp in new_facts.get("employees", []):
        if emp in existing_facts.get("employees", []):
            diff["confirmed"].append({"type": "employee", "id": emp})
        else:
            diff["added"].append({"type": "employee", "id": emp})

    # Relationships: check for contradictions
    for rel in new_facts.get("relationships", []):
        exists = any((r["subject"] == rel["subject"] and r["object"] == rel["object"] and r["type"] == rel["type"]) for r in existing_facts.get("relationships", []))
        if exists:
            diff["confirmed"].append({"type": "relationship", **rel})
        else:
            # check if there is an inverse (i.e., contradiction signal for this dataset is limited)
            contrad = False
            for er in existing_facts.get("relationships", []):
                if er["type"] == rel["type"] and er["subject"] == rel["subject"] and er["object"] != rel["object"]:
                    contrad = True
            if contrad:
                diff["contradicted"].append({"type": "relationship", **rel})
            else:
                diff["added"].append({"type": "relationship", **rel})

    logger.info(f"Computed diff: {diff}")
    return diff

# Compute affected graph neighborhood (impact radius)
def compute_impact_radius(driver, diff: Dict[str, Any], max_depth: int = 2) -> Dict[str, Any]:
    affected_nodes: Set[str] = set()
    affected_chunks: Set[str] = set()
    affected_plans: Set[str] = set()

    with get_session(driver) as session:
        # For each modified/contradicted relationship, BFS out to depth
        nodes_of_interest = []
        for entry in diff.get("modified", []) + diff.get("contradicted", []):
            if entry.get("type") == "relationship":
                nodes_of_interest.append(entry.get("subject"))
                nodes_of_interest.append(entry.get("object"))

        for nid in nodes_of_interest:
            # Collect 0..depth neighbors
            q = """
            MATCH (n:Person {id:$id})-[*1..$depth]-(m)
            RETURN DISTINCT m
            """
            res = session.run(q, id=nid, depth=max_depth).data()
            for r in res:
                node = r.get("m")
                if node and hasattr(node, "id"):
                    affected_nodes.add(node.id)

        # Affected chunks: find chunks that are PART_OF documents connected to these nodes
        if affected_nodes:
            res = session.run(
                "MATCH (n:Person) WHERE n.id IN $ids WITH n MATCH (n)-[*1]-(d:Document)<-[:PART_OF]-(c:Chunk) RETURN DISTINCT c.chunk_id AS chunk_id",
                ids=list(affected_nodes)
            ).data()
            for r in res:
                affected_chunks.add(r["chunk_id"])

    impact = {
        "affected_nodes": list(affected_nodes),
        "affected_chunks": list(affected_chunks),
        "affected_plans": list(affected_plans),
        "severity": "high" if diff.get("contradicted") else ("medium" if diff.get("modified") else "low")
    }
    logger.info(f"Impact report: {impact}")
    return impact

# Incremental re-embedding
def re_embed_chunks(driver, chunk_ids: List[str], semantic_threshold: float = 0.1):
    model = get_embedding_model()
    updated = []
    with get_session(driver) as session:
        for cid in chunk_ids:
            row = session.run(
                "MATCH (c:Chunk {chunk_id:$cid}) RETURN c.content AS content, c.summary AS summary, c.embedding AS embedding",
                cid=cid
            ).single()
            if not row:
                continue
            content = row["content"]
            old_summary = row["summary"] or ""
            new_summary = old_summary  # default

            # Re-summarize using simple heuristic: append a note about recent changes (placeholder for LLM)
            new_summary = old_summary + ""  # Placeholder: in future call LLM with updated graph context

            old_emb = row["embedding"]
            new_emb = model.encode(new_summary).tolist()

            # Compute simple cosine similarity
            try:
                import numpy as _np
                cos_sim = _np.dot(_np.array(old_emb), _np.array(new_emb)) / (_np.linalg.norm(_np.array(old_emb)) * _np.linalg.norm(_np.array(new_emb)) + 1e-8)
            except Exception:
                cos_sim = 1.0

            if 1.0 - cos_sim > semantic_threshold:
                session.run(
                    "MATCH (c:Chunk {chunk_id:$cid}) SET c.summary=$summary, c.embedding=$embedding",
                    cid=cid, summary=new_summary, embedding=new_emb
                )
                updated.append(cid)
    logger.info(f"Re-embedded chunks (changed): {updated}")
    return updated

# Conflict resolution (simple temporal precedence strategy)
def resolve_conflicts(driver, diff: Dict[str, Any], new_doc_id: str, strategy: str = "temporal"):
    with get_session(driver) as session:
        for entry in diff.get("contradicted", []):
            if entry.get("type") == "relationship":
                subj = entry["subject"]
                obj = entry["object"]
                rel_type = entry["type"]
                if strategy == "temporal":
                    # Mark existing edges as superseded if present
                    session.run(
                        "MATCH (a:Person {id:$a})-[r:%s]->(b:Person {id:$b}) SET r.status='superseded', r.superseded_by=$doc, r.superseded_at=timestamp() RETURN r" % rel_type,
                        a=subj, b=obj, doc=new_doc_id
                    )
                    # Create new edge (materialized as a simple relationship to the document's facts)
                    session.run(
                        "MATCH (a:Person {id:$a}), (d:Document {doc_id:$doc}) MERGE (a)-[r:%s {source_doc:$doc}]->(d) RETURN r" % rel_type,
                        a=subj, doc=new_doc_id
                    )
                elif strategy == "human":
                    # Flag nodes for review
                    session.run(
                        "MATCH (a:Person {id:$a}) SET a.pending_review = true RETURN a",
                        a=subj
                    )
    logger.info("Conflict resolution applied")

# Placeholder for plan invalidation
def invalidate_plans(affected_plans: List[str]):
    # No plan cache exists yet; record invalidations to a log for audit
    for p in affected_plans:
        logger.info(f"Invalidated plan: {p}")

# Proactive re-reasoning for monitored queries (stub)
def proactive_re_reasoning(impact_report: Dict[str, Any]):
    # Find monitored queries touching affected nodes and re-run orchestrator (not implemented)
    logger.info("Proactive re-reasoning would run for impact: %s", json.dumps(impact_report))

# Persist impact report to results/saia
def persist_impact_report(doc_id: str, impact_report: Dict[str, Any]):
    output_dir = RESULTS_DIR / "saia"
    os.makedirs(output_dir, exist_ok=True)
    path = output_dir / f"impact_{doc_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(impact_report, f, indent=2)
    logger.info(f"Persisted SAIA impact report to {path}")

# Main trigger function
def trigger_saia(doc_id: str, document_text: str):
    logger.info(f"SAIA triggered for {doc_id}")
    driver = get_neo4j_driver()
    try:
        new_facts = extract_facts(document_text)
        existing = query_existing_facts(driver, new_facts)
        diff = compute_diff(new_facts, existing)
        impact = compute_impact_radius(driver, diff)
        impact["diff"] = diff

        # Re-embed affected chunks incrementally
        updated_chunks = re_embed_chunks(driver, impact.get("affected_chunks", []))
        impact["re_embedded_chunks"] = updated_chunks

        # Resolve conflicts and invalidate plans
        resolve_conflicts(driver, diff, doc_id, strategy="temporal")
        invalidate_plans(impact.get("affected_plans", []))

        # Persist impact report
        persist_impact_report(doc_id, impact)

        # Proactive re-reasoning if high severity
        if impact.get("severity") == "high":
            proactive_re_reasoning(impact)

        return impact
    finally:
        driver.close()
