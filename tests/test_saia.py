import json

import under_development.saia as saia


def test_extract_facts_parses_expected_entities():
    text = (
        "EMP001 reports to EMP002 on 2026-03-15. "
        "EMP001 approved Budget A. Policy SEC-01 applies."
    )
    facts = saia.extract_facts(text)

    assert "EMP001" in facts["employees"]
    assert "EMP002" in facts["employees"]
    assert "2026-03-15" in facts["dates"]
    assert "SEC-01" in facts["policies"]
    assert any(r["type"] == "REPORTS_TO" for r in facts["relationships"])
    assert any(r["type"] == "APPROVES" for r in facts["relationships"])


def test_compute_diff_flags_contradictions():
    new_facts = {
        "employees": ["EMP001"],
        "relationships": [{"subject": "EMP001", "type": "REPORTS_TO", "object": "EMP003"}],
    }
    existing_facts = {
        "employees": ["EMP001"],
        "relationships": [{"subject": "EMP001", "type": "REPORTS_TO", "object": "EMP002"}],
    }

    diff = saia.compute_diff(new_facts, existing_facts)
    assert {"type": "employee", "id": "EMP001"} in diff["confirmed"]
    assert any(item["type"] == "REPORTS_TO" and item["object"] == "EMP003" for item in diff["contradicted"])


def test_get_session_uses_database(monkeypatch):
    class Driver:
        def __init__(self):
            self.kwargs = None

        def session(self, **kwargs):
            self.kwargs = kwargs
            return object()

    driver = Driver()
    monkeypatch.setattr(saia, "NEO4J_DATABASE", "neo_db")
    saia.get_session(driver)
    assert driver.kwargs == {"database": "neo_db"}


def test_persist_impact_report_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(saia, "RESULTS_DIR", tmp_path)
    payload = {"severity": "low", "affected_chunks": []}

    saia.persist_impact_report("doc-x", payload)
    out_file = tmp_path / "saia" / "impact_doc-x.json"
    assert out_file.exists()
    assert json.loads(out_file.read_text(encoding="utf-8")) == payload


def test_trigger_saia_orchestrates_and_closes_driver(monkeypatch):
    calls = {
        "resolve": False,
        "invalidate": False,
        "persist": False,
        "proactive": False,
    }

    class Driver:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    driver = Driver()
    monkeypatch.setattr(saia, "get_neo4j_driver", lambda: driver)
    monkeypatch.setattr(saia, "extract_facts", lambda _t: {"employees": [], "relationships": []})
    monkeypatch.setattr(saia, "query_existing_facts", lambda _d, _f: {"employees": [], "relationships": []})
    monkeypatch.setattr(
        saia,
        "compute_diff",
        lambda _n, _e: {"added": [], "modified": [], "contradicted": [{"type": "relationship"}], "confirmed": []},
    )
    monkeypatch.setattr(
        saia,
        "compute_impact_radius",
        lambda _d, _diff: {"affected_nodes": [], "affected_chunks": ["c1"], "affected_plans": ["p1"], "severity": "high"},
    )
    monkeypatch.setattr(saia, "re_embed_chunks", lambda _d, _chunks: ["c1"])
    monkeypatch.setattr(saia, "resolve_conflicts", lambda *_args, **_kwargs: calls.__setitem__("resolve", True))
    monkeypatch.setattr(saia, "invalidate_plans", lambda _plans: calls.__setitem__("invalidate", True))
    monkeypatch.setattr(saia, "persist_impact_report", lambda _doc, _impact: calls.__setitem__("persist", True))
    monkeypatch.setattr(saia, "proactive_re_reasoning", lambda _impact: calls.__setitem__("proactive", True))

    result = saia.trigger_saia("doc-1", "text")

    assert result["severity"] == "high"
    assert result["re_embedded_chunks"] == ["c1"]
    assert calls["resolve"] is True
    assert calls["invalidate"] is True
    assert calls["persist"] is True
    assert calls["proactive"] is True
    assert driver.closed is True
