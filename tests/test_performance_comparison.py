import json

from scripts import performance_comparison as perf


def test_save_qa_pairs_writes_output(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, "ROOT_DIR", tmp_path)
    qa_pairs = [{"question": "q1", "answer": "a1"}]

    ok = perf.save_qa_pairs(qa_pairs, "out/qa.json")
    output_file = tmp_path / "out" / "qa.json"

    assert ok is True
    assert output_file.exists()
    assert json.loads(output_file.read_text(encoding="utf-8")) == qa_pairs
