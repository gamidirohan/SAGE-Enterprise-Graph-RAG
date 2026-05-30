from app import policy_guard


def test_policy_guard_flags_multi_item_queries_with_insufficient_coverage():
    verdict = policy_guard.evaluate_answer(
        query="Who are the various managers we have?",
        answer="Bijade is your manager.",
        answer_payload={"evidence_refs": ["chunk:chunk-1"]},
        trace={
            "query_profile": {
                "expects_multiple_items": True,
                "minimum_unique_evidence": 2,
            },
            "coverage": {
                "distinct_evidence_count": 1,
            },
            "evidence": [{"chunk_id": "chunk-1"}],
        },
        plan={"query_profile": {"expects_multiple_items": True, "minimum_unique_evidence": 2}},
    )

    assert verdict["passed"] is False
    assert "insufficient_answer_coverage" in verdict["issues"]


def test_policy_guard_rejects_temporal_lookup_answer_that_omits_available_start_date():
    verdict = policy_guard.evaluate_answer(
        query="From when will Charlie start working on Project Proton?",
        answer="Charlie is assigned to Project Proton.",
        answer_payload={
            "summary": "Charlie is assigned to Project Proton.",
            "bullets": [],
            "evidence_refs": ["fact:fact-proton"],
        },
        trace={
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "fact_id": "fact-proton",
                    "fact": {
                        "claim_type": "ASSIGNMENT_STATE",
                        "temporal_start": "2026-05-12",
                    },
                    "document": {"doc_id": "chat-msg-proton"},
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is False
    assert "missing_required_answer_slot:temporal_start" in verdict["issues"]


def test_policy_guard_rejects_temporal_lookup_answer_when_source_text_has_start_date():
    verdict = policy_guard.evaluate_answer(
        query="From when will Charlie start working on Project Proton?",
        answer="Charlie is assigned to Project Proton.",
        answer_payload={
            "summary": "Charlie is assigned to Project Proton.",
            "bullets": [],
            "evidence_refs": ["fact:fact-proton"],
        },
        trace={
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "fact_id": "fact-proton",
                    "fact": {
                        "claim_type": "ASSIGNMENT_STATE",
                    },
                    "document": {
                        "doc_id": "chat-msg-proton",
                        "content": "Hi Charlie, I'll be you manager for project proton, and you'll work on it for 2 months starting tomorrow",
                    },
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is False
    assert "missing_required_answer_slot:temporal_start" in verdict["issues"]


def test_policy_guard_passes_temporal_lookup_answer_with_available_start_date():
    verdict = policy_guard.evaluate_answer(
        query="From when will Charlie start working on Project Proton?",
        answer="Charlie starts working on Project Proton on 2026-05-12.",
        answer_payload={
            "summary": "Charlie starts working on Project Proton on 2026-05-12.",
            "bullets": [],
            "evidence_refs": ["fact:fact-proton"],
        },
        trace={
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "fact_id": "fact-proton",
                    "fact": {
                        "claim_type": "ASSIGNMENT_STATE",
                        "temporal_start": "2026-05-12",
                    },
                    "document": {"doc_id": "chat-msg-proton"},
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is True
    assert "missing_required_answer_slot:temporal_start" not in verdict["issues"]


def test_policy_guard_rejects_duration_lookup_answer_that_omits_available_end_date():
    verdict = policy_guard.evaluate_answer(
        query="How long will Charlie work on Project Proton?",
        answer="Charlie starts working on Project Proton on 2026-05-12.",
        answer_payload={
            "summary": "Charlie starts working on Project Proton on 2026-05-12.",
            "bullets": [],
            "evidence_refs": ["fact:fact-proton"],
        },
        trace={
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "fact_id": "fact-proton",
                    "fact": {
                        "claim_type": "ASSIGNMENT_STATE",
                        "temporal_start": "2026-05-12",
                        "temporal_end": "2026-07-12",
                    },
                    "document": {"doc_id": "chat-msg-proton"},
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is False
    assert "missing_required_answer_slot:temporal_end" in verdict["issues"]


def test_policy_guard_accepts_duration_lookup_answer_with_source_duration():
    verdict = policy_guard.evaluate_answer(
        query="How long is Charlie going to work, and in which project",
        answer="Charlie will work on Project Proton for 2 months.",
        answer_payload={
            "summary": "Charlie will work on Project Proton for 2 months.",
            "bullets": [],
            "evidence_refs": ["fact:fact-proton"],
        },
        trace={
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "fact_id": "fact-proton",
                    "fact": {
                        "claim_type": "ASSIGNMENT_STATE",
                        "temporal_start": "2026-05-12",
                        "temporal_end": "2026-07-12",
                    },
                    "document": {
                        "doc_id": "chat-msg-proton",
                        "content": "Hi Charlie, I'll be you manager for project proton, and you'll work on it for 2 months starting tomorrow",
                    },
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is True
    assert "missing_required_answer_slot:temporal_end" not in verdict["issues"]


def test_policy_guard_rejects_manager_lookup_answer_backed_by_assignment_relation():
    verdict = policy_guard.evaluate_answer(
        query="Who is Charlie's Manager now?",
        answer="Charlie is assigned to Project Proton.",
        answer_payload={
            "summary": "Charlie is assigned to Project Proton.",
            "bullets": [],
            "evidence_refs": ["fact:fact-assignment"],
        },
        trace={
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "fact_id": "fact-assignment",
                    "fact": {"claim_type": "ASSIGNMENT_STATE"},
                    "document": {"doc_id": "chat-msg-assignment"},
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is False
    assert "missing_required_answer_relation:reports_to" in verdict["issues"]


def test_policy_guard_accepts_manager_lookup_answer_backed_by_reports_to_relation():
    verdict = policy_guard.evaluate_answer(
        query="Who is Charlie's Manager now?",
        answer="Charlie reports to Alice Manager.",
        answer_payload={
            "summary": "Charlie reports to Alice Manager.",
            "bullets": [],
            "evidence_refs": ["fact:fact-manager"],
        },
        trace={
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "fact_id": "fact-manager",
                    "fact": {"claim_type": "REPORTS_TO"},
                    "document": {"doc_id": "chat-msg-manager"},
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is True
    assert "missing_required_answer_relation:reports_to" not in verdict["issues"]


def test_policy_guard_does_not_require_reports_to_for_manager_of_project_lookup():
    verdict = policy_guard.evaluate_answer(
        query="Who was the manager of Project Alpha in 2022?",
        answer="Bob Smith was the manager of Project Alpha in 2022.",
        answer_payload={
            "summary": "Bob Smith was the manager of Project Alpha in 2022.",
            "bullets": [],
            "evidence_refs": ["chunk:chunk-role"],
        },
        trace={
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "chunk_id": "chunk-role",
                    "chunk_summary": "Bob Smith was the manager of Project Alpha in 2022.",
                    "document": {"doc_id": "chat-msg-role"},
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is True


def test_policy_guard_rejects_direct_attribute_answer_from_unfocused_evidence():
    verdict = policy_guard.evaluate_answer(
        query="What is the office address for HQ?",
        answer="Test Alice is busy.",
        answer_payload={
            "summary": "Test Alice is busy.",
            "bullets": [],
            "evidence_refs": ["fact:fact-busy"],
        },
        trace={
            "query_type": "general_search",
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "fact_id": "fact-busy",
                    "fact_summary": "Test Alice is busy.",
                    "fact": {"claim_type": "STATUS", "display_summary": "Test Alice is busy."},
                    "document": {"doc_id": "chat-msg-busy", "content": "Test Alice is busy."},
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is False
    assert "unfocused_evidence_for_direct_lookup" in verdict["issues"]
    assert "missing_required_answer_slot:address_or_location" in verdict["issues"]


def test_policy_guard_accepts_direct_attribute_answer_from_focused_evidence():
    verdict = policy_guard.evaluate_answer(
        query="What is the office address for HQ?",
        answer="HQ is located at 123 Enterprise Way, Suite 400.",
        answer_payload={
            "summary": "HQ is located at 123 Enterprise Way, Suite 400.",
            "bullets": [],
            "evidence_refs": ["chunk:chunk-hq"],
        },
        trace={
            "query_type": "general_search",
            "query_profile": {"expects_multiple_items": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "chunk_id": "chunk-hq",
                    "chunk_summary": "HQ is located at 123 Enterprise Way, Suite 400.",
                    "document": {"doc_id": "chat-msg-hq", "content": "HQ is located at 123 Enterprise Way, Suite 400."},
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False}},
    )

    assert verdict["passed"] is True
    assert "missing_required_answer_relation:reports_to" not in verdict["issues"]


def test_policy_guard_rejects_weak_focus_task_evidence_for_direct_deadline_lookup():
    verdict = policy_guard.evaluate_answer(
        query="By when does george has to submit the project alpha documents",
        answer="Fresh Alice 950 starts working on Project Alpha on 9pm.",
        answer_payload={
            "summary": "Fresh Alice 950 starts working on Project Alpha on 9pm.",
            "bullets": [],
            "evidence_refs": ["fact:fact-assignment"],
        },
        trace={
            "query_type": "task_commitment_lookup",
            "query_profile": {"expects_multiple_items": False, "requires_broad_coverage": False},
            "coverage": {"distinct_evidence_count": 1},
            "evidence": [
                {
                    "fact_id": "fact-assignment",
                    "fact_summary": "Fresh Alice 950 starts working on Project Alpha on 9pm.",
                    "fact": {"claim_type": "ASSIGNMENT_STATE", "display_summary": "Fresh Alice 950 starts working on Project Alpha on 9pm."},
                    "document": {"doc_id": "chat-msg-assignment", "content": "Fresh Alice 950 starts working on Project Alpha on 9pm."},
                }
            ],
        },
        plan={"query_profile": {"expects_multiple_items": False, "requires_broad_coverage": False}},
    )

    assert verdict["passed"] is False
    assert "unfocused_evidence_for_direct_lookup" in verdict["issues"]
