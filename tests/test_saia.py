import json

from app import saia


class _Result:
    def __init__(self, rows=None):
        self._rows = rows or []

    def data(self):
        return self._rows


class _Session:
    def __init__(self, handler=None):
        self.handler = handler or (lambda _query, _params, _calls: [])
        self.calls = []

    def run(self, query, **params):
        self.calls.append({"query": query, "params": params})
        return _Result(self.handler(query, params, self.calls))


class _Ctx:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class _Driver:
    def __init__(self, session):
        self._session = session
        self.closed = False

    def close(self):
        self.closed = True


class _Node(dict):
    def items(self):
        return super().items()


class _FakeNameSession:
    def run(self, query, **params):
        if "MATCH (p:Person)" not in query:
            return _Result([])
        value = params.get("value")
        rows = {
            "bob": [{"id": "bob-id", "labels": ["Person"]}],
            "alice": [{"id": "alice-id", "labels": ["Person"]}],
            "charlie": [{"id": "charlie-id", "labels": ["Person"]}],
        }.get(str(value).lower(), [])
        return _Result(rows)


def _patch_saia_runtime(monkeypatch, session):
    driver = _Driver(session)
    monkeypatch.setenv("SAIA_ENABLED", "true")
    monkeypatch.setattr(saia.utils, "create_neo4j_driver", lambda: driver)
    monkeypatch.setattr(saia.utils, "open_neo4j_session", lambda _driver, _database: _Ctx(session))
    monkeypatch.setattr(saia.utils, "generate_embedding", lambda _text: [0.1, 0.2])
    return driver


def test_normalize_temporal_reference_resolves_next_monday():
    result = saia.normalize_temporal_reference("next Monday", "2026-04-01T10:00:00Z")

    assert result["temporal_start"] == "2026-04-06"
    assert result["temporal_granularity"] == "date"


def test_extract_claims_from_text_resolves_direct_chat_commitment():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m1",
        source_message_id="m1",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("I'll send you the report tomorrow.", context)

    task_claims = [claim for claim in claims if claim["claim_type"] == "TASK_ASSIGNMENT"]
    assert len(task_claims) == 1
    claim = task_claims[0]
    assert claim["subject_entity_id"] == "u1"
    assert claim["object_entity_id"] == "u2"
    assert claim["resolution_status"] == "resolved"
    assert claim["temporal_start"] == "2026-04-02"
    assert "u1 will send report to u2" in claim["normalized_text"]


def test_extract_claims_from_text_supports_progressive_commitment_with_grounding():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m1b",
        source_message_id="m1b",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("I'll be sharing the report to you by 8pm today.", context)

    task_claims = [claim for claim in claims if claim["claim_type"] == "TASK_ASSIGNMENT"]
    assert len(task_claims) == 1
    claim = task_claims[0]
    grounding = json.loads(claim["grounding_json"])
    assert claim["subject_raw"] == "I"
    assert claim["subject_entity_id"] == "u1"
    assert claim["object_raw"] == "you"
    assert claim["object_entity_id"] == "u2"
    assert claim["value_text"] == "share report"
    assert claim["temporal_start"] == "2026-04-01T20:00:00+00:00"
    assert claim["normalized_text"] == "u1 will share report to u2 on 2026-04-01T20:00:00+00:00"
    assert grounding["temporal_expressions"] == ["8pm", "today"]
    assert grounding["references"][0]["raw"] == "I"
    assert grounding["references"][1]["raw"] == "you"


def test_resolve_reference_maps_we_and_us_to_group_scope():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-group-1",
        source_message_id="group-1",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2", "u3"],
        conversation_id="group:g1",
        conversation_type="group",
        group_id="group1",
        sent_at="2026-04-01T10:00:00Z",
    )

    we_resolution = saia._resolve_reference("we", context, allow_pronouns=True)
    us_resolution = saia._resolve_reference("us", context, allow_pronouns=True)

    assert we_resolution.status == "resolved"
    assert we_resolution.entity_id == "group1"
    assert us_resolution.status == "resolved"
    assert us_resolution.entity_id == "group1"


def test_resolve_reference_leaves_it_unresolved_until_coreference_support_exists():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-direct-1",
        source_message_id="direct-1",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
    )

    resolution = saia._resolve_reference("it", context, allow_pronouns=True)

    assert resolution.status == "unresolved"
    assert resolution.entity_id is None
    assert resolution.key is None


def test_extract_claims_from_text_canonicalizes_email_sender_and_receiver_to_person_ids():
    class _EmailIdentitySession:
        def run(self, query, **params):
            if "MATCH (p:Person)" not in query:
                return _Result([])
            value = str(params.get("value") or "").lower()
            if value == "alice@example.com":
                return _Result([{"id": "1", "labels": ["Person"], "display_name": "Alice Johnson"}])
            if value == "bob@example.com":
                return _Result([{"id": "2", "labels": ["Person"], "display_name": "Bob Smith"}])
            return _Result([])

    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-email-1",
        source_message_id="email-1",
        linked_message_id=None,
        sender_id="alice@example.com",
        receiver_ids=["bob@example.com"],
        conversation_id="direct:alice@example.com:bob@example.com",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("I'll send you the report tomorrow.", context, session=_EmailIdentitySession())

    task_claims = [claim for claim in claims if claim["claim_type"] == "TASK_ASSIGNMENT"]
    assert len(task_claims) == 1
    claim = task_claims[0]
    assert claim["subject_entity_id"] == "1"
    assert claim["object_entity_id"] == "2"
    assert "Alice Johnson will send report to Bob Smith" in claim["normalized_text"]


def test_extract_claims_from_text_merges_contextual_follow_up_fragment():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m1c",
        source_message_id="m1c",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("I'll be sharing the report to you by 8pm today. For Project Alpha.", context)

    task_claims = [claim for claim in claims if claim["claim_type"] == "TASK_ASSIGNMENT"]
    assert len(task_claims) == 1
    claim = task_claims[0]
    payload = json.loads(claim["payload_json"])
    assert claim["source_span_text"] == "I'll be sharing the report to you by 8pm today; For Project Alpha"
    assert claim["value_text"] == "share report for Project Alpha"
    assert payload["item"] == "report for Project Alpha"
    assert payload["task_signature"] == "share-report-for-project-alpha"
    assert payload["context_fragments"] == ["for Project Alpha"]
    assert payload["recipient_relation"] == "to"


def test_extract_claims_from_text_strips_temporal_correction_marker_from_commitment():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m1d",
        source_message_id="m1d",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-13T13:23:17Z",
    )

    claims = saia.extract_claims_from_text(
        "Correction: I'll send you the Project Alpha budget by 9pm tomorrow instead.",
        context,
    )

    task_claims = [claim for claim in claims if claim["claim_type"] == "TASK_ASSIGNMENT"]
    assert len(task_claims) == 1
    claim = task_claims[0]
    payload = json.loads(claim["payload_json"])
    assert claim["value_text"] == "send Project Alpha budget"
    assert claim["normalized_text"] == "u1 will send Project Alpha budget to u2 on 2026-04-14T21:00:00+00:00"
    assert payload["item"] == "Project Alpha budget"
    assert payload["task_signature"] == "send-project-alpha-budget"


def test_extract_claims_from_text_group_request_stays_noncanonical():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m2",
        source_message_id="m2",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2", "u3"],
        conversation_id="group:g1",
        conversation_type="group",
        group_id="g1",
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("Can you send the file?", context)

    assert len(claims) == 1
    claim = claims[0]
    assert claim["claim_type"] == "REQUEST"
    assert claim["object_entity_id"] is None
    assert claim["resolution_status"] == "unresolved"
    assert saia.should_promote_claim(claim) is False


def test_extract_claims_from_text_reports_to_uses_person_lookup():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m3",
        source_message_id="m3",
        linked_message_id=None,
        sender_id="u9",
        receiver_ids=["u8"],
        conversation_id="direct:u8:u9",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("Bob now reports to Alice.", context, session=_FakeNameSession())

    assert len(claims) == 1
    claim = claims[0]
    assert claim["claim_type"] == "REPORTS_TO"
    assert claim["subject_entity_id"] == "bob-id"
    assert claim["object_entity_id"] == "alice-id"
    assert claim["resolution_status"] == "resolved"


def test_extract_claims_from_text_reports_to_prefers_exact_user_person_match_over_duplicate_name():
    class _DuplicateNameSession:
        def run(self, query, **params):
            if "MATCH (p:Person)" not in query:
                return _Result([])
            value = str(params.get("value")).lower()
            if value == "george brown":
                return _Result(
                    [
                        {
                            "id": "external-george",
                            "labels": ["Person"],
                            "name": "George Brown",
                            "email": "",
                            "display_name": "George Brown",
                        },
                        {
                            "id": "7",
                            "labels": ["User", "Person"],
                            "name": "George Brown",
                            "email": "george@example.com",
                            "display_name": "George Brown",
                        },
                    ]
                )
            if value == "vinitha":
                return _Result(
                    [
                        {
                            "id": "vinitha-id",
                            "labels": ["User", "Person"],
                            "name": "Vinitha",
                            "email": "vinitha@example.com",
                            "display_name": "Vinitha",
                        }
                    ]
                )
            return _Result([])

    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m3b",
        source_message_id="m3b",
        linked_message_id=None,
        sender_id="u9",
        receiver_ids=["u8"],
        conversation_id="direct:u8:u9",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-13T13:37:07Z",
    )

    claims = saia.extract_claims_from_text(
        "George Brown now reports to Vinitha.",
        context,
        session=_DuplicateNameSession(),
    )

    assert len(claims) == 1
    claim = claims[0]
    assert claim["claim_type"] == "REPORTS_TO"
    assert claim["subject_entity_id"] == "7"
    assert claim["object_entity_id"] == "vinitha-id"
    assert claim["resolution_status"] == "resolved"


def test_process_chat_message_promotes_direct_commitment_to_canonical_fact(monkeypatch):
    session = _Session()
    driver = _patch_saia_runtime(monkeypatch, session)

    result = saia.process_chat_message(
        message_id="m1",
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
        content="I'll send you the report tomorrow.",
    )

    assert result["status"] == "completed"
    assert result["claims_extracted"] >= 1
    assert result["claims_canonicalized"] == 1
    assert driver.closed is True

    fact_writes = [
        call
        for call in session.calls
        if "SET f.canonical_key = $canonical_key" in call["query"]
        and call["params"].get("canonical_key") == "assignment::direct:u1:u2::send-report"
    ]
    assert len(fact_writes) == 1
    fact_params = fact_writes[0]["params"]
    assert fact_params["canonical_key"] == "assignment::direct:u1:u2::send-report"
    assert fact_params["subject_entity_id"] == "u1"
    assert fact_params["object_entity_id"] == "u2"
    assert fact_params["temporal_start"] == "2026-04-02"


def test_process_chat_message_confirms_existing_fact_without_creating_duplicate(monkeypatch):
    def handler(query, params, _calls):
        if "MATCH (f:CanonicalFact {canonical_key: $canonical_key, status: 'current'}) RETURN f" in query:
            return [
                {
                    "f": _Node(
                        {
                            "fact_id": "fact-existing",
                            "canonical_key": "assignment::direct:u1:u2::send-report",
                            "claim_type": "TASK_ASSIGNMENT",
                            "predicate": "TASK_COMMITMENT",
                            "subject_entity_id": "u1",
                            "subject_key": "u1",
                            "object_entity_id": "u2",
                            "object_key": "u2",
                            "value_text": "send report",
                            "temporal_start": "2026-04-02",
                            "temporal_granularity": "date",
                        }
                    )
                }
            ]
        return []

    session = _Session(handler=handler)
    _patch_saia_runtime(monkeypatch, session)

    result = saia.process_chat_message(
        message_id="m1-repeat",
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
        content="I'll send you the report tomorrow.",
    )

    assert result["status"] == "completed"
    assert result["claims_canonicalized"] == 1
    assert not any("MERGE (f:CanonicalFact {fact_id: $fact_id})" in call["query"] for call in session.calls)
    assert any("SUPPORTS" in call["query"] and call["params"].get("fact_id") == "fact-existing" for call in session.calls)
    assert any("support_count = coalesce(f.support_count, 0) + 1" in call["query"] and call["params"].get("fact_id") == "fact-existing" for call in session.calls)


def test_process_chat_message_group_request_creates_claim_but_no_canonical_fact(monkeypatch):
    session = _Session()
    _patch_saia_runtime(monkeypatch, session)

    result = saia.process_chat_message(
        message_id="m2",
        sender_id="u1",
        receiver_ids=["u2", "u3"],
        conversation_id="group:g1",
        conversation_type="group",
        group_id="g1",
        sent_at="2026-04-01T10:00:00Z",
        content="Can you send the file?",
    )

    assert result["status"] == "completed"
    assert result["claims_extracted"] == 1
    assert result["claims_canonicalized"] == 0
    assert any("MERGE (c:Claim {claim_id: $claim_id})" in call["query"] for call in session.calls)
    assert not any("MERGE (f:CanonicalFact {fact_id: $fact_id})" in call["query"] for call in session.calls)


def test_process_chat_message_supersedes_conflicting_reports_to_fact(monkeypatch):
    def handler(query, params, _calls):
        if "MATCH (p:Person)" in query:
            rows = {
                "bob": [{"id": "bob-id", "labels": ["Person"]}],
                "charlie": [{"id": "charlie-id", "labels": ["Person"]}],
            }.get(str(params.get("value")).lower(), [])
            return rows
        if "MATCH (f:CanonicalFact {canonical_key: $canonical_key, status: 'current'}) RETURN f" in query:
            return [
                {
                    "f": _Node(
                        {
                            "fact_id": "fact-old",
                            "claim_type": "REPORTS_TO",
                            "predicate": "REPORTS_TO",
                            "subject_entity_id": "bob-id",
                            "subject_key": "bob-id",
                            "object_entity_id": "alice-id",
                            "object_key": "alice-id",
                            "value_text": None,
                            "temporal_start": None,
                            "temporal_granularity": "unresolved",
                        }
                    )
                }
            ]
        return []

    session = _Session(handler=handler)
    _patch_saia_runtime(monkeypatch, session)

    result = saia.process_chat_message(
        message_id="m3",
        sender_id="u9",
        receiver_ids=["u8"],
        conversation_id="direct:u8:u9",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
        content="Bob now reports to Charlie.",
    )

    assert result["status"] == "completed"
    assert result["claims_canonicalized"] == 1
    assert result["conflicts_found"] == 1
    assert any("SUPERSEDED_BY" in call["query"] for call in session.calls)
    assert any(call["params"].get("existing_fact_id") == "fact-old" for call in session.calls if "existing_fact_id" in call["params"])


def test_process_message_attachment_links_provenance_without_rewriting_raw_content(monkeypatch):
    def handler(query, params, _calls):
        if "MATCH (p:Person)" in query:
            rows = {
                "bob": [{"id": "bob-id", "labels": ["Person"]}],
                "alice": [{"id": "alice-id", "labels": ["Person"]}],
            }.get(str(params.get("value")).lower(), [])
            return rows
        return []

    session = _Session(handler=handler)
    _patch_saia_runtime(monkeypatch, session)

    result = saia.process_message_attachment(
        doc_id="message-attachment-m7",
        linked_message_id="m7",
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
        content="Bob now reports to Alice.",
        attachment_name="org-update.txt",
    )

    assert result["status"] == "completed"
    assert result["claims_canonicalized"] == 1
    assert any("HAS_EVIDENCE_DOCUMENT" in call["query"] for call in session.calls)
    assert all("SET d.content" not in call["query"] for call in session.calls)
    assert all("SET m.content" not in call["query"] for call in session.calls)
    assert all("SET c.content" not in call["query"] for call in session.calls)


def test_extract_claims_from_text_group_meeting_supports_plain_hour():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m4",
        source_message_id="m4",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2", "u3"],
        conversation_id="group:g1",
        conversation_type="group",
        group_id="g1",
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("We have a meeting tomorrow at 10.", context)

    meeting_claims = [claim for claim in claims if claim["claim_type"] == "MEETING_EVENT"]
    assert len(meeting_claims) == 1
    claim = meeting_claims[0]
    assert claim["subject_entity_id"] == "g1"
    assert claim["temporal_start"] == "2026-04-02T10:00:00+00:00"
    assert claim["normalized_text"] == "meeting scheduled for 2026-04-02T10:00:00+00:00"


def test_extract_claims_from_text_supports_meet_verb_with_relative_day():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m5",
        source_message_id="m5",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("Let's meet next Monday.", context)

    meeting_claims = [claim for claim in claims if claim["claim_type"] == "MEETING_EVENT"]
    assert len(meeting_claims) == 1
    assert meeting_claims[0]["temporal_start"] == "2026-04-06"


def test_extract_claims_from_text_supports_generic_service_status():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m6",
        source_message_id="m6",
        linked_message_id=None,
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("The server is down.", context)

    status_claims = [claim for claim in claims if claim["claim_type"] == "STATUS_UPDATE"]
    assert len(status_claims) == 1
    claim = status_claims[0]
    assert claim["subject_key"] == "server"
    assert claim["value_text"] == "down"
    assert saia.should_promote_claim(claim) is True


def test_extract_claims_from_text_supports_assignment_state():
    context = saia.GroundingContext(
        source_kind="chat_message",
        source_doc_id="chat-msg-m7",
        source_message_id="m7",
        linked_message_id=None,
        sender_id="u9",
        receiver_ids=["u8"],
        conversation_id="direct:u8:u9",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
    )

    claims = saia.extract_claims_from_text("Bob is assigned to Project A.", context, session=_FakeNameSession())

    assignment_claims = [claim for claim in claims if claim["claim_type"] == "ASSIGNMENT_STATE"]
    assert len(assignment_claims) == 1
    claim = assignment_claims[0]
    assert claim["subject_entity_id"] == "bob-id"
    assert claim["object_key"] == "project-a"
    assert claim["value_text"] == "active"


def test_process_message_attachment_passive_approval_promotes_canonical_fact(monkeypatch):
    session = _Session()
    _patch_saia_runtime(monkeypatch, session)

    result = saia.process_message_attachment(
        doc_id="message-attachment-m8",
        linked_message_id="m8",
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
        content="The budget is approved.",
        attachment_name="approval.txt",
    )

    assert result["status"] == "completed"
    assert result["claims_canonicalized"] == 1
    fact_writes = [
        call
        for call in session.calls
        if "SET f.canonical_key = $canonical_key" in call["query"]
        and call["params"].get("canonical_key") == "approval::budget"
    ]
    assert len(fact_writes) == 1
    assert fact_writes[0]["params"]["subject_key"] == "budget"
    assert fact_writes[0]["params"]["value_text"] == "approved"


def test_process_chat_message_assignment_end_supersedes_current_fact(monkeypatch):
    def handler(query, params, _calls):
        if "MATCH (p:Person)" in query:
            rows = {
                "bob": [{"id": "bob-id", "labels": ["Person"]}],
            }.get(str(params.get("value")).lower(), [])
            return rows
        if "MATCH (f:CanonicalFact {canonical_key: $canonical_key, status: 'current'}) RETURN f" in query:
            return [
                {
                    "f": _Node(
                        {
                            "fact_id": "fact-assignment-old",
                            "claim_type": "ASSIGNMENT_STATE",
                            "predicate": "ASSIGNED_TO",
                            "subject_entity_id": "bob-id",
                            "subject_key": "bob-id",
                            "object_entity_id": None,
                            "object_key": "project-a",
                            "value_text": "active",
                            "temporal_start": None,
                            "temporal_granularity": "unresolved",
                        }
                    )
                }
            ]
        return []

    session = _Session(handler=handler)
    _patch_saia_runtime(monkeypatch, session)

    result = saia.process_chat_message(
        message_id="m9",
        sender_id="u9",
        receiver_ids=["u8"],
        conversation_id="direct:u8:u9",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
        content="Bob is no longer working on Project A.",
    )

    assert result["status"] == "completed"
    assert result["claims_canonicalized"] == 1
    assert result["conflicts_found"] == 1
    assert any("SUPERSEDED_BY" in call["query"] for call in session.calls)


def test_process_chat_message_corrected_commitment_supersedes_existing_fact(monkeypatch):
    def handler(query, params, _calls):
        if "MATCH (f:CanonicalFact {canonical_key: $canonical_key, status: 'current'}) RETURN f" in query:
            return [
                {
                    "f": _Node(
                        {
                            "fact_id": "fact-budget-old",
                            "canonical_key": "assignment::direct:u1:u2::send-project-alpha-budget",
                            "claim_type": "TASK_ASSIGNMENT",
                            "predicate": "TASK_COMMITMENT",
                            "subject_entity_id": "u1",
                            "subject_key": "u1",
                            "object_entity_id": "u2",
                            "object_key": "u2",
                            "value_text": "send Project Alpha budget",
                            "temporal_start": "2026-04-14T18:00:00+00:00",
                            "temporal_granularity": "datetime",
                        }
                    )
                }
            ]
        return []

    session = _Session(handler=handler)
    _patch_saia_runtime(monkeypatch, session)

    result = saia.process_chat_message(
        message_id="m-budget-correction",
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-13T13:23:17Z",
        content="Correction: I'll send you the Project Alpha budget by 9pm tomorrow instead.",
    )

    assert result["status"] == "completed"
    assert result["claims_canonicalized"] == 1
    assert result["conflicts_found"] == 1
    fact_writes = [
        call
        for call in session.calls
        if "SET f.canonical_key = $canonical_key" in call["query"]
        and call["params"].get("canonical_key") == "assignment::direct:u1:u2::send-project-alpha-budget"
    ]
    assert len(fact_writes) == 1
    assert fact_writes[0]["params"]["value_text"] == "send Project Alpha budget"
    assert fact_writes[0]["params"]["temporal_start"] == "2026-04-14T21:00:00+00:00"
    assert any("SUPERSEDED_BY" in call["query"] for call in session.calls)


def test_process_chat_message_records_mutation_action_for_new_fact(monkeypatch):
    session = _Session()
    _patch_saia_runtime(monkeypatch, session)

    result = saia.process_chat_message(
        message_id="m10",
        sender_id="u1",
        receiver_ids=["u2"],
        conversation_id="direct:u1:u2",
        conversation_type="direct",
        group_id=None,
        sent_at="2026-04-01T10:00:00Z",
        content="I'll send you the report tomorrow.",
    )

    assert result["status"] == "completed"
    mutation_updates = [
        call
        for call in session.calls
        if "c.mutation_action = $mutation_action" in call["query"]
    ]
    assert any(call["params"].get("mutation_action") == "insert_new_fact" for call in mutation_updates)


def test_collect_message_insight_returns_preview_claims_for_skipped_message(monkeypatch):
    monkeypatch.setenv("SAIA_ENABLED", "true")

    def handler(query, params, _calls):
        if "MATCH (m:Message {id: $message_id})" in query and "RETURN m" in query:
            return [
                {
                    "m": _Node(
                        {
                            "id": "m-preview",
                            "content": "I'll be sharing the report to you by 8pm today.",
                            "source": "chat_message",
                            "saia_status": "skipped",
                            "saia_processed_at": "2026-04-12T08:29:10.192913+00:00",
                            "sender_id": "u1",
                            "receiver_id": "u2",
                            "conversation_id": "direct:u1:u2",
                            "conversation_type": "direct",
                            "group_id": None,
                            "sent_at": "2026-04-12T08:29:06.790Z",
                            "is_ai_response": False,
                        }
                    )
                }
            ]
        if "UNWIND $entity_ids AS entity_id" in query:
            display_names = {
                "u1": "Alice",
                "u2": "Bob",
            }
            return [
                {"entity_id": entity_id, "display_name": display_names.get(entity_id, entity_id)}
                for entity_id in params.get("entity_ids", [])
            ]
        if "OPTIONAL MATCH (u:User {id: $entity_id})" in query:
            display_names = {
                "u1": "Alice",
                "u2": "Bob",
            }
            entity_id = params.get("entity_id")
            return [{"display_name": display_names.get(entity_id, entity_id)}]
        return []

    insight = saia.collect_message_insight(_Session(handler=handler), "m-preview")

    assert insight["saia_status"] == "skipped"
    assert insight["claims"] == []
    assert len(insight["preview_claims"]) == 1
    preview_claim = insight["preview_claims"][0]
    assert preview_claim["preview_only"] is True
    assert preview_claim["normalized_text"] == "Alice will share report to Bob on 2026-04-12T20:00:00+00:00"
    assert preview_claim["display_text"] == "Alice will share report to Bob on 2026-04-12T20:00:00+00:00"
    assert preview_claim["grounding"]["temporal_expressions"] == ["8pm", "today"]
    assert preview_claim["grounding"]["references"][0]["raw"] == "I"
    assert preview_claim["grounding"]["references"][1]["raw"] == "you"
    assert preview_claim["grounding"]["references"][0]["display_name"] == "Alice"
    assert preview_claim["grounding"]["references"][1]["display_name"] == "Bob"
    assert insight["summary"]["preview_claim_count"] == 1


def test_collect_message_insight_returns_preview_claims_when_saia_is_disabled(monkeypatch):
    monkeypatch.delenv("SAIA_ENABLED", raising=False)

    def handler(query, params, _calls):
        if "MATCH (m:Message {id: $message_id})" in query and "RETURN m" in query:
            return [
                {
                    "m": _Node(
                        {
                            "id": "m-disabled",
                            "content": "I'll send you the report tomorrow.",
                            "source": "chat_message",
                            "sender_id": "u1",
                            "receiver_id": "u2",
                            "conversation_id": "direct:u1:u2",
                            "conversation_type": "direct",
                            "group_id": None,
                            "sent_at": "2026-04-12T08:29:06.790Z",
                            "is_ai_response": False,
                        }
                    )
                }
            ]
        if "UNWIND $entity_ids AS entity_id" in query:
            display_names = {
                "u1": "Alice",
                "u2": "Bob",
            }
            return [
                {"entity_id": entity_id, "display_name": display_names.get(entity_id, entity_id)}
                for entity_id in params.get("entity_ids", [])
            ]
        if "OPTIONAL MATCH (u:User {id: $entity_id})" in query:
            display_names = {
                "u1": "Alice",
                "u2": "Bob",
            }
            entity_id = params.get("entity_id")
            return [{"display_name": display_names.get(entity_id, entity_id)}]
        return []

    insight = saia.collect_message_insight(_Session(handler=handler), "m-disabled")

    assert insight["saia_status"] == "disabled"
    assert insight["saia_error"] == "SAIA processing is disabled in backend configuration."
    assert insight["claims"] == []
    assert len(insight["preview_claims"]) == 1
    assert insight["preview_claims"][0]["preview_only"] is True
