"""Contract eligibility must be applied before lexical candidate budgets."""
from __future__ import annotations

import pytest

from ai_glasses_memory_assistant.memory_store import EventMemoryStore
from ai_glasses_memory_assistant.temporal_parser import TemporalResolution
from tests.helpers import CoreChatService, FakeAgent, pre_reply_recall

NOW = 1778131200.0


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GLASSES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("ai_glasses_memory_assistant.memory_store.time.time", lambda: NOW)
    result = EventMemoryStore(db_path=tmp_path / "events.db")
    yield result
    result._conn.close()


@pytest.mark.parametrize("query,content", [
    ("oak receipt", "Archived oak receipt, reference R-581."),
    ("清点", "已归档白桦清点凭据R-581。"),
    ("审计 copper", "项目审计纪要 copper R-581."),
])
@pytest.mark.parametrize("interference", ["profile", "assistant_preference", "observation"])
@pytest.mark.parametrize("without_fts", [False, True])
def test_contract_eligibility_precedes_each_candidate_limit(store, query, content, interference, without_fts):
    target = store.add_memory("u1", content, created_at=NOW-365*86400)
    for index in range(45):
        store.add_memory(
            "u1", query + " interests", created_at=NOW-index,
            kind="event" if interference == "observation" else interference,
            memory_type="observation" if interference == "observation" else "preference",
        )
    if without_fts:
        store._conn.execute("DROP TABLE memories_fts")
    result = store.search_with_ranking(
        "u1", query, limit=3, kind="event", exclude_memory_types=("observation",),
    )
    assert [memory.id for memory in result.memories] == [target.id]
    assert [row["id"] for row in result.candidate_ranking] == [target.id]
    # The management/capsule default remains unrestricted by kind/type.
    assert store.search_with_ranking("u1", query, limit=3).memories
    assert target.id not in {m.id for m in store.search("u1", query, limit=3)}


@pytest.mark.parametrize("query", ["anchor", " "])
def test_scoped_query_preserves_user_subject_and_lifecycle(store, query):
    named = store.create_named_subject("u1", "林澈")
    provisional = store.create_provisional_subject("u1", "speaker", source_scope="capture-test")
    target = store.add_memory("u1", "anchor older", subject_id=named.id, created_at=NOW-100)
    temp = store.add_memory("u1", "anchor temporary", subject_id=provisional.id)
    for state in ("stale", "superseded", "deleted", "other_user", "self", "observation", "profile"):
        memory = store.add_memory(
            "u2" if state == "other_user" else "u1", "anchor newer " + state,
            subject_id=None if state in {"other_user", "self"} else named.id,
            status=state if state in {"stale", "superseded"} else "active",
            kind="profile" if state == "profile" else "event",
            memory_type="observation" if state == "observation" else "event",
            created_at=NOW,
        )
        if state == "deleted":
            store.delete_memory(memory.user_id, memory.id)
    kwargs = {"kind": "event", "exclude_memory_types": ("observation",), "limit": 1}
    assert [m.id for m in store.search("u1", query, subject_ids=[named.id], **kwargs)] == [target.id]
    assert [m.id for m in store.search("u1", query, subject_ids=[provisional.id], **kwargs)] == [temp.id]
    assert store.search("u1", query, subject_ids=[], **kwargs) == []
    assert store.search("u1", query, subject_ids=[store.ensure_self_subject("u2").id], **kwargs) == []


@pytest.mark.parametrize("limit,expected", [(0, 1), (3, 3), (100, 20)])
def test_eligible_pool_keeps_limits_dedup_and_existing_ranking(store, limit, expected):
    for index in range(50):
        store.add_memory("u1", "anchor 会议 " + str(index), created_at=NOW-index)
        store.add_memory("u1", "anchor 会议 profile " + str(index), kind="profile", created_at=NOW+index)
    result = store.search_with_ranking("u1", "anchor 会议", limit=limit, kind="event", exclude_memory_types=("observation",))
    assert len(result.memories) == expected
    ids = [r["id"] for r in result.candidate_ranking]
    assert len(ids) == len(set(ids)) <= expected*8
    assert all(m.kind == "event" for m in result.memories)
    assert [r["total_score"] for r in result.ranking] == sorted((r["total_score"] for r in result.ranking), reverse=True)


@pytest.mark.parametrize("query,content", [("pine docket", "Filed pine docket D-613."), ("校验", "已归档银杉校验记录D-613。")])
def test_authorized_service_evidence_reaches_actual_main_input(tmp_path, monkeypatch, query, content):
    monkeypatch.setenv("AI_GLASSES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("ai_glasses_memory_assistant.memory_store.time.time", lambda: NOW)
    agent = FakeAgent(pre_reply=pre_reply_recall())
    service = CoreChatService(str(tmp_path), agent=agent)
    store = service.memory_store
    target = store.add_memory("u1", content, created_at=NOW-365*86400)
    for index in range(40):
        store.add_memory("u1", query + " interests", kind="profile", created_at=NOW-200-index)
    for index in range(60):
        store.add_memory("u1", f"Unrelated archive note {index}", created_at=NOW-index)
    response = service.chat(query, user_id="u1")
    assert response["debug"]["routing"]["pre_reply_decision_applied"]
    assert response["debug"]["planner"]["event_recall_strategy"] == "text_search"
    assert target.id in {row["id"] for row in response["recalled_memories"]}
    assert any(target.content in call["message"] for call in agent.calls if not call["system_message"])
    assert sum("unified pre-reply decision classifier" in (call["system_message"] or "") for call in agent.calls) == 1


def test_empty_and_skipped_contracts_do_not_manufacture_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GLASSES_HOME", str(tmp_path / "home"))
    service = CoreChatService(str(tmp_path), agent=FakeAgent(pre_reply=pre_reply_recall()))
    response = service.chat("pine docket", user_id="u1")
    assert response["recalled_memories"] == []
    service.memory_store.add_memory("u1", "pine docket evidence")
    events, trace = service._recall_event_memories(
        user_id="u1", message="pine docket", temporal=TemporalResolution(), reference_time=NOW,
        strategy="skipped",
    )
    assert events == []
    assert trace["candidate_trace"]["candidate_count"] == 0
