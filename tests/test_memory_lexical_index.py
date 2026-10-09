"""Focused lexical/index regressions; all stores use a disposable app home."""
from __future__ import annotations

import json

import pytest

from ai_glasses_memory_assistant import memory_store as memory_store_module
from ai_glasses_memory_assistant.memory_store import EventMemoryStore
from tests.helpers import CoreChatService
from ai_glasses_memory_assistant.temporal_parser import TemporalResolution


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GLASSES_HOME", str(tmp_path / "home"))
    result = EventMemoryStore(db_path=tmp_path / "events.db")
    yield result
    result._conn.close()


def fts_ids(store, query):
    # MATCH observes the actual inverted index, unlike external-content SELECT.
    return {
        row["id"] for row in store._conn.execute(
            "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid "
            "WHERE memories_fts MATCH ?", (store._fts_query(query),)
        )
    }


@pytest.mark.parametrize("term", ["会议", "巡检"])
def test_partial_fts_hits_include_chinese_substrings_and_dedupe(store, term):
    exact = store.add_memory("u1", f"{term} 预算", created_at=100)
    embedded = store.add_memory("u1", f"项目{term}纪要需要确认预算", created_at=200)
    assert fts_ids(store, term) == {exact.id}
    assert {row["id"] for row in store._search_like("u1", term, 20)} == {exact.id, embedded.id}
    result = store.search_with_ranking("u1", term, limit=5)
    assert {memory.id for memory in result.memories} == {exact.id, embedded.id}
    assert len(result.candidate_ranking) == 2
    assert [memory.id for memory in store.search("u1", term)] == [memory.id for memory in result.memories]


def test_merged_candidates_preserve_scope_active_filter_and_ranking(store, monkeypatch):
    monkeypatch.setattr(memory_store_module.time, "time", lambda: 1000.0)
    subject = store.create_named_subject("u1", "张三")
    exact = store.add_memory("u1", "会议 预算", subject_id=subject.id, created_at=100)
    embedded = store.add_memory("u1", "项目会议纪要", subject_id=subject.id, created_at=900)
    for content in ["会议 其他", "项目会议纪要其他"]:
        store.add_memory("u2", content, created_at=999)
        store.add_memory("u1", content, created_at=999)
        store.add_memory("u1", content, subject_id=subject.id, status="stale", created_at=999)
        store.add_memory("u1", content, subject_id=subject.id, status="superseded", created_at=999)
        deleted = store.add_memory("u1", content, subject_id=subject.id, created_at=999)
        store.delete_memory("u1", deleted.id)
    result = store.search_with_ranking("u1", "会议", subject_ids=[subject.id], limit=1)
    # The newer LIKE-only record competes in the same ranking, rather than being appended.
    assert [memory.id for memory in result.memories] == [embedded.id]
    assert {item["id"] for item in result.candidate_ranking} == {exact.id, embedded.id}
    assert store.search("u1", "会议", subject_ids=[]) == []
    foreign = store.ensure_self_subject("u2")
    assert store.search("u1", "会议", subject_ids=[foreign.id]) == []


@pytest.mark.parametrize("limit,expected", [(0, 1), (2, 2), (100, 20)])
def test_merged_candidates_keep_bounded_pools_and_output(store, limit, expected):
    for index in range(90):
        # Disjoint FTS and LIKE pools when the query has a second English term.
        store.add_memory("u1", f"项目会议纪要{index}", created_at=200 + index)
        store.add_memory("u1", f"anchor unrelated {index}", created_at=100 + index)
    result = store.search_with_ranking("u1", "会议 anchor", limit=limit)
    assert len(result.memories) == expected
    ids = [item["id"] for item in result.candidate_ranking]
    assert len(ids) == len(set(ids))
    assert len(ids) <= 8 * expected
    assert any("会议" in memory.content for memory in result.memories)
    assert [item["total_score"] for item in result.ranking] == sorted(
        [item["total_score"] for item in result.ranking], reverse=True
    )


def test_like_still_works_without_fts_and_for_zero_hits(store):
    embedded = store.add_memory("u1", "项目会议纪要")
    assert fts_ids(store, "会议") == set()
    assert [item.id for item in store.search("u1", "会议")] == [embedded.id]
    store._conn.execute("DROP TABLE memories_fts")
    assert [item.id for item in store.search("u1", "会议")] == [embedded.id]


def test_access_metadata_and_same_value_updates_do_not_write_fts(store):
    memory = store.add_memory("u1", "meeting notes", tags=["budget"])
    before = store._conn.total_changes
    accessed = store.record_memory_access("u1", [memory.id], accessed_at=123)
    assert accessed[0].access_count == 1
    assert accessed[0].last_accessed_at == 123
    assert accessed[0].strength >= memory.strength
    assert store._conn.total_changes - before == 1
    before = store._conn.total_changes
    store._conn.execute(
        "UPDATE memories SET content=content, tags=tags, deleted_at=deleted_at WHERE id=?",
        (memory.id,),
    )
    store._conn.commit()
    assert store._conn.total_changes - before == 1
    assert fts_ids(store, "meeting") == {memory.id}
    assert fts_ids(store, "budget") == {memory.id}


@pytest.mark.parametrize("field", ["content", "tags"])
def test_fts_tracks_each_indexed_field_independently(store, field):
    memory = store.add_memory("u1", "oldword", tags=["oldtag"])
    old_term, new_term, new_value = (
        ("oldword", "newword", "newword") if field == "content"
        else ("oldtag", "newtag", json.dumps(["newtag"]))
    )
    store._conn.execute(f"UPDATE memories SET {field}=? WHERE id=?", (new_value, memory.id))
    store._conn.commit()
    assert not fts_ids(store, old_term)
    assert fts_ids(store, new_term) == {memory.id}
    unchanged_term = "oldtag" if field == "content" else "oldword"
    assert fts_ids(store, unchanged_term) == {memory.id}


def test_status_changes_keep_index_and_filter_active_results(store):
    memory = store.add_memory("u1", "meeting notes")
    before = store._conn.total_changes
    assert store.mark_stale("u1", memory.id)
    assert store._conn.total_changes - before == 1
    assert fts_ids(store, "meeting") == {memory.id}
    assert store.search("u1", "meeting") == []


def test_fts_tracks_content_tags_delete_restore_and_physical_delete(store):
    memory = store.add_memory("u1", "oldword", tags=["oldtag"])
    store._conn.execute(
        "UPDATE memories SET content=?, tags=? WHERE id=?",
        ("newword", json.dumps(["newtag"]), memory.id),
    )
    store._conn.commit()
    assert not fts_ids(store, "oldword oldtag")
    assert fts_ids(store, "newword newtag") == {memory.id}
    assert store.delete_memory("u1", memory.id)
    assert not fts_ids(store, "newword newtag")
    # Deleted rows can change text without reappearing; restore indexes the new text.
    store._conn.execute("UPDATE memories SET content='restoredword' WHERE id=?", (memory.id,))
    assert not fts_ids(store, "restoredword")
    store._conn.execute(
        "UPDATE memories SET deleted_at=NULL, status='active' WHERE id=?", (memory.id,)
    )
    store._conn.commit()
    assert fts_ids(store, "restoredword") == {memory.id}
    assert fts_ids(store, "newtag") == {memory.id}
    store.purge_memory("u1", memory.id)
    assert not fts_ids(store, "restoredword newtag")


def test_reopening_replaces_legacy_update_trigger_without_rebuild(store):
    memory = store.add_memory("u1", "meeting notes")
    store._conn.executescript("""
        DROP TRIGGER memories_fts_update;
        CREATE TRIGGER memories_fts_update AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            SELECT 'delete', old.rowid, old.content, old.tags WHERE old.deleted_at IS NULL;
            INSERT INTO memories_fts(rowid, content, tags)
            SELECT new.rowid, new.content, new.tags WHERE new.deleted_at IS NULL;
        END;
    """)
    store._conn.close()
    reopened = EventMemoryStore(db_path=store.db_path)
    try:
        before = reopened._conn.total_changes
        reopened.record_memory_access("u1", [memory.id])
        assert reopened._conn.total_changes - before == 1
        assert fts_ids(reopened, "meeting") == {memory.id}
    finally:
        reopened._conn.close()


def test_service_text_search_consumes_both_lexical_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GLASSES_HOME", str(tmp_path / "home"))
    service = CoreChatService(str(tmp_path / "home"))
    exact = service.memory_store.add_memory("u1", "会议 预算")
    embedded = service.memory_store.add_memory("u1", "项目会议纪要需要确认预算")
    memories, debug = service._recall_event_memories(
        user_id="u1", message="会议", temporal=TemporalResolution(),
        reference_time=1000.0, strategy="text_search", subject_ids=[exact.subject_id],
    )
    assert {item.id for item in memories} == {exact.id, embedded.id}
    assert debug["lexical_fallback_used"] is False
    assert debug["recency_supplement_used"] is False
    assert debug["candidate_trace"]["candidate_count"] == 2


@pytest.mark.parametrize("term,embedded", [("tea", "team"), ("art", "cart")])
@pytest.mark.parametrize("fts_available", [True, False])
@pytest.mark.parametrize("distractor_count", [5, 25])
def test_english_internal_substrings_cannot_crowd_out_exact_word(
    store, term, embedded, fts_available, distractor_count
):
    exact = store.add_memory("u1", f"I prefer green {term}.", created_at=100)
    for index in range(distractor_count):
        store.add_memory("u1", f"The {embedded} meeting is scheduled for Friday.", created_at=200 + index)
    if not fts_available:
        store._conn.execute("DROP TABLE memories_fts")
    result = store.search_with_ranking("u1", term, limit=5)
    assert [memory.id for memory in result.memories] == [exact.id]
    assert [item["id"] for item in result.candidate_ranking] == [exact.id]
    assert result.ranking[0]["text_score"] == 1


@pytest.mark.parametrize("fts_available", [True, False])
def test_english_whole_words_match_case_punctuation_and_tags(store, fts_available):
    exact = store.add_memory("u1", "A cup of TEA, please.")
    tagged = store.add_memory("u1", "A beverage preference.", tags=["tea"])
    store.add_memory("u1", "The team meets.", tags=["teapot"])
    store.add_memory("u1", "tea42 pretea teaSuffix are identifiers.")
    if not fts_available:
        store._conn.execute("DROP TABLE memories_fts")
    result = store.search_with_ranking("u1", "tea", limit=5)
    assert {exact.id, tagged.id}.issubset({memory.id for memory in result.memories})
    assert all(item["text_score"] == 1 for item in result.ranking if item["id"] in {exact.id, tagged.id})
    assert all("team" not in memory.content for memory in result.memories)


@pytest.mark.parametrize("fts_available", [True, False])
@pytest.mark.parametrize("query", ["会议 tea", "会议tea"])
def test_mixed_queries_score_english_words_and_preserve_chinese_substrings(
    store, monkeypatch, fts_available, query
):
    monkeypatch.setattr(memory_store_module.time, "time", lambda: 1000.0)
    exact = store.add_memory("u1", "项目会议tea备忘", created_at=100)
    for index in range(25):
        store.add_memory("u1", "项目会议team纪要", created_at=900 + index)
    if not fts_available:
        store._conn.execute("DROP TABLE memories_fts")
    result = store.search_with_ranking("u1", query, limit=5)
    assert result.memories[0].id == exact.id
    scores = {item["id"]: item["text_score"] for item in result.candidate_ranking}
    assert scores[exact.id] == 1
    assert all(score < 1 for memory_id, score in scores.items() if memory_id != exact.id)


def test_english_fallback_respects_user_subject_and_active_filters(store):
    subject = store.create_named_subject("u1", "张三")
    exact = store.add_memory("u1", "I prefer tea.", subject_id=subject.id, created_at=100)
    store.add_memory("u2", "tea", created_at=999)
    store.add_memory("u1", "tea", created_at=999)
    store.add_memory("u1", "tea", subject_id=subject.id, status="stale", created_at=999)
    store.add_memory("u1", "tea", subject_id=subject.id, status="superseded", created_at=999)
    deleted = store.add_memory("u1", "tea", subject_id=subject.id, created_at=999)
    store.delete_memory("u1", deleted.id)
    for index in range(25):
        store.add_memory("u1", "The team meets.", subject_id=subject.id, created_at=900 + index)
    store._conn.execute("DROP TABLE memories_fts")
    assert [memory.id for memory in store.search("u1", "tea", limit=5, subject_ids=[subject.id])] == [exact.id]
