"""Truncation must retain the existing final scorer's best lexical evidence."""
from __future__ import annotations

import random
import sqlite3
import pytest
from ai_glasses_memory_assistant.memory_store import EventMemoryStore

NOW = 1778131200.0


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GLASSES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("ai_glasses_memory_assistant.memory_store.time.time", lambda: NOW)
    result = EventMemoryStore(tmp_path / "events.db")
    yield result
    result._conn.close()


@pytest.mark.parametrize("query", ["alder receipt", "勘验", "复核 titanium"])
@pytest.mark.parametrize("without_fts", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_early_budget_retains_old_evidence_selected_by_existing_score(store, query, without_fts, reverse):
    rows = [(query + " supporting detail "*30, NOW-365*86400, True)]
    rows += [(query + f" item {i}", NOW-100-i, False) for i in range(70)]
    target = None
    for content, stamp, gold in reversed(rows) if reverse else rows:
        memory = store.add_memory("u1", content, created_at=stamp, occurred_at=NOW if gold else None)
        if gold:
            target = memory
            store._conn.execute("UPDATE memories SET strength=.88, access_count=10, last_accessed_at=? WHERE id=?", (NOW, memory.id))
    store._conn.commit()
    if without_fts:
        store._conn.execute("DROP TABLE memories_fts")
    # All fixture rows meet the same lexical qualification; full set is eval-only.
    eligible = store.list_memories("u1", kind="event", exclude_memory_types=("observation",))
    reference = store._rank_memory_search_results(eligible, query, 8)
    assert reference.memories[0].id == target.id
    bounded = store.search_with_ranking("u1", query, limit=8, kind="event", exclude_memory_types=("observation",))
    assert [m.id for m in bounded.memories] == [m.id for m in reference.memories]
    assert {m.id for m in reference.memories} <= {r["id"] for r in bounded.candidate_ranking}
    assert len(bounded.candidate_ranking) <= 32


@pytest.mark.parametrize("without_fts", [False, True])
def test_candidate_order_matches_scorer_over_varied_metadata_and_tags(store, without_fts):
    rng = random.Random(7649)
    for i in range(90):
        memory = store.add_memory(
            "u1", "cedar evidence " + str(i), tags=["复验"],
            memory_type=rng.choice(["event", "task", "decision", "preference", "project_state"]),
            created_at=rng.choice([-100., 0., NOW-400*86400, NOW-2000]) + i,
            occurred_at=rng.choice([None, 0., NOW-200*86400, NOW-100]),
        )
        store._conn.execute("UPDATE memories SET strength=?,access_count=?,last_accessed_at=? WHERE id=?", (rng.uniform(-.2,1.2),rng.randrange(20),rng.choice([None,0.,NOW-200*86400,NOW-500]),memory.id))
    store._conn.commit()
    if without_fts:
        store._conn.execute("DROP TABLE memories_fts")
    for query in ("cedar", "复验", "cedar 复验"):
        # A scoring reference must use the SAME lexical universe, not all active rows.
        rows = {r["id"]: r for r in store._search_like("u1", query, 1000)}
        try:
            fts = store._conn.execute(
                "SELECT m.* FROM memories m WHERE user_id=? AND rowid IN "
                "(SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?)",
                ("u1", store._fts_query(query)),
            ).fetchall()
            rows.update({r["id"]: r for r in fts})
        except sqlite3.OperationalError:
            pass
        eligible = [store._row_to_event(r) for r in rows.values()]
        reference = store._rank_memory_search_results(eligible, query, 20)
        result = store.search_with_ranking("u1", query, limit=20)
        assert [m.id for m in result.memories] == [m.id for m in reference.memories]
        assert result.ranking == reference.ranking


def test_exact_ties_use_insertion_order_without_changes_to_scores(store):
    memories = [store.add_memory("u1", "anchor evidence", created_at=NOW) for _ in range(10)]
    result = store.search_with_ranking("u1", "anchor", limit=2)
    assert [m.id for m in result.memories] == [m.id for m in memories[:2]]
    assert result.ranking[0]["total_score"] == result.ranking[1]["total_score"]
