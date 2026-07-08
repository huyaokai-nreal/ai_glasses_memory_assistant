from __future__ import annotations

import tempfile

from ai_glasses_memory_assistant.memory_store import EventMemoryStore
from ai_glasses_memory_assistant.timeline_store import TimelineStore
from tests.helpers import isolated_app_home


def test_memory_store_is_user_scoped_and_searchable() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        store = EventMemoryStore()
        mine = store.add_memory("u1", "周五检查 demo", kind="event", memory_type="task", tags=["auto"])
        store.add_memory("u2", "周五检查 demo", kind="event", memory_type="task")

        assert [memory.id for memory in store.list_memories("u1")] == [mine.id]
        assert [memory.id for memory in store.search("u1", "demo")] == [mine.id]
        assert store.search("u2", "demo")[0].user_id == "u2"


def test_timeline_keeps_raw_evidence_redacted_and_searchable() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        store = TimelineStore()
        result = store.add_turn(
            "u1",
            "请记一下项目要上线，api_key=sk-abcdefghijklmnopqrstuvwxyz123456",
            created_at=1.0,
        )
        updated = store.update_turn_reply("u1", result.turn.id, "已经记下", legacy_session_id="s1", updated_at=2.0)
        chunks = store.search_chunks("u1", "项目")

        assert updated is not None
        assert updated.assistant_reply == "已经记下"
        assert result.redaction is not None and result.redaction.redacted
        assert "sk-" not in result.turn.raw_text
        assert chunks and chunks[0].parent_id == result.turn.id


def test_memory_jobs_survive_service_restart_boundary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        timeline = TimelineStore()
        timeline.upsert_memory_job(
            "u1",
            "job1",
            {"job_id": "job1", "user_id": "u1", "status": "pending", "memory_processing": {"status": "pending"}},
        )
        reloaded = TimelineStore()

        assert reloaded.get_memory_job("u1", "job1")["status"] == "pending"
        assert reloaded.get_memory_job("u2", "job1") is None
