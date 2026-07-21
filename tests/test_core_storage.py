from __future__ import annotations

import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

import pytest

from ai_glasses_memory_assistant import memory_store as memory_store_module
from ai_glasses_memory_assistant.memory_store import (
    EventMemoryStore,
    event_to_dict,
    voice_profile_to_dict,
)
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


def test_legacy_memories_are_migrated_to_each_users_self_subject() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'event',
                memory_type TEXT NOT NULL DEFAULT 'event',
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL DEFAULT 'chat',
                source_id TEXT NOT NULL DEFAULT '',
                ingestion_id TEXT NOT NULL DEFAULT '',
                evidence_ids TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL DEFAULT 0,
                occurred_at REAL,
                start_at REAL,
                end_at REAL,
                time_granularity TEXT NOT NULL DEFAULT 'unknown',
                temporal_text TEXT NOT NULL DEFAULT '',
                temporal_confidence REAL,
                privacy_level TEXT NOT NULL DEFAULT 'normal',
                status TEXT NOT NULL DEFAULT 'active',
                confidence REAL,
                superseded_by TEXT NOT NULL DEFAULT '',
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed_at REAL,
                strength REAL NOT NULL DEFAULT 0,
                deleted_at REAL
            );
            INSERT INTO memories (
                id, user_id, content, created_at, updated_at, status
            ) VALUES
                ('m1', 'u1', '旧记忆一', 1, 1, 'active'),
                ('m2', 'u2', '旧记忆二', 2, 2, 'active');
            """
        )
        conn.commit()
        conn.close()

        store = EventMemoryStore(db_path=db_path)
        first = store.list_memories("u1")[0]
        second = store.list_memories("u2")[0]

        assert first.subject_type == "self" and first.subject_name == "我"
        assert second.subject_type == "self" and second.subject_name == "我"
        assert first.subject_id != second.subject_id
        assert [subject.id for subject in store.list_subjects("u1")] == [first.subject_id]

        store._conn.execute(
            """
            UPDATE memories
            SET subject_id = 'missing-subject', subject_type = 'named', subject_name = '错误人物'
            WHERE id = 'm1'
            """
        )
        store._conn.commit()
        store._conn.close()

        reopened = EventMemoryStore(db_path=db_path)
        repaired = reopened.get_memory("u1", "m1")
        assert repaired.subject_id == first.subject_id
        assert repaired.subject_type == "self" and repaired.subject_name == "我"
        assert [subject.id for subject in reopened.list_subjects("u1")] == [first.subject_id]


def test_memories_and_search_can_be_scoped_to_independent_subjects() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        store = EventMemoryStore()
        self_memory = store.add_memory("u1", "我喜欢矿泉水", kind="profile", memory_type="preference")
        zhang = store.create_named_subject("u1", "张三")
        li = store.create_named_subject("u1", "李四")
        zhang_memory = store.add_memory(
            "u1",
            "喜欢苏打水",
            subject_id=zhang.id,
            kind="profile",
            memory_type="preference",
        )
        li_memory = store.add_memory(
            "u1",
            "喜欢茶",
            subject_id=li.id,
            kind="profile",
            memory_type="preference",
        )
        zhang_event = store.add_memory(
            "u1",
            "周五开会",
            subject_id=zhang.id,
            kind="event",
            memory_type="task",
            start_at=10.0,
        )
        li_untimed_event = store.add_memory(
            "u1",
            "准备材料",
            subject_id=li.id,
            kind="event",
            memory_type="task",
        )

        assert [
            item.id
            for item in store.list_memories("u1", kind="profile", subject_ids=[zhang.id])
        ] == [zhang_memory.id]
        assert [item.id for item in store.search("u1", "喜欢", subject_ids=[li.id])] == [li_memory.id]
        assert store.find_similar_memory(
            "u1",
            zhang_memory.content,
            kind="profile",
            memory_type="preference",
            subject_id=li.id,
        ) is None
        assert store.mark_superseded("u1", zhang_memory.id, li_memory.id) is False
        assert store.get_memory("u1", zhang_memory.id, subject_ids=[li.id]) is None
        assert store.get_memory("u1", zhang_memory.id, subject_ids=[zhang.id]).id == zhang_memory.id
        assert store.find_similar_memory(
            "u1",
            self_memory.content,
            kind="profile",
            memory_type="preference",
        ).id == self_memory.id
        assert [item.id for item in store.list_events_between(
            "u1", 0.0, 20.0, subject_ids=[zhang.id]
        )] == [zhang_event.id]
        assert [item.id for item in store.list_recent_untimed_events(
            "u1", subject_ids=[li.id]
        )] == [li_untimed_event.id]
        assert event_to_dict(zhang_memory)["subject_name"] == "张三"


def test_provisional_subjects_are_isolated_by_capture_scope() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        store = EventMemoryStore()
        first = store.create_provisional_subject("u1", "speaker_2", source_scope="capture-1")
        repeated = store.create_provisional_subject("u1", "speaker_2", source_scope="capture-1")
        assert store.resolve_subject(
            "u1",
            "speaker_2",
            subject_types={"provisional"},
            include_scoped_aliases=True,
        ).id == first.id
        second = store.create_provisional_subject("u1", "speaker_2", source_scope="capture-2")

        assert repeated.id == first.id
        assert second.id != first.id
        assert store.resolve_subject("u1", "speaker_2", source_scope="capture-1").id == first.id
        assert store.resolve_subject("u1", "speaker_2", source_scope="capture-2").id == second.id
        assert store.resolve_subject("u1", "speaker_2") is None
        assert store.resolve_subject(
            "u1",
            "speaker_2",
            subject_types={"provisional"},
            include_scoped_aliases=True,
        ) is None


def test_concurrent_subject_creation_reuses_one_self_and_named_subject() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        store = EventMemoryStore()

        def create_many(factory):
            barrier = Barrier(8)

            def create_once(_index):
                barrier.wait()
                return factory()

            with ThreadPoolExecutor(max_workers=8) as executor:
                return list(executor.map(create_once, range(8)))

        self_subjects = create_many(lambda: store.ensure_self_subject("u1"))
        named_subjects = create_many(lambda: store.create_named_subject("u1", "Alex"))

        assert len({subject.id for subject in self_subjects}) == 1
        assert len({subject.id for subject in named_subjects}) == 1
        assert [(subject.subject_type, subject.display_name) for subject in store.list_subjects("u1")] == [
            ("self", "我"),
            ("named", "Alex"),
        ]


def test_fts_rebuild_only_runs_when_external_content_tables_are_created(
    tmp_path,
    monkeypatch,
) -> None:
    statements: list[str] = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(memory_store_module.sqlite3, "connect", traced_connect)
    db_path = tmp_path / "fts.db"
    first = EventMemoryStore(db_path=db_path)
    first.add_memory("u1", "initial searchable memory")
    first._conn.close()

    rebuild_sql = "VALUES('rebuild')"
    assert sum(rebuild_sql in statement for statement in statements) == 2
    statements.clear()

    reopened = EventMemoryStore(db_path=db_path)
    assert not any(rebuild_sql in statement for statement in statements)
    assert [memory.content for memory in reopened.search("u1", "searchable")] == [
        "initial searchable memory"
    ]
    reopened._conn.close()


def test_subject_merge_moves_memories_aliases_and_voice_profiles_without_serializing_embedding() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        store = EventMemoryStore()
        provisional = store.create_provisional_subject("u1", "speaker_2", source_scope="capture-1")
        named = store.create_named_subject("u1", "张三")
        memory = store.add_memory(
            "u1",
            "喜欢苏打水",
            subject_id=provisional.id,
            kind="profile",
            memory_type="preference",
        )
        timed_event = store.add_memory(
            "u1",
            "周五参加评审",
            subject_id=provisional.id,
            kind="event",
            memory_type="task",
            start_at=10.0,
        )
        untimed_event = store.add_memory(
            "u1",
            "准备评审材料",
            subject_id=provisional.id,
            kind="event",
            memory_type="task",
        )
        conflicting_source_profile = store.store_voice_profile(
            "u1",
            provisional.id,
            embedding=[0.1, 0.2, 0.3],
            embedding_model="campplus",
            confidence=0.9,
            source_id="segment-1",
            created_at=2.0,
        )
        movable_profile = store.store_voice_profile(
            "u1",
            provisional.id,
            embedding=[0.2, 0.3, 0.4],
            embedding_model="campplus",
            confidence=0.8,
            source_id="segment-2",
            created_at=3.0,
        )
        target_profile = store.store_voice_profile(
            "u1",
            named.id,
            embedding=[0.9, 0.1, 0.1],
            embedding_model="campplus",
            confidence=0.95,
            source_id="segment-1",
            created_at=1.0,
        )
        repeated_target_profile = store.store_voice_profile(
            "u1",
            named.id,
            embedding=[0.8, 0.2, 0.1],
            embedding_model="campplus",
            confidence=0.7,
            source_id="segment-1",
        )
        store.add_subject_alias("u1", named.id, "老张")

        merged = store.merge_subjects("u1", provisional.id, named.id)
        refreshed = store.get_memory("u1", memory.id)
        profiles = store.list_voice_profiles(
            "u1",
            subject_ids=[named.id],
            embedding_model="campplus",
            embedding_dim=3,
        )

        assert merged.id == named.id
        assert refreshed.subject_id == named.id and refreshed.subject_name == "张三"
        assert store.resolve_subject("u1", "speaker_2", source_scope="capture-1").id == named.id
        assert store.resolve_subject("u1", "老张").id == named.id
        assert store.create_provisional_subject(
            "u1", "speaker_2", source_scope="capture-1"
        ).id == named.id
        assert [item.id for item in store.list_memories(
            "u1", subject_ids=[provisional.id]
        )] == [untimed_event.id, timed_event.id, memory.id]
        assert [item.id for item in store.search(
            "u1", "喜欢苏打水", subject_ids=[provisional.id]
        )] == [memory.id]
        assert [item.id for item in store.list_events_between(
            "u1", 0.0, 20.0, subject_ids=[provisional.id]
        )] == [timed_event.id]
        assert [item.id for item in store.list_recent_untimed_events(
            "u1", subject_ids=[provisional.id]
        )] == [untimed_event.id]
        assert store.get_memory("u1", memory.id, subject_ids=[provisional.id]).id == memory.id

        profiles_by_source = {item.source_id: item for item in profiles}
        assert repeated_target_profile.id == target_profile.id
        assert set(profiles_by_source) == {"segment-1", "segment-2"}
        assert profiles_by_source["segment-1"].id == target_profile.id
        assert profiles_by_source["segment-1"].embedding == (0.9, 0.1, 0.1)
        assert profiles_by_source["segment-2"].id == movable_profile.id
        assert all(item.subject_id == named.id for item in profiles)
        assert conflicting_source_profile.id not in {item.id for item in profiles}

        serialized_profile = voice_profile_to_dict(target_profile)
        assert set(serialized_profile) == {
            "id",
            "user_id",
            "subject_id",
            "embedding_model",
            "embedding_dim",
            "confidence",
            "source_id",
            "created_at",
        }

        assert store.merge_subjects("u1", provisional.id, named.id).id == named.id
        other = store.create_named_subject("u1", "李四")
        with pytest.raises(ValueError, match="already been merged"):
            store.merge_subjects("u1", provisional.id, other.id)
        assert store.get_memory("u1", memory.id).subject_id == named.id


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


def test_device_audio_event_queue_is_idempotent_and_recovers_running(tmp_path) -> None:
    store = TimelineStore(db_path=tmp_path / "timeline.db")
    event = {
        "schema_version": "audio_event.v1",
        "event_id": "event-1",
        "audio_session_id": "session-1",
        "segment_id": "segment-1",
        "type": "transcript_final",
        "lane": "ambient",
        "source_type": "ambient_audio",
        "start_ms": 0,
        "end_ms": 1000,
        "text": "明天提交材料",
        "final": True,
        "vad": {},
        "wake": {},
        "asr": {},
        "speaker": {"state": "user"},
        "overlap": {"state": "not_observed"},
        "audio_retention": "discarded_after_processing",
    }

    created = store.enqueue_device_audio_event(
        user_id="u1",
        event=event,
        capture_id="capture-1",
        private_payload={"speaker_embedding": [1.0, 0.0]},
    )
    duplicate = store.enqueue_device_audio_event(user_id="u1", event=event, capture_id="capture-other")

    assert created["created"] is True
    assert duplicate["created"] is False
    assert duplicate["capture_id"] == "capture-1"
    claimed = store.claim_next_device_audio_event("u1")
    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["attempt_count"] == 1
    assert store.recover_running_device_audio_events() == 1
    assert store.get_device_audio_event("u1", "event-1")["status"] == "pending"

    claimed_again = store.claim_next_device_audio_event("u1")
    assert claimed_again is not None
    failed = store.update_device_audio_event(
        user_id="u1",
        event_id="event-1",
        status="failed",
        error_type="ConnectionError",
    )
    assert failed is not None
    assert failed["attempt_count"] == 2
    assert store.device_audio_event_summary("u1")["failed"] == 1
    assert store.retry_failed_device_audio_events("u1", "event-1") == 1
    assert store.get_device_audio_event("u1", "event-1")["status"] == "pending"
