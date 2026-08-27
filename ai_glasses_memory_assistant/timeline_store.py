from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .app_home import get_data_dir
from .privacy_filter import RedactionResult, redact_sensitive_text
from .evidence_set import PagedSourceResult, decode_page_cursor, encode_page_cursor


# 原始时间线与结构化记忆同目录存储，但保持独立表，避免污染 event/profile 召回。
def default_data_dir() -> Path:
    return get_data_dir()


@dataclass(frozen=True)
class TimelineTurn:
    id: str
    user_id: str
    source: str
    raw_text: str
    assistant_reply: str
    interaction_id: str
    legacy_session_id: str
    created_at: float
    updated_at: float
    status: str = "active"


@dataclass(frozen=True)
class TimelineChunk:
    id: str
    user_id: str
    parent_type: str
    parent_id: str
    text: str
    chunk_index: int
    start_offset: int
    end_offset: int
    timestamp: float
    source: str
    metadata: dict[str, Any]
    status: str = "active"


@dataclass(frozen=True)
class TimelineWriteResult:
    turn: TimelineTurn
    chunks: list[TimelineChunk]
    redaction: RedactionResult | None = None


@dataclass(frozen=True)
class TimelineSearchResult:
    chunks: list[TimelineChunk]
    ranking: list[dict[str, Any]]
    candidate_ranking: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TimelinePurgeResult:
    purged_chunk_ids: list[str]
    purged_parent_ids: list[str]

    @property
    def purged_chunk_count(self) -> int:
        return len(self.purged_chunk_ids)

    @property
    def purged_parent_count(self) -> int:
        return len(self.purged_parent_ids)


@dataclass(frozen=True)
class TimelineChunkReference:
    chunk: TimelineChunk
    active_refs: int = 0
    retained_refs: int = 0


@dataclass(frozen=True)
class ReplyFeedback:
    id: str
    user_id: str
    turn_id: str
    rating: str
    note: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class SpeakerProfileRecord:
    user_id: str
    embedding: list[float]
    model_name: str
    source: str
    created_at: float
    updated_at: float
    sample_count: int = 1
    target_sample_count: int = 3
    calibration_status: str = "single_sample"
    profile_version: int = 1
    user_threshold: float | None = None
    other_threshold: float | None = None


@dataclass(frozen=True)
class SpeakerEnrollmentSampleRecord:
    user_id: str
    enrollment_session_id: str
    sample_index: int
    sample_total: int
    embedding: list[float]
    model_name: str
    source: str
    created_at: float
    updated_at: float


class TimelineStore:
    """User-scoped raw timeline store for full-text recall and evidence links."""

    def __init__(self, db_path: Path | None = None) -> None:
        data_dir = default_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (data_dir / "timeline.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # schema 使用独立 timeline/chunk 表，避免把大段原话塞进结构化 memories。
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_turns (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'chat',
                    raw_text TEXT NOT NULL,
                    assistant_reply TEXT NOT NULL DEFAULT '',
                    interaction_id TEXT NOT NULL DEFAULT '',
                    legacy_session_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    deleted_at REAL
                );
                CREATE TABLE IF NOT EXISTS captures (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'continuous_capture',
                    context TEXT NOT NULL DEFAULT '',
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'running',
                    deleted_at REAL
                );
                CREATE TABLE IF NOT EXISTS memory_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    deleted_at REAL
                );
                CREATE TABLE IF NOT EXISTS device_audio_events (
                    user_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    audio_session_id TEXT NOT NULL,
                    capture_id TEXT NOT NULL DEFAULT '',
                    event_payload TEXT NOT NULL,
                    private_payload TEXT NOT NULL DEFAULT '{}',
                    dispatch_payload TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    error_type TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    PRIMARY KEY(user_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    parent_type TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    start_offset INTEGER NOT NULL DEFAULT 0,
                    end_offset INTEGER NOT NULL DEFAULT 0,
                    timestamp REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    deleted_at REAL
                );
                CREATE TABLE IF NOT EXISTS discussion_slices (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    capture_id TEXT NOT NULL,
                    day_key TEXT NOT NULL,
                    first_chunk_id TEXT NOT NULL,
                    last_chunk_id TEXT NOT NULL,
                    start_at REAL NOT NULL,
                    end_at REAL NOT NULL,
                    chunk_ids TEXT NOT NULL DEFAULT '[]',
                    summary_payload TEXT NOT NULL DEFAULT '{}',
                    backend TEXT NOT NULL DEFAULT '',
                    error_type TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, capture_id, first_chunk_id, last_chunk_id)
                );
                CREATE TABLE IF NOT EXISTS discussion_topics (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    day_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    topic_key TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    key_points TEXT NOT NULL DEFAULT '[]',
                    decisions TEXT NOT NULL DEFAULT '[]',
                    tasks TEXT NOT NULL DEFAULT '[]',
                    open_questions TEXT NOT NULL DEFAULT '[]',
                    participant_labels TEXT NOT NULL DEFAULT '[]',
                    time_spans TEXT NOT NULL DEFAULT '[]',
                    slice_ids TEXT NOT NULL DEFAULT '[]',
                    evidence_ids TEXT NOT NULL DEFAULT '[]',
                    start_at REAL NOT NULL,
                    end_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deleted_at REAL
                );
                CREATE TABLE IF NOT EXISTS discussion_days (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    day_key TEXT NOT NULL,
                    overview TEXT NOT NULL DEFAULT '',
                    topic_ids TEXT NOT NULL DEFAULT '[]',
                    evidence_ids TEXT NOT NULL DEFAULT '[]',
                    topic_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ready',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deleted_at REAL,
                    UNIQUE(user_id, day_key)
                );
                CREATE TABLE IF NOT EXISTS reply_feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, turn_id)
                );
                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    user_id TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    model_name TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'campp_reference_enrollment',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 1,
                    target_sample_count INTEGER NOT NULL DEFAULT 3,
                    calibration_status TEXT NOT NULL DEFAULT 'single_sample',
                    profile_version INTEGER NOT NULL DEFAULT 1,
                    user_threshold REAL,
                    other_threshold REAL
                );
                CREATE TABLE IF NOT EXISTS speaker_enrollment_samples (
                    user_id TEXT NOT NULL,
                    enrollment_session_id TEXT NOT NULL,
                    sample_index INTEGER NOT NULL,
                    sample_total INTEGER NOT NULL,
                    embedding TEXT NOT NULL,
                    model_name TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'campp_reference_enrollment',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(user_id, enrollment_session_id, sample_index)
                );
                CREATE INDEX IF NOT EXISTS idx_raw_turns_user_created
                    ON raw_turns(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chunks_user_timestamp
                    ON chunks(user_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_chunks_parent
                    ON chunks(parent_type, parent_id, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_discussion_slices_capture
                    ON discussion_slices(user_id, capture_id, start_at);
                CREATE INDEX IF NOT EXISTS idx_discussion_slices_status
                    ON discussion_slices(user_id, status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_discussion_topics_day
                    ON discussion_topics(user_id, day_key, start_at);
                CREATE INDEX IF NOT EXISTS idx_discussion_days_user
                    ON discussion_days(user_id, day_key DESC);
                CREATE INDEX IF NOT EXISTS idx_reply_feedback_user_rating_updated
                    ON reply_feedback(user_id, rating, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_jobs_user_updated
                    ON memory_jobs(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_device_audio_events_user_status
                    ON device_audio_events(user_id, status, created_at);
                CREATE INDEX IF NOT EXISTS idx_speaker_profiles_updated
                    ON speaker_profiles(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_speaker_enrollment_samples_updated
                    ON speaker_enrollment_samples(user_id, updated_at DESC);
                """
            )
            self._ensure_column("speaker_profiles", "sample_count", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column("speaker_profiles", "target_sample_count", "INTEGER NOT NULL DEFAULT 3")
            self._ensure_column("speaker_profiles", "calibration_status", "TEXT NOT NULL DEFAULT 'single_sample'")
            self._ensure_column("speaker_profiles", "profile_version", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column("speaker_profiles", "user_threshold", "REAL")
            self._ensure_column("speaker_profiles", "other_threshold", "REAL")
            self._ensure_column("reply_feedback", "note", "TEXT NOT NULL DEFAULT ''")
            try:
                self._conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        text,
                        source,
                        content='chunks',
                        content_rowid='rowid'
                    )
                    """
                )
                self._conn.executescript(
                    """
                    DROP TRIGGER IF EXISTS chunks_fts_insert;
                    DROP TRIGGER IF EXISTS chunks_fts_delete;
                    DROP TRIGGER IF EXISTS chunks_fts_update;

                    CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
                    AFTER INSERT ON chunks WHEN new.deleted_at IS NULL BEGIN
                        INSERT INTO chunks_fts(rowid, text, source)
                        VALUES (new.rowid, new.text, new.source);
                    END;

                    CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
                    AFTER DELETE ON chunks WHEN old.deleted_at IS NULL BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, text, source)
                        VALUES ('delete', old.rowid, old.text, old.source);
                    END;

                    CREATE TRIGGER IF NOT EXISTS chunks_fts_update
                    AFTER UPDATE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, text, source)
                        SELECT 'delete', old.rowid, old.text, old.source
                        WHERE old.deleted_at IS NULL;
                        INSERT INTO chunks_fts(rowid, text, source)
                        SELECT new.rowid, new.text, new.source
                        WHERE new.deleted_at IS NULL;
                    END;
                    """
                )
            except sqlite3.OperationalError:
                # SQLite builds without FTS5 still search through LIKE fallback.
                pass
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        columns = {
            str(row["name"])
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in columns:
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def upsert_speaker_profile(
        self,
        *,
        user_id: str,
        embedding: list[float],
        model_name: str,
        source: str = "campp_reference_enrollment",
        updated_at: float | None = None,
        sample_count: int = 1,
        target_sample_count: int = 3,
        calibration_status: str = "single_sample",
        profile_version: int | None = None,
        user_threshold: float | None = None,
        other_threshold: float | None = None,
    ) -> SpeakerProfileRecord:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            raise ValueError("user_id is required")
        normalized_embedding = [float(value) for value in embedding if value is not None]
        if not normalized_embedding:
            raise ValueError("speaker embedding cannot be empty")
        now = float(updated_at if updated_at is not None else time.time())
        existing = self.get_speaker_profile(normalized_user_id)
        created_at = existing.created_at if existing is not None else now
        resolved_profile_version = (
            int(profile_version)
            if profile_version is not None
            else (existing.profile_version + 1 if existing is not None else 1)
        )
        payload = json.dumps(normalized_embedding, ensure_ascii=True)
        record = SpeakerProfileRecord(
            user_id=normalized_user_id,
            embedding=normalized_embedding,
            model_name=str(model_name or "").strip(),
            source=str(source or "campp_reference_enrollment").strip() or "campp_reference_enrollment",
            created_at=created_at,
            updated_at=now,
            sample_count=max(1, int(sample_count)),
            target_sample_count=max(1, int(target_sample_count)),
            calibration_status=str(calibration_status or "single_sample").strip() or "single_sample",
            profile_version=max(1, resolved_profile_version),
            user_threshold=float(user_threshold) if user_threshold is not None else None,
            other_threshold=float(other_threshold) if other_threshold is not None else None,
        )
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO speaker_profiles (
                    user_id, embedding, model_name, source, created_at, updated_at,
                    sample_count, target_sample_count, calibration_status, profile_version,
                    user_threshold, other_threshold
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    embedding=excluded.embedding,
                    model_name=excluded.model_name,
                    source=excluded.source,
                    updated_at=excluded.updated_at,
                    sample_count=excluded.sample_count,
                    target_sample_count=excluded.target_sample_count,
                    calibration_status=excluded.calibration_status,
                    profile_version=excluded.profile_version,
                    user_threshold=excluded.user_threshold,
                    other_threshold=excluded.other_threshold
                """,
                (
                    record.user_id,
                    payload,
                    record.model_name,
                    record.source,
                    record.created_at,
                    record.updated_at,
                    record.sample_count,
                    record.target_sample_count,
                    record.calibration_status,
                    record.profile_version,
                    record.user_threshold,
                    record.other_threshold,
                ),
            )
        return record

    def get_speaker_profile(self, user_id: str) -> SpeakerProfileRecord | None:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT user_id, embedding, model_name, source, created_at, updated_at,
                       sample_count, target_sample_count, calibration_status, profile_version,
                       user_threshold, other_threshold
                FROM speaker_profiles
                WHERE user_id = ?
                """,
                (normalized_user_id,),
            ).fetchone()
        if row is None:
            return None
        return SpeakerProfileRecord(
            user_id=str(row["user_id"]),
            embedding=self._embedding_from_json(str(row["embedding"] or "[]")),
            model_name=str(row["model_name"] or ""),
            source=str(row["source"] or ""),
            created_at=float(row["created_at"] or 0.0),
            updated_at=float(row["updated_at"] or 0.0),
            sample_count=int(row["sample_count"] or 1),
            target_sample_count=int(row["target_sample_count"] or 3),
            calibration_status=str(row["calibration_status"] or "single_sample"),
            profile_version=int(row["profile_version"] or 1),
            user_threshold=float(row["user_threshold"]) if row["user_threshold"] is not None else None,
            other_threshold=float(row["other_threshold"]) if row["other_threshold"] is not None else None,
        )

    def get_speaker_profile_summary(self, user_id: str) -> dict[str, Any]:
        record = self.get_speaker_profile(user_id)
        pending_samples = self.list_speaker_enrollment_samples(user_id)
        if record is None:
            return {
                "enrolled": False,
                "updated_at": None,
                "speaker_model": "",
                "speaker_source": "",
                "sample_count": len(pending_samples),
                "target_sample_count": pending_samples[0].sample_total if pending_samples else 3,
                "calibration_status": "pending" if pending_samples else "not_enrolled",
                "speaker_profile_version": None,
            }
        return {
            "enrolled": True,
            "updated_at": record.updated_at,
            "speaker_model": record.model_name,
            "speaker_source": record.source,
            "sample_count": record.sample_count,
            "target_sample_count": record.target_sample_count,
            "calibration_status": record.calibration_status,
            "speaker_profile_version": record.profile_version,
        }

    def save_speaker_enrollment_sample(
        self,
        *,
        user_id: str,
        enrollment_session_id: str,
        sample_index: int,
        sample_total: int,
        embedding: list[float],
        model_name: str,
        source: str = "campp_reference_enrollment",
        updated_at: float | None = None,
    ) -> SpeakerEnrollmentSampleRecord:
        normalized_user_id = str(user_id or "").strip()
        normalized_session_id = str(enrollment_session_id or "").strip()
        if not normalized_user_id:
            raise ValueError("user_id is required")
        if not normalized_session_id:
            raise ValueError("enrollment_session_id is required")
        normalized_embedding = [float(value) for value in embedding if value is not None]
        if not normalized_embedding:
            raise ValueError("speaker embedding cannot be empty")
        now = float(updated_at if updated_at is not None else time.time())
        payload = json.dumps(normalized_embedding, ensure_ascii=True)
        record = SpeakerEnrollmentSampleRecord(
            user_id=normalized_user_id,
            enrollment_session_id=normalized_session_id,
            sample_index=max(1, int(sample_index)),
            sample_total=max(1, int(sample_total)),
            embedding=normalized_embedding,
            model_name=str(model_name or "").strip(),
            source=str(source or "campp_reference_enrollment").strip() or "campp_reference_enrollment",
            created_at=now,
            updated_at=now,
        )
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO speaker_enrollment_samples (
                    user_id, enrollment_session_id, sample_index, sample_total,
                    embedding, model_name, source, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, enrollment_session_id, sample_index) DO UPDATE SET
                    embedding=excluded.embedding,
                    model_name=excluded.model_name,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (
                    record.user_id,
                    record.enrollment_session_id,
                    record.sample_index,
                    record.sample_total,
                    payload,
                    record.model_name,
                    record.source,
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def list_speaker_enrollment_samples(self, user_id: str, enrollment_session_id: str = "") -> list[SpeakerEnrollmentSampleRecord]:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return []
        params: list[Any] = [normalized_user_id]
        query = """
            SELECT user_id, enrollment_session_id, sample_index, sample_total,
                   embedding, model_name, source, created_at, updated_at
            FROM speaker_enrollment_samples
            WHERE user_id = ?
        """
        normalized_session_id = str(enrollment_session_id or "").strip()
        if normalized_session_id:
            query += " AND enrollment_session_id = ?"
            params.append(normalized_session_id)
        query += " ORDER BY updated_at ASC, sample_index ASC"
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [
            SpeakerEnrollmentSampleRecord(
                user_id=str(row["user_id"]),
                enrollment_session_id=str(row["enrollment_session_id"]),
                sample_index=int(row["sample_index"] or 0),
                sample_total=int(row["sample_total"] or 0),
                embedding=self._embedding_from_json(str(row["embedding"] or "[]")),
                model_name=str(row["model_name"] or ""),
                source=str(row["source"] or ""),
                created_at=float(row["created_at"] or 0.0),
                updated_at=float(row["updated_at"] or 0.0),
            )
            for row in rows
        ]

    def clear_speaker_enrollment_samples(self, user_id: str, enrollment_session_id: str = "") -> int:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return 0
        params: list[Any] = [normalized_user_id]
        query = "DELETE FROM speaker_enrollment_samples WHERE user_id = ?"
        normalized_session_id = str(enrollment_session_id or "").strip()
        if normalized_session_id:
            query += " AND enrollment_session_id = ?"
            params.append(normalized_session_id)
        with self._lock, self._conn:
            cursor = self._conn.execute(query, tuple(params))
        return int(cursor.rowcount or 0)

    @staticmethod
    def _embedding_from_json(raw: str) -> list[float]:
        try:
            payload = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        embedding: list[float] = []
        for item in payload:
            try:
                embedding.append(float(item))
            except (TypeError, ValueError):
                continue
        return embedding

    # 每个用户 turn 先沉淀原话 chunk，助手回复稍后可通过 update_turn_reply 补齐。
    def add_turn(
        self,
        user_id: str,
        raw_text: str,
        *,
        assistant_reply: str = "",
        source: str = "chat",
        interaction_id: str = "",
        legacy_session_id: str = "",
        created_at: float | None = None,
    ) -> TimelineWriteResult:
        # 原文脱敏
        redaction = redact_sensitive_text(str(raw_text or ""))
        raw_text = redaction.text.strip()
        if not raw_text:
            raise ValueError("timeline turn text cannot be empty")
        now = created_at if created_at is not None else time.time()
        with self._lock:
            # 创建timeline 记录
            turn = TimelineTurn(
                id=f"turn_{uuid.uuid4().hex[:16]}",
                user_id=user_id,
                source=str(source or "chat").strip() or "chat",
                raw_text=raw_text,
                assistant_reply=str(assistant_reply or "").strip(),
                interaction_id=str(interaction_id or "").strip(),
                legacy_session_id=str(legacy_session_id or "").strip(),
                created_at=now,
                updated_at=now,
                status="active",
            )
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO raw_turns (
                        id, user_id, source, raw_text, assistant_reply, interaction_id,
                        legacy_session_id, created_at, updated_at, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.id,
                        turn.user_id,
                        turn.source,
                        turn.raw_text,
                        turn.assistant_reply,
                        turn.interaction_id,
                        turn.legacy_session_id,
                        turn.created_at,
                        turn.updated_at,
                        turn.status,
                    ),
                )
            # 把这段原话切成 chunks 写到 chunks 表
            chunks = self.add_chunks(
                user_id,
                parent_type="turn",
                parent_id=turn.id,
                chunks=self._chunk_text(raw_text),
                source=turn.source,
                timestamp=now,
                metadata={
                    **({"legacy_session_id": turn.legacy_session_id} if turn.legacy_session_id else {}),
                    **_redaction_metadata(redaction),
                },
            )
            return TimelineWriteResult(turn=turn, chunks=chunks, redaction=redaction)

    # 回复生成后补齐 turn，确保 raw timeline 同时保留用户原话和助手应答。
    def update_turn_reply(
        self,
        user_id: str,
        turn_id: str,
        assistant_reply: str,
        *,
        legacy_session_id: str = "",
        updated_at: float | None = None,
    ) -> TimelineTurn | None:
        now = updated_at if updated_at is not None else time.time()
        with self._lock:
            if legacy_session_id:
                self._conn.execute(
                    """
                    UPDATE raw_turns
                    SET assistant_reply = ?, legacy_session_id = ?, updated_at = ?
                    WHERE user_id = ? AND id = ? AND deleted_at IS NULL
                    """,
                    (str(assistant_reply or "").strip(), legacy_session_id, now, user_id, turn_id),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE raw_turns
                    SET assistant_reply = ?, updated_at = ?
                    WHERE user_id = ? AND id = ? AND deleted_at IS NULL
                    """,
                    (str(assistant_reply or "").strip(), now, user_id, turn_id),
                )
            self._conn.commit()
            return self.get_turn(user_id, turn_id)

    def get_turn(self, user_id: str, turn_id: str) -> TimelineTurn | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, user_id, source, raw_text, assistant_reply, interaction_id,
                       legacy_session_id, created_at, updated_at, status
                FROM raw_turns
                WHERE user_id = ? AND id = ? AND deleted_at IS NULL AND status = 'active'
                """,
                (user_id, turn_id),
            ).fetchone()
            return self._row_to_turn(row) if row else None

    def upsert_reply_feedback(
        self,
        *,
        user_id: str,
        turn_id: str,
        rating: str,
        note: str = "",
        updated_at: float | None = None,
    ) -> ReplyFeedback:
        normalized_user_id = str(user_id or "").strip()
        normalized_turn_id = str(turn_id or "").strip()
        normalized_rating = str(rating or "").strip().lower()
        if not normalized_user_id:
            raise ValueError("user_id is required")
        if not normalized_turn_id:
            raise ValueError("turn_id is required")
        if normalized_rating not in {"satisfied", "needs_improvement"}:
            raise ValueError("rating must be satisfied or needs_improvement")
        normalized_note = redact_sensitive_text(str(note or "")).text.strip()[:1000]
        turn = self.get_turn(normalized_user_id, normalized_turn_id)
        if turn is None:
            raise ValueError("timeline turn not found")
        if not turn.assistant_reply:
            raise ValueError("timeline turn has no assistant reply")
        now = float(updated_at if updated_at is not None else time.time())
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT id, created_at FROM reply_feedback WHERE user_id = ? AND turn_id = ?",
                (normalized_user_id, normalized_turn_id),
            ).fetchone()
            feedback_id = str(existing["id"]) if existing else f"feedback_{uuid.uuid4().hex[:16]}"
            created_at = float(existing["created_at"]) if existing else now
            self._conn.execute(
                """
                INSERT INTO reply_feedback (id, user_id, turn_id, rating, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, turn_id) DO UPDATE SET
                    rating=excluded.rating,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (feedback_id, normalized_user_id, normalized_turn_id, normalized_rating, normalized_note, created_at, now),
            )
        return ReplyFeedback(
            id=feedback_id,
            user_id=normalized_user_id,
            turn_id=normalized_turn_id,
            rating=normalized_rating,
            note=normalized_note,
            created_at=created_at,
            updated_at=now,
        )

    def list_reply_feedback(
        self,
        user_id: str,
        *,
        rating: str = "",
        limit: int = 100,
    ) -> list[ReplyFeedback]:
        normalized_rating = str(rating or "").strip().lower()
        if normalized_rating and normalized_rating not in {"satisfied", "needs_improvement"}:
            raise ValueError("rating must be satisfied or needs_improvement")
        bounded_limit = max(1, min(int(limit), 500))
        query = "SELECT id, user_id, turn_id, rating, note, created_at, updated_at FROM reply_feedback WHERE user_id = ?"
        params: list[Any] = [str(user_id or "").strip()]
        if normalized_rating:
            query += " AND rating = ?"
            params.append(normalized_rating)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(bounded_limit)
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [
            ReplyFeedback(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                turn_id=str(row["turn_id"]),
                rating=str(row["rating"]),
                note=str(row["note"] or ""),
                created_at=float(row["created_at"]),
                updated_at=float(row["updated_at"]),
            )
            for row in rows
        ]

    def add_capture(
        self,
        user_id: str,
        *,
        source: str = "continuous_capture",
        context: str = "",
        started_at: float | None = None,
    ) -> dict[str, Any]:
        now = started_at if started_at is not None else time.time()
        capture_id = f"cap_{uuid.uuid4().hex[:16]}"
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO captures (
                    id, user_id, source, context, started_at, created_at, updated_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (capture_id, user_id, source, context, now, now, now, "running"),
            )
            self._conn.commit()
            return {
                "capture_id": capture_id,
                "user_id": user_id,
                "source": source,
                "context": context,
                "started_at": now,
                "status": "running",
            }

    def get_capture(self, user_id: str, capture_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, user_id, source, context, started_at, ended_at, summary,
                       created_at, updated_at, status
                FROM captures
                WHERE user_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (user_id, capture_id),
            ).fetchone()
            if row is None:
                return None
            chunks = self._conn.execute(
                """
                SELECT id, user_id, parent_type, parent_id, text, chunk_index,
                       start_offset, end_offset, timestamp, source, metadata, status
                FROM chunks
                WHERE user_id = ? AND parent_type = 'capture' AND parent_id = ?
                  AND deleted_at IS NULL AND status = 'active'
                ORDER BY chunk_index ASC
                """,
                (user_id, capture_id),
            ).fetchall()
            return {
                "capture_id": row["id"],
                "user_id": row["user_id"],
                "source": str(row["source"] or ""),
                "context": str(row["context"] or ""),
                "chunks": [
                    {
                        "text": str(chunk["text"] or ""),
                        "timestamp": float(chunk["timestamp"]),
                        "chunk_id": str(chunk["id"] or ""),
                        "metadata": self._metadata_from_json(str(chunk["metadata"] or "{}")),
                    }
                    for chunk in chunks
                ],
                "started_at": float(row["started_at"]),
                "ended_at": float(row["ended_at"]) if row["ended_at"] is not None else None,
                "summary": str(row["summary"] or ""),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]) if row["updated_at"] else float(row["created_at"]),
                "status": str(row["status"] or "running"),
            }

    # capture 片段直接作为 chunk 入库，stop 后再做结构化记忆抽取。
    def add_capture_chunk(
        self,
        user_id: str,
        capture_id: str,
        text: str,
        *,
        timestamp: float | None = None,
        source: str = "continuous_capture",
        metadata: dict[str, Any] | None = None,
    ) -> TimelineChunk:
        with self._lock:
            capture = self._conn.execute(
                """
                SELECT id FROM captures
                WHERE user_id = ? AND id = ? AND deleted_at IS NULL AND status = 'running'
                """,
                (user_id, capture_id),
            ).fetchone()
            if capture is None:
                raise ValueError("capture not found")
            existing = self._conn.execute(
                """
                SELECT COUNT(*) AS count FROM chunks
                WHERE user_id = ? AND parent_type = 'capture' AND parent_id = ? AND deleted_at IS NULL
                """,
                (user_id, capture_id),
            ).fetchone()
            chunk = self.add_chunks(
                user_id,
                parent_type="capture",
                parent_id=capture_id,
                chunks=[{"text": text, "start_offset": 0, "end_offset": len(str(text or ""))}],
                source=source,
                timestamp=timestamp,
                metadata=metadata,
                start_index=int(existing["count"] or 0),
            )[0]
            self._conn.execute(
                "UPDATE captures SET updated_at = ? WHERE user_id = ? AND id = ?",
                (time.time(), user_id, capture_id),
            )
            self._conn.commit()
            return chunk

    def finish_capture(
        self,
        user_id: str,
        capture_id: str,
        *,
        summary: str = "",
        ended_at: float | None = None,
        status: str = "stopped",
    ) -> bool:
        now = ended_at if ended_at is not None else time.time()
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE captures
                SET ended_at = ?, summary = ?, updated_at = ?, status = ?
                WHERE user_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (now, str(summary or "").strip(), now, str(status or "stopped"), user_id, capture_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def create_discussion_slice(
        self,
        *,
        user_id: str,
        capture_id: str,
        day_key: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        normalized = [chunk for chunk in chunks if str(chunk.get("chunk_id") or "").strip()]
        if not normalized:
            return None
        first_id = str(normalized[0]["chunk_id"])
        last_id = str(normalized[-1]["chunk_id"])
        slice_id = "dslice_" + uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{user_id}\n{capture_id}\n{first_id}\n{last_id}",
        ).hex[:20]
        now = time.time()
        chunk_ids = [str(chunk["chunk_id"]) for chunk in normalized]
        start_at = float(normalized[0].get("timestamp") or now)
        end_at = float(normalized[-1].get("timestamp") or start_at)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO discussion_slices (
                    id, user_id, capture_id, day_key, first_chunk_id, last_chunk_id,
                    start_at, end_at, chunk_ids, created_at, updated_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    slice_id,
                    user_id,
                    capture_id,
                    day_key,
                    first_id,
                    last_id,
                    start_at,
                    end_at,
                    json.dumps(chunk_ids, ensure_ascii=True),
                    now,
                    now,
                ),
            )
        return self.get_discussion_slice(user_id, slice_id)

    def get_discussion_slice(self, user_id: str, slice_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM discussion_slices WHERE user_id = ? AND id = ?",
                (user_id, slice_id),
            ).fetchone()
        return self._discussion_slice_payload(row) if row else None

    def list_discussion_slices(
        self,
        user_id: str,
        *,
        capture_id: str = "",
        statuses: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if capture_id:
            clauses.append("capture_id = ?")
            params.append(capture_id)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(sorted(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM discussion_slices WHERE {' AND '.join(clauses)} ORDER BY start_at, created_at",
                tuple(params),
            ).fetchall()
        return [self._discussion_slice_payload(row) for row in rows]

    def list_discussion_slices_for_recovery(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM discussion_slices
                WHERE status IN ('pending', 'running', 'failed')
                ORDER BY updated_at, start_at LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [self._discussion_slice_payload(row) for row in rows]

    def update_discussion_slice(
        self,
        *,
        user_id: str,
        slice_id: str,
        status: str,
        summary_payload: dict[str, Any] | None = None,
        backend: str = "",
        error_type: str = "",
    ) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                UPDATE discussion_slices
                SET status = ?, summary_payload = ?, backend = ?, error_type = ?, updated_at = ?
                WHERE user_id = ? AND id = ?
                """,
                (
                    status,
                    json.dumps(summary_payload or {}, ensure_ascii=False),
                    str(backend or ""),
                    str(error_type or ""),
                    time.time(),
                    user_id,
                    slice_id,
                ),
            )
        return cur.rowcount > 0

    def discussion_uncovered_capture_chunks(self, user_id: str, capture_id: str) -> list[dict[str, Any]]:
        capture = self.get_capture(user_id, capture_id)
        if not capture:
            return []
        covered = {
            chunk_id
            for item in self.list_discussion_slices(user_id, capture_id=capture_id)
            for chunk_id in item["chunk_ids"]
        }
        return [chunk for chunk in capture["chunks"] if str(chunk.get("chunk_id") or "") not in covered]

    def ambient_capture_ids_between(self, user_id: str, start_at: float, end_at: float) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT parent_id FROM chunks
                WHERE user_id = ? AND parent_type = 'capture' AND source = 'ambient_audio_text'
                  AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
                """,
                (user_id, float(start_at), float(end_at)),
            ).fetchall()
        return [str(row["parent_id"] or "") for row in rows if str(row["parent_id"] or "")]

    def list_ambient_capture_keys(self, *, limit: int = 500) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT chunks.user_id, chunks.parent_id, MIN(chunks.timestamp) AS first_at
                FROM chunks
                WHERE chunks.parent_type = 'capture' AND chunks.source = 'ambient_audio_text'
                  AND chunks.deleted_at IS NULL AND chunks.status = 'active'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM discussion_slices, json_each(discussion_slices.chunk_ids)
                      WHERE discussion_slices.user_id = chunks.user_id
                        AND discussion_slices.capture_id = chunks.parent_id
                        AND json_each.value = chunks.id
                  )
                GROUP BY chunks.user_id, chunks.parent_id
                ORDER BY first_at LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            (str(row["user_id"]), str(row["parent_id"]))
            for row in rows
            if str(row["user_id"] or "") and str(row["parent_id"] or "")
        ]

    def list_discussion_topics(self, user_id: str, day_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM discussion_topics
                WHERE user_id = ? AND day_key = ? AND deleted_at IS NULL AND status = 'active'
                ORDER BY start_at, created_at
                """,
                (user_id, day_key),
            ).fetchall()
        return [self._discussion_topic_payload(row) for row in rows]

    def upsert_discussion_topic(
        self,
        *,
        user_id: str,
        day_key: str,
        slice_id: str,
        contribution: dict[str, Any],
        start_at: float,
        end_at: float,
    ) -> dict[str, Any]:
        merge_id = str(contribution.get("merge_topic_id") or "").strip()
        with self._lock:
            existing_row = self._conn.execute(
                """
                SELECT * FROM discussion_topics
                WHERE user_id = ? AND day_key = ? AND id = ?
                  AND deleted_at IS NULL AND status = 'active'
                """,
                (user_id, day_key, merge_id),
            ).fetchone() if merge_id else None
            existing = self._discussion_topic_payload(existing_row) if existing_row else None
            now = time.time()
            topic_id = existing["id"] if existing else f"dtopic_{uuid.uuid4().hex[:20]}"
            time_spans = _merge_time_spans(
                list(existing["time_spans"] if existing else []),
                {"start_at": float(start_at), "end_at": float(end_at)},
            )
            values = {
                name: _merge_string_values(
                    list(existing[name] if existing else []),
                    list(contribution.get(name) or []),
                )
                for name in (
                    "key_points", "decisions", "tasks", "open_questions", "participant_labels"
                )
            }
            evidence_ids = _merge_string_values(
                list(existing["evidence_ids"] if existing else []),
                list(contribution.get("source_chunk_ids") or []),
                limit=None,
            )
            slice_ids = _merge_string_values(
                list(existing["slice_ids"] if existing else []), [slice_id], limit=200
            )
            summary = str(contribution.get("summary") or "").strip()
            created_at = float(existing["created_at"]) if existing else now
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO discussion_topics (
                        id, user_id, day_key, title, topic_key, summary, key_points,
                        decisions, tasks, open_questions, participant_labels, time_spans,
                        slice_ids, evidence_ids, start_at, end_at, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title, topic_key=excluded.topic_key, summary=excluded.summary,
                        key_points=excluded.key_points, decisions=excluded.decisions, tasks=excluded.tasks,
                        open_questions=excluded.open_questions,
                        participant_labels=excluded.participant_labels,
                        time_spans=excluded.time_spans, slice_ids=excluded.slice_ids,
                        evidence_ids=excluded.evidence_ids, start_at=excluded.start_at,
                        end_at=excluded.end_at, status='active', updated_at=excluded.updated_at,
                        deleted_at=NULL
                    """,
                    (
                        topic_id,
                        user_id,
                        day_key,
                        str(contribution.get("title") or (existing or {}).get("title") or "环境讨论"),
                        str(contribution.get("topic_key") or (existing or {}).get("topic_key") or ""),
                        summary or str((existing or {}).get("summary") or ""),
                        json.dumps(values["key_points"], ensure_ascii=False),
                        json.dumps(values["decisions"], ensure_ascii=False),
                        json.dumps(values["tasks"], ensure_ascii=False),
                        json.dumps(values["open_questions"], ensure_ascii=False),
                        json.dumps(values["participant_labels"], ensure_ascii=False),
                        json.dumps(time_spans, ensure_ascii=True),
                        json.dumps(slice_ids, ensure_ascii=True),
                        json.dumps(evidence_ids, ensure_ascii=True),
                        min(float(start_at), float(existing["start_at"]) if existing else float(start_at)),
                        max(float(end_at), float(existing["end_at"]) if existing else float(end_at)),
                        created_at,
                        now,
                    ),
                )
        return next(topic for topic in self.list_discussion_topics(user_id, day_key) if topic["id"] == topic_id)

    def rebuild_discussion_day(self, user_id: str, day_key: str) -> dict[str, Any] | None:
        topics = self.list_discussion_topics(user_id, day_key)
        if not topics:
            return None
        overview = "\n".join(
            f"{index}. {topic['title']}：{topic['summary']}"
            for index, topic in enumerate(topics, start=1)
        )
        topic_ids = [topic["id"] for topic in topics]
        evidence_ids = _merge_string_values(
            [], [evidence_id for topic in topics for evidence_id in topic["evidence_ids"]], limit=None
        )
        now = time.time()
        day_id = "dday_" + uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}\n{day_key}").hex[:20]
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO discussion_days (
                    id, user_id, day_key, overview, topic_ids, evidence_ids,
                    topic_count, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
                ON CONFLICT(user_id, day_key) DO UPDATE SET
                    overview=excluded.overview, topic_ids=excluded.topic_ids,
                    evidence_ids=excluded.evidence_ids, topic_count=excluded.topic_count,
                    status='ready', updated_at=excluded.updated_at, deleted_at=NULL
                """,
                (
                    day_id,
                    user_id,
                    day_key,
                    overview,
                    json.dumps(topic_ids, ensure_ascii=True),
                    json.dumps(evidence_ids, ensure_ascii=True),
                    len(topics),
                    now,
                    now,
                ),
            )
        return self.get_discussion_day(user_id, day_key)

    def list_discussion_days(self, user_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 365))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM discussion_days
                WHERE user_id = ? AND deleted_at IS NULL
                ORDER BY day_key DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._discussion_day_payload(row) for row in rows]

    def get_discussion_day(self, user_id: str, day_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM discussion_days
                WHERE user_id = ? AND day_key = ? AND deleted_at IS NULL
                """,
                (user_id, day_key),
            ).fetchone()
        if row is None:
            return None
        payload = self._discussion_day_payload(row)
        topics = self.list_discussion_topics(user_id, day_key)
        existing_ids = self.existing_chunk_ids(user_id, payload["evidence_ids"])
        for topic in topics:
            available = [item for item in topic["evidence_ids"] if item in existing_ids]
            topic["available_evidence_ids"] = available
            topic["evidence_status"] = "available" if len(available) == len(topic["evidence_ids"]) else (
                "partially_expired" if available else "expired"
            )
        payload["topics"] = topics
        payload["available_evidence_ids"] = [item for item in payload["evidence_ids"] if item in existing_ids]
        payload["evidence_status"] = (
            "available"
            if len(payload["available_evidence_ids"]) == len(payload["evidence_ids"])
            else "partially_expired" if payload["available_evidence_ids"] else "expired"
        )
        return payload

    def delete_discussion_day(
        self,
        *,
        user_id: str,
        day_key: str,
        scope: str,
        start_at: float,
        end_at: float,
    ) -> dict[str, Any]:
        normalized_scope = str(scope or "").strip().lower()
        if normalized_scope not in {"raw", "summary", "all"}:
            raise ValueError("discussion delete scope must be raw, summary, or all")
        raw_result = TimelinePurgeResult([], [])
        summary_count = 0
        if normalized_scope in {"raw", "all"}:
            raw_ids = self._ambient_chunk_ids_between(user_id, start_at, end_at, limit=100_000)
            raw_result = self.purge_chunks(user_id, raw_ids, preserve_capture_parents=True)
        if normalized_scope in {"summary", "all"}:
            with self._lock, self._conn:
                summary_count += int(self._conn.execute(
                    "DELETE FROM discussion_slices WHERE user_id = ? AND day_key = ?",
                    (user_id, day_key),
                ).rowcount or 0)
                summary_count += int(self._conn.execute(
                    "DELETE FROM discussion_topics WHERE user_id = ? AND day_key = ?",
                    (user_id, day_key),
                ).rowcount or 0)
                summary_count += int(self._conn.execute(
                    "DELETE FROM discussion_days WHERE user_id = ? AND day_key = ?",
                    (user_id, day_key),
                ).rowcount or 0)
        return {
            "day": day_key,
            "scope": normalized_scope,
            "purged_raw_count": raw_result.purged_chunk_count,
            "purged_parent_count": raw_result.purged_parent_count,
            "deleted_summary_record_count": summary_count,
        }

    def purge_expired_ambient_chunks(self, *, cutoff: float, limit: int = 500) -> TimelinePurgeResult:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id FROM chunks
                WHERE source = 'ambient_audio_text' AND timestamp < ?
                ORDER BY timestamp ASC LIMIT ?
                """,
                (float(cutoff), max(1, int(limit))),
            ).fetchall()
        return self.purge_chunks("", []) if not rows else self._purge_ambient_rows(rows)

    def _purge_ambient_rows(self, rows: list[sqlite3.Row]) -> TimelinePurgeResult:
        ids = [str(row["id"] or "") for row in rows]
        if not ids:
            return TimelinePurgeResult([], [])
        placeholders = ", ".join("?" for _ in ids)
        with self._lock:
            user_rows = self._conn.execute(
                f"SELECT DISTINCT user_id FROM chunks WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        purged_ids: list[str] = []
        purged_parents: list[str] = []
        for user_row in user_rows:
            result = self.purge_chunks(
                str(user_row["user_id"]),
                ids,
                preserve_capture_parents=True,
            )
            purged_ids.extend(result.purged_chunk_ids)
            purged_parents.extend(result.purged_parent_ids)
        return TimelinePurgeResult(purged_ids, purged_parents)

    def _ambient_chunk_ids_between(self, user_id: str, start_at: float, end_at: float, *, limit: int) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id FROM chunks
                WHERE user_id = ? AND source = 'ambient_audio_text'
                  AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp LIMIT ?
                """,
                (user_id, float(start_at), float(end_at), max(1, int(limit))),
            ).fetchall()
        return [str(row["id"] or "") for row in rows]

    @staticmethod
    def _discussion_slice_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "capture_id": str(row["capture_id"]),
            "day_key": str(row["day_key"]),
            "first_chunk_id": str(row["first_chunk_id"]),
            "last_chunk_id": str(row["last_chunk_id"]),
            "start_at": float(row["start_at"]),
            "end_at": float(row["end_at"]),
            "chunk_ids": _json_list(row["chunk_ids"]),
            "summary_payload": _json_object(row["summary_payload"]),
            "backend": str(row["backend"] or ""),
            "error_type": str(row["error_type"] or ""),
            "status": str(row["status"] or "pending"),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @staticmethod
    def _discussion_topic_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "day_key": str(row["day_key"]),
            "title": str(row["title"] or ""),
            "topic_key": str(row["topic_key"] or ""),
            "summary": str(row["summary"] or ""),
            "key_points": _json_list(row["key_points"]),
            "decisions": _json_list(row["decisions"]),
            "tasks": _json_list(row["tasks"]),
            "open_questions": _json_list(row["open_questions"]),
            "participant_labels": _json_list(row["participant_labels"]),
            "time_spans": _json_list(row["time_spans"]),
            "slice_ids": _json_list(row["slice_ids"]),
            "evidence_ids": _json_list(row["evidence_ids"]),
            "start_at": float(row["start_at"]),
            "end_at": float(row["end_at"]),
            "status": str(row["status"] or "active"),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @staticmethod
    def _discussion_day_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "day": str(row["day_key"]),
            "overview": str(row["overview"] or ""),
            "topic_ids": _json_list(row["topic_ids"]),
            "evidence_ids": _json_list(row["evidence_ids"]),
            "topic_count": int(row["topic_count"] or 0),
            "status": str(row["status"] or "ready"),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def upsert_memory_job(self, user_id: str, job_id: str, payload: dict[str, Any]) -> None:
        now = time.time()
        created_at = float(payload.get("created_at") or now)
        updated_at = float(payload.get("updated_at") or now)
        status = str(payload.get("status") or "pending")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_jobs (id, user_id, payload, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_id = excluded.user_id,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at,
                    status = excluded.status
                """,
                (
                    job_id,
                    user_id,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    created_at,
                    updated_at,
                    status,
                ),
            )
            self._conn.commit()

    def get_memory_job(self, user_id: str, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT payload FROM memory_jobs
                WHERE user_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (user_id, job_id),
            ).fetchone()
            if row is None:
                return None
            try:
                payload = json.loads(row["payload"] or "{}")
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None

    def enqueue_device_audio_event(
        self,
        *,
        user_id: str,
        event: dict[str, Any],
        capture_id: str = "",
        private_payload: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        event_id = str(event.get("event_id") or "").strip()
        audio_session_id = str(event.get("audio_session_id") or "").strip()
        if not user_id.strip() or not event_id or not audio_session_id:
            raise ValueError("device audio event identity is required")
        now = float(created_at if created_at is not None else time.time())
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO device_audio_events (
                    user_id, event_id, audio_session_id, capture_id, event_payload,
                    private_payload, dispatch_payload, status, attempt_count,
                    error_type, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, '{}', 'pending', 0, '', ?, ?, NULL)
                """,
                (
                    user_id,
                    event_id,
                    audio_session_id,
                    str(capture_id or "").strip(),
                    json.dumps(event, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(private_payload or {}, ensure_ascii=False, separators=(",", ":")),
                    now,
                    now,
                ),
            )
            self._conn.commit()
            row = self._device_audio_event_row(user_id, event_id)
            if row is None:
                raise RuntimeError("device audio event was not persisted")
            payload = self._device_audio_event_payload(row)
            payload["created"] = cursor.rowcount == 1
            return payload

    def get_device_audio_event(self, user_id: str, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._device_audio_event_row(user_id, event_id)
            return self._device_audio_event_payload(row) if row is not None else None

    def list_device_audio_events(
        self,
        user_id: str,
        *,
        statuses: set[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        normalized_statuses = sorted(str(item) for item in (statuses or set()) if str(item))
        if normalized_statuses:
            clauses.append("status IN (" + ",".join("?" for _ in normalized_statuses) + ")")
            params.extend(normalized_statuses)
        params.append(max(1, min(int(limit), 500)))
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM device_audio_events
                WHERE {" AND ".join(clauses)}
                ORDER BY created_at, event_id
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            return [self._device_audio_event_payload(row) for row in rows]

    def claim_next_device_audio_event(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    """
                    SELECT event_id FROM device_audio_events
                    WHERE user_id = ? AND status = 'pending'
                    ORDER BY created_at, event_id
                    LIMIT 1
                    """,
                    (user_id,),
                ).fetchone()
                if row is None:
                    return None
                event_id = str(row["event_id"])
                now = time.time()
                updated = self._conn.execute(
                    """
                    UPDATE device_audio_events
                    SET status = 'running', attempt_count = attempt_count + 1,
                        updated_at = ?, error_type = ''
                    WHERE user_id = ? AND event_id = ? AND status = 'pending'
                    """,
                    (now, user_id, event_id),
                )
                if updated.rowcount != 1:
                    return None
            claimed = self._device_audio_event_row(user_id, event_id)
            return self._device_audio_event_payload(claimed) if claimed is not None else None

    def update_device_audio_event(
        self,
        *,
        user_id: str,
        event_id: str,
        status: str,
        dispatch: dict[str, Any] | None = None,
        error_type: str = "",
    ) -> dict[str, Any] | None:
        if status not in {"pending", "running", "completed", "failed"}:
            raise ValueError("unsupported device audio event status")
        now = time.time()
        completed_at = now if status in {"completed", "failed"} else None
        with self._lock:
            self._conn.execute(
                """
                UPDATE device_audio_events
                SET status = ?, dispatch_payload = ?, error_type = ?,
                    updated_at = ?, completed_at = ?
                WHERE user_id = ? AND event_id = ?
                """,
                (
                    status,
                    json.dumps(dispatch or {}, ensure_ascii=False, separators=(",", ":")),
                    str(error_type or ""),
                    now,
                    completed_at,
                    user_id,
                    event_id,
                ),
            )
            self._conn.commit()
            row = self._device_audio_event_row(user_id, event_id)
            return self._device_audio_event_payload(row) if row is not None else None

    def recover_running_device_audio_events(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE device_audio_events
                SET status = 'pending', updated_at = ?, error_type = 'process_restarted'
                WHERE status = 'running'
                """,
                (time.time(),),
            )
            self._conn.commit()
            return max(0, int(cursor.rowcount))

    def retry_failed_device_audio_events(self, user_id: str, event_id: str = "") -> int:
        clauses = ["user_id = ?", "status = 'failed'"]
        params: list[Any] = [user_id]
        if event_id:
            clauses.append("event_id = ?")
            params.append(event_id)
        with self._lock:
            cursor = self._conn.execute(
                f"""
                UPDATE device_audio_events
                SET status = 'pending', updated_at = ?, completed_at = NULL, error_type = ''
                WHERE {" AND ".join(clauses)}
                """,
                tuple([time.time(), *params]),
            )
            self._conn.commit()
            return max(0, int(cursor.rowcount))

    def device_audio_event_summary(self, user_id: str) -> dict[str, int]:
        summary = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT status, COUNT(*) AS total FROM device_audio_events
                WHERE user_id = ? GROUP BY status
                """,
                (user_id,),
            ).fetchall()
        for row in rows:
            status = str(row["status"])
            if status in summary:
                summary[status] = int(row["total"])
        return summary

    def device_audio_event_users(self, *, statuses: set[str] | None = None) -> list[str]:
        clauses = []
        params: list[Any] = []
        normalized_statuses = sorted(str(item) for item in (statuses or set()) if str(item))
        if normalized_statuses:
            clauses.append("status IN (" + ",".join("?" for _ in normalized_statuses) + ")")
            params.extend(normalized_statuses)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT DISTINCT user_id FROM device_audio_events {where} ORDER BY user_id",
                tuple(params),
            ).fetchall()
            return [str(row["user_id"]) for row in rows]

    def _device_audio_event_row(self, user_id: str, event_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM device_audio_events WHERE user_id = ? AND event_id = ?",
            (user_id, event_id),
        ).fetchone()

    @staticmethod
    def _device_audio_event_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "user_id": str(row["user_id"]),
            "event_id": str(row["event_id"]),
            "audio_session_id": str(row["audio_session_id"]),
            "capture_id": str(row["capture_id"] or ""),
            "event": _json_object(row["event_payload"]),
            "private": _json_object(row["private_payload"]),
            "dispatch": _json_object(row["dispatch_payload"]),
            "status": str(row["status"]),
            "attempt_count": int(row["attempt_count"]),
            "error_type": str(row["error_type"] or ""),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "completed_at": float(row["completed_at"]) if row["completed_at"] is not None else None,
        }

    def add_chunks(
        self,
        user_id: str,
        *,
        parent_type: str,
        parent_id: str,
        chunks: list[dict[str, Any]],
        source: str = "",
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
        start_index: int = 0,
    ) -> list[TimelineChunk]:
        now = timestamp if timestamp is not None else time.time()
        saved: list[TimelineChunk] = []
        with self._lock:
            with self._conn:
                for offset, item in enumerate(chunks):
                    redaction = redact_sensitive_text(str(item.get("text") or item.get("content") or ""))
                    text = redaction.text.strip()
                    if not text:
                        continue
                    chunk_metadata = _merge_redaction_metadata(metadata, redaction)
                    chunk = TimelineChunk(
                        id=f"chunk_{uuid.uuid4().hex[:16]}",
                        user_id=user_id,
                        parent_type=str(parent_type or "").strip(),
                        parent_id=str(parent_id or "").strip(),
                        text=text,
                        chunk_index=start_index + offset,
                        start_offset=int(item.get("start_offset") or 0),
                        end_offset=int(item.get("end_offset") or len(text)),
                        timestamp=now,
                        source=str(source or "").strip(),
                        metadata=chunk_metadata,
                        status="active",
                    )
                    self._conn.execute(
                        """
                        INSERT INTO chunks (
                            id, user_id, parent_type, parent_id, text, chunk_index,
                            start_offset, end_offset, timestamp, source, metadata,
                            status, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.id,
                            chunk.user_id,
                            chunk.parent_type,
                            chunk.parent_id,
                            chunk.text,
                            chunk.chunk_index,
                            chunk.start_offset,
                            chunk.end_offset,
                            chunk.timestamp,
                            chunk.source,
                            json.dumps(chunk.metadata, ensure_ascii=False),
                            chunk.status,
                            time.time(),
                        ),
                    )
                    saved.append(chunk)
            return saved

    # 原文检索优先 FTS，中文或 FTS 不可用时回退到 LIKE。
    def search_chunks(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        exclude_parent_id: str = "",
    ) -> list[TimelineChunk]:
        return self.search_chunks_with_ranking(
            user_id,
            query,
            limit=limit,
            exclude_parent_id=exclude_parent_id,
        ).chunks

    # 原文召回同样保留旧返回类型，debug 通过 with_ranking 获取排序因子。
    def search_chunks_with_ranking(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        exclude_parent_id: str = "",
    ) -> TimelineSearchResult:
        query = str(query or "").strip()
        if not query:
            chunks = self.list_recent_chunks(user_id, limit=limit, exclude_parent_id=exclude_parent_id)
            return TimelineSearchResult(chunks=chunks, ranking=[])
        limit = max(1, min(int(limit), 20))
        candidate_limit = max(limit * 4, limit)
        with self._lock:
            rows = []
            try:
                rows = self._conn.execute(
                    """
                    SELECT c.id, c.user_id, c.parent_type, c.parent_id, c.text,
                           c.chunk_index, c.start_offset, c.end_offset, c.timestamp,
                           c.source, c.metadata, c.status
                    FROM chunks_fts f
                    JOIN chunks c ON c.rowid = f.rowid
                    WHERE chunks_fts MATCH ? AND c.user_id = ? AND c.deleted_at IS NULL
                      AND c.status = 'active' AND (? = '' OR c.parent_id != ?)
                    ORDER BY bm25(chunks_fts), c.timestamp DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), user_id, exclude_parent_id, exclude_parent_id, candidate_limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            if not rows:
                rows = self._search_chunks_like(user_id, query, candidate_limit, exclude_parent_id=exclude_parent_id)
            chunks = [self._row_to_chunk(row) for row in rows]
            return self._rank_chunk_search_results(chunks, query, limit)

    def list_recent_chunks(
        self,
        user_id: str,
        *,
        since: float | None = None,
        limit: int = 20,
        exclude_parent_id: str = "",
    ) -> list[TimelineChunk]:
        limit = max(1, min(int(limit), 100))
        params: list[Any] = [user_id]
        clauses = ["user_id = ?", "deleted_at IS NULL", "status = 'active'"]
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if exclude_parent_id:
            clauses.append("parent_id != ?")
            params.append(exclude_parent_id)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, user_id, parent_type, parent_id, text, chunk_index,
                       start_offset, end_offset, timestamp, source, metadata, status
                FROM chunks
                WHERE {" AND ".join(clauses)}
                ORDER BY timestamp DESC, chunk_index DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def page_active_chunks(
        self,
        user_id: str,
        *,
        cursor: str | None = None,
        page_size: int = 100,
        start_at: float | None = None,
        end_at: float | None = None,
        parent_types: list[str] | tuple[str, ...] | set[str] | None = None,
        exclude_parent_id: str = "",
    ) -> PagedSourceResult[TimelineChunk]:
        """Page active raw chunks in stable chronological order."""

        page_size = max(1, min(int(page_size), 500))
        clauses = ["user_id = ?", "deleted_at IS NULL", "status = 'active'"]
        params: list[Any] = [user_id]
        if start_at is not None:
            clauses.append("timestamp >= ?")
            params.append(float(start_at))
        if end_at is not None:
            clauses.append("timestamp < ?")
            params.append(float(end_at))
        normalized_parent_types = list(dict.fromkeys(
            str(item).strip() for item in (parent_types or []) if str(item).strip()
        ))
        if normalized_parent_types:
            clauses.append(f"parent_type IN ({', '.join('?' for _ in normalized_parent_types)})")
            params.extend(normalized_parent_types)
        if exclude_parent_id:
            clauses.append("parent_id != ?")
            params.append(exclude_parent_id)
        decoded_cursor = decode_page_cursor(cursor)
        if cursor and decoded_cursor is None:
            raise ValueError("invalid timeline page cursor")
        if decoded_cursor is not None:
            cursor_time, cursor_id = decoded_cursor
            clauses.append("(timestamp > ? OR (timestamp = ? AND id > ?))")
            params.extend([cursor_time, cursor_time, cursor_id])
        params.append(page_size + 1)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, user_id, parent_type, parent_id, text, chunk_index,
                       start_offset, end_offset, timestamp, source, metadata, status
                FROM chunks
                WHERE {' AND '.join(clauses)}
                ORDER BY timestamp ASC, id ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        exhausted = len(rows) <= page_size
        selected_rows = rows[:page_size]
        items = [self._row_to_chunk(row) for row in selected_rows]
        next_cursor = None
        if not exhausted and selected_rows:
            last = selected_rows[-1]
            next_cursor = encode_page_cursor(float(last["timestamp"]), str(last["id"]))
        return PagedSourceResult(items, next_cursor, len(selected_rows), exhausted)

    def list_adjacent_chunks(
        self,
        user_id: str,
        *,
        parent_id: str,
        chunk_index: int,
        before: int = 1,
        after: int = 1,
    ) -> list[TimelineChunk]:
        if not parent_id:
            return []
        lower = int(chunk_index) - max(0, int(before))
        upper = int(chunk_index) + max(0, int(after))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, user_id, parent_type, parent_id, text, chunk_index,
                       start_offset, end_offset, timestamp, source, metadata, status
                FROM chunks
                WHERE user_id = ? AND parent_id = ? AND deleted_at IS NULL
                  AND status = 'active' AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index ASC, id ASC
                """,
                (user_id, parent_id, lower, upper),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def list_chunks_by_ids(
        self,
        user_id: str,
        chunk_ids: list[str],
        *,
        limit: int = 20,
        include_deleted: bool = False,
    ) -> list[TimelineChunk]:
        ids = list(dict.fromkeys(str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()))
        if not ids:
            return []
        limit = max(1, min(int(limit), 100))
        placeholders = ", ".join("?" for _ in ids)
        active_clause = "" if include_deleted else "AND deleted_at IS NULL AND status = 'active'"
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, user_id, parent_type, parent_id, text, chunk_index,
                       start_offset, end_offset, timestamp, source, metadata, status
                FROM chunks
                WHERE user_id = ? AND id IN ({placeholders})
                  {active_clause}
                ORDER BY timestamp DESC, chunk_index DESC
                LIMIT ?
                """,
                tuple([user_id, *ids, limit]),
            ).fetchall()
            chunks = [self._row_to_chunk(row) for row in rows]
        by_id = {chunk.id: chunk for chunk in chunks}
        return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]

    def existing_chunk_ids(
        self,
        user_id: str,
        chunk_ids: list[str],
        *,
        include_deleted: bool = False,
    ) -> set[str]:
        ids = list(dict.fromkeys(str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()))
        active_clause = "" if include_deleted else "AND deleted_at IS NULL AND status = 'active'"
        existing: set[str] = set()
        batch_size = 500
        with self._lock:
            for offset in range(0, len(ids), batch_size):
                batch = ids[offset:offset + batch_size]
                placeholders = ", ".join("?" for _ in batch)
                rows = self._conn.execute(
                    f"""
                    SELECT id FROM chunks
                    WHERE user_id = ? AND id IN ({placeholders})
                      {active_clause}
                    """,
                    tuple([user_id, *batch]),
                ).fetchall()
                existing.update(str(row["id"]) for row in rows)
        return existing

    def list_chunk_references(
        self,
        user_id: str,
        chunk_ids: list[str],
        reference_counts: dict[str, Any],
        *,
        limit: int = 100,
        include_deleted: bool = True,
    ) -> list[TimelineChunkReference]:
        ids = list(dict.fromkeys(str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()))
        ids = ids[: max(1, int(limit))]
        chunks: list[TimelineChunk] = []
        for offset in range(0, len(ids), 100):
            batch = ids[offset:offset + 100]
            chunks.extend(self.list_chunks_by_ids(
                user_id,
                batch,
                limit=len(batch),
                include_deleted=include_deleted,
            ))
        return [
            TimelineChunkReference(
                chunk=chunk,
                active_refs=int(getattr(reference_counts.get(chunk.id), "active_refs", 0) or 0),
                retained_refs=int(getattr(reference_counts.get(chunk.id), "retained_refs", 0) or 0),
            )
            for chunk in chunks
        ]

    # 删除采用软删除；搜索和原文召回只读取 active chunks。
    def delete_chunks(self, user_id: str, chunk_ids: list[str], *, deleted_at: float | None = None) -> int:
        ids = list(dict.fromkeys(str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()))
        if not ids:
            return 0
        now = deleted_at if deleted_at is not None else time.time()
        placeholders = ", ".join("?" for _ in ids)
        with self._lock:
            parent_rows = self._conn.execute(
                f"""
                SELECT DISTINCT parent_id FROM chunks
                WHERE user_id = ? AND id IN ({placeholders})
                """,
                tuple([user_id, *ids]),
            ).fetchall()
            cur = self._conn.execute(
                f"""
                UPDATE chunks
                SET deleted_at = ?, status = 'deleted'
                WHERE user_id = ? AND id IN ({placeholders})
                  AND deleted_at IS NULL AND status = 'active'
                """,
                tuple([now, user_id, *ids]),
            )
            self._conn.commit()
            self._update_parent_status_for_chunks(user_id, [str(row["parent_id"] or "") for row in parent_rows])
            self._conn.commit()
            return cur.rowcount

    def purge_chunks(
        self,
        user_id: str,
        chunk_ids: list[str],
        *,
        preserve_capture_parents: bool = False,
    ) -> TimelinePurgeResult:
        ids = list(dict.fromkeys(str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()))
        if not ids:
            return TimelinePurgeResult(purged_chunk_ids=[], purged_parent_ids=[])
        placeholders = ", ".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, parent_type, parent_id
                FROM chunks
                WHERE user_id = ? AND id IN ({placeholders})
                """,
                tuple([user_id, *ids]),
            ).fetchall()
            if not rows:
                return TimelinePurgeResult(purged_chunk_ids=[], purged_parent_ids=[])
            purged_chunk_ids = [str(row["id"] or "") for row in rows if str(row["id"] or "")]
            parent_refs = list(dict.fromkeys(
                (str(row["parent_type"] or ""), str(row["parent_id"] or ""))
                for row in rows
                if str(row["parent_id"] or "")
            ))
            self._conn.execute(
                f"""
                DELETE FROM chunks
                WHERE user_id = ? AND id IN ({placeholders})
                """,
                tuple([user_id, *ids]),
            )
            purged_parent_ids: list[str] = []
            refresh_parent_ids: list[str] = []
            for parent_type, parent_id in parent_refs:
                remaining = self._conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM chunks
                    WHERE user_id = ? AND parent_id = ?
                    """,
                    (user_id, parent_id),
                ).fetchone()
                if int(remaining["count"] or 0) > 0:
                    refresh_parent_ids.append(parent_id)
                    continue
                if parent_type == "turn":
                    cur = self._conn.execute(
                        "DELETE FROM raw_turns WHERE user_id = ? AND id = ?",
                        (user_id, parent_id),
                    )
                elif parent_type == "capture":
                    if preserve_capture_parents:
                        continue
                    cur = self._conn.execute(
                        "DELETE FROM captures WHERE user_id = ? AND id = ?",
                        (user_id, parent_id),
                    )
                else:
                    continue
                if cur.rowcount > 0:
                    purged_parent_ids.append(parent_id)
            self._update_parent_status_for_chunks(user_id, refresh_parent_ids)
            self._conn.commit()
            return TimelinePurgeResult(
                purged_chunk_ids=purged_chunk_ids,
                purged_parent_ids=purged_parent_ids,
            )

    def _search_chunks_like(
        self,
        user_id: str,
        query: str,
        limit: int,
        *,
        exclude_parent_id: str = "",
    ) -> list[sqlite3.Row]:
        terms = self._fallback_terms(query)
        if not terms:
            return []
        params: list[Any] = [user_id]
        clauses = []
        exclude_clause = ""
        if exclude_parent_id:
            exclude_clause = "AND parent_id != ?"
            params.append(exclude_parent_id)
        for term in terms:
            clauses.append("text LIKE ?")
            params.append(f"%{term}%")
        params.append(max(limit * 4, limit))
        rows = self._conn.execute(
            f"""
            SELECT id, user_id, parent_type, parent_id, text, chunk_index,
                   start_offset, end_offset, timestamp, source, metadata, status
            FROM chunks
            WHERE user_id = ? AND deleted_at IS NULL AND status = 'active'
              {exclude_clause}
              AND ({" OR ".join(clauses)})
            ORDER BY timestamp DESC, chunk_index ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return sorted(
            rows,
            key=lambda row: (_match_score(str(row["text"] or ""), terms), float(row["timestamp"] or 0)),
            reverse=True,
        )[:limit]

    # Timeline 排序只暴露分数和 id，不把原文片段复制进 debug。
    def _rank_chunk_search_results(
        self,
        chunks: list[TimelineChunk],
        query: str,
        limit: int,
    ) -> TimelineSearchResult:
        terms = self._fallback_terms(query)
        now = time.time()
        ranked = []
        for chunk in chunks:
            text_score = _normalized_match_score(chunk.text, terms)
            recency_score = _timeline_recency_score(chunk.timestamp, now)
            total_score = 0.75 * text_score + 0.25 * recency_score
            ranking = {
                "id": chunk.id,
                "total_score": round(total_score, 4),
                "text_score": round(text_score, 4),
                "recency_score": round(recency_score, 4),
                "reason": "text_recency",
            }
            ranked.append((total_score, text_score, recency_score, chunk.timestamp, chunk, ranking))
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        selected = ranked[:limit]
        return TimelineSearchResult(
            chunks=[item[4] for item in selected],
            ranking=[item[5] for item in selected],
            candidate_ranking=[item[5] for item in ranked],
        )

    @staticmethod
    def _chunk_text(text: str, *, max_chars: int = 900) -> list[dict[str, Any]]:
        cleaned = str(text or "").strip()
        if not cleaned:
            return []
        chunks: list[dict[str, Any]] = []
        current = ""
        current_start = 0
        cursor = 0
        for part in _split_text_units(cleaned):
            start = cleaned.find(part, cursor)
            if start < 0:
                start = cursor
            end = start + len(part)
            if current and len(current) + len(part) + 1 > max_chars:
                chunks.append({
                    "text": current.strip(),
                    "start_offset": current_start,
                    "end_offset": cursor,
                })
                current = part
                current_start = start
            else:
                if not current:
                    current_start = start
                    current = part
                else:
                    current = f"{current}\n{part}" if "\n" in part else f"{current} {part}"
            cursor = end
        if current.strip():
            chunks.append({
                "text": current.strip(),
                "start_offset": current_start,
                "end_offset": min(len(cleaned), cursor),
            })
        if len(chunks) == 1 and len(chunks[0]["text"]) <= max_chars:
            return chunks
        return _split_oversized_chunks(chunks, max_chars=max_chars)

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = [part.strip().replace('"', '""') for part in query.split() if part.strip()]
        return " OR ".join(f'"{term}"' for term in terms) or '""'

    @staticmethod
    def _fallback_terms(query: str) -> list[str]:
        text = re.sub(r"[？?！!，,。；;：:\[\]()（）\"']", " ", str(query or ""))
        terms = [part for part in re.split(r"\s+", text.strip()) if part]
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text):
            if token not in terms:
                terms.append(token)
            if re.fullmatch(r"[\u4e00-\u9fff]{5,}", token):
                for size in (4, 3):
                    for idx in range(0, max(0, len(token) - size + 1)):
                        fragment = token[idx:idx + size]
                        if fragment not in terms:
                            terms.append(fragment)
        filtered = [term for term in terms if _is_useful_search_term(term)]
        return (filtered or terms)[:12]

    def _update_parent_status_for_chunks(self, user_id: str, parent_ids: list[str]) -> None:
        for parent_id in list(dict.fromkeys(parent_id for parent_id in parent_ids if parent_id)):
            row = self._conn.execute(
                """
                SELECT parent_type,
                       SUM(CASE WHEN deleted_at IS NULL AND status = 'active' THEN 1 ELSE 0 END) AS active_count,
                       COUNT(*) AS total_count
                FROM chunks
                WHERE user_id = ? AND parent_id = ?
                GROUP BY parent_type
                """,
                (user_id, parent_id),
            ).fetchone()
            if row is None:
                continue
            active_count = int(row["active_count"] or 0)
            total_count = int(row["total_count"] or 0)
            if active_count == total_count:
                status = "active"
                deleted_at = None
            elif active_count == 0:
                status = "deleted"
                deleted_at = time.time()
            else:
                status = "partial_deleted"
                deleted_at = None
            if row["parent_type"] == "turn":
                self._conn.execute(
                    """
                    UPDATE raw_turns
                    SET status = ?, deleted_at = ?, updated_at = ?
                    WHERE user_id = ? AND id = ?
                    """,
                    (status, deleted_at, time.time(), user_id, parent_id),
                )
            elif row["parent_type"] == "capture":
                self._conn.execute(
                    """
                    UPDATE captures
                    SET status = ?, deleted_at = ?, updated_at = ?
                    WHERE user_id = ? AND id = ?
                    """,
                    (status, deleted_at, time.time(), user_id, parent_id),
                )

    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> TimelineTurn:
        created_at = float(row["created_at"])
        return TimelineTurn(
            id=row["id"],
            user_id=row["user_id"],
            source=str(row["source"] or ""),
            raw_text=str(row["raw_text"] or ""),
            assistant_reply=str(row["assistant_reply"] or ""),
            interaction_id=str(row["interaction_id"] or ""),
            legacy_session_id=str(row["legacy_session_id"] or ""),
            created_at=created_at,
            updated_at=float(row["updated_at"]) if row["updated_at"] else created_at,
            status=str(row["status"] or "active"),
        )

    @staticmethod
    def _metadata_from_json(value: str) -> dict[str, Any]:
        try:
            metadata = json.loads(value or "{}")
        except json.JSONDecodeError:
            metadata = {}
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> TimelineChunk:
        metadata = TimelineStore._metadata_from_json(str(row["metadata"] or "{}"))
        return TimelineChunk(
            id=row["id"],
            user_id=row["user_id"],
            parent_type=str(row["parent_type"] or ""),
            parent_id=str(row["parent_id"] or ""),
            text=str(row["text"] or ""),
            chunk_index=int(row["chunk_index"] or 0),
            start_offset=int(row["start_offset"] or 0),
            end_offset=int(row["end_offset"] or 0),
            timestamp=float(row["timestamp"]),
            source=str(row["source"] or ""),
            metadata=metadata if isinstance(metadata, dict) else {},
            status=str(row["status"] or "active"),
        )


def chunk_to_dict(chunk: TimelineChunk) -> dict[str, Any]:
    public_metadata = _public_timeline_metadata(dict(chunk.metadata or {}))
    return {
        "id": chunk.id,
        "user_id": chunk.user_id,
        "parent_type": chunk.parent_type,
        "parent_id": chunk.parent_id,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "timestamp": chunk.timestamp,
        "source": chunk.source,
        "metadata": public_metadata,
        "status": chunk.status,
    }


def _public_timeline_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_timeline_metadata(item)
            for key, item in value.items()
            if str(key).casefold() not in {"speaker_embedding", "embedding"}
        }
    if isinstance(value, list):
        return [_public_timeline_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_public_timeline_metadata(item) for item in value]
    return value


def _redaction_metadata(redaction: RedactionResult) -> dict[str, Any]:
    return {
        "redacted": redaction.redacted,
        "redaction_categories": redaction.categories,
        "redaction_count": redaction.count,
    }


def _merge_redaction_metadata(metadata: dict[str, Any] | None, redaction: RedactionResult) -> dict[str, Any]:
    merged = dict(metadata or {})
    if redaction.redacted or "redacted" not in merged:
        merged.update(_redaction_metadata(redaction))
    return merged


def _json_list(value: Any) -> list[Any]:
    try:
        payload = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _json_object(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_string_values(
    existing: list[Any],
    incoming: list[Any],
    *,
    limit: int | None = 100,
) -> list[str]:
    values = list(dict.fromkeys(
        str(item).strip() for item in [*existing, *incoming] if str(item).strip()
    ))
    return values if limit is None else values[:limit]


def _merge_time_spans(existing: list[Any], incoming: dict[str, float]) -> list[dict[str, float]]:
    spans = [
        {"start_at": float(item["start_at"]), "end_at": float(item["end_at"])}
        for item in existing
        if isinstance(item, dict) and item.get("start_at") is not None and item.get("end_at") is not None
    ]
    spans.append({"start_at": float(incoming["start_at"]), "end_at": float(incoming["end_at"])})
    spans.sort(key=lambda item: (item["start_at"], item["end_at"]))
    return list({(item["start_at"], item["end_at"]): item for item in spans}.values())


_LOW_INFORMATION_TERMS = {
    "问题",
    "事情",
    "这个",
    "那个",
    "一下",
    "之前",
    "有没有",
    "是否",
    "什么",
    "怎么",
    "如何",
    "相关",
    "提到",
    "说过",
    "找一下",
}


def _is_useful_search_term(term: str) -> bool:
    cleaned = str(term or "").strip()
    if not cleaned or cleaned in _LOW_INFORMATION_TERMS:
        return False
    if re.fullmatch(r"[\u4e00-\u9fff]{1,2}", cleaned) and cleaned in _LOW_INFORMATION_TERMS:
        return False
    return True


def _match_score(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term and term in text)


def _normalized_match_score(text: str, terms: list[str]) -> float:
    useful_terms = [term for term in terms if term]
    if not useful_terms:
        return 0.0
    return min(1.0, _match_score(text, useful_terms) / len(useful_terms))


def _timeline_recency_score(timestamp: float | None, now: float) -> float:
    if timestamp is None:
        return 0.0
    age_seconds = max(0.0, now - float(timestamp))
    return max(0.0, min(1.0, 1.0 - (age_seconds / (30 * 24 * 60 * 60))))


def _split_text_units(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        return []
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= 900:
            units.append(paragraph)
            continue
        pieces = [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*", paragraph) if part.strip()]
        units.extend(pieces or [paragraph])
    return units


def _split_oversized_chunks(chunks: list[dict[str, Any]], *, max_chars: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        start_offset = int(chunk.get("start_offset") or 0)
        if len(text) <= max_chars:
            result.append(chunk)
            continue
        cursor = 0
        while cursor < len(text):
            piece = text[cursor:cursor + max_chars].strip()
            if piece:
                result.append({
                    "text": piece,
                    "start_offset": start_offset + cursor,
                    "end_offset": start_offset + cursor + len(piece),
                })
            cursor += max_chars
    return result
