from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app_home import get_data_dir
from .privacy_filter import RedactionResult, redact_sensitive_text


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
                CREATE INDEX IF NOT EXISTS idx_memory_jobs_user_updated
                    ON memory_jobs(user_id, updated_at DESC);
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
    ) -> bool:
        now = ended_at if ended_at is not None else time.time()
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE captures
                SET ended_at = ?, summary = ?, updated_at = ?, status = 'stopped'
                WHERE user_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (now, str(summary or "").strip(), now, user_id, capture_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

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

    def list_chunk_references(
        self,
        user_id: str,
        chunk_ids: list[str],
        reference_counts: dict[str, Any],
        *,
        limit: int = 100,
        include_deleted: bool = True,
    ) -> list[TimelineChunkReference]:
        chunks = self.list_chunks_by_ids(
            user_id,
            chunk_ids,
            limit=limit,
            include_deleted=include_deleted,
        )
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

    def purge_chunks(self, user_id: str, chunk_ids: list[str]) -> TimelinePurgeResult:
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
