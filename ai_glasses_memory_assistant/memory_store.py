from __future__ import annotations

import json
import hashlib
import math
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app_home import get_data_dir
from .memory_evidence import EvidenceReferenceCounts, normalize_evidence_ids
from .memory_lifecycle import (
    ACTIVE_MEMORY_STATUS,
    SUPERSEDED_MEMORY_STATUS,
    STALE_MEMORY_STATUS,
    can_transition_memory_status,
    is_active_memory_status,
    normalize_memory_status,
)
from .memory_kernel import source_trace
from .memory_subject_identity import normalize_subject_name, subject_name_key

_CJK_ACTION_OBJECT_PATTERN = re.compile(r"(?P<action>吃|喝|买|看|开|聊|做|去|约|取|交|写|改|补)(?:了|过|一下)?(?P<object>[\u4e00-\u9fffA-Za-z0-9]{1,8})")
_MEMORY_SUBJECT_TYPES = {"self", "named", "provisional"}


@dataclass(frozen=True)
class MemorySubject:
    id: str
    user_id: str
    subject_type: str
    display_name: str
    canonical_name: str
    source_scope: str
    created_at: float
    updated_at: float
    merged_into: str = ""


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    user_id: str
    subject_id: str
    embedding_model: str
    embedding_dim: int
    embedding: tuple[float, ...]
    confidence: float | None
    source_id: str
    created_at: float


def default_data_dir() -> Path:
    return get_data_dir()


@dataclass(frozen=True)
class LegacyClassification:
    value: str
    source: str
    reason: str
    role: str = "legacy_fallback"
    marker: str = ""

    def debug_payload(self, field: str) -> dict[str, str]:
        payload = {
            "field": field,
            "value": self.value,
            "source": self.source,
            "role": self.role,
            "reason": self.reason,
        }
        if self.marker:
            payload["marker"] = self.marker
        return payload


@dataclass(frozen=True)
class MemoryEvent:
    id: str
    user_id: str
    kind: str
    memory_type: str
    content: str
    tags: list[str]
    source: str
    source_id: str
    ingestion_id: str
    evidence_ids: list[str]
    subject_id: str
    subject_type: str
    subject_name: str
    created_at: float
    updated_at: float
    occurred_at: float | None
    start_at: float | None = None
    end_at: float | None = None
    time_granularity: str = "unknown"
    temporal_text: str = ""
    temporal_confidence: float | None = None
    privacy_level: str = "normal"
    status: str = "active"
    confidence: float | None = None
    superseded_by: str = ""
    access_count: int = 0
    last_accessed_at: float | None = None
    strength: float = 0.0


@dataclass(frozen=True)
class MemorySearchResult:
    memories: list[MemoryEvent]
    ranking: list[dict[str, Any]]


@dataclass(frozen=True)
class StrengthPolicy:
    cap: float
    decay_days: float
    floor: float
    reason: str


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    user_id: str
    filename: str
    title: str
    summary: str
    content: str
    content_hash: str
    source: str
    ingestion_id: str
    created_at: float
    updated_at: float
    status: str = "active"


class EventMemoryStore:
    """Small event-memory store for the stage-1 AI glasses MVP.

    This is intentionally separate from Hermes's curated MEMORY.md/USER.md
    store. AI-glasses memory needs timestamped life events that can be viewed,
    searched, and deleted independently from the agent's own operating notes.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        data_dir = default_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (data_dir / "events.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._subject_lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # schema 初始化同时兼容旧库补列，避免开发中的本地数据被迁移打断。
    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject_id TEXT NOT NULL DEFAULT '',
                subject_type TEXT NOT NULL DEFAULT 'self',
                subject_name TEXT NOT NULL DEFAULT '我',
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
            CREATE TABLE IF NOT EXISTS memory_subjects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                display_name TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                source_scope TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                merged_into TEXT NOT NULL DEFAULT '',
                deleted_at REAL
            );
            CREATE TABLE IF NOT EXISTS memory_subject_aliases (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                source_scope TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_voice_profiles (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                embedding_json TEXT NOT NULL,
                confidence REAL,
                source_id TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                ingestion_id TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                deleted_at REAL
            );
            """
        )
        self._ensure_column("memories", "kind", "TEXT NOT NULL DEFAULT 'event'")
        self._ensure_column("memories", "subject_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("memories", "subject_type", "TEXT NOT NULL DEFAULT 'self'")
        self._ensure_column("memories", "subject_name", "TEXT NOT NULL DEFAULT '我'")
        self._ensure_column("memories", "memory_type", "TEXT NOT NULL DEFAULT 'event'")
        self._ensure_column("memories", "occurred_at", "REAL")
        self._ensure_column("memories", "start_at", "REAL")
        self._ensure_column("memories", "end_at", "REAL")
        self._ensure_column("memories", "time_granularity", "TEXT NOT NULL DEFAULT 'unknown'")
        self._ensure_column("memories", "temporal_text", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("memories", "temporal_confidence", "REAL")
        self._ensure_column("memories", "source_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("memories", "ingestion_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("memories", "evidence_ids", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("memories", "updated_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("memories", "privacy_level", "TEXT NOT NULL DEFAULT 'normal'")
        self._ensure_column("memories", "status", "TEXT NOT NULL DEFAULT 'active'")
        self._ensure_column("memories", "confidence", "REAL")
        self._ensure_column("memories", "superseded_by", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("memories", "access_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("memories", "last_accessed_at", "REAL")
        self._ensure_column("memories", "strength", "REAL NOT NULL DEFAULT 0")
        # Subject creation and legacy-row assignment must commit or roll back together.
        with self._conn:
            self._backfill_memory_subjects()
        try:
            self._backfill_memory_strength()
        except sqlite3.DatabaseError:
            # 启动期迁移是历史数据增强，不能让损坏/只读旧库阻断 demo 基础聊天。
            pass
        self._ensure_column("documents", "updated_at", "REAL NOT NULL DEFAULT 0")
        self._ensure_column("documents", "status", "TEXT NOT NULL DEFAULT 'active'")
        self._ensure_column("documents", "deleted_at", "REAL")
        self._conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_user_created
                ON memories(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_user_subject_created
                ON memories(user_id, subject_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_user_kind_created
                ON memories(user_id, kind, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_user_kind_start
                ON memories(user_id, kind, start_at);
            CREATE INDEX IF NOT EXISTS idx_memories_user_type_created
                ON memories(user_id, memory_type, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_user_status_created
                ON memories(user_id, status, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_user_ingestion
                ON memories(user_id, ingestion_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_subjects_user_self
                ON memory_subjects(user_id)
                WHERE subject_type = 'self' AND merged_into = '' AND deleted_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_memory_subjects_user_type
                ON memory_subjects(user_id, subject_type, created_at);
            CREATE INDEX IF NOT EXISTS idx_memory_subject_alias_lookup
                ON memory_subject_aliases(user_id, normalized_alias, source_scope);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_subject_alias_unique
                ON memory_subject_aliases(user_id, subject_id, normalized_alias, source_scope);
            CREATE INDEX IF NOT EXISTS idx_memory_voice_profiles_subject
                ON memory_voice_profiles(user_id, subject_id, embedding_model, embedding_dim);
            CREATE INDEX IF NOT EXISTS idx_documents_user_created
                ON documents(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_documents_user_ingestion
                ON documents(user_id, ingestion_id);
            CREATE INDEX IF NOT EXISTS idx_documents_user_hash
                ON documents(user_id, content_hash);
            """
        )
        memories_fts_exists = self._table_exists("memories_fts")
        documents_fts_exists = self._table_exists("documents_fts")
        # FTS5 只作为搜索加速；不可用时仍保留 LIKE fallback。
        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    tags,
                    content='memories',
                    content_rowid='rowid'
                )
                """
            )
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    title,
                    summary,
                    content,
                    filename,
                    content='documents',
                    content_rowid='rowid'
                )
                """
            )
            self._conn.executescript(
                """
                DROP TRIGGER IF EXISTS memories_fts_insert;
                DROP TRIGGER IF EXISTS memories_fts_delete;
                DROP TRIGGER IF EXISTS memories_fts_update;
                DROP TRIGGER IF EXISTS documents_fts_insert;
                DROP TRIGGER IF EXISTS documents_fts_delete;
                DROP TRIGGER IF EXISTS documents_fts_update;

                CREATE TRIGGER IF NOT EXISTS memories_fts_insert
                AFTER INSERT ON memories WHEN new.deleted_at IS NULL BEGIN
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.rowid, new.content, new.tags);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_fts_delete
                AFTER DELETE ON memories WHEN old.deleted_at IS NULL BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.rowid, old.content, old.tags);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_fts_update
                AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    SELECT 'delete', old.rowid, old.content, old.tags
                    WHERE old.deleted_at IS NULL;
                    INSERT INTO memories_fts(rowid, content, tags)
                    SELECT new.rowid, new.content, new.tags
                    WHERE new.deleted_at IS NULL;
                END;

                CREATE TRIGGER IF NOT EXISTS documents_fts_insert
                AFTER INSERT ON documents WHEN new.deleted_at IS NULL BEGIN
                    INSERT INTO documents_fts(rowid, title, summary, content, filename)
                    VALUES (new.rowid, new.title, new.summary, new.content, new.filename);
                END;

                CREATE TRIGGER IF NOT EXISTS documents_fts_delete
                AFTER DELETE ON documents WHEN old.deleted_at IS NULL BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, title, summary, content, filename)
                    VALUES ('delete', old.rowid, old.title, old.summary, old.content, old.filename);
                END;

                CREATE TRIGGER IF NOT EXISTS documents_fts_update
                AFTER UPDATE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, title, summary, content, filename)
                    SELECT 'delete', old.rowid, old.title, old.summary, old.content, old.filename
                    WHERE old.deleted_at IS NULL;
                    INSERT INTO documents_fts(rowid, title, summary, content, filename)
                    SELECT new.rowid, new.title, new.summary, new.content, new.filename
                    WHERE new.deleted_at IS NULL;
                END;
                """
            )
            # External-content FTS tables only need a rebuild when first created over legacy rows.
            if not memories_fts_exists:
                self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            if not documents_fts_exists:
                self._conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        except sqlite3.OperationalError:
            # SQLite builds without FTS5 still work through LIKE fallback.
            pass
        self._conn.commit()

    # 增量补列用于本地 demo 数据平滑升级，不做破坏性迁移。
    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        existing = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def _table_exists(self, table: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None

    # 历史库只有 user_id；升级时为每个用户创建唯一的“我”，再把旧记忆归到该主体。
    def _backfill_memory_subjects(self) -> None:
        users = self._conn.execute(
            "SELECT DISTINCT user_id FROM memories WHERE TRIM(user_id) != ''"
        ).fetchall()
        for row in users:
            user_id = str(row["user_id"])
            subject = self._ensure_self_subject_row(user_id)
            self._conn.execute(
                """
                UPDATE memories
                SET subject_id = ?, subject_type = ?, subject_name = ?
                WHERE user_id = ? AND (
                    subject_id IS NULL
                    OR TRIM(subject_id) = ''
                    OR NOT EXISTS (
                        SELECT 1
                        FROM memory_subjects AS subject
                        WHERE subject.id = memories.subject_id
                          AND subject.user_id = memories.user_id
                          AND subject.deleted_at IS NULL
                    )
                )
                """,
                (subject.id, subject.subject_type, subject.display_name, user_id),
            )
        self._conn.execute(
            """
            UPDATE memories
            SET subject_type = COALESCE(
                    (
                        SELECT subject_type
                        FROM memory_subjects
                        WHERE id = memories.subject_id AND user_id = memories.user_id
                    ),
                    subject_type
                ),
                subject_name = COALESCE(
                    (
                        SELECT display_name
                        FROM memory_subjects
                        WHERE id = memories.subject_id AND user_id = memories.user_id
                    ),
                    subject_name
                )
            WHERE subject_id != ''
            """
        )

    def ensure_self_subject(self, user_id: str, *, display_name: str = "我") -> MemorySubject:
        with self._subject_lock:
            subject = self._ensure_self_subject_row(user_id, display_name=display_name)
            self._conn.commit()
            return subject

    def _ensure_self_subject_row(self, user_id: str, *, display_name: str = "我") -> MemorySubject:
        user_id = _require_nonempty(user_id, "user_id")
        row = self._conn.execute(
            """
            SELECT id, user_id, subject_type, display_name, canonical_name,
                   source_scope, created_at, updated_at, merged_into
            FROM memory_subjects
            WHERE user_id = ? AND subject_type = 'self'
              AND merged_into = '' AND deleted_at IS NULL
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if row:
            return self._row_to_subject(row)
        name = normalize_subject_name(display_name or "我") or "我"
        now = time.time()
        subject = MemorySubject(
            id=uuid.uuid4().hex,
            user_id=user_id,
            subject_type="self",
            display_name=name,
            canonical_name=subject_name_key(name),
            source_scope="",
            created_at=now,
            updated_at=now,
        )
        self._insert_subject(subject)
        self._insert_subject_alias(subject, name, source_scope="")
        return subject

    def create_named_subject(self, user_id: str, display_name: str) -> MemorySubject:
        user_id = _require_nonempty(user_id, "user_id")
        display_name = _require_nonempty(normalize_subject_name(display_name), "display_name")
        with self._subject_lock:
            existing = self.resolve_subject(user_id, display_name, source_scope="", subject_types={"named"})
            if existing is not None:
                return existing
            subject = self._new_subject(
                user_id,
                subject_type="named",
                display_name=display_name,
                source_scope="",
            )
            self._conn.commit()
            return subject

    def create_provisional_subject(
        self,
        user_id: str,
        display_name: str,
        *,
        source_scope: str,
    ) -> MemorySubject:
        user_id = _require_nonempty(user_id, "user_id")
        display_name = _require_nonempty(normalize_subject_name(display_name), "display_name")
        source_scope = _require_nonempty(source_scope, "source_scope")
        with self._subject_lock:
            existing = self.resolve_subject(user_id, display_name, source_scope=source_scope)
            if existing is not None:
                return existing
            subject = self._new_subject(
                user_id,
                subject_type="provisional",
                display_name=display_name,
                source_scope=source_scope,
            )
            self._conn.commit()
            return subject

    def _new_subject(
        self,
        user_id: str,
        *,
        subject_type: str,
        display_name: str,
        source_scope: str,
    ) -> MemorySubject:
        subject_type = _normalize_subject_type(subject_type)
        now = time.time()
        subject = MemorySubject(
            id=uuid.uuid4().hex,
            user_id=user_id,
            subject_type=subject_type,
            display_name=display_name,
            canonical_name=subject_name_key(display_name),
            source_scope=source_scope,
            created_at=now,
            updated_at=now,
        )
        self._insert_subject(subject)
        self._insert_subject_alias(subject, display_name, source_scope=source_scope)
        return subject

    def _insert_subject(self, subject: MemorySubject) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_subjects (
                id, user_id, subject_type, display_name, canonical_name,
                source_scope, created_at, updated_at, merged_into
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject.id,
                subject.user_id,
                subject.subject_type,
                subject.display_name,
                subject.canonical_name,
                subject.source_scope,
                subject.created_at,
                subject.updated_at,
                subject.merged_into,
            ),
        )

    def resolve_subject(
        self,
        user_id: str,
        name: str,
        *,
        source_scope: str | None = None,
        subject_types: set[str] | None = None,
        include_scoped_aliases: bool = False,
    ) -> MemorySubject | None:
        normalized_name = subject_name_key(name)
        if not normalized_name:
            return None
        allowed_types = {_normalize_subject_type(item) for item in subject_types} if subject_types else None
        rows = self._conn.execute(
            """
            SELECT s.id, s.user_id, s.subject_type, s.display_name, s.canonical_name,
                   s.source_scope, s.created_at, s.updated_at, s.merged_into,
                   a.source_scope AS alias_scope
            FROM memory_subject_aliases a
            JOIN memory_subjects s ON s.id = a.subject_id AND s.user_id = a.user_id
            WHERE a.user_id = ? AND a.normalized_alias = ?
              AND s.merged_into = '' AND s.deleted_at IS NULL
            ORDER BY s.created_at ASC
            """,
            (user_id, normalized_name),
        ).fetchall()
        candidates = [
            (self._row_to_subject(row), str(row["alias_scope"] or ""))
            for row in rows
            if allowed_types is None or str(row["subject_type"]) in allowed_types
        ]
        if source_scope is not None:
            exact = [subject for subject, alias_scope in candidates if alias_scope == source_scope]
            exact = _unique_subjects(exact)
            if len(exact) == 1:
                return exact[0]
            if len(exact) > 1:
                return None
            global_subjects = _unique_subjects(
                [subject for subject, alias_scope in candidates if alias_scope == ""]
            )
            return global_subjects[0] if len(global_subjects) == 1 else None
        if include_scoped_aliases:
            subjects = _unique_subjects([subject for subject, _ in candidates])
            return subjects[0] if len(subjects) == 1 else None
        # 没有采集范围时只解析全局别名，避免把上一次录音的 speaker_2 带到下一次录音。
        subjects = _unique_subjects(
            [subject for subject, alias_scope in candidates if alias_scope == ""]
        )
        return subjects[0] if len(subjects) == 1 else None

    def get_subject(self, user_id: str, subject_id: str) -> MemorySubject | None:
        subject = self._get_subject(user_id, subject_id)
        if subject is None:
            return None
        seen: set[str] = set()
        while subject.merged_into and subject.id not in seen:
            seen.add(subject.id)
            next_subject = self._get_subject(user_id, subject.merged_into)
            if next_subject is None:
                break
            subject = next_subject
        return subject

    def _get_subject(self, user_id: str, subject_id: str) -> MemorySubject | None:
        row = self._conn.execute(
            """
            SELECT id, user_id, subject_type, display_name, canonical_name,
                   source_scope, created_at, updated_at, merged_into
            FROM memory_subjects
            WHERE user_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (user_id, subject_id),
        ).fetchone()
        return self._row_to_subject(row) if row else None

    def list_subjects(self, user_id: str, *, include_merged: bool = False) -> list[MemorySubject]:
        merged_clause = "" if include_merged else "AND merged_into = ''"
        rows = self._conn.execute(
            f"""
            SELECT id, user_id, subject_type, display_name, canonical_name,
                   source_scope, created_at, updated_at, merged_into
            FROM memory_subjects
            WHERE user_id = ? AND deleted_at IS NULL {merged_clause}
            ORDER BY CASE subject_type WHEN 'self' THEN 0 WHEN 'named' THEN 1 ELSE 2 END,
                     created_at ASC
            """,
            (user_id,),
        ).fetchall()
        return [self._row_to_subject(row) for row in rows]

    def add_subject_alias(
        self,
        user_id: str,
        subject_id: str,
        alias: str,
        *,
        source_scope: str | None = None,
    ) -> MemorySubject:
        subject = self.get_subject(user_id, subject_id)
        if subject is None:
            raise ValueError("subject does not exist for user")
        alias = _require_nonempty(alias, "alias")
        alias_scope = subject.source_scope if source_scope is None else str(source_scope).strip()
        self._insert_subject_alias(subject, alias, source_scope=alias_scope)
        self._conn.commit()
        return subject

    def _insert_subject_alias(self, subject: MemorySubject, alias: str, *, source_scope: str) -> None:
        normalized = subject_name_key(alias)
        if not normalized:
            raise ValueError("alias cannot be empty")
        self._conn.execute(
            """
            INSERT OR IGNORE INTO memory_subject_aliases (
                id, user_id, subject_id, alias, normalized_alias, source_scope, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                subject.user_id,
                subject.id,
                alias.strip(),
                normalized,
                source_scope,
                time.time(),
            ),
        )

    def merge_subjects(
        self,
        user_id: str,
        source_subject_id: str,
        target_subject_id: str,
    ) -> MemorySubject:
        source = self._get_subject(user_id, source_subject_id)
        target = self.get_subject(user_id, target_subject_id)
        if source is None or target is None:
            raise ValueError("source and target subjects must exist for user")
        canonical_source = self.get_subject(user_id, source.id)
        if canonical_source is None:
            raise ValueError("source subject cannot be resolved")
        if canonical_source.id == target.id:
            return target
        if source.merged_into:
            raise ValueError("source subject has already been merged")
        if source.subject_type == "self":
            raise ValueError("self subject cannot be merged into another subject")
        now = time.time()
        source_aliases = self._conn.execute(
            """
            SELECT alias, source_scope
            FROM memory_subject_aliases
            WHERE user_id = ? AND subject_id = ?
            """,
            (user_id, source.id),
        ).fetchall()
        with self._conn:
            self._conn.execute(
                """
                UPDATE memories
                SET subject_id = ?, subject_type = ?, subject_name = ?, updated_at = ?
                WHERE user_id = ? AND subject_id = ?
                """,
                (target.id, target.subject_type, target.display_name, now, user_id, source.id),
            )
            for alias_row in source_aliases:
                self._insert_subject_alias(
                    target,
                    str(alias_row["alias"]),
                    source_scope=str(alias_row["source_scope"] or ""),
                )
            self._conn.execute(
                "DELETE FROM memory_subject_aliases WHERE user_id = ? AND subject_id = ?",
                (user_id, source.id),
            )
            self._conn.execute(
                """
                DELETE FROM memory_voice_profiles
                WHERE user_id = ? AND subject_id = ? AND TRIM(source_id) != ''
                  AND EXISTS (
                      SELECT 1
                      FROM memory_voice_profiles AS target_profile
                      WHERE target_profile.user_id = memory_voice_profiles.user_id
                        AND target_profile.subject_id = ?
                        AND target_profile.embedding_model = memory_voice_profiles.embedding_model
                        AND target_profile.embedding_dim = memory_voice_profiles.embedding_dim
                        AND target_profile.source_id = memory_voice_profiles.source_id
                  )
                """,
                (user_id, source.id, target.id),
            )
            self._conn.execute(
                """
                UPDATE memory_voice_profiles
                SET subject_id = ?
                WHERE user_id = ? AND subject_id = ? AND embedding_dim > 0
                  AND TRIM(embedding_model) != ''
                """,
                (target.id, user_id, source.id),
            )
            self._conn.execute(
                """
                UPDATE memory_subjects
                SET merged_into = ?, updated_at = ?
                WHERE user_id = ? AND id = ? AND merged_into = ''
                """,
                (target.id, now, user_id, source.id),
            )
        return target

    def store_voice_profile(
        self,
        user_id: str,
        subject_id: str,
        *,
        embedding: list[float] | tuple[float, ...],
        embedding_model: str,
        confidence: float | None = None,
        source_id: str = "",
        created_at: float | None = None,
    ) -> VoiceProfile:
        subject = self.get_subject(user_id, subject_id)
        if subject is None:
            raise ValueError("subject does not exist for user")
        model = _require_nonempty(normalize_subject_name(embedding_model).casefold(), "embedding_model")
        vector = _normalize_embedding(embedding)
        confidence = _normalize_optional_confidence(confidence)
        normalized_source_id = str(source_id or "").strip()
        if normalized_source_id:
            existing = self._conn.execute(
                """
                SELECT id, user_id, subject_id, embedding_model, embedding_dim,
                       embedding_json, confidence, source_id, created_at
                FROM memory_voice_profiles
                WHERE user_id = ? AND subject_id = ? AND embedding_model = ?
                  AND embedding_dim = ? AND source_id = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (user_id, subject.id, model, len(vector), normalized_source_id),
            ).fetchone()
            if existing is not None:
                return self._row_to_voice_profile(existing)
        profile = VoiceProfile(
            id=uuid.uuid4().hex,
            user_id=user_id,
            subject_id=subject.id,
            embedding_model=model,
            embedding_dim=len(vector),
            embedding=vector,
            confidence=confidence,
            source_id=normalized_source_id,
            created_at=time.time() if created_at is None else float(created_at),
        )
        self._conn.execute(
            """
            INSERT INTO memory_voice_profiles (
                id, user_id, subject_id, embedding_model, embedding_dim,
                embedding_json, confidence, source_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.id,
                profile.user_id,
                profile.subject_id,
                profile.embedding_model,
                profile.embedding_dim,
                json.dumps(profile.embedding),
                profile.confidence,
                profile.source_id,
                profile.created_at,
            ),
        )
        self._conn.commit()
        return profile

    def list_voice_profiles(
        self,
        user_id: str,
        *,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> list[VoiceProfile]:
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        normalized_ids = self._canonical_subject_ids(user_id, subject_ids)
        if subject_ids is not None:
            if not normalized_ids:
                return []
            clauses.append(f"subject_id IN ({', '.join('?' for _ in normalized_ids)})")
            params.extend(normalized_ids)
        if embedding_model is not None:
            clauses.append("embedding_model = ?")
            params.append(normalize_subject_name(embedding_model).casefold())
        if embedding_dim is not None:
            clauses.append("embedding_dim = ?")
            params.append(int(embedding_dim))
        rows = self._conn.execute(
            f"""
            SELECT id, user_id, subject_id, embedding_model, embedding_dim,
                   embedding_json, confidence, source_id, created_at
            FROM memory_voice_profiles
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at ASC
            """,
            tuple(params),
        ).fetchall()
        return [self._row_to_voice_profile(row) for row in rows]

    # 旧库补列后 strength 默认为 0，这里按现有置信度/类型补一次初始值。
    def _backfill_memory_strength(self) -> None:
        rows = self._conn.execute(
            """
            SELECT id, memory_type, confidence
            FROM memories
            WHERE strength IS NULL OR strength <= 0
            """
        ).fetchall()
        if not rows:
            return
        for row in rows:
            self._conn.execute(
                """
                UPDATE memories
                SET strength = ?
                WHERE id = ?
                """,
                (
                    _initial_memory_strength(
                        str(row["memory_type"] or ""),
                        float(row["confidence"]) if row["confidence"] is not None else None,
                    ),
                    row["id"],
                ),
            )

    # 文档归档保存完整原文，摘要只用于识别，不替代后续细节召回。
    def add_document(
        self,
        user_id: str,
        *,
        filename: str,
        title: str,
        summary: str,
        content: str,
        source: str,
        ingestion_id: str,
        created_at: float | None = None,
    ) -> DocumentRecord:
        content = str(content or "").strip()
        if not content:
            raise ValueError("document content cannot be empty")
        filename = str(filename or "uploaded.md").strip() or "uploaded.md"
        title = str(title or filename).strip() or filename
        summary = str(summary or title).strip() or title
        now = created_at if created_at is not None else time.time()
        document = DocumentRecord(
            id=uuid.uuid4().hex,
            user_id=user_id,
            filename=filename,
            title=title,
            summary=summary,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            source=str(source or "").strip(),
            ingestion_id=str(ingestion_id or "").strip(),
            created_at=now,
            updated_at=now,
            status=ACTIVE_MEMORY_STATUS,
        )
        self._conn.execute(
            """
            INSERT INTO documents (
                id, user_id, filename, title, summary, content, content_hash,
                source, ingestion_id, created_at, updated_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.id,
                document.user_id,
                document.filename,
                document.title,
                document.summary,
                document.content,
                document.content_hash,
                document.source,
                document.ingestion_id,
                document.created_at,
                document.updated_at,
                document.status,
            ),
        )
        self._conn.commit()
        return document

    def list_documents(self, user_id: str, *, limit: int = 20) -> list[DocumentRecord]:
        limit = max(1, min(int(limit), 100))
        rows = self._conn.execute(
            """
            SELECT id, user_id, filename, title, summary, content, content_hash,
                   source, ingestion_id, created_at, updated_at, status
            FROM documents
            WHERE user_id = ? AND deleted_at IS NULL AND status = 'active'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, user_id: str, document_id: str) -> DocumentRecord | None:
        row = self._conn.execute(
            """
            SELECT id, user_id, filename, title, summary, content, content_hash,
                   source, ingestion_id, created_at, updated_at, status
            FROM documents
            WHERE user_id = ? AND id = ? AND deleted_at IS NULL AND status = 'active'
            """,
            (user_id, document_id),
        ).fetchone()
        return self._row_to_document(row) if row else None

    def get_document_any_status(self, user_id: str, document_id: str) -> DocumentRecord | None:
        row = self._conn.execute(
            """
            SELECT id, user_id, filename, title, summary, content, content_hash,
                   source, ingestion_id, created_at, updated_at, status
            FROM documents
            WHERE user_id = ? AND id = ?
            """,
            (user_id, document_id),
        ).fetchone()
        return self._row_to_document(row) if row else None

    # 文档编辑会更新全文和哈希，后续细节问答始终召回最新原文。
    def update_document(
        self,
        user_id: str,
        document_id: str,
        *,
        filename: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
    ) -> DocumentRecord | None:
        existing = self.get_document(user_id, document_id)
        if existing is None:
            return None
        next_filename = str(filename).strip() if filename is not None else existing.filename
        next_title = str(title).strip() if title is not None else existing.title
        next_summary = str(summary).strip() if summary is not None else existing.summary
        next_content = str(content).strip() if content is not None else existing.content
        if not next_content:
            raise ValueError("document content cannot be empty")
        next_filename = next_filename or "uploaded.md"
        next_title = next_title or next_filename
        next_summary = next_summary or next_title
        now = time.time()
        self._conn.execute(
            """
            UPDATE documents
            SET filename = ?, title = ?, summary = ?, content = ?, content_hash = ?, updated_at = ?
            WHERE user_id = ? AND id = ? AND deleted_at IS NULL AND status = 'active'
            """,
            (
                next_filename,
                next_title,
                next_summary,
                next_content,
                hashlib.sha256(next_content.encode("utf-8")).hexdigest(),
                now,
                user_id,
                document_id,
            ),
        )
        self._conn.commit()
        return self.get_document(user_id, document_id)

    # 文档搜索优先全文索引，中文短词或 FTS 不可用时退回 LIKE。
    def search_documents(self, user_id: str, query: str, *, limit: int = 5) -> list[DocumentRecord]:
        query = query.strip()
        if not query:
            return self.list_documents(user_id, limit=limit)
        limit = max(1, min(int(limit), 20))
        rows = []
        try:
            rows = self._conn.execute(
                """
                SELECT d.id, d.user_id, d.filename, d.title, d.summary, d.content,
                       d.content_hash, d.source, d.ingestion_id, d.created_at,
                       d.updated_at, d.status
                FROM documents_fts f
                JOIN documents d ON d.rowid = f.rowid
                WHERE documents_fts MATCH ? AND d.user_id = ? AND d.deleted_at IS NULL
                  AND d.status = 'active'
                ORDER BY bm25(documents_fts), d.created_at DESC
                LIMIT ?
                """,
                (self._fts_query(query), user_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            rows = self._search_documents_like(user_id, query, limit)
        return [self._row_to_document(row) for row in rows]

    # 唯一新增记忆入口：统一规范 kind/type/privacy/status 和时间字段。
    def add_memory(
        self,
        user_id: str,
        content: str,
        *,
        subject_id: str | None = None,
        subject_type: str | None = None,
        subject_name: str | None = None,
        subject_scope: str = "",
        kind: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        source: str = "chat",
        source_id: str = "",
        ingestion_id: str = "",
        evidence_ids: list[str] | None = None,
        occurred_at: float | None = None,
        start_at: float | None = None,
        end_at: float | None = None,
        time_granularity: str = "unknown",
        temporal_text: str = "",
        temporal_confidence: float | None = None,
        privacy_level: str = "normal",
        status: str = "active",
        confidence: float | None = None,
        superseded_by: str = "",
        created_at: float | None = None,
    ) -> MemoryEvent:
        content = content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        memory_kind = normalize_memory_kind(kind or classify_memory_kind(content))
        memory_type = normalize_memory_type(memory_type or default_memory_type_for_kind(memory_kind))
        privacy_level = normalize_privacy_level(privacy_level)
        status = normalize_memory_status(status)
        subject = self._subject_for_memory(
            user_id,
            subject_id=subject_id,
            subject_type=subject_type,
            subject_name=subject_name,
            subject_scope=subject_scope or ingestion_id or source_id,
        )
        now = created_at if created_at is not None else time.time()
        if start_at is None and occurred_at is not None:
            start_at = occurred_at
        if occurred_at is None and start_at is not None:
            occurred_at = start_at
        event = MemoryEvent(
            id=uuid.uuid4().hex,
            user_id=user_id,
            kind=memory_kind,
            memory_type=memory_type,
            content=content,
            tags=tags or [],
            source=source,
            source_id=source_id.strip(),
            ingestion_id=ingestion_id.strip(),
            evidence_ids=evidence_ids or [],
            subject_id=subject.id,
            subject_type=subject.subject_type,
            subject_name=subject.display_name,
            created_at=now,
            updated_at=now,
            occurred_at=occurred_at,
            start_at=start_at,
            end_at=end_at,
            time_granularity=time_granularity or "unknown",
            temporal_text=temporal_text.strip(),
            temporal_confidence=temporal_confidence,
            privacy_level=privacy_level,
            status=status,
            confidence=confidence,
            superseded_by=superseded_by.strip(),
            strength=_initial_memory_strength(memory_type, confidence),
        )
        self._conn.execute(
            """
            INSERT INTO memories (
                id, user_id, subject_id, subject_type, subject_name,
                kind, memory_type, content, tags, source,
                source_id, ingestion_id, evidence_ids, created_at, updated_at,
                occurred_at, start_at, end_at, time_granularity,
                temporal_text, temporal_confidence, privacy_level, status,
                confidence, superseded_by, access_count, last_accessed_at, strength
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.user_id,
                event.subject_id,
                event.subject_type,
                event.subject_name,
                event.kind,
                event.memory_type,
                event.content,
                json.dumps(event.tags),
                event.source,
                event.source_id,
                event.ingestion_id,
                json.dumps(event.evidence_ids),
                event.created_at,
                event.updated_at,
                event.occurred_at,
                event.start_at,
                event.end_at,
                event.time_granularity,
                event.temporal_text,
                event.temporal_confidence,
                event.privacy_level,
                event.status,
                event.confidence,
                event.superseded_by,
                event.access_count,
                event.last_accessed_at,
                event.strength,
            ),
        )
        self._conn.commit()
        return event

    def _subject_for_memory(
        self,
        user_id: str,
        *,
        subject_id: str | None,
        subject_type: str | None,
        subject_name: str | None,
        subject_scope: str,
    ) -> MemorySubject:
        if subject_id:
            subject = self.get_subject(user_id, subject_id)
            if subject is None:
                raise ValueError("subject does not exist for user")
            return subject
        if not subject_name and not subject_type:
            return self.ensure_self_subject(user_id)
        normalized_type = _normalize_subject_type(subject_type or "named")
        if normalized_type == "self":
            return self.ensure_self_subject(user_id, display_name=subject_name or "我")
        if not subject_name:
            raise ValueError("subject_name is required for non-self memory")
        if normalized_type == "provisional":
            return self.create_provisional_subject(
                user_id,
                subject_name,
                source_scope=subject_scope,
            )
        return self.create_named_subject(user_id, subject_name)

    # 默认只列 active 记忆，删除和 superseded 记录不进入普通召回。
    def list_memories(
        self,
        user_id: str,
        *,
        limit: int = 100,
        kind: str | None = None,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[MemoryEvent]:
        clauses = ["user_id = ?", "deleted_at IS NULL", "status = ?"]
        params: list[Any] = [user_id, ACTIVE_MEMORY_STATUS]
        if kind:
            clauses.append("kind = ?")
            params.append(normalize_memory_kind(kind))
        normalized_subject_ids = self._canonical_subject_ids(user_id, subject_ids)
        if subject_ids is not None:
            if not normalized_subject_ids:
                return []
            clauses.append(f"subject_id IN ({', '.join('?' for _ in normalized_subject_ids)})")
            params.extend(normalized_subject_ids)
        params.append(max(1, min(int(limit), 500)))
        rows = self._conn.execute(
            f"""
            SELECT id, user_id, subject_id, subject_type, subject_name,
                   kind, content, tags, source, created_at,
                   occurred_at, start_at, end_at, time_granularity,
                   temporal_text, temporal_confidence, memory_type, source_id,
                   ingestion_id, evidence_ids, updated_at, privacy_level,
                   status, confidence, superseded_by, access_count,
                   last_accessed_at, strength
            FROM memories
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    # 搜索优先 FTS，失败或无结果时再退回 LIKE，兼容中文短词。
    def search(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[MemoryEvent]:
        return self.search_with_ranking(
            user_id,
            query,
            limit=limit,
            subject_ids=subject_ids,
        ).memories

    # 文本召回用可解释轻排序，旧 search() 仍保持只返回 MemoryEvent 列表。
    def search_with_ranking(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> MemorySearchResult:
        query = query.strip()
        if not query:
            memories = self.list_memories(user_id, limit=limit, subject_ids=subject_ids)
            return MemorySearchResult(memories=memories, ranking=[])
        limit = max(1, min(int(limit), 20))
        candidate_limit = max(limit * 4, limit)
        normalized_subject_ids = self._canonical_subject_ids(user_id, subject_ids)
        if subject_ids is not None and not normalized_subject_ids:
            return MemorySearchResult(memories=[], ranking=[])
        subject_clause = ""
        subject_params: list[str] = []
        if subject_ids is not None:
            subject_clause = f"AND m.subject_id IN ({', '.join('?' for _ in normalized_subject_ids)})"
            subject_params = normalized_subject_ids
        rows = []
        try:
            rows = self._conn.execute(
                f"""
                SELECT m.id, m.user_id, m.subject_id, m.subject_type, m.subject_name,
                       m.kind, m.content, m.tags, m.source,
                       m.created_at, m.occurred_at, m.start_at, m.end_at,
                       m.time_granularity, m.temporal_text, m.temporal_confidence,
                       m.memory_type, m.source_id, m.ingestion_id, m.evidence_ids,
                       m.updated_at, m.privacy_level, m.status, m.confidence,
                       m.superseded_by, m.access_count, m.last_accessed_at,
                       m.strength
                FROM memories_fts f
                JOIN memories m ON m.rowid = f.rowid
                WHERE memories_fts MATCH ? AND m.user_id = ? AND m.deleted_at IS NULL
                  AND m.status = ?
                  {subject_clause}
                ORDER BY bm25(memories_fts), m.created_at DESC
                LIMIT ?
                """,
                (
                    self._fts_query(query),
                    user_id,
                    ACTIVE_MEMORY_STATUS,
                    *subject_params,
                    candidate_limit,
                ),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            rows = self._search_like(
                user_id,
                query,
                candidate_limit,
                subject_ids=normalized_subject_ids if subject_ids is not None else None,
            )
        memories = [self._row_to_event(row) for row in rows]
        return self._rank_memory_search_results(memories, query, limit)

    # 事件时间查询使用半开区间重叠判断，适配跨时间段事件。
    def list_events_between(
        self,
        user_id: str,
        start_at: float,
        end_at: float,
        *,
        limit: int = 20,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[MemoryEvent]:
        if start_at >= end_at:
            return []
        limit = max(1, min(int(limit), 100))
        normalized_subject_ids = self._canonical_subject_ids(user_id, subject_ids)
        if subject_ids is not None and not normalized_subject_ids:
            return []
        subject_clause = ""
        if subject_ids is not None:
            subject_clause = f"AND subject_id IN ({', '.join('?' for _ in normalized_subject_ids)})"
        rows = self._conn.execute(
            f"""
            SELECT id, user_id, subject_id, subject_type, subject_name,
                   kind, content, tags, source, created_at,
                   occurred_at, start_at, end_at, time_granularity,
                   temporal_text, temporal_confidence, memory_type, source_id,
                   ingestion_id, evidence_ids, updated_at, privacy_level,
                   status, confidence, superseded_by, access_count,
                   last_accessed_at, strength
            FROM memories
            WHERE user_id = ?
              AND kind = 'event'
              AND deleted_at IS NULL
              AND status = ?
              {subject_clause}
              AND COALESCE(start_at, occurred_at) IS NOT NULL
              AND COALESCE(start_at, occurred_at) < ?
              AND COALESCE(end_at, start_at + 0.001, occurred_at + 0.001) > ?
            ORDER BY COALESCE(start_at, occurred_at) ASC, created_at DESC
            LIMIT ?
            """,
            (
                user_id,
                ACTIVE_MEMORY_STATUS,
                *normalized_subject_ids,
                end_at,
                start_at,
                limit,
            ),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    # 未来计划召回会用无时间事件兜底，避免漏掉未解析出时间的待办。
    def list_recent_untimed_events(
        self,
        user_id: str,
        *,
        limit: int = 5,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[MemoryEvent]:
        limit = max(1, min(int(limit), 20))
        normalized_subject_ids = self._canonical_subject_ids(user_id, subject_ids)
        if subject_ids is not None and not normalized_subject_ids:
            return []
        subject_clause = ""
        if subject_ids is not None:
            subject_clause = f"AND subject_id IN ({', '.join('?' for _ in normalized_subject_ids)})"
        rows = self._conn.execute(
            f"""
            SELECT id, user_id, subject_id, subject_type, subject_name,
                   kind, content, tags, source, created_at,
                   occurred_at, start_at, end_at, time_granularity,
                   temporal_text, temporal_confidence, memory_type, source_id,
                   ingestion_id, evidence_ids, updated_at, privacy_level,
                   status, confidence, superseded_by, access_count,
                   last_accessed_at, strength
            FROM memories
            WHERE user_id = ?
              AND kind = 'event'
              AND deleted_at IS NULL
              AND status = ?
              {subject_clause}
              AND COALESCE(start_at, occurred_at) IS NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, ACTIVE_MEMORY_STATUS, *normalized_subject_ids, limit),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _search_like(
        self,
        user_id: str,
        query: str,
        limit: int,
        *,
        subject_ids: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        terms = self._fallback_terms(query)
        clauses = []
        params: list[Any] = [user_id, ACTIVE_MEMORY_STATUS]
        term_params: list[Any] = []
        for term in terms:
            clauses.append("(content LIKE ? OR tags LIKE ?)")
            like = f"%{term}%"
            term_params.extend([like, like])
        if not clauses:
            return []
        subject_clause = ""
        if subject_ids is not None:
            if not subject_ids:
                return []
            subject_clause = f"AND subject_id IN ({', '.join('?' for _ in subject_ids)})"
            params.extend(subject_ids)
        params.extend(term_params)
        params.append(limit)
        return self._conn.execute(
            f"""
            SELECT id, user_id, subject_id, subject_type, subject_name,
                   kind, content, tags, source, created_at,
                   occurred_at, start_at, end_at, time_granularity,
                   temporal_text, temporal_confidence, memory_type, source_id,
                   ingestion_id, evidence_ids, updated_at, privacy_level,
                   status, confidence, superseded_by, access_count,
                   last_accessed_at, strength
            FROM memories
            WHERE user_id = ? AND deleted_at IS NULL AND status = ?
              {subject_clause}
              AND ({" OR ".join(clauses)})
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    def _search_documents_like(self, user_id: str, query: str, limit: int) -> list[sqlite3.Row]:
        terms = self._fallback_terms(query)
        clauses = []
        params: list[Any] = [user_id]
        for term in terms:
            clauses.append("(filename LIKE ? OR title LIKE ? OR summary LIKE ? OR content LIKE ?)")
            like = f"%{term}%"
            params.extend([like, like, like, like])
        if not clauses:
            return []
        params.append(limit)
        return self._conn.execute(
            f"""
            SELECT id, user_id, filename, title, summary, content, content_hash,
                   source, ingestion_id, created_at, updated_at, status
            FROM documents
            WHERE user_id = ? AND deleted_at IS NULL AND status = 'active'
              AND ({" OR ".join(clauses)})
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    # P2 轻排序只作用于文本搜索，不改变时间范围召回的时间语义。
    def _rank_memory_search_results(
        self,
        memories: list[MemoryEvent],
        query: str,
        limit: int,
    ) -> MemorySearchResult:
        terms = self._fallback_terms(query)
        ranked = []
        now = time.time()
        for memory in memories:
            text_score = _text_match_score(" ".join([memory.content, " ".join(memory.tags)]), terms)
            recency_score = _recency_score(memory.start_at or memory.occurred_at or memory.created_at, now)
            strength_policy = _effective_strength_policy(memory, now)
            strength_score = _clamp_score(strength_policy["effective_strength"])
            access_score = _access_score(memory.access_count)
            total_score = (
                0.55 * text_score
                + 0.20 * recency_score
                + 0.20 * strength_score
                + 0.05 * access_score
            )
            if access_score > 0 and text_score >= 0.99:
                total_score += min(0.02, 0.01 * access_score)
            ranking = {
                "id": memory.id,
                "total_score": round(total_score, 4),
                "text_score": round(text_score, 4),
                "recency_score": round(recency_score, 4),
                "strength_score": round(strength_score, 4),
                "base_strength_score": round(_clamp_score(memory.strength), 4),
                "effective_strength_score": round(strength_score, 4),
                "strength_cap": round(float(strength_policy["strength_cap"]), 4),
                "decay_factor": round(float(strength_policy["decay_factor"]), 4),
                "strength_policy_reason": str(strength_policy["strength_policy_reason"]),
                "access_score": round(access_score, 4),
                "reason": "text_recency_effective_strength_access",
            }
            ranked.append((total_score, text_score, access_score, recency_score, memory.created_at, memory, ranking))
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]), reverse=True)
        selected = ranked[:limit]
        return MemorySearchResult(
            memories=[item[5] for item in selected],
            ranking=[item[6] for item in selected],
        )

    # 删除采用软删除，保留 audit/历史排查所需的数据库记录。
    def delete_document(self, user_id: str, document_id: str) -> bool:
        now = time.time()
        cur = self._conn.execute(
            """
            UPDATE documents
            SET deleted_at = ?, status = 'deleted', updated_at = ?
            WHERE id = ? AND user_id = ? AND deleted_at IS NULL
            """,
            (now, now, document_id, user_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def purge_document(self, user_id: str, document_id: str) -> DocumentRecord | None:
        document = self.get_document_any_status(user_id, document_id)
        if document is None:
            return None
        cur = self._conn.execute(
            """
            DELETE FROM documents
            WHERE id = ? AND user_id = ?
            """,
            (document_id, user_id),
        )
        self._conn.commit()
        return document if cur.rowcount > 0 else None

    # 删除采用软删除，保留 audit/历史排查所需的数据库记录。
    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        cur = self._conn.execute(
            """
            UPDATE memories
            SET deleted_at = ?, status = 'deleted', updated_at = ?
            WHERE id = ? AND user_id = ? AND deleted_at IS NULL
            """,
            (time.time(), time.time(), memory_id, user_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def purge_memory(self, user_id: str, memory_id: str) -> MemoryEvent | None:
        memory = self.get_memory_any_status(user_id, memory_id)
        if memory is None:
            return None
        cur = self._conn.execute(
            """
            DELETE FROM memories
            WHERE id = ? AND user_id = ?
            """,
            (memory_id, user_id),
        )
        self._conn.commit()
        return memory if cur.rowcount > 0 else None

    def get_memory(
        self,
        user_id: str,
        memory_id: str,
        *,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> MemoryEvent | None:
        return self._get_memory(user_id, memory_id, subject_ids=subject_ids)

    def get_memory_any_status(
        self,
        user_id: str,
        memory_id: str,
        *,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> MemoryEvent | None:
        return self._get_memory(
            user_id,
            memory_id,
            include_deleted=True,
            subject_ids=subject_ids,
        )

    def active_memory_count_with_evidence(self, user_id: str, evidence_id: str) -> int:
        evidence_id = str(evidence_id or "").strip()
        if not evidence_id:
            return 0
        return self.evidence_reference_counts(user_id, [evidence_id])[evidence_id].active_refs

    def evidence_reference_counts(
        self,
        user_id: str,
        evidence_ids: list[str] | tuple[str, ...] | set[str],
    ) -> dict[str, EvidenceReferenceCounts]:
        ids = normalize_evidence_ids(evidence_ids)
        counts = {evidence_id: EvidenceReferenceCounts() for evidence_id in ids}
        if not ids:
            return counts
        retained_statuses = (ACTIVE_MEMORY_STATUS, STALE_MEMORY_STATUS, SUPERSEDED_MEMORY_STATUS)
        placeholders = ", ".join("?" for _ in retained_statuses)
        rows = self._conn.execute(
            f"""
            SELECT evidence_ids, status FROM memories
            WHERE user_id = ? AND deleted_at IS NULL AND status IN ({placeholders})
            """,
            tuple([user_id, *retained_statuses]),
        ).fetchall()
        wanted = set(ids)
        active_refs = {evidence_id: 0 for evidence_id in ids}
        retained_refs = {evidence_id: 0 for evidence_id in ids}
        for row in rows:
            try:
                row_evidence_ids = {
                    str(item).strip()
                    for item in json.loads(row["evidence_ids"] or "[]")
                    if str(item).strip()
                }
            except json.JSONDecodeError:
                row_evidence_ids = set()
            matched = wanted & row_evidence_ids
            if not matched:
                continue
            for evidence_id in matched:
                retained_refs[evidence_id] += 1
                if str(row["status"] or "") == ACTIVE_MEMORY_STATUS:
                    active_refs[evidence_id] += 1
        return {
            evidence_id: EvidenceReferenceCounts(
                active_refs=active_refs[evidence_id],
                retained_refs=retained_refs[evidence_id],
            )
            for evidence_id in ids
        }

    # 写入前用规范化文本和近似时间做轻量去重。
    def find_similar_memory(
        self,
        user_id: str,
        content: str,
        *,
        kind: str,
        memory_type: str,
        start_at: float | None = None,
        subject_id: str | None = None,
    ) -> MemoryEvent | None:
        normalized = _normalize_for_dedupe(content)
        if not normalized:
            return None
        scoped_subject_id = subject_id or self.ensure_self_subject(user_id).id
        candidates = self.list_memories(
            user_id,
            limit=50,
            kind=kind,
            subject_ids=[scoped_subject_id],
        )
        for memory in candidates:
            if memory.memory_type != memory_type or not is_active_memory_status(memory.status):
                continue
            if start_at is not None and memory.start_at is not None and abs(memory.start_at - start_at) > 3600:
                continue
            if _normalize_for_dedupe(memory.content) == normalized:
                return memory
        return None

    # 重复记忆不新增行，只合并 evidence/source 并提高置信度。
    def merge_memory_evidence(
        self,
        user_id: str,
        memory_id: str,
        *,
        evidence_ids: list[str] | None = None,
        source_id: str = "",
        confidence: float | None = None,
    ) -> MemoryEvent | None:
        existing = self._get_memory(user_id, memory_id)
        if existing is None or not is_active_memory_status(existing.status):
            return None
        merged_evidence = list(dict.fromkeys([*existing.evidence_ids, *(evidence_ids or [])]))
        if source_id and source_id not in merged_evidence:
            merged_evidence.append(source_id)
        next_confidence = existing.confidence
        if confidence is not None:
            next_confidence = max(confidence, existing.confidence or 0.0)
        next_source_id = existing.source_id or source_id.strip()
        self._conn.execute(
            """
            UPDATE memories
            SET evidence_ids = ?, source_id = ?, confidence = ?, updated_at = ?
            WHERE id = ? AND user_id = ? AND deleted_at IS NULL AND status = ?
            """,
            (
                json.dumps(merged_evidence),
                next_source_id,
                next_confidence,
                time.time(),
                memory_id,
                user_id,
                ACTIVE_MEMORY_STATUS,
            ),
        )
        self._conn.commit()
        return self._get_memory(user_id, memory_id)

    # superseded 表示被新记忆替代，但仍保留可追溯关系。
    def mark_superseded(self, user_id: str, memory_id: str, superseded_by: str) -> bool:
        existing = self._get_memory(user_id, memory_id)
        replacement = self._get_memory(user_id, superseded_by)
        if existing is None or not can_transition_memory_status(
            existing.status,
            SUPERSEDED_MEMORY_STATUS,
            superseded_by=superseded_by,
        ):
            return False
        if replacement is None or existing.subject_id != replacement.subject_id:
            return False
        cur = self._conn.execute(
            """
            UPDATE memories
            SET status = 'superseded', superseded_by = ?, updated_at = ?
            WHERE id = ? AND user_id = ? AND deleted_at IS NULL AND status = ?
            """,
            (superseded_by.strip(), time.time(), memory_id, user_id, ACTIVE_MEMORY_STATUS),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # stale 表示仍可追溯，但不再作为当前有效记忆参与召回。
    def mark_stale(self, user_id: str, memory_id: str) -> bool:
        existing = self._get_memory(user_id, memory_id)
        if existing is None or not can_transition_memory_status(existing.status, STALE_MEMORY_STATUS):
            return False
        cur = self._conn.execute(
            """
            UPDATE memories
            SET status = 'stale', updated_at = ?
            WHERE id = ? AND user_id = ? AND deleted_at IS NULL AND status = ?
            """,
            (time.time(), memory_id, user_id, ACTIVE_MEMORY_STATUS),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # 只在记忆真正进入回复上下文后记录访问，不让管理列表读取改变排序信号。
    def record_memory_access(
        self,
        user_id: str,
        memory_ids: list[str],
        *,
        accessed_at: float | None = None,
    ) -> list[MemoryEvent]:
        unique_ids = list(dict.fromkeys(str(memory_id).strip() for memory_id in memory_ids if str(memory_id).strip()))
        if not unique_ids:
            return []
        now = accessed_at if accessed_at is not None else time.time()
        updated: list[MemoryEvent] = []
        for memory_id in unique_ids:
            existing = self._get_memory(user_id, memory_id)
            if existing is None or not is_active_memory_status(existing.status):
                continue
            next_count = max(0, int(existing.access_count)) + 1
            base_strength = _initial_memory_strength(existing.memory_type, existing.confidence)
            strength_policy = _strength_policy_for_type(existing.memory_type)
            next_strength = min(
                strength_policy.cap,
                max(float(existing.strength or 0.0), base_strength + min(0.12, 0.03 * next_count)),
            )
            self._conn.execute(
                """
                UPDATE memories
                SET access_count = ?, last_accessed_at = ?, strength = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND deleted_at IS NULL AND status = ?
                """,
                (next_count, now, next_strength, now, memory_id, user_id, ACTIVE_MEMORY_STATUS),
            )
            refreshed = self._get_memory(user_id, memory_id)
            if refreshed is not None:
                updated.append(refreshed)
        self._conn.commit()
        return updated

    def _get_memory(
        self,
        user_id: str,
        memory_id: str,
        *,
        include_deleted: bool = False,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> MemoryEvent | None:
        deleted_clause = "" if include_deleted else "AND deleted_at IS NULL"
        normalized_subject_ids = self._canonical_subject_ids(user_id, subject_ids)
        if subject_ids is not None and not normalized_subject_ids:
            return None
        subject_clause = ""
        if subject_ids is not None:
            subject_clause = f"AND subject_id IN ({', '.join('?' for _ in normalized_subject_ids)})"
        row = self._conn.execute(
            f"""
            SELECT id, user_id, subject_id, subject_type, subject_name,
                   kind, content, tags, source, created_at,
                   occurred_at, start_at, end_at, time_granularity,
                   temporal_text, temporal_confidence, memory_type, source_id,
                   ingestion_id, evidence_ids, updated_at, privacy_level,
                   status, confidence, superseded_by, access_count,
                   last_accessed_at, strength
            FROM memories
            WHERE user_id = ? AND id = ? {deleted_clause} {subject_clause}
            """,
            (user_id, memory_id, *normalized_subject_ids),
        ).fetchone()
        return self._row_to_event(row) if row else None

    def _canonical_subject_ids(
        self,
        user_id: str,
        subject_ids: list[str] | tuple[str, ...] | set[str] | None,
    ) -> list[str]:
        canonical_ids: list[str] = []
        for subject_id in _normalize_subject_ids(subject_ids):
            subject = self.get_subject(user_id, subject_id)
            if subject is not None and subject.id not in canonical_ids:
                canonical_ids.append(subject.id)
        return canonical_ids

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = [part.strip().replace('"', '""') for part in query.split() if part.strip()]
        return " OR ".join(f'"{term}"' for term in terms) or '""'

    @staticmethod
    def _fallback_terms(query: str) -> list[str]:
        terms = [part.strip() for part in query.split() if part.strip()]
        cjk_terms = [
            "吃面",
            "今天",
            "昨天",
            "明天",
            "中午",
            "早上",
            "晚上",
            "咖啡",
            "午餐",
            "晚餐",
            "早餐",
            "喜欢",
            "不喜欢",
            "文档",
            "上传",
            "会议",
            "周报",
            "日报",
            "攻略",
            "门票",
            "路线",
            "安排",
        ]
        for term in cjk_terms:
            if term in query and term not in terms:
                terms.append(term)
        for match in _CJK_ACTION_OBJECT_PATTERN.finditer(query):
            phrase = match.group(0)
            if phrase not in terms:
                terms.append(phrase)
            obj = match.group("object")
            if obj not in terms:
                terms.append(obj)
        if not terms and query.strip():
            terms.append(query.strip())
        return terms[:10]

    # SQLite 行统一在这里转成 MemoryEvent，兼容旧字段缺失。
    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
        try:
            tags = json.loads(row["tags"] or "[]")
        except json.JSONDecodeError:
            tags = []
        try:
            evidence_ids = json.loads(row["evidence_ids"] or "[]") if "evidence_ids" in row.keys() else []
        except json.JSONDecodeError:
            evidence_ids = []
        created_at = float(row["created_at"])
        return MemoryEvent(
            id=row["id"],
            user_id=row["user_id"],
            subject_id=str(row["subject_id"] or "") if "subject_id" in row.keys() else "",
            subject_type=str(row["subject_type"] or "self") if "subject_type" in row.keys() else "self",
            subject_name=str(row["subject_name"] or "我") if "subject_name" in row.keys() else "我",
            kind=row["kind"] if "kind" in row.keys() else "event",
            memory_type=str(row["memory_type"] or "") if "memory_type" in row.keys() else normalize_memory_type(row["kind"]),
            content=row["content"],
            tags=tags if isinstance(tags, list) else [],
            source=row["source"],
            source_id=str(row["source_id"] or "") if "source_id" in row.keys() else "",
            ingestion_id=str(row["ingestion_id"] or "") if "ingestion_id" in row.keys() else "",
            evidence_ids=evidence_ids if isinstance(evidence_ids, list) else [],
            created_at=created_at,
            updated_at=float(row["updated_at"]) if "updated_at" in row.keys() and row["updated_at"] else created_at,
            occurred_at=float(row["occurred_at"]) if row["occurred_at"] is not None else None,
            start_at=float(row["start_at"]) if "start_at" in row.keys() and row["start_at"] is not None else None,
            end_at=float(row["end_at"]) if "end_at" in row.keys() and row["end_at"] is not None else None,
            time_granularity=str(row["time_granularity"] or "unknown") if "time_granularity" in row.keys() else "unknown",
            temporal_text=str(row["temporal_text"] or "") if "temporal_text" in row.keys() else "",
            temporal_confidence=(
                float(row["temporal_confidence"])
                if "temporal_confidence" in row.keys() and row["temporal_confidence"] is not None
                else None
            ),
            privacy_level=str(row["privacy_level"] or "normal") if "privacy_level" in row.keys() else "normal",
            status=str(row["status"] or "active") if "status" in row.keys() else "active",
            confidence=float(row["confidence"]) if "confidence" in row.keys() and row["confidence"] is not None else None,
            superseded_by=str(row["superseded_by"] or "") if "superseded_by" in row.keys() else "",
            access_count=max(0, int(row["access_count"])) if "access_count" in row.keys() and row["access_count"] is not None else 0,
            last_accessed_at=(
                float(row["last_accessed_at"])
                if "last_accessed_at" in row.keys() and row["last_accessed_at"] is not None
                else None
            ),
            strength=(
                float(row["strength"])
                if "strength" in row.keys() and row["strength"] is not None
                else _initial_memory_strength(
                    str(row["memory_type"] or "") if "memory_type" in row.keys() else "event",
                    float(row["confidence"]) if "confidence" in row.keys() and row["confidence"] is not None else None,
                )
            ),
        )

    @staticmethod
    def _row_to_subject(row: sqlite3.Row) -> MemorySubject:
        return MemorySubject(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            subject_type=str(row["subject_type"]),
            display_name=str(row["display_name"]),
            canonical_name=str(row["canonical_name"]),
            source_scope=str(row["source_scope"] or ""),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            merged_into=str(row["merged_into"] or ""),
        )

    @staticmethod
    def _row_to_voice_profile(row: sqlite3.Row) -> VoiceProfile:
        try:
            embedding = _normalize_embedding(json.loads(row["embedding_json"] or "[]"))
        except (json.JSONDecodeError, TypeError, ValueError):
            embedding = ()
        return VoiceProfile(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            subject_id=str(row["subject_id"]),
            embedding_model=str(row["embedding_model"]),
            embedding_dim=int(row["embedding_dim"]),
            embedding=embedding,
            confidence=float(row["confidence"]) if row["confidence"] is not None else None,
            source_id=str(row["source_id"] or ""),
            created_at=float(row["created_at"]),
        )

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        created_at = float(row["created_at"])
        return DocumentRecord(
            id=row["id"],
            user_id=row["user_id"],
            filename=str(row["filename"] or ""),
            title=str(row["title"] or ""),
            summary=str(row["summary"] or ""),
            content=str(row["content"] or ""),
            content_hash=str(row["content_hash"] or ""),
            source=str(row["source"] or ""),
            ingestion_id=str(row["ingestion_id"] or ""),
            created_at=created_at,
            updated_at=float(row["updated_at"]) if "updated_at" in row.keys() and row["updated_at"] else created_at,
            status=str(row["status"] or "active") if "status" in row.keys() else "active",
        )


# HTTP/API/debug 输出统一走这个序列化函数，避免字段不一致。
def event_to_dict(event: MemoryEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "user_id": event.user_id,
        "subject_id": event.subject_id,
        "subject_type": event.subject_type,
        "subject_name": event.subject_name,
        "kind": event.kind,
        "memory_type": event.memory_type,
        "content": event.content,
        "tags": event.tags,
        "source": event.source,
        "source_id": event.source_id,
        "ingestion_id": event.ingestion_id,
        "evidence_ids": event.evidence_ids,
        "created_at": event.created_at,
        "updated_at": event.updated_at,
        "occurred_at": event.occurred_at,
        "start_at": event.start_at,
        "end_at": event.end_at,
        "time_granularity": event.time_granularity,
        "temporal_text": event.temporal_text,
        "temporal_confidence": event.temporal_confidence,
        "privacy_level": event.privacy_level,
        "status": event.status,
        "confidence": event.confidence,
        "superseded_by": event.superseded_by,
        "access_count": event.access_count,
        "last_accessed_at": event.last_accessed_at,
        "strength": event.strength,
        "source_trace": source_trace(
            layer="reflection" if event.memory_type == "observation" else "structured_memory",
            user_id=event.user_id,
            source=event.source,
            source_id=event.source_id,
            ingestion_id=event.ingestion_id,
            evidence_ids=event.evidence_ids,
            privacy_level=event.privacy_level,
            status=event.status,
            confidence=event.confidence,
        ),
    }


def subject_to_dict(subject: MemorySubject) -> dict[str, Any]:
    return {
        "id": subject.id,
        "user_id": subject.user_id,
        "subject_type": subject.subject_type,
        "display_name": subject.display_name,
        "source_scope": subject.source_scope,
        "created_at": subject.created_at,
        "updated_at": subject.updated_at,
        "merged_into": subject.merged_into,
    }


# 对外只暴露声纹元数据；原始向量仅供本地匹配代码读取 VoiceProfile.embedding。
def voice_profile_to_dict(profile: VoiceProfile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "subject_id": profile.subject_id,
        "embedding_model": profile.embedding_model,
        "embedding_dim": profile.embedding_dim,
        "confidence": profile.confidence,
        "source_id": profile.source_id,
        "created_at": profile.created_at,
    }


def document_to_dict(document: DocumentRecord, *, include_content: bool = False) -> dict[str, Any]:
    payload = {
        "id": document.id,
        "user_id": document.user_id,
        "filename": document.filename,
        "title": document.title,
        "summary": document.summary,
        "source": document.source,
        "ingestion_id": document.ingestion_id,
        "content_hash": document.content_hash,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "status": document.status,
    }
    if include_content:
        payload["content"] = document.content
    payload["source_trace"] = source_trace(
        layer="document_archive",
        user_id=document.user_id,
        source=document.source,
        source_id=document.id,
        ingestion_id=document.ingestion_id,
        status=document.status,
    )
    return payload


def _require_nonempty(value: str, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


def _normalize_subject_type(subject_type: str) -> str:
    normalized = str(subject_type or "").strip().lower()
    if normalized not in _MEMORY_SUBJECT_TYPES:
        raise ValueError(f"unsupported subject_type: {subject_type}")
    return normalized


def _normalize_subject_ids(
    subject_ids: list[str] | tuple[str, ...] | set[str] | None,
) -> list[str]:
    if subject_ids is None:
        return []
    return list(
        dict.fromkeys(
            str(subject_id).strip()
            for subject_id in subject_ids
            if str(subject_id).strip()
        )
    )


def _unique_subjects(subjects: list[MemorySubject]) -> list[MemorySubject]:
    return list({subject.id: subject for subject in subjects}.values())


def _normalize_embedding(embedding: Any) -> tuple[float, ...]:
    if not isinstance(embedding, (list, tuple)) or not embedding:
        raise ValueError("embedding must be a non-empty list or tuple")
    vector = tuple(float(value) for value in embedding)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding values must be finite")
    if sum(value * value for value in vector) <= 0.0:
        raise ValueError("embedding must have a non-zero norm")
    return vector


def _normalize_optional_confidence(confidence: float | None) -> float | None:
    if confidence is None:
        return None
    numeric = float(confidence)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return numeric


# 外部传入的 kind 只允许进入当前支持的记忆大类。
def normalize_memory_kind(kind: str) -> str:
    normalized = (kind or "").strip().lower()
    if normalized in {"profile", "user", "preference", "identity", "routine"}:
        return "profile"
    if normalized in {"event", "activity", "plan", "schedule"}:
        return "event"
    if normalized in {"assistant_preference", "assistant", "assistant_setting", "persona"}:
        return "assistant_preference"
    return "event"


# memory_type 用于项目周报、提醒等二级语义，不等同于 kind。
def normalize_memory_type(memory_type: str) -> str:
    normalized = (memory_type or "").strip().lower()
    aliases = {
        "profile": "preference",
        "routine": "preference",
        "plan": "task",
        "schedule": "task",
        "todo": "task",
        "meeting": "event",
        "state": "project_state",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"fact", "event", "task", "preference", "decision", "project_state", "observation"}
    return normalized if normalized in allowed else "event"


def default_memory_type_for_kind(kind: str) -> str:
    normalized = normalize_memory_kind(kind)
    if normalized in {"profile", "assistant_preference"}:
        return "preference"
    return "event"


def normalize_privacy_level(privacy_level: str) -> str:
    normalized = (privacy_level or "").strip().lower()
    if normalized in {"sensitive", "requires_confirmation"}:
        return normalized
    return "normal"


def _initial_memory_strength(memory_type: str, confidence: float | None) -> float:
    if confidence is not None:
        return _clamp_score(confidence)
    weights = {
        "preference": 0.85,
        "observation": 0.8,
        "task": 0.75,
    }
    return weights.get(str(memory_type or "").strip().lower(), 0.7)


def _strength_policy_for_type(memory_type: str) -> StrengthPolicy:
    normalized = str(memory_type or "").strip().lower()
    policies = {
        "preference": StrengthPolicy(cap=0.9, decay_days=180.0, floor=0.72, reason="stable_preference_slow_decay"),
        "task": StrengthPolicy(cap=0.86, decay_days=30.0, floor=0.35, reason="task_status_recency_sensitive"),
        "project_state": StrengthPolicy(cap=0.86, decay_days=45.0, floor=0.4, reason="project_state_recency_sensitive"),
        "decision": StrengthPolicy(cap=0.88, decay_days=75.0, floor=0.5, reason="decision_moderate_decay"),
        "observation": StrengthPolicy(cap=0.86, decay_days=90.0, floor=0.55, reason="observation_summary_light_decay"),
    }
    return policies.get(normalized, StrengthPolicy(cap=0.88, decay_days=60.0, floor=0.45, reason="default_memory_moderate_decay"))


def _effective_strength_policy(memory: MemoryEvent, now: float) -> dict[str, Any]:
    policy = _strength_policy_for_type(memory.memory_type)
    base_strength = _clamp_score(memory.strength)
    capped_strength = min(base_strength, policy.cap)
    timestamp = memory.last_accessed_at or memory.updated_at or memory.created_at
    age_seconds = max(0.0, now - float(timestamp or now))
    decay_seconds = max(1.0, policy.decay_days * 24 * 60 * 60)
    decay_factor = max(policy.floor, 1.0 - (age_seconds / decay_seconds))
    return {
        "base_strength": base_strength,
        "capped_strength": capped_strength,
        "effective_strength": capped_strength * decay_factor,
        "strength_cap": policy.cap,
        "decay_factor": decay_factor,
        "strength_policy_reason": policy.reason,
    }


def effective_memory_strength(memory: MemoryEvent, *, now: float | None = None) -> float:
    return float(_effective_strength_policy(memory, time.time() if now is None else now)["effective_strength"])


def _clamp_score(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _text_match_score(text: str, terms: list[str]) -> float:
    useful_terms = [term.strip().lower() for term in terms if term.strip()]
    if not useful_terms:
        return 0.0
    haystack = str(text or "").lower()
    matched = sum(1 for term in useful_terms if term in haystack)
    return _clamp_score(matched / len(useful_terms))


def _recency_score(timestamp: float | None, now: float) -> float:
    if timestamp is None:
        return 0.0
    age_seconds = max(0.0, now - float(timestamp))
    return _clamp_score(1.0 - (age_seconds / (30 * 24 * 60 * 60)))


def _access_score(access_count: int) -> float:
    return _clamp_score(max(0, int(access_count or 0)) / 10)


def classify_memory_kind_with_reason(text: str) -> LegacyClassification:
    lowered = text.strip().lower()
    if _looks_like_legacy_direct_question(lowered):
        return LegacyClassification(
            value="event",
            source="legacy_default",
            reason="question_text_no_kind_marker",
        )
    profile_triggers = (
        "我叫",
        "我是",
        "我喜欢",
        "我不喜欢",
        "我的习惯",
        "我的偏好",
        "remember that i like",
        "my name is",
        "i am ",
        "i like",
        "i don't like",
        "my preference",
    )
    for trigger in profile_triggers:
        if trigger in lowered:
            return LegacyClassification(
                value="profile",
                source="legacy_phrase_classifier",
                reason="profile_trigger",
                marker=trigger,
            )
    return LegacyClassification(
        value="event",
        source="legacy_default",
        reason="no_profile_trigger",
    )


def classify_memory_kind(text: str) -> str:
    return classify_memory_kind_with_reason(text).value


# 导入/手动写入没有显式 type 时，用轻量信号给周报和提醒提供最低可用分类。
# P1.11: 这里仍是 legacy fallback；自然词本身不能承担开放语义裁决。
def classify_memory_type_with_reason(text: str, kind: str = "") -> LegacyClassification:
    lowered = text.strip().lower()
    if kind == "profile":
        return LegacyClassification(
            value="preference",
            source="legacy_kind_default",
            reason="profile_kind_defaults_to_preference",
        )
    if _looks_like_legacy_direct_question(lowered):
        return LegacyClassification(
            value="event",
            source="legacy_default",
            reason="question_text_no_type_marker",
        )
    preference_marker = _legacy_preference_marker(lowered)
    if preference_marker:
        return LegacyClassification(
            value="preference",
            source="legacy_phrase_classifier",
            reason="preference_marker",
            marker=preference_marker,
        )
    for marker in ("决定", "结论", "decision", "decided"):
        if marker in lowered:
            return LegacyClassification(
                value="decision",
                source="legacy_phrase_classifier",
                reason="decision_marker",
                marker=marker,
            )
    task_marker = _legacy_task_marker(lowered)
    if task_marker:
        return LegacyClassification(
            value="task",
            source="legacy_phrase_classifier",
            reason="task_marker",
            marker=task_marker,
        )
    project_state_marker = _legacy_project_state_marker(lowered)
    if project_state_marker:
        return LegacyClassification(
            value="project_state",
            source="legacy_phrase_classifier",
            reason="project_state_marker",
            marker=project_state_marker,
        )
    return LegacyClassification(
        value="event",
        source="legacy_default",
        reason="no_type_marker",
    )


def classify_memory_type(text: str, kind: str = "") -> str:
    return classify_memory_type_with_reason(text, kind).value


def _legacy_preference_marker(lowered: str) -> str:
    explicit_markers = (
        "我喜欢",
        "我不喜欢",
        "用户喜欢",
        "用户不喜欢",
        "我的偏好",
        "用户偏好",
        "我的习惯",
        "用户习惯",
        "prefer",
        "i like",
        "i don't like",
        "dislike",
    )
    for marker in explicit_markers:
        if marker in lowered:
            return marker
    for marker in ("偏好", "习惯"):
        if re.search(rf"(?:^|[，,;；:：\s]){re.escape(marker)}(?:[：:\s]|是)", lowered):
            return marker
    return ""


def _legacy_task_marker(lowered: str) -> str:
    for marker in ("负责", "待办", "任务", "要做", "周五前", "deadline", "todo"):
        if marker in lowered:
            return marker
    if "提醒" not in lowered:
        return ""
    reminder_context = (
        "提醒我",
        "提醒一下",
        "记得提醒",
        "帮我提醒",
        "需要提醒",
    )
    if any(marker in lowered for marker in reminder_context):
        return "提醒"
    temporal_context = (
        "今天",
        "明天",
        "后天",
        "上午",
        "下午",
        "晚上",
        "中午",
        "点",
        "周一",
        "周二",
        "周三",
        "周四",
        "周五",
        "周六",
        "周日",
        "下周",
        "deadline",
    )
    if any(marker in lowered for marker in temporal_context):
        return "提醒"
    return ""


def _legacy_project_state_marker(lowered: str) -> str:
    background_terms = (
        "背景",
        "说明",
        "目的",
        "介绍",
        "文档",
        "材料",
        "临时",
        "临时草稿",
        "待确认",
        "待确认草稿",
        "还没确认",
        "不是最终",
        "not final",
        "draft",
    )
    if any(term in lowered for term in background_terms):
        return ""
    for marker in ("风险", "卡点", "延期"):
        if marker in lowered:
            return marker
    status_markers = ("状态", "进展")
    current_context = (
        "当前",
        "现在",
        "目前",
        "主线",
        "项目状态",
        "项目进展",
        "最近",
        "这一轮",
        "本周",
    )
    if any(marker in lowered for marker in status_markers) and any(term in lowered for term in current_context):
        return next(marker for marker in status_markers if marker in lowered)
    return ""


def _normalize_for_dedupe(text: str) -> str:
    return "".join(str(text or "").lower().split()).strip("。,.，")


# 旧规则抽取只作为 fallback，主链路优先用 turn_planner/LLM classifier。
def extract_memory_candidates(message: str) -> list[dict[str, Any]]:
    """Conservative rule-based extractor for the MVP.

    This keeps stage 1 deterministic. Later we can replace this with a small
    extraction model or Hermes tool call once the event schema stabilizes.
    """
    text = message.strip()
    if not text:
        return []
    content = _strip_legacy_memory_request(text)
    if _looks_like_legacy_direct_question(text) and content == text:
        return []
    triggers = (
        "记住",
        "帮我记",
        "记一下",
        "记录",
        "我叫",
        "我是",
        "我喜欢",
        "我不喜欢",
        "我今天",
        "今天我",
        "昨天我",
        "明天",
        "remember",
        "my name is",
        "i like",
        "i don't like",
        "today i",
        "yesterday i",
        "tomorrow",
    )
    lowered = text.lower()
    if any(trigger in lowered for trigger in triggers):
        kind_classification = classify_memory_kind_with_reason(content)
        type_classification = classify_memory_type_with_reason(content, kind_classification.value)
        reason = (
            "legacy_memory_extractor"
            f"; kind={kind_classification.source}:{kind_classification.reason}"
            f"; memory_type={type_classification.source}:{type_classification.reason}"
        )
        return [{
            "content": content,
            "kind": kind_classification.value,
            "memory_type": type_classification.value,
            "reason": reason,
            "source": "rule_fallback",
            "classification": {
                "kind": kind_classification.debug_payload("kind"),
                "memory_type": type_classification.debug_payload("memory_type"),
            },
        }]
    return []


def _looks_like_legacy_direct_question(text: str) -> bool:
    stripped = str(text or "").strip()
    if stripped.endswith(("?", "？")):
        return True
    return any(marker in stripped for marker in ("什么", "啥", "吗", "如何", "怎么", "谁", "哪里", "哪儿", "是否"))


def _strip_legacy_memory_request(text: str) -> str:
    original = " ".join(str(text or "").split()).strip()
    stripped = original.strip(" ：:，,")
    polite_prefixes = (
        "你能帮我",
        "能不能帮我",
        "可以帮我",
        "麻烦帮我",
        "请帮我",
        "帮我",
        "请",
    )
    changed = True
    while changed:
        changed = False
        for prefix in polite_prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :].strip(" ：:，,")
                changed = True
                break
    matched_command = False
    for marker in ("记一下", "帮我记一下", "帮我记", "记录一下", "记录", "记住", "remember"):
        if stripped.startswith(marker):
            stripped = stripped[len(marker) :].strip(" ：:，,")
            matched_command = True
            break
    if not matched_command:
        return original
    suffixes = ("可以吗", "好吗", "行吗", "可以不", "好不好", "吗", "么", "吗?", "吗？")
    changed = True
    while changed:
        changed = False
        stripped = stripped.strip(" 。！？!?，,；;：: ")
        for suffix in suffixes:
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)].strip(" 。！？!?，,；;：: ")
                changed = True
                break
    return stripped
