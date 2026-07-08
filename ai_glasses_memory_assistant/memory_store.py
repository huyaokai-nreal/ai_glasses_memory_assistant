from __future__ import annotations

import json
import hashlib
import re
import sqlite3
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

_CJK_ACTION_OBJECT_PATTERN = re.compile(r"(?P<action>吃|喝|买|看|开|聊|做|去|约|取|交|写|改|补)(?:了|过|一下)?(?P<object>[\u4e00-\u9fffA-Za-z0-9]{1,8})")


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
            CREATE INDEX IF NOT EXISTS idx_documents_user_created
                ON documents(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_documents_user_ingestion
                ON documents(user_id, ingestion_id);
            CREATE INDEX IF NOT EXISTS idx_documents_user_hash
                ON documents(user_id, content_hash);
            """
        )
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
                id, user_id, kind, memory_type, content, tags, source,
                source_id, ingestion_id, evidence_ids, created_at, updated_at,
                occurred_at, start_at, end_at, time_granularity,
                temporal_text, temporal_confidence, privacy_level, status,
                confidence, superseded_by, access_count, last_accessed_at, strength
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.user_id,
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

    # 默认只列 active 记忆，删除和 superseded 记录不进入普通召回。
    def list_memories(
        self,
        user_id: str,
        *,
        limit: int = 100,
        kind: str | None = None,
    ) -> list[MemoryEvent]:
        params: list[Any] = [user_id]
        kind_clause = ""
        if kind:
            kind_clause = "AND kind = ?"
            params.append(normalize_memory_kind(kind))
        params.append(max(1, min(int(limit), 500)))
        rows = self._conn.execute(
            """
            SELECT id, user_id, kind, content, tags, source, created_at,
                   occurred_at, start_at, end_at, time_granularity,
                   temporal_text, temporal_confidence, memory_type, source_id,
                   ingestion_id, evidence_ids, updated_at, privacy_level,
                   status, confidence, superseded_by, access_count,
                   last_accessed_at, strength
            FROM memories
            WHERE user_id = ? AND deleted_at IS NULL AND status = ?
            {kind_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """.format(kind_clause=kind_clause),
            tuple([user_id, ACTIVE_MEMORY_STATUS, *params[1:]]),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    # 搜索优先 FTS，失败或无结果时再退回 LIKE，兼容中文短词。
    def search(self, user_id: str, query: str, *, limit: int = 5) -> list[MemoryEvent]:
        return self.search_with_ranking(user_id, query, limit=limit).memories

    # 文本召回用可解释轻排序，旧 search() 仍保持只返回 MemoryEvent 列表。
    def search_with_ranking(self, user_id: str, query: str, *, limit: int = 5) -> MemorySearchResult:
        query = query.strip()
        if not query:
            memories = self.list_memories(user_id, limit=limit)
            return MemorySearchResult(memories=memories, ranking=[])
        limit = max(1, min(int(limit), 20))
        candidate_limit = max(limit * 4, limit)
        rows = []
        try:
            rows = self._conn.execute(
                """
                SELECT m.id, m.user_id, m.kind, m.content, m.tags, m.source,
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
                ORDER BY bm25(memories_fts), m.created_at DESC
                LIMIT ?
                """,
                (self._fts_query(query), user_id, ACTIVE_MEMORY_STATUS, candidate_limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            rows = self._search_like(user_id, query, candidate_limit)
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
    ) -> list[MemoryEvent]:
        if start_at >= end_at:
            return []
        limit = max(1, min(int(limit), 100))
        rows = self._conn.execute(
            """
            SELECT id, user_id, kind, content, tags, source, created_at,
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
              AND COALESCE(start_at, occurred_at) IS NOT NULL
              AND COALESCE(start_at, occurred_at) < ?
              AND COALESCE(end_at, start_at + 0.001, occurred_at + 0.001) > ?
            ORDER BY COALESCE(start_at, occurred_at) ASC, created_at DESC
            LIMIT ?
            """,
            (user_id, ACTIVE_MEMORY_STATUS, end_at, start_at, limit),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    # 未来计划召回会用无时间事件兜底，避免漏掉未解析出时间的待办。
    def list_recent_untimed_events(
        self,
        user_id: str,
        *,
        limit: int = 5,
    ) -> list[MemoryEvent]:
        limit = max(1, min(int(limit), 20))
        rows = self._conn.execute(
            """
            SELECT id, user_id, kind, content, tags, source, created_at,
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
              AND COALESCE(start_at, occurred_at) IS NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, ACTIVE_MEMORY_STATUS, limit),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _search_like(self, user_id: str, query: str, limit: int) -> list[sqlite3.Row]:
        terms = self._fallback_terms(query)
        clauses = []
        params: list[Any] = [user_id]
        for term in terms:
            clauses.append("(content LIKE ? OR tags LIKE ?)")
            like = f"%{term}%"
            params.extend([like, like])
        if not clauses:
            return []
        params.append(limit)
        return self._conn.execute(
            f"""
            SELECT id, user_id, kind, content, tags, source, created_at,
                   occurred_at, start_at, end_at, time_granularity,
                   temporal_text, temporal_confidence, memory_type, source_id,
                   ingestion_id, evidence_ids, updated_at, privacy_level,
                   status, confidence, superseded_by, access_count,
                   last_accessed_at, strength
            FROM memories
            WHERE user_id = ? AND deleted_at IS NULL AND status = ?
              AND ({" OR ".join(clauses)})
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple([user_id, ACTIVE_MEMORY_STATUS, *params[1:]]),
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

    def get_memory(self, user_id: str, memory_id: str) -> MemoryEvent | None:
        return self._get_memory(user_id, memory_id)

    def get_memory_any_status(self, user_id: str, memory_id: str) -> MemoryEvent | None:
        return self._get_memory(user_id, memory_id, include_deleted=True)

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
    ) -> MemoryEvent | None:
        normalized = _normalize_for_dedupe(content)
        if not normalized:
            return None
        candidates = self.list_memories(user_id, limit=50, kind=kind)
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
        if existing is None or not can_transition_memory_status(
            existing.status,
            SUPERSEDED_MEMORY_STATUS,
            superseded_by=superseded_by,
        ):
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

    def _get_memory(self, user_id: str, memory_id: str, *, include_deleted: bool = False) -> MemoryEvent | None:
        deleted_clause = "" if include_deleted else "AND deleted_at IS NULL"
        row = self._conn.execute(
            f"""
            SELECT id, user_id, kind, content, tags, source, created_at,
                   occurred_at, start_at, end_at, time_granularity,
                   temporal_text, temporal_confidence, memory_type, source_id,
                   ingestion_id, evidence_ids, updated_at, privacy_level,
                   status, confidence, superseded_by, access_count,
                   last_accessed_at, strength
            FROM memories
            WHERE user_id = ? AND id = ? {deleted_clause}
            """,
            (user_id, memory_id),
        ).fetchone()
        return self._row_to_event(row) if row else None

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
