from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
"""


class AppSessionStore:
    """Minimal session store used by the AI glasses demo and Hermes fallback."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = Lock()
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()

    def create_session(self, session_id: str, source: str, **kwargs: Any) -> str:
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO sessions (
                    id, source, user_id, model, model_config, system_prompt,
                    parent_session_id, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    source,
                    kwargs.get("user_id"),
                    kwargs.get("model"),
                    json.dumps(kwargs.get("model_config")) if kwargs.get("model_config") else None,
                    kwargs.get("system_prompt"),
                    kwargs.get("parent_session_id"),
                    time.time(),
                ),
            )
            self._conn.commit()
        return session_id

    def end_session(self, session_id: str, end_reason: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            )
            self._conn.commit()

    def reopen_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (session_id,),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                (system_prompt, session_id),
            )
            self._conn.commit()

    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: Any = None,
        tool_call_id: str | None = None,
        tool_calls: Any = None,
        tool_name: str | None = None,
        token_count: int | None = None,
        finish_reason: str | None = None,
        reasoning: str | None = None,
        reasoning_content: str | None = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        codex_message_items: Any = None,
    ) -> None:
        stored_content = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else content
        tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        reasoning_details_json = json.dumps(reasoning_details, ensure_ascii=False) if reasoning_details else None
        codex_reasoning_json = json.dumps(codex_reasoning_items, ensure_ascii=False) if codex_reasoning_items else None
        codex_message_json = json.dumps(codex_message_items, ensure_ascii=False) if codex_message_items else None
        tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else int(bool(tool_calls))
        with self._lock:
            self._conn.execute(
                """INSERT INTO messages (
                    session_id, role, content, tool_call_id, tool_calls, tool_name,
                    timestamp, token_count, finish_reason, reasoning,
                    reasoning_content, reasoning_details, codex_reasoning_items,
                    codex_message_items
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    stored_content,
                    tool_call_id,
                    tool_calls_json,
                    tool_name,
                    time.time(),
                    token_count,
                    finish_reason,
                    reasoning,
                    reasoning_content,
                    reasoning_details_json,
                    codex_reasoning_json,
                    codex_message_json,
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET message_count = message_count + 1, tool_call_count = tool_call_count + ? WHERE id = ?",
                (tool_call_count, session_id),
            )
            self._conn.commit()

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
                (session_id,),
            ).fetchall()
        return [self._decode_message(dict(row)) for row in rows]

    def update_token_counts(
        self,
        session_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: float | None = None,
        actual_cost_usd: float | None = None,
        cost_status: str | None = None,
        cost_source: str | None = None,
        pricing_version: str | None = None,
        billing_provider: str | None = None,
        billing_base_url: str | None = None,
        billing_mode: str | None = None,
        api_call_count: int = 0,
        **_: Any,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE sessions SET
                    input_tokens = input_tokens + ?,
                    output_tokens = output_tokens + ?,
                    cache_read_tokens = cache_read_tokens + ?,
                    cache_write_tokens = cache_write_tokens + ?,
                    reasoning_tokens = reasoning_tokens + ?,
                    estimated_cost_usd = COALESCE(?, estimated_cost_usd),
                    actual_cost_usd = COALESCE(?, actual_cost_usd),
                    cost_status = COALESCE(?, cost_status),
                    cost_source = COALESCE(?, cost_source),
                    pricing_version = COALESCE(?, pricing_version),
                    billing_provider = COALESCE(?, billing_provider),
                    billing_base_url = COALESCE(?, billing_base_url),
                    billing_mode = COALESCE(?, billing_mode),
                    api_call_count = api_call_count + ?
                WHERE id = ?""",
                (
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    reasoning_tokens,
                    estimated_cost_usd,
                    actual_cost_usd,
                    cost_status,
                    cost_source,
                    pricing_version,
                    billing_provider,
                    billing_base_url,
                    billing_mode,
                    api_call_count,
                    session_id,
                ),
            )
            self._conn.commit()

    def set_session_title(self, session_id: str, title: str) -> bool:
        normalized = title.strip() if title else None
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (normalized, session_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def get_session_title(self, session_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT title FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return row["title"] if row else None

    def get_next_title_in_lineage(self, title: str | None) -> str | None:
        if not title:
            return None
        return f"{title} (continued)"

    def message_count(self, session_id: str | None = None) -> int:
        with self._lock:
            if session_id:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS count FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) AS count FROM messages").fetchone()
        return int(row["count"] if row else 0)

    @staticmethod
    def _decode_message(row: dict[str, Any]) -> dict[str, Any]:
        content = row.get("content")
        if isinstance(content, str) and content[:1] in {"[", "{"}:
            try:
                row["content"] = json.loads(content)
            except json.JSONDecodeError:
                pass
        for key in ("tool_calls", "reasoning_details", "codex_reasoning_items", "codex_message_items"):
            if row.get(key):
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    row[key] = None
        return row


def create_session_store(db_path: str | Path) -> AppSessionStore:
    return AppSessionStore(db_path=db_path)
