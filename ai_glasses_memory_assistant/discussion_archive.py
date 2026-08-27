from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


def _env_positive_int(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class DiscussionArchiveSettings:
    raw_retention_days: int = 30
    idle_gap_seconds: int = 180
    max_slice_seconds: int = 900
    max_slice_segments: int = 40
    query_wait_seconds: int = 15
    purge_batch_size: int = 500
    recovery_batch_size: int = 500

    @classmethod
    def from_env(cls) -> "DiscussionArchiveSettings":
        return cls(
            raw_retention_days=_env_positive_int(
                "AI_GLASSES_DISCUSSION_RAW_RETENTION_DAYS", cls.raw_retention_days
            ),
            idle_gap_seconds=_env_positive_int(
                "AI_GLASSES_DISCUSSION_IDLE_GAP_SECONDS", cls.idle_gap_seconds
            ),
            max_slice_seconds=_env_positive_int(
                "AI_GLASSES_DISCUSSION_MAX_SLICE_SECONDS", cls.max_slice_seconds
            ),
            max_slice_segments=_env_positive_int(
                "AI_GLASSES_DISCUSSION_MAX_SLICE_SEGMENTS", cls.max_slice_segments
            ),
            query_wait_seconds=_env_positive_int(
                "AI_GLASSES_DISCUSSION_QUERY_WAIT_SECONDS", cls.query_wait_seconds
            ),
            purge_batch_size=_env_positive_int(
                "AI_GLASSES_DISCUSSION_PURGE_BATCH_SIZE", cls.purge_batch_size
            ),
            recovery_batch_size=_env_positive_int(
                "AI_GLASSES_DISCUSSION_RECOVERY_BATCH_SIZE", cls.recovery_batch_size
            ),
        )


def timezone_info(name: str):
    if name:
        try:
            return ZoneInfo(name)
        except (KeyError, ValueError):
            pass
    return datetime.fromtimestamp(0).astimezone().tzinfo


def local_day_key(timestamp: float, timezone: str = "") -> str:
    return datetime.fromtimestamp(float(timestamp), timezone_info(timezone)).date().isoformat()


def should_close_before_append(
    chunks: list[dict[str, Any]],
    next_chunk: dict[str, Any],
    *,
    settings: DiscussionArchiveSettings,
    timezone: str = "",
) -> bool:
    if not chunks:
        return False
    first_at = float(chunks[0].get("timestamp") or 0)
    last_at = float(chunks[-1].get("timestamp") or 0)
    next_at = float(next_chunk.get("timestamp") or 0)
    return bool(
        local_day_key(first_at, timezone) != local_day_key(next_at, timezone)
        or next_at - last_at >= settings.idle_gap_seconds
        or next_at - first_at >= settings.max_slice_seconds
        or len(chunks) >= settings.max_slice_segments
    )


def summarize_discussion_slice(
    agent: Any | None,
    *,
    chunks: list[dict[str, Any]],
    existing_topics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    allowed_chunk_ids = {
        str(chunk.get("chunk_id") or "") for chunk in chunks if str(chunk.get("chunk_id") or "")
    }
    allowed_topic_ids = {
        str(topic.get("id") or "") for topic in existing_topics if str(topic.get("id") or "")
    }
    if agent is not None:
        prompt = {
            "chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "timestamp": chunk.get("timestamp"),
                    "text": chunk.get("text"),
                    "speaker_label": (chunk.get("metadata") or {}).get("speaker_label"),
                }
                for chunk in chunks
            ],
            "existing_topics": [
                {
                    "id": topic.get("id"),
                    "title": topic.get("title"),
                    "summary": topic.get("summary"),
                    "topic_key": topic.get("topic_key"),
                }
                for topic in existing_topics
            ],
        }
        try:
            result = agent.run_conversation(
                json.dumps(prompt, ensure_ascii=False),
                system_message=(
                    "You are an internal discussion archive summarizer. Output strict JSON only. "
                    "Split the supplied transcript chunks into coherent topics. Use only supplied chunk IDs. "
                    "Write every user-facing text field (title, topic_key, summary, key_points, decisions, "
                    "tasks, and open_questions) in Simplified Chinese, even when the source contains another "
                    "language. Keep proper names accurate and do not add facts while translating. "
                    "For each topic return title, topic_key, summary, key_points, decisions, tasks, "
                    "open_questions, participant_labels, source_chunk_ids, and merge_topic_id. "
                    "merge_topic_id must be empty or one supplied existing topic ID. If merging, summary and "
                    "lists must describe the combined topic, not only the new chunks. Shape: "
                    '{"topics":[{"title":"","topic_key":"","summary":"","key_points":[],"decisions":[],"tasks":[],"open_questions":[],"participant_labels":[],"source_chunk_ids":[],"merge_topic_id":""}]}'
                ),
                conversation_history=[],
                persist_user_message=None,
            )
            raw = str(result.get("final_response") or "").strip()
            payload = _parse_json_object(raw)
            topics = _validated_topics(
                payload.get("topics"),
                allowed_chunk_ids=allowed_chunk_ids,
                allowed_topic_ids=allowed_topic_ids,
            )
            if topics:
                return topics, "llm"
        except Exception:
            pass
    return [_fallback_topic(chunks, existing_topics)], "deterministic_fallback"


def _validated_topics(
    value: Any,
    *,
    allowed_chunk_ids: set[str],
    allowed_topic_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    topics: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        source_ids = [
            chunk_id
            for chunk_id in _string_list(item.get("source_chunk_ids"), limit=100)
            if chunk_id in allowed_chunk_ids and chunk_id not in used_ids
        ]
        title = _text(item.get("title"), 120)
        summary = _text(item.get("summary"), 1600)
        if not source_ids or not title or not summary:
            continue
        used_ids.update(source_ids)
        merge_topic_id = str(item.get("merge_topic_id") or "").strip()
        if merge_topic_id not in allowed_topic_ids:
            merge_topic_id = ""
        topics.append({
            "title": title,
            "topic_key": _topic_key(item.get("topic_key") or title),
            "summary": summary,
            "key_points": _string_list(item.get("key_points")),
            "decisions": _string_list(item.get("decisions")),
            "tasks": _string_list(item.get("tasks")),
            "open_questions": _string_list(item.get("open_questions")),
            "participant_labels": _string_list(item.get("participant_labels")),
            "source_chunk_ids": source_ids,
            "merge_topic_id": merge_topic_id,
        })
    missing = allowed_chunk_ids - used_ids
    if missing and topics:
        topics[-1]["source_chunk_ids"].extend(sorted(missing))
    return topics


def _fallback_topic(chunks: list[dict[str, Any]], existing_topics: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(chunk.get("text") or "").strip() for chunk in chunks if str(chunk.get("text") or "").strip()]
    first = texts[0] if texts else "环境讨论"
    title = _text(re.split(r"[。！？!?；;]", first, maxsplit=1)[0], 40) or "环境讨论"
    topic_key = _topic_key(title)
    merge_topic_id = next(
        (
            str(topic.get("id") or "")
            for topic in existing_topics
            if str(topic.get("topic_key") or "") == topic_key
        ),
        "",
    )
    participants = list(dict.fromkeys(
        str((chunk.get("metadata") or {}).get("speaker_label") or "").strip()
        for chunk in chunks
        if str((chunk.get("metadata") or {}).get("speaker_label") or "").strip()
    ))
    return {
        "title": title,
        "topic_key": topic_key,
        "summary": _text("；".join(texts), 1200),
        "key_points": texts[:5],
        "decisions": [],
        "tasks": [],
        "open_questions": [],
        "participant_labels": participants,
        "source_chunk_ids": [str(chunk.get("chunk_id") or "") for chunk in chunks],
        "merge_topic_id": merge_topic_id,
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("discussion summary did not return JSON")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("discussion summary JSON must be an object")
    return payload


def _string_list(value: Any, *, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(_text(item, 240) for item in value if _text(item, 240)))[:limit]


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _topic_key(value: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or "").casefold())[:80]
