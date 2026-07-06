from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any


TEMPORAL_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class TemporalResolution:
    has_temporal_expression: bool = False
    temporal_text: str = ""
    kind: str = "unknown"
    start_at: float | None = None
    end_at: float | None = None
    granularity: str = "unknown"
    timezone: str = ""
    normalized_text: str = ""
    confidence: float | None = None
    backend: str = "none"
    raw: str = ""
    reason: str = ""
    error: str = ""

    # 只有时间范围完整且置信度足够时，才允许用于事件查询/写入。
    @property
    def usable_range(self) -> bool:
        confidence = self.confidence if self.confidence is not None else 0.0
        return (
            self.has_temporal_expression
            and self.start_at is not None
            and self.end_at is not None
            and self.start_at < self.end_at
            and confidence >= TEMPORAL_CONFIDENCE_THRESHOLD
        )

    # debug 保留后端、置信度和错误，方便定位时间解析误差。
    def debug_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": self.backend,
            "has_temporal_expression": self.has_temporal_expression,
            "temporal_text": self.temporal_text,
            "kind": self.kind,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "granularity": self.granularity,
            "timezone": self.timezone,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "usable_range": self.usable_range,
            "reason": self.reason,
            "broad_time_range": bool(
                broad_time_period_label(self.temporal_text)
                and not _has_explicit_clock_time(self.temporal_text)
            ),
        }
        if self.raw:
            payload["raw"] = self.raw
        if self.error:
            payload["error"] = self.error
        return payload


# LLM 时间解析只在本地 parser 不够时使用，返回结构化时间范围。
def resolve_temporal_expression(
    agent: Any,
    message: str,
    *,
    reference_time: float,
    timezone: str = "",
) -> TemporalResolution:
    """Resolve natural-language time expressions through an isolated LLM call.

    Product code should not encode a growing table of phrases such as today,
    last month, or last year. The LLM resolves the phrase to a concrete time
    range; this module only validates the JSON and returns a safe structure.
    """

    reference_dt = datetime.fromtimestamp(reference_time).astimezone()
    tz = timezone or reference_dt.tzname() or "local"
    prompt = f"""Resolve temporal expressions for a text-first AI glasses memory assistant.

Return JSON only, with this exact shape:
{{
  "has_temporal_expression": false,
  "temporal_text": "",
  "kind": "unknown|instant|date|date_range|recurring",
  "start_at_iso": null,
  "end_at_iso": null,
  "granularity": "unknown|minute|hour|day|week|month|year",
  "normalized_text": "",
  "confidence": 0.0,
  "reason": ""
}}

Rules:
- Resolve time relative to the reference time and timezone below.
- Support broad natural-language ranges such as recent days, last week, last month, last year, holidays, seasons, explicit dates, and date ranges.
- For upcoming-plan questions such as "最近要做什么", "接下来有什么", "这两天安排", "今天还有什么", or "明天要干嘛", resolve toward the user's near future schedule instead of past history. If no tighter range is stated, use a half-open range from the reference time to 7 days after the reference time.
- For a day-level expression, return a half-open local day range: start at 00:00 and end at the next 00:00.
- For broad part-of-day expressions without a specific clock time, such as "上午", "中午", "下午", "今晚", "今天晚上", or "晚上", return a range rather than an instant. Do not treat the range start as the user's exact intended time.
- For broad evening expressions without a specific clock time, such as "今晚", "今天晚上", or "晚上", use the local half-open range 18:00 to next-day 00:00. Do not narrow broad evening to a one-hour block.
- For an instant event, prefer returning both start_at_iso and end_at_iso. If the duration is unknown, use a one-hour block.
- For month/year/week ranges, return the full local half-open range.
- normalized_text should remove only the temporal phrase when that is safe; otherwise keep it empty.
- If the text has no meaningful temporal expression, set has_temporal_expression=false and leave start/end null.
- Do not decide whether to save memory. Only resolve time.

Reference time: {reference_dt.isoformat()}
Timezone: {tz}
User text:
{message}
"""
    try:
        # 使用隔离的内部 prompt，不把 temporal parser 对话写入主历史。
        result = agent.run_conversation(
            prompt,
            system_message=(
                "You are an internal temporal parser. "
                "Output strict JSON only. Do not call tools."
            ),
            conversation_history=[],
            persist_user_message=None,
        )
        raw = (result.get("final_response") or "").strip()
        payload = _parse_json_object(raw)
        resolution = _resolution_from_payload(payload, raw=raw, backend="llm", timezone=tz)
        return _normalize_broad_evening_resolution(resolution, message=message)
    except Exception as exc:
        return TemporalResolution(
            timezone=tz,
            backend="unavailable",
            error=str(exc),
        )


# LLM JSON 在这里收敛为安全结构，非法时间范围直接降级为 unknown。
def _resolution_from_payload(
    payload: dict[str, Any],
    *,
    raw: str,
    backend: str,
    timezone: str,
) -> TemporalResolution:
    has_temporal_expression = bool(payload.get("has_temporal_expression"))
    kind = _allowed(
        str(payload.get("kind") or "unknown").strip().lower(),
        {"instant", "date", "date_range", "recurring", "unknown"},
        "unknown",
    )
    granularity = _allowed(
        str(payload.get("granularity") or "unknown").strip().lower(),
        {"minute", "hour", "day", "week", "month", "year", "unknown"},
        "unknown",
    )
    start_at = _parse_iso_timestamp(payload.get("start_at_iso"))
    end_at = _parse_iso_timestamp(payload.get("end_at_iso"))
    confidence = _optional_float(payload.get("confidence"))
    if kind == "instant" and start_at is not None and end_at is None:
        end_at = (datetime.fromtimestamp(start_at).astimezone() + timedelta(hours=1)).timestamp()
    if start_at is None or end_at is None or start_at >= end_at:
        start_at = None
        end_at = None
        if kind != "unknown":
            kind = "unknown"
    return TemporalResolution(
        has_temporal_expression=has_temporal_expression and start_at is not None and end_at is not None,
        temporal_text=str(payload.get("temporal_text") or "").strip(),
        kind=kind,
        start_at=start_at,
        end_at=end_at,
        granularity=granularity,
        timezone=timezone,
        normalized_text=str(payload.get("normalized_text") or "").strip(),
        confidence=confidence,
        backend=backend,
        raw=raw,
        reason=str(payload.get("reason") or "").strip(),
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}


def _parse_iso_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _allowed(value: str, allowed: set[str], default: str) -> str:
    return value if value in allowed else default


# 防止 LLM 把宽泛晚上误缩成一小时，统一为 18:00 到 24:00。
def _normalize_broad_evening_resolution(
    resolution: TemporalResolution,
    *,
    message: str,
) -> TemporalResolution:
    if not resolution.has_temporal_expression or resolution.start_at is None:
        return resolution
    text = f"{message} {resolution.temporal_text}".strip()
    if not _has_broad_evening_expression(text) or _has_explicit_clock_time(text):
        return resolution
    start_dt = datetime.fromtimestamp(resolution.start_at).astimezone()
    evening_start = start_dt.replace(hour=18, minute=0, second=0, microsecond=0)
    evening_end = (evening_start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    start_at = evening_start.timestamp()
    end_at = evening_end.timestamp()
    if resolution.end_at is not None and resolution.start_at <= start_at and resolution.end_at >= end_at:
        return resolution
    return replace(
        resolution,
        kind="date_range",
        start_at=start_at,
        end_at=end_at,
        granularity="hour",
        reason=(resolution.reason + " Normalized broad evening range to 18:00-24:00.").strip(),
    )


def _has_broad_evening_expression(text: str) -> bool:
    return any(term in text for term in ("今晚", "今天晚上", "晚上"))


# 提供用户可见的宽泛时间段标签，不把范围起点暴露成精确时间。
def broad_time_period_label(text: str) -> str:
    for marker, label in (
        ("早上", "早上"),
        ("上午", "上午"),
        ("中午", "中午"),
        ("下午", "下午"),
        ("今天晚上", "晚上"),
        ("今晚", "晚上"),
        ("晚上", "晚上"),
    ):
        if marker in text:
            return label
    return ""


def _has_explicit_clock_time(text: str) -> bool:
    return bool(re.search(r"(\d{1,2}|[一二三四五六七八九十两半]+)\s*[点:：]", text))
