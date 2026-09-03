from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .memory_candidate import IntentDecision, MemoryWriteCandidate
from .intent_policy import sensitive_input_reason
from .temporal_parser import TemporalResolution
from .audio_engine.contracts import AudioEvent, AudioEventPlan


COMPLETE_SET_ANSWER_INTENTS = frozenset({"count_or_total", "multi_fact"})

_ENGLISH_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTH_FIRST_DATE_RE = re.compile(
    r"\b(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"(?P<start>\d{1,2})(?:st|nd|rd|th)?(?:\s*(?:-|to|and)\s*(?P<end>\d{1,2})(?:st|nd|rd|th)?)?\b",
    re.IGNORECASE,
)
_DAY_FIRST_DATE_RE = re.compile(
    r"\b(?P<start>\d{1,2})(?:st|nd|rd|th)?(?:\s*(?:-|to|and)\s*(?P<end>\d{1,2})(?:st|nd|rd|th)?)?\s+"
    r"(?:of\s+)?(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_NUMERIC_MONTH_DAY_RE = re.compile(r"(?<!\d)(?P<month>0?[1-9]|1[0-2])/(?P<start>0?[1-9]|[12]\d|3[01])(?!\d)")


def coverage_requirement_for_answer(
    answer_intent: str,
    answer_obligations: list[str],
) -> str:
    """Map the classifier-owned answer contract to a non-semantic execution mode."""

    if answer_intent in COMPLETE_SET_ANSWER_INTENTS and "count_scope" in answer_obligations:
        return "complete_set"
    return "best_evidence"


@dataclass(frozen=True)
class TurnPlan:
    needs_location: bool = False
    location_text: str = ""
    needs_web_search: bool = False
    web_query: str | None = None
    web_reason: str = ""
    needs_profile_memory: bool = False
    needs_event_memory: bool = False
    needs_observation_memory: bool = False
    needs_timeline_recall: bool = False
    timeline_query: str | None = None
    timeline_reason: str = ""
    needs_discussion_recall: bool = False
    discussion_query: str | None = None
    recall_goal: str = "none"
    conversation_action: str = ""
    recall_subject_names: list[str] = field(default_factory=list)
    recall_subject_scope: str = "self"
    evidence_scope: str = "personal"
    discussion_relation_scope: str = "topic"
    memory_write_candidates: list[MemoryWriteCandidate] = field(default_factory=list)
    temporal_scope: TemporalResolution = field(default_factory=TemporalResolution)
    reply_mode: str = "llm"
    fast_path: bool = False
    fast_path_kind: str = ""
    event_recall_strategy: str = "skipped"
    temporal_query: dict[str, Any] = field(default_factory=dict)
    document_query: dict[str, Any] = field(default_factory=dict)
    answer_intent: str = "direct_answer"
    answer_focus: str = ""
    answer_obligations: list[str] = field(default_factory=list)
    uncertainty_policy: str = "none"
    coverage_requirement: str = "best_evidence"
    reason: str = ""

    # 让本地 planner 的结果兼容旧的 IntentDecision 调用点。
    def as_intent_decision(self) -> IntentDecision:
        return IntentDecision(
            needs_web_search=self.needs_web_search,
            web_query=self.web_query,
            web_reason=self.web_reason,
            memory_write_candidates=self.memory_write_candidates,
            confidence=1.0,
            backend="local_planner",
        )

    # debug 中明确标出哪些 LLM 阶段已被本地 planner 替代。
    def debug_payload(self) -> dict[str, Any]:
        skipped = ["intent_classification", "query_temporal_resolution"]
        return {
            "backend": "local",
            "needs_location": self.needs_location,
            "location_text": self.location_text,
            "needs_web_search": self.needs_web_search,
            "web_query": self.web_query,
            "web_reason": self.web_reason,
            "needs_profile_memory": self.needs_profile_memory,
            "needs_event_memory": self.needs_event_memory,
            "needs_observation_memory": self.needs_observation_memory,
            "needs_timeline_recall": self.needs_timeline_recall,
            "timeline_query": self.timeline_query,
            "timeline_reason": self.timeline_reason,
            "needs_discussion_recall": self.needs_discussion_recall,
            "discussion_query": self.discussion_query,
            "recall_goal": self.recall_goal,
            "conversation_action": self.conversation_action,
            "recall_subject_names": list(self.recall_subject_names),
            "recall_subject_scope": self.recall_subject_scope,
            "evidence_scope": self.evidence_scope,
            "discussion_relation_scope": self.discussion_relation_scope,
            "memory_write_count": len(self.memory_write_candidates),
            "reply_mode": self.reply_mode,
            "fast_path": self.fast_path,
            "fast_path_kind": self.fast_path_kind,
            "event_recall_strategy": self.event_recall_strategy,
            "temporal_query": dict(self.temporal_query),
            "document_query": dict(self.document_query),
            "answer_intent": self.answer_intent,
            "answer_focus": self.answer_focus,
            "answer_obligations": list(self.answer_obligations),
            "uncertainty_policy": self.uncertainty_policy,
            "coverage_requirement": self.coverage_requirement,
            "reason": self.reason,
            "skipped_stages": skipped,
            "temporal": self.temporal_scope.debug_payload(),
        }

    # PreReplyDecision 是普通问题唯一的开放语义路由权威。
    # 此方法把 LLM 决策映射成可执行的 TurnPlan 字段，不添加基于消息文本的
    # 第二语义判断、不 OR 合并 baseline 猜测、不恢复显式 none 决策。
    def apply_pre_reply_decision(self, decision: Any) -> "TurnPlan":
        recall_type = str(getattr(decision, "memory_recall_type", "") or "none")
        recall_goal = str(getattr(decision, "recall_goal", "") or "none")
        route_profile = bool(getattr(decision, "needs_profile_memory", False)) or recall_type == "profile"
        route_event = bool(getattr(decision, "needs_event_memory", False)) or recall_type in {"event", "observation"}
        route_timeline = bool(getattr(decision, "needs_timeline_recall", False)) or recall_type == "timeline"
        route_discussion = bool(getattr(decision, "needs_discussion_recall", False))
        route_observation = recall_type == "observation"
        turn_intent = str(getattr(decision, "turn_intent", "") or "chat")
        cross_kind_specific_fact = (
            recall_goal == "specific_fact"
            and route_profile
            and not self.temporal_scope.has_temporal_expression
        )
        if cross_kind_specific_fact:
            route_event = True
        reply_mode = str(getattr(decision, "reply_mode", "") or "llm")
        profile_context_for_llm = turn_intent == "mixed" and route_profile and recall_goal == "summary"
        decision_timeline_query = getattr(decision, "timeline_query", None)
        timeline_query = decision_timeline_query if route_timeline and decision_timeline_query else None
        timeline_reason = "pre_reply_decision_timeline_query" if timeline_query else ""
        event_recall_strategy = str(getattr(decision, "event_recall_strategy", "") or "skipped")
        decision_reason = f"pre_reply_decision:{decision.reason}" if getattr(decision, "reason", "") else "pre_reply_decision"
        if cross_kind_specific_fact:
            decision_reason += "|specific_fact_cross_kind"
        if profile_context_for_llm:
            decision_reason += "|profile_context_for_llm"
        temporal_query = dict(getattr(decision, "temporal_query", {}) or {})
        temporal_scope = _temporal_resolution_from_decision(temporal_query, fallback=self.temporal_scope)
        answer_intent = str(getattr(decision, "answer_intent", "") or "direct_answer")
        answer_focus = str(getattr(decision, "answer_focus", "") or "").strip()
        answer_obligations = list(dict.fromkeys(
            str(item).strip()
            for item in (getattr(decision, "answer_obligations", []) or [])
            if str(item).strip()
        ))
        uncertainty_policy = str(getattr(decision, "uncertainty_policy", "") or "none")
        return TurnPlan(
            needs_location=bool(getattr(decision, "needs_location", False)),
            location_text=str(getattr(decision, "location_text", "") or ""),
            needs_web_search=bool(getattr(decision, "needs_web_search", False)),
            web_query=getattr(decision, "web_query", None),
            web_reason=str(getattr(decision, "web_reason", "") or ""),
            needs_profile_memory=route_profile,
            needs_event_memory=route_event,
            needs_observation_memory=route_observation,
            needs_timeline_recall=route_timeline,
            timeline_query=timeline_query,
            timeline_reason=timeline_reason,
            needs_discussion_recall=route_discussion,
            discussion_query=(
                str(getattr(decision, "discussion_query", "") or self.discussion_query or "").strip() or None
                if route_discussion else None
            ),
            recall_goal=recall_goal,
            conversation_action=str(getattr(decision, "conversation_action", "") or ""),
            recall_subject_names=list(getattr(decision, "recall_subject_names", []) or []),
            recall_subject_scope=str(getattr(decision, "recall_subject_scope", "") or "self"),
            evidence_scope=str(getattr(decision, "evidence_scope", "") or "personal"),
            discussion_relation_scope=str(
                getattr(decision, "discussion_relation_scope", "") or "topic"
            ),
            memory_write_candidates=[],
            temporal_scope=temporal_scope,
            reply_mode=reply_mode,
            fast_path=self.fast_path,
            fast_path_kind=self.fast_path_kind,
            event_recall_strategy=event_recall_strategy,
            temporal_query=dict(getattr(decision, "temporal_query", {}) or {}),
            document_query=dict(getattr(decision, "document_query", {}) or {}),
            answer_intent=answer_intent,
            answer_focus=answer_focus,
            answer_obligations=answer_obligations,
            uncertainty_policy=uncertainty_policy,
            # Coverage is part of the same PreReplyDecision contract. Do not
            # reconstruct it here from user text or a second semantic planner.
            coverage_requirement=str(
                getattr(decision, "coverage_requirement", "") or "best_evidence"
            ),
            reason=decision_reason,
        )


def _temporal_resolution_from_decision(
    query: dict[str, Any],
    *,
    fallback: TemporalResolution,
) -> TemporalResolution:
    """Convert classifier-owned temporal parameters without inspecting user wording."""

    if not query or not bool(query.get("has_expression")):
        return fallback
    raw_protocol = query.get("protocol")
    protocol = dict(raw_protocol) if isinstance(raw_protocol, dict) else {}
    start_at = query.get("start_at")
    end_at = query.get("end_at")
    if isinstance(start_at, bool) or not isinstance(start_at, (int, float)) or not math.isfinite(start_at):
        start_at = None
    if isinstance(end_at, bool) or not isinstance(end_at, (int, float)) or not math.isfinite(end_at):
        end_at = None
    if start_at is None or end_at is None or end_at <= start_at:
        return TemporalResolution(
            has_temporal_expression=True,
            temporal_text=str(query.get("normalized_query") or "").strip(),
            timezone=str(query.get("timezone") or ""),
            granularity=str(query.get("granularity") or "unknown"),
            backend="pre_reply_decision_invalid_range",
            confidence=0.0,
            reason="ppd_temporal_contract_invalid",
            error=str(protocol.get("error") or "structured_temporal_query_missing_valid_range"),
            protocol=protocol,
        )
    return TemporalResolution(
        has_temporal_expression=True,
        start_at=start_at,
        end_at=end_at,
        granularity=str(query.get("granularity") or "unknown"),
        timezone=str(query.get("timezone") or ""),
        normalized_text=str(query.get("normalized_query") or "").strip(),
        confidence=1.0,
        backend="pre_reply_decision",
        reason="ppd_temporal_contract_applied",
        protocol=protocol,
    )


def plan_audio_event(event: AudioEvent) -> AudioEventPlan:
    """Route audio events without allowing partial text to acquire side effects."""

    if event.event_type == "transcript_partial" or not event.final:
        return AudioEventPlan("ui_only", "partial_or_state_event", memory_eligible=False)
    if event.event_type == "speech_rejected":
        return AudioEventPlan("drop", "speech_rejected", memory_eligible=False)
    if event.event_type == "speaker_update" and event.lane == "enrollment":
        return AudioEventPlan("enroll", "speaker_enrollment_final", memory_eligible=False)
    if event.event_type != "transcript_final" or not event.text.strip():
        return AudioEventPlan("drop", "final_without_transcript", memory_eligible=False)

    speaker_state = str(event.speaker.get("state") or "unknown")
    speaker_reason = str(event.speaker.get("reason") or "")
    overlap_state = str(event.overlap.get("state") or "unknown")
    safe_identity = speaker_state == "user" and overlap_state == "not_observed"
    if event.lane == "ambient":
        return AudioEventPlan("capture", "ambient_final", memory_eligible=safe_identity)
    if event.lane == "assistant":
        if speaker_state == "other":
            return AudioEventPlan("drop", "assistant_other_speaker", memory_eligible=False)
        if speaker_state == "unknown" and speaker_reason != "reference_unavailable":
            return AudioEventPlan("drop", "assistant_low_confidence_speaker", memory_eligible=False)
        return AudioEventPlan("chat", "assistant_final", memory_eligible=safe_identity)
    return AudioEventPlan("drop", "unsupported_audio_lane", memory_eligible=False)


def native_location_preflight(message: str) -> dict[str, Any]:
    """Keep the Android compatibility hook non-authoritative.

    Location need is an output of ``PreReplyDecision``.  This hook may still be
    called by native code, but it must never turn message wording into a device
    location request.
    """

    return {
        "needed": False,
        "reason": "pre_reply_decision_deferred",
        "authority": "pre_reply_decision",
    }

# 确定性 preflight 入口：只做输入校验、安全/隐私门控、确定性 fast path
# 和原始时间解析。开放语义判断（memory/web/location/discussion 召回）由
# PreReplyDecision 单一权威负责。planner 不再自行推断这些语义。
def plan_turn(
    message: str,
    *,
    reference_time: float,
    timezone: str = "",
    allow_continuous_capture: bool = False,
) -> TurnPlan:
    text = _compact(message)
    sensitive_reason = sensitive_input_reason(text)
    long_input = _long_input_signal(text)
    if sensitive_reason:
        return TurnPlan(
            reply_mode="sensitive_credential_rejected",
            fast_path=True,
            fast_path_kind="sensitive_credential",
            reason=f"safety_gate:{sensitive_reason}",
        )
    if allow_continuous_capture and long_input["should_capture"]:
        return TurnPlan(
            reply_mode="continuous_capture",
            fast_path=True,
            fast_path_kind="continuous_capture",
            reason="structural_long_input:" + str(long_input["reason"]),
        )

    # 非 fast-path 下，planner 只提供确定性 preflight baseline；
    # 开放语义决策（discussion recall、memory、web、location）由 PreReplyDecision 单一权威负责。
    return TurnPlan(
        needs_location=False,
        location_text="",
        needs_web_search=False,
        web_query=None,
        web_reason="",
        needs_profile_memory=False,
        needs_event_memory=False,
        needs_observation_memory=False,
        needs_timeline_recall=False,
        timeline_query=None,
        timeline_reason="",
        needs_discussion_recall=False,
        discussion_query=None,
        recall_goal="none",
        conversation_action="",
        memory_write_candidates=[],
        temporal_scope=TemporalResolution(backend="awaiting_pre_reply_decision"),
        reply_mode="llm",
        event_recall_strategy="skipped",
        reason="default_pre_reply_decision_required",
    )


# Local parsing only normalizes an already authorized temporal value; it does
# not select an open-semantic recall route.
def resolve_temporal_local(
    message: str,
    *,
    reference_time: float,
    timezone: str = "",
) -> TemporalResolution:
    tzinfo = _timezone_info(timezone)
    reference_dt = datetime.fromtimestamp(reference_time, tzinfo)
    tz_name = timezone or reference_dt.tzname() or "local"
    text = _compact(message)
    temporal_text = ""
    normalized_text = ""
    start_dt: datetime | None = None
    end_dt: datetime | None = None
    granularity = "unknown"
    reason = ""

    # 日期、星期、时间段和钟点分开解析，最后合成半开时间范围。
    day_base, day_text = _resolve_day_base(text, reference_dt)
    weekday_base, weekday_text = _resolve_weekday(text, reference_dt)
    if weekday_base is not None:
        day_base = weekday_base
        day_text = weekday_text

    time_window = _resolve_time_window(text)
    clock = _resolve_clock(text)
    english_calendar = _resolve_english_calendar_range(text, reference_dt)

    if any(marker in text for marker in ("接下来", "后面", "未来", "这两天")):
        start_dt = reference_dt
        end_dt = reference_dt + timedelta(days=7)
        temporal_text = _first_match(text, ("接下来", "后面几天", "后面", "未来", "这两天"))
        granularity = "day"
        reason = "future temporal expression resolved to the next seven days"
    elif english_calendar is not None:
        start_dt, end_dt, temporal_text = english_calendar
        normalized_text = _strip_memory_prefix(_remove_first(text, temporal_text))
        granularity = "day"
        reason = "local explicit English calendar date"
    elif "最近" in text or "近期" in text:
        start_dt = _start_of_day(reference_dt - timedelta(days=7))
        end_dt = reference_dt
        temporal_text = "近期" if "近期" in text else "最近"
        granularity = "day"
        reason = "recent recall resolved to the previous seven days"
    elif day_base is not None:
        if clock is not None:
            start_dt = _at_time(day_base, clock)
            end_dt = start_dt + timedelta(hours=1)
            window_text = time_window[2] if time_window is not None else ""
            temporal_text = _join_temporal_text(_join_temporal_text(day_text, window_text), clock[1])
            normalized_text = _strip_memory_prefix(_remove_first(text, temporal_text))
            granularity = "hour"
            reason = "local day and clock expression"
        elif time_window is not None:
            start_time, end_time, window_text = time_window
            start_dt = _at_time(day_base, (start_time, window_text))
            if end_time == dt_time(0, 0):
                end_dt = _start_of_day(day_base + timedelta(days=1))
            else:
                end_dt = _at_time(day_base, (end_time, window_text))
            temporal_text = _join_temporal_text(day_text, window_text)
            normalized_text = _strip_memory_prefix(_remove_first(text, temporal_text))
            granularity = "hour"
            reason = "local day and part-of-day expression"
        else:
            start_dt = _start_of_day(day_base)
            end_dt = start_dt + timedelta(days=1)
            temporal_text = day_text
            normalized_text = _strip_memory_prefix(_remove_first(text, temporal_text))
            granularity = "day"
            reason = "local day expression"
    elif time_window is not None:
        start_time, end_time, window_text = time_window
        start_dt = _at_time(reference_dt, (start_time, window_text))
        if clock is not None:
            start_dt = _at_time(reference_dt, clock)
            end_dt = start_dt + timedelta(hours=1)
            temporal_text = _join_temporal_text(window_text, clock[1])
            reason = "local part-of-day clock expression"
        else:
            end_dt = _start_of_day(reference_dt + timedelta(days=1)) if end_time == dt_time(0, 0) else _at_time(reference_dt, (end_time, window_text))
            temporal_text = window_text
            reason = "local part-of-day range expression"
        normalized_text = _strip_memory_prefix(_remove_first(text, temporal_text))
        granularity = "hour"

    # Unresolved wording stays opaque; a structured classifier/temporal provider
    # may supply numeric bounds later, but this helper never creates a route.
    if start_dt is None or end_dt is None:
        return TemporalResolution(timezone=tz_name, backend="local_none", reason="no structured temporal range")

    return TemporalResolution(
        has_temporal_expression=True,
        temporal_text=temporal_text,
        kind="instant" if granularity == "hour" and clock is not None else "date_range",
        start_at=start_dt.timestamp(),
        end_at=end_dt.timestamp(),
        granularity=granularity,
        timezone=tz_name,
        normalized_text=normalized_text,
        confidence=0.95,
        backend="local",
        reason=reason,
    )


def _resolve_english_calendar_range(
    text: str,
    reference_dt: datetime,
) -> tuple[datetime, datetime, str] | None:
    """Resolve unambiguous English calendar dates for local event writing."""

    match = _MONTH_FIRST_DATE_RE.search(text) or _DAY_FIRST_DATE_RE.search(text)
    if match is not None:
        month = _ENGLISH_MONTHS[match.group("month").casefold()]
        start_day = int(match.group("start"))
        end_day = int(match.group("end") or start_day)
    else:
        match = _NUMERIC_MONTH_DAY_RE.search(text)
        if match is None:
            return None
        month = int(match.group("month"))
        start_day = int(match.group("start"))
        end_day = start_day
    year = _nearest_calendar_year(reference_dt, month, start_day)
    try:
        start_dt = reference_dt.replace(
            year=year, month=month, day=start_day, hour=0, minute=0, second=0, microsecond=0
        )
        end_dt = reference_dt.replace(
            year=year, month=month, day=end_day, hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
    except ValueError:
        return None
    return start_dt, end_dt, match.group(0)


def _nearest_calendar_year(reference_dt: datetime, month: int, day: int) -> int:
    """Infer only the closest year for an explicit month/day without inventing a date."""

    try:
        candidate = reference_dt.replace(month=month, day=day)
    except ValueError:
        return reference_dt.year
    if candidate - reference_dt > timedelta(days=183):
        return reference_dt.year - 1
    if reference_dt - candidate > timedelta(days=183):
        return reference_dt.year + 1
    return reference_dt.year


def _canonical_text(message: str) -> str:
    return _compact(message).strip("。！？!? ")


def _long_input_signal(text: str) -> dict[str, Any]:
    if len(text) < 90:
        return {"should_capture": False, "reason": "too_short"}
    sentence_count = len([part for part in re.split(r"[。！？!?；;，,\n]+", text) if part.strip()])
    line_count = len([part for part in text.splitlines() if part.strip()])
    if sentence_count < 3 and line_count < 3:
        return {"should_capture": False, "reason": "not_enough_segments"}
    return {
        "should_capture": True,
        "reason": f"chars={len(text)},sentences={sentence_count},lines={line_count}",
    }


def _resolve_day_base(text: str, reference_dt: datetime) -> tuple[datetime | None, str]:
    if "后天" in text:
        return reference_dt + timedelta(days=2), "后天"
    if "明天" in text:
        return reference_dt + timedelta(days=1), "明天"
    if "昨天" in text:
        return reference_dt - timedelta(days=1), "昨天"
    if "今天" in text or "今晚" in text:
        return reference_dt, "今晚" if "今晚" in text else "今天"
    return None, ""


def _resolve_weekday(text: str, reference_dt: datetime) -> tuple[datetime | None, str]:
    weekdays = {
        "周一": 0,
        "周二": 1,
        "周三": 2,
        "周四": 3,
        "周五": 4,
        "周六": 5,
        "周日": 6,
        "周天": 6,
        "星期一": 0,
        "星期二": 1,
        "星期三": 2,
        "星期四": 3,
        "星期五": 4,
        "星期六": 5,
        "星期日": 6,
        "星期天": 6,
    }
    for marker, weekday in weekdays.items():
        if marker not in text:
            continue
        delta = (weekday - reference_dt.weekday()) % 7
        return reference_dt + timedelta(days=delta), marker
    return None, ""


def _resolve_time_window(text: str) -> tuple[dt_time, dt_time, str] | None:
    if "上午" in text or "早上" in text:
        return dt_time(6, 0), dt_time(12, 0), "上午" if "上午" in text else "早上"
    if "中午" in text:
        return dt_time(11, 0), dt_time(14, 0), "中午"
    if "下午" in text:
        return dt_time(12, 0), dt_time(18, 0), "下午"
    if "今晚" in text:
        return dt_time(18, 0), dt_time(0, 0), "今晚"
    if "傍晚" in text:
        return dt_time(17, 0), dt_time(20, 0), "傍晚"
    if "晚上" in text:
        return dt_time(18, 0), dt_time(0, 0), "晚上"
    return None


def _resolve_clock(text: str) -> tuple[dt_time, str] | None:
    match = re.search(r"(?P<hour>\d{1,2}|[一二三四五六七八九十两]+)\s*[点:：]\s*(?P<minute>\d{1,2}|半)?", text)
    if not match:
        return None
    hour = _parse_hour(match.group("hour"))
    minute_text = match.group("minute") or ""
    minute = 30 if minute_text == "半" else int(minute_text) if minute_text.isdigit() else 0
    if hour is None or hour > 23 or minute > 59:
        return None
    if any(marker in text for marker in ("下午", "晚上", "今晚")) and 1 <= hour < 12:
        hour += 12
    clock_text = match.group(0)
    return dt_time(hour, minute), clock_text


def _parse_hour(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if text == "十":
        return 10
    if text.startswith("十") and len(text) == 2:
        return 10 + digits.get(text[1], 0)
    if text.endswith("十") and len(text) == 2:
        return digits.get(text[0], 0) * 10
    if "十" in text:
        left, right = text.split("十", 1)
        return digits.get(left, 1) * 10 + digits.get(right, 0)
    return digits.get(text)


def _timezone_info(timezone: str):
    if timezone:
        try:
            return ZoneInfo(timezone)
        except Exception:
            pass
    return datetime.fromtimestamp(0).astimezone().tzinfo


def _at_time(day: datetime, value: tuple[dt_time, str]) -> datetime:
    tm, _ = value
    return day.replace(hour=tm.hour, minute=tm.minute, second=0, microsecond=0)


def _start_of_day(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def _join_temporal_text(day_text: str, time_text: str) -> str:
    if day_text and time_text and time_text not in day_text:
        return f"{day_text}{time_text}"
    return day_text or time_text


def _strip_memory_prefix(text: str) -> str:
    # Memory-write intent is represented by the structured candidate; this
    # helper only trims structural punctuation around an authorized value.
    return _compact(text).strip(" ：:，,")


def _remove_first(text: str, phrase: str) -> str:
    if not phrase:
        return text
    return text.replace(phrase, "", 1).strip(" ，,。.!！?？：:")


def _first_match(text: str, terms: tuple[str, ...]) -> str:
    for term in terms:
        if term in text:
            return term
    return ""


def _compact(message: str) -> str:
    return " ".join(str(message or "").strip().split())
