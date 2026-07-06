from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .memory_candidate import IntentDecision, MemoryWriteCandidate
from .intent_policy import is_question
from .temporal_parser import TemporalResolution


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
    recall_goal: str = "none"
    conversation_action: str = ""
    memory_write_candidates: list[MemoryWriteCandidate] = field(default_factory=list)
    temporal_scope: TemporalResolution = field(default_factory=TemporalResolution)
    reply_mode: str = "llm"
    fast_path: bool = False
    fast_path_kind: str = ""
    event_recall_strategy: str = "skipped"
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
            "recall_goal": self.recall_goal,
            "conversation_action": self.conversation_action,
            "memory_write_count": len(self.memory_write_candidates),
            "reply_mode": self.reply_mode,
            "fast_path": self.fast_path,
            "fast_path_kind": self.fast_path_kind,
            "event_recall_strategy": self.event_recall_strategy,
            "reason": self.reason,
            "skipped_stages": skipped,
            "temporal": self.temporal_scope.debug_payload(),
        }

    # llm_first 下让 PreReplyDecision 成为最终执行判断源；planner 只负责承载执行字段。
    def apply_pre_reply_decision(self, decision: Any) -> "TurnPlan":
        recall_type = str(getattr(decision, "memory_recall_type", "") or "none")
        route_profile = bool(getattr(decision, "needs_profile_memory", False)) or recall_type == "profile"
        route_event = bool(getattr(decision, "needs_event_memory", False)) or recall_type in {"event", "observation"}
        route_timeline = bool(getattr(decision, "needs_timeline_recall", False)) or recall_type == "timeline"
        route_observation = recall_type == "observation"
        reply_mode = str(getattr(decision, "reply_mode", "") or "llm")
        if reply_mode == "llm":
            if route_timeline:
                reply_mode = "local_timeline_recall"
            elif route_profile:
                reply_mode = "local_profile_recall"
            elif route_event:
                reply_mode = "local_event_recall"
        decision_timeline_query = getattr(decision, "timeline_query", None)
        timeline_query = decision_timeline_query if route_timeline and decision_timeline_query else None
        timeline_reason = "pre_reply_decision_timeline_query" if timeline_query else ""
        event_recall_strategy = str(getattr(decision, "event_recall_strategy", "") or "skipped")
        if route_observation:
            event_recall_strategy = "observation_review"
        elif route_event and event_recall_strategy == "skipped":
            event_recall_strategy = "text_search"
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
            recall_goal=str(getattr(decision, "recall_goal", "") or "none"),
            conversation_action=str(getattr(decision, "conversation_action", "") or ""),
            memory_write_candidates=[],
            temporal_scope=self.temporal_scope,
            reply_mode=reply_mode,
            fast_path=self.fast_path,
            fast_path_kind=self.fast_path_kind,
            event_recall_strategy=event_recall_strategy,
            reason=f"pre_reply_decision:{decision.reason}" if getattr(decision, "reason", "") else "pre_reply_decision",
        )


# 本地规划入口：把用户一句话转成“是否读记忆/查 web/用定位/本地回复”的执行计划。
def plan_turn(message: str, *, reference_time: float, timezone: str = "") -> TurnPlan:
    text = _compact(message)
    lowered = text.lower()
    canonical = _canonical_text(text).lower()
    if _is_greeting(canonical):
        return TurnPlan(
            reply_mode="greeting",
            fast_path=True,
            fast_path_kind="greeting",
            reason="matched_greeting_fast_path",
        )
    if _is_identity_query(canonical):
        return TurnPlan(
            reply_mode="identity_query",
            fast_path=True,
            fast_path_kind="identity_query",
            reason="matched_identity_query_read_only",
        )
    identity_name = _identity_statement_name(text)
    if identity_name:
        return TurnPlan(
            memory_write_candidates=[
                MemoryWriteCandidate(
                    content=f"用户名字叫{identity_name}",
                    kind="profile",
                    memory_type="preference",
                    confidence=1.0,
                    reason="matched_identity_statement_fast_path",
                )
            ],
            reply_mode="identity_statement",
            fast_path=True,
            fast_path_kind="identity_statement",
            reason="matched_identity_statement_fast_path",
        )
    sensitive_credential = _is_sensitive_credential_statement(text)
    long_input = _long_input_signal(text)
    if sensitive_credential and not long_input["should_capture"]:
        return TurnPlan(
            reply_mode="sensitive_credential_rejected",
            fast_path=True,
            fast_path_kind="sensitive_credential",
            reason="matched_sensitive_credential_input",
        )
    # 时间解析是确定性执行辅助，不承担开放语义判断。
    temporal = resolve_temporal_local(text, reference_time=reference_time, timezone=timezone)
    if long_input["should_capture"]:
        return TurnPlan(
            temporal_scope=temporal,
            reply_mode="continuous_capture",
            fast_path=True,
            fast_path_kind="continuous_capture",
            reason="matched_long_input_capture:" + str(long_input["reason"]),
        )

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
        recall_goal="none",
        conversation_action="",
        memory_write_candidates=[],
        temporal_scope=temporal,
        reply_mode="llm",
        event_recall_strategy="skipped",
        reason="default_pre_reply_decision_required",
    )


# 本地时间解析覆盖高频短语，避免每个“昨天/明天/今晚”都调用 LLM。
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

    if any(marker in text for marker in ("接下来", "后面", "未来", "这两天")):
        start_dt = reference_dt
        end_dt = reference_dt + timedelta(days=7)
        temporal_text = _first_match(text, ("接下来", "后面几天", "后面", "未来", "这两天"))
        granularity = "day"
        reason = "future temporal expression resolved to the next seven days"
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

    # 复杂时间只标记为可延迟处理，不在同步路径硬猜范围。
    if start_dt is None or end_dt is None:
        if _has_complex_temporal_hint(text):
            return TemporalResolution(
                has_temporal_expression=True,
                temporal_text=_first_match(text, _COMPLEX_TEMPORAL_HINTS),
                timezone=tz_name,
                confidence=0.4,
                backend="llm_deferred_or_fallback",
                reason="complex temporal expression deferred from synchronous path",
            )
        return TemporalResolution(timezone=tz_name, backend="local_none", reason="no local temporal expression")

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


_MEMORY_COMMAND_MARKERS = ("记一下", "记下来", "记录", "帮我记", "记住", "remember")
_GREETING_EXACT = ("你好", "您好", "早上好", "下午好", "晚上好", "hello", "hi", "hey")
_IDENTITY_QUERY_EXACT = ("我是谁", "我叫什么", "你是谁", "你叫什么", "你知道我是谁吗", "你知道我叫什么吗")
_SENSITIVE_CREDENTIAL_MARKERS = (
    "api key",
    "apikey",
    "api_key",
    "access token",
    "token",
    "secret",
    "password",
    "credential",
    "密码",
    "口令",
    "银行卡密码",
    "pin",
)
_CREDENTIAL_VALUE_MARKERS = ("是", "=", "：", ":", "为", "记住", "记录", "告诉你")
_EVENT_TIME_MARKERS = (
    "今天",
    "明天",
    "后天",
    "昨天",
    "周一",
    "周二",
    "周三",
    "周四",
    "周五",
    "周六",
    "周日",
    "星期",
    "上午",
    "下午",
    "晚上",
    "中午",
)
_COMPLEX_TEMPORAL_HINTS = ("上周", "下周", "上个月", "下个月", "去年", "明年", "春节", "假期", "月底", "年初")
_FUTURE_TEMPORAL_HINTS = ("下周", "下个", "明天", "后天", "接下来", "后面", "未来")
_LONG_INPUT_PROJECT_MARKERS = (
    "项目",
    "demo",
    "阶段",
    "目标",
    "进展",
    "状态",
    "负责",
    "风险",
    "卡点",
    "决定",
    "结论",
    "接下来",
    "下一步",
)
_LONG_INPUT_DAILY_MARKERS = (
    "上午",
    "中午",
    "下午",
    "晚上",
    "早上",
    "今天",
    "昨天",
    "后来",
    "然后",
    "接着",
    "路上",
    "回家",
    "去了",
    "见了",
    "聊到",
    "说到",
)
_LONG_INPUT_ACTION_MARKERS = (
    "负责",
    "决定",
    "提醒",
    "约",
    "交",
    "取",
    "复盘",
    "讨论",
    "推进",
    "适配",
    "补",
    "做",
    "去",
    "聊",
    "说",
    "提到",
)
_LONG_QUESTION_REQUEST_MARKERS = (
    "解释",
    "介绍",
    "说明",
    "讲讲",
    "写一篇",
    "帮我写",
    "分析一下",
    "总结一下",
    "举例",
    "代码",
    "原理",
    "机制",
    "为什么",
)


def _canonical_text(message: str) -> str:
    return _compact(message).strip("。！？!? ")


def _is_greeting(text: str) -> bool:
    return text in _GREETING_EXACT


def _is_identity_query(text: str) -> bool:
    return text in _IDENTITY_QUERY_EXACT or any(
        text.startswith(term) and _is_question_like(text)
        for term in _IDENTITY_QUERY_EXACT
    )


def _identity_statement_name(text: str) -> str:
    compact = str(text or "").strip()
    if not compact.startswith("我叫"):
        return ""
    return compact.removeprefix("我叫").strip("。！？!? ，,；;：:")


def _is_sensitive_credential_statement(text: str) -> bool:
    lowered = text.lower()
    if any(marker in text for marker in ("纠正一下", "说错了", "我刚才说错了")):
        return False
    if not any(marker in lowered for marker in _SENSITIVE_CREDENTIAL_MARKERS):
        return False
    if _is_question_like(text) and not any(marker in lowered for marker in _MEMORY_COMMAND_MARKERS):
        return False
    return any(marker in text for marker in _CREDENTIAL_VALUE_MARKERS) or any(
        marker in lowered for marker in _MEMORY_COMMAND_MARKERS
    )


def _long_input_signal(text: str) -> dict[str, Any]:
    explicit_transcript = _explicit_short_transcript_signal(text)
    if explicit_transcript:
        return {"should_capture": True, "reason": explicit_transcript}
    if len(text) < 90:
        return {"should_capture": False, "reason": "too_short"}
    sentence_count = len([part for part in re.split(r"[。！？!?；;，,\n]+", text) if part.strip()])
    line_count = len([part for part in text.splitlines() if part.strip()])
    if sentence_count < 3 and line_count < 3:
        return {"should_capture": False, "reason": "not_enough_segments"}
    if _looks_like_long_question_or_generation_request(text):
        return {"should_capture": False, "reason": "long_question_or_generation_request"}

    project_hits = _marker_hit_count(text, _LONG_INPUT_PROJECT_MARKERS)
    daily_hits = _marker_hit_count(text, _LONG_INPUT_DAILY_MARKERS)
    action_hits = _marker_hit_count(text, _LONG_INPUT_ACTION_MARKERS)
    temporal_hits = _marker_hit_count(text, _EVENT_TIME_MARKERS + _COMPLEX_TEMPORAL_HINTS)
    person_hits = len(set(re.findall(r"\b[A-Z][a-z]{1,20}\b", text)))
    first_person_hits = text.count("我") + text.count("我们")
    topic_score = sum(1 for count in (project_hits, daily_hits, action_hits, temporal_hits, person_hits) if count > 0)

    should_capture = (
        topic_score >= 3
        and (action_hits >= 2 or project_hits >= 2 or daily_hits >= 3 or person_hits >= 2 or first_person_hits >= 2)
        and (first_person_hits > 0 or project_hits > 0)
    )
    reason = (
        f"chars={len(text)},sentences={sentence_count},lines={line_count},"
        f"topics={topic_score},project={project_hits},daily={daily_hits},"
        f"actions={action_hits},time={temporal_hits},people={person_hits}"
    )
    return {"should_capture": should_capture, "reason": reason}


def _explicit_short_transcript_signal(text: str) -> str:
    lowered = text.lower()
    intro_hits = any(marker in text for marker in ("口述", "转写", "录音", "语音", "先记一段", "补一段"))
    capture_intent_hits = any(marker in text for marker in ("先记一段", "记一段", "口述一段", "转写文字", "转写内容", "补一段文字"))
    action_hits = _marker_hit_count(text, _LONG_INPUT_ACTION_MARKERS)
    temporal_hits = _marker_hit_count(text, _EVENT_TIME_MARKERS + _COMPLEX_TEMPORAL_HINTS)
    correction_hits = any(marker in text for marker in ("哦不对", "不对", "应该是", "改成", "更正"))
    do_not_remember_hits = any(marker in text for marker in ("不用记", "不要记", "别沉淀", "先别沉淀", "不用保存"))
    person_hits = len(set(re.findall(r"\b[A-Z][a-z]{1,20}\b", text)))
    if (
        intro_hits
        and capture_intent_hits
        and (action_hits > 0 or temporal_hits > 0 or correction_hits or do_not_remember_hits or person_hits > 0)
    ):
        return (
            "explicit_short_transcript:"
            f"actions={action_hits},time={temporal_hits},people={person_hits},"
            f"correction={correction_hits},do_not_remember={do_not_remember_hits}"
        )
    return ""


def _looks_like_long_question_or_generation_request(text: str) -> bool:
    if not _is_question_like(text) and not any(marker in text for marker in ("请", "帮我", "详细", "写一篇")):
        return False
    request_hits = _marker_hit_count(text, _LONG_QUESTION_REQUEST_MARKERS)
    personal_trace_hits = _marker_hit_count(text, _LONG_INPUT_DAILY_MARKERS + _LONG_INPUT_PROJECT_MARKERS)
    return request_hits >= 2 and personal_trace_hits <= 1


def _marker_hit_count(text: str, markers: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for marker in markers if marker.lower() in lowered)


def _is_question_like(text: str) -> bool:
    return is_question(text) or any(marker in text for marker in ("什么", "哪些", "有没有", "是不是", "是否", "啥", "干嘛"))


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
        if delta == 0 and _has_future_temporal_hint(text):
            delta = 7
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


_MEMORY_COMMAND_SHELL_RE = re.compile(
    r"^(?:"
    r"(?:那个|这个|就是|然后|还有|另外|对了|诶|欸|哎|唉|额|呃|嗯|啊|好|好的)[，,、\s]*)*"
    r"(?:"
    r"(?:麻烦|拜托|请)?你(?:先|顺手|帮忙|再)?"
    r"|(?:麻烦|拜托|请)(?:你)?"
    r"|(?:能不能|可不可以|可以|能)(?:帮我|替我|给我|帮忙)?"
    r"|(?:先|顺手|再)?(?:帮我|替我|给我|帮忙)"
    r")?"
    r"(?:先|顺手|再)?"
    r"(?:帮我|替我|给我|帮忙)?"
    r"(?:记(?:一下|下|着|住|下来|个事)|记录(?:一下|下)?|存(?:一下|下)?|remember)"
    r"(?:一下|下|哈|吧|个事)?"
    r"[，,：:\s]*",
    re.IGNORECASE,
)


def _strip_memory_command_shell(text: str) -> tuple[str, bool]:
    stripped = _compact(text)
    match = _MEMORY_COMMAND_SHELL_RE.match(stripped)
    if not match:
        return stripped, False
    cleaned = stripped[match.end() :].strip(" ：:，,")
    return (cleaned or stripped), bool(cleaned)


def _strip_memory_prefix(text: str) -> str:
    stripped, shell_removed = _strip_memory_command_shell(text)
    if shell_removed:
        return _strip_memory_request_suffix(stripped)
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
    for marker in ("记一下", "帮我记一下", "帮我记", "记录一下", "记录", "记住", "remember"):
        if stripped.startswith(marker):
            stripped = stripped[len(marker) :].strip(" ：:，,")
            break
    stripped = re.sub(r"^(?:确认一下|确认下)[，,：:\\s]*", "", stripped).strip(" ：:，,")
    stripped = re.sub(r"[，,。\\s]*(?:这个)?记下来[。.!！]?$", "", stripped).strip(" ：:，,")
    return _strip_memory_request_suffix(stripped)


def _strip_memory_request_suffix(text: str) -> str:
    stripped = _compact(text).strip(" ：:，,")
    suffixes = (
        "可以吗",
        "好吗",
        "行吗",
        "可以不",
        "好不好",
        "吗",
        "么",
        "吗?",
        "吗？",
    )
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


def _remove_first(text: str, phrase: str) -> str:
    if not phrase:
        return text
    return text.replace(phrase, "", 1).strip(" ，,。.!！?？：:")


def _first_match(text: str, terms: tuple[str, ...]) -> str:
    for term in terms:
        if term in text:
            return term
    return ""


def _has_complex_temporal_hint(text: str) -> bool:
    return any(term in text for term in _COMPLEX_TEMPORAL_HINTS)


def _has_future_temporal_hint(text: str) -> bool:
    return any(term in text for term in _FUTURE_TEMPORAL_HINTS)


def _compact(message: str) -> str:
    return " ".join(str(message or "").strip().split())
