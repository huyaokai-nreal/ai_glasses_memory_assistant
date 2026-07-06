from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_EXTRACTABLE_ROLES = {"memory_candidate", "correction"}
_DO_NOT_REMEMBER_MARKERS = (
    "别记这个",
    "不用记",
    "不要记这个",
    "不要保存",
    "真的不要保存",
    "这个别保存",
    "别保存这个",
)
_CONTRAST_SPLIT_RE = re.compile(r"(?:，|,)?(?:但|但是|不过|可是|然后|另外|还有)")
_CORRECTION_FOCUS_RE = re.compile(r"(?:改成|应该是|更正为|换成|实际是|准确说)[:：,， ]*(.+)$")
_SEGMENT_SPLIT_RE = re.compile(r"[，,。.!！?？；;]|(?:但|但是|不过|可是|然后|另外|还有)")
_BACKGROUND_NOISE_TERMS = (
    "背景",
    "有点吵",
    "噪声",
    "口头禅",
    "停顿",
    "重复词",
    "音量",
    "先不用沉淀",
    "先保留下来",
    "讨论过程",
    "上下文",
)


@dataclass(frozen=True)
class SegmentSemanticDecision:
    segment_index: int
    raw_span: str
    semantic_role: str = "memory_candidate"
    noise_level: str = "low"
    contains_filler: bool = False
    do_not_remember_scope: str = ""
    should_extract: bool = True
    candidate_span: str = ""
    candidate_hint: str = ""
    confidence: float | None = None
    reason: str = ""
    backend: str = "rule_fallback"
    raw: str = ""
    error: str = ""

    def extraction_text(self) -> str:
        return (self.candidate_span or self.raw_span).strip()

    def debug_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "segment_index": self.segment_index,
            "raw_span": self.raw_span,
            "semantic_role": self.semantic_role,
            "noise_level": self.noise_level,
            "contains_filler": self.contains_filler,
            "do_not_remember_scope": self.do_not_remember_scope,
            "should_extract": self.should_extract,
            "candidate_span": self.candidate_span,
            "candidate_hint": self.candidate_hint,
            "confidence": self.confidence,
            "reason": self.reason,
            "backend": self.backend,
        }
        if self.raw:
            payload["raw"] = self.raw
        if self.error:
            payload["error"] = self.error
        return payload


def classify_segment_semantics(
    agent: Any,
    *,
    segment: str,
    segment_index: int,
    marker_kinds: list[str] | None = None,
) -> SegmentSemanticDecision:
    text = str(segment or "").strip()
    marker_list = [str(marker).strip() for marker in marker_kinds or [] if str(marker).strip()]
    prompt = f"""Classify this transcript segment for an AI glasses memory assistant.

Return JSON only, with this exact shape:
{{
  "semantic_role": "memory_candidate|temporary_context|chitchat|noise_only|correction|do_not_remember|sensitive_reject",
  "noise_level": "none|low|medium|high",
  "contains_filler": false,
  "do_not_remember_scope": "",
  "should_extract": false,
  "candidate_span": "",
  "candidate_hint": "profile|task|decision|project_state|event|preference|correction|sensitive|",
  "confidence": 0.0,
  "reason": ""
}}

Rules:
- Treat marker_kinds as weak signals, not final judgments.
- Do not skip an entire segment only because it contains filler such as 嗯, 那个, 哈哈.
- If a segment contains both do-not-remember content and a separate useful task/fact, set should_extract=true and put only the useful span in candidate_span.
- If "不用记/不要保存/别记" only applies to a local span, write that span in do_not_remember_scope.
- If the whole segment is only filler, background noise, casual chatter, or explicitly do-not-remember content, set should_extract=false.
- Never invent facts outside raw_span.

segment_index: {segment_index}
marker_kinds: {json.dumps(marker_list, ensure_ascii=False)}
raw_span:
{text}
"""
    try:
        result = agent.run_conversation(
            prompt,
            system_message=(
                "You are an internal segment semantic cleaner. "
                "Output strict JSON only. Do not call tools."
            ),
            conversation_history=[],
            persist_user_message=None,
        )
        raw = str(result.get("final_response") or "").strip()
        payload = _parse_json_object(raw)
        return _decision_from_payload(payload, segment_index=segment_index, raw_span=text, raw=raw, backend="llm")
    except Exception as exc:
        fallback = fallback_segment_semantic_decision(
            segment=text,
            segment_index=segment_index,
            marker_kinds=marker_list,
            backend="rule_fallback",
        )
        return SegmentSemanticDecision(
            segment_index=fallback.segment_index,
            raw_span=fallback.raw_span,
            semantic_role=fallback.semantic_role,
            noise_level=fallback.noise_level,
            contains_filler=fallback.contains_filler,
            do_not_remember_scope=fallback.do_not_remember_scope,
            should_extract=fallback.should_extract,
            candidate_span=fallback.candidate_span,
            candidate_hint=fallback.candidate_hint,
            confidence=fallback.confidence,
            reason=fallback.reason,
            backend="rule_fallback",
            error=str(exc),
        )


def fallback_segment_semantic_decision(
    *,
    segment: str,
    segment_index: int,
    marker_kinds: list[str] | None = None,
    backend: str = "rule_fallback",
) -> SegmentSemanticDecision:
    text = str(segment or "").strip()
    markers = set(marker_kinds or [])
    filler_markers = {"filler:um", "filler:that", "filler:laugh"}
    noise_markers = {"noise:background"}
    contains_filler = bool(markers & filler_markers)
    if "correction:do_not_remember" in markers:
        do_not_remember_scope, extractable_span = _split_local_do_not_remember_scope(text)
        if extractable_span:
            return SegmentSemanticDecision(
                segment_index=segment_index,
                raw_span=text,
                semantic_role="memory_candidate",
                noise_level="low",
                contains_filler=contains_filler,
                do_not_remember_scope=do_not_remember_scope,
                should_extract=True,
                candidate_span=extractable_span,
                candidate_hint="task",
                confidence=0.72,
                reason="matched_local_do_not_remember_with_remaining_span",
                backend=backend,
            )
        return SegmentSemanticDecision(
            segment_index=segment_index,
            raw_span=text,
            semantic_role="do_not_remember",
            noise_level="low",
            contains_filler=contains_filler,
            do_not_remember_scope=do_not_remember_scope or text,
            should_extract=False,
            confidence=0.7,
            reason="matched_do_not_remember_marker",
            backend=backend,
        )
    if {"correction:negate_previous", "correction:replace_with"} & markers:
        correction_span = _correction_focus_span(text)
        return SegmentSemanticDecision(
            segment_index=segment_index,
            raw_span=text,
            semantic_role="correction",
            noise_level="low",
            contains_filler=contains_filler,
            should_extract=bool(correction_span),
            candidate_span=correction_span,
            candidate_hint="correction",
            confidence=0.68,
            reason="matched_correction_marker",
            backend=backend,
        )
    scoped_span = _extract_high_value_span(text)
    if scoped_span and scoped_span != text:
        return SegmentSemanticDecision(
            segment_index=segment_index,
            raw_span=text,
            semantic_role="memory_candidate",
            noise_level="low" if markers else "none",
            contains_filler=contains_filler,
            should_extract=True,
            candidate_span=scoped_span,
            candidate_hint=_candidate_hint_for_span(scoped_span),
            confidence=0.66,
            reason="matched_high_value_span_with_background_filtered",
            backend=backend,
        )
    if markers and markers <= (filler_markers | noise_markers):
        role = "noise_only" if markers & noise_markers else "chitchat"
        return SegmentSemanticDecision(
            segment_index=segment_index,
            raw_span=text,
            semantic_role=role,
            noise_level="high" if markers & noise_markers else "low",
            contains_filler=contains_filler,
            should_extract=False,
            confidence=0.65,
            reason="matched_only_filler_or_noise_markers",
            backend=backend,
        )
    return SegmentSemanticDecision(
        segment_index=segment_index,
        raw_span=text,
        semantic_role="memory_candidate",
        noise_level="low" if markers else "none",
        contains_filler=contains_filler,
        should_extract=bool(text),
        candidate_span=text,
        confidence=0.55,
        reason="fallback_extract_nonempty_segment",
        backend=backend,
    )


def _split_local_do_not_remember_scope(text: str) -> tuple[str, str]:
    scope = str(text or "").strip()
    for marker in _DO_NOT_REMEMBER_MARKERS:
        if marker not in scope:
            continue
        before, after = scope.split(marker, 1)
        local_scope = f"{before}{marker}".strip(" ，,。.!！?？；;")
        remainder = _CONTRAST_SPLIT_RE.sub(" ", after, count=1).strip(" ，,。.!！?？；;")
        return local_scope, remainder
    return scope, ""


def _correction_focus_span(text: str) -> str:
    content = str(text or "").strip(" ，,。.!！?？；;")
    match = _CORRECTION_FOCUS_RE.search(content)
    if match:
        focused = match.group(1).strip(" ，,。.!！?？；;")
        if focused:
            return focused
    parts = [
        part.strip(" ，,。.!！?？；;")
        for part in re.split(r"[，,。.!！?？；;]", content)
        if part.strip(" ，,。.!！?？；;")
    ]
    return parts[-1] if parts else content


def _extract_high_value_span(text: str) -> str:
    content = str(text or "").strip(" ，,。.!！?？；;")
    if not content:
        return ""
    parts = [
        part.strip(" ，,。.!！?？；;")
        for part in _SEGMENT_SPLIT_RE.split(content)
        if part.strip(" ，,。.!！?？；;")
    ]
    high_value_parts = [
        part for part in parts
        if _looks_like_high_value_fact(part) and not _looks_like_background_only(part)
    ]
    if not high_value_parts:
        return ""
    return "；".join(dict.fromkeys(high_value_parts))


def _candidate_hint_for_span(text: str) -> str:
    content = str(text or "")
    if any(marker in content for marker in ("负责", "截止", "交第一版", "待办", "计划", "验收", "验证")):
        return "task"
    if any(marker in content for marker in ("决定", "结论", "确定", "先做")):
        return "decision"
    if any(marker in content for marker in ("风险", "卡点", "不稳定", "失败", "状态", "进展")):
        return "project_state"
    return ""


def _looks_like_high_value_fact(text: str) -> bool:
    content = str(text or "")
    return any(
        marker in content
        for marker in (
            "负责",
            "截止",
            "交第一版",
            "待办",
            "下一步",
            "准备",
            "计划",
            "启动",
            "上线",
            "发布",
            "交付",
            "跑起来",
            "决定",
            "结论",
            "确定",
            "先做",
            "风险",
            "卡点",
            "失败",
            "不稳定",
            "进展",
            "状态",
            "周一",
            "周二",
            "周三",
            "周四",
            "周五",
            "周六",
            "周日",
            "今天",
            "明天",
            "后天",
            "上午",
            "下午",
            "晚上",
        )
    )


def _looks_like_background_only(text: str) -> bool:
    content = str(text or "")
    return any(term in content for term in _BACKGROUND_NOISE_TERMS)


def _decision_from_payload(
    payload: dict[str, Any],
    *,
    segment_index: int,
    raw_span: str,
    raw: str,
    backend: str,
) -> SegmentSemanticDecision:
    role = str(payload.get("semantic_role") or "").strip().lower()
    if role not in {
        "memory_candidate",
        "temporary_context",
        "chitchat",
        "noise_only",
        "correction",
        "do_not_remember",
        "sensitive_reject",
    }:
        role = "memory_candidate"
    candidate_span = str(payload.get("candidate_span") or "").strip()
    should_extract = bool(payload.get("should_extract")) and role in _EXTRACTABLE_ROLES
    if should_extract and not candidate_span:
        candidate_span = raw_span
    return SegmentSemanticDecision(
        segment_index=segment_index,
        raw_span=raw_span,
        semantic_role=role,
        noise_level=str(payload.get("noise_level") or "low").strip().lower(),
        contains_filler=bool(payload.get("contains_filler")),
        do_not_remember_scope=str(payload.get("do_not_remember_scope") or "").strip(),
        should_extract=should_extract,
        candidate_span=candidate_span,
        candidate_hint=str(payload.get("candidate_hint") or "").strip().lower(),
        confidence=_optional_float(payload.get("confidence")),
        reason=str(payload.get("reason") or "").strip(),
        backend=backend,
        raw=raw,
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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
