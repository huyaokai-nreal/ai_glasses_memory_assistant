from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .memory_confidence import (
    MEMORY_WRITE_MIN_CONFIDENCE,
    confidence_policy_payload,
)


@dataclass(frozen=True)
class MemoryWriteGateResult:
    allowed: bool
    reason: str
    requires_confirmation: bool = False
    privacy_level: str = "normal"
    confidence_policy: dict[str, Any] = field(default_factory=dict)
    safety_policy: dict[str, Any] = field(default_factory=dict)
    question_policy: dict[str, Any] = field(default_factory=dict)


MIN_MEMORY_CONFIDENCE = MEMORY_WRITE_MIN_CONFIDENCE

# 写入门控触发词只服务长期记忆安全边界；turn routing 归 turn_planner.py。
POLICY_TERMS: dict[str, tuple[str, ...]] = {
    "question_markers": (
        "?",
        "？",
        "什么",
        "啥",
        "吗",
        "如何",
        "怎么",
        "干嘛",
        "谁",
        "哪里",
        "哪儿",
        "有没有",
        "是不是",
        "是否",
        "多少",
        "几",
    ),
    "meta_profile": (
        "用户询问",
        "用户问",
        "用户想知道",
        "asked",
        "asks",
        "question",
    ),
    "transient_context": (
        "刚刚",
        "刚才",
        "路过",
        "经过",
        "当前位置",
        "现在在",
        "临时",
        "临时想",
        "临时方案",
        "先想一下",
        "先想想",
        "还没确认",
        "待确认",
        "等会再确认",
        "之后再确认",
        "可能不是最终",
        "不是最终方案",
        "还不确定",
        "暂时",
        "先别写成偏好",
        "不要记成偏好",
        "只是临时想法",
        "只是试试",
        "先试试",
        "临时偏好",
        "tentative",
        "draft",
        "not final",
    ),
    "memory_request": (
        "记住",
        "记一下",
        "帮我记",
        "帮忙记",
        "保存",
        "记录",
        "备忘",
        "提醒我",
        "remember",
        "save this",
    ),
}


# 长期记忆写入的最后门控：敏感信息、低置信度和问题文本都在这里拦截。
def should_write_memory_candidate(candidate: Any, message: str) -> MemoryWriteGateResult:
    content = str(getattr(candidate, "content", "") or "").strip()
    kind = str(getattr(candidate, "kind", "") or "").strip().lower()
    source_type = str(getattr(candidate, "source_type", "") or "chat").strip().lower()
    speaker_hint = str(getattr(candidate, "speaker_hint", "") or "").strip().lower()
    subject_type = str(getattr(candidate, "subject_type", "") or "self").strip().lower()
    subject_name = str(getattr(candidate, "subject_name", "") or "").strip()
    do_not_remember_scope = str(getattr(candidate, "do_not_remember_scope", "") or "").strip()
    confidence = getattr(candidate, "confidence", None)
    if not content:
        return MemoryWriteGateResult(False, "empty_candidate")
    if not bool(getattr(candidate, "memory_eligible", True)):
        return MemoryWriteGateResult(False, "audio_memory_ineligible")
    overlap_state = str(getattr(candidate, "overlap_state", "") or "").strip().lower()
    if overlap_state in {"suspected", "unknown"}:
        return MemoryWriteGateResult(False, f"audio_overlap_{overlap_state}")
    speaker_state = str(getattr(candidate, "speaker_state", "") or "").strip().lower()
    if speaker_state in {"other", "unknown", "environment"}:
        return MemoryWriteGateResult(False, f"audio_speaker_{speaker_state}")
    if do_not_remember_scope:
        return MemoryWriteGateResult(False, "explicit_do_not_remember")
    if source_type == "ambient_audio":
        return MemoryWriteGateResult(False, "ambient_only")
    if source_type == "multi_speaker_transcript":
        if subject_type not in {"self", "named", "provisional"}:
            return MemoryWriteGateResult(False, "invalid_memory_subject")
        if subject_type in {"named", "provisional"} and not subject_name:
            return MemoryWriteGateResult(False, "missing_memory_subject_name")
    if source_type == "wake_query":
        if speaker_hint == "other":
            return MemoryWriteGateResult(False, "third_party_speech_blocked")
        if speaker_hint == "unknown":
            return MemoryWriteGateResult(False, "unknown_speaker_blocked")
    sensitive_reason = _sensitive_reason(content)
    if sensitive_reason and not _looks_like_nonsecret_document_schedule(content, message, kind):
        final_reason = "do_not_memorize_sensitive_audio" if source_type in {"ambient_audio", "wake_query"} else sensitive_reason
        return MemoryWriteGateResult(
            False,
            final_reason,
            requires_confirmation=True,
            privacy_level="requires_confirmation",
            safety_policy=_safety_policy_payload(final_reason),
        )
    privacy_level = str(getattr(candidate, "privacy_level", "") or "").strip().lower()
    if privacy_level in {"sensitive", "requires_confirmation"}:
        return MemoryWriteGateResult(
            False,
            "candidate_privacy_requires_confirmation",
            requires_confirmation=True,
            privacy_level="requires_confirmation",
            safety_policy=_safety_policy_payload("candidate_privacy_requires_confirmation"),
        )
    if confidence is not None and confidence < MIN_MEMORY_CONFIDENCE:
        return MemoryWriteGateResult(
            False,
            "candidate_confidence_below_threshold",
            confidence_policy=confidence_policy_payload(
                purpose="memory_write_candidate",
                confidence=confidence,
                min_confidence=MIN_MEMORY_CONFIDENCE,
                treatment="reject",
                backend=str(getattr(candidate, "source", "") or ""),
                reason="candidate_confidence_below_threshold",
            ),
        )
    if kind == "profile" and _contains_any(content.lower(), POLICY_TERMS["meta_profile"]):
        return MemoryWriteGateResult(False, "profile_candidate_is_meta_description")
    transient_marker = _transient_context_marker(content)
    if transient_marker:
        return MemoryWriteGateResult(
            False,
            "transient_ambient_chitchat" if source_type == "wake_query" else "candidate_is_transient_context",
        )
    if _looks_like_direct_source_question(content):
        return MemoryWriteGateResult(
            False,
            "candidate_is_question_text",
            question_policy=_question_policy_payload(
                "candidate_is_question_text",
                treatment="reject_candidate_question_text",
                source="candidate",
            ),
        )
    if kind == "assistant_preference":
        return MemoryWriteGateResult(True, "allowed")
    if _looks_like_direct_source_question(message) and not _looks_like_memory_request(message):
        return MemoryWriteGateResult(
            False,
            "source_question_without_memory_request",
            question_policy=_question_policy_payload(
                "source_question_without_memory_request",
                treatment="reject_source_question_without_memory_request",
                source="source_message",
            ),
        )
    return MemoryWriteGateResult(True, "allowed")


def transient_context_marker(text: str) -> str:
    return _transient_context_marker(text)


def _transient_context_marker(text: str) -> str:
    lowered = str(text or "").lower()
    for marker in POLICY_TERMS["transient_context"]:
        if marker.lower() in lowered:
            return marker
    return ""


# 敏感信息检测是保守拒绝策略，命中后需要确认而不是默认落库。
def _sensitive_reason(content: str) -> str:
    lowered = content.lower()
    secret_terms = (
        "密码",
        "口令",
        "api key",
        "apikey",
        "token",
        "access token",
        "secret",
        "银行卡",
        "身份证",
        "护照",
        "社保",
        "病历",
        "诊断结果",
        "医疗记录",
        "工资明细",
        "收入明细",
        "负债明细",
        "验证码",
        "校验码",
        "短信码",
        "bearer",
        "jwt",
    )
    if any(term in lowered for term in secret_terms):
        return "candidate_contains_sensitive_term"
    secret_patterns = (
        r"sk-proj-[A-Za-z0-9\-_]{20,}",
        r"(?:sk|pk|rk|ak)-[A-Za-z0-9][A-Za-z0-9\-_]{19,}",
        r"Bearer\s+[A-Za-z0-9._\-+/=]{20,}",
        r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        r"gh[pus]_[A-Za-z0-9]{36,}",
        r"github_pat_[A-Za-z0-9_]{22,}",
    )
    if any(re.search(pattern, content, flags=re.IGNORECASE) for pattern in secret_patterns):
        return "candidate_contains_sensitive_secret"
    if re.search(r"(?:验证码|校验码|短信码|动态码)\s*(?:是|:|：)?\s*\d{4,8}", content):
        return "candidate_contains_sensitive_code"
    digits = "".join(ch for ch in content if ch.isdigit())
    if len(digits) >= 16:
        return "candidate_contains_long_sensitive_number"
    return ""


def _safety_policy_payload(reason: str) -> dict[str, Any]:
    return {
        "role": "hard_safety",
        "reason": reason,
        "treatment": "reject_or_require_confirmation",
        "affects_final_decision": True,
        "overrides_llm": True,
    }


def _question_policy_payload(reason: str, *, treatment: str, source: str) -> dict[str, Any]:
    return {
        "role": "weak_signal",
        "reason": reason,
        "source": source,
        "treatment": treatment,
        "affects_final_decision": True,
        "overrides_llm": False,
    }


def _looks_like_nonsecret_document_schedule(content: str, message: str, kind: str) -> bool:
    if kind != "event":
        return False
    text = f"{message}\n{content}".lower()
    document_terms = ("护照", "身份证", "证件")
    action_terms = ("取", "拿", "办", "补办", "递交", "提交", "带", "领取", "更换", "办理")
    schedule_terms = (
        "记一下",
        "记录",
        "今天",
        "明天",
        "后天",
        "周一",
        "周二",
        "周三",
        "周四",
        "周五",
        "周六",
        "周日",
        "上午",
        "下午",
        "晚上",
        "中午",
        "点",
    )
    secret_markers = ("号码", "编号", "证号", "护照号", "身份证号", "密码", "口令", "验证码", "校验码")
    if not any(term in text for term in document_terms):
        return False
    if not any(term in text for term in action_terms):
        return False
    if not any(term in text for term in schedule_terms):
        return False
    if any(marker in text for marker in secret_markers):
        return False
    for term in document_terms:
        if re.search(rf"{re.escape(term)}\s*(?:是|为|:|：)?\s*[A-Za-z]?\d{{5,}}", text, flags=re.IGNORECASE):
            return False
    return True


def is_question(message: str) -> bool:
    return _looks_like_question(message)


# fast path 回复必须短且确定，避免引入主模型成本。
def fast_reply_for_greeting() -> str:
    return "你好，我在。"


def profile_statement_reply(message: str) -> str:
    text = _canonical_text(message)
    if text.startswith("我叫"):
        name = text.removeprefix("我叫").strip()
        if name:
            return f"好的，{name}，我记住了。"
    return "好的，我记住了。"


def event_record_reply(message: str) -> str:
    return f"我先记下：{_compact(message)}。"


def _compact(message: str) -> str:
    return " ".join(message.strip().split())


def _canonical_text(message: str) -> str:
    return _compact(message).strip("。！？!? ")


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _looks_like_question(message: str) -> bool:
    lowered = message.lower()
    return _contains_any(lowered, POLICY_TERMS["question_markers"])


def _looks_like_direct_source_question(message: str) -> bool:
    text = _compact(message).strip()
    if not text:
        return False
    if text.rstrip().endswith(("?", "？")):
        return True
    final_clause = re.split(r"[。！？!?；;\n]+", text)[-1].strip("。！？!?，,；;：: ")
    if not final_clause:
        return False
    if re.match(
        r"^(?:who|what|when|where|why|how|which|do|does|did|is|are|was|were|can|could|would|should|will|have|has|may|might)\b",
        final_clause,
        flags=re.IGNORECASE,
    ):
        return True
    question_suffixes = (
        "什么",
        "啥",
        "干嘛",
        "吗",
        "呢",
        "如何",
        "怎么",
        "为什么",
        "为何",
        "哪里",
        "哪儿",
        "谁",
        "多少",
        "几",
        "什么意思",
        "啥意思",
        "怎么回事",
        "是什么",
        "是啥",
    )
    if any(final_clause.endswith(marker) for marker in question_suffixes):
        return True
    if re.search(r"(?:是什么|是啥|什么是|啥是)\S*$", final_clause):
        return True
    return bool(re.search(r"(?:^|[，,：:\s])(?:是不是|有没有|是否)", final_clause))


def _looks_like_memory_request(message: str) -> bool:
    lowered = message.lower()
    return _contains_any(lowered, POLICY_TERMS["memory_request"])
