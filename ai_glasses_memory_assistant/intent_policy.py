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

# 长期记忆写入的最后门控：敏感信息、低置信度和显式候选边界都在这里拦截。
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
    if kind == "assistant_preference":
        return MemoryWriteGateResult(True, "allowed")
    return MemoryWriteGateResult(True, "allowed")


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


def sensitive_input_reason(content: str) -> str:
    """Detect credential-shaped input without routing ordinary discussion.

    The write gate remains conservative, but chat preflight only short-circuits
    when a secret value, code, or identifier shape is present.  A sentence
    merely discussing password policy therefore still reaches PreReplyDecision.
    """
    text = str(content or "")
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in (
        r"sk-proj-[A-Za-z0-9\-_]{20,}",
        r"(?:sk|pk|rk|ak)-[A-Za-z0-9][A-Za-z0-9\-_]{19,}",
        r"Bearer\s+[A-Za-z0-9._\-+/=]{20,}",
        r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        r"gh[pus]_[A-Za-z0-9]{36,}",
        r"github_pat_[A-Za-z0-9_]{22,}",
        r"(?:验证码|校验码|短信码|动态码)\s*(?:是|为|:|：|=)?\s*\d{4,8}",
        r"(?:身份证|身份证号|护照|护照号|银行卡|卡号)\s*(?:是|为|:|：|=)?\s*[A-Za-z0-9\-]{6,}",
        r"(?:密码|口令|password|passcode|api[_ ]?key|access token|secret)\s*(?:是|为|:|：|=)\s*\S+",
    )):
        return "candidate_contains_sensitive_input"
    digits = "".join(ch for ch in text if ch.isdigit())
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
