from __future__ import annotations

import re
from typing import Any


def is_explanation_query(message: str) -> bool:
    text = str(message or "").strip()
    if not text:
        return False
    markers = (
        "为什么这么说",
        "你为什么这么说",
        "依据是什么",
        "你的依据是什么",
        "为什么这么回答",
        "为什么这么答",
        "为什么没记住",
        "为什么没有记住",
        "为什么没保存",
        "为什么没有保存",
        "为什么没写进去",
        "为什么没有写进去",
    )
    return any(marker in text for marker in markers)


def explanation_reply(
    *,
    message: str,
    source_summary: dict[str, Any] | None,
    memory_processing: dict[str, Any] | None,
    recall_arbitration: dict[str, Any] | None,
    recent_context_capsule: dict[str, Any] | None,
    recalled_memories: list[dict[str, Any]] | None = None,
    saved_memories: list[dict[str, Any]] | None = None,
    evidence_quotes: list[str] | None = None,
) -> str:
    text = str(message or "")
    source_summary = dict(source_summary or {})
    memory_processing = dict(memory_processing or {})
    recall_arbitration = dict(recall_arbitration or {})
    recent_context_capsule = dict(recent_context_capsule or {})
    recalled_memories = [dict(item) for item in recalled_memories or [] if isinstance(item, dict)]
    saved_memories = [dict(item) for item in saved_memories or [] if isinstance(item, dict)]
    evidence_quotes = [str(item).strip() for item in evidence_quotes or [] if str(item).strip()]

    if any(marker in text for marker in ("为什么没记住", "为什么没有记住", "为什么没保存", "为什么没有保存", "为什么没写进去", "为什么没有写进去")):
        status = str(memory_processing.get("status") or "not_needed")
        stage = str(memory_processing.get("stage") or "")
        stage_reason = str(memory_processing.get("stage_reason") or "")
        stage_explanation = str(memory_processing.get("stage_explanation") or "")
        saved_count = int(memory_processing.get("saved_count") or 0)
        skip_policy = dict(memory_processing.get("skip_policy") or {})
        safety_policy = dict(memory_processing.get("safety_policy") or {})
        local_scope = matching_local_do_not_remember_scope(
            text,
            list(memory_processing.get("local_do_not_remember_scopes") or []),
        )
        if local_scope:
            return (
                f"「{local_scope}」没有保存，是因为它在这轮被识别为局部“不要记/不用记”的范围；"
                "后面可保存的内容仍会继续走候选和写入门控。"
            )
        if status == "saved":
            return f"这轮其实已经保存了 {saved_count} 条记忆。"
        if status in {"pending", "running"}:
            return "这轮还在后台整理记忆，暂时还没有最终保存结果。"
        if skip_policy.get("role") == "ephemeral_context":
            return "这轮被当成临时上下文或还没确认的想法，所以不应写入长期记忆。"
        if status == "skipped":
            return stage_explanation or "这轮没有需要长期保存的内容，所以没有写入长期记忆。"
        if status == "rejected":
            return stage_explanation or f"这轮在 {stage or 'gate'} 阶段被拒绝了，原因是 {stage_reason or '门控拒绝'}。"
        if status == "failed":
            error_type = str(memory_processing.get("error_type") or "unknown")
            if stage_explanation:
                return f"{stage_explanation} 错误类型是 {error_type}。"
            return f"这轮在 {stage or 'write_failure'} 阶段失败了，错误类型是 {error_type}。"
        if safety_policy.get("role") == "hard_safety":
            return "这轮在安全门控阶段被拒绝了，所以没有进入长期记忆写入。"
        return "这轮没有进入需要保存长期记忆的路径。"

    lines: list[str] = []
    primary_source = str(source_summary.get("primary_source") or "none")
    primary_label = str(source_summary.get("primary_source_label") or "无可用来源")
    primary_explanation = str(source_summary.get("primary_source_explanation") or "")
    if primary_source == "none":
        lines.append("当前没有可用来源，所以我不能把这轮回答说成有明确依据。")
    else:
        lines.append(f"这次回答主要依据是{primary_label}。")
        if primary_explanation:
            lines.append(primary_explanation)
        memory_basis = explanation_memory_basis(
            primary_source=primary_source,
            recalled_memories=recalled_memories,
            saved_memories=saved_memories,
            evidence_quotes=evidence_quotes,
        )
        if memory_basis:
            lines.append(memory_basis)
    dropped_sources = list(recall_arbitration.get("decisions") or [])
    dropped_reasons = [
        str(item.get("reason") or "").strip()
        for item in dropped_sources
        if isinstance(item, dict) and str(item.get("reason") or "").strip()
    ]
    if dropped_reasons:
        lines.append(f"另外有一些来源被压掉了，主要是因为：{ '；'.join(list(dict.fromkeys(dropped_reasons))[:3]) }。")
    injection_reason = str(recent_context_capsule.get("injection_reason") or "")
    injected = recent_context_capsule.get("injected_to_main_llm")
    if injected is True:
        lines.append(f"最近上下文这轮有参与主回答，原因是 {injection_reason or '需要承接上文'}。")
    elif injection_reason:
        lines.append(f"最近上下文这轮没有参与主回答，原因是 {injection_reason}。")
    if not lines:
        lines.append("当前没有可用来源。")
    return " ".join(line for line in lines if line).strip()


def explanation_memory_basis(
    *,
    primary_source: str,
    recalled_memories: list[dict[str, Any]],
    saved_memories: list[dict[str, Any]],
    evidence_quotes: list[str] | None = None,
) -> str:
    if primary_source not in {"profile", "structured_memory"}:
        return ""
    memory = primary_explanation_memory(
        primary_source=primary_source,
        recalled_memories=recalled_memories,
        saved_memories=saved_memories,
    )
    if not memory:
        return ""
    content = str(memory.get("content") or "").strip()
    parts = [f"具体依据是这条记忆：“{content}”。"]
    quote_text = format_explanation_evidence_quotes(evidence_quotes or [])
    if quote_text:
        parts.append(quote_text)
    source_trace = memory.get("source_trace") if isinstance(memory.get("source_trace"), dict) else {}
    source_id = str(source_trace.get("source_id") or memory.get("source_id") or "").strip()
    ingestion_id = str(source_trace.get("ingestion_id") or memory.get("ingestion_id") or "").strip()
    evidence_ids = memory_payload_evidence_ids(memory)
    trace_parts = []
    if source_id:
        trace_parts.append(f"source_id={source_id}")
    if ingestion_id:
        trace_parts.append(f"ingestion_id={ingestion_id}")
    evidence_text = ",".join(str(item) for item in evidence_ids if str(item).strip())
    if evidence_text:
        trace_parts.append(f"evidence_ids={evidence_text}")
    if trace_parts:
        parts.append(f"trace: {'; '.join(trace_parts)}。")
    return " ".join(parts)


def primary_explanation_memory(
    *,
    primary_source: str,
    recalled_memories: list[dict[str, Any]],
    saved_memories: list[dict[str, Any]],
) -> dict[str, Any] | None:
    memories = [*recalled_memories, *saved_memories]
    for item in memories:
        if not str(item.get("content") or "").strip():
            continue
        if primary_source == "profile" and item.get("kind") not in {"profile", "assistant_preference"}:
            continue
        return item
    return next((item for item in memories if str(item.get("content") or "").strip()), None)


def memory_payload_evidence_ids(memory: dict[str, Any]) -> list[str]:
    source_trace = memory.get("source_trace") if isinstance(memory.get("source_trace"), dict) else {}
    evidence_ids = source_trace.get("evidence_ids")
    if not isinstance(evidence_ids, list):
        evidence_ids = memory.get("evidence_ids") if isinstance(memory.get("evidence_ids"), list) else []
    return list(dict.fromkeys(str(item).strip() for item in evidence_ids if str(item).strip()))


def format_explanation_evidence_quotes(evidence_quotes: list[str]) -> str:
    quotes = [str(item).strip() for item in evidence_quotes if str(item).strip()]
    if not quotes:
        return ""
    trimmed = [quote if len(quote) <= 160 else f"{quote[:157]}..." for quote in quotes[:2]]
    if len(trimmed) == 1:
        return f"它来自你当时这句原话：“{trimmed[0]}”。"
    joined = "；".join(f"“{quote}”" for quote in trimmed)
    return f"它来自你当时这些原话：{joined}。"


def matching_local_do_not_remember_scope(message: str, scopes: list[Any]) -> str:
    text = str(message or "").strip()
    normalized_text = re.sub(r"[\s，,。.!！?？；;：“”\"'‘’（）()、]+", "", text)
    for scope in scopes:
        scope_text = str(scope or "").strip()
        if not scope_text:
            continue
        normalized_scope = re.sub(r"[\s，,。.!！?？；;：“”\"'‘’（）()、]+", "", scope_text)
        if normalized_scope and (normalized_scope in normalized_text or scope_text in text):
            return scope_text
    return ""
