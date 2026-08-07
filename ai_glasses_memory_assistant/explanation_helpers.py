from __future__ import annotations

import re
from typing import Any


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
