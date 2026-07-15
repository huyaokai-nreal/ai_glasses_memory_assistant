from __future__ import annotations

from typing import Any

from .memory_candidate import MemoryWriteCandidate
from .memory_store import (
    classify_memory_kind_with_reason,
    classify_memory_type_with_reason,
    normalize_memory_kind,
    normalize_memory_type,
)


def import_items_from_payload(*, items: list[dict[str, Any]] | None, text: str) -> list[dict[str, Any]]:
    if items:
        return [item for item in items if isinstance(item, dict)]
    chunks = []
    for line in str(text or "").splitlines():
        cleaned = line.strip(" \t-•0123456789.、")
        if cleaned:
            chunks.append({"content": cleaned})
    if chunks:
        return chunks
    cleaned = str(text or "").strip()
    return [{"content": cleaned}] if cleaned else []


def classify_import_item(item: dict[str, Any], content: str) -> tuple[str, str, list[dict[str, str]]]:
    classification_debug: list[dict[str, str]] = []
    explicit_kind = str(item.get("kind") or "").strip()
    explicit_type = str(item.get("memory_type") or "").strip()
    explicit_memory_type = normalize_memory_type(explicit_type) if explicit_type else ""
    if explicit_kind:
        kind = normalize_memory_kind(explicit_kind)
    elif explicit_memory_type:
        kind = "profile" if explicit_memory_type == "preference" else "event"
    else:
        kind_classification = classify_memory_kind_with_reason(content)
        kind = kind_classification.value
        classification_debug.append(kind_classification.debug_payload("kind"))
    if explicit_memory_type:
        memory_type = explicit_memory_type
    else:
        type_classification = classify_memory_type_with_reason(content, kind)
        memory_type = normalize_memory_type(type_classification.value)
        classification_debug.append(type_classification.debug_payload("memory_type"))
    if not explicit_kind and memory_type == "preference":
        kind = "profile"
    return kind, memory_type, classification_debug


def candidate_from_import_item(
    item: dict[str, Any],
    *,
    content: str,
    kind: str,
    memory_type: str,
    source_id: str,
    ingestion_id: str,
    source: str,
    confidence: float,
    classification_debug: list[dict[str, str]] | None = None,
) -> MemoryWriteCandidate:
    evidence_ids = item.get("evidence_ids")
    if not isinstance(evidence_ids, list):
        evidence_ids = [source_id]
    classification_debug = classification_debug or []
    fallback_reasons = [
        f"{entry.get('field')}={entry.get('source')}:{entry.get('reason')}"
        for entry in classification_debug
        if entry.get("field") and entry.get("source") and entry.get("reason")
    ]
    base_reason = str(item.get("reason") or f"imported_from_{source}")
    reason = base_reason
    if fallback_reasons:
        reason = f"{base_reason}; " + "; ".join(fallback_reasons)
    return MemoryWriteCandidate(
        content=content,
        kind=kind,
        memory_type=memory_type,
        privacy_level=str(item.get("privacy_level") or "normal"),
        confidence=confidence,
        reason=reason,
        source=source,
        source_id=source_id,
        ingestion_id=ingestion_id,
        evidence_ids=[str(eid) for eid in evidence_ids if str(eid).strip()],
        source_type=str(item.get("source_type") or source or "manual_import"),
        speaker_hint=str(item.get("speaker_hint") or ""),
        do_not_remember_scope=str(item.get("do_not_remember_scope") or ""),
        subject_id=str(item.get("subject_id") or "").strip(),
        subject_type=str(item.get("subject_type") or "self").strip().lower(),
        subject_name=str(item.get("subject_name") or "").strip(),
        subject_scope=str(item.get("subject_scope") or "").strip(),
    )
