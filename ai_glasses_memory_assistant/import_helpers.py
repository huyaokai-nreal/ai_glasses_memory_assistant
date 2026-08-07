from __future__ import annotations

import json
from typing import Any

from .memory_candidate import MemoryWriteCandidate
from .memory_store import (
    normalize_memory_kind,
    normalize_memory_type,
)
from .turn_semantic_classifier import classify_pre_reply_decision


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


def classify_import_item(
    item: dict[str, Any],
    content: str,
    *,
    semantic_agent: Any | None = None,
) -> tuple[str, str, list[dict[str, str]]]:
    """Use explicit fields or a structured provider classification.

    Missing fields are intentionally pending when no provider is available;
    content is never inspected for a kind/type phrase fallback.
    """

    classification_debug: list[dict[str, str]] = []
    explicit_kind = str(item.get("kind") or "").strip()
    explicit_type = str(item.get("memory_type") or "").strip()
    kind = normalize_memory_kind(explicit_kind) if explicit_kind else ""
    memory_type = normalize_memory_type(explicit_type) if explicit_type else ""
    if kind and memory_type:
        classification_debug.append({
            "source": "explicit_fields",
            "status": "accepted",
            "reason": "kind_and_memory_type_provided",
        })
        return kind, memory_type, classification_debug
    if semantic_agent is None:
        classification_debug.append({
            "source": "structured_classifier",
            "status": "classification_pending",
            "reason": "provider_unavailable",
        })
        return "", "", classification_debug
    system_message = (
        "You are a structured memory import classifier. Return JSON only with "
        "kind (profile|event|assistant_preference), memory_type "
        "(fact|event|task|preference|decision|project_state|observation), "
        "confidence (0..1), and reason. Do not infer from a fixed phrase list."
    )
    try:
        result = semantic_agent.run_conversation(content, system_message=system_message)
        raw = str((result or {}).get("final_response") or "")
        parsed = json.loads(raw)
        classified_kind = normalize_memory_kind(str(parsed.get("kind") or ""))
        classified_type = normalize_memory_type(str(parsed.get("memory_type") or ""))
        confidence = float(parsed.get("confidence"))
        if not str(parsed.get("kind") or "").strip() or not str(parsed.get("memory_type") or "").strip():
            raise ValueError("missing kind or memory_type")
        classification_debug.append({
            "source": "structured_classifier",
            "status": "accepted" if confidence >= 0.75 else "rejected_low_confidence",
            "reason": str(parsed.get("reason") or "structured_classification"),
            "confidence": str(confidence),
        })
        if confidence < 0.75:
            return "", "", classification_debug
        return kind or classified_kind, memory_type or classified_type, classification_debug
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError, KeyError):
        # Reuse the repository's unified structured classifier when a provider
        # only exposes the pre-reply contract. This is still model output, not
        # a phrase-based fallback; invalid or unavailable output remains pending.
        try:
            decision = classify_pre_reply_decision(semantic_agent, content)
            classified_kind = normalize_memory_kind(decision.memory_kind)
            classified_type = normalize_memory_type(decision.memory_type)
            confidence = float(decision.confidence or 0.0)
            if (
                decision.error
                or decision.backend != "llm"
                or decision.memory_action != "write"
                or not classified_kind
                or not classified_type
            ):
                raise ValueError("unusable_unified_classification")
            classification_debug.append({
                "source": "structured_classifier",
                "backend": "unified_pre_reply_decision",
                "status": "accepted" if confidence >= 0.75 else "rejected_low_confidence",
                "reason": decision.reason or "unified_structured_classification",
                "confidence": str(confidence),
            })
            if confidence < 0.75:
                return "", "", classification_debug
            return kind or classified_kind, memory_type or classified_type, classification_debug
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError, KeyError):
            classification_debug.append({
                "source": "structured_classifier",
                "status": "classification_pending",
                "reason": "invalid_or_incomplete_provider_result",
            })
            return "", "", classification_debug


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
