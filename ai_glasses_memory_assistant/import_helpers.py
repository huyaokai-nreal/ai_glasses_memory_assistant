from __future__ import annotations

import json
import os
from typing import Any

from .memory_candidate import MemoryWriteCandidate
from .memory_store import (
    normalize_memory_kind,
    normalize_memory_type,
)
from .turn_semantic_classifier import _parse_json_object, classify_pre_reply_decision


# Batch classification covers many fragments in one call, so the output budget
# is larger than the per-fragment classifier's; thinking stays off for both.
IMPORT_BATCH_CLASSIFIER_OPTIONS: dict[str, Any] = {
    "response_format": {"type": "json_object"},
    "max_tokens": 4096,
    "disable_thinking": True,
}
IMPORT_BATCH_MIN_CONFIDENCE = 0.75


def import_batch_classification_enabled() -> bool:
    """Batch import classification is on by default; disable with AI_GLASSES_IMPORT_BATCH_CLASSIFY=0."""
    raw = str(os.environ.get("AI_GLASSES_IMPORT_BATCH_CLASSIFY") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


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


def _coerce_confidence(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_import_items_batch(
    contents: list[str],
    *,
    semantic_agent: Any | None,
    max_items_per_chunk: int = 20,
    max_chars_per_chunk: int = 6000,
) -> list[dict[str, Any]] | None:
    """Classify many import fragments with one LLM call per chunk.

    Returns one decision dict per input index; fragments the batch failed to
    classify keep an empty dict so the caller falls back to the per-fragment
    classifier for just those. Returns None only when the agent is missing or
    a chunk fails outright, in which case every fragment falls back.
    """
    if semantic_agent is None or not contents:
        return None
    decisions: list[dict[str, Any]] = [{} for _ in contents]
    start = 0
    while start < len(contents):
        chunk: list[tuple[int, str]] = []
        chunk_chars = 0
        index = start
        while index < len(contents):
            content = str(contents[index] or "")
            if chunk and (
                len(chunk) >= max_items_per_chunk
                or chunk_chars + len(content) > max_chars_per_chunk
            ):
                break
            chunk.append((index, content))
            chunk_chars += len(content)
            index += 1
        # Any batch failure degrades to the per-fragment path; imports never break.
        try:
            if not _classify_import_items_chunk(chunk, semantic_agent, decisions):
                return None
        except Exception:
            return None
        start = index
    return decisions


def _classify_import_items_chunk(
    chunk: list[tuple[int, str]],
    semantic_agent: Any,
    decisions: list[dict[str, Any]],
) -> bool:
    system_message = (
        "You are a structured memory import classifier. For each numbered fragment "
        "decide whether it should be stored as a long-term personal memory. Return "
        'JSON only: {"results": [{"index": <number>, "memory_action": "write"|"none", '
        '"kind": "profile"|"event"|"assistant_preference"|"", "memory_type": '
        '"fact"|"event"|"task"|"preference"|"decision"|"project_state"|"observation"|"", '
        '"confidence": 0.0, "reason": "..."}]}. Use memory_action "none" with empty '
        "kind/type for chatter, questions, transient or low-value fragments; do not "
        "guess types. Emit exactly one entry per input index."
    )
    numbered = "\n".join(f"{index}: {content}" for index, content in chunk)
    prompt = f"Fragments:\n{numbered}"
    result = semantic_agent.run_conversation(
        prompt,
        system_message=system_message,
        **IMPORT_BATCH_CLASSIFIER_OPTIONS,
    )
    raw = str((result or {}).get("final_response") or "")
    parsed = _parse_json_object(raw)
    results = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(results, list) or not results:
        return False
    chunk_indices = {index for index, _ in chunk}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        try:
            entry_index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if entry_index not in chunk_indices:
            continue
        confidence = _coerce_confidence(entry.get("confidence"))
        kind = normalize_memory_kind(str(entry.get("kind") or ""))
        memory_type = normalize_memory_type(str(entry.get("memory_type") or ""))
        memory_action = (
            "write" if str(entry.get("memory_action") or "").strip().lower() == "write" else "none"
        )
        accepted = (
            bool(kind)
            and bool(memory_type)
            and memory_action == "write"
            and confidence is not None
            and confidence >= IMPORT_BATCH_MIN_CONFIDENCE
        )
        decisions[entry_index] = {
            "source": "structured_classifier_batch",
            "status": "accepted" if accepted else "rejected_low_confidence",
            "kind": kind,
            "memory_type": memory_type,
            "memory_action": memory_action,
            "confidence": confidence,
            "reason": str(entry.get("reason") or "structured_batch_classification"),
        }
    return True


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
