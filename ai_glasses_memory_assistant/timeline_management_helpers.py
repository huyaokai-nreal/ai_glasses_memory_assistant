from __future__ import annotations

from typing import Any

from .timeline_store import TimelineChunk, chunk_to_dict


def managed_timeline_chunk_payload(chunk: TimelineChunk, counts: Any | None = None) -> dict[str, Any]:
    payload = chunk_to_dict(chunk)
    payload["active_refs"] = int(getattr(counts, "active_refs", 0) or 0)
    payload["retained_refs"] = int(getattr(counts, "retained_refs", 0) or 0)
    payload["can_soft_delete"] = payload["active_refs"] <= 0
    payload["can_purge"] = payload["retained_refs"] <= 0
    return payload


def timeline_delete_result(
    chunk_id: str,
    *,
    action: str,
    status: str,
    reason: str,
    active_refs: int,
    retained_refs: int,
) -> dict[str, Any]:
    explanation_map = {
        "chunk_not_found": "来源不存在，可能之前已经被删掉了。",
        "chunk_already_deleted": "来源已经不再 active，所以不会再参与普通召回。",
        "active_memory_reference": "这段来源还被 active 记忆引用，所以现在不能删除。",
        "no_retained_memory_reference": "这段来源已经没有 retained 引用，所以这次直接彻底删除。",
        "retained_inactive_memory_reference": "这段来源还被 inactive 记忆保留引用，所以这次只做软删除。",
        "no_active_memory_reference": "这段来源不再被 active 记忆使用，所以这次做软删除。",
    }
    return {
        "chunk_id": chunk_id,
        "action": action,
        "status": status,
        "reason": reason,
        "active_refs": active_refs,
        "retained_refs": retained_refs,
        "explanation": explanation_map.get(reason, ""),
    }


def timeline_delete_summary(
    chunk_ids: list[str],
    results: list[dict[str, Any]],
    *,
    soft_deleted_count: int = 0,
) -> dict[str, Any]:
    deleted_or_inactive_source_ids = [
        str(result.get("chunk_id") or "")
        for result in results
        if str(result.get("action") or "") in {"soft_deleted", "purged", "already_deleted"}
    ]
    return {
        "requested_count": len(chunk_ids),
        "deleted_count": soft_deleted_count,
        "purged_count": sum(1 for result in results if result["action"] == "purged"),
        "retained_count": sum(1 for result in results if result["action"] == "retained"),
        "not_found_count": sum(1 for result in results if result["action"] == "not_found"),
        "deleted_or_inactive_source_ids": deleted_or_inactive_source_ids,
        "explanation": (
            "这些来源删除后，后续原话召回只会使用仍然 active 的 chunk。"
            if deleted_or_inactive_source_ids
            else "没有命中可处理的来源。"
        ),
        "results": results,
    }


def empty_purge_result(*, deleted: bool) -> dict[str, Any]:
    return {
        "deleted": deleted,
        "purged": False,
        "purged_chunk_count": 0,
        "purged_parent_count": 0,
        "soft_deleted_chunk_count": 0,
        "retained_evidence_count": 0,
        "audit_records_removed": 0,
    }


def timeline_redaction_debug(redaction: Any | None) -> dict[str, Any]:
    if redaction is None:
        return {
            "redacted": False,
            "redaction_categories": [],
            "redaction_count": 0,
        }
    return {
        "redacted": bool(getattr(redaction, "redacted", False)),
        "redaction_categories": list(getattr(redaction, "categories", []) or []),
        "redaction_count": int(getattr(redaction, "count", 0) or 0),
    }


def timeline_chunk_redaction_debug(chunk: TimelineChunk) -> dict[str, Any]:
    return {
        "redacted": bool(chunk.metadata.get("redacted")),
        "redaction_categories": list(chunk.metadata.get("redaction_categories") or []),
        "redaction_count": int(chunk.metadata.get("redaction_count") or 0),
    }
