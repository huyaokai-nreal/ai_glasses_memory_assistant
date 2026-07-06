from __future__ import annotations

from typing import Any


ACTIVE_MEMORY_STATUS = "active"
STALE_MEMORY_STATUS = "stale"
SUPERSEDED_MEMORY_STATUS = "superseded"
DELETED_MEMORY_STATUS = "deleted"

VALID_MEMORY_STATUSES = {
    ACTIVE_MEMORY_STATUS,
    STALE_MEMORY_STATUS,
    SUPERSEDED_MEMORY_STATUS,
    DELETED_MEMORY_STATUS,
}


def normalize_memory_status(status: str) -> str:
    normalized = (status or "").strip().lower()
    return normalized if normalized in VALID_MEMORY_STATUSES else ACTIVE_MEMORY_STATUS


def is_active_memory_status(status: str) -> bool:
    return normalize_memory_status(status) == ACTIVE_MEMORY_STATUS


def can_transition_memory_status(
    from_status: str,
    to_status: str,
    *,
    superseded_by: str = "",
) -> bool:
    source = normalize_memory_status(from_status)
    target = normalize_memory_status(to_status)
    if source == DELETED_MEMORY_STATUS or source == target:
        return False
    if target == DELETED_MEMORY_STATUS:
        return True
    if source != ACTIVE_MEMORY_STATUS:
        return False
    if target == STALE_MEMORY_STATUS:
        return True
    if target == SUPERSEDED_MEMORY_STATUS:
        return bool(str(superseded_by or "").strip())
    return False


def lifecycle_transition_payload(
    *,
    memory_id: str,
    from_status: str,
    to_status: str,
    reason: str,
    superseded_by: str = "",
) -> dict[str, Any]:
    payload = {
        "memory_id": str(memory_id or ""),
        "from_status": normalize_memory_status(from_status),
        "to_status": normalize_memory_status(to_status),
        "reason": str(reason or ""),
    }
    if superseded_by:
        payload["superseded_by"] = str(superseded_by)
    return payload
