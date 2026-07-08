from __future__ import annotations

from dataclasses import dataclass


TIMELINE_CHUNK_EVIDENCE_PREFIX = "chunk_"


@dataclass(frozen=True)
class EvidenceReferenceCounts:
    active_refs: int = 0
    retained_refs: int = 0


def normalize_evidence_ids(evidence_ids: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    return list(dict.fromkeys(str(evidence_id).strip() for evidence_id in (evidence_ids or []) if str(evidence_id).strip()))


def is_timeline_chunk_evidence_id(evidence_id: str) -> bool:
    return str(evidence_id or "").strip().startswith(TIMELINE_CHUNK_EVIDENCE_PREFIX)


def timeline_chunk_evidence_ids(evidence_ids: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    return [evidence_id for evidence_id in normalize_evidence_ids(evidence_ids) if is_timeline_chunk_evidence_id(evidence_id)]


def plan_timeline_evidence_cleanup(
    evidence_ids: list[str] | tuple[str, ...] | set[str] | None,
    reference_counts: dict[str, EvidenceReferenceCounts],
    *,
    hard_purge: bool,
) -> dict[str, list[str]]:
    soft_delete_chunk_ids: list[str] = []
    purge_chunk_ids: list[str] = []
    retained_chunk_ids: list[str] = []
    for evidence_id in timeline_chunk_evidence_ids(evidence_ids):
        counts = reference_counts.get(evidence_id) or EvidenceReferenceCounts()
        if counts.active_refs > 0:
            retained_chunk_ids.append(evidence_id)
            continue
        if hard_purge and counts.retained_refs <= 0:
            purge_chunk_ids.append(evidence_id)
            continue
        soft_delete_chunk_ids.append(evidence_id)
    return {
        "soft_delete_chunk_ids": soft_delete_chunk_ids,
        "purge_chunk_ids": purge_chunk_ids,
        "retained_chunk_ids": retained_chunk_ids,
    }
