from __future__ import annotations

from typing import Any


MEMORY_KERNEL_VERSION = "2026-05-17.v1"

_LAYER_DELETE_POLICY = {
    "raw_timeline": "soft_delete_chunk; parent turn/capture becomes partial_deleted or deleted",
    "structured_memory": "soft_delete_memory; delete unshared raw evidence chunks",
    "reflection": "supersede_or_soft_delete; never replace source evidence silently",
    "runtime_recall": "ephemeral_context_only; nothing is deleted from runtime recall",
    "document_archive": "soft_delete_document; document no longer participates in recall",
}

_LAYERS = (
    {
        "name": "raw_timeline",
        "stores": ("raw_turns", "captures", "chunks"),
        "purpose": "durable original user text and capture chunks used for full-text recall and evidence",
    },
    {
        "name": "structured_memory",
        "stores": ("memories", "documents"),
        "purpose": "user-visible profile, event, assistant preference, task, decision, and document records",
    },
    {
        "name": "reflection",
        "stores": ("memories.memory_type=observation",),
        "purpose": "background summaries derived from evidence-backed structured memories",
    },
    {
        "name": "runtime_recall",
        "stores": ("debug.recall", "memory-context"),
        "purpose": "small top-k context injected for the current turn only",
    },
)


def memory_kernel_contract() -> dict[str, Any]:
    return {
        "version": MEMORY_KERNEL_VERSION,
        "layers": [
            {
                **layer,
                "delete_policy": _LAYER_DELETE_POLICY[layer["name"]],
            }
            for layer in _LAYERS
        ],
        "write_contract": {
            "required_fields": (
                "user_id",
                "source",
                "source_id",
                "ingestion_id",
                "evidence_ids",
                "privacy_level",
                "status",
                "confidence",
            ),
            "gate": "should_write_memory_candidate",
            "rule": "write candidates first, then gate, dedupe, persist, audit",
        },
        "recall_contract": {
            "rule": "planner/router opens a specific gate; only relevant top-k evidence is injected",
            "default": "ordinary chat and factual QA do not read long-term memory",
        },
    }


def memory_kernel_summary() -> dict[str, Any]:
    return {
        "version": MEMORY_KERNEL_VERSION,
        "layer_names": [layer["name"] for layer in _LAYERS],
    }


def source_trace(
    *,
    layer: str,
    user_id: str,
    source: str = "",
    source_id: str = "",
    ingestion_id: str = "",
    evidence_ids: list[str] | None = None,
    privacy_level: str = "normal",
    status: str = "active",
    confidence: float | None = None,
) -> dict[str, Any]:
    return {
        "kernel_version": MEMORY_KERNEL_VERSION,
        "layer": layer,
        "user_id": user_id,
        "source": source,
        "source_id": source_id,
        "ingestion_id": ingestion_id,
        "evidence_ids": _unique_strings(evidence_ids or []),
        "privacy_level": privacy_level or "normal",
        "status": status or "active",
        "confidence": confidence,
        "delete_policy": _LAYER_DELETE_POLICY.get(layer, "soft_delete"),
    }


def recall_trace(
    *,
    layer: str,
    strategy: str,
    count: int,
    reason: str = "",
    query: str = "",
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "kernel_version": MEMORY_KERNEL_VERSION,
        "layer": layer,
        "strategy": strategy,
        "count": max(0, int(count)),
        "reason": reason,
        "query": query,
        "evidence_ids": _unique_strings(evidence_ids or []),
        "delete_policy": _LAYER_DELETE_POLICY.get(layer, "soft_delete"),
    }


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))
