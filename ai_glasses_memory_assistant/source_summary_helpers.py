from __future__ import annotations

from typing import Any


def source_summary(
    *,
    recalled_memories: list[dict[str, Any]],
    recalled_timeline_chunks: list[dict[str, Any]],
    recalled_documents: list[dict[str, Any]],
    saved_memories: list[dict[str, Any]],
    input_source: str = "chat",
    pending_confirmation_count: int = 0,
    rejected_count: int = 0,
    primary_source: str = "",
    primary_source_reason: str = "",
    dropped_sources: list[dict[str, Any]] | None = None,
    deleted_or_inactive_source_ids: list[str] | None = None,
    discussion_topic_count: int = 0,
) -> dict[str, Any]:
    structured = [item for item in [*recalled_memories, *saved_memories] if isinstance(item, dict)]
    profile_count = sum(1 for item in structured if item.get("kind") in {"profile", "assistant_preference"})
    event_count = sum(1 for item in structured if item.get("kind") == "event")
    observation_count = sum(1 for item in structured if item.get("memory_type") == "observation")
    payload = {
        "input_source": input_source,
        "structured_memory_count": len(structured),
        "profile_count": profile_count,
        "event_count": event_count,
        "observation_count": observation_count,
        "timeline_chunk_count": len(recalled_timeline_chunks),
        "document_count": len(recalled_documents),
        "discussion_topic_count": int(discussion_topic_count),
        "saved_memory_count": len(saved_memories),
        "pending_confirmation_count": pending_confirmation_count,
        "rejected_count": rejected_count,
        "source_types": source_types_for_summary(
            structured_memory_count=len(structured),
            timeline_chunk_count=len(recalled_timeline_chunks),
            document_count=len(recalled_documents),
            observation_count=observation_count,
            saved_memory_count=len(saved_memories),
            discussion_topic_count=discussion_topic_count,
        ),
    }
    payload.update(
        source_basis_summary(
            primary_source=primary_source,
            primary_source_reason=primary_source_reason,
            structured_memory_count=len(structured),
            observation_count=observation_count,
            timeline_chunk_count=len(recalled_timeline_chunks),
            document_count=len(recalled_documents),
            saved_memory_count=len(saved_memories),
            discussion_topic_count=discussion_topic_count,
            dropped_sources=dropped_sources or [],
            deleted_or_inactive_source_ids=deleted_or_inactive_source_ids or [],
        )
    )
    return payload


def source_summary_from_debug(
    *,
    recalled_memories: list[dict[str, Any]],
    recalled_timeline_chunks: list[dict[str, Any]],
    recalled_documents: list[dict[str, Any]],
    saved_memories: list[dict[str, Any]],
    debug: dict[str, Any] | None,
    input_source: str = "chat",
    pending_confirmation_count: int = 0,
    rejected_count: int = 0,
    fallback_primary_source: str = "",
    fallback_primary_source_reason: str = "",
) -> dict[str, Any]:
    debug = dict(debug or {})
    arbitration = dict(debug.get("memory", {}).get("recall_arbitration") or {})
    discussion_topic_count = len(debug.get("discussion_archive", {}).get("topics") or [])
    return source_summary(
        recalled_memories=recalled_memories,
        recalled_timeline_chunks=recalled_timeline_chunks,
        recalled_documents=recalled_documents,
        saved_memories=saved_memories,
        input_source=input_source,
        pending_confirmation_count=pending_confirmation_count,
        rejected_count=rejected_count,
        primary_source=(
            "discussion_archive"
            if discussion_topic_count else str(arbitration.get("primary_source") or fallback_primary_source or "")
        ),
        primary_source_reason=(
            "discussion_archive_requested_range"
            if discussion_topic_count else str(arbitration.get("primary_source_reason") or fallback_primary_source_reason or "")
        ),
        dropped_sources=list(arbitration.get("decisions") or []),
        deleted_or_inactive_source_ids=deleted_or_inactive_source_ids_from_debug(debug),
        discussion_topic_count=discussion_topic_count,
    )


def audit_summary(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source_summary") if isinstance(record.get("source_summary"), dict) else {}
    return {
        "record_type": str(record.get("record_type") or "chat_turn"),
        "source": str(record.get("source") or source.get("input_source") or "chat"),
        "structured_memory_count": int(source.get("structured_memory_count") or 0),
        "timeline_chunk_count": int(source.get("timeline_chunk_count") or 0),
        "document_count": int(source.get("document_count") or 0),
        "saved_memory_count": int(source.get("saved_memory_count") or 0),
    }


def source_basis_summary(
    *,
    primary_source: str,
    primary_source_reason: str,
    structured_memory_count: int,
    observation_count: int,
    timeline_chunk_count: int,
    document_count: int,
    saved_memory_count: int,
    dropped_sources: list[dict[str, Any]],
    deleted_or_inactive_source_ids: list[str],
    discussion_topic_count: int = 0,
) -> dict[str, Any]:
    effective_primary = infer_primary_source(
        primary_source=primary_source,
        structured_memory_count=structured_memory_count,
        observation_count=observation_count,
        timeline_chunk_count=timeline_chunk_count,
        document_count=document_count,
        discussion_topic_count=discussion_topic_count,
    )
    return {
        "primary_source": effective_primary,
        "primary_source_label": primary_source_label(effective_primary),
        "primary_source_explanation": primary_source_explanation(
            primary_source=effective_primary,
            primary_source_reason=primary_source_reason,
            structured_memory_count=structured_memory_count,
            observation_count=observation_count,
            timeline_chunk_count=timeline_chunk_count,
            document_count=document_count,
            saved_memory_count=saved_memory_count,
            discussion_topic_count=discussion_topic_count,
            deleted_or_inactive_source_ids=deleted_or_inactive_source_ids,
        ),
        "dropped_source_count": len(dropped_sources),
        "dropped_sources": [dict(item) for item in dropped_sources if isinstance(item, dict)][:8],
        "deleted_or_inactive_source_ids": list(dict.fromkeys(deleted_or_inactive_source_ids)),
    }


def infer_primary_source(
    *,
    primary_source: str,
    structured_memory_count: int,
    observation_count: int,
    timeline_chunk_count: int,
    document_count: int,
    discussion_topic_count: int = 0,
) -> str:
    normalized = str(primary_source or "").strip()
    if normalized:
        return normalized
    if discussion_topic_count:
        return "discussion_archive"
    if document_count:
        return "document"
    if timeline_chunk_count:
        return "raw_timeline"
    if observation_count:
        return "observation"
    if structured_memory_count:
        return "structured_memory"
    return "none"


def primary_source_label(source: str) -> str:
    labels = {
        "structured_memory": "结构化记忆",
        "observation": "观察总结",
        "document": "文档原文",
        "raw_timeline": "timeline 原话",
        "discussion_archive": "全天讨论归档",
        "profile": "稳定画像",
        "none": "无可用来源",
    }
    return labels.get(str(source or ""), "混合来源")


def primary_source_explanation(
    *,
    primary_source: str,
    primary_source_reason: str,
    structured_memory_count: int,
    observation_count: int,
    timeline_chunk_count: int,
    document_count: int,
    saved_memory_count: int,
    deleted_or_inactive_source_ids: list[str],
    discussion_topic_count: int = 0,
) -> str:
    source = str(primary_source or "none")
    if source == "document":
        if primary_source_reason == "document_detail_primary":
            return "这次主要依据上传文档原文，结构化记忆或 observation 只作为背景。"
        return "这次主要依据上传文档原文，而不是摘要记忆。"
    if source == "raw_timeline":
        if deleted_or_inactive_source_ids:
            return "这次只能依据仍然 active 的 timeline 原话；部分旧来源已删除或不再可用。"
        return "这次主要依据 timeline 原话 chunk，而不是后续总结。"
    if source == "discussion_archive":
        return "这次主要依据按时间和话题形成的全天讨论摘要；可用原文证据只用于核对。"
    if source == "observation":
        return "这次主要依据 observation 总结，并结合相关结构化记忆。"
    if source == "structured_memory":
        if document_count:
            return "这次主要依据结构化任务、决策或项目状态，文档只作为背景补充。"
        return "这次主要依据结构化任务、决策或项目状态。"
    if source == "profile":
        return "这次主要依据稳定画像记忆。"
    if deleted_or_inactive_source_ids:
        return "当前没有可用来源了，相关原始证据已删除或不再 active。"
    if saved_memory_count and not (structured_memory_count or observation_count or timeline_chunk_count or document_count):
        return "这次没有召回旧来源，只有本轮新保存的记忆。"
    return "当前没有可用来源。"


def source_types_for_summary(
    *,
    structured_memory_count: int,
    timeline_chunk_count: int,
    document_count: int,
    observation_count: int,
    saved_memory_count: int,
    discussion_topic_count: int = 0,
) -> list[str]:
    source_types: list[str] = []
    if structured_memory_count:
        source_types.append("structured_memory")
    if observation_count:
        source_types.append("observation")
    if timeline_chunk_count:
        source_types.append("raw_timeline")
    if document_count:
        source_types.append("document")
    if saved_memory_count:
        source_types.append("memory_write")
    if discussion_topic_count:
        source_types.append("discussion_archive")
    return source_types


def deleted_or_inactive_source_ids_from_debug(debug: dict[str, Any] | None) -> list[str]:
    debug = dict(debug or {})
    deleted_ids: list[str] = []
    timeline_recall = dict(debug.get("timeline", {}).get("recall") or {})
    if str(timeline_recall.get("reason") or "") in {
        "raw_evidence_source_missing_or_deleted",
        "summary_recall_source_missing_or_deleted",
    }:
        deleted_ids.extend(str(item) for item in timeline_recall.get("missing_source_ids") or [] if str(item).strip())
    arbitration = dict(debug.get("memory", {}).get("recall_arbitration") or {})
    empty_guard = dict(arbitration.get("empty_evidence_guard") or {})
    deleted_ids.extend(str(item) for item in empty_guard.get("missing_source_ids") or [] if str(item).strip())
    return list(dict.fromkeys(deleted_ids))
