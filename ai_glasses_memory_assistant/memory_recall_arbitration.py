from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecallArbitrationResult:
    profile_memories: list[Any]
    event_memories: list[Any]
    timeline_chunks: list[Any]
    primary_source: str
    debug: dict[str, Any]
    empty_evidence_guard: dict[str, Any] = field(default_factory=dict)


def arbitrate_recall_sources(
    *,
    message: str,
    recall_goal: str,
    reply_mode: str = "",
    event_recall_strategy: str = "",
    profile_memories: list[Any],
    event_memories: list[Any],
    timeline_chunks: list[Any],
    document_mode: str,
    document_count: int,
) -> RecallArbitrationResult:
    """Apply deterministic priority rules after retrieval and before reply synthesis."""
    recall_goal = str(recall_goal or "none")
    reply_mode = str(reply_mode or "")
    event_recall_strategy = str(event_recall_strategy or "")
    document_mode = str(document_mode or "")
    kept_profiles = list(profile_memories)
    kept_events = list(event_memories)
    kept_timeline = list(timeline_chunks)
    decisions: list[dict[str, Any]] = []
    empty_guard = {
        "triggered": False,
        "reason": "",
        "reply": "",
    }
    summary_scope = ""
    dynamic_evidence = False
    profile_treatment = ""
    primary_source = _primary_source(
        recall_goal=recall_goal,
        profile_memories=kept_profiles,
        event_memories=kept_events,
        timeline_chunks=kept_timeline,
        document_mode=document_mode,
        document_count=document_count,
    )

    document_with_structured_summary = (
        document_mode in {"full_document", "sections"}
        and document_count
        and recall_goal == "summary"
        and any(_memory_type(memory) in {"project_state", "task", "decision"} for memory in kept_events)
    )
    if document_with_structured_summary:
        primary_source = "structured_memory"
        observations = [memory for memory in kept_events if _memory_type(memory) == "observation"]
        if observations:
            decisions.extend(_drop_decisions(observations, reason="summary_structured_memory_primary_over_observation"))
            kept_events = [memory for memory in kept_events if _memory_type(memory) != "observation"]
        decisions.append(
            {
                "source": "document",
                "action": "background",
                "reason": "summary_structured_memory_primary_document_background",
            }
        )
    elif document_mode in {"full_document", "sections"} and document_count:
        primary_source = "document"
        if kept_profiles:
            decisions.extend(_drop_decisions(kept_profiles, reason="document_detail_primary_over_profile"))
            kept_profiles = []
        if kept_events:
            decisions.extend(_drop_decisions(kept_events, reason="document_detail_primary_over_structured_memory"))
            kept_events = []
        if kept_timeline and recall_goal != "raw_evidence":
            decisions.extend(_drop_timeline_decisions(kept_timeline, reason="document_detail_primary_over_timeline"))
            kept_timeline = []

    elif recall_goal == "raw_evidence":
        primary_source = "raw_timeline" if kept_timeline else "none"
        if kept_profiles:
            decisions.extend(_drop_decisions(kept_profiles, reason="raw_evidence_primary_over_profile"))
            kept_profiles = []
        if kept_events:
            decisions.extend(_drop_decisions(kept_events, reason="raw_evidence_primary_over_structured_memory"))
            kept_events = []
        if not kept_timeline:
            empty_guard = {
                "triggered": True,
                "reason": "raw_evidence_requested_but_no_timeline_chunks",
                "reply": "我没有找到之前相关的原话。",
            }

    elif recall_goal == "specific_fact":
        observations = [memory for memory in kept_events if _memory_type(memory) == "observation"]
        direct_events = [memory for memory in kept_events if _memory_type(memory) != "observation"]
        if observations:
            decisions.extend(_drop_decisions(observations, reason="specific_fact_drops_observation"))
        kept_events = direct_events
        primary_source = _specific_fact_primary_source(kept_profiles, kept_events)
        if (
            reply_mode in {"local_event_recall", "local_profile_recall"}
            and event_recall_strategy != "ambiguous_recent_upcoming_plan"
            and not kept_profiles
            and not kept_events
            and not document_count
            and not kept_timeline
        ):
            empty_guard = {
                "triggered": True,
                "reason": "specific_fact_requested_but_no_direct_evidence",
                "reply": "",
            }
            primary_source = "none"

    elif recall_goal == "summary":
        summary_scope = "structured_unspecified"
        dynamic_evidence = _summary_has_dynamic_evidence(kept_events, kept_timeline, document_count)
        profile_treatment = ""
        if kept_profiles:
            profile_treatment = "kept_structured_summary_scope"
        primary_source = _summary_primary_source(kept_events, kept_profiles, kept_timeline, document_count)

    debug = {
        "primary_source": primary_source,
        "recall_goal": recall_goal,
        "summary_profile_policy": {
            "scope": summary_scope,
            "dynamic_evidence": dynamic_evidence,
            "profile_treatment": profile_treatment,
        },
        "kept_counts": {
            "profile": len(kept_profiles),
            "event": len(kept_events),
            "timeline": len(kept_timeline),
            "document": int(document_count or 0),
        },
        "dropped_counts": _dropped_counts(decisions),
        "decisions": decisions,
        "empty_evidence_guard": empty_guard,
    }
    if document_with_structured_summary:
        debug["primary_source_reason"] = "summary_structured_memory_primary_with_document_background"
    return RecallArbitrationResult(
        profile_memories=kept_profiles,
        event_memories=kept_events,
        timeline_chunks=kept_timeline,
        primary_source=primary_source,
        debug=debug,
        empty_evidence_guard=empty_guard,
    )


def recall_arbitration_with_reason(arbitration_debug: dict[str, Any]) -> dict[str, Any]:
    payload = dict(arbitration_debug or {})
    primary_source = str(payload.get("primary_source") or "none")
    reason_map = {
        "document": "document_detail_primary",
        "raw_timeline": "raw_timeline_primary",
        "structured_memory": "structured_memory_primary",
        "observation": "observation_primary",
        "profile": "profile_primary",
        "none": "no_available_source",
    }
    payload["primary_source_reason"] = str(payload.get("primary_source_reason") or reason_map.get(primary_source, "mixed_source_priority"))
    return payload


def _primary_source(
    *,
    recall_goal: str,
    profile_memories: list[Any],
    event_memories: list[Any],
    timeline_chunks: list[Any],
    document_mode: str,
    document_count: int,
) -> str:
    if document_mode in {"full_document", "sections"} and document_count:
        return "document"
    if recall_goal == "raw_evidence":
        return "raw_timeline" if timeline_chunks else "none"
    if any(_memory_type(memory) == "observation" for memory in event_memories):
        return "observation"
    if any(_kind(memory) == "event" for memory in event_memories):
        return "structured_memory"
    if profile_memories:
        return "profile"
    if timeline_chunks:
        return "raw_timeline"
    return "none"


def _specific_fact_primary_source(profile_memories: list[Any], event_memories: list[Any]) -> str:
    if event_memories:
        return "structured_memory"
    if profile_memories:
        return "profile"
    return "none"


def _summary_primary_source(
    event_memories: list[Any],
    profile_memories: list[Any],
    timeline_chunks: list[Any],
    document_count: int,
) -> str:
    if any(_memory_type(memory) == "observation" for memory in event_memories):
        return "observation"
    if any(_kind(memory) == "event" for memory in event_memories):
        return "structured_memory"
    if profile_memories:
        return "profile"
    if document_count:
        return "document"
    if timeline_chunks:
        return "raw_timeline"
    return "none"


def _summary_has_dynamic_evidence(event_memories: list[Any], timeline_chunks: list[Any], document_count: int) -> bool:
    return bool(event_memories or timeline_chunks or document_count)


def _drop_decisions(memories: list[Any], *, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "source": _source_for_memory(memory),
            "id": str(getattr(memory, "id", "") or ""),
            "kind": _kind(memory),
            "memory_type": _memory_type(memory),
            "action": "drop",
            "reason": reason,
        }
        for memory in memories
    ]


def _drop_timeline_decisions(chunks: list[Any], *, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "source": "raw_timeline",
            "id": str(getattr(chunk, "id", "") or ""),
            "action": "drop",
            "reason": reason,
        }
        for chunk in chunks
    ]


def _dropped_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "profile": 0,
        "event": 0,
        "observation": 0,
        "timeline": 0,
    }
    for decision in decisions:
        source = str(decision.get("source") or "")
        memory_type = str(decision.get("memory_type") or "")
        if source == "raw_timeline":
            counts["timeline"] += 1
        elif memory_type == "observation":
            counts["observation"] += 1
        elif source == "profile":
            counts["profile"] += 1
        elif source == "structured_memory":
            counts["event"] += 1
    return counts


def _source_for_memory(memory: Any) -> str:
    if _kind(memory) == "profile":
        return "profile"
    if _memory_type(memory) == "observation":
        return "observation"
    return "structured_memory"


def _kind(memory: Any) -> str:
    return str(getattr(memory, "kind", "") or "")


def _memory_type(memory: Any) -> str:
    return str(getattr(memory, "memory_type", "") or "")
