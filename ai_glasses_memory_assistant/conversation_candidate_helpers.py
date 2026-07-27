from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .conversation_helpers import ConversationSession
from .memory_candidate import MemoryWriteCandidate
from .privacy_filter import redact_sensitive_text


@dataclass(frozen=True)
class ConversationExtractionUnit:
    text: str
    speaker_label: str
    speaker_role: str
    turn_index: int
    fragment_index: int
    subject_type: str
    subject_name: str
    subject_id: str = ""
    subject_scope: str = ""
    evidence_id: str = ""
    audio_event_id: str = ""
    speaker_state: str = ""
    overlap_state: str = ""
    memory_eligible: bool = True

    def policy_context(self) -> dict[str, Any]:
        return {
            "source_type": "multi_speaker_transcript",
            "speaker_label": self.speaker_label,
            "speaker_role": self.speaker_role,
            "turn_index": self.turn_index,
            "fragment_index": self.fragment_index,
            "subject_type": self.subject_type,
            "subject_name": self.subject_name,
            "subject_id": self.subject_id,
            "subject_scope": self.subject_scope,
            "evidence_id": self.evidence_id,
            "audio_event_id": self.audio_event_id,
            "speaker_state": self.speaker_state,
            "overlap_state": self.overlap_state,
            "memory_eligible": self.memory_eligible,
        }

    def debug_payload(self) -> dict[str, Any]:
        return {
            **self.policy_context(),
            "text": self.text,
        }


def conversation_memory_candidates(
    session: ConversationSession,
) -> tuple[list[MemoryWriteCandidate], dict[str, Any]]:
    debug = session.debug_payload()
    debug["candidate_count"] = 0
    debug["candidate_strategy"] = "structural_only"
    debug["candidate_turn_indices"] = []
    debug["candidate_facts"] = []
    debug["saved_candidates"] = []
    debug["rejected_turns"] = []
    debug["rejected_reasons"] = []
    debug["candidate_previews"] = []
    return [], debug


def conversation_extraction_plan(
    session: ConversationSession,
    *,
    subject_scope: str = "",
) -> tuple[list[ConversationExtractionUnit], dict[str, Any]]:
    debug = session.debug_payload()
    debug["candidate_count"] = 0
    debug["candidate_strategy"] = "structured_semantic_input"
    debug["candidate_facts"] = []
    debug["saved_candidates"] = []
    debug["gate_rejected_candidates"] = []
    rejected_turns: list[dict[str, Any]] = []
    aliases_by_turn: dict[int, list[dict[str, Any]]] = {}
    for item in session.speaker_aliases:
        turn_index = int(item.get("turn_index", -1))
        if turn_index >= 0:
            aliases_by_turn.setdefault(turn_index, []).append(dict(item))
    units: list[ConversationExtractionUnit] = []
    for turn in session.turns:
        for fragment_index, fragment in enumerate(_conversation_fragments(turn.text)):
            aliases = aliases_by_turn.get(turn.turn_index, [])
            cleaned_fragment = _remove_alias_declarations(fragment, aliases)
            if aliases and not cleaned_fragment:
                rejected_turns.append(_rejected_turn(
                    turn,
                    reason="speaker_alias_declaration",
                    fragment_index=fragment_index,
                ))
                continue
            redaction = redact_sensitive_text(cleaned_fragment)
            if redaction.redacted:
                rejected_turns.append(_rejected_turn(
                    turn,
                    reason="sensitive_fragment_filtered",
                    fragment_index=fragment_index,
                ))
                continue
            units.append(ConversationExtractionUnit(
                text=cleaned_fragment,
                speaker_label=turn.speaker_label,
                speaker_role=turn.speaker_role,
                turn_index=turn.turn_index,
                fragment_index=fragment_index,
                subject_type=(
                    "self"
                    if turn.speaker_role == "user"
                    else "provisional"
                    if turn.speaker_role == "unknown_speaker"
                    else "named"
                ),
                subject_name="" if turn.speaker_role == "user" else turn.speaker_label,
                subject_scope=subject_scope,
            ))

    debug["candidate_turn_indices"] = list(dict.fromkeys(unit.turn_index for unit in units))
    debug["semantic_units"] = [unit.debug_payload() for unit in units]
    debug["rejected_turns"] = rejected_turns
    debug["rejected_reasons"] = list(dict.fromkeys(
        str(item.get("reason") or "") for item in rejected_turns if str(item.get("reason") or "")
    ))
    return units, debug


def _conversation_fragments(text: str) -> list[str]:
    return [
        fragment.strip(" ，,。.!！?？；;")
        for fragment in re.split(r"[\n\r]+|[，,；;]|(?<=[。！？!?])|(?<=\.)\s+", str(text or ""))
        if fragment.strip(" ，,。.!！?？；;")
    ]


def _remove_alias_declarations(fragment: str, aliases: list[dict[str, Any]]) -> str:
    cleaned = str(fragment or "")
    for item in aliases:
        source = re.escape(str(item.get("source_label") or "").strip())
        target = re.escape(str(item.get("target_label") or "").strip())
        if not source or not target:
            continue
        cleaned = re.sub(
            rf"{source}\s*(?:是|就是|叫)\s*{target}",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned.strip(" ，,。.!！?？；;")


def _rejected_turn(
    turn: Any,
    *,
    reason: str,
    fragment_index: int | None = None,
) -> dict[str, Any]:
    payload = {
        "turn_index": int(getattr(turn, "turn_index", -1)),
        "speaker_label": str(getattr(turn, "speaker_label", "") or ""),
        "speaker_role": str(getattr(turn, "speaker_role", "") or ""),
        "reason": reason,
    }
    if fragment_index is not None:
        payload["fragment_index"] = fragment_index
    return payload
