from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConversationParticipant:
    label: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "role": self.role,
        }


@dataclass(frozen=True)
class ConversationTurn:
    speaker_label: str
    speaker_role: str
    text: str
    timestamp_text: str = ""
    turn_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_label": self.speaker_label,
            "speaker_role": self.speaker_role,
            "text": self.text,
            "timestamp_text": self.timestamp_text,
            "turn_index": self.turn_index,
        }


@dataclass(frozen=True)
class ConversationSession:
    turns: list[ConversationTurn]
    participants: dict[str, str]
    source: str = "speaker_labeled_transcript"
    speaker_aliases: list[dict[str, Any]] = field(default_factory=list)
    alias_applied_turns: list[dict[str, Any]] = field(default_factory=list)

    def debug_payload(self) -> dict[str, Any]:
        return {
            "detected": True,
            "source": self.source,
            "turn_count": len(self.turns),
            "participants": dict(self.participants),
            "speaker_aliases": [dict(alias) for alias in self.speaker_aliases],
            "alias_applied_turns": [dict(item) for item in self.alias_applied_turns],
            "parsed_turns": [turn.to_dict() for turn in self.turns],
            "known_participants": [
                label for label, role in self.participants.items() if role == "known_person"
            ],
            "unknown_speaker_count": sum(1 for role in self.participants.values() if role == "unknown_speaker"),
            "user_turn_count": sum(1 for turn in self.turns if turn.speaker_role == "user"),
            "non_user_turn_count": sum(1 for turn in self.turns if turn.speaker_role != "user"),
        }


def parse_speaker_labeled_transcript(text: str) -> ConversationSession | None:
    turns: list[ConversationTurn] = []
    for line in str(text or "").splitlines():
        if not line.strip():
            continue
        parsed = parse_speaker_labeled_line(line)
        if parsed is None:
            continue
        timestamp_text, speaker_label, utterance = parsed
        if not speaker_label or not utterance:
            continue
        speaker_role = conversation_speaker_role(speaker_label)
        turns.append(ConversationTurn(
            speaker_label=speaker_label,
            speaker_role=speaker_role,
            text=utterance,
            timestamp_text=timestamp_text,
            turn_index=len(turns),
        ))
    if len(turns) < 2:
        return None
    turns, participants, speaker_aliases, alias_applied_turns = apply_conversation_speaker_aliases(turns)
    return ConversationSession(
        turns=turns,
        participants=participants,
        speaker_aliases=speaker_aliases,
        alias_applied_turns=alias_applied_turns,
    )


def parse_speaker_labeled_line(line: str) -> tuple[str, str, str] | None:
    raw = str(line or "").strip()
    if not raw:
        return None
    timestamped = re.match(r"^\[([^\]]+)\]\s*\[([^\]]+)\]\s*(.+?)\s*$", raw)
    if timestamped:
        return timestamped.group(1).strip(), timestamped.group(2).strip(), timestamped.group(3).strip()
    bracketed = re.match(r"^\[([^\]]+)\]\s*(.+?)\s*$", raw)
    if bracketed:
        label = bracketed.group(1).strip()
        if looks_like_timestamp_label(label):
            return None
        return "", label, bracketed.group(2).strip()
    colon = re.match(r"^([\u4e00-\u9fffA-Za-z0-9_][\u4e00-\u9fffA-Za-z0-9_\- ]{0,30})\s*[:：]\s*(.+?)\s*$", raw)
    if colon:
        return "", colon.group(1).strip(), colon.group(2).strip()
    return None


def looks_like_timestamp_label(label: str) -> bool:
    text = str(label or "").strip()
    return bool(re.match(r"^\d{1,2}[:：]\d{2}(?::\d{2})?$", text))


def conversation_speaker_role(label: str) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"用户", "我", "本人", "佩戴者", "user", "me", "wearer"}:
        return "user"
    if (
        normalized.startswith("speaker_")
        or normalized.startswith("spk_")
        or normalized in {"unknown", "unknown_speaker", "未知", "未知说话人", "其他人"}
    ):
        return "unknown_speaker"
    return "known_person"


def apply_conversation_speaker_aliases(
    turns: list[ConversationTurn],
) -> tuple[list[ConversationTurn], dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    alias_map: dict[str, str] = {}
    speaker_aliases: list[dict[str, Any]] = []
    for turn in turns:
        if turn.speaker_role != "user":
            continue
        for source_label, target_label in conversation_aliases_from_text(turn.text):
            if not source_label or not target_label:
                continue
            if conversation_speaker_role(source_label) != "unknown_speaker":
                continue
            if conversation_speaker_role(target_label) != "known_person":
                continue
            alias_map[source_label] = target_label
            speaker_aliases.append({
                "source_label": source_label,
                "target_label": target_label,
                "alias_source": "user_named_speaker",
                "turn_index": turn.turn_index,
            })

    participants: dict[str, str] = {}
    alias_applied_turns: list[dict[str, Any]] = []
    normalized_turns: list[ConversationTurn] = []
    for turn in turns:
        original_role = turn.speaker_role
        original_label = turn.speaker_label
        participants.setdefault(original_label, original_role)
        alias_label = alias_map.get(original_label)
        if original_role == "unknown_speaker" and alias_label:
            participants[original_label] = "known_person"
            participants.setdefault(alias_label, "known_person")
            alias_applied_turns.append({
                "turn_index": turn.turn_index,
                "from": original_label,
                "to": alias_label,
            })
            normalized_turns.append(ConversationTurn(
                speaker_label=alias_label,
                speaker_role="known_person",
                text=turn.text,
                timestamp_text=turn.timestamp_text,
                turn_index=turn.turn_index,
            ))
            continue
        normalized_turns.append(turn)
        participants.setdefault(turn.speaker_label, turn.speaker_role)
    return normalized_turns, participants, speaker_aliases, alias_applied_turns


def conversation_aliases_from_text(text: str) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    raw = str(text or "")
    pattern = re.compile(
        r"(speaker_\d+|spk_\d+|unknown_speaker|unknown|未知说话人)"
        r"\s*(?:是|就是|叫)\s*"
        r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_\- ]{0,20})",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(raw):
        source_label = match.group(1).strip()
        target_label = match.group(2).strip(" ，,。.!！?？：:；;")
        if 1 < len(target_label) <= 20:
            aliases.append((source_label, target_label))
    return aliases
