from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AudioEventType = Literal[
    "session_state",
    "speech_start",
    "wake_detected",
    "transcript_partial",
    "transcript_final",
    "speech_rejected",
    "speaker_update",
    "error",
]


@dataclass(frozen=True)
class AudioEvent:
    event_id: str
    audio_session_id: str
    segment_id: str
    event_type: AudioEventType
    lane: str
    source_type: str
    start_ms: int = 0
    end_ms: int = 0
    text: str = ""
    final: bool = False
    vad: dict[str, Any] = field(default_factory=dict)
    wake: dict[str, Any] = field(default_factory=dict)
    asr: dict[str, Any] = field(default_factory=dict)
    speaker: dict[str, Any] = field(default_factory=dict)
    overlap: dict[str, Any] = field(default_factory=dict)
    audio_retention: str = "buffered_in_memory"
    speaker_embedding: tuple[float, ...] = field(default_factory=tuple, repr=False)
    speaker_embedding_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "audio_event.v1",
            "event_id": self.event_id,
            "audio_session_id": self.audio_session_id,
            "segment_id": self.segment_id,
            "type": self.event_type,
            "lane": self.lane,
            "source_type": self.source_type,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "final": self.final,
            "vad": dict(self.vad),
            "wake": dict(self.wake),
            "asr": dict(self.asr),
            "speaker": dict(self.speaker),
            "overlap": dict(self.overlap),
            "audio_retention": self.audio_retention,
        }


@dataclass(frozen=True)
class AudioEventPlan:
    action: Literal["ui_only", "chat", "capture", "enroll", "drop"]
    reason: str
    memory_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "memory_eligible": self.memory_eligible,
        }
