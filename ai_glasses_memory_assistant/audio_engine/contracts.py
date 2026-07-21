from __future__ import annotations

import math
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

_AUDIO_EVENT_TYPES = {
    "session_state",
    "speech_start",
    "wake_detected",
    "transcript_partial",
    "transcript_final",
    "speech_rejected",
    "speaker_update",
    "error",
}
_AUDIO_EVENT_LANES = {"ambient", "assistant", "enrollment"}
_FINAL_EVENT_TYPES = {"transcript_final", "speech_rejected", "speaker_update"}
_AUDIO_RETENTION_VALUES = {
    "buffered_in_memory",
    "discarded_after_processing",
    "discarded_after_failure",
}
_MAX_EVENT_TEXT_CHARS = 32_000
_MAX_SPEAKER_EMBEDDING_VALUES = 4_096


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

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        private_payload: dict[str, Any] | None = None,
    ) -> "AudioEvent":
        if not isinstance(payload, dict):
            raise ValueError("audio event must be an object")
        if payload.get("schema_version") != "audio_event.v1":
            raise ValueError("unsupported audio event schema_version")
        event_type = _required_string(payload, "type")
        if event_type not in _AUDIO_EVENT_TYPES:
            raise ValueError("unsupported audio event type")
        lane = _required_string(payload, "lane")
        if lane not in _AUDIO_EVENT_LANES:
            raise ValueError("unsupported audio event lane")
        final = payload.get("final")
        if not isinstance(final, bool):
            raise ValueError("audio event final must be boolean")
        if event_type == "transcript_partial" and final:
            raise ValueError("partial transcript cannot be final")
        if event_type in _FINAL_EVENT_TYPES and not final:
            raise ValueError(f"{event_type} must be final")
        start_ms = _non_negative_int(payload, "start_ms")
        end_ms = _non_negative_int(payload, "end_ms")
        if end_ms < start_ms:
            raise ValueError("audio event end_ms cannot be before start_ms")
        text = str(payload.get("text") or "")
        if len(text) > _MAX_EVENT_TEXT_CHARS:
            raise ValueError("audio event text is too large")
        audio_retention = str(payload.get("audio_retention") or "buffered_in_memory")
        if audio_retention not in _AUDIO_RETENTION_VALUES:
            raise ValueError("unsupported audio retention state")
        private = private_payload if isinstance(private_payload, dict) else {}
        embedding = _speaker_embedding(private.get("speaker_embedding"))
        return cls(
            event_id=_required_string(payload, "event_id"),
            audio_session_id=_required_string(payload, "audio_session_id"),
            segment_id=_required_string(payload, "segment_id"),
            event_type=event_type,
            lane=lane,
            source_type=_required_string(payload, "source_type"),
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            final=final,
            vad=_object(payload, "vad"),
            wake=_object(payload, "wake"),
            asr=_object(payload, "asr"),
            speaker=_object(payload, "speaker"),
            overlap=_object(payload, "overlap"),
            audio_retention=audio_retention,
            speaker_embedding=embedding,
            speaker_embedding_model=str(private.get("speaker_embedding_model") or "").strip(),
        )

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


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"audio event {key} is required")
    return value.strip()


def _non_negative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"audio event {key} must be a non-negative integer")
    return value


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"audio event {key} must be an object")
    return dict(value)


def _speaker_embedding(value: Any) -> tuple[float, ...]:
    if value in (None, [], ()):
        return ()
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_SPEAKER_EMBEDDING_VALUES:
        raise ValueError("speaker embedding must be a bounded array")
    try:
        embedding = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("speaker embedding must contain numbers") from exc
    if not embedding or not all(math.isfinite(item) for item in embedding):
        raise ValueError("speaker embedding must contain finite numbers")
    return embedding
