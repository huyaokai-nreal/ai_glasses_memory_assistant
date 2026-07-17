from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryWriteCandidate:
    content: str
    kind: str
    memory_type: str = ""
    privacy_level: str = "normal"
    confidence: float | None = None
    reason: str = ""
    source: str = ""
    source_id: str = ""
    ingestion_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    source_type: str = "chat"
    speaker_hint: str = ""
    do_not_remember_scope: str = ""
    subject_id: str = ""
    subject_type: str = "self"
    subject_name: str = ""
    subject_scope: str = ""
    audio_event_id: str = ""
    speaker_state: str = ""
    overlap_state: str = ""
    memory_eligible: bool = True


@dataclass(frozen=True)
class IntentDecision:
    needs_web_search: bool = False
    web_query: str | None = None
    web_reason: str = ""
    is_weather_query: bool = False
    weather_location_source: str = "none"
    weather_place_text: str = ""
    memory_write_candidates: list[MemoryWriteCandidate] = field(default_factory=list)
    confidence: float | None = None
    backend: str = "fallback"
    raw: str = ""
    error: str = ""

    def debug_payload(self) -> dict:
        payload: dict = {
            "backend": self.backend,
            "needs_web_search": self.needs_web_search,
            "web_query": self.web_query,
            "web_reason": self.web_reason,
            "is_weather_query": self.is_weather_query,
            "weather_location_source": self.weather_location_source,
            "weather_place_text": self.weather_place_text,
            "confidence": self.confidence,
            "memory_write_count": len(self.memory_write_candidates),
        }
        if self.raw:
            payload["raw"] = self.raw
        if self.error:
            payload["error"] = self.error
        return payload
