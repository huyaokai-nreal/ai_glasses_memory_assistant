from __future__ import annotations

import base64
import binascii
import secrets
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock, RLock
from typing import Any, Callable

import numpy as np

from .backends import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioBackendRegistry,
    cosine_similarity,
)
from .contracts import AudioEvent
from .settings import AudioEngineSettings, DEFAULT_AUDIO_SETTINGS


MAX_PUSH_BYTES = SAMPLE_RATE * 2 * 2


def _event_id() -> str:
    return f"aevt_{uuid.uuid4().hex[:16]}"


def _segment_id() -> str:
    return f"seg_{uuid.uuid4().hex[:12]}"


def _session_locked(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapped(self: "AudioSession", *args: Any, **kwargs: Any) -> Any:
        with self._state_lock:
            return method(self, *args, **kwargs)

    return wrapped


@dataclass
class AnonymousGroup:
    label: str
    centroid: tuple[float, ...]
    segment_count: int = 1
    duration_seconds: float = 0.0

    def add(self, embedding: tuple[float, ...], duration_seconds: float) -> None:
        previous = np.asarray(self.centroid, dtype=np.float32)
        current = np.asarray(embedding, dtype=np.float32)
        merged = previous * self.segment_count + current
        norm = float(np.linalg.norm(merged))
        if norm > 0.0:
            self.centroid = tuple(float(item) for item in merged / norm)
        self.segment_count += 1
        self.duration_seconds += duration_seconds


@dataclass
class AudioSession:
    session_id: str
    token: str
    user_id: str
    mode: str
    registry: AudioBackendRegistry
    settings: AudioEngineSettings = field(default_factory=AudioEngineSettings.from_env)
    clock: Callable[[], float] = time.monotonic
    capture_id: str = ""
    reference_embedding: tuple[float, ...] = ()
    user_threshold: float | None = None
    other_threshold: float | None = None
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    last_sequence: int = 0
    response_cache: OrderedDict[int, list[AudioEvent]] = field(default_factory=OrderedDict)
    closed: bool = False
    vad: Any = None
    kws: Any = None
    pre_roll: deque[np.ndarray] = field(default_factory=deque)
    pending_samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    active_samples: list[np.ndarray] = field(default_factory=list)
    speaker_window: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    speech_active: bool = False
    segment_start_sample: int = 0
    total_samples: int = 0
    segment_id: str = ""
    wake_active: bool = False
    wake_keyword: str = ""
    interaction_state: str = "ambient_listening"
    wake_query_deadline: float | None = None
    query_speaker_state: str = "unknown"
    query_speaker_reason: str = "not_checked"
    asr_cursor: int = 0
    asr_cache: dict[str, Any] = field(default_factory=dict)
    text_parts: list[str] = field(default_factory=list)
    anonymous_groups: list[AnonymousGroup] = field(default_factory=list)
    dispatch_count: int = 0
    playback_id: str = ""
    playback_deadline: float | None = None
    _state_lock: Any = field(default_factory=RLock, repr=False)

    def __post_init__(self) -> None:
        self.vad = self.vad or self.registry.vad.create()
        self.kws = self.kws or self.registry.kws.create()
        self.pre_roll = deque(maxlen=self.settings.pre_roll_frames)
        now = self.clock()
        self.created_at = now
        self.updated_at = now

    def public_payload(self) -> dict[str, Any]:
        return {
            "audio_session_id": self.session_id,
            "session_token": self.token,
            "user_id": self.user_id,
            "mode": self.mode,
            "capture_id": self.capture_id,
            "status": "stopped" if self.closed else "running",
            "format": {
                "encoding": "pcm_s16le",
                "sample_rate": self.settings.sample_rate,
                "channels": 1,
                "frame_samples": self.settings.frame_samples,
                "recommended_push_samples": self.settings.recommended_push_samples,
                "recommended_push_ms": round(
                    self.settings.recommended_push_samples / self.settings.sample_rate * 1000
                ),
            },
            "interaction_state": self.interaction_state,
            "playback": {"active": bool(self.playback_id)},
        }

    @_session_locked
    def push(self, *, sequence: int, pcm16_base64: str) -> tuple[list[AudioEvent], bool]:
        if self.closed:
            raise ValueError("audio session is stopped")
        if sequence in self.response_cache:
            return list(self.response_cache[sequence]), True
        expected = self.last_sequence + 1
        if sequence != expected:
            raise ValueError(f"audio sequence out of order: expected {expected}")
        try:
            raw = base64.b64decode(str(pcm16_base64 or ""), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("pcm16_base64 is invalid") from exc
        if not raw or len(raw) % 2:
            raise ValueError("pcm16 payload must contain whole 16-bit samples")
        if len(raw) > MAX_PUSH_BYTES:
            raise ValueError("pcm16 payload is too large")
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        events = self._expire_wake_query()
        events.extend(self._accept_samples(samples))
        self.last_sequence = sequence
        self.updated_at = self.clock()
        self.response_cache[sequence] = list(events)
        while len(self.response_cache) > self.settings.sequence_cache_limit:
            self.response_cache.popitem(last=False)
        return events, False

    @_session_locked
    def control(self, action: str, *, playback_id: str = "") -> list[AudioEvent]:
        normalized_action = str(action or "").strip()
        normalized_playback_id = str(playback_id or "").strip()
        now = self.clock()
        if normalized_action == "playback_started":
            if not normalized_playback_id:
                raise ValueError("playback_id is required")
            self.playback_id = normalized_playback_id
            self.playback_deadline = now + self.settings.playback_timeout_seconds
            self.updated_at = now
            return [self._session_event("playback_started")]
        if normalized_action == "playback_finished":
            if not normalized_playback_id:
                raise ValueError("playback_id is required")
            if self.playback_id != normalized_playback_id:
                return [self._session_event("stale_playback_finished")]
            self.playback_id = ""
            self.playback_deadline = None
            self.updated_at = now
            return [self._session_event("playback_finished")]
        if normalized_action != "wake_ack_finished":
            raise ValueError("unsupported audio control action")
        if self.interaction_state != "wake_acknowledging":
            raise ValueError("wake acknowledgement is not pending")
        self.playback_id = ""
        self.playback_deadline = None
        self._discard_segment(reset_vad=True)
        self.interaction_state = "awaiting_query"
        self.wake_query_deadline = now + self.settings.wake_query_start_timeout_seconds
        self.updated_at = now
        return [self._session_event("awaiting_query")]

    @_session_locked
    def stop(self) -> list[AudioEvent]:
        if self.closed:
            return []
        events: list[AudioEvent] = []
        if self.speech_active and self.pending_samples.size:
            self.active_samples.append(self.pending_samples.astype(np.float32, copy=False))
            self.total_samples += int(self.pending_samples.size)
        if self.speech_active and self.active_samples and self.interaction_state != "wake_acknowledging":
            events.extend(self._finalize_segment(reason="session_stop"))
        self.pending_samples = np.zeros(0, dtype=np.float32)
        self.active_samples = []
        self.closed = True
        self.playback_id = ""
        self.playback_deadline = None
        self.updated_at = self.clock()
        self._reset_kws()
        events.append(self._session_event("stopped"))
        return events

    @_session_locked
    def abort(self) -> None:
        self.pending_samples = np.zeros(0, dtype=np.float32)
        self.active_samples = []
        self.asr_cache = {}
        self.text_parts = []
        self.anonymous_groups = []
        self.pre_roll.clear()
        self.speaker_window = np.zeros(0, dtype=np.float32)
        self.speech_active = False
        self.segment_id = ""
        self.dispatch_count = 0
        self.playback_id = ""
        self.playback_deadline = None
        self.closed = True
        self.updated_at = self.clock()
        self._reset_kws()

    @_session_locked
    def expire_if_idle(self, cutoff: float) -> bool:
        now = self.clock()
        if self.playback_id and self.playback_deadline is not None and now < self.playback_deadline:
            return False
        if self.playback_id:
            self.playback_id = ""
            self.playback_deadline = None
        if self.closed or self.dispatch_count or self.updated_at >= cutoff:
            return False
        self.abort()
        return True

    @_session_locked
    def begin_dispatch(self) -> None:
        if self.closed:
            raise ValueError("audio session is stopped")
        self.dispatch_count += 1
        self.updated_at = self.clock()

    @_session_locked
    def end_dispatch(self) -> None:
        self.dispatch_count = max(0, self.dispatch_count - 1)
        self.updated_at = self.clock()

    def _accept_samples(self, samples: np.ndarray) -> list[AudioEvent]:
        self.pending_samples = np.concatenate([self.pending_samples, samples])
        events: list[AudioEvent] = []
        while self.pending_samples.size >= self.settings.frame_samples:
            frame = self.pending_samples[:self.settings.frame_samples]
            self.pending_samples = self.pending_samples[self.settings.frame_samples:]
            events.extend(self._accept_frame(frame))
            self.total_samples += self.settings.frame_samples
        return events

    def _accept_frame(self, frame: np.ndarray) -> list[AudioEvent]:
        events: list[AudioEvent] = []
        vad_transition = self.vad.accept(frame)
        active = bool(vad_transition)
        started = False
        if active and not self.speech_active:
            started = True
            self.speech_active = True
            self.pre_roll.append(np.asarray(frame, dtype=np.float32))
            pre_roll_samples = sum(item.size for item in self.pre_roll)
            self.segment_start_sample = max(0, self.total_samples + frame.size - pre_roll_samples)
            self.segment_id = _segment_id()
            self.active_samples = list(self.pre_roll)
            self.pre_roll.clear()
            if self.interaction_state == "awaiting_query":
                self.interaction_state = "query_speech"
                self.wake_query_deadline = None
                self.wake_active = True
            else:
                self.wake_active = False
            self.asr_cursor = 0
            self.asr_cache = {}
            self.text_parts = []
            self.speaker_window = np.zeros(0, dtype=np.float32)
            self.query_speaker_state = "unknown"
            self.query_speaker_reason = "not_checked"
            events.append(self._speech_event("speech_start", final=False, text=""))
        if active:
            if not started:
                self.active_samples.append(np.asarray(frame, dtype=np.float32))
            self._append_speaker_window(frame)
            if self.mode == "ambient" and self.interaction_state == "ambient_listening":
                keyword = str(self.kws.accept(frame) or "").strip()
                if keyword:
                    if self.registry.streaming_asr.capability().status == "ready":
                        self.interaction_state = "wake_acknowledging"
                        self.wake_keyword = keyword
                        self._reset_kws()
                        events.append(self._wake_event(keyword))
                    else:
                        events.append(self._error_event("streaming_asr_unavailable"))
                        self._reset_kws()
            if self.interaction_state == "query_speech":
                events.extend(self._maybe_partial())
            elif (
                self.interaction_state == "ambient_listening"
                and self._active_audio().size >= self.settings.ambient_segment_samples
            ):
                events.extend(self._finalize_segment(reason="max_segment_duration"))
            return events
        if self.speech_active:
            if self.interaction_state == "wake_acknowledging":
                self._discard_segment(reset_vad=False)
            else:
                events.extend(self._finalize_segment(reason="vad_inactive"))
        self.pre_roll.append(np.asarray(frame, dtype=np.float32))
        return events

    def _maybe_partial(self) -> list[AudioEvent]:
        audio = self._active_audio()
        events: list[AudioEvent] = []
        while audio.size - self.asr_cursor >= self.settings.partial_samples:
            chunk = audio[self.asr_cursor:self.asr_cursor + self.settings.partial_samples]
            delta = str(self.registry.streaming_asr.transcribe(chunk, cache=self.asr_cache, is_final=False) or "").strip()
            if delta:
                self.text_parts.append(delta)
            self.asr_cursor += self.settings.partial_samples
            text = "".join(self.text_parts).strip()
            speaker = self._query_speaker_metadata()
            self.query_speaker_state = str(speaker.get("state") or "unknown")
            self.query_speaker_reason = str(speaker.get("reason") or "speaker_not_checked")
            if text and self.query_speaker_state == "user":
                events.append(self._speech_event("transcript_partial", final=False, text=text))
        if self.asr_cursor:
            remaining = audio[self.asr_cursor:]
            self.active_samples = [remaining] if remaining.size else []
            self.asr_cursor = 0
        return events

    def _finalize_segment(self, *, reason: str) -> list[AudioEvent]:
        audio = self._active_audio()
        events: list[AudioEvent] = []
        speaker_audio = self.speaker_window if self.interaction_state == "query_speech" else audio
        speaker = self._speaker_metadata(speaker_audio)
        overlap = self._overlap_metadata(speaker_audio, speaker)
        if self.mode == "speaker_enroll":
            text = ""
            source_type = "speaker_enroll"
            lane = "enrollment"
            asr_backend = "not_used"
        elif self.interaction_state == "query_speech":
            tail = audio[self.asr_cursor:]
            delta = str(self.registry.streaming_asr.transcribe(tail, cache=self.asr_cache, is_final=True) or "").strip()
            if delta:
                self.text_parts.append(delta)
            text = "".join(self.text_parts).strip()
            asr_backend = "funasr_streaming"
            source_type = "wake_query"
            lane = "assistant"
        else:
            text = str(self.registry.offline_asr.transcribe(audio) or "").strip()
            source_type = "ambient_audio"
            lane = "ambient"
            asr_backend = "sensevoice"
        if self.mode == "speaker_enroll" and speaker.get("_embedding"):
            events.append(self._speech_event(
                "speaker_update",
                final=True,
                text="",
                lane=lane,
                source_type=source_type,
                speaker=speaker,
                overlap=overlap,
                reason=reason,
                asr_backend=asr_backend,
            ))
        elif lane == "assistant" and overlap.get("state") == "suspected":
            events.append(self._speech_event(
                "speech_rejected",
                final=True,
                text="",
                lane=lane,
                source_type=source_type,
                speaker=speaker,
                overlap=overlap,
                reason="assistant_overlap_suspected",
                asr_backend=asr_backend,
            ))
        elif lane == "assistant" and self.reference_embedding and speaker.get("state") != "user":
            events.append(self._speech_event(
                "speech_rejected",
                final=True,
                text="",
                lane=lane,
                source_type=source_type,
                speaker=speaker,
                overlap=overlap,
                reason="assistant_speaker_not_verified",
                asr_backend=asr_backend,
            ))
        elif not text:
            events.append(self._speech_event(
                "speech_rejected",
                final=True,
                text="",
                lane=lane,
                source_type=source_type,
                speaker=speaker,
                overlap=overlap,
                reason="empty_transcript_or_environment",
                asr_backend=asr_backend,
            ))
        else:
            events.append(self._speech_event(
                "transcript_final",
                final=True,
                text=text,
                lane=lane,
                source_type=source_type,
                speaker=speaker,
                overlap=overlap,
                reason=reason,
                asr_backend=asr_backend,
            ))
        was_query = self.interaction_state == "query_speech"
        self._reset_segment()
        if was_query:
            self.interaction_state = "ambient_listening"
            self.wake_keyword = ""
            self._reset_kws()
        return events

    def _query_speaker_metadata(self) -> dict[str, Any]:
        analysis = self.registry.speaker.analyze(self.speaker_window)
        embedding = tuple(analysis.embedding or ())
        similarity = cosine_similarity(embedding, self.reference_embedding)
        if not self.reference_embedding:
            return {"state": "unknown", "reason": "reference_unavailable"}
        if similarity is None:
            return {"state": "unknown", "reason": "speaker_embedding_unavailable"}
        if self.user_threshold is not None and similarity >= self.user_threshold:
            return {"state": "user", "reason": "calibrated_user_match", "similarity": similarity}
        if self.other_threshold is not None and similarity <= self.other_threshold:
            return {"state": "other", "reason": "calibrated_other_match", "similarity": similarity}
        return {"state": "unknown", "reason": "speaker_similarity_ambiguous", "similarity": similarity}

    def _append_speaker_window(self, frame: np.ndarray) -> None:
        self.speaker_window = np.concatenate([self.speaker_window, np.asarray(frame, dtype=np.float32)])
        if self.speaker_window.size > self.settings.speaker_window_samples:
            self.speaker_window = self.speaker_window[-self.settings.speaker_window_samples:]

    def _expire_wake_query(self) -> list[AudioEvent]:
        if (
            self.interaction_state != "awaiting_query"
            or self.wake_query_deadline is None
            or self.clock() < self.wake_query_deadline
        ):
            return []
        self.interaction_state = "ambient_listening"
        self.wake_query_deadline = None
        self.wake_keyword = ""
        self._reset_kws()
        return [self._session_event("wake_timeout")]

    def _discard_segment(self, *, reset_vad: bool) -> None:
        self._reset_segment()
        if reset_vad and hasattr(self.vad, "reset"):
            self.vad.reset()

    def _reset_segment(self) -> None:
        self.active_samples = []
        self.speaker_window = np.zeros(0, dtype=np.float32)
        self.speech_active = False
        self.segment_id = ""
        self.wake_active = False
        self.asr_cursor = 0
        self.asr_cache = {}
        self.text_parts = []

    def _reset_kws(self) -> None:
        if hasattr(self.kws, "reset"):
            self.kws.reset()

    def _speaker_metadata(self, audio: np.ndarray) -> dict[str, Any]:
        analysis = self.registry.speaker.analyze(audio)
        embedding = tuple(analysis.embedding or ())
        similarity = cosine_similarity(embedding, self.reference_embedding)
        if similarity is None:
            state = "unknown"
            reason = "reference_unavailable" if not self.reference_embedding else "speaker_embedding_unavailable"
        elif self.user_threshold is not None and similarity >= self.user_threshold:
            state = "user"
            reason = "calibrated_user_match"
        elif self.other_threshold is not None and similarity <= self.other_threshold:
            state = "other"
            reason = "calibrated_other_match"
        else:
            state = "unknown"
            reason = "speaker_similarity_ambiguous"
        group_label = ""
        persist_eligible = False
        duration = audio.size / SAMPLE_RATE
        if embedding:
            group = self._anonymous_group(embedding, duration)
            group_label = group.label
            persist_eligible = group.segment_count >= 2 or group.duration_seconds >= 3.0
        return {
            "state": state,
            "reason": reason,
            "similarity": round(similarity, 4) if similarity is not None else None,
            "model": analysis.model_name,
            "voice_group": group_label,
            "identity_reliable": False,
            "profile_persist_eligible": persist_eligible,
            "_embedding": embedding,
        }

    def _anonymous_group(self, embedding: tuple[float, ...], duration: float) -> AnonymousGroup:
        best: AnonymousGroup | None = None
        best_similarity = -1.0
        for group in self.anonymous_groups:
            similarity = cosine_similarity(embedding, group.centroid)
            if similarity is not None and similarity > best_similarity:
                best = group
                best_similarity = similarity
        if best is not None and best_similarity >= 0.72:
            best.add(embedding, duration)
            return best
        group = AnonymousGroup(
            label=f"PRED_SPK{len(self.anonymous_groups) + 1:04d}",
            centroid=embedding,
            duration_seconds=duration,
        )
        self.anonymous_groups.append(group)
        return group

    def _overlap_metadata(self, audio: np.ndarray, speaker: dict[str, Any]) -> dict[str, Any]:
        if not speaker.get("_embedding") or audio.size < SAMPLE_RATE * 2:
            return {"state": "unknown", "reason": "insufficient_speaker_evidence"}
        if audio.size < SAMPLE_RATE * 4:
            return {"state": "not_observed", "reason": "single_consistent_speaker_window"}
        window_embeddings: list[tuple[float, ...]] = []
        for start in range(0, audio.size, SAMPLE_RATE * 2):
            window = audio[start:start + SAMPLE_RATE * 2]
            if window.size < SAMPLE_RATE:
                continue
            analysis = self.registry.speaker.analyze(window)
            if analysis.embedding:
                window_embeddings.append(tuple(analysis.embedding))
        if len(window_embeddings) < 2:
            return {"state": "unknown", "reason": "insufficient_speaker_windows"}
        similarities = [
            cosine_similarity(window_embeddings[index - 1], window_embeddings[index])
            for index in range(1, len(window_embeddings))
        ]
        valid = [value for value in similarities if value is not None]
        if valid and min(valid) < 0.65:
            return {"state": "suspected", "reason": "speaker_window_conflict"}
        return {"state": "not_observed", "reason": "speaker_windows_consistent"}

    def _active_audio(self) -> np.ndarray:
        if not self.active_samples:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.active_samples).astype(np.float32, copy=False)

    def _speech_event(
        self,
        event_type: str,
        *,
        final: bool,
        text: str,
        lane: str | None = None,
        source_type: str | None = None,
        speaker: dict[str, Any] | None = None,
        overlap: dict[str, Any] | None = None,
        reason: str = "",
        asr_backend: str = "",
    ) -> AudioEvent:
        public_speaker = dict(speaker or {})
        embedding = tuple(public_speaker.pop("_embedding", ()) or ())
        start_ms = round(self.segment_start_sample / SAMPLE_RATE * 1000)
        end_ms = round((self.total_samples + FRAME_SAMPLES) / SAMPLE_RATE * 1000)
        return AudioEvent(
            event_id=_event_id(),
            audio_session_id=self.session_id,
            segment_id=self.segment_id,
            event_type=event_type,
            lane=lane or ("enrollment" if self.mode == "speaker_enroll" else "ambient"),
            source_type=source_type or ("speaker_enroll" if self.mode == "speaker_enroll" else "ambient_audio"),
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            final=final,
            vad={"state": "speech_end" if final else event_type, "backend": getattr(self.vad, "backend", "custom"), "reason": reason},
            wake={
                "detected": bool(self.wake_keyword),
                "keyword": self.wake_keyword,
                "backend": getattr(self.kws, "reason", "custom"),
                "phase": self.interaction_state,
            },
            asr={"backend": asr_backend or ("funasr_streaming" if (lane or "") == "assistant" or self.wake_active else "sensevoice")},
            speaker=public_speaker,
            overlap=dict(overlap or {"state": "unknown", "reason": "not_evaluated"}),
            audio_retention="discarded_after_processing" if final else "buffered_in_memory",
            speaker_embedding=embedding,
            speaker_embedding_model=str(public_speaker.get("model") or ""),
        )

    def _wake_event(self, keyword: str) -> AudioEvent:
        return AudioEvent(
            event_id=_event_id(),
            audio_session_id=self.session_id,
            segment_id=self.segment_id,
            event_type="wake_detected",
            lane="assistant",
            source_type="wake_query",
            start_ms=round(self.total_samples / SAMPLE_RATE * 1000),
            end_ms=round((self.total_samples + FRAME_SAMPLES) / SAMPLE_RATE * 1000),
            wake={
                "detected": True,
                "keyword": keyword,
                "backend": getattr(self.kws, "reason", "kws"),
                "phase": "wake_acknowledging",
                "ack_text": self.settings.wake_ack_text,
            },
        )

    def _error_event(self, reason: str) -> AudioEvent:
        return AudioEvent(
            event_id=_event_id(),
            audio_session_id=self.session_id,
            segment_id=self.segment_id,
            event_type="error",
            lane="assistant",
            source_type="wake_query",
            final=False,
            asr={"reason": reason},
            wake={"phase": self.interaction_state},
        )

    def _session_event(self, status: str) -> AudioEvent:
        return AudioEvent(
            event_id=_event_id(),
            audio_session_id=self.session_id,
            segment_id="",
            event_type="session_state",
            lane=self.mode,
            source_type=self.mode,
            final=False,
            vad={"state": status},
            wake={"phase": self.interaction_state},
            audio_retention="discarded_after_processing" if status == "stopped" else "buffered_in_memory",
        )


class AudioSessionManager:
    def __init__(
        self,
        *,
        registry: AudioBackendRegistry | None = None,
        clock: Callable[[], float] | None = None,
        settings: AudioEngineSettings | None = None,
        idle_seconds: float | None = None,
    ) -> None:
        self.settings = settings or AudioEngineSettings.from_env()
        self.registry = registry or AudioBackendRegistry(settings=self.settings)
        self.clock = clock or time.monotonic
        self.idle_seconds = (
            float(idle_seconds) if idle_seconds is not None else self.settings.idle_timeout_seconds
        )
        self._sessions: dict[str, AudioSession] = {}
        self._lock = Lock()

    def capabilities(self) -> dict[str, Any]:
        components = self.registry.capabilities()
        vad_usable = components["vad"]["status"] in {"ready", "degraded"}
        ambient_transcription_ready = vad_usable and components["utterance_asr"]["status"] == "ready"
        speaker_enrollment_ready = vad_usable and components["speaker"]["status"] == "ready"
        assistant_query_ready = (
            vad_usable
            and components["kws"]["status"] == "ready"
            and components["streaming_asr"]["status"] == "ready"
        )
        return {
            "schema_version": "audio_event.v1",
            "transport": "http_pcm_batches",
            "audio_input_ready": True,
            "format": {
                "encoding": "pcm_s16le",
                "sample_rate": self.settings.sample_rate,
                "channels": 1,
                "frame_samples": self.settings.frame_samples,
                "recommended_push_samples": self.settings.recommended_push_samples,
            },
            "components": components,
            "wake_policy": "kws_only",
            "ambient_transcription_ready": ambient_transcription_ready,
            "speaker_enrollment_ready": speaker_enrollment_ready,
            "assistant_query_ready": assistant_query_ready,
            "assistant_wake_ready": assistant_query_ready,
            "wake_query_start_timeout_seconds": self.settings.wake_query_start_timeout_seconds,
            "worklet_flush_timeout_seconds": self.settings.worklet_flush_timeout_seconds,
            "playback_timeout_seconds": self.settings.playback_timeout_seconds,
            "raw_audio_persistence": False,
            "partial_side_effects": False,
        }

    def start(
        self,
        *,
        user_id: str,
        mode: str,
        capture_id: str = "",
        reference_embedding: tuple[float, ...] = (),
        user_threshold: float | None = None,
        other_threshold: float | None = None,
    ) -> AudioSession:
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in {"ambient", "speaker_enroll"}:
            raise ValueError("audio mode must be ambient or speaker_enroll")
        session = AudioSession(
            session_id=f"audio_{uuid.uuid4().hex[:16]}",
            token=secrets.token_urlsafe(24),
            user_id=str(user_id or "").strip(),
            mode=normalized_mode,
            capture_id=str(capture_id or "").strip(),
            registry=self.registry,
            settings=self.settings,
            clock=self.clock,
            reference_embedding=tuple(reference_embedding or ()),
            user_threshold=user_threshold,
            other_threshold=other_threshold,
        )
        if not session.user_id:
            raise ValueError("user_id cannot be empty")
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, *, user_id: str, session_id: str, token: str) -> AudioSession:
        with self._lock:
            session = self._sessions.get(str(session_id or ""))
        if session is None or session.user_id != user_id or not secrets.compare_digest(session.token, str(token or "")):
            raise ValueError("audio session not found")
        return session

    def active_for_user(self, user_id: str) -> list[AudioSession]:
        normalized_user_id = str(user_id or "").strip()
        with self._lock:
            return [
                session
                for session in self._sessions.values()
                if session.user_id == normalized_user_id and not session.closed
            ]

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def expire_idle(self) -> list[AudioSession]:
        cutoff = self.clock() - self.idle_seconds
        with self._lock:
            candidates = list(self._sessions.values())
        expired = [session for session in candidates if session.expire_if_idle(cutoff)]
        with self._lock:
            for session in expired:
                if self._sessions.get(session.session_id) is session:
                    self._sessions.pop(session.session_id, None)
        return expired

    def close(self) -> list[AudioSession]:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.abort()
        return sessions
