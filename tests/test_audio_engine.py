from __future__ import annotations

import base64
import http.client
import json
import socket
import tempfile
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from ai_glasses_memory_assistant.audio_engine.backends import (
    AudioBackendRegistry,
    BackendCapability,
    KeywordSpotterSession,
    SpeakerAnalysis,
)
from ai_glasses_memory_assistant.audio_engine import AudioEvent
from ai_glasses_memory_assistant.audio_engine.runtime import AudioSessionManager
from ai_glasses_memory_assistant.audio_engine.settings import AudioEngineSettings
from ai_glasses_memory_assistant.memory_candidate import MemoryWriteCandidate
from ai_glasses_memory_assistant.intent_policy import should_write_memory_candidate
from ai_glasses_memory_assistant.server import GlassesHandler
from tests.helpers import CoreChatService, FakeAgent, isolated_app_home, pre_reply_write


class ScriptedVad:
    backend = "fake_vad"

    def __init__(self, states: list[bool]) -> None:
        self.states = deque(states)

    def accept(self, frame: np.ndarray) -> bool:
        return self.states.popleft() if self.states else False


class FakeVadFactory:
    def capability(self) -> BackendCapability:
        return BackendCapability("ready", "fake_vad", "test")

    def create(self) -> ScriptedVad:
        return ScriptedVad([])


class FakeStreamingAsr:
    def capability(self) -> BackendCapability:
        return BackendCapability("ready", "fake_streaming_asr", "test")

    def transcribe(self, audio: np.ndarray, *, cache: dict, is_final: bool) -> str:
        return " final" if is_final else "partial"


class UnavailableStreamingAsr(FakeStreamingAsr):
    def capability(self) -> BackendCapability:
        return BackendCapability("unavailable", "fake_streaming_asr", "test_unavailable")


class CommandStreamingAsr(FakeStreamingAsr):
    def transcribe(self, audio: np.ndarray, *, cache: dict, is_final: bool) -> str:
        return "记住明天提交材料" if is_final else ""


class FakeOfflineAsr:
    def capability(self) -> BackendCapability:
        return BackendCapability("ready", "fake_offline_asr", "test")

    def transcribe(self, audio: np.ndarray) -> str:
        return "ambient transcript"


class FakeSpeaker:
    def __init__(self, embedding: tuple[float, ...] = (1.0, 0.0, 0.0)) -> None:
        self.embedding = embedding

    def capability(self) -> BackendCapability:
        return BackendCapability("ready", "fake_speaker", "test")

    def analyze(self, audio: np.ndarray) -> SpeakerAnalysis:
        return SpeakerAnalysis(embedding=self.embedding, model_name="fake_speaker", reason="test")


class RecordingOfflineAsr(FakeOfflineAsr):
    def __init__(self) -> None:
        self.sample_counts: list[int] = []

    def transcribe(self, audio: np.ndarray) -> str:
        self.sample_counts.append(int(audio.size))
        return super().transcribe(audio)


class ScriptedSpeaker(FakeSpeaker):
    def __init__(self, embeddings: list[tuple[float, ...]]) -> None:
        super().__init__()
        self.embeddings = deque(embeddings)

    def analyze(self, audio: np.ndarray) -> SpeakerAnalysis:
        embedding = self.embeddings.popleft() if self.embeddings else self.embedding
        return SpeakerAnalysis(embedding=embedding, model_name="fake_speaker", reason="test")


class FakeKwsSession:
    reason = "fake_kws"

    def __init__(self, keyword: str = "") -> None:
        self.keyword = keyword
        self.reset_count = 0

    def accept(self, audio: np.ndarray) -> str:
        keyword, self.keyword = self.keyword, ""
        return keyword

    def reset(self) -> None:
        self.reset_count += 1


class FakeKwsFactory:
    def __init__(self, keyword: str = "") -> None:
        self.keyword = keyword

    def capability(self) -> BackendCapability:
        return BackendCapability("ready", "fake_kws", "test")

    def create(self) -> FakeKwsSession:
        return FakeKwsSession(self.keyword)


class UnavailableOfflineAsr(FakeOfflineAsr):
    def capability(self) -> BackendCapability:
        return BackendCapability("unavailable", "fake_offline_asr", "model_dir_missing")


class UnavailableSpeaker(FakeSpeaker):
    def capability(self) -> BackendCapability:
        return BackendCapability("unavailable", "fake_speaker", "model_dir_missing")


class UnavailableKwsFactory(FakeKwsFactory):
    def capability(self) -> BackendCapability:
        return BackendCapability("degraded", "fake_kws", "model_or_keywords_missing")


class UnavailableVadFactory(FakeVadFactory):
    def capability(self) -> BackendCapability:
        return BackendCapability("unavailable", "fake_vad", "vad_unavailable")


class DegradedVadFactory(FakeVadFactory):
    def capability(self) -> BackendCapability:
        return BackendCapability("degraded", "fake_vad", "energy_fallback")


def fake_registry(
    *,
    keyword: str = "",
    speaker_embedding: tuple[float, ...] = (1.0, 0.0, 0.0),
) -> AudioBackendRegistry:
    return AudioBackendRegistry(
        streaming_asr=FakeStreamingAsr(),
        offline_asr=FakeOfflineAsr(),
        speaker=FakeSpeaker(speaker_embedding),
        kws=FakeKwsFactory(keyword),
        vad=FakeVadFactory(),
    )


def unavailable_streaming_registry() -> AudioBackendRegistry:
    registry = fake_registry()
    registry.streaming_asr = UnavailableStreamingAsr()
    return registry


def pcm_frames(count: int) -> str:
    samples = np.full(512 * count, 1000, dtype="<i2")
    return base64.b64encode(samples.tobytes()).decode("ascii")


def assistant_final_event(session_id: str, event_id: str, text: str = "语音问题") -> AudioEvent:
    return AudioEvent(
        event_id=event_id,
        audio_session_id=session_id,
        segment_id=f"segment-{event_id}",
        event_type="transcript_final",
        lane="assistant",
        source_type="wake_query",
        text=text,
        final=True,
        speaker={"state": "user"},
        overlap={"state": "not_observed"},
        audio_retention="discarded_after_processing",
    )


def test_audio_event_v1_strict_round_trip_and_private_embedding() -> None:
    original = assistant_final_event("session-1", "event-1", "明天提交材料")
    parsed = AudioEvent.from_dict(
        original.to_dict(),
        private_payload={
            "speaker_embedding": [1.0, 0.0],
            "speaker_embedding_model": "android-speaker-v1",
        },
    )

    assert parsed.to_dict() == original.to_dict()
    assert parsed.speaker_embedding == (1.0, 0.0)
    assert parsed.speaker_embedding_model == "android-speaker-v1"
    assert "speaker_embedding" not in parsed.to_dict()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": "audio_event.v2"}, "schema_version"),
        ({"type": "transcript_partial", "final": True}, "cannot be final"),
        ({"type": "transcript_final", "final": False}, "must be final"),
        ({"start_ms": 20, "end_ms": 10}, "cannot be before"),
        ({"speaker": []}, "speaker must be an object"),
    ],
)
def test_audio_event_v1_rejects_invalid_native_payload(change: dict, message: str) -> None:
    payload = assistant_final_event("session-1", "event-1").to_dict()
    payload.update(change)

    with pytest.raises(ValueError, match=message):
        AudioEvent.from_dict(payload)


def test_audio_event_v1_rejects_non_finite_private_embedding() -> None:
    payload = assistant_final_event("session-1", "event-1").to_dict()

    with pytest.raises(ValueError, match="finite"):
        AudioEvent.from_dict(payload, private_payload={"speaker_embedding": [float("nan")]})


def test_device_speaker_classification_reuses_persisted_profile_thresholds() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        assert service.classify_device_speaker(
            user_id="u1", embedding=[1.0, 0.0], model_name="android-speaker-v1"
        )["reason"] == "reference_unavailable"
        assert service.classify_device_speaker(
            user_id="u1", embedding=[], model_name="android-speaker-v1"
        )["reason"] == "reference_unavailable"
        service.timeline_store.upsert_speaker_profile(
            user_id="u1",
            embedding=[1.0, 0.0],
            model_name="android-speaker-v1",
            user_threshold=0.8,
            other_threshold=0.2,
        )

        matched = service.classify_device_speaker(
            user_id="u1", embedding=[1.0, 0.0], model_name="android-speaker-v1"
        )
        rejected = service.classify_device_speaker(
            user_id="u1", embedding=[0.0, 1.0], model_name="android-speaker-v1"
        )
        mismatch = service.classify_device_speaker(
            user_id="u1", embedding=[1.0, 0.0], model_name="other-model"
        )
        unavailable = service.classify_device_speaker(
            user_id="u1", embedding=[], model_name="android-speaker-v1"
        )

        assert matched["state"] == "user"
        assert rejected["state"] == "other"
        assert unavailable["reason"] == "speaker_embedding_unavailable"
        assert mismatch == {
            "state": "unknown",
            "reason": "speaker_model_mismatch",
            "model": "other-model",
        }
        service.close()


def test_device_audio_events_share_final_dispatch_and_queue_offline_chat() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.set_device_network_state(online=False)
        capture = service.start_device_capture(user_id="u1")
        ambient = AudioEvent(
            event_id="device-ambient-1",
            audio_session_id="device-session-1",
            segment_id="device-segment-1",
            event_type="transcript_final",
            lane="ambient",
            source_type="ambient_audio",
            start_ms=0,
            end_ms=1000,
            text="明天下午提交材料",
            final=True,
            speaker={"state": "user"},
            overlap={"state": "not_observed"},
            audio_retention="discarded_after_processing",
        )
        queued_ambient = service.ingest_device_audio_event(
            user_id="u1",
            capture_id=capture["capture_id"],
            event_payload=ambient.to_dict(),
        )
        completed_ambient = service.wait_device_audio_event(
            user_id="u1", event_id=ambient.event_id, timeout=2.0
        )

        assert queued_ambient["created"] is True
        assert completed_ambient["status"] == "completed"
        persisted_capture = service.timeline_store.get_capture("u1", capture["capture_id"])
        assert [chunk["text"] for chunk in persisted_capture["chunks"]] == ["明天下午提交材料"]
        capture_status = service.device_capture_status(user_id="u1", capture_id=capture["capture_id"])
        assert capture_status == {
            "capture_id": capture["capture_id"],
            "status": "running",
            "chunk_count": 1,
            "last_chunk_id": persisted_capture["chunks"][0]["chunk_id"],
            "last_segment_id": "device-segment-1",
            "last_captured_at": persisted_capture["chunks"][0]["timestamp"],
        }
        assert "text" not in capture_status
        assert "metadata" not in capture_status

        duplicate = service.ingest_device_audio_event(
            user_id="u1",
            capture_id=capture["capture_id"],
            event_payload=ambient.to_dict(),
        )
        assert duplicate["created"] is False
        assert len(service.timeline_store.get_capture("u1", capture["capture_id"])["chunks"]) == 1

        partial_payload = ambient.to_dict()
        partial_payload.update({
            "event_id": "device-partial-1",
            "type": "transcript_partial",
            "final": False,
        })
        partial = service.ingest_device_audio_event(
            user_id="u1",
            capture_id=capture["capture_id"],
            event_payload=partial_payload,
        )
        assert partial["queued"] is False
        assert partial["dispatch"]["action"] == "ui_only"

        query = assistant_final_event("device-session-1", "device-query-1", "我明天要做什么")
        pending_query = service.ingest_device_audio_event(
            user_id="u1",
            capture_id=capture["capture_id"],
            event_payload=query.to_dict(),
        )
        assert pending_query["status"] == "pending"
        assert service.wait_device_audio_event(
            user_id="u1", event_id=query.event_id, timeout=0.05
        )["status"] == "pending"

        service.set_device_network_state(online=True)
        completed_query = service.wait_device_audio_event(
            user_id="u1", event_id=query.event_id, timeout=2.0
        )
        assert completed_query["status"] == "completed"
        assert completed_query["dispatch"]["action"] == "chat"
        assert completed_query["dispatch"]["result"]["reply"] == "主回复"
        service.close()


def test_device_speaker_enrollment_aggregates_three_private_samples_once() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        session_id = "android-enrollment-1"
        embeddings = ([1.0, 0.0], [0.999, 0.02], [0.998, -0.02])
        completed = []

        for index, embedding in enumerate(embeddings, start=1):
            event = AudioEvent(
                event_id=f"device-enrollment-{index}",
                audio_session_id=session_id,
                segment_id=f"enrollment-segment-{index}",
                event_type="speaker_update",
                lane="enrollment",
                source_type="speaker_enrollment",
                start_ms=0,
                end_ms=2_000,
                final=True,
                speaker={"state": "enrollment", "model": "android-speaker-v1"},
                overlap={"state": "unknown", "reason": "enrollment_not_evaluated"},
                audio_retention="discarded_after_processing",
            )
            queued = service.ingest_device_audio_event(
                user_id="u1",
                event_payload=event.to_dict(),
                private_payload={
                    "speaker_embedding": embedding,
                    "speaker_embedding_model": "android-speaker-v1",
                    "enrollment_session_id": session_id,
                    "sample_index": index,
                    "sample_total": 3,
                },
            )
            assert queued["created"] is True
            completed.append(service.wait_device_audio_event(
                user_id="u1", event_id=event.event_id, timeout=2.0
            ))

        assert [item["dispatch"]["result"]["status"] for item in completed] == [
            "pending",
            "pending",
            "ok",
        ]
        profile = service.get_speaker_profile(user_id="u1")
        assert profile["enrolled"] is True
        assert profile["sample_count"] == 3
        profile_version = profile["speaker_profile_version"]

        duplicate = service.ingest_device_audio_event(
            user_id="u1",
            event_payload=AudioEvent(
                event_id="device-enrollment-3",
                audio_session_id=session_id,
                segment_id="enrollment-segment-3",
                event_type="speaker_update",
                lane="enrollment",
                source_type="speaker_enrollment",
                end_ms=2_000,
                final=True,
                speaker={"state": "enrollment"},
                overlap={"state": "unknown"},
                audio_retention="discarded_after_processing",
            ).to_dict(),
            private_payload={
                "speaker_embedding": embeddings[-1],
                "speaker_embedding_model": "android-speaker-v1",
                "enrollment_session_id": session_id,
                "sample_index": 3,
                "sample_total": 3,
            },
        )
        assert duplicate["created"] is False
        assert service.get_speaker_profile(user_id="u1")["speaker_profile_version"] == profile_version

        public = {
            "completed": completed,
            "queue": service.device_audio_event_queue(user_id="u1"),
            "audit": service.read_audit_records(user_id="u1", limit=20),
        }
        serialized = json.dumps(public, ensure_ascii=False)
        assert "speaker_embedding" not in serialized
        assert "[1.0, 0.0]" not in serialized
        service.close()


def stream_ambient_capture(
    service: CoreChatService,
    *,
    user_id: str,
    vad_states: list[bool],
    push_frame_counts: list[int],
) -> tuple[dict, list[dict], dict]:
    started = service.start_audio_session(user_id=user_id, mode="ambient")
    session = service.audio_sessions.get(
        user_id=user_id,
        session_id=started["audio_session_id"],
        token=started["session_token"],
    )
    session.vad = ScriptedVad(vad_states)
    pushes = [
        service.push_audio_session(
            user_id=user_id,
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=sequence,
            pcm16_base64=pcm_frames(frame_count),
        )
        for sequence, frame_count in enumerate(push_frame_counts, start=1)
    ]
    stopped = service.stop_audio_session(
        user_id=user_id,
        audio_session_id=session.session_id,
        session_token=session.token,
    )
    return started, pushes, stopped


def attach_fake_audio(service: CoreChatService, *, keyword: str = "") -> None:
    service.audio_sessions = AudioSessionManager(registry=fake_registry(keyword=keyword), clock=service._clock)


def test_service_loads_app_dotenv_before_creating_audio_backends() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        home = Path(tmpdir)
        asr_dir = home / "sensevoice"
        speaker_dir = home / "campp"
        asr_dir.mkdir()
        speaker_dir.mkdir()
        (home / ".env").write_text(
            "\n".join([
                f"AI_GLASSES_ASR_MODEL_DIR={asr_dir}",
                f"AI_GLASSES_SPEAKER_MODEL_DIR={speaker_dir}",
            ]),
            encoding="utf-8",
        )
        with isolated_app_home(tmpdir, clear=True):
            service = CoreChatService(tmpdir)
            registry = service.audio_sessions.registry
            assert registry.offline_asr.model_dir == str(asr_dir)
            assert registry.speaker.model_dir == str(speaker_dir)
            assert registry.offline_asr.capability().reason != "model_dir_missing"
            assert registry.speaker.capability().reason != "model_dir_missing"
            serialized_capabilities = json.dumps(service.audio_capabilities(), ensure_ascii=False)
            assert str(home) not in serialized_capabilities
            assert str(asr_dir) not in serialized_capabilities
            assert str(speaker_dir) not in serialized_capabilities
            service.close()


def test_service_uses_one_registry_for_streaming_legacy_and_speaker_models() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        registry = service.audio_sessions.registry

        assert not hasattr(service, "audio_processor")
        assert registry.offline_asr._runner is registry.offline_adapter.asr_runner
        assert registry.speaker._runner is registry.offline_adapter.speaker_runner
        service.close()


def test_actionable_audio_capabilities_follow_required_backends() -> None:
    ready = AudioSessionManager(registry=fake_registry()).capabilities()
    assert ready["audio_input_ready"] is True
    assert ready["ambient_transcription_ready"] is True
    assert ready["speaker_enrollment_ready"] is True
    assert ready["assistant_query_ready"] is True
    assert ready["assistant_wake_ready"] is True

    degraded_vad = fake_registry()
    degraded_vad.vad = DegradedVadFactory()
    degraded = AudioSessionManager(registry=degraded_vad).capabilities()
    assert degraded["audio_input_ready"] is True
    assert degraded["ambient_transcription_ready"] is False
    assert degraded["speaker_enrollment_ready"] is False
    assert degraded["assistant_query_ready"] is False

    no_offline_asr = fake_registry()
    no_offline_asr.offline_asr = UnavailableOfflineAsr()
    offline_missing = AudioSessionManager(registry=no_offline_asr).capabilities()
    assert offline_missing["ambient_transcription_ready"] is False
    assert offline_missing["speaker_enrollment_ready"] is True
    assert offline_missing["assistant_query_ready"] is True

    no_streaming_asr = fake_registry()
    no_streaming_asr.streaming_asr = UnavailableStreamingAsr()
    streaming_missing = AudioSessionManager(registry=no_streaming_asr).capabilities()
    assert streaming_missing["ambient_transcription_ready"] is True
    assert streaming_missing["speaker_enrollment_ready"] is True
    assert streaming_missing["assistant_query_ready"] is False

    no_speaker = fake_registry()
    no_speaker.speaker = UnavailableSpeaker()
    speaker_missing = AudioSessionManager(registry=no_speaker).capabilities()
    assert speaker_missing["ambient_transcription_ready"] is True
    assert speaker_missing["speaker_enrollment_ready"] is False
    assert speaker_missing["assistant_query_ready"] is True

    no_kws = fake_registry()
    no_kws.kws = UnavailableKwsFactory()
    assert AudioSessionManager(registry=no_kws).capabilities()["assistant_query_ready"] is False

    no_vad = fake_registry()
    no_vad.vad = UnavailableVadFactory()
    vad_missing = AudioSessionManager(registry=no_vad).capabilities()
    assert vad_missing["ambient_transcription_ready"] is False
    assert vad_missing["speaker_enrollment_ready"] is False
    assert vad_missing["assistant_query_ready"] is False


def test_service_rejects_continuous_audio_start_when_vad_is_degraded() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        registry = fake_registry()
        registry.vad = DegradedVadFactory()
        service.audio_sessions = AudioSessionManager(registry=registry, clock=service._clock)

        with pytest.raises(ValueError, match="ambient audio is unavailable"):
            service.start_audio_session(user_id="u1", mode="ambient")
        with pytest.raises(ValueError, match="speaker enrollment is unavailable"):
            service.start_audio_session(user_id="u1", mode="speaker_enroll")

        assert service._captures == {}
        assert service.audio_sessions.active_for_user("u1") == []
        service.close()


def test_partial_has_no_side_effect_and_duplicate_final_is_consumed_once() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service, keyword="hermes")
        service.chat = Mock(return_value={"reply": "voice reply"})
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.reference_embedding = (1.0, 0.0, 0.0)
        session.user_threshold = 0.8
        session.other_threshold = 0.2
        session.vad = ScriptedVad([True])

        wake = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=1,
            pcm16_base64=pcm_frames(1),
        )
        assert [event["type"] for event in wake["events"]].count("wake_detected") == 1
        service.control_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            action="wake_ack_finished",
        )
        session.vad = ScriptedVad([True] * 16 + [False])

        partial = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=2,
            pcm16_base64=pcm_frames(16),
        )
        assert any(event["type"] == "transcript_partial" for event in partial["events"])
        assert all("result" not in dispatch for dispatch in partial["dispatches"])
        service.chat.assert_not_called()

        final = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=3,
            pcm16_base64=pcm_frames(1),
        )
        duplicate = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=3,
            pcm16_base64=pcm_frames(1),
        )

        assert any(event["type"] == "transcript_final" for event in final["events"])
        assert duplicate["duplicate"] is True
        assert duplicate["dispatches"] == final["dispatches"]
        chat_job = next(dispatch["job"] for dispatch in final["dispatches"] if dispatch["action"] == "chat")
        assert "result" not in chat_job
        completed = service.wait_audio_dispatch_job(user_id="u1", job_id=chat_job["job_id"])
        assert completed["status"] == "completed"
        assert completed["result"]["reply"] == "voice reply"
        service.chat.assert_called_once()


def test_long_running_push_bounds_response_and_dispatch_caches() -> None:
    settings = AudioEngineSettings(sequence_cache_limit=3)
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.audio_sessions = AudioSessionManager(
            registry=fake_registry(),
            settings=settings,
            clock=service._clock,
        )
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.vad = ScriptedVad([False] * 6)

        latest = None
        for sequence in range(1, 7):
            latest = service.push_audio_session(
                user_id="u1",
                audio_session_id=session.session_id,
                session_token=session.token,
                sequence=sequence,
                pcm16_base64=pcm_frames(1),
            )

        assert list(session.response_cache) == [4, 5, 6]
        assert [
            sequence
            for audio_session_id, sequence in service._audio_dispatch_cache
            if audio_session_id == session.session_id
        ] == [4, 5, 6]

        duplicate = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=6,
            pcm16_base64=pcm_frames(1),
        )
        assert duplicate["duplicate"] is True
        assert duplicate["dispatches"] == latest["dispatches"]

        with pytest.raises(ValueError, match="out of order"):
            service.push_audio_session(
                user_id="u1",
                audio_session_id=session.session_id,
                session_token=session.token,
                sequence=3,
                pcm16_base64=pcm_frames(1),
            )
        assert list(session.response_cache) == [4, 5, 6]


def test_pcm_push_uses_centralized_raw_byte_limit() -> None:
    settings = AudioEngineSettings(
        pcm_push_max_bytes=1024,
        streaming_request_max_bytes=8192,
    )
    session = AudioSessionManager(registry=fake_registry(), settings=settings).start(
        user_id="u1",
        mode="ambient",
    )
    session.vad = ScriptedVad([False])
    session.push(sequence=1, pcm16_base64=pcm_frames(1))
    with pytest.raises(ValueError, match="too large"):
        session.push(sequence=2, pcm16_base64=pcm_frames(2))


def test_completed_audio_dispatch_jobs_are_bounded_per_session() -> None:
    settings = AudioEngineSettings(sequence_cache_limit=3)
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.audio_sessions = AudioSessionManager(
            registry=fake_registry(),
            settings=settings,
            clock=service._clock,
        )
        service.chat = Mock(return_value={"reply": "voice reply"})
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        job_ids = []
        for index in range(6):
            job = service._start_audio_chat_dispatch_job(
                session=session,
                event=assistant_final_event(session.session_id, f"bounded-{index}"),
                memory_eligible=True,
            )
            job_ids.append(job["job_id"])
            completed = service.wait_audio_dispatch_job(user_id="u1", job_id=job["job_id"])
            assert completed["status"] == "completed"

        retained = service._audio_dispatch_jobs_for_session(session.session_id)
        assert [job["job_id"] for job in retained] == job_ids[-3:]
        assert service.read_audio_dispatch_job(user_id="u1", job_id=job_ids[0]) is None


def test_concurrent_duplicate_final_reuses_job_without_blocking_pcm_push() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service, keyword="hermes")
        chat_started = threading.Event()
        release_chat = threading.Event()

        def chat_result(*_args, **_kwargs) -> dict[str, str]:
            chat_started.set()
            assert release_chat.wait(timeout=1)
            return {"reply": "voice reply"}

        service.chat = Mock(side_effect=chat_result)
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.vad = ScriptedVad([True])
        service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=1,
            pcm16_base64=pcm_frames(1),
        )
        service.control_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            action="wake_ack_finished",
        )
        session.vad = ScriptedVad([True, False])
        first = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=2,
            pcm16_base64=pcm_frames(2),
        )
        assert chat_started.wait(timeout=1)
        session.updated_at = 0.0
        service._expire_audio_sessions()
        assert session.closed is False
        later_push = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=3,
            pcm16_base64=pcm_frames(1),
        )
        second = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=2,
            pcm16_base64=pcm_frames(2),
        )

        assert first["duplicate"] is False
        assert later_push["accepted_sequence"] == 3
        assert second["duplicate"] is True
        assert first["dispatches"] == second["dispatches"]
        chat_job = first["dispatches"][-1]["job"]
        assert chat_job["status"] in {"pending", "running"}
        assert service.read_audio_dispatch_job(user_id="u2", job_id=chat_job["job_id"]) is None

        release_chat.set()
        completed = service.wait_audio_dispatch_job(user_id="u1", job_id=chat_job["job_id"])
        assert completed["status"] == "completed"
        assert completed["result"]["reply"] == "voice reply"
        service.chat.assert_called_once()


def test_audio_dispatch_failure_and_interrupted_stop_have_terminal_states() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        first_started = threading.Event()
        release_first = threading.Event()

        def chat_result(message: str, **_kwargs) -> dict[str, str]:
            if message == "first":
                first_started.set()
                assert release_first.wait(timeout=1)
                return {"reply": "first reply"}
            raise RuntimeError("model unavailable")

        service.chat = Mock(side_effect=chat_result)
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        first = service._start_audio_chat_dispatch_job(
            session=session,
            event=assistant_final_event(session.session_id, "event-1", "first"),
            memory_eligible=True,
        )
        assert first_started.wait(timeout=1)
        pending = service._start_audio_chat_dispatch_job(
            session=session,
            event=assistant_final_event(session.session_id, "event-2", "second"),
            memory_eligible=True,
        )

        stopped = service.stop_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            interrupted=True,
        )
        pending_state = service.read_audio_dispatch_job(user_id="u1", job_id=pending["job_id"])
        assert stopped["status"] == "interrupted"
        assert pending_state["status"] == "cancelled"
        assert pending_state["reason"] == "audio_session_interrupted"

        release_first.set()
        first_state = service.wait_audio_dispatch_job(user_id="u1", job_id=first["job_id"])
        service.wait_audio_dispatch_job(user_id="u1", job_id=pending["job_id"])
        assert first_state["status"] == "completed"
        assert service.chat.call_count == 1

        second_session = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=second_session["audio_session_id"],
            token=second_session["session_token"],
        )
        failed = service._start_audio_chat_dispatch_job(
            session=session,
            event=assistant_final_event(session.session_id, "event-3", "fail"),
            memory_eligible=True,
        )
        failed_state = service.wait_audio_dispatch_job(user_id="u1", job_id=failed["job_id"])
        assert failed_state["status"] == "failed"
        assert failed_state["error_type"] == "RuntimeError"
        assert "result" not in failed_state


def test_service_close_marks_running_audio_dispatch_interrupted() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        chat_started = threading.Event()
        release_chat = threading.Event()

        def chat_result(*_args, **_kwargs) -> dict[str, str]:
            chat_started.set()
            assert release_chat.wait(timeout=1)
            return {"reply": "late reply"}

        service.chat = Mock(side_effect=chat_result)
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        job = service._start_audio_chat_dispatch_job(
            session=session,
            event=assistant_final_event(session.session_id, "event-close"),
            memory_eligible=True,
        )
        assert chat_started.wait(timeout=1)

        service.close(timeout=0)
        state_after_close = service.read_audio_dispatch_job(user_id="u1", job_id=job["job_id"])
        assert state_after_close["status"] == "interrupted"
        assert state_after_close["reason"] == "service_close_timeout"
        release_chat.set()
        service.wait_audio_dispatch_job(user_id="u1", job_id=job["job_id"])
        assert service.read_audio_dispatch_job(user_id="u1", job_id=job["job_id"])["status"] == "interrupted"


def test_session_stop_tail_final_can_start_audio_dispatch_job() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        service.chat = Mock(return_value={"reply": "tail reply"})
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.reference_embedding = (1.0, 0.0, 0.0)
        session.user_threshold = 0.8
        session.other_threshold = 0.2
        session.interaction_state = "query_speech"
        session.speech_active = True
        session.segment_id = "tail-segment"
        session.active_samples = [np.full(512, 0.1, dtype=np.float32)]
        session.speaker_window = np.full(512, 0.1, dtype=np.float32)

        stopped = service.stop_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
        )
        job = next(dispatch["job"] for dispatch in stopped["dispatches"] if dispatch["action"] == "chat")
        completed = service.wait_audio_dispatch_job(user_id="u1", job_id=job["job_id"])
        assert completed["status"] == "completed"
        assert completed["result"]["reply"] == "tail reply"
        service.chat.assert_called_once()


def test_audio_sequence_and_user_session_are_isolated() -> None:
    manager = AudioSessionManager(registry=fake_registry())
    first = manager.start(user_id="u1", mode="ambient")
    second = manager.start(user_id="u2", mode="ambient")

    with pytest.raises(ValueError, match="out of order"):
        first.push(sequence=2, pcm16_base64=pcm_frames(1))
    with pytest.raises(ValueError, match="not found"):
        manager.get(user_id="u2", session_id=first.session_id, token=first.token)
    assert manager.get(user_id="u2", session_id=second.session_id, token=second.token) is second


def test_same_user_ambient_takeover_interrupts_old_capture_without_memory_job() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        first = service.start_audio_session(user_id="u1", mode="ambient")
        service.append_capture_chunk(
            user_id="u1",
            capture_id=first["capture_id"],
            text="已有环境文本",
            metadata={"source_type": "ambient_audio"},
        )

        second = service.start_audio_session(user_id="u1", mode="ambient")
        old_capture = service.timeline_store.get_capture("u1", first["capture_id"])
        active = service.audio_sessions.active_for_user("u1")

        assert second["replaced_audio_session_ids"] == [first["audio_session_id"]]
        assert [session.session_id for session in active] == [second["audio_session_id"]]
        assert old_capture is not None and old_capture["status"] == "interrupted"
        assert service._memory_jobs == {}
        with pytest.raises(ValueError, match="not found"):
            service.push_audio_session(
                user_id="u1",
                audio_session_id=first["audio_session_id"],
                session_token=first["session_token"],
                sequence=1,
                pcm16_base64=pcm_frames(1),
            )
        with pytest.raises(ValueError, match="not found"):
            service.stop_audio_session(
                user_id="u1",
                audio_session_id=first["audio_session_id"],
                session_token=first["session_token"],
            )


def test_concurrent_same_user_starts_leave_one_active_session() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        barrier = threading.Barrier(3)
        results: list[dict] = []

        def start_from_tab() -> None:
            barrier.wait()
            results.append(service.start_audio_session(user_id="u1", mode="ambient"))

        first = threading.Thread(target=start_from_tab)
        second = threading.Thread(target=start_from_tab)
        first.start()
        second.start()
        barrier.wait()
        first.join(timeout=1)
        second.join(timeout=1)

        assert len(results) == 2
        active = service.audio_sessions.active_for_user("u1")
        assert len(active) == 1
        assert active[0].session_id in {result["audio_session_id"] for result in results}
        assert sum(bool(result["replaced_audio_session_ids"]) for result in results) == 1
        captures = [service.timeline_store.get_capture("u1", result["capture_id"]) for result in results]
        assert sorted(capture["status"] for capture in captures if capture is not None) == ["interrupted", "running"]
        assert service._memory_jobs == {}


def test_takeover_waits_for_current_final_dispatch_before_interrupting_capture() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        first = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=first["audio_session_id"],
            token=first["session_token"],
        )
        session.vad = ScriptedVad([True, False])
        dispatch_started = threading.Event()
        release_dispatch = threading.Event()
        original_consume = service._consume_audio_events

        def slow_consume(*, session, events):
            dispatch_started.set()
            assert release_dispatch.wait(timeout=1)
            return original_consume(session=session, events=events)

        service._consume_audio_events = slow_consume
        push_results: list[dict] = []
        start_results: list[dict] = []
        push_thread = threading.Thread(target=lambda: push_results.append(service.push_audio_session(
            user_id="u1",
            audio_session_id=first["audio_session_id"],
            session_token=first["session_token"],
            sequence=1,
            pcm16_base64=pcm_frames(2),
        )))
        push_thread.start()
        assert dispatch_started.wait(timeout=1)
        start_thread = threading.Thread(
            target=lambda: start_results.append(service.start_audio_session(user_id="u1", mode="ambient"))
        )
        start_thread.start()
        time.sleep(0.02)
        assert start_thread.is_alive()

        release_dispatch.set()
        push_thread.join(timeout=1)
        start_thread.join(timeout=1)

        assert any(dispatch["action"] == "capture" for dispatch in push_results[0]["dispatches"])
        assert start_results[0]["replaced_audio_session_ids"] == [first["audio_session_id"]]
        old_capture = service.timeline_store.get_capture("u1", first["capture_id"])
        assert old_capture is not None and old_capture["status"] == "interrupted"
        assert len(old_capture["chunks"]) == 1
        assert service._memory_jobs == {}


def test_ambient_and_enrollment_are_exclusive_per_user_but_users_are_isolated() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        u1_ambient = service.start_audio_session(user_id="u1", mode="ambient")
        u2_ambient = service.start_audio_session(user_id="u2", mode="ambient")

        enrollment = service.start_audio_session(user_id="u1", mode="speaker_enroll")
        assert enrollment["replaced_audio_session_ids"] == [u1_ambient["audio_session_id"]]
        assert [(session.user_id, session.mode) for session in service.audio_sessions.active_for_user("u1")] == [
            ("u1", "speaker_enroll"),
        ]
        assert [session.session_id for session in service.audio_sessions.active_for_user("u2")] == [
            u2_ambient["audio_session_id"],
        ]

        resumed = service.start_audio_session(user_id="u1", mode="ambient")
        assert resumed["replaced_audio_session_ids"] == [enrollment["audio_session_id"]]
        assert service.audio_sessions.active_for_user("u1")[0].mode == "ambient"
        with pytest.raises(ValueError, match="audio mode"):
            service.start_audio_session(user_id="u1", mode="unsupported")
        assert service.audio_sessions.active_for_user("u1")[0].session_id == resumed["audio_session_id"]


def test_vad_activation_preserves_pre_roll_without_duplicating_silence_frames() -> None:
    registry = fake_registry()
    recorder = RecordingOfflineAsr()
    registry.offline_asr = recorder
    manager = AudioSessionManager(registry=registry)
    session = manager.start(user_id="u1", mode="ambient")
    session.vad = ScriptedVad([False, False, True, False])

    events, _ = session.push(sequence=1, pcm16_base64=pcm_frames(4))

    assert any(event.event_type == "transcript_final" for event in events)
    assert recorder.sample_counts == [3 * 512]


def test_wake_ack_timeout_resets_to_ambient_without_query() -> None:
    now = [0.0]
    settings = AudioEngineSettings(wake_query_start_timeout_seconds=10.0)
    manager = AudioSessionManager(
        registry=fake_registry(keyword="hermes"),
        settings=settings,
        clock=lambda: now[0],
    )
    session = manager.start(user_id="u1", mode="ambient")
    session.vad = ScriptedVad([True])
    wake_events, _ = session.push(sequence=1, pcm16_base64=pcm_frames(1))
    assert sum(event.event_type == "wake_detected" for event in wake_events) == 1

    session.control("wake_ack_finished")
    now[0] = 10.1
    session.vad = ScriptedVad([False])
    timeout_events, _ = session.push(sequence=2, pcm16_base64=pcm_frames(1))

    assert session.interaction_state == "ambient_listening"
    assert [event.vad["state"] for event in timeout_events] == ["wake_timeout"]
    assert session.kws.reset_count >= 2


def test_query_without_reference_hides_partial_but_keeps_final_text() -> None:
    manager = AudioSessionManager(registry=fake_registry(keyword="hermes"))
    session = manager.start(user_id="u1", mode="ambient")
    session.vad = ScriptedVad([True])
    session.push(sequence=1, pcm16_base64=pcm_frames(1))
    session.control("wake_ack_finished")
    session.vad = ScriptedVad([True] * 16 + [False])

    events, _ = session.push(sequence=2, pcm16_base64=pcm_frames(17))

    assert not any(event.event_type == "transcript_partial" for event in events)
    final = next(event for event in events if event.event_type == "transcript_final")
    assert final.text == "partialfinal"
    assert final.speaker["state"] == "unknown"


def test_enrolled_other_speaker_produces_textless_rejection() -> None:
    manager = AudioSessionManager(
        registry=fake_registry(keyword="hermes", speaker_embedding=(0.0, 1.0, 0.0))
    )
    session = manager.start(
        user_id="u1",
        mode="ambient",
        reference_embedding=(1.0, 0.0, 0.0),
        user_threshold=0.8,
        other_threshold=0.2,
    )
    session.vad = ScriptedVad([True])
    session.push(sequence=1, pcm16_base64=pcm_frames(1))
    session.control("wake_ack_finished")
    session.vad = ScriptedVad([True, False])

    events, _ = session.push(sequence=2, pcm16_base64=pcm_frames(2))

    rejected = next(event for event in events if event.event_type == "speech_rejected")
    assert rejected.text == ""
    assert rejected.speaker["state"] == "other"


def test_suspected_overlap_in_query_produces_textless_rejection() -> None:
    settings = AudioEngineSettings(partial_samples=100_000, speaker_window_samples=64_000)
    registry = fake_registry(keyword="hermes")
    registry.speaker = ScriptedSpeaker([
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ])
    manager = AudioSessionManager(registry=registry, settings=settings)
    session = manager.start(
        user_id="u1",
        mode="ambient",
        reference_embedding=(1.0, 0.0, 0.0),
        user_threshold=0.8,
        other_threshold=0.2,
    )
    session.vad = ScriptedVad([True])
    session.push(sequence=1, pcm16_base64=pcm_frames(1))
    session.control("wake_ack_finished")
    session.vad = ScriptedVad([True] * 125 + [False])

    sequence = 2
    for frame_count in (60, 60, 5):
        events, _ = session.push(sequence=sequence, pcm16_base64=pcm_frames(frame_count))
        assert not any(event.final for event in events)
        sequence += 1
    events, _ = session.push(sequence=sequence, pcm16_base64=pcm_frames(1))

    rejected = next(event for event in events if event.event_type == "speech_rejected")
    assert rejected.text == ""
    assert rejected.overlap["state"] == "suspected"


def test_verified_wake_final_passes_audio_context_through_main_memory_gate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=FakeAgent(pre_reply=pre_reply_write("明天提交材料")),
        )
        registry = fake_registry(keyword="hermes")
        registry.streaming_asr = CommandStreamingAsr()
        service.audio_sessions = AudioSessionManager(registry=registry, clock=service._clock)
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.reference_embedding = (1.0, 0.0, 0.0)
        session.user_threshold = 0.8
        session.other_threshold = 0.2
        session.vad = ScriptedVad([True])
        service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=1,
            pcm16_base64=pcm_frames(1),
        )
        service.control_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            action="wake_ack_finished",
        )
        session.vad = ScriptedVad([True] * 63 + [False])
        service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=2,
            pcm16_base64=pcm_frames(60),
        )
        service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=3,
            pcm16_base64=pcm_frames(3),
        )
        final = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=4,
            pcm16_base64=pcm_frames(1),
        )

        chat_dispatch = next(item for item in final["dispatches"] if item["action"] == "chat")
        chat_job = service.wait_audio_dispatch_job(user_id="u1", job_id=chat_dispatch["job"]["job_id"])
        event = next(item for item in final["events"] if item["type"] == "transcript_final")
        assert event["speaker"]["state"] == "user"
        assert event["overlap"]["state"] == "not_observed"
        assert chat_job["result"]["debug"]["audio_event"] == {
            "event_id": event["event_id"],
            "speaker_state": "user",
            "overlap_state": "not_observed",
        }
        assert [memory.content for memory in service.memory_store.list_memories("u1")] == ["明天提交材料"]
        serialized = json.dumps(final, ensure_ascii=False)
        assert "speaker_embedding" not in serialized
        assert "[1.0, 0.0, 0.0]" not in serialized


def test_query_longer_than_thirty_seconds_keeps_bounded_pcm_until_vad_final() -> None:
    manager = AudioSessionManager(registry=fake_registry(keyword="hermes"))
    session = manager.start(
        user_id="u1",
        mode="ambient",
        reference_embedding=(1.0, 0.0, 0.0),
        user_threshold=0.8,
        other_threshold=0.2,
    )
    session.vad = ScriptedVad([True])
    session.push(sequence=1, pcm16_base64=pcm_frames(1))
    session.control("wake_ack_finished")

    speech_frames = 969
    session.vad = ScriptedVad([True] * speech_frames + [False])
    sequence = 2
    remaining = speech_frames
    final_events = []
    while remaining:
        frame_count = min(60, remaining)
        events, _ = session.push(sequence=sequence, pcm16_base64=pcm_frames(frame_count))
        assert not any(event.event_type == "transcript_final" for event in events)
        sequence += 1
        remaining -= frame_count
    assert session.interaction_state == "query_speech"
    assert session._active_audio().size < session.settings.partial_samples
    assert session.speaker_window.size <= session.settings.speaker_window_samples

    final_events, _ = session.push(sequence=sequence, pcm16_base64=pcm_frames(1))
    assert sum(event.event_type == "transcript_final" for event in final_events) == 1


def test_ambient_still_transcribes_when_streaming_query_asr_is_unavailable() -> None:
    registry = unavailable_streaming_registry()
    registry.kws = FakeKwsFactory("hermes")
    manager = AudioSessionManager(registry=registry)
    session = manager.start(user_id="u1", mode="ambient")
    session.vad = ScriptedVad([True, False])

    events, _ = session.push(sequence=1, pcm16_base64=pcm_frames(2))
    final = next(event for event in events if event.event_type == "transcript_final")

    assert final.text == "ambient transcript"
    assert final.asr["backend"] == "sensevoice"
    assert any(event.event_type == "error" for event in events)
    assert manager.capabilities()["assistant_wake_ready"] is False
    assert manager.capabilities()["assistant_query_ready"] is False
    assert manager.capabilities()["ambient_transcription_ready"] is True


def test_ambient_final_appends_capture_and_interrupted_stop_skips_memory_job() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.vad = ScriptedVad([True, False])

        result = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=1,
            pcm16_base64=pcm_frames(2),
        )
        capture = service.timeline_store.get_capture("u1", started["capture_id"])

        assert any(dispatch["action"] == "capture" for dispatch in result["dispatches"])
        assert capture is not None and [chunk["text"] for chunk in capture["chunks"]] == ["ambient transcript"]
        assert capture["chunks"][0]["metadata"]["memory_eligible"] is False

        stopped = service.stop_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            interrupted=True,
        )
        assert stopped["capture"]["status"] == "interrupted"
        assert stopped["capture"]["memory_processing"]["reason"] == "audio_session_interrupted"
        assert service.memory_store.list_memories("u1") == []


def test_pause_for_enrollment_flushes_tail_without_creating_memory_job() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.vad = ScriptedVad([True, False])
        service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=1,
            pcm16_base64=pcm_frames(2),
        )
        session.vad = ScriptedVad([True])
        service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=2,
            pcm16_base64=pcm_frames(1),
        )

        paused = service.stop_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            stop_reason="pause_for_enrollment",
        )
        capture = service.timeline_store.get_capture("u1", started["capture_id"])

        assert paused["status"] == "paused"
        assert paused["stop_reason"] == "pause_for_enrollment"
        assert paused["capture"] == {
            "capture_id": started["capture_id"],
            "status": "paused",
            "chunk_count": 2,
            "memory_processing": {"status": "not_needed", "reason": "pause_for_enrollment"},
        }
        assert capture is not None and capture["status"] == "paused"
        assert [chunk["text"] for chunk in capture["chunks"]] == ["ambient transcript", "ambient transcript"]
        assert session.active_samples == []
        assert session.pending_samples.size == 0
        assert service._memory_jobs == {}
        assert service.memory_store.list_memories("u1") == []


def test_pause_for_enrollment_rejects_non_ambient_session_without_stopping_it() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        started = service.start_audio_session(user_id="u1", mode="speaker_enroll")

        with pytest.raises(ValueError, match="requires an ambient session"):
            service.stop_audio_session(
                user_id="u1",
                audio_session_id=started["audio_session_id"],
                session_token=started["session_token"],
                stop_reason="pause_for_enrollment",
            )
        active = service.audio_sessions.active_for_user("u1")
        assert [session.session_id for session in active] == [started["audio_session_id"]]


def test_ambient_start_failure_marks_created_capture_interrupted() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        service.audio_sessions.start = Mock(side_effect=RuntimeError("session init failed"))

        with pytest.raises(RuntimeError, match="session init failed"):
            service.start_audio_session(user_id="u1", mode="ambient")

        assert len(service._captures) == 1
        capture = next(iter(service._captures.values()))
        assert capture["status"] == "interrupted"


def test_interrupted_stop_discards_active_speech_without_final_dispatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        service.chat = Mock(return_value={"reply": "unexpected"})
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.vad = ScriptedVad([True])
        service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=1,
            pcm16_base64=pcm_frames(1),
        )

        stopped = service.stop_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            interrupted=True,
        )

        assert stopped["events"] == []
        assert stopped["dispatches"] == []
        assert session.active_samples == []
        service.chat.assert_not_called()


def test_idle_timeout_aborts_pcm_and_marks_ambient_capture_interrupted() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.active_samples = [np.ones(512, dtype=np.float32)]
        session.updated_at = 0.0

        service._expire_audio_sessions()
        capture = service.timeline_store.get_capture("u1", started["capture_id"])

        assert session.closed is True
        assert session.active_samples == []
        assert capture is not None and capture["status"] == "interrupted"


def test_playback_state_prevents_false_idle_expiry_and_eventually_cleans_up() -> None:
    now = [0.0]
    settings = AudioEngineSettings(
        idle_timeout_seconds=30.0,
        playback_timeout_seconds=120.0,
    )
    manager = AudioSessionManager(
        registry=fake_registry(),
        settings=settings,
        clock=lambda: now[0],
    )
    session = manager.start(user_id="u1", mode="ambient")

    started = session.control("playback_started", playback_id="playback-1")
    now[0] = 31.0
    assert manager.expire_idle() == []
    assert session.closed is False
    assert started[0].vad["state"] == "playback_started"
    assert session.public_payload()["playback"] == {"active": True}

    session.control("playback_started", playback_id="playback-2")
    stale = session.control("playback_finished", playback_id="playback-1")
    assert stale[0].vad["state"] == "stale_playback_finished"
    assert session.public_payload()["playback"] == {"active": True}

    now[0] = 152.0
    assert manager.expire_idle() == [session]
    assert session.closed is True
    assert session.public_payload()["playback"] == {"active": False}


def test_matching_playback_finish_restores_normal_idle_timeout() -> None:
    now = [0.0]
    manager = AudioSessionManager(
        registry=fake_registry(),
        settings=AudioEngineSettings(idle_timeout_seconds=30.0, playback_timeout_seconds=120.0),
        clock=lambda: now[0],
    )
    session = manager.start(user_id="u1", mode="ambient")
    session.control("playback_started", playback_id="playback-1")
    now[0] = 45.0
    finished = session.control("playback_finished", playback_id="playback-1")
    assert finished[0].vad["state"] == "playback_finished"
    assert manager.expire_idle() == []

    now[0] = 60.0
    duplicate = session.control("playback_finished", playback_id="playback-1")
    assert duplicate[0].vad["state"] == "stale_playback_finished"
    now[0] = 76.0
    assert manager.expire_idle() == [session]
    assert session.closed is True


def test_background_reaper_expires_idle_session_without_another_request() -> None:
    settings = AudioEngineSettings(
        idle_timeout_seconds=0.05,
        reaper_interval_seconds=0.01,
    )
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.audio_sessions = AudioSessionManager(registry=fake_registry(), settings=settings)
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.active_samples = [np.ones(512, dtype=np.float32)]
        service.start_audio_reaper()
        deadline = time.monotonic() + 1.0
        while not session.closed and time.monotonic() < deadline:
            time.sleep(0.01)

        capture = service.timeline_store.get_capture("u1", started["capture_id"])
        assert session.closed is True
        assert session.active_samples == []
        assert capture is not None and capture["status"] == "interrupted"
        service.close()


def test_service_close_interrupts_active_capture_and_clears_session_state() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        started = service.start_audio_session(user_id="u1", mode="ambient")
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.active_samples = [np.ones(512, dtype=np.float32)]
        service.start_audio_reaper()

        service.close()

        assert session.closed is True
        assert session.active_samples == []
        assert service._audio_reaper_thread is None
        assert service.audio_sessions._sessions == {}
        assert service._audio_session_metadata == {}
        assert service._captures[started["capture_id"]]["status"] == "interrupted"


def test_kws_sessions_share_serialized_inference_but_keep_separate_streams() -> None:
    class Stream:
        def __init__(self) -> None:
            self.ready = True

        def accept_waveform(self, _sample_rate: int, _audio: np.ndarray) -> None:
            self.ready = True

    class Model:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        def is_ready(self, stream: Stream) -> bool:
            return stream.ready

        def decode_stream(self, stream: Stream) -> None:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            stream.ready = False
            self.active -= 1

        def get_result(self, _stream: Stream) -> str:
            return "hermes"

        def reset_stream(self, stream: Stream) -> None:
            stream.ready = True

    model = Model()
    inference_lock = threading.Lock()
    sessions = [
        KeywordSpotterSession(model, Stream(), lock=inference_lock, reason="test")
        for _ in range(2)
    ]

    for _round in range(2):
        results: list[str] = []
        threads = [
            threading.Thread(
                target=lambda session=session: results.append(
                    session.accept(np.ones(512, dtype=np.float32))
                )
            )
            for session in sessions
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1)
        assert sorted(results) == ["hermes", "hermes"]
    assert model.max_active == 1


def test_speaker_enrollment_uses_internal_embedding_without_exposing_it() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        started = service.start_audio_session(user_id="u1", mode="speaker_enroll", sample_total=1)
        session = service.audio_sessions.get(
            user_id="u1",
            session_id=started["audio_session_id"],
            token=started["session_token"],
        )
        session.vad = ScriptedVad([True, False])

        result = service.push_audio_session(
            user_id="u1",
            audio_session_id=session.session_id,
            session_token=session.token,
            sequence=1,
            pcm16_base64=pcm_frames(2),
        )

        assert result["dispatches"][-1]["action"] == "enroll"
        assert result["dispatches"][-1]["result"]["enrolled"] is True
        serialized = json.dumps(result, ensure_ascii=False)
        assert "speaker_embedding" not in serialized
        assert "[1.0, 0.0, 0.0]" not in serialized


def test_anonymous_voice_group_metadata_is_user_scoped_and_deletable() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        anonymous = service.memory_store.create_provisional_subject("u1", "PRED_SPK0001", source_scope="capture-1")
        named = service.memory_store.create_named_subject("u1", "张三")
        for source_id in ("chunk-1", "chunk-2"):
            service.memory_store.store_voice_profile(
                "u1",
                anonymous.id,
                embedding=[1.0, 0.0],
                embedding_model="fake_speaker",
                source_id=source_id,
            )
        service.memory_store.store_voice_profile(
            "u1",
            named.id,
            embedding=[0.0, 1.0],
            embedding_model="fake_speaker",
            source_id="named-chunk",
        )

        payload = service.list_anonymous_voice_groups(user_id="u1")
        assert len(payload["groups"]) == 1
        assert payload["groups"][0]["group_id"] == anonymous.id
        assert payload["groups"][0]["profile_count"] == 2
        serialized = json.dumps(payload, ensure_ascii=False)
        assert "speaker_embedding" not in serialized
        assert "[1.0, 0.0]" not in serialized
        assert service.list_anonymous_voice_groups(user_id="u2")["groups"] == []

        deleted = service.delete_anonymous_voice_group(user_id="u1", group_id=anonymous.id)
        assert deleted["deleted_profile_count"] == 2
        assert service.memory_store.list_voice_profiles("u1", subject_ids=[anonymous.id]) == []
        assert len(service.memory_store.list_voice_profiles("u1", subject_ids=[named.id])) == 1


def test_streaming_anonymous_voice_matches_across_captures_and_stays_private() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)

        _, single_pushes, single_stop = stream_ambient_capture(
            service,
            user_id="single-user",
            vad_states=[True, False],
            push_frame_counts=[2],
        )
        single_final = next(
            event
            for payload in single_pushes
            for event in payload["events"]
            if event["type"] == "transcript_final"
        )
        assert single_final["speaker"]["profile_persist_eligible"] is False
        assert service.memory_store.list_voice_profiles("single-user") == []
        assert single_stop["capture"]["status"] == "stopped"

        _, first_pushes, first_stop = stream_ambient_capture(
            service,
            user_id="u1",
            vad_states=[True, False, True, False],
            push_frame_counts=[4],
        )
        first_finals = [
            event
            for payload in first_pushes
            for event in payload["events"]
            if event["type"] == "transcript_final"
        ]
        assert [event["speaker"]["profile_persist_eligible"] for event in first_finals] == [False, True]
        first_identities = first_stop["capture"]["import_result"]["conversation_session"]["voice_identity"]
        first_subject_id = first_identities[0]["subject_id"]
        assert {identity["subject_id"] for identity in first_identities} == {first_subject_id}
        assert first_identities[0]["profile_stored"] is False
        assert first_identities[1]["profile_stored"] is True
        first_subject = service.memory_store.get_subject("u1", first_subject_id)
        assert first_subject is not None
        assert first_subject.subject_type == "provisional"
        assert first_subject.display_name == "PRED_SPK0001"
        assert len(service.memory_store.list_voice_profiles("u1", subject_ids=[first_subject_id])) == 1

        _, long_pushes, long_stop = stream_ambient_capture(
            service,
            user_id="long-user",
            vad_states=[True] * 94 + [False],
            push_frame_counts=[60, 35],
        )
        long_final = next(
            event
            for payload in long_pushes
            for event in payload["events"]
            if event["type"] == "transcript_final"
        )
        assert long_final["speaker"]["profile_persist_eligible"] is True
        assert long_stop["capture"]["chunk_count"] == 1
        assert len(service.memory_store.list_voice_profiles("long-user")) == 1

        _, second_pushes, second_stop = stream_ambient_capture(
            service,
            user_id="u1",
            vad_states=[True, False],
            push_frame_counts=[2],
        )
        second_identity = second_stop["capture"]["import_result"]["conversation_session"]["voice_identity"][0]
        assert second_identity["decision"] == "matched"
        assert second_identity["reason"] == "voice_profile_matched"
        assert second_identity["subject_id"] == first_subject_id
        assert second_identity["subject_type"] == "provisional"
        assert second_identity["subject_name"] == "PRED_SPK0001"
        assert len(service.memory_store.list_voice_profiles("u1")) == 1

        _, u2_pushes, u2_stop = stream_ambient_capture(
            service,
            user_id="u2",
            vad_states=[True, False, True, False],
            push_frame_counts=[4],
        )
        u2_identities = u2_stop["capture"]["import_result"]["conversation_session"]["voice_identity"]
        u2_subject_id = u2_identities[0]["subject_id"]
        assert u2_subject_id != first_subject_id
        assert {identity["subject_id"] for identity in u2_identities} == {u2_subject_id}
        assert service.memory_store.get_subject("u2", u2_subject_id).subject_type == "provisional"
        assert len(service.memory_store.list_voice_profiles("u2")) == 1

        u1_groups = service.list_anonymous_voice_groups(user_id="u1")
        u2_groups = service.list_anonymous_voice_groups(user_id="u2")
        assert [(group["label"], group["identity_reliable"]) for group in u1_groups["groups"]] == [
            ("PRED_SPK0001", False),
        ]
        assert len(u2_groups["groups"]) == 1
        assert all(
            subject.subject_type != "named"
            for subject in service.memory_store.list_subjects("u1")
            if subject.display_name.startswith("PRED_SPK")
        )

        public_payload = {
            "first_pushes": first_pushes,
            "first_stop": first_stop,
            "second_pushes": second_pushes,
            "second_stop": second_stop,
            "u2_pushes": u2_pushes,
            "u2_stop": u2_stop,
            "u1_groups": u1_groups,
            "audit": service.read_audit_records(user_id="u1", limit=100),
        }
        serialized = json.dumps(public_payload, ensure_ascii=False)
        assert '"speaker_embedding"' not in serialized
        assert '"embedding":' not in serialized
        assert "[1.0, 0.0, 0.0]" not in serialized

        deleted = service.delete_anonymous_voice_group(user_id="u1", group_id=first_subject_id)
        assert deleted == {
            "deleted": True,
            "group_id": first_subject_id,
            "deleted_profile_count": 1,
        }
        assert service.list_anonymous_voice_groups(user_id="u1")["groups"] == []
        assert len(service.list_anonymous_voice_groups(user_id="u2")["groups"]) == 1
        assert len(service.memory_store.list_voice_profiles("u2")) == 1


def test_stdlib_http_audio_routes_preserve_session_contract() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        attach_fake_audio(service)
        GlassesHandler.service = service
        server = ThreadingHTTPServer(("127.0.0.1", 0), GlassesHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("GET", "/api/audio/capabilities")
            response = connection.getresponse()
            capabilities = json.loads(response.read())
            assert response.status == 200
            assert capabilities["schema_version"] == "audio_event.v1"
            assert capabilities["playback_timeout_seconds"] == 300.0
            assert capabilities["audio_input_ready"] is True
            assert capabilities["ambient_transcription_ready"] is True
            assert capabilities["speaker_enrollment_ready"] is True
            assert capabilities["assistant_query_ready"] is True

            connection.request(
                "POST",
                "/api/audio/session/start",
                body=json.dumps({"user_id": "u1", "mode": "ambient"}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            started = json.loads(response.read())
            assert response.status == 200

            push_body = json.dumps({
                "user_id": "u1",
                "audio_session_id": started["audio_session_id"],
                "session_token": started["session_token"],
                "sequence": 1,
                "pcm16_base64": pcm_frames(1),
            })
            for duplicate in (False, True):
                connection.request(
                    "POST",
                    "/api/audio/session/push",
                    body=push_body,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                pushed = json.loads(response.read())
                assert response.status == 200
                assert pushed["duplicate"] is duplicate

            session = service.audio_sessions.get(
                user_id="u1",
                session_id=started["audio_session_id"],
                token=started["session_token"],
            )
            service.chat = Mock(return_value={"reply": "voice reply"})
            job = service._start_audio_chat_dispatch_job(
                session=session,
                event=assistant_final_event(session.session_id, "http-event"),
                memory_eligible=True,
            )
            service.wait_audio_dispatch_job(user_id="u1", job_id=job["job_id"])
            connection.request(
                "GET",
                f"/api/audio/dispatch/jobs?user_id=u1&job_id={job['job_id']}",
            )
            response = connection.getresponse()
            queried_job = json.loads(response.read())["job"]
            assert response.status == 200
            assert queried_job["status"] == "completed"
            assert queried_job["result"]["reply"] == "voice reply"

            for action, expected_state in (
                ("playback_started", "playback_started"),
                ("playback_finished", "playback_finished"),
            ):
                connection.request(
                    "POST",
                    "/api/audio/session/control",
                    body=json.dumps({
                        "user_id": "u1",
                        "audio_session_id": started["audio_session_id"],
                        "session_token": started["session_token"],
                        "action": action,
                        "playback_id": "http-playback",
                    }),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                controlled = json.loads(response.read())
                assert response.status == 200
                assert controlled["events"][0]["vad"]["state"] == expected_state

            connection.request(
                "POST",
                "/api/audio/session/stop",
                body=json.dumps({
                    "user_id": "u1",
                    "audio_session_id": started["audio_session_id"],
                    "session_token": started["session_token"],
                    "interrupted": True,
                }),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            stopped = json.loads(response.read())
            assert response.status == 200
            assert stopped["status"] == "interrupted"
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            GlassesHandler.service = None


def test_http_audio_body_limits_reject_before_unbounded_reads() -> None:
    settings = AudioEngineSettings(
        pcm_push_max_bytes=1024,
        streaming_request_max_bytes=8192,
        legacy_audio_request_max_bytes=1024,
        speaker_enrollment_request_max_bytes=1024,
        request_body_read_timeout_seconds=0.1,
    )
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.audio_sessions = AudioSessionManager(
            registry=fake_registry(),
            settings=settings,
            clock=service._clock,
        )
        GlassesHandler.service = service
        server = ThreadingHTTPServer(("127.0.0.1", 0), GlassesHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(
            path: str,
            *,
            content_length: str | None,
            body: bytes = b"",
            transfer_encoding: str = "",
            shutdown_write: bool = False,
        ) -> tuple[int, dict]:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.putrequest("POST", path)
            connection.putheader("Content-Type", "application/json")
            if content_length is not None:
                connection.putheader("Content-Length", content_length)
            if transfer_encoding:
                connection.putheader("Transfer-Encoding", transfer_encoding)
            connection.endheaders()
            if body:
                connection.send(body)
            if shutdown_write and connection.sock is not None:
                connection.sock.shutdown(socket.SHUT_WR)
            response = connection.getresponse()
            payload = json.loads(response.read())
            status = response.status
            connection.close()
            return status, payload

        try:
            for path, limit in (
                ("/api/audio/session/push", settings.streaming_request_max_bytes),
                ("/api/audio/segment/process", settings.legacy_audio_request_max_bytes),
                ("/api/speaker/enroll", settings.speaker_enrollment_request_max_bytes),
            ):
                status, payload = request(path, content_length=str(limit + 1))
                assert status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                assert payload["detail"] == "audio request body is too large"

            assert request("/api/audio/session/push", content_length="-1")[0] == HTTPStatus.BAD_REQUEST
            assert request("/api/audio/session/push", content_length="invalid")[0] == HTTPStatus.BAD_REQUEST
            assert request("/api/audio/session/push", content_length="9" * 5000)[0] == HTTPStatus.BAD_REQUEST
            assert request("/api/audio/session/push", content_length=None)[0] == HTTPStatus.LENGTH_REQUIRED
            assert request(
                "/api/audio/session/push",
                content_length=None,
                transfer_encoding="chunked",
            )[0] == HTTPStatus.BAD_REQUEST
            assert request(
                "/api/audio/session/push",
                content_length="10",
            )[0] == HTTPStatus.REQUEST_TIMEOUT
            assert request(
                "/api/audio/session/push",
                content_length="10",
                body=b"{}",
                shutdown_write=True,
            )[0] == HTTPStatus.BAD_REQUEST

            legal_body = json.dumps({"user_id": "u1", "mode": "ambient"}).encode("utf-8")
            status, payload = request(
                "/api/audio/session/start",
                content_length=str(len(legal_body)),
                body=legal_body,
            )
            assert status == HTTPStatus.OK
            assert payload["audio_session_id"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            GlassesHandler.service = None


def test_browser_exposes_one_standby_control_and_flushes_before_stop() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "static" / "index.html").read_text(encoding="utf-8")
    app = (root / "static" / "app.js").read_text(encoding="utf-8")
    worklet = (root / "static" / "audio-worklet.js").read_text(encoding="utf-8")

    assert html.count('id="ambient-standby-toggle"') == 1
    assert '/static/app.js?v=mobile-ui-3' in html
    assert 'id="ambient-wake-button"' not in html
    assert 'id="voice-input-button"' not in html
    assert "manual_wake" not in app
    assert "MediaRecorder" not in app
    assert "/api/audio/dispatch/jobs" in app
    assert '"playback_started"' in app
    assert '"playback_finished"' in app
    assert "playback_id: playbackId" in app
    assert "收音=可用" in app
    assert "全天转写=可用" in app
    assert "语音唤醒问答=可用" in app
    assert "缺少 Silero VAD，仅能做收音诊断" in app
    assert "声纹录入=可用" in app
    assert "ambient_transcription_ready" in app
    assert "speaker_enrollment_ready" in app
    assert "assistant_query_ready" in app
    assert 'stopReason: "pause_for_enrollment"' in app
    assert 'stop_reason: stopReason' in app
    assert "resumeAmbientUserId" in app
    assert app.index("await flushUnifiedAudio(active)") < app.index(
        'requestJSON("/api/audio/session/stop"'
    )
    assert "this.emitPcm(this.output.length)" in worklet
    assert 'type: "flushed"' in worklet


@pytest.mark.parametrize(
    ("speaker_state", "overlap_state", "expected_reason"),
    [
        ("other", "not_observed", "audio_speaker_other"),
        ("unknown", "not_observed", "audio_speaker_unknown"),
        ("user", "suspected", "audio_overlap_suspected"),
        ("user", "unknown", "audio_overlap_unknown"),
    ],
)
def test_audio_memory_candidate_privacy_states_are_conservative(
    speaker_state: str,
    overlap_state: str,
    expected_reason: str,
) -> None:
    candidate = MemoryWriteCandidate(
        content="明天提交材料",
        kind="event",
        memory_type="task",
        source_type="wake_query",
        speaker_state=speaker_state,
        overlap_state=overlap_state,
    )
    decision = should_write_memory_candidate(candidate, "记住明天提交材料")
    assert decision.allowed is False
    assert decision.reason == expected_reason
