from __future__ import annotations

import json
import tempfile
from typing import Any

from ai_glasses_memory_assistant import (
    capture_helpers,
    conversation_candidate_helpers,
    conversation_helpers,
    explanation_helpers,
)
from ai_glasses_memory_assistant.agent_bridge import AI_GLASSES_SYSTEM_PROMPT, GlassesChatService
from ai_glasses_memory_assistant.audio_processing import AudioSegmentProcessResult
from ai_glasses_memory_assistant.intent_policy import should_write_memory_candidate
from ai_glasses_memory_assistant.memory_candidate import MemoryWriteCandidate
from ai_glasses_memory_assistant.temporal_parser import TemporalResolution
from ai_glasses_memory_assistant.timeline_store import chunk_to_dict
from ai_glasses_memory_assistant.turn_planner import TurnPlan
from tests.helpers import CoreChatService, FakeAgent, isolated_app_home, pre_reply_recall, pre_reply_write


class TranscriptCandidateAgent(FakeAgent):
    def __init__(self, candidates_by_segment: dict[str, tuple[str, str, str]]) -> None:
        super().__init__()
        self.candidates_by_segment = candidates_by_segment

    def run_conversation(
        self,
        message: str,
        system_message: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        persist_user_message: str | None = None,
    ) -> dict[str, Any]:
        if system_message and "segment semantic cleaner" in system_message:
            raw_span = str(message or "").rsplit("raw_span:", 1)[-1].strip()
            return {
                "final_response": json.dumps({
                    "semantic_role": "memory_candidate",
                    "noise_level": "none",
                    "contains_filler": False,
                    "do_not_remember_scope": "",
                    "should_extract": True,
                    "candidate_span": raw_span,
                    "candidate_hint": "",
                    "confidence": 0.95,
                    "reason": "core transcript fixture",
                }, ensure_ascii=False),
            }
        if system_message and "unified pre-reply decision classifier" in system_message:
            segment = str(message or "").rsplit("User message:\n", 1)[-1].strip()
            candidate = self.candidates_by_segment.get(segment)
            if candidate is None:
                return super().run_conversation(
                    message,
                    system_message=system_message,
                    conversation_history=conversation_history,
                    persist_user_message=persist_user_message,
                )
            content, kind, memory_type = candidate
            payload = pre_reply_write(content, kind=kind, memory_type=memory_type)
            payload["memory_candidates"] = [{
                "content": content,
                "kind": kind,
                "memory_type": memory_type,
                "subject_type": "self",
                "subject_name": "",
                "confidence": 0.95,
            }]
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        return super().run_conversation(
            message,
            system_message=system_message,
            conversation_history=conversation_history,
            persist_user_message=persist_user_message,
        )


class NoiseTranscriptCandidateAgent(TranscriptCandidateAgent):
    def run_conversation(self, message: str, **kwargs) -> dict[str, Any]:
        if "segment semantic cleaner" in str(kwargs.get("system_message") or ""):
            return {
                "final_response": json.dumps({
                    "semantic_role": "noise_only",
                    "noise_level": "high",
                    "contains_filler": False,
                    "do_not_remember_scope": "",
                    "should_extract": False,
                    "candidate_span": "",
                    "candidate_hint": "",
                    "confidence": 0.96,
                    "reason": "test_noise_only",
                }, ensure_ascii=False),
            }
        return super().run_conversation(message, **kwargs)


class UnavailableConversationTypingAgent(FakeAgent):
    def run_conversation(self, message: str, **kwargs: Any) -> dict[str, Any]:
        if "unified pre-reply decision classifier" in str(kwargs.get("system_message") or ""):
            raise RuntimeError("synthetic classifier outage")
        return super().run_conversation(message, **kwargs)


class FailingSemanticTranscriptCandidateAgent(TranscriptCandidateAgent):
    def run_conversation(self, message: str, **kwargs) -> dict[str, Any]:
        if "segment semantic cleaner" in str(kwargs.get("system_message") or ""):
            raise RuntimeError("semantic cleaner unavailable")
        return super().run_conversation(message, **kwargs)


class LowConfidenceTranscriptCandidateAgent(TranscriptCandidateAgent):
    def run_conversation(self, message: str, **kwargs) -> dict[str, Any]:
        if "segment semantic cleaner" in str(kwargs.get("system_message") or ""):
            raw_span = str(message or "").rsplit("raw_span:", 1)[-1].strip()
            return {
                "final_response": json.dumps({
                    "semantic_role": "memory_candidate",
                    "noise_level": "low",
                    "contains_filler": False,
                    "do_not_remember_scope": "",
                    "should_extract": True,
                    "candidate_span": raw_span,
                    "candidate_hint": "event",
                    "confidence": 0.4,
                    "reason": "test_low_confidence",
                }, ensure_ascii=False),
            }
        return super().run_conversation(message, **kwargs)


class ThreadedCoreChatService(CoreChatService):
    def _start_background_long_input_processing(self, **kwargs) -> None:
        GlassesChatService._start_background_long_input_processing(self, **kwargs)


class SubjectOverrideTranscriptCandidateAgent(TranscriptCandidateAgent):
    def __init__(
        self,
        candidates_by_segment: dict[str, tuple[str, str, str]],
        subject_overrides: dict[str, dict[str, str]],
    ) -> None:
        super().__init__(candidates_by_segment)
        self.subject_overrides = subject_overrides

    def run_conversation(self, message: str, **kwargs) -> dict[str, Any]:
        result = super().run_conversation(message, **kwargs)
        system_message = str(kwargs.get("system_message") or "")
        if "unified pre-reply decision classifier" not in system_message:
            return result
        segment = str(message or "").rsplit("User message:\n", 1)[-1].strip()
        override = self.subject_overrides.get(segment)
        if not override:
            return result
        payload = json.loads(str(result.get("final_response") or "{}"))
        for candidate in payload.get("memory_candidates") or []:
            candidate.update(override)
        result["final_response"] = json.dumps(payload, ensure_ascii=False)
        return result


class SequenceAudioProcessor:
    def __init__(self, results: list[AudioSegmentProcessResult]) -> None:
        self.results = list(results)

    def process(self, **_kwargs) -> AudioSegmentProcessResult:
        return self.results.pop(0)


def _assert_no_raw_embedding(payload: Any) -> None:
    if isinstance(payload, dict):
        assert "speaker_embedding" not in payload
        assert "embedding" not in payload
        for value in payload.values():
            _assert_no_raw_embedding(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_no_raw_embedding(value)


def test_capture_helpers_preserve_capture_text_contract() -> None:
    text = " 第一段   内容\n\n第二段\n第三段\n第四段\n第五段\n第六段"

    assert capture_helpers.summarize_capture_text(text) == "第一段 内容；第二段；第三段；第四段；第五段"
    assert capture_helpers.continuous_capture_reply(["a"]) == "收到，我先把这段长输入整理到时间线里，有价值的内容会后台沉淀。"
    assert capture_helpers.continuous_capture_reply(["a", "b"]) == "收到，我先把这段长输入按 2 段整理，有价值的内容会后台沉淀。"


def test_capture_creates_provisional_subject_and_saves_its_memory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=TranscriptCandidateAgent({"我带材料": ("带材料", "event", "task")}),
        )
        capture = service.start_capture(user_id="u1")
        service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="我带材料",
            metadata={
                "speaker_label": "speaker_2",
                "speaker_embedding": [1.0, 0.0, 0.0],
                "speaker_embedding_model": "campplus",
                "speaker_confidence": 0.94,
            },
        )

        result = service.stop_capture(user_id="u1", capture_id=capture["capture_id"])
        memories = service.memory_store.list_memories("u1")

        assert {(memory.subject_type, memory.subject_name, memory.content) for memory in memories} == {
            ("provisional", "speaker_2", "带材料"),
        }
        assert len(service.memory_store.list_voice_profiles("u1")) == 1
        assert result["import_result"]["saved_count"] == 1
        assert result["import_result"]["saved_memories"][0]["id"] == memories[0].id
        identity = result["import_result"]["conversation_session"]["voice_identity"][0]
        assert identity["subject_id"] == memories[0].subject_id
        assert identity["profile_stored"] is True


def test_android_ambient_stop_keeps_unlabeled_unknown_final_timeline_only() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=TranscriptCandidateAgent({"主要就是这些还红包": ("主要就是这些还红包", "event", "event")}),
        )
        capture = service.start_device_capture(user_id="u1")
        appended = service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="主要就是这些还红包",
            metadata={
                "audio_event_id": "audio-unknown-1",
                "speaker_hint": "unknown",
                "speaker_state": "unknown",
                "overlap_state": "not_observed",
                "memory_eligible": False,
            },
        )

        result = service.stop_capture(user_id="u1", capture_id=capture["capture_id"])

        assert service.memory_store.list_memories("u1") == []
        assert service.timeline_store.list_chunks_by_ids("u1", [appended["chunk_id"]], limit=1)[0].text == "主要就是这些还红包"
        assert result["import_result"]["saved_count"] == 0
        assert "audio_speaker_unknown" in result["import_result"]["memory_job"]["rejected_reasons"]


def test_android_ambient_stop_fails_closed_when_audio_identity_metadata_is_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=TranscriptCandidateAgent({"我买车了": ("我买车了", "event", "fact")}),
        )
        capture = service.start_device_capture(user_id="u1")
        appended = service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="我买车了",
            metadata={
                "audio_event_id": "audio-missing-identity",
                "memory_eligible": True,
            },
        )

        result = service.stop_capture(user_id="u1", capture_id=capture["capture_id"])

        assert service.memory_store.list_memories("u1") == []
        assert service.timeline_store.list_chunks_by_ids("u1", [appended["chunk_id"]], limit=1)[0].text == "我买车了"
        unit_result = result["import_result"]["memory_job"]["extraction_trace"]["conversation_session"]["unit_gate_results"][0]
        assert unit_result["status"] == "rejected"
        assert unit_result["reason"] == "audio_speaker_unknown"


def test_android_ambient_stop_saves_only_trusted_semantically_valid_user_chunk() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=TranscriptCandidateAgent({
                "我买车了": ("我买车了", "event", "fact"),
                "安あ": ("安あ", "event", "event"),
            }),
        )
        capture = service.start_device_capture(user_id="u1")
        first = service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="我买车了",
            metadata={
                "audio_event_id": "audio-user-1",
                "speaker_hint": "user",
                "speaker_state": "user",
                "overlap_state": "not_observed",
                "memory_eligible": True,
            },
        )
        second = service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="安あ",
            metadata={
                "audio_event_id": "audio-unknown-2",
                "speaker_hint": "unknown",
                "speaker_state": "unknown",
                "overlap_state": "not_observed",
                "memory_eligible": False,
            },
        )

        result = service.stop_capture(user_id="u1", capture_id=capture["capture_id"])
        memories = service.memory_store.list_memories("u1")

        assert [(memory.content, memory.evidence_ids) for memory in memories] == [("我买车了", [first["chunk_id"]])]
        assert result["import_result"]["saved_count"] == 1
        assert "audio_speaker_unknown" in result["import_result"]["memory_job"]["rejected_reasons"]
        assert service.timeline_store.list_chunks_by_ids("u1", [second["chunk_id"]], limit=1)[0].text == "安あ"
        unit_results = result["import_result"]["memory_job"]["extraction_trace"]["conversation_session"]["unit_gate_results"]
        assert [(item["chunk_id"], item["status"], item["reason"]) for item in unit_results] == [
            (first["chunk_id"], "saved", "saved"),
            (second["chunk_id"], "rejected", "audio_speaker_unknown"),
        ]


def test_android_ambient_semantic_noise_and_fallback_fail_closed() -> None:
    cases = (
        (NoiseTranscriptCandidateAgent, "semantic_noise_only"),
        (FailingSemanticTranscriptCandidateAgent, "semantic_backend_untrusted"),
        (LowConfidenceTranscriptCandidateAgent, "semantic_confidence_below_threshold"),
    )
    for agent_type, expected_reason in cases:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            service = CoreChatService(
                tmpdir,
                agent=agent_type({"今はご前": ("今はご前", "event", "event")}),
            )
            capture = service.start_device_capture(user_id="u1")
            appended = service.append_capture_chunk(
                user_id="u1",
                capture_id=capture["capture_id"],
                text="今はご前",
                metadata={
                    "audio_event_id": f"audio-{expected_reason}",
                    "speaker_hint": "user",
                    "speaker_state": "user",
                    "overlap_state": "not_observed",
                    "memory_eligible": True,
                },
            )

            result = service.stop_capture(user_id="u1", capture_id=capture["capture_id"])

            assert service.memory_store.list_memories("u1") == []
            assert service.timeline_store.list_chunks_by_ids("u1", [appended["chunk_id"]], limit=1)[0].text == "今はご前"
            assert expected_reason in result["import_result"]["memory_job"]["rejected_reasons"]
            unit_result = result["import_result"]["memory_job"]["extraction_trace"]["conversation_session"]["unit_gate_results"][0]
            assert unit_result["audio_event_id"] == f"audio-{expected_reason}"
            assert unit_result["chunk_id"] == appended["chunk_id"]
            assert unit_result["status"] == "rejected"
            assert unit_result["reason"] == expected_reason


def test_capture_voice_match_reuses_subject_across_different_speaker_labels() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=TranscriptCandidateAgent({
                "我带材料": ("带材料", "event", "task"),
                "我准备报价": ("准备报价", "event", "task"),
            }),
        )
        first = service.start_capture(user_id="u1")
        service.append_capture_chunk(
            user_id="u1",
            capture_id=first["capture_id"],
            text="我带材料",
            metadata={
                "speaker_label": "speaker_2",
                "speaker_embedding": [1.0, 0.0, 0.0],
                "speaker_embedding_model": "campplus",
                "speaker_confidence": 0.95,
            },
        )
        service.stop_capture(user_id="u1", capture_id=first["capture_id"])
        first_memory = service.memory_store.list_memories("u1")[0]

        second = service.start_capture(user_id="u1")
        service.append_capture_chunk(
            user_id="u1",
            capture_id=second["capture_id"],
            text="我准备报价",
            metadata={
                "speaker_label": "speaker_9",
                "speaker_embedding": [0.999, 0.01, 0.0],
                "speaker_embedding_model": "campplus",
                "speaker_confidence": 0.96,
            },
        )

        result = service.stop_capture(user_id="u1", capture_id=second["capture_id"])
        memories = service.memory_store.list_memories("u1")
        second_memory = next(memory for memory in memories if memory.content == "准备报价")

        assert second_memory.subject_id == first_memory.subject_id
        assert second_memory.subject_name == "speaker_2"
        identity = result["import_result"]["conversation_session"]["voice_identity"][0]
        assert identity["decision"] == "matched"
        assert identity["reason"] == "voice_profile_matched"


def test_audio_segment_path_passes_internal_voice_embedding_to_capture_without_exposing_it() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=TranscriptCandidateAgent({
                "I will bring the notes": ("bring the notes", "event", "task"),
                "I will prepare the summary": ("prepare the summary", "event", "task"),
            }),
        )
        service.audio_sessions.registry.offline_adapter = SequenceAudioProcessor([
            AudioSegmentProcessResult(
                status="processed",
                transcript="I will bring the notes",
                metadata={"speaker_hint": "unknown", "speaker_confidence": 0.95},
                speaker_embedding=(1.0, 0.0, 0.0),
                speaker_embedding_model="campplus",
            ),
            AudioSegmentProcessResult(
                status="processed",
                transcript="I will prepare the summary",
                metadata={"speaker_hint": "unknown", "speaker_confidence": 0.96},
                speaker_embedding=(0.999, 0.01, 0.0),
                speaker_embedding_model="campplus",
            ),
        ])

        first = service.start_capture(user_id="u1")
        first_payload = service.process_audio_segment(
            user_id="u1",
            capture_id=first["capture_id"],
            speaker_label="speaker_2",
        )
        _assert_no_raw_embedding(first_payload)
        first_stopped = service.stop_capture(user_id="u1", capture_id=first["capture_id"])
        first_identity = first_stopped["import_result"]["conversation_session"]["voice_identity"][0]

        second = service.start_capture(user_id="u1")
        second_payload = service.process_audio_segment(
            user_id="u1",
            capture_id=second["capture_id"],
            speaker_label="speaker_9",
        )
        _assert_no_raw_embedding(second_payload)
        second_stopped = service.stop_capture(user_id="u1", capture_id=second["capture_id"])
        second_identity = second_stopped["import_result"]["conversation_session"]["voice_identity"][0]

        assert second_identity["decision"] == "matched"
        assert second_identity["subject_id"] == first_identity["subject_id"]
        assert len(service.memory_store.list_voice_profiles("u1")) == 2
        assert service.memory_store.list_memories("u1") == []


def test_explicit_alias_merges_existing_provisional_memory_and_voice_profile() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=TranscriptCandidateAgent({
                "I will bring the notes": ("bring the notes", "event", "task"),
                "I confirm": ("confirmed", "event", "fact"),
            }),
        )
        capture = service.start_capture(user_id="u1")
        service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="I will bring the notes",
            metadata={
                "speaker_label": "speaker_2",
                "speaker_embedding": [1.0, 0.0, 0.0],
                "speaker_embedding_model": "campplus",
                "speaker_confidence": 0.95,
            },
        )
        service.stop_capture(user_id="u1", capture_id=capture["capture_id"])
        provisional_memory = service.memory_store.list_memories("u1")[0]
        provisional_subject_id = provisional_memory.subject_id

        response = service.chat(
            "[user] speaker_2 是 Alex\n[Alex] I confirm",
            user_id="u1",
            input_mode="speaker_transcript",
        )
        refreshed = service.memory_store.get_memory("u1", provisional_memory.id)
        named = service.memory_store.resolve_subject("u1", "Alex", subject_types={"named"})
        profiles = service.memory_store.list_voice_profiles("u1", subject_ids=[named.id])

        assert refreshed.subject_id == named.id
        assert (refreshed.subject_type, refreshed.subject_name) == ("named", "Alex")
        assert service.memory_store.get_subject("u1", provisional_subject_id).id == named.id
        assert service.memory_store.resolve_subject(
            "u1",
            "speaker_2",
            source_scope=capture["capture_id"],
        ).id == named.id
        assert profiles and all(profile.subject_id == named.id for profile in profiles)
        assert response["debug"]["conversation_session"]["subject_alias_actions"][0]["action"] == (
            "provisional_merged"
        )


def test_capture_ambiguous_voice_match_creates_new_provisional_subject() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=TranscriptCandidateAgent({"我准备材料": ("准备材料", "event", "task")}),
        )
        first = service.memory_store.create_provisional_subject("u1", "speaker_a", source_scope="seed-a")
        second = service.memory_store.create_provisional_subject("u1", "speaker_b", source_scope="seed-b")
        service.memory_store.store_voice_profile(
            "u1",
            first.id,
            embedding=[1.0, 0.0],
            embedding_model="campplus",
            source_id="seed-a",
        )
        service.memory_store.store_voice_profile(
            "u1",
            second.id,
            embedding=[0.99875, 0.04998],
            embedding_model="campplus",
            source_id="seed-b",
        )
        capture = service.start_capture(user_id="u1")
        service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="我准备材料",
            metadata={
                "speaker_label": "speaker_7",
                "speaker_embedding": [0.99969, 0.025],
                "speaker_embedding_model": "campplus",
                "speaker_confidence": 0.92,
            },
        )

        result = service.stop_capture(user_id="u1", capture_id=capture["capture_id"])
        memory = next(memory for memory in service.memory_store.list_memories("u1") if memory.content == "准备材料")
        identity = result["import_result"]["conversation_session"]["voice_identity"][0]

        assert memory.subject_id not in {first.id, second.id}
        assert (memory.subject_type, memory.subject_name) == ("provisional", "speaker_7")
        assert identity["decision"] == "provisional"
        assert identity["reason"] == "voice_profile_ambiguous_similarity_margin"
        assert identity["margin"] < 0.05


def test_capture_raw_embedding_is_not_exposed_in_public_payloads_or_audit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=TranscriptCandidateAgent({"我带材料": ("带材料", "event", "task")}),
        )
        capture = service.start_capture(user_id="u1")
        appended = service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="我带材料",
            metadata={
                "speaker_label": "speaker_2",
                "speaker_embedding": [0.1, 0.2, 0.3],
                "speaker_embedding_model": "campplus",
                "speaker_confidence": 0.91,
                "speaker": {
                    "embedding": [0.4, 0.5, 0.6],
                    "samples": [{"speaker_embedding": [0.7, 0.8, 0.9]}],
                },
            },
        )

        result = service.stop_capture(user_id="u1", capture_id=capture["capture_id"])
        chunk = service.timeline_store.list_chunks_by_ids("u1", [appended["chunk_id"]])[0]
        public_chunk = chunk_to_dict(chunk)
        audits = service.read_audit_records(user_id="u1", limit=20)

        _assert_no_raw_embedding(result)
        _assert_no_raw_embedding(public_chunk)
        _assert_no_raw_embedding(result.get("import_result", {}).get("conversation_session", {}))
        _assert_no_raw_embedding(audits)


def test_conversation_helpers_preserve_speaker_labeled_parser_contract() -> None:
    transcript = "\n".join([
        "[09:31][user] speaker_2 是 Alex",
        "[09:32][speaker_2] ready",
        "Beta: noted",
    ])

    helper_session = conversation_helpers.parse_speaker_labeled_transcript(transcript)

    assert helper_session is not None
    assert helper_session.turns[0].timestamp_text == "09:31"
    assert helper_session.turns[1].speaker_label == "Alex"
    debug = helper_session.debug_payload()
    assert debug["participants"]["speaker_2"] == "known_person"
    assert debug["participants"]["Alex"] == "known_person"
    assert debug["speaker_aliases"] == [{
        "source_label": "speaker_2",
        "target_label": "Alex",
        "alias_source": "user_named_speaker",
        "turn_index": 0,
    }]
    assert debug["alias_applied_turns"] == [{"turn_index": 1, "from": "speaker_2", "to": "Alex"}]
    assert debug["parsed_turns"][1]["speaker_label"] == "Alex"


def test_conversation_helpers_keep_predicted_voice_labels_anonymous_until_named() -> None:
    assert conversation_helpers.conversation_speaker_role("PRED_SPK0001") == "unknown_speaker"
    assert conversation_helpers.conversation_speaker_role("speaker_2") == "unknown_speaker"
    assert conversation_helpers.conversation_speaker_role("Alex") == "known_person"

    session = conversation_helpers.parse_speaker_labeled_transcript(
        "[user] PRED_SPK0001 是 Alex\n[PRED_SPK0001] I confirm"
    )

    assert session is not None
    assert session.turns[1].speaker_label == "Alex"
    assert session.turns[1].speaker_role == "known_person"
    assert session.speaker_aliases[0]["source_label"] == "PRED_SPK0001"


def test_conversation_candidate_helpers_stay_structural_only() -> None:
    transcript = "\n".join([
        "[09:31][user] context",
        "[09:32][Alpha] ready",
        "[09:33][speaker_2] hidden",
    ])
    session = conversation_helpers.parse_speaker_labeled_transcript(transcript)
    assert session is not None

    helper_candidates, helper_debug = conversation_candidate_helpers.conversation_memory_candidates(session)

    assert helper_candidates == []
    assert helper_debug["candidate_strategy"] == "structural_only"
    assert helper_debug["candidate_count"] == 0
    assert helper_debug["candidate_turn_indices"] == []
    assert helper_debug["candidate_facts"] == []
    assert helper_debug["rejected_reasons"] == []


def test_conversation_extraction_plan_enforces_structural_privacy_boundaries() -> None:
    transcript = "\n".join([
        "[用户] 我负责方案",
        "[speaker_2] 我带材料",
        "[用户] speaker_2 是李四",
        "[李四] 验证码是 482931，我负责报价",
    ])
    session = conversation_helpers.parse_speaker_labeled_transcript(transcript)
    assert session is not None

    units, debug = conversation_candidate_helpers.conversation_extraction_plan(session)

    assert [(unit.speaker_role, unit.text) for unit in units] == [
        ("user", "我负责方案"),
        ("known_person", "我带材料"),
        ("known_person", "我负责报价"),
    ]
    assert "speaker_alias_declaration" in debug["rejected_reasons"]
    assert "sensitive_fragment_filtered" in debug["rejected_reasons"]
    assert all("482931" not in unit.text for unit in units)


def test_conversation_extraction_plan_keeps_named_people_without_user_turn() -> None:
    session = conversation_helpers.parse_speaker_labeled_transcript(
        "张三：我负责材料\n李四：我准备报价"
    )
    assert session is not None

    units, debug = conversation_candidate_helpers.conversation_extraction_plan(session)

    assert [(unit.subject_type, unit.subject_name, unit.text) for unit in units] == [
        ("named", "张三", "我负责材料"),
        ("named", "李四", "我准备报价"),
    ]
    assert debug["rejected_reasons"] == []
    assert debug["rejected_turns"] == []


def test_multi_speaker_memory_gate_accepts_isolated_subjects_and_blocks_sensitive_content() -> None:
    subjects = [
        ("self", "", "user"),
        ("named", "张三", "other"),
        ("provisional", "speaker_2", "unknown"),
    ]

    for subject_type, subject_name, speaker_hint in subjects:
        safe = MemoryWriteCandidate(
            content="喜欢安静环境",
            kind="profile",
            memory_type="preference",
            source_type="multi_speaker_transcript",
            speaker_hint=speaker_hint,
            subject_type=subject_type,
            subject_name=subject_name,
        )
        sensitive = MemoryWriteCandidate(
            content="验证码是 482931",
            kind="profile",
            memory_type="fact",
            source_type="multi_speaker_transcript",
            speaker_hint=speaker_hint,
            subject_type=subject_type,
            subject_name=subject_name,
        )

        assert should_write_memory_candidate(safe, "多人转写").allowed is True
        sensitive_gate = should_write_memory_candidate(sensitive, "多人转写")
        assert sensitive_gate.allowed is False
        assert sensitive_gate.requires_confirmation is True


def test_chat_routes_multi_speaker_transcript_through_structural_gate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = TranscriptCandidateAgent({
            "我负责方案": ("负责方案", "event", "task"),
            "我负责报价": ("负责报价", "event", "task"),
        })
        service = CoreChatService(tmpdir, agent=agent)

        response = service.chat(
            "[用户] 我负责方案\n[张三] 我负责报价",
            user_id="u1",
            session_id="transcript-session",
            input_mode="speaker_transcript",
        )
        job = service.read_memory_job(
            user_id="u1",
            job_id=response["debug"]["memory_processing"]["job_id"],
        )
        memories = service.memory_store.list_memories("u1")

        assert response["saved_memories"] == []
        assert response["debug"]["memory_processing"]["status"] == "pending"
        assert response["debug"]["conversation_session"]["detected"] is True
        assert "multi_speaker_structural_gate" in response["debug"]["steps"]
        assert job is not None and job["status"] == "saved"
        assert response["session_id"] == job["session_id"] == "transcript-session"
        assert {(memory.subject_type, memory.subject_name) for memory in memories} == {
            ("self", "我"),
            ("named", "张三"),
        }
        evidence_texts = {
            memory.subject_name: {
                chunk.text
                for chunk in service.timeline_store.list_chunks_by_ids("u1", memory.evidence_ids)
            }
            for memory in memories
        }
        assert evidence_texts == {
            "我": {"我负责方案"},
            "张三": {"我负责报价"},
        }


def test_chat_saves_named_people_without_user_turn_and_keeps_bob_preference_separate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = TranscriptCandidateAgent({
            "我负责客户合同": ("负责客户合同", "event", "task"),
            "我最近喜欢喝冰美式": ("最近喜欢喝冰美式", "profile", "preference"),
        })
        service = CoreChatService(tmpdir, agent=agent)

        response = service.chat(
            "张三：我负责客户合同\nBob：我最近喜欢喝冰美式",
            user_id="u1",
            input_mode="speaker_transcript",
        )
        memories = service.memory_store.list_memories("u1")

        assert response["saved_memories"] == []
        assert {(memory.subject_name, memory.content) for memory in memories} == {
            ("张三", "负责客户合同"),
            ("Bob", "最近喜欢喝冰美式"),
        }
        assert next(memory for memory in memories if memory.subject_name == "Bob").memory_type == "preference"


def test_multi_speaker_llm_candidate_cannot_override_trusted_turn_subject() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = SubjectOverrideTranscriptCandidateAgent(
            {
                "I own the task": ("owns the task", "event", "task"),
                "I confirm": ("confirmed", "event", "fact"),
            },
            {
                "I own the task": {
                    "subject_type": "named",
                    "subject_name": "Injected Person",
                    "subject_scope": "injected-scope",
                }
            },
        )
        service = CoreChatService(tmpdir, agent=agent)

        service.chat(
            "Alex: I own the task\nBeta: I confirm",
            user_id="u1",
            input_mode="speaker_transcript",
        )
        memories = service.memory_store.list_memories("u1")

        assert {(memory.subject_name, memory.content) for memory in memories} == {
            ("Alex", "owns the task"),
            ("Beta", "confirmed"),
        }
        assert service.memory_store.resolve_subject("u1", "Injected Person") is None


def test_chat_preserves_each_provisional_speaker_as_an_independent_subject() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = TranscriptCandidateAgent({
            "我带材料": ("带材料", "event", "task"),
            "我准备报价": ("准备报价", "event", "task"),
        })
        service = CoreChatService(tmpdir, agent=agent)

        service.chat(
            "speaker_2：我带材料\nspeaker_3：我准备报价",
            user_id="u1",
            input_mode="speaker_transcript",
        )
        memories = service.memory_store.list_memories("u1")

        assert {(memory.subject_type, memory.subject_name, memory.content) for memory in memories} == {
            ("provisional", "speaker_2", "带材料"),
            ("provisional", "speaker_3", "准备报价"),
        }
        assert len({memory.subject_id for memory in memories}) == 2


def test_ordinary_colon_form_chat_does_not_trigger_speaker_transcript_mode() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)

        response = service.chat("主题：下周计划\n备注：先看需求", user_id="u1")

        assert response["debug"]["input_mode"] == "chat"
        assert "conversation_session" not in response["debug"]
        assert response["debug"]["memory_processing"]["status"] == "not_needed"
        assert service.memory_store.list_memories("u1") == []


def test_wait_memory_job_finishes_real_worker_before_store_cleanup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = TranscriptCandidateAgent({
            "我负责材料": ("负责材料", "event", "task"),
            "我负责报价": ("负责报价", "event", "task"),
        })
        service = ThreadedCoreChatService(tmpdir, agent=agent)
        try:
            response = service.chat(
                "张三：我负责材料\n李四：我负责报价",
                user_id="u1",
                input_mode="speaker_transcript",
            )
            job_id = response["debug"]["memory_processing"]["job_id"]

            job = service.wait_memory_job(user_id="u1", job_id=job_id, timeout=5.0)

            assert job is not None and job["status"] == "saved"
            assert job["session_id"] == response["session_id"]
            assert job["saved_count"] == 2
        finally:
            service.close()


def test_explanation_helpers_preserve_reply_contract() -> None:
    memory = {
        "kind": "profile",
        "content": "用户喜欢低糖拿铁",
        "source_trace": {
            "source_id": "turn-1",
            "ingestion_id": "ingestion-1",
            "evidence_ids": ["chunk-1", "chunk-1"],
        },
    }

    assert explanation_helpers.is_explanation_query("你为什么这么说？") is True
    reply = explanation_helpers.explanation_reply(
        message="你为什么这么说？",
        source_summary={
            "primary_source": "profile",
            "primary_source_label": "稳定画像",
            "primary_source_explanation": "这次主要依据稳定画像记忆。",
        },
        memory_processing={"status": "not_needed"},
        recall_arbitration={"decisions": [{"reason": "raw_timeline_lower_priority"}]},
        recent_context_capsule={
            "injected_to_main_llm": False,
            "injection_reason": "no_recent_reference_markers",
        },
        recalled_memories=[memory],
        saved_memories=[],
        evidence_quotes=["我喜欢低糖拿铁"],
    )

    assert "这次回答主要依据是稳定画像" in reply
    assert "具体依据是这条记忆" in reply
    assert "evidence_ids=chunk-1" in reply
    assert "raw_timeline_lower_priority" in reply

    skipped_reply = explanation_helpers.explanation_reply(
        message="为什么没保存咖啡这段？",
        source_summary={},
        memory_processing={
            "status": "skipped",
            "local_do_not_remember_scopes": ["咖啡这段"],
        },
        recall_arbitration={},
        recent_context_capsule={},
    )

    assert "咖啡这段" in skipped_reply
    assert "局部“不要记/不用记”的范围" in skipped_reply


def test_chat_returns_reply_and_persists_timeline_audit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        response = service.chat("帮我总结一下这个系统", user_id="u1")

        assert response["reply"] == "主回复"
        assert response["debug"]["timeline"]["persisted"] is True
        assert service.audit_path.exists()
        assert service.fake_agent.calls


def test_chat_saves_memory_candidate_after_reply() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=pre_reply_write("周五检查 demo", memory_type="task"))
        service = CoreChatService(tmpdir, agent=agent)
        response = service.chat("记一下周五检查 demo", user_id="u1")
        memories = service.memory_store.list_memories("u1")

        assert response["reply"]
        assert "记住" in response["reply"]
        assert response["saved_memories"]
        assert memories[0].content == "周五检查 demo"
        assert memories[0].evidence_ids


def test_named_correction_is_saved_once_under_the_semantic_subject() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_write("喜欢茶", kind="profile", memory_type="preference")
        decision["flags"]["correction"] = True
        decision["memory_candidates"] = [{
            "content": "喜欢茶",
            "kind": "profile",
            "memory_type": "preference",
            "subject_type": "named",
            "subject_name": "张三",
            "confidence": 0.95,
        }]
        service = CoreChatService(tmpdir, agent=FakeAgent(pre_reply=decision))

        service.chat("纠正一下，张三喜欢茶", user_id="u1")
        memories = service.memory_store.list_memories("u1")

        assert [(memory.subject_type, memory.subject_name, memory.content) for memory in memories] == [
            ("named", "张三", "喜欢茶"),
        ]
        assert memories[0].source == "correction"


def test_chat_rejects_sensitive_memory_but_keeps_reply() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=pre_reply_write("api_key=sk-abcdefghijklmnopqrstuvwxyz123456", memory_type="fact"))
        service = CoreChatService(tmpdir, agent=agent)
        response = service.chat("记一下 api_key=sk-abcdefghijklmnopqrstuvwxyz123456", user_id="u1")

        assert response["reply"]
        assert response["saved_memories"] == []
        assert "保存" in response["reply"] or response["debug"]["fast_path"] is True
        assert service.memory_store.list_memories("u1") == []


def test_chat_recalls_user_scoped_memory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=pre_reply_recall())
        service = CoreChatService(tmpdir, agent=agent)
        mine = service.memory_store.add_memory("u1", "周五检查 demo", kind="event", memory_type="task", created_at=1.0)
        service.memory_store.add_memory("u2", "周五检查 demo", kind="event", memory_type="task", created_at=1.0)

        response = service.chat("我周五要做什么", user_id="u1")

        assert response["recalled_memories"]
        assert response["recalled_memories"][0]["id"] == mine.id
        assert "周五检查 demo" in response["reply"]


def test_tailored_advice_uses_bounded_self_profile_context_for_main_reply() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="profile", goal="summary")
        decision["turn_intent"] = "mixed"
        decision["reason"] = "tailored_advice_profile_context"
        agent = FakeAgent(pre_reply=decision, reply="Choose a quiet workspace and reserve uninterrupted time.")
        service = CoreChatService(tmpdir, agent=agent)
        mine = service.memory_store.add_memory(
            "u1",
            "Prefers quiet workspaces for concentrated tasks.",
            kind="profile",
            memory_type="preference",
        )
        other = service.memory_store.add_memory(
            "u2",
            "Prefers collaborative open-plan workspaces.",
            kind="profile",
            memory_type="preference",
        )

        response = service.chat(
            "Based on the work habits I have shared, what focus environment would suit me?",
            user_id="u1",
        )

        assert [memory["id"] for memory in response["recalled_memories"]] == [mine.id]
        assert other.id not in {memory["id"] for memory in response["recalled_memories"]}
        assert response["reply"] == "Choose a quiet workspace and reserve uninterrupted time."
        assert response["debug"]["planner"]["needs_profile_memory"] is True
        assert response["debug"]["planner"]["needs_event_memory"] is False
        assert response["debug"]["planner"]["needs_timeline_recall"] is False
        assert response["debug"]["planner"]["reply_mode"] == "llm"
        assert response["debug"]["planner"]["reason"].endswith("profile_context_for_llm")
        main_call = next(call for call in reversed(agent.calls) if not call["system_message"])
        assert "Prefers quiet workspaces for concentrated tasks." in main_call["message"]
        assert "Prefers collaborative open-plan workspaces." not in main_call["message"]


def test_main_memory_prompt_answers_from_relevant_context_before_abstaining() -> None:
    assert "When the recalled context is directly relevant" in AI_GLASSES_SYSTEM_PROMPT
    prompt = " ".join(AI_GLASSES_SYSTEM_PROMPT.lower().split())
    assert "abstain only when the recalled context does not support" in prompt


def test_tailored_advice_can_recall_profile_and_episodic_history_together() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="profile", goal="summary")
        decision["turn_intent"] = "mixed"
        decision["needs_profile_memory"] = True
        decision["needs_event_memory"] = True
        decision["event_recall_strategy"] = "text_search"
        decision["reason"] = "tailored recommendation needs stable preference and prior activity"
        agent = FakeAgent(pre_reply=decision, reply="Use Adobe Premiere Pro resources focused on advanced color grading.")
        service = CoreChatService(tmpdir, agent=agent)
        profile = service.memory_store.add_memory(
            "u1",
            "Prefers Adobe Premiere Pro for video editing.",
            kind="profile",
            memory_type="preference",
        )
        activity = service.memory_store.add_memory(
            "u1",
            "I'm learning the Lumetri Color Panel and advanced color grading in Adobe Premiere Pro.",
            kind="event",
            memory_type="event",
        )
        unrelated = service.memory_store.add_memory(
            "u1",
            "周末准备骑车去郊外。",
            kind="event",
            memory_type="event",
        )

        response = service.chat(
            "Can you recommend resources where I can learn more about video editing?",
            user_id="u1",
        )

        recalled_ids = {item["id"] for item in response["recalled_memories"]}
        assert profile.id in recalled_ids
        assert activity.id in recalled_ids
        assert unrelated.id not in recalled_ids
        assert response["debug"]["planner"]["needs_profile_memory"] is True
        assert response["debug"]["planner"]["needs_event_memory"] is True
        assert response["debug"]["planner"]["event_recall_strategy"] == "text_search"
        assert response["debug"]["memory"]["event_recall"]["candidate_trace"]["selected_ids"]
        main_call = next(call for call in reversed(agent.calls) if not call["system_message"])
        assert "Lumetri Color Panel" in main_call["message"]


def test_malformed_tailored_profile_recall_still_uses_self_dietary_context_for_main_reply() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = {
            "turn_intent": "mixed",
            "reply_mode": "llm",
            "answer_source": "llm",
            "memory_action": "none|write|recall|correction|explain",
            "memory_recall_type": "none|profile|event|timeline|observation",
            "recall_goal": "none|summary|raw_evidence|specific_fact",
            "needs_profile_memory": True,
            "needs_event_memory": False,
            "needs_timeline_recall": False,
            "needs_discussion_recall": False,
            "recall_subject_scope": "self",
            "recall_subject_names": [],
            "event_recall_strategy": "skipped",
            "flags": {},
            "confidence": 0.95,
            "reason": "synthetic tailored dietary advice",
        }
        agent = FakeAgent(pre_reply=decision, reply="Choose a dairy-free lunch with a clear ingredient list.")
        service = CoreChatService(tmpdir, agent=agent)
        mine = service.memory_store.add_memory(
            "u1",
            "The user avoids dairy because lactose causes discomfort.",
            kind="profile",
            memory_type="fact",
        )
        other = service.memory_store.add_memory(
            "u2",
            "The user prefers extra-cheese pasta.",
            kind="profile",
            memory_type="preference",
        )

        response = service.chat("What lunch would fit my dietary needs today?", user_id="u1")

        assert response["reply"] == "Choose a dairy-free lunch with a clear ingredient list."
        assert [memory["id"] for memory in response["recalled_memories"]] == [mine.id]
        assert other.id not in {memory["id"] for memory in response["recalled_memories"]}
        assert response["debug"]["planner"]["reply_mode"] == "llm"
        assert response["debug"]["planner"]["reason"].endswith("profile_context_for_llm")
        assert "recovered_malformed_tailored_profile_recall" in response["debug"]["pre_reply_decision"]["warnings"]
        main_call = next(call for call in reversed(agent.calls) if not call["system_message"])
        assert "The user avoids dairy because lactose causes discomfort." in main_call["message"]
        assert "The user prefers extra-cheese pasta." not in main_call["message"]


def test_generic_advice_does_not_recall_profile_context() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir, agent=FakeAgent())
        service.memory_store.add_memory(
            "u1",
            "Prefers quiet workspaces for concentrated tasks.",
            kind="profile",
            memory_type="preference",
        )

        response = service.chat("What are three ways to plan a focused workday?", user_id="u1")

        assert response["recalled_memories"] == []
        assert response["debug"]["planner"]["needs_profile_memory"] is False
        assert response["debug"]["planner"]["reply_mode"] == "llm"


def test_specific_personal_fact_recall_searches_profile_and_event_together() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=pre_reply_recall(recall_type="profile"), reply="是的，我记得你买车了，而且车停在楼下。")
        service = CoreChatService(tmpdir, agent=agent)
        unrelated_profile = service.memory_store.add_memory(
            "u1",
            "我叫小明",
            kind="profile",
            memory_type="fact",
        )
        first = service.memory_store.add_memory("u1", "我买车了", kind="event", memory_type="fact")
        second = service.memory_store.add_memory("u1", "我的车停在楼下", kind="event", memory_type="event")

        response = service.chat("我有车吗", user_id="u1")
        recalled_ids = {item["id"] for item in response["recalled_memories"]}

        assert first.id in recalled_ids
        assert second.id in recalled_ids
        assert unrelated_profile.id not in recalled_ids
        assert "买车" in response["reply"]
        assert response["debug"]["memory"]["cross_kind_recall"]["applied"] is True
        assert response["debug"]["memory"]["cross_kind_recall"]["event_count"] == 2


def test_specific_personal_fact_keeps_conflicting_profile_and_event_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(
            pre_reply=pre_reply_recall(recall_type="profile"),
            reply="我的记忆存在冲突：一条说没有车，另一条说后来买车了。",
        )
        service = CoreChatService(tmpdir, agent=agent)
        profile = service.memory_store.add_memory("u1", "我没有车", kind="profile", memory_type="fact")
        event = service.memory_store.add_memory("u1", "我后来买车了", kind="event", memory_type="fact")

        response = service.chat("我有车吗", user_id="u1")

        assert {item["id"] for item in response["recalled_memories"]} == {profile.id, event.id}
        assert "冲突" in response["reply"]
        assert response["debug"]["memory"]["recall_arbitration"]["kept_counts"] == {
            "profile": 1,
            "event": 1,
            "timeline": 0,
            "document": 0,
        }
        cross_kind = response["debug"]["memory"]["cross_kind_recall"]
        assert cross_kind["adopted_memory_ids"] == [profile.id, event.id]
        assert cross_kind["reply_path"] == "main_llm"
        main_call = next(call for call in reversed(agent.calls) if not call["system_message"])
        assert "我没有车" in main_call["message"]
        assert "我后来买车了" in main_call["message"]
        assert "state the conflict clearly" in main_call["message"]


def test_specific_personal_fact_without_evidence_uses_empty_evidence_guard() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=FakeAgent(
                pre_reply=pre_reply_recall(recall_type="profile"),
                reply="我猜你可能有车。",
            ),
        )

        response = service.chat("我有车吗", user_id="u1")

        assert response["reply"] == "我没有查到这件事的具体记录。"
        assert response["recalled_memories"] == []
        guard = response["debug"]["memory"]["recall_arbitration"]["empty_evidence_guard"]
        assert guard["triggered"] is True
        assert guard["reason"] == "specific_fact_requested_but_no_direct_evidence"
        assert response["debug"]["memory"]["cross_kind_recall"]["reply_path"] == "local"


def test_temporal_specific_fact_query_remains_event_focused() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="event")
        service = CoreChatService(tmpdir, agent=FakeAgent(pre_reply=decision))
        service.memory_store.add_memory("u1", "我叫小明", kind="profile", memory_type="fact")
        service.memory_store.add_memory(
            "u1",
            "我昨天开车去了公司",
            kind="event",
            memory_type="event",
            occurred_at=1778044800.0,
        )

        response = service.chat("我昨天开车去哪里了", user_id="u1")

        assert all(item["kind"] == "event" for item in response["recalled_memories"])
        assert response["debug"]["memory"]["cross_kind_recall"]["applied"] is False


def test_identity_weekly_report_and_reminders_default_to_self_subject() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        named = service.memory_store.create_named_subject("u1", "Alex")
        service.memory_store.add_memory(
            "u1",
            "用户喜欢安静环境",
            kind="profile",
            memory_type="preference",
        )
        service.memory_store.add_memory(
            "u1",
            "喜欢热闹环境",
            subject_id=named.id,
            kind="profile",
            memory_type="preference",
        )
        service.memory_store.add_memory(
            "u1",
            "用户提交周报",
            kind="event",
            memory_type="task",
            start_at=1778131200.0 + 3600,
        )
        service.memory_store.add_memory(
            "u1",
            "准备外部材料",
            subject_id=named.id,
            kind="event",
            memory_type="task",
            start_at=1778131200.0 + 3600,
        )

        identity = service.chat("我是谁", user_id="u1")
        weekly = service.weekly_report(
            user_id="u1",
            start_at=1778131200.0 - 60,
            end_at=1778131200.0 + 7200,
        )
        reminders = service.check_reminders(user_id="u1", now=1778131200.0)

        assert {(item["subject_type"], item["content"]) for item in identity["recalled_memories"]} == {
            ("self", "用户喜欢安静环境"),
        }
        assert {item["content"] for item in weekly["source_memories"]} == {"用户提交周报"}
        assert [item["content"] for item in reminders["reminders"]] == ["用户提交周报"]


def test_all_subject_recall_takes_precedence_over_mentioned_named_person() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall()
        decision["recall_subject_scope"] = "all"
        service = CoreChatService(tmpdir, agent=FakeAgent(pre_reply=decision))
        first = service.memory_store.create_named_subject("u1", "Alex")
        second = service.memory_store.create_named_subject("u1", "Beta")
        service.memory_store.add_memory(
            "u1",
            "负责发布检查",
            subject_id=first.id,
            kind="event",
            memory_type="task",
        )
        service.memory_store.add_memory(
            "u1",
            "负责发布确认",
            subject_id=second.id,
            kind="event",
            memory_type="task",
        )

        response = service.chat("除了 Alex，还有谁负责发布？", user_id="u1")

        assert {item["subject_name"] for item in response["recalled_memories"]} == {"Alex", "Beta"}
        assert response["debug"]["memory"]["subject_recall"]["effective_scope"] == "all"


def test_unresolved_subject_names_fall_back_to_self_for_first_person_event_query() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        self_subject = service.memory_store.ensure_self_subject("u1")
        memory = service.memory_store.add_memory(
            "u1",
            "参加了 Data Analysis using Python webinar",
            subject_id=self_subject.id,
            kind="event",
            memory_type="event",
        )
        planner = TurnPlan(
            needs_event_memory=True,
            event_recall_strategy="text_search",
            recall_subject_scope="named",
            recall_subject_names=["Data Analysis using Python webinar"],
        )

        selected_ids, debug = service._recall_subject_selection(
            user_id="u1",
            message="Which event did I attend first, the Data Analysis using Python webinar or the workshop?",
            planner=planner,
        )

        assert selected_ids == [self_subject.id]
        assert debug["effective_scope"] == "self"
        assert debug["unresolved_names"] == ["Data Analysis using Python webinar"]
        assert debug["fallback_applied"] is True
        assert debug["fallback_reason"] == "unresolved_named_entities_with_self_reference"
        assert [item.id for item in service.memory_store.list_memories("u1", subject_ids=selected_ids)] == [memory.id]


def test_unresolved_named_person_does_not_fall_back_to_self() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        self_subject = service.memory_store.ensure_self_subject("u1")
        service.memory_store.add_memory(
            "u1",
            "我的私人任务",
            subject_id=self_subject.id,
            kind="event",
            memory_type="task",
        )
        planner = TurnPlan(
            needs_event_memory=True,
            event_recall_strategy="text_search",
            recall_subject_scope="named",
            recall_subject_names=["Alex"],
        )

        selected_ids, debug = service._recall_subject_selection(
            user_id="u1",
            message="What did Alex do?",
            planner=planner,
        )

        assert selected_ids == []
        assert debug["effective_scope"] == "named"
        assert debug["unresolved_names"] == ["Alex"]
        assert debug["fallback_applied"] is False
        assert debug["fallback_reason"] == "unresolved_named_subjects"


def test_resolved_named_subject_still_limits_recall() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        alex = service.memory_store.create_named_subject("u1", "Alex")
        service.memory_store.add_memory(
            "u1",
            "Alex 负责发布检查",
            subject_id=alex.id,
            kind="event",
            memory_type="task",
        )
        planner = TurnPlan(
            needs_event_memory=True,
            recall_subject_scope="named",
            recall_subject_names=["Alex"],
        )

        selected_ids, debug = service._recall_subject_selection(
            user_id="u1",
            message="What did Alex do?",
            planner=planner,
        )

        assert selected_ids == [alex.id]
        assert debug["effective_scope"] == "named"
        assert debug["resolved_subjects"][0]["subject_name"] == "Alex"
        assert debug["unresolved_names"] == []
        assert debug["fallback_applied"] is False


def test_ambiguous_provisional_name_does_not_select_the_first_subject() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall()
        decision["recall_subject_scope"] = "named"
        decision["recall_subject_names"] = ["speaker_2"]
        service = CoreChatService(tmpdir, agent=FakeAgent(pre_reply=decision))
        first = service.memory_store.create_provisional_subject("u1", "speaker_2", source_scope="capture-a")
        second = service.memory_store.create_provisional_subject("u1", "speaker_2", source_scope="capture-b")
        service.memory_store.add_memory(
            "u1",
            "first scoped task",
            subject_id=first.id,
            kind="event",
            memory_type="task",
        )
        service.memory_store.add_memory(
            "u1",
            "second scoped task",
            subject_id=second.id,
            kind="event",
            memory_type="task",
        )

        response = service.chat("speaker_2 的任务是什么？", user_id="u1")

        assert response["recalled_memories"] == []
        assert response["debug"]["memory"]["subject_recall"]["ambiguous_names"] == ["speaker_2"]


def test_chat_recalls_named_people_separately_for_comparison() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="profile")
        decision["recall_subject_scope"] = "named"
        decision["recall_subject_names"] = ["张三", "李四"]
        service = CoreChatService(tmpdir, agent=FakeAgent(pre_reply=decision))
        zhang = service.memory_store.create_named_subject("u1", "张三")
        li = service.memory_store.create_named_subject("u1", "李四")
        service.memory_store.add_memory(
            "u1",
            "喜欢苏打水",
            subject_id=zhang.id,
            kind="profile",
            memory_type="preference",
        )
        service.memory_store.add_memory(
            "u1",
            "喜欢茶",
            subject_id=li.id,
            kind="profile",
            memory_type="preference",
        )
        service.memory_store.add_memory(
            "u1",
            "喜欢矿泉水",
            kind="profile",
            memory_type="preference",
        )

        response = service.chat("张三和李四分别喜欢喝什么？", user_id="u1")

        assert {
            (memory["subject_name"], memory["content"])
            for memory in response["recalled_memories"]
        } == {
            ("张三", "喜欢苏打水"),
            ("李四", "喜欢茶"),
        }
        assert response["debug"]["memory"]["subject_recall"]["effective_scope"] == "named"


def test_markdown_document_import_and_recall_use_document_helpers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        result = service.import_memory_events(
            user_id="u1",
            text="# 南太行自驾攻略\n## 费用\n红旗渠门票 80 元",
            source="markdown_upload",
            context="南太行自驾攻略.md",
        )

        recall = service._recall_documents_for_query("u1", "南太行自驾攻略里红旗渠门票多少钱？")

        assert result["document"]["title"] == "南太行自驾攻略"
        assert recall.mode == "full_document"
        assert recall.reason == "document_title_match"
        assert "红旗渠门票 80 元" in recall.context


def test_text_import_uses_import_helpers_without_bypassing_memory_gate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        result = service.import_memory_events(
            user_id="u1",
            text="- 周五检查 demo\n- 周六整理复盘",
            source="manual_import",
        )

        saved = service.memory_store.list_memories("u1")

        assert result["candidate_count"] == 2
        assert result["saved_count"] == 2
        assert {memory.content for memory in saved} == {"检查 demo", "整理复盘"}
        assert all(memory.evidence_ids for memory in saved)


def test_conversation_import_preserves_pairs_and_only_saves_user_facts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        result = service.import_conversation_events(
            user_id="u1",
            session_id="history-1",
            turns=[
                {
                    "role": "user",
                    "content": "I attended the Python webinar two months ago. Should I use Seaborn?",
                    "occurred_at": 10.0,
                    "source_id": "turn-user-1",
                },
                {
                    "role": "assistant",
                    "content": "Use Seaborn. api_key=sk-abcdefghijklmnopqrstuvwxyz123456",
                    "occurred_at": 11.0,
                    "source_id": "turn-assistant-1",
                },
                {
                    "role": "assistant",
                    "content": "Assistant-only historical note",
                    "occurred_at": 12.0,
                    "source_id": "turn-assistant-2",
                },
                {
                    "role": "user",
                    "content": "I attended the time management workshop last Saturday.",
                    "occurred_at": 13.0,
                    "source_id": "turn-user-2",
                },
                {"role": "user", "content": "", "occurred_at": 14.0},
            ],
            source="conversation_archive",
            context="imported chat history",
        )

        first_turn = service.timeline_store.get_turn("u1", result["timeline_turn_ids"][0])
        memories = service.memory_store.list_memories("u1")
        assistant_chunks = service.timeline_store.search_chunks("u1", "historical note")

        assert result["imported_turn_count"] == 4
        assert result["skipped_empty_turn_count"] == 1
        assert result["paired_turn_count"] == 1
        assert result["user_only_turn_count"] == 1
        assert result["assistant_only_turn_count"] == 1
        assert result["failed_count"] == 0
        assert first_turn is not None
        assert first_turn.raw_text == "I attended the Python webinar two months ago. Should I use Seaborn?"
        assert "Use Seaborn" in first_turn.assistant_reply
        assert "sk-" not in first_turn.assistant_reply
        assert assistant_chunks and assistant_chunks[0].metadata["role"] == "assistant"
        assert all("Use Seaborn" not in memory.content for memory in memories)
        assert all("Assistant-only" not in memory.content for memory in memories)
        assert {memory.content for memory in memories} == {
            "I attended the Python webinar two months ago",
            "I attended the time management workshop last Saturday",
        }
        assert all(memory.evidence_ids for memory in memories)
        service.close()


def test_assistant_history_recall_uses_timeline_evidence_without_creating_memory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="timeline", goal="raw_evidence")
        decision["needs_timeline_recall"] = True
        decision["timeline_query"] = "Use Seaborn"
        decision["memory_recall_type"] = "timeline"
        decision["reason"] = "user asks what the assistant previously recommended"
        agent = FakeAgent(pre_reply=decision, reply="上次我建议使用 Seaborn。")
        service = CoreChatService(tmpdir, agent=agent)
        turn_result = service.timeline_store.add_turn(
            "u1",
            "Which library should I use for a quick chart?",
            created_at=10.0,
        )
        service.timeline_store.update_turn_reply(
            "u1",
            turn_result.turn.id,
            "Use Seaborn for the quick chart.",
            updated_at=11.0,
        )
        service.timeline_store.add_chunks(
            "u1",
            parent_type="turn",
            parent_id=turn_result.turn.id,
            chunks=[{"text": "Use Seaborn for the quick chart."}],
            source="chat",
            timestamp=11.0,
            metadata={"role": "assistant"},
            start_index=1,
        )

        response = service.chat("What did you recommend for the quick chart?", user_id="u1")

        assert response["recalled_memories"] == []
        assert response["recalled_timeline_chunks"]
        assert any("Use Seaborn" in chunk["text"] for chunk in response["recalled_timeline_chunks"])
        assert response["debug"]["planner"]["needs_timeline_recall"] is True
        assert response["debug"]["memory"]["recall_arbitration"]["primary_source"] == "raw_timeline"
        assert response["debug"]["memory"]["recall_arbitration"]["kept_counts"]["profile"] == 0


def test_conversation_import_semantically_types_default_preference_without_changing_events() -> None:
    preference = "I concentrate best in quiet places with uninterrupted time"
    event = "I attended the planning workshop yesterday"
    agent = TranscriptCandidateAgent({
        preference: (preference, "profile", "preference"),
        event: (event, "event", "event"),
    })
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir, agent=agent)
        result = service.import_conversation_events(
            user_id="u1",
            session_id="work-history",
            turns=[
                {"role": "user", "content": preference, "occurred_at": 10.0},
                {"role": "user", "content": event, "occurred_at": 11.0},
            ],
        )
        memories = {memory.content: memory for memory in service.memory_store.list_memories("u1")}
        overlays = [
            decision
            for imported in result["import_results"]
            for decision in imported["classification_decisions"]
            if decision["role"] == "conversation_import_semantic_type_overlay"
        ]

        assert memories[preference].kind == "profile"
        assert memories[preference].memory_type == "preference"
        assert memories[event].kind == "event"
        assert memories[event].memory_type == "event"
        assert any(item["applied"] and item["legacy_kind"] == "event" for item in overlays)
        assert any(item["fallback_reason"] == "semantic_type_matches_legacy" for item in overlays)
        service.close()


def test_conversation_import_preference_remains_user_isolated_during_profile_recall() -> None:
    preference = "I concentrate best in quiet places with uninterrupted time"
    agent = TranscriptCandidateAgent({
        preference: (preference, "profile", "preference"),
    })
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir, agent=agent)
        service.import_conversation_events(
            user_id="u1",
            session_id="u1-history",
            turns=[{"role": "user", "content": preference, "occurred_at": 10.0}],
        )
        service.import_conversation_events(
            user_id="u2",
            session_id="u2-history",
            turns=[{"role": "user", "content": preference, "occurred_at": 10.0}],
        )
        agent.pre_reply = pre_reply_recall(recall_type="profile", goal="summary")
        response = service.chat("Summarize my working preferences", user_id="u1")

        u1_profile = service.memory_store.list_memories("u1", kind="profile")
        u2_profile = service.memory_store.list_memories("u2", kind="profile")

        assert len(u1_profile) == 1
        assert len(u2_profile) == 1
        assert [item["id"] for item in response["recalled_memories"]] == [u1_profile[0].id]
        assert u2_profile[0].id not in {item["id"] for item in response["recalled_memories"]}
        service.close()


def test_conversation_import_sensitive_fragment_skips_semantic_typing_and_memory_write() -> None:
    agent = FakeAgent(pre_reply=pre_reply_write("unrelated", kind="profile", memory_type="preference"))
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir, agent=agent)
        result = service.import_conversation_events(
            user_id="u1",
            session_id="sensitive-history",
            turns=[{
                "role": "user",
                "content": "My api_key=sk-abcdefghijklmnopqrstuvwxyz123456",
                "occurred_at": 10.0,
            }],
        )

        assert not service.memory_store.list_memories("u1")
        assert not agent.calls
        assert "sensitive_fragment_filtered" in result["conversation_extraction"]["rejected_reasons"]
        service.close()


def test_conversation_import_semantic_typing_failure_keeps_legacy_event() -> None:
    content = "I concentrate best in quiet places with uninterrupted time"
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir, agent=UnavailableConversationTypingAgent())
        result = service.import_conversation_events(
            user_id="u1",
            session_id="typing-fallback",
            turns=[{"role": "user", "content": content, "occurred_at": 10.0}],
        )
        memory = service.memory_store.list_memories("u1")[0]
        overlay = next(
            decision
            for decision in result["import_results"][0]["classification_decisions"]
            if decision["role"] == "conversation_import_semantic_type_overlay"
        )

        assert memory.kind == "event"
        assert memory.memory_type == "event"
        assert overlay["applied"] is False
        assert overlay["fallback_reason"] == "semantic_decision_unavailable"
        service.close()


def test_conversation_import_defers_observation_reflection_until_all_fragments_saved() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        reflection_calls: list[dict[str, Any]] = []

        def record_reflection(**kwargs: Any) -> list[dict[str, Any]]:
            reflection_calls.append(kwargs)
            return []

        service._maybe_start_observation_reflect_for_memories = record_reflection  # type: ignore[method-assign]
        result = service.import_conversation_events(
            user_id="u1",
            session_id="history-reflection",
            turns=[
                {"role": "user", "content": "I attended the planning workshop.", "occurred_at": 10.0},
                {"role": "user", "content": "I scheduled the follow-up for Friday.", "occurred_at": 11.0},
            ],
            source="conversation_archive",
        )

        assert result["failed_count"] == 0
        assert result["saved_count"] == 2
        assert len(reflection_calls) == 1
        call = reflection_calls[0]
        assert call["session_id"] == "history-reflection"
        assert {memory.content for memory in call["memories"]} == {
            "I attended the planning workshop",
            "I scheduled the follow-up for Friday",
        }
        service.close()


def test_conversation_import_skips_observation_reflection_after_fragment_failure() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        reflection_calls: list[dict[str, Any]] = []
        import_calls = 0
        original_import = service.import_memory_events

        def fail_second_import(**kwargs: Any) -> dict[str, Any]:
            nonlocal import_calls
            import_calls += 1
            if import_calls == 2:
                raise RuntimeError("simulated fragment failure")
            return original_import(**kwargs)

        def record_reflection(**kwargs: Any) -> list[dict[str, Any]]:
            reflection_calls.append(kwargs)
            return []

        service.import_memory_events = fail_second_import  # type: ignore[method-assign]
        service._maybe_start_observation_reflect_for_memories = record_reflection  # type: ignore[method-assign]
        result = service.import_conversation_events(
            user_id="u1",
            session_id="history-incomplete",
            turns=[
                {"role": "user", "content": "I attended the planning workshop.", "occurred_at": 10.0},
                {"role": "user", "content": "I scheduled the follow-up for Friday.", "occurred_at": 11.0},
            ],
            source="conversation_archive",
        )

        assert result["failed_count"] == 1
        assert result["saved_count"] == 1
        assert result["failures"][0]["error"] == "simulated fragment failure"
        assert not reflection_calls
        service.close()


def test_conversation_import_rejects_unsupported_roles() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)

        try:
            service.import_conversation_events(
                user_id="u1",
                session_id="history-1",
                turns=[{"role": "system", "content": "hidden instruction", "occurred_at": 1.0}],
                source="conversation_archive",
            )
        except ValueError as exc:
            assert "role must be user or assistant" in str(exc)
        else:
            raise AssertionError("unsupported roles must fail validation")


def test_temporal_event_recall_falls_back_to_matching_text_when_recorded_later() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.memory_store.add_memory(
            "u1",
            "I took my bike in for repairs in February",
            kind="event",
            memory_type="event",
            occurred_at=1678406400.0,
        )
        temporal = TemporalResolution(
            has_temporal_expression=True,
            start_at=1675209600.0,
            end_at=1677628800.0,
            granularity="month",
            confidence=1.0,
            backend="test",
        )

        memories, debug = service._recall_event_memories(
            user_id="u1",
            message="Which vehicle did I take care of first in February, the bike or the car?",
            temporal=temporal,
            reference_time=1678406400.0,
            strategy="temporal_range",
        )

        assert [memory.content for memory in memories] == ["I took my bike in for repairs in February"]
        assert debug["text_fallback_used"] is True
        assert debug["candidate_trace"]["selected_ids"] == [memories[0].id]
        assert debug["candidate_trace"]["candidate_count"] >= 1
        assert debug["candidate_trace"]["candidates"][0]["id"] == memories[0].id


def test_upcoming_event_recall_trace_keeps_timed_and_untimed_candidates_bounded() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        timed = service.memory_store.add_memory(
            "u1",
            "I will review the project brief tomorrow",
            kind="event",
            memory_type="task",
            start_at=1778217600.0,
            end_at=1778221200.0,
        )
        untimed = service.memory_store.add_memory(
            "u1",
            "I need to send the project brief",
            kind="event",
            memory_type="task",
        )
        closed = service.memory_store.add_memory(
            "u1",
            "I already sent the project brief",
            kind="event",
            memory_type="task",
            tags=["task_status:completed"],
        )

        memories, debug = service._recall_event_memories(
            user_id="u1",
            message="What do I need to do next?",
            temporal=TemporalResolution(),
            reference_time=1778131200.0,
            strategy="upcoming_plan",
        )

        selected_ids = {memory.id for memory in memories}
        trace = debug["candidate_trace"]
        assert timed.id in selected_ids
        assert untimed.id in selected_ids
        assert closed.id not in selected_ids
        assert trace["candidate_count"] == len(trace["candidates"])
        assert trace["candidate_count"] <= 32
        assert trace["selected_ids"] == sorted(selected_ids)


def test_text_event_recall_exposes_bounded_candidate_trace_without_changing_selection() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        first = service.memory_store.add_memory("u1", "project atlas kickoff", kind="event", memory_type="event")
        service.memory_store.add_memory("u1", "project atlas budget", kind="event", memory_type="event")
        other = service.memory_store.add_memory("u2", "project atlas kickoff", kind="event", memory_type="event")
        expected = service.memory_store.search("u1", "What happened during the atlas kickoff?", limit=8)

        memories, debug = service._recall_event_memories(
            user_id="u1",
            message="What happened during the atlas kickoff?",
            temporal=TemporalResolution(),
            reference_time=100.0,
            strategy="text_search",
        )

        trace = debug["candidate_trace"]
        assert first.id in {memory.id for memory in memories}
        assert [memory.id for memory in memories] == [memory.id for memory in expected]
        assert trace["selected_ids"] == sorted(memory.id for memory in memories)
        assert trace["candidate_count"] == len(trace["candidates"])
        assert trace["candidate_count"] <= 32
        assert {item["id"] for item in trace["candidates"]} >= {first.id}
        assert other.id not in {item["id"] for item in trace["candidates"]}


def test_timeline_recall_trace_is_bounded_and_user_isolated() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        mine = service.timeline_store.add_turn(
            "u1",
            "I need a dairy-free lunch because lactose causes discomfort.",
            created_at=10.0,
        ).chunks[0]
        other = service.timeline_store.add_turn(
            "u2",
            "I need a dairy-free lunch because lactose causes discomfort.",
            created_at=11.0,
        ).chunks[0]
        planner = TurnPlan(
            needs_timeline_recall=True,
            timeline_query="dairy-free lunch",
            recall_goal="raw_evidence",
        )
        expected = service.timeline_store.search_chunks("u1", "dairy-free lunch", limit=5)

        chunks, debug = service._recall_timeline_chunks(user_id="u1", planner=planner)

        trace = debug["candidate_trace"]
        assert [chunk.id for chunk in chunks] == [chunk.id for chunk in expected]
        assert mine.id in {chunk.id for chunk in chunks}
        assert trace["selected_ids"] == sorted(chunk.id for chunk in chunks)
        assert trace["candidate_count"] == len(trace["candidates"])
        assert trace["candidate_count"] <= 20
        assert other.id not in {item["id"] for item in trace["candidates"]}


def test_specific_fact_recall_supplements_structured_memory_with_timeline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir, agent=FakeAgent(pre_reply=pre_reply_recall()))
        service.memory_store.add_memory("u1", "I attended the workshop", kind="event", memory_type="event")
        service.timeline_store.add_turn(
            "u1",
            "I attended the Data Analysis using Python webinar before the workshop.",
            created_at=10.0,
        )

        response = service.chat(
            "Which event happened first, the webinar or the workshop?",
            user_id="u1",
            session_id="recall",
            memory_writes_allowed=False,
        )

        assert response["recalled_timeline_chunks"]
        assert response["debug"]["timeline"]["recall"]["supplemental"] is True


def test_memory_job_payload_helpers_preserve_polling_contract() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        job = service._create_memory_job(
            user_id="u1",
            session_id="s1",
            mode="reply_first_background",
            candidate_count=1,
            created_at=1.0,
            evidence_ids=["chunk1"],
        )
        saved = service.memory_store.add_memory(
            "u1",
            "周五检查 demo",
            kind="event",
            memory_type="task",
            evidence_ids=["chunk1"],
        )

        updated = service._update_memory_job(
            user_id="u1",
            job_id=job["job_id"],
            status="saved",
            saved_memories=[saved],
            completed=True,
            extraction_backend="core_test",
        )
        polled = service.read_memory_job(user_id="u1", job_id=job["job_id"])

        assert updated is not None
        assert polled is not None
        assert polled["status"] == "saved"
        assert polled["saved_count"] == 1
        assert polled["saved_memory_ids"] == [saved.id]
        assert polled["memory_processing"]["status"] == "saved"
        assert polled["memory_processing"]["stage_reason"] == "memory_saved"
        assert service.read_memory_job(user_id="u2", job_id=job["job_id"]) is None
