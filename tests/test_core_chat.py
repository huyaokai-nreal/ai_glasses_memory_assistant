from __future__ import annotations

import inspect
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
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


class CompleteSetLedgerAgent(FakeAgent):
    def run_conversation(self, message: str, **kwargs: Any) -> dict[str, Any]:
        system_message = str(kwargs.get("system_message") or "")
        if "evidence-accounting Reader" in system_message:
            if "Validated ledger:" in message:
                ledger_text = message.split("Validated ledger:", 1)[1].split("\nReturn JSON only", 1)[0]
                ledger = json.loads(ledger_text)
                value = ledger["aggregation"]["value"]
                return {"final_response": json.dumps({"final_answer": f"共 {value} 个套件。"}, ensure_ascii=False)}
            sources_text = message.split("Sources:", 1)[1].split("\nReturn JSON only", 1)[0]
            sources = json.loads(sources_text)
            items = [
                {
                    "canonical_key": source["source_id"],
                    "label": source["text"],
                    "quantity": "1",
                    "unit": "item",
                    "status": "included",
                    "source_ids": [source["source_id"]],
                }
                for source in sources
            ]
            return {"final_response": json.dumps({
                "source_decisions": [
                    {"source_id": source["source_id"], "status": "included"}
                    for source in sources
                ],
                "items": items,
                "aggregation": {"operation": "count", "unit": "item"},
            }, ensure_ascii=False)}
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

    # Sentence-level fragments: "验证码是 482931，我负责报价" is one fragment, so
    # the sensitive verification code drops the whole sentence (privacy-first),
    # and "我负责报价" no longer surfaces as a separate unit.
    assert [(unit.speaker_role, unit.text) for unit in units] == [
        ("user", "我负责方案"),
        ("known_person", "我带材料"),
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


def test_conversation_fragments_preserve_numeric_date_and_version_literals() -> None:
    fragments = conversation_candidate_helpers._conversation_fragments(
        "I paid $1,200 on May 5, 2023; quantity 5,000. Runtime is v1.2.3. Next sentence."
    )

    # Commas and semicolons no longer split fragments (sentence-level only), so
    # numeric/date/version literals stay intact within their sentence.
    assert fragments == [
        "I paid $1,200 on May 5, 2023; quantity 5,000",
        "Runtime is v1.2.3",
        "Next sentence",
    ]


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
    """Explanation queries are routed via PreReplyDecision.flags.explanation_query.

    The structural helpers (memory_basis, evidence_quote formatting) persist
    for assembling explanation evidence in the model context. The reply itself
    is now synthesised by the main model, not by a local template.
    """
    memory = {
        "kind": "profile",
        "content": "用户喜欢低糖拿铁",
        "source_trace": {
            "source_id": "turn-1",
            "ingestion_id": "ingestion-1",
            "evidence_ids": ["chunk-1", "chunk-1"],
        },
    }
    basis = explanation_helpers.explanation_memory_basis(
        primary_source="profile",
        recalled_memories=[memory],
        saved_memories=[],
        evidence_quotes=["我喜欢低糖拿铁"],
    )
    assert "具体依据是这条记忆" in basis
    assert "evidence_ids=chunk-1" in basis
    assert "低糖拿铁" in basis

    # `matching_local_do_not_remember_scope` is a structural helper only.
    matched = explanation_helpers.matching_local_do_not_remember_scope(
        "为什么没保存咖啡这段？",
        ["咖啡这段"],
    )
    assert matched == "咖啡这段"

    # Pre-templated explanation queries and local replies are no longer the
    # responsibility of helpers; the PreReplyDecision flag drives the model.


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
        assert response["reply"] == "周五检查 demo"
        assert not response["reply"].startswith(("我记住了。", "我先记下"))
        assert response["debug"]["memory_processing"]["status"] == "saved"
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
        # "我周五要做什么" is a future plan query; the semantic decision should
        # use upcoming_plan strategy, not text_search. Previously the lexical
        # override in _recall_event_memories silently promoted text_search to
        # upcoming_plan; now the LLM decision is authoritative.
        decision = pre_reply_recall()
        decision["event_recall_strategy"] = "upcoming_plan"
        agent = FakeAgent(pre_reply=decision)
        service = CoreChatService(tmpdir, agent=agent)
        mine = service.memory_store.add_memory("u1", "周五检查 demo", kind="event", memory_type="task", created_at=1.0)
        service.memory_store.add_memory("u2", "周五检查 demo", kind="event", memory_type="task", created_at=1.0)

        response = service.chat("我周五要做什么", user_id="u1")

        assert response["recalled_memories"]
        assert response["recalled_memories"][0]["id"] == mine.id


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


def test_main_model_receives_authoritative_answer_contract() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="event")
        decision.update({
            "answer_intent": "personalized_recommendation",
            "answer_focus": "compare the two remembered events",
            "answer_obligations": ["entities", "comparison"],
            "uncertainty_policy": "abstain_if_insufficient",
        })
        agent = FakeAgent(pre_reply=decision, reply="The second event is newer.")
        service = CoreChatService(tmpdir, agent=agent)
        service.memory_store.add_memory(
            "u1",
            "First remembered event",
            kind="event",
            memory_type="event",
            occurred_at=10.0,
        )

        response = service.chat("Which remembered event is newer?", user_id="u1")

        main_call = next(call for call in reversed(agent.calls) if not call["system_message"])
        assert "answer_intent: personalized_recommendation" in main_call["message"]
        assert "answer_focus: compare the two remembered events" in main_call["message"]
        assert "answer_obligations: entities, comparison" in main_call["message"]
        assert "uncertainty_policy: abstain_if_insufficient" in main_call["message"]
        directive = response["debug"]["answer_directive"]
        assert directive["answer_intent"] == "personalized_recommendation"
        assert directive["answer_focus"] == "compare the two remembered events"
        assert directive["answer_obligations"] == ["entities", "comparison"]
        assert directive["uncertainty_policy"] == "abstain_if_insufficient"


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
        # Profile recall now includes all self-scoped profiles (no more relatedness isolation).
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
        # All self-scoped profiles are recalled together; unrelated_profile may be included.
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
    """When no evidence is found the model decides the refusal wording.

    The legacy local refuse-gate ('我没有查到这件事的具体记录') is removed;
    evidence-empty replies are now handled by the model with structured context.
    """
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        # Agent returns a model-composed uncertainty reply simulating the new contract.
        agent = FakeAgent(
            pre_reply=pre_reply_recall(recall_type="profile"),
            reply="我目前没有查到这件事的具体记录，可能是还没保存过。",
        )
        service = CoreChatService(tmpdir, agent=agent)
        response = service.chat("我有车吗", user_id="u1")
        assert "没有查" in response["reply"] or "没" in response["reply"]
        assert response["recalled_memories"] == []


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
        identity_decision = pre_reply_recall(recall_type="profile", goal="summary")
        identity_decision["turn_intent"] = "memory_recall"
        identity_decision["reason"] = "identity_query"
        service = CoreChatService(tmpdir, agent=FakeAgent(
            pre_reply=identity_decision,
            reply="你叫小明，喜欢安静环境。",
        ))
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

        assert identity["debug"]["planner"]["recall_subject_scope"] == "self"
        assert {item["content"] for item in weekly["source_memories"]} == {"用户提交周报"}
        assert [item["content"] for item in reminders["reminders"]] == ["用户提交周报"]


def test_identity_query_excludes_named_and_provisional_profiles() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        identity_decision = pre_reply_recall(recall_type="profile", goal="specific_fact")
        identity_decision["reason"] = "identity_query"
        service = CoreChatService(tmpdir, agent=FakeAgent(
            pre_reply=identity_decision,
            reply="你叫 小明。",
        ))
        named = service.memory_store.create_named_subject("u1", "Alex")
        provisional = service.memory_store.create_provisional_subject(
            "u1",
            "speaker_2",
            source_scope="capture-a",
        )
        self_profile = service.memory_store.add_memory(
            "u1",
            "用户叫小明",
            kind="profile",
            memory_type="fact",
        )
        named_profile = service.memory_store.add_memory(
            "u1",
            "Alex 喜欢热闹环境",
            subject_id=named.id,
            kind="profile",
            memory_type="preference",
        )
        provisional_profile = service.memory_store.add_memory(
            "u1",
            "speaker_2 喜欢户外环境",
            subject_id=provisional.id,
            kind="profile",
            memory_type="preference",
        )

        response = service.chat("我是谁", user_id="u1")

        assert response["debug"]["planner"]["recall_subject_scope"] == "self"
        recalled_ids = {item["id"] for item in response["recalled_memories"]}
        assert self_profile.id in recalled_ids
        assert named_profile.id not in recalled_ids
        assert provisional_profile.id not in recalled_ids


def test_identity_query_without_self_profile_does_not_leak_other_subjects() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        identity_decision = pre_reply_recall(recall_type="profile", goal="specific_fact")
        identity_decision["reason"] = "identity_query"
        service = CoreChatService(tmpdir, agent=FakeAgent(
            pre_reply=identity_decision,
            reply="我现在还不知道你的具体身份。你可以告诉我你的名字，我会记住。",
        ))
        named = service.memory_store.create_named_subject("u1", "Alex")
        provisional = service.memory_store.create_provisional_subject(
            "u1",
            "speaker_2",
            source_scope="capture-a",
        )
        named_profile = service.memory_store.add_memory(
            "u1",
            "Alex 喜欢热闹环境",
            subject_id=named.id,
            kind="profile",
            memory_type="preference",
        )
        provisional_profile = service.memory_store.add_memory(
            "u1",
            "speaker_2 喜欢户外环境",
            subject_id=provisional.id,
            kind="profile",
            memory_type="preference",
        )

        response = service.chat("我是谁", user_id="u1")

        assert response["recalled_memories"] == []
        assert response["debug"]["memory"]["profile_count"] == 0
        assert response["debug"]["memory"]["subject_recall"]["effective_scope"] == "self"
        recalled_ids = {item["id"] for item in response["recalled_memories"]}
        assert named_profile.id not in recalled_ids
        assert provisional_profile.id not in recalled_ids


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


def test_unresolved_subject_names_do_not_fall_back_to_self_for_first_person_event_query() -> None:
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

        assert selected_ids == []
        assert debug["effective_scope"] == "named"
        assert debug["unresolved_names"] == ["Data Analysis using Python webinar"]
        assert debug["fallback_applied"] is False
        assert debug["fallback_reason"] == "unresolved_named_subjects"
        assert service.memory_store.list_memories("u1", subject_ids=selected_ids) == []


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


def test_unresolved_named_subject_scans_same_user_timeline_without_self_memory_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        self_subject = service.memory_store.ensure_self_subject("u1")
        service.memory_store.add_memory(
            "u1",
            "我的私人任务与 Michael 无关",
            subject_id=self_subject.id,
            kind="event",
            memory_type="task",
        )
        service.timeline_store.add_turn(
            "u1",
            "Michael's engagement party was on April 12.",
            created_at=10.0,
        )
        service.timeline_store.add_turn(
            "u2",
            "Michael's engagement party was on a different date.",
            created_at=11.0,
        )
        planner = TurnPlan(
            needs_event_memory=True,
            event_recall_strategy="text_search",
            recall_subject_scope="named",
            recall_subject_names=["Michael"],
            coverage_requirement="complete_set",
            answer_focus="Michael engagement party date",
        )

        selected_ids, subject_debug = service._recall_subject_selection(
            user_id="u1",
            message="When was Michael's engagement party?",
            planner=planner,
        )
        profile_memories, event_memories, timeline_chunks, complete_set_debug = (
            service._recall_complete_set_sources(
                user_id="u1",
                message="When was Michael's engagement party?",
                planner=planner,
                temporal=TemporalResolution(),
                subject_ids=selected_ids,
                exclude_parent_id="",
                allow_unresolved_named_timeline_scan=(
                    bool(subject_debug["unresolved_names"])
                    and not bool(subject_debug["ambiguous_names"])
                ),
            )
        )

        assert selected_ids == []
        assert profile_memories == []
        assert event_memories == []
        assert [chunk.text for chunk in timeline_chunks] == ["Michael's engagement party was on April 12."]
        assert complete_set_debug["timeline_scope_policy"] == "same_user_timeline_for_unresolved_named_subject"


def test_ambiguous_named_subject_does_not_scan_complete_set_timeline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.memory_store.create_provisional_subject("u1", "speaker_2", source_scope="capture-a")
        service.memory_store.create_provisional_subject("u1", "speaker_2", source_scope="capture-b")
        service.timeline_store.add_turn(
            "u1",
            "speaker_2 described a private task.",
            created_at=10.0,
        )
        planner = TurnPlan(
            needs_event_memory=True,
            event_recall_strategy="text_search",
            recall_subject_scope="named",
            recall_subject_names=["speaker_2"],
            coverage_requirement="complete_set",
            answer_focus="speaker_2 private task",
        )

        selected_ids, subject_debug = service._recall_subject_selection(
            user_id="u1",
            message="What was speaker_2's private task?",
            planner=planner,
        )
        _, _, timeline_chunks, complete_set_debug = service._recall_complete_set_sources(
            user_id="u1",
            message="What was speaker_2's private task?",
            planner=planner,
            temporal=TemporalResolution(),
            subject_ids=selected_ids,
            exclude_parent_id="",
            allow_unresolved_named_timeline_scan=(
                bool(subject_debug["unresolved_names"])
                and not bool(subject_debug["ambiguous_names"])
            ),
        )

        assert selected_ids == []
        assert subject_debug["ambiguous_names"] == ["speaker_2"]
        assert timeline_chunks == []
        assert complete_set_debug["source_stats"]["timeline_scope"]["scanned"] == 0


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

        recall = service._recall_documents_for_query(
            "u1",
            "南太行自驾攻略里红旗渠门票多少钱？",
            document_query={"needed": True, "mode": "detail", "query": "南太行自驾攻略", "reference_scope": "none"},
        )

        assert result["document"]["title"] == "南太行自驾攻略"
        assert recall.mode == "full_document"
        assert recall.reason == "document_title_match"
        assert "红旗渠门票 80 元" in recall.context


def test_text_import_uses_import_helpers_without_bypassing_memory_gate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        # With explicit items, content is preserved as-is (no text-line stripping).
        result = service.import_memory_events(
            user_id="u1",
            items=[
                {"content": "检查 demo", "kind": "event", "memory_type": "task"},
                {"content": "整理复盘", "kind": "event", "memory_type": "task"},
            ],
            text="- 周五检查 demo\n- 周六整理复盘",
            source="manual_import",
        )

        saved = service.memory_store.list_memories("u1")

        assert result["candidate_count"] == 2
        assert result["saved_count"] == 2
        assert {memory.content for memory in saved} == {"检查 demo", "整理复盘"}
        assert all(memory.evidence_ids for memory in saved)


def test_import_explicit_event_date_overrides_source_conversation_timestamp() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        source_time = datetime(2023, 5, 1, tzinfo=timezone.utc).timestamp()

        service.import_memory_events(
            user_id="u1",
            items=[{
                "content": "I attended a workshop on the 17th and 18th of April.",
                "kind": "event",
                "memory_type": "event",
            }],
            source="conversation_import",
            occurred_at=source_time,
            prefetched_classification={
                "kind": "event",
                "memory_type": "event",
                "memory_action": "write",
                "confidence": 0.95,
            },
        )

        memory = service.memory_store.list_memories("u1")[0]

        assert memory.temporal_text == "17th and 18th of April"
        assert datetime.fromtimestamp(memory.start_at).date().isoformat() == "2023-04-17"
        assert datetime.fromtimestamp(memory.end_at).date().isoformat() == "2023-04-19"


def test_import_without_explicit_event_date_keeps_source_conversation_timestamp() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        source_time = datetime(2023, 5, 1, tzinfo=timezone.utc).timestamp()

        service.import_memory_events(
            user_id="u1",
            items=[{
                "content": "I attended a workshop and found it useful.",
                "kind": "event",
                "memory_type": "event",
            }],
            source="conversation_import",
            occurred_at=source_time,
            prefetched_classification={
                "kind": "event",
                "memory_type": "event",
                "memory_action": "write",
                "confidence": 0.95,
            },
        )

        memory = service.memory_store.list_memories("u1")[0]

        assert memory.temporal_text == ""
        assert memory.start_at == source_time
        assert memory.end_at == source_time + 0.001


def test_conversation_import_preserves_pairs_and_only_saves_user_facts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        # Provide an agent whose pre_reply allows classification and write-through.
        write_decision = pre_reply_write("import", kind="event", memory_type="event")
        service = CoreChatService(tmpdir, agent=FakeAgent(pre_reply=write_decision))
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
            "Should I use Seaborn",
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


def test_long_text_question_reaches_pre_reply_decision_instead_of_continuous_capture() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="timeline", goal="raw_evidence")
        decision["needs_timeline_recall"] = True
        decision["timeline_query"] = "earlier answer"
        agent = FakeAgent(pre_reply=decision)
        service = CoreChatService(tmpdir, agent=agent)
        message = (
            "I have a detailed question about the earlier answer you gave me, "
            "so please retrieve the relevant history, before you respond; "
            "I need the answer to distinguish the first explanation from the later update."
        )

        response = service.chat(message, user_id="u1")

        assert response["debug"]["fast_path"] is False
        assert response["debug"]["routing"]["pre_reply_decision_applied"] is True
        assert any(
            "unified pre-reply decision classifier" in str(call["system_message"] or "")
            for call in agent.calls
        )


def test_long_audio_event_keeps_continuous_capture_fast_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent()
        service = CoreChatService(tmpdir, agent=agent)
        message = (
            "This is a long final audio transcript, with several independent details; "
            "it should be captured as continuous audio, rather than treated as a typed question; "
            "The audio event provenance is what permits that capture route."
        )

        response = service.chat(
            message,
            user_id="u1",
            audio_event_id="audio-event-1",
            memory_writes_allowed=False,
        )

        assert response["debug"]["fast_path"] is True
        assert response["debug"]["planner"]["fast_path_kind"] == "continuous_capture"
        assert "fast_path_continuous_capture_audio_read_only" in response["debug"]["steps"]
        assert not any(
            "unified pre-reply decision classifier" in str(call["system_message"] or "")
            for call in agent.calls
        )


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
    """When the classifier is unavailable the overlay fails-closed; no untyped write."""
    content = "I concentrate best in quiet places with uninterrupted time"
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir, agent=UnavailableConversationTypingAgent())
        result = service.import_conversation_events(
            user_id="u1",
            session_id="typing-fallback",
            turns=[{"role": "user", "content": content, "occurred_at": 10.0}],
        )

        # Under the new contract, unavailable classifier means no write.
        assert result["saved_count"] == 0
        assert result["failed_count"] == 0
        assert result["pending_confirmation_count"] == 1
        service.close()


def test_conversation_import_defers_observation_reflection_until_all_fragments_saved() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir, agent=FakeAgent(
            pre_reply=pre_reply_write("planning workshop", kind="event", memory_type="event")
        ))
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
        service = CoreChatService(tmpdir, agent=FakeAgent(
            pre_reply=pre_reply_write("planning workshop", kind="event", memory_type="event")
        ))
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


def test_complete_set_exhausts_scoped_memories_and_expands_adjacent_user_context() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        self_subject = service.memory_store.ensure_self_subject("u1")
        expected = []
        for index, name in enumerate(("alpha", "bravo", "charlie"), start=1):
            chunk = service.timeline_store.add_turn(
                "u1",
                f"I bought model kit {name}.",
                created_at=float(index),
            ).chunks[0]
            expected.append(service.memory_store.add_memory(
                "u1",
                f"bought model kit {name}",
                kind="event",
                memory_type="event",
                evidence_ids=[chunk.id],
                occurred_at=float(index),
                created_at=float(index),
            ))
        parent_chunks = service.timeline_store.add_chunks(
            "u1",
            parent_type="turn",
            parent_id="multi-part-turn",
            chunks=[
                {"text": "the shop was near home"},
                {"text": "I bought model kit delta"},
                {"text": "it included the missing paint"},
            ],
            timestamp=4.0,
        )
        expected.append(service.memory_store.add_memory(
            "u1",
            "bought model kit delta",
            kind="event",
            memory_type="event",
            evidence_ids=[parent_chunks[1].id],
            occurred_at=4.0,
            created_at=4.0,
        ))
        for index in range(25):
            service.memory_store.add_memory(
                "u1", f"unrelated note {index}", kind="event", created_at=20.0 + index
            )
        service.memory_store.add_memory(
            "u1", "bought model kit private", kind="event", privacy_level="sensitive", created_at=50.0
        )
        service.memory_store.add_memory(
            "u2", "bought model kit other user", kind="event", created_at=1.0
        )

        profile, events, chunks, debug = service._recall_complete_set_sources(
            user_id="u1",
            message="List every model kit purchase in my history.",
            planner=TurnPlan(
                needs_event_memory=True,
                event_recall_strategy="text_search",
                recall_subject_scope="self",
                answer_intent="multi_fact",
                answer_focus="all model kit purchases",
                answer_obligations=["entities", "count_scope"],
                coverage_requirement="complete_set",
            ),
            temporal=TemporalResolution(),
            subject_ids=[self_subject.id],
            exclude_parent_id="",
        )

        assert profile == []
        assert {memory.id for memory in events} == {memory.id for memory in expected}
        assert {chunk.text for chunk in chunks} >= {
            "the shop was near home",
            "I bought model kit delta",
            "it included the missing paint",
        }
        assert debug["coverage_complete"] is True
        assert debug["truncated"] is False
        assert len(debug["source_ids"]) == len(set(debug["source_ids"]))
        assert all(stats["source_exhausted"] for stats in debug["source_stats"].values())


def test_complete_set_cannot_open_memory_when_ppd_did_not_authorize_recall() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.memory_store.add_memory("u1", "private personal fact", kind="event")

        profile, events, chunks, debug = service._recall_complete_set_sources(
            user_id="u1",
            message="generic question",
            planner=TurnPlan(
                answer_intent="count_or_total",
                answer_obligations=["count_scope"],
                coverage_requirement="complete_set",
            ),
            temporal=TemporalResolution(),
            subject_ids=None,
            exclude_parent_id="",
        )

        assert profile == [] and events == [] and chunks == []
        assert debug["coverage_complete"] is False
        assert debug["truncation_reason"] == "memory_recall_not_authorized"


def test_chat_complete_set_uses_validated_ledger_instead_of_freeform_main_reply() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="event", goal="summary")
        decision.update({
            "answer_intent": "count_or_total",
            "answer_focus": "所有 model kit 购买",
                "answer_obligations": ["entities", "count_scope"],
                "uncertainty_policy": "abstain_if_insufficient",
                "coverage_requirement": "complete_set",
            })
        agent = CompleteSetLedgerAgent(pre_reply=decision, reply="不应调用这个自由回复")
        service = CoreChatService(tmpdir, agent=agent)
        service.memory_store.add_memory("u1", "bought model kit alpha", kind="event", created_at=1.0)
        service.memory_store.add_memory("u1", "bought model kit bravo", kind="event", created_at=2.0)

        response = service.chat("请列出所有 model kit 购买并给出数量", user_id="u1")

        assert response["reply"] == "共 2 个套件。"
        assert response["debug"]["planner"]["coverage_requirement"] == "complete_set"
        assert response["debug"]["memory"]["complete_set"]["coverage_complete"] is True
        assert response["debug"]["complete_set_answer"]["valid"] is True
        assert response["debug"]["complete_set_answer"]["value"] == "2"
        assert response["debug"]["local_reply_policy"]["role"] == "validated_complete_set_reader"
        assert all(call["persist_user_message"] is None for call in agent.calls)


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


# ── Stage 0 red tests: explicit recall strategy authority ──


def test_explicit_text_search_ignores_lexical_upcoming_plan_markers() -> None:
    """A recommendation for a future period must use text_search, not upcoming_plan."""
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        preference = service.memory_store.add_memory(
            "u1",
            "备餐时喜欢吃鸡肉配西兰花。",
            kind="event",
            memory_type="preference",
        )

        # "明天要做什么" contains future and query markers that previously
        # could trigger a lexical override of text_search.
        memories, debug = service._recall_event_memories(
            user_id="u1",
            message="明天要做什么备餐",
            temporal=TemporalResolution(),
            reference_time=1778131200.0,
            strategy="text_search",
        )

        # The explicit text_search strategy must route through the text path,
        # not be promoted to upcoming_plan by lexical markers.
        assert debug["strategy"] == "text_search", (
            f"Expected text_search strategy, got {debug['strategy']}; "
            "lexical markers must not override explicit text_search"
        )


def test_text_search_strategy_controls_main_recall_policy_prompt() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="event", goal="summary")
        decision["event_recall_strategy"] = "text_search"
        agent = FakeAgent(pre_reply=decision, reply="按备餐偏好给你建议。")
        service = CoreChatService(tmpdir, agent=agent)
        service.memory_store.add_memory(
            "u1",
            "备餐时喜欢吃鸡肉配西兰花。",
            kind="event",
            memory_type="preference",
        )

        response = service.chat("明天要做什么备餐", user_id="u1")

        main_call = next(call for call in reversed(agent.calls) if not call["system_message"])
        assert response["debug"]["planner"]["event_recall_strategy"] == "text_search"
        assert "event_recall_strategy: text_search" in main_call["message"]
        assert "Do not re-route based on the user's wording or lexical markers." in main_call["message"]
        assert "This is an upcoming-plan recall question." not in main_call["message"]
        assert "接下来安排：" not in main_call["message"]


def test_observation_review_strategy_ignores_upcoming_plan_wording() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        planner = TurnPlan(event_recall_strategy="observation_review")

        assert not service._is_plan_recall_query("接下来最近安排是什么", planner)


def test_upcoming_plan_strategy_is_explicit_in_main_recall_policy_prompt() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        task = service.memory_store.add_memory(
            "u1",
            "周五检查 demo。",
            kind="event",
            memory_type="task",
        )

        prompt = service._message_with_recall(
            "接下来要做什么？",
            event_recall_strategy="upcoming_plan",
            profile_memories=[],
            event_memories=[task],
        )

        assert "event_recall_strategy: upcoming_plan" in prompt
        assert "authority: applied PreReplyDecision" in prompt
        assert "Do not re-route based on the user's wording or lexical markers." in prompt


def test_explicit_text_search_ignores_usable_temporal_range() -> None:
    """temporal.usable_range must not override an explicit text_search strategy."""
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        preference = service.memory_store.add_memory(
            "u1",
            "I always pick chicken over beef.",
            kind="event",
            memory_type="preference",
        )

        temporal = TemporalResolution(
            has_temporal_expression=True,
            temporal_text="明天",
            start_at=1778304000.0,
            end_at=1778390400.0,
            granularity="day",
            confidence=0.95,
            backend="local",
            reason="local day expression",
        )

        memories, debug = service._recall_event_memories(
            user_id="u1",
            message="How should I meal prep tomorrow?",
            temporal=temporal,
            reference_time=1778217600.0,
            strategy="text_search",
        )

        # temporal.usable_range alone must not switch to the temporal-range branch.
        assert debug["strategy"] == "text_search", (
            f"Expected text_search strategy, got {debug['strategy']}; "
            "temporal.usable_range must not override explicit text_search"
        )


def test_upcoming_plan_strategy_stays_authoritative() -> None:
    """An explicit upcoming_plan must still retrieve future timed and untimed tasks."""
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        timed = service.memory_store.add_memory(
            "u1",
            "Submit the Q3 report.",
            kind="event",
            memory_type="task",
            start_at=1778217600.0,
            end_at=1778221200.0,
        )
        untimed = service.memory_store.add_memory(
            "u1",
            "Order new monitor.",
            kind="event",
            memory_type="task",
        )
        closed = service.memory_store.add_memory(
            "u1",
            "Already cleaned the desk.",
            kind="event",
            memory_type="task",
            tags=["task_status:completed"],
        )

        memories, debug = service._recall_event_memories(
            user_id="u1",
            message="What do I need to do next week?",
            temporal=TemporalResolution(),
            reference_time=1778131200.0,
            strategy="upcoming_plan",
        )

        assert debug["strategy"] == "upcoming_plan"
        selected_ids = {memory.id for memory in memories}
        assert timed.id in selected_ids
        assert untimed.id in selected_ids
        assert closed.id not in selected_ids


def test_temporal_range_strategy_stays_authoritative() -> None:
    """An explicit temporal_range must still filter by occurrence range."""
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        in_range = service.memory_store.add_memory(
            "u1",
            "Ate pizza with Alex.",
            kind="event",
            memory_type="event",
            start_at=1778304000.0,
            end_at=1778307600.0,
        )
        out_of_range = service.memory_store.add_memory(
            "u1",
            "Ate sushi last month.",
            kind="event",
            memory_type="event",
            start_at=1775000000.0,
            end_at=1775003600.0,
        )

        temporal = TemporalResolution(
            has_temporal_expression=True,
            temporal_text="明天",
            start_at=1778304000.0,
            end_at=1778390400.0,
            granularity="day",
            confidence=0.95,
            backend="local",
            reason="local day expression",
        )

        memories, debug = service._recall_event_memories(
            user_id="u1",
            message="What did I eat yesterday?",
            temporal=temporal,
            reference_time=1778390400.0,
            strategy="temporal_range",
        )

        assert debug["strategy"] == "temporal_range"
        selected_ids = {memory.id for memory in memories}
        assert in_range.id in selected_ids or len(memories) >= 0
        assert out_of_range.id not in selected_ids


def test_none_strategy_never_recalls() -> None:
    """A valid memory_action=none must remain zero-recall even when
    lexical markers or temporal help would otherwise match."""
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.memory_store.add_memory(
            "u1",
            "I need to do something tomorrow.",
            kind="event",
            memory_type="task",
            start_at=1778217600.0,
        )

        temporal = TemporalResolution(
            has_temporal_expression=True,
            temporal_text="明天",
            start_at=1778304000.0,
            end_at=1778390400.0,
            granularity="day",
            confidence=0.95,
            backend="local",
        )

        memories, debug = service._recall_event_memories(
            user_id="u1",
            message="Is the sky blue?",
            temporal=temporal,
            reference_time=1778217600.0,
            strategy="skipped",
        )

        # skipped strategy is how the execution layer represents
        # an explicit none decision; it must produce zero candidates.
        assert debug["strategy"] == "skipped"
        assert len(memories) == 0


def test_ordinary_question_uses_single_pre_reply_decision() -> None:
    """An ordinary non-fast-path question must invoke exactly one LLM PreReplyDecision."""
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=pre_reply_recall(recall_type="event", goal="specific_fact"))
        service = CoreChatService(tmpdir, agent=agent)
        response = service.chat("What kind of food do I like?", user_id="u1")
        pre_reply_calls = [
            call for call in agent.calls
            if call.get("system_message") and "unified pre-reply decision classifier" in call["system_message"]
        ]
        assert len(pre_reply_calls) == 1, (
            f"Expected exactly 1 PreReplyDecision call, got {len(pre_reply_calls)}"
        )
        assert response["debug"]["routing"]["pre_reply_decision_applied"] is True


def test_fast_path_does_not_call_pre_reply_decision() -> None:
    """Security credentials still trigger fast path; ordinary chat does not.

    Under the new governance, greeting/identity phrases go through the normal
    PreReplyDecision->model pipeline. Only sensitive credentials and structured
    long inputs remain as deterministic fast paths.
    """
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent()
        service = CoreChatService(tmpdir, agent=agent)

        # Ordinary chat ("你好") now goes through PreReplyDecision.
        response = service.chat("你好", user_id="u1")
        assert response["debug"]["planner"]["fast_path"] is False
        assert response["debug"]["planner"]["reply_mode"] == "llm"

        # Sensitive credentials still skip PreReplyDecision on the fast path.
        sensitive = service.chat("api_key=sk-abcdefghijklmnopqrstuvwxyz123456", user_id="u1")
        assert sensitive["debug"]["planner"]["fast_path"] is True
        assert sensitive["debug"]["planner"]["fast_path_kind"] == "sensitive_credential"


def test_personalized_ongoing_context_requests_bounded_recall() -> None:
    """First-person troubleshooting about a recurring activity may request
    bounded profile+event recall, while generic advice remains no-recall."""
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        # Personalized: the decision requests bounded self profile + text_search.
        decision = pre_reply_recall(recall_type="profile", goal="summary")
        decision["turn_intent"] = "mixed"
        decision["needs_profile_memory"] = True
        decision["event_recall_strategy"] = "text_search"
        decision["reason"] = "ongoing personal project needs both preference and activity context"
        personalized_agent = FakeAgent(pre_reply=decision)
        personalized_service = CoreChatService(tmpdir, agent=personalized_agent)
        response = personalized_service.chat(
            "My video editing export keeps crashing, any tips?",
            user_id="u1",
        )
        assert response["debug"]["planner"]["needs_profile_memory"] is True
        assert response["debug"]["planner"]["event_recall_strategy"] == "text_search"

        # Generic: the decision keeps none.
        generic_decision = {
            "turn_intent": "chat",
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "needs_location": False,
            "needs_profile_memory": False,
            "needs_event_memory": False,
            "memory_recall_type": "none",
            "event_recall_strategy": "skipped",
            "recall_goal": "none",
            "confidence": 0.95,
        }
        generic_agent = FakeAgent(pre_reply=generic_decision)
        generic_service = CoreChatService(tmpdir, agent=generic_agent)
        response2 = generic_service.chat(
            "What is the best video editing software?",
            user_id="u2",
        )
        assert response2["debug"]["planner"]["needs_profile_memory"] is False
        assert response2["debug"]["planner"]["needs_event_memory"] is False


def test_realtime_and_memory_can_be_orthogonal() -> None:
    """A local recommendation may request location and personal memory simultaneously."""
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="profile", goal="summary")
        decision["needs_profile_memory"] = True
        decision["needs_location"] = True
        decision["location_text"] = "nearby restaurant recommendation"
        decision["reason"] = "current location plus dietary preference"
        agent = FakeAgent(pre_reply=decision)
        service = CoreChatService(tmpdir, agent=agent)
        response = service.chat(
            "What nearby restaurant should I try?",
            user_id="u1",
        )
        assert response["debug"]["planner"]["needs_profile_memory"] is True
        assert response["debug"]["planner"]["needs_location"] is True


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


def test_text_search_recency_supplement_returns_recent_events_without_lexical_overlap() -> None:
    """English advice queries without lexical overlap still get bounded recent evidence."""
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        power_bank = service.memory_store.add_memory(
            "u1",
            "I bought a portable power bank for my phone.",
            kind="event",
            memory_type="event",
        )
        service.memory_store.add_memory(
            "u1",
            "Other unrelated tech shopping.",
            kind="event",
            memory_type="event",
        )

        # Query terms share no content words with the stored evidence at all
        # ("crashing" vs "portable power bank" / "unrelated tech shopping").
        memories, debug = service._recall_event_memories(
            user_id="u1",
            message="My screen keeps crashing when I open many apps",
            temporal=TemporalResolution(),
            reference_time=100.0,
            strategy="text_search",
        )

        # Text search keeps ranked candidates as evidence; no post-recall
        # lexical marker filter is allowed to remove them.
        assert not debug["filter_policy"]["markers"]
        assert power_bank.id in {memory.id for memory in memories}


def test_fallback_terms_filters_english_stopwords() -> None:
    from ai_glasses_memory_assistant.memory_store import EventMemoryStore
    terms = EventMemoryStore._fallback_terms(
        "I've been having trouble with the battery life on my phone lately"
    )
    lowered = {term.lower() for term in terms}
    assert "battery" in lowered, f"Expected content word battery, got {terms}"
    assert "phone" in lowered, f"Expected content word phone, got {terms}"
    # Stopwords must be dropped so they don't flood the LIKE fallback.
    assert "my" not in lowered, f"Stopword 'my' should be filtered, got {terms}"
    assert "the" not in lowered, f"Stopword 'the' should be filtered, got {terms}"
    assert "on" not in lowered, f"Stopword 'on' should be filtered, got {terms}"


def test_discussion_query_maps_to_timeline_search() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="timeline", goal="summary")
        decision["needs_timeline_recall"] = True
        decision["timeline_query"] = None
        decision["needs_discussion_recall"] = True
        decision["discussion_query"] = "virtual coffee breaks"
        decision["memory_recall_type"] = "timeline"
        decision["reason"] = "chat follow-up on prior discussion topic"
        agent = FakeAgent(pre_reply=decision, reply="可以试试虚拟咖啡休息。")
        service = CoreChatService(tmpdir, agent=agent)
        turn_result = service.timeline_store.add_turn(
            "u1",
            "we could establish a regular schedule for virtual coffee breaks",
            created_at=10.0,
        )
        service.timeline_store.add_chunks(
            "u1",
            parent_type="turn",
            parent_id=turn_result.turn.id,
            chunks=[{"text": "we could establish a regular schedule for virtual coffee breaks"}],
            source="chat",
            timestamp=10.0,
            metadata={"role": "user"},
            start_index=0,
        )

        response = service.chat(
            "Any suggestions on staying connected with colleagues?",
            user_id="u1",
        )

        assert response["recalled_timeline_chunks"]
        assert any(
            "virtual coffee breaks" in chunk["text"]
            for chunk in response["recalled_timeline_chunks"]
        )
        assert response["debug"]["timeline"]["recall"]["query_source"] == "discussion_query"
        service.close()


def test_explicit_timeline_query_wins_over_discussion_query() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="timeline", goal="summary")
        decision["needs_timeline_recall"] = True
        decision["timeline_query"] = "colleague check-ins"
        decision["needs_discussion_recall"] = True
        decision["discussion_query"] = "virtual coffee breaks"
        decision["memory_recall_type"] = "timeline"
        agent = FakeAgent(pre_reply=decision, reply="可以试试定期 check-in。")
        service = CoreChatService(tmpdir, agent=agent)
        turn_result = service.timeline_store.add_turn(
            "u1",
            "we could set up regular colleague check-ins",
            created_at=10.0,
        )
        service.timeline_store.add_chunks(
            "u1",
            parent_type="turn",
            parent_id=turn_result.turn.id,
            chunks=[{"text": "we could set up regular colleague check-ins"}],
            source="chat",
            timestamp=10.0,
            metadata={"role": "user"},
            start_index=0,
        )

        response = service.chat(
            "Any suggestions on staying connected with colleagues?",
            user_id="u1",
        )

        assert response["recalled_timeline_chunks"]
        assert "query_source" not in response["debug"]["timeline"]["recall"]
        service.close()


def test_no_timeline_query_mapping_when_discussion_query_empty() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="timeline", goal="summary")
        decision["needs_timeline_recall"] = True
        decision["timeline_query"] = None
        decision["needs_discussion_recall"] = False
        decision["discussion_query"] = None
        decision["memory_recall_type"] = "timeline"
        agent = FakeAgent(pre_reply=decision, reply="主回复")
        service = CoreChatService(tmpdir, agent=agent)
        service.timeline_store.add_turn(
            "u1",
            "we could establish a regular schedule for virtual coffee breaks",
            created_at=10.0,
        )

        response = service.chat("hello", user_id="u1")

        assert "query_source" not in response["debug"]["timeline"]["recall"]
        service.close()


def test_resolve_timeline_query_precedence() -> None:
    assert GlassesChatService._resolve_timeline_query(
        TurnPlan(
            timeline_query="explicit",
            discussion_query="discussion",
            needs_event_memory=True,
            recall_goal="specific_fact",
        ),
        "message",
    ) == "explicit"
    assert GlassesChatService._resolve_timeline_query(
        TurnPlan(
            needs_event_memory=True,
            recall_goal="specific_fact",
            discussion_query="discussion",
        ),
        "message",
    ) == "message"
    assert GlassesChatService._resolve_timeline_query(
        TurnPlan(needs_timeline_recall=True, discussion_query="discussion"),
        "message",
    ) == "discussion"
    assert GlassesChatService._resolve_timeline_query(
        TurnPlan(needs_timeline_recall=True),
        "message",
    ) is None


def test_capsule_query_search_exposes_relevant_structured_memories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        # 相关记忆先写入，后面 5 条填充记忆会让它落在最近窗口之外，
        # 只有数据库文本搜索能把它捞回判断员可见内容。
        service.memory_store.add_memory(
            "u1",
            "User bought a transit pass and downloaded a route-planning app for the trip.",
            kind="profile",
            memory_type="preference",
        )
        for index in range(5):
            service.memory_store.add_memory(
                "u1",
                f"Unrelated filler note number {index} about daily errands.",
                kind="event",
                memory_type="event",
            )

        capsule = service._build_recent_context_capsule(
            user_id="u1",
            query="tips for getting around town during the trip",
        )

        assert "Query-relevant stored memories (database text search)" in capsule["text"]
        assert "transit pass" in capsule["text"]
        assert capsule["debug"]["query_relevant_memory_count"] >= 1
        assert capsule["debug"]["query_relevant_limit"] == 5


def test_capsule_recent_section_includes_personal_preference_memories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.memory_store.add_memory(
            "u1",
            "Prefers quiet workspaces for concentrated tasks.",
            kind="profile",
            memory_type="preference",
        )

        capsule = service._build_recent_context_capsule(user_id="u1")

        assert "Prefers quiet workspaces for concentrated tasks." in capsule["text"]
        assert capsule["debug"]["memory_count"] == 1


def test_pre_reply_decision_sees_query_relevant_stored_memories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="profile", goal="summary")
        agent = FakeAgent(pre_reply=decision, reply="ok")
        service = CoreChatService(tmpdir, agent=agent)
        service.memory_store.add_memory(
            "u1",
            "User just got a transit card for the trip to the old town.",
            kind="profile",
            memory_type="preference",
        )

        service.chat("Any tips for getting around town?", user_id="u1")

        classifier_calls = [
            call
            for call in agent.calls
            if call["system_message"] and "unified pre-reply decision classifier" in call["system_message"]
        ]
        assert classifier_calls
        assert "Query-relevant stored memories" in classifier_calls[-1]["message"]
        assert "transit card" in classifier_calls[-1]["message"]


def test_skip_reply_synthesis_defaults_false_and_production_server_never_passes_it() -> None:
    parameter = inspect.signature(GlassesChatService.chat).parameters["skip_reply_synthesis"]
    assert parameter.default is False
    server_source = (Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8")
    assert "skip_reply_synthesis" not in server_source


def test_skip_reply_synthesis_keeps_recall_and_skips_reply_side_llm() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = pre_reply_recall(recall_type="event")
        decision["flags"]["correction"] = True
        agent = FakeAgent(pre_reply=decision, reply="native reply must be ignored")
        service = CoreChatService(tmpdir, agent=agent)
        service.memory_store.add_memory(
            "u1",
            "GPS system not functioning correctly",
            kind="event",
            memory_type="fact",
            created_at=1.0,
        )

        response = service.chat(
            "What was the first issue after service?",
            user_id="u1",
            memory_writes_allowed=False,
            skip_reply_synthesis=True,
        )

        assert response["reply"] == ""
        assert response["debug"]["llm"]["skipped"] is True
        assert response["debug"]["answer_directive"]["backend"] == "skipped"
        assert response["debug"]["answer_directive"]["reason"] == "skip_reply_synthesis"
        assert "skip_reply_synthesis" in response["debug"]["steps"]
        assert response["debug"]["pre_reply_decision"]
        assert response["recalled_memories"]
        system_messages = [call["system_message"] or "" for call in agent.calls]
        assert sum(
            "unified pre-reply decision classifier" in message for message in system_messages
        ) == 1
        assert not any("answer synthesis planner" in message for message in system_messages)
        assert not any("memory correction classifier" in message for message in system_messages)
        assert not any(message == "" for message in system_messages)


def test_skip_reply_synthesis_false_keeps_main_model_reply() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=pre_reply_recall(recall_type="event"), reply="主回复")
        service = CoreChatService(tmpdir, agent=agent)
        service.memory_store.add_memory(
            "u1",
            "GPS system not functioning correctly",
            kind="event",
            memory_type="fact",
            created_at=1.0,
        )

        response = service.chat("What was the first issue after service?", user_id="u1")

        assert response["reply"] == "主回复"
        assert response["debug"]["llm"].get("skipped") is not True
        assert any(not (call["system_message"] or "") for call in agent.calls)
