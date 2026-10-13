import json
from dataclasses import replace

import pytest

from ai_glasses_memory_assistant.intent_policy import MemoryWritePolicy, should_write_memory_candidate
from ai_glasses_memory_assistant.memory_candidate import MemoryWriteCandidate
from ai_glasses_memory_assistant.temporal_parser import TemporalResolution
from tests.helpers import CoreChatService, isolated_app_home


@pytest.mark.parametrize("value", [
    [], {"extra": 1}, {"version": 2}, {"version": True},
    {"by_type": {"unknown": {}}}, {"by_type": []},
    {"by_type": {"fact": {"enabled": "true"}}},
    {"by_type": {"fact": {"min_confidence": .59}}},
    {"by_type": {"fact": {"min_confidence": float("nan")}}},
    {"by_type": {"fact": {"min_confidence": True}}},
    {"by_type": {"fact": {"bypass_privacy": True}}},
])
def test_invalid_policy_is_rejected_instead_of_silently_changing_behavior(monkeypatch, value):
    monkeypatch.setenv("AI_GLASSES_MEMORY_WRITE_POLICY_JSON", json.dumps(value))
    with pytest.raises(ValueError):
        MemoryWritePolicy.from_env()


def test_type_policy_changes_real_writes_without_content_keyword_rules(tmp_path, monkeypatch):
    config = {"version": 1, "by_type": {"fact": {"min_confidence": .9}, "task": {"enabled": False}}}
    monkeypatch.setenv("AI_GLASSES_MEMORY_WRITE_POLICY_JSON", json.dumps(config))
    with isolated_app_home(str(tmp_path)):
        service = CoreChatService(str(tmp_path))
        candidates = [MemoryWriteCandidate("Acquired a red backpack.", "event", "fact", confidence=.8),
                      MemoryWriteCandidate("Enjoys hiking.", "profile", "preference", confidence=.8),
                      MemoryWriteCandidate("Submit the application tomorrow.", "event", "task", confidence=.95)]
        result = service._save_memory_candidates(candidates=candidates, message="Personal facts and plans.",
            user_id="u", agent=None, reference_time=100, query_temporal=TemporalResolution(), saved_temporal_debug=[])
        assert [m.content for m in result.saved] == ["Enjoys hiking."]
        assert {r['reason'] for r in result.rejected} == {"candidate_confidence_below_type_policy", "memory_type_disabled_by_policy"}
        service.close()


def test_configured_body_handling_cannot_enable_query_label_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GLASSES_MEMORY_WRITE_POLICY_JSON", json.dumps({"by_type": {"fact": {"normalize_candidate_time": True}}}))
    with isolated_app_home(str(tmp_path)):
        service = CoreChatService(str(tmp_path))
        candidate = MemoryWriteCandidate("Bought an orange suitcase.", "event", "fact", confidence=.9)
        temporal = TemporalResolution(start_at=100,end_at=200,confidence=1,normalized_text="time-label",backend="pre_reply_decision")
        result = service._save_memory_candidates(candidates=[candidate],message=candidate.content,user_id="u",agent=None,
            reference_time=100,query_temporal=temporal,saved_temporal_debug=[])
        assert result.saved[0].content == candidate.content
        service.close()


def test_confidence_feature_and_privacy_gate_are_independent_of_type_config(monkeypatch):
    monkeypatch.setenv("AI_GLASSES_MEMORY_WRITE_POLICY_JSON", json.dumps({"by_type": {"fact": {"require_confidence": True}}}))
    policy = MemoryWritePolicy.from_env().for_type("fact")
    candidate = MemoryWriteCandidate("Owns a folding chair.","event","fact")
    assert should_write_memory_candidate(candidate,candidate.content,write_policy=policy).reason == "confidence_required_by_policy"
    assert should_write_memory_candidate(replace(candidate,confidence=.9),candidate.content,write_policy=policy).allowed
    secret = replace(candidate,content="密码是123456",confidence=.99)
    assert not should_write_memory_candidate(secret,secret.content,write_policy=policy).allowed


def test_policy_snapshot_and_candidate_temporal_strategy_are_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_GLASSES_MEMORY_WRITE_POLICY_JSON", json.dumps({"by_type": {"fact": {"normalize_candidate_time": False}}}))
    with isolated_app_home(str(tmp_path)):
        service = CoreChatService(str(tmp_path))
        monkeypatch.setenv("AI_GLASSES_MEMORY_WRITE_POLICY_JSON", "{}")
        candidate = MemoryWriteCandidate("明天完成设备交接。","event","fact",confidence=.99)
        result = service._save_memory_candidates(candidates=[candidate],message=candidate.content,user_id="u",agent=None,
            reference_time=1778131200,query_temporal=TemporalResolution(),saved_temporal_debug=[])
        assert result.saved[0].content == candidate.content and result.saved[0].start_at is not None
        assert not service.memory_write_policy.for_type("fact").normalize_candidate_time
        service.close()
