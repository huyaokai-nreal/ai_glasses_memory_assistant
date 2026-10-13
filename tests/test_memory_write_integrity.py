import json
import sqlite3
from unittest.mock import patch

import pytest

from ai_glasses_memory_assistant.memory_candidate import MemoryWriteCandidate
from ai_glasses_memory_assistant.temporal_parser import TemporalResolution
from tests.helpers import CoreChatService, FakeAgent, isolated_app_home, pre_reply_write


@pytest.mark.parametrize("content", [
    "Bought a red travel bag; collect it on Monday.",
    "周四要把银色钥匙交给林老师。",
    "Booked a workshop with Jordan for next month.",
    "今天为办公室购买了两把椅子。",
])
@pytest.mark.parametrize("label,fragment", [
    ("period_label", ""), ("Monday", "Monday"), ("date-range", "non_matching"),
])
def test_query_time_metadata_never_replaces_body_and_commit_survives_reopen(tmp_path, content, label, fragment):
    with isolated_app_home(str(tmp_path)):
        service = CoreChatService(str(tmp_path))
        temporal = TemporalResolution(has_temporal_expression=True, start_at=100, end_at=200,
                                      confidence=1, backend="pre_reply_decision",
                                      normalized_text=label, temporal_text=fragment)
        saved = service._save_memory_candidates(
            candidates=[MemoryWriteCandidate(content=content, kind="event", memory_type="fact", confidence=.99)],
            message=content, user_id="u", agent=None, reference_time=100,
            query_temporal=temporal, saved_temporal_debug=[],
        ).saved
        assert len(saved) == 1 and saved[0].content == content
        service.close()
        with sqlite3.connect(str(tmp_path / "events.db")) as db:
            assert db.execute("select content,start_at,end_at from memories where user_id='u'").fetchone() == (content, 100, 200)


def test_local_candidate_temporal_normalization_remains_compatible(tmp_path):
    with isolated_app_home(str(tmp_path)):
        service = CoreChatService(str(tmp_path))
        result = service._save_memory_candidates(
            candidates=[MemoryWriteCandidate(content="明天完成设备交接。", kind="event", memory_type="fact", confidence=.99)],
            message="明天完成设备交接。", user_id="u", agent=None, reference_time=1778131200,
            query_temporal=TemporalResolution(), saved_temporal_debug=[],
        )
        assert result.saved[0].content == "完成设备交接"
        assert result.saved[0].start_at is not None
        service.close()


def write_decision(content, **changes):
    return {**pre_reply_write(content, memory_type="fact"), **changes}


def test_deferred_confirmation_uses_pending_state_even_if_model_claims_saved(tmp_path):
    with isolated_app_home(str(tmp_path)):
        agent = FakeAgent(pre_reply=write_decision("Purchased a blue travel bag."), reply="I have already saved it.")
        service = CoreChatService(str(tmp_path), agent=agent)
        response = service.chat("I purchased a blue travel bag.", user_id="u", defer_memory_writes=True)
        assert response["reply"] == "收到，正在保存。"
        assert response["debug"]["memory_processing"]["status"] == "pending"
        assert service._sessions[response["session_id"]].history[-1]["content"] == response["reply"]
        audit = [json.loads(line) for line in service.audit_path.read_text().splitlines()]
        assert next(row for row in audit if "reply" in row)["reply"] == response["reply"]
        job = service.read_memory_job(user_id="u", job_id=response["debug"]["memory_processing"]["job_id"])
        assert job["status"] == "saved" and job["saved_count"] == 1
        assert service.memory_store.list_memories("u")[0].content == "Purchased a blue travel bag."
        service.close()


def test_sync_save_and_failure_have_truthful_receipts(tmp_path):
    with isolated_app_home(str(tmp_path)):
        agent = FakeAgent(pre_reply=write_decision("Bought a folding chair."), reply="Saved.")
        service = CoreChatService(str(tmp_path), agent=agent)
        good = service.chat("Bought a folding chair.", user_id="good")
        assert good["reply"] == "Saved." and len(good["saved_memories"]) == 1
        with patch.object(service.memory_store, "add_memory", side_effect=OSError("injected storage failure")):
            bad = service.chat("Bought a folding chair.", user_id="bad")
        assert bad["debug"]["memory_processing"]["status"] == "failed"
        assert "保存失败" in bad["reply"] and bad["reply"] != "Saved."
        assert not service.memory_store.list_memories("bad")
        job = service.read_memory_job(user_id="bad", job_id=bad["debug"]["memory_processing"]["job_id"])
        assert job["status"] == "failed"
        service.close()


def test_rejected_candidate_does_not_claim_saved(tmp_path):
    with isolated_app_home(str(tmp_path)):
        decision = write_decision("Purchased a folding chair.", confidence=.01)
        service = CoreChatService(str(tmp_path), agent=FakeAgent(pre_reply=decision, reply="Saved."))
        response = service.chat("Purchased a folding chair.", user_id="u")
        assert response["debug"]["memory_processing"]["status"] == "rejected"
        assert "未保存" in response["reply"] and not response["saved_memories"]
        service.close()


def test_mixed_question_keeps_answer_and_receives_pending_context(tmp_path):
    with isolated_app_home(str(tmp_path)):
        decision = write_decision("Purchased a folding chair.", turn_intent="mixed")
        agent = FakeAgent(pre_reply=decision, reply="Use a dry cloth to clean it.")
        service = CoreChatService(str(tmp_path), agent=agent)
        response = service.chat("Purchased a folding chair. How should I clean it?", user_id="u", defer_memory_writes=True)
        assert response["reply"] == "Use a dry cloth to clean it."
        assert any("<memory-write-state>" in str(call["message"]) for call in agent.calls)
        service.close()
