import json
import os
import time
from datetime import datetime, timezone
from dataclasses import replace

import pytest

from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
from ai_glasses_memory_assistant.answer_synthesizer import synthesize_complete_set_answer
from ai_glasses_memory_assistant.evidence_set import EvidenceCandidate
from ai_glasses_memory_assistant.temporal_parser import TemporalResolution
from tests.helpers import CoreChatService, FakeAgent


@pytest.fixture
def host_zone():
    original = os.environ.get("TZ")
    def set_zone(value):
        os.environ["TZ"] = value
        time.tzset()
    yield set_zone
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


@pytest.mark.parametrize("zone,expected", [
    ("Asia/Shanghai", "1月2日，星期五，0点30分"),
    ("+08:00", "1月2日，星期五，0点30分"),
    ("UTC", "1月1日，星期四，16点30分"),
    ("", "1月1日，星期四，16点30分"),
    ("not/a_zone", "1月1日，星期四，16点30分"),
])
def test_current_clock_uses_service_zone_and_preserves_host_fallback(host_zone, zone, expected):
    host_zone("UTC")
    ref = datetime(2026, 1, 1, 16, 30, tzinfo=timezone.utc).timestamp()
    assert expected in GlassesChatService._local_current_time_reply(ref, timezone=zone)


def test_calendar_route_consumes_ppd_target_without_another_model_call(tmp_path, host_zone):
    host_zone("UTC")
    agent = FakeAgent(pre_reply={
        "turn_intent": "chat", "reply_mode": "local_calendar", "answer_source": "local_clock",
        "memory_action": "none", "memory_recall_type": "none", "recall_goal": "none",
        "answer_intent": "direct_fact", "confidence": .99,
        "temporal_query": {"has_expression": True, "start_at": "2026-01-02T00:00:00+08:00",
                           "end_at": "2026-01-03T00:00:00+08:00", "timezone": "Asia/Shanghai",
                           "granularity": "day"},
    })
    service = CoreChatService(str(tmp_path), agent=agent)
    service.timezone = "Asia/Shanghai"
    try:
        result = service.chat("请告诉我所问那天的日期", user_id="u")
        assert result["reply"] == "你问的日期是2026年1月2日，星期五。"
        assert result["debug"]["planner"]["reply_mode"] == "local_calendar"
        assert result["debug"]["local_reply_policy"]["skips_main_llm"] is True
        assert len(agent.calls) == 1
    finally:
        service.close()


def test_calendar_dst_range_and_invalid_target_do_not_fall_back_to_now(host_zone):
    host_zone("UTC")
    temporal = TemporalResolution(has_temporal_expression=True, confidence=1, timezone="America/New_York",
        start_at=datetime.fromisoformat("2026-03-08T00:00:00-05:00").timestamp(),
        end_at=datetime.fromisoformat("2026-03-09T00:00:00-04:00").timestamp())
    assert GlassesChatService._local_calendar_reply(temporal) == "你问的日期是2026年3月8日，星期天。"
    temporal = replace(temporal, end_at=datetime.fromisoformat("2026-03-10T00:00:00-04:00").timestamp())
    assert "3月8日" in GlassesChatService._local_calendar_reply(temporal)
    assert "3月9日" in GlassesChatService._local_calendar_reply(temporal)
    assert "没能可靠确定" in GlassesChatService._local_calendar_reply(TemporalResolution())


class LedgerAgent:
    def __init__(self, decisions, items, answer="1"):
        self.decisions, self.items, self.answer = decisions, items, answer
        self.prompts = []

    def run_conversation(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if "Validated ledger:" in prompt:
            return {"final_response": json.dumps({"final_answer": self.answer})}
        return {"final_response": json.dumps({"source_decisions": self.decisions, "items": self.items,
                                              "aggregation": {"operation": "count", "unit": "item"}})}


def item(label, key, ids):
    return dict(canonical_key=key, label=label, quantity="1", unit="item", status="included", source_ids=ids)


def test_entity_list_preserves_multiple_facts_and_uncertain_supplement_without_forcing_total():
    candidates = [EvidenceCandidate("a", "structured_memory", "Plan A and B tomorrow."),
                  EvidenceCandidate("b", "structured_memory", "Plan C has no known date.")]
    agent = LedgerAgent([dict(source_id="a", status="included"), dict(source_id="b", status="uncertain")],
                        [item("Plan A tomorrow", "a", ["a"]), item("Plan B tomorrow", "b", ["a"])], answer="0")
    result = synthesize_complete_set_answer(agent, message="List the plans", candidates=candidates,
        coverage_complete=True, answer_contract=dict(answer_intent="multi_fact", answer_obligations=["entities", "temporal_relation"]))
    assert result.valid and result.selected_source_ids == ("a",)
    assert "Plan A tomorrow" in result.final_answer and "Plan B tomorrow" in result.final_answer
    assert "尚不确定" in result.final_answer and "Plan C has no known date" in result.final_answer
    assert result.final_answer != "0" and len(agent.prompts) == 1


def test_requested_count_still_requires_verified_total_and_unknown_scope_cannot_be_zero():
    candidates = [EvidenceCandidate("a", "structured_memory", "One confirmed item.")]
    agent = LedgerAgent([dict(source_id="a", status="included")], [item("the item", "a", ["a"])], answer="1")
    contract = dict(answer_intent="count_or_total", answer_obligations=["count_scope"])
    result = synthesize_complete_set_answer(agent, message="How many?", candidates=candidates,
                                           coverage_complete=True, answer_contract=contract)
    assert result.valid and result.final_answer == "1" and len(agent.prompts) == 2
    unknown = LedgerAgent([dict(source_id="a", status="uncertain")], [], answer="0")
    result = synthesize_complete_set_answer(unknown, message="How many?", candidates=candidates,
                                           coverage_complete=True, answer_contract=contract)
    assert not result.valid and result.error == "uncertain_scope" and "无法可靠" in result.final_answer
    assert len(unknown.prompts) == 1


def test_complete_set_and_direct_share_time_relationship_policy(tmp_path):
    service = CoreChatService(str(tmp_path))
    try:
        temporal = TemporalResolution(has_temporal_expression=True, start_at=100, end_at=200,
                                       confidence=1, timezone="UTC")
        memory = service.memory_store.add_memory("u", "An earlier event, recorded later.",
                                               created_at=1000, start_at=120, end_at=140)
        candidate = EvidenceCandidate("a", "structured_memory", memory.content,
                                      temporal_note=service._memory_temporal_note(memory, temporal, "UTC"))
        policy = service._temporal_evidence_policy(temporal, "UTC")
        agent = LedgerAgent([dict(source_id="a", status="uncertain")], [])
        result = synthesize_complete_set_answer(agent, message="List that day's events", candidates=[candidate],
            coverage_complete=True, answer_contract=dict(answer_intent="multi_fact", answer_obligations=["entities"],
                                                         temporal_evidence_policy=policy))
        direct = service._message_with_recall("that day", profile_memories=[], event_memories=[memory], query_temporal=temporal)
        assert policy in direct and json.dumps(policy)[1:-1] in agent.prompts[0]
        assert "stored interval:" in agent.prompts[0] and "recorded_at:" in agent.prompts[0]
        assert "event binding: unknown" in agent.prompts[0] and "尚不确定" in result.final_answer
    finally:
        service.close()
