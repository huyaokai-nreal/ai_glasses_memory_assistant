from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest

from tests.helpers import CoreChatService, FakeAgent, isolated_app_home, pre_reply_recall


@dataclass(frozen=True)
class EvidenceSpec:
    content: str
    kind: str
    memory_type: str
    occurred_at: float | None = None
    start_at: float | None = None


@dataclass(frozen=True)
class ReaderScenario:
    scenario_id: str
    query: str
    alternate_query: str | None
    decision: dict[str, Any]
    evidence: tuple[EvidenceSpec, ...]
    reply: str
    required_concepts: tuple[tuple[str, ...], ...] = ()
    refusal_allowed: bool = False
    unsupported_terms: tuple[str, ...] = ()


_REFUSAL_MARKERS = (
    "no record",
    "no relevant record",
    "i don't know",
    "i do not know",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "not enough information",
    "不知道",
    "不清楚",
    "无法确定",
    "没有记录",
    "没查到",
    "无法回答",
)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def canonical_refusal(text: str, *, answer_aliases: tuple[str, ...] = ()) -> bool:
    """Test-only detector for a clear refusal, never used by production routing."""
    normalized = _normalized(text)
    if not normalized or not any(marker in normalized for marker in _REFUSAL_MARKERS):
        return False
    return not any(_normalized(alias) in normalized for alias in answer_aliases if alias)


def _decision(
    *,
    recall_type: str,
    answer_intent: str,
    focus: str,
    obligations: list[str],
    uncertainty_policy: str,
    needs_profile: bool = False,
    needs_event: bool = False,
    strategy: str = "skipped",
    goal: str = "specific_fact",
) -> dict[str, Any]:
    decision = pre_reply_recall(recall_type=recall_type, goal=goal)
    decision.update(
        {
            "answer_intent": answer_intent,
            "answer_focus": focus,
            "answer_obligations": obligations,
            "uncertainty_policy": uncertainty_policy,
            "needs_profile_memory": needs_profile,
            "needs_event_memory": needs_event,
            "memory_recall_type": recall_type,
            "recall_type": recall_type,
            "event_recall_strategy": strategy,
            "recall_goal": goal,
            "confidence": 0.95,
        }
    )
    return decision


def _generic_decision() -> dict[str, Any]:
    decision = pre_reply_recall(recall_type="event", goal="none")
    decision.update(
        {
            "turn_intent": "chat",
            "memory_action": "none",
            "memory_recall_type": "none",
            "recall_type": "none",
            "needs_profile_memory": False,
            "needs_event_memory": False,
            "event_recall_strategy": "skipped",
            "recall_goal": "none",
            "answer_intent": "generic_recommendation",
            "answer_focus": "give general focus advice without personal memory",
            "answer_obligations": ["incremental_next_step"],
            "uncertainty_policy": "none",
        }
    )
    return decision


def _abstain_decision(*, recall_type: str, needs_profile: bool, needs_event: bool) -> dict[str, Any]:
    return _decision(
        recall_type=recall_type,
        answer_intent="abstain",
        focus="state whether the requested activity is supported by recalled evidence",
        obligations=["entities"],
        uncertainty_policy="abstain_if_insufficient",
        needs_profile=needs_profile,
        needs_event=needs_event,
        strategy="text_search" if needs_event else "skipped",
        goal="specific_fact",
    )


READER_SCENARIOS: tuple[ReaderScenario, ...] = (
    ReaderScenario(
        scenario_id="direct_fact",
        query="我喜欢什么咖啡？",
        alternate_query="What coffee do I like?",
        decision=_decision(
            recall_type="profile",
            answer_intent="direct_fact",
            focus="answer the user's coffee preference",
            obligations=["entities"],
            uncertainty_policy="abstain_if_insufficient",
            needs_profile=True,
            goal="specific_fact",
        ),
        evidence=(EvidenceSpec("用户喜欢低糖拿铁。", "profile", "preference"),),
        reply="你喜欢低糖拿铁。",
        required_concepts=(("低糖拿铁", "low-sugar latte", "low sugar latte"),),
    ),
    ReaderScenario(
        scenario_id="count_or_total",
        query="How many open tasks do I have, and what are they?",
        alternate_query="我有几个未完成的任务，分别是什么？",
        decision=_decision(
            recall_type="event",
            answer_intent="count_or_total",
            focus="count and enumerate the open tasks",
            obligations=["entities", "count_scope"],
            uncertainty_policy="abstain_if_insufficient",
            needs_event=True,
            strategy="text_search",
        ),
        evidence=(
            EvidenceSpec("Open task: renew passport.", "event", "task"),
            EvidenceSpec("Open task: book dentist appointment.", "event", "task"),
            EvidenceSpec("Open task: submit tax form.", "event", "task"),
        ),
        reply="There are 3 open tasks: renew passport, book dentist appointment, and submit tax form.",
        required_concepts=(
            ("3", "three", "三个", "三"),
            ("renew passport",),
            ("book dentist",),
            ("submit tax",),
        ),
    ),
    ReaderScenario(
        scenario_id="temporal_compare",
        query="哪一件事发生得更晚？",
        alternate_query="Which event happened later?",
        decision=_decision(
            recall_type="event",
            answer_intent="temporal_compare",
            focus="compare which remembered event happened later",
            obligations=["entities", "temporal_relation"],
            uncertainty_policy="abstain_if_insufficient",
            needs_event=True,
            strategy="text_search",
        ),
        evidence=(
            EvidenceSpec("Started walking practice.", "event", "event", occurred_at=10.0),
            EvidenceSpec("Started swimming practice.", "event", "event", occurred_at=20.0),
        ),
        reply="游泳练习发生得更晚，时间上排在步行练习之后。",
        required_concepts=(("游泳", "swimming"), ("更晚", "later", "after")),
    ),
    ReaderScenario(
        scenario_id="causal_explanation",
        query="Why have I been tired lately?",
        alternate_query="我最近为什么总觉得累？",
        decision=_decision(
            recall_type="event",
            answer_intent="causal_explanation",
            focus="explain the supported possible causes of recent fatigue",
            obligations=["entities", "qualifiers"],
            uncertainty_policy="abstain_if_insufficient",
            needs_event=True,
            strategy="text_search",
        ),
        evidence=(
            EvidenceSpec("Slept only five hours recently.", "event", "event"),
            EvidenceSpec("Drank four coffees recently.", "event", "event"),
        ),
        reply="可能的原因包括最近只睡了五个小时，以及咖啡喝得比较多。",
        required_concepts=(("五个小时", "five hours", "five"), ("咖啡", "coffee")),
    ),
    ReaderScenario(
        scenario_id="personalized_recommendation",
        query="根据我的习惯，我适合在哪里工作？",
        alternate_query="Where should I work based on my habits?",
        decision=_decision(
            recall_type="profile",
            answer_intent="personalized_recommendation",
            focus="recommend a work setting from the user's preferences and history",
            obligations=["entities", "incremental_next_step"],
            uncertainty_policy="state_limits_when_context_is_sparse",
            needs_profile=True,
            needs_event=True,
            strategy="text_search",
            goal="summary",
        ),
        evidence=(
            EvidenceSpec("Prefers quiet workspaces.", "profile", "preference"),
            EvidenceSpec("Focused better during the last quiet-cafe work session.", "event", "event"),
        ),
        reply="可以优先选择安静的咖啡馆或安静的工作区，并预留一段不被打扰的时间。",
        required_concepts=(("安静", "quiet"), ("工作区", "workspace", "咖啡馆", "cafe")),
    ),
    ReaderScenario(
        scenario_id="generic_recommendation",
        query="How can I improve focus?",
        alternate_query="如何提高专注力？",
        decision=_generic_decision(),
        evidence=(),
        reply="Try one short distraction-free work block, remove notifications, and take a planned break.",
        refusal_allowed=False,
    ),
    ReaderScenario(
        scenario_id="conflicting_evidence",
        query="我现在有车吗？",
        alternate_query="Do I currently own a car?",
        decision=_decision(
            recall_type="profile",
            answer_intent="direct_fact",
            focus="determine current car ownership while preserving conflicting and temporal evidence",
            obligations=["negation_constraints", "temporal_relation"],
            uncertainty_policy="state_limits_when_context_is_sparse",
            needs_profile=True,
            needs_event=True,
            strategy="text_search",
            goal="specific_fact",
        ),
        evidence=(
            EvidenceSpec("我没有车。", "profile", "fact", occurred_at=10.0),
            EvidenceSpec("我后来买车了。", "event", "fact", occurred_at=20.0),
        ),
        reply="记录存在时间冲突：之前说没有车，后来记录显示买了车；按较新的记录，现在更可能有车。",
        required_concepts=(
            ("没有车", "no car", "don't own"),
            ("买车", "bought a car", "有车", "own a car"),
            ("后来", "later", "较新的", "newer", "conflict", "冲突"),
        ),
    ),
    ReaderScenario(
        scenario_id="weak_background",
        query="上周参加了哪个会议？",
        alternate_query="Which meeting did I attend last week?",
        decision=_abstain_decision(recall_type="profile", needs_profile=True, needs_event=False),
        evidence=(EvidenceSpec("用户是软件工程师。", "profile", "fact"),),
        reply="我没有找到上周参加具体会议的相关记录，无法确认。",
        refusal_allowed=True,
        unsupported_terms=("aurora", "conference name"),
    ),
    ReaderScenario(
        scenario_id="no_evidence",
        query="Did I attend the Aurora conference?",
        alternate_query="我参加过 Aurora 会议吗？",
        decision=_abstain_decision(recall_type="event", needs_profile=False, needs_event=True),
        evidence=(),
        reply="I don't have a relevant record to confirm whether you attended it.",
        refusal_allowed=True,
        unsupported_terms=("aurora conference", "attended the aurora conference"),
    ),
)


def _seed_scenario(service: CoreChatService, scenario: ReaderScenario) -> list[str]:
    ids: list[str] = []
    for evidence in scenario.evidence:
        memory = service.memory_store.add_memory(
            "u1",
            evidence.content,
            kind=evidence.kind,
            memory_type=evidence.memory_type,
            occurred_at=evidence.occurred_at,
            start_at=evidence.start_at,
            confidence=0.95,
        )
        ids.append(memory.id)
    return ids


def _assert_reply_contract(scenario: ReaderScenario, reply: str) -> None:
    normalized = _normalized(reply)
    if scenario.refusal_allowed:
        assert not any(_normalized(term) in normalized for term in scenario.unsupported_terms)
        return
    aliases = tuple(alias for concept in scenario.required_concepts for alias in concept)
    assert not canonical_refusal(reply, answer_aliases=aliases), reply
    for concept in scenario.required_concepts:
        assert any(_normalized(alias) in normalized for alias in concept), (
            scenario.scenario_id,
            concept,
            reply,
        )


@pytest.mark.parametrize("scenario", READER_SCENARIOS, ids=lambda item: item.scenario_id)
def test_fixed_reader_evidence_contract(scenario: ReaderScenario) -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=scenario.decision, reply=scenario.reply)
        service = CoreChatService(tmpdir, agent=agent)
        expected_ids = _seed_scenario(service, scenario)

        response = service.chat(
            scenario.query,
            user_id="u1",
            memory_writes_allowed=False,
        )

        debug = response["debug"]
        assert debug["memory_writes_allowed"] is False
        assert response["saved_memories"] == []
        pre_reply = debug["pre_reply_decision"]
        planner = debug["planner"]
        directive = debug["answer_directive"]
        recalled = response["recalled_memories"]
        recalled_ids = [item["id"] for item in recalled]
        assert set(recalled_ids) == set(expected_ids)
        assert pre_reply["answer_intent"] == scenario.decision["answer_intent"]
        assert pre_reply["answer_focus"] == scenario.decision["answer_focus"]
        assert pre_reply["answer_obligations"] == scenario.decision["answer_obligations"]
        assert pre_reply["uncertainty_policy"] == scenario.decision["uncertainty_policy"]
        assert planner["event_recall_strategy"] == scenario.decision["event_recall_strategy"]
        assert directive["answer_focus"] == scenario.decision["answer_focus"]
        assert directive["answer_obligations"] == scenario.decision["answer_obligations"]
        assert directive["uncertainty_policy"] == scenario.decision["uncertainty_policy"]

        main_call = next(call for call in reversed(agent.calls) if not call["system_message"])
        prompt = str(main_call["message"])
        for evidence, evidence_id in zip(scenario.evidence, expected_ids):
            assert evidence.content in prompt, (scenario.scenario_id, evidence_id)
        assert "<answer-directive>" in prompt
        assert f"answer_focus: {scenario.decision['answer_focus']}" in prompt
        obligations = ", ".join(scenario.decision["answer_obligations"])
        assert f"answer_obligations: {obligations}" in prompt
        assert f"uncertainty_policy: {scenario.decision['uncertainty_policy']}" in prompt
        assert f"event_recall_strategy: {scenario.decision['event_recall_strategy']}" in prompt
        _assert_reply_contract(scenario, response["reply"])


def test_canonical_refusal_detector_is_test_only_and_allows_qualified_answer() -> None:
    assert canonical_refusal("I don't know; there is no record.") is True
    assert canonical_refusal(
        "I do not have the exact wording, but the record says low-sugar latte.",
        answer_aliases=("low-sugar latte",),
    ) is False
