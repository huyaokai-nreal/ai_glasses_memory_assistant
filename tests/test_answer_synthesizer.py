from __future__ import annotations

import json

from ai_glasses_memory_assistant.answer_synthesizer import (
    AnswerDirective,
    INCOMPLETE_COMPLETE_SET_REPLY,
    _fallback_directive,
    apply_answer_contract,
    synthesize_complete_set_answer,
    synthesize_answer_directive,
)
from ai_glasses_memory_assistant.evidence_set import EvidenceCandidate


def test_reader_directive_requires_using_relevant_evidence_before_abstaining() -> None:
    directive = AnswerDirective(
        answer_intent="summary",
        answer_focus="summarize the directly relevant evidence",
        answer_obligations=["entities", "qualifiers"],
        uncertainty_policy="abstain_if_insufficient",
    )
    instruction = directive.instruction_text()

    assert "answer_focus: summarize the directly relevant evidence" in instruction
    assert "answer_obligations: entities, qualifiers" in instruction
    assert "uncertainty_policy: abstain_if_insufficient" in instruction
    assert "directly relevant structured memory evidence is present" in instruction
    assert "abstain only when the recalled evidence does not support" in instruction
    assert directive.debug_payload()["answer_obligations"] == ["entities", "qualifiers"]


def test_reader_fallback_keeps_summary_evidence_boundary() -> None:
    directive = _fallback_directive(
        {"recall_goal": "summary"},
        {"profile_count": 0, "event_count": 2, "observation_count": 0, "timeline_count": 0},
        error="synthetic provider failure",
    )

    assert directive.answer_intent == "summary"
    assert directive.evidence_policy == "separate_background"
    assert directive.uncertainty_policy == "state_limits_when_context_is_sparse"
    assert "prefer source evidence" in " ".join(directive.filtering_rules)


class _ConflictingAnswerPlanner:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def run_conversation(self, message: str, **kwargs):
        if self.fail:
            raise RuntimeError("synthetic answer planner failure")
        return {
            "final_response": (
                '{"answer_focus":"provider focus",'
                '"answer_obligations":["comparison"],'
                '"uncertainty_policy":"none",'
                '"answer_intent":"summary",'
                '"organization":"thematic"}'
            ),
        }


def _answer_contract() -> dict[str, object]:
    return {
        "answer_intent": "personalized_recommendation",
        "answer_focus": "compare the two supported facts",
        "answer_obligations": ["entities", "comparison", "entities", "invalid"],
        "uncertainty_policy": "abstain_if_insufficient",
    }


def test_route_answer_contract_overrides_conflicting_provider_directive() -> None:
    directive = synthesize_answer_directive(
        _ConflictingAnswerPlanner(),
        message="Which fact is newer?",
        route_debug={},
        intent_debug={},
        temporal_debug={},
        evidence_summary={"event_count": 2},
        answer_contract=_answer_contract(),
    )

    assert directive.answer_focus == "compare the two supported facts"
    assert directive.answer_obligations == ["entities", "comparison"]
    assert directive.uncertainty_policy == "abstain_if_insufficient"
    assert directive.answer_intent == "personalized_recommendation"
    assert directive.organization == "thematic"


def test_route_answer_contract_survives_synthesis_fallback() -> None:
    directive = synthesize_answer_directive(
        _ConflictingAnswerPlanner(fail=True),
        message="Which fact is newer?",
        route_debug={},
        intent_debug={},
        temporal_debug={},
        evidence_summary={"event_count": 2},
        answer_contract=_answer_contract(),
    )

    assert directive.backend == "fallback"
    assert directive.answer_focus == "compare the two supported facts"
    assert directive.answer_obligations == ["entities", "comparison"]
    assert directive.uncertainty_policy == "abstain_if_insufficient"
    assert directive.answer_intent == "personalized_recommendation"


def test_empty_route_answer_contract_keeps_non_forcing_defaults() -> None:
    directive = apply_answer_contract(
        AnswerDirective(),
        {"answer_focus": "", "answer_obligations": [], "uncertainty_policy": "none"},
    )

    assert directive.answer_focus == ""
    assert directive.answer_obligations == []
    assert "answer_obligations: none" in directive.instruction_text()


def test_instruction_text_adds_obligation_guidance() -> None:
    directive = apply_answer_contract(
        AnswerDirective(),
        {
            "answer_intent": "personalized_recommendation",
            "answer_obligations": ["negation_constraints", "incremental_next_step", "comparison"],
            "uncertainty_policy": "state_limits_when_context_is_sparse",
        },
    )

    text = directive.instruction_text()
    assert "answer_intent: personalized_recommendation" in text
    assert "would not prefer or should avoid" in text
    assert "already owns, tried, prepared, or planned" in text
    assert "address both sides of the comparison" in text


def test_instruction_text_covers_existing_fact_obligations_without_expanding_them() -> None:
    directive = apply_answer_contract(
        AnswerDirective(),
        {
            "answer_intent": "count_or_total",
            "answer_focus": "count the devices and report each supported model and purchase date",
            "answer_obligations": ["count_scope", "entities", "qualifiers", "temporal_relation"],
            "uncertainty_policy": "abstain_if_insufficient",
        },
    )

    text = directive.instruction_text()
    assert directive.answer_obligations == ["count_scope", "entities", "qualifiers", "temporal_relation"]
    assert "name each requested object or identity" in text
    assert "model, type, category, specification, ratio" in text
    assert "date, duration, ordering, or per-entity time relation" in text
    assert "Do not list entities unless entities is also an obligation" in text
    assert "do not invent missing entities" in text


def test_count_scope_alone_does_not_add_entities_obligation() -> None:
    directive = apply_answer_contract(
        AnswerDirective(),
        {
            "answer_intent": "count_or_total",
            "answer_focus": "count all supported plants",
            "answer_obligations": ["count_scope"],
        },
    )

    assert directive.answer_obligations == ["count_scope"]
    assert "Cover the entities obligation" not in directive.instruction_text()
    assert "Do not list entities unless entities is also an obligation" in directive.instruction_text()


def test_complete_set_directive_cannot_claim_total_when_coverage_is_incomplete() -> None:
    directive = apply_answer_contract(
        AnswerDirective(),
        {
            "answer_intent": "count_or_total",
            "answer_focus": "all scoped purchases",
            "answer_obligations": ["count_scope"],
            "uncertainty_policy": "abstain_if_insufficient",
            "coverage_requirement": "complete_set",
            "coverage_complete": False,
        },
    )

    text = directive.instruction_text()
    assert directive.coverage_requirement == "complete_set"
    assert directive.coverage_complete is False
    assert "account for every supplied source" in text
    assert "Do not state an absolute count or total" in text


class _LedgerAgent:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.messages: list[str] = []

    def run_conversation(self, message: str, **_kwargs):
        self.messages.append(message)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return {"final_response": json.dumps(response)}


class _FailingLedgerAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run_conversation(self, _message: str, **_kwargs):
        self.calls += 1
        raise RuntimeError("synthetic provider failure")


class _RawLedgerAgent:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def run_conversation(self, _message: str, **_kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return {"final_response": response}


def test_product_complete_set_reader_returns_only_validated_answer() -> None:
    agent = _LedgerAgent([
        {
            "items": [{
                "canonical_key": "kit-alpha",
                "label": "Alpha",
                "quantity": "1",
                "unit": "item",
                "status": "included",
                "source_ids": ["memory:m1", "timeline:t1"],
            }],
            "aggregation": {"operation": "count", "value": "1", "unit": "item"},
            "final_answer": "1 item",
        },
        {"final_answer": "共 1 个：Alpha。"},
    ])
    result = synthesize_complete_set_answer(
        agent,
        message="我买过哪些套件，一共有几个？",
        answer_contract={
            "answer_intent": "count_or_total",
            "answer_focus": "所有套件购买",
            "answer_obligations": ["entities", "count_scope"],
        },
        candidates=[
            EvidenceCandidate(source_id="memory:m1", source_type="memory", text="买了 Alpha"),
            EvidenceCandidate(source_id="timeline:t1", source_type="timeline", text="我买了 Alpha"),
        ],
        coverage_complete=True,
    )

    assert result.valid is True
    assert result.final_answer == "共 1 个：Alpha。"
    assert result.value == "1"
    assert result.api_calls == 2
    assert "readable item label must preserve any source-supported entity" in agent.messages[0]
    assert "Fixed answer contract:" in agent.messages[1]
    assert '"answer_obligations": ["entities", "count_scope"]' in agent.messages[1]
    assert "When entities is required, name the readable ledger item labels" in agent.messages[1]
    assert "Count scope alone does not require an entity list" in agent.messages[1]


def test_product_complete_set_reader_fails_without_complete_coverage() -> None:
    agent = _LedgerAgent([])
    result = synthesize_complete_set_answer(
        agent,
        message="总数？",
        answer_contract={"answer_intent": "count_or_total", "answer_obligations": ["count_scope"]},
        candidates=[EvidenceCandidate(source_id="memory:m1", source_type="memory", text="one")],
        coverage_complete=False,
    )

    assert result.valid is False
    assert result.final_answer == INCOMPLETE_COMPLETE_SET_REPLY
    assert result.error == "coverage_incomplete"
    assert result.reader_status == "insufficient_evidence"
    assert result.failure_stage == "coverage"
    assert agent.calls == 0


def test_product_complete_set_reader_recomputes_intermediate_total() -> None:
    agent = _LedgerAgent([
        {
            "source_decisions": [{"source_id": "memory:note", "status": "included"}],
            "items": [
                {
                    "canonical_key": "architecture-review",
                    "label": "architecture review",
                    "quantity": "1",
                    "unit": "item",
                    "status": "included",
                    "source_ids": ["memory:note"],
                },
                {
                    "canonical_key": "release-review",
                    "label": "release review",
                    "quantity": "1",
                    "unit": "item",
                    "status": "included",
                    "source_ids": ["memory:note"],
                },
            ],
            "aggregation": {"operation": "count", "value": "500", "unit": "item"},
        },
        {"final_answer": "共 2 次评审。"},
    ])

    result = synthesize_complete_set_answer(
        agent,
        message="我参加了几次评审？",
        answer_contract={
            "answer_intent": "count_or_total",
            "answer_focus": "参加过的评审",
            "answer_obligations": ["count_scope"],
        },
        candidates=[
            EvidenceCandidate(
                source_id="memory:note",
                source_type="memory",
                text="参加了架构评审和发布评审。",
            ),
        ],
        coverage_complete=True,
    )

    assert result.valid is True
    assert result.value == "2"
    assert result.final_answer == "共 2 次评审。"
    assert result.reader_status == "answered"
    assert result.failure_stage == ""
    assert [attempt["stage"] for attempt in result.execution_attempts] == [
        "batch_ledger",
        "final_answer",
    ]
    assert all(attempt["input_chars"] > 0 for attempt in result.execution_attempts)
    assert all(attempt["duration_seconds"] >= 0 for attempt in result.execution_attempts)


def test_product_complete_set_execution_failure_is_not_evidence_insufficiency() -> None:
    agent = _LedgerAgent([{"items": []}, {"items": []}])

    result = synthesize_complete_set_answer(
        agent,
        message="一共有几个？",
        answer_contract={"answer_intent": "count_or_total", "answer_obligations": ["count_scope"]},
        candidates=[EvidenceCandidate(source_id="memory:m1", source_type="memory", text="一个项目")],
        coverage_complete=True,
    )

    assert result.valid is False
    assert result.coverage_complete is True
    assert "1条候选证据" in result.final_answer
    assert "完整整理未能通过验证" in result.final_answer
    assert result.reader_status == "execution_failed"
    assert result.failure_stage == "batch_ledger"
    assert [attempt["outcome"] for attempt in result.execution_attempts] == ["invalid", "invalid"]


def test_product_complete_set_provider_failure_is_reported_as_execution_failure() -> None:
    agent = _FailingLedgerAgent()

    result = synthesize_complete_set_answer(
        agent,
        message="一共有几个？",
        answer_contract={"answer_intent": "count_or_total", "answer_obligations": ["count_scope"]},
        candidates=[EvidenceCandidate(source_id="memory:m1", source_type="memory", text="一个项目")],
        coverage_complete=True,
    )

    assert "1条候选证据" in result.final_answer
    assert result.reader_status == "execution_failed"
    assert result.failure_stage == "provider"
    assert result.error == "provider_output_failure"
    assert [attempt["outcome"] for attempt in result.execution_attempts] == [
        "provider_error",
        "provider_error",
    ]


def test_product_complete_set_invalid_json_is_reported_as_execution_failure() -> None:
    agent = _RawLedgerAgent(["", "not-json"])

    result = synthesize_complete_set_answer(
        agent,
        message="一共有几个？",
        answer_contract={"answer_intent": "count_or_total", "answer_obligations": ["count_scope"]},
        candidates=[EvidenceCandidate(source_id="memory:m1", source_type="memory", text="一个项目")],
        coverage_complete=True,
    )

    assert "1条候选证据" in result.final_answer
    assert result.reader_status == "execution_failed"
    assert result.failure_stage == "json"
    assert result.error == "invalid_json_output"
    assert [attempt["outcome"] for attempt in result.execution_attempts] == [
        "invalid_json",
        "invalid_json",
    ]


def test_product_complete_set_execution_failure_preserves_structured_scope() -> None:
    result = synthesize_complete_set_answer(
        None,
        message="任意问法不应改变降级范围",
        answer_contract={"answer_focus": "已授权的讨论范围"},
        candidates=[
            EvidenceCandidate(source_id="discussion:a", source_type="discussion_archive", text="A"),
            EvidenceCandidate(source_id="discussion:b", source_type="discussion_archive", text="B"),
        ],
        coverage_complete=True,
    )

    assert "已授权的讨论范围" in result.final_answer
    assert "2条候选证据" in result.final_answer
    assert result.reader_status == "execution_failed"
    assert result.failure_stage == "provider"
