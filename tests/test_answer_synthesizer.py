from __future__ import annotations

from ai_glasses_memory_assistant.answer_synthesizer import (
    AnswerDirective,
    _fallback_directive,
    apply_answer_contract,
    synthesize_answer_directive,
)


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
