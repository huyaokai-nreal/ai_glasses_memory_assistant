from __future__ import annotations

from ai_glasses_memory_assistant.answer_synthesizer import AnswerDirective, _fallback_directive


def test_reader_directive_requires_using_relevant_evidence_before_abstaining() -> None:
    instruction = AnswerDirective(answer_intent="summary").instruction_text()

    assert "directly relevant structured memory evidence is present" in instruction
    assert "abstain only when the recalled evidence does not support" in instruction


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
