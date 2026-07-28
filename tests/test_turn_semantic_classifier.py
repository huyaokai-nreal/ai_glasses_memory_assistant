import json

from ai_glasses_memory_assistant.turn_semantic_classifier import (
    _decision_from_payload,
    classify_pre_reply_decision,
)


class CapturingClassifierAgent:
    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    def run_conversation(self, message: str, **kwargs: object) -> dict[str, str]:
        self.calls.append({"message": message, **kwargs})
        return {"final_response": json.dumps(self.decision)}


def test_read_only_recall_survives_low_confidence() -> None:
    decision = _decision_from_payload(
        {
            "memory_action": "recall",
            "memory_recall_type": "event",
            "needs_event_memory": True,
            "recall_goal": "specific_fact",
            "event_recall_strategy": "text_search",
            "confidence": 0.7,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.error == ""
    assert decision.needs_event_memory is True
    assert decision.recall_goal == "specific_fact"
    assert "confidence_below_threshold_read_only_recall_preserved" in decision.warnings


def test_invalid_unrelated_fields_do_not_discard_valid_recall() -> None:
    decision = _decision_from_payload(
        {
            "memory_action": "recall",
            "needs_event_memory": True,
            "recall_goal": "specific_fact",
            "event_recall_strategy": "text_search",
            "reply_mode": "answer_from_memory",
            "answer_source": "memory",
            "scope": "personal",
            "memory_recall_type": "event_lookup",
            "confidence": 0.95,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.error == ""
    assert decision.needs_event_memory is True
    assert decision.memory_recall_type == "event"
    assert any(item.startswith("invalid_reply_mode") for item in decision.warnings)


def test_classifier_contract_keeps_tailored_advice_as_profile_context_for_reply() -> None:
    agent = CapturingClassifierAgent({
        "turn_intent": "mixed",
        "memory_action": "recall",
        "memory_recall_type": "profile",
        "needs_profile_memory": True,
        "recall_goal": "summary",
        "confidence": 0.95,
    })

    decision = classify_pre_reply_decision(
        agent,
        "Based on the work habits I have shared, what focus routine would suit me?",
    )

    assert decision.turn_intent == "mixed"
    assert decision.needs_profile_memory is True
    assert decision.needs_event_memory is False
    assert decision.needs_timeline_recall is False
    assert decision.memory_recall_type == "profile"
    assert decision.recall_goal == "summary"
    classifier_prompt = str(agent.calls[0]["message"])
    assert "advice or a recommendation tailored to their own existing preferences" in classifier_prompt
    assert "generic advice not tailored to the user's own existing preferences" in classifier_prompt
