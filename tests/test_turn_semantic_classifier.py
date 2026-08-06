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


def test_malformed_mixed_profile_recall_recovers_existing_summary_contract() -> None:
    decision = _decision_from_payload(
        {
            "turn_intent": "mixed",
            "reply_mode": "llm",
            "memory_action": "none|write|recall|correction|explain",
            "memory_recall_type": "none|profile|event|timeline|observation",
            "recall_goal": "none|summary|raw_evidence|specific_fact",
            "needs_profile_memory": True,
            "needs_event_memory": False,
            "needs_timeline_recall": False,
            "needs_discussion_recall": False,
            "recall_subject_scope": "self",
            "recall_subject_names": [],
            "confidence": 0.95,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.memory_action == "recall"
    assert decision.memory_recall_type == "profile"
    assert decision.recall_goal == "summary"
    assert "recovered_malformed_tailored_profile_recall" in decision.warnings


def test_explicit_no_recall_is_not_recovered_as_profile_summary() -> None:
    decision = _decision_from_payload(
        {
            "turn_intent": "chat",
            "reply_mode": "llm",
            "memory_action": "none",
            "memory_recall_type": "none",
            "recall_goal": "none",
            "needs_profile_memory": False,
            "confidence": 0.95,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.memory_action == "none"
    assert decision.memory_recall_type == "none"
    assert decision.recall_goal == "none"
    assert "recovered_malformed_tailored_profile_recall" not in decision.warnings


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
    assert "Personal ongoing context" in classifier_prompt
    assert "Generic/factual advice" in classifier_prompt
    assert "Current local recommendation" in classifier_prompt
    assert "If the user asks what the assistant previously said" in classifier_prompt
    assert "generic advice not tailored to the user's own existing preferences" in classifier_prompt
    assert "event_recall_strategy=text_search" in classifier_prompt
    assert "Time distinction" in classifier_prompt


def test_classifier_contract_opens_recall_for_advice_building_on_owned_items() -> None:
    """Advice grounded in user-owned items/experiments must open bounded recall."""
    agent = CapturingClassifierAgent({
        "turn_intent": "mixed",
        "memory_action": "recall",
        "memory_recall_type": "profile",
        "needs_profile_memory": True,
        "needs_event_memory": True,
        "event_recall_strategy": "text_search",
        "recall_goal": "summary",
        "confidence": 0.95,
    })

    decision = classify_pre_reply_decision(
        agent,
        "My kitchen's becoming a bit of a mess again. Any tips for keeping it clean?",
    )

    assert decision.needs_profile_memory is True
    assert decision.needs_event_memory is True
    classifier_prompt = str(agent.calls[0]["message"])
    assert "Advice building on user-owned items/experiments" in classifier_prompt
    assert "slow cooker" in classifier_prompt
    assert "utensil holder" in classifier_prompt
    assert "portable power bank" in classifier_prompt
    # The negative example anchor must exist so generic advice stays none.
    assert "no user-specific anchor" in classifier_prompt
