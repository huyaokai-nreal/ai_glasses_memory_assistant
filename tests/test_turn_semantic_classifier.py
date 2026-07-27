from ai_glasses_memory_assistant.turn_semantic_classifier import _decision_from_payload


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
