from ai_glasses_memory_assistant.turn_planner import plan_turn


LONG_MULTI_SENTENCE_TEXT = (
    "I have a detailed question about the project history, and need a direct answer; "
    "please use the information I previously shared, to explain what changed; "
    "I also need the answer to distinguish the earlier decision from the later follow-up."
)


def test_long_text_question_does_not_enter_continuous_capture_without_audio_provenance() -> None:
    plan = plan_turn(LONG_MULTI_SENTENCE_TEXT, reference_time=1000.0)

    assert plan.fast_path is False
    assert plan.reply_mode == "llm"


def test_long_audio_event_keeps_continuous_capture_fast_path() -> None:
    plan = plan_turn(
        LONG_MULTI_SENTENCE_TEXT,
        reference_time=1000.0,
        allow_continuous_capture=True,
    )

    assert plan.fast_path is True
    assert plan.fast_path_kind == "continuous_capture"
