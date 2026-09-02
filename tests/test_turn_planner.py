from datetime import datetime, timezone

from ai_glasses_memory_assistant.turn_planner import plan_turn, resolve_temporal_local


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


def test_local_temporal_parser_preserves_english_calendar_date_ranges_for_event_writes() -> None:
    resolution = resolve_temporal_local(
        "I attended a two-day workshop on the 17th and 18th of April.",
        reference_time=datetime(2023, 5, 1, tzinfo=timezone.utc).timestamp(),
        timezone="UTC",
    )

    assert resolution.usable_range is True
    assert resolution.temporal_text == "17th and 18th of April"
    assert resolution.start_at == datetime(2023, 4, 17, tzinfo=timezone.utc).timestamp()
    assert resolution.end_at == datetime(2023, 4, 19, tzinfo=timezone.utc).timestamp()


def test_local_temporal_parser_preserves_numeric_english_calendar_dates() -> None:
    resolution = resolve_temporal_local(
        "I visited the gallery on 2/15.",
        reference_time=datetime(2023, 3, 1, tzinfo=timezone.utc).timestamp(),
        timezone="UTC",
    )

    assert resolution.usable_range is True
    assert resolution.temporal_text == "2/15"
    assert resolution.start_at == datetime(2023, 2, 15, tzinfo=timezone.utc).timestamp()
    assert resolution.end_at == datetime(2023, 2, 16, tzinfo=timezone.utc).timestamp()


def test_local_temporal_parser_does_not_treat_a_model_scale_as_a_date() -> None:
    resolution = resolve_temporal_local(
        "I am building a 1/72 scale model aircraft.",
        reference_time=datetime(2023, 3, 1, tzinfo=timezone.utc).timestamp(),
        timezone="UTC",
    )

    assert resolution.usable_range is False
