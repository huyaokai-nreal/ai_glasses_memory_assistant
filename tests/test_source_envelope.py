from __future__ import annotations

import json

import pytest

from ai_glasses_memory_assistant.source_envelope import (
    SourceEnvelopeError,
    SourceEnvelopeInput,
    SourceEnvelopeIntegrityError,
    build_source_envelopes,
    deserialize_source_envelopes,
    select_complete_envelopes_for_text_budget,
    serialize_source_envelopes,
    source_ids_by_alias,
)


def test_multiline_source_round_trip_keeps_text_and_record_boundaries() -> None:
    original = build_source_envelopes([
        SourceEnvelopeInput(
            source_id="timeline:42",
            source_type="timeline",
            text="第一行\n第二行含有 [source=not-a-boundary]，以及 emoji ✨\n最后一行",
        ),
        SourceEnvelopeInput(
            source_id="memory:7",
            source_type="structured_memory",
            text="同样的标题\n但这是另一条来源。",
        ),
    ])

    serialized = serialize_source_envelopes(original)

    assert "\n" not in serialized
    assert deserialize_source_envelopes(serialized) == original
    assert source_ids_by_alias(original) == {"S1": "timeline:42", "S2": "memory:7"}


def test_source_aliases_follow_input_priority_and_ids_are_unique() -> None:
    envelopes = build_source_envelopes([
        SourceEnvelopeInput("timeline:late", "timeline", "later-ranked source"),
        SourceEnvelopeInput("memory:early", "memory", "earlier-ranked source"),
    ])

    assert [envelope.alias for envelope in envelopes] == ["S1", "S2"]
    assert source_ids_by_alias(envelopes)["S2"] == "memory:early"
    with pytest.raises(SourceEnvelopeError, match="duplicate source_id"):
        build_source_envelopes([
            SourceEnvelopeInput("memory:one", "memory", "first"),
            SourceEnvelopeInput("memory:one", "timeline", "second"),
        ])


def test_deserialization_rejects_tampered_multiline_text() -> None:
    serialized = serialize_source_envelopes(build_source_envelopes([
        SourceEnvelopeInput("timeline:1", "timeline", "original\nsecond line"),
    ]))
    payload = json.loads(serialized)
    payload["sources"][0]["text"] = "original\nreplaced line"

    with pytest.raises(SourceEnvelopeIntegrityError, match="hash mismatch"):
        deserialize_source_envelopes(json.dumps(payload, ensure_ascii=False))


def test_text_budget_keeps_a_complete_priority_prefix() -> None:
    envelopes = build_source_envelopes([
        SourceEnvelopeInput("timeline:1", "timeline", "alpha\nbeta"),
        SourceEnvelopeInput("timeline:2", "timeline", "gamma"),
        SourceEnvelopeInput("timeline:3", "timeline", "delta"),
    ])

    selection = select_complete_envelopes_for_text_budget(envelopes, text_char_budget=10)

    assert [item.source_id for item in selection.included] == ["timeline:1"]
    assert [item.source_id for item in selection.omitted] == ["timeline:2", "timeline:3"]
    assert selection.included[0].text == "alpha\nbeta"


def test_text_budget_rejects_negative_values() -> None:
    envelopes = build_source_envelopes([
        SourceEnvelopeInput("timeline:1", "timeline", "source"),
    ])

    with pytest.raises(SourceEnvelopeError, match="non-negative"):
        select_complete_envelopes_for_text_budget(envelopes, text_char_budget=-1)
