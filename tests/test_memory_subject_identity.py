from __future__ import annotations

import math

import pytest

from ai_glasses_memory_assistant.memory_subject_identity import (
    VOICE_MATCH_REASON_AMBIGUOUS_MARGIN,
    VOICE_MATCH_REASON_BELOW_THRESHOLD,
    VOICE_MATCH_REASON_INCOMING_EMBEDDING_INVALID,
    VOICE_MATCH_REASON_MATCHED,
    VOICE_MATCH_REASON_NO_COMPATIBLE_DIMENSION,
    VOICE_MATCH_REASON_NO_COMPATIBLE_MODEL,
    VOICE_MATCH_REASON_NO_COMPATIBLE_REFERENCES,
    VOICE_REFERENCE_REASON_EMBEDDING_DIMENSION_MISMATCH,
    VOICE_REFERENCE_REASON_EMBEDDING_MODEL_MISMATCH,
    VoiceProfileReference,
    match_voice_profile,
    normalize_subject_name,
    subject_name_key,
    update_voice_profile_centroid,
)


def _reference(
    subject_id: str,
    embedding: list[float],
    *,
    name: str | None = None,
    model: str = "campplus",
    subject_type: str = "named",
    sample_count: int = 1,
) -> VoiceProfileReference:
    return VoiceProfileReference(
        subject_id=subject_id,
        subject_name=name or subject_id.title(),
        embedding=embedding,
        embedding_model=model,
        subject_type=subject_type,
        sample_count=sample_count,
    )


def _match(
    embedding: list[float],
    references: list[VoiceProfileReference],
    *,
    model: str = "campplus",
):
    return match_voice_profile(
        embedding,
        embedding_model=model,
        references=references,
        provisional_subject_id="provisional-1",
        provisional_subject_name="speaker_1",
    )


def test_subject_name_normalization_is_stable_and_language_neutral() -> None:
    assert normalize_subject_name("  Ａｌｉｃｅ\tSmith\n") == "Alice Smith"
    assert subject_name_key(" ALICE  Smith ") == "alice smith"
    assert normalize_subject_name("") == ""


def test_voice_match_selects_one_clear_compatible_subject() -> None:
    result = _match(
        [1.0, 0.0],
        [
            _reference("alice", [1.0, 0.0], name=" Alice ", subject_type="self"),
            _reference("bob", [0.0, 1.0]),
        ],
    )

    assert result.matched is True
    assert result.reason == VOICE_MATCH_REASON_MATCHED
    assert result.subject_id == "alice"
    assert result.subject_name == "Alice"
    assert result.subject_type == "self"
    assert result.similarity == 1.0
    assert result.runner_up_similarity == 0.0
    assert result.margin == 1.0
    assert result.candidate_scores == (("alice", 1.0), ("bob", 0.0))


def test_voice_match_falls_back_when_best_score_is_below_threshold() -> None:
    result = _match(
        [1.0, 0.0],
        [_reference("alice", [0.70, math.sqrt(1.0 - 0.70**2)])],
    )

    assert result.matched is False
    assert result.reason == VOICE_MATCH_REASON_BELOW_THRESHOLD
    assert result.subject_id == "provisional-1"
    assert result.subject_type == "provisional"
    assert result.similarity == 0.7


def test_voice_match_falls_back_when_two_distinct_subjects_are_too_close() -> None:
    result = _match(
        [1.0, 0.0],
        [
            _reference("alice", [1.0, 0.0]),
            _reference("bob", [0.98, math.sqrt(1.0 - 0.98**2)]),
        ],
    )

    assert result.matched is False
    assert result.reason == VOICE_MATCH_REASON_AMBIGUOUS_MARGIN
    assert result.subject_id == "provisional-1"
    assert result.similarity == 1.0
    assert result.runner_up_similarity == 0.98
    assert result.margin == 0.02


def test_duplicate_references_for_one_subject_do_not_create_false_ambiguity() -> None:
    result = _match(
        [1.0, 0.0],
        [
            _reference("alice", [1.0, 0.0]),
            _reference("alice", [0.99, math.sqrt(1.0 - 0.99**2)]),
        ],
    )

    assert result.matched is True
    assert result.compatible_reference_count == 1
    assert result.runner_up_similarity is None


def test_voice_match_rejects_model_mismatch_without_comparing_embeddings() -> None:
    result = _match([1.0, 0.0], [_reference("alice", [1.0, 0.0], model="other-model")])

    assert result.reason == VOICE_MATCH_REASON_NO_COMPATIBLE_MODEL
    assert result.candidate_scores == ()
    assert result.rejected_references == (("alice", VOICE_REFERENCE_REASON_EMBEDDING_MODEL_MISMATCH),)


def test_voice_match_rejects_dimension_mismatch_instead_of_truncating() -> None:
    result = _match([1.0, 0.0], [_reference("alice", [1.0, 0.0, 0.0])])

    assert result.reason == VOICE_MATCH_REASON_NO_COMPATIBLE_DIMENSION
    assert result.rejected_references == (("alice", VOICE_REFERENCE_REASON_EMBEDDING_DIMENSION_MISMATCH),)


def test_voice_match_can_use_compatible_reference_while_reporting_rejected_ones() -> None:
    result = _match(
        [1.0, 0.0],
        [
            _reference("alice", [1.0, 0.0]),
            _reference("bob", [1.0, 0.0, 0.0]),
        ],
    )

    assert result.matched is True
    assert result.subject_id == "alice"
    assert result.compatible_reference_count == 1
    assert result.rejected_references == (("bob", VOICE_REFERENCE_REASON_EMBEDDING_DIMENSION_MISMATCH),)


def test_voice_match_rejects_invalid_incoming_embedding() -> None:
    result = _match([0.0, 0.0], [_reference("alice", [1.0, 0.0])])

    assert result.reason == VOICE_MATCH_REASON_INCOMING_EMBEDDING_INVALID
    assert result.subject_type == "provisional"


def test_voice_match_uses_generic_reason_for_mixed_reference_failures() -> None:
    result = _match(
        [1.0, 0.0],
        [
            _reference("alice", [1.0, 0.0], model="other-model"),
            _reference("bob", [1.0, 0.0, 0.0]),
        ],
    )

    assert result.reason == VOICE_MATCH_REASON_NO_COMPATIBLE_REFERENCES


def test_voice_centroid_update_weights_existing_samples() -> None:
    reference = _reference("alice", [1.0, 0.0], sample_count=3)

    updated = update_voice_profile_centroid(reference, [0.0, 1.0], embedding_model="CAMPPLUS")

    assert updated is not None
    assert updated.embedding == pytest.approx((0.75, 0.25))
    assert updated.sample_count == 4
    assert reference.embedding == [1.0, 0.0]
    assert reference.sample_count == 3


@pytest.mark.parametrize(
    ("embedding", "model"),
    [
        ([0.0, 1.0], "other-model"),
        ([0.0, 1.0, 0.0], "campplus"),
        ([0.0, 0.0], "campplus"),
    ],
)
def test_voice_centroid_update_rejects_incompatible_sample(embedding: list[float], model: str) -> None:
    reference = _reference("alice", [1.0, 0.0], sample_count=2)

    assert update_voice_profile_centroid(reference, embedding, embedding_model=model) is None
