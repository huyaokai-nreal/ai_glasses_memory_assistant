from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Sequence


DEFAULT_VOICE_MATCH_THRESHOLD = 0.72
DEFAULT_VOICE_MATCH_MIN_MARGIN = 0.05

VOICE_MATCH_DECISION_MATCHED = "matched"
VOICE_MATCH_DECISION_PROVISIONAL = "provisional"

VOICE_MATCH_REASON_MATCHED = "voice_profile_matched"
VOICE_MATCH_REASON_INCOMING_EMBEDDING_INVALID = "incoming_embedding_invalid"
VOICE_MATCH_REASON_INCOMING_MODEL_MISSING = "incoming_embedding_model_missing"
VOICE_MATCH_REASON_NO_REFERENCES = "voice_profile_no_references"
VOICE_MATCH_REASON_NO_COMPATIBLE_MODEL = "voice_profile_no_compatible_model"
VOICE_MATCH_REASON_NO_COMPATIBLE_DIMENSION = "voice_profile_no_compatible_dimension"
VOICE_MATCH_REASON_NO_VALID_REFERENCE_EMBEDDING = "voice_profile_no_valid_reference_embedding"
VOICE_MATCH_REASON_NO_COMPATIBLE_REFERENCES = "voice_profile_no_compatible_references"
VOICE_MATCH_REASON_BELOW_THRESHOLD = "voice_profile_below_similarity_threshold"
VOICE_MATCH_REASON_AMBIGUOUS_MARGIN = "voice_profile_ambiguous_similarity_margin"

VOICE_REFERENCE_REASON_SUBJECT_ID_MISSING = "subject_id_missing"
VOICE_REFERENCE_REASON_EMBEDDING_MODEL_MISSING = "embedding_model_missing"
VOICE_REFERENCE_REASON_EMBEDDING_MODEL_MISMATCH = "embedding_model_mismatch"
VOICE_REFERENCE_REASON_EMBEDDING_INVALID = "embedding_invalid"
VOICE_REFERENCE_REASON_EMBEDDING_DIMENSION_MISMATCH = "embedding_dimension_mismatch"


VoiceMatchDecision = Literal["matched", "provisional"]


def normalize_subject_name(value: str) -> str:
    """Normalize display names without applying language- or domain-specific aliases."""

    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split())


def subject_name_key(value: str) -> str:
    """Return a stable comparison key; user scoping remains the caller's responsibility."""

    return normalize_subject_name(value).casefold()


@dataclass(frozen=True)
class VoiceProfileReference:
    subject_id: str
    subject_name: str
    embedding: Sequence[float]
    embedding_model: str
    subject_type: str = "named"
    sample_count: int = 1


@dataclass(frozen=True)
class VoiceMatchResult:
    decision: VoiceMatchDecision
    subject_id: str
    subject_name: str
    subject_type: str
    reason: str
    similarity: float | None = None
    runner_up_similarity: float | None = None
    margin: float | None = None
    compatible_reference_count: int = 0
    candidate_scores: tuple[tuple[str, float], ...] = ()
    rejected_references: tuple[tuple[str, str], ...] = ()

    @property
    def matched(self) -> bool:
        return self.decision == VOICE_MATCH_DECISION_MATCHED


def match_voice_profile(
    embedding: Sequence[float],
    *,
    embedding_model: str,
    references: Iterable[VoiceProfileReference],
    provisional_subject_id: str,
    provisional_subject_name: str,
    threshold: float = DEFAULT_VOICE_MATCH_THRESHOLD,
    min_margin: float = DEFAULT_VOICE_MATCH_MIN_MARGIN,
) -> VoiceMatchResult:
    """Resolve one embedding conservatively or keep it as a provisional subject."""

    provisional_id = str(provisional_subject_id or "").strip()
    if not provisional_id:
        raise ValueError("provisional_subject_id must not be empty")
    provisional_name = normalize_subject_name(provisional_subject_name)
    if not provisional_name:
        raise ValueError("provisional_subject_name must not be empty")
    if not -1.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between -1.0 and 1.0")
    if not 0.0 <= float(min_margin) <= 2.0:
        raise ValueError("min_margin must be between 0.0 and 2.0")

    reference_list = tuple(references)
    incoming = _coerce_embedding(embedding)
    if incoming is None:
        return _provisional_result(
            provisional_id,
            provisional_name,
            VOICE_MATCH_REASON_INCOMING_EMBEDDING_INVALID,
        )

    incoming_model = _model_key(embedding_model)
    if not incoming_model:
        return _provisional_result(
            provisional_id,
            provisional_name,
            VOICE_MATCH_REASON_INCOMING_MODEL_MISSING,
        )
    if not reference_list:
        return _provisional_result(
            provisional_id,
            provisional_name,
            VOICE_MATCH_REASON_NO_REFERENCES,
        )

    rejected: list[tuple[str, str]] = []
    best_by_subject: dict[str, tuple[float, VoiceProfileReference]] = {}
    for reference in reference_list:
        subject_id = str(reference.subject_id or "").strip()
        if not subject_id:
            rejected.append(("", VOICE_REFERENCE_REASON_SUBJECT_ID_MISSING))
            continue
        reference_model = _model_key(reference.embedding_model)
        if not reference_model:
            rejected.append((subject_id, VOICE_REFERENCE_REASON_EMBEDDING_MODEL_MISSING))
            continue
        if reference_model != incoming_model:
            rejected.append((subject_id, VOICE_REFERENCE_REASON_EMBEDDING_MODEL_MISMATCH))
            continue
        reference_embedding = _coerce_embedding(reference.embedding)
        if reference_embedding is None:
            rejected.append((subject_id, VOICE_REFERENCE_REASON_EMBEDDING_INVALID))
            continue
        if len(reference_embedding) != len(incoming):
            rejected.append((subject_id, VOICE_REFERENCE_REASON_EMBEDDING_DIMENSION_MISMATCH))
            continue
        similarity = _cosine_similarity(incoming, reference_embedding)
        existing = best_by_subject.get(subject_id)
        if existing is None or similarity > existing[0]:
            best_by_subject[subject_id] = (similarity, reference)

    rejected_tuple = tuple(sorted(rejected))
    if not best_by_subject:
        return _provisional_result(
            provisional_id,
            provisional_name,
            _no_compatible_reference_reason(rejected_tuple),
            rejected_references=rejected_tuple,
        )

    ranked = sorted(
        best_by_subject.items(),
        key=lambda item: (
            -item[1][0],
            item[0],
            subject_name_key(item[1][1].subject_name),
        ),
    )
    top_subject_id, (top_similarity, top_reference) = ranked[0]
    runner_up_similarity = ranked[1][1][0] if len(ranked) > 1 else None
    margin = top_similarity - runner_up_similarity if runner_up_similarity is not None else None
    candidate_scores = tuple((subject_id, round(score, 4)) for subject_id, (score, _) in ranked)
    common = {
        "similarity": round(top_similarity, 4),
        "runner_up_similarity": round(runner_up_similarity, 4) if runner_up_similarity is not None else None,
        "margin": round(margin, 4) if margin is not None else None,
        "compatible_reference_count": len(ranked),
        "candidate_scores": candidate_scores,
        "rejected_references": rejected_tuple,
    }

    if top_similarity < float(threshold):
        return _provisional_result(
            provisional_id,
            provisional_name,
            VOICE_MATCH_REASON_BELOW_THRESHOLD,
            **common,
        )
    if margin is not None and margin + 1e-12 < float(min_margin):
        return _provisional_result(
            provisional_id,
            provisional_name,
            VOICE_MATCH_REASON_AMBIGUOUS_MARGIN,
            **common,
        )

    return VoiceMatchResult(
        decision=VOICE_MATCH_DECISION_MATCHED,
        subject_id=top_subject_id,
        subject_name=normalize_subject_name(top_reference.subject_name),
        subject_type=str(top_reference.subject_type or "named").strip() or "named",
        reason=VOICE_MATCH_REASON_MATCHED,
        **common,
    )


def update_voice_profile_centroid(
    reference: VoiceProfileReference,
    embedding: Sequence[float],
    *,
    embedding_model: str,
) -> VoiceProfileReference | None:
    """Add one sample to a centroid only when model and dimensions are compatible."""

    reference_model = _model_key(reference.embedding_model)
    incoming_model = _model_key(embedding_model)
    if not reference_model or reference_model != incoming_model:
        return None
    current = _coerce_embedding(reference.embedding)
    incoming = _coerce_embedding(embedding)
    if current is None or incoming is None or len(current) != len(incoming):
        return None
    try:
        sample_count = int(reference.sample_count)
    except (TypeError, ValueError):
        return None
    if sample_count <= 0:
        return None
    next_count = sample_count + 1
    centroid = tuple(
        ((current_value * sample_count) + incoming_value) / next_count
        for current_value, incoming_value in zip(current, incoming)
    )
    return replace(reference, embedding=centroid, sample_count=next_count)


def _provisional_result(
    subject_id: str,
    subject_name: str,
    reason: str,
    **kwargs: object,
) -> VoiceMatchResult:
    return VoiceMatchResult(
        decision=VOICE_MATCH_DECISION_PROVISIONAL,
        subject_id=subject_id,
        subject_name=subject_name,
        subject_type="provisional",
        reason=reason,
        **kwargs,
    )


def _model_key(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _coerce_embedding(value: Sequence[float]) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        embedding = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not embedding or any(not math.isfinite(item) for item in embedding):
        return None
    if sum(item * item for item in embedding) <= 0.0:
        return None
    return embedding


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    similarity = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def _no_compatible_reference_reason(rejected: tuple[tuple[str, str], ...]) -> str:
    reasons = {reason for _, reason in rejected}
    if reasons and reasons <= {
        VOICE_REFERENCE_REASON_EMBEDDING_MODEL_MISSING,
        VOICE_REFERENCE_REASON_EMBEDDING_MODEL_MISMATCH,
    }:
        return VOICE_MATCH_REASON_NO_COMPATIBLE_MODEL
    if reasons == {VOICE_REFERENCE_REASON_EMBEDDING_DIMENSION_MISMATCH}:
        return VOICE_MATCH_REASON_NO_COMPATIBLE_DIMENSION
    if reasons == {VOICE_REFERENCE_REASON_EMBEDDING_INVALID}:
        return VOICE_MATCH_REASON_NO_VALID_REFERENCE_EMBEDDING
    return VOICE_MATCH_REASON_NO_COMPATIBLE_REFERENCES
