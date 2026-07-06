from __future__ import annotations

from typing import Any


MEMORY_WRITE_MIN_CONFIDENCE = 0.6
QUESTION_PROFILE_MIN_CONFIDENCE = 0.85
PREFERENCE_DEDUPE_MIN_CONFIDENCE = 0.75
STRUCTURED_EVENT_DEDUPE_MIN_CONFIDENCE = 0.78
CORRECTION_DETECTION_MIN_CONFIDENCE = 0.75
OBSERVATION_UPDATE_MIN_CONFIDENCE = 0.75
CORRECTION_TARGET_MIN_CONFIDENCE = 0.75

CONFIDENCE_MEANINGS = {
    "memory_write_candidate": "candidate fact/value extraction confidence",
    "question_profile_write": "stable profile extraction confidence from a question-like source",
    "dedupe_relationship": "relationship classification confidence, not factual truth",
    "correction_detection": "correction intent confidence, not factual truth",
    "correction_target_resolution": "corrected memory target confidence, not factual truth",
    "observation_update_relationship": "observation update relationship confidence, not factual truth",
}


def confidence_policy_payload(
    *,
    purpose: str,
    confidence: Any,
    min_confidence: float,
    treatment: str,
    backend: str = "",
    reason: str = "",
) -> dict[str, Any]:
    score = _optional_float(confidence)
    return {
        "purpose": purpose,
        "meaning": CONFIDENCE_MEANINGS.get(purpose, "confidence score"),
        "confidence": score,
        "min_confidence": min_confidence,
        "passed": bool(score is not None and score >= min_confidence),
        "treatment": treatment,
        "backend": backend,
        "reason": reason,
        "band": _confidence_band(score, min_confidence),
    }


def attach_confidence_policy(
    payload: dict[str, Any],
    *,
    purpose: str,
    min_confidence: float,
    treatment: str,
) -> dict[str, Any]:
    enriched = dict(payload)
    enriched["confidence_policy"] = confidence_policy_payload(
        purpose=purpose,
        confidence=enriched.get("confidence"),
        min_confidence=min_confidence,
        treatment=treatment,
        backend=str(enriched.get("backend") or ""),
        reason=str(enriched.get("reason") or ""),
    )
    return enriched


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _confidence_band(score: float | None, min_confidence: float) -> str:
    if score is None:
        return "missing"
    if score < min_confidence:
        return "below_threshold"
    if score >= 0.9:
        return "high"
    return "usable"
