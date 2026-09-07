"""Focused tests for overlap-aware AliMeeting diarization scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "evals" / "audio_frontend"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from score_diarization import aggregate_cases, score_case, score_predictions  # noqa: E402
from score_auto_mask_retention import evaluate_retention  # noqa: E402


REFERENCE = [
    {"speaker": "A", "start_s": 0.0, "end_s": 2.0},
    {"speaker": "B", "start_s": 1.0, "end_s": 3.0},
]


def test_score_case_is_permutation_invariant_and_overlap_aware() -> None:
    hypothesis = [
        {"speaker": "speaker_1", "start_s": 0.0, "end_s": 2.0},
        {"speaker": "speaker_0", "start_s": 1.0, "end_s": 3.0},
    ]
    result = score_case(REFERENCE, hypothesis, 3.0, frame_seconds=0.1)
    assert result["errors"] == 0
    assert result["der"] == 0.0
    assert result["overlap_detected_frames"] == result["overlap_reference_frames"]
    assert result["overlap_frame_recall"] == 1.0
    assert result["overlap_frame_precision"] == 1.0


def test_score_case_counts_missing_second_overlap_speaker() -> None:
    hypothesis = [{"speaker": "speaker_0", "start_s": 0.0, "end_s": 3.0}]
    result = score_case(REFERENCE, hypothesis, 3.0, frame_seconds=0.1)
    assert result["miss_frames"] > 0
    assert result["overlap_detected_frames"] == 0
    assert result["der"] > 0
    assert result["overlap_frame_recall"] == 0.0
    assert result["overlap_frame_precision"] is None
    aggregate = aggregate_cases({"case": result})
    assert aggregate["der"] > 0
    assert aggregate["overlap_frame_recall"] == 0.0


def test_score_case_reports_overlap_speaker_recall() -> None:
    hypothesis = [
        {"speaker": "x", "start_s": 0.0, "end_s": 2.0},
        {"speaker": "y", "start_s": 1.0, "end_s": 2.0},
    ]
    result = score_case(REFERENCE, hypothesis, 3.0, frame_seconds=0.1)
    assert result["overlap_speaker_recall"] == pytest.approx(1.0)


def test_score_predictions_applies_gate() -> None:
    predictions = {
        "ch0": {
            "case": [
                {"speaker": "x", "start_s": 0.0, "end_s": 2.0},
                {"speaker": "y", "start_s": 1.0, "end_s": 3.0},
            ]
        }
    }
    result = score_predictions({"case": REFERENCE}, {"case": 3.0}, predictions, frame_seconds=0.1)
    assert result["variants"]["ch0"]["gate"]["passed"] is True


def test_auto_mask_retention_compares_same_variant() -> None:
    def scored(overlap: float, non: float, case: float, rtf: float | None = None):
        value = {
            "overlap_exposed": {"normalized_cer": overlap},
            "non_overlap": {"normalized_cer": non},
            "all": {"normalized_cer": overlap},
            "per_case": {f"c{i}": {"normalized_cer": case} for i in range(8)},
        }
        if rtf is not None:
            value["frontend_rtf"] = rtf
        return value

    oracle_payload = {"variants": {"ch0": scored(0.43, 0.11, 0.5), "micA": scored(0.24, 0.11, 0.3)}}
    auto_payload = {"variants": {"micA": scored(0.25, 0.11, 0.32, 0.01)}}
    aggregate = {"der": 0.20, "overlap_frame_recall": 0.75}
    threshold_payload = {"variants": {"micA_mean": {
        "selected_threshold": 0.35,
        "selected": {"tune": {"aggregate": aggregate}, "holdout": {"aggregate": aggregate}},
    }}}
    result = evaluate_retention(
        oracle_payload, auto_payload, threshold_payload,
        enhance_variant="micA", diar_variant="micA_mean",
    )
    assert result["passed"] is True
    assert result["overlap_gain"]["retention"] > 0.8
