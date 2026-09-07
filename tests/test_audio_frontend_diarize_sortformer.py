"""Unit tests for Sortformer input/output normalization without NeMo weights."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "evals" / "audio_frontend"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from diarize_sortformer import fixed_average, parse_segment, segment_confidence  # noqa: E402
from tune_sortformer_threshold import probabilities_to_segments  # noqa: E402


def test_fixed_average_uses_only_selected_channels() -> None:
    data = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    np.testing.assert_allclose(fixed_average(data, [0, 2]), [2, 6])


def test_parse_segment_accepts_nemo_string_and_tuple() -> None:
    assert parse_segment("0.080 1.200 speaker_0") == (0.08, 1.2, "speaker_0")
    assert parse_segment((1, 2, "speaker_1")) == (1.0, 2.0, "speaker_1")


def test_segment_confidence_selects_speaker_frames() -> None:
    probs = np.zeros((10, 4), dtype=np.float32)
    probs[1:4, 2] = 0.8
    confidence = segment_confidence(probs, 0.08, 0.32, "speaker_2")
    assert confidence is not None
    assert abs(confidence - 0.8) < 1e-6


def test_probabilities_to_segments_preserves_overlap() -> None:
    probs = np.zeros((5, 2), dtype=np.float32)
    probs[1:4, 0] = 0.8
    probs[2:5, 1] = 0.7
    segments = probabilities_to_segments(probs, 0.5, frame_seconds=0.1)
    assert segments == [
        {"speaker": "speaker_0", "start_s": 0.1, "end_s": 0.4, "confidence": pytest.approx(0.8)},
        {"speaker": "speaker_1", "start_s": 0.2, "end_s": 0.5, "confidence": pytest.approx(0.7)},
    ]
