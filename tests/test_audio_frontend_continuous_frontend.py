"""Unit tests for the continuous multichannel frontend (watermark, MVDR, identity).

Pure synthetic signals and synthetic probability matrices: no AliMeeting audio,
no Sortformer weights, no network. These tests run in the ``py311`` evaluation
environment and must stay importable from ``hermes`` as well.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "evals" / "audio_frontend"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from run_continuous_frontend import (  # noqa: E402
    CASE_FIELD_WHITELIST,
    CORE_BLOCK_SECONDS,
    FRAME_SECONDS,
    GOLD_FORBIDDEN_FIELDS,
    FrontendInputError,
    _frame_bounds,
    activity_mask,
    check_segment_integrity,
    contiguous_runs,
    load_probabilities,
    max_energy_channel,
    plan_block_segments,
    plan_blocks,
    probability_frame_count,
    process_variant,
    N_FFT,
    segment_id,
    select_case_fields,
    track_name,
    write_segment_wav,
)

SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _synthetic_probabilities(duration_s: float, active_ranges: dict[int, tuple[float, float]]) -> np.ndarray:
    frames = probability_frame_count(duration_s)
    probabilities = np.zeros((frames, 4), dtype=np.float32)
    for track, (start_s, end_s) in active_ranges.items():
        start = int(np.floor(start_s / FRAME_SECONDS))
        end = int(np.ceil(end_s / FRAME_SECONDS))
        probabilities[start:end, track] = 0.9
    return probabilities


def _directional_mix(duration_s: float = 4.0) -> np.ndarray:
    """8-channel mix: 1 kHz from broadside, 3 kHz delayed by 2 samples/channel."""
    total = int(duration_s * SAMPLE_RATE)
    t = np.arange(total) / SAMPLE_RATE
    talker_a = np.sin(2 * np.pi * 1000.0 * t)
    talker_b = np.sin(2 * np.pi * 3000.0 * t)
    a_mask = (t >= 0.0) & (t < 2.5)
    b_mask = (t >= 1.0) & (t < 3.5)
    data = np.zeros((total, 8), dtype=np.float32)
    for channel in range(8):
        shifted_b = np.roll(talker_b, 2 * channel)
        column = np.zeros(total, dtype=np.float32)
        column[a_mask] += talker_a[a_mask]
        column[b_mask] += shifted_b[b_mask]
        data[:, channel] = column
    data += np.random.default_rng(7).normal(0.0, 0.001, size=data.shape).astype(np.float32)
    return data


def _band_power(signal: np.ndarray, frequency: float) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / SAMPLE_RATE)
    index = int(np.argmin(np.abs(freqs - frequency)))
    window = slice(max(0, index - 3), index + 4)
    return float(np.sum(spectrum[window] ** 2))


def _write_wav(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_16")
    return path


def _run_variant(
    tmp_path: Path,
    *,
    duration_s: float,
    active_ranges: dict[int, tuple[float, float]],
    variant: str,
    channels: list[int],
    data: np.ndarray | None = None,
) -> dict:
    wav_path = _write_wav(tmp_path / "mix.wav", data if data is not None else _directional_mix(duration_s))
    activity = activity_mask(_synthetic_probabilities(duration_s, active_ranges), 0.30)
    blocks = plan_blocks(duration_s)
    return process_variant(
        variant=variant,
        channels=channels,
        wav_path=wav_path,
        activity=activity,
        probabilities=_synthetic_probabilities(duration_s, active_ranges),
        blocks=blocks,
        duration_s=duration_s,
        valid_frames=probability_frame_count(duration_s),
        out_dir=tmp_path / "out",
        session_id="TEST_SESSION",
        case_id="TEST-full",
    )


# --------------------------------------------------------------------------- #
# Watermark / block ownership
# --------------------------------------------------------------------------- #


def test_plan_blocks_cover_duration_and_flush_last_block() -> None:
    blocks = plan_blocks(75.0)
    assert [(block.core_start_s, block.core_end_s) for block in blocks] == [(0.0, 30.0), (30.0, 60.0), (60.0, 75.0)]
    assert blocks[0].read_end_s == 33.2
    assert blocks[-1].read_end_s == 75.0  # nothing left to look ahead at


def test_plan_blocks_own_every_frame_exactly_once() -> None:
    duration = 1573.85
    blocks = plan_blocks(duration)
    valid = probability_frame_count(duration)
    owned: list[tuple[int, int]] = []
    for block in blocks:
        owned.append(_frame_bounds(block, duration, valid))
    assert owned[0][0] == 0
    for (_, previous_end), (next_start, _) in zip(owned, owned[1:]):
        assert previous_end == next_start
    assert owned[-1][1] == valid


def test_cross_boundary_utterance_is_split_at_block_edges() -> None:
    duration = 70.0
    # 26.4 s and 70.0 s sit exactly on the 80 ms probability grid.
    activity = activity_mask(_synthetic_probabilities(duration, {0: (26.4, 70.0)}), 0.30)
    segments: list[tuple[int, float, float]] = []
    for block in plan_blocks(duration):
        lo, hi = _frame_bounds(block, duration, probability_frame_count(duration))
        segments.extend(plan_block_segments(activity, lo, hi, duration))
    assert [(round(start, 2), round(end, 2)) for _, start, end in segments] == [(26.4, 30.0), (30.0, 60.0), (60.0, 70.0)]
    for (_, _, previous_end), (_, next_start, _) in zip(segments, segments[1:]):
        assert previous_end <= next_start + 1e-9


def test_long_utterance_never_exceeds_core_block_length() -> None:
    duration = 100.0
    activity = activity_mask(_synthetic_probabilities(duration, {0: (0.0, 100.0)}), 0.30)
    longest = 0.0
    for block in plan_blocks(duration):
        lo, hi = _frame_bounds(block, duration, probability_frame_count(duration))
        for _, start, end in plan_block_segments(activity, lo, hi, duration):
            longest = max(longest, end - start)
    assert longest <= CORE_BLOCK_SECONDS + 1e-6


# --------------------------------------------------------------------------- #
# Activity, runs and identity
# --------------------------------------------------------------------------- #


def test_contiguous_runs_and_activity_threshold() -> None:
    assert contiguous_runs(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]
    mask = activity_mask(np.array([[0.29], [0.30], [0.90]], dtype=np.float32), 0.30)
    assert mask.tolist() == [[False], [True], [True]]


def test_two_overlapping_tracks_produce_independent_segments() -> None:
    duration = 4.0
    probabilities = _synthetic_probabilities(duration, {0: (0.0, 2.5), 1: (1.0, 3.5)})
    activity = activity_mask(probabilities, 0.30)
    segments = plan_block_segments(activity, 0, activity.shape[0], duration)
    tracks = sorted({track for track, _, _ in segments})
    assert tracks == [0, 1]
    assert [(round(start, 2), round(end, 2)) for track, start, end in segments if track == 0] == [(0.0, 2.56)]
    assert [(round(start, 2), round(end, 2)) for track, start, end in segments if track == 1] == [(0.96, 3.52)]


def test_segment_id_is_deterministic_and_variant_scoped() -> None:
    first = segment_id("S1", "spk_01", 6.64, 22.72, "all8")
    assert first == segment_id("S1", "spk_01", 6.64, 22.72, "all8")
    assert first != segment_id("S1", "spk_01", 6.64, 22.72, "ch0")
    assert first != segment_id("S1", "spk_02", 6.64, 22.72, "all8")
    assert track_name(0) == "spk_01"
    assert track_name(3) == "spk_04"


def test_check_segment_integrity_detects_duplicates_overlaps_and_range() -> None:
    good = [
        {"segment_id": "a", "variant": "all8", "track": "spk_01", "start_s": 0.0, "end_s": 1.0},
        {"segment_id": "b", "variant": "all8", "track": "spk_01", "start_s": 1.0, "end_s": 2.0},
    ]
    assert check_segment_integrity(good, 10.0)["passed"] is True
    duplicate = good + [dict(good[0])]
    assert check_segment_integrity(duplicate, 10.0)["duplicate_segment_ids"] == 1
    overlapping = good + [
        {"segment_id": "c", "variant": "all8", "track": "spk_01", "start_s": 1.5, "end_s": 2.5}
    ]
    assert check_segment_integrity(overlapping, 10.0)["same_track_overlaps"] == 1
    out_of_range = good + [
        {"segment_id": "d", "variant": "all8", "track": "spk_01", "start_s": 9.0, "end_s": 12.0}
    ]
    assert check_segment_integrity(out_of_range, 10.0)["out_of_range_intervals"] == 1


# --------------------------------------------------------------------------- #
# Gold isolation and fail-closed inputs
# --------------------------------------------------------------------------- #


def test_select_case_fields_drops_gold_fields() -> None:
    case = {
        "case_id": "R1-full",
        "session_id": "R1",
        "cut_wav_path": "/tmp/mix.wav",
        "cut_wav_sha256": "abc",
        "window_start_s": 0.0,
        "window_end_s": 10.0,
        "textgrid_path": "/tmp/gold.TextGrid",
        "textgrid_sha256": "def",
        "diarization_reference_intervals": [{"speaker": "S", "start_s": 0.0, "end_s": 1.0}],
    }
    selected = select_case_fields(case)
    assert set(selected) == set(CASE_FIELD_WHITELIST)
    for field in GOLD_FORBIDDEN_FIELDS:
        assert field not in selected
        assert field not in CASE_FIELD_WHITELIST


def test_probability_loader_rejects_model_drift(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-manifest.json").write_text(
        '{"schema": "eval_ali_sortformer_predictions.v1", "model": {"sha256": "deadbeef"}, "runtime": {}, "postprocessing_threshold": 0.30}',
        encoding="utf-8",
    )
    try:
        load_probabilities(run_dir, "case", "micA_mean", 0.30)
    except FrontendInputError as exc:
        assert "model drift" in str(exc)
    else:  # pragma: no cover - the call must fail
        raise AssertionError("model drift must fail closed")


def test_probability_loader_rejects_threshold_mismatch(tmp_path: Path) -> None:
    from run_continuous_frontend import MODEL_SHA256, NEMO_COMMIT

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-manifest.json").write_text(
        '{"schema": "eval_ali_sortformer_predictions.v1", "model": {"sha256": "%s"}, "runtime": {"nemo_commit": "%s"}, "postprocessing_threshold": 0.30}'
        % (MODEL_SHA256, NEMO_COMMIT),
        encoding="utf-8",
    )
    try:
        load_probabilities(run_dir, "case", "micA_mean", 0.35)
    except FrontendInputError as exc:
        assert "threshold mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("threshold mismatch must fail closed")


# --------------------------------------------------------------------------- #
# Enhancement: MVDR, fallback and all8/ch0 alignment
# --------------------------------------------------------------------------- #


def test_all8_suppresses_interferer_more_than_ch0(tmp_path: Path) -> None:
    duration = 4.0
    ranges = {0: (0.0, 2.5), 1: (1.0, 3.5)}
    data = _directional_mix(duration)
    enhanced = _run_variant(tmp_path / "all8", duration_s=duration, active_ranges=ranges, variant="all8", channels=list(range(8)), data=data)
    baseline = _run_variant(tmp_path / "ch0", duration_s=duration, active_ranges=ranges, variant="ch0", channels=[0], data=data)

    track_segments = {row["track"]: row for row in enhanced["rows"]}
    assert set(track_segments) == {"spk_01", "spk_02"}
    all8_wave, _ = sf.read(str(tmp_path / "all8" / "out" / "segments" / "all8" / f"{track_segments['spk_01']['segment_id']}.wav"), dtype="float32")
    ch0_row = next(row for row in baseline["rows"] if row["track"] == "spk_01")
    ch0_wave, _ = sf.read(str(tmp_path / "ch0" / "out" / "segments" / "ch0" / f"{ch0_row['segment_id']}.wav"), dtype="float32")

    all8_ratio = _band_power(all8_wave, 3000.0) / max(_band_power(all8_wave, 1000.0), 1e-12)
    ch0_ratio = _band_power(ch0_wave, 3000.0) / max(_band_power(ch0_wave, 1000.0), 1e-12)
    assert all8_ratio < ch0_ratio, f"all8={all8_ratio} ch0={ch0_ratio}"


def test_all8_and_ch0_segments_are_time_aligned(tmp_path: Path) -> None:
    duration = 4.0
    ranges = {0: (0.0, 2.5), 1: (1.0, 3.5)}
    data = _directional_mix(duration)
    enhanced = _run_variant(tmp_path / "a8", duration_s=duration, active_ranges=ranges, variant="all8", channels=list(range(8)), data=data)
    baseline = _run_variant(tmp_path / "c0", duration_s=duration, active_ranges=ranges, variant="ch0", channels=[0], data=data)
    left = {(row["track"], row["start_s"], row["end_s"]) for row in enhanced["rows"]}
    right = {(row["track"], row["start_s"], row["end_s"]) for row in baseline["rows"]}
    assert left == right
    assert all(row["enhancement"] == "pass_through" for row in baseline["rows"])
    assert all(row["enhancement"] == "mvdr" for row in enhanced["rows"])


def test_fallback_reason_recorded_when_target_never_solo(tmp_path: Path) -> None:
    duration = 3.0
    # Track 0 is only ever active together with track 1 -> no solo covariance.
    ranges = {0: (0.5, 1.5), 1: (0.0, 2.0)}
    result = _run_variant(
        tmp_path / "fallback",
        duration_s=duration,
        active_ranges=ranges,
        variant="all8",
        channels=list(range(8)),
        data=_directional_mix(duration),
    )
    first = next(row for row in result["rows"] if row["track"] == "spk_01")
    assert first["enhancement"] == "fallback_max_energy_channel"
    assert first["fallback_reason"] == "insufficient_target_solo_frames"
    assert result["stats"]["fallback_segments"] >= 1


def test_max_energy_channel_picks_the_loudest_channel() -> None:
    Zxx = np.zeros((3, 4, 2), dtype=np.complex128)
    Zxx[2] = 5.0 + 0j
    Zxx[0] = 1.0 + 0j
    assert max_energy_channel(Zxx, np.array([True, True, True, True])) == 2


def test_segments_stay_inside_the_processed_window(tmp_path: Path) -> None:
    duration = 4.0
    ranges = {0: (0.0, 2.5), 1: (1.0, 3.5)}
    result = _run_variant(
        tmp_path / "range",
        duration_s=duration,
        active_ranges=ranges,
        variant="ch0",
        channels=[0],
        data=_directional_mix(duration),
    )
    assert check_segment_integrity(result["rows"], duration)["passed"] is True
    assert all(row["end_s"] <= duration + 1e-9 for row in result["rows"])


# --------------------------------------------------------------------------- #
# WAV write contract: on-disk frame count == sample_count == round((end-start)*sr)
# --------------------------------------------------------------------------- #


def test_write_segment_wav_enforces_exact_frame_count() -> None:
    # A synthetic ISTFT output shorter than the declared interval: the helper must
    # pad with silence so the file on disk has exactly round((end_s-start_s)*16000)
    # frames. start_s=1.0 and end_s=6.3 straddle the 1s/CORE_BLOCK_SECONDS block grid,
    # exercising a segment that crosses block boundaries (P3-4).
    out = tmp_path_fixture() / "seg.wav"
    waveform = np.full(50_000, 0.25, dtype=np.float32)
    first_center = 1.0 + N_FFT / (2.0 * SAMPLE_RATE)
    sample_count, _ = write_segment_wav(
        waveform, start_s=1.0, end_s=6.3, first_center=first_center, wav_path_out=out
    )
    expected = int(round((6.3 - 1.0) * SAMPLE_RATE))
    assert expected == 84_800
    assert sample_count == expected
    data, sr = sf.read(str(out))
    assert sr == SAMPLE_RATE
    assert data.shape[0] == expected  # wav frames == sample_count == round((end-start)*sr)
    assert np.allclose(data[:10_000], 0.25)  # signal preserved
    assert np.allclose(data[-10_000:], 0.0)   # tail padded with silence


def test_write_segment_wav_pads_short_waveform_with_silence() -> None:
    out = tmp_path_fixture() / "seg2.wav"
    # Much shorter than the target window => padded with silence to the exact count.
    waveform = np.full(5_000, 0.5, dtype=np.float32)
    first_center = 0.0 + N_FFT / (2.0 * SAMPLE_RATE)
    sample_count, _ = write_segment_wav(
        waveform, start_s=0.0, end_s=1.0, first_center=first_center, wav_path_out=out
    )
    expected = int(round(1.0 * SAMPLE_RATE))
    assert expected == 16_000
    assert sample_count == expected
    data, _ = sf.read(str(out))
    assert data.shape[0] == expected
    assert np.allclose(data[:5_000], 0.5)
    assert np.allclose(data[5_000:], 0.0)


def tmp_path_fixture() -> Path:
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="fe-wav-test-"))
    path.mkdir(parents=True, exist_ok=True)
    return path

