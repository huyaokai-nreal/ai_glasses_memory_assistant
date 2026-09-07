"""Unit tests for the P2 GSS scoring script.

These verify the importable, fail-closed pieces without the ASR model or GPU:
- discover_gss_outputs maps <variant>/<segment_id>.wav correctly.
- discover fails closed on unknown / missing segments.
- ASR model-hash drift between prepare and score fails closed.
- CER aggregation matches the P0 reference definitions.
- score_variant transcribes via a fake ASR and aggregates consistently.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "evals" / "audio_frontend"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from score_multichannel_cer import (  # noqa: E402
    AsrHashMismatchError,
    DuplicateSegmentError,
    MissingSegmentsError,
    UnknownSegmentError,
    _assert_asr_hash,
    discover_gss_outputs,
    evaluate_gates,
    index_segments,
    load_frontend_rtfs,
    score_variant,
)
from p0_oracle_interval_ablation import IntervalResult, aggregate  # noqa: E402

SAMPLE_RATE = 16_000


@pytest.fixture
def workdir() -> Path:
    """Fresh unique temp dir; avoids the sandbox workdir broker on re-runs."""
    path = Path(tempfile.mkdtemp(prefix="af_score_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _make_mono_wav(path: Path, seconds: float = 0.3) -> None:
    data = np.zeros(int(seconds * SAMPLE_RATE), dtype="int16")
    sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_16")


def _make_prep_manifest(tmp: Path) -> tuple[Path, dict[str, object]]:
    seg1 = {
        "segment_id": "R1__SPK1__0500__1000",
        "case_id": "R1-smoke",
        "speaker": "SPK1",
        "original_start_s": 0.5,
        "original_end_s": 1.0,
        "rel_start_s": 0.5,
        "rel_end_s": 1.0,
        "text": "你好",
        "overlap_exposed": False,
    }
    seg2 = {
        "segment_id": "R1__SPK2__1200__1500",
        "case_id": "R1-smoke",
        "speaker": "SPK2",
        "original_start_s": 1.2,
        "original_end_s": 1.5,
        "rel_start_s": 1.2,
        "rel_end_s": 1.5,
        "text": "世界",
        "overlap_exposed": True,
    }
    mani = {
        "schema": "eval_ali_gss_prep.v1",
        "asr_model_sha256": "abc123",
        "channel_variants": [
            {"name": "ch0", "channels": [0]},
            {"name": "micA", "channels": [0, 2, 4, 6]},
            {"name": "micB", "channels": [1, 3, 5, 7]},
            {"name": "all8", "channels": [0, 1, 2, 3, 4, 5, 6, 7]},
        ],
        "cases": [{"case_id": "R1-smoke", "session_id": "R1", "segments": [seg1, seg2]}],
    }
    p = tmp / "prep-manifest.json"
    p.write_text(json.dumps(mani, ensure_ascii=False), encoding="utf-8")
    return p, mani


def _write_gss_outputs(gss_root: Path, mani: dict[str, object]) -> None:
    for variant in mani["channel_variants"]:  # type: ignore[attr-defined]
        vdir = gss_root / variant["name"]  # type: ignore[index]
        vdir.mkdir(parents=True, exist_ok=True)
        for case in mani["cases"]:  # type: ignore[attr-defined]
            for seg in case["segments"]:  # type: ignore[index]
                _make_mono_wav(vdir / f"{seg['segment_id']}.wav")  # type: ignore[index]


def test_index_segments_flattens_all_cases() -> None:
    _, mani = _make_prep_manifest(Path("/tmp"))  # not used on disk here
    idx = index_segments(mani)  # type: ignore[arg-type]
    assert set(idx) == {"R1__SPK1__0500__1000", "R1__SPK2__1200__1500"}
    assert idx["R1__SPK1__0500__1000"]["case_id"] == "R1-smoke"


def test_index_segments_duplicate_manifest_id_fails_closed() -> None:
    _, mani = _make_prep_manifest(Path("/tmp"))
    duplicate = dict(mani["cases"][0]["segments"][0])  # type: ignore[index]
    mani["cases"][0]["segments"].append(duplicate)  # type: ignore[index]
    with pytest.raises(DuplicateSegmentError):
        index_segments(mani)  # type: ignore[arg-type]


def test_discover_success(workdir: Path) -> None:
    _, mani = _make_prep_manifest(workdir)
    gss = workdir / "gss"
    _write_gss_outputs(gss, mani)
    discovered = discover_gss_outputs(gss, mani)  # type: ignore[arg-type]
    assert set(discovered) == {"ch0", "micA", "micB", "all8"}
    assert set(discovered["ch0"]) == {"R1__SPK1__0500__1000", "R1__SPK2__1200__1500"}


def test_discover_can_scope_to_one_variant(workdir: Path) -> None:
    _, mani = _make_prep_manifest(workdir)
    gss = workdir / "gss"
    vdir = gss / "micA"
    vdir.mkdir(parents=True)
    for seg in mani["cases"][0]["segments"]:  # type: ignore[index]
        _make_mono_wav(vdir / f"{seg['segment_id']}.wav")
    discovered = discover_gss_outputs(gss, mani, {"micA"})  # type: ignore[arg-type]
    assert set(discovered) == {"micA"}


def test_discover_unknown_segment_fails_closed(workdir: Path) -> None:
    _, mani = _make_prep_manifest(workdir)
    gss = workdir / "gss"
    vdir = gss / "ch0"
    vdir.mkdir(parents=True)
    _make_mono_wav(vdir / "NOT_A_REAL_SEGMENT__X__0000__0000.wav")
    with pytest.raises(UnknownSegmentError):
        discover_gss_outputs(gss, mani)  # type: ignore[arg-type]


def test_discover_missing_segment_fails_closed(workdir: Path) -> None:
    _, mani = _make_prep_manifest(workdir)
    gss = workdir / "gss"
    vdir = gss / "ch0"
    vdir.mkdir(parents=True)
    # Write only one of the two expected segments for ch0.
    _make_mono_wav(vdir / "R1__SPK1__0500__1000.wav")
    with pytest.raises(MissingSegmentsError):
        discover_gss_outputs(gss, mani)  # type: ignore[arg-type]


def test_assert_asr_hash_mismatch_fails_closed() -> None:
    manifest = {"asr_model_sha256": "abc123"}
    with pytest.raises(AsrHashMismatchError):
        _assert_asr_hash({"asr_model_sha256": "different"}, manifest)  # type: ignore[arg-type]
    # Equal hash passes without raising.
    _assert_asr_hash({"asr_model_sha256": "abc123"}, manifest)  # type: ignore[arg-type]


def test_aggregate_matches_p0_reference() -> None:
    perfect = IntervalResult(
        case_id="R1-smoke", speaker="SPK1", start_s=0.5, end_s=1.0, duration_s=0.5,
        overlap_exposed=False, reference_chars=4, errors=0, substitutions=0, deletions=0, insertions=0,
    )
    assert aggregate([perfect])["normalized_cer"] == 0.0
    with_del = IntervalResult(
        case_id="R1-smoke", speaker="SPK1", start_s=0.5, end_s=1.0, duration_s=0.5,
        overlap_exposed=False, reference_chars=4, errors=2, substitutions=0, deletions=2, insertions=0,
    )
    assert aggregate([with_del])["normalized_cer"] == 0.5
    assert aggregate([with_del])["rates_per_reference_char"]["deletion"] == 0.5


def test_score_variant_uses_fake_asr_and_aggregates(workdir: Path) -> None:
    _, mani = _make_prep_manifest(workdir)
    gss = workdir / "gss"
    _write_gss_outputs(gss, mani)

    class FakeAsr:
        def transcribe(self, audio):  # noqa: ANN001
            return "你好"  # matches seg1 ("你好") perfectly; mismatches seg2 ("世界")

    discovered = discover_gss_outputs(gss, mani)  # type: ignore[arg-type]
    result = score_variant(FakeAsr(), mani, "ch0", discovered)  # type: ignore[arg-type]
    # seg1 perfect, seg2 mismatched -> overall CER between 0 and 1, per-case present.
    assert result["segments"] == 2
    assert "R1-smoke" in result["per_case"]
    assert result["all"]["intervals"] == 2


def test_load_frontend_rtfs_reads_enhance_manifest(workdir: Path) -> None:
    path = workdir / "enhance-manifest.json"
    path.write_text(json.dumps({"variants": {"ch0": {"cpu_rtf": 0.01}, "micB": {"cpu_rtf": 0.08}}}))
    assert load_frontend_rtfs(path) == {"ch0": 0.01, "micB": 0.08}


def test_evaluate_gates_uses_scored_baseline() -> None:
    def score(overlap: float, non: float, per_case: list[float], rtf: float) -> dict[str, object]:
        return {
            "overlap_exposed": {"normalized_cer": overlap},
            "non_overlap": {"normalized_cer": non},
            "per_case": {f"c{i}": {"normalized_cer": value} for i, value in enumerate(per_case)},
            "frontend_rtf": rtf,
        }

    baseline = [0.5] * 8
    scores = {
        "ch0": score(0.43, 0.11, baseline, 0.001),
        "micB": score(0.24, 0.10, [0.3] * 8, 0.01),
        "micA": score(0.24, 0.12, [0.3] * 8, 0.01),
    }
    result = evaluate_gates(scores)  # type: ignore[arg-type]
    assert result["passing_candidates"] == ["micB"]
    assert result["variants"]["micB"]["cases_improved_vs_baseline"] == 8
    assert result["variants"]["micA"]["checks"]["non_overlap_cer"] is False
