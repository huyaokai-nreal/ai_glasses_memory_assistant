"""Unit tests for the P2 GSS input-package preparation script.

These run on synthetic fixtures only (no 5 GB AliMeeting data, no ASR, no GPU).
They verify the four required properties:
- RTTM times are zeroed to the window start.
- Short / boundary-truncated intervals are NOT re-added (prep only follows per_interval.jsonl).
- The four channel variants are fixed and written to the manifest.
- Re-running identical input yields stable cut hashes and an identical manifest (ignoring timestamp).
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

from prepare_ali_multichannel import (  # noqa: E402
    CHANNEL_VARIANTS,
    SAMPLE_RATE,
    prepare_package,
    segment_id,
)


@pytest.fixture
def workdir() -> Path:
    """Fresh unique temp dir; avoids the sandbox workdir broker on re-runs."""
    path = Path(tempfile.mkdtemp(prefix="af_prepare_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _make_8ch_wav(path: Path, seconds: float = 2.0) -> None:
    data = np.zeros((int(seconds * SAMPLE_RATE), 8), dtype="int16")
    # Put a tiny non-zero pilot on channel 0 so the file is not all-silent.
    data[:1000, 0] = 1
    sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_16")


def _build_source_run(tmp: Path) -> tuple[Path, Path]:
    """Create a minimal source run with one case and three reference intervals.

    Only the 0.5-1.0s interval is in per_interval.jsonl (>=0.5s, inside window).
    A 0.2s short interval and a boundary-truncated interval are deliberately kept
    OUT of per_interval.jsonl to prove they are never re-added.
    """
    far = tmp / "R1_MS801.wav"
    _make_8ch_wav(far, seconds=2.0)
    reference_intervals = [
        {"speaker": "SPK1", "start_s": 0.5, "end_s": 1.0, "text": "你好世界"},
        {"speaker": "SPK1", "start_s": 1.2, "end_s": 1.4, "text": "短"},  # 0.2s short
        {"speaker": "SPK2", "start_s": -0.1, "end_s": 0.3, "text": "边界"},  # boundary truncated
    ]
    manifest = {
        "runtime": {"ambient_audio_profile": {"asr_backend": "sherpa_sensevoice", "asr_model_sha256": "abc123"}},
        "cases": [
            {
                "case_id": "R1-smoke",
                "session_id": "R1",
                "audio_path": str(far),
                "start_s": 0.0,
                "end_s": 1.5,
                "reference_intervals": reference_intervals,
            }
        ],
    }
    (tmp / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rows = [
        {
            "case_id": "R1-smoke",
            "speaker": "SPK1",
            "start_s": 0.5,
            "end_s": 1.0,
            "duration_s": 0.5,
            "overlap_exposed": False,
            "reference_chars": 4,
            "errors": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
        }
    ]
    per = tmp / "per_interval.jsonl"
    per.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return tmp, per


def test_segment_id_is_deterministic_and_variant_independent() -> None:
    a = segment_id("R1-smoke", "SPK1", 0.5, 1.0)
    b = segment_id("R1-smoke", "SPK1", 0.5, 1.0)
    assert a == b
    assert a.startswith("R1-smoke__SPK1__")


def test_rttm_times_zeroed_and_free_of_gold_text(workdir: Path) -> None:
    src, per = _build_source_run(workdir)
    out = workdir / "out"
    mani = prepare_package(source_run=src, per_interval_path=per, out_dir=out, gss_commit="c1", gss_license="MIT")

    assert mani["total_segments"] == 1
    seg = mani["cases"][0]["segments"][0]
    assert seg["rel_start_s"] == 0.5
    assert seg["rel_end_s"] == 1.0

    rttm = (out / mani["cases"][0]["rttm_path"]).read_text(encoding="utf-8")
    assert "SPEAKER R1 1 0.500 0.500 <NA> <NA> SPK1 <NA>" in rttm
    assert "你好世界" not in rttm  # RTTM must not carry gold text


def test_short_and_boundary_intervals_not_readded(workdir: Path) -> None:
    src, per = _build_source_run(workdir)
    out = workdir / "out"
    mani = prepare_package(source_run=src, per_interval_path=per, out_dir=out, gss_commit="c1", gss_license="MIT")
    # Only the per_interval row becomes a segment; the 0.2s and boundary intervals are absent.
    ids = {s["segment_id"] for c in mani["cases"] for s in c["segments"]}
    assert len(ids) == 1
    assert mani["total_segments"] == 1


def test_four_channel_variants_fixed(workdir: Path) -> None:
    src, per = _build_source_run(workdir)
    out = workdir / "out"
    mani = prepare_package(source_run=src, per_interval_path=per, out_dir=out, gss_commit="c1", gss_license="MIT")
    assert [v["name"] for v in mani["channel_variants"]] == ["ch0", "micA", "micB", "all8"]
    assert [v["channels"] for v in mani["channel_variants"]] == [
        [0],
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]


def test_rerun_produces_stable_hash_and_manifest(workdir: Path) -> None:
    src, per = _build_source_run(workdir)
    out1 = workdir / "o1"
    out2 = workdir / "o2"
    m1 = prepare_package(source_run=src, per_interval_path=per, out_dir=out1, gss_commit="c1", gss_license="MIT")
    m2 = prepare_package(source_run=src, per_interval_path=per, out_dir=out2, gss_commit="c1", gss_license="MIT")

    assert m1["cases"][0]["cut_wav_sha256"] == m2["cases"][0]["cut_wav_sha256"]
    ids1 = [s["segment_id"] for c in m1["cases"] for s in c["segments"]]
    ids2 = [s["segment_id"] for c in m2["cases"] for s in c["segments"]]
    assert ids1 == ids2
    m1c = {k: v for k, v in m1.items() if k != "created_at"}
    m2c = {k: v for k, v in m2.items() if k != "created_at"}
    assert m1c == m2c


def test_refuses_to_overwrite_existing_package(workdir: Path) -> None:
    src, per = _build_source_run(workdir)
    out = workdir / "out"
    prepare_package(source_run=src, per_interval_path=per, out_dir=out, gss_commit="c1", gss_license="MIT")
    with pytest.raises(FileExistsError):
        prepare_package(source_run=src, per_interval_path=per, out_dir=out, gss_commit="c1", gss_license="MIT")
