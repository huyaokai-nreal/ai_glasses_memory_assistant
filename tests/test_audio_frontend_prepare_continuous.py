"""Tests for the no-copy full-session AliMeeting manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "evals" / "audio_frontend"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from prepare_ali_continuous import build_manifest  # noqa: E402
from score_diarization import load_complete_references  # noqa: E402


def test_build_manifest_uses_full_audio_and_direct_references(tmp_path: Path) -> None:
    audio_dir = tmp_path / "Eval_Ali_far" / "audio_dir"
    grid_dir = tmp_path / "Eval_Ali_far" / "textgrid_dir"
    audio_dir.mkdir(parents=True)
    grid_dir.mkdir(parents=True)
    sf.write(
        str(audio_dir / "R1_MS1.wav"),
        np.zeros((16_000, 8), dtype=np.int16),
        16_000,
        subtype="PCM_16",
    )
    (grid_dir / "R1.TextGrid").write_text(
        '\n'.join([
            'File type = "ooTextFile"', 'Object class = "TextGrid"',
            'item [1]:', 'class = "IntervalTier"', 'name = "N_SPK1"',
            'intervals [1]:', 'xmin = 0.1', 'xmax = 0.9', 'text = "你好"',
        ]),
        encoding="utf-8",
    )
    manifest = build_manifest(tmp_path)
    assert manifest["total_reference_intervals"] == 1
    assert manifest["cases"][0]["window_end_s"] == 1.0

    out = tmp_path / "prep-manifest.json"
    out.write_text(json.dumps(manifest), encoding="utf-8")
    references, durations = load_complete_references(out)
    assert durations == {"R1-full": 1.0}
    assert references["R1-full"] == [{"speaker": "N_SPK1", "start_s": 0.1, "end_s": 0.9}]
