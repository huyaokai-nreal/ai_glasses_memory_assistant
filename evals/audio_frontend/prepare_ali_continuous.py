#!/usr/bin/env python3
"""Build a manifest for full-session AliMeeting diarization without copying PCM."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_glasses_memory_assistant.evals.eval_ali import parse_textgrid, sha256_file  # noqa: E402

SCHEMA = "eval_ali_continuous_prep.v1"


def build_manifest(eval_root: Path) -> dict:
    far_root = eval_root / "Eval_Ali_far"
    cases = []
    for textgrid in sorted((far_root / "textgrid_dir").glob("*.TextGrid")):
        session_id = textgrid.stem
        audio_candidates = sorted((far_root / "audio_dir").glob(f"{session_id}_*.wav"))
        if len(audio_candidates) != 1:
            raise ValueError(f"expected one far audio for {session_id}, got {audio_candidates}")
        audio = audio_candidates[0]
        with sf.SoundFile(str(audio)) as source:
            if source.samplerate != 16_000 or source.channels != 8:
                raise ValueError(f"{audio}: expected 16kHz 8ch")
            duration = source.frames / source.samplerate
        intervals = [
            {"speaker": item.speaker, "start_s": item.start_s, "end_s": item.end_s}
            for item in parse_textgrid(textgrid)
            if item.text.strip() and item.end_s > item.start_s
        ]
        cases.append({
            "case_id": f"{session_id}-full",
            "session_id": session_id,
            "cut_wav_path": str(audio.resolve()),
            "cut_wav_sha256": sha256_file(audio),
            "textgrid_path": str(textgrid.resolve()),
            "textgrid_sha256": sha256_file(textgrid),
            "window_start_s": 0.0,
            "window_end_s": duration,
            "diarization_reference_intervals": intervals,
        })
    if not cases:
        raise ValueError(f"no AliMeeting sessions under {far_root}")
    return {
        "schema": SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sample_rate": 16_000,
        "channels": 8,
        "cases": cases,
        "total_audio_seconds": sum(case["window_end_s"] for case in cases),
        "total_reference_intervals": sum(len(case["diarization_reference_intervals"]) for case in cases),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=ROOT / "data" / "Eval_Ali")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite continuous manifest: {args.out}")
    args.out.mkdir(parents=True)
    manifest = build_manifest(args.eval_root.resolve())
    (args.out / "prep-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(args.out / "prep-manifest.json"),
        "cases": len(manifest["cases"]),
        "hours": manifest["total_audio_seconds"] / 3600,
        "reference_intervals": manifest["total_reference_intervals"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
