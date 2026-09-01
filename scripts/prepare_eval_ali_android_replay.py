#!/usr/bin/env python3
"""Prepare Eval_Ali V2 smoke windows as 16 kHz mono PCM16 WAV files for Android debug replay.

The generated WAV files are intentionally outside the V2 report directory. Android exports only
final event JSON after replay; neither this script nor the scorer puts audio into reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_glasses_memory_assistant.evals.eval_ali import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "Eval_Ali")
    parser.add_argument("--out", type=Path, required=True, help="New output directory for debug replay WAV files.")
    parser.add_argument("--channel", type=int, default=0, help="Zero-based source channel; the comparison matrix fixes this to 0.")
    return parser.parse_args()


def ensure_safe_output(path: Path) -> Path:
    output = path.resolve()
    reports = (ROOT / "reports" / "eval_ali").resolve()
    if output == reports or reports in output.parents:
        raise ValueError("--out must not be inside reports/eval_ali because V2 reports must not retain audio")
    if output.exists():
        raise FileExistsError(f"--out already exists: {output}")
    return output


def main() -> int:
    args = parse_args()
    output = ensure_safe_output(args.out)
    manifest = build_manifest(args.root.resolve(), "smoke")
    cases = manifest["cases"]
    if args.channel < 0:
        raise ValueError("--channel must be non-negative")
    output.mkdir(parents=True)
    prepared: list[dict[str, object]] = []
    for case in cases:
        source = Path(str(case["audio_path"]))
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        if sample_rate != 16_000:
            raise ValueError(f"{source}: expected 16 kHz Eval_Ali audio, got {sample_rate}")
        if args.channel >= audio.shape[1]:
            raise ValueError(f"{source}: channel {args.channel} is unavailable (channels={audio.shape[1]})")
        start = round(float(case["start_s"]) * sample_rate)
        end = round(float(case["end_s"]) * sample_rate)
        window = np.asarray(audio[start:end, args.channel], dtype=np.float32)
        target = output / f"{case['case_id']}.wav"
        sf.write(target, window, sample_rate, subtype="PCM_16")
        prepared.append({"case_id": case["case_id"], "wav": target.name, "sample_rate": sample_rate, "channels": 1, "duration_seconds": len(window) / sample_rate})
    (output / "replay-manifest.json").write_text(
        json.dumps({"schema": "eval_ali_android_replay_input.v1", "preset": "smoke", "channel": args.channel, "cases": prepared}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(prepared)} Android debug replay WAV files in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
