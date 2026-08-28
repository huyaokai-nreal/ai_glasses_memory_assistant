#!/usr/bin/env python3
"""Lock a comparable Eval_Ali offline streaming baseline from three runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=3, type=Path, help="Three completed offline run directories")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def compatibility_key(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest.get("schema"), "preset": manifest.get("preset"), "channel": manifest.get("channel"),
        "delivery_mode": manifest.get("delivery_mode"), "frame_samples": manifest.get("frame_samples"),
        "safety_policy": manifest.get("safety_policy"), "runtime": manifest.get("runtime"),
        "cases": [{key: item.get(key) for key in ("case_id", "audio_sha256", "textgrid_sha256", "start_s", "end_s")}
                  for item in manifest.get("cases") or []],
    }


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((path / "run-manifest.json").read_text(encoding="utf-8"))
    scores = json.loads((path / "scores.json").read_text(encoding="utf-8"))
    if scores.get("status") != "complete" or scores.get("completed_cases") != 8:
        raise ValueError(f"baseline run is incomplete: {path}")
    return manifest, scores


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loaded = [load_run(path.resolve()) for path in args.runs]
    key = compatibility_key(loaded[0][0])
    if any(compatibility_key(manifest) != key for manifest, _ in loaded[1:]):
        raise SystemExit("the three offline runs are not comparable")

    def median(path: tuple[str, ...]) -> float | None:
        values: list[float] = []
        for _, scores in loaded:
            value: Any = scores
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                values.append(float(value))
        return statistics.median(values) if values else None

    payload = {
        "schema": "eval_ali_offline_locked_baseline.v1",
        "runs": [str(path.resolve()) for path in args.runs],
        "compatibility": key,
        "scores": {
            "normalized_cer": median(("normalized_cer",)),
            "vad": {"f1": median(("vad", "f1"))},
            "throughput": {"wall_seconds_per_audio_second": median(("throughput", "wall_seconds_per_audio_second"))},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
