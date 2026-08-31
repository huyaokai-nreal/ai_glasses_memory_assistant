#!/usr/bin/env python3
"""Lock a comparable V2 ambient-audio memory-closure baseline from three runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=3, type=Path, help="Three complete V2 full-run directories")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def compatibility(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: manifest.get(key) for key in ("schema", "preset", "gold_sha256", "delivery_mode", "runtime", "cases")}


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((path / "run-manifest.json").read_text(encoding="utf-8"))
    scores = json.loads((path / "scores.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "ambient_audio_memory_v2_manifest.v1" or manifest.get("preset") != "full":
        raise ValueError(f"baseline run is not a full V2 run: {path}")
    if scores.get("status") != "complete" or scores.get("completed_cases") != 8:
        raise ValueError(f"baseline run is incomplete: {path}")
    if int((scores.get("health") or {}).get("memory_saved") or 0) != 0:
        raise ValueError(f"baseline violates ambient privacy gate: {path}")
    return manifest, scores


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loaded = [load_run(path.resolve()) for path in args.runs]
    key = compatibility(loaded[0][0])
    if any(compatibility(manifest) != key for manifest, _ in loaded[1:]):
        raise SystemExit("the three V2 runs are not comparable")

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
        "schema": "ambient_audio_memory_v2_locked_baseline.v1",
        "runs": [str(path.resolve()) for path in args.runs],
        "compatibility": key,
        "scores": {
            "health": {
                "normalized_cer": median(("health", "normalized_cer")),
                "vad_f1": median(("health", "vad", "f1")),
            },
            "closure": {
                "passed_cases": median(("closure", "passed_cases")),
                "passed_questions": median(("closure", "passed_questions")),
                "end_to_end_passed": all(bool((scores.get("closure") or {}).get("end_to_end_passed")) for _, scores in loaded),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
