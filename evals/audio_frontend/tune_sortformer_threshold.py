#!/usr/bin/env python3
"""Tune Sortformer activity threshold on half the meetings and verify on holdout.

The default 0.5 postprocessing misses too many overlap frames. This script uses
the raw probabilities already emitted by the pinned model, scans a small fixed
threshold grid, selects on four meetings, and reports the untouched remaining
four meetings separately. It does not rerun the neural model or alter ASR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from score_diarization import (  # noqa: E402
    DEFAULT_FRAME_SECONDS,
    load_complete_references,
    score_predictions,
)

SCHEMA = "eval_ali_sortformer_threshold_scan.v1"
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50)


def probabilities_to_segments(
    probabilities: np.ndarray,
    threshold: float,
    *,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
) -> list[dict[str, Any]]:
    """Convert T x S activity probabilities into anonymous speaker segments."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError(f"expected [frames, speakers], got {probabilities.shape}")
    segments: list[dict[str, Any]] = []
    for speaker_index in range(probabilities.shape[1]):
        active = probabilities[:, speaker_index] >= threshold
        padded = np.pad(active.astype(np.int8), (1, 1))
        changes = np.flatnonzero(np.diff(padded))
        for start_frame, end_frame in zip(changes[0::2], changes[1::2]):
            if end_frame <= start_frame:
                continue
            segments.append({
                "speaker": f"speaker_{speaker_index}",
                "start_s": float(start_frame * frame_seconds),
                "end_s": float(end_frame * frame_seconds),
                "confidence": float(np.mean(probabilities[start_frame:end_frame, speaker_index])),
            })
    return sorted(segments, key=lambda item: (item["start_s"], item["speaker"]))


def load_probability_index(run_dir: Path) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {}
    for line in (Path(run_dir) / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = row.get("probabilities_path")
        if not path:
            raise ValueError("prediction run does not contain saved probabilities")
        index.setdefault(row["input_variant"], {})[row["case_id"]] = Path(path)
    return index


def subset_score(
    references: dict[str, list[dict[str, Any]]],
    durations: dict[str, float],
    predictions: dict[str, list[dict[str, Any]]],
    case_ids: list[str],
) -> dict[str, Any]:
    result = score_predictions(
        {case_id: references[case_id] for case_id in case_ids},
        {case_id: durations[case_id] for case_id in case_ids},
        {"candidate": {case_id: predictions[case_id] for case_id in case_ids}},
    )["variants"]["candidate"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-manifest", type=Path, required=True)
    parser.add_argument("--prediction-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    references, durations = load_complete_references(args.prep_manifest)
    probability_index = load_probability_index(args.prediction_run)
    case_ids = sorted(references)
    # Alternation keeps both halves mixed across the sorted 4/3/2-speaker cases;
    # a contiguous split made the holdout mostly easier two-speaker meetings.
    tune_cases = case_ids[::2]
    holdout_cases = case_ids[1::2]
    variants: dict[str, Any] = {}
    for variant, paths in probability_index.items():
        if set(paths) != set(case_ids):
            raise ValueError(f"{variant}: probability case set does not match references")
        candidates: list[dict[str, Any]] = []
        for threshold in THRESHOLDS:
            predictions = {
                case_id: probabilities_to_segments(np.load(paths[case_id], allow_pickle=False), threshold)
                for case_id in case_ids
            }
            candidates.append({
                "threshold": threshold,
                "tune": subset_score(references, durations, predictions, tune_cases),
                "holdout": subset_score(references, durations, predictions, holdout_cases),
                "full": subset_score(references, durations, predictions, case_ids),
            })
        eligible = [
            item for item in candidates
            if item["tune"]["aggregate"]["overlap_frame_recall"] >= 0.70
        ]
        selected = min(
            eligible or candidates,
            key=lambda item: (
                item["tune"]["aggregate"]["der"],
                -item["tune"]["aggregate"]["overlap_frame_recall"],
            ),
        )
        holdout_gate = selected["holdout"]["gate"]["passed"]
        variants[variant] = {
            "selection_rule": "lowest tune DER among thresholds with tune overlap recall >= 0.70",
            "selected_threshold": selected["threshold"],
            "selected": selected,
            "holdout_gate_passed": holdout_gate,
            "candidates": candidates,
        }

    payload = {
        "schema": SCHEMA,
        "prediction_run": str(args.prediction_run.resolve()),
        "frame_seconds": DEFAULT_FRAME_SECONDS,
        "thresholds": THRESHOLDS,
        "split_method": "alternating_sorted_case_ids",
        "tune_cases": tune_cases,
        "holdout_cases": holdout_cases,
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "selected": {
            name: {
                "threshold": result["selected_threshold"],
                "holdout_der": result["selected"]["holdout"]["aggregate"]["der"],
                "holdout_overlap_recall": result["selected"]["holdout"]["aggregate"]["overlap_frame_recall"],
                "passed": result["holdout_gate_passed"],
            }
            for name, result in variants.items()
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
