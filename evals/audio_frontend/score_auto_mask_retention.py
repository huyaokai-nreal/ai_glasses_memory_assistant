#!/usr/bin/env python3
"""Evaluate how much oracle spatial-enhancement gain survives automatic masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "eval_ali_auto_mask_retention.v1"


def evaluate_retention(
    oracle_payload: dict[str, Any],
    auto_payload: dict[str, Any],
    threshold_payload: dict[str, Any],
    *,
    enhance_variant: str,
    diar_variant: str,
) -> dict[str, Any]:
    baseline = oracle_payload["variants"]["ch0"]
    oracle = oracle_payload["variants"][enhance_variant]
    automatic = auto_payload["variants"][enhance_variant]
    selected_diar = threshold_payload["variants"][diar_variant]
    baseline_overlap = float(baseline["overlap_exposed"]["normalized_cer"])
    oracle_overlap = float(oracle["overlap_exposed"]["normalized_cer"])
    auto_overlap = float(automatic["overlap_exposed"]["normalized_cer"])
    oracle_gain = baseline_overlap - oracle_overlap
    auto_gain = baseline_overlap - auto_overlap
    if oracle_gain <= 0:
        raise ValueError("oracle candidate did not improve the baseline")
    retention = auto_gain / oracle_gain
    improved_cases = sum(
        1
        for case_id, score in automatic["per_case"].items()
        if score["normalized_cer"] < baseline["per_case"][case_id]["normalized_cer"]
    )
    selected = selected_diar["selected"]
    tune = selected["tune"]["aggregate"]
    holdout = selected["holdout"]["aggregate"]
    checks = {
        "diarization_tune_der": tune["der"] <= 0.25,
        "diarization_tune_overlap_recall": tune["overlap_frame_recall"] >= 0.70,
        "diarization_holdout_der": holdout["der"] <= 0.25,
        "diarization_holdout_overlap_recall": holdout["overlap_frame_recall"] >= 0.70,
        "overlap_cer": auto_overlap <= 0.30,
        "oracle_gain_retained": retention >= 0.80,
        "cases_improved": improved_cases >= 7,
        "frontend_rtf": automatic.get("frontend_rtf") is not None
        and automatic["frontend_rtf"] <= 1.0,
    }
    return {
        "schema": SCHEMA,
        "enhance_variant": enhance_variant,
        "diar_variant": diar_variant,
        "activity_threshold": selected_diar["selected_threshold"],
        "baseline": {
            "overlap_cer": baseline_overlap,
            "non_overlap_cer": baseline["non_overlap"]["normalized_cer"],
        },
        "oracle": {
            "overlap_cer": oracle_overlap,
            "non_overlap_cer": oracle["non_overlap"]["normalized_cer"],
        },
        "automatic": {
            "overlap_cer": auto_overlap,
            "non_overlap_cer": automatic["non_overlap"]["normalized_cer"],
            "all_cer": automatic["all"]["normalized_cer"],
            "frontend_rtf": automatic.get("frontend_rtf"),
            "cases_improved_vs_ch0": improved_cases,
        },
        "overlap_gain": {
            "oracle_absolute": oracle_gain,
            "automatic_absolute": auto_gain,
            "retention": retention,
        },
        "diarization_selected": {"tune": tune, "holdout": holdout},
        "checks": checks,
        "passed": all(checks.values()),
        "scope": {
            "oracle_segment_boundaries": True,
            "evaluation_speaker_mapping": True,
            "continuous_asr": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-scores", type=Path, required=True)
    parser.add_argument("--auto-scores", type=Path, required=True)
    parser.add_argument("--threshold-scan", type=Path, required=True)
    parser.add_argument("--enhance-variant", default="micA")
    parser.add_argument("--diar-variant", default="micA_mean")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_retention(
        json.loads(args.oracle_scores.read_text(encoding="utf-8")),
        json.loads(args.auto_scores.read_text(encoding="utf-8")),
        json.loads(args.threshold_scan.read_text(encoding="utf-8")),
        enhance_variant=args.enhance_variant,
        diar_variant=args.diar_variant,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "passed": result["passed"],
        "automatic_overlap_cer": result["automatic"]["overlap_cer"],
        "retention": result["overlap_gain"]["retention"],
        "checks": result["checks"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
