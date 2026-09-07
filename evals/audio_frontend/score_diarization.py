#!/usr/bin/env python3
"""Score automatic diarization on the complete AliMeeting window annotation.

The stage-12 package contains 255 intervals selected for CER. DER must not use
that subset because omitted short or boundary-clipped speech would be counted as
false alarm. This scorer therefore reloads every source ``reference_interval``
and clips it to each 75-second window before frame scoring.

Predictions are JSONL records with ``case_id``, ``input_variant`` and a
``segments`` list. Each segment contains relative ``start_s``, ``end_s`` and an
anonymous ``speaker``. Speaker labels are mapped to reference labels separately
for each meeting by maximum overlap; the mapping is evaluation-only.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = "eval_ali_diarization_score.v1"
PREDICTION_SCHEMA = "eval_ali_sortformer_predictions.v1"
DEFAULT_FRAME_SECONDS = 0.08


class DiarizationInputError(ValueError):
    """The reference or prediction evidence is incomplete or inconsistent."""


def load_complete_references(prep_manifest: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    """Load and window-clip all source annotations, including short intervals."""
    prep = json.loads(Path(prep_manifest).read_text(encoding="utf-8"))
    source_cases: dict[str, dict[str, Any]] = {}
    if prep.get("source_run"):
        source_path = Path(prep["source_run"]) / "run-manifest.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_cases = {case["case_id"]: case for case in source.get("cases") or []}
    references: dict[str, list[dict[str, Any]]] = {}
    durations: dict[str, float] = {}
    for prep_case in prep.get("cases") or []:
        case_id = prep_case["case_id"]
        start = float(prep_case["window_start_s"])
        end = float(prep_case["window_end_s"])
        if end <= start:
            raise DiarizationInputError(f"invalid window for {case_id}: {start}..{end}")
        direct_intervals = prep_case.get("diarization_reference_intervals")
        source_case = source_cases.get(case_id)
        if direct_intervals is None and source_case is None:
            raise DiarizationInputError(f"reference intervals are missing for {case_id}")
        clipped: list[dict[str, Any]] = []
        for interval in direct_intervals if direct_intervals is not None else source_case.get("reference_intervals") or []:
            if direct_intervals is not None:
                clipped_start = max(0.0, float(interval["start_s"]))
                clipped_end = min(end - start, float(interval["end_s"]))
                speaker = str(interval["speaker"])
                if clipped_end > clipped_start:
                    clipped.append({"speaker": speaker, "start_s": clipped_start, "end_s": clipped_end})
                continue
            clipped_start = max(start, float(interval["start_s"]))
            clipped_end = min(end, float(interval["end_s"]))
            if clipped_end <= clipped_start:
                continue
            clipped.append({
                "speaker": str(interval["speaker"]),
                "start_s": clipped_start - start,
                "end_s": clipped_end - start,
            })
        if not clipped:
            raise DiarizationInputError(f"no complete reference intervals for {case_id}")
        references[case_id] = clipped
        durations[case_id] = end - start
    if not references:
        raise DiarizationInputError("prep manifest contains no cases")
    return references, durations


def load_predictions(path: Path) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Return variant -> case -> segments and the prediction run manifest."""
    path = Path(path)
    manifest_path = path / "run-manifest.json" if path.is_dir() else path.with_name("run-manifest.json")
    jsonl_path = path / "predictions.jsonl" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PREDICTION_SCHEMA:
        raise DiarizationInputError(f"unexpected prediction schema: {manifest.get('schema')}")
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        variant = str(row["input_variant"])
        case_id = str(row["case_id"])
        if case_id in predictions.setdefault(variant, {}):
            raise DiarizationInputError(f"duplicate prediction: {variant}/{case_id}")
        predictions[variant][case_id] = list(row.get("segments") or [])
    return predictions, manifest


def _activity_matrix(
    segments: Iterable[dict[str, Any]],
    speakers: list[str],
    centers: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((len(speakers), centers.shape[0]), dtype=bool)
    speaker_index = {speaker: index for index, speaker in enumerate(speakers)}
    for segment in segments:
        start = max(0.0, float(segment["start_s"]))
        end = float(segment["end_s"])
        if end <= start:
            continue
        index = speaker_index[str(segment["speaker"])]
        matrix[index] |= (centers >= start) & (centers < end)
    return matrix


def optimal_mapping(ref: np.ndarray, hyp: np.ndarray, ref_speakers: list[str], hyp_speakers: list[str]) -> dict[str, str | None]:
    """Find the one-to-one hypothesis/reference mapping with most co-activity."""
    if not hyp_speakers:
        return {}
    overlap = hyp.astype(np.int64) @ ref.astype(np.int64).T
    dummy_count = max(0, len(hyp_speakers) - len(ref_speakers))
    targets: list[str | None] = list(ref_speakers) + [None] * dummy_count
    # Distinguish dummy slots while permuting, then collapse them back to None.
    indexed_targets = list(enumerate(targets))
    best_score = -1
    best: tuple[tuple[int, str | None], ...] | None = None
    for assignment in itertools.permutations(indexed_targets, len(hyp_speakers)):
        score = 0
        for hyp_index, (_, target) in enumerate(assignment):
            if target is not None:
                score += int(overlap[hyp_index, ref_speakers.index(target)])
        if score > best_score:
            best_score = score
            best = assignment
    assert best is not None
    return {hyp_speakers[i]: best[i][1] for i in range(len(hyp_speakers))}


def score_case(
    reference_segments: list[dict[str, Any]],
    hypothesis_segments: list[dict[str, Any]],
    duration_s: float,
    *,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
) -> dict[str, Any]:
    """Compute overlap-aware, zero-collar frame DER and overlap recall."""
    centers = np.arange(frame_seconds / 2, duration_s, frame_seconds)
    ref_speakers = sorted({str(segment["speaker"]) for segment in reference_segments})
    hyp_speakers = sorted({str(segment["speaker"]) for segment in hypothesis_segments})
    ref = _activity_matrix(reference_segments, ref_speakers, centers)
    hyp = _activity_matrix(hypothesis_segments, hyp_speakers, centers)
    mapping = optimal_mapping(ref, hyp, ref_speakers, hyp_speakers)

    mapped_hyp = np.zeros_like(ref)
    for hyp_index, speaker in enumerate(hyp_speakers):
        target = mapping[speaker]
        if target is not None:
            mapped_hyp[ref_speakers.index(target)] |= hyp[hyp_index]

    ref_count = ref.sum(axis=0)
    hyp_count = hyp.sum(axis=0)
    correct = (ref & mapped_hyp).sum(axis=0)
    miss = np.maximum(0, ref_count - hyp_count)
    false_alarm = np.maximum(0, hyp_count - ref_count)
    confusion = np.minimum(ref_count, hyp_count) - correct
    denominator = int(ref_count.sum())
    if denominator <= 0:
        raise DiarizationInputError("reference contains no scored speaker frames")

    overlap_ref = ref_count >= 2
    overlap_hyp = hyp_count >= 2
    overlap_ref_frames = int(overlap_ref.sum())
    overlap_hyp_frames = int(overlap_hyp.sum())
    overlap_detected_frames = int((overlap_ref & overlap_hyp).sum())
    overlap_correct_speaker_frames = int(correct[overlap_ref].sum())
    overlap_ref_speaker_frames = int(ref_count[overlap_ref].sum())
    errors = int(miss.sum() + false_alarm.sum() + confusion.sum())
    return {
        "reference_speakers": len(ref_speakers),
        "hypothesis_speakers": len(hyp_speakers),
        "speaker_mapping": mapping,
        "reference_speaker_frames": denominator,
        "miss_frames": int(miss.sum()),
        "false_alarm_frames": int(false_alarm.sum()),
        "confusion_frames": int(confusion.sum()),
        "errors": errors,
        "der": errors / denominator,
        "overlap_reference_frames": overlap_ref_frames,
        "overlap_hypothesis_frames": overlap_hyp_frames,
        "overlap_detected_frames": overlap_detected_frames,
        "overlap_frame_recall": overlap_detected_frames / overlap_ref_frames if overlap_ref_frames else None,
        "overlap_frame_precision": overlap_detected_frames / overlap_hyp_frames if overlap_hyp_frames else None,
        "overlap_reference_speaker_frames": overlap_ref_speaker_frames,
        "overlap_correct_speaker_frames": overlap_correct_speaker_frames,
        "overlap_speaker_recall": (
            overlap_correct_speaker_frames / overlap_ref_speaker_frames if overlap_ref_speaker_frames else None
        ),
    }


def aggregate_cases(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "reference_speaker_frames", "miss_frames", "false_alarm_frames",
        "confusion_frames", "errors", "overlap_reference_frames",
        "overlap_hypothesis_frames", "overlap_detected_frames",
        "overlap_reference_speaker_frames", "overlap_correct_speaker_frames",
    )
    totals = {key: sum(int(case[key]) for case in cases.values()) for key in keys}
    ref = totals["reference_speaker_frames"]
    overlap_ref = totals["overlap_reference_frames"]
    overlap_hyp = totals["overlap_hypothesis_frames"]
    overlap_spk_ref = totals["overlap_reference_speaker_frames"]
    totals.update({
        "der": totals["errors"] / ref if ref else None,
        "miss_rate": totals["miss_frames"] / ref if ref else None,
        "false_alarm_rate": totals["false_alarm_frames"] / ref if ref else None,
        "confusion_rate": totals["confusion_frames"] / ref if ref else None,
        "overlap_frame_recall": totals["overlap_detected_frames"] / overlap_ref if overlap_ref else None,
        "overlap_frame_precision": totals["overlap_detected_frames"] / overlap_hyp if overlap_hyp else None,
        "overlap_speaker_recall": totals["overlap_correct_speaker_frames"] / overlap_spk_ref if overlap_spk_ref else None,
        "cases": len(cases),
    })
    return totals


def score_predictions(
    references: dict[str, list[dict[str, Any]]],
    durations: dict[str, float],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    expected_cases = set(references)
    for variant, predicted_cases in predictions.items():
        if set(predicted_cases) != expected_cases:
            missing = sorted(expected_cases - set(predicted_cases))
            extra = sorted(set(predicted_cases) - expected_cases)
            raise DiarizationInputError(f"{variant}: case mismatch missing={missing} extra={extra}")
        cases = {
            case_id: score_case(references[case_id], predicted_cases[case_id], durations[case_id], frame_seconds=frame_seconds)
            for case_id in sorted(references)
        }
        aggregate = aggregate_cases(cases)
        variants[variant] = {
            "aggregate": aggregate,
            "per_case": cases,
            "gate": {
                "der_max": 0.25,
                "overlap_frame_recall_min": 0.70,
                "checks": {
                    "der": aggregate["der"] is not None and aggregate["der"] <= 0.25,
                    "overlap_frame_recall": aggregate["overlap_frame_recall"] is not None
                    and aggregate["overlap_frame_recall"] >= 0.70,
                },
            },
        }
        variants[variant]["gate"]["passed"] = all(variants[variant]["gate"]["checks"].values())
    return {"schema": SCHEMA, "frame_seconds": frame_seconds, "variants": variants}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction run dir or predictions.jsonl.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-seconds", type=float, default=DEFAULT_FRAME_SECONDS)
    parser.add_argument("--allow-case-subset", action="store_true", help="Score only predicted cases for a smoke run.")
    args = parser.parse_args(argv)
    if args.frame_seconds <= 0:
        parser.error("--frame-seconds must be positive")

    references, durations = load_complete_references(args.prep_manifest)
    predictions, run_manifest = load_predictions(args.predictions)
    if args.allow_case_subset:
        predicted_case_sets = [set(cases) for cases in predictions.values()]
        if not predicted_case_sets or any(cases != predicted_case_sets[0] for cases in predicted_case_sets[1:]):
            raise DiarizationInputError("prediction variants do not contain the same smoke case subset")
        selected_cases = predicted_case_sets[0]
        if not selected_cases or not selected_cases <= set(references):
            raise DiarizationInputError("predicted smoke case subset is empty or unknown")
        references = {case_id: references[case_id] for case_id in sorted(selected_cases)}
        durations = {case_id: durations[case_id] for case_id in sorted(selected_cases)}
    result = score_predictions(references, durations, predictions, frame_seconds=args.frame_seconds)
    result["prediction_manifest"] = run_manifest
    result["reference_interval_count"] = sum(len(items) for items in references.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "score": str(args.out),
        "reference_intervals": result["reference_interval_count"],
        "variants": {
            name: {"der": value["aggregate"]["der"], "overlap_recall": value["aggregate"]["overlap_frame_recall"], "passed": value["gate"]["passed"]}
            for name, value in result["variants"].items()
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
