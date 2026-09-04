#!/usr/bin/env python3
"""Measure ASR error on oracle speech intervals, split by overlap exposure.

This diagnostic reuses the exact ASR model recorded by an Eval_Ali run manifest.
It scores fully contained TextGrid intervals one by one and reports separate
non-overlap and overlap-exposed CER components. It does not perform diarization
or source separation: overlap intervals still contain mixed channel-0 audio.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_glasses_memory_assistant.audio_engine.backends import OfflineAsrBackend
from ai_glasses_memory_assistant.evals.eval_ali import EvalAliCase, ReferenceInterval, score_cer, sha256_file

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class IntervalResult:
    case_id: str
    speaker: str
    start_s: float
    end_s: float
    duration_s: float
    overlap_exposed: bool
    reference_chars: int
    errors: int
    substitutions: int
    deletions: int
    insertions: int


def case_from_dict(payload: dict[str, Any]) -> EvalAliCase:
    return EvalAliCase(
        **{key: value for key, value in payload.items() if key != "reference_intervals"},
        reference_intervals=tuple(ReferenceInterval(**item) for item in payload["reference_intervals"]),
    )


def overlaps_other_speaker(
    interval: ReferenceInterval,
    intervals: Iterable[ReferenceInterval],
) -> bool:
    return any(
        other is not interval
        and other.speaker != interval.speaker
        and min(interval.end_s, other.end_s) > max(interval.start_s, other.start_s)
        for other in intervals
    )


def score_interval(
    *,
    case: EvalAliCase,
    interval: ReferenceInterval,
    all_intervals: tuple[ReferenceInterval, ...],
    hypothesis: str,
) -> IntervalResult:
    normalized = score_cer(interval.text, hypothesis)["normalized"]
    return IntervalResult(
        case_id=case.case_id,
        speaker=interval.speaker,
        start_s=interval.start_s,
        end_s=interval.end_s,
        duration_s=interval.end_s - interval.start_s,
        overlap_exposed=overlaps_other_speaker(interval, all_intervals),
        reference_chars=int(normalized["reference_chars"]),
        errors=int(normalized["errors"]),
        substitutions=int(normalized["substitutions"]),
        deletions=int(normalized["deletions"]),
        insertions=int(normalized["insertions"]),
    )


def aggregate(rows: Iterable[IntervalResult]) -> dict[str, Any]:
    materialized = list(rows)
    reference_chars = sum(row.reference_chars for row in materialized)
    errors = sum(row.errors for row in materialized)
    substitutions = sum(row.substitutions for row in materialized)
    deletions = sum(row.deletions for row in materialized)
    insertions = sum(row.insertions for row in materialized)

    return summarize_cer_counts(
        intervals=len(materialized),
        audio_seconds=sum(row.duration_s for row in materialized),
        reference_chars=reference_chars,
        errors=errors,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
    )


def summarize_cer_counts(
    *,
    intervals: int,
    audio_seconds: float,
    reference_chars: int,
    errors: int,
    substitutions: int,
    deletions: int,
    insertions: int,
) -> dict[str, Any]:
    def rate(value: int, denominator: int) -> float | None:
        return value / denominator if denominator else None

    return {
        "intervals": intervals,
        "audio_seconds": audio_seconds,
        "reference_chars": reference_chars,
        "errors": errors,
        "normalized_cer": rate(errors, reference_chars),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "rates_per_reference_char": {
            "substitution": rate(substitutions, reference_chars),
            "deletion": rate(deletions, reference_chars),
            "insertion": rate(insertions, reference_chars),
        },
        "shares_of_errors": {
            "substitution": rate(substitutions, errors),
            "deletion": rate(deletions, errors),
            "insertion": rate(insertions, errors),
        },
    }


def load_source_run(path: Path) -> tuple[dict[str, Any], list[EvalAliCase]]:
    manifest_path = path / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = ((manifest.get("runtime") or {}).get("ambient_audio_profile") or {})
    if profile.get("asr_backend") != "sherpa_sensevoice":
        raise ValueError("source run must use sherpa_sensevoice for reproducible interval scoring")
    model_dir = Path(str(profile.get("asr_model_dir") or ""))
    model_path = model_dir / "model.int8.onnx"
    tokens_path = model_dir / "tokens.txt"
    if not model_path.is_file() or not tokens_path.is_file():
        raise ValueError("source run ASR model files are unavailable")
    if profile.get("asr_model_sha256") and sha256_file(model_path) != profile["asr_model_sha256"]:
        raise ValueError("source run ASR model hash changed")
    if profile.get("tokens_sha256") and sha256_file(tokens_path) != profile["tokens_sha256"]:
        raise ValueError("source run ASR token hash changed")
    os.environ["AI_GLASSES_AMBIENT_ASR_BACKEND"] = str(profile["asr_backend"])
    os.environ["AI_GLASSES_AMBIENT_ASR_MODEL_DIR"] = str(model_dir)
    return profile, [case_from_dict(item) for item in manifest.get("cases") or []]


def transcribe_interval(asr: OfflineAsrBackend, source: sf.SoundFile, interval: ReferenceInterval) -> str:
    return transcribe_range(asr, source, interval.start_s, interval.end_s)


def transcribe_range(asr: OfflineAsrBackend, source: sf.SoundFile, start_s: float, end_s: float) -> str:
    start = round(start_s * SAMPLE_RATE)
    frames = round((end_s - start_s) * SAMPLE_RATE)
    source.seek(start)
    audio = source.read(frames, dtype="float32", always_2d=True)
    if len(audio) != frames:
        raise ValueError("source WAV ended before the reference interval")
    return str(asr.transcribe(np.asarray(audio[:, 0], dtype=np.float32)) or "").strip()


def run_segment_length_ablation(
    source_run: Path,
    cases: Iterable[EvalAliCase],
    asr: OfflineAsrBackend,
    *,
    segment_lengths: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    """Keep production VAD regions fixed and only vary maximum ASR segment length."""

    counters = {
        length: {"intervals": 0, "audio_seconds": 0.0, "reference_chars": 0, "errors": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
        for length in segment_lengths
    }
    for case in cases:
        health = json.loads((source_run / case.case_id / "health.json").read_text(encoding="utf-8"))
        regions = [
            event for event in health.get("events") or []
            if event.get("type") in {"transcript_final", "speech_rejected"}
        ]
        with sf.SoundFile(case.audio_path) as source:
            for length in segment_lengths:
                hypothesis: list[str] = []
                for event in regions:
                    cursor = float(event["start_s"])
                    end_s = float(event["end_s"])
                    while cursor < end_s - 1e-6:
                        stop = min(end_s, cursor + length)
                        hypothesis.append(transcribe_range(asr, source, cursor, stop))
                        cursor = stop
                normalized = score_cer(str(health.get("reference_text") or ""), "".join(hypothesis))["normalized"]
                target = counters[length]
                target["intervals"] += len(regions)
                target["audio_seconds"] += sum(float(event["end_s"]) - float(event["start_s"]) for event in regions)
                for name in ("reference_chars", "errors", "substitutions", "deletions", "insertions"):
                    target[name] += int(normalized[name])
    return {
        str(length): summarize_cer_counts(**values)
        for length, values in counters.items()
    }


def run_cases(
    cases: Iterable[EvalAliCase],
    asr: OfflineAsrBackend,
    *,
    min_interval_seconds: float,
) -> tuple[list[IntervalResult], dict[str, int]]:
    results: list[IntervalResult] = []
    excluded = {"boundary_truncated": 0, "shorter_than_minimum": 0}
    for case in cases:
        intervals = case.reference_intervals
        with sf.SoundFile(case.audio_path) as source:
            if source.samplerate != SAMPLE_RATE or source.channels != 8:
                raise ValueError(f"unexpected Eval_Ali source format: {case.audio_path}")
            for interval in intervals:
                if interval.start_s < case.start_s or interval.end_s > case.end_s:
                    excluded["boundary_truncated"] += 1
                    continue
                if interval.end_s - interval.start_s < min_interval_seconds:
                    excluded["shorter_than_minimum"] += 1
                    continue
                hypothesis = transcribe_interval(asr, source, interval)
                results.append(
                    score_interval(
                        case=case,
                        interval=interval,
                        all_intervals=intervals,
                        hypothesis=hypothesis,
                    )
                )
    return results, excluded


def write_summary(path: Path, scores: dict[str, Any]) -> None:
    all_rows = scores["all"]
    clean = scores["non_overlap"]
    overlap = scores["overlap_exposed"]
    lines = [
        "# Eval_Ali oracle 时间边界 ASR 诊断",
        "",
        "- 本诊断使用金标时间边界切音频；它不执行声纹分离或说话人分轨。",
        "- 重叠区间仍是 channel 0 混音，因此只能观察重叠暴露的负担。",
        "- 为避免首尾窗口截断制造假删除，只评分完全位于 case 窗口内的参考区间。",
        f"- ASR profile：`{scores['source_profile']}`；模型 hash：`{scores['asr_model_sha256']}`",
        f"- 有效区间：{all_rows['intervals']}；排除：{scores['excluded']}",
        "",
        "## 结果",
        "",
        f"- 全部 oracle 区间 CER：{all_rows['normalized_cer']}；删除率：{all_rows['rates_per_reference_char']['deletion']}",
        f"- 无重叠区间 CER：{clean['normalized_cer']}；删除率：{clean['rates_per_reference_char']['deletion']}",
        f"- 重叠暴露区间 CER：{overlap['normalized_cer']}；删除率：{overlap['rates_per_reference_char']['deletion']}",
        "",
        "## 固定生产 VAD 区间的段长消融",
        "",
    ]
    lines.extend(
        f"- 最长 {length} 秒：CER {row['normalized_cer']}；删除率 {row['rates_per_reference_char']['deletion']}"
        for length, row in scores["segment_length_ablation"].items()
    )
    lines.extend([
        "",
        "这里的差值同时包含重叠混音、区间长度和说话方式差异，不能单独证明需要哪一种 diarization 架构。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True, help="Completed Eval_Ali run containing run-manifest.json.")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "p0_oracle_interval")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit-cases", type=int, default=0, help="Debug-only case limit.")
    parser.add_argument("--min-interval-seconds", type=float, default=0.5)
    parser.add_argument("--segment-lengths", default="5,10,15,30", help="Comma-separated maximum segment seconds.")
    args = parser.parse_args(argv)
    if args.min_interval_seconds <= 0:
        parser.error("--min-interval-seconds must be positive")

    try:
        segment_lengths = tuple(dict.fromkeys(float(item) for item in args.segment_lengths.split(",") if item.strip()))
    except ValueError:
        parser.error("--segment-lengths must contain comma-separated numbers")
    if not segment_lengths or any(item <= 0 for item in segment_lengths):
        parser.error("--segment-lengths must contain positive values")

    source_run = args.source_run.resolve()
    profile, cases = load_source_run(source_run)
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    if not cases:
        raise ValueError("source run manifest contains no cases")
    asr = OfflineAsrBackend()
    capability = asr.capability()
    if capability.status != "ready":
        raise ValueError(f"ASR is not ready: {capability.reason}")

    results, excluded = run_cases(cases, asr, min_interval_seconds=args.min_interval_seconds)
    segment_length_ablation = run_segment_length_ablation(
        source_run,
        cases,
        asr,
        segment_lengths=segment_lengths,
    )
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "per_interval.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    scores = {
        "schema": "eval_ali_oracle_interval_ablation.v1",
        "source_run": str(args.source_run.resolve()),
        "source_profile": profile.get("name"),
        "asr_model_sha256": profile.get("asr_model_sha256"),
        "cases": len(cases),
        "minimum_interval_seconds": args.min_interval_seconds,
        "excluded": excluded,
        "all": aggregate(results),
        "non_overlap": aggregate(row for row in results if not row.overlap_exposed),
        "overlap_exposed": aggregate(row for row in results if row.overlap_exposed),
        "segment_length_ablation": segment_length_ablation,
        "limitations": [
            "Oracle TextGrid boundaries are unavailable in real use.",
            "Overlap audio remains a single mixed channel; this is not diarization or source separation.",
            "Interval duration and speaking style can differ between the two groups.",
        ],
    }
    (run_dir / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(run_dir / "summary.md", scores)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
