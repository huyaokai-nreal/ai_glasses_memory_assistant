#!/usr/bin/env python3
"""Compare far-field mixed audio with aligned close-talk speaker tracks.

The existing P0 interval run supplies far-field ASR counts. This tool reuses
the exact intervals and ASR checkpoint, transcribes the matching AliMeeting
near-field speaker track, and reports paired CER deltas. Near-field tracks are
an oracle diagnostic; they are not outputs of a deployable separator.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ai_glasses_memory_assistant.audio_engine.backends import OfflineAsrBackend
from ai_glasses_memory_assistant.evals.eval_ali import EvalAliCase, ReferenceInterval, parse_textgrid
from p0_oracle_interval_ablation import (
    IntervalResult,
    load_source_run,
    score_interval,
    summarize_cer_counts,
    transcribe_interval,
)


@dataclass(frozen=True)
class PairedIntervalResult:
    case_id: str
    speaker: str
    start_s: float
    end_s: float
    duration_s: float
    overlap_exposed: bool
    reference_chars: int
    far_errors: int
    far_substitutions: int
    far_deletions: int
    far_insertions: int
    near_errors: int
    near_substitutions: int
    near_deletions: int
    near_insertions: int


def interval_key(case_id: str, speaker: str, start_s: float, end_s: float) -> tuple[str, str, float, float]:
    return case_id, speaker, round(start_s, 6), round(end_s, 6)


def near_paths(eval_root: Path, case: EvalAliCase, speaker: str) -> tuple[Path, Path]:
    if not speaker.startswith("N_SPK"):
        raise ValueError(f"unexpected AliMeeting speaker label: {speaker!r}")
    stem = f"{case.session_id}_{speaker}"
    audio = eval_root / "Eval_Ali_near" / "audio_dir" / f"{stem}.wav"
    textgrid = eval_root / "Eval_Ali_near" / "textgrid_dir" / f"{stem}.TextGrid"
    if not audio.is_file() or not textgrid.is_file():
        raise ValueError(f"matching AliMeeting near track is missing for {case.case_id}/{speaker}")
    return audio, textgrid


def _reference_signature(interval: ReferenceInterval) -> tuple[float, float, str]:
    return round(interval.start_s, 6), round(interval.end_s, 6), interval.text.strip()


def validate_near_gold(case: EvalAliCase, speaker: str, textgrid_path: Path) -> None:
    far = Counter(_reference_signature(item) for item in case.reference_intervals if item.speaker == speaker)
    near = Counter(_reference_signature(item) for item in parse_textgrid(textgrid_path) if item.text.strip())
    if not far or any(near[signature] < count for signature, count in far.items()):
        raise ValueError(f"far/near TextGrid mismatch for {case.case_id}/{speaker}")


def pair_result(far: IntervalResult, near: IntervalResult) -> PairedIntervalResult:
    if (
        far.case_id != near.case_id
        or far.speaker != near.speaker
        or round(far.start_s, 6) != round(near.start_s, 6)
        or round(far.end_s, 6) != round(near.end_s, 6)
        or far.reference_chars != near.reference_chars
        or far.overlap_exposed != near.overlap_exposed
    ):
        raise ValueError("far and near interval results are not aligned")
    return PairedIntervalResult(
        case_id=far.case_id,
        speaker=far.speaker,
        start_s=far.start_s,
        end_s=far.end_s,
        duration_s=far.duration_s,
        overlap_exposed=far.overlap_exposed,
        reference_chars=far.reference_chars,
        far_errors=far.errors,
        far_substitutions=far.substitutions,
        far_deletions=far.deletions,
        far_insertions=far.insertions,
        near_errors=near.errors,
        near_substitutions=near.substitutions,
        near_deletions=near.deletions,
        near_insertions=near.insertions,
    )


def _aggregate_side(rows: list[PairedIntervalResult], prefix: str) -> dict[str, Any]:
    return summarize_cer_counts(
        intervals=len(rows),
        audio_seconds=sum(row.duration_s for row in rows),
        reference_chars=sum(row.reference_chars for row in rows),
        errors=sum(getattr(row, f"{prefix}_errors") for row in rows),
        substitutions=sum(getattr(row, f"{prefix}_substitutions") for row in rows),
        deletions=sum(getattr(row, f"{prefix}_deletions") for row in rows),
        insertions=sum(getattr(row, f"{prefix}_insertions") for row in rows),
    )


def aggregate_paired(rows: Iterable[PairedIntervalResult]) -> dict[str, Any]:
    materialized = list(rows)
    far = _aggregate_side(materialized, "far")
    near = _aggregate_side(materialized, "near")
    far_cer = far["normalized_cer"]
    near_cer = near["normalized_cer"]
    far_deletion = far["rates_per_reference_char"]["deletion"]
    near_deletion = near["rates_per_reference_char"]["deletion"]
    return {
        "intervals": len(materialized),
        "audio_seconds": sum(row.duration_s for row in materialized),
        "reference_chars": sum(row.reference_chars for row in materialized),
        "far_mixed": far,
        "near_oracle": near,
        "near_minus_far": {
            "cer_absolute": near_cer - far_cer if near_cer is not None and far_cer is not None else None,
            "deletion_rate_absolute": near_deletion - far_deletion if near_deletion is not None and far_deletion is not None else None,
        },
        "paired_interval_outcomes": {
            "near_better": sum(row.near_errors < row.far_errors for row in materialized),
            "equal": sum(row.near_errors == row.far_errors for row in materialized),
            "near_worse": sum(row.near_errors > row.far_errors for row in materialized),
        },
    }


def load_far_results(path: Path) -> tuple[dict[str, Any], list[IntervalResult]]:
    scores = json.loads((path / "scores.json").read_text(encoding="utf-8"))
    if scores.get("schema") != "eval_ali_oracle_interval_ablation.v1":
        raise ValueError("far interval run has an unsupported schema")
    rows = [
        IntervalResult(**json.loads(line))
        for line in (path / "per_interval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int((scores.get("all") or {}).get("intervals") or -1):
        raise ValueError("far interval detail count differs from scores.json")
    return scores, rows


def run_near_oracle(
    *,
    cases: Iterable[EvalAliCase],
    far_rows: Iterable[IntervalResult],
    eval_root: Path,
    asr: OfflineAsrBackend,
) -> list[PairedIntervalResult]:
    case_by_id = {case.case_id: case for case in cases}
    reference_by_key = {
        interval_key(case.case_id, item.speaker, item.start_s, item.end_s): item
        for case in case_by_id.values()
        for item in case.reference_intervals
    }
    grouped: dict[tuple[str, str], list[IntervalResult]] = defaultdict(list)
    for row in far_rows:
        if row.case_id not in case_by_id:
            raise ValueError(f"far interval references an unknown case: {row.case_id}")
        grouped[(row.case_id, row.speaker)].append(row)

    paired_by_key: dict[tuple[str, str, float, float], PairedIntervalResult] = {}
    for (case_id, speaker), rows in grouped.items():
        case = case_by_id[case_id]
        audio_path, textgrid_path = near_paths(eval_root, case, speaker)
        validate_near_gold(case, speaker, textgrid_path)
        with sf.SoundFile(audio_path) as source:
            if source.samplerate != 16_000 or source.channels != 1:
                raise ValueError(f"unexpected AliMeeting near format: {audio_path}")
            for far in rows:
                key = interval_key(far.case_id, far.speaker, far.start_s, far.end_s)
                interval = reference_by_key.get(key)
                if interval is None:
                    raise ValueError(f"far interval is absent from source manifest: {key}")
                hypothesis = transcribe_interval(asr, source, interval)
                near = score_interval(
                    case=case,
                    interval=interval,
                    all_intervals=case.reference_intervals,
                    hypothesis=hypothesis,
                )
                paired_by_key[key] = pair_result(far, near)

    ordered = [
        paired_by_key[interval_key(row.case_id, row.speaker, row.start_s, row.end_s)]
        for row in far_rows
    ]
    if len(ordered) != len(paired_by_key):
        raise ValueError("duplicate far interval keys are not supported")
    return ordered


def write_summary(path: Path, scores: dict[str, Any]) -> None:
    lines = [
        "# Eval_Ali far/near 配对 ASR 上限诊断",
        "",
        "- far 是现有远场阵列 channel 0 混音；near 是同一说话人的近讲耳麦轨。",
        "- near 是 oracle 上限，不是分离模型输出，也不是 Android 真机成绩。",
        "- 两侧使用相同金标区间、相同参考文本和相同 ASR checkpoint。",
        "",
        "## 结果",
        "",
    ]
    for label, title in (("all", "全部"), ("non_overlap", "无重叠"), ("overlap_exposed", "重叠暴露")):
        row = scores[label]
        lines.append(
            f"- {title}：far CER {row['far_mixed']['normalized_cer']} → near CER {row['near_oracle']['normalized_cer']}；"
            f"far 删除率 {row['far_mixed']['rates_per_reference_char']['deletion']} → near 删除率 {row['near_oracle']['rates_per_reference_char']['deletion']}；"
            f"区间 {row['intervals']}。"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "far/near 差值同时包含距离、混响、阵列/耳麦响应、串音和重叠影响；它用于估计前端可改善空间，不能单独归因给某一种分离算法。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--far-interval-run", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=ROOT / "data" / "Eval_Ali")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "p1_overlap_oracle")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit-cases", type=int, default=0, help="Debug-only case limit.")
    args = parser.parse_args(argv)

    far_run = args.far_interval_run.resolve()
    far_scores, far_rows = load_far_results(far_run)
    source_profile, cases = load_source_run(Path(far_scores["source_run"]))
    if far_scores.get("asr_model_sha256") != source_profile.get("asr_model_sha256"):
        raise ValueError("far interval run and source manifest use different ASR hashes")
    if args.limit_cases:
        allowed = {case.case_id for case in cases[: args.limit_cases]}
        cases = [case for case in cases if case.case_id in allowed]
        far_rows = [row for row in far_rows if row.case_id in allowed]
    if not far_rows:
        raise ValueError("no far interval rows selected")

    asr = OfflineAsrBackend()
    capability = asr.capability()
    if capability.status != "ready":
        raise ValueError(f"ASR is not ready: {capability.reason}")
    paired = run_near_oracle(cases=cases, far_rows=far_rows, eval_root=args.eval_root.resolve(), asr=asr)
    scores = {
        "schema": "eval_ali_far_near_overlap_baseline.v1",
        "far_interval_run": str(far_run),
        "source_run": far_scores["source_run"],
        "source_profile": source_profile.get("name"),
        "asr_model_sha256": source_profile.get("asr_model_sha256"),
        "cases": len(cases),
        "all": aggregate_paired(paired),
        "non_overlap": aggregate_paired(row for row in paired if not row.overlap_exposed),
        "overlap_exposed": aggregate_paired(row for row in paired if row.overlap_exposed),
        "limitations": [
            "Near-field audio is an oracle close-talk recording, not separator output.",
            "Oracle TextGrid boundaries are unavailable in real use.",
            "Far/near differences combine acoustics, distance, cross-talk, and overlap.",
            "This offline benchmark does not validate Android microphone routing or device acoustics.",
        ],
    }
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "per_interval.jsonl").open("w", encoding="utf-8") as handle:
        for row in paired:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    (run_dir / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(run_dir / "summary.md", scores)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
