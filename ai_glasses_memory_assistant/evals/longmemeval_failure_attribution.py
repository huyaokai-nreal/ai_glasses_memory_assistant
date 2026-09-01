"""Zero-model attribution for completed LongMemEval report artifacts.

This module deliberately uses only standard-library file parsing.  It is an
offline diagnostic: it never imports the chat service, Reader, PPD, or an LLM
client, and it never changes product behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANALYSIS_SCHEMA = "longmemeval.failure-attribution.v1"
PRIMARY_CAUSES = (
    "route_not_requested",
    "write_or_type_verified",
    "retrieval_empty",
    "selection_or_truncation_verified",
    "reader_evidence_use_verified",
    "insufficient_artifact_evidence",
)
BEHAVIOR_OVERLAYS = (
    "should_abstain_but_answered",
    "should_answer_but_refused",
    "other_judge_mismatch",
)
_RECALL_FLAG_NAMES = (
    "needs_profile_memory",
    "needs_event_memory",
    "needs_observation_memory",
    "needs_timeline_recall",
    "needs_discussion_recall",
)
_REQUIRED_CASE_ARTIFACTS = (
    Path("result.json"),
    Path("audit.redacted.jsonl"),
    Path("evidence-export.json"),
    Path("database/events.db"),
    Path("database/timeline.db"),
)


class AttributionError(ValueError):
    """Raised when the frozen report cannot support a trustworthy analysis."""


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise AttributionError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AttributionError(f"invalid JSON artifact: {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_run(path: Path, run_dir: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _find_detail_path(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*.details.json"))
    if len(candidates) != 1:
        raise AttributionError(
            f"expected exactly one root-level *.details.json in {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _index_details(details: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(details, list):
        raise AttributionError("details artifact must be a JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(details):
        if not isinstance(row, dict):
            raise AttributionError(f"details row {position} is not an object")
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise AttributionError(f"details row {position} has no question_id")
        if question_id in indexed:
            raise AttributionError(f"duplicate details question_id: {question_id}")
        indexed[question_id] = row
    return indexed


def _index_case_artifacts(run_dir: Path) -> dict[str, dict[str, Path]]:
    case_root = run_dir / "cases"
    if not case_root.is_dir():
        raise AttributionError(f"missing cases directory: {case_root}")
    indexed: dict[str, dict[str, Path]] = {}
    for result_path in sorted(case_root.glob("*/result.json")):
        result = _load_json(result_path)
        if not isinstance(result, dict):
            raise AttributionError(f"case result is not an object: {result_path}")
        question_id = str(result.get("question_id") or "").strip()
        if not question_id:
            raise AttributionError(f"case result has no question_id: {result_path}")
        if question_id in indexed:
            raise AttributionError(f"duplicate case artifact question_id: {question_id}")
        case_dir = result_path.parent
        artifacts = {"result": result_path}
        for relative_path in _REQUIRED_CASE_ARTIFACTS[1:]:
            candidate = case_dir / relative_path
            if not candidate.is_file():
                raise AttributionError(f"missing case artifact for {question_id}: {candidate}")
            artifacts[relative_path.as_posix()] = candidate
        indexed[question_id] = artifacts
    return indexed


def _final_turn_decision(detail: dict[str, Any]) -> dict[str, Any]:
    debug = detail.get("response_debug")
    if not isinstance(debug, dict):
        return {}
    turn_decision = debug.get("turn_decision")
    if isinstance(turn_decision, dict) and isinstance(turn_decision.get("final"), dict):
        return turn_decision["final"]
    pre_reply = debug.get("pre_reply_decision")
    return pre_reply if isinstance(pre_reply, dict) else {}


def _offline_evidence(detail: dict[str, Any]) -> dict[str, Any]:
    evidence = detail.get("offline_evidence")
    return evidence if isinstance(evidence, dict) else {}


def _is_verified(value: Any) -> bool:
    return value is True or value == "verified"


def _recall_requested(decision: dict[str, Any]) -> bool:
    return any(bool(decision.get(name)) for name in _RECALL_FLAG_NAMES)


def _reader_debug(detail: dict[str, Any]) -> dict[str, Any]:
    debug = detail.get("reader_debug")
    return debug if isinstance(debug, dict) else {}


def _behavior_overlay(detail: dict[str, Any]) -> str:
    reader_debug = _reader_debug(detail)
    refusal = reader_debug.get("refusal")
    if bool(detail.get("is_abstention")) and refusal is False:
        return "should_abstain_but_answered"
    if not bool(detail.get("is_abstention")) and refusal is True:
        return "should_answer_but_refused"
    return "other_judge_mismatch"


def _observations(detail: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    observations: list[str] = []
    reader_debug = _reader_debug(detail)
    if _recall_requested(decision):
        observations.append("recall_requested")
    if str(detail.get("recall_context") or "").strip():
        observations.append("recall_context_nonempty")
    if reader_debug.get("refusal") is True:
        observations.append("reader_refusal_observed")
    if reader_debug.get("relevant_evidence"):
        observations.append("reader_reported_relevant_evidence")
    if bool((decision.get("temporal_query") or {}).get("has_expression")):
        observations.append("temporal_query_present")
    recall_chars = int(detail.get("recall_context_chars") or 0)
    reader_chars = int(detail.get("reader_input_chars") or 0)
    if recall_chars > reader_chars >= 0:
        observations.append("reader_input_truncated")
    if not reader_debug:
        observations.append("reader_debug_missing")
    return observations


def _classify_primary(detail: dict[str, Any], decision: dict[str, Any]) -> tuple[str, list[str]]:
    evidence = _offline_evidence(detail)
    context = str(detail.get("recall_context") or "").strip()
    proof_paths: list[str] = []
    if _is_verified(evidence.get("write_or_type_loss")):
        return "write_or_type_verified", ["offline_evidence.write_or_type_loss"]
    if _is_verified(evidence.get("candidate_support")) and not bool(evidence.get("selected_support")):
        return "selection_or_truncation_verified", [
            "offline_evidence.candidate_support",
            "offline_evidence.selected_support",
        ]
    if _is_verified(evidence.get("selected_support")):
        return "reader_evidence_use_verified", ["offline_evidence.selected_support"]
    if not _recall_requested(decision) and not context:
        return "route_not_requested", ["response_debug.turn_decision.final", "recall_context"]
    if _recall_requested(decision) and not context:
        return "retrieval_empty", ["response_debug.turn_decision.final", "recall_context"]
    return "insufficient_artifact_evidence", proof_paths


def _source_paths(case_artifacts: dict[str, Path], run_dir: Path, detail_path: Path) -> dict[str, str]:
    paths = {"details": _relative_to_run(detail_path, run_dir)}
    for name, path in sorted(case_artifacts.items()):
        paths[name] = _relative_to_run(path, run_dir)
    return paths


def analyze_run(
    run_dir: Path,
    *,
    expected_total: int = 500,
    expected_failures: int = 146,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a frozen run and return an auditable, model-free attribution set."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run-manifest.json"
    latest_path = run_dir / "eval-latest.json"
    detail_path = _find_detail_path(run_dir)
    run_manifest = _load_json(manifest_path)
    latest = _load_json(latest_path)
    details = _index_details(_load_json(detail_path))
    if not isinstance(latest, dict):
        raise AttributionError("eval-latest.json must be an object")
    judge_map = (
        latest.get("summary", {})
        .get("official_judge", {})
        .get("per_question", {})
    )
    if not isinstance(judge_map, dict):
        raise AttributionError("official_judge.per_question must be an object")
    judge_ids = set(judge_map)
    detail_ids = set(details)
    if len(detail_ids) != expected_total:
        raise AttributionError(f"expected {expected_total} unique detail IDs, found {len(detail_ids)}")
    if judge_ids != detail_ids:
        missing_in_judge = sorted(detail_ids - judge_ids)
        missing_in_details = sorted(judge_ids - detail_ids)
        raise AttributionError(
            "judge/detail ID mismatch: "
            f"missing_in_judge={missing_in_judge[:3]}, missing_in_details={missing_in_details[:3]}"
        )
    invalid_scores = {question_id: score for question_id, score in judge_map.items() if score not in {0, 1}}
    if invalid_scores:
        raise AttributionError(f"judge scores must be 0 or 1: {sorted(invalid_scores)[:3]}")
    failure_ids = sorted(question_id for question_id, score in judge_map.items() if score == 0)
    if len(failure_ids) != expected_failures:
        raise AttributionError(f"expected {expected_failures} official judge failures, found {len(failure_ids)}")
    case_artifacts = _index_case_artifacts(run_dir)
    missing_cases = sorted(set(failure_ids) - set(case_artifacts))
    if missing_cases:
        raise AttributionError(f"missing case artifacts for official failures: {missing_cases[:3]}")

    rows: list[dict[str, Any]] = []
    for question_id in failure_ids:
        detail = details[question_id]
        decision = _final_turn_decision(detail)
        primary_cause, proof_paths = _classify_primary(detail, decision)
        if primary_cause not in PRIMARY_CAUSES:
            raise AssertionError(f"unsupported primary cause: {primary_cause}")
        behavior_overlay = _behavior_overlay(detail)
        if behavior_overlay not in BEHAVIOR_OVERLAYS:
            raise AssertionError(f"unsupported behavior overlay: {behavior_overlay}")
        rows.append(
            {
                "question_id": question_id,
                "question_type": str(detail.get("question_type") or "unknown"),
                "official_judge_score": 0,
                "primary_cause": primary_cause,
                "confidence": "verified" if primary_cause != "insufficient_artifact_evidence" else "insufficient",
                "behavior_overlay": behavior_overlay,
                "review_required": primary_cause == "insufficient_artifact_evidence",
                "proof_paths": proof_paths,
                "observations": _observations(detail, decision),
                "source_paths": _source_paths(case_artifacts[question_id], run_dir, detail_path),
            }
        )

    source_files = (manifest_path, latest_path, detail_path)
    analysis_manifest = {
        "schema": ANALYSIS_SCHEMA,
        "source_run_dir": str(run_dir),
        "source_snapshot": run_manifest.get("source_snapshot"),
        "current_git_sha": _git_sha(),
        "input_files": [
            {"path": _relative_to_run(path, run_dir), "sha256": _sha256_file(path)}
            for path in source_files
        ],
        "input_counts": {
            "detail_ids": len(detail_ids),
            "judge_ids": len(judge_ids),
            "official_judge_failures": len(failure_ids),
            "case_artifacts_for_failures": len(failure_ids),
        },
        "expected_counts": {"total": expected_total, "official_judge_failures": expected_failures},
        "zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0},
    }
    return analysis_manifest, rows


def _count_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(str(row[key]) for row in rows)
    return [{key: name, "count": counts[name]} for name in sorted(counts)]


def _by_type_and_cause(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[str(row["question_type"])][str(row["primary_cause"])] += 1
    return [
        {
            "question_type": question_type,
            "wrong": sum(grouped[question_type].values()),
            "primary_causes": [
                {"primary_cause": cause, "count": grouped[question_type][cause]}
                for cause in sorted(grouped[question_type])
            ],
        }
        for question_type in sorted(grouped)
    ]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    review_rows = [row for row in rows if bool(row["review_required"])]
    return {
        "schema": ANALYSIS_SCHEMA,
        "total_official_judge_failures": len(rows),
        "primary_causes": _count_rows(rows, "primary_cause"),
        "confidence": _count_rows(rows, "confidence"),
        "behavior_overlays": _count_rows(rows, "behavior_overlay"),
        "by_question_type": _by_type_and_cause(rows),
        "review_queue_count": len(review_rows),
        "zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0},
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _render_summary(summary: dict[str, Any], analysis_manifest: dict[str, Any]) -> str:
    lines = [
        "# LongMemEval official-judge failure attribution",
        "",
        "This is an artifact-only diagnostic. It made no Qwen, judge, or network calls.",
        "",
        f"- Source run: `{analysis_manifest['source_run_dir']}`",
        f"- Official-judge failures: {summary['total_official_judge_failures']}",
        "- Zero-model gate: `qwen_calls=0`, `judge_calls=0`, `network_calls=0`",
        "",
        "## Primary causes",
        "",
        "| Cause | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {row['primary_cause']} | {row['count']} |" for row in summary["primary_causes"])
    lines.extend(["", "## By question type", "", "| Type | Wrong | Verified causes |", "| --- | ---: | --- |"])
    for row in summary["by_question_type"]:
        causes = ", ".join(f"{item['primary_cause']}={item['count']}" for item in row["primary_causes"])
        lines.append(f"| {row['question_type']} | {row['wrong']} | {causes} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Rows marked `insufficient_artifact_evidence` are review items, not evidence that Reader, retrieval, or writing caused the failure. Do not authorize a product repair from this report alone unless one general root cause has direct proof across at least five distinct cases.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_review_queue(rows: list[dict[str, Any]]) -> str:
    review_rows = [row for row in rows if bool(row["review_required"])]
    lines = [
        "# Review queue: insufficient artifact evidence",
        "",
        "These rows intentionally do not assert a write, selection, or Reader root cause.",
        "",
        "| Question ID | Type | Behavior overlay | Observations |",
        "| --- | --- | --- | --- |",
    ]
    for row in review_rows:
        observations = ", ".join(row["observations"]) or "none"
        lines.append(
            f"| {row['question_id']} | {row['question_type']} | {row['behavior_overlay']} | {observations} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_analysis(out_dir: Path, analysis_manifest: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    """Write a new immutable analysis directory after all validation succeeds."""
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise AttributionError(f"output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        summary = summarize_rows(rows)
        _write_json(temporary_dir / "analysis-manifest.json", analysis_manifest)
        _write_jsonl(temporary_dir / "failure-attribution.jsonl", rows)
        _write_json(temporary_dir / "summary.json", summary)
        (temporary_dir / "summary.md").write_text(
            _render_summary(summary, analysis_manifest), encoding="utf-8"
        )
        (temporary_dir / "review-queue.md").write_text(_render_review_queue(rows), encoding="utf-8")
        os.replace(temporary_dir, out_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a zero-Qwen LongMemEval failure-attribution report.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Frozen LongMemEval run directory to inspect read-only.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New immutable output directory; it must not exist.")
    parser.add_argument("--expected-total", type=int, default=500, help="Expected number of detail/judge IDs.")
    parser.add_argument("--expected-failures", type=int, default=146, help="Expected official-judge failure count.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis_manifest, rows = analyze_run(
            args.run_dir,
            expected_total=args.expected_total,
            expected_failures=args.expected_failures,
        )
        out_dir = write_analysis(args.out_dir, analysis_manifest, rows)
    except AttributionError as exc:
        print(f"failure attribution blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), **summarize_rows(rows)["zero_model_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
