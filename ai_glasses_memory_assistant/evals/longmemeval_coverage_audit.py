"""Zero-model audit for direct evidence that was omitted from Reader context.

This evaluator-only module compares a frozen LongMemEval answer reference with
the redacted per-case SQLite exports and the recorded recall candidate trace.
It identifies reviewable coverage failures; it never imports product code,
starts a model, or changes runtime behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "longmemeval.coverage-audit.v1"
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS = frozenset({"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "with"})


class CoverageAuditError(ValueError):
    """Raised when the frozen run cannot support a trustworthy coverage audit."""


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise CoverageAuditError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CoverageAuditError(f"invalid JSON artifact: {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: Any) -> str:
    return _NORMALIZE_RE.sub(" ", str(value or "").casefold()).strip()


def _reference_terms(value: Any) -> set[str]:
    """Keep only answer-reference terms that can be checked without a model."""
    return {term for term in _normalize(value).split() if term not in _STOPWORDS}


def _matches_reference(text: Any, terms: set[str]) -> bool:
    return bool(terms) and terms.issubset(set(_normalize(text).split()))


def _find_details_path(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("*.details.json"))
    if len(paths) != 1:
        raise CoverageAuditError(
            f"expected exactly one root-level *.details.json in {run_dir}, found {len(paths)}"
        )
    return paths[0]


def _index_details(details: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(details, list):
        raise CoverageAuditError("details artifact must be a JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    for position, detail in enumerate(details):
        if not isinstance(detail, dict):
            raise CoverageAuditError(f"details row {position} is not an object")
        question_id = str(detail.get("question_id") or "").strip()
        if not question_id:
            raise CoverageAuditError(f"details row {position} has no question_id")
        if question_id in indexed:
            raise CoverageAuditError(f"duplicate details question_id: {question_id}")
        indexed[question_id] = detail
    return indexed


def _index_cases(run_dir: Path) -> dict[str, Path]:
    cases_dir = run_dir / "cases"
    if not cases_dir.is_dir():
        raise CoverageAuditError(f"missing cases directory: {cases_dir}")
    indexed: dict[str, Path] = {}
    for result_path in sorted(cases_dir.glob("*/result.json")):
        result = _load_json(result_path)
        if not isinstance(result, dict):
            raise CoverageAuditError(f"case result is not an object: {result_path}")
        question_id = str(result.get("question_id") or "").strip()
        if not question_id:
            raise CoverageAuditError(f"case result has no question_id: {result_path}")
        if question_id in indexed:
            raise CoverageAuditError(f"duplicate case artifact question_id: {question_id}")
        case_dir = result_path.parent
        for relative in (Path("database/events.db"), Path("database/timeline.db")):
            if not (case_dir / relative).is_file():
                raise CoverageAuditError(f"missing case artifact for {question_id}: {case_dir / relative}")
        indexed[question_id] = case_dir
    return indexed


def _timeline_recall(detail: dict[str, Any]) -> dict[str, Any]:
    debug = detail.get("response_debug")
    if not isinstance(debug, dict):
        return {}
    timeline = debug.get("timeline")
    if not isinstance(timeline, dict):
        return {}
    recall = timeline.get("recall")
    return recall if isinstance(recall, dict) else {}


def _candidate_ids(trace: dict[str, Any], key: str) -> set[str]:
    values = trace.get(key)
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if str(value)}


def _timeline_support(case_dir: Path, terms: set[str], excluded_parent_id: str) -> list[dict[str, str]]:
    database = case_dir / "database" / "timeline.db"
    try:
        with sqlite3.connect(str(database)) as connection:
            rows = connection.execute("SELECT id, parent_id, text FROM chunks").fetchall()
    except sqlite3.Error as exc:
        raise CoverageAuditError(f"cannot read timeline database: {database}: {exc}") from exc
    support: list[dict[str, str]] = []
    for source_id, parent_id, text in rows:
        if str(parent_id or "") == excluded_parent_id:
            continue
        if _matches_reference(text, terms):
            support.append({"source_type": "timeline", "source_id": str(source_id)})
    return support


def _memory_support(case_dir: Path, terms: set[str]) -> list[dict[str, str]]:
    database = case_dir / "database" / "events.db"
    try:
        with sqlite3.connect(str(database)) as connection:
            rows = connection.execute("SELECT id, content FROM memories").fetchall()
    except sqlite3.Error as exc:
        raise CoverageAuditError(f"cannot read events database: {database}: {exc}") from exc
    return [
        {"source_type": "structured_memory", "source_id": str(source_id)}
        for source_id, text in rows
        if _matches_reference(text, terms)
    ]


def _coverage_state(*, timeline_support: list[dict[str, str]], recall: dict[str, Any]) -> str:
    trace = recall.get("candidate_trace")
    if not isinstance(trace, dict):
        return "timeline_trace_missing"
    source_ids = {item["source_id"] for item in timeline_support}
    selected = _candidate_ids(trace, "selected_ids")
    dropped = _candidate_ids(trace, "dropped_candidate_ids")
    if source_ids & selected:
        return "selected_source_absent_from_context"
    if source_ids & dropped:
        return "candidate_dropped_before_context"
    if not str(trace.get("query") or "").strip():
        return "empty_timeline_query"
    return "stored_source_not_in_candidate_trace"


def analyze_run(run_dir: Path, *, expected_total: int = 500) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Analyze one frozen run without importing or calling any model code."""
    run_dir = run_dir.resolve()
    latest_path = run_dir / "eval-latest.json"
    details_path = _find_details_path(run_dir)
    latest = _load_json(latest_path)
    details = _index_details(_load_json(details_path))
    if len(details) != expected_total:
        raise CoverageAuditError(f"expected {expected_total} unique detail IDs, found {len(details)}")
    judge_map = latest.get("summary", {}).get("official_judge", {}).get("per_question", {})
    if not isinstance(judge_map, dict):
        raise CoverageAuditError("official_judge.per_question must be an object")
    if set(judge_map) != set(details):
        raise CoverageAuditError("judge/detail ID mismatch")
    if any(score not in {0, 1} for score in judge_map.values()):
        raise CoverageAuditError("judge scores must be 0 or 1")
    cases = _index_cases(run_dir)
    failure_ids = sorted(question_id for question_id, score in judge_map.items() if score == 0)
    if missing := sorted(set(failure_ids) - set(cases)):
        raise CoverageAuditError(f"missing case artifacts for official failures: {missing[:3]}")

    rows: list[dict[str, Any]] = []
    for question_id in failure_ids:
        detail = details[question_id]
        answer = _normalize(detail.get("answer"))
        terms = _reference_terms(detail.get("answer"))
        context = detail.get("recall_context")
        # Very short numeric references are too ambiguous to be direct proof.
        if len(answer) < 3 or answer.isdigit() or _matches_reference(context, terms):
            continue
        recall = _timeline_recall(detail)
        excluded_parent_id = str(recall.get("excluded_parent_id") or "")
        timeline_support = _timeline_support(cases[question_id], terms, excluded_parent_id)
        memory_support = _memory_support(cases[question_id], terms)
        if not timeline_support and not memory_support:
            continue
        state = _coverage_state(timeline_support=timeline_support, recall=recall)
        trace = recall.get("candidate_trace") if isinstance(recall.get("candidate_trace"), dict) else {}
        rows.append(
            {
                "question_id": question_id,
                "question_type": str(detail.get("question_type") or "unknown"),
                "ground_truth_reference_used_for_diagnostic": True,
                "reader_context_contains_reference": False,
                "coverage_state": state,
                "timeline_support": timeline_support,
                "structured_memory_support": memory_support,
                "timeline_query_present": bool(str(trace.get("query") or "").strip()),
                "proof_paths": [
                    "details.answer",
                    "details.recall_context",
                    "response_debug.timeline.recall.candidate_trace",
                    "database/timeline.db:chunks",
                    "database/events.db:memories",
                ],
                "source_paths": {
                    "details": details_path.name,
                    "timeline_db": f"cases/{cases[question_id].name}/database/timeline.db",
                    "events_db": f"cases/{cases[question_id].name}/database/events.db",
                },
            }
        )
    manifest = {
        "schema": AUDIT_SCHEMA,
        "source_run": str(run_dir),
        "expected_total": expected_total,
        "official_judge_failures": len(failure_ids),
        "input_sha256": {
            "eval-latest.json": _sha256_file(latest_path),
            details_path.name: _sha256_file(details_path),
        },
        "zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0},
    }
    return manifest, rows


def write_audit(output_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write immutable JSONL and summary outputs for later manual review."""
    if output_dir.exists():
        raise CoverageAuditError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "coverage-audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    by_state = Counter(row["coverage_state"] for row in rows)
    by_type = Counter(row["question_type"] for row in rows)
    summary = {
        "schema": AUDIT_SCHEMA,
        "direct_source_absent_from_context": len(rows),
        "by_coverage_state": dict(sorted(by_state.items())),
        "by_question_type": dict(sorted(by_type.items())),
        "zero_model_gate": manifest["zero_model_gate"],
    }
    (output_dir / "analysis-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# Direct evidence omitted from Reader context", "", "Manual review is required before any product change.", ""]
    for row in rows:
        lines.append(f"- `{row['question_id']}`: `{row['coverage_state']}` ({row['question_type']})")
    (output_dir / "review-queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=500)
    args = parser.parse_args(argv)
    try:
        manifest, rows = analyze_run(args.run_dir, expected_total=args.expected_total)
        write_audit(args.output_dir, manifest, rows)
    except CoverageAuditError as exc:
        print(f"coverage audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output_dir), "direct_source_absent_from_context": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
