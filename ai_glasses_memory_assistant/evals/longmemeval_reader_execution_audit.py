"""Read-only, zero-model audit of LongMemEval Reader execution failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ANALYSIS_SCHEMA = "longmemeval.reader-execution-audit.v1"


class ReaderExecutionAuditError(ValueError):
    """Raised when a frozen report cannot support a trustworthy audit."""


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ReaderExecutionAuditError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReaderExecutionAuditError(f"invalid JSON artifact: {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_single_detail_path(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("*.details.json"))
    if len(paths) != 1:
        raise ReaderExecutionAuditError(
            f"expected exactly one root-level *.details.json in {run_dir}, found {len(paths)}"
        )
    return paths[0]


def _index_detail_types(details: Any) -> dict[str, str]:
    if not isinstance(details, list):
        raise ReaderExecutionAuditError("details artifact must be a JSON array")
    indexed: dict[str, str] = {}
    for position, row in enumerate(details):
        if not isinstance(row, dict):
            raise ReaderExecutionAuditError(f"details row {position} is not an object")
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise ReaderExecutionAuditError(f"details row {position} has no question_id")
        if question_id in indexed:
            raise ReaderExecutionAuditError(f"duplicate details question_id: {question_id}")
        indexed[question_id] = str(row.get("question_type") or "unknown")
    return indexed


def _normalized_attempts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    attempts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        errors = item.get("validation_errors")
        attempts.append(
            {
                "stage": str(item.get("stage") or ""),
                "attempt": int(item.get("attempt") or 0),
                "outcome": str(item.get("outcome") or ""),
                "validation_errors": [str(error) for error in errors] if isinstance(errors, list) else [],
            }
        )
    return attempts


def analyze_run(
    run_dir: Path,
    *,
    expected_total: int = 500,
    expected_failures: int = 121,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return only current official-judge failures that hit Reader execution failure."""

    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run-manifest.json"
    latest_path = run_dir / "eval-latest.json"
    detail_path = _find_single_detail_path(run_dir)
    run_manifest = _load_json(manifest_path)
    latest = _load_json(latest_path)
    detail_types = _index_detail_types(_load_json(detail_path))
    judge_map = latest.get("summary", {}).get("official_judge", {}).get("per_question", {}) if isinstance(latest, dict) else {}
    if not isinstance(judge_map, dict):
        raise ReaderExecutionAuditError("official_judge.per_question must be an object")
    if set(judge_map) != set(detail_types):
        raise ReaderExecutionAuditError("judge/detail ID mismatch")
    if len(detail_types) != expected_total:
        raise ReaderExecutionAuditError(f"expected {expected_total} detail IDs, found {len(detail_types)}")
    failures = sorted(question_id for question_id, score in judge_map.items() if score == 0)
    if len(failures) != expected_failures:
        raise ReaderExecutionAuditError(
            f"expected {expected_failures} official judge failures, found {len(failures)}"
        )

    case_results: dict[str, tuple[Path, dict[str, Any]]] = {}
    for result_path in sorted((run_dir / "cases").glob("*/result.json")):
        result = _load_json(result_path)
        question_id = str(result.get("question_id") or "").strip() if isinstance(result, dict) else ""
        if not question_id:
            raise ReaderExecutionAuditError(f"case result has no question_id: {result_path}")
        if question_id in case_results:
            raise ReaderExecutionAuditError(f"duplicate case artifact question_id: {question_id}")
        case_results[question_id] = (result_path, result)
    missing = sorted(set(failures) - set(case_results))
    if missing:
        raise ReaderExecutionAuditError(f"missing case result for official failures: {missing[:3]}")

    rows: list[dict[str, Any]] = []
    for question_id in failures:
        result_path, result = case_results[question_id]
        debug = result.get("reader_debug") if isinstance(result.get("reader_debug"), dict) else {}
        if debug.get("reader_status") != "execution_failed":
            continue
        validation_errors = debug.get("ledger_validation_errors")
        errors = [str(error) for error in validation_errors] if isinstance(validation_errors, list) else []
        rows.append(
            {
                "question_id": question_id,
                "question_type": detail_types[question_id],
                "official_judge_score": 0,
                "reader_status": "execution_failed",
                "failure_stage": str(debug.get("failure_stage") or ""),
                "ledger_error": str(debug.get("ledger_error") or ""),
                "validation_errors": errors,
                "execution_attempts": _normalized_attempts(debug.get("execution_attempts")),
                "source_path": result_path.relative_to(run_dir).as_posix(),
            }
        )

    analysis_manifest = {
        "schema": ANALYSIS_SCHEMA,
        "source_run_dir": str(run_dir),
        "source_snapshot": run_manifest.get("source_snapshot") if isinstance(run_manifest, dict) else None,
        "input_files": [
            {"path": path.relative_to(run_dir).as_posix(), "sha256": _sha256_file(path)}
            for path in (manifest_path, latest_path, detail_path)
        ],
        "input_counts": {"detail_ids": len(detail_types), "official_judge_failures": len(failures)},
        "expected_counts": {"total": expected_total, "official_judge_failures": expected_failures},
        "zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0},
    }
    return analysis_manifest, rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        (str(row["failure_stage"]), str(row["ledger_error"]), tuple(row["validation_errors"]))
        for row in rows
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "execution_failed_official_judge_failures": len(rows),
        "exact_failure_signatures": [
            {
                "failure_stage": signature[0],
                "ledger_error": signature[1],
                "validation_errors": list(signature[2]),
                "count": count,
            }
            for signature, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0},
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_analysis(out_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    """Write a new immutable report directory after analysis completes."""

    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise ReaderExecutionAuditError(f"output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        summary = summarize_rows(rows)
        _write_json(temporary_dir / "analysis-manifest.json", manifest)
        _write_json(temporary_dir / "summary.json", summary)
        with (temporary_dir / "reader-execution-failures.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary_dir, out_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a zero-Qwen Reader execution-failure audit.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=500)
    parser.add_argument("--expected-failures", type=int, default=121)
    args = parser.parse_args(argv)
    try:
        manifest, rows = analyze_run(
            args.run_dir, expected_total=args.expected_total, expected_failures=args.expected_failures
        )
        out_dir = write_analysis(args.out_dir, manifest, rows)
    except ReaderExecutionAuditError as exc:
        print(f"reader execution audit blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), **summarize_rows(rows)["zero_model_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
