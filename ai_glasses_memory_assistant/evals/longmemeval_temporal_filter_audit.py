"""Zero-model audit of temporal filters applied during failed LongMemEval recall.

This evaluation-only module reads frozen report artifacts.  It never imports
product code, opens an LLM client, calls a network service, or reads Oracle
answers.  It records observable filter behavior and leaves the semantic role
of a date phrase for explicit human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "longmemeval.temporal-filter-audit.v1"


class TemporalFilterAuditError(ValueError):
    """Raised when frozen artifacts cannot support a trustworthy audit."""


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise TemporalFilterAuditError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TemporalFilterAuditError(f"invalid JSON artifact: {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_details_path(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("*.details.json"))
    if len(paths) != 1:
        raise TemporalFilterAuditError(
            f"expected exactly one root-level *.details.json in {run_dir}, found {len(paths)}"
        )
    return paths[0]


def _index_details(details: Any, *, expected_total: int) -> dict[str, dict[str, Any]]:
    if not isinstance(details, list):
        raise TemporalFilterAuditError("details artifact must be a JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    for position, item in enumerate(details):
        if not isinstance(item, dict):
            raise TemporalFilterAuditError(f"details row {position} is not an object")
        question_id = str(item.get("question_id") or "").strip()
        if not question_id:
            raise TemporalFilterAuditError(f"details row {position} has no question_id")
        if question_id in indexed:
            raise TemporalFilterAuditError(f"duplicate details question_id: {question_id}")
        indexed[question_id] = item
    if len(indexed) != expected_total:
        raise TemporalFilterAuditError(f"expected {expected_total} detail IDs, found {len(indexed)}")
    return indexed


def _load_attribution_rows(attribution_dir: Path) -> list[dict[str, Any]]:
    path = attribution_dir / "failure-attribution.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise TemporalFilterAuditError(f"missing attribution rows: {path}") from exc
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TemporalFilterAuditError(f"invalid attribution row {position}: {exc}") from exc
        if not isinstance(row, dict):
            raise TemporalFilterAuditError(f"attribution row {position} is not an object")
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise TemporalFilterAuditError(f"attribution row {position} has no question_id")
        if question_id in seen:
            raise TemporalFilterAuditError(f"duplicate attribution question_id: {question_id}")
        seen.add(question_id)
        rows.append(row)
    return rows


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _final_decision(detail: dict[str, Any]) -> dict[str, Any]:
    debug = _dict(detail.get("response_debug"))
    turn_decision = _dict(debug.get("turn_decision"))
    final = _dict(turn_decision.get("final"))
    return final or _dict(debug.get("pre_reply_decision"))


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _usable_range(value: dict[str, Any]) -> bool:
    start_at = _number(value.get("start_at"))
    end_at = _number(value.get("end_at"))
    return start_at is not None and end_at is not None and end_at > start_at


def _event_recall(detail: dict[str, Any]) -> dict[str, Any]:
    memory = _dict(_dict(detail.get("response_debug")).get("memory"))
    return _dict(memory.get("event_recall"))


def _filter_scope(event_recall: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    complete_set = _dict(event_recall.get("complete_set"))
    complete_scope = _dict(complete_set.get("scope"))
    direct_scope = {
        "start_at": event_recall.get("start_at"),
        "end_at": event_recall.get("end_at"),
    }
    if _usable_range(direct_scope):
        return True, {"source": "event_recall", **direct_scope}
    if _usable_range(complete_scope):
        return True, {
            "source": "complete_set.scope",
            "start_at": complete_scope.get("start_at"),
            "end_at": complete_scope.get("end_at"),
        }
    return False, {"source": "none", "start_at": None, "end_at": None}


def analyze_run(
    run_dir: Path,
    attribution_dir: Path,
    *,
    expected_total: int = 500,
    expected_retrieval_empty: int = 11,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return observable temporal-filter facts for current retrieval-empty rows."""
    run_dir = run_dir.resolve()
    attribution_dir = attribution_dir.resolve()
    details_path = _find_details_path(run_dir)
    details = _index_details(_load_json(details_path), expected_total=expected_total)
    attribution_manifest = _load_json(attribution_dir / "analysis-manifest.json")
    source_run = Path(str(_dict(attribution_manifest).get("source_run_dir") or "")).resolve()
    if source_run != run_dir:
        raise TemporalFilterAuditError(
            f"attribution source run mismatch: {source_run} != {run_dir}"
        )
    attribution_rows = _load_attribution_rows(attribution_dir)
    targets = [row for row in attribution_rows if row.get("primary_cause") == "retrieval_empty"]
    if len(targets) != expected_retrieval_empty:
        raise TemporalFilterAuditError(
            f"expected {expected_retrieval_empty} retrieval_empty rows, found {len(targets)}"
        )

    rows: list[dict[str, Any]] = []
    for target in sorted(targets, key=lambda item: str(item["question_id"])):
        question_id = str(target["question_id"])
        detail = details.get(question_id)
        if detail is None:
            raise TemporalFilterAuditError(f"retrieval-empty row missing detail: {question_id}")
        response_debug = _dict(detail.get("response_debug"))
        runtime_temporal = _dict(_dict(response_debug.get("temporal")).get("query"))
        decision = _final_decision(detail)
        decision_temporal = _dict(decision.get("temporal_query"))
        event_recall = _event_recall(detail)
        filter_applied, filter_scope = _filter_scope(event_recall)
        complete_set = _dict(event_recall.get("complete_set"))
        rows.append(
            {
                "question_id": question_id,
                "question_type": str(detail.get("question_type") or "unknown"),
                "reader_context_empty": not bool(str(detail.get("recall_context") or "").strip()),
                "decision": {
                    "event_recall_strategy": str(decision.get("event_recall_strategy") or ""),
                    "coverage_requirement": str(decision.get("coverage_requirement") or ""),
                    "temporal_query_has_expression": bool(decision_temporal.get("has_expression")),
                    "temporal_query_granularity": str(decision_temporal.get("granularity") or "unknown"),
                },
                "runtime_temporal": {
                    "backend": str(runtime_temporal.get("backend") or ""),
                    "has_temporal_expression": bool(runtime_temporal.get("has_temporal_expression")),
                    "usable_range": _usable_range(runtime_temporal),
                    "granularity": str(runtime_temporal.get("granularity") or "unknown"),
                    "reason": str(runtime_temporal.get("reason") or ""),
                },
                "record_time_filter": {
                    "applied": filter_applied,
                    "scope": filter_scope,
                    "event_recall_count": int(event_recall.get("count") or 0),
                    "complete_set_candidate_count": int(complete_set.get("candidate_count") or 0),
                    "complete_set_coverage_complete": complete_set.get("coverage_complete"),
                },
                "semantic_role": "manual_review_required",
                "semantic_role_reason": (
                    "The audit intentionally does not infer whether a temporal phrase qualifies "
                    "the answer target or the occurrence time of supporting evidence."
                ),
                "proof_paths": [
                    "response_debug.turn_decision.final",
                    "response_debug.temporal.query",
                    "response_debug.memory.event_recall",
                    "recall_context",
                ],
                "source_paths": dict(target.get("source_paths") or {}),
            }
        )

    manifest = {
        "schema": AUDIT_SCHEMA,
        "source_run_dir": str(run_dir),
        "attribution_dir": str(attribution_dir),
        "input_sha256": {
            details_path.name: _sha256_file(details_path),
            "failure-attribution.jsonl": _sha256_file(attribution_dir / "failure-attribution.jsonl"),
        },
        "input_counts": {
            "detail_ids": len(details),
            "retrieval_empty_rows": len(rows),
        },
        "zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0},
    }
    return manifest, rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = Counter(str(row["decision"]["event_recall_strategy"]) for row in rows)
    return {
        "schema": AUDIT_SCHEMA,
        "retrieval_empty_rows": len(rows),
        "record_time_filter_applied": sum(bool(row["record_time_filter"]["applied"]) for row in rows),
        "runtime_usable_ranges": sum(bool(row["runtime_temporal"]["usable_range"]) for row in rows),
        "by_event_recall_strategy": dict(sorted(strategies.items())),
        "semantic_role_review_required": len(rows),
        "zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0},
    }


def write_audit(out_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    """Write a new immutable audit directory only after full validation."""
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise TemporalFilterAuditError(f"output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        summary = _summary(rows)
        (temporary_dir / "analysis-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (temporary_dir / "temporal-filter-audit.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (temporary_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        review = [
            "# Temporal filter review queue",
            "",
            "All rows require human semantic review; this audit does not read Oracle answers or infer phrase meaning.",
            "",
            "| Question ID | Strategy | Record-time filter applied | Runtime usable range |",
            "| --- | --- | --- | --- |",
        ]
        for row in rows:
            review.append(
                "| {question_id} | {strategy} | {filter} | {range} |".format(
                    question_id=row["question_id"],
                    strategy=row["decision"]["event_recall_strategy"] or "none",
                    filter="yes" if row["record_time_filter"]["applied"] else "no",
                    range="yes" if row["runtime_temporal"]["usable_range"] else "no",
                )
            )
        (temporary_dir / "review-queue.md").write_text("\n".join(review) + "\n", encoding="utf-8")
        temporary_dir.replace(out_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=500)
    parser.add_argument("--expected-retrieval-empty", type=int, default=11)
    args = parser.parse_args(argv)
    try:
        manifest, rows = analyze_run(
            args.run_dir,
            args.attribution_dir,
            expected_total=args.expected_total,
            expected_retrieval_empty=args.expected_retrieval_empty,
        )
        out_dir = write_audit(args.out_dir, manifest, rows)
    except TemporalFilterAuditError as exc:
        print(f"temporal filter audit blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), **_summary(rows)["zero_model_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
