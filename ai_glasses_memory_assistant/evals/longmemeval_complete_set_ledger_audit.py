"""Zero-model provenance gate for valid complete-set Reader ledgers.

The audit does not decide which facts answer a benchmark question.  It proves a
narrower prerequisite for any Reader behavior change: every source declared to
the complete-set Reader can be reconstructed from the frozen detail artifact,
including its full text and a stable hash, and every ledger item points only to
one of those sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "longmemeval.complete-set-ledger-audit.v1"


class CompleteSetLedgerAuditError(ValueError):
    """Raised when a frozen run cannot support a trustworthy audit."""


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise CompleteSetLedgerAuditError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CompleteSetLedgerAuditError(f"invalid JSON artifact: {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _find_details_path(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("*.details.json"))
    if len(paths) != 1:
        raise CompleteSetLedgerAuditError(
            f"expected exactly one root-level *.details.json in {run_dir}, found {len(paths)}"
        )
    return paths[0]


def _index_details(details: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(details, list):
        raise CompleteSetLedgerAuditError("details artifact must be a JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    for position, detail in enumerate(details):
        if not isinstance(detail, dict):
            raise CompleteSetLedgerAuditError(f"details row {position} is not an object")
        question_id = str(detail.get("question_id") or "").strip()
        if not question_id:
            raise CompleteSetLedgerAuditError(f"details row {position} has no question_id")
        if question_id in indexed:
            raise CompleteSetLedgerAuditError(f"duplicate details question_id: {question_id}")
        indexed[question_id] = detail
    return indexed


def _reader_debug(detail: dict[str, Any]) -> dict[str, Any]:
    value = detail.get("reader_debug")
    return value if isinstance(value, dict) else {}


def _source_inventory(detail: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sources: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for prefix, records, text_key in (
        ("memory", detail.get("recalled_memories"), "content"),
        ("timeline", detail.get("recalled_timeline_chunks"), "text"),
    ):
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("id") or "").strip()
            if not record_id:
                continue
            source_id = f"{prefix}:{record_id}"
            if source_id in sources:
                duplicate_ids.append(source_id)
                continue
            text = str(record.get(text_key) or "").strip()
            sources[source_id] = {
                "source_id": source_id,
                "text_sha256": _sha256_text(text),
                "text_chars": len(text),
                "multiline": "\n" in text,
                "nonempty": bool(text),
            }
    return sources, sorted(set(duplicate_ids))


def audit_detail(question_id: str, detail: dict[str, Any]) -> dict[str, Any] | None:
    """Audit one valid-ledger failure, or return ``None`` when not selected."""
    debug = _reader_debug(detail)
    task = debug.get("answer_task")
    if not isinstance(task, dict) or task.get("coverage_requirement") != "complete_set":
        return None
    if debug.get("reader_status") != "answered" or debug.get("ledger_valid") is not True:
        return None
    expected_ids = [str(value) for value in task.get("source_ids") or [] if str(value)]
    if len(expected_ids) != len(set(expected_ids)):
        raise CompleteSetLedgerAuditError(f"duplicate declared source IDs for {question_id}")
    sources, duplicate_inventory_ids = _source_inventory(detail)
    ledger_items = debug.get("ledger_items")
    if not isinstance(ledger_items, list):
        raise CompleteSetLedgerAuditError(f"valid ledger has no ledger_items list for {question_id}")
    ledger_item_source_ids = sorted({
        str(source_id)
        for item in ledger_items
        if isinstance(item, dict)
        for source_id in (item.get("source_ids") or [])
        if str(source_id)
    })
    expected = set(expected_ids)
    actual = set(sources)
    missing_declared_sources = sorted(expected - actual)
    unexpected_inventory_sources = sorted(actual - expected)
    unknown_ledger_item_sources = sorted(set(ledger_item_source_ids) - actual)
    empty_declared_sources = sorted(source_id for source_id in expected if not sources.get(source_id, {}).get("nonempty"))
    accounted_count = debug.get("ledger_accounted_source_count")
    accounted_matches_declared = isinstance(accounted_count, int) and accounted_count == len(expected)
    return {
        "question_id": question_id,
        "question_type": str(detail.get("question_type") or "unknown"),
        "reader_status": "answered",
        "ledger_valid": True,
        "declared_source_count": len(expected),
        "inventory_source_count": len(actual),
        "ledger_accounted_source_count": accounted_count,
        "ledger_item_count": len(ledger_items),
        "ledger_item_source_count": len(ledger_item_source_ids),
        "ledger_value": str(debug.get("ledger_value") or ""),
        "ledger_operation": str(debug.get("ledger_operation") or ""),
        "source_inventory": [sources[source_id] for source_id in sorted(expected) if source_id in sources],
        "missing_declared_sources": missing_declared_sources,
        "unexpected_inventory_sources": unexpected_inventory_sources,
        "duplicate_inventory_source_ids": duplicate_inventory_ids,
        "empty_declared_sources": empty_declared_sources,
        "unknown_ledger_item_sources": unknown_ledger_item_sources,
        "accounted_matches_declared": accounted_matches_declared,
        "reader_input_reversible": (
            not missing_declared_sources
            and not duplicate_inventory_ids
            and not empty_declared_sources
            and not unknown_ledger_item_sources
            and accounted_matches_declared
        ),
        "proof_paths": [
            "details.reader_debug.answer_task.source_ids",
            "details.recalled_memories",
            "details.recalled_timeline_chunks",
            "details.reader_debug.ledger_items",
            "details.reader_debug.ledger_accounted_source_count",
        ],
    }


def analyze_run(run_dir: Path, *, expected_total: int = 500) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one frozen run without importing product or model code."""
    run_dir = run_dir.resolve()
    latest_path = run_dir / "eval-latest.json"
    details_path = _find_details_path(run_dir)
    latest = _load_json(latest_path)
    details = _index_details(_load_json(details_path))
    if len(details) != expected_total:
        raise CompleteSetLedgerAuditError(f"expected {expected_total} unique detail IDs, found {len(details)}")
    judge_map = latest.get("summary", {}).get("official_judge", {}).get("per_question", {})
    if not isinstance(judge_map, dict) or set(judge_map) != set(details):
        raise CompleteSetLedgerAuditError("judge/detail ID mismatch")
    if any(score not in {0, 1} for score in judge_map.values()):
        raise CompleteSetLedgerAuditError("judge scores must be 0 or 1")
    rows = [
        row
        for question_id in sorted(question_id for question_id, score in judge_map.items() if score == 0)
        if (row := audit_detail(question_id, details[question_id])) is not None
    ]
    manifest = {
        "schema": AUDIT_SCHEMA,
        "source_run": str(run_dir),
        "expected_total": expected_total,
        "official_judge_failures": sum(score == 0 for score in judge_map.values()),
        "selected_valid_ledger_failures": len(rows),
        "input_sha256": {
            "eval-latest.json": _sha256_file(latest_path),
            details_path.name: _sha256_file(details_path),
        },
        "zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0},
    }
    return manifest, rows


def write_audit(output_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write immutable JSONL, JSON, and Markdown summaries."""
    if output_dir.exists():
        raise CompleteSetLedgerAuditError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "analysis-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ledger-provenance.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    reversible = sum(row["reader_input_reversible"] for row in rows)
    summary = {
        "schema": AUDIT_SCHEMA,
        "selected_valid_ledger_failures": len(rows),
        "reader_input_reversible": reversible,
        "nonreversible_rows": len(rows) - reversible,
        "by_question_type": dict(sorted(Counter(row["question_type"] for row in rows).items())),
        "by_operation": dict(sorted(Counter(row["ledger_operation"] for row in rows).items())),
        "zero_model_gate": manifest["zero_model_gate"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        "# Complete-set ledger provenance audit\n\n"
        f"- Selected valid-ledger judge failures: {len(rows)}\n"
        f"- Reversible Reader inputs: {reversible}/{len(rows)}\n"
        "- Model calls: Qwen 0, judge 0, network 0\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-total", type=int, default=500)
    args = parser.parse_args(argv)
    try:
        manifest, rows = analyze_run(args.run_dir, expected_total=args.expected_total)
        write_audit(args.out_dir, manifest, rows)
    except CompleteSetLedgerAuditError as exc:
        print(f"complete-set ledger audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.out_dir), **manifest}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
