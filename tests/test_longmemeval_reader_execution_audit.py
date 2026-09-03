from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_glasses_memory_assistant.evals import longmemeval_reader_execution_audit as audit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(tmp_path: Path, *, duplicate_detail: bool = False) -> Path:
    run_dir = tmp_path / "run"
    details = [
        {"question_id": "failed", "question_type": "multi-session"},
        {"question_id": "healthy", "question_type": "temporal-reasoning"},
    ]
    if duplicate_detail:
        details.append({"question_id": "failed", "question_type": "multi-session"})
    _write_json(run_dir / "run-manifest.json", {"source_snapshot": {"git_head": "frozen"}})
    _write_json(
        run_dir / "eval-latest.json",
        {"summary": {"official_judge": {"per_question": {"failed": 0, "healthy": 1}}}},
    )
    _write_json(run_dir / "sample.details.json", details)
    _write_json(
        run_dir / "cases" / "case-failed" / "result.json",
        {
            "question_id": "failed",
            "reader_debug": {
                "reader_status": "execution_failed",
                "failure_stage": "batch_ledger",
                "ledger_error": "batch_validation_failed",
                "ledger_validation_errors": ["included_source_without_item"],
                "execution_attempts": [{
                    "stage": "batch_ledger",
                    "attempt": 2,
                    "outcome": "invalid",
                    "validation_errors": ["included_source_without_item"],
                }],
            },
        },
    )
    _write_json(run_dir / "cases" / "case-healthy" / "result.json", {"question_id": "healthy"})
    return run_dir


def test_audit_reads_raw_case_diagnostics_without_model_calls(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)

    manifest, rows = audit.analyze_run(run_dir, expected_total=2, expected_failures=1)

    assert manifest["zero_model_gate"] == {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0}
    assert rows[0]["validation_errors"] == ["included_source_without_item"]
    assert rows[0]["execution_attempts"][0]["attempt"] == 2
    assert audit.summarize_rows(rows)["exact_failure_signatures"][0]["count"] == 1


def test_audit_rejects_duplicate_detail_ids(tmp_path: Path) -> None:
    with pytest.raises(audit.ReaderExecutionAuditError, match="duplicate details question_id"):
        audit.analyze_run(_make_run(tmp_path, duplicate_detail=True), expected_total=3, expected_failures=1)


def test_audit_writes_immutable_output(tmp_path: Path) -> None:
    manifest, rows = audit.analyze_run(_make_run(tmp_path), expected_total=2, expected_failures=1)
    out_dir = tmp_path / "analysis"

    audit.write_analysis(out_dir, manifest, rows)

    assert (out_dir / "reader-execution-failures.jsonl").is_file()
    with pytest.raises(audit.ReaderExecutionAuditError, match="already exists"):
        audit.write_analysis(out_dir, manifest, rows)


def test_module_has_no_product_or_model_imports() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")

    assert "GlassesChatService" not in source
    assert "import openai" not in source
