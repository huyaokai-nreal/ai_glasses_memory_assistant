from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_glasses_memory_assistant.evals import longmemeval_temporal_filter_audit as audit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _detail(question_id: str, *, strategy: str, temporal: dict[str, object], event_recall: dict[str, object]) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": "temporal-reasoning",
        "recall_context": "",
        "response_debug": {
            "turn_decision": {
                "final": {
                    "event_recall_strategy": strategy,
                    "coverage_requirement": "best_evidence",
                    "temporal_query": {"has_expression": True, "granularity": "day"},
                }
            },
            "temporal": {"query": temporal},
            "memory": {"event_recall": event_recall},
        },
    }


def _attribution(run_dir: Path, attribution_dir: Path, question_ids: list[str]) -> None:
    _write_json(attribution_dir / "analysis-manifest.json", {"source_run_dir": str(run_dir.resolve())})
    rows = [
        {"question_id": question_id, "primary_cause": "retrieval_empty", "source_paths": {"result": "case/result.json"}}
        for question_id in question_ids
    ]
    (attribution_dir / "failure-attribution.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (attribution_dir / "failure-attribution.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_analyze_run_records_applied_temporal_range_without_semantic_guess(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    attribution_dir = tmp_path / "attribution"
    detail = _detail(
        "one",
        strategy="temporal_range",
        temporal={"backend": "llm", "start_at": 10, "end_at": 20, "granularity": "day"},
        event_recall={"strategy": "temporal_range", "start_at": 10, "end_at": 20, "count": 0},
    )
    _write_json(run_dir / "run.details.json", [detail])
    _attribution(run_dir, attribution_dir, ["one"])

    manifest, rows = audit.analyze_run(run_dir, attribution_dir, expected_total=1, expected_retrieval_empty=1)

    assert manifest["zero_model_gate"] == {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0}
    assert rows[0]["record_time_filter"] == {
        "applied": True,
        "scope": {"source": "event_recall", "start_at": 10, "end_at": 20},
        "event_recall_count": 0,
        "complete_set_candidate_count": 0,
        "complete_set_coverage_complete": None,
    }
    assert rows[0]["semantic_role"] == "manual_review_required"


def test_analyze_run_records_complete_set_filter_and_text_search_without_filter(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    attribution_dir = tmp_path / "attribution"
    filtered = _detail(
        "filtered",
        strategy="text_search",
        temporal={"backend": "llm", "start_at": 10, "end_at": 20},
        event_recall={"strategy": "text_search", "complete_set": {"scope": {"start_at": 10, "end_at": 20}, "candidate_count": 0, "coverage_complete": True}},
    )
    unfiltered = _detail(
        "unfiltered",
        strategy="text_search",
        temporal={"backend": "llm", "start_at": None, "end_at": None},
        event_recall={"strategy": "text_search", "count": 0},
    )
    _write_json(run_dir / "run.details.json", [filtered, unfiltered])
    _attribution(run_dir, attribution_dir, ["filtered", "unfiltered"])

    _manifest, rows = audit.analyze_run(run_dir, attribution_dir, expected_total=2, expected_retrieval_empty=2)

    by_id = {row["question_id"]: row for row in rows}
    assert by_id["filtered"]["record_time_filter"]["scope"]["source"] == "complete_set.scope"
    assert by_id["unfiltered"]["record_time_filter"]["applied"] is False


def test_analyze_run_rejects_attribution_from_another_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    attribution_dir = tmp_path / "attribution"
    _write_json(run_dir / "run.details.json", [_detail("one", strategy="text_search", temporal={}, event_recall={})])
    _attribution(tmp_path / "other-run", attribution_dir, ["one"])

    with pytest.raises(audit.TemporalFilterAuditError, match="source run mismatch"):
        audit.analyze_run(run_dir, attribution_dir, expected_total=1, expected_retrieval_empty=1)


def test_write_audit_is_immutable(tmp_path: Path) -> None:
    out_dir = tmp_path / "audit"
    manifest = {"zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0}}
    row = {
        "question_id": "one",
        "decision": {"event_recall_strategy": "text_search"},
        "record_time_filter": {"applied": False},
        "runtime_temporal": {"usable_range": False},
    }

    audit.write_audit(out_dir, manifest, [row])

    assert (out_dir / "review-queue.md").is_file()
    with pytest.raises(audit.TemporalFilterAuditError, match="already exists"):
        audit.write_audit(out_dir, manifest, [row])


def test_module_has_no_product_model_or_oracle_imports() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")

    assert "from ai_glasses_memory_assistant" not in source
    assert "import openai" not in source
    assert "GlassesChatService" not in source
