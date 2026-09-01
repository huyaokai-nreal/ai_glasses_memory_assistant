from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_glasses_memory_assistant.evals import longmemeval_failure_attribution as attribution


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _detail(
    question_id: str,
    *,
    decision: dict[str, object] | None = None,
    context: str = "",
    is_abstention: bool = False,
    reader_debug: dict[str, object] | None = None,
    offline_evidence: dict[str, object] | None = None,
    recall_context_chars: int = 0,
    reader_input_chars: int = 0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "question_id": question_id,
        "question_type": "multi-session",
        "recall_context": context,
        "recall_context_chars": recall_context_chars,
        "reader_input_chars": reader_input_chars,
        "is_abstention": is_abstention,
        "response_debug": {"turn_decision": {"final": decision or {}}},
    }
    if reader_debug is not None:
        row["reader_debug"] = reader_debug
    if offline_evidence is not None:
        row["offline_evidence"] = offline_evidence
    return row


def _write_case(run_dir: Path, question_id: str) -> None:
    case_dir = run_dir / "cases" / f"case-{question_id}"
    _write_json(case_dir / "result.json", {"question_id": question_id})
    (case_dir / "audit.redacted.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(case_dir / "evidence-export.json", {"exported": []})
    (case_dir / "database").mkdir(parents=True, exist_ok=True)
    (case_dir / "database/events.db").write_bytes(b"events")
    (case_dir / "database/timeline.db").write_bytes(b"timeline")


def _make_run(tmp_path: Path, details: list[dict[str, object]], judge: dict[str, int]) -> Path:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "run-manifest.json", {"source_snapshot": {"git_head": "frozen"}})
    _write_json(run_dir / "eval-latest.json", {"summary": {"official_judge": {"per_question": judge}}})
    _write_json(run_dir / "run.details.json", details)
    for row in details:
        _write_case(run_dir, str(row["question_id"]))
    return run_dir


def test_analyze_run_covers_primary_causes_and_behavior_overlays(tmp_path: Path) -> None:
    details = [
        _detail("route", reader_debug={"refusal": True}),
        _detail("empty", decision={"needs_event_memory": True}, is_abstention=True, reader_debug={"refusal": False}),
        _detail("write", offline_evidence={"write_or_type_loss": "verified"}, reader_debug={}),
        _detail("selection", offline_evidence={"candidate_support": "verified", "selected_support": False}, reader_debug={}),
        _detail("reader", context="evidence", offline_evidence={"selected_support": "verified"}, reader_debug={}),
        _detail(
            "unknown",
            decision={"needs_event_memory": True, "temporal_query": {"has_expression": True}},
            context="some evidence",
            reader_debug={"relevant_evidence": ["some evidence"]},
            recall_context_chars=20,
            reader_input_chars=10,
        ),
    ]
    judge = {str(row["question_id"]): 0 for row in details}
    run_dir = _make_run(tmp_path, details, judge)

    manifest, rows = attribution.analyze_run(run_dir, expected_total=6, expected_failures=6)

    assert [row["primary_cause"] for row in rows] == [
        "retrieval_empty",
        "reader_evidence_use_verified",
        "route_not_requested",
        "selection_or_truncation_verified",
        "insufficient_artifact_evidence",
        "write_or_type_verified",
    ]
    by_id = {row["question_id"]: row for row in rows}
    assert by_id["route"]["behavior_overlay"] == "should_answer_but_refused"
    assert by_id["empty"]["behavior_overlay"] == "should_abstain_but_answered"
    assert by_id["unknown"]["behavior_overlay"] == "other_judge_mismatch"
    assert by_id["unknown"]["review_required"] is True
    assert "reader_input_truncated" in by_id["unknown"]["observations"]
    assert manifest["zero_model_gate"] == {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0}


def test_analyze_run_rejects_duplicate_detail_id(tmp_path: Path) -> None:
    details = [_detail("same"), _detail("same")]
    run_dir = _make_run(tmp_path, details, {"same": 0})

    with pytest.raises(attribution.AttributionError, match="duplicate details question_id"):
        attribution.analyze_run(run_dir, expected_total=2, expected_failures=1)


def test_analyze_run_rejects_missing_judge_mapping(tmp_path: Path) -> None:
    details = [_detail("one"), _detail("two")]
    run_dir = _make_run(tmp_path, details, {"one": 0})

    with pytest.raises(attribution.AttributionError, match="judge/detail ID mismatch"):
        attribution.analyze_run(run_dir, expected_total=2, expected_failures=1)


def test_analyze_run_rejects_missing_case_artifact(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, [_detail("one")], {"one": 0})
    (run_dir / "cases" / "case-one" / "database/timeline.db").unlink()

    with pytest.raises(attribution.AttributionError, match="missing case artifact"):
        attribution.analyze_run(run_dir, expected_total=1, expected_failures=1)


def test_write_analysis_is_immutable_and_keeps_review_queue(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, [_detail("one", context="unknown")], {"one": 0})
    manifest, rows = attribution.analyze_run(run_dir, expected_total=1, expected_failures=1)
    out_dir = tmp_path / "analysis"

    attribution.write_analysis(out_dir, manifest, rows)

    assert (out_dir / "analysis-manifest.json").is_file()
    assert (out_dir / "failure-attribution.jsonl").is_file()
    assert "one" in (out_dir / "review-queue.md").read_text(encoding="utf-8")
    written_manifest = json.loads((out_dir / "analysis-manifest.json").read_text(encoding="utf-8"))
    assert written_manifest["zero_model_gate"]["qwen_calls"] == 0
    with pytest.raises(attribution.AttributionError, match="already exists"):
        attribution.write_analysis(out_dir, manifest, rows)


def test_module_has_no_product_or_model_imports() -> None:
    source = Path(attribution.__file__).read_text(encoding="utf-8")

    assert "from ai_glasses_memory_assistant" not in source
    assert "import openai" not in source
    assert "GlassesChatService" not in source
