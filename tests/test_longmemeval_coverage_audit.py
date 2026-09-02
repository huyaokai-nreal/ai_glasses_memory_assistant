import json
import sqlite3
from pathlib import Path

import pytest

from ai_glasses_memory_assistant.evals import longmemeval_coverage_audit as audit


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _case(run_dir: Path, question_id: str, *, chunks: list[tuple[str, str, str]]) -> None:
    case_dir = run_dir / "cases" / f"case-{question_id}"
    _json(case_dir / "result.json", {"question_id": question_id})
    database_dir = case_dir / "database"
    database_dir.mkdir(parents=True)
    with sqlite3.connect(database_dir / "timeline.db") as connection:
        connection.execute("CREATE TABLE chunks (id TEXT, parent_id TEXT, text TEXT)")
        connection.executemany("INSERT INTO chunks VALUES (?, ?, ?)", chunks)
    with sqlite3.connect(database_dir / "events.db") as connection:
        connection.execute("CREATE TABLE memories (id TEXT, content TEXT)")


def _detail(question_id: str, *, answer: str, context: str, recall: dict[str, object]) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": "multi-session",
        "answer": answer,
        "recall_context": context,
        "response_debug": {"timeline": {"recall": recall}},
    }


def _run(tmp_path: Path, details: list[dict[str, object]], judge: dict[str, int]) -> Path:
    run_dir = tmp_path / "run"
    _json(run_dir / "eval-latest.json", {"summary": {"official_judge": {"per_question": judge}}})
    _json(run_dir / "run.details.json", details)
    return run_dir


def test_audit_marks_direct_timeline_candidate_dropped_before_context(tmp_path: Path) -> None:
    detail = _detail(
        "one",
        answer="Blue Harbor",
        context="other recalled evidence",
        recall={
            "excluded_parent_id": "question-turn",
            "candidate_trace": {"query": "where did I go", "selected_ids": ["other"], "dropped_candidate_ids": ["answer"]},
        },
    )
    run_dir = _run(tmp_path, [detail], {"one": 0})
    _case(run_dir, "one", chunks=[("answer", "history-turn", "I visited Blue Harbor."), ("query", "question-turn", "Blue Harbor")])

    manifest, rows = audit.analyze_run(run_dir, expected_total=1)

    assert manifest["zero_model_gate"]["qwen_calls"] == 0
    assert rows[0]["coverage_state"] == "candidate_dropped_before_context"
    assert rows[0]["timeline_support"] == [{"source_type": "timeline", "source_id": "answer"}]


def test_audit_excludes_answer_echoed_only_by_final_question(tmp_path: Path) -> None:
    detail = _detail(
        "one",
        answer="Blue Harbor",
        context="other recalled evidence",
        recall={"excluded_parent_id": "question-turn", "candidate_trace": {"query": "where", "selected_ids": [], "dropped_candidate_ids": []}},
    )
    run_dir = _run(tmp_path, [detail], {"one": 0})
    _case(run_dir, "one", chunks=[("query", "question-turn", "Was it Blue Harbor?")])

    _manifest, rows = audit.analyze_run(run_dir, expected_total=1)

    assert rows == []


def test_audit_accepts_reordered_direct_reference_terms(tmp_path: Path) -> None:
    detail = _detail(
        "one",
        answer="Data Analysis webinar",
        context="other recalled evidence",
        recall={
            "excluded_parent_id": "question-turn",
            "candidate_trace": {"query": "which webinar", "selected_ids": [], "dropped_candidate_ids": ["answer"]},
        },
    )
    run_dir = _run(tmp_path, [detail], {"one": 0})
    _case(run_dir, "one", chunks=[("answer", "history-turn", "I attended a webinar on data analysis."), ("query", "question-turn", "Data Analysis webinar?")])

    _manifest, rows = audit.analyze_run(run_dir, expected_total=1)

    assert rows[0]["coverage_state"] == "candidate_dropped_before_context"


def test_audit_rejects_existing_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(audit.CoverageAuditError, match="already exists"):
        audit.write_audit(output_dir, {"zero_model_gate": {}}, [])


def test_module_has_no_product_or_model_imports() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "from ai_glasses_memory_assistant" not in source
    assert "import openai" not in source
    assert "GlassesChatService" not in source
