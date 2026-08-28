from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _tool_module():
    path = Path(__file__).resolve().parents[1] / "android" / "tools" / "replay_usage_snapshot.py"
    spec = importlib.util.spec_from_file_location("replay_usage_snapshot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_replay_selects_only_needs_improvement_and_keeps_source_unchanged(tmp_path, capsys) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    feedback = [
        {"feedback_id": "bad", "rating": "needs_improvement", "user_message": "环境里发生了什么"},
        {"feedback_id": "good", "rating": "satisfied", "user_message": "谢谢"},
    ]
    path = snapshot / "feedback_index.json"
    original = json.dumps(feedback, ensure_ascii=False)
    path.write_text(original, encoding="utf-8")

    assert _tool_module().main([str(snapshot), "--select-only"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["selected_count"] == 1
    assert result["feedback"][0]["feedback_id"] == "bad"
    assert path.read_text(encoding="utf-8") == original


def test_snapshot_replay_exports_only_complete_set_operational_diagnostics() -> None:
    payload = _tool_module().replay_complete_set_debug({
        "debug": {
            "complete_set_answer": {
                "valid": False,
                "reader_status": "execution_failed",
                "failure_stage": "json",
                "validation_errors": ["reader_invalid_json"],
                "raw": "must not be exported",
            },
        },
        "discussion_recall": {
            "complete_set_answer": {
                "candidate_count": 6,
                "reader_status": "execution_failed",
                "failure_stage": "json",
                "answer_contract": {"must_not": "be exported"},
            },
        },
    })

    assert payload == {
        "valid": False,
        "reader_status": "execution_failed",
        "failure_stage": "json",
        "validation_errors": ["reader_invalid_json"],
        "discussion_candidate_count": 6,
        "discussion_reader_status": "execution_failed",
        "discussion_failure_stage": "json",
    }
