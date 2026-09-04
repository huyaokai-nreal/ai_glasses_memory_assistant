from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_glasses_memory_assistant.evals import longmemeval_reader_input_manifest as manifest_tool


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_source_run(tmp_path: Path, *, completed: bool = True) -> Path:
    run_dir = tmp_path / "source-run"
    _write_json(run_dir / "run-manifest.json", {"source_snapshot": {"head": "frozen"}})
    case_dir = run_dir / "cases" / "001-target"
    _write_json(
        case_dir / "result.json",
        {
            "question_id": "target",
            "question": "What happened?",
            "recall_context": "[memory id=m1] verified fact",
            "recalled_memories": [{"id": "m1", "content": "verified fact"}],
            "recalled_timeline_chunks": [],
            "recalled_documents": [],
            "response_debug": {
                "pre_reply_decision": {"answer_intent": "direct_answer", "answer_obligations": ["count"]},
                "planner": {"coverage_requirement": "complete_set"},
                "memory": {"complete_set": {"coverage_complete": True, "source_ids": ["m1"]}},
            },
        },
    )
    if completed:
        _write_json(case_dir / "completed.json", {"question_id": "target"})
    return run_dir


def _write_cohort(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "cohort.json"
    _write_json(path, payload)
    return path


def test_builds_reversible_manifest_without_model_or_product_calls(tmp_path: Path) -> None:
    run_dir = _make_source_run(tmp_path)
    cohort = _write_cohort(tmp_path, {"groups": {"target": ["target"]}})
    source_result = run_dir / "cases" / "001-target" / "result.json"
    before = source_result.read_bytes()

    manifest = manifest_tool.build_input_manifest(run_dir, cohort)

    assert manifest["zero_model_gate"] == {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0}
    assert manifest["case_count"] == 1
    assert manifest["cases"][0]["recall_context_chars"] == len("[memory id=m1] verified fact")
    assert manifest["cases"][0]["source_envelope_counts"] == {"memories": 1, "timeline_chunks": 0, "documents": 0}
    assert source_result.read_bytes() == before


def test_rejects_duplicate_or_missing_cohort_ids(tmp_path: Path) -> None:
    run_dir = _make_source_run(tmp_path)
    duplicate = _write_cohort(tmp_path, {"groups": {"target": ["target"], "health": ["target"]}})
    with pytest.raises(manifest_tool.ReaderInputManifestError, match="duplicate cohort question_id"):
        manifest_tool.build_input_manifest(run_dir, duplicate)

    missing = _write_cohort(tmp_path, {"groups": {"target": ["missing"]}})
    with pytest.raises(manifest_tool.ReaderInputManifestError, match="missing from source run"):
        manifest_tool.build_input_manifest(run_dir, missing)


def test_rejects_missing_completed_marker_and_source_hash_mismatch(tmp_path: Path) -> None:
    incomplete_run = _make_source_run(tmp_path / "incomplete", completed=False)
    cohort = _write_cohort(tmp_path / "incomplete", {"groups": {"target": ["target"]}})
    with pytest.raises(manifest_tool.ReaderInputManifestError, match="no completed marker"):
        manifest_tool.build_input_manifest(incomplete_run, cohort)

    run_dir = _make_source_run(tmp_path / "hash")
    mismatched = _write_cohort(
        tmp_path / "hash",
        {"source_run": {"run_manifest_sha256": "not-the-source-hash"}, "groups": {"target": ["target"]}},
    )
    with pytest.raises(manifest_tool.ReaderInputManifestError, match="SHA-256 mismatch"):
        manifest_tool.build_input_manifest(run_dir, mismatched)


def test_output_is_atomic_and_immutable(tmp_path: Path) -> None:
    manifest = manifest_tool.build_input_manifest(
        _make_source_run(tmp_path), _write_cohort(tmp_path, {"groups": {"target": ["target"]}})
    )
    out_dir = tmp_path / "out"

    manifest_tool.write_input_manifest(out_dir, manifest)

    assert (out_dir / "input-manifest.json").is_file()
    with pytest.raises(manifest_tool.ReaderInputManifestError, match="already exists"):
        manifest_tool.write_input_manifest(out_dir, manifest)


def test_module_has_no_product_reader_or_model_imports() -> None:
    source = Path(manifest_tool.__file__).read_text(encoding="utf-8")

    assert "GlassesChatService" not in source
    assert "longmemeval_runner" not in source
    assert "import openai" not in source
