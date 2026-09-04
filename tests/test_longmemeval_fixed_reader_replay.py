from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_glasses_memory_assistant.evals import longmemeval_fixed_reader_replay as replay
from ai_glasses_memory_assistant.evals import longmemeval_reader_input_manifest as input_manifest


_READER_CONFIG = {
    "reader_provider": "llama_cpp",
    "reader_model": "qwen3.8-27b-32k",
    "reader_base_url": "http://10.252.17.5:11438/v1",
    "reader_max_context_chars": 16000,
    "reader_max_tokens": 4096,
    "reader_temperature": 0.0,
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "source"
    _write_json(
        run_dir / "run-manifest.json",
        {"source_snapshot": {"head": "frozen"}, "config": _READER_CONFIG},
    )
    case_dir = run_dir / "cases" / "001-target"
    _write_json(
        case_dir / "result.json",
        {
            "question_id": "target",
            "question": "What happened?",
            "question_type": "multi-session",
            "question_date": "2024-01-01",
            "recall_context": "[memory id=m1] verified fact",
            "recalled_memories": [{"id": "m1", "content": "verified fact"}],
            "recalled_timeline_chunks": [],
            "recalled_documents": [],
            "response_debug": {
                "pre_reply_decision": {
                    "answer_intent": "direct_answer",
                    "answer_obligations": ["count"],
                    "answer_focus": "facts",
                    "uncertainty_policy": "abstain_if_insufficient",
                },
                "planner": {"coverage_requirement": "best_evidence"},
                "memory": {"complete_set": {"coverage_complete": True, "source_ids": ["m1"]}},
            },
        },
    )
    _write_json(case_dir / "completed.json", {"question_id": "target"})
    cohort = tmp_path / "cohort.json"
    _write_json(cohort, {"groups": {"target": ["target"]}})
    frozen_dir = tmp_path / "frozen"
    input_manifest.write_input_manifest(frozen_dir, input_manifest.build_input_manifest(run_dir, cohort))
    return run_dir, frozen_dir / "input-manifest.json"


class FakeReader:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.last_debug: dict[str, object] | None = None

    def answer(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        self.last_debug = {
            "reader_status": "answered",
            "output_schema": "evidence_and_final_answer",
            "refusal": False,
            "selected_source_ids": ["m1"],
            "api_calls": 0,
            "execution_attempts": [],
        }
        return "verified answer"


def test_verifies_all_frozen_bytes_before_reader_calls(tmp_path: Path) -> None:
    run_dir, manifest_path = _source_and_manifest(tmp_path)

    manifest, verified = replay.verify_frozen_inputs(run_dir, manifest_path)
    reader = FakeReader()
    records = replay.run_verified_replay(verified, reader)

    assert manifest["case_count"] == 1
    assert len(reader.calls) == 1
    assert records[0]["output_schema"] == "evidence_and_final_answer"
    assert records[0]["hypothesis_sha256"] != "verified answer"
    assert replay.summarize_records(records)["judge_calls"] == 0


def test_blocks_before_reader_construction_when_source_changes(tmp_path: Path) -> None:
    run_dir, manifest_path = _source_and_manifest(tmp_path)
    result_path = run_dir / "cases" / "001-target" / "result.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["recall_context"] = "changed context"
    _write_json(result_path, data)

    with pytest.raises(replay.FixedReaderReplayError, match="frozen input"):
        replay.verify_frozen_inputs(run_dir, manifest_path)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("answer_task", "answer_task_sha256"),
        ("reader_config", "frozen Reader configuration hash mismatch"),
        ("context_limit", "reader_invocation_sha256"),
    ],
)
def test_blocks_task_or_reader_configuration_drift_before_fake_reader(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    run_dir, manifest_path = _source_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "answer_task":
        source = run_dir / "cases" / "001-target" / "result.json"
        result = json.loads(source.read_text(encoding="utf-8"))
        result["response_debug"]["pre_reply_decision"]["answer_obligations"] = ["different"]
        _write_json(source, result)
        # Preserve the outer byte hash to prove the invocation hash checks the
        # concrete Reader task, not merely a stale manifest field.
        manifest["cases"][0]["source_result_sha256"] = replay._sha256_file(source)
    elif mutation == "reader_config":
        manifest["reader_config"]["temperature"] = 0.7
    else:
        manifest["reader_config"]["max_context_chars"] = 12
        manifest["reader_config_sha256"] = replay._canonical_json_hash(manifest["reader_config"])
    _write_json(manifest_path, manifest)

    reader = FakeReader()
    with pytest.raises(replay.FixedReaderReplayError, match=expected):
        replay.verify_frozen_inputs(run_dir, manifest_path)
    assert reader.calls == []


def test_blocks_reader_or_answer_task_code_snapshot_before_fake_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, manifest_path = _source_and_manifest(tmp_path)
    monkeypatch.setattr(input_manifest, "_reader_source_path", lambda: Path(__file__))

    reader = FakeReader()
    with pytest.raises(replay.FixedReaderReplayError, match="Reader implementation snapshot changed"):
        replay.verify_frozen_inputs(run_dir, manifest_path)
    assert reader.calls == []


def test_blocks_answer_task_code_snapshot_before_fake_reader(tmp_path: Path) -> None:
    run_dir, manifest_path = _source_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["answer_task_code_sha256"] = "changed"
    _write_json(manifest_path, manifest)

    reader = FakeReader()
    with pytest.raises(replay.FixedReaderReplayError, match="answer-task implementation snapshot changed"):
        replay.verify_frozen_inputs(run_dir, manifest_path)
    assert reader.calls == []


def test_writer_is_immutable_and_does_not_store_answer_text(tmp_path: Path) -> None:
    run_dir, manifest_path = _source_and_manifest(tmp_path)
    manifest, verified = replay.verify_frozen_inputs(run_dir, manifest_path)
    records = replay.run_verified_replay(verified, FakeReader())
    out_dir = tmp_path / "replay"

    replay.write_replay(out_dir, input_manifest=manifest, records=records)

    output = (out_dir / "reader-output.jsonl").read_text(encoding="utf-8")
    assert "verified answer" not in output
    assert (out_dir / "summary.json").is_file()
    with pytest.raises(replay.FixedReaderReplayError, match="already exists"):
        replay.write_replay(out_dir, input_manifest=manifest, records=records)


def test_local_qwen_gate_rejects_different_provider_without_constructing_reader() -> None:
    config = replay.ReaderConfig(
        provider="other",
        model="other",
        base_url="http://other",
        api_key="key",
        timeout=1,
        max_tokens=1,
        temperature=0.0,
        max_context_chars=1,
        thinking="disabled",
    )
    with pytest.raises(replay.FixedReaderReplayError, match="local Qwen"):
        replay._local_qwen_reader(config)


def test_replay_has_no_judge_entrypoint() -> None:
    source = Path(replay.__file__).read_text(encoding="utf-8")
    assert "official_judge" not in source
    assert "judge_question" not in source
