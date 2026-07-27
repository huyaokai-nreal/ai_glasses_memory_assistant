from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from ai_glasses_memory_assistant.app_home import APP_HOME_ENV
from ai_glasses_memory_assistant.evals.longmemeval_adapter import (
    LongMemEvalItem,
    LongMemEvalSession,
    LongMemEvalTurn,
)
from ai_glasses_memory_assistant.evals import longmemeval_runner as runner


class FakeReader:
    def __init__(self, answer: str = "GPS system not functioning correctly") -> None:
        self.config = runner.ReaderConfig(
            provider="fake",
            model="fake-reader",
            base_url="https://example.invalid",
            api_key="secret-not-logged",
            max_context_chars=32,
        )
        self.answer_text = answer
        self.calls: list[dict[str, str]] = []

    def answer(self, **kwargs: str) -> str:
        self.calls.append(dict(kwargs))
        return self.answer_text


class FakeImportService:
    def __init__(self, shared: dict[str, Any]) -> None:
        self.shared = shared
        self.closed = False
        self.home = os.environ.get(APP_HOME_ENV)

    def import_conversation_events(self, **kwargs: Any) -> dict[str, Any]:
        assert not self.closed
        self.shared.setdefault("imports", []).append(kwargs)
        return {
            "saved_count": 1,
            "rejected_count": 0,
            "pending_confirmation_count": 0,
            "failed_count": 0,
            "paired_turn_count": 1,
            "user_only_turn_count": 0,
            "assistant_only_turn_count": 0,
            "timeline_turn_count": 1,
            "timeline_chunk_count": 1,
        }

    def import_memory_events(self, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("LongMemEval must use the conversation import contract")

    def close(self, *, timeout: float) -> None:
        self.closed = True
        self.shared["import_closed"] = timeout


class FakeQueryService:
    def __init__(self, shared: dict[str, Any]) -> None:
        self.shared = shared
        self.closed = False
        self.home = os.environ.get(APP_HOME_ENV)

    def chat(self, message: str, **kwargs: Any) -> dict[str, Any]:
        assert self.shared.get("import_closed") is not None
        self.shared["query"] = {"message": message, **kwargs}
        return {
            "reply": "native reply must be ignored",
            "api_calls": 2,
            "recalled_memories": [
                {"content": "GPS system not functioning correctly"},
                {"content": "GPS system not functioning correctly"},
            ],
            "recalled_timeline_chunks": [{"text": "First service completed"}],
            "recalled_documents": [],
            "debug": {"memory": {"event_recall_count": 1}},
        }

    def close(self, *, timeout: float) -> None:
        self.closed = True
        self.shared["query_closed"] = timeout


def make_item(question_id: str = "gpt4_test") -> LongMemEvalItem:
    return LongMemEvalItem(
        question_id=question_id,
        question_type="temporal-reasoning",
        question="What was the first issue after service?",
        answer="GPS system not functioning correctly",
        question_date="2023/04/10 (Mon) 23:07",
        sessions=[
            LongMemEvalSession(
                session_id="later",
                date="2023/04/10 (Mon) 17:50",
                turns=[
                    LongMemEvalTurn(role="user", content="The GPS stopped working."),
                    LongMemEvalTurn(role="assistant", content="The GPS system has an issue."),
                ],
            ),
            LongMemEvalSession(
                session_id="earlier",
                date="2023/04/10 (Mon) 14:47",
                turns=[
                    LongMemEvalTurn(role="user", content="The first service is complete."),
                    LongMemEvalTurn(role="assistant", content="Everything initially looked fine."),
                ],
            ),
        ],
        answer_session_ids=["later"],
        is_abstention=False,
    )


def test_import_replay_preserves_session_roles_times_without_answer_metadata() -> None:
    item = make_item()
    shared: dict[str, Any] = {}
    service = FakeImportService(shared)
    clock = runner.MutableClock(0.0)

    stats = runner.ingest_history_via_import(service, item, user_id="u1", clock=clock)

    imports = shared["imports"]
    assert [entry["session_id"] for entry in imports] == ["earlier", "later"]
    assert [[turn["role"] for turn in entry["turns"]] for entry in imports] == [
        ["user", "assistant"],
        ["user", "assistant"],
    ]
    assert [[turn["content"] for turn in entry["turns"]] for entry in imports] == [
        ["The first service is complete.", "Everything initially looked fine."],
        ["The GPS stopped working.", "The GPS system has an issue."],
    ]
    occurred_at = [turn["occurred_at"] for entry in imports for turn in entry["turns"]]
    assert occurred_at == sorted(occurred_at)
    assert len(set(occurred_at)) == 4
    forbidden_keys = {"answer", "has_answer", "answer_session_ids"}
    assert all(forbidden_keys.isdisjoint(entry) for entry in imports)
    assert all(forbidden_keys.isdisjoint(turn) for entry in imports for turn in entry["turns"])
    assert stats == {
        "history_mode": "import",
        "session_count": 2,
        "history_turn_count": 4,
        "imported_turn_count": 4,
        "saved_memory_count": 2,
        "rejected_memory_count": 0,
        "pending_confirmation_count": 0,
        "skipped_empty_turn_count": 0,
        "failed_import_count": 0,
        "paired_turn_count": 2,
        "user_only_turn_count": 0,
        "assistant_only_turn_count": 0,
        "timeline_turn_count": 2,
        "timeline_chunk_count": 2,
        "import_failures": [],
    }


def test_import_replay_keeps_assistant_only_content_as_timeline_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(APP_HOME_ENV, str(tmp_path / "app-home"))
    item = make_item("multiline")
    item.sessions[0].turns[:] = [
        LongMemEvalTurn(role="assistant", content="Alex: owns roadmap\nBeta: owns demo"),
    ]
    item.sessions[:] = [item.sessions[0]]
    clock = runner.MutableClock(runner._timestamp(item.question_date) or 0.0)
    service = runner.GlassesChatService(clock=clock)
    try:
        stats = runner.ingest_history_via_import(service, item, user_id="u1", clock=clock)
        memories = service.memory_store.list_memories("u1")
    finally:
        service.close()

    chunks = service.timeline_store.search_chunks("u1", "roadmap")

    assert stats["saved_memory_count"] == 0
    assert stats["assistant_only_turn_count"] == 1
    assert memories == []
    assert chunks and chunks[0].text == "Alex: owns roadmap\nBeta: owns demo"


def test_item_run_closes_import_before_recall_and_uses_independent_reader() -> None:
    item = make_item()
    shared: dict[str, Any] = {}
    services: list[Any] = []

    def service_factory(_clock: runner.MutableClock) -> Any:
        service = FakeImportService(shared) if not services else FakeQueryService(shared)
        services.append(service)
        return service

    reader = FakeReader()
    previous_home = os.environ.get(APP_HOME_ENV)
    result = runner.run_longmemeval_item(
        item,
        reader=reader,
        background_wait=3.0,
        history_mode="import",
        keep_home=False,
        original_app_home=previous_home,
        service_factory=service_factory,
    )

    assert result["status"] == "success"
    assert result["hypothesis"] == reader.answer_text
    assert result["reply"] != "native reply must be ignored"
    assert shared["query"]["memory_writes_allowed"] is False
    assert services[0].closed is True
    assert services[1].closed is True
    assert services[0].home == services[1].home
    assert services[0].home is not None and not Path(services[0].home).exists()
    assert reader.calls[0]["memory_context"] == (
        "Structured memories:\n- [source=structured_memory] GPS system not functioning correctly\n\n"
        "Timeline evidence:\n- [source=timeline] First service completed"
    )
    assert result["recall_context_chars"] == len(reader.calls[0]["memory_context"])
    assert result["reader_input_chars"] < result["recall_context_chars"]


def test_item_run_stops_before_reader_when_import_has_failed_fragments() -> None:
    item = make_item()
    shared: dict[str, Any] = {}
    service = FakeImportService(shared)

    def failed_import(**_kwargs: Any) -> dict[str, Any]:
        return {
            "saved_count": 0,
            "rejected_count": 0,
            "pending_confirmation_count": 0,
            "failed_count": 1,
            "failures": [{"turn_index": 0, "error_type": "ValueError", "error": "bad fragment"}],
            "paired_turn_count": 0,
            "user_only_turn_count": 0,
            "assistant_only_turn_count": 0,
            "timeline_turn_count": 0,
            "timeline_chunk_count": 0,
        }

    service.import_conversation_events = failed_import  # type: ignore[method-assign]
    reader = FakeReader()
    result = runner.run_longmemeval_item(
        item,
        reader=reader,
        background_wait=1.0,
        history_mode="import",
        keep_home=False,
        original_app_home=None,
        service_factory=lambda _clock: service,
    )

    assert result["status"] == "error"
    assert result["stage"] == "import"
    assert result["failed_import_count"] == 2
    assert len(result["import_failures"]) == 2
    assert reader.calls == []


def test_benchmark_continues_after_failure_and_writes_two_field_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [make_item("ok"), make_item("failed")]
    paths = runner.RunPaths(
        output_dir=tmp_path,
        brief_output=tmp_path / "longmemeval_oracle_memory.jsonl",
        detail_output=tmp_path / "longmemeval_oracle_memory.jsonl.details.json",
    )
    runner.prepare_run_paths(paths, overwrite=False)

    def fake_run(item: LongMemEvalItem, **_kwargs: Any) -> dict[str, Any]:
        if item.question_id == "failed":
            return {
                "question_id": item.question_id,
                "question_type": item.question_type,
                "status": "error",
                "stage": "reader",
                "hypothesis": "",
                "answer_hit": False,
                "recall_hit": False,
                "recall_context_chars": 0,
                "measured_seconds": 0.1,
            }
        return {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "status": "success",
            "stage": "complete",
            "hypothesis": "answer",
            "answer_hit": True,
            "recall_hit": True,
            "recall_context_chars": 12,
            "measured_seconds": 0.1,
        }

    monkeypatch.setattr(runner, "run_longmemeval_item", fake_run)
    progress_output = StringIO()
    progress = runner.ProgressReporter(len(items), stream=progress_output)
    runs = runner.run_benchmark_items(
        items,
        reader=FakeReader(),
        paths=paths,
        progress=progress,
        app_llm_env={},
        original_app_home=None,
        background_wait=1.0,
        history_mode="import",
        keep_homes=False,
    )
    runner.write_json_output(paths.detail_output, runs)

    brief_rows = [json.loads(line) for line in paths.brief_output.read_text(encoding="utf-8").splitlines()]
    details = json.loads(paths.detail_output.read_text(encoding="utf-8"))
    assert brief_rows == [{"question_id": "ok", "hypothesis": "answer"}]
    assert list(brief_rows[0]) == ["question_id", "hypothesis"]
    assert [row["status"] for row in details] == ["success", "error"]
    assert "2/2" in progress_output.getvalue()
    assert "success=1 failed=1" in progress_output.getvalue()
    assert "phase=error" in progress_output.getvalue()


def test_recall_context_deduplicates_and_counts_before_reader_truncation() -> None:
    context = runner.build_recall_context({
        "recalled_memories": [
            {"content": "alpha", "occurred_at": 1672531200, "time_granularity": "day"},
            {"content": "alpha", "occurred_at": 1672531200, "time_granularity": "day"},
            {"content": "beta"},
        ],
        "recalled_timeline_chunks": [{"text": "raw evidence", "timestamp": 1672617600}],
        "recalled_documents": [{"title": "Doc", "summary": "summary"}],
    })

    assert context.count("] alpha") == 1
    assert "source=structured_memory" in context
    assert "event_time=2023-01-01" in context
    assert "source=timeline" in context
    assert "recorded_at=2023-01-02" in context
    assert "Document evidence:\n- Doc: summary" in context
    assert len(runner.truncate_text(context, 30)) < len(context)
    assert len(context) > 30


def test_import_replay_preserves_fragment_failures_for_diagnosis() -> None:
    item = make_item()
    shared: dict[str, Any] = {}
    service = FakeImportService(shared)

    def failed_import(**kwargs: Any) -> dict[str, Any]:
        shared.setdefault("imports", []).append(kwargs)
        return {
            "saved_count": 1,
            "rejected_count": 0,
            "pending_confirmation_count": 0,
            "failed_count": 1,
            "failures": [{"turn_index": 1, "error_type": "ValueError", "error": "bad fragment"}],
            "paired_turn_count": 1,
            "user_only_turn_count": 0,
            "assistant_only_turn_count": 0,
            "timeline_turn_count": 1,
            "timeline_chunk_count": 1,
        }

    service.import_conversation_events = failed_import  # type: ignore[method-assign]
    stats = runner.ingest_history_via_import(service, item, user_id="u1", clock=runner.MutableClock(0.0))

    assert stats["failed_import_count"] == 2
    assert stats["import_failures"] == [
        {
            "session_id": "earlier",
            "failed_count": 1,
            "failures": [{"turn_index": 1, "error_type": "ValueError", "error": "bad fragment"}],
        },
        {
            "session_id": "later",
            "failed_count": 1,
            "failures": [{"turn_index": 1, "error_type": "ValueError", "error": "bad fragment"}],
        },
    ]


def test_reader_prompt_and_answer_parser_match_external_contract() -> None:
    prompt = runner.build_reader_prompt(
        question="Where?",
        question_type="single-session-user",
        question_date="2023/01/01 12:00",
        memory_context="- at home",
    )

    assert "Use only the memory context below" in prompt
    assert '"relevant_evidence"' in prompt
    assert runner.extract_reader_final_answer(
        '```json\n{"relevant_evidence": ["at home"], "final_answer": "home"}\n```'
    ) == "home"


def test_longmemeval_timestamp_and_progress_are_human_readable() -> None:
    assert runner._timestamp("2023/04/10 (Mon) 23:07") is not None
    output = StringIO()
    progress = runner.ProgressReporter(2, stream=output)

    progress.update(question_id="q1", phase="import 1/4")
    progress.complete(question_id="q1", succeeded=True)

    rendered = output.getvalue()
    assert "phase=import 1/4" in rendered
    assert "1/2" in rendered
    assert "success=1 failed=0" in rendered
    assert "ETA=" in rendered
    assert "secret" not in rendered


def test_explicit_question_ids_are_not_truncated_by_default_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "oracle.json"
    dataset_path.write_text(json.dumps([
        {
            "question_id": f"q{index}",
            "question_type": "temporal-reasoning",
            "question": "When?",
            "answer": "today",
            "question_date": "2023/01/01 (Sun) 00:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
        }
        for index in range(21)
    ]), encoding="utf-8")
    parser = runner.build_parser()
    args = parser.parse_args([
        "--dataset-path", str(dataset_path),
        *[part for index in range(21) for part in ("--question-id", f"q{index}")],
    ])

    assert len(runner.select_items(args)) == 21
