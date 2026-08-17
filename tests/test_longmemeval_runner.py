from __future__ import annotations

import json
import os
import sqlite3
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


# Module-level state is per-process, so the factory is picklable for
# ProcessPoolExecutor workers (each worker gets its own copy of the globals).
_PROCESS_SHARED: dict[str, Any] = {}


class _FakeEvalService:
    """One fake covering both import and recall phases for worker processes."""

    def __init__(self, shared: dict[str, Any]) -> None:
        self.shared = shared
        self.closed = False

    def import_conversation_events(self, **kwargs: Any) -> dict[str, Any]:
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
        self.shared["import_closed"] = timeout
        self.shared["query_closed"] = timeout


def _process_service_factory(_clock: runner.MutableClock) -> Any:
    return _FakeEvalService(_PROCESS_SHARED)


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


def test_summary_keeps_question_type_groups_exclusive_and_abstention_overlay() -> None:
    runs = [
        {
            "question_type": "temporal-reasoning",
            "is_abstention": False,
            "status": "success",
            "answer_hit": True,
            "recall_hit": True,
            "measured_seconds": 1.0,
        },
        {
            "question_type": "temporal-reasoning",
            "is_abstention": True,
            "status": "success",
            "answer_hit": True,
            "recall_hit": False,
            "measured_seconds": 2.0,
        },
        {
            "question_type": "single-session-user",
            "is_abstention": False,
            "status": "success",
            "answer_hit": False,
            "recall_hit": True,
            "measured_seconds": 3.0,
        },
    ]

    summary = runner.summarize_longmemeval_runs(runs)

    assert summary["overall"]["total"] == 3
    assert summary["non_abstention"]["total"] == 2
    assert summary["abstention"]["total"] == 1
    assert "correctness_rate" not in summary["abstention"]
    assert "answer_hit_rate" not in summary["overall"]
    assert "recall_hit_rate" not in summary["overall"]
    assert set(summary["by_question_type"]) == {"temporal-reasoning", "single-session-user"}
    assert sum(item["total"] for item in summary["by_question_type"].values()) == 3


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


def test_build_context_enables_skip_reply_synthesis_for_recall() -> None:
    item = make_item()
    shared: dict[str, Any] = {}
    services: list[Any] = []

    def service_factory(_clock: runner.MutableClock) -> Any:
        service = FakeImportService(shared) if not services else FakeQueryService(shared)
        services.append(service)
        return service

    context = runner.build_question_memory_context(
        item,
        background_wait=1.0,
        history_mode="import",
        original_app_home=None,
        service_factory=service_factory,
    )

    assert context["status"] == "success"
    assert shared["query"]["skip_reply_synthesis"] is True
    assert shared["query"]["memory_writes_allowed"] is False


def test_two_phase_split_matches_serial_item_run() -> None:
    item = make_item()

    def run_serial() -> dict[str, Any]:
        shared: dict[str, Any] = {}
        services: list[Any] = []

        def factory(_clock: runner.MutableClock) -> Any:
            service = FakeImportService(shared) if not services else FakeQueryService(shared)
            services.append(service)
            return service

        return runner.run_longmemeval_item(
            item,
            reader=FakeReader(),
            background_wait=1.0,
            history_mode="import",
            keep_home=False,
            original_app_home=None,
            service_factory=factory,
        )

    def run_split() -> dict[str, Any]:
        shared: dict[str, Any] = {}
        services: list[Any] = []

        def factory(_clock: runner.MutableClock) -> Any:
            service = FakeImportService(shared) if not services else FakeQueryService(shared)
            services.append(service)
            return service

        reader = FakeReader()
        context = runner.build_question_memory_context(
            item,
            background_wait=1.0,
            history_mode="import",
            original_app_home=None,
            service_factory=factory,
        )
        result = runner.answer_question_from_memory_context(context, reader=reader)
        runner._remove_ephemeral_app_home(result, keep_homes=False)
        return result

    serial = run_serial()
    split = run_split()
    for key in (
        "question_id",
        "status",
        "stage",
        "hypothesis",
        "reply",
        "answer_hit",
        "recall_hit",
        "recall_context",
        "recall_context_chars",
        "recalled_memory_count",
        "recalled_timeline_count",
        "api_calls",
        "failed_import_count",
        "app_home",
    ):
        assert split[key] == serial[key], key
    assert set(split["phase_seconds"]) == set(serial["phase_seconds"])


def test_parallel_workers_match_serial_order_and_results(tmp_path: Path) -> None:
    items = [make_item(f"gpt4_test_{index}") for index in range(3)]
    paths = runner.RunPaths(
        output_dir=tmp_path / "out",
        brief_output=tmp_path / "out" / "brief.jsonl",
        detail_output=tmp_path / "out" / "details.json",
    )

    def run_with(workers: int) -> list[dict[str, Any]]:
        return runner.run_benchmark_items(
            items,
            reader=FakeReader(),
            paths=paths,
            progress=runner.ProgressReporter(len(items), stream=StringIO()),
            app_llm_env={},
            original_app_home=None,
            background_wait=1.0,
            history_mode="import",
            keep_homes=False,
            service_factory=_process_service_factory,
            workers=workers,
        )

    serial = run_with(workers=1)
    parallel = run_with(workers=2)

    assert [run["question_id"] for run in serial] == [item.question_id for item in items]
    assert [run["question_id"] for run in parallel] == [item.question_id for item in items]
    for serial_run, parallel_run in zip(serial, parallel):
        assert parallel_run["status"] == "success"
        assert parallel_run["hypothesis"] == serial_run["hypothesis"]
        assert parallel_run["recall_context"] == serial_run["recall_context"]
        assert parallel_run["recalled_memory_count"] == serial_run["recalled_memory_count"]
        assert "reader" in parallel_run["phase_seconds"]


def test_two_level_cache_hits_skip_import_and_recall(tmp_path: Path) -> None:
    dataset_path = tmp_path / "oracle.json"
    dataset_path.write_text("[]", encoding="utf-8")
    item = make_item()
    counters: dict[str, int] = {"import": 0, "recall": 0}
    shared: dict[str, Any] = {}
    services: list[Any] = []

    def factory(_clock: runner.MutableClock) -> Any:
        if not services:
            counters["import"] += 1
            service: Any = FakeImportService(shared)
        else:
            counters["recall"] += 1
            service = FakeQueryService(shared)
        services.append(service)
        return service

    cache_root = runner.prepare_cache_root(
        tmp_path / "cache" / "oracle",
        dataset_path=dataset_path,
        history_mode="import",
    )
    paths = runner.RunPaths(
        tmp_path / "out",
        tmp_path / "out" / "brief.jsonl",
        tmp_path / "out" / "details.json",
    )

    def run_once() -> dict[str, Any]:
        return runner.run_benchmark_items(
            [item],
            reader=FakeReader(),
            paths=paths,
            progress=runner.ProgressReporter(1, stream=StringIO()),
            app_llm_env={},
            original_app_home=None,
            background_wait=1.0,
            history_mode="import",
            keep_homes=False,
            service_factory=factory,
            workers=1,
            cache_root=cache_root,
            cache_enabled=True,
        )[0]

    question_cache = cache_root / "gpt4_test"
    first = run_once()
    assert counters == {"import": 1, "recall": 1}
    assert first["cache_hits"] == {"l1": False, "l2": False}
    assert (question_cache / "app_home" / "import_meta.json").is_file()
    assert (question_cache / "recall.json").is_file()

    second = run_once()
    assert counters == {"import": 1, "recall": 1}
    assert second["cache_hits"] == {"l1": True, "l2": True}
    assert second["phase_seconds"]["import"] == 0.0
    assert second["phase_seconds"]["recall"] == 0.0
    assert second["recall_context"] == first["recall_context"]
    assert second["hypothesis"] == first["hypothesis"]

    (question_cache / "recall.json").unlink()
    third = run_once()
    assert counters == {"import": 1, "recall": 2}
    assert third["cache_hits"] == {"l1": True, "l2": False}
    assert third["phase_seconds"]["import"] == 0.0
    assert third["recall_context"] == first["recall_context"]


def test_cache_rebuilds_when_manifest_key_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "oracle.json"
    dataset_path.write_text("[]", encoding="utf-8")
    cache_root = tmp_path / "cache" / "oracle"
    runner.prepare_cache_root(cache_root, dataset_path=dataset_path, history_mode="import")
    stale = cache_root / "gpt4_test" / "recall.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")

    runner.prepare_cache_root(cache_root, dataset_path=dataset_path, history_mode="timeline")
    assert not (cache_root / "gpt4_test").exists()
    manifest = json.loads((cache_root / "cache-manifest.json").read_text(encoding="utf-8"))
    assert manifest["history_mode"] == "timeline"

    monkeypatch.setattr(runner, "LONGMEMEVAL_CACHE_VERSION", "v9")
    runner.prepare_cache_root(cache_root, dataset_path=dataset_path, history_mode="timeline")
    manifest = json.loads((cache_root / "cache-manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "v9"


def test_corrupt_recall_cache_is_treated_as_miss_and_rebuilt(tmp_path: Path) -> None:
    dataset_path = tmp_path / "oracle.json"
    dataset_path.write_text("[]", encoding="utf-8")
    item = make_item()
    counters: dict[str, int] = {"import": 0, "recall": 0}
    shared: dict[str, Any] = {}
    services: list[Any] = []

    def factory(_clock: runner.MutableClock) -> Any:
        if not services:
            counters["import"] += 1
            service: Any = FakeImportService(shared)
        else:
            counters["recall"] += 1
            service = FakeQueryService(shared)
        services.append(service)
        return service

    cache_root = runner.prepare_cache_root(
        tmp_path / "cache" / "oracle",
        dataset_path=dataset_path,
        history_mode="import",
    )
    question_cache = cache_root / "gpt4_test"
    question_cache.mkdir(parents=True)
    (question_cache / "recall.json").write_text("not json", encoding="utf-8")

    context = runner.build_question_memory_context(
        item,
        background_wait=1.0,
        history_mode="import",
        original_app_home=None,
        service_factory=factory,
        cache_dir=question_cache,
        cache_enabled=True,
    )

    assert context["status"] == "success"
    assert counters == {"import": 1, "recall": 1}
    assert json.loads((question_cache / "recall.json").read_text(encoding="utf-8"))["recall_context"]


def test_no_cache_flag_disables_reads_and_writes(tmp_path: Path) -> None:
    parser = runner.build_parser()
    args = parser.parse_args(["--no-cache"])
    assert args.no_cache is True
    assert runner.resolve_cache_root(parser.parse_args([])).is_absolute()
    assert runner.prepare_cache_root(
        None,
        dataset_path=tmp_path / "oracle.json",
        history_mode="import",
    ) is None


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
    assert "synthesize only the user preference or constraint directly supported" in prompt
    assert "answer from them instead of claiming that memory is unavailable" in prompt
    assert '"relevant_evidence"' in prompt
    assert "build the answer as an incremental next step" in prompt
    assert "Never add unmentioned brand names" in prompt
    assert "cover that dimension explicitly in the final answer" in prompt
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


def test_checkpoint_persists_each_case_and_resume_skips_completed_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "oracle.json"
    dataset_path.write_text("[]", encoding="utf-8")
    items = [make_item("first"), make_item("second")]
    paths = runner.RunPaths(
        output_dir=tmp_path / "checkpoint",
        brief_output=tmp_path / "checkpoint" / "oracle_memory.jsonl",
        detail_output=tmp_path / "checkpoint" / "oracle_memory.jsonl.details.json",
    )
    config = {"dataset_path": str(dataset_path), "history_mode": "import", "reader_model": "fake"}
    manifest = runner.build_run_manifest(items=items, config=config)
    assert runner.prepare_checkpoint_run(paths, manifest=manifest, overwrite=False, resume=False) == {}
    calls: list[str] = []

    def fake_run(item: LongMemEvalItem, **_kwargs: Any) -> dict[str, Any]:
        calls.append(item.question_id)
        home = tmp_path / "homes" / f"glasses-longmemeval-{item.question_id}"
        (home / "data").mkdir(parents=True)
        return {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "status": "success" if item.question_id == "first" else "error",
            "stage": "complete" if item.question_id == "first" else "reader",
            "hypothesis": "answer" if item.question_id == "first" else "",
            "answer_hit": item.question_id == "first",
            "recall_hit": item.question_id == "first",
            "recall_context": "evidence" if item.question_id == "first" else "",
            "recalled_memory_count": 1 if item.question_id == "first" else 0,
            "recalled_timeline_count": 0,
            "failed_import_count": 0,
            "measured_seconds": 0.1,
            "app_home": str(home),
        }

    monkeypatch.setattr(runner, "run_longmemeval_item", fake_run)
    progress = runner.ProgressReporter(len(items), stream=StringIO())
    runs = runner.run_benchmark_items(
        items,
        reader=FakeReader(),
        paths=paths,
        progress=progress,
        app_llm_env={},
        original_app_home=None,
        background_wait=0,
        history_mode="import",
        keep_homes=False,
        config=config,
    )

    assert calls == ["first", "second"]
    assert [run["failure_classification"]["stage"] for run in runs] == [
        "evidence_supported_answer",
        "write_or_type_loss",
    ]
    assert (paths.output_dir / "cases" / "001-first" / "completed.json").is_file()
    assert (paths.output_dir / "cases" / "002-second" / "completed.json").is_file()
    assert not (tmp_path / "homes" / "glasses-longmemeval-first").exists()
    assert [json.loads(line) for line in paths.brief_output.read_text(encoding="utf-8").splitlines()] == [
        {"question_id": "first", "hypothesis": "answer"}
    ]

    completed = runner.prepare_checkpoint_run(paths, manifest=manifest, overwrite=False, resume=True)
    resumed = runner.run_benchmark_items(
        items,
        reader=FakeReader(),
        paths=paths,
        progress=runner.ProgressReporter(len(items), stream=StringIO()),
        app_llm_env={},
        original_app_home=None,
        background_wait=0,
        history_mode="import",
        keep_homes=False,
        config=config,
        completed_runs=completed,
        max_new_cases=1,
    )
    assert calls == ["first", "second"]
    assert [item["question_id"] for item in resumed] == ["first", "second"]


def test_max_new_cases_only_limits_current_invocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = runner.RunPaths(tmp_path / "checkpoint", tmp_path / "checkpoint" / "brief.jsonl", tmp_path / "checkpoint" / "details.json")
    runner.prepare_run_paths(paths, overwrite=False)
    items = [make_item("first"), make_item("second")]
    calls: list[str] = []

    def fake_run(item: LongMemEvalItem, **_kwargs: Any) -> dict[str, Any]:
        calls.append(item.question_id)
        return {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "status": "success",
            "hypothesis": "answer",
            "answer_hit": True,
            "recall_hit": True,
            "recall_context_chars": 1,
            "measured_seconds": 0.1,
            "app_home": "<temporary>",
        }

    monkeypatch.setattr(runner, "run_longmemeval_item", fake_run)
    runs = runner.run_benchmark_items(
        items,
        reader=FakeReader(),
        paths=paths,
        progress=runner.ProgressReporter(len(items), stream=StringIO()),
        app_llm_env={},
        original_app_home=None,
        background_wait=0,
        history_mode="import",
        keep_homes=False,
        max_new_cases=1,
    )

    assert calls == ["first"]
    assert [run["question_id"] for run in runs] == ["first"]


def test_checkpoint_rejects_different_manifest(tmp_path: Path) -> None:
    dataset_path = tmp_path / "oracle.json"
    dataset_path.write_text("[]", encoding="utf-8")
    paths = runner.RunPaths(tmp_path / "checkpoint", tmp_path / "checkpoint" / "brief.jsonl", tmp_path / "checkpoint" / "details.json")
    first = runner.build_run_manifest(items=[make_item("first")], config={"dataset_path": str(dataset_path)})
    runner.prepare_checkpoint_run(paths, manifest=first, overwrite=False, resume=False)
    second = runner.build_run_manifest(items=[make_item("second")], config={"dataset_path": str(dataset_path)})

    with pytest.raises(ValueError, match="Checkpoint manifest"):
        runner.prepare_checkpoint_run(paths, manifest=second, overwrite=False, resume=True)


def test_case_evidence_redacts_sqlite_and_audit_payloads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    data_dir = home / "data"
    data_dir.mkdir(parents=True)
    secret = "password: correct-horse-battery-staple"
    database = data_dir / "events.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE memories (id TEXT, content TEXT)")
        connection.execute("INSERT INTO memories VALUES (?, ?)", ("m1", secret))
    (data_dir / "chat_audit.jsonl").write_text(json.dumps({"message": secret}) + "\n", encoding="utf-8")

    destination = tmp_path / "evidence"
    exported = runner.export_case_evidence(home, destination)

    assert exported["exported"] == ["audit.redacted.jsonl", "database/events.db"]
    assert secret not in (destination / "audit.redacted.jsonl").read_text(encoding="utf-8")
    with sqlite3.connect(destination / "database" / "events.db") as connection:
        content = connection.execute("SELECT content FROM memories WHERE id = 'm1'").fetchone()[0]
    assert content == runner.redact_sensitive_text(secret).text


def test_case_evidence_exports_real_memory_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "glasses-longmemeval-real"
    monkeypatch.setenv(APP_HOME_ENV, str(home))
    service = runner.GlassesChatService()
    try:
        service.memory_store.add_memory("u1", "normal memory", kind="event", memory_type="event")
    finally:
        service.close()

    exported = runner.export_case_evidence(home, tmp_path / "evidence")

    assert "database/events.db" in exported["exported"]
    assert "database/timeline.db" in exported["exported"]


def test_offline_failure_taxonomy_uses_final_planner_flags_not_debug_dictionary_truthiness() -> None:
    route_off = {
        "status": "success",
        "is_abstention": False,
        "answer_hit": False,
        "recall_hit": False,
        "recall_context": "",
        "recalled_memory_count": 0,
        "recalled_timeline_count": 0,
        "response_debug": {
            "turn_decision": {"final": {"needs_profile_memory": False, "needs_event_memory": False, "needs_timeline_recall": False}},
            "memory": {"event_recall": {"strategy": "skipped_by_planner"}},
            "timeline": {"recall": {"strategy": "skipped_by_planner"}},
        },
    }
    assert runner.classify_offline_failure(route_off)["stage"] == "routing_miss"

    requested_empty = {
        **route_off,
        "response_debug": {
            "turn_decision": {"final": {"needs_profile_memory": True, "needs_event_memory": False, "needs_timeline_recall": False}},
            "memory": {"event_recall": {"strategy": "skipped_by_planner"}},
            "timeline": {"recall": {"strategy": "skipped_by_planner"}},
        },
    }
    assert runner.classify_offline_failure(requested_empty)["stage"] == "retrieval_coverage_loss"

    context_without_verified_evidence = {
        **requested_empty,
        "recall_context": "some retrieved memory",
        "recalled_memory_count": 1,
        "response_debug": {
            **requested_empty["response_debug"],
            "memory": {"event_recall": {"strategy": "text_search"}, "candidate_trace": {"candidates": []}},
        },
    }
    assert runner.classify_offline_failure(context_without_verified_evidence)["stage"] == "unclassified_insufficient_evidence"


def test_offline_failure_taxonomy_only_assigns_ranking_or_reader_after_explicit_review_evidence() -> None:
    base = {
        "status": "success",
        "is_abstention": False,
        "answer_hit": False,
        "recall_context": "some retrieved memory",
        "response_debug": {"turn_decision": {"final": {"needs_event_memory": True}}},
    }
    ranking = {**base, "offline_evidence": {"candidate_support": "verified", "selected_support": False}}
    reader = {**base, "offline_evidence": {"selected_support": "verified"}}

    assert runner.classify_offline_failure(ranking)["stage"] == "ranking_or_context_loss"
    assert runner.classify_offline_failure(reader)["stage"] == "reader_synthesis_loss"


# ── Stage 3: Reader debug observability ──


def test_parse_reader_structured_extracts_evidence_and_answer() -> None:
    result = runner._parse_reader_structured(
        '{"relevant_evidence": ["line 1", "line 2"], "final_answer": "the answer"}'
    )
    assert result is not None
    assert result["relevant_evidence"] == ["line 1", "line 2"]
    assert result["final_answer"] == "the answer"


def test_parse_reader_structured_accepts_only_final_answer() -> None:
    result = runner._parse_reader_structured(
        '{"final_answer": "just the answer"}'
    )
    assert result is not None
    assert result["relevant_evidence"] == []
    assert result["final_answer"] == "just the answer"


def test_parse_reader_structured_rejects_invalid_inputs() -> None:
    assert runner._parse_reader_structured("") is None
    assert runner._parse_reader_structured("not json") is None
    assert runner._parse_reader_structured('{"other": "field"}') is None


def test_fake_reader_compatible_with_reader_debug() -> None:
    reader = FakeReader(answer="the answer")
    result = reader.answer(
        question="q", question_type="single", question_date="2024",
        memory_context="some context",
    )
    assert result == "the answer"
    # FakeReader does not set last_debug; callers must use getattr safely.
    assert getattr(reader, "last_debug", None) is None


# ── Stage 5: phase timing ──


def test_summarize_phase_seconds_aggregates_count_mean_median_p95() -> None:
    runs = [
        {"status": "success", "phase_seconds": {"import": 1.0, "import_close": 0.2, "recall": 2.0, "reader": 3.0}},
        {"status": "success", "phase_seconds": {"import": 2.0, "import_close": 0.3, "recall": 3.0, "reader": 4.0}},
        {"status": "success", "phase_seconds": {"import": 3.0, "import_close": 0.1, "recall": 4.0, "reader": 5.0}},
    ]
    summary = runner._summarize_phase_seconds(runs)
    assert summary["import"]["count"] == 3
    assert summary["import"]["mean"] == 2.0
    assert summary["import"]["median"] == 2.0
    assert summary["import"]["p95"] == 3.0  # 3 items, 95th percentile = last
    assert summary["reader"]["mean"] == 4.0


def test_summarize_phase_seconds_handles_empty_and_partial_runs() -> None:
    assert runner._summarize_phase_seconds([])["import"]["count"] == 0
    runs = [{"status": "error", "phase_seconds": {"import": 5.0}}]
    # Error runs are excluded; only success runs count.
    assert runner._summarize_phase_seconds(runs)["import"]["count"] == 0


# ── Stage A: reader refusal-with-evidence retry ──


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.sent_prompts: list[str] = []

    def create(self, **kwargs) -> Any:
        content = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        messages = kwargs.get("messages") or []
        if messages and isinstance(messages[0], dict):
            self.sent_prompts.append(str(messages[0].get("content") or ""))
        return type("R", (), {"choices": [_FakeChoice(content)]})()


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = type("C", (), {"completions": _FakeCompletions(responses)})()


def _make_reader(responses: list[str]) -> runner.OpenAIReader:
    import ai_glasses_memory_assistant.evals.longmemeval_runner as runner_mod
    reader = runner.OpenAIReader.__new__(runner.OpenAIReader)
    reader.config = runner.ReaderConfig(
        provider="fake",
        model="fake-reader",
        base_url="https://example.invalid",
        api_key="secret-not-logged",
        max_context_chars=16000,
        max_tokens=1024,
        temperature=0.0,
    )
    reader._client = _FakeClient(responses)
    reader.last_debug = None
    return reader


def test_reader_refuses_with_evidence_triggers_retry() -> None:
    refusal = json.dumps({
        "relevant_evidence": ["I'm trying to learn Adobe Premiere Pro advanced settings"],
        "final_answer": runner.UNKNOWN_ANSWER,
    })
    follow_up = json.dumps({
        "relevant_evidence": ["I'm trying to learn Adobe Premiere Pro advanced settings"],
        "final_answer": "Use Adobe Premiere Pro tutorials for advanced color grading.",
    })
    reader = _make_reader([refusal, follow_up])

    answer = reader.answer(
        question="Recommend resources for video editing?",
        question_type="single-session-preference",
        question_date="2023/05/20 12:00",
        memory_context="Structured memories:\n- I'm trying to learn Adobe Premiere Pro advanced settings",
    )

    assert answer == "Use Adobe Premiere Pro tutorials for advanced color grading."
    assert reader.last_debug is not None
    assert reader.last_debug["refusal_retry"] is True
    assert reader.last_debug["refusal"] is False
    assert reader.last_debug["relevant_evidence"]


def test_reader_refusal_without_evidence_with_context_retries() -> None:
    refusal = json.dumps({
        "relevant_evidence": [],
        "final_answer": runner.UNKNOWN_ANSWER,
    })
    follow_up = json.dumps({
        "relevant_evidence": ["I enjoy listening to podcasts during my commute"],
        "final_answer": "Try history podcasts during your commute.",
    })
    reader = _make_reader([refusal, follow_up])

    answer = reader.answer(
        question="Recommend resources?",
        question_type="single-session-preference",
        question_date="2023/05/20 12:00",
        memory_context="Structured memories:\n- something unrelated",
    )

    assert answer == "Try history podcasts during your commute."
    assert reader.last_debug is not None
    assert reader.last_debug["refusal_retry"] is True
    assert reader.last_debug["refusal"] is False
    assert reader.last_debug["relevant_evidence"]
    assert reader._client.chat.completions.calls == 2


def test_reader_refusal_without_evidence_retry_still_refuses() -> None:
    refusal = json.dumps({
        "relevant_evidence": [],
        "final_answer": runner.UNKNOWN_ANSWER,
    })
    reader = _make_reader([refusal, refusal])

    answer = reader.answer(
        question="Recommend resources?",
        question_type="single-session-preference",
        question_date="2023/05/20 12:00",
        memory_context="Structured memories:\n- something unrelated",
    )

    assert answer == runner.UNKNOWN_ANSWER
    assert reader.last_debug is not None
    assert reader.last_debug["refusal_retry"] is True
    assert reader.last_debug["refusal"] is True
    assert reader._client.chat.completions.calls == 2


def test_reader_empty_context_does_not_retry() -> None:
    reader = _make_reader([])

    answer = reader.answer(
        question="Recommend resources?",
        question_type="single-session-preference",
        question_date="2023/05/20 12:00",
        memory_context="",
    )

    assert answer == runner.UNKNOWN_ANSWER
    assert reader.last_debug is not None
    assert reader.last_debug["refusal"] is True
    assert reader.last_debug.get("refusal_retry", False) is False
    assert reader._client.chat.completions.calls == 0


def test_render_markdown_prefers_official_judge_over_local_metrics() -> None:
    payload = {
        "generated_at": "2026-08-05T18:00:00+08:00",
        "config": {"dataset_path": "oracle.json", "history_mode": "import", "reader_model": "deepseek-v4-flash"},
        "summary": {
            "overall": {"total": 30, "succeeded": 30, "failed": 0, "mean_seconds": 60.0},
            "non_abstention": {"total": 30, "succeeded": 30, "failed": 0, "mean_seconds": 60.0},
            "abstention": {"total": 0, "succeeded": 0, "failed": 0, "mean_seconds": 0.0},
            "recall_context": {"total_chars": 40000, "mean_chars": 1333},
            "by_question_type": {"single-session-preference": {"total": 30, "succeeded": 30}},
            "failures": [],
            "official_judge": {
                "judge_model": "deepseek-chat",
                "protocol": "longmemeval-official-qa-v1",
                "upstream_reference_judge": "gpt-4o-2024-08-06",
                "judge_model_substitution": True,
                "judge_max_tokens": 512,
                "upstream_max_tokens": 10,
                "overall": {
                    "total": 30,
                    "correct": 12,
                    "correct_rate": 0.4,
                    "refusals": 1,
                    "parse_errors": 0,
                },
                "by_question_type": {
                    "single-session-preference": {"total": 30, "correct": 12, "correct_rate": 0.4},
                },
                "by_abstention": {
                    "answerable": {"total": 29, "correct": 11, "correct_rate": 0.3793},
                    "abstention": {"total": 1, "correct": 1, "correct_rate": 1.0},
                },
                "per_question": {"32260d93": 0, "8a2466db": 1},
            },
        },
        "runs": [],
    }
    markdown = runner.render_markdown(payload)

    assert "总览（官方 judge）" in markdown
    assert "官方 judge 命中率" in markdown
    assert "40.0%" in markdown
    assert "12/30" in markdown
    assert "非拒答命中率" not in markdown
    assert "longmemeval-official-qa-v1" in markdown
    assert "gpt-4o-2024-08-06" in markdown
    assert "judge 模型替代 | 是" in markdown
    assert "512（上游默认 10）" in markdown
    assert "官方 judge 分题型结果" in markdown
    assert "single-session-preference | 12/30 | 40.0%" in markdown
    assert "官方 judge 可回答性切片" in markdown
    assert "abstention | 1/1 | 100.0%" in markdown
    # Local substring rate must NOT be presented as the headline metric.
    assert "回答命中率（本地 substring）" not in markdown
    assert "召回命中率" not in markdown
    assert "官方 judge 逐题明细" in markdown
    assert "✅ 1" in markdown
    assert "❌ 0" in markdown
    # 说明段落仍保留“本地指标只进 details”的口径提示。
    assert "不展示任何本地命中率作为成绩" in markdown


def test_render_markdown_without_official_judge_shows_no_local_rates() -> None:
    payload = {
        "generated_at": "2026-08-13T18:00:00+08:00",
        "config": {"dataset_path": "oracle.json", "history_mode": "import", "reader_model": "fake"},
        "summary": {
            "overall": {"total": 3, "succeeded": 3, "failed": 0, "mean_seconds": 1.5},
            "non_abstention": {"total": 3, "succeeded": 3, "failed": 0, "mean_seconds": 1.5},
            "abstention": {"total": 0, "succeeded": 0, "failed": 0, "mean_seconds": 0.0},
            "recall_context": {"total_chars": 300, "mean_chars": 100},
            "phase_seconds": {},
            "by_question_type": {"single-session-user": {"total": 3, "succeeded": 3}},
            "failures": [],
        },
        "runs": [],
    }
    markdown = runner.render_markdown(payload)

    assert "总览" in markdown
    assert "成功数" in markdown
    assert "回答命中率" not in markdown
    assert "召回命中率" not in markdown
    assert "官方 judge" not in markdown or "正式成绩需运行官方 judge" in markdown


# ── General answer contract: fixed-evidence tests ──


def test_answer_task_renders_only_when_provided() -> None:
    legacy = runner.build_reader_prompt(
        question="q",
        question_type="single-session-user",
        question_date="2023/01/01 12:00",
        memory_context="- at home",
    )
    contracted = runner.build_reader_prompt(
        question="q",
        question_type="single-session-user",
        question_date="2023/01/01 12:00",
        memory_context="- at home",
        answer_task={
            "answer_intent": "personalized_recommendation",
            "answer_focus": "recommend accessories for existing camera",
            "answer_obligations": ["incremental_next_step", "negation_constraints"],
            "uncertainty_policy": "state_limits_when_context_is_sparse",
        },
    )
    # Legacy prompt must remain byte-for-byte the default when no task is given.
    assert "Answer task contract:" not in legacy
    assert "coverage" not in legacy
    # Contracted prompt adds the generic contract, never preference-only rules.
    assert "Answer task contract:" in contracted
    assert "answer_intent: personalized_recommendation" in contracted
    assert "answer_obligations: incremental_next_step, negation_constraints" in contracted
    assert "uncertainty_policy: state_limits_when_context_is_sparse" in contracted
    assert "Working order for final_answer:" in contracted
    assert "coverage" in contracted
    # The 13 base rules are still present.
    assert "Use only the memory context below" in contracted
    assert "synthesize only the user preference or constraint directly supported" in contracted


def test_answer_task_renders_obligation_hints() -> None:
    prompt = runner.build_reader_prompt(
        question="q",
        question_type="single-session-user",
        question_date="2023/01/01 12:00",
        memory_context="- at home",
        answer_task={
            "answer_intent": "personalized_recommendation",
            "answer_obligations": ["negation_constraints", "incremental_next_step", "comparison"],
            "uncertainty_policy": "state_limits_when_context_is_sparse",
        },
    )

    assert "Negation constraints: explicitly state what the user would not prefer" in prompt
    assert "Incremental next step: build the answer on top of what the user already owns" in prompt
    assert "Comparison: cover both sides of the comparison explicitly." in prompt
    assert "cover that dimension explicitly in the final answer" in prompt
    assert "make final_answer cover each obligation" in prompt


def test_reader_forward_answer_task_and_records_coverage() -> None:
    fixed = json.dumps({
        "relevant_evidence": ["I bought a portable power bank for my phone."],
        "coverage": ["incremental_next_step"],
        "final_answer": "Use your existing power bank to top up during the day.",
    })
    reader = _make_reader([fixed])
    task = {
        "answer_intent": "personalized_recommendation",
        "answer_obligations": ["incremental_next_step"],
        "uncertainty_policy": "state_limits_when_context_is_sparse",
    }

    answer = reader.answer(
        question="Battery tips?",
        question_type="single-session-preference",
        question_date="2023/05/20 12:00",
        memory_context="Structured memories:\n- I bought a portable power bank for my phone.",
        answer_task=task,
    )

    assert answer == "Use your existing power bank to top up during the day."
    assert reader.last_debug is not None
    assert reader.last_debug["answer_task"] == task
    assert reader.last_debug["coverage"] == ["incremental_next_step"]
    # The prompt sent to the LLM must contain the rendered contract.
    sent = reader._client.chat.completions.sent_prompts[0]
    assert "Answer task contract:" in sent
    assert "answer_intent: personalized_recommendation" in sent


def test_parse_reader_structured_accepts_coverage_field() -> None:
    parsed = runner._parse_reader_structured(
        '{"relevant_evidence": ["e1"], "coverage": ["comparison", "negation_constraints"], "final_answer": "ok"}'
    )
    assert parsed is not None
    assert parsed["coverage"] == ["comparison", "negation_constraints"]
    assert parsed["final_answer"] == "ok"
    # Legacy reader output without coverage still parses.
    legacy = runner._parse_reader_structured('{"relevant_evidence": ["e1"], "final_answer": "ok"}')
    assert legacy is not None
    assert legacy["coverage"] == []


def test_extract_answer_task_returns_none_without_decision() -> None:
    assert runner._extract_answer_task({}) is None
    assert runner._extract_answer_task({"debug": {"planner": {"decision": {}}}}) is None
    # Legacy decision without contract fields -> None.
    legacy = runner._extract_answer_task({
        "debug": {"pre_reply_decision": {"memory_action": "none", "answer_intent": "direct_answer"}}
    })
    # direct_answer default with no obligations is treated as legacy no-op.
    assert legacy is None


def test_extract_answer_task_carries_contract_fields() -> None:
    task = runner._extract_answer_task({
        "debug": {
            "pre_reply_decision": {
                "answer_intent": "causal_explanation",
                "answer_focus": "explain sneezing causes",
                "answer_obligations": ["entities", "qualifiers"],
                "uncertainty_policy": "abstain_if_insufficient",
            }
        }
    })
    assert task is not None
    assert task["answer_intent"] == "causal_explanation"
    assert task["answer_obligations"] == ["entities", "qualifiers"]
    assert task["uncertainty_policy"] == "abstain_if_insufficient"


def test_causal_explanation_covers_multiple_supported_causes() -> None:
    """75f70248-style: cat shedding AND deep-cleaning dust both in evidence.

    Fixed evidence test: the contract instructs the reader to consider all
    supported obligations; here the fake reader follows it and returns a
    multi-cause answer.
    """
    fixed = json.dumps({
        "relevant_evidence": [
            "Your cat Luna sheds a lot in the living room.",
            "You deep-cleaned the living room last weekend.",
        ],
        "coverage": ["entities", "qualifiers"],
        "final_answer": "Yes — both your cat Luna's shedding and last weekend's deep cleaning could be contributing.",
    })
    reader = _make_reader([fixed])
    answer = reader.answer(
        question="I've been sneezing quite a bit. Do you think it might be my living room?",
        question_type="single-session-preference",
        question_date="2023/05/20 12:00",
        memory_context="Structured memories:\n- Your cat Luna sheds a lot in the living room.\n- You deep-cleaned the living room last weekend.",
        answer_task={"answer_intent": "causal_explanation", "answer_obligations": ["entities", "qualifiers"]},
    )
    assert "cat" in answer and "deep" in answer.lower()


def test_incremental_next_step_does_not_repeat_experiment() -> None:
    """38146c39-style: user already tried turbinado sugar -> next step, not repeat."""
    fixed = json.dumps({
        "relevant_evidence": ["I've been experimenting with sugars and found turbinado adds richer flavor."],
        "coverage": ["incremental_next_step"],
        "final_answer": "Since turbinado already worked for you, try pairing it with a touch of flaky salt next.",
    })
    reader = _make_reader([fixed])
    answer = reader.answer(
        question="My chocolate chip cookies need something extra. Any advice?",
        question_type="single-session-preference",
        question_date="2023/05/20 12:00",
        memory_context="Structured memories:\n- I've been experimenting with sugars and found turbinado adds richer flavor.",
        answer_task={"answer_intent": "personalized_recommendation", "answer_obligations": ["incremental_next_step"]},
    )
    assert "turbinado already worked" in answer
    assert "flaky salt" in answer


def test_generic_recommendation_does_not_force_recall() -> None:
    """generic_recommendation contract: the reader uses evidence as-is; it does
    not manufacture personal constraints. Fixed evidence test."""
    fixed = json.dumps({
        "relevant_evidence": [],
        "coverage": [],
        "final_answer": runner.UNKNOWN_ANSWER,
    })
    reader = _make_reader([fixed])
    answer = reader.answer(
        question="What's a good beginner camera?",
        question_type="single-session-preference",
        question_date="2023/05/20 12:00",
        memory_context="",
        answer_task={"answer_intent": "generic_recommendation", "answer_obligations": []},
    )
    # No memory context -> refusal is correct; the contract must not force coverage.
    assert answer == runner.UNKNOWN_ANSWER
    assert reader.last_debug is not None
    assert reader.last_debug["refusal"] is True


def test_direct_fact_not_rendered_as_preference_summary() -> None:
    fixed = json.dumps({
        "relevant_evidence": ["The project sync meeting is scheduled for Friday 10:00."],
        "coverage": ["entities"],
        "final_answer": "Your sync meeting is Friday at 10:00.",
    })
    reader = _make_reader([fixed])
    answer = reader.answer(
        question="When is my project sync meeting?",
        question_type="single-session-user",
        question_date="2023/05/20 12:00",
        memory_context="Structured memories:\n- The project sync meeting is scheduled for Friday 10:00.",
        answer_task={"answer_intent": "direct_fact", "answer_obligations": ["entities"]},
    )
    assert "Friday at 10:00" in answer
    # No preference-style padding in the fixed answer.
    assert "prefer" not in answer.lower() or "prefer" not in fixed
