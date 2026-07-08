from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
from datetime import datetime
from types import MethodType
from pathlib import Path
from typing import Any

from ai_glasses_memory_assistant.app_home import APP_HOME_ENV
from ai_glasses_memory_assistant.agent_bridge import ChatSession, GlassesChatService
from ai_glasses_memory_assistant.env_loader import load_app_dotenv, restore_app_llm_env, snapshot_app_llm_env
from ai_glasses_memory_assistant.evals.metrics import evaluate_turn, summarize_runs
from ai_glasses_memory_assistant.evals.report import write_reports
from ai_glasses_memory_assistant.memory_store import EventMemoryStore, event_to_dict
from ai_glasses_memory_assistant.timeline_store import chunk_to_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS = REPO_ROOT / "evals" / "scenarios.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"


class EvalFailingMemoryStore(EventMemoryStore):
    """Minimal failing store for live evals that need background write failures."""

    def add_memory(self, *args, **kwargs):
        raise RuntimeError("simulated write failure")


# 离线 eval 入口：直接跑真实 service.chat，不启动 Web UI。
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline live-LLM evals for the AI glasses demo.")
    parser.add_argument("--mode", choices=("live",), default="live", help="Use the real demo LLM path.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--scenario-id", action="append", default=[], help="Run only matching scenario id; can be repeated.")
    parser.add_argument("--category", action="append", default=[], help="Run only matching scenario category; can be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of scenarios; 0 means all.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--background-wait", type=float, default=15.0)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any hard check fails.")
    parser.add_argument("--keep-homes", action="store_true", help="Keep temporary AI_GLASSES_HOME directories for debugging.")
    args = parser.parse_args(argv)

    # 每个场景会切换临时 app home，先加载原环境里的 Hermes provider 配置。
    original_hermes_home = _legacy_hermes_home()
    original_app_home_env = os.environ.get(APP_HOME_ENV)
    load_app_dotenv()
    app_llm_env = snapshot_app_llm_env()
    # 迁移期仍加载 Hermes .env，给 Hermes runtime provider 提供旧 provider key 兜底。
    _load_legacy_hermes_dotenv(original_hermes_home)
    restore_app_llm_env(app_llm_env)

    scenarios = load_scenarios(args.scenarios)
    scenarios = filter_scenarios(scenarios, scenario_ids=args.scenario_id, categories=args.category)
    if args.limit > 0:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        raise SystemExit("No eval scenarios matched the requested filters.")
    runs: list[dict[str, Any]] = []

    for scenario in scenarios:
        for repeat_index in range(1, max(1, args.repeat) + 1):
            runs.append(run_scenario(
                scenario=scenario,
                repeat_index=repeat_index,
                background_wait=args.background_wait,
                keep_home=args.keep_homes,
                original_app_home_env=original_app_home_env,
            ))

    summary = summarize_runs(runs)
    paths = write_reports(
        output_dir=args.report_dir,
        summary=summary,
        runs=runs,
        config={
            "mode": args.mode,
            "scenarios": str(args.scenarios),
            "scenario_count": len(scenarios),
            "scenario_ids": [scenario.get("id") for scenario in scenarios],
            "repeat": max(1, args.repeat),
            "background_wait": args.background_wait,
        },
    )
    print(f"Markdown report: {paths['markdown']}")
    print(f"JSON report: {paths['json']}")
    print(
        "Summary: "
        f"active_pass_rate={summary['active_pass_rate']}, "
        f"active_failed_turns={summary['active_failed_turns']}, "
        f"target_failed_turns={summary['target_failed_turns']}, "
        f"p95={summary['latency_seconds']['p95']}s"
    )
    return 1 if args.strict and summary["active_failed_turns"] else 0


def _legacy_hermes_home() -> Any | None:
    try:
        from hermes_constants import get_hermes_home
    except ImportError:
        return None
    return get_hermes_home()


def _load_legacy_hermes_dotenv(hermes_home: Any | None) -> None:
    if hermes_home is None:
        return
    try:
        from hermes_cli.env_loader import load_hermes_dotenv
    except ImportError:
        return
    load_hermes_dotenv(hermes_home=hermes_home)


# 场景文件是 jsonl；读取时顺便规范 active/target 状态。
def load_scenarios(path: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                scenario = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(scenario, dict) or not scenario.get("id"):
                raise ValueError(f"{path}:{line_number}: scenario must be an object with id")
            scenario["status"] = _scenario_status(scenario)
            scenarios.append(scenario)
    return scenarios


# category 和 scenario-id 过滤只影响要跑哪些场景，不改变场景内容。
def filter_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    scenario_ids: list[str],
    categories: list[str],
) -> list[dict[str, Any]]:
    wanted_ids = {str(item) for item in scenario_ids if str(item).strip()}
    wanted_categories = {str(item) for item in categories if str(item).strip()}
    return [
        scenario
        for scenario in scenarios
        if (not wanted_ids or str(scenario.get("id")) in wanted_ids)
        and (not wanted_categories or str(scenario.get("category")) in wanted_categories)
    ]


# 单个场景使用隔离数据库和固定 reference_time，保证多轮可复现。
def run_scenario(
    *,
    scenario: dict[str, Any],
    repeat_index: int,
    background_wait: float,
    keep_home: bool,
    original_app_home_env: str | None,
) -> dict[str, Any]:
    # 临时 AI_GLASSES_HOME 隔离记忆、audit 和 session，避免污染真实 demo 数据。
    app_home = Path(tempfile.mkdtemp(prefix=f"glasses-eval-{scenario['id']}-"))
    os.environ[APP_HOME_ENV] = str(app_home)
    reference_time = _timestamp(scenario.get("reference_time")) or time.time()
    store = _build_eval_memory_store(scenario=scenario)
    service = GlassesChatService(
        memory_store=store,
        clock=lambda: reference_time,
        timezone=str(scenario.get("timezone") or "Asia/Shanghai"),
    )
    _install_eval_pre_reply_decision_agent(service, scenario)
    run: dict[str, Any] = {
        "scenario_id": scenario.get("id"),
        "name": scenario.get("name"),
        "category": scenario.get("category"),
        "status": _scenario_status(scenario),
        "repeat_index": repeat_index,
        "app_home": str(app_home) if keep_home else "<temporary>",
        "hermes_home": str(app_home) if keep_home else "<temporary>",
        "turns": [],
    }
    try:
        default_user_id = str(scenario.get("user_id") or "local-user")
        _preload_memories(store, scenario, default_user_id=default_user_id)
        _preload_timeline(service, scenario, default_user_id=default_user_id)
        _preload_documents(service, scenario, default_user_id=default_user_id)
        for turn_index, turn in enumerate(scenario.get("turns") or [], start=1):
            turn_result = run_turn(
                service=service,
                store=store,
                scenario=scenario,
                turn=turn,
                repeat_index=repeat_index,
                turn_index=turn_index,
                background_wait=background_wait,
            )
            run["turns"].append(turn_result)
    finally:
        if original_app_home_env is None:
            os.environ.pop(APP_HOME_ENV, None)
        else:
            os.environ[APP_HOME_ENV] = original_app_home_env
        if not keep_home:
            shutil.rmtree(app_home, ignore_errors=True)
    run["passed"] = all(turn.get("passed") for turn in run["turns"])
    return run


def _build_eval_memory_store(*, scenario: dict[str, Any]) -> EventMemoryStore:
    mode = str(scenario.get("memory_store_mode") or "").strip()
    if mode == "failing_write":
        return EvalFailingMemoryStore()
    return EventMemoryStore()


# 单轮 eval 先记录执行前记忆，再比较本轮新增记忆和回复。
def run_turn(
    *,
    service: GlassesChatService,
    store: EventMemoryStore,
    scenario: dict[str, Any],
    turn: dict[str, Any],
    repeat_index: int,
    turn_index: int,
    background_wait: float,
) -> dict[str, Any]:
    user_id = str(turn.get("user_id") or scenario.get("user_id") or "local-user")
    before_ids = {memory.id for memory in store.list_memories(user_id, limit=500)}
    response: dict[str, Any] | None = None
    exception = ""
    exception_traceback = ""
    started = time.perf_counter()
    action = str(turn.get("action") or "chat")
    try:
        if action == "reminder_check":
            response = _run_reminder_check(service=service, turn=turn, scenario=scenario, user_id=user_id)
        elif action == "add_memory":
            response = _run_add_memory_action(store=store, turn=turn, scenario=scenario, user_id=user_id)
        elif action == "delete_memory":
            response = _run_delete_memory_action(service=service, store=store, turn=turn, user_id=user_id)
        elif action == "purge_memory":
            response = _run_purge_memory_action(service=service, store=store, turn=turn, user_id=user_id)
        elif action == "timeline_search":
            response = _run_timeline_search_action(service=service, turn=turn, user_id=user_id)
        elif action == "delete_timeline_chunks":
            response = _run_delete_timeline_chunks_action(service=service, turn=turn, user_id=user_id)
        elif action == "memory_search":
            response = _run_memory_search_action(store=store, turn=turn, user_id=user_id)
        elif action == "weekly_report":
            response = _run_weekly_report_action(service=service, turn=turn, scenario=scenario, user_id=user_id)
        elif action == "memory_import":
            response = _run_memory_import_action(service=service, turn=turn, scenario=scenario, user_id=user_id)
        elif action == "document_update":
            response = _run_document_update_action(service=service, turn=turn, user_id=user_id)
        elif action == "document_delete":
            response = _run_document_delete_action(service=service, turn=turn, user_id=user_id)
        elif action == "chat_expect_failure":
            response = _run_chat_expect_failure_action(service=service, turn=turn, scenario=scenario, user_id=user_id)
        elif action == "ambient_wake_query":
            response = _run_ambient_wake_query_action(service=service, turn=turn, scenario=scenario, user_id=user_id)
        elif action == "chat":
            response = service.chat(
                str(turn.get("message") or ""),
                user_id=user_id,
                session_id=turn.get("session_id") or scenario.get("session_id"),
                location=turn.get("location") if isinstance(turn.get("location"), dict) else None,
            )
        else:
            raise ValueError(f"unsupported eval turn action: {action}")
    except Exception as exc:
        exception = repr(exc)
        exception_traceback = traceback.format_exc()
    measured_seconds = round(time.perf_counter() - started, 6)
    expect = dict(turn.get("expect") or {})
    # 后台写入可能晚于回复返回；先等已知 job 到终态，再读取新增记忆。
    background_deadline = time.perf_counter() + max(0.0, background_wait)
    if response and not exception:
        terminal_memory_jobs: list[dict[str, Any]] = []
        timed_out_job_ids: list[str] = []
        for job_id in _memory_processing_job_ids(response):
            job = _wait_for_memory_job_terminal(
                service,
                user_id=user_id,
                job_id=job_id,
                timeout=_remaining_wait(background_deadline),
            )
            if isinstance(job, dict):
                terminal_memory_jobs.append(job)
            else:
                timed_out_job_ids.append(job_id)
        terminal_memory_jobs.extend(_wait_for_known_memory_jobs_terminal(
            service,
            user_id=user_id,
            timeout=_remaining_wait(background_deadline),
        ))
        if terminal_memory_jobs or timed_out_job_ids:
            debug = response.setdefault("debug", {})
            if isinstance(debug, dict):
                if terminal_memory_jobs:
                    debug["memory_jobs"] = terminal_memory_jobs
                if timed_out_job_ids:
                    debug["memory_jobs_timeout"] = timed_out_job_ids
    if expect.get("saved_contains") and not exception:
        _wait_for_saved_terms(
            store,
            user_id,
            before_ids,
            expect["saved_contains"],
            timeout=_remaining_wait(background_deadline),
        )
    expected_saved_count = expect.get("saved_count_min")
    if expected_saved_count is None:
        exact_saved_count = expect.get("saved_count")
        if isinstance(exact_saved_count, int) and exact_saved_count > 0:
            expected_saved_count = exact_saved_count
    if expected_saved_count and not exception:
        _wait_for_saved_count(
            store,
            user_id,
            before_ids,
            min_count=int(expected_saved_count),
            timeout=_remaining_wait(background_deadline),
        )
    all_memories = [event_to_dict(memory) for memory in store.list_memories(user_id, limit=500)]
    new_memories = [item for item in all_memories if item.get("id") not in before_ids]
    evaluation = evaluate_turn(
        response=response,
        exception=exception,
        expect=expect,
        new_memories=new_memories,
        all_memories=all_memories,
    )
    return {
        "scenario_id": scenario.get("id"),
        "category": scenario.get("category"),
        "status": _scenario_status(scenario),
        "action": action,
        "repeat_index": repeat_index,
        "turn_index": turn_index,
        "message": turn.get("message"),
        "user_id": user_id,
        "expect": expect,
        "response": response,
        "exception": exception,
        "traceback": exception_traceback,
        "measured_seconds": measured_seconds,
        "new_memories": new_memories,
        "all_memories": all_memories,
        **evaluation,
    }


# 手动提醒检查走真实 service；主动推送 runtime 仍不是当前 eval action 的职责。
def _run_reminder_check(
    *,
    service: GlassesChatService,
    turn: dict[str, Any],
    scenario: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    now = _timestamp(turn.get("now") or scenario.get("reference_time")) or time.time()
    result = service.check_reminders(user_id=user_id, now=now)
    reminders = [item for item in result.get("reminders") or [] if isinstance(item, dict)]
    return {
        "action": "reminder_check",
        "reply": "",
        "triggered": bool(reminders),
        "status": result.get("status", "ready"),
        "completed": True,
        "api_calls": 0,
        "recalled_memories": [
            {
                "id": item.get("memory_id", ""),
                "content": item.get("content", ""),
                "memory_type": "task",
                "evidence_ids": item.get("evidence_ids", []),
            }
            for item in reminders
        ],
        "saved_memories": [],
        "debug": {
            "reminder_check": {
                "status": result.get("status", "ready"),
                "reason": result.get("debug", {}).get("reason", "manual_check_only"),
                "now": now,
                "reminder_count": result.get("reminder_count", 0),
                "location": turn.get("location") if isinstance(turn.get("location"), dict) else None,
                "user_allowed": bool(turn.get("user_allowed")),
            },
            "timing": {"total_seconds": 0.0, "stages": []},
        },
    }


def _run_add_memory_action(
    *,
    store: EventMemoryStore,
    turn: dict[str, Any],
    scenario: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    evidence_ids = turn.get("evidence_ids") if isinstance(turn.get("evidence_ids"), list) else []
    if not evidence_ids and turn.get("evidence_from_memory_contains") is not None:
        source_memory_id = _resolve_eval_memory_id(store, user_id, turn.get("evidence_from_memory_contains"))
        source_memory = store.get_memory_any_status(user_id, source_memory_id) if source_memory_id else None
        evidence_ids = list(source_memory.evidence_ids) if source_memory is not None else []
    memory = store.add_memory(
        user_id,
        str(turn.get("content") or ""),
        kind=str(turn.get("kind") or "event"),
        memory_type=str(turn.get("memory_type") or "") or None,
        tags=turn.get("tags") if isinstance(turn.get("tags"), list) else ["eval-action"],
        source=str(turn.get("source") or "eval-action"),
        source_id=str(turn.get("source_id") or ""),
        ingestion_id=str(turn.get("ingestion_id") or scenario.get("id") or ""),
        evidence_ids=[str(item) for item in evidence_ids if str(item).strip()],
        status=str(turn.get("status") or "active"),
        superseded_by=str(turn.get("superseded_by") or ""),
        confidence=_optional_float(turn.get("confidence")),
    )
    return {
        "action": "add_memory",
        "reply": "",
        "api_calls": 0,
        "saved_memories": [event_to_dict(memory)],
        "recalled_memories": [],
        "recalled_timeline_chunks": [],
        "debug": {"eval_action": {"action": "add_memory", "memory_id": memory.id}},
    }


def _run_delete_memory_action(
    *,
    service: GlassesChatService,
    store: EventMemoryStore,
    turn: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    memory_id = _resolve_eval_memory_id(
        store,
        user_id,
        turn.get("memory_id") or turn.get("memory_content_contains"),
        status=str(turn.get("memory_status") or ""),
    )
    deleted = service.delete_memory(user_id=user_id, memory_id=memory_id)
    return {
        "action": "delete_memory",
        "reply": "",
        "api_calls": 0,
        "deleted": deleted,
        "recalled_memories": [],
        "recalled_timeline_chunks": [],
        "debug": {"eval_action": {"action": "delete_memory", "memory_id": memory_id, "deleted": deleted}},
    }


def _run_purge_memory_action(
    *,
    service: GlassesChatService,
    store: EventMemoryStore,
    turn: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    memory_id = _resolve_eval_memory_id(
        store,
        user_id,
        turn.get("memory_id") or turn.get("memory_content_contains"),
        status=str(turn.get("memory_status") or ""),
    )
    purge = service.purge_memory(user_id=user_id, memory_id=memory_id)
    return {
        "action": "purge_memory",
        "reply": "",
        "api_calls": 0,
        **purge,
        "recalled_memories": [],
        "recalled_timeline_chunks": [],
        "debug": {"eval_action": {"action": "purge_memory", "memory_id": memory_id, **purge}},
    }


def _run_timeline_search_action(
    *,
    service: GlassesChatService,
    turn: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    query = str(turn.get("query") or turn.get("message") or "")
    if bool(turn.get("include_deleted")):
        ids = [
            str(row["id"] or "")
            for row in service.timeline_store._conn.execute(
                """
                SELECT id FROM chunks
                WHERE user_id = ? AND text LIKE ?
                ORDER BY timestamp DESC, chunk_index DESC
                LIMIT ?
                """,
                (user_id, f"%{query}%", int(turn.get("limit") or 5)),
            ).fetchall()
        ]
        chunks = service.timeline_store.list_chunks_by_ids(user_id, ids, limit=int(turn.get("limit") or 5), include_deleted=True)
    else:
        chunks = service.timeline_store.search_chunks(user_id, query, limit=int(turn.get("limit") or 5))
    return {
        "action": "timeline_search",
        "reply": "",
        "api_calls": 0,
        "recalled_memories": [],
        "recalled_timeline_chunks": [chunk_to_dict(chunk) for chunk in chunks],
        "debug": {
            "eval_action": {
                "action": "timeline_search",
                "query": query,
                "include_deleted": bool(turn.get("include_deleted")),
                "chunk_count": len(chunks),
            }
        },
    }


def _run_delete_timeline_chunks_action(
    *,
    service: GlassesChatService,
    turn: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    explicit_ids = [str(item) for item in turn.get("chunk_ids", [])] if isinstance(turn.get("chunk_ids"), list) else []
    query = str(turn.get("query") or turn.get("message") or "")
    if explicit_ids:
        chunk_ids = explicit_ids
    else:
        chunks = service.timeline_store.search_chunks(user_id, query, limit=int(turn.get("limit") or 5))
        chunk_ids = [chunk.id for chunk in chunks]
    result = service.delete_timeline_chunks(
        user_id=user_id,
        chunk_ids=chunk_ids,
        purge=bool(turn.get("purge", False)),
    )
    return {
        "action": "delete_timeline_chunks",
        "reply": "",
        "api_calls": 0,
        "recalled_memories": [],
        "recalled_timeline_chunks": [],
        "debug": {
            "eval_action": {
                "action": "delete_timeline_chunks",
                "query": query,
                "chunk_ids": chunk_ids,
                **result,
            }
        },
    }


def _run_memory_search_action(
    *,
    store: EventMemoryStore,
    turn: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    query = str(turn.get("query") or turn.get("message") or "")
    result = store.search_with_ranking(user_id, query, limit=int(turn.get("limit") or 5))
    memories = [event_to_dict(memory) for memory in result.memories]
    return {
        "action": "memory_search",
        "reply": "\n".join(str(item.get("content") or "") for item in memories),
        "api_calls": 0,
        "recalled_memories": memories,
        "recalled_timeline_chunks": [],
        "debug": {
            "eval_action": {
                "action": "memory_search",
                "query": query,
                "memory_count": len(memories),
                "ranking": result.ranking,
            }
        },
    }


def _run_ambient_wake_query_action(
    *,
    service: GlassesChatService,
    turn: dict[str, Any],
    scenario: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    capture = service.start_capture(
        user_id=user_id,
        source=str(turn.get("source") or "ambient_audio_text"),
        context=str(turn.get("context") or "ambient wakeword eval"),
    )
    base_time = _timestamp(scenario.get("reference_time")) or time.time()
    for index, item in enumerate(turn.get("ambient_chunks") or []):
        if isinstance(item, dict):
            text = str(item.get("text") or "")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        else:
            text = str(item or "")
            metadata = {}
        service.append_capture_chunk(
            user_id=user_id,
            capture_id=capture["capture_id"],
            text=text,
            timestamp=base_time + index,
            metadata={
                "source_type": "ambient_audio_text",
                "audio_retention": "not_recorded_browser_asr_text_only",
                **metadata,
            },
        )
    response = service.chat(
        str(turn.get("message") or ""),
        user_id=user_id,
        session_id=turn.get("session_id") or scenario.get("session_id"),
        ambient_capture_id=capture["capture_id"],
        wake_session={
            "ambient_capture_id": capture["capture_id"],
            "wake_detected_at": base_time + max(0, len(turn.get("ambient_chunks") or []) - 1),
            "pre_wake_segment_ids": [
                str(chunk.get("chunk_id") or "")
                for chunk in service.timeline_store.get_capture(user_id, capture["capture_id"]).get("chunks", [])
            ],
            "post_wake_query_segment_ids": [],
            "wake_query_text": str(turn.get("message") or ""),
            "wake_mode": "button",
            "status": "consumed",
        },
    )
    response.setdefault("debug", {})["eval_action"] = {
        "action": "ambient_wake_query",
        "capture_id": capture["capture_id"],
        "ambient_chunk_count": len(turn.get("ambient_chunks") or []),
    }
    return response


def _run_weekly_report_action(
    *,
    service: GlassesChatService,
    turn: dict[str, Any],
    scenario: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    reference = _timestamp(turn.get("now") or scenario.get("reference_time")) or time.time()
    start_at = _timestamp(turn.get("start_at")) or (reference - 7 * 24 * 60 * 60)
    end_at = _timestamp(turn.get("end_at")) or reference
    report = service.weekly_report(user_id=user_id, start_at=start_at, end_at=end_at)
    return {
        "action": "weekly_report",
        "reply": str(report.get("draft") or ""),
        "completed": True,
        "api_calls": 0,
        "recalled_memories": [],
        "saved_memories": [],
        "debug": {
            "weekly_report": {
                "project_count": report.get("project_count", 0),
                "projects": report.get("projects", []),
            },
            "timing": {"total_seconds": 0.0, "stages": []},
        },
    }


def _run_memory_import_action(
    *,
    service: GlassesChatService,
    turn: dict[str, Any],
    scenario: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    result = service.import_memory_events(
        user_id=user_id,
        items=turn.get("items") if isinstance(turn.get("items"), list) else None,
        text=str(turn.get("text") or turn.get("message") or ""),
        source=str(turn.get("source") or scenario.get("source") or "manual_import"),
        context=str(turn.get("context") or scenario.get("context") or ""),
        confirm=bool(turn.get("confirm", False)),
        occurred_at=_timestamp(turn.get("occurred_at")),
    )
    return {
        "action": "memory_import",
        "reply": str(result.get("reply") or ""),
        "completed": True,
        "api_calls": 0,
        "recalled_memories": [],
        "recalled_timeline_chunks": [],
        "saved_memories": result.get("saved_memories", []),
        "debug": {
            "eval_action": {
                "action": "memory_import",
                "source": result.get("source"),
                "context": result.get("context"),
                "saved_count": result.get("saved_count", 0),
                "source_summary": result.get("source_summary", {}),
                "cleaning_trace": result.get("cleaning_trace", {}),
            },
            "timing": {"total_seconds": 0.0, "stages": []},
        },
    }


def _run_document_update_action(
    *,
    service: GlassesChatService,
    turn: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    document = _resolve_eval_document(service, user_id, turn)
    updated = None
    if document is not None:
        updated = service.memory_store.update_document(
            user_id,
            document.id,
            filename=str(turn.get("filename") or document.filename),
            title=str(turn.get("title") or document.title),
            summary=str(turn.get("summary") or document.summary),
            content=str(turn.get("text") or turn.get("content") or document.content),
        )
    return {
        "action": "document_update",
        "reply": "",
        "api_calls": 0,
        "recalled_memories": [],
        "recalled_timeline_chunks": [],
        "recalled_documents": [service._document_payload(updated)] if updated is not None else [],
        "debug": {
            "eval_action": {
                "action": "document_update",
                "selector": _document_selector_debug(turn),
                "document_id": document.id if document is not None else "",
                "updated": updated is not None,
                "filename": updated.filename if updated is not None else "",
                "title": updated.title if updated is not None else "",
            },
            "timing": {"total_seconds": 0.0, "stages": []},
        },
    }


def _run_document_delete_action(
    *,
    service: GlassesChatService,
    turn: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    document = _resolve_eval_document(service, user_id, turn)
    deleted = False
    if document is not None:
        deleted = service.memory_store.delete_document(user_id, document.id)
    return {
        "action": "document_delete",
        "reply": "",
        "api_calls": 0,
        "deleted": deleted,
        "recalled_memories": [],
        "recalled_timeline_chunks": [],
        "recalled_documents": [],
        "debug": {
            "eval_action": {
                "action": "document_delete",
                "selector": _document_selector_debug(turn),
                "document_id": document.id if document is not None else "",
                "deleted": deleted,
            },
            "timing": {"total_seconds": 0.0, "stages": []},
        },
    }


def _run_chat_expect_failure_action(
    *,
    service: GlassesChatService,
    turn: dict[str, Any],
    scenario: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    message = str(turn.get("message") or "")
    failure_mode = str(turn.get("failure_mode") or "").strip()
    original_list_events_between = getattr(service.memory_store, "list_events_between", None)
    original_new_session = getattr(service, "_new_session", None)
    original_memory_snapshot = getattr(service, "_memory_snapshot", None)
    expected_error_type = ""
    caught_exception = ""
    try:
        if failure_mode == "memory_retrieval_database_error":
            def _raise_database_error(*args, **kwargs):
                raise sqlite3.DatabaseError("database disk image is malformed")
            service.memory_store.list_events_between = _raise_database_error  # type: ignore[attr-defined]
            expected_error_type = "DatabaseError"
        elif failure_mode == "assistant_response_database_error":
            if original_new_session is None:
                raise ValueError("assistant_response_database_error requires _new_session")

            def _patch_agent_run_conversation(agent: Any) -> None:
                original_run_conversation = getattr(agent, "run_conversation")

                def _raise_database_error(message, system_message=None, conversation_history=None, persist_user_message=None):
                    if system_message:
                        return original_run_conversation(
                            message,
                            system_message=system_message,
                            conversation_history=conversation_history,
                            persist_user_message=persist_user_message,
                        )
                    raise sqlite3.DatabaseError("database disk image is malformed")

                agent.run_conversation = _raise_database_error  # type: ignore[assignment]

            for existing_session in getattr(service, "_sessions", {}).values():
                agent = getattr(existing_session, "agent", None)
                if agent is not None:
                    _patch_agent_run_conversation(agent)

            def _failing_new_session(self: GlassesChatService, *, user_id: str, session_id: str | None = None) -> ChatSession:
                session = original_new_session(user_id=user_id, session_id=session_id)
                _patch_agent_run_conversation(session.agent)
                return session

            service._new_session = MethodType(_failing_new_session, service)
            expected_error_type = "DatabaseError"
        elif failure_mode == "memory_snapshot_database_error":
            if original_memory_snapshot is None:
                raise ValueError("memory_snapshot_database_error requires _memory_snapshot")

            def _raise_memory_snapshot_error(self: GlassesChatService, target_user_id: str) -> dict[str, Any]:
                raise sqlite3.DatabaseError("database disk image is malformed")

            service._memory_snapshot = MethodType(_raise_memory_snapshot_error, service)
            expected_error_type = "DatabaseError"
        else:
            raise ValueError(f"unsupported eval failure mode: {failure_mode}")
        try:
            service.chat(
                message,
                user_id=user_id,
                session_id=turn.get("session_id") or scenario.get("session_id"),
                location=turn.get("location") if isinstance(turn.get("location"), dict) else None,
            )
        except Exception as exc:
            caught_exception = repr(exc)
        audit_records = service.read_audit_records(user_id=user_id, limit=int(turn.get("audit_limit") or 10))
        failed_record = next(
            (record for record in reversed(audit_records) if record.get("record_type") == "chat_turn_failed"),
            {},
        )
        return {
            "action": "chat_expect_failure",
            "reply": "",
            "api_calls": 0,
            "completed": True,
            "recalled_memories": [],
            "recalled_timeline_chunks": [],
            "saved_memories": [],
            "debug": {
                "eval_action": {
                    "action": "chat_expect_failure",
                    "failure_mode": failure_mode,
                    "expected_error_type": expected_error_type,
                    "caught_exception": caught_exception,
                    "audit_record": failed_record,
                },
                "timing": {"total_seconds": 0.0, "stages": []},
            },
        }
    finally:
        if original_list_events_between is not None:
            service.memory_store.list_events_between = original_list_events_between  # type: ignore[attr-defined]
        if original_new_session is not None:
            service._new_session = original_new_session  # type: ignore[assignment]
        if original_memory_snapshot is not None:
            service._memory_snapshot = original_memory_snapshot  # type: ignore[assignment]


def _resolve_eval_memory_id(store: EventMemoryStore, user_id: str, selector: Any, *, status: str = "") -> str:
    selector_text = str(selector or "").strip()
    if not selector_text:
        return ""
    if selector_text.startswith("mem_"):
        return selector_text
    terms = [str(item) for item in selector] if isinstance(selector, list) else [selector_text]
    status = str(status or "").strip()
    clauses = ["user_id = ?"]
    params: list[Any] = [user_id]
    if status:
        clauses.append("status = ?")
        params.append(status)
    rows = store._conn.execute(
        f"""
        SELECT id, content FROM memories
        WHERE {" AND ".join(clauses)}
        ORDER BY updated_at DESC, created_at DESC
        """,
        tuple(params),
    ).fetchall()
    for row in rows:
        content = str(row["content"] or "")
        if all(str(term) in content for term in terms):
            return str(row["id"] or "")
    return selector_text


def _resolve_eval_document(
    service: GlassesChatService,
    user_id: str,
    turn: dict[str, Any],
):
    document_id = str(turn.get("document_id") or "").strip()
    if document_id:
        return service.memory_store.get_document(user_id, document_id)
    selector = _document_selector_debug(turn)
    if selector:
        documents = service.memory_store.search_documents(user_id, selector, limit=10)
    else:
        documents = service.memory_store.list_documents(user_id, limit=10)
    if not documents:
        return None
    terms = [str(term) for term in turn.get("document_contains", [])] if isinstance(turn.get("document_contains"), list) else []
    if terms:
        for document in documents:
            haystack = "\n".join([document.filename, document.title, document.summary, document.content])
            if all(term in haystack for term in terms):
                return document
        return None
    return documents[0]


def _document_selector_debug(turn: dict[str, Any]) -> str:
    return str(
        turn.get("document_query")
        or turn.get("query")
        or turn.get("filename")
        or turn.get("title")
        or ""
    ).strip()


def _scenario_status(scenario: dict[str, Any]) -> str:
    status = str(scenario.get("status") or "active").strip().lower()
    return "target" if status == "target" else "active"


# 场景预置记忆直接写库，用于测试召回、提醒和隔离边界。
def _preload_memories(store: EventMemoryStore, scenario: dict[str, Any], *, default_user_id: str) -> None:
    for item in scenario.get("preload") or []:
        if not isinstance(item, dict):
            continue
        store.add_memory(
            str(item.get("user_id") or default_user_id),
            str(item.get("content") or ""),
            kind=str(item.get("kind") or "event"),
            tags=item.get("tags") if isinstance(item.get("tags"), list) else ["eval-preload"],
            source=str(item.get("source") or "eval-preload"),
            source_id=str(item.get("source_id") or ""),
            ingestion_id=str(item.get("ingestion_id") or ""),
            evidence_ids=item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else None,
            memory_type=str(item.get("memory_type") or "") or None,
            occurred_at=_timestamp(item.get("occurred_at") or item.get("occurred_at_iso")),
            start_at=_timestamp(item.get("start_at") or item.get("start_at_iso")),
            end_at=_timestamp(item.get("end_at") or item.get("end_at_iso")),
            time_granularity=str(item.get("time_granularity") or "unknown"),
            temporal_text=str(item.get("temporal_text") or ""),
            temporal_confidence=_optional_float(item.get("temporal_confidence")),
            privacy_level=str(item.get("privacy_level") or "normal"),
            status=str(item.get("status") or "active"),
            confidence=_optional_float(item.get("confidence")),
            superseded_by=str(item.get("superseded_by") or ""),
        )
        updates: dict[str, Any] = {}
        for field in ("created_at", "updated_at", "last_accessed_at"):
            timestamp = _timestamp(item.get(field) or item.get(f"{field}_iso"))
            if timestamp is not None:
                updates[field] = timestamp
        if "access_count" in item:
            updates["access_count"] = max(0, int(item.get("access_count") or 0))
        if "strength" in item:
            updates["strength"] = max(0.0, min(1.0, float(item.get("strength") or 0.0)))
        if updates:
            memory_id = _resolve_eval_memory_id(store, str(item.get("user_id") or default_user_id), item.get("content"))
            assignments = ", ".join(f"{field} = ?" for field in updates)
            store._conn.execute(
                f"UPDATE memories SET {assignments} WHERE id = ?",
                tuple([*updates.values(), memory_id]),
            )
            store._conn.commit()


# 预置 Markdown 文档必须走 service 的导入入口，确保 eval 覆盖真实 document-first 归档路径。
def _preload_documents(service: GlassesChatService, scenario: dict[str, Any], *, default_user_id: str) -> None:
    for item in scenario.get("preload_documents") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("content") or "").strip()
        if not text:
            continue
        service.import_memory_events(
            user_id=str(item.get("user_id") or default_user_id),
            text=text,
            source="markdown_upload",
            context=str(item.get("filename") or item.get("context") or "uploaded.md"),
        )


# 预置 timeline 原话时直接走 timeline_store，便于复现实话追问与 audit replay。
def _preload_timeline(service: GlassesChatService, scenario: dict[str, Any], *, default_user_id: str) -> None:
    for item in scenario.get("timeline_turns") or []:
        if not isinstance(item, dict):
            continue
        message = str(item.get("message") or item.get("text") or "").strip()
        if not message:
            continue
        service.timeline_store.add_turn(
            str(item.get("user_id") or default_user_id),
            message,
            created_at=_timestamp(item.get("created_at") or item.get("created_at_iso")),
        )


class _EvalPreReplyDecisionAgent:
    def __init__(
        self,
        pre_reply_payloads: dict[str, Any] | None = None,
        *,
        dedupe_payloads: dict[str, Any] | None = None,
        correction_target_payloads: dict[str, Any] | None = None,
        correction_payloads: dict[str, Any] | None = None,
        observation_update_payloads: dict[str, Any] | None = None,
        intent_payloads: dict[str, Any] | None = None,
        segment_payloads: dict[str, Any] | None = None,
        live_correction_target_agent: Any | None = None,
    ) -> None:
        self.pre_reply_payloads = pre_reply_payloads or {}
        self.dedupe_payloads = dedupe_payloads or {}
        self.correction_target_payloads = correction_target_payloads or {}
        self.correction_payloads = correction_payloads or {}
        self.observation_update_payloads = observation_update_payloads or {}
        self.intent_payloads = intent_payloads or {}
        self.segment_payloads = segment_payloads or {}
        self.live_correction_target_agent = live_correction_target_agent
        self.model = "eval-pre-reply-decision"
        self.provider = "eval"
        self.api_mode = "eval"
        self.enabled_toolsets = []

    def run_conversation(
        self,
        message: str,
        system_message: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        persist_user_message: str | None = None,
    ) -> dict[str, Any]:
        if system_message and "unified pre-reply decision classifier" in system_message:
            user_message = _pre_reply_user_message(message)
            payload = self.pre_reply_payloads.get(user_message) or self.pre_reply_payloads.get("*") or {}
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "memory dedupe classifier" in system_message:
            list_header = (
                "Current active structured memories:"
                if "Current active structured memories:" in message
                else "Current active profile preferences:"
            )
            payload = _lookup_eval_payload(
                self.dedupe_payloads,
                _json_content_between(
                    message,
                    start_header="New candidate:",
                    end_header=list_header,
                ),
            )
            payload = _with_resolved_id(
                payload,
                message,
                list_header=list_header,
                selector_key="memory_content_contains",
                output_key="memory_id",
                id_keys=("memory_id", "id"),
            )
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "memory correction classifier" in system_message:
            payload = _lookup_eval_payload(
                self.correction_payloads,
                _text_between(message, start_header="User message:", end_header="Return one JSON object only."),
            )
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "memory correction target resolver" in system_message:
            payload = _lookup_eval_payload(
                self.correction_target_payloads,
                _text_between(message, start_header="User message:", end_header="Replacement memory:"),
            )
            payload = _with_resolved_id(
                payload,
                message,
                list_header="Candidate active memories:",
                selector_key="memory_content_contains",
                output_key="memory_id",
                id_keys=("memory_id", "id"),
            )
            if payload:
                return {"final_response": json.dumps(payload, ensure_ascii=False)}
            if self.live_correction_target_agent is not None:
                return self.live_correction_target_agent.run_conversation(
                    message,
                    system_message=system_message,
                    conversation_history=conversation_history,
                    persist_user_message=persist_user_message,
                )
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and (
            "observation update classifier" in system_message
            or "observation memory update classifier" in system_message
        ):
            payload = _lookup_eval_payload(
                self.observation_update_payloads,
                _json_content_between(
                    message,
                    start_header="New observation candidate:",
                    end_header="Current active observations:",
                ),
            )
            payload = _with_resolved_id(
                payload,
                message,
                list_header="Current active observations:",
                selector_key="observation_content_contains",
                output_key="observation_id",
                id_keys=("observation_id", "memory_id", "id"),
            )
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "intent classifier" in system_message:
            payload = _lookup_eval_payload(self.intent_payloads, str(message or ""))
            if not payload:
                payload = {"memory_write_candidates": [], "confidence": 1.0}
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "segment semantic cleaner" in system_message:
            raw_span = str(message or "")
            if "raw_span:" in raw_span:
                raw_span = raw_span.rsplit("raw_span:", 1)[-1].strip()
            payload = _lookup_eval_payload(
                self.segment_payloads,
                raw_span,
            )
            if not payload:
                payload = {
                    "semantic_role": "memory_candidate",
                    "noise_level": "none",
                    "contains_filler": False,
                    "do_not_remember_scope": "",
                    "should_extract": True,
                    "candidate_span": "",
                    "candidate_hint": "",
                    "confidence": 0.7,
                    "reason": "eval default extract segment",
                }
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "temporal parser" in system_message:
            return {"final_response": json.dumps({"has_temporal_expression": False, "confidence": 0.0}, ensure_ascii=False)}
        reply = _eval_reply_from_memory_context(message)
        return {
            "final_response": reply,
            "messages": [{"role": "assistant", "content": reply}],
            "api_calls": 1,
            "completed": True,
        }


def _install_eval_pre_reply_decision_agent(service: GlassesChatService, scenario: dict[str, Any]) -> None:
    pre_reply_payloads = _normalized_payload_map(scenario.get("pre_reply_payloads"))
    dedupe_payloads = _normalized_payload_map(scenario.get("dedupe_payloads"))
    correction_target_payloads = _normalized_payload_map(scenario.get("correction_target_payloads"))
    correction_payloads = _normalized_payload_map(scenario.get("correction_payloads"))
    observation_update_payloads = _normalized_payload_map(scenario.get("observation_update_payloads"))
    intent_payloads = _normalized_payload_map(scenario.get("intent_payloads"))
    segment_payloads = _normalized_payload_map(scenario.get("segment_payloads"))
    live_correction_target_agent = None
    if str(scenario.get("correction_target_resolver") or "").strip().lower() == "live":
        live_session = service._new_session(
            user_id=str(scenario.get("user_id") or "local-user"),
            session_id=f"{scenario.get('session_id') or scenario.get('id') or 'eval'}-live-correction-target",
        )
        live_correction_target_agent = live_session.agent
    if not any((
        pre_reply_payloads,
        dedupe_payloads,
        correction_target_payloads,
        correction_payloads,
        observation_update_payloads,
        intent_payloads,
        segment_payloads,
        live_correction_target_agent is not None,
    )):
        return
    eval_agent = _EvalPreReplyDecisionAgent(
        pre_reply_payloads,
        dedupe_payloads=dedupe_payloads,
        correction_target_payloads=correction_target_payloads,
        correction_payloads=correction_payloads,
        observation_update_payloads=observation_update_payloads,
        intent_payloads=intent_payloads,
        segment_payloads=segment_payloads,
        live_correction_target_agent=live_correction_target_agent,
    )
    seed_session_id = str(scenario.get("session_id") or "")
    service._sessions[seed_session_id] = ChatSession(id=seed_session_id, agent=eval_agent)

    def _new_session(self: GlassesChatService, *, user_id: str, session_id: str | None = None) -> ChatSession:
        sid = session_id or "eval-pre-reply-decision-session"
        return ChatSession(id=sid, agent=eval_agent)

    service._new_session = MethodType(_new_session, service)


def _normalized_payload_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): dict(payload)
        for key, payload in value.items()
        if isinstance(payload, dict)
    }


def _lookup_eval_payload(payloads: dict[str, dict[str, Any]], text: str) -> dict[str, Any]:
    if not payloads:
        return {}
    text = str(text or "").strip()
    if text in payloads:
        return dict(payloads[text])
    normalized_text = _compact_lookup_text(text)
    for key, payload in payloads.items():
        normalized_key = _compact_lookup_text(key)
        if key == "*" or (normalized_key and (normalized_key in normalized_text or normalized_text in normalized_key)):
            return dict(payload)
    return {}


def _with_resolved_id(
    payload: dict[str, Any],
    prompt: str,
    *,
    list_header: str,
    selector_key: str,
    output_key: str,
    id_keys: tuple[str, ...],
) -> dict[str, Any]:
    if not payload or payload.get(output_key):
        return payload
    selector = payload.pop(selector_key, None)
    if selector is None:
        return payload
    selector_terms = [str(item) for item in selector] if isinstance(selector, list) else [str(selector)]
    candidates = _json_between(prompt, start_header=list_header, end_header="Return one JSON object only.")
    if not isinstance(candidates, list):
        return payload
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = str(candidate.get("content") or "")
        if not all(term in content for term in selector_terms):
            continue
        for id_key in id_keys:
            resolved = str(candidate.get(id_key) or "").strip()
            if resolved:
                payload[output_key] = resolved
                return payload
    return payload


def _json_content_between(prompt: str, *, start_header: str, end_header: str) -> str:
    raw = _text_between(prompt, start_header=start_header, end_header=end_header)
    payload = _json_between(prompt, start_header=start_header, end_header=end_header)
    if isinstance(payload, dict):
        return str(payload.get("content") or "")
    return raw


def _json_between(prompt: str, *, start_header: str, end_header: str) -> Any:
    text = _text_between(prompt, start_header=start_header, end_header=end_header)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _text_between(prompt: str, *, start_header: str, end_header: str) -> str:
    if start_header not in prompt:
        return ""
    text = prompt.split(start_header, 1)[1]
    if end_header in text:
        text = text.split(end_header, 1)[0]
    return text.strip()


def _compact_lookup_text(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def _pre_reply_user_message(prompt: str) -> str:
    marker = "User message:"
    if marker not in prompt:
        return prompt.strip()
    return prompt.rsplit(marker, 1)[1].strip()


def _eval_reply_from_memory_context(prompt: str) -> str:
    ambient_context = _text_between(
        prompt,
        start_header="Recent ambient audio transcript before wake word:",
        end_header="Recent raw user timeline topics:",
    )
    if ambient_context:
        lines = [
            line.strip()
            for line in ambient_context.splitlines()
            if re.match(r"^\d+\.\s+", line.strip())
        ]
        evidence = "；".join(lines[:3])
        if "unknown" in prompt.lower() or "不确定" in prompt:
            return "我能看到刚才的原话，但情绪判断不确定，所以我不会强行断言你的情绪。现场线索：" + evidence
        if "别记" in prompt or "不用记" in prompt:
            return "我会只把刚才这段当作当前语境来回答，不写成长期记忆。现场线索：" + evidence
        return "刚才这段听起来你有点烦躁，我先站在你这边说：让你不舒服的主要是这些原话：" + evidence
    direct_evidence = _memory_context_section_lines(prompt, "Direct structured memory evidence:")
    if direct_evidence:
        return "我查到这些近期活动：" + "；".join(direct_evidence[:3])
    compare_evidence = _document_compare_evidence(prompt)
    if compare_evidence:
        return compare_evidence
    document_evidence = _document_context_evidence(prompt)
    if document_evidence:
        return "我查到文档里写的是：" + document_evidence
    timeline_evidence = _memory_context_section_lines(prompt, "Raw user timeline chunks recalled for this query:")
    if timeline_evidence:
        return "我找到这些相关原话：" + "；".join(timeline_evidence[:3])
    return "主回复"


def _document_compare_evidence(prompt: str) -> str:
    context = _text_between(prompt, start_header="<document-context>", end_header="</document-context>")
    if not context or "Cross-document comparison candidates:" not in context:
        return ""
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[System note:") or line == "Cross-document comparison candidates:":
            continue
        if re.match(r"^\d+\.\s+title:\s+", line):
            if current:
                blocks.append(current)
            current = {"title": line.split("title:", 1)[-1].strip()}
            continue
        if current is None:
            continue
        if line.startswith("high_level_summary:"):
            current["summary"] = line.split("high_level_summary:", 1)[-1].strip()
        elif line.startswith("high_level_excerpt:"):
            current["excerpt"] = line.split("high_level_excerpt:", 1)[-1].strip()
    if current:
        blocks.append(current)
    if len(blocks) < 2:
        return ""
    parts = []
    for block in blocks[:2]:
        title = block.get("title", "").strip()
        summary = block.get("excerpt", "").strip() or block.get("summary", "").strip()
        if title and summary:
            parts.append(f"{title}更偏向{summary}")
        elif title:
            parts.append(title)
    if len(parts) < 2:
        return ""
    return "我看到这两份文档的高层差异是：" + "；".join(parts)


def _document_context_evidence(prompt: str) -> str:
    context = _text_between(prompt, start_header="<document-context>", end_header="</document-context>")
    if not context:
        return ""
    lines = []
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[System note:"):
            continue
        if line.startswith(("document_id:", "filename:", "title:", "uploaded_at:", "recall_mode:")):
            continue
        if line == "Archived user documents:" or line.startswith("summary:"):
            continue
        lines.append(line)
    return "；".join(lines[:3])


def _memory_context_section_lines(prompt: str, header: str) -> list[str]:
    if header not in prompt:
        return []
    lines = prompt.splitlines()
    collecting = False
    evidence: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line == header:
            collecting = True
            continue
        if not collecting:
            continue
        if not line:
            break
        if line.endswith(":") or line.startswith("<"):
            break
        cleaned = line
        if ". " in cleaned:
            prefix, rest = cleaned.split(". ", 1)
            if prefix.isdigit():
                cleaned = rest
        cleaned = cleaned.strip()
        if cleaned:
            evidence.append(cleaned)
    return evidence


# 轮询新增记忆内容，覆盖 reply-first 后台写入的异步完成窗口。
def _wait_for_saved_terms(
    store: EventMemoryStore,
    user_id: str,
    before_ids: set[str],
    terms: Any,
    *,
    timeout: float,
) -> None:
    expected_terms = [str(term) for term in terms] if isinstance(terms, list) else [str(terms)]
    deadline = time.perf_counter() + max(0.0, timeout)
    while time.perf_counter() < deadline:
        memories = [
            event_to_dict(memory)
            for memory in store.list_memories(user_id, limit=500)
            if memory.id not in before_ids
        ]
        text = "\n".join(str(memory.get("content") or "") for memory in memories)
        if all(term in text for term in expected_terms):
            return
        time.sleep(0.2)


def _wait_for_saved_count(
    store: EventMemoryStore,
    user_id: str,
    before_ids: set[str],
    *,
    min_count: int,
    timeout: float,
) -> None:
    if min_count <= 0:
        return
    deadline = time.perf_counter() + max(0.0, timeout)
    while time.perf_counter() < deadline:
        count = sum(
            1
            for memory in store.list_memories(user_id, limit=500)
            if memory.id not in before_ids
        )
        if count >= min_count:
            return
        time.sleep(0.2)


def _memory_processing_job_ids(response: dict[str, Any]) -> list[str]:
    debug = response.get("debug") if isinstance(response, dict) else {}
    processing = debug.get("memory_processing") if isinstance(debug, dict) else {}
    if not isinstance(processing, dict):
        return []
    job_ids: list[str] = []
    for key in ("job_id", "observation_job_id"):
        job_id = str(processing.get(key) or "").strip()
        if job_id and job_id not in job_ids:
            job_ids.append(job_id)
    return job_ids


def _wait_for_memory_job_terminal(
    service: GlassesChatService,
    *,
    user_id: str,
    job_id: str,
    timeout: float,
) -> dict[str, Any] | None:
    read_job = getattr(service, "read_memory_job", None)
    if not callable(read_job):
        return None
    terminal_statuses = {"saved", "rejected", "skipped", "failed"}
    deadline = time.perf_counter() + max(0.0, timeout)
    while time.perf_counter() < deadline:
        job = read_job(user_id=user_id, job_id=job_id)
        if isinstance(job, dict) and str(job.get("status") or "") in terminal_statuses:
            return dict(job)
        time.sleep(0.2)
    return None


def _wait_for_known_memory_jobs_terminal(
    service: GlassesChatService,
    *,
    user_id: str,
    timeout: float,
) -> list[dict[str, Any]]:
    jobs = getattr(service, "_memory_jobs", None)
    if not isinstance(jobs, dict):
        return []
    terminal_statuses = {"saved", "rejected", "skipped", "failed"}
    deadline = time.perf_counter() + max(0.0, timeout)
    while time.perf_counter() < deadline:
        pending_job_ids = [
            str(job.get("job_id") or "")
            for job in list(jobs.values())
            if isinstance(job, dict)
            and job.get("user_id") == user_id
            and str(job.get("status") or "") not in terminal_statuses
        ]
        if not any(pending_job_ids):
            return [
                dict(job)
                for job in list(jobs.values())
                if isinstance(job, dict)
                and job.get("user_id") == user_id
                and str(job.get("status") or "") in terminal_statuses
            ]
        time.sleep(0.2)
    return [
        dict(job)
        for job in list(jobs.values())
        if isinstance(job, dict)
        and job.get("user_id") == user_id
        and str(job.get("status") or "") in terminal_statuses
    ]


def _remaining_wait(deadline: float) -> float:
    return max(0.0, deadline - time.perf_counter())


def _timestamp(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return _optional_float(text)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
