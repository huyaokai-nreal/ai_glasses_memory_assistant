#!/usr/bin/env python3
"""Replay every needs-improvement turn from a local Android usage snapshot.

The source snapshot is read-only. Replay uses copies of its databases in a temporary
app home, so any new raw turns, jobs, or audit records are discarded on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_LLM_ENV = (
    "AI_GLASSES_LLM_PROVIDER",
    "AI_GLASSES_LLM_MODEL",
    "AI_GLASSES_LLM_BASE_URL",
    "AI_GLASSES_LLM_API_KEY",
)

COMPLETE_SET_DEBUG_FIELDS = (
    "valid",
    "coverage_complete",
    "batch_count",
    "api_calls",
    "reader_status",
    "failure_stage",
    "error",
    "validation_errors",
    "execution_attempts",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Extracted usage snapshot directory")
    parser.add_argument("--select-only", action="store_true", help="Print selected feedback without calling a model")
    parser.add_argument("--replay", action="store_true", help="Replay selected turns against copied databases")
    parser.add_argument("--output", type=Path, help="Optional JSON result path; stdout is always used when omitted")
    args = parser.parse_args(argv)
    if args.select_only == args.replay:
        parser.error("choose exactly one of --select-only or --replay")
    return args


def selected_feedback(snapshot: Path) -> list[dict[str, Any]]:
    feedback_path = snapshot / "feedback_index.json"
    try:
        payload = json.loads(feedback_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"usage snapshot is missing {feedback_path.name}") from exc
    if not isinstance(payload, list):
        raise ValueError("feedback_index.json must contain a list")
    return [
        dict(item) for item in payload
        if isinstance(item, dict) and str(item.get("rating") or "") == "needs_improvement"
    ]


def verify_local_llm_environment() -> dict[str, str]:
    values = {name: str(os.getenv(name) or "").strip() for name in REQUIRED_LLM_ENV}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError("replay requires explicit local LLM configuration: " + ", ".join(missing))
    provider = values["AI_GLASSES_LLM_PROVIDER"].lower()
    base_url = values["AI_GLASSES_LLM_BASE_URL"].lower()
    if provider not in {"llama_cpp", "ollama"} or "deepseek" in base_url:
        raise RuntimeError("replay refuses cloud/DeepSeek configuration; set explicit local Qwen AI_GLASSES_LLM_* values")
    return values


def replay_complete_set_debug(response: dict[str, Any]) -> dict[str, Any]:
    """Export only operational diagnostics, never prompts or source text."""

    debug = dict(response.get("debug") or {})
    complete_set = dict(debug.get("complete_set_answer") or {})
    discussion = dict(response.get("discussion_recall") or {})
    discussion_complete_set = dict(discussion.get("complete_set_answer") or {})
    payload = {
        key: complete_set.get(key)
        for key in COMPLETE_SET_DEBUG_FIELDS
        if key in complete_set
    }
    if discussion_complete_set:
        payload["discussion_candidate_count"] = discussion_complete_set.get("candidate_count")
        payload["discussion_reader_status"] = discussion_complete_set.get("reader_status")
        payload["discussion_failure_stage"] = discussion_complete_set.get("failure_stage")
    return payload


@contextmanager
def copied_snapshot_databases(snapshot: Path) -> Iterator[tuple[Path, Path, Path]]:
    source_database = snapshot / "database"
    required = ("events.db", "timeline.db")
    missing = [name for name in required if not (source_database / name).is_file()]
    if missing:
        raise ValueError("usage snapshot is missing database/" + ", database/".join(missing))
    with tempfile.TemporaryDirectory(prefix="ai-glasses-snapshot-replay-") as tmpdir:
        root = Path(tmpdir)
        for name in required:
            shutil.copy2(source_database / name, root / name)
        app_home = root / "app-home"
        app_home.mkdir()
        yield root / "events.db", root / "timeline.db", app_home


@contextmanager
def temporary_app_home(app_home: Path) -> Iterator[None]:
    previous = os.environ.get("AI_GLASSES_HOME")
    os.environ["AI_GLASSES_HOME"] = str(app_home)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("AI_GLASSES_HOME", None)
        else:
            os.environ["AI_GLASSES_HOME"] = previous


def replay(snapshot: Path, feedback: list[dict[str, Any]]) -> dict[str, Any]:
    verify_local_llm_environment()
    from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
    from ai_glasses_memory_assistant.memory_store import EventMemoryStore
    from ai_glasses_memory_assistant.timeline_store import TimelineStore

    results: list[dict[str, Any]] = []
    with copied_snapshot_databases(snapshot) as (events_path, timeline_path, app_home), temporary_app_home(app_home):
        for item in feedback:
            # Recreate the service with the original turn timestamp as its
            # clock. This keeps relative questions such as "今天" anchored to
            # the snapshot turn without adding a production chat API parameter.
            reference_time = float(item.get("turn_created_at") or 0.0)
            service = GlassesChatService(
                memory_store=EventMemoryStore(db_path=events_path),
                timeline_store=TimelineStore(db_path=timeline_path),
                clock=(lambda fixed_time=reference_time: fixed_time) if reference_time > 0 else None,
            )
            try:
                response = service.chat(
                    str(item.get("user_message") or ""),
                    user_id=str(item.get("user_id") or ""),
                )
                discussion = dict(response.get("discussion_recall") or {})
                provenance = dict(discussion.get("evidence_provenance") or {})
                results.append({
                    "feedback_id": item.get("feedback_id"),
                    "turn_id": item.get("turn_id"),
                    "user_message": item.get("user_message"),
                    "reply": response.get("reply"),
                    "evidence_scope": provenance.get("evidence_scope"),
                    "discussion_relation_scope": (discussion.get("coverage") or {}).get("relation_scope"),
                    "coverage": discussion.get("coverage") or {},
                    "evidence_provenance": provenance,
                    "discussion_topic_ids": [topic.get("id") for topic in discussion.get("topics") or []],
                    "complete_set_debug": replay_complete_set_debug(response),
                })
            finally:
                service.close()
    return {"mode": "replay", "selected_count": len(feedback), "results": results}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = args.snapshot.expanduser().resolve()
    feedback = selected_feedback(snapshot)
    payload = (
        {"mode": "select_only", "selected_count": len(feedback), "feedback": feedback}
        if args.select_only else replay(snapshot, feedback)
    )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
