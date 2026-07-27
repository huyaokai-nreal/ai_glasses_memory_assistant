from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, TextIO

from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
from ai_glasses_memory_assistant.app_home import APP_HOME_ENV
from ai_glasses_memory_assistant.env_loader import load_app_dotenv, restore_app_llm_env, snapshot_app_llm_env
from ai_glasses_memory_assistant.evals.longmemeval_adapter import (
    DEFAULT_ORACLE_PATH,
    LongMemEvalItem,
    LongMemEvalSession,
    answer_terms,
    load_longmemeval_items,
)
from ai_glasses_memory_assistant.evals.metrics import contains_any
from ai_glasses_memory_assistant.llm_runtime import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_FALLBACK_PROVIDER,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    LLM_PROVIDER_ENV,
)


DEFAULT_OUTPUT_ROOT_DIR = Path("reports") / "longmemeval"
DEFAULT_READER_MAX_CONTEXT_CHARS = 16000
DEFAULT_READER_MAX_TOKENS = 4096
DEFAULT_READER_TIMEOUT = 120
UNKNOWN_ANSWER = "I don't know based on the available memory."
_LONGMEMEVAL_WEEKDAY_RE = re.compile(r"\s+\([^)]*\)\s+")


@dataclass(frozen=True)
class ReaderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout: int = DEFAULT_READER_TIMEOUT
    max_tokens: int = DEFAULT_READER_MAX_TOKENS
    temperature: float = 0.0
    max_context_chars: int = DEFAULT_READER_MAX_CONTEXT_CHARS


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path
    brief_output: Path
    detail_output: Path


@dataclass
class MutableClock:
    value: float

    def __call__(self) -> float:
        return self.value


class Reader(Protocol):
    config: ReaderConfig

    def answer(
        self,
        *,
        question: str,
        question_type: str,
        question_date: str,
        memory_context: str,
    ) -> str:
        ...


class OpenAIReader:
    def __init__(self, config: ReaderConfig) -> None:
        from openai import OpenAI

        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=float(config.timeout),
        )

    def answer(
        self,
        *,
        question: str,
        question_type: str,
        question_date: str,
        memory_context: str,
    ) -> str:
        if not memory_context.strip():
            return UNKNOWN_ANSWER
        reader_context = truncate_text(memory_context, self.config.max_context_chars)
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": build_reader_prompt(
                    question=question,
                    question_type=question_type,
                    question_date=question_date,
                    memory_context=reader_context,
                ),
            }],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("Reader LLM returned no choices")
        message = getattr(choices[0], "message", None)
        content = _chat_message_text(getattr(message, "content", "") if message is not None else "")
        answer = extract_reader_final_answer(content)
        return answer or UNKNOWN_ANSWER


class ProgressReporter:
    def __init__(self, total: int, *, stream: TextIO = sys.stderr, width: int = 20) -> None:
        self.total = max(0, int(total))
        self.stream = stream
        self.width = max(10, int(width))
        self.started = time.perf_counter()
        self.completed = 0
        self.succeeded = 0
        self.failed = 0
        self._interactive = bool(getattr(stream, "isatty", lambda: False)())
        self._line_open = False

    def update(self, *, question_id: str, phase: str) -> None:
        line = self._render(question_id=question_id, phase=phase)
        if self._interactive:
            self.stream.write("\r" + line)
            self.stream.flush()
            self._line_open = True
        else:
            self.stream.write(line + "\n")
            self.stream.flush()

    def complete(self, *, question_id: str, succeeded: bool) -> None:
        self.completed += 1
        if succeeded:
            self.succeeded += 1
        else:
            self.failed += 1
        self.update(question_id=question_id, phase="complete" if succeeded else "error")

    def finish(self) -> None:
        if self._interactive and self._line_open:
            self.stream.write("\n")
            self.stream.flush()
            self._line_open = False

    def _render(self, *, question_id: str, phase: str) -> str:
        ratio = (self.completed / self.total) if self.total else 1.0
        filled = min(self.width, int(ratio * self.width))
        if filled >= self.width:
            bar = "=" * self.width
        else:
            bar = "=" * filled + ">" + "." * max(0, self.width - filled - 1)
        elapsed = max(0.0, time.perf_counter() - self.started)
        eta = None
        if self.completed > 0 and self.completed < self.total:
            eta = (elapsed / self.completed) * (self.total - self.completed)
        return (
            f"[{bar}] {self.completed}/{self.total} {ratio * 100:5.1f}%  "
            f"success={self.succeeded} failed={self.failed}  "
            f"phase={phase} question={question_id}  "
            f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}"
        )


ServiceFactory = Callable[[MutableClock], Any]
ProgressCallback = Callable[[str], None]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    load_app_dotenv()
    reader_config = resolve_reader_config(args)
    app_llm_env = snapshot_app_llm_env()
    items = select_items(args)
    paths = resolve_run_paths(args)
    prepare_run_paths(paths, overwrite=bool(args.overwrite))

    reader = OpenAIReader(reader_config)
    progress = ProgressReporter(len(items))
    original_app_home = os.environ.get(APP_HOME_ENV)
    try:
        runs = run_benchmark_items(
            items,
            reader=reader,
            paths=paths,
            progress=progress,
            app_llm_env=app_llm_env,
            original_app_home=original_app_home,
            background_wait=max(0.0, float(args.background_wait)),
            history_mode=str(args.history_mode),
            keep_homes=bool(args.keep_homes),
        )
    finally:
        progress.finish()
        _restore_app_home(original_app_home)

    summary = summarize_longmemeval_runs(runs)
    config = {
        "dataset_path": str(args.dataset_path),
        "limit": args.limit,
        "question_ids": args.question_id,
        "question_types": args.question_type,
        "history_mode": args.history_mode,
        "background_wait": args.background_wait,
        "reader_provider": reader_config.provider,
        "reader_model": reader_config.model,
        "reader_base_url": reader_config.base_url,
        "reader_max_context_chars": reader_config.max_context_chars,
        "reader_max_tokens": reader_config.max_tokens,
        "reader_temperature": reader_config.temperature,
    }
    write_json_output(paths.detail_output, runs)
    write_longmemeval_report(output_dir=paths.output_dir, summary=summary, runs=runs, config=config)

    failed = sum(1 for run in runs if run.get("status") != "success")
    console_summary = {
        "instances_requested": len(items),
        "instances_succeeded": len(items) - failed,
        "instances_failed": failed,
        "output": str(paths.brief_output),
        "detail_output": str(paths.detail_output),
        "json_report": str(paths.output_dir / "eval-latest.json"),
        "markdown_report": str(paths.output_dir / "eval-latest.md"),
        "reader_model": reader_config.model,
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LongMemEval through the AI glasses memory assistant.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_ORACLE_PATH)
    parser.add_argument("--limit", type=int, default=20, help="0 means all matching instances.")
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--question-type", action="append", default=[])
    parser.add_argument("--include-abstention", action="store_true", default=True)
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument("--output-dir", "--report-dir", dest="output_dir", type=Path)
    parser.add_argument("--detail-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--background-wait", type=float, default=15.0)
    parser.add_argument("--history-mode", choices=("import", "timeline", "chat"), default="import")
    parser.add_argument("--keep-homes", action="store_true")
    parser.add_argument("--reader-provider")
    parser.add_argument("--reader-model")
    parser.add_argument("--reader-base-url")
    parser.add_argument("--reader-api-key")
    parser.add_argument("--reader-timeout", type=int, default=DEFAULT_READER_TIMEOUT)
    parser.add_argument("--reader-max-tokens", type=int, default=DEFAULT_READER_MAX_TOKENS)
    parser.add_argument("--reader-temperature", type=float, default=0.0)
    parser.add_argument(
        "--reader-max-context-chars",
        type=int,
        default=DEFAULT_READER_MAX_CONTEXT_CHARS,
    )
    return parser


def resolve_reader_config(args: argparse.Namespace) -> ReaderConfig:
    provider = str(args.reader_provider or os.getenv(LLM_PROVIDER_ENV) or DEEPSEEK_FALLBACK_PROVIDER).strip()
    model = str(args.reader_model or os.getenv(LLM_MODEL_ENV) or "").strip()
    base_url = str(args.reader_base_url or os.getenv(LLM_BASE_URL_ENV) or "").strip().rstrip("/")
    api_key = str(
        args.reader_api_key
        or os.getenv(LLM_API_KEY_ENV)
        or (os.getenv(DEEPSEEK_API_KEY_ENV) if provider == DEEPSEEK_FALLBACK_PROVIDER else "")
        or ""
    ).strip()
    missing = []
    if not model:
        missing.append("--reader-model or AI_GLASSES_LLM_MODEL")
    if not base_url:
        missing.append("--reader-base-url or AI_GLASSES_LLM_BASE_URL")
    if not api_key:
        missing.append("--reader-api-key, AI_GLASSES_LLM_API_KEY, or DEEPSEEK_API_KEY")
    if missing:
        raise ValueError("LongMemEval reader requires: " + ", ".join(missing))
    return ReaderConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=max(1, int(args.reader_timeout)),
        max_tokens=max(1, int(args.reader_max_tokens)),
        temperature=float(args.reader_temperature),
        max_context_chars=max(0, int(args.reader_max_context_chars)),
    )


def select_items(args: argparse.Namespace) -> list[LongMemEvalItem]:
    items = load_longmemeval_items(
        args.dataset_path,
        limit=0,
        question_types={str(item) for item in args.question_type if str(item).strip()} or None,
        include_abstention=bool(args.include_abstention),
    )
    requested_ids = [str(item).strip() for item in args.question_id if str(item).strip()]
    if requested_ids:
        by_id = {item.question_id: item for item in items}
        missing = [question_id for question_id in requested_ids if question_id not in by_id]
        if missing:
            raise ValueError("Unknown LongMemEval question_id: " + ", ".join(missing))
        items = [by_id[question_id] for question_id in requested_ids]
    if int(args.limit) > 0 and not requested_ids:
        items = items[: int(args.limit)]
    if not items:
        raise ValueError("No LongMemEval items matched the requested filters.")
    return items


def resolve_run_paths(args: argparse.Namespace) -> RunPaths:
    input_stem = _safe_output_name(Path(args.dataset_path).stem)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.output_root_dir) / f"{input_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    brief_output = output_dir / f"{input_stem}_memory.jsonl"
    if args.detail_output:
        detail_output = Path(args.detail_output)
        if not detail_output.is_absolute():
            detail_output = output_dir / detail_output
    else:
        detail_output = output_dir / f"{input_stem}_memory.jsonl.details.json"
    return RunPaths(output_dir=output_dir, brief_output=brief_output, detail_output=detail_output)


def prepare_run_paths(paths: RunPaths, *, overwrite: bool) -> None:
    outputs = [
        paths.brief_output,
        paths.detail_output,
        paths.output_dir / "eval-latest.json",
        paths.output_dir / "eval-latest.md",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists. Pass --overwrite to replace:\n  "
            + "\n  ".join(str(path) for path in existing)
        )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.detail_output.parent.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.unlink()
    paths.brief_output.write_text("", encoding="utf-8")


def run_benchmark_items(
    items: list[LongMemEvalItem],
    *,
    reader: Reader,
    paths: RunPaths,
    progress: ProgressReporter,
    app_llm_env: dict[str, str],
    original_app_home: str | None,
    background_wait: float,
    history_mode: str,
    keep_homes: bool,
    service_factory: ServiceFactory | None = None,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for item in items:
        restore_app_llm_env(app_llm_env)
        phase_callback = lambda phase, item=item: progress.update(question_id=item.question_id, phase=phase)
        result = run_longmemeval_item(
            item,
            reader=reader,
            background_wait=background_wait,
            history_mode=history_mode,
            keep_home=keep_homes,
            original_app_home=original_app_home,
            progress_callback=phase_callback,
            service_factory=service_factory,
        )
        runs.append(result)
        succeeded = result.get("status") == "success"
        if succeeded:
            write_jsonl_row(paths.brief_output, {
                "question_id": result["question_id"],
                "hypothesis": result["hypothesis"],
            })
        progress.complete(question_id=item.question_id, succeeded=succeeded)
    return runs


def run_longmemeval_item(
    item: LongMemEvalItem,
    *,
    reader: Reader,
    background_wait: float,
    history_mode: str,
    keep_home: bool,
    original_app_home: str | None,
    progress_callback: ProgressCallback | None = None,
    service_factory: ServiceFactory | None = None,
) -> dict[str, Any]:
    app_home = Path(tempfile.mkdtemp(prefix=f"glasses-longmemeval-{item.question_id}-"))
    os.environ[APP_HOME_ENV] = str(app_home)
    question_timestamp = _timestamp(item.question_date) or time.time()
    clock = MutableClock(question_timestamp)
    factory = service_factory or _default_service_factory
    user_id = f"longmemeval-{item.question_id}"
    started = time.perf_counter()
    stage = "initialization"
    import_service: Any | None = None
    query_service: Any | None = None
    import_stats: dict[str, Any] = _empty_import_stats(item)
    try:
        stage = "import"
        _report_phase(progress_callback, "import")
        import_service = factory(clock)
        if history_mode == "import":
            import_stats = ingest_history_via_import(
                import_service,
                item,
                user_id=user_id,
                clock=clock,
                progress_callback=progress_callback,
            )
        elif history_mode == "chat":
            ingest_history_via_chat(
                import_service,
                item,
                user_id=user_id,
                background_wait=background_wait,
            )
            import_stats = _legacy_import_stats(item, mode="chat")
        else:
            ingest_history_to_timeline(import_service, item, user_id=user_id)
            import_stats = _legacy_import_stats(item, mode="timeline")

        stage = "import_close"
        _report_phase(progress_callback, "wait-import")
        _close_service(import_service, timeout=background_wait)
        import_service = None

        if int(import_stats.get("failed_import_count") or 0):
            return {
                "question_id": item.question_id,
                "question_type": item.question_type,
                "is_abstention": item.is_abstention,
                "question": item.question,
                "answer": item.answer,
                "question_date": item.question_date,
                "hypothesis": "",
                "reply": "",
                "answer_hit": False,
                "recall_hit": False,
                **import_stats,
                "recall_context": "",
                "recall_context_chars": 0,
                "reader_input_chars": 0,
                "exception": "conversation import contained failed fragments",
                "error": "conversation import contained failed fragments",
                "measured_seconds": round(time.perf_counter() - started, 6),
                "app_home": str(app_home) if keep_home else "<temporary>",
                "status": "error",
                "stage": "import",
                "passed": False,
            }

        stage = "recall"
        _report_phase(progress_callback, "recall")
        clock.value = question_timestamp
        query_service = factory(clock)
        response = query_service.chat(
            item.question,
            user_id=user_id,
            session_id=f"{item.question_id}-question",
            memory_writes_allowed=False,
        )
        memory_context = build_recall_context(response)
        recall_context_chars = len(memory_context)

        stage = "reader"
        _report_phase(progress_callback, "reader")
        hypothesis = reader.answer(
            question=item.question,
            question_type=item.question_type,
            question_date=item.question_date,
            memory_context=memory_context,
        )
        recalled_memories = list(response.get("recalled_memories") or [])
        recalled_timeline_chunks = list(response.get("recalled_timeline_chunks") or [])
        recalled_documents = list(response.get("recalled_documents") or [])
        answer_hit = score_answer(hypothesis, item.answer, is_abstention=item.is_abstention)
        recall_hit = score_recall_context(memory_context, item)
        native_api_calls = int(response.get("api_calls") or 0)
        return {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "is_abstention": item.is_abstention,
            "question": item.question,
            "answer": item.answer,
            "question_date": item.question_date,
            "hypothesis": hypothesis,
            "reply": hypothesis,
            "answer_hit": answer_hit,
            "recall_hit": recall_hit,
            **import_stats,
            "recalled_memory_count": len(recalled_memories),
            "recalled_timeline_count": len(recalled_timeline_chunks),
            "recalled_document_count": len(recalled_documents),
            "recall_context": memory_context,
            "recall_context_chars": recall_context_chars,
            "reader_input_chars": len(truncate_text(memory_context, reader.config.max_context_chars)),
            "api_calls": native_api_calls + (1 if memory_context.strip() else 0),
            "measured_seconds": round(time.perf_counter() - started, 6),
            "response_debug": response.get("debug", {}),
            "recalled_memories": recalled_memories,
            "recalled_timeline_chunks": recalled_timeline_chunks,
            "recalled_documents": recalled_documents,
            "app_home": str(app_home) if keep_home else "<temporary>",
            "status": "success",
            "stage": "complete",
            "passed": bool(answer_hit),
        }
    except Exception as exc:
        return {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "is_abstention": item.is_abstention,
            "question": item.question,
            "answer": item.answer,
            "question_date": item.question_date,
            "hypothesis": "",
            "reply": "",
            "answer_hit": False,
            "recall_hit": False,
            **import_stats,
            "recall_context": "",
            "recall_context_chars": 0,
            "reader_input_chars": 0,
            "exception": repr(exc),
            "error": str(exc),
            "measured_seconds": round(time.perf_counter() - started, 6),
            "app_home": str(app_home) if keep_home else "<temporary>",
            "status": "error",
            "stage": stage,
            "passed": False,
        }
    finally:
        _close_service(import_service, timeout=background_wait, suppress_errors=True)
        _close_service(query_service, timeout=background_wait, suppress_errors=True)
        _restore_app_home(original_app_home)
        if not keep_home:
            shutil.rmtree(app_home, ignore_errors=True)


def ingest_history_via_import(
    service: Any,
    item: LongMemEvalItem,
    *,
    user_id: str,
    clock: MutableClock,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    sessions = sorted_history_sessions(item.sessions)
    total_turns = sum(len(session.turns) for session in sessions)
    imported_turn_count = 0
    saved_memory_count = 0
    rejected_memory_count = 0
    pending_confirmation_count = 0
    skipped_empty_turn_count = 0
    failed_import_count = 0
    paired_turn_count = 0
    user_only_turn_count = 0
    assistant_only_turn_count = 0
    timeline_turn_count = 0
    timeline_chunk_count = 0
    import_failures: list[dict[str, Any]] = []
    processed_turn_count = 0
    for session_index, session in enumerate(sessions):
        session_timestamp = _timestamp(session.date)
        if session_timestamp is None:
            session_timestamp = (_timestamp(item.question_date) or clock.value) + session_index
        conversation_turns: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(session.turns):
            content = str(turn.content or "").strip()
            occurred_at = session_timestamp + turn_index
            clock.value = occurred_at
            role = str(turn.role or "unknown").strip().lower() or "unknown"
            source_id = f"{item.question_id}:{session.session_id}:{turn_index}"
            conversation_turns.append({
                "role": role,
                "content": content,
                "occurred_at": occurred_at,
                "source_id": source_id,
            })
        result = service.import_conversation_events(
            user_id=user_id,
            session_id=session.session_id,
            turns=conversation_turns,
            source="longmemeval_oracle",
            context=f"question_id={item.question_id};session_id={session.session_id}",
        )
        imported_turn_count += int(result.get("imported_turn_count", len(conversation_turns)) or 0)
        saved_memory_count += int(result.get("saved_count") or 0)
        rejected_memory_count += int(result.get("rejected_count") or 0)
        pending_confirmation_count += int(result.get("pending_confirmation_count") or 0)
        skipped_empty_turn_count += int(result.get("skipped_empty_turn_count") or 0)
        failed_import_count += int(result.get("failed_count") or 0)
        session_failures = list(result.get("failures") or [])
        if session_failures:
            import_failures.append({
                "session_id": str(session.session_id),
                "failed_count": int(result.get("failed_count") or len(session_failures)),
                "failures": session_failures,
            })
        paired_turn_count += int(result.get("paired_turn_count") or 0)
        user_only_turn_count += int(result.get("user_only_turn_count") or 0)
        assistant_only_turn_count += int(result.get("assistant_only_turn_count") or 0)
        timeline_turn_count += int(result.get("timeline_turn_count") or 0)
        timeline_chunk_count += int(result.get("timeline_chunk_count") or 0)
        processed_turn_count += len(conversation_turns)
        _report_phase(progress_callback, f"import {processed_turn_count}/{total_turns}")
    return {
        "history_mode": "import",
        "session_count": len(sessions),
        "history_turn_count": total_turns,
        "imported_turn_count": imported_turn_count,
        "saved_memory_count": saved_memory_count,
        "rejected_memory_count": rejected_memory_count,
        "pending_confirmation_count": pending_confirmation_count,
        "skipped_empty_turn_count": skipped_empty_turn_count,
        "failed_import_count": failed_import_count,
        "paired_turn_count": paired_turn_count,
        "user_only_turn_count": user_only_turn_count,
        "assistant_only_turn_count": assistant_only_turn_count,
        "timeline_turn_count": timeline_turn_count,
        "timeline_chunk_count": timeline_chunk_count,
        "import_failures": import_failures,
    }


def ingest_history_to_timeline(service: Any, item: LongMemEvalItem, *, user_id: str) -> None:
    for session in sorted_history_sessions(item.sessions):
        text = "\n".join(f"{turn.role}: {turn.content}" for turn in session.turns)
        if not text.strip():
            continue
        service.timeline_store.add_turn(
            user_id,
            text,
            source="longmemeval_history",
            interaction_id=session.session_id,
            created_at=_timestamp(session.date),
        )


def ingest_history_via_chat(
    service: Any,
    item: LongMemEvalItem,
    *,
    user_id: str,
    background_wait: float,
) -> None:
    for session in sorted_history_sessions(item.sessions):
        for turn in session.turns:
            if turn.role != "user":
                continue
            response = service.chat(
                turn.content,
                user_id=user_id,
                session_id=f"{item.question_id}-{session.session_id}",
            )
            wait_for_response_jobs(service, response, user_id=user_id, timeout=background_wait)


def wait_for_response_jobs(
    service: Any,
    response: dict[str, Any],
    *,
    user_id: str,
    timeout: float,
) -> None:
    job_ids = _memory_processing_job_ids(response)
    deadline = time.perf_counter() + max(0.0, timeout)
    for job_id in job_ids:
        while time.perf_counter() < deadline:
            job = service.read_memory_job(user_id=user_id, job_id=job_id)
            if not isinstance(job, dict):
                break
            if str(job.get("status") or "") in {"saved", "rejected", "failed", "not_needed"}:
                break
            time.sleep(0.05)


def sorted_history_sessions(sessions: list[LongMemEvalSession]) -> list[LongMemEvalSession]:
    indexed = list(enumerate(sessions))
    indexed.sort(key=lambda value: (
        _timestamp(value[1].date) is None,
        _timestamp(value[1].date) or float(value[0]),
        value[0],
    ))
    return [session for _index, session in indexed]


def build_recall_context(response: dict[str, Any]) -> str:
    sections: list[str] = []
    memories = _format_memory_evidence(response.get("recalled_memories"))
    timeline_chunks = _format_timeline_evidence(response.get("recalled_timeline_chunks"))
    documents = _unique_document_texts(response.get("recalled_documents"))
    if memories:
        sections.append("Structured memories:\n" + "\n".join(f"- {text}" for text in memories))
    if timeline_chunks:
        sections.append("Timeline evidence:\n" + "\n".join(f"- {text}" for text in timeline_chunks))
    if documents:
        sections.append("Document evidence:\n" + "\n".join(f"- {text}" for text in documents))
    return "\n\n".join(sections)


def build_reader_prompt(
    *,
    question: str,
    question_type: str,
    question_date: str,
    memory_context: str,
) -> str:
    return (
        "You are answering a LongMemEval benchmark question using retrieved memory only.\n"
        "Follow these rules strictly:\n"
        "1. Use only the memory context below. Do not invent details.\n"
        "2. First identify the 1-5 most relevant memory lines or snippets.\n"
        "3. Ignore irrelevant background memories once you have found the relevant evidence.\n"
        "4. If the answer text itself appears in memory, return it instead of abstaining.\n"
        "5. For counting, totaling, or comparison questions, gather all relevant items before answering.\n"
        "6. For temporal or knowledge-update questions, compare the event_time or recorded_at labels before the question date.\n"
        "7. If the memory truly lacks the answer, set final_answer to exactly: "
        f'"{UNKNOWN_ANSWER}"\n'
        "8. Keep the final answer concise and direct.\n\n"
        f"Question type: {question_type}\n"
        f"Question date: {question_date}\n"
        f"Question: {question}\n\n"
        "Memory context:\n"
        f"{memory_context or '[empty]'}\n\n"
        "Return valid JSON only, with no markdown or extra text. Use exactly this schema:\n"
        "{\n"
        '  "relevant_evidence": ["<most relevant memory snippet>", "<optional>"],\n'
        f'  "final_answer": "<final answer or exactly {UNKNOWN_ANSWER}>"\n'
        "}\n"
    )


def extract_reader_final_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    candidates = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        answer = parsed.get("final_answer", parsed.get("answer"))
        if answer is not None and str(answer).strip():
            return str(answer).strip()
    return ""


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 64)
    return text[:keep] + "\n\n[truncated for reader]\n"


def score_answer(reply: str, answer: str, *, is_abstention: bool) -> bool:
    if is_abstention:
        unknown_markers = ["不知道", "不清楚", "没有", "无法确定", "not sure", "don't know", "unknown"]
        return contains_any(reply, unknown_markers)
    terms = answer_terms(answer)
    return bool(terms and contains_any(reply, terms))


def score_recall_context(memory_context: str, item: LongMemEvalItem) -> bool:
    terms = answer_terms(item.answer)
    return bool(terms and not item.is_abstention and contains_any(memory_context, terms))


def score_recall(response: dict[str, Any], memories: list[dict[str, Any]], item: LongMemEvalItem) -> bool:
    del memories
    return score_recall_context(build_recall_context(response), item)


def summarize_longmemeval_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[str(run.get("question_type") or "unknown")].append(run)
        if run.get("is_abstention"):
            groups["abstention"].append(run)
    return {
        "overall": _summarize_group(runs),
        "by_question_type": {name: _summarize_group(items) for name, items in sorted(groups.items())},
        "recall_context": _summarize_recall_chars(runs),
        "failures": [
            {
                "question_id": run.get("question_id"),
                "question_type": run.get("question_type"),
                "question": run.get("question"),
                "answer": run.get("answer"),
                "reply": run.get("reply"),
                "stage": run.get("stage", ""),
                "exception": run.get("exception", ""),
            }
            for run in runs
            if run.get("status") != "success"
        ][:50],
    }


def write_longmemeval_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": config,
        "summary": summary,
        "runs": runs,
    }
    (output_dir / "eval-latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "eval-latest.md").write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    recall_summary = summary.get("recall_context") or {}
    lines = [
        "# LongMemEval 端到端测评报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 数据集：{payload['config'].get('dataset_path')}",
        f"- 历史导入模式：{payload['config'].get('history_mode')}",
        f"- Reader：{payload['config'].get('reader_model')}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 题目数 | {summary['overall']['total']} |",
        f"| 成功数 | {summary['overall']['succeeded']} |",
        f"| 失败数 | {summary['overall']['failed']} |",
        f"| 回答命中率 | {_pct(summary['overall']['answer_hit_rate'])} |",
        f"| 召回命中率 | {_pct(summary['overall']['recall_hit_rate'])} |",
        f"| 平均耗时 | {summary['overall']['mean_seconds']}s |",
        f"| 总召回字符数 | {recall_summary.get('total_chars', 0)} |",
        f"| 平均召回字符数 | {recall_summary.get('mean_chars', 0)} |",
        "",
        "## 分项",
        "",
        "| question_type | 题目数 | 回答命中率 | 召回命中率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, item in summary["by_question_type"].items():
        lines.append(
            f"| {name} | {item['total']} | {_pct(item['answer_hit_rate'])} | {_pct(item['recall_hit_rate'])} |"
        )
    lines.extend(["", "## 失败样本", ""])
    failures = summary.get("failures") or []
    if not failures:
        lines.append("本次没有运行失败样本。")
    else:
        for item in failures[:20]:
            lines.extend([
                f"### {item.get('question_id')} / {item.get('question_type')}",
                "",
                f"- 问题：{item.get('question')}",
                f"- 标准答案：{item.get('answer')}",
                f"- 失败阶段：{item.get('stage') or ''}",
                f"- 异常：{item.get('exception') or ''}",
                "",
            ])
    lines.extend([
        "## 说明",
        "",
        "- `*_memory.jsonl` 只包含 `question_id` 和 `hypothesis`，用于交给统一 evaluator。",
        "- `*.details.json` 保存完整召回文本和 Reader 截断前的 `recall_context_chars`。",
        "- 本报告的字符串命中率是本地诊断指标，不等于 LongMemEval 官方 GPT judge 分数。",
        "- `history-mode=import` 逐 turn 走本系统导入门控；timeline/chat 仅保留为兼容诊断模式。",
    ])
    return "\n".join(lines) + "\n"


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _summarize_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    succeeded = sum(1 for run in runs if run.get("status") == "success")
    if total == 0:
        return {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "answer_hit_rate": 0.0,
            "recall_hit_rate": 0.0,
            "mean_seconds": 0.0,
        }
    return {
        "total": total,
        "succeeded": succeeded,
        "failed": total - succeeded,
        "answer_hit_rate": round(sum(1 for run in runs if run.get("answer_hit")) / total, 4),
        "recall_hit_rate": round(sum(1 for run in runs if run.get("recall_hit")) / total, 4),
        "mean_seconds": round(sum(float(run.get("measured_seconds") or 0.0) for run in runs) / total, 4),
    }


def _summarize_recall_chars(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(run.get("recall_context_chars") or 0) for run in runs if run.get("status") == "success"]
    if not values:
        return {"total_chars": 0, "mean_chars": 0.0, "min_chars": 0, "max_chars": 0, "empty_count": 0}
    return {
        "total_chars": sum(values),
        "mean_chars": round(sum(values) / len(values), 2),
        "min_chars": min(values),
        "max_chars": max(values),
        "empty_count": sum(1 for value in values if value == 0),
    }


def _memory_processing_job_ids(response: dict[str, Any]) -> list[str]:
    debug = response.get("debug") if isinstance(response.get("debug"), dict) else {}
    memory_processing = debug.get("memory_processing") if isinstance(debug.get("memory_processing"), dict) else {}
    ids = []
    for key in ("job_id", "memory_job_id"):
        value = memory_processing.get(key)
        if value:
            ids.append(str(value))
    return ids


def _timestamp(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    normalized = _LONGMEMEVAL_WEEKDAY_RE.sub(" ", text)
    try:
        return datetime.strptime(normalized, "%Y/%m/%d %H:%M").timestamp()
    except ValueError:
        return None


def _pct(value: float) -> str:
    return f"{round(float(value) * 100, 2)}%"


def _safe_output_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(value or "").strip()
    ).strip("._")
    return cleaned or "longmemeval"


def _chat_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, dict):
                    text = text.get("value")
                if text is not None:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(value).strip()


def _unique_texts(value: Any, *, key: str) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get(key) or "").strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def _format_memory_evidence(value: Any) -> list[str]:
    formatted: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        event_time = _format_context_timestamp(item.get("start_at") or item.get("occurred_at"))
        granularity = str(item.get("time_granularity") or "").strip()
        key = (content, event_time, granularity)
        if key in seen:
            continue
        seen.add(key)
        labels = ["source=structured_memory"]
        if event_time:
            labels.append(f"event_time={event_time}")
        if granularity:
            labels.append(f"granularity={granularity}")
        formatted.append(f"[{'; '.join(labels)}] {content}")
    return formatted


def _format_timeline_evidence(value: Any) -> list[str]:
    formatted: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        recorded_at = _format_context_timestamp(item.get("timestamp"))
        key = (text, recorded_at)
        if key in seen:
            continue
        seen.add(key)
        labels = ["source=timeline"]
        if recorded_at:
            labels.append(f"recorded_at={recorded_at}")
        formatted.append(f"[{'; '.join(labels)}] {text}")
    return formatted


def _format_context_timestamp(value: Any) -> str:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return ""
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="minutes")


def _unique_document_texts(value: Any) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("filename") or "").strip()
        body = str(item.get("summary") or item.get("content") or "").strip()
        text = ": ".join(part for part in (title, body) if part)
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def _empty_import_stats(item: LongMemEvalItem) -> dict[str, Any]:
    return {
        "history_mode": "import",
        "session_count": len(item.sessions),
        "history_turn_count": sum(len(session.turns) for session in item.sessions),
        "imported_turn_count": 0,
        "saved_memory_count": 0,
        "rejected_memory_count": 0,
        "pending_confirmation_count": 0,
        "skipped_empty_turn_count": 0,
        "failed_import_count": 0,
        "paired_turn_count": 0,
        "user_only_turn_count": 0,
        "assistant_only_turn_count": 0,
        "timeline_turn_count": 0,
        "timeline_chunk_count": 0,
        "import_failures": [],
    }


def _legacy_import_stats(item: LongMemEvalItem, *, mode: str) -> dict[str, Any]:
    stats = _empty_import_stats(item)
    stats["history_mode"] = mode
    stats["imported_turn_count"] = stats["history_turn_count"]
    return stats


def _default_service_factory(clock: MutableClock) -> GlassesChatService:
    return GlassesChatService(clock=clock)


def _close_service(
    service: Any | None,
    *,
    timeout: float,
    suppress_errors: bool = False,
) -> None:
    if service is None:
        return
    close = getattr(service, "close", None)
    if not callable(close):
        return
    try:
        close(timeout=max(0.0, timeout))
    except TypeError:
        close()
    except Exception:
        if not suppress_errors:
            raise


def _restore_app_home(original_app_home: str | None) -> None:
    if original_app_home is None:
        os.environ.pop(APP_HOME_ENV, None)
    else:
        os.environ[APP_HOME_ENV] = original_app_home


def _report_phase(callback: ProgressCallback | None, phase: str) -> None:
    if callback is not None:
        callback(phase)


if __name__ == "__main__":
    raise SystemExit(main())
