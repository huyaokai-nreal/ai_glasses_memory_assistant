from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
from ai_glasses_memory_assistant.app_home import APP_HOME_ENV
from ai_glasses_memory_assistant.env_loader import load_app_dotenv, restore_app_llm_env, snapshot_app_llm_env
from ai_glasses_memory_assistant.evals.longmemeval_adapter import (
    DEFAULT_ORACLE_PATH,
    LongMemEvalItem,
    answer_terms,
    load_longmemeval_items,
)
from ai_glasses_memory_assistant.evals.metrics import contains_any
from ai_glasses_memory_assistant.memory_store import EventMemoryStore, event_to_dict
from ai_glasses_memory_assistant.timeline_store import chunk_to_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LongMemEval through the AI glasses memory assistant.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_ORACLE_PATH)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--question-type", action="append", default=[])
    parser.add_argument("--include-abstention", action="store_true", default=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports") / "longmemeval")
    parser.add_argument("--background-wait", type=float, default=15.0)
    parser.add_argument("--history-mode", choices=("timeline", "chat"), default="timeline")
    parser.add_argument("--keep-homes", action="store_true")
    args = parser.parse_args(argv)

    load_app_dotenv()
    app_llm_env = snapshot_app_llm_env()
    items = load_longmemeval_items(
        args.dataset_path,
        limit=max(0, args.limit),
        question_types={str(item) for item in args.question_type if str(item).strip()} or None,
        include_abstention=bool(args.include_abstention),
    )
    if not items:
        raise SystemExit("No LongMemEval items matched the requested filters.")

    original_app_home = os.environ.get(APP_HOME_ENV)
    runs: list[dict[str, Any]] = []
    for item in items:
        restore_app_llm_env(app_llm_env)
        runs.append(
            run_longmemeval_item(
                item,
                background_wait=max(0.0, args.background_wait),
                history_mode=args.history_mode,
                keep_home=bool(args.keep_homes),
                original_app_home=original_app_home,
            )
        )
    if original_app_home is None:
        os.environ.pop(APP_HOME_ENV, None)
    else:
        os.environ[APP_HOME_ENV] = original_app_home

    summary = summarize_longmemeval_runs(runs)
    write_longmemeval_report(
        output_dir=args.report_dir,
        summary=summary,
        runs=runs,
        config={
            "dataset_path": str(args.dataset_path),
            "limit": args.limit,
            "question_types": args.question_type,
            "history_mode": args.history_mode,
            "background_wait": args.background_wait,
        },
    )
    print(f"Markdown report: {args.report_dir / 'eval-latest.md'}")
    print(f"JSON report: {args.report_dir / 'eval-latest.json'}")
    print(f"Summary: overall={summary['overall']['answer_hit_rate']}, total={summary['overall']['total']}")
    return 0


def run_longmemeval_item(
    item: LongMemEvalItem,
    *,
    background_wait: float,
    history_mode: str,
    keep_home: bool,
    original_app_home: str | None,
) -> dict[str, Any]:
    app_home = Path(tempfile.mkdtemp(prefix=f"glasses-longmemeval-{item.question_id}-"))
    os.environ[APP_HOME_ENV] = str(app_home)
    store = EventMemoryStore()
    service = GlassesChatService(memory_store=store)
    user_id = f"longmemeval-{item.question_id}"
    started = time.perf_counter()
    try:
        if history_mode == "chat":
            ingest_history_via_chat(service, item, user_id=user_id, background_wait=background_wait)
        else:
            ingest_history_to_timeline(service, item, user_id=user_id)
        response = service.chat(item.question, user_id=user_id, session_id=f"{item.question_id}-question")
        wait_for_response_jobs(service, response, user_id=user_id, timeout=background_wait)
        memories = [event_to_dict(memory) for memory in store.list_memories(user_id, limit=500)]
        timeline_chunks = service.timeline_store.search_chunks(user_id, item.question, limit=10)
        answer_hit = score_answer(response.get("reply", ""), item.answer, is_abstention=item.is_abstention)
        recall_hit = score_recall(response, memories, item)
        return {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "is_abstention": item.is_abstention,
            "question": item.question,
            "answer": item.answer,
            "reply": response.get("reply", ""),
            "answer_hit": answer_hit,
            "recall_hit": recall_hit,
            "session_count": len(item.sessions),
            "history_turn_count": sum(len(session.turns) for session in item.sessions),
            "memory_count": len(memories),
            "timeline_match_count": len(timeline_chunks),
            "api_calls": response.get("api_calls", 0),
            "measured_seconds": round(time.perf_counter() - started, 6),
            "response_debug": response.get("debug", {}),
            "recalled_memories": response.get("recalled_memories", []),
            "recalled_timeline_chunks": response.get("recalled_timeline_chunks", []),
            "sample_timeline_matches": [chunk_to_dict(chunk) for chunk in timeline_chunks[:3]],
            "app_home": str(app_home) if keep_home else "<temporary>",
            "passed": bool(answer_hit),
        }
    except Exception as exc:
        return {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "is_abstention": item.is_abstention,
            "question": item.question,
            "answer": item.answer,
            "reply": "",
            "answer_hit": False,
            "recall_hit": False,
            "exception": repr(exc),
            "measured_seconds": round(time.perf_counter() - started, 6),
            "app_home": str(app_home) if keep_home else "<temporary>",
            "passed": False,
        }
    finally:
        if original_app_home is None:
            os.environ.pop(APP_HOME_ENV, None)
        else:
            os.environ[APP_HOME_ENV] = original_app_home
        if not keep_home:
            shutil.rmtree(app_home, ignore_errors=True)


def ingest_history_to_timeline(service: GlassesChatService, item: LongMemEvalItem, *, user_id: str) -> None:
    for session in item.sessions:
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
    service: GlassesChatService,
    item: LongMemEvalItem,
    *,
    user_id: str,
    background_wait: float,
) -> None:
    for session in item.sessions:
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
    service: GlassesChatService,
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


def score_answer(reply: str, answer: str, *, is_abstention: bool) -> bool:
    if is_abstention:
        unknown_markers = ["不知道", "不清楚", "没有", "无法确定", "not sure", "don't know", "unknown"]
        return contains_any(reply, unknown_markers)
    terms = answer_terms(answer)
    if not terms:
        return False
    return contains_any(reply, terms)


def score_recall(response: dict[str, Any], memories: list[dict[str, Any]], item: LongMemEvalItem) -> bool:
    terms = answer_terms(item.answer)
    if not terms or item.is_abstention:
        return False
    recalled_text = "\n".join(
        [
            str(response.get("reply") or ""),
            *[str(memory.get("content") or "") for memory in response.get("recalled_memories") or []],
            *[str(chunk.get("text") or "") for chunk in response.get("recalled_timeline_chunks") or []],
            *[str(memory.get("content") or "") for memory in memories],
        ]
    )
    return contains_any(recalled_text, terms)


def summarize_longmemeval_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[str(run.get("question_type") or "unknown")].append(run)
        if run.get("is_abstention"):
            groups["abstention"].append(run)
    return {
        "overall": _summarize_group(runs),
        "by_question_type": {name: _summarize_group(items) for name, items in sorted(groups.items())},
        "failures": [
            {
                "question_id": run.get("question_id"),
                "question_type": run.get("question_type"),
                "question": run.get("question"),
                "answer": run.get("answer"),
                "reply": run.get("reply"),
                "exception": run.get("exception", ""),
            }
            for run in runs
            if not run.get("answer_hit")
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
    (output_dir / "eval-latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "eval-latest.md").write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# LongMemEval 端到端测评报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 数据集：{payload['config'].get('dataset_path')}",
        f"- 历史导入模式：{payload['config'].get('history_mode')}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 题目数 | {summary['overall']['total']} |",
        f"| 回答命中率 | {_pct(summary['overall']['answer_hit_rate'])} |",
        f"| 召回命中率 | {_pct(summary['overall']['recall_hit_rate'])} |",
        f"| 平均耗时 | {summary['overall']['mean_seconds']}s |",
        "",
        "## 分项",
        "",
        "| question_type | 题目数 | 回答命中率 | 召回命中率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, item in summary["by_question_type"].items():
        lines.append(f"| {name} | {item['total']} | {_pct(item['answer_hit_rate'])} | {_pct(item['recall_hit_rate'])} |")
    lines.extend(["", "## 失败样本", ""])
    failures = summary.get("failures") or []
    if not failures:
        lines.append("本次没有失败样本。")
    else:
        for item in failures[:20]:
            lines.extend([
                f"### {item.get('question_id')} / {item.get('question_type')}",
                "",
                f"- 问题：{item.get('question')}",
                f"- 标准答案：{item.get('answer')}",
                f"- 系统回答：{item.get('reply')}",
                f"- 异常：{item.get('exception') or ''}",
                "",
            ])
    lines.extend([
        "## 说明",
        "",
        "- 第一版硬判使用答案字符串归一化命中，不等于 LongMemEval 官方 GPT judge 分数。",
        "- `history-mode=timeline` 只把历史作为原文 timeline 导入，主要测长历史召回和回答；`history-mode=chat` 会更接近完整写入链路，但成本更高。",
        "- 原始 LongMemEval JSON 应放在 `data/benchmarks/longmemeval/`，并由 `.gitignore` 排除。",
    ])
    return "\n".join(lines) + "\n"


def _summarize_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    if total == 0:
        return {"total": 0, "answer_hit_rate": 0.0, "recall_hit_rate": 0.0, "mean_seconds": 0.0}
    return {
        "total": total,
        "answer_hit_rate": round(sum(1 for run in runs if run.get("answer_hit")) / total, 4),
        "recall_hit_rate": round(sum(1 for run in runs if run.get("recall_hit")) / total, 4),
        "mean_seconds": round(sum(float(run.get("measured_seconds") or 0.0) for run in runs) / total, 4),
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
        return None


def _pct(value: float) -> str:
    return f"{round(float(value) * 100, 2)}%"


if __name__ == "__main__":
    raise SystemExit(main())
