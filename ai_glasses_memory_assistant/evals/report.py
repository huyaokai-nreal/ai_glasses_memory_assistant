from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# 同时输出 JSON 和 Markdown：JSON 给排障，Markdown 给人工快速阅读。
def write_reports(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": config,
        "summary": summary,
        "runs": runs,
    }
    json_path = output_dir / "eval-latest.json"
    md_path = output_dir / "eval-latest.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


# Markdown 报告突出总览、性能、失败样本和 target 缺口。
def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# AI 眼镜真实 LLM 离线评测报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 评测模式：{payload['config'].get('mode')}",
        f"- 重复次数：{payload['config'].get('repeat')}",
        f"- 场景数：{payload['config'].get('scenario_count')}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 总运行数 | {summary['total_runs']} |",
        f"| 总轮次 | {summary['total_turns']} |",
        f"| 通过轮次 | {summary['passed_turns']} |",
        f"| 失败轮次 | {summary['failed_turns']} |",
        f"| 通过率 | {_pct(summary['pass_rate'])} |",
        f"| Active 轮次 | {summary['active_turns']} |",
        f"| Active 失败轮次 | {summary['active_failed_turns']} |",
        f"| Active 通过率 | {_pct(summary['active_pass_rate'])} |",
        f"| Target 轮次 | {summary['target_turns']} |",
        f"| Target 缺口轮次 | {summary['target_failed_turns']} |",
        f"| Target 通过率 | {_pct(summary['target_pass_rate'])} |",
        f"| 异常率 | {_pct(summary['exception_rate'])} |",
        f"| 完成率 | {_pct(summary['completed_rate'])} |",
        f"| 回复事实命中率 | {_pct(summary['reply_fact_hit_rate'])} |",
        f"| 记忆召回命中率 | {_pct(summary['memory_recall_hit_rate'])} |",
        f"| 记忆写入命中率 | {_pct(summary['memory_write_hit_rate'])} |",
        f"| 多用户隔离通过率 | {_pct(summary['isolation_pass_rate'])} |",
        f"| 重复稳定率 | {_pct(summary['repeat_stability_rate'])} |",
        f"| API calls 总数 | {summary['api_calls_total']} |",
        f"| API calls 均值 | {summary['api_calls_mean']} |",
        "",
        "## 性能",
        "",
        "| 范围 | p50 | p95 | max |",
        "| --- | ---: | ---: | ---: |",
        _latency_row("total", summary["latency_seconds"]),
    ]
    for name, values in sorted(summary.get("stage_latency_seconds", {}).items()):
        lines.append(_latency_row(name, values))
    lines.extend(["", "## 待复盘样本", ""])
    failures = [
        turn
        for run in payload["runs"]
        for turn in run.get("turns", [])
        if turn.get("status", "active") != "target" and not turn.get("passed")
    ]
    if not failures:
        lines.append("本次没有自动硬判失败样本。")
    else:
        for turn in failures:
            lines.extend(_failure_block(turn))
    lines.extend(["", "## Target 场景缺口", ""])
    target_failures = [
        turn
        for run in payload["runs"]
        for turn in run.get("turns", [])
        if turn.get("status") == "target" and not turn.get("passed")
    ]
    if not target_failures:
        lines.append("本次没有 target 场景缺口。")
    else:
        for turn in target_failures:
            lines.extend(_failure_block(turn))
    lines.extend(["", "## 慢场景 Top 10", ""])
    slowest = summary.get("slowest_turns") or []
    if not slowest:
        lines.append("没有可用耗时数据。")
    else:
        lines.extend([
            "| 场景 | repeat | turn | seconds | passed | message |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ])
        for item in slowest:
            lines.append(
                f"| {item['scenario_id']} | {item['repeat_index']} | {item['turn_index']} | "
                f"{item['seconds']} | {item['passed']} | {_cell(item['message'])} |"
            )
    lines.extend([
        "",
        "## 说明",
        "",
        "- 本报告不使用 demo LLM 自评；自动硬判只基于金标准、召回/写入证据和回复文本。",
        "- `status=target` 的场景用于暴露产品能力缺口，不计入 `--strict` 的阻断失败。",
        "- `eval-latest.json` 保留完整 `debug`、`timing`、召回记忆和保存记忆，供人工复核。",
    ])
    return "\n".join(lines) + "\n"


# 失败块保留输入、回复、召回和新增记忆，便于定位是哪一层错了。
def _failure_block(turn: dict[str, Any]) -> list[str]:
    response = turn.get("response") or {}
    recalled = response.get("recalled_memories") or []
    saved = turn.get("new_memories") or []
    lines = [
        f"### {turn.get('scenario_id')} / repeat {turn.get('repeat_index')} / turn {turn.get('turn_index')}",
        "",
        f"- 输入：{turn.get('message')}",
        f"- 回复：{response.get('reply') if response else ''}",
        f"- 异常：{turn.get('exception') or ''}",
        f"- 失败原因：{'; '.join(turn.get('failures') or [])}",
        f"- 召回记忆：{'; '.join(str(item.get('content') or '') for item in recalled)}",
        f"- 新增记忆：{'; '.join(str(item.get('content') or '') for item in saved)}",
    ]
    if turn.get("traceback"):
        lines.extend(["", "```text", str(turn["traceback"]).strip(), "```"])
    lines.append("")
    return lines


def _latency_row(name: str, values: dict[str, Any]) -> str:
    return f"| {name} | {values.get('p50', 0)} | {values.get('p95', 0)} | {values.get('max', 0)} |"


def _pct(value: float) -> str:
    return f"{round(value * 100, 2)}%"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
