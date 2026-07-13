from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .memory_store import MemoryEvent


def format_weekly_report(projects: list[dict[str, Any]]) -> str:
    if not projects:
        return "本周还没有可汇总的项目记忆。"
    lines = ["本周项目进展草稿："]
    for project in projects:
        lines.append(f"\n## {project['project']}")
        if project["decisions"]:
            lines.append("会议结论/决策：" + "；".join(project["decisions"]))
        if project["tasks"]:
            lines.append("待办/计划：" + "；".join(project["tasks"]))
        if project.get("completed_tasks"):
            lines.append("已完成任务：" + "；".join(project["completed_tasks"]))
        if project.get("cancelled_tasks"):
            lines.append("已取消任务：" + "；".join(project["cancelled_tasks"]))
        if project["risks"]:
            lines.append("风险/卡点：" + "；".join(project["risks"]))
        if project["completed"]:
            lines.append("完成/进展：" + "；".join(project["completed"]))
        if project.get("document_summaries"):
            lines.append("文档/背景：" + "；".join(project["document_summaries"]))
        if project["evidence_ids"]:
            lines.append("依据：" + "，".join(project["evidence_ids"][:6]))
    return "\n".join(lines)


def attention_items_reply(
    memories: list[MemoryEvent],
    *,
    format_event: Callable[[MemoryEvent], str],
) -> str:
    if not memories:
        return "我没有查到接下来特别需要注意的待办或风险。"
    tasks = [memory for memory in memories if memory.memory_type == "task"]
    risks = [memory for memory in memories if memory.memory_type == "project_state"]
    others = [memory for memory in memories if memory.memory_type not in {"task", "project_state"}]
    lines = ["接下来需要注意："]
    if tasks:
        lines.append("待办/安排：" + "；".join(format_event(memory) for memory in tasks[:4]))
    if risks:
        lines.append("风险/卡点：" + "；".join(memory.content for memory in risks[:4]))
    if others:
        lines.append("相关事项：" + "；".join(memory.content for memory in others[:3]))
    return "\n".join(lines)
