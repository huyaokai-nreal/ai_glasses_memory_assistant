from __future__ import annotations

import re
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


def project_name_for_memory(memory: MemoryEvent) -> str:
    for tag in memory.tags:
        if str(tag).startswith("project:"):
            return str(tag).split(":", 1)[1] or "未分类项目"
    text = memory.content
    match = re.search(
        r"(?:项目|project)[:： ]?(?P<name>[\w\u4e00-\u9fff-]{2,24}?)(?:决定|会议|风险|卡点|进展|延期|完成|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("name").strip(" ：:")
    match = re.search(
        r"(?P<name>[\w\u4e00-\u9fff -]{2,24}?)(?:项目|project)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return normalize_project_name(match.group("name"))
    return "未分类项目"


def normalize_project_name(project: str) -> str:
    cleaned = str(project or "").strip(" ：:-")
    for prefix in ("最近事件线索包括", "事件线索包括", "最近线索包括"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" ：:-")
    return re.sub(r"\s+", "", cleaned) or "未分类项目"


def looks_like_background_only_observation(content: str) -> bool:
    text = str(content or "")
    return any(marker in text for marker in ("背景", "说明", "目的")) and not any(
        marker in text for marker in ("风险", "卡点", "任务", "待办", "决定", "状态", "进展")
    )


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
