from __future__ import annotations


def summarize_capture_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return "；".join(lines[:5])


def continuous_capture_reply(segments: list[str]) -> str:
    if len(segments) <= 1:
        return "收到，我先把这段长输入整理到时间线里，有价值的内容会后台沉淀。"
    return f"收到，我先把这段长输入按 {len(segments)} 段整理，有价值的内容会后台沉淀。"
