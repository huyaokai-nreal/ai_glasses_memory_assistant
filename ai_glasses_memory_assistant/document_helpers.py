from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .memory_store import DocumentRecord


DOCUMENT_TITLE_MATCH_THRESHOLD = 0.72
DOCUMENT_TITLE_AMBIGUOUS_MARGIN = 0.08


@dataclass(frozen=True)
class DocumentRecallResult:
    documents: list[DocumentRecord] = field(default_factory=list)
    context: str = ""
    mode: str = "skipped"
    reason: str = ""


@dataclass(frozen=True)
class DocumentTitleMatch:
    document: DocumentRecord
    score: float


def title_for_markdown_document(text: str, filename: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading:
            return clean_markdown_title(heading.group(1))
    return clean_markdown_title(Path(filename).stem or filename)


def summary_for_markdown_document(text: str, title: str) -> str:
    headings = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading:
            cleaned = clean_markdown_title(heading.group(1))
            if cleaned and cleaned != title:
                headings.append(cleaned)
        if len(headings) >= 4:
            break
    if headings:
        return f"{title}；主要包含：" + "、".join(headings)
    return title


def clean_markdown_title(text: str) -> str:
    cleaned = re.sub(r"[*_`~#>\[\]()]+", "", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:：|")
    return cleaned or "上传文档"


def document_compare_anchor_terms(message: str, anchor: str = "") -> list[str]:
    text = str(message or "")
    for noise in (
        "对比一下",
        "比较一下",
        "这两份",
        "这两篇",
        "两份",
        "两篇",
        "两个文档",
        "文档",
        "文件",
        "有什么不同",
        "有什么区别",
        "有何不同",
        "有何区别",
    ):
        text = text.replace(noise, " ")
    base = anchor or text
    normalized = normalize_document_title_text(base)
    if not normalized:
        return []
    tokens = document_title_tokens(base)
    if tokens:
        return tokens
    if len(normalized) >= 4:
        return [normalized[: min(len(normalized), 6)]]
    return [normalized]


def has_ambiguous_document_title_match(matches: list[DocumentTitleMatch]) -> bool:
    if len(matches) < 2:
        return False
    return matches[0].score - matches[1].score < DOCUMENT_TITLE_AMBIGUOUS_MARGIN


def document_title_match_score(query: str, tokens: list[str], document: DocumentRecord) -> float:
    candidates = [
        normalize_document_title_text(document.title),
        normalize_document_title_text(Path(document.filename).stem),
    ]
    best = 0.0
    for candidate in candidates:
        if not candidate:
            continue
        if query == candidate:
            best = max(best, 1.0)
        elif query in candidate or candidate in query:
            best = max(best, 0.95)
        if tokens and all(token in candidate for token in tokens):
            best = max(best, 0.9)
        best = max(best, SequenceMatcher(None, query, candidate).ratio())
    return best


def normalize_document_title_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"\.(md|markdown)$", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[\s\-_#*`~>\[\]()（）【】《》“”\"'‘’:：,，.。!！?？/\\|]+", "", normalized)
    return normalized


def document_title_tokens(message: str) -> list[str]:
    if not re.search(r"\s", str(message or "")):
        return []
    tokens = []
    for raw_token in re.split(r"\s+", str(message or "")):
        token = normalize_document_title_text(raw_token)
        if len(token) >= 2:
            tokens.append(token)
    return tokens


def document_query_terms(message: str) -> str:
    text = str(message or "")
    noise = (
        "我什么时候上传过",
        "我上传过",
        "什么时候上传",
        "什么文档",
        "什么文件",
        "文档",
        "文件",
        "里面",
        "那份",
        "这份",
        "的",
        "吗",
        "？",
        "?",
    )
    for item in noise:
        text = text.replace(item, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or str(message or "")


def document_metadata_context(documents: list[DocumentRecord]) -> str:
    lines = ["<document-context>", "Archived user documents:"]
    for idx, document in enumerate(documents, start=1):
        lines.append(
            f"{idx}. title: {document.title}\n"
            f"filename: {document.filename}\n"
            f"uploaded_at: {_format_timestamp(document.created_at)}\n"
            f"summary: {document.summary}"
        )
    lines.append("</document-context>")
    return "\n".join(lines)


def document_compare_metadata_context(documents: list[DocumentRecord], *, message: str = "") -> str:
    lines = [
        "<document-context>",
        "[System note: The user is asking for a cross-document comparison. Compare only high-level metadata and summaries. Do not answer with document-only fine details.]",
        "Cross-document comparison candidates:",
    ]
    for idx, document in enumerate(documents, start=1):
        summary = " ".join(str(document.summary or "").split())
        excerpt = document_compare_high_level_excerpt(document, message=message)
        lines.append(
            f"{idx}. title: {document.title}\n"
            f"filename: {document.filename}\n"
            f"uploaded_at: {_format_timestamp(document.created_at)}\n"
            f"high_level_summary: {summary}\n"
            f"high_level_excerpt: {excerpt}"
        )
    lines.append("</document-context>")
    return "\n".join(lines)


def document_compare_high_level_excerpt(document: DocumentRecord, *, message: str = "") -> str:
    content = str(document.content or "")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("#"):
            continue
        cleaned = line.lstrip("-* ").strip()
        if cleaned:
            return cleaned[:120]
    return str(document.title or document.filename or "上传文档")


def document_detail_context(
    message: str,
    document: DocumentRecord,
    *,
    prefer_sections: bool = False,
) -> tuple[str, str]:
    max_full_chars = 12000
    if not prefer_sections and len(document.content) <= max_full_chars:
        body = document.content
        mode = "full_document"
    else:
        body = select_document_sections(message, document.content)
        mode = "sections" if body != document.content else "full_document"
    context = "\n".join([
        "<document-context>",
        "[System note: Use the archived source document below as evidence for this turn. Do not answer document details from summary alone.]",
        f"document_id: {document.id}",
        f"filename: {document.filename}",
        f"title: {document.title}",
        f"uploaded_at: {_format_timestamp(document.created_at)}",
        f"recall_mode: {mode}",
        "",
        body,
        "</document-context>",
    ])
    return context, mode


def select_document_sections(message: str, content: str) -> str:
    terms = [term for term in re.split(r"\s+", document_query_terms(message)) if len(term) >= 2]
    sections = []
    current = []
    for line in content.splitlines():
        if re.match(r"^#{1,6}\s+", line) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    heading_matches = []
    for section in sections:
        lines = section.splitlines()
        first_line = lines[0] if lines else ""
        heading_match = re.match(r"^#{1,6}\s+(.+)$", first_line)
        if not heading_match:
            continue
        heading = clean_markdown_title(heading_match.group(1))
        if heading and any(term in heading or heading in term for term in terms):
            heading_matches.append(section)
    matches = heading_matches or [section for section in sections if any(term in section for term in terms)]
    if not matches:
        matches = sections[:3]
    selected = "\n\n".join(matches[:5]).strip()
    return selected[:12000] if selected else content[:12000]


def project_name_for_document(document: DocumentRecord) -> str:
    text = " ".join([document.title, document.summary, document.filename, document.content[:500]])
    match = re.search(
        r"(?:^|[\s，。；;#])(?:项目|project)[:： ]+(?P<name>[\w\u4e00-\u9fff -]{2,24})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return normalize_project_name(match.group("name"))
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


def weekly_document_summary(document: DocumentRecord) -> str:
    summary = str(document.summary or document.title or document.filename).strip()
    if summary == document.title:
        for line in document.content.splitlines():
            cleaned = clean_markdown_title(line)
            if cleaned and cleaned != document.title:
                summary = f"{document.title}：{cleaned}"
                break
    if document.title and document.title not in summary:
        summary = f"{document.title}：{summary}"
    return summary[:180]


def document_summary_is_background_only(document: DocumentRecord) -> bool:
    text = " ".join(
        part for part in (
            str(document.title or "").strip(),
            str(document.summary or "").strip(),
            str(document.content or "")[:200].strip(),
        )
        if part
    )
    return any(marker in text for marker in ("背景", "说明", "目的"))


def _format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="minutes")
