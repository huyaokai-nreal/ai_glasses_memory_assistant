from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .intent_policy import is_question
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


def document_recall_phrase_policy(message: str, recall: DocumentRecallResult) -> dict[str, Any]:
    type_markers = document_reference_type_markers(message)
    reference_markers = recent_document_reference_markers(message)
    explicit_document_terms = [
        marker
        for marker in ("文档", "文件", "上传", "md", "Markdown", "markdown")
        if marker in str(message or "")
    ]
    payload: dict[str, Any] = {
        "role": "none",
        "reason": recall.reason or "not_checked",
        "type_markers": type_markers,
        "reference_markers": reference_markers,
        "explicit_document_terms": explicit_document_terms,
        "final_action": recall.mode,
    }
    if recall.reason == "not_document_query":
        if type_markers:
            payload.update({
                "role": "weak_signal",
                "treatment": "ignored_without_document_context",
            })
        return payload
    if recall.reason == "ambiguous_document_title_match":
        payload.update({
            "role": "retrieval_guard",
            "treatment": "metadata_only_until_disambiguated",
        })
        return payload
    if recall.reason == "cross_document_compare_query":
        payload.update({
            "role": "retrieval_guard",
            "treatment": "metadata_only_for_cross_document_compare",
        })
        return payload
    if recall.reason in {"recent_document_reference", "recent_document_type_reference"}:
        payload.update({
            "role": "retrieval_hint",
            "treatment": "select_recent_document_candidate",
        })
        return payload
    if recall.reason == "recent_document_followup":
        payload.update({
            "role": "retrieval_hint",
            "treatment": "inherit_recent_document_candidate",
        })
        return payload
    if recall.reason in {"document_title_match", "document_detail_query", "document_overview_query", "upload_history_query"}:
        payload.update({
            "role": "retrieval_hint",
            "treatment": "load_document_context" if recall.context else "metadata_reply",
        })
        return payload
    if recall.mode in {"none", "skipped"}:
        payload.update({
            "role": "none",
            "treatment": "no_document_context_loaded",
        })
        return payload
    payload.update({
        "role": "retrieval_hint",
        "treatment": "document_context_available",
    })
    return payload


def is_document_query(message: str) -> bool:
    text = str(message or "")
    if any(marker in text for marker in ("文档", "文件", "上传", "md", "Markdown", "markdown")):
        return True
    if any(marker in text for marker in ("攻略", "会议纪要", "周报", "日报")):
        return any(
            context in text
            for context in (
                "这份",
                "那份",
                "上一份",
                "上次",
                "最近上传",
                "最新的",
                "刚上传",
                "刚刚上传",
                "刚才上传",
                "里面",
                "里",
            )
        )
    return False


def is_document_history_query(message: str) -> bool:
    text = str(message or "")
    return "上传" in text and any(marker in text for marker in ("什么时候", "哪些", "什么文档", "什么文件", "上传过"))


def is_document_overview_query(message: str) -> bool:
    text = str(message or "")
    return "讲了什么" in text or "关于什么" in text or "是什么文档" in text


def is_cross_document_compare_query(message: str) -> bool:
    text = str(message or "")
    compare_markers = ("对比", "比较", "区别", "不同", "差异")
    plural_markers = ("两份", "两篇", "两个文档", "这两份", "这两篇", "两版")
    return any(marker in text for marker in compare_markers) and any(marker in text for marker in plural_markers)


def is_recent_document_reference(message: str) -> bool:
    if not recent_document_reference_markers(message):
        return False
    return is_document_query(str(message or ""))


def is_recent_document_followup_query(message: str) -> bool:
    text = str(message or "").strip()
    if not text:
        return False
    if is_document_query(text):
        return False
    if not is_question(text):
        return False
    if not any(marker in text for marker in ("那", "这", "里面", "里", "那个", "这个")):
        return False
    if any(marker in text for marker in ("为什么这么说", "依据是什么", "依据", "为什么没记住", "为什么没保存")):
        return False
    return len(text) <= 24


def recent_document_reference_markers(message: str) -> list[str]:
    text = str(message or "")
    return [
        marker
        for marker in (
            "刚刚那份",
            "刚刚这份",
            "刚才那份",
            "刚才这份",
            "这份",
            "那份",
            "上一份",
            "最近上传",
            "最近的",
            "最新的",
            "刚上传",
            "刚刚上传",
            "刚才上传",
        )
        if marker in text
    ]


def message_matches_recent_document_followup(document: DocumentRecord, message: str) -> bool:
    haystack = " ".join([document.title, document.summary, document.content or ""])
    normalized_haystack = haystack.lower()
    text = str(message or "").strip().lower()
    stop_terms = {
        "这个", "那个", "这次", "那次", "这里", "那里", "里面", "怎么", "什么", "一下",
        "然后", "还是", "继续", "刚才", "刚刚", "这份", "那份", "这个材料", "那个材料",
        "写的", "说的", "问的", "内容", "细节", "问题", "回答", "依据",
    }
    candidates: set[str] = set()
    for size in range(2, min(4, len(text)) + 1):
        for idx in range(0, len(text) - size + 1):
            piece = text[idx:idx + size]
            if piece in stop_terms:
                continue
            if all("\u4e00" <= ch <= "\u9fff" for ch in piece) or piece.isascii():
                candidates.add(piece)
    return any(piece in normalized_haystack for piece in candidates)


def document_reference_type_markers(message: str) -> list[str]:
    text = str(message or "")
    return [marker for marker in ("会议纪要", "周报", "日报", "攻略") if marker in text]


def document_matches_reference_type(document: DocumentRecord, type_markers: list[str]) -> bool:
    haystack = " ".join([document.filename, document.title, document.summary])
    return any(marker in haystack for marker in type_markers)


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
    requested_markers = []
    text = str(message or "")
    if any(marker in text for marker in ("风险", "卡点")):
        requested_markers.extend(("风险", "卡点"))
    if any(marker in text for marker in ("建议", "做法")):
        requested_markers.extend(("建议", "做法"))
    if any(marker in text for marker in ("适用", "适合", "场景", "对象", "人群")):
        requested_markers.extend(("适用场景", "适合", "对象", "人群"))
    if any(marker in text for marker in ("结论", "决定", "侧重点", "重点")):
        requested_markers.extend(("结论", "决定", "重点"))
    heading_markers = tuple(dict.fromkeys([*requested_markers, "一句话总结", "摘要", "总结"]))
    for idx, line in enumerate(lines):
        if any(marker in line for marker in heading_markers):
            for candidate in lines[idx + 1:]:
                if candidate.startswith("#"):
                    break
                cleaned = candidate.lstrip("-* ").strip()
                if cleaned:
                    return cleaned[:120]
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


def document_history_reply(documents: list[DocumentRecord]) -> str:
    lines = ["你上传过这些文档："]
    for document in documents:
        lines.append(f"- {document.filename}：{_format_timestamp(document.created_at)}，关于{document.title}。")
    return "\n".join(lines)


def document_overview_reply(documents: list[DocumentRecord]) -> str:
    document = documents[0]
    return f"这份文档是《{document.title}》。{document.summary}"


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
