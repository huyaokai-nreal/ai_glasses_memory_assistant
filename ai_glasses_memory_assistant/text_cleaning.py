from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .privacy_filter import RedactionResult, redact_sensitive_text


_FILLER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("filler:um", re.compile(r"(?:嗯+|呃+|啊+|额+)", re.IGNORECASE)),
    ("filler:that", re.compile(r"(?:那个|这个|就是|就是说|怎么说呢)", re.IGNORECASE)),
    ("filler:laugh", re.compile(r"(?:哈哈+|呵呵+|lol|haha+)", re.IGNORECASE)),
    ("noise:background", re.compile(r"(?:有点吵|太吵了|背景.*?吵|听不清|杂音)", re.IGNORECASE)),
)

_CORRECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("correction:negate_previous", re.compile(r"(?:哦?不对|刚才说错了|说错了|不是这个|不对)")),
    ("correction:replace_with", re.compile(r"(?:改成|应该是|更正为|换成|实际是|准确说)")),
    ("correction:do_not_remember", re.compile(r"(?:别记这个|不用记|不要记这个|不要保存|真的不要保存|这个别保存|别保存这个)")),
)

_NEGATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("negation:not_preference", re.compile(r"(?:不是喜欢|不喜欢|并不喜欢|不是.{0,24}(?:而是|是))")),
    ("negation:not_plan", re.compile(r"(?:不用|不要|取消|先不|不考虑|别提醒)")),
)

_SEGMENT_SPLIT_RE = re.compile(r"[\n。！？!?；;]+|(?<=，)(?=(?:但是|不过|然后|后来|哦|啊|嗯|那个|对了|另外|还有))")
_SPACE_RE = re.compile(r"[ \t\r\f\v]+")


@dataclass(frozen=True)
class TextMarker:
    kind: str
    text: str
    start: int
    end: int

    def debug_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class TextSegment:
    text: str
    index: int
    marker_kinds: list[str] = field(default_factory=list)

    def debug_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "marker_kinds": list(self.marker_kinds),
        }


@dataclass(frozen=True)
class TextCleaningTrace:
    raw_length: int
    normalized_text: str
    redaction: RedactionResult
    noise_markers: list[TextMarker] = field(default_factory=list)
    correction_markers: list[TextMarker] = field(default_factory=list)
    negation_markers: list[TextMarker] = field(default_factory=list)
    segments: list[TextSegment] = field(default_factory=list)

    def debug_payload(self) -> dict[str, Any]:
        return {
            "raw_length": self.raw_length,
            "normalized_text": self.normalized_text,
            **self.redaction.debug_payload(),
            "noise_markers": [marker.debug_payload() for marker in self.noise_markers],
            "correction_markers": [marker.debug_payload() for marker in self.correction_markers],
            "negation_markers": [marker.debug_payload() for marker in self.negation_markers],
            "segments": [segment.debug_payload() for segment in self.segments],
            "summary": {
                "noise_marker_count": len(self.noise_markers),
                "correction_marker_count": len(self.correction_markers),
                "negation_marker_count": len(self.negation_markers),
                "segment_count": len(self.segments),
            },
        }


def clean_text_for_memory(raw_text: str, *, max_segments: int = 12) -> TextCleaningTrace:
    """Create a raw-preserving diagnostic trace for memory ingestion text."""
    normalized = _normalize_text(raw_text)
    redaction = redact_sensitive_text(normalized)
    safe_text = redaction.text.strip()
    noise_markers = _find_markers(safe_text, _FILLER_PATTERNS)
    correction_markers = _find_markers(safe_text, _CORRECTION_PATTERNS)
    negation_markers = _find_markers(safe_text, _NEGATION_PATTERNS)
    all_markers = [*noise_markers, *correction_markers, *negation_markers]
    segments = _segment_text(safe_text, all_markers, max_segments=max_segments)
    return TextCleaningTrace(
        raw_length=len(str(raw_text or "")),
        normalized_text=safe_text,
        redaction=redaction,
        noise_markers=noise_markers,
        correction_markers=correction_markers,
        negation_markers=negation_markers,
        segments=segments,
    )


def _normalize_text(raw_text: str) -> str:
    text = unicodedata.normalize("NFKC", str(raw_text or ""))
    text = text.replace("\u200b", "")
    text = _SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"([。！？!?，,；;])\1{1,}", r"\1", text)
    return text.strip()


def _find_markers(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[TextMarker]:
    markers: list[TextMarker] = []
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            markers.append(TextMarker(kind=kind, text=match.group(0), start=match.start(), end=match.end()))
    markers.sort(key=lambda marker: (marker.start, marker.end, marker.kind))
    return markers


def _segment_text(text: str, markers: list[TextMarker], *, max_segments: int) -> list[TextSegment]:
    if not text:
        return []
    raw_segments = [segment.strip(" \t，,") for segment in _SEGMENT_SPLIT_RE.split(text)]
    segments: list[TextSegment] = []
    cursor = 0
    for raw_segment in raw_segments:
        if not raw_segment:
            continue
        start = text.find(raw_segment, cursor)
        if start < 0:
            start = cursor
        end = start + len(raw_segment)
        marker_kinds = sorted({
            marker.kind
            for marker in markers
            if marker.start < end and marker.end > start
        })
        segments.append(TextSegment(text=raw_segment, index=len(segments), marker_kinds=marker_kinds))
        cursor = end
        if len(segments) >= max_segments:
            break
    return segments
