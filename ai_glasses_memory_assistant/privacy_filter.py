from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redacted: bool
    categories: list[str]
    count: int

    def debug_payload(self) -> dict[str, object]:
        return {
            "redacted": self.redacted,
            "redaction_categories": self.categories,
            "redaction_count": self.count,
        }


_REDACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private", re.compile(r"<private>[\s\S]*?</private>", re.IGNORECASE)),
    (
        "token",
        re.compile(
            r"(?:api[_-]?key|secret|token|password|credential|auth)"
            r"(?:\s*(?:是|为)\s*|\s*[=:：]\s*)"
            r"[\"']?[A-Za-z0-9_\-/.+]{20,}[\"']?",
            re.IGNORECASE,
        ),
    ),
    ("token", re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]{20,}", re.IGNORECASE)),
    ("token", re.compile(r"sk-proj-[A-Za-z0-9\-_]{20,}")),
    ("token", re.compile(r"(?:sk|pk|rk|ak)-[A-Za-z0-9][A-Za-z0-9\-_]{19,}")),
    ("token", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
    ("token", re.compile(r"gh[pus]_[A-Za-z0-9]{36,}")),
    ("token", re.compile(r"github_pat_[A-Za-z0-9_]{22,}")),
    ("token", re.compile(r"xoxb-[A-Za-z0-9\-]+")),
    ("token", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("token", re.compile(r"AIza[A-Za-z0-9\-_]{35}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("token", re.compile(r"npm_[A-Za-z0-9]{36}")),
    ("token", re.compile(r"glpat-[A-Za-z0-9\-_]{20,}")),
    ("token", re.compile(r"dop_v1_[A-Za-z0-9]{64}")),
    ("code", re.compile(r"(?:验证码|校验码|短信码|动态码)\s*(?:是|:|：)?\s*\d{4,8}")),
    ("card", re.compile(r"(?<!\d)(?:\d[ -]?){15,19}(?!\d)")),
    ("id", re.compile(r"(?<![0-9Xx])\d{17}[0-9Xx](?![0-9Xx])")),
)


def redact_sensitive_text(text: str) -> RedactionResult:
    result = str(text or "")
    categories: list[str] = []
    count = 0
    for category, pattern in _REDACTION_PATTERNS:
        placeholder = f"[已脱敏:{category}]"

        def replace(_match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            categories.append(category)
            return placeholder

        result = pattern.sub(replace, result)
    unique_categories = list(dict.fromkeys(categories))
    return RedactionResult(
        text=result,
        redacted=count > 0,
        categories=unique_categories,
        count=count,
    )
