from __future__ import annotations

from html.parser import HTMLParser
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class WebSearchResponse:
    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "duckduckgo_html_fallback"

    def context_text(self, *, limit: int = 5) -> str:
        lines: list[str] = []
        for idx, item in enumerate(self.results[:limit], start=1):
            lines.append(
                f"{idx}. {item.get('title') or 'Untitled'}\n"
                f"URL: {item.get('url') or ''}\n"
                f"Summary: {item.get('description') or item.get('snippet') or ''}"
            )
        return "\n\n".join(lines)


# 解析 DuckDuckGo HTML fallback，只抽取标题、链接和摘要三类上下文。
class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._current_title: list[str] = []
        self._current_url = ""
        self._current_snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: v or "" for k, v in attrs}
        classes = attrs_dict.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._in_title = True
            self._current_title = []
            self._current_snippet = []
            self._current_url = self._clean_url(attrs_dict.get("href", ""))
        elif tag in {"a", "div"} and "result__snippet" in classes:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
        if tag in {"a", "div"} and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)

    def close(self) -> None:
        self._flush()
        super().close()

    # 一个搜索结果结束时统一去重入队。
    def _flush(self) -> None:
        title = " ".join("".join(self._current_title).split())
        if not title or not self._current_url:
            return
        snippet = " ".join("".join(self._current_snippet).split())
        if not any(r["url"] == self._current_url for r in self.results):
            self.results.append({
                "title": title,
                "url": self._current_url,
                "description": snippet,
            })
        self._current_title = []
        self._current_snippet = []
        self._current_url = ""

    @staticmethod
    def _clean_url(url: str) -> str:
        if not url:
            return ""
        if url.startswith("//duckduckgo.com/l/"):
            parsed = urlparse("https:" + url)
            uddg = parse_qs(parsed.query).get("uddg", [""])[0]
            return unquote(uddg) if uddg else url
        return url


# 无 API key 搜索兜底，失败会由上层记录 debug 并继续对话。
def duckduckgo_search(query: str, *, limit: int = 5, timeout: float = 8.0) -> list[dict[str, Any]]:
    """No-key web search fallback for the local demo."""
    safe_limit = max(1, min(int(limit), 10))
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 AI-Glasses-Memory-Assistant/0.1",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = _DuckDuckGoParser()
    parser.feed(html)
    parser.close()
    return parser.results[:safe_limit]


def search_web(query: str, *, limit: int = 5, timeout: float = 8.0) -> WebSearchResponse:
    """Project-owned web search boundary for realtime context lookups."""
    results = duckduckgo_search(query, limit=limit, timeout=timeout)
    return WebSearchResponse(query=query, results=results)
