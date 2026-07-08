from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass


DEFAULT_TTS_PROVIDER = "edge"
DEFAULT_EDGE_VOICE = "zh-CN-XiaoyiNeural"
DEFAULT_EDGE_RATE = "+8%"
DEFAULT_EDGE_PITCH = "+0Hz"
MAX_TTS_TEXT_CHARS = 800

_RATE_RE = re.compile(r"^[+-]?\d{1,3}%$")
_MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]*)\]\((?:[^)]+)\)")
_RAW_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_BOLD_RE = re.compile(r"(\*\*|__)([^*_]+?)\1")
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_STRIKETHROUGH_RE = re.compile(r"~~([^~\n]+?)~~")


@dataclass(frozen=True)
class TTSAudio:
    audio: bytes
    media_type: str = "audio/mpeg"
    provider: str = DEFAULT_TTS_PROVIDER
    voice: str = DEFAULT_EDGE_VOICE


class TTSServiceError(RuntimeError):
    """Raised when speech synthesis cannot be completed."""


# 播报前清理 Markdown/URL，避免把格式符号和链接读出来。
def strip_tts_markup(text: str) -> str:
    """Remove visual Markdown that sounds unnatural when spoken."""
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    source = re.sub(r"```[A-Za-z0-9_+-]*\n?", "\n", source)
    source = source.replace("```", "\n")
    source = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1).strip(), source)
    source = _RAW_URL_RE.sub("", source)

    lines = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^>\s*", "", line)
        line = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", line)
        line = re.sub(r"^\[[ xX]\]\s+", "", line)
        if line:
            lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _BOLD_RE.sub(r"\2", cleaned)
    cleaned = _ITALIC_STAR_RE.sub(r"\1", cleaned)
    cleaned = _STRIKETHROUGH_RE.sub(r"\1", cleaned)
    cleaned = cleaned.replace("*", "").replace("`", "")
    return cleaned


# TTS 输入统一在这里限长和判空，保护前端误传大段内容。
def normalize_tts_text(text: str) -> str:
    normalized = " ".join(strip_tts_markup(text).split())
    if not normalized:
        raise ValueError("text is required")
    if len(normalized) > MAX_TTS_TEXT_CHARS:
        raise ValueError(f"text must be {MAX_TTS_TEXT_CHARS} characters or fewer")
    return normalized


# Edge TTS 语速只允许小范围调整，避免生成不可听的语音。
def normalize_edge_rate(rate: str | None) -> str:
    candidate = str(rate or DEFAULT_EDGE_RATE).strip()
    if not _RATE_RE.match(candidate):
        raise ValueError("rate must look like '+8%' or '-5%'")
    value = int(candidate[:-1])
    if value < -50 or value > 50:
        raise ValueError("rate must be between -50% and +50%")
    return f"{value:+d}%"


# 当前 demo 只允许中文 Neural voice，保持语音入口体验一致。
def normalize_edge_voice(voice: str | None) -> str:
    candidate = str(voice or os.getenv("AI_GLASSES_TTS_VOICE") or DEFAULT_EDGE_VOICE).strip()
    if not candidate:
        return DEFAULT_EDGE_VOICE
    if not candidate.startswith("zh-CN-") or not candidate.endswith("Neural"):
        raise ValueError("voice must be a zh-CN Neural voice")
    return candidate


# 异步 TTS 入口保留给未来异步封装；标准库 server 会通过同步 wrapper 调用。
async def synthesize_speech_async(text: str, *, voice: str | None = None, rate: str | None = None) -> TTSAudio:
    provider = os.getenv("AI_GLASSES_TTS_PROVIDER", DEFAULT_TTS_PROVIDER).strip().lower() or DEFAULT_TTS_PROVIDER
    if provider != "edge":
        raise TTSServiceError(f"unsupported TTS provider: {provider}")

    normalized_text = normalize_tts_text(text)
    normalized_voice = normalize_edge_voice(voice)
    normalized_rate = normalize_edge_rate(rate)

    try:
        import edge_tts
    except ImportError as exc:
        raise TTSServiceError("edge-tts is not installed") from exc

    try:
        communicate = edge_tts.Communicate(
            normalized_text,
            normalized_voice,
            rate=normalized_rate,
            pitch=DEFAULT_EDGE_PITCH,
        )
        audio = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                audio.extend(chunk.get("data") or b"")
    except Exception as exc:
        raise TTSServiceError("Edge TTS synthesis failed") from exc

    if not audio:
        raise TTSServiceError("Edge TTS returned no audio")
    return TTSAudio(audio=bytes(audio), provider=provider, voice=normalized_voice)


# 标准库 HTTP handler 使用同步入口，内部仍复用同一套异步实现。
def synthesize_speech(text: str, *, voice: str | None = None, rate: str | None = None) -> TTSAudio:
    return asyncio.run(synthesize_speech_async(text, voice=voice, rate=rate))
