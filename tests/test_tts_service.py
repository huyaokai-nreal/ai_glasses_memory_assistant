from __future__ import annotations

import io
import json
import sys
import types
import unittest
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_glasses_memory_assistant import tts_service
from ai_glasses_memory_assistant.server import GlassesHandler
from ai_glasses_memory_assistant.tts_service import MAX_TTS_TEXT_CHARS, TTSAudio, TTSServiceError


# FakeEdgeCommunicate 模拟 edge_tts 流式返回，避免测试依赖真实网络/provider。
class FakeEdgeCommunicate:
    def __init__(self, text: str, voice: str, rate: str, pitch: str) -> None:
        self.text = text
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def stream(self):
        yield {"type": "audio", "data": b"mp3"}
        yield {"type": "WordBoundary", "data": b"ignored"}
        yield {"type": "audio", "data": b"-bytes"}


# 服务层测试覆盖文本清理、voice/rate 校验和 provider 异常。
class TTSServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_normalizes_text_and_rejects_empty_text(self) -> None:
        self.assertEqual(tts_service.normalize_tts_text("  你好\n世界  "), "你好 世界")
        with self.assertRaisesRegex(ValueError, "text is required"):
            tts_service.normalize_tts_text("   ")

    def test_normalizes_markdown_for_speech(self) -> None:
        weather_reply = "北京当前实时天气如下：\n\n- **气温**：约 22°C\n- **天气**：晴/多云"
        normalized = tts_service.normalize_tts_text(weather_reply)

        self.assertNotIn("**", normalized)
        self.assertNotIn("- ", normalized)
        self.assertIn("气温", normalized)
        self.assertIn("22°C", normalized)
        self.assertIn("天气", normalized)

    def test_normalizes_links_without_reading_urls(self) -> None:
        normalized = tts_service.normalize_tts_text("你可以打开 [看天气](https://www.kantianqi.com/) 查看，或访问 https://weather.cma.cn/")

        self.assertIn("看天气", normalized)
        self.assertNotIn("https://", normalized)
        self.assertNotIn("weather.cma.cn", normalized)

    def test_rejects_overlong_text(self) -> None:
        with self.assertRaisesRegex(ValueError, str(MAX_TTS_TEXT_CHARS)):
            tts_service.normalize_tts_text("你" * (MAX_TTS_TEXT_CHARS + 1))

    def test_normalizes_edge_voice_and_rate(self) -> None:
        self.assertEqual(tts_service.normalize_edge_voice(None), "zh-CN-XiaoyiNeural")
        self.assertEqual(tts_service.normalize_edge_voice("zh-CN-XiaoxiaoNeural"), "zh-CN-XiaoxiaoNeural")
        self.assertEqual(tts_service.normalize_edge_rate(None), "+8%")
        self.assertEqual(tts_service.normalize_edge_rate("8%"), "+8%")
        with self.assertRaisesRegex(ValueError, "zh-CN Neural"):
            tts_service.normalize_edge_voice("en-US-JennyNeural")
        with self.assertRaisesRegex(ValueError, "between"):
            tts_service.normalize_edge_rate("+80%")

    async def test_synthesize_speech_streams_edge_audio_without_writing_files(self) -> None:
        fake_edge = types.SimpleNamespace(Communicate=FakeEdgeCommunicate)
        with patch.dict(sys.modules, {"edge_tts": fake_edge}), patch.dict("os.environ", {"AI_GLASSES_TTS_PROVIDER": "edge"}, clear=False):
            result = await tts_service.synthesize_speech_async("你好", voice="zh-CN-XiaoyiNeural", rate="+8%")

        self.assertEqual(result.audio, b"mp3-bytes")
        self.assertEqual(result.media_type, "audio/mpeg")
        self.assertEqual(result.provider, "edge")
        self.assertEqual(result.voice, "zh-CN-XiaoyiNeural")

    async def test_missing_edge_dependency_is_clear(self) -> None:
        with patch.dict(sys.modules, {"edge_tts": None}), patch.dict("os.environ", {"AI_GLASSES_TTS_PROVIDER": "edge"}, clear=False):
            with self.assertRaisesRegex(TTSServiceError, "edge-tts is not installed"):
                await tts_service.synthesize_speech_async("你好")


# HTTP 测试直接构造标准库 handler，验证 /api/tts 的状态码和二进制响应。
class TTSHTTPTests(unittest.TestCase):
    def request_tts(self, payload: dict) -> tuple[int, dict[str, str], bytes]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler = GlassesHandler.__new__(GlassesHandler)
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/api/tts"
        handler.command = "POST"
        handler.request_version = "HTTP/1.1"
        handler.requestline = "POST /api/tts HTTP/1.1"
        handler.client_address = ("127.0.0.1", 0)
        handler.server = types.SimpleNamespace(server_version="test", sys_version="")
        handler.log_request = lambda *args, **kwargs: None

        handler.do_POST()

        response = handler.wfile.getvalue()
        head, response_body = response.split(b"\r\n\r\n", 1)
        lines = head.split(b"\r\n")
        status = int(lines[0].split()[1])
        headers = BytesParser(policy=default).parsebytes(b"\r\n".join(lines[1:]) + b"\r\n\r\n")
        return status, dict(headers.items()), response_body

    def test_tts_endpoint_returns_mp3_bytes(self) -> None:
        speech = TTSAudio(audio=b"mp3", provider="edge", voice="zh-CN-XiaoyiNeural")
        with patch("ai_glasses_memory_assistant.server.synthesize_speech", return_value=speech):
            status, headers, body = self.request_tts({"text": "你好"})

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/mpeg")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-TTS-Voice"], "zh-CN-XiaoyiNeural")
        self.assertEqual(body, b"mp3")

    def test_tts_endpoint_returns_400_for_bad_input(self) -> None:
        with patch("ai_glasses_memory_assistant.server.synthesize_speech", side_effect=ValueError("text is required")):
            status, headers, body = self.request_tts({"text": ""})

        self.assertEqual(status, 400)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertIn("text is required", body.decode("utf-8"))

    def test_tts_endpoint_returns_503_when_provider_unavailable(self) -> None:
        with patch(
            "ai_glasses_memory_assistant.server.synthesize_speech",
            side_effect=TTSServiceError("edge-tts is not installed"),
        ):
            status, headers, body = self.request_tts({"text": "你好"})

        self.assertEqual(status, 503)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertIn("edge-tts is not installed", body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
