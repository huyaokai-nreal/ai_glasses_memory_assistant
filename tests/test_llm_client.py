from __future__ import annotations

import json
from unittest.mock import patch

from ai_glasses_memory_assistant.llm_client import StdlibOpenAICompatibleLLMClient


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_stdlib_client_matches_llm_client_contract_without_logging_key() -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response({"choices": [{"message": {"content": "移动端回复"}}]})

    client = StdlibOpenAICompatibleLLMClient(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/",
        api_key="device-secret",
        provider="deepseek",
        system_prompt="system",
    )
    with patch("ai_glasses_memory_assistant.llm_client.urlopen", side_effect=fake_urlopen):
        result = client.run_conversation("你好")

    assert result["final_response"] == "移动端回复"
    assert result["completed"] is True
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer device-secret"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "你好"},
    ]
    assert "device-secret" not in json.dumps(result, ensure_ascii=False)


def test_stdlib_client_applies_structured_accounting_options_only_when_requested() -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return _Response({"choices": [{"message": {"content": '{"items":[]}'}}]})

    client = StdlibOpenAICompatibleLLMClient(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="device-secret",
        provider="deepseek",
    )
    with patch("ai_glasses_memory_assistant.llm_client.urlopen", side_effect=fake_urlopen):
        result = client.run_conversation(
            "Return JSON",
            response_format={"type": "json_object"},
            disable_thinking=True,
            max_tokens=4096,
        )

    assert result["final_response"] == '{"items":[]}'
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert captured["payload"]["max_tokens"] == 4096
