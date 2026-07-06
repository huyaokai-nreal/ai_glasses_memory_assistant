from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ai_glasses_memory_assistant.llm_client import (
    HermesLLMClient,
    OpenAICompatibleLLMClient,
    create_hermes_llm_client,
)


class HermesLLMClientTest(unittest.TestCase):
    def test_openai_compatible_client_returns_hermes_style_result(self) -> None:
        captured: dict = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="主回复")
                        )
                    ]
                )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.chat = SimpleNamespace(completions=FakeCompletions())

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = FakeOpenAI

        with patch.dict("sys.modules", {"openai": openai_module}):
            client = OpenAICompatibleLLMClient(
                model="deepseek-v4-flash",
                base_url="https://api.deepseek.com",
                api_key="key",
                provider="deepseek",
                api_mode="chat_completions",
                system_prompt="system",
            )
            result = client.run_conversation(
                "message with memory context",
                conversation_history=[{"role": "user", "content": "old"}],
                persist_user_message="raw message",
            )

        self.assertEqual(captured["client_kwargs"], {"api_key": "key", "base_url": "https://api.deepseek.com"})
        self.assertEqual(captured["model"], "deepseek-v4-flash")
        self.assertEqual(captured["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(captured["messages"][1], {"role": "user", "content": "old"})
        self.assertEqual(captured["messages"][2], {"role": "user", "content": "message with memory context"})
        self.assertEqual(result["final_response"], "主回复")
        self.assertEqual(result["api_calls"], 1)
        self.assertTrue(result["completed"])
        self.assertEqual(result["messages"][-2], {"role": "user", "content": "raw message"})
        self.assertEqual(result["messages"][-1], {"role": "assistant", "content": "主回复"})

    def test_openai_compatible_client_system_message_overrides_default_prompt(self) -> None:
        captured: dict = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = FakeOpenAI

        with patch.dict("sys.modules", {"openai": openai_module}):
            client = OpenAICompatibleLLMClient(
                model="m",
                base_url="https://api.deepseek.com",
                api_key="key",
                system_prompt="default",
            )
            client.run_conversation("hello", system_message="override")

        self.assertEqual(captured["messages"][0], {"role": "system", "content": "override"})

    def test_adapter_forwards_run_conversation_and_attributes(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.enabled_toolsets = []
                self.step_callback = None
                self.calls: list[dict] = []

            def run_conversation(self, user_message, system_message=None, conversation_history=None, **kwargs):
                self.calls.append({
                    "user_message": user_message,
                    "system_message": system_message,
                    "conversation_history": conversation_history,
                    "kwargs": kwargs,
                })
                return {"final_response": "ok", "messages": []}

            def chat(self, message):
                return f"chat:{message}"

        agent = FakeAgent()
        client = HermesLLMClient(agent)

        callback = object()
        client.step_callback = callback
        result = client.run_conversation("hello", system_message="sys", persist_user_message="hello")

        self.assertEqual(result["final_response"], "ok")
        self.assertIs(agent.step_callback, callback)
        self.assertEqual(client.enabled_toolsets, [])
        self.assertEqual(client.chat("hi"), "chat:hi")
        self.assertEqual(agent.calls[0]["user_message"], "hello")
        self.assertEqual(agent.calls[0]["system_message"], "sys")
        self.assertEqual(agent.calls[0]["kwargs"]["persist_user_message"], "hello")

    def test_create_hermes_llm_client_preserves_aia_agent_parameters(self) -> None:
        captured: dict = {}

        class CapturingAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.enabled_toolsets = kwargs.get("enabled_toolsets")

            def run_conversation(self, *args, **kwargs):
                return {"final_response": "", "messages": []}

        run_agent = types.ModuleType("run_agent")
        run_agent.AIAgent = CapturingAgent
        config = SimpleNamespace(
            model="demo-model",
            reasoning_config={"enabled": False},
        )
        runtime = {
            "provider": "custom",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "key",
            "command": None,
            "args": None,
            "credential_pool": None,
        }

        with patch.dict("sys.modules", {"run_agent": run_agent}):
            client = create_hermes_llm_client(
                config=config,
                runtime=runtime,
                api_mode="chat_completions",
                session_id="s1",
                session_db=object(),
                user_id="u1",
                system_prompt="system",
            )

        self.assertIsInstance(client, HermesLLMClient)
        self.assertEqual(captured["model"], "demo-model")
        self.assertEqual(captured["provider"], "custom")
        self.assertEqual(captured["base_url"], "http://127.0.0.1:11434/v1")
        self.assertEqual(captured["api_key"], "key")
        self.assertEqual(captured["api_mode"], "chat_completions")
        self.assertEqual(captured["reasoning_config"], {"enabled": False})
        self.assertTrue(captured["quiet_mode"])
        self.assertEqual(captured["platform"], "ai_glasses_web")
        self.assertEqual(captured["session_id"], "s1")
        self.assertEqual(captured["user_id"], "u1")
        self.assertTrue(captured["skip_context_files"])
        self.assertTrue(captured["skip_memory"])
        self.assertEqual(captured["enabled_toolsets"], [])
        self.assertEqual(captured["ephemeral_system_prompt"], "system")


if __name__ == "__main__":
    unittest.main()
