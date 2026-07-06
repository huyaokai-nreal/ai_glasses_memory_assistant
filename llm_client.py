from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    def run_conversation(
        self,
        user_message: str,
        system_message: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...


class HermesLLMClient:
    """Thin adapter around Hermes AIAgent while the demo still uses Hermes runtime."""

    def __init__(self, agent: Any) -> None:
        object.__setattr__(self, "_agent", agent)

    @property
    def agent(self) -> Any:
        return self._agent

    def run_conversation(
        self,
        user_message: str,
        system_message: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._agent.run_conversation(
            user_message,
            system_message=system_message,
            conversation_history=conversation_history,
            **kwargs,
        )

    def chat(self, message: str) -> str:
        return self._agent.chat(message)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_agent":
            object.__setattr__(self, name, value)
            return
        setattr(self._agent, name, value)


def create_hermes_llm_client(
    *,
    config: Any,
    runtime: dict[str, Any],
    api_mode: str | None,
    session_id: str,
    session_db: Any,
    user_id: str,
    system_prompt: str,
) -> HermesLLMClient:
    from run_agent import AIAgent

    agent = AIAgent(
        model=config.model,
        provider=runtime.get("provider"),
        base_url=runtime.get("base_url"),
        api_key=runtime.get("api_key"),
        api_mode=api_mode,
        acp_command=runtime.get("command"),
        acp_args=runtime.get("args"),
        credential_pool=runtime.get("credential_pool"),
        reasoning_config=config.reasoning_config,
        quiet_mode=True,
        platform="ai_glasses_web",
        session_id=session_id,
        session_db=session_db,
        user_id=user_id,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        ephemeral_system_prompt=system_prompt,
    )
    return HermesLLMClient(agent)


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        provider: str = "openai_compatible",
        api_mode: str = "chat_completions",
        system_prompt: str = "",
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.api_mode = api_mode
        self.enabled_toolsets: list[str] = []
        self.step_callback = None
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self._system_prompt = system_prompt
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def run_conversation(
        self,
        user_message: str,
        system_message: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        persist_user_message = kwargs.get("persist_user_message")
        messages = self._build_messages(
            user_message=user_message,
            system_message=system_message,
            conversation_history=conversation_history,
        )
        if callable(self.step_callback):
            self.step_callback(0, [])
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        final_response = self._response_text(response)
        stored_user_content = persist_user_message if persist_user_message is not None else user_message
        returned_messages = list(conversation_history or [])
        returned_messages.append({"role": "user", "content": stored_user_content})
        returned_messages.append({"role": "assistant", "content": final_response})
        return {
            "final_response": final_response,
            "messages": returned_messages,
            "api_calls": 1,
            "completed": True,
        }

    def chat(self, message: str) -> str:
        result = self.run_conversation(message)
        return str(result.get("final_response") or "")

    def _build_messages(
        self,
        *,
        user_message: str,
        system_message: str | None,
        conversation_history: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        effective_system = system_message if system_message is not None else self._system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.extend(dict(item) for item in conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _response_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(getattr(item, "text", "") or ""))
            return "".join(parts)
        return str(content or "")


def create_openai_compatible_llm_client(
    *,
    model: str,
    base_url: str,
    api_key: str,
    provider: str,
    api_mode: str,
    system_prompt: str,
) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        api_mode=api_mode,
        system_prompt=system_prompt,
    )
