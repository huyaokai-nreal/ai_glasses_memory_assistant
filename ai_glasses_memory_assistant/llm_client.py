from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
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
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict):
            request_kwargs["response_format"] = response_format
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max(1, int(max_tokens))
        # DeepSeek thinking is useful for open-ended replies, but a fixed-schema
        # accounting pass needs to spend its output budget on the JSON ledger.
        if kwargs.get("disable_thinking") and self.provider == "deepseek":
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = self._client.chat.completions.create(
            **request_kwargs,
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
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content", "") if isinstance(message, dict) else ""
            if isinstance(content, list):
                return "".join(
                    str(item.get("text") or "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            return str(content or "")
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


class StdlibOpenAICompatibleLLMClient(OpenAICompatibleLLMClient):
    """OpenAI-compatible transport for embedded runtimes without the SDK."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        provider: str = "openai_compatible",
        api_mode: str = "chat_completions",
        system_prompt: str = "",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_mode = api_mode
        self.enabled_toolsets: list[str] = []
        self.step_callback = None
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self._system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds

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
        request_payload: dict[str, Any] = {"model": self.model, "messages": messages}
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict):
            request_payload["response_format"] = response_format
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            request_payload["max_tokens"] = max(1, int(max_tokens))
        if kwargs.get("disable_thinking") and self.provider == "deepseek":
            request_payload["thinking"] = {"type": "disabled"}
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = _http_error_detail(exc)
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("LLM response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("LLM response must be a JSON object")
        final_response = self._response_text(payload)
        returned_messages = list(conversation_history or [])
        returned_messages.append({
            "role": "user",
            "content": persist_user_message if persist_user_message is not None else user_message,
        })
        returned_messages.append({"role": "assistant", "content": final_response})
        return {
            "final_response": final_response,
            "messages": returned_messages,
            "api_calls": 1,
            "completed": True,
        }


def _http_error_detail(exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return "request rejected"
    if not isinstance(payload, dict):
        return "request rejected"
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("type") or "request rejected")[:300]
    return str(error or "request rejected")[:300]


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


def create_stdlib_openai_compatible_llm_client(
    *,
    model: str,
    base_url: str,
    api_key: str,
    provider: str,
    api_mode: str,
    system_prompt: str,
) -> StdlibOpenAICompatibleLLMClient:
    return StdlibOpenAICompatibleLLMClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        api_mode=api_mode,
        system_prompt=system_prompt,
    )
