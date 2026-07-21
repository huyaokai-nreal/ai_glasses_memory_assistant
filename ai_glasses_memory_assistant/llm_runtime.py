from __future__ import annotations

import os
from dataclasses import dataclass

from .env_loader import load_app_dotenv
from .llm_client import create_openai_compatible_llm_client, create_stdlib_openai_compatible_llm_client


LLM_PROVIDER_ENV = "AI_GLASSES_LLM_PROVIDER"
LLM_MODEL_ENV = "AI_GLASSES_LLM_MODEL"
LLM_BASE_URL_ENV = "AI_GLASSES_LLM_BASE_URL"
LLM_API_KEY_ENV = "AI_GLASSES_LLM_API_KEY"
LLM_API_MODE_ENV = "AI_GLASSES_LLM_API_MODE"
LLM_TRANSPORT_ENV = "AI_GLASSES_LLM_TRANSPORT"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
SUPPORTED_LLM_API_MODE = "chat_completions"
SUPPORTED_LLM_TRANSPORTS = {"openai_sdk", "stdlib_http"}
DEEPSEEK_FALLBACK_PROVIDER = "deepseek"


@dataclass(frozen=True)
class DemoLLMConfig:
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    api_mode: str = ""
    transport: str = "openai_sdk"


def _demo_llm_config() -> DemoLLMConfig:
    provider = os.getenv(LLM_PROVIDER_ENV, DEEPSEEK_FALLBACK_PROVIDER).strip() or DEEPSEEK_FALLBACK_PROVIDER
    model = os.getenv(LLM_MODEL_ENV, "").strip()
    base_url = os.getenv(LLM_BASE_URL_ENV, "").strip().rstrip("/")
    api_key = os.getenv(LLM_API_KEY_ENV, "").strip()
    if not api_key and provider == DEEPSEEK_FALLBACK_PROVIDER:
        api_key = os.getenv(DEEPSEEK_API_KEY_ENV, "").strip()
    api_mode = os.getenv(LLM_API_MODE_ENV, SUPPORTED_LLM_API_MODE).strip() or SUPPORTED_LLM_API_MODE
    if api_mode != SUPPORTED_LLM_API_MODE:
        raise ValueError(
            f"{LLM_API_MODE_ENV}={api_mode!r} is not supported; use "
            f"{SUPPORTED_LLM_API_MODE!r}."
        )
    transport = os.getenv(LLM_TRANSPORT_ENV, "openai_sdk").strip() or "openai_sdk"
    if transport not in SUPPORTED_LLM_TRANSPORTS:
        raise ValueError(
            f"{LLM_TRANSPORT_ENV}={transport!r} is not supported; use one of "
            + ", ".join(sorted(SUPPORTED_LLM_TRANSPORTS))
        )
    missing = []
    if not model:
        missing.append(LLM_MODEL_ENV)
    if not base_url:
        missing.append(LLM_BASE_URL_ENV)
    if not api_key:
        missing.append(f"{LLM_API_KEY_ENV} or {DEEPSEEK_API_KEY_ENV}")
    if missing:
        raise ValueError(
            "OpenAI-compatible LLM requires: " + ", ".join(missing)
        )
    return DemoLLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        transport=transport,
    )


def create_demo_llm_client(*, system_prompt: str):
    load_app_dotenv()
    config = _demo_llm_config()
    factory = (
        create_stdlib_openai_compatible_llm_client
        if config.transport == "stdlib_http"
        else create_openai_compatible_llm_client
    )
    return factory(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        provider=config.provider,
        api_mode=config.api_mode or SUPPORTED_LLM_API_MODE,
        system_prompt=system_prompt,
    )
