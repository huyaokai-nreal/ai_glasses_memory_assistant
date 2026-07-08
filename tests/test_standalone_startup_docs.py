from __future__ import annotations

from pathlib import Path

from ai_glasses_memory_assistant import agent_bridge
from ai_glasses_memory_assistant.env_loader import APP_LLM_ENV_NAMES


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PACKAGE_ROOT / "README.md"
CONTEXT_README_PATH = PACKAGE_ROOT / "docs" / "context" / "README.md"
CODE_MAP_PATH = PACKAGE_ROOT / "docs" / "context" / "code-map.md"
SYSTEM_FLOW_PATH = PACKAGE_ROOT / "docs" / "context" / "system-flow-current.md"
STANDALONE_PYPROJECT_PATH = PACKAGE_ROOT / "pyproject.standalone.toml"


def _combined_docs() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in [README_PATH, CONTEXT_README_PATH, CODE_MAP_PATH, SYSTEM_FLOW_PATH]
    )


def test_three_context_docs_are_the_only_context_markdown_files() -> None:
    context_files = sorted(path.name for path in (PACKAGE_ROOT / "docs" / "context").glob("*.md"))
    assert context_files == ["README.md", "code-map.md", "system-flow-current.md"]


def test_developer_docs_keep_runtime_env_contract_visible() -> None:
    text = _combined_docs()

    for env_name in APP_LLM_ENV_NAMES:
        assert env_name in text

    assert agent_bridge.OPENAI_COMPATIBLE_BACKEND in text
    assert agent_bridge.HERMES_BACKEND in text
    assert agent_bridge.DEEPSEEK_FALLBACK_PROVIDER in text
    assert agent_bridge.OLLAMA_DEFAULT_API_MODE in text
    assert agent_bridge.DEEPSEEK_API_KEY_ENV in text
    assert "AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1" in text


def test_developer_docs_keep_supported_entrypoints_and_validation_current() -> None:
    text = _combined_docs()

    assert "python -m ai_glasses_memory_assistant.server" in text
    assert "python -m ai_glasses_memory_assistant.app" not in text
    assert "--certfile" in text
    assert "--keyfile" in text
    assert "conda run -n hermes python -m pytest tests/test_standalone_startup_docs.py -q" in text
    assert "conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py" in text
    assert "conda run -n hermes python -m unittest discover tests -q" in text


def test_developer_docs_keep_dependency_and_boundary_contract_visible() -> None:
    text = _combined_docs()

    assert "openai" in text
    assert "fastapi" not in text.lower()
    assert "pydantic" not in text.lower()
    assert "uvicorn" not in text.lower()
    assert "edge-tts" in text
    assert "funasr" in text
    assert "pytest" in text
    assert "always-on audio runtime" in text
    assert "主动提醒" in text
    assert "reliable worker" not in text.lower()


def test_standalone_packaging_draft_keeps_entrypoints_and_extras_current() -> None:
    text = STANDALONE_PYPROJECT_PATH.read_text(encoding="utf-8")

    assert "openai" in text
    assert "fastapi" not in text
    assert "pydantic" not in text
    assert "uvicorn" not in text
    assert "edge-tts" in text
    assert "funasr" in text
    assert "pytest" in text
    assert "ai-glasses-memory-assistant" in text
    assert "ai_glasses_memory_assistant.server:main" in text
    assert "ai-glasses-memory-assistant-fastapi" not in text
    assert "ai_glasses_memory_assistant.app:main" not in text
    assert "ai_glasses_memory_assistant.evals" in text
    assert "[tool.setuptools.package-data]" in text
