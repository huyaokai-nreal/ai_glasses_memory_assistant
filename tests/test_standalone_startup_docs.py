from __future__ import annotations

from pathlib import Path

from ai_glasses_memory_assistant import agent_bridge
from ai_glasses_memory_assistant.env_loader import APP_LLM_ENV_NAMES


DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "context"
DOC_PATH = DOCS_DIR / "standalone-startup.md"
DEPLOYMENT_DOC_PATH = DOCS_DIR / "standalone-deployment.md"
PACKAGING_DOC_PATH = DOCS_DIR / "standalone-packaging.md"
BASELINE_DOC_PATH = DOCS_DIR / "standalone-eval-baseline.md"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PACKAGE_ROOT / "README.md"
STANDALONE_PYPROJECT_PATH = PACKAGE_ROOT / "pyproject.standalone.toml"


def test_standalone_startup_doc_matches_runtime_env_contract() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    for env_name in APP_LLM_ENV_NAMES:
        assert env_name in text

    assert agent_bridge.OPENAI_COMPATIBLE_BACKEND in text
    assert agent_bridge.HERMES_BACKEND in text
    assert agent_bridge.DEEPSEEK_FALLBACK_PROVIDER in text
    assert agent_bridge.OLLAMA_DEFAULT_API_MODE in text
    assert agent_bridge.DEEPSEEK_API_KEY_ENV in text


def test_standalone_startup_doc_keeps_supported_entrypoints_current() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "python -m ai_glasses_memory_assistant.server" in text
    assert "python -m ai_glasses_memory_assistant.app" in text
    assert "--certfile" in text
    assert "--keyfile" in text


def test_standalone_deployment_doc_keeps_dependency_manifest_draft_current() -> None:
    text = DEPLOYMENT_DOC_PATH.read_text(encoding="utf-8")

    assert "python -m ai_glasses_memory_assistant.server" in text
    assert "python -m ai_glasses_memory_assistant.app" in text
    assert "AI_GLASSES_LLM_BACKEND=openai_compatible" in text
    assert "AI_GLASSES_LLM_BACKEND=hermes" in text
    assert "AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1" in text
    assert "legacy" in text
    assert "openai" in text
    assert "fastapi" in text
    assert "pydantic" in text
    assert "uvicorn" in text
    assert "edge-tts" in text
    assert "funasr" in text
    assert "pytest" in text
    assert "迁出前" in text
    assert "target_failed_turns" in text


def test_standalone_packaging_draft_keeps_entrypoints_and_extras_current() -> None:
    text = STANDALONE_PYPROJECT_PATH.read_text(encoding="utf-8")

    assert "openai" in text
    assert "fastapi" in text
    assert "pydantic" in text
    assert "uvicorn" in text
    assert "edge-tts" in text
    assert "funasr" in text
    assert "pytest" in text
    assert "ai-glasses-memory-assistant" in text
    assert "ai_glasses_memory_assistant.server:main" in text
    assert "ai-glasses-memory-assistant-fastapi" in text
    assert "ai_glasses_memory_assistant.app:main" in text
    assert "ai_glasses_memory_assistant.evals" in text
    assert "static/*.html" in text
    assert "static/*.css" in text
    assert "static/*.js" in text
    assert "evals/*.jsonl" in text
    assert "docs/context/*.md" in text


def test_standalone_packaging_doc_and_readme_keep_deployment_contract_visible() -> None:
    packaging_text = PACKAGING_DOC_PATH.read_text(encoding="utf-8")
    readme_text = README_PATH.read_text(encoding="utf-8")
    combined = packaging_text + "\n" + readme_text

    assert "pyproject.standalone.toml" in combined
    assert "AI_GLASSES_HOME" in combined
    assert "AI_GLASSES_LLM_BACKEND=openai_compatible" in combined
    assert "AI_GLASSES_LLM_BACKEND=hermes" in combined
    assert "AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1" in combined
    assert "legacy" in combined
    assert "AI_GLASSES_ASR_MODEL_DIR" in combined
    assert "AI_GLASSES_EMOTION_MODEL_DIR" in combined
    assert "AI_GLASSES_SPEAKER_MODEL_DIR" in combined
    assert "static/" in combined
    assert "FastAPI" in combined
    assert "target_failed_turns" in combined


def test_standalone_eval_baseline_doc_keeps_runner_contract_visible() -> None:
    text = BASELINE_DOC_PATH.read_text(encoding="utf-8")

    assert "python -m ai_glasses_memory_assistant.evals.runner" in text
    assert "--mode live" in text
    assert "--category memory_mechanism" in text
    assert "--category audit_replay" in text
    assert "--category public_preference_write" in text
    assert "--report-dir ai_glasses_memory_assistant/reports/standalone-migration-baseline" in text
    assert "eval-latest.json" in text
    assert "eval-latest.md" in text
    assert "summary.active_pass_rate" in text
    assert "summary.target_failed_turns" in text
    assert "runs[].turns[].failures" in text
    assert "runs[].turns[].response.debug" in text
    assert "runs[].turns[].new_memories" in text
    assert "runs[].turns[].response.recalled_memories" in text
    assert "debug.tools" in text
    assert "debug.location" in text
    assert "debug.weather" in text
    assert "AI_GLASSES_LLM_BACKEND" in text
    assert "DEEPSEEK_API_KEY" in text
    assert "方案" in text
    assert "没有执行正式 baseline" in text
