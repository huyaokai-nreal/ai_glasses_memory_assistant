from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_glasses_memory_assistant import agent_bridge
from ai_glasses_memory_assistant.app_home import get_app_home, get_data_dir
from ai_glasses_memory_assistant.env_loader import APP_LLM_ENV_NAMES, candidate_env_paths, load_app_dotenv
from ai_glasses_memory_assistant.server_config import parse_server_bind, startup_message


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_app_home_contract_and_legacy_fallback() -> None:
    with patch.dict("os.environ", {"AI_GLASSES_HOME": "/tmp/glasses", "HERMES_HOME": "/tmp/hermes"}, clear=True):
        assert get_app_home() == Path("/tmp/glasses")
        assert get_data_dir() == Path("/tmp/glasses/data")

    with patch.dict("os.environ", {"HERMES_HOME": "/tmp/hermes"}, clear=True), patch("pathlib.Path.home", return_value=Path("/tmp/home")):
        assert get_app_home() == Path("/tmp/home/.ai-glasses-memory-assistant")
        assert get_data_dir() == Path("/tmp/hermes/ai_glasses_memory_assistant")


def test_app_home_dotenv_loads_without_overriding_existing_env(tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("AI_GLASSES_LLM_MODEL=from-file\nAI_GLASSES_LLM_API_KEY=from-file\n", encoding="utf-8")

    with patch.dict("os.environ", {"AI_GLASSES_HOME": str(home), "AI_GLASSES_LLM_API_KEY": "already-set"}, clear=True):
        assert candidate_env_paths()[0] == env_file
        assert load_app_dotenv() == [env_file]
        assert os.environ["AI_GLASSES_LLM_MODEL"] == "from-file"
        assert os.environ["AI_GLASSES_LLM_API_KEY"] == "already-set"


def test_server_bind_contract_keeps_single_stdlib_entrypoint() -> None:
    default_bind = parse_server_bind([])
    assert default_bind.host == "0.0.0.0"
    assert default_bind.port == 8765
    assert "Same LAN device" in startup_message(default_bind, lan_ip="192.168.1.23")

    https_bind = parse_server_bind(["--certfile", "certs/cert.pem", "--keyfile", "certs/key.pem"])
    assert "https://192.168.1.23:8765" in startup_message(https_bind, lan_ip="192.168.1.23")
    assert "LAN HTTP note" not in startup_message(https_bind, lan_ip="192.168.1.23")

    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parse_server_bind(["--certfile", "certs/cert.pem"])


def test_docs_keep_current_startup_contract_visible() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            PACKAGE_ROOT / "README.md",
            PACKAGE_ROOT / "docs" / "context" / "README.md",
            PACKAGE_ROOT / "docs" / "context" / "code-map.md",
            PACKAGE_ROOT / "docs" / "context" / "system-flow-current.md",
        ]
    )
    context_docs = sorted(path.name for path in (PACKAGE_ROOT / "docs" / "context").glob("*.md"))

    assert context_docs == ["README.md", "code-map.md", "system-flow-current.md"]
    assert "python -m ai_glasses_memory_assistant.server" in docs
    assert "python -m ai_glasses_memory_assistant.app" not in docs
    assert "fastapi" not in docs.lower()
    assert "uvicorn" not in docs.lower()
    assert "tests/test_core_startup.py" in docs
    assert "tests/test_agent_bridge_policy.py" not in docs
    assert agent_bridge.OPENAI_COMPATIBLE_BACKEND in docs
    assert "AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1" in docs
    for env_name in APP_LLM_ENV_NAMES:
        assert env_name in docs
