from __future__ import annotations

import os
from pathlib import Path


APP_HOME_ENV = "AI_GLASSES_HOME"
LEGACY_HERMES_HOME_ENV = "HERMES_HOME"
DEFAULT_APP_HOME_NAME = ".ai-glasses-memory-assistant"


def get_app_home() -> Path:
    """Return the app's own home directory."""
    explicit_home = os.environ.get(APP_HOME_ENV)
    if explicit_home:
        return Path(explicit_home).expanduser()

    return Path.home() / DEFAULT_APP_HOME_NAME


def get_data_dir() -> Path:
    legacy_home = os.environ.get(LEGACY_HERMES_HOME_ENV)
    if not os.environ.get(APP_HOME_ENV) and legacy_home:
        return Path(legacy_home).expanduser() / "ai_glasses_memory_assistant"

    return get_app_home() / "data"
