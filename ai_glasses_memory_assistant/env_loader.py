from __future__ import annotations

import os
from pathlib import Path

from .app_home import get_app_home


APP_LLM_ENV_NAMES = (
    "AI_GLASSES_LLM_PROVIDER",
    "AI_GLASSES_LLM_MODEL",
    "AI_GLASSES_LLM_BASE_URL",
    "AI_GLASSES_LLM_API_KEY",
    "AI_GLASSES_LLM_API_MODE",
)


def candidate_env_paths() -> list[Path]:
    """Return .env files this app owns, ordered from most to least specific."""
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        get_app_home() / ".env",
    ]
    if not os.environ.get("AI_GLASSES_HOME") and not os.environ.get("HERMES_HOME"):
        paths.append(repo_root / ".env")
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def load_app_dotenv(paths: list[Path] | None = None) -> list[Path]:
    """Load app-owned .env files without overriding existing environment vars."""
    loaded: list[Path] = []
    for path in paths if paths is not None else candidate_env_paths():
        env_path = Path(path).expanduser()
        if not env_path.exists() or not env_path.is_file():
            continue
        _load_env_file(env_path)
        loaded.append(env_path)
    return loaded


def snapshot_app_llm_env() -> dict[str, str]:
    return {name: os.environ[name] for name in APP_LLM_ENV_NAMES if name in os.environ}


def restore_app_llm_env(snapshot: dict[str, str]) -> None:
    for name, value in snapshot.items():
        os.environ[name] = value


def _load_env_file(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig") as file:
        for raw_line in file:
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _strip_optional_quotes(value.strip())


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
