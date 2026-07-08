from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def isolated_app_home(tmpdir: str, extra_env: Mapping[str, str] | None = None, *, clear: bool = False) -> Iterator[None]:
    """Use the current app-owned home path for tests by default."""
    env = {**dict(extra_env or {}), "AI_GLASSES_HOME": tmpdir}
    with patch.dict("os.environ", env, clear=clear):
        yield
