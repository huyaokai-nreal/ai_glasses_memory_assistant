from __future__ import annotations

import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).parents[1] / "android" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from run_device_acceptance import _devtools_error_message


def test_devtools_error_message_includes_a_bounded_exception_description() -> None:
    message = _devtools_error_message(
        "DevTools returned an error",
        {"result": {"result": {"description": "TypeError: bridge method is undefined"}}},
    )

    assert message == "DevTools returned an error: TypeError: bridge method is undefined"


def test_devtools_error_message_limits_long_payloads() -> None:
    message = _devtools_error_message("DevTools expression failed", {"exceptionDetails": {"text": "x" * 700}})

    assert len(message) == len("DevTools expression failed: ") + 600
