from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from ai_glasses_memory_assistant.app_home import get_app_home, get_data_dir


class AppHomeTests(unittest.TestCase):
    def test_ai_glasses_home_takes_priority(self) -> None:
        with patch.dict("os.environ", {"AI_GLASSES_HOME": "/tmp/glasses", "HERMES_HOME": "/tmp/hermes"}, clear=True):
            self.assertEqual(get_app_home(), Path("/tmp/glasses"))
            self.assertEqual(get_data_dir(), Path("/tmp/glasses/data"))

    def test_legacy_hermes_home_fallback_keeps_existing_tests_isolated(self) -> None:
        with patch.dict("os.environ", {"HERMES_HOME": "/tmp/hermes"}, clear=True), patch("pathlib.Path.home", return_value=Path("/tmp/home")):
            self.assertEqual(get_app_home(), Path("/tmp/home/.ai-glasses-memory-assistant"))
            self.assertEqual(get_data_dir(), Path("/tmp/hermes/ai_glasses_memory_assistant"))

    def test_default_home_is_independent_from_hermes(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch("pathlib.Path.home", return_value=Path("/tmp/home")):
            self.assertEqual(get_app_home(), Path("/tmp/home/.ai-glasses-memory-assistant"))
            self.assertEqual(get_data_dir(), Path("/tmp/home/.ai-glasses-memory-assistant/data"))


if __name__ == "__main__":
    unittest.main()
