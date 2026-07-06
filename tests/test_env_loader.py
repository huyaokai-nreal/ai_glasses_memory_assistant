from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_glasses_memory_assistant.app_home import APP_HOME_ENV
from ai_glasses_memory_assistant.env_loader import (
    candidate_env_paths,
    load_app_dotenv,
    restore_app_llm_env,
    snapshot_app_llm_env,
)


class EnvLoaderTest(unittest.TestCase):
    def test_loads_app_home_dotenv_without_overriding_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                APP_HOME_ENV: tmpdir,
                "AI_GLASSES_LLM_MODEL": "from-shell",
            },
            clear=True,
        ):
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join([
                    "# app-owned env",
                    "AI_GLASSES_LLM_PROVIDER=custom",
                    "AI_GLASSES_LLM_MODEL=from-dotenv",
                    "export AI_GLASSES_LLM_API_KEY='dotenv-key'",
                    'AI_GLASSES_LLM_BASE_URL="http://127.0.0.1:11434/v1"',
                    "AI_GLASSES_LLM_API_MODE=chat_completions",
                    "AI_GLASSES_LLM_REASONING_ENABLED=false",
                    "INVALID_LINE_WITHOUT_EQUALS",
                ]),
                encoding="utf-8",
            )

            loaded = load_app_dotenv()

            self.assertEqual(loaded, [env_path])
            self.assertEqual(os.environ["AI_GLASSES_LLM_PROVIDER"], "custom")
            self.assertEqual(os.environ["AI_GLASSES_LLM_MODEL"], "from-shell")
            self.assertEqual(os.environ["AI_GLASSES_LLM_API_KEY"], "dotenv-key")
            self.assertEqual(os.environ["AI_GLASSES_LLM_BASE_URL"], "http://127.0.0.1:11434/v1")
            self.assertEqual(os.environ["AI_GLASSES_LLM_API_MODE"], "chat_completions")
            self.assertEqual(os.environ["AI_GLASSES_LLM_REASONING_ENABLED"], "false")
            self.assertNotIn("INVALID_LINE_WITHOUT_EQUALS", os.environ)

    def test_loads_explicit_paths_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {}, clear=True):
            first = Path(tmpdir) / "first.env"
            second = Path(tmpdir) / "second.env"
            missing = Path(tmpdir) / "missing.env"
            first.write_text("AI_GLASSES_LLM_PROVIDER=first\n", encoding="utf-8")
            second.write_text("AI_GLASSES_LLM_PROVIDER=second\nAI_GLASSES_LLM_MODEL=model2\n", encoding="utf-8")

            loaded = load_app_dotenv([missing, first, second])

            self.assertEqual(loaded, [first, second])
            self.assertEqual(os.environ["AI_GLASSES_LLM_PROVIDER"], "first")
            self.assertEqual(os.environ["AI_GLASSES_LLM_MODEL"], "model2")

    def test_candidate_paths_include_app_home_before_package_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {APP_HOME_ENV: tmpdir}, clear=True):
            paths = candidate_env_paths()

        self.assertEqual(paths[0], Path(tmpdir) / ".env")
        self.assertEqual(paths[1].name, ".env")
        self.assertEqual(paths[1].parent.name, "ai_glasses_memory_assistant")

    def test_restore_app_llm_env_preserves_app_values_and_keeps_fallback_new_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AI_GLASSES_LLM_MODEL": "app-model",
                "DEEPSEEK_API_KEY": "provider-key",
            },
            clear=True,
        ):
            snapshot = snapshot_app_llm_env()
            os.environ["AI_GLASSES_LLM_MODEL"] = "hermes-model"
            os.environ["AI_GLASSES_LLM_API_KEY"] = "hermes-key"

            restore_app_llm_env(snapshot)

            self.assertEqual(os.environ["AI_GLASSES_LLM_MODEL"], "app-model")
            self.assertEqual(os.environ["AI_GLASSES_LLM_API_KEY"], "hermes-key")
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "provider-key")


if __name__ == "__main__":
    unittest.main()
