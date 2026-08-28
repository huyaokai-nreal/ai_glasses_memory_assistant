from __future__ import annotations

import contextlib
import errno
import http.client
import io
import json
import os
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_glasses_memory_assistant import agent_bridge, server
from ai_glasses_memory_assistant import android_runtime, mobile_runtime
from ai_glasses_memory_assistant.app_home import get_app_home, get_data_dir
from ai_glasses_memory_assistant.env_loader import APP_LLM_ENV_NAMES, candidate_env_paths, load_app_dotenv
from ai_glasses_memory_assistant.server import ExclusiveThreadingHTTPServer, ThreadingHTTPSServer
from ai_glasses_memory_assistant.server_config import parse_server_bind, startup_message


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_android_diagnostic_bundle_removes_voiceprints_and_secrets(tmp_path) -> None:
    app_home = tmp_path / "runtime"
    data_dir = app_home / "data"
    data_dir.mkdir(parents=True)
    timeline_path = data_dir / "timeline.db"
    with sqlite3.connect(timeline_path) as connection:
        connection.executescript(
            """
            CREATE TABLE speaker_profiles (user_id TEXT, embedding TEXT);
            CREATE TABLE speaker_enrollment_samples (user_id TEXT, embedding TEXT);
            CREATE TABLE device_audio_events (
                event_id TEXT,
                private_payload TEXT,
                event_payload TEXT,
                dispatch_payload TEXT
            );
            CREATE TABLE chunks (metadata TEXT);
            """
        )
        connection.execute("INSERT INTO speaker_profiles VALUES (?, ?)", ("u1", "[0.1,0.2]"))
        connection.execute("INSERT INTO speaker_enrollment_samples VALUES (?, ?)", ("u1", "[0.3,0.4]"))
        connection.execute(
            "INSERT INTO device_audio_events VALUES (?, ?, ?, ?)",
            (
                "event-1",
                json.dumps({"speaker_embedding": [0.5, 0.6]}),
                json.dumps({"text": "hello"}),
                json.dumps({"embedding": [0.7, 0.8], "state": "completed"}),
            ),
        )
        connection.execute(
            "INSERT INTO chunks VALUES (?)",
            (json.dumps({"speaker_embedding": [0.9, 1.0], "speaker": "user"}),),
        )
    events_path = data_dir / "events.db"
    with sqlite3.connect(events_path) as connection:
        connection.execute("CREATE TABLE memory_voice_profiles (id TEXT, embedding_json TEXT)")
        connection.execute("INSERT INTO memory_voice_profiles VALUES (?, ?)", ("voice-1", "[1.0,0.0]"))
    (data_dir / "chat_audit.jsonl").write_text(
        json.dumps({"authorization": "Bearer secret-token", "speaker_embedding": [0.2, 0.4]}) + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "diagnostic.zip"

    result = json.loads(android_runtime.create_diagnostic_bundle(
        str(app_home),
        str(destination),
        json.dumps({"api_key": "secret-key", "model": "X4000"}),
    ))

    assert result["redaction"]["voice_profile_rows_removed"] == 3
    assert result["redaction"]["private_audio_payloads_cleared"] == 1
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            "database/events.db",
            "database/timeline.db",
            "audit/chat_audit.redacted.jsonl",
            "diagnostics.json",
        }
        metadata = json.loads(archive.read("diagnostics.json"))
        assert metadata["device"]["api_key"] == "[REDACTED]"
        audit = archive.read("audit/chat_audit.redacted.jsonl").decode("utf-8")
        assert "secret-token" not in audit
        assert "0.2" not in audit
        extracted = tmp_path / "exported-timeline.db"
        extracted.write_bytes(archive.read("database/timeline.db"))
    with sqlite3.connect(extracted) as connection:
        assert connection.execute("SELECT COUNT(*) FROM speaker_profiles").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM speaker_enrollment_samples").fetchone()[0] == 0
        assert connection.execute("SELECT private_payload FROM device_audio_events").fetchone()[0] == "{}"
        dispatch = json.loads(connection.execute("SELECT dispatch_payload FROM device_audio_events").fetchone()[0])
        assert dispatch["embedding"] == "[REDACTED]"
        chunk = json.loads(connection.execute("SELECT metadata FROM chunks").fetchone()[0])
        assert chunk["speaker_embedding"] == "[REDACTED]"


def test_android_usage_snapshot_keeps_feedback_text_memories_discussions_and_audit(tmp_path) -> None:
    app_home = tmp_path / "runtime"
    data_dir = app_home / "data"
    data_dir.mkdir(parents=True)
    timeline_path = data_dir / "timeline.db"
    with sqlite3.connect(timeline_path) as connection:
        connection.executescript(
            """
            CREATE TABLE raw_turns (
                id TEXT, user_id TEXT, source TEXT, raw_text TEXT, assistant_reply TEXT, created_at REAL
            );
            CREATE TABLE reply_feedback (
                id TEXT, user_id TEXT, turn_id TEXT, rating TEXT, note TEXT, created_at REAL, updated_at REAL
            );
            CREATE TABLE chunks (metadata TEXT, text TEXT);
            CREATE TABLE device_audio_events (private_payload TEXT, event_payload TEXT, dispatch_payload TEXT);
            CREATE TABLE discussion_days (id TEXT, overview TEXT);
            CREATE TABLE discussion_topics (id TEXT, title TEXT, summary TEXT);
            CREATE TABLE speaker_profiles (user_id TEXT, embedding TEXT);
            """
        )
        connection.execute(
            "INSERT INTO raw_turns VALUES (?, ?, ?, ?, ?, ?)",
            ("turn-1", "u1", "chat", "沃尔玛会议具体讲了什么？", "讨论了供应链和门店策略。", 1.0),
        )
        connection.execute(
            "INSERT INTO reply_feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("feedback-1", "u1", "turn-1", "needs_improvement", "没有引用每日回顾。", 2.0, 3.0),
        )
        connection.execute("INSERT INTO chunks VALUES (?, ?)", (json.dumps({"speaker": "unknown"}), "ASR 最终原文"))
        connection.execute(
            "INSERT INTO device_audio_events VALUES (?, ?, ?)",
            (
                json.dumps({"speaker_embedding": [0.1, 0.2], "safe": "保留"}),
                json.dumps({"text": "ASR 最终原文", "raw_pcm": "bytes"}),
                json.dumps({"state": "completed"}),
            ),
        )
        connection.execute("INSERT INTO discussion_days VALUES (?, ?)", ("day-1", "沃尔玛会议每日回顾"))
        connection.execute("INSERT INTO discussion_topics VALUES (?, ?, ?)", ("topic-1", "沃尔玛会议", "讨论供应链"))
        connection.execute("INSERT INTO speaker_profiles VALUES (?, ?)", ("u1", "[0.1,0.2]"))
    events_path = data_dir / "events.db"
    with sqlite3.connect(events_path) as connection:
        connection.execute("CREATE TABLE memories (id TEXT, content TEXT)")
        connection.execute("INSERT INTO memories VALUES (?, ?)", ("memory-1", "我下午六点开会"))
    with sqlite3.connect(data_dir / "sessions.db") as connection:
        connection.execute("CREATE TABLE sessions (id TEXT, content TEXT)")
        connection.execute("INSERT INTO sessions VALUES (?, ?)", ("session-1", "完整会话原文"))
    (data_dir / "chat_audit.jsonl").write_text(
        json.dumps({"record_type": "reply_feedback", "timeline_turn_id": "turn-1", "note": "没有引用每日回顾。", "authorization": "Bearer secret", "speaker_embedding": [0.3]}) + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "usage.zip"

    result = json.loads(android_runtime.create_usage_data_bundle(
        str(app_home), str(destination), json.dumps({"api_key": "secret-key", "model": "X4000"})
    ))

    assert result["schema"] == "ai_glasses_usage_snapshot.v1"
    with zipfile.ZipFile(destination) as archive:
        assert {"database/timeline.db", "database/events.db", "database/sessions.db", "audit/chat_audit.jsonl", "feedback_index.json", "analysis_request.md", "manifest.json"}.issubset(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["feedback_needs_improvement_count"] == 1
        assert manifest["record_counts"]["timeline.db"]["discussion_days"] == 1
        assert manifest["device"]["api_key"] == "[EXCLUDED]"
        feedback = json.loads(archive.read("feedback_index.json"))
        assert feedback == [{
            "assistant_reply": "讨论了供应链和门店策略。",
            "feedback_created_at": 2.0,
            "feedback_id": "feedback-1",
            "feedback_updated_at": 3.0,
            "note": "没有引用每日回顾。",
            "rating": "needs_improvement",
            "source": "chat",
            "turn_created_at": 1.0,
            "turn_id": "turn-1",
            "user_id": "u1",
            "user_message": "沃尔玛会议具体讲了什么？",
        }]
        audit = archive.read("audit/chat_audit.jsonl").decode("utf-8")
        assert "没有引用每日回顾。" in audit
        assert "Bearer secret" not in audit
        assert "0.3" not in audit
        exported_timeline = tmp_path / "usage-timeline.db"
        exported_timeline.write_bytes(archive.read("database/timeline.db"))
        exported_events = tmp_path / "usage-events.db"
        exported_events.write_bytes(archive.read("database/events.db"))
    with sqlite3.connect(exported_timeline) as connection:
        assert connection.execute("SELECT raw_text FROM raw_turns").fetchone()[0] == "沃尔玛会议具体讲了什么？"
        assert connection.execute("SELECT note FROM reply_feedback").fetchone()[0] == "没有引用每日回顾。"
        assert connection.execute("SELECT overview FROM discussion_days").fetchone()[0] == "沃尔玛会议每日回顾"
        private_payload, event_payload = connection.execute("SELECT private_payload, event_payload FROM device_audio_events").fetchone()
        assert json.loads(private_payload)["speaker_embedding"] == "[EXCLUDED]"
        assert json.loads(private_payload)["safe"] == "保留"
        assert json.loads(event_payload)["raw_pcm"] == "[EXCLUDED]"
        assert connection.execute("SELECT COUNT(*) FROM speaker_profiles").fetchone()[0] == 0
    with sqlite3.connect(exported_events) as connection:
        assert connection.execute("SELECT content FROM memories").fetchone()[0] == "我下午六点开会"


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


def test_demo_llm_config_requires_openai_compatible_fields() -> None:
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError) as exc_info:
        agent_bridge._demo_llm_config()

    message = str(exc_info.value)
    assert "OpenAI-compatible LLM requires" in message
    assert "AI_GLASSES_LLM_MODEL" in message
    assert "AI_GLASSES_LLM_BASE_URL" in message
    assert "AI_GLASSES_LLM_API_KEY or DEEPSEEK_API_KEY" in message


def test_demo_llm_config_uses_deepseek_api_key_fallback() -> None:
    with patch.dict(
        "os.environ",
        {
            "AI_GLASSES_LLM_PROVIDER": "deepseek",
            "AI_GLASSES_LLM_MODEL": "deepseek-v4-flash",
            "AI_GLASSES_LLM_BASE_URL": "https://api.deepseek.com/",
            "DEEPSEEK_API_KEY": "test-key",
        },
        clear=True,
    ):
        config = agent_bridge._demo_llm_config()

    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-flash"
    assert config.base_url == "https://api.deepseek.com"
    assert config.api_key == "test-key"
    assert config.api_mode == agent_bridge.SUPPORTED_LLM_API_MODE
    assert config.transport == "openai_sdk"


def test_demo_llm_config_accepts_embedded_stdlib_transport() -> None:
    with patch.dict(
        "os.environ",
        {
            "AI_GLASSES_LLM_PROVIDER": "deepseek",
            "AI_GLASSES_LLM_MODEL": "deepseek-v4-flash",
            "AI_GLASSES_LLM_BASE_URL": "https://api.deepseek.com",
            "AI_GLASSES_LLM_API_KEY": "test-key",
            "AI_GLASSES_LLM_TRANSPORT": "stdlib_http",
        },
        clear=True,
    ):
        config = agent_bridge._demo_llm_config()

    assert config.transport == "stdlib_http"


def test_demo_llm_config_rejects_unsupported_api_mode() -> None:
    with patch.dict(
        "os.environ",
        {
            "AI_GLASSES_LLM_PROVIDER": "deepseek",
            "AI_GLASSES_LLM_MODEL": "deepseek-v4-flash",
            "AI_GLASSES_LLM_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_API_KEY": "test-key",
            "AI_GLASSES_LLM_API_MODE": "responses",
        },
        clear=True,
    ), pytest.raises(ValueError) as exc_info:
        agent_bridge._demo_llm_config()

    assert "AI_GLASSES_LLM_API_MODE='responses' is not supported" in str(exc_info.value)
    assert agent_bridge.SUPPORTED_LLM_API_MODE in str(exc_info.value)


def test_demo_llm_config_ignores_removed_backend_switch() -> None:
    with patch.dict(
        "os.environ",
        {
            "AI_GLASSES_LLM_BACKEND": "hermes",
            "AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK": "1",
            "AI_GLASSES_LLM_PROVIDER": "deepseek",
            "AI_GLASSES_LLM_MODEL": "deepseek-v4-flash",
            "AI_GLASSES_LLM_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_API_KEY": "test-key",
        },
        clear=True,
    ):
        config = agent_bridge._demo_llm_config()

    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-flash"
    assert not hasattr(config, "backend")


def test_server_bind_contract_keeps_single_stdlib_entrypoint() -> None:
    default_bind = parse_server_bind([])
    assert default_bind.host == "0.0.0.0"
    assert default_bind.port == 8765
    assert "Same LAN device" in startup_message(default_bind, lan_ip="192.168.1.23")
    assert "microphone require HTTPS" in startup_message(default_bind, lan_ip="192.168.1.23")
    assert ExclusiveThreadingHTTPServer.allow_reuse_address is False
    assert issubclass(ThreadingHTTPSServer, ExclusiveThreadingHTTPServer)

    https_bind = parse_server_bind(["--certfile", "certs/cert.pem", "--keyfile", "certs/key.pem"])
    assert "https://192.168.1.23:8765" in startup_message(https_bind, lan_ip="192.168.1.23")
    assert "LAN HTTP note" not in startup_message(https_bind, lan_ip="192.168.1.23")

    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parse_server_bind(["--certfile", "certs/cert.pem"])


def test_android_runtime_binds_loopback_and_requires_its_token(tmp_path) -> None:
    home = tmp_path / "home"
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("android runtime", encoding="utf-8")
    config = {
        "app_home": str(home),
        "static_dir": str(static),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": "test-key",
        "owner_id": "device-owner",
    }

    payload = json.loads(android_runtime.start(json.dumps(config)))
    parsed = server.urlparse(payload["base_url"])
    try:
        capture = json.loads(android_runtime.start_capture("device-owner"))
        public_device = json.loads(android_runtime.set_device_state(json.dumps({
            "capture_id": capture["capture_id"],
            "latest_partial": "识别中",
            "partial_sequence": 2,
            "enrollment_state": "recording",
            "enrollment_sample_count": 1,
            "audio_rms_dbfs": -32.5,
            "vad_segment_count": 2,
            "speaker_embedding": [1.0, 0.0],
        })))
        assert public_device["latest_partial"] == "识别中"
        assert public_device["audio_rms_dbfs"] == -32.5
        assert "speaker_embedding" not in public_device

        unauthorized = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
        unauthorized.request("GET", "/api/runtime")
        assert unauthorized.getresponse().status == 401
        unauthorized.close()

        authorized = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
        authorized.request(
            "GET",
            "/api/runtime",
            headers={"X-AI-Glasses-Local-Token": payload["local_token"]},
        )
        response = authorized.getresponse()
        runtime_payload = json.loads(response.read())
        authorized.close()

        assert response.status == 200
        assert runtime_payload["platform"] == "android"
        assert runtime_payload["audio_input_owner"] == "native"
        assert runtime_payload["owner_id"] == "device-owner"
        assert runtime_payload["device_event_queue"] == {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        assert runtime_payload["device"]["enrollment_state"] == "recording"
        assert runtime_payload["ambient_context"] == {
            "capture_id": capture["capture_id"],
            "status": "running",
            "chunk_count": 0,
            "last_chunk_id": "",
            "last_segment_id": "",
            "last_captured_at": None,
        }
        assert "speaker_embedding" not in runtime_payload["device"]
        assert "ambient_context" not in runtime_payload["device"]
        assert (home / "data" / "events.db").exists()
        assert (home / "data" / "timeline.db").exists()
        with pytest.raises(ValueError, match="device owner"):
            android_runtime.queue_status("another-owner")
    finally:
        assert json.loads(android_runtime.stop()) == {"running": False}


def test_android_runtime_rejects_insecure_or_incomplete_config() -> None:
    with pytest.raises(ValueError, match="missing"):
        android_runtime.start("{}")
    with pytest.raises(ValueError, match="must use https"):
        android_runtime.start(json.dumps({
            "app_home": "/tmp/app",
            "static_dir": "/tmp/static",
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_url": "http://api.example.test",
            "api_key": "test-key",
            "owner_id": "owner",
        }))


def test_mobile_runtime_forces_ios_platform_marker(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_start(raw: str) -> str:
        captured.update(json.loads(raw))
        return '{"running": true}'

    monkeypatch.setattr(mobile_runtime, "_start", fake_start)

    assert mobile_runtime.start(json.dumps({"platform": "android", "owner_id": "ios-owner"})) == '{"running": true}'
    assert captured == {"platform": "ios", "owner_id": "ios-owner"}


def test_server_exits_clearly_when_port_is_already_used() -> None:
    address_in_use = OSError(errno.EADDRINUSE, "Address already in use")
    with (
        patch.object(server, "parse_server_bind", return_value=parse_server_bind([])),
        patch.object(server, "ExclusiveThreadingHTTPServer", side_effect=address_in_use),
        pytest.raises(SystemExit, match="port is already used"),
    ):
        server.main()


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
    assert "OpenAI-compatible" in docs
    assert "AI_GLASSES_LLM_BACKEND" not in docs
    assert "AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK" not in docs
    assert "ollama" not in docs.lower()
    for env_name in APP_LLM_ENV_NAMES:
        assert env_name in docs


def test_cleanup_scanner_is_repo_local_and_read_only() -> None:
    script = PACKAGE_ROOT / "scripts" / "scan_cleanup_candidates.py"
    result = subprocess.run(
        [sys.executable, str(script), "--repo", str(PACKAGE_ROOT), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report["read_only"] is True
    assert report["repo"] == str(PACKAGE_ROOT)
    assert "safe_artifacts" in report
    assert "review_required" in report
    assert any(policy["scope"] == "memory_privacy_core" for policy in report["protected_policies"])
    assert any(item["path"] == "ai_glasses_memory_assistant/agent_bridge.py" for item in report["largest_files"])
    wrappers = report["review_required"]["thin_helper_wrappers"]
    assert isinstance(wrappers, list)
    if wrappers:
        assert all("python_reference_count" in item for item in wrappers)
        assert all("python_reference_files" in item for item in wrappers)
    groups = report["review_groups"]["thin_helper_wrappers"]
    assert set(groups) == {"zero_python_refs", "internal_only_python_refs", "test_referenced"}
    assert len(wrappers) == sum(len(items) for items in groups.values())
    assert report["summary"]["zero_reference_wrapper_count"] == len(groups["zero_python_refs"])
    assert report["summary"]["test_referenced_wrapper_count"] == len(groups["test_referenced"])
