from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).parents[1] / "android" / "tools"
TOOL_PATH = TOOLS_DIR / "pull_device_diagnostics.py"
sys.path.insert(0, str(TOOLS_DIR))
SPEC = importlib.util.spec_from_file_location("pull_device_diagnostics", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_select_device_requires_serial_only_for_multiple_devices() -> None:
    devices = [
        {"serial": "one", "model": "Phone1"},
        {"serial": "two", "model": "Phone2"},
    ]

    with pytest.raises(RuntimeError, match="pass --serial"):
        tool.select_device(devices, None)

    assert tool.select_device(devices, "two")["model"] == "Phone2"
    assert tool.select_device([devices[0]], None)["serial"] == "one"


@pytest.mark.parametrize("job_status", ["saved", "skipped", "rejected", "failed"])
def test_wait_for_capture_stop_accepts_every_terminal_memory_status(job_status: str) -> None:
    class FakeDevtools:
        def evaluate(self, expression):
            return {
                "audio": {
                    "state": "idle",
                    "last_stopped_capture_id": "capture-1",
                    "last_stop_status": "completed",
                    "last_stop_memory_job_id": "job-1",
                    "device_event_queue": {"pending": 0, "running": 0},
                },
                "memory_job": {"job_id": "job-1", "status": job_status},
            }

    state, timeout_stage = tool.wait_for_capture_stop(
        FakeDevtools(),
        capture_id="capture-1",
        timeout_seconds=1,
        poll_interval=0,
    )

    assert timeout_stage == ""
    assert state["memory_job"]["status"] == job_status


def test_wait_for_capture_stop_reports_failure() -> None:
    class FakeDevtools:
        def evaluate(self, expression):
            return {
                "audio": {
                    "state": "idle",
                    "last_stopped_capture_id": "capture-1",
                    "last_stop_status": "failed",
                    "last_stop_error": "database locked",
                },
                "memory_job": None,
            }

    with pytest.raises(RuntimeError, match="database locked"):
        tool.wait_for_capture_stop(FakeDevtools(), capture_id="capture-1", timeout_seconds=1)


def test_wait_for_capture_stop_returns_partial_timeout_evidence(monkeypatch) -> None:
    class FakeDevtools:
        def evaluate(self, expression):
            return {
                "audio": {
                    "state": "stopping",
                    "last_stopped_capture_id": "",
                    "last_stop_status": "",
                    "device_event_queue": {"pending": 1, "running": 0},
                },
                "memory_job": None,
            }

    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(tool.time, "monotonic", lambda: next(ticks))

    state, timeout_stage = tool.wait_for_capture_stop(
        FakeDevtools(),
        capture_id="capture-1",
        timeout_seconds=1,
        poll_interval=0,
    )

    assert timeout_stage == "capture_stop_or_memory_job_timeout"
    assert state["audio"]["state"] == "stopping"


def test_pull_remote_file_streams_binary_without_a_shell(tmp_path: Path, monkeypatch) -> None:
    payload = b"PK\x03\x04binary"
    calls = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, stdout, stderr):
            calls.append(command)
            stdout.write(payload)

        def communicate(self):
            return None, b""

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    destination = tmp_path / "bundle.zip"
    relative = "cache/adb-diagnostics/ai-glasses-diagnostic-" + "a" * 32 + ".zip"

    tool.pull_remote_file("adb", "serial-1", relative, destination)

    assert destination.read_bytes() == payload
    assert calls == [["adb", "-s", "serial-1", "exec-out", "run-as", tool.PACKAGE, "cat", relative]]


def test_archive_validation_and_extraction_reject_path_traversal(tmp_path: Path) -> None:
    valid = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid, "w") as archive:
        archive.writestr("diagnostics.json", "{}")
        archive.writestr("audit/chat_audit.redacted.jsonl", "{}\n")

    assert tool.validate_archive(valid) == ["diagnostics.json", "audit/chat_audit.redacted.jsonl"]
    extracted = tool.extract_archive(valid, tmp_path / "out")
    assert extracted == ["diagnostics.json", "audit/chat_audit.redacted.jsonl"]

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("diagnostics.json", "{}")
        archive.writestr("../secret.txt", "secret")
    with pytest.raises(ValueError, match="unsafe diagnostic archive member"):
        tool.validate_archive(unsafe)

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not-a-zip")
    with pytest.raises(ValueError, match="not a valid ZIP"):
        tool.validate_archive(corrupt)


def test_public_collection_state_omits_transcripts_and_private_payloads() -> None:
    state = {
        "audio": {
            "state": "idle",
            "capture_id": "capture-1",
            "final_query": "private question",
            "latest_partial": "private partial",
            "device_event_queue": {"pending": 0},
        },
        "memory_job": {
            "job_id": "job-1",
            "status": "rejected",
            "rejected_reasons": ["audio_speaker_unknown"],
            "memory_processing": {"raw": "private"},
        },
    }

    public = tool.public_collection_state(state)
    encoded = json.dumps(public)

    assert public["audio"]["capture_id"] == "capture-1"
    assert public["memory_job"]["rejected_reasons"] == ["audio_speaker_unknown"]
    assert "private question" not in encoded
    assert "private partial" not in encoded
    assert "memory_processing" not in encoded


def test_remote_snapshot_cleanup_falls_back_to_bounded_run_as_rm(monkeypatch) -> None:
    relative = "cache/adb-diagnostics/ai-glasses-diagnostic-" + "b" * 32 + ".zip"
    calls = []

    class FakeDevtools:
        pass

    class FakeAdb:
        def shell(self, *args, check=True):
            calls.append((args, check))
            return ""

    monkeypatch.setattr(tool, "delete_remote_snapshot", lambda devtools, path: False)

    assert tool.remove_remote_snapshot(FakeDevtools(), FakeAdb(), relative) is True
    assert calls == [(('run-as', tool.PACKAGE, 'rm', '-f', relative), False)]
    assert tool.remove_remote_snapshot(FakeDevtools(), FakeAdb(), "cache/other.zip") is False


def test_atomic_json_writer_replaces_existing_pointer(tmp_path: Path) -> None:
    destination = tmp_path / "latest.json"
    destination.write_text("old", encoding="utf-8")

    tool.write_json(destination, {"path": "/new"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"path": "/new"}
    assert not destination.with_suffix(".json.part").exists()
