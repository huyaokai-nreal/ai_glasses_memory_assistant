from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TOOL_PATH = Path(__file__).parents[1] / "android" / "tools" / "run_device_acceptance.py"
SPEC = importlib.util.spec_from_file_location("run_device_acceptance", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_public_snapshot_expression_omits_private_payloads() -> None:
    expression = tool.PUBLIC_SNAPSHOT_EXPRESSION

    for forbidden in ("api_key", "speaker_embedding", "private_payload", "latitude:", "longitude:"):
        assert forbidden not in expression
    assert "audit_types" in expression
    assert "chunk_count" in expression


def test_report_contains_only_supplied_evidence(tmp_path: Path) -> None:
    report = {
        "device_serial": "serial-1",
        "started_at": "now",
        "checks": [{"name": "capture", "status": "pass", "evidence": {"chunk_count": 2}}],
    }

    json_path, markdown_path = tool.write_report(report, tmp_path)

    assert '"chunk_count": 2' in json_path.read_text(encoding="utf-8")
    assert "PASS" in markdown_path.read_text(encoding="utf-8")


def test_permission_granted_parses_package_runtime_permissions() -> None:
    class FakeAdb:
        def shell(self, *args, check=True):
            return """
              runtime permissions:
                android.permission.ACCESS_FINE_LOCATION: granted=false, flags=[]
                android.permission.ACCESS_COARSE_LOCATION: granted=true, flags=[]
            """

    adb = FakeAdb()
    assert tool.permission_granted(adb, "android.permission.ACCESS_COARSE_LOCATION") is True
    assert tool.permission_granted(adb, "android.permission.ACCESS_FINE_LOCATION") is False


def test_reconnect_removes_old_forward(monkeypatch) -> None:
    calls = []

    class FakeSocket:
        def close(self):
            calls.append("close")

    class FakeAdb:
        def run(self, *args, check=True):
            calls.append(args)

    replacement = object()
    monkeypatch.setattr(tool, "connect_devtools", lambda adb: (replacement, "45678"))

    connected, port = tool.reconnect_devtools(FakeAdb(), FakeSocket(), "12345")

    assert connected is replacement
    assert port == "45678"
    assert calls == ["close", ("forward", "--remove", "tcp:12345")]
