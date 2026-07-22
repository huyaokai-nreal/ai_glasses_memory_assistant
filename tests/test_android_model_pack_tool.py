from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).parents[1] / "android" / "tools" / "install_local_model_pack.py"
SPEC = importlib.util.spec_from_file_location("install_local_model_pack", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
model_pack_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = model_pack_tool
SPEC.loader.exec_module(model_pack_tool)


def _write_pack(tmp_path: Path, *, content: bytes = b"model") -> tuple[Path, dict]:
    pack_dir = tmp_path / "pack"
    model_path = pack_dir / "models" / "shared.onnx"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(content)
    relative = "models/shared.onnx"
    digest = hashlib.sha256(content).hexdigest()
    components = {
        name: {"engine": name, "roles": {"model": relative}}
        for name in model_pack_tool.REQUIRED_COMPONENTS
    }
    manifest = {
        "schema": model_pack_tool.SCHEMA,
        "version": "test-pack-v1",
        "sherpa_onnx_version": model_pack_tool.SHERPA_VERSION,
        "install_mode": model_pack_tool.INSTALL_MODE,
        "files": [{"path": relative, "sha256": digest, "size_bytes": len(content)}],
        "components": components,
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pack_dir, manifest


def _args(pack_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        adb="adb",
        serial="serial-1",
        package="com.aiglasses.memoryassistant.demo",
        pack_dir=pack_dir,
    )


def test_load_and_verify_pack_accepts_valid_local_pack(tmp_path: Path) -> None:
    pack_dir, manifest = _write_pack(tmp_path)

    assert model_pack_tool.load_and_verify_pack(pack_dir) == manifest


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda manifest: manifest.update(sherpa_onnx_version="1.13.3"), "sherpa-onnx"),
        (lambda manifest: manifest["files"][0].update(path="../shared.onnx"), "unsafe model path"),
        (lambda manifest: manifest["files"][0].update(sha256="0" * 64), "SHA-256 mismatch"),
        (lambda manifest: manifest["files"].append(dict(manifest["files"][0])), "duplicate model path"),
    ],
)
def test_load_and_verify_pack_rejects_invalid_pack(tmp_path: Path, change, message: str) -> None:
    pack_dir, manifest = _write_pack(tmp_path)
    change(manifest)
    (pack_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        model_pack_tool.load_and_verify_pack(pack_dir)


def test_adb_command_propagates_child_process_failure(monkeypatch) -> None:
    failure = subprocess.CalledProcessError(1, ["adb"])
    monkeypatch.setattr(model_pack_tool.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(failure))

    with pytest.raises(subprocess.CalledProcessError):
        model_pack_tool.adb_command("adb", "serial-1", "get-state")


def test_install_switches_pointer_only_after_device_hash_matches(tmp_path: Path, monkeypatch) -> None:
    pack_dir, manifest = _write_pack(tmp_path)
    calls: list[tuple[str, ...]] = []

    def fake_adb_command(adb: str, serial: str, *args: str, capture: bool = False):
        calls.append(args)
        stdout = ""
        if args == ("get-state",):
            stdout = "device\n"
        elif args[:4] == ("shell", "run-as", _args(pack_dir).package, "sha256sum"):
            stdout = f"{manifest['files'][0]['sha256']}  model\n"
        return subprocess.CompletedProcess([adb, serial, *args], 0, stdout=stdout, stderr="")

    monkeypatch.setattr(model_pack_tool, "adb_command", fake_adb_command)
    monkeypatch.setattr(model_pack_tool.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0))

    model_pack_tool.install(_args(pack_dir), manifest)

    destination_move = next(
        index
        for index, args in enumerate(calls)
        if len(args) >= 6 and args[-3] == "mv" and ".adb-staging-" in args[-2]
    )
    pointer_move = next(index for index, args in enumerate(calls) if args[-2:] == ("files/models/.current.json.part", "files/models/current.json"))
    assert destination_move < pointer_move


def test_install_keeps_current_pointer_when_device_hash_fails(tmp_path: Path, monkeypatch) -> None:
    pack_dir, manifest = _write_pack(tmp_path)
    calls: list[tuple[str, ...]] = []
    cleanup_calls: list[tuple] = []

    def fake_adb_command(adb: str, serial: str, *args: str, capture: bool = False):
        calls.append(args)
        stdout = "device\n" if args == ("get-state",) else ""
        if "sha256sum" in args:
            stdout = f"{'0' * 64}  model\n"
        return subprocess.CompletedProcess([adb, serial, *args], 0, stdout=stdout, stderr="")

    monkeypatch.setattr(model_pack_tool, "adb_command", fake_adb_command)
    monkeypatch.setattr(
        model_pack_tool.subprocess,
        "run",
        lambda *args, **kwargs: cleanup_calls.append(args) or subprocess.CompletedProcess([], 0),
    )

    with pytest.raises(RuntimeError, match="device SHA-256 mismatch"):
        model_pack_tool.install(_args(pack_dir), manifest)

    assert not any(args[-2:] == ("files/models/.current.json.part", "files/models/current.json") for args in calls)
    assert cleanup_calls
