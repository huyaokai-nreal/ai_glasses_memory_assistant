#!/usr/bin/env python3
"""Atomically install a verified adb_local model pack into a debug APK."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Any


SCHEMA = "model_pack.v1"
INSTALL_MODE = "adb_local"
SHERPA_VERSION = "1.13.4"
REQUIRED_COMPONENTS = {"vad", "kws", "online_asr", "ambient_asr", "speaker"}
SAFE_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA256 = re.compile(r"[a-f0-9]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="ADB device serial")
    parser.add_argument("--pack-dir", type=Path, required=True, help="Directory containing manifest.json")
    parser.add_argument("--package", default="com.aiglasses.memoryassistant.demo")
    parser.add_argument("--adb", default="adb")
    return parser.parse_args()


def load_and_verify_pack(pack_dir: Path) -> dict[str, Any]:
    root = pack_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("install_mode") != INSTALL_MODE:
        raise ValueError(f"manifest must use {SCHEMA} with install_mode={INSTALL_MODE}")
    if manifest.get("sherpa_onnx_version") != SHERPA_VERSION:
        raise ValueError(f"manifest must use sherpa-onnx {SHERPA_VERSION}")
    version = str(manifest.get("version") or "")
    if not SAFE_PART.fullmatch(version):
        raise ValueError("manifest version is invalid")
    components = manifest.get("components")
    if not isinstance(components, dict) or not REQUIRED_COMPONENTS.issubset(components):
        raise ValueError("manifest is missing required model components")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a non-empty array")
    verified_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("manifest file must be an object")
        relative = safe_relative_path(str(item.get("path") or ""))
        if relative in verified_paths:
            raise ValueError(f"duplicate model path: {relative}")
        if str(item.get("url") or "").strip():
            raise ValueError(f"adb_local model file must not declare a URL: {relative}")
        digest = str(item.get("sha256") or "").lower()
        if not SHA256.fullmatch(digest):
            raise ValueError(f"invalid SHA-256: {relative}")
        expected_size = int(item.get("size_bytes") or 0)
        source = root / relative
        if not source.is_file() or source.stat().st_size != expected_size:
            raise ValueError(f"model file is missing or has wrong size: {relative}")
        if file_sha256(source) != digest:
            raise ValueError(f"model SHA-256 mismatch: {relative}")
        verified_paths.add(relative)

    for name, component in components.items():
        roles = component.get("roles") if isinstance(component, dict) else None
        if not isinstance(roles, dict) or not roles:
            raise ValueError(f"component roles are missing: {name}")
        for relative in roles.values():
            if safe_relative_path(str(relative)) not in verified_paths:
                raise ValueError(f"component {name} references an unknown file")
    return manifest


def safe_relative_path(raw: str) -> str:
    normalized = raw.strip().replace("\\", "/")
    parts = normalized.split("/")
    if not normalized or normalized.startswith("/") or not all(SAFE_PART.fullmatch(part) for part in parts):
        raise ValueError(f"unsafe model path: {raw}")
    return "/".join(parts)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adb_command(adb: str, serial: str, *args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [adb, "-s", serial, *args],
        check=True,
        text=True,
        capture_output=capture,
    )


def install(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    version = str(manifest["version"])
    token = uuid.uuid4().hex[:12]
    remote_tar = f"/data/local/tmp/ai-glasses-model-{token}.tar"
    remote_pointer = f"/data/local/tmp/ai-glasses-current-{token}.json"
    staging = f"files/models/.adb-staging-{token}"
    destination_name = f"{version}-{token}"
    destination = f"files/models/packs/{destination_name}"

    state = adb_command(args.adb, args.serial, "get-state", capture=True).stdout.strip()
    if state != "device":
        raise RuntimeError(f"ADB device is not ready: {state}")
    adb_command(args.adb, args.serial, "shell", "run-as", args.package, "pwd", capture=True)
    services = adb_command(
        args.adb,
        args.serial,
        "shell",
        "dumpsys",
        "activity",
        "services",
        args.package,
        capture=True,
    ).stdout
    if ".AudioCaptureService" in services:
        raise RuntimeError("continuous capture is running; stop it before installing models")

    with tempfile.TemporaryDirectory(prefix="ai-glasses-model-pack-") as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / "pack.tar"
        pointer = temporary_path / "current.json"
        pointer.write_text(
            json.dumps({"version": version, "directory": destination_name}, ensure_ascii=True),
            encoding="utf-8",
        )
        with tarfile.open(archive, "w") as output:
            output.add(args.pack_dir / "manifest.json", arcname="manifest.json")
            for item in manifest["files"]:
                relative = safe_relative_path(str(item["path"]))
                output.add(args.pack_dir / relative, arcname=relative)

        try:
            adb_command(args.adb, args.serial, "push", str(archive), remote_tar)
            adb_command(args.adb, args.serial, "push", str(pointer), remote_pointer)
            adb_command(args.adb, args.serial, "shell", "run-as", args.package, "mkdir", "-p", staging)
            adb_command(
                args.adb,
                args.serial,
                "shell",
                "run-as",
                args.package,
                "tar",
                "-xf",
                remote_tar,
                "-C",
                staging,
            )
            for item in manifest["files"]:
                relative = safe_relative_path(str(item["path"]))
                result = adb_command(
                    args.adb,
                    args.serial,
                    "shell",
                    "run-as",
                    args.package,
                    "sha256sum",
                    f"{staging}/{relative}",
                    capture=True,
                )
                if result.stdout.split()[0].lower() != str(item["sha256"]).lower():
                    raise RuntimeError(f"device SHA-256 mismatch: {relative}")
            adb_command(args.adb, args.serial, "shell", "run-as", args.package, "mkdir", "-p", "files/models/packs")
            adb_command(args.adb, args.serial, "shell", "run-as", args.package, "mv", staging, destination)
            adb_command(
                args.adb,
                args.serial,
                "shell",
                "run-as",
                args.package,
                "cp",
                remote_pointer,
                "files/models/.current.json.part",
            )
            adb_command(
                args.adb,
                args.serial,
                "shell",
                "run-as",
                args.package,
                "mv",
                "files/models/.current.json.part",
                "files/models/current.json",
            )
        except Exception:
            subprocess.run(
                [args.adb, "-s", args.serial, "shell", "run-as", args.package, "rm", "-rf", staging],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            raise
        finally:
            subprocess.run(
                [args.adb, "-s", args.serial, "shell", "rm", "-f", remote_tar, remote_pointer],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    print(f"Installed model pack {version} on {args.serial}")


def main() -> None:
    args = parse_args()
    args.pack_dir = args.pack_dir.resolve()
    manifest = load_and_verify_pack(args.pack_dir)
    install(args, manifest)


if __name__ == "__main__":
    main()
