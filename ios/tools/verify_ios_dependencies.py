#!/usr/bin/env python3
"""Verify native iOS dependency slices and the final app link layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ios/tools/ios_dependency_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plist(path: Path) -> dict:
    with path.open("rb") as source:
        value = plistlib.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"plist is not an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_dependencies(deps_dir: Path, manifest_path: Path) -> None:
    lock = json.loads(manifest_path.read_text(encoding="utf-8"))
    ort_root = deps_dir / "onnxruntime.xcframework/ios-arm64/onnxruntime.framework"
    sherpa_root = deps_dir / "sherpa-onnx.xcframework/ios-arm64"
    ort_binary = ort_root / "onnxruntime"
    ort_header = ort_root / "Headers/onnxruntime_c_api.h"
    sherpa_archive = sherpa_root / "sherpa-onnx.a"

    require(ort_binary.is_file(), f"missing ONNX Runtime binary: {ort_binary}")
    require(ort_header.is_file(), f"missing ONNX Runtime header: {ort_header}")
    require(sherpa_archive.is_file(), f"missing sherpa archive: {sherpa_archive}")

    ort_info = plist(ort_root / "Info.plist")
    actual_ort_version = str(ort_info.get("CFBundleShortVersionString") or "")
    require(actual_ort_version == lock["onnxruntime_version"], f"ORT version mismatch: {actual_ort_version}")

    header_text = ort_header.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"#define\s+ORT_API_VERSION\s+(\d+)", header_text)
    require(match is not None, "ORT_API_VERSION is missing from the header")
    require(int(match.group(1)) == int(lock["ort_api_version"]), "ORT API version mismatch")

    require(sha256(ort_binary) == lock["sha256"]["onnxruntime_binary"], "ORT binary checksum mismatch")
    require(sha256(ort_header) == lock["sha256"]["onnxruntime_header"], "ORT header checksum mismatch")
    require(sha256(sherpa_archive) == lock["sha256"]["sherpa_onnx_archive"], "sherpa checksum mismatch")

    lipo = subprocess.run(["lipo", "-info", str(ort_binary)], capture_output=True, text=True, check=True)
    require("arm64" in lipo.stdout, "ORT binary has no arm64 slice")
    print(f"verified sherpa-onnx {lock['sherpa_onnx_version']} + ORT {actual_ort_version} (API {match.group(1)})")


def verify_app(app_path: Path, manifest_path: Path) -> None:
    lock = json.loads(manifest_path.read_text(encoding="utf-8"))
    executable = app_path / app_path.stem
    require(executable.is_file(), f"missing app executable: {executable}")
    require(not (app_path / "Frameworks/onnxruntime.framework").exists(), "duplicate embedded ORT framework")
    require((app_path / "Frameworks/Python.framework").exists(), "missing embedded Python framework")
    require((app_path / "PythonRuntime").is_dir(), "missing bundled PythonRuntime")

    symbols = subprocess.run(["nm", "-gU", str(executable)], capture_output=True, text=True, check=True)
    require("_OrtGetApiBase" in symbols.stdout, "final app does not contain ORT API entrypoint")
    print(f"verified app link layout and bundled PythonRuntime (ORT {lock['onnxruntime_version']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deps-dir", type=Path, default=ROOT / "ios/.deps")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--app-path", type=Path)
    args = parser.parse_args()
    verify_dependencies(args.deps_dir.resolve(), args.manifest.resolve())
    if args.app_path:
        verify_app(args.app_path.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
