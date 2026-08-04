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


def verify_numpy(numpy_dir: Path, expected_version: str, label: str) -> None:
    require(numpy_dir.is_dir(), f"missing {label} numpy directory: {numpy_dir}")
    version_py = numpy_dir / "version.py"
    require(version_py.is_file(), f"missing {label} numpy version.py")
    version_text = version_py.read_text(encoding="utf-8")
    version_match = re.search(r'version\s*=\s*[\"\']([^\"\']+)[\"\']', version_text)
    require(version_match is not None, f"cannot determine {label} numpy version")
    require(version_match.group(1) == expected_version,
            f"{label} numpy version mismatch: {version_match.group(1)} != {expected_version}")
    core = numpy_dir / "core"
    so_files = list(core.glob("_multiarray_umath*.so"))
    require(so_files, f"{label} numpy _multiarray_umath native extension is missing")
    all_extensions = list(numpy_dir.rglob("*.so"))
    require(all_extensions, f"{label} numpy has no native extensions")
    for so in all_extensions:
        result = subprocess.run(["lipo", "-info", str(so)], capture_output=True, text=True)
        require(result.returncode == 0 and "arm64" in result.stdout,
                f"{label} numpy native extension {so.name} is not arm64")
    print(f"verified {label} numpy {expected_version} arm64 ({len(all_extensions)} native extension(s))")


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
    require((app_path / "PythonRuntime/lib/python3.11/encodings/__init__.py").is_file(),
            "missing Python stdlib encodings; the pure-Python standard library is not bundled")

    numpy_dir = app_path / "PythonRuntime/lib/python3.11/site-packages/numpy"
    verify_numpy(numpy_dir, lock["numpy_version"], "bundled")

    symbols = subprocess.run(["nm", "-gU", str(executable)], capture_output=True, text=True, check=True)
    require("_OrtGetApiBase" in symbols.stdout, "final app does not contain ORT API entrypoint")
    print(f"verified app link layout and bundled PythonRuntime (ORT {lock['onnxruntime_version']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deps-dir", type=Path, default=ROOT / "ios/.deps")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--app-path", type=Path)
    parser.add_argument("--python-smoke", type=Path,
                        help="optional iOS Python executable/device wrapper used to run import numpy")
    args = parser.parse_args()
    verify_dependencies(args.deps_dir.resolve(), args.manifest.resolve())
    verify_numpy(args.deps_dir.resolve() / "numpy-arm64/numpy", json.loads(args.manifest.read_text())["numpy_version"], "dependency")
    if args.app_path:
        verify_app(args.app_path.resolve(), args.manifest.resolve())
    if args.python_smoke:
        smoke = subprocess.run([str(args.python_smoke), "-c", "import numpy; print(numpy.__version__)"] ,
                                capture_output=True, text=True)
        require(smoke.returncode == 0 and smoke.stdout.strip() == json.loads(args.manifest.read_text())["numpy_version"],
                f"iOS Python numpy import failed: {smoke.stderr.strip()}")
        print("verified numpy import smoke check")
    else:
        print("numpy import smoke check skipped (no --python-smoke executable supplied)")


if __name__ == "__main__":
    main()
