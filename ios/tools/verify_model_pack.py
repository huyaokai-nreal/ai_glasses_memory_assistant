#!/usr/bin/env python3
"""Validate an Android-compatible model_pack.v1 before bundling it into iOS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_COMPONENTS = {"vad", "kws", "online_asr", "ambient_asr", "speaker"}
SHERPA_VERSION = "1.13.4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(pack_dir: Path) -> dict:
    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "model_pack.v1":
        raise ValueError("model schema must be model_pack.v1")
    if manifest.get("sherpa_onnx_version") != SHERPA_VERSION:
        raise ValueError(f"model pack must require sherpa-onnx {SHERPA_VERSION}")
    components = manifest.get("components")
    if not isinstance(components, dict) or not REQUIRED_COMPONENTS.issubset(components):
        raise ValueError("model pack does not contain all five required components")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model pack has no declared files")

    declared: set[str] = set()
    for item in files:
        relative = Path(str(item.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe model path: {relative}")
        normalized = relative.as_posix()
        if normalized in declared:
            raise ValueError(f"duplicate model path: {normalized}")
        declared.add(normalized)
        expected_size = int(item.get("size_bytes") or 0)
        expected_hash = str(item.get("sha256") or "").lower()
        target = pack_dir / relative
        if not target.is_file() or target.stat().st_size != expected_size:
            raise ValueError(f"model file size mismatch: {normalized}")
        if sha256(target) != expected_hash:
            raise ValueError(f"model SHA-256 mismatch: {normalized}")

    for component, value in components.items():
        roles = value.get("roles") if isinstance(value, dict) else None
        if not isinstance(roles, dict) or not roles:
            raise ValueError(f"component roles are invalid: {component}")
        for path in roles.values():
            if str(path) not in declared:
                raise ValueError(f"component {component} references an undeclared file")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate(args.pack_dir.resolve())
    print(f"verified {manifest['version']}: {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
