#!/usr/bin/env python3
"""Assemble a local Android model pack with Ten VAD and SenseVoice 2025 INT8.

This copies a verified existing Android pack and replaces only the ambient VAD/ASR artifacts.
It never downloads models and refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


MAX_AMBIENT_ASR_BYTES = 300 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if root.resolve() not in candidate.parents:
        raise ValueError(f"model pack path escapes its root: {relative_path}")
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", type=Path, required=True, help="Existing verified Android pack directory.")
    parser.add_argument("--candidate-sensevoice-dir", type=Path, required=True, help="Directory containing model.int8.onnx and tokens.txt.")
    parser.add_argument("--ten-vad-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New pack directory; must not exist.")
    parser.add_argument("--version", default="x4000-sherpa-1.13.4-ten-sensevoice-2025-v1")
    return parser.parse_args()


def update_file_record(records: list[dict[str, Any]], path: str, source: Path) -> None:
    record = next((item for item in records if item.get("path") == path), None)
    replacement = {"path": path, "sha256": sha256(source), "size_bytes": source.stat().st_size}
    if record is None:
        records.append(replacement)
    else:
        record.update(replacement)


def main() -> int:
    args = parse_args()
    source_pack, candidate_dir, ten_vad, output = (args.source_pack.resolve(), args.candidate_sensevoice_dir.resolve(), args.ten_vad_model.resolve(), args.out.resolve())
    manifest_path = source_pack / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest is missing: {manifest_path}")
    if output.exists():
        raise FileExistsError(f"--out already exists: {output}")
    model, tokens = candidate_dir / "model.int8.onnx", candidate_dir / "tokens.txt"
    for artifact in (model, tokens, ten_vad):
        if not artifact.is_file():
            raise FileNotFoundError(f"required candidate artifact is missing: {artifact}")
    if model.stat().st_size > MAX_AMBIENT_ASR_BYTES:
        raise ValueError(f"candidate model exceeds 300 MiB ambient-ASR budget: {model.stat().st_size} bytes")

    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = manifest.get("components") or {}
    vad, ambient_asr = components.get("vad"), components.get("ambient_asr")
    if not isinstance(vad, dict) or not isinstance(ambient_asr, dict):
        raise ValueError("source pack lacks vad or ambient_asr components")
    old_paths = {str(value) for value in (vad.get("roles") or {}).values()} | {str(value) for value in (ambient_asr.get("roles") or {}).values()}
    output.mkdir(parents=True)
    for record in manifest.get("files") or []:
        relative_path = str(record.get("path") or "")
        if not relative_path or relative_path in old_paths:
            continue
        source = contained(source_pack, relative_path)
        if not source.is_file():
            raise FileNotFoundError(f"source pack artifact is missing: {source}")
        target = contained(output, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    replacements = {
        "vad/ten-vad.onnx": ten_vad,
        "sensevoice/model.int8.onnx": model,
        "sensevoice/tokens.txt": tokens,
    }
    for relative_path, source in replacements.items():
        target = contained(output, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest = copy.deepcopy(manifest)
    manifest["version"] = args.version
    manifest["components"]["vad"] = {
        **vad,
        "engine": "ten_vad",
        "roles": {"model": "vad/ten-vad.onnx"},
        "options": {**(vad.get("options") or {}), "window_size": 256},
    }
    manifest["components"]["ambient_asr"] = {
        **ambient_asr,
        "engine": "sense_voice",
        "roles": {"model": "sensevoice/model.int8.onnx", "tokens": "sensevoice/tokens.txt"},
        "options": {**(ambient_asr.get("options") or {}), "language": "auto", "use_itn": True},
    }
    records = [dict(item) for item in manifest.get("files") or [] if str(item.get("path") or "") not in old_paths]
    for relative_path, source in replacements.items():
        update_file_record(records, relative_path, source)
    manifest["files"] = sorted(records, key=lambda item: str(item["path"]))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Assembled {args.version} at {output}; ambient model={model.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
