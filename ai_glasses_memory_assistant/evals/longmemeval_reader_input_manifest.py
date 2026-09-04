"""Freeze reversible Reader inputs from an immutable LongMemEval run.

The tool deliberately does not import the product service, Reader, judge, or
model client.  It only prepares a hash-checked input manifest for a later,
separately-authorized fixed-context Reader replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from ai_glasses_memory_assistant.evals.longmemeval_answer_task import extract_answer_task_from_debug


MANIFEST_SCHEMA = "longmemeval.reader-input-manifest.v2"
DEFAULT_READER_TIMEOUT = 120
DEFAULT_READER_MAX_TOKENS = 4096
DEFAULT_READER_MAX_CONTEXT_CHARS = 16000
DEFAULT_READER_TEMPERATURE = 0.0
DEFAULT_READER_THINKING = "disabled"


class ReaderInputManifestError(ValueError):
    """Raised when source artifacts cannot support a fixed-input replay."""


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ReaderInputManifestError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReaderInputManifestError(f"invalid JSON artifact: {path}: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _read_groups(cohort: Any) -> dict[str, list[str]]:
    if not isinstance(cohort, dict):
        raise ReaderInputManifestError("cohort must be a JSON object")
    groups = cohort.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ReaderInputManifestError("cohort.groups must be a nonempty object")
    selected: dict[str, list[str]] = {}
    seen: set[str] = set()
    for group, values in groups.items():
        name = str(group).strip()
        if not name:
            raise ReaderInputManifestError("cohort group name must be nonempty")
        if not isinstance(values, list) or not values:
            raise ReaderInputManifestError(f"cohort group {name} must be a nonempty list")
        ids: list[str] = []
        for value in values:
            question_id = str(value).strip()
            if not question_id:
                raise ReaderInputManifestError(f"cohort group {name} has an empty question_id")
            if question_id in seen:
                raise ReaderInputManifestError(f"duplicate cohort question_id: {question_id}")
            seen.add(question_id)
            ids.append(question_id)
        selected[name] = ids
    return selected


def _reader_source_path() -> Path:
    """Locate the Reader implementation without importing product code."""

    return Path(__file__).with_name("longmemeval_" + "runner.py")


def _reader_config_from_run_manifest(run_manifest: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Freeze all public Reader settings, including historical defaults."""

    config = run_manifest.get("config") if isinstance(run_manifest, dict) else None
    if not isinstance(config, dict):
        raise ReaderInputManifestError("source run-manifest has no config object")
    required = ("reader_provider", "reader_model", "reader_base_url")
    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for key in required:
        value = str(config.get(key) or "").strip()
        if not value:
            raise ReaderInputManifestError(f"source run-manifest missing {key}")
        values[key.removeprefix("reader_")] = value.rstrip("/") if key == "reader_base_url" else value
        provenance[key.removeprefix("reader_")] = "source run-manifest"
    defaults: dict[str, Any] = {
        "timeout": DEFAULT_READER_TIMEOUT,
        "max_tokens": DEFAULT_READER_MAX_TOKENS,
        "temperature": DEFAULT_READER_TEMPERATURE,
        "max_context_chars": DEFAULT_READER_MAX_CONTEXT_CHARS,
        "thinking": DEFAULT_READER_THINKING,
    }
    for name, default in defaults.items():
        key = f"reader_{name}"
        value = config.get(key, default)
        if name in {"timeout", "max_tokens", "max_context_chars"}:
            try:
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise ReaderInputManifestError(f"source run-manifest has invalid {key}") from exc
            if value < 1:
                raise ReaderInputManifestError(f"source run-manifest has invalid {key}")
        elif name == "temperature":
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ReaderInputManifestError(f"source run-manifest has invalid {key}") from exc
        else:
            value = str(value or default).strip()
            if value not in {"disabled", "enabled"}:
                raise ReaderInputManifestError(f"source run-manifest has invalid {key}")
        values[name] = value
        provenance[name] = "source run-manifest" if key in config else "historical runner default"
    return values, provenance


def build_reader_invocation_payload(
    result: dict[str, Any], *, reader_config: dict[str, Any]
) -> dict[str, Any]:
    """Return every non-secret value forwarded to the Reader implementation."""

    return {
        "reader_config": reader_config,
        "question": str(result.get("question") or ""),
        "question_type": str(result.get("question_type") or ""),
        "question_date": str(result.get("question_date") or ""),
        "memory_context": str(result.get("recall_context") or ""),
        "answer_task": extract_answer_task_from_debug(result.get("response_debug")),
        "recalled_memories": list(result.get("recalled_memories") or []),
        "recalled_timeline_chunks": list(result.get("recalled_timeline_chunks") or []),
    }


def _index_cases(run_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((run_dir / "cases").glob("*/result.json")):
        result = _load_json(path)
        if not isinstance(result, dict):
            raise ReaderInputManifestError(f"case result is not an object: {path}")
        question_id = str(result.get("question_id") or "").strip()
        if not question_id:
            raise ReaderInputManifestError(f"case result has no question_id: {path}")
        if question_id in indexed:
            raise ReaderInputManifestError(f"duplicate case result question_id: {question_id}")
        completed_path = path.parent / "completed.json"
        if not completed_path.is_file():
            raise ReaderInputManifestError(f"case has no completed marker: {question_id}")
        indexed[question_id] = (path, result)
    return indexed


def build_input_manifest(source_run_dir: Path, cohort_path: Path) -> dict[str, Any]:
    """Validate source artifacts and return one fixed Reader-input manifest."""

    source_run_dir = source_run_dir.resolve()
    cohort_path = cohort_path.resolve()
    cohort = _load_json(cohort_path)
    groups = _read_groups(cohort)
    run_manifest_path = source_run_dir / "run-manifest.json"
    run_manifest = _load_json(run_manifest_path)
    expected_hash = (
        cohort.get("source_run", {}).get("run_manifest_sha256")
        if isinstance(cohort.get("source_run"), dict)
        else None
    )
    actual_hash = _sha256_file(run_manifest_path)
    if expected_hash is not None and str(expected_hash) != actual_hash:
        raise ReaderInputManifestError("source run-manifest SHA-256 mismatch")
    reader_config, reader_config_provenance = _reader_config_from_run_manifest(run_manifest)
    reader_code_path = _reader_source_path()
    if not reader_code_path.is_file():
        raise ReaderInputManifestError(f"missing Reader implementation snapshot: {reader_code_path}")
    indexed = _index_cases(source_run_dir)
    cases: list[dict[str, Any]] = []
    for group, question_ids in groups.items():
        for question_id in question_ids:
            if question_id not in indexed:
                raise ReaderInputManifestError(f"cohort question_id missing from source run: {question_id}")
            result_path, result = indexed[question_id]
            context = str(result.get("recall_context") or "")
            source_envelope = {
                "recalled_memories": result.get("recalled_memories") or [],
                "recalled_timeline_chunks": result.get("recalled_timeline_chunks") or [],
                "recalled_documents": result.get("recalled_documents") or [],
            }
            invocation_payload = build_reader_invocation_payload(result, reader_config=reader_config)
            answer_task = invocation_payload["answer_task"]
            cases.append(
                {
                    "group": group,
                    "question_id": question_id,
                    "source_path": result_path.relative_to(source_run_dir).as_posix(),
                    "source_result_sha256": _sha256_file(result_path),
                    "question_sha256": _sha256_bytes(str(result.get("question") or "").encode("utf-8")),
                    "answer_task_sha256": _canonical_json_hash(answer_task),
                    "recall_context_sha256": _sha256_bytes(context.encode("utf-8")),
                    "recall_context_chars": len(context),
                    "source_envelope_sha256": _canonical_json_hash(source_envelope),
                    "source_envelope_counts": {
                        "memories": len(source_envelope["recalled_memories"]),
                        "timeline_chunks": len(source_envelope["recalled_timeline_chunks"]),
                        "documents": len(source_envelope["recalled_documents"]),
                    },
                    "reader_invocation_sha256": _canonical_json_hash(invocation_payload),
                }
            )
    return {
        "schema": MANIFEST_SCHEMA,
        "purpose": "fixed-context Reader replay preparation only; no Reader or judge executed",
        "source_run_dir": str(source_run_dir),
        "source_run_manifest_sha256": actual_hash,
        "source_snapshot": run_manifest.get("source_snapshot") if isinstance(run_manifest, dict) else None,
        "reader_config": reader_config,
        "reader_config_provenance": reader_config_provenance,
        "reader_config_sha256": _canonical_json_hash(reader_config),
        "reader_code_sha256": _sha256_file(reader_code_path),
        "answer_task_code_sha256": _sha256_file(Path(__file__).with_name("longmemeval_answer_task.py")),
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256_file(cohort_path),
        "groups": {group: len(question_ids) for group, question_ids in groups.items()},
        "case_count": len(cases),
        "cases": cases,
        "zero_model_gate": {"qwen_calls": 0, "judge_calls": 0, "network_calls": 0},
    }


def write_input_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    """Atomically create a new immutable output directory."""

    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise ReaderInputManifestError(f"output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        (temporary_dir / "input-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, out_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze hash-checked Reader inputs from a completed LongMemEval run.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_input_manifest(args.source_run_dir, args.cohort)
        out_dir = write_input_manifest(args.out_dir, manifest)
    except ReaderInputManifestError as exc:
        print(f"reader input manifest blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), **manifest["zero_model_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
