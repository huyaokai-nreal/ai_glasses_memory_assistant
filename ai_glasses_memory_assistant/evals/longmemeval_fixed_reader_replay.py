"""Run a local Reader only after hash-verifying frozen LongMemEval inputs.

This evaluator never imports history, builds recall, invokes PPD, or runs a
judge.  The input manifest is prepared by ``longmemeval_reader_input_manifest``
and must match the completed source artifacts immediately before the first
Reader request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from ai_glasses_memory_assistant.evals import longmemeval_reader_input_manifest as input_manifest
from ai_glasses_memory_assistant.evals.longmemeval_runner import (
    OpenAIReader,
    ReaderConfig,
)


REPLAY_SCHEMA = "longmemeval.fixed-reader-replay.v2"
LOCAL_QWEN_PROVIDER = "llama_cpp"
LOCAL_QWEN_MODEL = "qwen3.8-27b-32k"
LOCAL_QWEN_BASE_URL = "http://10.252.17.5:11438/v1"


class FixedReaderReplayError(ValueError):
    """Raised when a fixed-context Reader replay cannot safely start."""


class ReaderLike(Protocol):
    last_debug: dict[str, Any] | None

    def answer(
        self,
        *,
        question: str,
        question_type: str,
        question_date: str,
        memory_context: str,
        answer_task: dict[str, Any] | None = None,
        recalled_memories: list[dict[str, Any]] | None = None,
        recalled_timeline_chunks: list[dict[str, Any]] | None = None,
    ) -> str: ...


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FixedReaderReplayError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FixedReaderReplayError(f"invalid JSON artifact: {path}: {exc}") from exc


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


def _safe_result_path(source_run_dir: Path, relative_path: str) -> Path:
    candidate = (source_run_dir / relative_path).resolve()
    try:
        candidate.relative_to(source_run_dir)
    except ValueError as exc:
        raise FixedReaderReplayError(f"manifest source path escapes source run: {relative_path}") from exc
    return candidate


def verify_frozen_inputs(source_run_dir: Path, manifest_path: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Verify every source byte before a Reader may be constructed."""

    source_run_dir = source_run_dir.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != input_manifest.MANIFEST_SCHEMA:
        raise FixedReaderReplayError("unsupported reader input manifest schema")
    reader_config = manifest.get("reader_config")
    if not isinstance(reader_config, dict):
        raise FixedReaderReplayError("input manifest has no frozen Reader configuration")
    if str(manifest.get("reader_config_sha256") or "") != _canonical_json_hash(reader_config):
        raise FixedReaderReplayError("frozen Reader configuration hash mismatch")
    if str(manifest.get("reader_code_sha256") or "") != _sha256_file(input_manifest._reader_source_path()):
        raise FixedReaderReplayError("Reader implementation snapshot changed after input freeze")
    answer_task_path = Path(input_manifest.__file__).with_name("longmemeval_answer_task.py")
    if str(manifest.get("answer_task_code_sha256") or "") != _sha256_file(answer_task_path):
        raise FixedReaderReplayError("answer-task implementation snapshot changed after input freeze")
    run_manifest_path = source_run_dir / "run-manifest.json"
    if str(manifest.get("source_run_manifest_sha256") or "") != _sha256_file(run_manifest_path):
        raise FixedReaderReplayError("source run-manifest SHA-256 changed after input freeze")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise FixedReaderReplayError("input manifest cases must be a nonempty list")
    verified: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for entry in cases:
        if not isinstance(entry, dict):
            raise FixedReaderReplayError("input manifest case must be an object")
        question_id = str(entry.get("question_id") or "").strip()
        if not question_id:
            raise FixedReaderReplayError("input manifest case has no question_id")
        if question_id in seen:
            raise FixedReaderReplayError(f"duplicate input manifest question_id: {question_id}")
        seen.add(question_id)
        result_path = _safe_result_path(source_run_dir, str(entry.get("source_path") or ""))
        result = _load_json(result_path)
        if not isinstance(result, dict) or str(result.get("question_id") or "") != question_id:
            raise FixedReaderReplayError(f"source result identity mismatch: {question_id}")
        if not (result_path.parent / "completed.json").is_file():
            raise FixedReaderReplayError(f"source case lost completed marker: {question_id}")
        context = str(result.get("recall_context") or "")
        envelope = {
            "recalled_memories": result.get("recalled_memories") or [],
            "recalled_timeline_chunks": result.get("recalled_timeline_chunks") or [],
            "recalled_documents": result.get("recalled_documents") or [],
        }
        invocation_payload = input_manifest.build_reader_invocation_payload(
            result,
            reader_config=reader_config,
        )
        actual_hashes = {
            "source_result_sha256": _sha256_file(result_path),
            "question_sha256": _sha256_bytes(str(result.get("question") or "").encode("utf-8")),
            "answer_task_sha256": _canonical_json_hash(invocation_payload["answer_task"]),
            "recall_context_sha256": _sha256_bytes(context.encode("utf-8")),
            "source_envelope_sha256": _canonical_json_hash(envelope),
            "reader_invocation_sha256": _canonical_json_hash(invocation_payload),
        }
        for key, actual in actual_hashes.items():
            if str(entry.get(key) or "") != actual:
                raise FixedReaderReplayError(f"frozen input {key} mismatch: {question_id}")
        verified.append((entry, result))
    if len(verified) != int(manifest.get("case_count") or -1):
        raise FixedReaderReplayError("input manifest case_count mismatch")
    return manifest, verified


def _reader_output_record(entry: dict[str, Any], result: dict[str, Any], reader: ReaderLike) -> dict[str, Any]:
    answer_task = input_manifest.extract_answer_task_from_debug(result.get("response_debug"))
    hypothesis = reader.answer(
        question=str(result.get("question") or ""),
        question_type=str(result.get("question_type") or ""),
        question_date=str(result.get("question_date") or ""),
        memory_context=str(result.get("recall_context") or ""),
        answer_task=answer_task,
        recalled_memories=list(result.get("recalled_memories") or []),
        recalled_timeline_chunks=list(result.get("recalled_timeline_chunks") or []),
    )
    reader_debug = reader.last_debug if isinstance(reader.last_debug, dict) else {}
    return {
        "group": entry.get("group"),
        "question_id": entry.get("question_id"),
        "input_hashes": {
            key: entry.get(key)
            for key in (
                "question_sha256",
                "answer_task_sha256",
                "recall_context_sha256",
                "source_envelope_sha256",
                "reader_invocation_sha256",
            )
        },
        "hypothesis_sha256": _sha256_bytes(str(hypothesis).encode("utf-8")),
        "reader_status": str(reader_debug.get("reader_status") or "unknown"),
        "failure_stage": str(reader_debug.get("failure_stage") or ""),
        "output_schema": str(reader_debug.get("output_schema") or "unknown"),
        "refusal": bool(reader_debug.get("refusal")),
        "selected_source_count": len(reader_debug.get("selected_source_ids") or []),
        "api_calls": int(reader_debug.get("api_calls") or 0),
        "execution_attempts": list(reader_debug.get("execution_attempts") or []),
    }


def run_verified_replay(verified: list[tuple[dict[str, Any], dict[str, Any]]], reader: ReaderLike) -> list[dict[str, Any]]:
    """Call only the supplied Reader on inputs verified before construction."""

    return [_reader_output_record(entry, result, reader) for entry, result in verified]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, dict[str, Any]] = {}
    for group in sorted({str(record.get("group") or "unknown") for record in records}):
        rows = [record for record in records if str(record.get("group") or "unknown") == group]
        by_group[group] = {
            "count": len(rows),
            "reader_status": dict(Counter(str(row["reader_status"]) for row in rows)),
            "output_schema": dict(Counter(str(row["output_schema"]) for row in rows)),
            "refusal_count": sum(bool(row["refusal"]) for row in rows),
        }
    return {
        "schema": REPLAY_SCHEMA,
        "case_count": len(records),
        "by_group": by_group,
        "judge_calls": 0,
        "note": "Reader-only observability; no judge or score is present.",
    }


def write_replay(out_dir: Path, *, input_manifest: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    """Atomically write a new immutable Reader-only replay artifact."""

    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FixedReaderReplayError(f"output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        replay_manifest = {
            "schema": REPLAY_SCHEMA,
            "input_manifest_sha256": _sha256_bytes(
                json.dumps(input_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
            "case_count": len(records),
            "reader_config": input_manifest.get("reader_config"),
            "reader_config_sha256": input_manifest.get("reader_config_sha256"),
            "reader_code_sha256": input_manifest.get("reader_code_sha256"),
            "answer_task_code_sha256": input_manifest.get("answer_task_code_sha256"),
            "judge_calls": 0,
        }
        (temporary_dir / "replay-manifest.json").write_text(
            json.dumps(replay_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (temporary_dir / "reader-output.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        (temporary_dir / "summary.json").write_text(
            json.dumps(summarize_records(records), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_dir, out_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return out_dir


def _frozen_reader_config(manifest: dict[str, Any], *, api_key: str) -> ReaderConfig:
    raw = manifest.get("reader_config")
    if not isinstance(raw, dict):
        raise FixedReaderReplayError("input manifest has no frozen Reader configuration")
    try:
        config = ReaderConfig(
            provider=str(raw["provider"]),
            model=str(raw["model"]),
            base_url=str(raw["base_url"]),
            api_key=api_key,
            timeout=int(raw["timeout"]),
            max_tokens=int(raw["max_tokens"]),
            temperature=float(raw["temperature"]),
            max_context_chars=int(raw["max_context_chars"]),
            thinking=str(raw["thinking"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FixedReaderReplayError("input manifest has an invalid frozen Reader configuration") from exc
    if not config.api_key.strip():
        raise FixedReaderReplayError("fixed Reader replay requires --reader-api-key for authentication")
    if config.timeout < 1 or config.max_tokens < 1 or config.max_context_chars < 1:
        raise FixedReaderReplayError("input manifest has an invalid frozen Reader configuration")
    if config.thinking not in {"disabled", "enabled"}:
        raise FixedReaderReplayError("input manifest has an invalid frozen Reader configuration")
    return config


def _add_reader_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reader-api-key", required=True)


def _local_qwen_reader(config: ReaderConfig) -> OpenAIReader:
    if config.provider != LOCAL_QWEN_PROVIDER or config.model != LOCAL_QWEN_MODEL or config.base_url != LOCAL_QWEN_BASE_URL:
        raise FixedReaderReplayError("fixed Reader replay requires the configured local Qwen endpoint and model")
    return OpenAIReader(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run only hash-verified fixed-context LongMemEval Reader inputs.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    _add_reader_arguments(parser)
    args = parser.parse_args(argv)
    try:
        manifest, verified = verify_frozen_inputs(args.source_run_dir, args.input_manifest)
        reader = _local_qwen_reader(_frozen_reader_config(manifest, api_key=args.reader_api_key))
        records = run_verified_replay(verified, reader)
        out_dir = write_replay(args.out_dir, input_manifest=manifest, records=records)
    except FixedReaderReplayError as exc:
        print(f"fixed Reader replay blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "qwen_calls": sum(row["api_calls"] for row in records), "judge_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
