from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, TextIO

from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
from ai_glasses_memory_assistant.app_home import APP_HOME_ENV
from ai_glasses_memory_assistant.env_loader import load_app_dotenv, restore_app_llm_env, snapshot_app_llm_env
from ai_glasses_memory_assistant.evidence_set import (
    EvidenceCandidate,
    batch_evidence_candidates,
    expand_consolidated_ledger,
    validate_aggregation_ledger,
)
from ai_glasses_memory_assistant.evals.longmemeval_adapter import (
    DEFAULT_ORACLE_PATH,
    LongMemEvalItem,
    LongMemEvalSession,
    answer_terms,
    load_longmemeval_items,
)
from ai_glasses_memory_assistant.evals.longmemeval_answer_task import extract_answer_task_from_debug
from ai_glasses_memory_assistant.evals.metrics import contains_any
from ai_glasses_memory_assistant.llm_runtime import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_FALLBACK_PROVIDER,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    LLM_PROVIDER_ENV,
)
from ai_glasses_memory_assistant.privacy_filter import redact_sensitive_text
from ai_glasses_memory_assistant.source_envelope import (
    SourceEnvelope,
    SourceEnvelopeInput,
    build_source_envelopes,
)


DEFAULT_OUTPUT_ROOT_DIR = Path("reports") / "longmemeval"
DEFAULT_CACHE_ROOT_DIR = Path("reports") / "longmemeval" / "cache"
DEFAULT_READER_MAX_CONTEXT_CHARS = 16000
DEFAULT_READER_MAX_TOKENS = 4096
DEFAULT_READER_TIMEOUT = 120
UNKNOWN_ANSWER = "I don't know based on the available memory."
INCOMPLETE_EVIDENCE_ANSWER = "I couldn't complete the evidence check, so I can't give a reliable total."
READER_EXECUTION_FAILED_ANSWER = "I found memory evidence, but the Reader could not complete processing. Please retry."
COMPLETE_SET_MAX_READER_CALLS = 10
_LONGMEMEVAL_WEEKDAY_RE = re.compile(r"\s+\([^)]*\)\s+")
CHECKPOINT_SCHEMA_VERSION = "longmemeval.checkpoint.v2"
# 缓存失效键之一：改动记忆提取/召回口径时必须手动 bump，避免旧缓存被误用。
# v4: RC1 路由补"follow-up 前序对话→needs_event_memory"规则 + RC2a named scope 解析失败回退 self，
# 二者都改变了召回的 subject/授权语义，必须使 8-31 旧缓存失效。
LONGMEMEVAL_CACHE_VERSION = "v4"


@dataclass(frozen=True)
class ReaderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout: int = DEFAULT_READER_TIMEOUT
    max_tokens: int = DEFAULT_READER_MAX_TOKENS
    temperature: float = 0.0
    max_context_chars: int = DEFAULT_READER_MAX_CONTEXT_CHARS
    thinking: str = "disabled"


def _thinking_extra_body(provider: str, *, thinking: str) -> dict[str, Any] | None:
    """Return the provider-specific request body that disables thinking mode.

    Qwen/DeepSeek default thinking (CoT) burns the output budget on fixed-schema
    reader/judge calls. Disable it explicitly; enabled/other providers get None.
    """
    if str(thinking or "disabled").strip().lower() == "enabled":
        return None
    provider = str(provider or "").strip().lower()
    if provider == "deepseek":
        return {"thinking": {"type": "disabled"}}
    if provider == "llama_cpp":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path
    brief_output: Path
    detail_output: Path


@dataclass
class MutableClock:
    value: float

    def __call__(self) -> float:
        return self.value


@dataclass(frozen=True)
class _RecallScoreItem:
    answer: str
    is_abstention: bool


class Reader(Protocol):
    config: ReaderConfig

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
    ) -> str:
        ...


class OpenAIReader:
    def __init__(self, config: ReaderConfig) -> None:
        from openai import OpenAI

        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=float(config.timeout),
        )
        self.last_debug: dict[str, Any] | None = None

    def _extra_body_kwargs(self) -> dict[str, Any]:
        extra = _thinking_extra_body(self.config.provider, thinking=self.config.thinking)
        return {"extra_body": extra} if extra else {}

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
    ) -> str:
        started = time.perf_counter()
        debug: dict[str, Any] = {
            "provider": self.config.provider,
            "model": self.config.model,
            "reader_input_chars": 0,
            "reader_output_chars": 0,
            "parse_success": False,
            "output_schema": "not_called",
            "relevant_evidence": [],
            "refusal": False,
            "error": "",
            "reader_status": "",
            "failure_stage": "",
            "selected_source_ids": [],
            "execution_attempts": [],
            "reader_duration_seconds": 0.0,
            "api_calls": 0,
        }
        if answer_task:
            debug["answer_task"] = answer_task
        try:
            if not memory_context.strip():
                debug["refusal"] = True
                debug["reader_status"] = "insufficient_evidence"
                debug["failure_stage"] = "context"
                debug["reader_duration_seconds"] = round(time.perf_counter() - started, 6)
                self.last_debug = debug
                return UNKNOWN_ANSWER
            if answer_task and answer_task.get("coverage_requirement") == "complete_set":
                answer = self._answer_complete_set(
                    question=question,
                    question_type=question_type,
                    question_date=question_date,
                    memory_context=memory_context,
                    answer_task=answer_task,
                    debug=debug,
                    recalled_memories=recalled_memories,
                    recalled_timeline_chunks=recalled_timeline_chunks,
                )
                debug["reader_duration_seconds"] = round(time.perf_counter() - started, 6)
                debug["refusal"] = debug.get("reader_status") == "insufficient_evidence"
                self.last_debug = debug
                return answer
            reader_context = truncate_text(memory_context, self.config.max_context_chars)
            debug["reader_input_chars"] = len(reader_context)
            debug["api_calls"] += 1
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[{
                    "role": "user",
                    "content": build_reader_prompt(
                        question=question,
                        question_type=question_type,
                        question_date=question_date,
                        memory_context=reader_context,
                        answer_task=answer_task,
                    ),
                }],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=False,
                **self._extra_body_kwargs(),
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise RuntimeError("Reader LLM returned no choices")
            message = getattr(choices[0], "message", None)
            content = _chat_message_text(getattr(message, "content", "") if message is not None else "")
            debug["reader_output_chars"] = len(content)
            parsed = _parse_reader_structured(content)
            if parsed is not None:
                debug["parse_success"] = True
                debug["output_schema"] = parsed["output_schema"]
                debug["relevant_evidence"] = parsed.get("relevant_evidence", [])
                debug["coverage"] = parsed.get("coverage", [])
                answer = parsed.get("final_answer", "")
            else:
                debug["output_schema"] = "unstructured_text"
                answer = extract_reader_final_answer(content)
            # A refusal with usable context contradicts rule 8: either evidence was
            # selected but unused, or no evidence was selected at all. Retry once
            # with a follow-up instruction instead of accepting the refusal.
            refusal_retry = False
            retry_instruction = ""
            if parsed is not None and answer and answer.strip() == UNKNOWN_ANSWER:
                if debug.get("relevant_evidence"):
                    refusal_retry = True
                    retry_instruction = (
                        "\n\nYou identified relevant evidence above but still returned the unknown answer. "
                        "Answer final_answer from that evidence; do not repeat the unknown answer."
                    )
                elif memory_context.strip():
                    refusal_retry = True
                    retry_instruction = (
                        "\n\nYou were given memory context above but selected no relevant evidence and "
                        "returned the unknown answer. Identify the relevant evidence from the context and "
                        "answer final_answer from it; do not repeat the unknown answer."
                    )
            if refusal_retry:
                retry_prompt = (
                    build_reader_prompt(
                        question=question,
                        question_type=question_type,
                        question_date=question_date,
                        memory_context=memory_context,
                        answer_task=answer_task,
                    )
                    + retry_instruction
                )
                debug["api_calls"] += 1
                retry_response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": retry_prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=False,
                    **self._extra_body_kwargs(),
                )
                retry_choices = getattr(retry_response, "choices", None) or []
                if retry_choices:
                    retry_message = getattr(retry_choices[0], "message", None)
                    retry_content = _chat_message_text(
                        getattr(retry_message, "content", "") if retry_message is not None else ""
                    )
                    debug["reader_retry_output_chars"] = len(retry_content)
                    retry_parsed = _parse_reader_structured(retry_content)
                    if retry_parsed is not None:
                        debug["parse_success"] = True
                        debug["retry_output_schema"] = retry_parsed["output_schema"]
                        debug["output_schema"] = retry_parsed["output_schema"]
                        debug["relevant_evidence"] = retry_parsed.get("relevant_evidence", [])
                        debug["coverage"] = retry_parsed.get("coverage", [])
                        answer = retry_parsed.get("final_answer", "")
                    else:
                        debug["retry_output_schema"] = "unstructured_text"
                        debug["output_schema"] = "unstructured_text"
                        answer = extract_reader_final_answer(retry_content)
            debug["refusal_retry"] = refusal_retry
            if not answer or answer.strip() == UNKNOWN_ANSWER:
                debug["refusal"] = True
                debug["reader_status"] = "insufficient_evidence"
                debug["failure_stage"] = "evidence"
            else:
                debug["reader_status"] = "answered"
                debug["failure_stage"] = ""
            debug["reader_duration_seconds"] = round(time.perf_counter() - started, 6)
            self.last_debug = debug
            return answer or UNKNOWN_ANSWER
        except Exception as exc:
            debug["error"] = str(exc)
            debug["reader_status"] = "execution_failed"
            debug["failure_stage"] = "provider"
            debug["reader_duration_seconds"] = round(time.perf_counter() - started, 6)
            self.last_debug = debug
            raise

    def _answer_complete_set(
        self,
        *,
        question: str,
        question_type: str,
        question_date: str,
        memory_context: str,
        answer_task: dict[str, Any],
        debug: dict[str, Any],
        recalled_memories: list[dict[str, Any]] | None = None,
        recalled_timeline_chunks: list[dict[str, Any]] | None = None,
    ) -> str:
        debug["complete_set"] = True

        def fail(
            *,
            status: str,
            stage: str,
            error: str,
            validation_errors: list[str] | None = None,
        ) -> str:
            debug["reader_status"] = status
            debug["failure_stage"] = stage
            debug["ledger_error"] = error
            if validation_errors is not None:
                debug["ledger_validation_errors"] = validation_errors
            return (
                INCOMPLETE_EVIDENCE_ANSWER
                if status == "insufficient_evidence"
                else READER_EXECUTION_FAILED_ANSWER
            )

        def record_attempt(
            *,
            stage: str,
            attempt: int,
            provider_error: bool,
            json_error: bool,
            validation: Any,
            input_chars: int,
            duration_seconds: float,
        ) -> None:
            attempts = debug.setdefault("execution_attempts", [])
            attempts.append({
                "stage": stage,
                "attempt": attempt,
                "outcome": (
                    "provider_error"
                    if provider_error
                    else "invalid_json"
                    if json_error
                    else ("valid" if validation.valid else "invalid")
                ),
                "input_chars": input_chars,
                "duration_seconds": round(duration_seconds, 6),
                "validation_errors": list(validation.errors),
            })

        if answer_task.get("coverage_complete") is not True or answer_task.get("truncated") is True:
            return fail(
                status="insufficient_evidence",
                stage="coverage",
                error=str(answer_task.get("truncation_reason") or "coverage_incomplete"),
            )
        if recalled_memories or recalled_timeline_chunks:
            candidates, _envelopes = _complete_set_candidates_from_structured(
                recalled_memories or [],
                recalled_timeline_chunks or [],
            )
        else:
            # Legacy fallback for artifacts/replays that only carry the rendered
            # line-based context without the structured payloads.
            candidates = _complete_set_candidates_from_context(memory_context)
        expected_source_ids = {
            str(source_id)
            for source_id in (answer_task.get("source_ids") or [])
            if str(source_id)
        }
        actual_source_ids = {candidate.source_id for candidate in candidates}
        if not candidates or (expected_source_ids and actual_source_ids != expected_source_ids):
            debug["expected_source_count"] = len(expected_source_ids)
            debug["actual_source_count"] = len(actual_source_ids)
            return fail(
                status="execution_failed",
                stage="source_delivery",
                error="reader_context_source_mismatch",
            )
        batches, batch_truncated = _batch_complete_set_candidates(
            candidates,
            max_context_chars=self.config.max_context_chars,
            # Reserve one call for cross-batch consolidation and one for final wording.
            max_batches=COMPLETE_SET_MAX_READER_CALLS - 2,
            max_candidates_per_batch=max(1, self.config.max_tokens // 256),
        )
        debug["ledger_batch_count"] = len(batches)
        if batch_truncated:
            return fail(
                status="execution_failed",
                stage="batch_budget",
                error="reader_batch_budget",
            )

        validated_batches = []
        for batch_index, batch in enumerate(batches):
            prompt = build_complete_set_ledger_prompt(
                question=question,
                question_type=question_type,
                question_date=question_date,
                answer_task=answer_task,
                candidates=batch,
                batch_index=batch_index,
                batch_count=len(batches),
            )
            validation = None
            provider_failures = 0
            json_failures = 0
            for attempt in range(2):
                retry_suffix = ""
                if attempt:
                    retry_suffix = (
                        "\n\nThe previous ledger failed deterministic validation. Classify every source_id "
                        "exactly once in source_decisions and put only supported facts in items. A source_id "
                        "may appear in several items only when that source explicitly states several facts."
                    )
                provider_error = False
                json_error = False
                attempt_prompt = prompt + retry_suffix
                attempt_started = time.perf_counter()
                try:
                    content = self._complete_text(attempt_prompt, debug)
                except Exception as exc:
                    provider_failures += 1
                    provider_error = True
                    debug["error"] = str(exc)
                    content = ""
                parsed, parsed_json = _parse_json_object_with_status(content)
                if not provider_error and not parsed_json:
                    json_failures += 1
                    json_error = True
                validation = validate_aggregation_ledger(
                    parsed,
                    batch,
                    require_final_answer=False,
                    allow_duplicate_source_ids=len(batches) > 1,
                    validate_claimed_value=False,
                )
                record_attempt(
                    stage="batch_ledger",
                    attempt=attempt + 1,
                    provider_error=provider_error,
                    json_error=json_error,
                    validation=validation,
                    input_chars=len(attempt_prompt),
                    duration_seconds=time.perf_counter() - attempt_started,
                )
                if validation.valid:
                    break
            if validation is None or not validation.valid:
                debug["ledger_failed_batch"] = batch_index
                provider_failed = provider_failures == 2
                json_failed = json_failures == 2
                output_failed = provider_failures + json_failures > 0
                return fail(
                    status="execution_failed",
                    stage=(
                        "provider" if provider_failed else
                        "json" if json_failed else
                        "output" if output_failed else
                        "batch_ledger"
                    ),
                    error=(
                        "provider_output_failure" if provider_failed else
                        "invalid_json_output" if json_failed else
                        "reader_output_failure" if output_failed else
                        "batch_validation_failed"
                    ),
                    validation_errors=list(validation.errors if validation else ()),
                )
            validated_batches.append(validation)

        ledger_validation = validated_batches[0]
        if len(validated_batches) > 1:
            consolidation_prompt = build_complete_set_consolidation_prompt(
                question=question,
                answer_task=answer_task,
                batch_ledgers=[{
                    "items": [
                        {**item, "quantity": str(item.get("quantity", "1"))}
                        for item in validation.items
                    ],
                    "source_decisions": validation.source_decisions,
                    "aggregation": {
                        "operation": validation.operation,
                        "value": str(validation.value),
                        "unit": validation.unit,
                    },
                } for validation in validated_batches],
            )
            final_validation = None
            provider_failures = 0
            json_failures = 0
            for attempt in range(2):
                suffix = "" if not attempt else (
                    "\n\nThe previous consolidated ledger failed deterministic validation. Return the full "
                    "ledger again, classify every source exactly once in source_decisions, retain every "
                    "supported fact item, and exclude facts outside answer_focus."
                )
                provider_error = False
                json_error = False
                attempt_prompt = consolidation_prompt + suffix
                attempt_started = time.perf_counter()
                try:
                    content = self._complete_text(attempt_prompt, debug)
                except Exception as exc:
                    provider_failures += 1
                    provider_error = True
                    debug["error"] = str(exc)
                    content = ""
                parsed, parsed_json = _parse_json_object_with_status(content)
                if not provider_error and not parsed_json:
                    json_failures += 1
                    json_error = True
                final_validation = validate_aggregation_ledger(
                    expand_consolidated_ledger(parsed),
                    candidates,
                    require_final_answer=False,
                    validate_claimed_value=False,
                )
                record_attempt(
                    stage="consolidation",
                    attempt=attempt + 1,
                    provider_error=provider_error,
                    json_error=json_error,
                    validation=final_validation,
                    input_chars=len(attempt_prompt),
                    duration_seconds=time.perf_counter() - attempt_started,
                )
                if final_validation.valid:
                    break
            if final_validation is None or not final_validation.valid:
                provider_failed = provider_failures == 2
                json_failed = json_failures == 2
                output_failed = provider_failures + json_failures > 0
                return fail(
                    status="execution_failed",
                    stage=(
                        "provider" if provider_failed else
                        "json" if json_failed else
                        "output" if output_failed else
                        "consolidation"
                    ),
                    error=(
                        "provider_output_failure" if provider_failed else
                        "invalid_json_output" if json_failed else
                        "reader_output_failure" if output_failed else
                        "consolidated_ledger_validation_failed"
                    ),
                    validation_errors=list(final_validation.errors if final_validation else ()),
                )
            ledger_validation = final_validation

        if ledger_validation.value is None:
            return fail(
                status="execution_failed",
                stage="aggregation",
                error="combined_ledger_validation_failed",
                validation_errors=list(ledger_validation.errors),
            )
        merged_payload: dict[str, Any] = {
            "items": [
                {**item, "quantity": str(item.get("quantity", "1"))}
                for item in ledger_validation.items
            ],
            "aggregation": {
                "operation": ledger_validation.operation,
                "value": str(ledger_validation.value),
                "unit": ledger_validation.unit or "item",
            },
        }
        if ledger_validation.source_decisions:
            merged_payload["source_decisions"] = ledger_validation.source_decisions
        final_prompt = build_complete_set_final_prompt(
            question=question,
            answer_task=answer_task,
            ledger=merged_payload,
        )
        final_validation = None
        provider_failures = 0
        json_failures = 0
        for attempt in range(2):
            suffix = "" if not attempt else (
                "\n\nThe previous final_answer did not contain the verified numeric value. "
                "Return JSON again and include that exact value."
            )
            provider_error = False
            json_error = False
            attempt_prompt = final_prompt + suffix
            attempt_started = time.perf_counter()
            try:
                content = self._complete_text(attempt_prompt, debug)
            except Exception as exc:
                provider_failures += 1
                provider_error = True
                debug["error"] = str(exc)
                content = ""
            parsed, parsed_json = _parse_json_object_with_status(content)
            if not provider_error and not parsed_json:
                json_failures += 1
                json_error = True
            final_answer = str(parsed.get("final_answer") or "") if isinstance(parsed, dict) else ""
            final_validation = validate_aggregation_ledger(
                {**merged_payload, "final_answer": final_answer},
                candidates,
            )
            record_attempt(
                stage="final_answer",
                attempt=attempt + 1,
                provider_error=provider_error,
                json_error=json_error,
                validation=final_validation,
                input_chars=len(attempt_prompt),
                duration_seconds=time.perf_counter() - attempt_started,
            )
            if final_validation.valid:
                break
        if final_validation is None or not final_validation.valid:
            provider_failed = provider_failures == 2
            json_failed = json_failures == 2
            output_failed = provider_failures + json_failures > 0
            return fail(
                status="execution_failed",
                stage=(
                    "provider" if provider_failed else
                    "json" if json_failed else
                    "output" if output_failed else
                    "final_answer"
                ),
                error=(
                    "provider_output_failure" if provider_failed else
                    "invalid_json_output" if json_failed else
                    "reader_output_failure" if output_failed else
                    "final_answer_validation_failed"
                ),
                validation_errors=list(final_validation.errors if final_validation else ()),
            )
        debug["parse_success"] = True
        debug["ledger_valid"] = True
        debug["ledger_item_count"] = len(final_validation.items)
        debug["ledger_accounted_source_count"] = len(final_validation.accounted_source_ids)
        debug["ledger_operation"] = final_validation.operation
        debug["ledger_value"] = str(final_validation.value)
        debug["ledger_unit"] = final_validation.unit
        # Retain the already validated source disposition for offline selection audits.
        debug["ledger_source_decisions"] = list(final_validation.source_decisions)
        debug["ledger_items"] = [
            {**item, "quantity": str(item.get("quantity", "0"))}
            for item in final_validation.items
        ]
        debug["reader_status"] = "answered"
        debug["failure_stage"] = ""
        debug["selected_source_ids"] = list(final_validation.selected_source_ids)
        debug["error"] = ""
        debug["reader_input_chars"] = len(memory_context)
        debug["reader_output_chars"] = len(final_validation.final_answer)
        return final_validation.final_answer

    def _complete_text(self, prompt: str, debug: dict[str, Any]) -> str:
        debug["api_calls"] = int(debug.get("api_calls") or 0) + 1
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        # The ledger is deterministic extraction into a fixed schema. Default
        # thinking mode (DeepSeek/Qwen) can consume the whole output budget
        # before emitting content, so disable it for these ledger calls.
        extra = _thinking_extra_body(self.config.provider, thinking=self.config.thinking)
        if extra:
            request_kwargs["extra_body"] = extra
        response = self._client.chat.completions.create(
            **request_kwargs,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("Reader LLM returned no choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = _chat_message_text(getattr(message, "content", "") if message is not None else "")
        attempts = debug.setdefault("completion_attempts", [])
        attempts.append({
            "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
            "content_chars": len(content),
            "reasoning_chars": len(str(getattr(message, "reasoning_content", "") or "")),
        })
        return content


def _parse_json_object_with_status(text: str) -> tuple[dict[str, Any], bool]:
    raw = str(text or "").strip()
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload, True
    return {}, False


def _parse_json_object(text: str) -> dict[str, Any]:
    return _parse_json_object_with_status(text)[0]


def _complete_set_candidates_from_context(memory_context: str) -> list[EvidenceCandidate]:
    candidates: list[EvidenceCandidate] = []
    seen: set[str] = set()
    for line in str(memory_context or "").splitlines():
        match = re.match(r"^- \[([^]]+)\]\s*(.*)$", line.strip())
        if not match:
            continue
        labels: dict[str, str] = {}
        for part in match.group(1).split(";"):
            key, separator, value = part.strip().partition("=")
            if separator:
                labels[key.strip()] = value.strip()
        source_id = labels.get("source_id", "")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        occurred_at = _iso_timestamp(labels.get("event_time", ""))
        recorded_at = _iso_timestamp(labels.get("recorded_at", ""))
        candidates.append(EvidenceCandidate(
            source_id=source_id,
            source_type=labels.get("source", "memory_context"),
            text=match.group(2).strip(),
            occurred_at=occurred_at,
            recorded_at=recorded_at,
            memory_id=source_id.removeprefix("memory:") if source_id.startswith("memory:") else "",
            evidence_ids=(
                tuple(
                    evidence_id.strip().removeprefix("timeline:")
                    for evidence_id in labels.get("evidence_ids", "").split(",")
                    if evidence_id.strip()
                )
                if source_id.startswith("memory:")
                else (source_id.removeprefix("timeline:"),)
            ),
            status=labels.get("status", "active"),
            superseded_by=labels.get("superseded_by", ""),
        ))
    return candidates


def _iso_timestamp(value: str) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def _context_timestamp_float(value: Any) -> float | None:
    """Return the same minute-truncated timestamp the line-based context emits.

    ``_format_context_timestamp`` rounds to minutes before serializing, and the
    line-based parser converts it back.  Reusing that exact round-trip keeps the
    structured path's occurred_at/recorded_at byte-consistent with the legacy
    path for single-line sources, so only multi-line text retention changes.
    """

    return _iso_timestamp(_format_context_timestamp(value))


def _complete_set_candidates_from_structured(
    memories: list[dict[str, Any]],
    timeline_chunks: list[dict[str, Any]],
) -> tuple[list[EvidenceCandidate], tuple[SourceEnvelope, ...]]:
    """Build complete-set candidates losslessly from structured recall payloads.

    The legacy ``_complete_set_candidates_from_context`` round-trip keeps only
    the first line of a multi-line source, because its `[label] text` format and
    line-based parser cannot distinguish embedded newlines from record
    boundaries.  This path reads the structured memory and Timeline payloads
    directly so every source's full decoded text survives, then derives stable
    ``S1...Sn`` aliases through the source envelope so the alias-to-real-ID
    mapping and text integrity remain verifiable without any model call.
    """

    candidates: list[EvidenceCandidate] = []
    seen: set[str] = set()

    def _add(source_id: str, candidate: EvidenceCandidate) -> None:
        if source_id and source_id not in seen:
            seen.add(source_id)
            candidates.append(candidate)

    for item in memories if isinstance(memories, list) else []:
        if not isinstance(item, dict):
            continue
        memory_id = str(item.get("id") or "").strip()
        if not memory_id:
            continue
        evidence_ids = tuple(
            dict.fromkeys(
                str(evidence_id).strip().removeprefix("timeline:")
                for evidence_id in (item.get("evidence_ids") or [])
                if str(evidence_id).strip()
            )
        )
        _add(
            f"memory:{memory_id}",
            EvidenceCandidate(
                source_id=f"memory:{memory_id}",
                source_type="structured_memory",
                text=str(item.get("content") or ""),
                occurred_at=_context_timestamp_float(
                    item.get("start_at") or item.get("occurred_at")
                ),
                recorded_at=None,
                memory_id=memory_id,
                evidence_ids=evidence_ids,
                status=str(item.get("status") or "active"),
                superseded_by=str(item.get("superseded_by") or ""),
            ),
        )

    for item in timeline_chunks if isinstance(timeline_chunks, list) else []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("id") or "").strip()
        if not chunk_id:
            continue
        _add(
            f"timeline:{chunk_id}",
            EvidenceCandidate(
                source_id=f"timeline:{chunk_id}",
                source_type="timeline",
                text=str(item.get("text") or ""),
                occurred_at=None,
                recorded_at=_context_timestamp_float(item.get("timestamp")),
                memory_id="",
                evidence_ids=(chunk_id,),
                status=str(item.get("status") or "active"),
                superseded_by="",
            ),
        )

    envelopes = build_source_envelopes([
        SourceEnvelopeInput(
            source_id=candidate.source_id,
            source_type=candidate.source_type,
            text=candidate.text,
        )
        for candidate in candidates
    ])
    return candidates, envelopes


def _batch_complete_set_candidates(
    candidates: list[EvidenceCandidate],
    *,
    max_context_chars: int,
    max_batches: int,
    max_candidates_per_batch: int | None = None,
) -> tuple[list[list[EvidenceCandidate]], bool]:
    return batch_evidence_candidates(
        candidates,
        max_context_chars=max_context_chars,
        max_batches=max_batches,
        max_candidates_per_batch=max_candidates_per_batch,
    )


def build_complete_set_ledger_prompt(
    *,
    question: str,
    question_type: str,
    question_date: str,
    answer_task: dict[str, Any],
    candidates: list[EvidenceCandidate],
    batch_index: int,
    batch_count: int,
) -> str:
    sources = [
        {
            "source_id": candidate.source_id,
            "source_type": candidate.source_type,
            "text": candidate.text,
            "occurred_at": candidate.occurred_at,
            "recorded_at": candidate.recorded_at,
            "status": candidate.status,
            "superseded_by": candidate.superseded_by,
        }
        for candidate in candidates
    ]
    fixed_task = {
        key: answer_task.get(key)
        for key in ("answer_intent", "answer_focus", "answer_obligations", "uncertainty_policy")
        if key in answer_task
    }
    return (
        "You are the evidence-accounting Reader for a memory question. The answer task below is fixed. "
        "Do not change its scope, intent, or obligations and do not request more memory.\n"
        "For every input source_id, decide within answer_focus whether it is included, excluded, or uncertain. "
        "Record that classification in source_decisions, where every source_id appears exactly once. "
        "Put only included facts in items. Each readable item label must preserve any source-supported entity, "
        "qualifier, or temporal detail needed by the fixed obligations, without inventing details or changing schema. "
        "Item source_ids are provenance: the same source_id may appear in "
        "several items only when that one source explicitly states several distinct in-scope facts. "
        "A source is included when any part supplies an in-scope fact; ignore extra background in that source. "
        "Repeated support for one fact uses the same canonical_key; excluded is never a deduplication marker. "
        "Use the same canonical_key for repeated mentions of one item. "
        "Use occurred_at, recorded_at, status, and superseded_by so later corrections or invalidations override older state. "
        "Choose count or sum and a unit, but do not calculate a value or write final_answer; deterministic "
        "code does both later. Do not infer facts absent from source text.\n"
        f"Batch: {batch_index + 1}/{batch_count}\n"
        f"Question type: {question_type}\nQuestion date: {question_date}\nQuestion: {question}\n"
        f"Fixed answer task: {json.dumps(fixed_task, ensure_ascii=False, sort_keys=True)}\n"
        f"Sources: {json.dumps(sources, ensure_ascii=False, sort_keys=True)}\n"
        "Return valid JSON only with exactly this shape:\n"
        '{"source_decisions":[{"source_id":"source-id","status":"included|excluded|uncertain"}],'
        '"items":[{"canonical_key":"stable event or item key","label":"user-readable label",'
        '"quantity":"1","unit":"item","status":"included|excluded|uncertain",'
        '"source_ids":["source-id"]}],'
        '"aggregation":{"operation":"count|sum","unit":"item"}}'
    )


def build_complete_set_final_prompt(
    *,
    question: str,
    answer_task: dict[str, Any],
    ledger: dict[str, Any],
) -> str:
    fixed_task = {
        key: answer_task.get(key)
        for key in (
            "answer_intent",
            "answer_focus",
            "answer_obligations",
            "uncertainty_policy",
            "coverage_requirement",
            "coverage_complete",
        )
        if key in answer_task
    }
    return (
        "Write only the final user-facing answer from this already validated evidence ledger. "
        "Do not add, remove, merge, or reinterpret ledger items. Include the exact verified aggregation.value. "
        "Use the language of the question and keep the answer concise. Cover every fixed answer obligation "
        "that the validated ledger supports. When entities is required, name the readable ledger item labels; "
        "preserve supported qualifiers or temporal relations when those obligations are required. Count scope "
        "alone does not require an entity list. If the ledger does not support an obligation, follow the fixed "
        "uncertainty policy instead of inventing content.\n"
        f"Question: {question}\n"
        f"Fixed answer task: {json.dumps(fixed_task, ensure_ascii=False, sort_keys=True)}\n"
        f"Validated ledger: {json.dumps(ledger, ensure_ascii=False, sort_keys=True)}\n"
        'Return JSON only: {"final_answer":"..."}'
    )


def build_complete_set_consolidation_prompt(
    *,
    question: str,
    answer_task: dict[str, Any],
    batch_ledgers: list[dict[str, object]],
) -> str:
    fixed_task = {
        key: answer_task.get(key)
        for key in ("answer_intent", "answer_focus", "answer_obligations", "uncertainty_policy")
        if key in answer_task
    }
    return (
        "You are the final evidence-accounting Reader for already reviewed batch ledgers. The answer task "
        "is fixed; do not change scope, recall memory, or act as a planner. Re-evaluate batch-local included "
        "items against answer_focus, exclude items that do not satisfy the full focus, and merge repeated "
        "facts across batches under one stable canonical_key. Classify every source exactly once in "
        "source_decisions. Put only supported facts in items; item source_ids are provenance and may repeat "
        "only when one source explicitly supports several distinct facts. Choose count or sum and a unit, "
        "but do not calculate a value or write final_answer; deterministic code does both later.\n"
        f"Question: {question}\n"
        f"Fixed answer task: {json.dumps(fixed_task, ensure_ascii=False, sort_keys=True)}\n"
        f"Batch ledgers: {json.dumps(batch_ledgers, ensure_ascii=False, sort_keys=True)}\n"
        "Return valid JSON only with exactly this shape:\n"
        '{"source_decisions":[{"source_id":"source-id","status":"included|excluded|uncertain"}],'
        '"items":[{"canonical_key":"stable key","label":"readable label","quantity":"1",'
        '"unit":"item","status":"included","source_ids":["source-id"]}],'
        '"aggregation":{"operation":"count|sum","unit":"item"}}'
    )


def _parse_reader_structured(text: str) -> dict[str, Any] | None:
    """Parse Reader JSON and preserve whether it carried an evidence field."""
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        evidence = parsed.get("relevant_evidence")
        final = parsed.get("final_answer")
        coverage = parsed.get("coverage")
        normalized_coverage: list[str] = []
        if isinstance(coverage, list):
            normalized_coverage = [str(item).strip()[:100] for item in coverage if str(item).strip()]
        elif isinstance(coverage, str):
            normalized_coverage = [part.strip()[:100] for part in coverage.split(",") if part.strip()]
        if evidence is not None and isinstance(evidence, list):
            return {
                "output_schema": "evidence_and_final_answer",
                "relevant_evidence": [str(item).strip()[:500] for item in evidence if str(item).strip()],
                "coverage": normalized_coverage,
                "final_answer": str(final).strip() if final is not None else "",
            }
        if final is not None and str(final).strip():
            return {
                "output_schema": "final_answer_only",
                "relevant_evidence": [],
                "coverage": normalized_coverage,
                "final_answer": str(final).strip(),
            }
    return None


class ProgressReporter:
    def __init__(self, total: int, *, stream: TextIO = sys.stderr, width: int = 20) -> None:
        self.total = max(0, int(total))
        self.stream = stream
        self.width = max(10, int(width))
        self.started = time.perf_counter()
        self.completed = 0
        self.succeeded = 0
        self.failed = 0
        self._interactive = bool(getattr(stream, "isatty", lambda: False)())
        self._line_open = False

    def update(self, *, question_id: str, phase: str) -> None:
        line = self._render(question_id=question_id, phase=phase)
        if self._interactive:
            self.stream.write("\r" + line)
            self.stream.flush()
            self._line_open = True
        else:
            self.stream.write(line + "\n")
            self.stream.flush()

    def complete(self, *, question_id: str, succeeded: bool) -> None:
        self.completed += 1
        if succeeded:
            self.succeeded += 1
        else:
            self.failed += 1
        self.update(question_id=question_id, phase="complete" if succeeded else "error")

    def finish(self) -> None:
        if self._interactive and self._line_open:
            self.stream.write("\n")
            self.stream.flush()
            self._line_open = False

    def _render(self, *, question_id: str, phase: str) -> str:
        ratio = (self.completed / self.total) if self.total else 1.0
        filled = min(self.width, int(ratio * self.width))
        if filled >= self.width:
            bar = "=" * self.width
        else:
            bar = "=" * filled + ">" + "." * max(0, self.width - filled - 1)
        elapsed = max(0.0, time.perf_counter() - self.started)
        eta = None
        if self.completed > 0 and self.completed < self.total:
            eta = (elapsed / self.completed) * (self.total - self.completed)
        return (
            f"[{bar}] {self.completed}/{self.total} {ratio * 100:5.1f}%  "
            f"success={self.succeeded} failed={self.failed}  "
            f"phase={phase} question={question_id}  "
            f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}"
        )


ServiceFactory = Callable[[MutableClock], Any]
ProgressCallback = Callable[[str], None]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    load_app_dotenv()
    reader_config = resolve_reader_config(args)
    app_llm_env = snapshot_app_llm_env()
    items = select_items(args)
    workers = max(1, int(args.workers))
    cache_enabled = not bool(args.no_cache)
    cache_root = prepare_cache_root(
        resolve_cache_root(args) if cache_enabled else None,
        dataset_path=Path(args.dataset_path).resolve(),
        history_mode=str(args.history_mode),
        app_model=app_llm_env.get("AI_GLASSES_LLM_MODEL", "") or os.getenv("AI_GLASSES_LLM_MODEL", ""),
    )
    config = {
        "dataset_path": str(args.dataset_path),
        "limit": args.limit,
        "question_ids": args.question_id,
        "question_types": args.question_type,
        "history_mode": args.history_mode,
        "background_wait": args.background_wait,
        "workers": workers,
        "cache": {
            "enabled": cache_enabled,
            "root": str(cache_root) if cache_root is not None else "",
            "version": LONGMEMEVAL_CACHE_VERSION,
        },
        "reader_provider": reader_config.provider,
        "reader_model": reader_config.model,
        "reader_base_url": reader_config.base_url,
        "reader_max_context_chars": reader_config.max_context_chars,
        "reader_max_tokens": reader_config.max_tokens,
        "reader_temperature": reader_config.temperature,
    }
    paths = resolve_run_paths(args)
    manifest = build_run_manifest(items=items, config=config)
    completed_runs = prepare_checkpoint_run(
        paths,
        manifest=manifest,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )

    reader = OpenAIReader(reader_config)
    progress = ProgressReporter(len(items))
    original_app_home = os.environ.get(APP_HOME_ENV)
    try:
        runs = run_benchmark_items(
            items,
            reader=reader,
            paths=paths,
            progress=progress,
            app_llm_env=app_llm_env,
            original_app_home=original_app_home,
            background_wait=max(0.0, float(args.background_wait)),
            history_mode=str(args.history_mode),
            keep_homes=bool(args.keep_homes),
            config=config,
            completed_runs=completed_runs,
            max_new_cases=max(0, int(args.max_new_cases)),
            workers=workers,
            cache_root=cache_root,
            cache_enabled=cache_enabled,
        )
    finally:
        progress.finish()
        _restore_app_home(original_app_home)

    summary = summarize_longmemeval_runs(runs)
    write_progress_outputs(paths=paths, runs=runs, config=config)

    failed = sum(1 for run in runs if run.get("status") != "success")
    console_summary = {
        "instances_requested": len(items),
        "instances_succeeded": len(items) - failed,
        "instances_failed": failed,
        "output": str(paths.brief_output),
        "detail_output": str(paths.detail_output),
        "json_report": str(paths.output_dir / "eval-latest.json"),
        "markdown_report": str(paths.output_dir / "eval-latest.md"),
        "reader_model": reader_config.model,
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LongMemEval through the AI glasses memory assistant.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_ORACLE_PATH)
    parser.add_argument("--limit", type=int, default=20, help="0 means all matching instances.")
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--question-type", action="append", default=[])
    parser.add_argument("--include-abstention", action="store_true", default=True)
    parser.add_argument("--output-root-dir", type=Path, default=DEFAULT_OUTPUT_ROOT_DIR)
    parser.add_argument("--output-dir", "--report-dir", dest="output_dir", type=Path)
    parser.add_argument("--detail-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume a matching checkpointed run.")
    parser.add_argument(
        "--max-new-cases",
        type=int,
        default=0,
        help="Run at most this many unfinished cases; 0 means all unfinished cases.",
    )
    parser.add_argument("--background-wait", type=float, default=15.0)
    parser.add_argument("--history-mode", choices=("import", "timeline", "chat"), default="import")
    parser.add_argument("--keep-homes", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for memory-context building; the Reader stays serial.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_ROOT_DIR,
        help="Root for the two-level per-question cache.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the two-level per-question cache (import + recall).",
    )
    parser.add_argument("--reader-provider")
    parser.add_argument("--reader-model")
    parser.add_argument("--reader-base-url")
    parser.add_argument("--reader-api-key")
    parser.add_argument("--reader-timeout", type=int, default=DEFAULT_READER_TIMEOUT)
    parser.add_argument("--reader-max-tokens", type=int, default=DEFAULT_READER_MAX_TOKENS)
    parser.add_argument("--reader-temperature", type=float, default=0.0)
    parser.add_argument(
        "--reader-thinking",
        choices=("disabled", "enabled"),
        default="disabled",
        help="Disable provider thinking mode for reader calls (default: disabled).",
    )
    parser.add_argument(
        "--reader-max-context-chars",
        type=int,
        default=DEFAULT_READER_MAX_CONTEXT_CHARS,
    )
    return parser


def resolve_reader_config(args: argparse.Namespace) -> ReaderConfig:
    provider = str(args.reader_provider or os.getenv(LLM_PROVIDER_ENV) or DEEPSEEK_FALLBACK_PROVIDER).strip()
    model = str(args.reader_model or os.getenv(LLM_MODEL_ENV) or "").strip()
    base_url = str(args.reader_base_url or os.getenv(LLM_BASE_URL_ENV) or "").strip().rstrip("/")
    api_key = str(
        args.reader_api_key
        or os.getenv(LLM_API_KEY_ENV)
        or (os.getenv(DEEPSEEK_API_KEY_ENV) if provider == DEEPSEEK_FALLBACK_PROVIDER else "")
        or ""
    ).strip()
    missing = []
    if not model:
        missing.append("--reader-model or AI_GLASSES_LLM_MODEL")
    if not base_url:
        missing.append("--reader-base-url or AI_GLASSES_LLM_BASE_URL")
    if not api_key:
        missing.append("--reader-api-key, AI_GLASSES_LLM_API_KEY, or DEEPSEEK_API_KEY")
    if missing:
        raise ValueError("LongMemEval reader requires: " + ", ".join(missing))
    return ReaderConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=max(1, int(args.reader_timeout)),
        max_tokens=max(1, int(args.reader_max_tokens)),
        temperature=float(args.reader_temperature),
        max_context_chars=max(0, int(args.reader_max_context_chars)),
        thinking=str(args.reader_thinking),
    )


def select_items(args: argparse.Namespace) -> list[LongMemEvalItem]:
    items = load_longmemeval_items(
        args.dataset_path,
        limit=0,
        question_types={str(item) for item in args.question_type if str(item).strip()} or None,
        include_abstention=bool(args.include_abstention),
    )
    requested_ids = [str(item).strip() for item in args.question_id if str(item).strip()]
    if requested_ids:
        by_id = {item.question_id: item for item in items}
        missing = [question_id for question_id in requested_ids if question_id not in by_id]
        if missing:
            raise ValueError("Unknown LongMemEval question_id: " + ", ".join(missing))
        items = [by_id[question_id] for question_id in requested_ids]
    if int(args.limit) > 0 and not requested_ids:
        items = items[: int(args.limit)]
    if not items:
        raise ValueError("No LongMemEval items matched the requested filters.")
    return items


def resolve_cache_root(args: argparse.Namespace) -> Path:
    cache_base = Path(args.cache_dir or DEFAULT_CACHE_ROOT_DIR)
    return (cache_base / _safe_output_name(Path(args.dataset_path).stem)).resolve()


def prepare_cache_root(
    cache_root: Path | None,
    *,
    dataset_path: Path,
    history_mode: str,
    app_model: str = "",
) -> Path | None:
    """Validate the dataset-level cache manifest; rebuild on any key mismatch."""
    if cache_root is None:
        return None
    manifest = {
        "version": LONGMEMEVAL_CACHE_VERSION,
        "dataset_sha256": _file_sha256(dataset_path),
        "history_mode": str(history_mode),
        # The cache snapshots each question's import library + recall, which are
        # produced by the app LLM and the memory-pipeline code. Both must be part
        # of the key, or a model/code change silently reuses a stale snapshot
        # (the cross-model contamination bug). ``source_snapshot`` is the git
        # HEAD + uncommitted-diff hash from _source_snapshot().
        "app_model": str(app_model or ""),
        "source_snapshot": _source_snapshot(),
    }
    manifest_path = cache_root / "cache-manifest.json"
    existing = None
    if manifest_path.is_file():
        try:
            existing = _load_json_object(manifest_path)
        except (ValueError, OSError, json.JSONDecodeError):
            existing = None
    if existing != manifest:
        if cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(manifest_path, manifest)
    return cache_root


def _question_cache_dir(cache_root: Path | None, question_id: str) -> Path | None:
    if cache_root is None:
        return None
    return cache_root / _safe_output_name(question_id)


def resolve_run_paths(args: argparse.Namespace) -> RunPaths:
    input_stem = _safe_output_name(Path(args.dataset_path).stem)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.output_root_dir) / f"{input_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    brief_output = output_dir / f"{input_stem}_memory.jsonl"
    if args.detail_output:
        detail_output = Path(args.detail_output)
        if not detail_output.is_absolute():
            detail_output = output_dir / detail_output
    else:
        detail_output = output_dir / f"{input_stem}_memory.jsonl.details.json"
    return RunPaths(output_dir=output_dir, brief_output=brief_output, detail_output=detail_output)


def prepare_run_paths(paths: RunPaths, *, overwrite: bool) -> None:
    outputs = [
        paths.brief_output,
        paths.detail_output,
        paths.output_dir / "eval-latest.json",
        paths.output_dir / "eval-latest.md",
        paths.output_dir / "run-manifest.json",
        paths.output_dir / "cases",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists. Pass --overwrite to replace:\n  "
            + "\n  ".join(str(path) for path in existing)
        )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.detail_output.parent.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    paths.brief_output.write_text("", encoding="utf-8")


def build_run_manifest(*, items: list[LongMemEvalItem], config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(str(config["dataset_path"])).resolve()
    return {
        "schema": CHECKPOINT_SCHEMA_VERSION,
        "dataset_sha256": _file_sha256(dataset_path),
        "source_snapshot": _source_snapshot(),
        "config": config,
        "question_ids": [item.question_id for item in items],
    }


def manifests_compatible_for_resume(existing: dict[str, Any], manifest: dict[str, Any]) -> bool:
    # Resume only needs to guard against changes that can alter per-case results.
    # `config.workers` is pure parallelism and never changes an individual case's
    # output, so it must not block a legitimate resume. `source_snapshot` is kept
    # for provenance but a launcher-only diff (e.g. scripts/run.sh) also must not
    # block; we still require the dataset, schema, question set, and the
    # result-affecting config (history mode, reader, cache, background_wait) to match.
    if existing.get("schema") != manifest.get("schema"):
        return False
    if existing.get("dataset_sha256") != manifest.get("dataset_sha256"):
        return False
    if existing.get("question_ids") != manifest.get("question_ids"):
        return False
    ec = dict(existing.get("config", {}))
    mc = dict(manifest.get("config", {}))
    ec.pop("workers", None)
    mc.pop("workers", None)
    return ec == mc


def prepare_checkpoint_run(
    paths: RunPaths,
    *,
    manifest: dict[str, Any],
    overwrite: bool,
    resume: bool,
) -> dict[str, dict[str, Any]]:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    manifest_path = paths.output_dir / "run-manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No checkpoint manifest to resume: {manifest_path}")
        existing = _load_json_object(manifest_path)
        if not manifests_compatible_for_resume(existing, manifest):
            raise ValueError("Checkpoint manifest does not match this dataset, configuration, or source snapshot")
        return load_completed_case_runs(paths, question_ids=list(manifest["question_ids"]))

    prepare_run_paths(paths, overwrite=overwrite)
    cases_dir = paths.output_dir / "cases"
    if cases_dir.exists():
        shutil.rmtree(cases_dir)
    cases_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest_path, manifest)
    return {}


def load_completed_case_runs(paths: RunPaths, *, question_ids: list[str]) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for index, question_id in enumerate(question_ids, start=1):
        marker = case_dir(paths, index=index, question_id=question_id) / "completed.json"
        if not marker.is_file():
            continue
        result = _load_json_object(marker)
        if result.get("question_id") != question_id:
            raise ValueError(f"Invalid checkpoint result for {question_id}")
        completed[question_id] = result
    return completed


def case_dir(paths: RunPaths, *, index: int, question_id: str) -> Path:
    return paths.output_dir / "cases" / f"{index:03d}-{_safe_output_name(question_id)}"


def persist_case_artifact(
    paths: RunPaths,
    *,
    index: int,
    result: dict[str, Any],
    config: dict[str, Any],
    keep_homes: bool,
) -> None:
    destination = case_dir(paths, index=index, question_id=str(result.get("question_id") or "unknown"))
    if destination.exists():
        raise FileExistsError(f"Checkpoint case directory already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".case-", dir=destination.parent))
    app_home = Path(str(result.get("app_home") or ""))
    try:
        sanitized_result = GlassesChatService._redact_audit_payload(dict(result))
        if not keep_homes:
            sanitized_result["app_home"] = "<temporary>"
        _atomic_write_json(temporary / "result.json", sanitized_result)
        _atomic_write_json(temporary / "config.json", config)
        evidence = export_case_evidence(app_home, temporary)
        _atomic_write_json(temporary / "evidence-export.json", evidence)
        # The directory is renamed only after this terminal marker exists.
        _atomic_write_json(temporary / "completed.json", sanitized_result)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def export_case_evidence(app_home: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    data_dir = app_home / "data"
    exported: list[str] = []
    skipped: dict[str, str] = {}
    audit_path = data_dir / "chat_audit.jsonl"
    if audit_path.is_file():
        lines: list[str] = []
        for raw_line in audit_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            lines.append(json.dumps(GlassesChatService._redact_audit_payload(record), ensure_ascii=False))
        (destination / "audit.redacted.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported.append("audit.redacted.jsonl")
    database_dir = destination / "database"
    for name in ("events.db", "timeline.db"):
        source = data_dir / name
        if not source.is_file():
            skipped[name] = "missing"
            continue
        try:
            database_dir.mkdir(parents=True, exist_ok=True)
            target = database_dir / name
            _backup_sqlite(source, target)
            _redact_sqlite_text(target)
            exported.append(f"database/{name}")
        except (OSError, sqlite3.Error) as exc:
            skipped[name] = type(exc).__name__
    return {"exported": exported, "skipped": skipped}


def _backup_sqlite(source: Path, target: Path) -> None:
    source_connection = sqlite3.connect(str(source))
    target_connection = sqlite3.connect(str(target))
    try:
        source_connection.backup(target_connection)
    finally:
        target_connection.close()
        source_connection.close()


def _redact_sqlite_text(path: Path) -> None:
    connection = sqlite3.connect(str(path))
    try:
        tables = [str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        for table in tables:
            if table.startswith("sqlite_"):
                continue
            columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            for _index, name, column_type, *_rest in columns:
                if "TEXT" not in str(column_type or "").upper():
                    continue
                for row_id, value in connection.execute(
                    f'SELECT rowid, "{name}" FROM "{table}" WHERE "{name}" IS NOT NULL'
                ).fetchall():
                    redacted = redact_sensitive_text(str(value)).text
                    if redacted != value:
                        connection.execute(
                            f'UPDATE "{table}" SET "{name}" = ? WHERE rowid = ?',
                            (redacted, row_id),
                        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()


def classify_offline_failure(result: dict[str, Any]) -> dict[str, str]:
    """Classify only report artifacts; this is never used by runtime code."""
    if bool(result.get("is_abstention")) and bool(result.get("answer_hit")):
        return {"stage": "expected_abstention", "reason": "expected_abstention_answered"}
    if result.get("status") != "success" or int(result.get("failed_import_count") or 0):
        return {"stage": "write_or_type_loss", "reason": "import_or_runtime_failure"}
    if bool(result.get("answer_hit")):
        return {"stage": "evidence_supported_answer", "reason": "local_answer_hit"}
    context = str(result.get("recall_context") or "")
    debug = result.get("response_debug") if isinstance(result.get("response_debug"), dict) else {}
    memory_debug = debug.get("memory") if isinstance(debug.get("memory"), dict) else {}
    timeline_debug = debug.get("timeline") if isinstance(debug.get("timeline"), dict) else {}
    turn_decision = debug.get("turn_decision") if isinstance(debug.get("turn_decision"), dict) else {}
    planner = turn_decision.get("final") if isinstance(turn_decision.get("final"), dict) else debug.get("planner")
    planner = planner if isinstance(planner, dict) else {}
    route_requested = any(bool(planner.get(field)) for field in (
        "needs_profile_memory",
        "needs_event_memory",
        "needs_timeline_recall",
        "needs_discussion_recall",
    ))
    event_strategy = str((memory_debug.get("event_recall") or {}).get("strategy") or "") if isinstance(memory_debug.get("event_recall"), dict) else ""
    timeline_strategy = str((timeline_debug.get("recall") or {}).get("strategy") or "") if isinstance(timeline_debug.get("recall"), dict) else ""
    recall_executed = (
        int(result.get("recalled_memory_count") or 0) > 0
        or int(result.get("recalled_timeline_count") or 0) > 0
        or (bool(event_strategy) and not event_strategy.startswith("skipped"))
        or (bool(timeline_strategy) and not timeline_strategy.startswith("skipped"))
    )
    if not route_requested and not recall_executed and not context:
        return {"stage": "routing_miss", "reason": "all_recall_routes_off"}
    offline_evidence = result.get("offline_evidence") if isinstance(result.get("offline_evidence"), dict) else {}
    if offline_evidence.get("candidate_support") == "verified" and not bool(offline_evidence.get("selected_support")):
        return {"stage": "ranking_or_context_loss", "reason": "verified_candidate_not_selected"}
    if offline_evidence.get("selected_support") == "verified" and not bool(result.get("answer_hit")):
        return {"stage": "reader_synthesis_loss", "reason": "verified_selected_evidence_not_used"}
    if route_requested and not context:
        return {"stage": "retrieval_coverage_loss", "reason": "requested_route_returned_empty_context"}
    return {"stage": "unclassified_insufficient_evidence", "reason": "requires_offline_evidence_review"}


def write_progress_outputs(*, paths: RunPaths, runs: list[dict[str, Any]], config: dict[str, Any]) -> None:
    write_json_output(paths.detail_output, runs)
    summary = summarize_longmemeval_runs(runs)
    write_longmemeval_report(output_dir=paths.output_dir, summary=summary, runs=runs, config=config)


def _append_brief_result_once(path: Path, result: dict[str, Any]) -> None:
    question_id = str(result["question_id"])
    if path.is_file():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(raw_line).get("question_id") == question_id:
                    return
            except json.JSONDecodeError:
                continue
    write_jsonl_row(path, {"question_id": question_id, "hypothesis": result["hypothesis"]})


def _remove_ephemeral_app_home(result: dict[str, Any], *, keep_homes: bool) -> None:
    if keep_homes:
        return
    app_home = Path(str(result.get("app_home") or ""))
    if app_home.is_dir() and app_home.name.startswith("glasses-longmemeval-"):
        shutil.rmtree(app_home, ignore_errors=True)
    result["app_home"] = "<temporary>"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_snapshot() -> dict[str, str]:
    def git_output(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"head": git_output("rev-parse", "HEAD"), "tracked_diff_sha256": hashlib.sha256(
        git_output("diff", "--binary", "HEAD").encode("utf-8")
    ).hexdigest()}


def run_benchmark_items(
    items: list[LongMemEvalItem],
    *,
    reader: Reader,
    paths: RunPaths,
    progress: ProgressReporter,
    app_llm_env: dict[str, str],
    original_app_home: str | None,
    background_wait: float,
    history_mode: str,
    keep_homes: bool,
    service_factory: ServiceFactory | None = None,
    config: dict[str, Any] | None = None,
    completed_runs: dict[str, dict[str, Any]] | None = None,
    max_new_cases: int = 0,
    workers: int = 1,
    cache_root: Path | None = None,
    cache_enabled: bool = False,
) -> list[dict[str, Any]]:
    completed_runs = completed_runs or {}
    if workers > 1:
        return _run_benchmark_items_parallel(
            items,
            reader=reader,
            paths=paths,
            progress=progress,
            app_llm_env=app_llm_env,
            original_app_home=original_app_home,
            background_wait=background_wait,
            history_mode=history_mode,
            keep_homes=keep_homes,
            service_factory=service_factory,
            config=config,
            completed_runs=completed_runs,
            max_new_cases=max_new_cases,
            workers=workers,
            cache_root=cache_root,
            cache_enabled=cache_enabled,
        )
    runs: list[dict[str, Any]] = []
    executed_count = 0
    for index, item in enumerate(items, start=1):
        completed = completed_runs.get(item.question_id)
        if completed is not None:
            runs.append(completed)
            progress.complete(question_id=item.question_id, succeeded=completed.get("status") == "success")
            continue
        if max_new_cases and executed_count >= max_new_cases:
            break
        restore_app_llm_env(app_llm_env)
        phase_callback = lambda phase, item=item: progress.update(question_id=item.question_id, phase=phase)
        if cache_enabled and cache_root is not None:
            context = build_question_memory_context(
                item,
                background_wait=background_wait,
                history_mode=history_mode,
                original_app_home=original_app_home,
                progress_callback=phase_callback,
                service_factory=service_factory,
                cache_dir=_question_cache_dir(cache_root, item.question_id),
                cache_enabled=True,
            )
            result = answer_question_from_memory_context(context, reader=reader)
        else:
            result = run_longmemeval_item(
                item,
                reader=reader,
                background_wait=background_wait,
                history_mode=history_mode,
                keep_home=keep_homes or config is not None,
                original_app_home=original_app_home,
                progress_callback=phase_callback,
                service_factory=service_factory,
            )
        result["failure_classification"] = classify_offline_failure(result)
        succeeded = result.get("status") == "success"
        if succeeded:
            _append_brief_result_once(paths.brief_output, result)
        if config is not None:
            # Keep the isolated app home until its sanitized evidence snapshot is durable.
            persist_case_artifact(paths, index=index, result=result, config=config, keep_homes=keep_homes)
        _remove_ephemeral_app_home(result, keep_homes=keep_homes)
        if config is not None:
            write_progress_outputs(paths=paths, runs=[*runs, result], config=config)
        runs.append(result)
        executed_count += 1
        progress.complete(question_id=item.question_id, succeeded=succeeded)
    return runs


def _run_benchmark_items_parallel(
    items: list[LongMemEvalItem],
    *,
    reader: Reader,
    paths: RunPaths,
    progress: ProgressReporter,
    app_llm_env: dict[str, str],
    original_app_home: str | None,
    background_wait: float,
    history_mode: str,
    keep_homes: bool,
    service_factory: ServiceFactory | None,
    config: dict[str, Any] | None,
    completed_runs: dict[str, dict[str, Any]],
    max_new_cases: int,
    workers: int,
    cache_root: Path | None,
    cache_enabled: bool,
) -> list[dict[str, Any]]:
    # 先按串行语义筛选出本轮要执行的题，保留 checkpoint/max_new_cases 行为。
    submitted: list[tuple[int, LongMemEvalItem]] = []
    executed_count = 0
    for index, item in enumerate(items, start=1):
        if item.question_id in completed_runs:
            progress.complete(
                question_id=item.question_id,
                succeeded=completed_runs[item.question_id].get("status") == "success",
            )
            continue
        if max_new_cases and executed_count >= max_new_cases:
            break
        submitted.append((index, item))
        executed_count += 1

    results_by_index: dict[int, dict[str, Any]] = {}
    if submitted:
        restore_app_llm_env(app_llm_env)
        payloads = [
            (
                index,
                item,
                background_wait,
                history_mode,
                original_app_home,
                app_llm_env,
                service_factory,
                _question_cache_dir(cache_root, item.question_id) if cache_enabled else None,
                cache_enabled,
            )
            for index, item in submitted
        ]
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {
                executor.submit(_build_context_worker, payload): payload
                for payload in payloads
            }
            for future in as_completed(futures):
                index, item = futures[future][0], futures[future][1]
                progress.update(question_id=item.question_id, phase="context")
                try:
                    context = future.result()
                except Exception as exc:
                    # 防御性兜底：worker 本身已把题目级错误转成 error dict。
                    context = _worker_error_context(item, exc)
                # Reader 保持在主进程串行执行，避免并发限流。
                result = answer_question_from_memory_context(context, reader=reader)
                result["failure_classification"] = classify_offline_failure(result)
                succeeded = result.get("status") == "success"
                if succeeded:
                    _append_brief_result_once(paths.brief_output, result)
                if config is not None:
                    persist_case_artifact(
                        paths,
                        index=index,
                        result=result,
                        config=config,
                        keep_homes=keep_homes,
                    )
                _remove_ephemeral_app_home(result, keep_homes=keep_homes)
                results_by_index[index] = result
                progress.complete(question_id=item.question_id, succeeded=succeeded)

    # 按题号顺序组装最终报告，语义与串行路径完全一致。
    ordered: list[dict[str, Any]] = []
    executed_count = 0
    for index, item in enumerate(items, start=1):
        completed = completed_runs.get(item.question_id)
        if completed is not None:
            ordered.append(completed)
            continue
        if max_new_cases and executed_count >= max_new_cases:
            break
        executed_count += 1
        result = results_by_index.get(index)
        if result is None:
            continue
        ordered.append(result)
    if config is not None:
        write_progress_outputs(paths=paths, runs=ordered, config=config)
    return ordered


def _build_context_worker(payload: tuple) -> dict[str, Any]:
    """Module-level (picklable) worker entry for parallel context building."""
    (
        _index,
        item,
        background_wait,
        history_mode,
        original_app_home,
        app_llm_env,
        service_factory,
        cache_dir,
        cache_enabled,
    ) = payload
    return build_question_memory_context(
        item,
        background_wait=background_wait,
        history_mode=history_mode,
        original_app_home=original_app_home,
        service_factory=service_factory,
        app_llm_env=app_llm_env,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )


def _worker_error_context(item: LongMemEvalItem, exc: BaseException) -> dict[str, Any]:
    return {
        "question_id": item.question_id,
        "question_type": item.question_type,
        "is_abstention": item.is_abstention,
        "question": item.question,
        "answer": item.answer,
        "question_date": item.question_date,
        "hypothesis": "",
        "reply": "",
        "answer_hit": False,
        "recall_hit": False,
        **_empty_import_stats(item),
        "recall_context": "",
        "recall_context_chars": 0,
        "reader_input_chars": 0,
        "api_calls": 0,
        "response_debug": {},
        "recalled_memories": [],
        "recalled_timeline_chunks": [],
        "recalled_documents": [],
        "exception": repr(exc),
        "error": str(exc),
        "measured_seconds": 0.0,
        "phase_seconds": {},
        "app_home": "",
        "cache_hits": {"l1": False, "l2": False},
        "status": "error",
        "stage": "memory_context",
        "passed": False,
    }


def build_question_memory_context(
    item: LongMemEvalItem,
    *,
    background_wait: float,
    history_mode: str,
    original_app_home: str | None,
    progress_callback: ProgressCallback | None = None,
    service_factory: ServiceFactory | None = None,
    app_llm_env: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    cache_enabled: bool = False,
) -> dict[str, Any]:
    """Build the isolated app home + recall context for one benchmark item.

    Import and recall stay together (the expensive phases) so this whole function
    can run in a worker process; the Reader answers in the parent. Item-level
    failures never raise: the caller receives an error dict with status/stage.
    """
    if app_llm_env:
        restore_app_llm_env(app_llm_env)
    started = time.perf_counter()
    identity = {
        "question_id": item.question_id,
        "question_type": item.question_type,
        "is_abstention": item.is_abstention,
        "question": item.question,
        "answer": item.answer,
        "question_date": item.question_date,
    }
    app_home = Path(tempfile.mkdtemp(prefix=f"glasses-longmemeval-{item.question_id}-"))
    os.environ[APP_HOME_ENV] = str(app_home)
    question_timestamp = _timestamp(item.question_date) or time.time()
    clock = MutableClock(question_timestamp)
    factory = service_factory or _default_service_factory
    user_id = f"longmemeval-{item.question_id}"
    started = time.perf_counter()
    stage = "initialization"
    import_service: Any | None = None
    query_service: Any | None = None
    import_stats: dict[str, Any] = _empty_import_stats(item)
    phase_seconds: dict[str, float] = {}
    cache_hits = {"l1": False, "l2": False}
    try:
        cached = _load_cached_recall(cache_dir) if cache_enabled and cache_dir is not None else None
        if cached is not None:
            # L2 命中：import + recall 全部跳过，直接进入 Reader。
            _materialize_cached_app_home(cache_dir, app_home)
            cache_hits = {"l1": (cache_dir / "app_home").is_dir(), "l2": True}
            # 热跑阶段成本：import/recall 被缓存跳过，按 0 计入本次 run。
            phase_seconds = {"import": 0.0, "import_close": 0.0, "recall": 0.0}
            import_stats = dict(cached.get("import_stats") or _empty_import_stats(item))
            return {
                **identity,
                "hypothesis": "",
                "reply": "",
                "answer_hit": False,
                "recall_hit": False,
                **import_stats,
                "recall_context": str(cached.get("recall_context") or ""),
                "recall_context_chars": int(cached.get("recall_context_chars") or 0),
                "reader_input_chars": 0,
                "api_calls": int(cached.get("api_calls") or 0),
                "response_debug": cached.get("response_debug")
                if isinstance(cached.get("response_debug"), dict)
                else {},
                "recalled_memories": list(cached.get("recalled_memories") or []),
                "recalled_timeline_chunks": list(cached.get("recalled_timeline_chunks") or []),
                "recalled_documents": list(cached.get("recalled_documents") or []),
                "recalled_memory_count": len(cached.get("recalled_memories") or []),
                "recalled_timeline_count": len(cached.get("recalled_timeline_chunks") or []),
                "recalled_document_count": len(cached.get("recalled_documents") or []),
                "app_home": str(app_home),
                "measured_seconds": round(time.perf_counter() - started, 6),
                "phase_seconds": phase_seconds,
                "cache_hits": cache_hits,
                "status": "success",
                "stage": "recall",
            }

        l1_hit = (
            cache_enabled
            and cache_dir is not None
            and (cache_dir / "app_home").is_dir()
        )
        if l1_hit:
            # L1 命中：跳过 import，直接从缓存的隔离 app home 做 recall。
            _materialize_cached_app_home(cache_dir, app_home)
            cached_meta = _load_import_meta(cache_dir)
            if cached_meta:
                import_stats = cached_meta
            cache_hits["l1"] = True
            phase_seconds["import"] = 0.0
            phase_seconds["import_close"] = 0.0
        else:
            import_started = time.perf_counter()
            stage = "import"
            _report_phase(progress_callback, "import")
            import_service = factory(clock)
            if history_mode == "import":
                import_stats = ingest_history_via_import(
                    import_service,
                    item,
                    user_id=user_id,
                    clock=clock,
                    progress_callback=progress_callback,
                )
            elif history_mode == "chat":
                ingest_history_via_chat(
                    import_service,
                    item,
                    user_id=user_id,
                    background_wait=background_wait,
                )
                import_stats = _legacy_import_stats(item, mode="chat")
            else:
                ingest_history_to_timeline(import_service, item, user_id=user_id)
                import_stats = _legacy_import_stats(item, mode="timeline")

            phase_seconds["import"] = round(time.perf_counter() - import_started, 6)
            import_close_started = time.perf_counter()
            stage = "import_close"
            _report_phase(progress_callback, "wait-import")
            _close_service(import_service, timeout=background_wait)
            import_service = None
            phase_seconds["import_close"] = round(time.perf_counter() - import_close_started, 6)

            if int(import_stats.get("failed_import_count") or 0):
                return {
                    **identity,
                    "hypothesis": "",
                    "reply": "",
                    "answer_hit": False,
                    "recall_hit": False,
                    **import_stats,
                    "recall_context": "",
                    "recall_context_chars": 0,
                    "reader_input_chars": 0,
                    "api_calls": 0,
                    "response_debug": {},
                    "recalled_memories": [],
                    "recalled_timeline_chunks": [],
                    "recalled_documents": [],
                    "exception": "conversation import contained failed fragments",
                    "error": "conversation import contained failed fragments",
                    "measured_seconds": round(time.perf_counter() - started, 6),
                    "phase_seconds": phase_seconds,
                    "app_home": str(app_home),
                    "cache_hits": cache_hits,
                    "status": "error",
                    "stage": "import",
                    "passed": False,
                }
            if cache_enabled and cache_dir is not None:
                _cache_app_home(app_home, cache_dir, import_stats)

        stage = "recall"
        _report_phase(progress_callback, "recall")
        recall_started = time.perf_counter()
        clock.value = question_timestamp
        query_service = factory(clock)
        response = query_service.chat(
            item.question,
            user_id=user_id,
            session_id=f"{item.question_id}-question",
            memory_writes_allowed=False,
            # 评测提速：runner 不消费回复，跳过 reply 侧 LLM；召回口径完全不变。
            skip_reply_synthesis=True,
        )
        memory_context = build_recall_context(response)
        recall_context_chars = len(memory_context)
        phase_seconds["recall"] = round(time.perf_counter() - recall_started, 6)
        recalled_memories = list(response.get("recalled_memories") or [])
        recalled_timeline_chunks = list(response.get("recalled_timeline_chunks") or [])
        recalled_documents = list(response.get("recalled_documents") or [])
        native_api_calls = int(response.get("api_calls") or 0)
        _close_service(query_service, timeout=background_wait)
        query_service = None
        if cache_enabled and cache_dir is not None:
            _cache_recall(cache_dir, {
                "recall_context": memory_context,
                "recall_context_chars": recall_context_chars,
                "response_debug": response.get("debug", {}),
                "recalled_memories": recalled_memories,
                "recalled_timeline_chunks": recalled_timeline_chunks,
                "recalled_documents": recalled_documents,
                "api_calls": native_api_calls,
                "import_stats": import_stats,
                "phase_seconds": phase_seconds,
            })
        return {
            **identity,
            "hypothesis": "",
            "reply": "",
            "answer_hit": False,
            "recall_hit": False,
            **import_stats,
            "recalled_memory_count": len(recalled_memories),
            "recalled_timeline_count": len(recalled_timeline_chunks),
            "recalled_document_count": len(recalled_documents),
            "recall_context": memory_context,
            "recall_context_chars": recall_context_chars,
            "reader_input_chars": 0,
            "phase_seconds": phase_seconds,
            "api_calls": native_api_calls,
            "measured_seconds": round(time.perf_counter() - started, 6),
            "response_debug": response.get("debug", {}),
            "recalled_memories": recalled_memories,
            "recalled_timeline_chunks": recalled_timeline_chunks,
            "recalled_documents": recalled_documents,
            "app_home": str(app_home),
            "cache_hits": cache_hits,
            "status": "success",
            "stage": "recall",
        }
    except Exception as exc:
        return {
            **identity,
            "hypothesis": "",
            "reply": "",
            "answer_hit": False,
            "recall_hit": False,
            **import_stats,
            "recall_context": "",
            "recall_context_chars": 0,
            "reader_input_chars": 0,
            "api_calls": 0,
            "response_debug": {},
            "recalled_memories": [],
            "recalled_timeline_chunks": [],
            "recalled_documents": [],
            "exception": repr(exc),
            "error": str(exc),
            "measured_seconds": round(time.perf_counter() - started, 6),
            "phase_seconds": phase_seconds,
            "app_home": str(app_home),
            "cache_hits": cache_hits,
            "status": "error",
            "stage": stage,
            "passed": False,
        }
    finally:
        _close_service(import_service, timeout=background_wait, suppress_errors=True)
        _close_service(query_service, timeout=background_wait, suppress_errors=True)
        _restore_app_home(original_app_home)


def answer_question_from_memory_context(
    context: dict[str, Any],
    *,
    reader: Reader,
) -> dict[str, Any]:
    """Run the Reader and scoring on an already-built memory context."""
    result = dict(context)
    if result.get("status") != "success":
        result["hypothesis"] = ""
        result["reply"] = ""
        result["answer_hit"] = False
        result["recall_hit"] = False
        result["recall_context"] = ""
        result["recall_context_chars"] = 0
        result["reader_input_chars"] = 0
        result["passed"] = False
        return result
    memory_context = str(result.get("recall_context") or "")
    reader_started = time.perf_counter()
    answer_task = _extract_answer_task({"debug": result.get("response_debug") or {}})
    recalled_memories = list(result.get("recalled_memories") or [])
    recalled_timeline_chunks = list(result.get("recalled_timeline_chunks") or [])
    hypothesis = reader.answer(
        question=str(result.get("question") or ""),
        question_type=str(result.get("question_type") or ""),
        question_date=str(result.get("question_date") or ""),
        memory_context=memory_context,
        answer_task=answer_task,
        recalled_memories=recalled_memories,
        recalled_timeline_chunks=recalled_timeline_chunks,
    )
    reader_seconds = time.perf_counter() - reader_started
    phase_seconds = dict(result.get("phase_seconds") or {})
    phase_seconds["reader"] = round(reader_seconds, 6)
    recalled_documents = list(result.get("recalled_documents") or [])
    answer_hit = score_answer(
        hypothesis,
        str(result.get("answer") or ""),
        is_abstention=bool(result.get("is_abstention")),
    )
    recall_hit = score_recall_context(
        memory_context,
        _RecallScoreItem(answer=str(result.get("answer") or ""), is_abstention=bool(result.get("is_abstention"))),
    )
    native_api_calls = int(result.get("api_calls") or 0)
    reader_debug = getattr(reader, "last_debug", None)
    reader_api_calls = (
        int(reader_debug.get("api_calls") or 0)
        if isinstance(reader_debug, dict)
        else (1 if memory_context.strip() else 0)
    )
    reader_input_chars = len(truncate_text(memory_context, reader.config.max_context_chars))
    if isinstance(reader_debug, dict):
        actual_reader_input_chars = reader_debug.get("reader_input_chars")
        if isinstance(actual_reader_input_chars, int) and actual_reader_input_chars >= 0:
            reader_input_chars = actual_reader_input_chars
    result.update({
        "hypothesis": hypothesis,
        "reply": hypothesis,
        "answer_hit": answer_hit,
        "recall_hit": recall_hit,
        "recalled_memory_count": len(recalled_memories),
        "recalled_timeline_count": len(recalled_timeline_chunks),
        "recalled_document_count": len(recalled_documents),
        "recall_context": memory_context,
        "recall_context_chars": len(memory_context),
        "reader_input_chars": reader_input_chars,
        "reader_debug": reader_debug if isinstance(reader_debug, dict) else None,
        "phase_seconds": phase_seconds,
        "api_calls": native_api_calls + reader_api_calls,
        "measured_seconds": round(float(result.get("measured_seconds") or 0.0) + reader_seconds, 6),
        "status": "success",
        "stage": "complete",
        "passed": bool(answer_hit),
    })
    return result


def run_longmemeval_item(
    item: LongMemEvalItem,
    *,
    reader: Reader,
    background_wait: float,
    history_mode: str,
    keep_home: bool,
    original_app_home: str | None,
    progress_callback: ProgressCallback | None = None,
    service_factory: ServiceFactory | None = None,
) -> dict[str, Any]:
    """Serial wrapper over the two eval phases; kept for compatibility and tests."""
    context = build_question_memory_context(
        item,
        background_wait=background_wait,
        history_mode=history_mode,
        original_app_home=original_app_home,
        progress_callback=progress_callback,
        service_factory=service_factory,
    )
    result = answer_question_from_memory_context(context, reader=reader)
    if not keep_home:
        _remove_ephemeral_app_home(result, keep_homes=False)
    return result


def _load_cached_recall(cache_dir: Path) -> dict[str, Any] | None:
    recall_path = cache_dir / "recall.json"
    if not recall_path.is_file():
        return None
    try:
        payload = _load_json_object(recall_path)
    except (ValueError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload.get("recall_context"), str):
        return None
    return payload


def _materialize_cached_app_home(cache_dir: Path, app_home: Path) -> None:
    source = cache_dir / "app_home"
    if not source.is_dir():
        return
    shutil.copytree(source, app_home, dirs_exist_ok=True)


def _cache_app_home(app_home: Path, cache_dir: Path, import_stats: dict[str, Any]) -> None:
    data_source = app_home / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cache-app-home-", dir=cache_dir))
    if data_source.is_dir():
        shutil.copytree(data_source, temporary / "data")
    _atomic_write_json(temporary / "import_meta.json", import_stats)
    target = cache_dir / "app_home"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    os.replace(temporary, target)


def _load_import_meta(cache_dir: Path) -> dict[str, Any] | None:
    meta_path = cache_dir / "app_home" / "import_meta.json"
    if not meta_path.is_file():
        return None
    try:
        payload = _load_json_object(meta_path)
    except (ValueError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_recall(cache_dir: Path, payload: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(cache_dir / "recall.json", payload)


def ingest_history_via_import(
    service: Any,
    item: LongMemEvalItem,
    *,
    user_id: str,
    clock: MutableClock,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    sessions = sorted_history_sessions(item.sessions)
    total_turns = sum(len(session.turns) for session in sessions)
    imported_turn_count = 0
    saved_memory_count = 0
    rejected_memory_count = 0
    pending_confirmation_count = 0
    skipped_empty_turn_count = 0
    failed_import_count = 0
    paired_turn_count = 0
    user_only_turn_count = 0
    assistant_only_turn_count = 0
    timeline_turn_count = 0
    timeline_chunk_count = 0
    import_failures: list[dict[str, Any]] = []
    processed_turn_count = 0
    for session_index, session in enumerate(sessions):
        session_timestamp = _timestamp(session.date)
        if session_timestamp is None:
            session_timestamp = (_timestamp(item.question_date) or clock.value) + session_index
        conversation_turns: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(session.turns):
            content = str(turn.content or "").strip()
            occurred_at = session_timestamp + turn_index
            clock.value = occurred_at
            role = str(turn.role or "unknown").strip().lower() or "unknown"
            source_id = f"{item.question_id}:{session.session_id}:{turn_index}"
            conversation_turns.append({
                "role": role,
                "content": content,
                "occurred_at": occurred_at,
                "source_id": source_id,
            })
        result = service.import_conversation_events(
            user_id=user_id,
            session_id=session.session_id,
            turns=conversation_turns,
            source="longmemeval_oracle",
            context=f"question_id={item.question_id};session_id={session.session_id}",
        )
        imported_turn_count += int(result.get("imported_turn_count", len(conversation_turns)) or 0)
        saved_memory_count += int(result.get("saved_count") or 0)
        rejected_memory_count += int(result.get("rejected_count") or 0)
        pending_confirmation_count += int(result.get("pending_confirmation_count") or 0)
        skipped_empty_turn_count += int(result.get("skipped_empty_turn_count") or 0)
        failed_import_count += int(result.get("failed_count") or 0)
        session_failures = list(result.get("failures") or [])
        if session_failures:
            import_failures.append({
                "session_id": str(session.session_id),
                "failed_count": int(result.get("failed_count") or len(session_failures)),
                "failures": session_failures,
            })
        paired_turn_count += int(result.get("paired_turn_count") or 0)
        user_only_turn_count += int(result.get("user_only_turn_count") or 0)
        assistant_only_turn_count += int(result.get("assistant_only_turn_count") or 0)
        timeline_turn_count += int(result.get("timeline_turn_count") or 0)
        timeline_chunk_count += int(result.get("timeline_chunk_count") or 0)
        processed_turn_count += len(conversation_turns)
        _report_phase(progress_callback, f"import {processed_turn_count}/{total_turns}")
    return {
        "history_mode": "import",
        "session_count": len(sessions),
        "history_turn_count": total_turns,
        "imported_turn_count": imported_turn_count,
        "saved_memory_count": saved_memory_count,
        "rejected_memory_count": rejected_memory_count,
        "pending_confirmation_count": pending_confirmation_count,
        "skipped_empty_turn_count": skipped_empty_turn_count,
        "failed_import_count": failed_import_count,
        "paired_turn_count": paired_turn_count,
        "user_only_turn_count": user_only_turn_count,
        "assistant_only_turn_count": assistant_only_turn_count,
        "timeline_turn_count": timeline_turn_count,
        "timeline_chunk_count": timeline_chunk_count,
        "import_failures": import_failures,
    }


def ingest_history_to_timeline(service: Any, item: LongMemEvalItem, *, user_id: str) -> None:
    for session in sorted_history_sessions(item.sessions):
        text = "\n".join(f"{turn.role}: {turn.content}" for turn in session.turns)
        if not text.strip():
            continue
        service.timeline_store.add_turn(
            user_id,
            text,
            source="longmemeval_history",
            interaction_id=session.session_id,
            created_at=_timestamp(session.date),
        )


def ingest_history_via_chat(
    service: Any,
    item: LongMemEvalItem,
    *,
    user_id: str,
    background_wait: float,
) -> None:
    for session in sorted_history_sessions(item.sessions):
        for turn in session.turns:
            if turn.role != "user":
                continue
            response = service.chat(
                turn.content,
                user_id=user_id,
                session_id=f"{item.question_id}-{session.session_id}",
            )
            wait_for_response_jobs(service, response, user_id=user_id, timeout=background_wait)


def wait_for_response_jobs(
    service: Any,
    response: dict[str, Any],
    *,
    user_id: str,
    timeout: float,
) -> None:
    job_ids = _memory_processing_job_ids(response)
    deadline = time.perf_counter() + max(0.0, timeout)
    for job_id in job_ids:
        while time.perf_counter() < deadline:
            job = service.read_memory_job(user_id=user_id, job_id=job_id)
            if not isinstance(job, dict):
                break
            if str(job.get("status") or "") in {"saved", "rejected", "failed", "not_needed"}:
                break
            time.sleep(0.05)


def sorted_history_sessions(sessions: list[LongMemEvalSession]) -> list[LongMemEvalSession]:
    indexed = list(enumerate(sessions))
    indexed.sort(key=lambda value: (
        _timestamp(value[1].date) is None,
        _timestamp(value[1].date) or float(value[0]),
        value[0],
    ))
    return [session for _index, session in indexed]


def _extract_answer_task(response: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the generic answer contract from the production decision.

    Returns None (legacy behavior) when the response has no pre-reply decision
    or none of the contract fields are present. Never leaks benchmark labels.
    """
    return extract_answer_task_from_debug(response.get("debug"))


def build_recall_context(response: dict[str, Any]) -> str:
    sections: list[str] = []
    answer_task = _extract_answer_task(response)
    include_source_ids = bool(
        answer_task and answer_task.get("coverage_requirement") == "complete_set"
    )
    memories = _format_memory_evidence(
        response.get("recalled_memories"),
        include_source_ids=include_source_ids,
    )
    timeline_chunks = _format_timeline_evidence(
        response.get("recalled_timeline_chunks"),
        include_source_ids=include_source_ids,
    )
    documents = _unique_document_texts(response.get("recalled_documents"))
    if memories:
        sections.append("Structured memories:\n" + "\n".join(f"- {text}" for text in memories))
    if timeline_chunks:
        sections.append("Timeline evidence:\n" + "\n".join(f"- {text}" for text in timeline_chunks))
    if documents:
        sections.append("Document evidence:\n" + "\n".join(f"- {text}" for text in documents))
    return "\n\n".join(sections)


def build_reader_prompt(
    *,
    question: str,
    question_type: str,
    question_date: str,
    memory_context: str,
    answer_task: dict[str, Any] | None = None,
) -> str:
    task_section = _render_answer_task(answer_task)
    schema_extra = (
        '  "coverage": ["<obligation name the evidence supports, e.g. incremental_next_step>", "<optional>"],\n'
        if answer_task else ""
    )
    return (
        "You are answering a LongMemEval benchmark question using retrieved memory only.\n"
        "Follow these rules strictly:\n"
        "1. Use only the memory context below. Do not invent details.\n"
        "2. First identify the 1-5 most relevant memory lines or snippets.\n"
        "3. Ignore irrelevant background memories once you have found the relevant evidence.\n"
        "4. If the answer text itself appears in memory, return it instead of abstaining.\n"
        "5. For recommendation, preference, or constraint questions, synthesize only the user preference or constraint directly supported by the relevant snippets; an exact answer sentence is not required.\n"
        "6. For counting, totaling, or comparison questions, gather all relevant items before answering.\n"
        "7. For temporal or knowledge-update questions, compare the event_time or recorded_at labels before the question date.\n"
        "8. If the relevant snippets support the requested fact or preference, answer from them instead of claiming that memory is unavailable. Whenever you list any relevant_evidence, you MUST base final_answer on that evidence; only an empty relevant_evidence list justifies the unknown answer.\n"
        "9. If the memory truly lacks the answer, set final_answer to exactly: "
        f'"{UNKNOWN_ANSWER}"\n'
        "10. Keep the final answer concise and direct.\n"
        "11. When memory shows the user already owns, tried, or experimented with something (an ingredient, a device, an accessory, a habit), build the answer as an incremental next step on top of that existing thing; do not repeat suggesting what they already did.\n"
        "12. Only use information explicitly present in the memory context. Never add unmentioned brand names, specific venues, model numbers, prices, or quantities.\n"
        "13. If the question implies a comparison (A vs B) or a negative constraint (what the user would not prefer), "
        "cover that dimension explicitly in the final answer (for example state what to avoid or what the user would "
        "not prefer) instead of a one-sided recommendation.\n"
        "14. When the answer task contract lists answer_obligations, make final_answer cover each obligation that the "
        "selected evidence supports and that would change the conclusion.\n\n"
        f"{task_section}"
        f"Question type: {question_type}\n"
        f"Question date: {question_date}\n"
        f"Question: {question}\n\n"
        "Memory context:\n"
        f"{memory_context or '[empty]'}\n\n"
        "Return valid JSON only, with no markdown or extra text. Use exactly this schema:\n"
        "{\n"
        '  "relevant_evidence": ["<most relevant memory snippet>", "<optional>"],\n'
        f"{schema_extra}"
        f'  "final_answer": "<final answer or exactly {UNKNOWN_ANSWER}>"\n'
        "}\n"
    )


def _render_answer_task(answer_task: dict[str, Any] | None) -> str:
    """Render the generic answer contract section for the Reader prompt.

    Only appended when an answer_task is provided; None keeps the legacy prompt
    byte-for-byte identical. The contract is product-generic (intent/obligations),
    never LongMemEval-specific.
    """
    if not answer_task:
        return ""
    intent = str(answer_task.get("answer_intent") or "direct_answer")
    focus = str(answer_task.get("answer_focus") or "").strip()
    obligations = answer_task.get("answer_obligations") or []
    if isinstance(obligations, str):
        obligations = [part.strip() for part in obligations.split(",") if part.strip()]
    uncertainty = str(answer_task.get("uncertainty_policy") or "none")
    coverage_requirement = str(answer_task.get("coverage_requirement") or "best_evidence")
    coverage_complete = answer_task.get("coverage_complete")
    lines = ["Answer task contract:", f"- answer_intent: {intent}"]
    if focus:
        lines.append(f"- answer_focus: {focus}")
    if obligations:
        lines.append("- answer_obligations: " + ", ".join(str(item) for item in obligations))
    lines.append(f"- uncertainty_policy: {uncertainty}")
    lines.append(f"- coverage_requirement: {coverage_requirement}")
    if coverage_requirement == "complete_set":
        lines.append(f"- coverage_complete: {coverage_complete is True}")
    lines.extend([
        "",
        "Working order for final_answer:",
        "1. Select the relevant evidence from the memory context.",
        "2. For each listed answer_obligation, decide whether the selected evidence supports it.",
        "3. Answer the user's question directly.",
        "4. Cover the obligations that the selected evidence supports and that would change the conclusion.",
        "5. Do not add content not supported by the evidence.",
        "6. Only abstain when the evidence is insufficient for the question.",
        "",
    ])
    obligation_hints = {
        "entities": "Entities: name every requested object or identity supported by the selected evidence; do not invent missing entities.",
        "qualifiers": "Qualifiers: preserve requested supported models, types, categories, specifications, ratios, or other non-temporal attributes for each relevant entity.",
        "negation_constraints": "Negation constraints: explicitly state what the user would not prefer or should avoid when the evidence supports it.",
        "incremental_next_step": "Incremental next step: build the answer on top of what the user already owns, tried, prepared, or planned.",
        "comparison": "Comparison: cover both sides of the comparison explicitly.",
        "temporal_relation": "Temporal relation: explicitly state the supported date, duration, ordering, or per-entity time relation requested by the user.",
        "count_scope": "Count scope: account for every supplied source in the fixed answer_focus before stating a total; do not list entities unless entities is also required.",
    }
    for obligation in obligations:
        hint = obligation_hints.get(str(obligation).strip())
        if hint:
            lines.append(hint)
    return "\n".join(lines) + "\n"


def extract_reader_final_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    candidates = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        answer = parsed.get("final_answer", parsed.get("answer"))
        if answer is not None and str(answer).strip():
            return str(answer).strip()
    return ""


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 64)
    return text[:keep] + "\n\n[truncated for reader]\n"


def score_answer(reply: str, answer: str, *, is_abstention: bool) -> bool:
    if is_abstention:
        unknown_markers = ["不知道", "不清楚", "没有", "无法确定", "not sure", "don't know", "unknown"]
        return contains_any(reply, unknown_markers)
    terms = answer_terms(answer)
    return bool(terms and contains_any(reply, terms))


def score_recall_context(memory_context: str, item: LongMemEvalItem) -> bool:
    terms = answer_terms(item.answer)
    return bool(terms and not item.is_abstention and contains_any(memory_context, terms))


def score_recall(response: dict[str, Any], memories: list[dict[str, Any]], item: LongMemEvalItem) -> bool:
    del memories
    return score_recall_context(build_recall_context(response), item)


def summarize_longmemeval_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[str(run.get("question_type") or "unknown")].append(run)
    non_abstention_runs = [run for run in runs if not run.get("is_abstention")]
    abstention_runs = [run for run in runs if run.get("is_abstention")]
    abstention_summary = _summarize_group(abstention_runs)
    return {
        "overall": _summarize_group(runs),
        "non_abstention": _summarize_group(non_abstention_runs),
        # Abstention is an orthogonal expected-outcome overlay, not a question type.
        "abstention": abstention_summary,
        "by_question_type": {name: _summarize_group(items) for name, items in sorted(groups.items())},
        "recall_context": _summarize_recall_chars(runs),
        "phase_seconds": _summarize_phase_seconds(runs),
        "failures": [
            {
                "question_id": run.get("question_id"),
                "question_type": run.get("question_type"),
                "question": run.get("question"),
                "answer": run.get("answer"),
                "reply": run.get("reply"),
                "stage": run.get("stage", ""),
                "exception": run.get("exception", ""),
            }
            for run in runs
            if run.get("status") != "success"
        ][:50],
    }


def write_longmemeval_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": config,
        "summary": summary,
        "runs": runs,
    }
    (output_dir / "eval-latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "eval-latest.md").write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    recall_summary = summary.get("recall_context") or {}
    official = summary.get("official_judge") or {}
    lines = [
        "# LongMemEval 端到端测评报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 数据集：{payload['config'].get('dataset_path')}",
        f"- 历史导入模式：{payload['config'].get('history_mode')}",
        f"- Reader：{payload['config'].get('reader_model')}",
        "",
    ]
    if official:
        official_overall = official.get("overall") or {}
        lines.extend([
            "## 总览（官方 judge）",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| 题目数 | {official_overall.get('total', summary['overall']['total'])} |",
            f"| 官方 judge 命中率 | {_pct(official_overall.get('correct_rate', 0.0))} |",
            f"| 命中题数 | {official_overall.get('correct', 0)}/{official_overall.get('total', 0)} |",
            f"| 固定未知回答数（诊断，不参与改分） | {official_overall.get('refusals', 0)} |",
            f"| 官方 judge 判为错误 | {official_overall.get('wrong', 0)} |",
            f"| judge 解析失败（未判定） | {official_overall.get('parse_errors', 0)} |",
            f"| judge 模型 | {official.get('judge_model', '')} |",
            f"| judge 完成时间 | {official.get('judged_at', '')} |",
            f"| 判分协议 | {official.get('protocol', '')} |",
            f"| 协议来源 | {official.get('protocol_source', '')} |",
            f"| 上游参考 judge | {official.get('upstream_reference_judge', '')} |",
            f"| judge 模型替代 | {'是' if official.get('judge_model_substitution') else '否'} |",
            f"| judge 输出 token 上限 | {official.get('judge_max_tokens', '')}（上游默认 {official.get('upstream_max_tokens', '')}） |",
            "",
            "## 官方 judge 分题型结果",
            "",
            "| 题型 | 命中 | 命中率 |",
            "| --- | ---: | ---: |",
        ])
        for question_type, metrics in sorted((official.get("by_question_type") or {}).items()):
            lines.append(
                f"| {question_type} | {metrics.get('correct', 0)}/{metrics.get('total', 0)} "
                f"| {_pct(metrics.get('correct_rate', 0.0))} |"
            )
        lines.extend([
            "",
            "## 官方 judge 可回答性切片",
            "",
            "| 切片 | 命中 | 命中率 |",
            "| --- | ---: | ---: |",
        ])
        for group, metrics in sorted((official.get("by_abstention") or {}).items()):
            lines.append(
                f"| {group} | {metrics.get('correct', 0)}/{metrics.get('total', 0)} "
                f"| {_pct(metrics.get('correct_rate', 0.0))} |"
            )
        lines.extend([
            "",
            "## 官方 judge 逐题明细",
            "",
            "| question_id | score |",
            "| --- | ---: |",
        ])
        per_question = official.get("per_question") or {}
        reasons_map = official.get("reasons") or {}
        for qid, score in sorted(per_question.items()):
            reason = reasons_map.get(qid, "")
            if score:
                mark = "✅ 1"
            elif reason == "judge parse error":
                mark = "⚠️ 0（解析失败）"
            elif reason == "refusal":
                mark = "🚫 0（拒答）"
            else:
                mark = "❌ 0（判定错误）"
            lines.append(f"| {qid} | {mark} |")
        lines.append("")
        lines.extend([
            "## 阶段耗时",
            "",
        ])
    else:
        lines.extend([
            "## 总览",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| 题目数 | {summary['overall']['total']} |",
            f"| 成功数 | {summary['overall']['succeeded']} |",
            f"| 失败数 | {summary['overall']['failed']} |",
            f"| 平均耗时 | {summary['overall']['mean_seconds']}s |",
            f"| 总召回字符数 | {recall_summary.get('total_chars', 0)} |",
            f"| 平均召回字符数 | {recall_summary.get('mean_chars', 0)} |",
            "",
            "## 阶段耗时",
            "",
        ])
    phase_summary = summary.get("phase_seconds") or {}
    if phase_summary:
        lines.extend([
            "| 阶段 | 样本数 | 均值(s) | 中位数(s) | P95(s) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for phase_name in ("import", "import_close", "recall", "reader"):
            phase = phase_summary.get(phase_name) or {}
            lines.append(
                f"| {phase_name} | {phase.get('count', 0)} | "
                f"{phase.get('mean', 0)} | {phase.get('median', 0)} | "
                f"{phase.get('p95', 0)} |"
            )
        lines.append("")
    lines.extend([
        "## 分项",
        "",
        "| question_type | 题目数 | 成功数 |",
        "| --- | ---: | ---: |",
    ])
    for name, item in summary["by_question_type"].items():
        lines.append(
            f"| {name} | {item['total']} | {item['succeeded']} |"
        )
    lines.extend(["", "## 失败样本", ""])
    failures = summary.get("failures") or []
    if not failures:
        lines.append("本次没有运行失败样本。")
    else:
        for item in failures[:20]:
            lines.extend([
                f"### {item.get('question_id')} / {item.get('question_type')}",
                "",
                f"- 问题：{item.get('question')}",
                f"- 标准答案：{item.get('answer')}",
                f"- 失败阶段：{item.get('stage') or ''}",
                f"- 异常：{item.get('exception') or ''}",
                "",
            ])
    lines.extend([
        "## 说明",
        "",
        "- `*_memory.jsonl` 只包含 `question_id` 和 `hypothesis`，用于交给统一 evaluator。",
        "- `*.details.json` 保存完整召回文本和 Reader 截断前的 `recall_context_chars`。",
        "- 未运行官方 judge 时，本报告不展示任何本地命中率作为成绩；本地 `answer_hit`/`recall_hit` 只保留在 `*.details.json` 每题调试字段，用于区分「没翻到」与「翻到没答对」。正式成绩需运行官方 judge。",
        "- 若本报告包含「官方 judge」段落，其命中率为 LLM 语义等价判定结果，是唯一可信的正确率指标。",
        "- 「judge 解析失败」表示 judge 未产生有效 JSON 判定（空/截断响应重试后仍失败），按 0 分计入但**不代表答案一定错误**，需人工复核。",
        "- `by_question_type` 是互斥分组；abstention 是单独的 expected-outcome overlay，不重复计入题数。",
        "- `history-mode=import` 逐 turn 走本系统导入门控；timeline/chat 仅保留为兼容诊断模式。",
    ])
    return "\n".join(lines) + "\n"


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _summarize_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    succeeded = sum(1 for run in runs if run.get("status") == "success")
    if total == 0:
        return {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "mean_seconds": 0.0,
        }
    return {
        "total": total,
        "succeeded": succeeded,
        "failed": total - succeeded,
        "mean_seconds": round(sum(float(run.get("measured_seconds") or 0.0) for run in runs) / total, 4),
    }


def _summarize_recall_chars(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(run.get("recall_context_chars") or 0) for run in runs if run.get("status") == "success"]
    if not values:
        return {"total_chars": 0, "mean_chars": 0.0, "min_chars": 0, "max_chars": 0, "empty_count": 0}
    return {
        "total_chars": sum(values),
        "mean_chars": round(sum(values) / len(values), 2),
        "min_chars": min(values),
        "max_chars": max(values),
        "empty_count": sum(1 for value in values if value == 0),
    }


def _summarize_phase_seconds(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-phase timings: count, mean, median, p95."""
    phases = ["import", "import_close", "recall", "reader"]
    result: dict[str, Any] = {}
    for phase in phases:
        values = [
            float(run.get("phase_seconds", {}).get(phase, 0.0) or 0.0)
            for run in runs
            if run.get("status") == "success" and isinstance(run.get("phase_seconds"), dict)
        ]
        if not values:
            result[phase] = {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
            continue
        sorted_values = sorted(values)
        n = len(sorted_values)
        median = sorted_values[n // 2] if n % 2 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        result[phase] = {
            "count": n,
            "mean": round(sum(values) / n, 4),
            "median": round(median, 4),
            "p95": round(sorted_values[int(n * 0.95)] if int(n * 0.95) < n else sorted_values[-1], 4),
        }
    return result


def _memory_processing_job_ids(response: dict[str, Any]) -> list[str]:
    debug = response.get("debug") if isinstance(response.get("debug"), dict) else {}
    memory_processing = debug.get("memory_processing") if isinstance(debug.get("memory_processing"), dict) else {}
    ids = []
    for key in ("job_id", "memory_job_id"):
        value = memory_processing.get(key)
        if value:
            ids.append(str(value))
    return ids


def _timestamp(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    normalized = _LONGMEMEVAL_WEEKDAY_RE.sub(" ", text)
    try:
        return datetime.strptime(normalized, "%Y/%m/%d %H:%M").timestamp()
    except ValueError:
        return None


def _pct(value: float) -> str:
    return f"{round(float(value) * 100, 2)}%"


def _safe_output_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(value or "").strip()
    ).strip("._")
    return cleaned or "longmemeval"


def _chat_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, dict):
                    text = text.get("value")
                if text is not None:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(value).strip()


def _unique_texts(value: Any, *, key: str) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get(key) or "").strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def _format_memory_evidence(value: Any, *, include_source_ids: bool = False) -> list[str]:
    formatted: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        event_time = _format_context_timestamp(item.get("start_at") or item.get("occurred_at"))
        granularity = str(item.get("time_granularity") or "").strip()
        memory_id = str(item.get("id") or "").strip()
        key = (content, event_time, granularity, memory_id if include_source_ids else "")
        if key in seen:
            continue
        seen.add(key)
        labels = ["source=structured_memory"]
        if include_source_ids and memory_id:
            labels.append(f"source_id=memory:{memory_id}")
            evidence_ids = [
                str(evidence_id).strip().removeprefix("timeline:")
                for evidence_id in (item.get("evidence_ids") or [])
                if str(evidence_id).strip()
            ]
            if evidence_ids:
                labels.append("evidence_ids=" + ",".join(dict.fromkeys(evidence_ids)))
        if event_time:
            labels.append(f"event_time={event_time}")
        if granularity:
            labels.append(f"granularity={granularity}")
        if include_source_ids:
            labels.append(f"status={str(item.get('status') or 'active')}")
            superseded_by = str(item.get("superseded_by") or "").strip()
            if superseded_by:
                labels.append(f"superseded_by={superseded_by}")
        formatted.append(f"[{'; '.join(labels)}] {content}")
    return formatted


def _format_timeline_evidence(value: Any, *, include_source_ids: bool = False) -> list[str]:
    formatted: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        recorded_at = _format_context_timestamp(item.get("timestamp"))
        chunk_id = str(item.get("id") or "").strip()
        key = (text, recorded_at, chunk_id if include_source_ids else "")
        if key in seen:
            continue
        seen.add(key)
        labels = ["source=timeline"]
        if include_source_ids and chunk_id:
            labels.append(f"source_id=timeline:{chunk_id}")
        if recorded_at:
            labels.append(f"recorded_at={recorded_at}")
        if include_source_ids:
            labels.append(f"status={str(item.get('status') or 'active')}")
        formatted.append(f"[{'; '.join(labels)}] {text}")
    return formatted


def _format_context_timestamp(value: Any) -> str:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return ""
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="minutes")


def _unique_document_texts(value: Any) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("filename") or "").strip()
        body = str(item.get("summary") or item.get("content") or "").strip()
        text = ": ".join(part for part in (title, body) if part)
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def _empty_import_stats(item: LongMemEvalItem) -> dict[str, Any]:
    return {
        "history_mode": "import",
        "session_count": len(item.sessions),
        "history_turn_count": sum(len(session.turns) for session in item.sessions),
        "imported_turn_count": 0,
        "saved_memory_count": 0,
        "rejected_memory_count": 0,
        "pending_confirmation_count": 0,
        "skipped_empty_turn_count": 0,
        "failed_import_count": 0,
        "paired_turn_count": 0,
        "user_only_turn_count": 0,
        "assistant_only_turn_count": 0,
        "timeline_turn_count": 0,
        "timeline_chunk_count": 0,
        "import_failures": [],
    }


def _legacy_import_stats(item: LongMemEvalItem, *, mode: str) -> dict[str, Any]:
    stats = _empty_import_stats(item)
    stats["history_mode"] = mode
    stats["imported_turn_count"] = stats["history_turn_count"]
    return stats


def _default_service_factory(clock: MutableClock) -> GlassesChatService:
    return GlassesChatService(clock=clock)


def _close_service(
    service: Any | None,
    *,
    timeout: float,
    suppress_errors: bool = False,
) -> None:
    if service is None:
        return
    close = getattr(service, "close", None)
    if not callable(close):
        return
    try:
        close(timeout=max(0.0, timeout))
    except TypeError:
        close()
    except Exception:
        if not suppress_errors:
            raise


def _restore_app_home(original_app_home: str | None) -> None:
    if original_app_home is None:
        os.environ.pop(APP_HOME_ENV, None)
    else:
        os.environ[APP_HOME_ENV] = original_app_home


def _report_phase(callback: ProgressCallback | None, phase: str) -> None:
    if callback is not None:
        callback(phase)


if __name__ == "__main__":
    raise SystemExit(main())
