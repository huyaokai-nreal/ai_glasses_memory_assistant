from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any

from .evidence_set import (
    EvidenceCandidate,
    batch_evidence_candidates,
    expand_consolidated_ledger,
    validate_aggregation_ledger,
)

from .turn_semantic_classifier import (
    VALID_ANSWER_INTENTS,
    VALID_ANSWER_OBLIGATIONS,
    VALID_UNCERTAINTY_POLICIES,
)


@dataclass(frozen=True)
class AnswerDirective:
    answer_intent: str = "direct_answer"
    answer_focus: str = ""
    answer_obligations: list[str] = field(default_factory=list)
    organization: str = "direct"
    evidence_policy: str = "use_available_context"
    filtering_rules: list[str] = field(default_factory=list)
    uncertainty_policy: str = "state_limits_when_context_is_sparse"
    coverage_requirement: str = "best_evidence"
    coverage_complete: bool | None = None
    style: str = "short_direct"
    confidence: float | None = None
    reason: str = ""
    backend: str = "none"
    raw: str = ""
    error: str = ""

    def debug_payload(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "answer_intent": self.answer_intent,
            "answer_focus": self.answer_focus,
            "answer_obligations": list(self.answer_obligations),
            "organization": self.organization,
            "evidence_policy": self.evidence_policy,
            "filtering_rules": self.filtering_rules,
            "uncertainty_policy": self.uncertainty_policy,
            "coverage_requirement": self.coverage_requirement,
            "coverage_complete": self.coverage_complete,
            "style": self.style,
            "confidence": self.confidence,
            "reason": self.reason,
            **({"raw": self.raw} if self.raw else {}),
            **({"error": self.error} if self.error else {}),
        }

    def instruction_text(self) -> str:
        lines = [
            f"answer_intent: {self.answer_intent}",
            f"answer_focus: {self.answer_focus or '(none specified)'}",
            f"answer_obligations: {', '.join(self.answer_obligations) if self.answer_obligations else 'none'}",
            f"organization: {self.organization}",
            f"evidence_policy: {self.evidence_policy}",
            f"uncertainty_policy: {self.uncertainty_policy}",
            f"coverage_requirement: {self.coverage_requirement}",
            f"coverage_complete: {self.coverage_complete if self.coverage_complete is not None else 'unknown'}",
            f"style: {self.style}",
        ]
        if "negation_constraints" in self.answer_obligations:
            lines.append(
                "Cover the negation_constraints obligation: explicitly state what the user "
                "would not prefer or should avoid, where the evidence supports it."
            )
        if "incremental_next_step" in self.answer_obligations:
            lines.append(
                "Cover the incremental_next_step obligation: build the answer as the next step "
                "on top of what the user already owns, tried, prepared, or planned."
            )
        if "comparison" in self.answer_obligations:
            lines.append(
                "Cover the comparison obligation: address both sides of the comparison explicitly."
            )
        if "entities" in self.answer_obligations:
            lines.append(
                "Cover the entities obligation: name each requested object or identity that the "
                "selected evidence supports; do not invent missing entities."
            )
        if "qualifiers" in self.answer_obligations:
            lines.append(
                "Cover the qualifiers obligation: preserve the requested supported model, type, "
                "category, specification, ratio, or other non-temporal attribute for each relevant entity."
            )
        if "temporal_relation" in self.answer_obligations:
            lines.append(
                "Cover the temporal_relation obligation: explicitly state the supported date, duration, "
                "ordering, or per-entity time relation requested by the user."
            )
        if "count_scope" in self.answer_obligations:
            lines.append(
                "Cover the count_scope obligation: state the verified total for the fixed scope after "
                "accounting for the supplied evidence. Do not list entities unless entities is also an obligation."
            )
        if self.coverage_requirement == "complete_set":
            lines.append(
                "Use only the fixed answer_focus to account for every supplied source before aggregating; "
                "do not change the recall scope or reinterpret the answer intent."
            )
            if self.coverage_complete is not True:
                lines.append(
                    "The evidence coverage is incomplete. Do not state an absolute count or total; "
                    "say that the evidence check could not be completed."
                )
        if self.filtering_rules:
            lines.append("filtering_rules:")
            lines.extend(f"- {rule}" for rule in self.filtering_rules)
        if self.reason:
            lines.append(f"reason: {self.reason}")
        lines.extend([
            "Use these directives to compose the user-facing answer.",
            "Separate direct evidence from background context when they answer different parts of the question.",
            "If directly relevant recalled memories conflict, state the conflict clearly and do not silently choose one.",
            "Do not turn stable identity/profile facts into recent activities unless the user explicitly asks for identity or background.",
            "When evidence is sparse, say that the summary is based only on currently recalled memory.",
            "When directly relevant structured memory evidence is present, use it to answer instead of claiming that no memory is available; abstain only when the recalled evidence does not support the requested fact or recommendation.",
        ])
        return "\n".join(lines)


INCOMPLETE_COMPLETE_SET_REPLY = "证据处理未完成，我现在不能可靠地给出完整数量或总数。"
COMPLETE_SET_EXECUTION_FAILED_REPLY = "已定位到相关证据，但完整整理未能通过验证，因此我不能可靠地给出完整结论。"


class _ReaderJSONDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class CompleteSetAnswer:
    final_answer: str
    valid: bool
    coverage_complete: bool
    batch_count: int = 0
    api_calls: int = 0
    operation: str = ""
    value: str = ""
    unit: str = ""
    error: str = ""
    validation_errors: tuple[str, ...] = ()
    reader_status: str = ""
    failure_stage: str = ""
    selected_source_ids: tuple[str, ...] = ()
    execution_attempts: tuple[dict[str, Any], ...] = ()

    def debug_payload(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "coverage_complete": self.coverage_complete,
            "batch_count": self.batch_count,
            "api_calls": self.api_calls,
            "operation": self.operation,
            "value": self.value,
            "unit": self.unit,
            "error": self.error,
            "validation_errors": list(self.validation_errors),
            "reader_status": self.reader_status,
            "failure_stage": self.failure_stage,
            "selected_source_ids": list(self.selected_source_ids),
            "execution_attempts": list(self.execution_attempts),
        }


def synthesize_complete_set_answer(
    agent: Any,
    *,
    message: str,
    answer_contract: dict[str, Any],
    candidates: list[EvidenceCandidate],
    coverage_complete: bool,
    max_context_chars: int = 16_000,
    max_batches: int = 10,
    ledger_max_tokens: int = 4096,
) -> CompleteSetAnswer:
    """Account fixed-scope evidence and validate aggregation without re-planning recall."""

    if not coverage_complete:
        return CompleteSetAnswer(
            final_answer=INCOMPLETE_COMPLETE_SET_REPLY,
            valid=False,
            coverage_complete=False,
            error="coverage_incomplete",
            reader_status="insufficient_evidence",
            failure_stage="coverage",
        )
    if agent is None:
        return CompleteSetAnswer(
            final_answer=_complete_set_execution_failure_reply(answer_contract, candidates),
            valid=False,
            coverage_complete=coverage_complete,
            error="reader_unavailable",
            reader_status="execution_failed",
            failure_stage="provider",
        )
    if not candidates:
        return CompleteSetAnswer(
            final_answer=INCOMPLETE_COMPLETE_SET_REPLY,
            valid=False,
            coverage_complete=coverage_complete,
            error="evidence_unavailable",
            reader_status="insufficient_evidence",
            failure_stage="evidence",
        )
    batches, truncated = batch_evidence_candidates(
        candidates,
        max_context_chars=max_context_chars,
        # Reserve one call for cross-batch consolidation and one for final wording.
        max_batches=max(1, max_batches - 2),
        max_candidates_per_batch=max(1, ledger_max_tokens // 256),
    )
    if truncated:
        return CompleteSetAnswer(
            final_answer=_complete_set_execution_failure_reply(answer_contract, candidates),
            valid=False,
            coverage_complete=True,
            batch_count=len(batches),
            error="reader_batch_budget",
            reader_status="execution_failed",
            failure_stage="batch_budget",
        )

    api_calls = 0
    execution_attempts: list[dict[str, Any]] = []
    validations = []
    for batch_index, batch in enumerate(batches):
        prompt = _complete_set_ledger_prompt(
            message=message,
            answer_contract=answer_contract,
            candidates=batch,
            batch_index=batch_index,
            batch_count=len(batches),
        )
        validation = None
        provider_failures = 0
        json_failures = 0
        for attempt in range(2):
            suffix = "" if not attempt else (
                "\nThe previous ledger failed deterministic validation. Correct it: classify every source_id "
                "exactly once in source_decisions; put only supported facts in items; a source_id may appear "
                "in several items only when that one source explicitly states several distinct facts."
            )
            api_calls += 1
            provider_error = False
            json_error = False
            attempt_prompt = prompt + suffix
            attempt_started = time.perf_counter()
            try:
                payload = _run_internal_json(agent, attempt_prompt, max_tokens=ledger_max_tokens)
            except _ReaderJSONDecodeError:
                json_failures += 1
                json_error = True
                payload = {}
            except Exception:
                provider_failures += 1
                provider_error = True
                payload = {}
            validation = validate_aggregation_ledger(
                payload,
                batch,
                require_final_answer=False,
                allow_duplicate_source_ids=len(batches) > 1,
                validate_claimed_value=False,
            )
            execution_attempts.append({
                "stage": "batch_ledger",
                "attempt": attempt + 1,
                "outcome": (
                    "provider_error" if provider_error else
                    "invalid_json" if json_error else
                    "valid" if validation.valid else "invalid"
                ),
                "input_chars": len(attempt_prompt),
                "duration_seconds": round(time.perf_counter() - attempt_started, 6),
                "validation_errors": list(validation.errors),
            })
            if validation.valid:
                break
        if validation is None or not validation.valid:
            provider_failed = provider_failures == 2
            json_failed = json_failures == 2
            output_failed = provider_failures + json_failures > 0
            return CompleteSetAnswer(
                final_answer=_complete_set_execution_failure_reply(answer_contract, candidates),
                valid=False,
                coverage_complete=True,
                batch_count=len(batches),
                api_calls=api_calls,
                error=(
                    "provider_output_failure" if provider_failed else
                    "invalid_json_output" if json_failed else
                    "reader_output_failure" if output_failed else
                    "batch_validation_failed"
                ),
                validation_errors=tuple(validation.errors if validation else ()),
                reader_status="execution_failed",
                failure_stage=(
                    "provider" if provider_failed else
                    "json" if json_failed else
                    "output" if output_failed else
                    "batch_ledger"
                ),
                execution_attempts=tuple(execution_attempts),
            )
        validations.append(validation)

    ledger_validation = validations[0]
    if len(validations) > 1:
        consolidation_prompt = _complete_set_consolidation_prompt(
            message=message,
            answer_contract=answer_contract,
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
            } for validation in validations],
        )
        final_validation = None
        provider_failures = 0
        json_failures = 0
        for attempt in range(2):
            suffix = "" if not attempt else (
                "\nThe previous consolidated ledger failed deterministic validation. Return the full "
                "ledger again, classify every source exactly once in source_decisions, retain every "
                "supported fact item, and exclude facts outside answer_focus."
            )
            api_calls += 1
            provider_error = False
            json_error = False
            attempt_prompt = consolidation_prompt + suffix
            attempt_started = time.perf_counter()
            try:
                payload = _run_internal_json(
                    agent,
                    attempt_prompt,
                    max_tokens=ledger_max_tokens,
                )
            except _ReaderJSONDecodeError:
                json_failures += 1
                json_error = True
                payload = {}
            except Exception:
                provider_failures += 1
                provider_error = True
                payload = {}
            final_validation = validate_aggregation_ledger(
                expand_consolidated_ledger(payload),
                candidates,
                require_final_answer=False,
                validate_claimed_value=False,
            )
            execution_attempts.append({
                "stage": "consolidation",
                "attempt": attempt + 1,
                "outcome": (
                    "provider_error" if provider_error else
                    "invalid_json" if json_error else
                    "valid" if final_validation.valid else "invalid"
                ),
                "input_chars": len(attempt_prompt),
                "duration_seconds": round(time.perf_counter() - attempt_started, 6),
                "validation_errors": list(final_validation.errors),
            })
            if final_validation.valid:
                break
        if final_validation is None or not final_validation.valid:
            provider_failed = provider_failures == 2
            json_failed = json_failures == 2
            output_failed = provider_failures + json_failures > 0
            return CompleteSetAnswer(
                final_answer=_complete_set_execution_failure_reply(answer_contract, candidates),
                valid=False,
                coverage_complete=True,
                batch_count=len(batches),
                api_calls=api_calls,
                error=(
                    "provider_output_failure" if provider_failed else
                    "invalid_json_output" if json_failed else
                    "reader_output_failure" if output_failed else
                    "consolidated_ledger_validation_failed"
                ),
                validation_errors=tuple(final_validation.errors if final_validation else ()),
                reader_status="execution_failed",
                failure_stage=(
                    "provider" if provider_failed else
                    "json" if json_failed else
                    "output" if output_failed else
                    "consolidation"
                ),
                execution_attempts=tuple(execution_attempts),
            )
        ledger_validation = final_validation

    operation = ledger_validation.operation
    value = ledger_validation.value
    unit = ledger_validation.unit or "item"
    if value is None:
        return CompleteSetAnswer(
            final_answer=_complete_set_execution_failure_reply(answer_contract, candidates),
            valid=False,
            coverage_complete=True,
            batch_count=len(batches),
            api_calls=api_calls,
            error="combined_ledger_validation_failed",
            validation_errors=tuple(ledger_validation.errors),
            reader_status="execution_failed",
            failure_stage="aggregation",
            execution_attempts=tuple(execution_attempts),
        )
    ledger = {
        "items": [
            {**item, "quantity": str(item.get("quantity", "1"))}
            for item in ledger_validation.items
        ],
        "aggregation": {
            "operation": operation,
            "value": str(value),
            "unit": unit,
        },
    }
    if ledger_validation.source_decisions:
        ledger["source_decisions"] = ledger_validation.source_decisions
    final_validation = None
    final_prompt = _complete_set_final_prompt(
        message=message,
        answer_contract=answer_contract,
        ledger=ledger,
    )
    provider_failures = 0
    json_failures = 0
    for attempt in range(2):
        suffix = "" if not attempt else (
            "\nThe previous final answer omitted the verified value. Return it again with the exact value."
        )
        api_calls += 1
        provider_error = False
        json_error = False
        attempt_prompt = final_prompt + suffix
        attempt_started = time.perf_counter()
        try:
            payload = _run_internal_json(agent, attempt_prompt, max_tokens=ledger_max_tokens)
        except _ReaderJSONDecodeError:
            json_failures += 1
            json_error = True
            payload = {}
        except Exception:
            provider_failures += 1
            provider_error = True
            payload = {}
        final_validation = validate_aggregation_ledger(
            {**ledger, "final_answer": str(payload.get("final_answer") or "")},
            candidates,
        )
        execution_attempts.append({
            "stage": "final_answer",
            "attempt": attempt + 1,
            "outcome": (
                "provider_error" if provider_error else
                "invalid_json" if json_error else
                "valid" if final_validation.valid else "invalid"
            ),
            "input_chars": len(attempt_prompt),
            "duration_seconds": round(time.perf_counter() - attempt_started, 6),
            "validation_errors": list(final_validation.errors),
        })
        if final_validation.valid:
            break
    if final_validation is None or not final_validation.valid:
        provider_failed = provider_failures == 2
        json_failed = json_failures == 2
        output_failed = provider_failures + json_failures > 0
        return CompleteSetAnswer(
            final_answer=_complete_set_execution_failure_reply(answer_contract, candidates),
            valid=False,
            coverage_complete=True,
            batch_count=len(batches),
            api_calls=api_calls,
            operation=operation,
            value=str(value),
            unit=unit,
            error=(
                "provider_output_failure" if provider_failed else
                "invalid_json_output" if json_failed else
                "reader_output_failure" if output_failed else
                "final_answer_validation_failed"
            ),
            validation_errors=tuple(final_validation.errors if final_validation else ()),
            reader_status="execution_failed",
            failure_stage=(
                "provider" if provider_failed else
                "json" if json_failed else
                "output" if output_failed else
                "final_answer"
            ),
            execution_attempts=tuple(execution_attempts),
        )
    return CompleteSetAnswer(
        final_answer=final_validation.final_answer,
        valid=True,
        coverage_complete=True,
        batch_count=len(batches),
        api_calls=api_calls,
        operation=final_validation.operation,
        value=str(final_validation.value),
        unit=final_validation.unit,
        reader_status="answered",
        selected_source_ids=final_validation.selected_source_ids,
        execution_attempts=tuple(execution_attempts),
    )


def _complete_set_execution_failure_reply(
    answer_contract: dict[str, Any],
    candidates: list[EvidenceCandidate],
) -> str:
    """Keep the authorized evidence boundary visible when Reader execution fails."""

    focus = str(answer_contract.get("answer_focus") or "").strip()
    scope = f"“{focus}”范围内" if focus else "当前授权范围内"
    return (
        f"已定位到{scope}{len(candidates)}条候选证据，但完整整理未能通过验证；"
        "因此我不能可靠地给出完整结论。"
    )


def _complete_set_ledger_prompt(
    *,
    message: str,
    answer_contract: dict[str, Any],
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
    return (
        "Account evidence for the fixed answer contract below. Do not change answer_focus, intent, "
        "recall scope, or obligations. For every source_id choose included, excluded, or uncertain. "
        "Record that classification in source_decisions, where each source_id must appear exactly once. "
        "Put only included facts in items. Each readable item label must preserve any source-supported entity, "
        "qualifier, or temporal detail needed by the fixed obligations, without inventing details or changing schema. "
        "Item source_ids are provenance: the same source_id may appear in "
        "several items only when that one source explicitly states several distinct in-scope facts. "
        "A source is included when any part of it supplies an in-scope fact; ignore its extra background. "
        "Repeated support for one fact uses the same canonical_key; excluded is never a deduplication marker. "
        "Use one stable canonical_key for repeated mentions of the same item and use times/status to "
        "apply later corrections. Choose count or sum and a unit, but do not calculate a value or write the "
        "final answer; deterministic code does both later. Do not invent facts.\n"
        f"Batch: {batch_index + 1}/{batch_count}\nQuestion: {message}\n"
        f"Fixed answer contract: {json.dumps(answer_contract, ensure_ascii=False, sort_keys=True)}\n"
        f"Sources: {json.dumps(sources, ensure_ascii=False, sort_keys=True)}\n"
        "Return JSON only: "
        '{"source_decisions":[{"source_id":"source-id","status":"included|excluded|uncertain"}],'
        '"items":[{"canonical_key":"stable key","label":"readable label","quantity":"1",'
        '"unit":"item","status":"included|excluded|uncertain","source_ids":["source-id"]}],'
        '"aggregation":{"operation":"count|sum","unit":"item"}}'
    )


def _complete_set_final_prompt(
    *,
    message: str,
    answer_contract: dict[str, Any],
    ledger: dict[str, Any],
) -> str:
    fixed_contract = {
        key: answer_contract.get(key)
        for key in (
            "answer_intent",
            "answer_focus",
            "answer_obligations",
            "uncertainty_policy",
            "coverage_requirement",
            "coverage_complete",
            "evidence_scope",
            "discussion_relation_scope",
            "speaker_counts",
            "anonymous_track_count",
            "source_attribution_rules",
        )
        if key in answer_contract
    }
    return (
        "Write the final answer from this validated ledger. Do not add, remove, merge, or reinterpret "
        "ledger items. Include the exact aggregation.value and use the user's language. Cover every fixed "
        "answer obligation that the validated ledger supports. When entities is required, name the readable "
        "ledger item labels; preserve supported qualifiers or temporal relations when those obligations are "
        "required. Count scope alone does not require an entity list. If the fixed contract includes "
        "source_attribution_rules, treat them as hard evidence limits: never infer a person, participant "
        "count, or user action from names, teams, or source counts. If the ledger does not support an "
        "obligation, follow the fixed uncertainty policy instead of inventing content.\n"
        f"Question: {message}\n"
        f"Fixed answer contract: {json.dumps(fixed_contract, ensure_ascii=False, sort_keys=True)}\n"
        f"Validated ledger: {json.dumps(ledger, ensure_ascii=False, sort_keys=True)}\n"
        'Return JSON only: {"final_answer":"..."}'
    )


def _complete_set_consolidation_prompt(
    *,
    message: str,
    answer_contract: dict[str, Any],
    batch_ledgers: list[dict[str, object]],
) -> str:
    fixed_contract = {
        key: answer_contract.get(key)
        for key in (
            "answer_intent",
            "answer_focus",
            "answer_obligations",
            "uncertainty_policy",
            "evidence_scope",
            "discussion_relation_scope",
            "speaker_counts",
            "anonymous_track_count",
            "source_attribution_rules",
        )
        if key in answer_contract
    }
    return (
        "Consolidate already reviewed batch ledgers under the fixed answer contract. Do not change scope, "
        "recall memory, or act as a planner. Re-evaluate batch-local included items against answer_focus, "
        "exclude items outside the full focus, and merge repeated facts across batches under one stable "
        "canonical_key. Classify every input source exactly once in source_decisions. Put only supported "
        "facts in items; item source_ids are provenance and may repeat only when one source explicitly "
        "supports several distinct facts. Choose count or sum and a unit, but do not calculate the value or "
        "write the final answer; deterministic code does both later.\n"
        f"Question: {message}\n"
        f"Fixed answer contract: {json.dumps(fixed_contract, ensure_ascii=False, sort_keys=True)}\n"
        f"Batch ledgers: {json.dumps(batch_ledgers, ensure_ascii=False, sort_keys=True)}\n"
        "Return JSON only: "
        '{"source_decisions":[{"source_id":"source-id","status":"included|excluded|uncertain"}],'
        '"items":[{"canonical_key":"stable key","label":"readable label","quantity":"1",'
        '"unit":"item","status":"included","source_ids":["source-id"]}],'
        '"aggregation":{"operation":"count|sum","unit":"item"}}'
    )


def _run_internal_json(agent: Any, prompt: str, *, max_tokens: int) -> dict[str, Any]:
    result = agent.run_conversation(
        prompt,
        system_message=(
            "You are an internal evidence-accounting Reader. Output strict JSON only. "
            "The answer contract is fixed; you are not a planner and must not call tools."
        ),
        conversation_history=[],
        persist_user_message=None,
        response_format={"type": "json_object"},
        disable_thinking=True,
        max_tokens=max(1, int(max_tokens)),
    )
    payload, parsed = _parse_json_object_with_status(str(result.get("final_response") or "").strip())
    if not parsed:
        raise _ReaderJSONDecodeError("reader_invalid_json")
    return payload


@dataclass(frozen=True)
class TextEmotionDirective:
    label: str = "unknown"
    confidence: str = "low"
    should_affect_reply: bool = False
    reason: str = ""
    backend: str = "none"
    raw: str = ""
    error: str = ""

    def debug_payload(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "label": self.label,
            "confidence": self.confidence,
            "should_affect_reply": self.should_affect_reply,
            "reason": self.reason,
            **({"raw": self.raw} if self.raw else {}),
            **({"error": self.error} if self.error else {}),
        }


def synthesize_answer_directive(
    agent: Any,
    *,
    message: str,
    route_debug: dict[str, Any],
    intent_debug: dict[str, Any],
    temporal_debug: dict[str, Any],
    evidence_summary: dict[str, Any],
    answer_contract: dict[str, Any] | None = None,
) -> AnswerDirective:
    normalized_contract = _normalize_answer_contract(answer_contract)
    prompt = (
        "Plan how the final user-facing answer should be organized for an AI glasses memory assistant.\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "answer_intent": "direct_answer|summary|specific_fact|raw_evidence|clarification",\n'
        '  "answer_focus": "...",\n'
        '  "answer_obligations": ["entities|qualifiers|negation_constraints|comparison|temporal_relation|count_scope|incremental_next_step"],\n'
        '  "organization": "direct|thematic|chronological|evidence_first|mixed",\n'
        '  "evidence_policy": "use_available_context|separate_background|direct_evidence_only|quote_raw_evidence",\n'
        '  "filtering_rules": ["..."],\n'
        '  "uncertainty_policy": "none|state_limits_when_context_is_sparse|abstain_if_insufficient|ask_clarifying_if_no_evidence",\n'
        '  "style": "short_direct|concise_structured|natural_brief",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "..."\n'
        "}\n\n"
        "Decision rules:\n"
        "- Treat the structured route as authoritative for recall, web, location, and reply mode.\n"
        "- Use temporal/evidence signals to organize wording, not to reopen routing decisions.\n"
        "- Treat intent fields as memory-extraction metadata only.\n"
        "- For summary or review questions, prefer thematic or chronological organization and require evidence boundaries.\n"
        "- Treat reflected observations as useful high-level hints, not the only source of truth.\n"
        "- Treat stable profile facts as background unless they directly answer the user's question.\n"
        "- If directly relevant recalled memories conflict, require the final answer to state the conflict instead of guessing.\n"
        "- If directly relevant structured memory evidence is present, use it to answer instead of claiming that no memory is available; abstain only when the recalled evidence does not support the requested fact or recommendation.\n"
        "- If raw timeline evidence is requested, preserve quote/evidence style.\n"
        "- If this is ordinary chat with no recalled evidence, keep a direct natural answer.\n\n"
        "User message:\n"
        f"{message}\n\n"
        "Structured route:\n"
        f"{json.dumps(route_debug, ensure_ascii=False, sort_keys=True)}\n\n"
        "Intent decision:\n"
        f"{json.dumps(intent_debug, ensure_ascii=False, sort_keys=True)}\n\n"
        "Authoritative answer contract (must not be changed):\n"
        f"{json.dumps(normalized_contract, ensure_ascii=False, sort_keys=True)}\n\n"
        "Temporal resolution:\n"
        f"{json.dumps(temporal_debug, ensure_ascii=False, sort_keys=True)}\n\n"
        "Evidence summary:\n"
        f"{json.dumps(evidence_summary, ensure_ascii=False, sort_keys=True)}"
    )
    try:
        result = agent.run_conversation(
            prompt,
            system_message=(
                "You are an internal answer synthesis planner. "
                "Output strict JSON only. Do not call tools."
            ),
            conversation_history=[],
            persist_user_message=None,
        )
        raw = str(result.get("final_response") or "").strip()
        payload = _parse_json_object(raw)
        return apply_answer_contract(
            _directive_from_payload(payload, raw=raw, backend="llm"),
            normalized_contract,
        )
    except Exception as exc:
        return apply_answer_contract(
            _fallback_directive(route_debug, evidence_summary, error=str(exc)),
            normalized_contract,
        )


def classify_text_emotion(
    agent: Any,
    *,
    text: str,
    context: dict[str, Any] | None = None,
) -> TextEmotionDirective:
    prompt = (
        "Classify the user's text emotion for an AI glasses assistant.\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "label": "unknown|calm|happy|annoyed|sad|anxious",\n'
        '  "confidence": "low|medium|high",\n'
        '  "should_affect_reply": false,\n'
        '  "reason": "..."\n'
        "}\n\n"
        "Decision rules:\n"
        "- Judge only from explicit text semantics, not from imagined tone.\n"
        "- Be conservative. If the text is ambiguous or too short, use unknown.\n"
        "- Do not use keyword spotting as the decision rule; consider the whole utterance.\n"
        "- should_affect_reply can be true only when confidence is high and the emotion is explicit enough to softly influence the response.\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Optional context:\n"
        f"{json.dumps(context or {}, ensure_ascii=False, sort_keys=True)}"
    )
    try:
        result = agent.run_conversation(
            prompt,
            system_message=(
                "You are an internal text emotion classifier. "
                "Output strict JSON only. Do not call tools."
            ),
            conversation_history=[],
            persist_user_message=None,
        )
        raw = str(result.get("final_response") or "").strip()
        payload = _parse_json_object(raw)
        return _text_emotion_from_payload(payload, raw=raw, backend="llm")
    except Exception as exc:
        return TextEmotionDirective(
            backend="fallback",
            error=str(exc),
            reason="text_emotion_classifier_failed",
        )


def _directive_from_payload(payload: dict[str, Any], *, raw: str, backend: str) -> AnswerDirective:
    valid_intents = {"direct_answer", "summary", "specific_fact", "raw_evidence", "clarification"}
    valid_organizations = {"direct", "thematic", "chronological", "evidence_first", "mixed"}
    valid_evidence_policies = {
        "use_available_context",
        "separate_background",
        "direct_evidence_only",
        "quote_raw_evidence",
    }
    valid_uncertainty = {
        "none",
        "state_limits_when_context_is_sparse",
        "abstain_if_insufficient",
        "ask_clarifying_if_no_evidence",
    }
    valid_styles = {"short_direct", "concise_structured", "natural_brief"}

    answer_intent = _choice(payload.get("answer_intent"), valid_intents, "direct_answer")
    organization = _choice(payload.get("organization"), valid_organizations, "direct")
    evidence_policy = _choice(payload.get("evidence_policy"), valid_evidence_policies, "use_available_context")
    uncertainty_policy = _choice(
        payload.get("uncertainty_policy"),
        valid_uncertainty,
        "state_limits_when_context_is_sparse",
    )
    style = _choice(payload.get("style"), valid_styles, "short_direct")
    filtering_rules = [
        str(item).strip()
        for item in payload.get("filtering_rules") or []
        if str(item).strip()
    ][:6]
    parsed_contract = _normalize_answer_contract(payload)
    return AnswerDirective(
        answer_intent=answer_intent,
        answer_focus=str(parsed_contract.get("answer_focus") or ""),
        answer_obligations=list(parsed_contract.get("answer_obligations") or []),
        organization=organization,
        evidence_policy=evidence_policy,
        filtering_rules=filtering_rules,
        uncertainty_policy=uncertainty_policy,
        style=style,
        confidence=_optional_float(payload.get("confidence")),
        reason=str(payload.get("reason") or "").strip(),
        backend=backend,
        raw=raw,
    )


def apply_answer_contract(
    directive: AnswerDirective,
    answer_contract: dict[str, Any] | None,
) -> AnswerDirective:
    """Keep route-owned answer obligations intact after provider synthesis."""

    normalized = _normalize_answer_contract(answer_contract)
    if not normalized:
        return directive
    return replace(
        directive,
        **{
            key: value
            for key, value in normalized.items()
            if key in {
                "answer_intent", "answer_focus", "answer_obligations", "uncertainty_policy",
                "coverage_requirement", "coverage_complete",
            }
        },
    )


def _normalize_answer_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, Any] = {}
    if "answer_intent" in value:
        answer_intent = str(value.get("answer_intent") or "").strip()
        if answer_intent in VALID_ANSWER_INTENTS:
            normalized["answer_intent"] = answer_intent
    if "answer_focus" in value:
        normalized["answer_focus"] = str(value.get("answer_focus") or "").strip()
    if "answer_obligations" in value:
        raw_obligations = value.get("answer_obligations")
        if isinstance(raw_obligations, str):
            raw_obligations = [part.strip() for part in raw_obligations.split(",") if part.strip()]
        obligations: list[str] = []
        if isinstance(raw_obligations, (list, tuple)):
            for item in raw_obligations[:8]:
                obligation = str(item).strip()
                if obligation in VALID_ANSWER_OBLIGATIONS and obligation not in obligations:
                    obligations.append(obligation)
        normalized["answer_obligations"] = obligations
    if "uncertainty_policy" in value:
        uncertainty_policy = str(value.get("uncertainty_policy") or "").strip()
        if uncertainty_policy in VALID_UNCERTAINTY_POLICIES:
            normalized["uncertainty_policy"] = uncertainty_policy
    if str(value.get("coverage_requirement") or "") in {"best_evidence", "complete_set"}:
        normalized["coverage_requirement"] = str(value["coverage_requirement"])
    if "coverage_complete" in value and isinstance(value.get("coverage_complete"), bool):
        normalized["coverage_complete"] = bool(value["coverage_complete"])
    return normalized


def _text_emotion_from_payload(payload: dict[str, Any], *, raw: str, backend: str) -> TextEmotionDirective:
    valid_labels = {"unknown", "calm", "happy", "annoyed", "sad", "anxious"}
    valid_confidence = {"low", "medium", "high"}
    label = _choice(payload.get("label"), valid_labels, "unknown")
    confidence = _choice(payload.get("confidence"), valid_confidence, "low")
    should_affect_reply = bool(payload.get("should_affect_reply")) and confidence == "high" and label != "unknown"
    return TextEmotionDirective(
        label=label,
        confidence=confidence,
        should_affect_reply=should_affect_reply,
        reason=str(payload.get("reason") or "").strip(),
        backend=backend,
        raw=raw,
    )


def _fallback_directive(
    route_debug: dict[str, Any],
    evidence_summary: dict[str, Any],
    *,
    error: str,
) -> AnswerDirective:
    recall_goal = str(route_debug.get("recall_goal") or "none")
    has_evidence = any(int(evidence_summary.get(key) or 0) for key in (
        "profile_count",
        "event_count",
        "observation_count",
        "timeline_count",
        "document_count",
        "web_context_count",
    ))
    if recall_goal == "raw_evidence":
        return AnswerDirective(
            answer_intent="raw_evidence",
            organization="evidence_first",
            evidence_policy="quote_raw_evidence",
            uncertainty_policy="state_limits_when_context_is_sparse",
            style="concise_structured",
            backend="fallback",
            error=error,
        )
    if recall_goal == "summary" or evidence_summary.get("observation_count"):
        return AnswerDirective(
            answer_intent="summary",
            organization="thematic",
            evidence_policy="separate_background",
            filtering_rules=[
                "Separate recent activities from stable profile/background facts.",
                "Use reflected observations as high-level hints and prefer source evidence when available.",
            ],
            uncertainty_policy="state_limits_when_context_is_sparse",
            style="concise_structured",
            backend="fallback",
            error=error,
        )
    return AnswerDirective(
        answer_intent="specific_fact" if has_evidence else "direct_answer",
        organization="direct",
        evidence_policy="use_available_context",
        uncertainty_policy="state_limits_when_context_is_sparse" if has_evidence else "none",
        style="short_direct",
        backend="fallback",
        error=error,
    )


def _choice(value: Any, choices: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in choices else default


def _parse_json_object_with_status(text: str) -> tuple[dict[str, Any], bool]:
    try:
        return _parse_json_object(text), True
    except (TypeError, ValueError):
        return {}, False


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
