from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from ai_glasses_memory_assistant.agent_bridge import AI_GLASSES_SYSTEM_PROMPT
from ai_glasses_memory_assistant.llm_runtime import create_demo_llm_client
from tests.helpers import CoreChatService, isolated_app_home
from tests.test_reader_contract import (
    READER_SCENARIOS,
    ReaderScenario,
    _assert_reply_contract,
    canonical_refusal,
    _seed_scenario,
)


REPORT_ENV = "AI_GLASSES_READER_CONTRACT_REPORT"
SMOKE_ENV = "AI_GLASSES_READER_CONTRACT_SMOKE"
REPEATS_ENV = "AI_GLASSES_READER_CONTRACT_REPEATS"


class RecordingReaderAgent:
    """Keep routing deterministic while delegating only the final Reader call."""

    def __init__(self, scenario: ReaderScenario, delegate: Any) -> None:
        self.scenario = scenario
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []
        self.model = getattr(delegate, "model", "")
        self.provider = getattr(delegate, "provider", "")
        self.api_mode = getattr(delegate, "api_mode", "")
        self.enabled_toolsets = list(getattr(delegate, "enabled_toolsets", []) or [])
        self.step_callback = None
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None

    def run_conversation(self, message: str, system_message: str | None = None, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({
            "message": message,
            "system_message": system_message,
            "persist_user_message": kwargs.get("persist_user_message"),
        })
        system = str(system_message or "").lower()
        if "unified pre-reply decision classifier" in system:
            return {"final_response": json.dumps(self.scenario.decision, ensure_ascii=False)}
        if "answer synthesis planner" in system:
            return {
                "final_response": json.dumps(
                    {
                        "answer_intent": "direct_answer",
                        "answer_focus": "provider must not replace the route contract",
                        "answer_obligations": ["comparison"],
                        "uncertainty_policy": "none",
                        "organization": "direct",
                        "evidence_policy": "use_available_context",
                        "style": "short_direct",
                    },
                    ensure_ascii=False,
                )
            }
        if system_message:
            # Temporal/correction helper calls are deterministic test plumbing;
            # they must not consume provider calls or influence the Reader result.
            if "temporal" in system or "time expression" in system:
                return {"final_response": json.dumps({"kind": "none", "confidence": 0.0})}
            if "correction" in system:
                return {"final_response": json.dumps({"is_correction": False, "confidence": 0.0})}
            return {"final_response": "{}"}
        return self.delegate.run_conversation(message, **kwargs)


def _report_path() -> Path:
    configured = str(os.getenv(REPORT_ENV) or "").strip()
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "ai-glasses-reader-contract.jsonl"


def _write_report(rows: list[dict[str, Any]]) -> None:
    path = _report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _result_row(
    *,
    scenario: ReaderScenario,
    attempt: int,
    status: str,
    agent: RecordingReaderAgent | None = None,
    expected_ids: list[str] | None = None,
    recalled_ids: list[str] | None = None,
    prompt_presence: dict[str, bool] | None = None,
    contract_match: bool = False,
    reply: str = "",
    reply_class: str = "",
    failure_stage: str = "",
    error_type: str = "",
) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "attempt": attempt,
        "status": status,
        "provider": getattr(agent, "provider", "") if agent else "",
        "model": getattr(agent, "model", "") if agent else "",
        "expected_evidence_ids": list(expected_ids or []),
        "recalled_evidence_ids": list(recalled_ids or []),
        "prompt_evidence_presence": dict(prompt_presence or {}),
        "answer_contract_match": bool(contract_match),
        "reply_class": reply_class or ("answered" if reply else ""),
        "reply_sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest() if reply else "",
        "failure_stage": failure_stage,
        "error_type": error_type,
    }


def _smoke_repeats() -> int:
    try:
        return max(1, min(2, int(os.getenv(REPEATS_ENV, "2"))))
    except ValueError:
        return 2


def _is_provider_transport_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "network",
            "llm http",
            "llm request",
            "response was not valid json",
            "rate limit",
        )
    )


def _query_for_attempt(scenario: ReaderScenario, attempt: int) -> str:
    if attempt > 1 and scenario.alternate_query:
        return scenario.alternate_query
    return scenario.query


@pytest.mark.skipif(os.getenv(SMOKE_ENV) != "1", reason="skipped: opt_in_required")
def test_reader_contract_with_real_provider() -> None:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    blocked = 0
    try:
        delegate = create_demo_llm_client(system_prompt=AI_GLASSES_SYSTEM_PROMPT)
    except Exception as exc:
        _write_report([{
            "status": "blocked",
            "failure_stage": "provider_transport",
            "error_type": type(exc).__name__,
            "reason": "provider_unavailable",
        }])
        pytest.skip(f"blocked: provider_unavailable:{type(exc).__name__}")

    for scenario in READER_SCENARIOS:
        for attempt in range(1, _smoke_repeats() + 1):
            with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
                agent = RecordingReaderAgent(scenario, delegate)
                service = CoreChatService(tmpdir, agent=agent)  # type: ignore[arg-type]
                expected_ids = _seed_scenario(service, scenario)
                debug: dict[str, Any] = {}
                recalled_ids: list[str] = []
                presence: dict[str, bool] = {}
                contract_match = False
                response: dict[str, Any] = {}
                try:
                    response = service.chat(
                        _query_for_attempt(scenario, attempt),
                        user_id="u1",
                        memory_writes_allowed=False,
                    )
                    debug = response["debug"]
                    if debug.get("memory_writes_allowed") is not False or response.get("saved_memories") != []:
                        raise AssertionError("provider smoke attempted a long-term memory write")
                    recalled_ids = [item["id"] for item in response["recalled_memories"]]
                    main_call = next(call for call in reversed(agent.calls) if not call["system_message"])
                    prompt = str(main_call["message"])
                    presence = {
                        evidence_id: evidence.content in prompt
                        for evidence, evidence_id in zip(scenario.evidence, expected_ids)
                    }
                    directive = debug["answer_directive"]
                    route_contract_match = all(
                        debug["pre_reply_decision"].get(key) == scenario.decision.get(key)
                        for key in (
                            "answer_intent",
                            "answer_focus",
                            "answer_obligations",
                            "uncertainty_policy",
                        )
                    )
                    planner_strategy_match = (
                        debug["planner"].get("event_recall_strategy")
                        == scenario.decision.get("event_recall_strategy")
                    )
                    contract_match = all(
                        directive.get(key) == scenario.decision.get(key)
                        for key in ("answer_focus", "answer_obligations", "uncertainty_policy")
                    ) and route_contract_match and planner_strategy_match
                    if set(recalled_ids) != set(expected_ids):
                        raise AssertionError("recall evidence IDs do not match the fixture")
                    if not all(presence.values()):
                        raise AssertionError("recalled evidence content is missing from the Reader prompt")
                    if not contract_match:
                        raise AssertionError("AnswerDirective contract differs from PreReplyDecision")
                    _assert_reply_contract(scenario, response["reply"])
                    rows.append(
                        _result_row(
                            scenario=scenario,
                            attempt=attempt,
                            status="pass",
                            agent=agent,
                            expected_ids=expected_ids,
                            recalled_ids=recalled_ids,
                            prompt_presence=presence,
                            contract_match=contract_match,
                            reply=str(response["reply"]),
                            reply_class=(
                                "canonical_refusal"
                                if canonical_refusal(str(response["reply"]))
                                else "answered"
                            ),
                        )
                    )
                except Exception as exc:
                    if not _is_provider_transport_error(exc):
                        failure_stage = "provider_reader"
                        if debug.get("pre_reply_decision") and set(recalled_ids) != set(expected_ids):
                            failure_stage = "routing_or_recall"
                        rows.append(
                            _result_row(
                                scenario=scenario,
                                attempt=attempt,
                                status="fail",
                                agent=agent,
                                expected_ids=expected_ids,
                                recalled_ids=recalled_ids,
                                prompt_presence=presence,
                                contract_match=contract_match,
                                reply=str(response.get("reply") or ""),
                                reply_class=(
                                    "canonical_refusal"
                                    if response.get("reply") and canonical_refusal(str(response["reply"]))
                                    else ""
                                ),
                                failure_stage=failure_stage,
                                error_type=type(exc).__name__,
                            )
                        )
                        failures.append(f"{scenario.scenario_id} attempt {attempt}: {failure_stage}:{type(exc).__name__}")
                        continue
                    blocked += 1
                    rows.append(
                        _result_row(
                            scenario=scenario,
                            attempt=attempt,
                            status="blocked",
                            agent=agent,
                            expected_ids=expected_ids,
                            error_type=type(exc).__name__,
                            failure_stage="provider_transport",
                        )
                    )

    _write_report(rows)
    if failures:
        pytest.fail("; ".join(failures))
    if blocked and blocked == len(READER_SCENARIOS) * _smoke_repeats():
        pytest.skip("blocked: all provider calls unavailable")
