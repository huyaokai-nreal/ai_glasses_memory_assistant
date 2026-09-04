"""Pure LongMemEval Reader answer-task construction.

This module deliberately accepts only recorded debug data.  Keeping this
normalization outside the product runner lets input-freezing tools verify the
exact same Reader contract without importing the product service or a model
client.
"""

from __future__ import annotations

from typing import Any


def extract_answer_task_from_debug(debug: Any) -> dict[str, Any] | None:
    """Return the Reader contract encoded by a recorded pre-reply decision.

    ``None`` preserves the legacy no-op behavior when no contract field can
    affect Reader behavior.  This function must stay semantically identical to
    the runner's private compatibility wrapper.
    """

    if not isinstance(debug, dict):
        return None
    decision = debug.get("pre_reply_decision")
    if not isinstance(decision, dict):
        planner = debug.get("planner")
        decision = planner.get("decision") if isinstance(planner, dict) else None
    if not isinstance(decision, dict):
        return None
    intent = decision.get("answer_intent")
    obligations = decision.get("answer_obligations")
    focus = decision.get("answer_focus")
    uncertainty = decision.get("uncertainty_policy")
    planner_debug = debug.get("planner") if isinstance(debug.get("planner"), dict) else {}
    coverage_requirement = str(planner_debug.get("coverage_requirement") or "")
    memory_debug = debug.get("memory") if isinstance(debug.get("memory"), dict) else {}
    complete_set_debug = memory_debug.get("complete_set")
    if not isinstance(complete_set_debug, dict):
        complete_set_debug = {}
    normalized_obligations: list[str] = []
    if isinstance(obligations, list):
        normalized_obligations = [str(item).strip() for item in obligations if str(item).strip()]
    elif isinstance(obligations, str):
        normalized_obligations = [part.strip() for part in obligations.split(",") if part.strip()]
    if (
        str(intent or "") in {"", "direct_answer"}
        and not normalized_obligations
        and not str(focus or "").strip()
        and coverage_requirement != "complete_set"
    ):
        return None
    task: dict[str, Any] = {}
    if intent:
        task["answer_intent"] = str(intent)
    if normalized_obligations:
        task["answer_obligations"] = normalized_obligations
    if focus:
        task["answer_focus"] = str(focus)
    if uncertainty:
        task["uncertainty_policy"] = str(uncertainty)
    if coverage_requirement in {"best_evidence", "complete_set"}:
        task["coverage_requirement"] = coverage_requirement
    if coverage_requirement == "complete_set":
        task["coverage_complete"] = complete_set_debug.get("coverage_complete") is True
        task["truncated"] = complete_set_debug.get("truncated") is True
        task["truncation_reason"] = str(complete_set_debug.get("truncation_reason") or "")
        task["source_ids"] = [
            str(source_id)
            for source_id in (complete_set_debug.get("source_ids") or [])
            if str(source_id)
        ]
    return task
