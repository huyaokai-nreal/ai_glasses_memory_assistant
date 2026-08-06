from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from .turn_semantic_classifier import VALID_ANSWER_OBLIGATIONS, VALID_UNCERTAINTY_POLICIES


@dataclass(frozen=True)
class AnswerDirective:
    answer_intent: str = "direct_answer"
    answer_focus: str = ""
    answer_obligations: list[str] = field(default_factory=list)
    organization: str = "direct"
    evidence_policy: str = "use_available_context"
    filtering_rules: list[str] = field(default_factory=list)
    uncertainty_policy: str = "state_limits_when_context_is_sparse"
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
            f"style: {self.style}",
        ]
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
            if key in {"answer_focus", "answer_obligations", "uncertainty_policy"}
        },
    )


def _normalize_answer_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, Any] = {}
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
