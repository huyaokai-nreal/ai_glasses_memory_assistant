from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


_EXTRACTABLE_ROLES = {"memory_candidate", "correction"}


@dataclass(frozen=True)
class SegmentSemanticDecision:
    segment_index: int
    raw_span: str
    semantic_role: str = "memory_candidate"
    noise_level: str = "low"
    contains_filler: bool = False
    do_not_remember_scope: str = ""
    should_extract: bool = True
    candidate_span: str = ""
    candidate_hint: str = ""
    confidence: float | None = None
    reason: str = ""
    backend: str = "unavailable"
    raw: str = ""
    error: str = ""

    def extraction_text(self) -> str:
        return (self.candidate_span or self.raw_span).strip()

    def debug_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "segment_index": self.segment_index,
            "raw_span": self.raw_span,
            "semantic_role": self.semantic_role,
            "noise_level": self.noise_level,
            "contains_filler": self.contains_filler,
            "do_not_remember_scope": self.do_not_remember_scope,
            "should_extract": self.should_extract,
            "candidate_span": self.candidate_span,
            "candidate_hint": self.candidate_hint,
            "confidence": self.confidence,
            "reason": self.reason,
            "backend": self.backend,
        }
        if self.raw:
            payload["raw"] = self.raw
        if self.error:
            payload["error"] = self.error
        return payload


def classify_segment_semantics(
    agent: Any,
    *,
    segment: str,
    segment_index: int,
    marker_kinds: list[str] | None = None,
) -> SegmentSemanticDecision:
    text = str(segment or "").strip()
    prompt = f"""Classify this transcript segment for an AI glasses memory assistant.

Return JSON only, with this exact shape:
{{
  "semantic_role": "memory_candidate|temporary_context|chitchat|noise_only|correction|do_not_remember|sensitive_reject",
  "noise_level": "none|low|medium|high",
  "contains_filler": false,
  "do_not_remember_scope": "",
  "should_extract": false,
  "candidate_span": "",
  "candidate_hint": "profile|task|decision|project_state|event|preference|correction|sensitive|",
  "confidence": 0.0,
  "reason": ""
}}

Rules:
- Decide from the supplied segment and explicit capture metadata only; do not
  rely on a language-specific marker table.
- Do not skip an entire segment only because it contains filler or casual chatter.
- If a segment contains both do-not-remember content and a separate useful task/fact, set should_extract=true and put only the useful span in candidate_span.
- If the whole segment is only filler, background noise, or explicitly do-not-remember content, set should_extract=false.
- Never invent facts outside raw_span.

segment_index: {segment_index}
raw_span:
{text}
"""
    try:
        result = agent.run_conversation(
            prompt,
            system_message=(
                "You are an internal segment semantic cleaner. "
                "Output strict JSON only. Do not call tools."
            ),
            conversation_history=[],
            persist_user_message=None,
        )
        raw = str(result.get("final_response") or "").strip()
        payload = _parse_json_object(raw)
        return _decision_from_payload(payload, segment_index=segment_index, raw_span=text, raw=raw, backend="llm")
    except Exception as exc:
        return SegmentSemanticDecision(
            segment_index=segment_index,
            raw_span=text,
            semantic_role="unknown",
            noise_level="unknown",
            should_extract=False,
            confidence=0.0,
            reason="structured_segment_classifier_error",
            backend="unavailable",
            error=str(exc),
        )


def _decision_from_payload(
    payload: dict[str, Any],
    *,
    segment_index: int,
    raw_span: str,
    raw: str,
    backend: str,
) -> SegmentSemanticDecision:
    role = str(payload.get("semantic_role") or "").strip().lower()
    if role not in {
        "memory_candidate",
        "temporary_context",
        "chitchat",
        "noise_only",
        "correction",
        "do_not_remember",
        "sensitive_reject",
    }:
        role = "memory_candidate"
    candidate_span = str(payload.get("candidate_span") or "").strip()
    should_extract = bool(payload.get("should_extract")) and role in _EXTRACTABLE_ROLES
    if should_extract and not candidate_span:
        candidate_span = raw_span
    return SegmentSemanticDecision(
        segment_index=segment_index,
        raw_span=raw_span,
        semantic_role=role,
        noise_level=str(payload.get("noise_level") or "low").strip().lower(),
        contains_filler=bool(payload.get("contains_filler")),
        do_not_remember_scope=str(payload.get("do_not_remember_scope") or "").strip(),
        should_extract=should_extract,
        candidate_span=candidate_span,
        candidate_hint=str(payload.get("candidate_hint") or "").strip().lower(),
        confidence=_optional_float(payload.get("confidence")),
        reason=str(payload.get("reason") or "").strip(),
        backend=backend,
        raw=raw,
    )


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
