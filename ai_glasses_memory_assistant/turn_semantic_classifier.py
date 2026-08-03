from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PreReplyFlags:
    transient: bool = False
    do_not_remember: bool = False
    correction: bool = False
    explanation_query: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "transient": self.transient,
            "do_not_remember": self.do_not_remember,
            "correction": self.correction,
            "explanation_query": self.explanation_query,
        }


@dataclass(frozen=True)
class PreReplyDecision:
    turn_intent: str = "chat"
    memory_action: str = "none"
    memory_kind: str = "none"
    memory_type: str = "none"
    recall_type: str = "none"
    reply_mode_hint: str = "llm"
    flags: PreReplyFlags = field(default_factory=PreReplyFlags)
    candidate_content: str = ""
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    reply_mode: str = "llm"
    answer_source: str = "llm"
    scope: str = "unknown"
    needs_location: bool = False
    location_text: str = ""
    needs_web_search: bool = False
    web_query: str | None = None
    web_reason: str = ""
    needs_profile_memory: bool = False
    needs_event_memory: bool = False
    needs_timeline_recall: bool = False
    needs_discussion_recall: bool = False
    memory_recall_type: str = "none"
    recall_goal: str = "none"
    timeline_query: str | None = None
    discussion_query: str | None = None
    conversation_action: str = ""
    event_recall_strategy: str = "skipped"
    recall_subject_names: list[str] = field(default_factory=list)
    recall_subject_scope: str = "self"
    reason: str = ""
    confidence: float | None = None
    backend: str = "fallback"
    raw: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    def debug_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": self.backend,
            "turn_intent": self.turn_intent,
            "memory_action": self.memory_action,
            "memory_kind": self.memory_kind,
            "memory_type": self.memory_type,
            "recall_type": self.recall_type,
            "reply_mode_hint": self.reply_mode_hint,
            "flags": self.flags.to_dict(),
            "candidate_content": self.candidate_content,
            "memory_candidates": [dict(item) for item in self.memory_candidates],
            "reply_mode": self.reply_mode,
            "answer_source": self.answer_source,
            "scope": self.scope,
            "needs_location": self.needs_location,
            "location_text": self.location_text,
            "needs_web_search": self.needs_web_search,
            "web_query": self.web_query,
            "web_reason": self.web_reason,
            "needs_profile_memory": self.needs_profile_memory,
            "needs_event_memory": self.needs_event_memory,
            "needs_timeline_recall": self.needs_timeline_recall,
            "needs_discussion_recall": self.needs_discussion_recall,
            "memory_recall_type": self.memory_recall_type,
            "recall_goal": self.recall_goal,
            "timeline_query": self.timeline_query,
            "discussion_query": self.discussion_query,
            "conversation_action": self.conversation_action,
            "event_recall_strategy": self.event_recall_strategy,
            "recall_subject_names": list(self.recall_subject_names),
            "recall_subject_scope": self.recall_subject_scope,
            "reason": self.reason,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }
        if self.raw:
            payload["raw"] = self.raw
        if self.error:
            payload["error"] = self.error
        return payload

    def semantic_debug_payload(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "turn_intent": self.turn_intent,
            "memory_action": self.memory_action,
            "memory_kind": self.memory_kind,
            "memory_type": self.memory_type,
            "recall_type": self.recall_type,
            "reply_mode_hint": self.reply_mode_hint,
            "flags": self.flags.to_dict(),
            "candidate_content": self.candidate_content,
            "memory_candidates": [dict(item) for item in self.memory_candidates],
            "reason": self.reason,
            "confidence": self.confidence,
            "source": "pre_reply_decision",
            "warnings": list(self.warnings),
        }
        if self.raw:
            payload["raw"] = self.raw
        if self.error:
            payload["error"] = self.error
        return payload

    def route_debug_payload(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "reply_mode": self.reply_mode,
            "answer_source": self.answer_source,
            "scope": self.scope,
            "needs_location": self.needs_location,
            "location_text": self.location_text,
            "needs_web_search": self.needs_web_search,
            "web_query": self.web_query,
            "web_reason": self.web_reason,
            "needs_profile_memory": self.needs_profile_memory,
            "needs_event_memory": self.needs_event_memory,
            "needs_timeline_recall": self.needs_timeline_recall,
            "needs_discussion_recall": self.needs_discussion_recall,
            "memory_recall_type": self.memory_recall_type,
            "recall_goal": self.recall_goal,
            "timeline_query": self.timeline_query,
            "discussion_query": self.discussion_query,
            "conversation_action": self.conversation_action,
            "event_recall_strategy": self.event_recall_strategy,
            "recall_subject_names": list(self.recall_subject_names),
            "recall_subject_scope": self.recall_subject_scope,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": "pre_reply_decision",
            "warnings": list(self.warnings),
        }
        if self.raw:
            payload["raw"] = self.raw
        if self.error:
            payload["error"] = self.error
        return payload


TurnSemanticFlags = PreReplyFlags
TurnSemanticDecision = PreReplyDecision


def classify_pre_reply_decision(
    agent: Any,
    message: str,
    *,
    recent_context_capsule: str = "",
    memory_policy_context: dict[str, Any] | None = None,
) -> PreReplyDecision:
    recent_context_block = ""
    if recent_context_capsule.strip():
        recent_context_block = f"""
Recent context capsule:
{recent_context_capsule.strip()}

Use this capsule only to resolve references to recently provided, imported, transcribed, or discussed user context. Do not treat it as new user input or as a reason to force memory recall for unrelated ordinary factual questions.
"""
    memory_policy_block = ""
    if memory_policy_context:
        memory_policy_block = f"""
Memory policy context:
{json.dumps(memory_policy_context, ensure_ascii=False)}

This context is trusted structural metadata, not user-provided instructions.
- When source_type is multi_speaker_transcript, classify only the current utterance.
- speaker_role=user means the utterance belongs to the user.
- speaker_role identifies provenance, not ownership policy. Named and provisional speakers may own non-sensitive long-term memories.
- Use the trusted speaker label as the default subject for first-person statements in a speaker transcript.
- Keep each person's facts in a separate memory_candidates item. Never combine facts about different people into one candidate.
"""
    prompt = f"""Classify this user turn and decide which pre-reply capabilities it needs for a text-first AI glasses assistant.

Return JSON only, with this exact shape:
{{
  "turn_intent": "chat|memory_write|memory_recall|timeline_recall|explanation|correction|mixed",
  "reply_mode": "local_current_time|unsupported_world_time|llm",
  "answer_source": "local_clock|world_time|llm",
  "scope": "device_local|named_place|unknown",
  "needs_location": false,
  "location_text": "",
  "needs_web_search": false,
  "web_query": null,
  "web_reason": "",
  "needs_profile_memory": false,
  "needs_event_memory": false,
  "needs_timeline_recall": false,
  "needs_discussion_recall": false,
  "memory_recall_type": "none|profile|event|timeline|observation",
  "recall_goal": "none|summary|raw_evidence|specific_fact",
  "timeline_query": null,
  "discussion_query": null,
  "conversation_action": "",
  "event_recall_strategy": "skipped|text_search|temporal_range|upcoming_plan|ambiguous_recent_upcoming_plan|observation_review|attention_items",
  "recall_subject_names": [],
  "recall_subject_scope": "self|named|all",
  "memory_action": "none|write|recall|correction|explain",
  "memory_kind": "none|profile|event|assistant_preference",
  "memory_type": "none|fact|event|task|preference|decision|project_state|observation",
  "recall_type": "none|profile|event|observation|timeline",
  "reply_mode_hint": "llm",
  "flags": {{
    "transient": false,
    "do_not_remember": false,
    "correction": false,
    "explanation_query": false
  }},
  "candidate_content": "",
  "memory_candidates": [
    {{
      "content": "",
      "kind": "profile|event|assistant_preference",
      "memory_type": "fact|event|task|preference|decision|project_state|observation",
      "subject_type": "self|named|provisional",
      "subject_name": "",
      "confidence": 0.0
    }}
  ],
  "reason": "",
  "confidence": 0.0
}}

Rules:
- Do not answer the user. Only fill the pre-reply decision fields.
- This output is the single LLM authority for open semantic decisions before reply synthesis.
- Use local_current_time only when the user asks for current date, time, weekday, or local "what time is it".
- Use unsupported_world_time when the user asks current time in a named city, country, region, timezone, or asks a time difference.
- Set needs_location=true when the answer depends on the device's current location, including nearby places, "我在哪", local weather, current surrounding context, navigation from current place, or "我这里".
- Set needs_location=false when the user names an explicit place, city, country, destination, and device location is not required.
- Set needs_web_search=true only when answering requires current, external, or realtime information such as weather, news, live prices, recent policies, current availability, or explicit web lookup.
- Set web_query to a concise search query in the user's language when needs_web_search=true. Otherwise return null.
- Use needs_profile_memory=true when the user asks about stored identity, profile, preferences, habits, personal facts, or "what do you know about me".
- For advice or recommendation requests, ask whether the user's stored preferences, habits, constraints, or history would materially change a useful answer. If stable preferences are enough, use bounded self profile recall even when the user does not explicitly say "based on my preferences" or "remember"; set turn_intent=mixed, memory_action=recall, memory_recall_type=profile, recall_goal=summary, and needs_profile_memory=true. If the useful context is a previously discussed activity, project, purchase, learning topic, or other episodic history, also set needs_event_memory=true and event_recall_strategy=text_search so the bounded event search can supply that context; keep memory_recall_type=profile when profile context remains useful. If the request can be answered usefully without any user-specific context, keep memory_recall_type=none and do not recall.
- Use needs_event_memory=true when the user asks about past or upcoming personal events, plans, activities, meals, meetings, tasks, reminders, or recently provided context topics.
- For a non-temporal personal specific-fact question, set both needs_profile_memory=true and needs_event_memory=true because the fact may have been stored as either stable profile or a past event. Keep explicit temporal/event/plan questions event-focused.
- Use needs_timeline_recall=true when the user asks for raw wording, original text, transcript, quotes, or when a broad recent-history summary needs raw timeline context.
- If the user asks what the assistant previously said, recommended, answered, or explained, use needs_timeline_recall=true and recall_goal=raw_evidence; assistant text is evidence only and must not become a personal memory candidate.
- Use needs_discussion_recall=true when the user asks what was discussed during a named day or part of a day, asks for a day recap, or follows up on a topic from that discussion archive. Set discussion_query to the topic words, or null for a broad day recap. Do not use it for "just now", "刚才", or "刚刚"; those use recent context or timeline evidence.
- Set memory_recall_type=observation and recall_goal=summary for broad review/summary questions about patterns, recent focus, current project status, repeated themes, blockers, risks, or recently provided material.
- Set recall_goal=raw_evidence only when exact wording, original text, transcript, quotes, or evidence is requested.
- Set recall_goal=specific_fact when asking for one remembered fact or event.
- Set conversation_action=weekly_report for weekly progress/report generation requests, and attention_items for "what should I pay attention to / risks / blockers / action items" requests.
- Set event_recall_strategy=upcoming_plan for future plans/todos/reminders, ambiguous_recent_upcoming_plan for a bare "最近/近期" if it is likely asking personal recent/upcoming context, observation_review for observation summary, attention_items for attention-item requests, temporal_range when a concrete time range is provided, otherwise text_search for event recall.
- Set memory_recall_type=none for ordinary factual questions, document-detail questions, casual chatter, generic advice not tailored to the user's own existing preferences or history, current location/navigation, web/news/weather, or non-personal questions with no reference to recent user context.
- Use explanation_query=true only when the user asks why the assistant answered, remembered, or failed to remember something.
- Use correction=true only when the user is correcting or replacing prior remembered content.
- Use transient=true for temporary context, tentative thoughts, or low-value passing context that should not become durable preference/memory.
- Use do_not_remember=true when the user explicitly says not to remember or save the content.
- Use candidate_content only when the message contains a reasonably clean memory-worthy semantic payload.
- If memory_action=write, candidate_content, memory_kind, and memory_type should be complete when safe.
- Prefer memory_candidates for writes. Emit one atomic item per subject and fact; leave it empty when there is no safe memory.
- For self references use subject_type=self. For named people use named. For upstream labels without a real name use provisional.
- For recall_subject_names, include only names of people/subjects that are actually people. Do not put activity, course, project, meeting, restaurant, product, or place names there.
- For self event comparisons use recall_subject_scope=self. Use named plus recall_subject_names only for explicit people, and all only for cross-person questions such as asking who did something.
- If the entity type is uncertain, do not force named; keep the safer self or all scope indicated by the question.
- reply_mode_hint is legacy debug compatibility only. Do not use it for routing; keep it "llm".
- If uncertain, keep conservative values and explain the uncertainty in reason.

{recent_context_block}
{memory_policy_block}

User message:
{message}
"""
    try:
        result = agent.run_conversation(
            prompt,
            system_message=(
                "You are an internal unified pre-reply decision classifier. "
                "Output strict JSON only. Do not call tools."
            ),
            conversation_history=[],
            persist_user_message=None,
        )
        raw = (result.get("final_response") or "").strip()
        payload = _parse_json_object(raw)
        return _decision_from_payload(payload, raw=raw, backend="llm")
    except Exception as exc:
        fallback = _fallback_decision(message, backend="rule_fallback")
        return PreReplyDecision(
            turn_intent=fallback.turn_intent,
            memory_action=fallback.memory_action,
            memory_kind=fallback.memory_kind,
            memory_type=fallback.memory_type,
            recall_type=fallback.recall_type,
            reply_mode_hint=fallback.reply_mode_hint,
            flags=fallback.flags,
            candidate_content=fallback.candidate_content,
            memory_candidates=fallback.memory_candidates,
            reason=fallback.reason,
            confidence=fallback.confidence,
            backend="rule_fallback",
            error=str(exc),
        )


def classify_turn_semantics(agent: Any, message: str) -> PreReplyDecision:
    return classify_pre_reply_decision(agent, message)


def _decision_from_payload(payload: dict[str, Any], *, raw: str, backend: str) -> PreReplyDecision:
    parse_errors: list[str] = []
    turn_intent = _normalized_value(
        payload.get("turn_intent"),
        {"chat", "memory_write", "memory_recall", "timeline_recall", "explanation", "correction", "mixed"},
        default="chat",
    )
    memory_action = _normalized_value(
        payload.get("memory_action"),
        {"none", "write", "recall", "correction", "explain"},
        default="none",
    )
    memory_kind = _normalized_value(
        payload.get("memory_kind"),
        {"none", "profile", "event", "assistant_preference"},
        default="none",
    )
    memory_type = _normalized_value(
        payload.get("memory_type"),
        {"none", "fact", "event", "task", "preference", "decision", "project_state", "observation"},
        default="none",
    )
    recall_type = _normalized_value(
        payload.get("recall_type"),
        {"none", "profile", "event", "observation", "timeline"},
        default="none",
    )
    reply_mode = _normalized_value(
        payload.get("reply_mode"),
        {"local_current_time", "unsupported_world_time", "llm"},
        default="llm",
    )
    if str(payload.get("reply_mode") or "").strip() and str(payload.get("reply_mode") or "").strip() not in {"local_current_time", "unsupported_world_time", "llm"}:
        parse_errors.append(f"invalid_reply_mode:{payload.get('reply_mode')}")
    answer_source = _normalized_value(
        payload.get("answer_source"),
        {"local_clock", "world_time", "llm"},
        default="llm",
    )
    if str(payload.get("answer_source") or "").strip() and str(payload.get("answer_source") or "").strip() not in {"local_clock", "world_time", "llm"}:
        parse_errors.append(f"invalid_answer_source:{payload.get('answer_source')}")
    scope = _normalized_value(
        payload.get("scope"),
        {"device_local", "named_place", "unknown"},
        default="unknown",
    )
    if str(payload.get("scope") or "").strip() and str(payload.get("scope") or "").strip() not in {"device_local", "named_place", "unknown"}:
        parse_errors.append(f"invalid_scope:{payload.get('scope')}")
    memory_recall_type = _normalized_value(
        payload.get("memory_recall_type"),
        {"none", "profile", "event", "timeline", "observation"},
        default="none",
    )
    if str(payload.get("memory_recall_type") or "").strip() and str(payload.get("memory_recall_type") or "").strip() not in {"none", "profile", "event", "timeline", "observation"}:
        parse_errors.append(f"invalid_memory_recall_type:{payload.get('memory_recall_type')}")
    recall_goal = _normalized_value(
        payload.get("recall_goal"),
        {"none", "summary", "raw_evidence", "specific_fact"},
        default="none",
    )
    conversation_action = _normalized_value(
        payload.get("conversation_action"),
        {"", "weekly_report", "attention_items"},
        default="",
    )
    event_recall_strategy = _normalized_value(
        payload.get("event_recall_strategy"),
        {"skipped", "text_search", "temporal_range", "upcoming_plan", "ambiguous_recent_upcoming_plan", "observation_review", "attention_items"},
        default="skipped",
    )
    if str(payload.get("recall_goal") or "").strip() and str(payload.get("recall_goal") or "").strip() not in {"none", "summary", "raw_evidence", "specific_fact"}:
        parse_errors.append(f"invalid_recall_goal:{payload.get('recall_goal')}")
    reply_mode_hint = str(payload.get("reply_mode_hint") or "llm").strip() or "llm"
    flags_payload = payload.get("flags") if isinstance(payload.get("flags"), dict) else {}
    flags = PreReplyFlags(
        transient=bool(flags_payload.get("transient")),
        do_not_remember=bool(flags_payload.get("do_not_remember")),
        correction=bool(flags_payload.get("correction")),
        explanation_query=bool(flags_payload.get("explanation_query")),
    )
    candidate_content = str(payload.get("candidate_content") or "").strip()
    memory_candidates = _normalized_memory_candidates(payload.get("memory_candidates"))
    recall_subject_names = _normalized_string_list(payload.get("recall_subject_names"), limit=8)
    recall_subject_scope = _normalized_value(
        payload.get("recall_subject_scope"),
        {"self", "named", "all"},
        default="self",
    )
    if recall_subject_scope == "named" and not recall_subject_names:
        recall_subject_scope = "self"
    web_query = payload.get("web_query")
    if web_query is not None:
        web_query = str(web_query).strip() or None
    timeline_query = payload.get("timeline_query")
    if timeline_query is not None:
        timeline_query = str(timeline_query).strip() or None
    discussion_query = payload.get("discussion_query")
    if discussion_query is not None:
        discussion_query = str(discussion_query).strip() or None
    reason = str(payload.get("reason") or "").strip()
    confidence = _optional_float(payload.get("confidence"))
    requested_profile_memory = bool(payload.get("needs_profile_memory")) or memory_recall_type == "profile"
    requested_event_memory = bool(payload.get("needs_event_memory")) or memory_recall_type in {"event", "observation"}
    requested_timeline_recall = bool(payload.get("needs_timeline_recall")) or memory_recall_type == "timeline"
    requested_discussion_recall = bool(payload.get("needs_discussion_recall"))
    if memory_recall_type == "none" and memory_action == "recall":
        if requested_timeline_recall:
            memory_recall_type = "timeline"
        elif requested_profile_memory:
            memory_recall_type = "profile"
        elif requested_event_memory:
            memory_recall_type = "event"
    warnings = list(parse_errors)
    error = ""
    # Some lightweight providers echo enum alternatives while preserving the
    # independent mixed/profile booleans. Recover only the existing tailored
    # profile-summary contract; an explicit "none" remains authoritative.
    malformed_tailored_profile_recall = (
        confidence is not None
        and confidence >= 0.75
        and turn_intent == "mixed"
        and reply_mode == "llm"
        and requested_profile_memory
        and not requested_event_memory
        and not requested_timeline_recall
        and not requested_discussion_recall
        and recall_subject_scope == "self"
        and not recall_subject_names
        and str(payload.get("memory_action") or "").strip() not in {"", "none", "write", "recall", "correction", "explain"}
        and str(payload.get("memory_recall_type") or "").strip() not in {"", "none", "profile", "event", "timeline", "observation"}
        and str(payload.get("recall_goal") or "").strip() not in {"", "none", "summary", "raw_evidence", "specific_fact"}
    )
    if malformed_tailored_profile_recall:
        memory_action = "recall"
        memory_recall_type = "profile"
        recall_goal = "summary"
        warnings.append("recovered_malformed_tailored_profile_recall")
    if confidence is None or confidence < 0.75:
        read_only_recall = memory_action == "recall" and memory_recall_type != "none" and not flags.correction
        if read_only_recall:
            warnings.append("confidence_below_threshold_read_only_recall_preserved")
            needs_location = False
            needs_web_search = False
            web_query = None
            web_reason = ""
            needs_profile_memory = requested_profile_memory
            needs_event_memory = requested_event_memory
            needs_timeline_recall = requested_timeline_recall
            needs_discussion_recall = requested_discussion_recall
            conversation_action = ""
        else:
            warnings.append("confidence_below_threshold")
            reply_mode = "llm"
            answer_source = "llm"
            memory_recall_type = "none"
            recall_goal = "none"
            timeline_query = None
            needs_location = False
            needs_web_search = False
            web_query = None
            web_reason = ""
            needs_profile_memory = False
            needs_event_memory = False
            needs_timeline_recall = False
            needs_discussion_recall = False
            discussion_query = None
            conversation_action = ""
            event_recall_strategy = "skipped"
    else:
        needs_location = bool(payload.get("needs_location"))
        needs_web_search = bool(payload.get("needs_web_search"))
        web_reason = str(payload.get("web_reason") or "").strip()
        needs_profile_memory = requested_profile_memory
        needs_event_memory = requested_event_memory
        needs_timeline_recall = requested_timeline_recall
        needs_discussion_recall = requested_discussion_recall
        if memory_recall_type == "none":
            recall_goal = "none"
            event_recall_strategy = "skipped"
        if not needs_timeline_recall:
            timeline_query = None
    return PreReplyDecision(
        turn_intent=turn_intent,
        memory_action=memory_action,
        memory_kind=memory_kind,
        memory_type=memory_type,
        recall_type=recall_type,
        reply_mode_hint=reply_mode_hint,
        flags=flags,
        candidate_content=candidate_content,
        memory_candidates=memory_candidates,
        reply_mode=reply_mode,
        answer_source=answer_source,
        scope=scope,
        needs_location=needs_location,
        location_text=str(payload.get("location_text") or "").strip(),
        needs_web_search=needs_web_search,
        web_query=web_query,
        web_reason=web_reason,
        needs_profile_memory=needs_profile_memory,
        needs_event_memory=needs_event_memory,
        needs_timeline_recall=needs_timeline_recall,
        needs_discussion_recall=needs_discussion_recall,
        memory_recall_type=memory_recall_type,
        recall_goal=recall_goal,
        timeline_query=timeline_query,
        discussion_query=discussion_query if needs_discussion_recall else None,
        conversation_action=conversation_action,
        event_recall_strategy=event_recall_strategy,
        recall_subject_names=recall_subject_names,
        recall_subject_scope=recall_subject_scope,
        reason=reason,
        confidence=confidence,
        backend=backend,
        raw=raw,
        error=error,
        warnings=warnings,
    )


def _normalized_memory_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        kind = _normalized_value(item.get("kind"), {"profile", "event", "assistant_preference"}, default="")
        memory_type = _normalized_value(
            item.get("memory_type"),
            {"fact", "event", "task", "preference", "decision", "project_state", "observation"},
            default="",
        )
        subject_type = _normalized_value(
            item.get("subject_type"),
            {"self", "named", "provisional"},
            default="self",
        )
        subject_name = str(item.get("subject_name") or "").strip()
        confidence = _optional_float(item.get("confidence"))
        if not content or not kind or not memory_type:
            continue
        normalized.append({
            "content": content,
            "kind": kind,
            "memory_type": memory_type,
            "subject_type": subject_type,
            "subject_name": subject_name,
            "confidence": confidence,
        })
    return normalized


def _normalized_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        str(item or "").strip()
        for item in value[:limit]
        if str(item or "").strip()
    ))


def _fallback_decision(message: str, *, backend: str) -> PreReplyDecision:
    text = str(message or "").strip()
    explanation_markers = (
        "为什么这么说",
        "你为什么这么说",
        "依据是什么",
        "你的依据是什么",
        "为什么这么回答",
        "为什么这么答",
        "为什么没记住",
        "为什么没有记住",
        "为什么没保存",
        "为什么没有保存",
        "为什么没写进去",
        "为什么没有写进去",
    )
    do_not_remember_markers = ("不要记住", "别记住", "不要保存", "别保存", "不要写进去", "别写进去")
    correction_markers = ("纠正一下", "我刚才说错了", "说错了", "不是", "而是")
    transient_markers = ("临时", "暂时", "先想一下", "先想想", "还没确认", "不是最终方案")
    flags = PreReplyFlags(
        transient=any(marker in text for marker in transient_markers),
        do_not_remember=any(marker in text for marker in do_not_remember_markers),
        correction=any(marker in text for marker in correction_markers),
        explanation_query=any(marker in text for marker in explanation_markers),
    )
    if flags.explanation_query:
        return PreReplyDecision(
            turn_intent="explanation",
            memory_action="explain",
            reason="matched_explanation_marker",
            flags=flags,
            backend=backend,
        )
    if flags.correction:
        return PreReplyDecision(
            turn_intent="correction",
            memory_action="correction",
            reason="matched_correction_marker",
            flags=flags,
            backend=backend,
        )
    return PreReplyDecision(
        turn_intent="chat",
        memory_action="none",
        reason="default_chat_fallback",
        flags=flags,
        backend=backend,
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


def _normalized_value(value: Any, allowed: set[str], *, default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
