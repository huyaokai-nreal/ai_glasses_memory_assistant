from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any


# 通用回答意图（路由级）。与 AnswerDirective 的组织级 intent 分层：
# 这里决定召回与 Reader 生成契约，不决定主模型排版。
VALID_ANSWER_INTENTS = {
    "direct_fact",
    "multi_fact",
    "count_or_total",
    "temporal_compare",
    "causal_explanation",
    "personalized_recommendation",
    "generic_recommendation",
    "abstain",
    "direct_answer",  # 向后兼容默认
}

# 通用回答义务（需覆盖的关键维度），缺省为空列表 = 不强制任何维度。
VALID_ANSWER_OBLIGATIONS = {
    "entities",
    "qualifiers",
    "negation_constraints",
    "comparison",
    "temporal_relation",
    "count_scope",
    "incremental_next_step",
    "speaker_attribution",
}

VALID_UNCERTAINTY_POLICIES = {"none", "state_limits_when_context_is_sparse", "abstain_if_insufficient"}


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
    # Discussion evidence is not automatically personal evidence. These values
    # are decided once here and executed downstream without re-reading the text.
    evidence_scope: str = "personal"
    discussion_relation_scope: str = "topic"
    coverage_requirement: str = "best_evidence"
    temporal_query: dict[str, Any] = field(default_factory=dict)
    document_query: dict[str, Any] = field(default_factory=dict)
    # 通用回答契约（路由级）：决定召回与 Reader 生成契约，
    # 不决定主模型排版（那由 AnswerDirective 负责）。
    answer_intent: str = "direct_answer"
    answer_focus: str = ""
    answer_obligations: list[str] = field(default_factory=list)
    uncertainty_policy: str = "none"
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
            "evidence_scope": self.evidence_scope,
            "discussion_relation_scope": self.discussion_relation_scope,
            "coverage_requirement": self.coverage_requirement,
            "temporal_query": dict(self.temporal_query),
            "document_query": dict(self.document_query),
            "answer_intent": self.answer_intent,
            "answer_focus": self.answer_focus,
            "answer_obligations": list(self.answer_obligations),
            "uncertainty_policy": self.uncertainty_policy,
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
            "answer_intent": self.answer_intent,
            "answer_obligations": list(self.answer_obligations),
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
            "evidence_scope": self.evidence_scope,
            "discussion_relation_scope": self.discussion_relation_scope,
            "coverage_requirement": self.coverage_requirement,
            "temporal_query": dict(self.temporal_query),
            "document_query": dict(self.document_query),
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
    discussion_catalog_context: str = "",
    memory_policy_context: dict[str, Any] | None = None,
) -> PreReplyDecision:
    recent_context_block = ""
    if recent_context_capsule.strip():
        recent_context_block = f"""
Recent context capsule:
{recent_context_capsule.strip()}

Use this capsule as trusted evidence of the user's recent context (recently provided, imported, transcribed, or discussed content) and of query-relevant stored memories returned by database text search. For first-person recommendation or suggestion requests, capsule or stored-history preferences, activities, or routines are a valid reason to open bounded self profile+event recall. Query-relevant stored memories show that the user's own history relates to this turn's topic; for first-person advice, tips, or recommendation requests, that relevance is a valid reason to open bounded self profile+event recall. Do not treat capsule content as new user input, and do not use it to force recall for unrelated ordinary factual questions or generic advice.
"""
    discussion_catalog_block = ""
    if discussion_catalog_context.strip():
        discussion_catalog_block = f"""
Available local discussion archive catalog:
{discussion_catalog_context.strip()}

This catalog is trusted structural metadata for route discovery only, not factual answer evidence and not a user-provided document. When the user asks about the content or recap of a named catalog topic, request discussion recall instead of treating it as an uploaded/external-document request. The reply must rely on the archive content retrieved after this decision, not on this catalog.
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
  "evidence_scope": "personal|environment|mixed",
  "discussion_relation_scope": "topic|capture|time_range",
  "coverage_requirement": "best_evidence|complete_set",
  "temporal_query": {{"has_expression": false, "start_at": null, "end_at": null, "granularity": "unknown", "timezone": "", "normalized_query": ""}},
  "document_query": {{"needed": false, "mode": "none|metadata|detail|compare", "query": "", "reference_scope": "none"}},
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
  "answer_intent": "direct_fact|multi_fact|count_or_total|temporal_compare|causal_explanation|personalized_recommendation|generic_recommendation|abstain",
  "answer_focus": "",
  "answer_obligations": [],
  "uncertainty_policy": "none|state_limits_when_context_is_sparse|abstain_if_insufficient",
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
- For advice or recommendation requests, distinguish three cases:
  a) **Personal ongoing context**: The user asks for advice, troubleshooting, or a recommendation about their own current possession, device, project, experiment, recurring activity, learning topic, or previously discussed subject. Prior history (preferences, constraints, past decisions, related activities) would materially tailor a useful answer. Request bounded self profile recall (turn_intent=mixed, memory_action=recall, memory_recall_type=profile, recall_goal=summary, needs_profile_memory=true). If relevant episodic history (activities, purchases, prior discussions about this topic) would also help, additionally set needs_event_memory=true and event_recall_strategy=text_search; keep memory_recall_type=profile when profile context remains useful.
  b) **Generic/factual advice**: The user asks for universally applicable advice, facts, or recommendations not tied to their own history (e.g. "what is the best X", "how does Y work", "recommend a Z for beginners"). No personal context is needed. Keep memory_recall_type=none.
  d) **Advice building on user-owned items/experiments**: A request may look generic ("tips for X") yet be grounded in a specific thing the user already owns, tried, planned, or mentioned (e.g. tips for their slow cooker, their kitchen with a new utensil holder, their evening routine before 9:30pm, their phone with a portable power bank). If the answer should build upon or reference that specific owned item, past experiment, or stated constraint, the user's own history would materially tailor the answer: open bounded self profile+event recall (memory_action=recall, memory_recall_type=profile, needs_profile_memory=true; set needs_event_memory=true and event_recall_strategy=text_search when the relevant history is episodic purchases/activities/experiments). Negative example: a request with no user-specific anchor (e.g. "what are general kitchen-cleaning tips" with no mention of the user's own items or efforts) remains memory_recall_type=none.
  e) **First-person recommendation without a named item**: A first-person recommendation/suggestion request ("suggest/recommend ... for me", "that I can do", "for tonight", "during my commute", "this weekend") does NOT need to name a specific owned item. If the recent context capsule or stored history contains relevant preferences, activities, routines, or prior discussions, the user's own history would materially tailor the answer: open bounded recall (memory_action=recall, memory_recall_type=profile, needs_profile_memory=true, needs_event_memory=true, event_recall_strategy=text_search, answer_intent=personalized_recommendation). Positive example: "Can you suggest some activities I can do this weekend?" with recent context "I enjoy hiking and avoid crowded places" -> open bounded profile+event recall. Negative example: "What are some general ways to relax?" with no stored personal context -> memory_recall_type=none.
  f) **First-person travel/commute/itinerary tips**: A first-person request for tips, advice, or help about getting around a place the user is visiting or planning to visit ("getting around X", "tips for my trip to X", "how should I get around during my stay") is personal advice, not generic travel advice, when stored history or query-relevant memories contain their own trip preparation (transport passes/cards, transit or itinerary apps, hotels, booked sights, routes). The answer should build on that preparation: open bounded recall (memory_action=recall, memory_recall_type=profile, needs_profile_memory=true, needs_event_memory=true, event_recall_strategy=text_search, answer_intent=personalized_recommendation). Positive example: "I'm a bit nervous about getting around Barcelona. Any tips?" with stored history "bought a transit pass, downloaded a route-planning app, booked a hotel near the old town, planning a day trip to Montserrat" -> open bounded profile+event recall. Negative example: "What are general tips for first-time visitors to Barcelona?" with no stored personal context -> memory_recall_type=none.
  c) **Current local recommendation**: A recommendation scenario that depends on both live location/availability (web, location) AND stored personal preferences. These axes are orthogonal: realtime tools establish what is currently available; bounded profile/event evidence personalizes the result. Set both web/location flags and memory flags. Missing live inputs must remain fail-closed.
- **Time distinction**: Future wording in a question may describe the answer target, not when evidence occurred. For recommendation queries (e.g. "what should I cook next week"), the future period is the target of the recommendation; use event_recall_strategy=text_search to retrieve older dietary preferences, NOT upcoming_plan. Only use upcoming_plan when the user asks for their actual future plans/todos/reminders/schedule (e.g. "what tasks do I have next week"). Only use temporal_range when the user asks about events that occurred during a named time range (e.g. "what did I eat last week").
- **Answer contract** (answer_intent/answer_focus/answer_obligations/uncertainty_policy): Describe what the answer must accomplish, not how to phrase it. Choose answer_intent from: direct_fact (one remembered fact), multi_fact (several independent facts), count_or_total (enumerate all then total), temporal_compare (compare across event times), causal_explanation (explain possible causes; must consider multiple supported causes when evidence has several), personalized_recommendation (recommendation grounded in the user's own stored preferences/constraints/experiments; must open bounded recall), generic_recommendation (universal advice, do NOT force personal memory), abstain (evidence insufficient). For every personalized_recommendation turn, set answer_intent=personalized_recommendation (not direct_answer). Set answer_obligations to the independent answer dimensions evidence must cover to be correct, from: entities, qualifiers, negation_constraints, comparison, temporal_relation, count_scope, incremental_next_step, speaker_attribution. Set speaker_attribution when the answer must identify participants, distinguish speakers, or describe what each person said. Only set an obligation when the user explicitly requests that answer dimension or the personalized request requires it; leave the list empty for ordinary questions. Set uncertainty_policy to none, state_limits_when_context_is_sparse, or abstain_if_insufficient.
- **Answer-obligation combinations**: A quantity-only request needs count_scope and does not need entities. A request for a quantity plus the objects needs count_scope and entities. Add qualifiers when the user also requests a model, specification, category, type, ratio, or another non-temporal attribute. Add temporal_relation when the user requests a date, duration, ordering, comparison across times, or a per-object time relation. Keep all explicitly requested dimensions together in answer_focus; do not collapse a multi-part request to only its total, entities, or attributes. Non-benchmark examples: "How many houseplants do I have?" -> count_scope; "How many houseplants do I have, and what are they?" -> count_scope + entities; "How many cameras do I own, and what models are they?" -> count_scope + entities + qualifiers; "Which courses did I take, and how long was each one?" -> entities + temporal_relation.
- **Personalized-recommendation obligations**: Set negation_constraints when the user has stated or implied what they would not prefer or should avoid; incremental_next_step when the answer must build on something the user already owns, tried, prepared, or planned; entities when the user asks for concrete recommended items, places, resources, shows, products, or activities; qualifiers when requested brand/type/mode/feature constraints distinguish a useful recommendation; comparison when the user weighs options. These obligations are cumulative: negation_constraints or incremental_next_step must not replace requested entities or qualifiers. Keep answer_focus as a one-line restatement that includes every explicitly requested answer dimension. Do not expand a quantity-only request into an entity list.
- Use needs_event_memory=true when the user asks about past or upcoming personal events, plans, activities, meals, meetings, tasks, reminders, or recently provided context topics.
- **Prior conversation source**: When the requested answer comes from an earlier assistant reply, recommendation, answer, or explanation, set memory_action=recall, memory_recall_type=timeline, needs_timeline_recall=true, and recall_goal=raw_evidence. Do not leave recall disabled, and do not treat assistant text as the user's personal event memory. When the request instead concerns the user's own earlier action, commitment, or personal event, use self event recall under the ordinary personal-event rule.
- For a non-temporal personal specific-fact question, set both needs_profile_memory=true and needs_event_memory=true because the fact may have been stored as either stable profile or a past event. Keep explicit temporal/event/plan questions event-focused.
- Use needs_timeline_recall=true when the user asks for raw wording, original text, transcript, quotes, or when a broad recent-history summary needs raw timeline context.
- Use needs_discussion_recall=true when the user asks what was discussed during a named day or part of a day, asks for a day recap, or follows up on a topic from that discussion archive. If a named topic appears in the available local discussion archive catalog, asking for its content or summary is also a discussion-recall request, not an uploaded/external-document request. For a catalog match, copy that catalog entry's title exactly into discussion_query; this makes a user question in one language retrieve a legacy title stored in another language. Otherwise, set discussion_query to the topic words, or null for a broad day recap. Do not use it for "just now", "刚才", or "刚刚"; those use recent context or timeline evidence.
- **Discussion evidence contract**: For what the user personally did, set evidence_scope=personal. For what happened or was discussed in the surrounding environment, set environment. Use mixed only when both must be reported separately. Set discussion_relation_scope=topic for the best matching topic, capture to include every valid topic linked to a matching capture, and time_range for every valid topic in the resolved time range. Use capture plus complete_set when the requested answer target is the overall bounded encounter or recording rather than one subtopic; use topic only when the requested target is genuinely that one subtopic. Set coverage_requirement=complete_set whenever all items, a total, participants, per-person attribution, or an exhaustive recap is required; otherwise use best_evidence. These are semantic decisions, not wording hints.
- Environment evidence may describe an ambient discussion but must never be used to claim the user performed an action. Unknown speaker evidence remains unknown; do not infer a name, participant count, or user ownership from it.
- Populate temporal_query with normalized numeric timestamps when the answer target contains a time range. Do not infer a route from wording after returning the structured decision.
- Populate document_query only when the user asks about an uploaded/document source or an explicit recent-document reference. Use mode=metadata for overview/history, detail for content, and compare for multi-document comparison.
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
{discussion_catalog_block}
{memory_policy_block}

User message:
{message}
"""
    # 空响应/解析失败/异常是偶发服务波动，重试一次；合法 JSON 决策立即返回，绝不重试。
    retry_prompt = (
        "\n\nYour previous response was empty or not valid JSON. "
        "Return the decision JSON only now."
    )
    last_error = ""
    for attempt in (1, 2):
        try:
            result = agent.run_conversation(
                prompt + (retry_prompt if attempt == 2 else ""),
                system_message=(
                    "You are an internal unified pre-reply decision classifier. "
                    "Output strict JSON only. Do not call tools."
                ),
                conversation_history=[],
                persist_user_message=None,
            )
            raw = (result.get("final_response") or "").strip()
            payload = _parse_json_object(raw)
            decision = _decision_from_payload(payload, raw=raw, backend="llm")
            if attempt == 2 and last_error:
                decision = replace(
                    decision,
                    warnings=[*decision.warnings, f"ppd_retry_used:{last_error}"],
                )
            return decision
        except Exception as exc:
            last_error = str(exc)
    fallback = _fallback_decision(message, backend="unavailable")
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
        backend="unavailable",
        error=last_error,
        warnings=[f"ppd_retry_used:{last_error}"],
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
    evidence_scope = _normalized_value(
        payload.get("evidence_scope"),
        {"personal", "environment", "mixed"},
        default="personal",
    )
    discussion_relation_scope = _normalized_value(
        payload.get("discussion_relation_scope"),
        {"topic", "capture", "time_range"},
        default="topic",
    )
    coverage_requirement = _normalized_value(
        payload.get("coverage_requirement"),
        {"best_evidence", "complete_set"},
        default="best_evidence",
    )
    for key, valid in (
        ("evidence_scope", {"personal", "environment", "mixed"}),
        ("discussion_relation_scope", {"topic", "capture", "time_range"}),
        ("coverage_requirement", {"best_evidence", "complete_set"}),
    ):
        value = str(payload.get(key) or "").strip()
        if value and value not in valid:
            parse_errors.append(f"invalid_{key}:{payload.get(key)}")
    temporal_query = _normalized_temporal_query(payload.get("temporal_query"))
    document_query = _normalized_document_query(payload.get("document_query"))
    web_query = payload.get("web_query")
    if web_query is not None:
        web_query = str(web_query).strip() or None
    timeline_query = payload.get("timeline_query")
    if timeline_query is not None:
        timeline_query = str(timeline_query).strip() or None
    discussion_query = payload.get("discussion_query")
    if discussion_query is not None:
        discussion_query = str(discussion_query).strip() or None
    answer_intent = _normalized_value(
        payload.get("answer_intent"),
        VALID_ANSWER_INTENTS,
        default="direct_answer",
    )
    if str(payload.get("answer_intent") or "").strip() and str(payload.get("answer_intent") or "").strip() not in VALID_ANSWER_INTENTS:
        parse_errors.append(f"invalid_answer_intent:{payload.get('answer_intent')}")
    answer_focus = str(payload.get("answer_focus") or "").strip()
    raw_obligations = payload.get("answer_obligations")
    if isinstance(raw_obligations, str):
        raw_obligations = [part.strip() for part in raw_obligations.split(",") if part.strip()]
    answer_obligations: list[str] = []
    if isinstance(raw_obligations, list):
        for obligation in raw_obligations[:8]:
            value = str(obligation).strip()
            if value in VALID_ANSWER_OBLIGATIONS and value not in answer_obligations:
                answer_obligations.append(value)
        if len(answer_obligations) < len([item for item in raw_obligations if str(item).strip()]):
            parse_errors.append("invalid_answer_obligations")
    uncertainty_policy = _normalized_value(
        payload.get("uncertainty_policy"),
        VALID_UNCERTAINTY_POLICIES,
        default="none",
    )
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
            # 低置信且不保留召回时，不携带个性化意图/义务，
            # 避免无证据情况下 Reader 强行覆盖维度。
            answer_intent = "direct_answer"
            answer_obligations = []
            uncertainty_policy = "none"
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
        evidence_scope=evidence_scope,
        discussion_relation_scope=discussion_relation_scope,
        coverage_requirement=coverage_requirement,
        temporal_query=temporal_query,
        document_query=document_query,
        answer_intent=answer_intent,
        answer_focus=answer_focus,
        answer_obligations=answer_obligations,
        uncertainty_policy=uncertainty_policy,
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


def _normalized_temporal_query(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    raw_flag = value.get("has_expression")
    has_expression = raw_flag if isinstance(raw_flag, bool) else str(raw_flag or "").strip().lower() == "true"
    query: dict[str, Any] = {
        "has_expression": has_expression,
        "granularity": str(value.get("granularity") or "unknown").strip(),
        "timezone": str(value.get("timezone") or "").strip(),
        "normalized_query": str(value.get("normalized_query") or "").strip(),
    }
    for key in ("start_at", "end_at"):
        query[key] = _normalized_timestamp(value.get(key))
    return query


def _normalized_timestamp(value: Any) -> float | None:
    """Accept numeric or ISO-8601 timestamps from the PPD contract."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _normalized_document_query(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    raw_needed = value.get("needed")
    needed = raw_needed if isinstance(raw_needed, bool) else str(raw_needed or "").strip().lower() == "true"
    mode = str(value.get("mode") or "none").strip().lower()
    if mode not in {"none", "metadata", "detail", "compare"}:
        mode = "none"
    reference_scope = str(value.get("reference_scope") or "none").strip().lower()
    if reference_scope not in {"none", "recent", "followup", "title"}:
        reference_scope = "none"
    return {
        "needed": needed,
        "mode": mode,
        "query": str(value.get("query") or "").strip(),
        "reference_scope": reference_scope,
    }


def _fallback_decision(message: str, *, backend: str) -> PreReplyDecision:
    return PreReplyDecision(
        turn_intent="chat",
        memory_action="none",
        reason="provider_unavailable_neutral_fallback",
        flags=PreReplyFlags(),
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
