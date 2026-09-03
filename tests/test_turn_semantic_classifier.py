import json
from datetime import datetime, timedelta, timezone

from ai_glasses_memory_assistant.turn_semantic_classifier import (
    _decision_from_payload,
    classify_pre_reply_decision,
)
from ai_glasses_memory_assistant.turn_planner import TurnPlan


def test_discussion_scope_contract_is_text_independent_after_pre_reply_decision() -> None:
    payload = {
        "memory_action": "recall",
        "needs_discussion_recall": True,
        "discussion_query": "任意主题",
        "evidence_scope": "environment",
        "discussion_relation_scope": "capture",
        "coverage_requirement": "complete_set",
        "confidence": 0.95,
    }
    decision = _decision_from_payload(payload, raw="{}", backend="llm")

    first = TurnPlan().apply_pre_reply_decision(decision)
    second = TurnPlan().apply_pre_reply_decision(decision)

    assert first.evidence_scope == second.evidence_scope == "environment"
    assert first.discussion_relation_scope == second.discussion_relation_scope == "capture"
    assert first.coverage_requirement == second.coverage_requirement == "complete_set"


def test_invalid_discussion_scope_contract_fails_closed() -> None:
    decision = _decision_from_payload(
        {
            "evidence_scope": "guessed_identity",
            "discussion_relation_scope": "nearby_topics",
            "coverage_requirement": "everything",
            "confidence": 0.95,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.evidence_scope == "personal"
    assert decision.discussion_relation_scope == "topic"
    assert decision.coverage_requirement == "best_evidence"


def test_speaker_attribution_is_a_classifier_owned_answer_obligation() -> None:
    decision = _decision_from_payload(
        {
            "answer_obligations": ["speaker_attribution"],
            "confidence": 0.95,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.answer_obligations == ["speaker_attribution"]


def test_ppd_temporal_protocol_normalizes_epoch_iso_and_explicit_timezone() -> None:
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    cases = [
        (start.timestamp(), end.timestamp(), "", "epoch"),
        (str(start.timestamp()), str(end.timestamp()), "", "epoch_string"),
        ("2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z", "", "iso8601_offset"),
        ("2026-09-01T08:00:00+08:00", "2026-09-02T08:00:00+08:00", "", "iso8601_offset"),
        ("2026-09-01T08:00:00", "2026-09-02T08:00:00", "+08:00", "iso8601_query_timezone"),
    ]

    for raw_start, raw_end, timezone_name, expected_format in cases:
        decision = _decision_from_payload(
            {
                "temporal_query": {
                    "has_expression": True,
                    "start_at": raw_start,
                    "end_at": raw_end,
                    "timezone": timezone_name,
                    "granularity": "day",
                },
                "confidence": 0.95,
            },
            raw="{}",
            backend="llm",
        )

        query = decision.temporal_query
        assert query["start_at"] == start.timestamp()
        assert query["end_at"] == end.timestamp()
        assert query["protocol"]["status"] == "valid"
        assert query["protocol"]["formats"]["start_at"] == expected_format
        assert query["protocol"]["range_semantics"] == "[start_at,end_at)"


def test_ppd_temporal_protocol_rejects_invalid_or_ambiguous_ranges() -> None:
    cases = [
        (True, 1788307200, "", "boolean_timestamp"),
        (float("inf"), 1788307200, "", "non_finite_timestamp"),
        (None, 1788307200, "", "missing_temporal_bound"),
        ("not-a-timestamp", 1788307200, "", "unsupported_timestamp_format"),
        ("2026-09-01T00:00:00", "2026-09-02T00:00:00", "", "timezone_required_for_naive_iso8601"),
        (1788307200, 1788307200, "", "end_at_must_be_after_start_at"),
    ]

    for raw_start, raw_end, timezone_name, expected_error in cases:
        decision = _decision_from_payload(
            {
                "temporal_query": {
                    "has_expression": True,
                    "start_at": raw_start,
                    "end_at": raw_end,
                    "timezone": timezone_name,
                },
                "confidence": 0.95,
            },
            raw="{}",
            backend="llm",
        )
        plan = TurnPlan().apply_pre_reply_decision(decision)

        assert decision.temporal_query["protocol"]["status"] == "invalid"
        assert decision.temporal_query["protocol"]["error"] == expected_error
        assert plan.temporal_scope.usable_range is False
        assert plan.temporal_scope.error == expected_error


def test_equivalent_ppd_epoch_and_iso_ranges_produce_the_same_execution_scope() -> None:
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    queries = [
        {"has_expression": True, "start_at": start.timestamp(), "end_at": end.timestamp()},
        {"has_expression": True, "start_at": "2026-09-01T00:00:00Z", "end_at": "2026-09-02T00:00:00Z"},
    ]

    scopes = [
        TurnPlan().apply_pre_reply_decision(
            _decision_from_payload({"temporal_query": query, "confidence": 0.95}, raw="{}", backend="llm")
        ).temporal_scope
        for query in queries
    ]

    assert all(scope.usable_range for scope in scopes)
    assert (scopes[0].start_at, scopes[0].end_at) == (scopes[1].start_at, scopes[1].end_at)


class CapturingClassifierAgent:
    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    def run_conversation(self, message: str, **kwargs: object) -> dict[str, str]:
        self.calls.append({"message": message, **kwargs})
        return {"final_response": json.dumps(self.decision)}


def test_read_only_recall_survives_low_confidence() -> None:
    decision = _decision_from_payload(
        {
            "memory_action": "recall",
            "memory_recall_type": "event",
            "needs_event_memory": True,
            "recall_goal": "specific_fact",
            "event_recall_strategy": "text_search",
            "confidence": 0.7,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.error == ""
    assert decision.needs_event_memory is True
    assert decision.recall_goal == "specific_fact"
    assert "confidence_below_threshold_read_only_recall_preserved" in decision.warnings


def test_invalid_unrelated_fields_do_not_discard_valid_recall() -> None:
    decision = _decision_from_payload(
        {
            "memory_action": "recall",
            "needs_event_memory": True,
            "recall_goal": "specific_fact",
            "event_recall_strategy": "text_search",
            "reply_mode": "answer_from_memory",
            "answer_source": "memory",
            "scope": "personal",
            "memory_recall_type": "event_lookup",
            "confidence": 0.95,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.error == ""
    assert decision.needs_event_memory is True
    assert decision.memory_recall_type == "event"
    assert any(item.startswith("invalid_reply_mode") for item in decision.warnings)


def test_malformed_mixed_profile_recall_recovers_existing_summary_contract() -> None:
    decision = _decision_from_payload(
        {
            "turn_intent": "mixed",
            "reply_mode": "llm",
            "memory_action": "none|write|recall|correction|explain",
            "memory_recall_type": "none|profile|event|timeline|observation",
            "recall_goal": "none|summary|raw_evidence|specific_fact",
            "needs_profile_memory": True,
            "needs_event_memory": False,
            "needs_timeline_recall": False,
            "needs_discussion_recall": False,
            "recall_subject_scope": "self",
            "recall_subject_names": [],
            "confidence": 0.95,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.memory_action == "recall"
    assert decision.memory_recall_type == "profile"
    assert decision.recall_goal == "summary"
    assert "recovered_malformed_tailored_profile_recall" in decision.warnings


def test_explicit_no_recall_is_not_recovered_as_profile_summary() -> None:
    decision = _decision_from_payload(
        {
            "turn_intent": "chat",
            "reply_mode": "llm",
            "memory_action": "none",
            "memory_recall_type": "none",
            "recall_goal": "none",
            "needs_profile_memory": False,
            "confidence": 0.95,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.memory_action == "none"
    assert decision.memory_recall_type == "none"
    assert decision.recall_goal == "none"
    assert "recovered_malformed_tailored_profile_recall" not in decision.warnings


def test_classifier_contract_keeps_tailored_advice_as_profile_context_for_reply() -> None:
    agent = CapturingClassifierAgent({
        "turn_intent": "mixed",
        "memory_action": "recall",
        "memory_recall_type": "profile",
        "needs_profile_memory": True,
        "recall_goal": "summary",
        "confidence": 0.95,
    })

    decision = classify_pre_reply_decision(
        agent,
        "Based on the work habits I have shared, what focus routine would suit me?",
    )

    assert decision.turn_intent == "mixed"
    assert decision.needs_profile_memory is True
    assert decision.needs_event_memory is False
    assert decision.needs_timeline_recall is False
    assert decision.memory_recall_type == "profile"
    assert decision.recall_goal == "summary"
    classifier_prompt = str(agent.calls[0]["message"])
    assert "Personal ongoing context" in classifier_prompt
    assert "Generic/factual advice" in classifier_prompt
    assert "Current local recommendation" in classifier_prompt
    assert "Prior conversation source" in classifier_prompt
    assert "memory_recall_type=timeline" in classifier_prompt
    assert "generic advice not tailored to the user's own existing preferences" in classifier_prompt
    assert "event_recall_strategy=text_search" in classifier_prompt
    assert "Time distinction" in classifier_prompt
    assert "quantity-only request needs count_scope and does not need entities" in classifier_prompt
    assert '"How many houseplants do I have?" -> count_scope' in classifier_prompt
    assert '"How many houseplants do I have, and what are they?" -> count_scope + entities' in classifier_prompt
    assert '"How many cameras do I own, and what models are they?" -> count_scope + entities + qualifiers' in classifier_prompt
    assert '"Which courses did I take, and how long was each one?" -> entities + temporal_relation' in classifier_prompt
    assert "must not replace requested entities or qualifiers" in classifier_prompt
    shape_prefix = classifier_prompt.split("Rules:", 1)[0]
    assert '"answer_intent"' in shape_prefix
    assert '"answer_focus"' in shape_prefix
    assert '"answer_obligations"' in shape_prefix
    assert '"uncertainty_policy"' in shape_prefix


def test_classifier_contract_opens_recall_for_advice_building_on_owned_items() -> None:
    """Advice grounded in user-owned items/experiments must open bounded recall."""
    agent = CapturingClassifierAgent({
        "turn_intent": "mixed",
        "memory_action": "recall",
        "memory_recall_type": "profile",
        "needs_profile_memory": True,
        "needs_event_memory": True,
        "event_recall_strategy": "text_search",
        "recall_goal": "summary",
        "confidence": 0.95,
    })

    decision = classify_pre_reply_decision(
        agent,
        "My kitchen's becoming a bit of a mess again. Any tips for keeping it clean?",
    )

    assert decision.needs_profile_memory is True
    assert decision.needs_event_memory is True
    classifier_prompt = str(agent.calls[0]["message"])
    assert "Advice building on user-owned items/experiments" in classifier_prompt
    assert "slow cooker" in classifier_prompt
    assert "utensil holder" in classifier_prompt
    assert "portable power bank" in classifier_prompt
    # The negative example anchor must exist so generic advice stays none.
    assert "no user-specific anchor" in classifier_prompt


class FlakyClassifierAgent:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def run_conversation(self, message: str, **kwargs: object) -> dict[str, str]:
        self.calls.append({"message": message, **kwargs})
        raw = self.responses.pop(0) if self.responses else ""
        return {"final_response": raw}


def test_ppd_retries_on_empty_response_then_succeeds() -> None:
    valid = {
        "turn_intent": "mixed",
        "memory_action": "recall",
        "memory_recall_type": "profile",
        "needs_profile_memory": True,
        "needs_event_memory": True,
        "event_recall_strategy": "text_search",
        "recall_goal": "summary",
        "confidence": 0.9,
    }
    agent = FlakyClassifierAgent(["", json.dumps(valid)])

    decision = classify_pre_reply_decision(agent, "Can you recommend a show for me tonight?")

    assert len(agent.calls) == 2
    assert decision.backend == "llm"
    assert decision.memory_recall_type == "profile"
    assert decision.error == ""
    assert any(str(warning).startswith("ppd_retry_used:") for warning in decision.warnings)


def test_ppd_retries_on_malformed_json_then_succeeds() -> None:
    agent = FlakyClassifierAgent(["not json at all", json.dumps({"memory_action": "none", "confidence": 0.8})])

    decision = classify_pre_reply_decision(agent, "hello")

    assert len(agent.calls) == 2
    assert decision.backend == "llm"
    assert decision.memory_action == "none"


def test_ppd_falls_back_after_two_failures() -> None:
    agent = FlakyClassifierAgent(["", ""])

    decision = classify_pre_reply_decision(agent, "hello")

    assert len(agent.calls) == 2
    assert decision.backend == "unavailable"
    assert decision.memory_action == "none"
    assert decision.error
    assert any(str(warning).startswith("ppd_retry_used:") for warning in decision.warnings)


def test_ppd_does_not_retry_valid_decision() -> None:
    agent = FlakyClassifierAgent([json.dumps({"memory_action": "none", "confidence": 0.9})])

    decision = classify_pre_reply_decision(agent, "hello")

    assert len(agent.calls) == 1
    assert decision.backend == "llm"
    assert all(not str(warning).startswith("ppd_retry_used:") for warning in decision.warnings)


def test_classifier_prompt_contains_first_person_recommendation_rule() -> None:
    agent = CapturingClassifierAgent({
        "memory_action": "recall",
        "memory_recall_type": "profile",
        "needs_profile_memory": True,
        "needs_event_memory": True,
        "event_recall_strategy": "text_search",
        "answer_intent": "personalized_recommendation",
        "confidence": 0.95,
    })

    classify_pre_reply_decision(
        agent,
        "Can you suggest some activities I can do this weekend?",
        recent_context_capsule="I enjoy hiking and avoid crowded places.",
    )

    prompt = str(agent.calls[0]["message"])
    assert "trusted evidence of the user's recent context" in prompt
    assert "First-person recommendation without a named item" in prompt
    assert "does NOT need to name a specific owned item" in prompt
    assert "this weekend" in prompt
    assert "I enjoy hiking and avoid crowded places" in prompt
    assert "What are some general ways to relax?" in prompt


def test_classifier_catalog_routes_named_discussion_without_becoming_evidence() -> None:
    agent = CapturingClassifierAgent({
        "turn_intent": "memory_recall",
        "memory_action": "recall",
        "needs_discussion_recall": True,
        "discussion_query": "沃尔玛会议",
        "recall_goal": "summary",
        "confidence": 0.95,
    })

    decision = classify_pre_reply_decision(
        agent,
        "沃尔玛高层会议讲了什么？",
        discussion_catalog_context=(
            "Available local discussion archive topics (routing metadata only):\n"
            "1. 2026-08-27 · Walmart Executive Meeting on Retail Strategy"
        ),
    )

    assert decision.needs_discussion_recall is True
    prompt = str(agent.calls[0]["message"])
    assert "Walmart Executive Meeting on Retail Strategy" in prompt
    assert "route discovery only, not factual answer evidence" in prompt
    assert "not an uploaded/external-document request" in prompt
    assert "copy that catalog entry's title exactly into discussion_query" in prompt


def test_classifier_prompt_contains_first_person_travel_rule() -> None:
    agent = CapturingClassifierAgent({
        "memory_action": "recall",
        "memory_recall_type": "profile",
        "needs_profile_memory": True,
        "needs_event_memory": True,
        "event_recall_strategy": "text_search",
        "answer_intent": "personalized_recommendation",
        "confidence": 0.95,
    })

    classify_pre_reply_decision(
        agent,
        "I'm a bit nervous about getting around Barcelona. Any tips?",
        recent_context_capsule=(
            "Query-relevant stored memories (database text search):\n"
            "1. 我 · profile/preference · bought a transit pass and a route-planning app for the trip."
        ),
    )

    prompt = str(agent.calls[0]["message"])
    assert "First-person travel/commute/itinerary tips" in prompt
    assert "getting around" in prompt
    assert "transit pass" in prompt
    assert "memory_recall_type=none" in prompt
    assert "query-relevant stored memories" in prompt


def test_classifier_prompt_mentions_query_relevant_memory_evidence() -> None:
    agent = CapturingClassifierAgent({
        "memory_action": "recall",
        "memory_recall_type": "profile",
        "needs_profile_memory": True,
        "needs_event_memory": True,
        "event_recall_strategy": "text_search",
        "answer_intent": "personalized_recommendation",
        "answer_obligations": ["negation_constraints", "incremental_next_step"],
        "confidence": 0.95,
    })

    classify_pre_reply_decision(
        agent,
        "Can you suggest some activities I can do this weekend?",
        recent_context_capsule="Query-relevant stored memories (database text search):\n1. hiking",
    )

    prompt = str(agent.calls[0]["message"])
    assert "query-relevant stored memories returned by database text search" in prompt
    assert "for first-person advice, tips, or recommendation requests, that relevance is a valid reason" in prompt


def test_personalized_recommendation_intent_survives_normalization() -> None:
    decision = _decision_from_payload(
        {
            "memory_action": "recall",
            "memory_recall_type": "profile",
            "needs_profile_memory": True,
            "needs_event_memory": True,
            "event_recall_strategy": "text_search",
            "answer_intent": "personalized_recommendation",
            "answer_obligations": ["qualifiers", "negation_constraints"],
            "confidence": 0.9,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.answer_intent == "personalized_recommendation"
    assert decision.answer_obligations == ["qualifiers", "negation_constraints"]
    assert decision.needs_profile_memory is True
    assert decision.needs_event_memory is True


def test_low_confidence_non_recall_downgrades_personalized_recommendation() -> None:
    decision = _decision_from_payload(
        {
            "memory_action": "none",
            "memory_recall_type": "none",
            "answer_intent": "personalized_recommendation",
            "confidence": 0.5,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.answer_intent == "direct_answer"
    assert any(warning == "confidence_below_threshold" for warning in decision.warnings)


def test_low_confidence_read_only_recall_preserves_personalized_intent() -> None:
    decision = _decision_from_payload(
        {
            "memory_action": "recall",
            "memory_recall_type": "profile",
            "needs_profile_memory": True,
            "needs_event_memory": True,
            "event_recall_strategy": "text_search",
            "answer_intent": "personalized_recommendation",
            "confidence": 0.5,
        },
        raw="{}",
        backend="llm",
    )

    assert decision.answer_intent == "personalized_recommendation"
    assert any(
        warning == "confidence_below_threshold_read_only_recall_preserved"
        for warning in decision.warnings
    )
