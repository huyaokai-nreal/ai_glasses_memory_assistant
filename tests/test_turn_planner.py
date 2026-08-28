from ai_glasses_memory_assistant.turn_planner import TurnPlan, coverage_requirement_for_answer
from ai_glasses_memory_assistant.turn_semantic_classifier import PreReplyDecision


def test_complete_set_is_derived_only_from_structured_answer_contract() -> None:
    decision = PreReplyDecision(
        memory_action="recall",
        needs_event_memory=True,
        memory_recall_type="event",
        event_recall_strategy="text_search",
        answer_intent="multi_fact",
        answer_focus="all purchases in the requested scope",
        answer_obligations=["entities", "count_scope"],
        uncertainty_policy="abstain_if_insufficient",
        coverage_requirement="complete_set",
    )

    plan = TurnPlan().apply_pre_reply_decision(decision)

    assert plan.answer_intent == decision.answer_intent
    assert plan.answer_focus == decision.answer_focus
    assert plan.answer_obligations == decision.answer_obligations
    assert plan.uncertainty_policy == decision.uncertainty_policy
    assert plan.coverage_requirement == "complete_set"
    assert plan.debug_payload()["coverage_requirement"] == "complete_set"


def test_complete_set_mapping_uses_no_user_text() -> None:
    assert coverage_requirement_for_answer("direct_fact", ["count_scope"]) == "best_evidence"
    assert coverage_requirement_for_answer("count_or_total", []) == "best_evidence"
    assert coverage_requirement_for_answer("count_or_total", ["count_scope"]) == "complete_set"
    assert coverage_requirement_for_answer("multi_fact", ["count_scope"]) == "complete_set"


def test_answer_obligation_combination_and_order_survive_turn_plan() -> None:
    decision = PreReplyDecision(
        answer_intent="count_or_total",
        answer_focus="count the cameras, list them, and include each supported model",
        answer_obligations=["count_scope", "entities", "qualifiers", "temporal_relation"],
        uncertainty_policy="abstain_if_insufficient",
        coverage_requirement="complete_set",
    )

    plan = TurnPlan().apply_pre_reply_decision(decision)

    assert plan.answer_focus == decision.answer_focus
    assert plan.answer_obligations == decision.answer_obligations
    assert plan.uncertainty_policy == decision.uncertainty_policy
    assert plan.coverage_requirement == "complete_set"


def test_generic_decision_keeps_best_evidence_and_no_memory_route() -> None:
    plan = TurnPlan().apply_pre_reply_decision(
        PreReplyDecision(
            memory_action="none",
            memory_recall_type="none",
            answer_intent="direct_fact",
            answer_obligations=[],
        )
    )

    assert plan.coverage_requirement == "best_evidence"
    assert plan.needs_profile_memory is False
    assert plan.needs_event_memory is False
    assert plan.needs_timeline_recall is False
