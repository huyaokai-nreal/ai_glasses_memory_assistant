from decimal import Decimal

from ai_glasses_memory_assistant.evidence_set import (
    EvidenceCandidate,
    EvidenceSourceStats,
    build_evidence_set,
    decode_page_cursor,
    encode_page_cursor,
    expand_consolidated_ledger,
    validate_aggregation_ledger,
    batch_evidence_candidates,
)


def test_page_cursor_round_trips_and_rejects_invalid_values() -> None:
    cursor = encode_page_cursor(12.5, "source-2")

    assert decode_page_cursor(cursor) == (12.5, "source-2")
    assert decode_page_cursor("not-json") is None
    assert decode_page_cursor('[1]') is None


def test_rrf_orders_complete_set_without_deleting_unranked_candidates() -> None:
    candidates = [
        EvidenceCandidate(source_id="a", source_type="memory", text="A"),
        EvidenceCandidate(source_id="b", source_type="timeline", text="B"),
        EvidenceCandidate(source_id="c", source_type="timeline", text="C"),
    ]
    evidence = build_evidence_set(
        candidates,
        source_stats={
            "memory": EvidenceSourceStats(scanned=1, candidate=1, selected=1, source_exhausted=True),
            "timeline": EvidenceSourceStats(scanned=2, candidate=2, selected=2, source_exhausted=True),
        },
        rankings={"semantic": ["b", "a"], "temporal": ["a", "b"]},
    )

    assert {item.source_id for item in evidence.candidates} == {"a", "b", "c"}
    assert evidence.candidates[-1].source_id == "c"
    assert evidence.coverage_complete is True
    assert evidence.truncated is False


def test_unexhausted_or_truncated_source_prevents_complete_coverage() -> None:
    evidence = build_evidence_set(
        [EvidenceCandidate(source_id="a", source_type="memory", text="A")],
        source_stats={
            "memory": EvidenceSourceStats(scanned=1, candidate=1, selected=1, source_exhausted=False),
        },
        truncation_reason="reader_batch_budget",
    )

    assert evidence.coverage_complete is False
    assert evidence.truncated is True
    assert evidence.truncation_reason == "reader_batch_budget"


def test_ledger_accounts_every_source_deduplicates_and_recomputes_decimal_sum() -> None:
    candidates = [
        EvidenceCandidate(source_id="s1", source_type="timeline", text="same item", occurred_at=1.0),
        EvidenceCandidate(source_id="s2", source_type="timeline", text="same item again", occurred_at=2.0),
        EvidenceCandidate(source_id="s3", source_type="timeline", text="second item", occurred_at=3.0),
    ]
    validation = validate_aggregation_ledger(
        {
            "items": [
                {
                    "canonical_key": "item-a",
                    "label": "Item A",
                    "quantity": "1.20",
                    "unit": "USD",
                    "status": "included",
                    "source_ids": ["s1", "s2"],
                },
                {
                    "canonical_key": "item-b",
                    "label": "Item B",
                    "quantity": "0.80",
                    "unit": "USD",
                    "status": "included",
                    "source_ids": ["s3"],
                },
            ],
            "aggregation": {"operation": "sum", "value": "2.00", "unit": "USD"},
            "final_answer": "The verified total is 2.00 USD.",
        },
        candidates,
    )

    assert validation.valid is True
    assert str(validation.value) == "2.00"
    assert validation.accounted_source_ids == ("s1", "s2", "s3")


def test_ledger_uses_latest_status_for_same_canonical_item() -> None:
    candidates = [
        EvidenceCandidate(source_id="old", source_type="memory", text="planned", occurred_at=1.0),
        EvidenceCandidate(source_id="new", source_type="memory", text="cancelled", occurred_at=2.0),
    ]
    validation = validate_aggregation_ledger(
        {
            "items": [
                {
                    "canonical_key": "task-a",
                    "label": "Task A",
                    "quantity": "1",
                    "unit": "item",
                    "status": "included",
                    "source_ids": ["old"],
                },
                {
                    "canonical_key": "task-a",
                    "label": "Task A cancelled",
                    "quantity": "1",
                    "unit": "item",
                    "status": "excluded",
                    "source_ids": ["new"],
                },
            ],
            "aggregation": {"operation": "count", "value": "0", "unit": "item"},
            "final_answer": "There are 0 active items.",
        },
        candidates,
    )

    assert validation.valid is True
    assert validation.value == 0
    assert validation.items[0]["status"] == "excluded"
    assert validation.items[0]["source_ids"] == ["old", "new"]


def test_ledger_rejects_missing_duplicate_superseded_and_wrong_total() -> None:
    candidates = [
        EvidenceCandidate(source_id="s1", source_type="memory", text="old", superseded_by="m2"),
        EvidenceCandidate(source_id="s2", source_type="memory", text="new"),
    ]
    validation = validate_aggregation_ledger(
        {
            "items": [{
                "canonical_key": "item-a",
                "label": "Item A",
                "quantity": "1",
                "unit": "item",
                "status": "included",
                "source_ids": ["s1"],
            }],
            "aggregation": {"operation": "count", "value": "2", "unit": "item"},
            "final_answer": "There are 2 items.",
        },
        candidates,
    )

    assert validation.valid is False
    assert "item_0_includes_superseded_source" in validation.errors
    assert "unaccounted_source_ids" in validation.errors
    assert "aggregation_value_mismatch" in validation.errors
    assert "final_answer_missing_verified_value" in validation.errors


def test_reader_batch_budget_marks_complete_set_incomplete() -> None:
    candidates = [
        EvidenceCandidate(
            source_id=f"s{index}",
            source_type="timeline",
            text="x" * 900,
        )
        for index in range(12)
    ]

    batches, truncated = batch_evidence_candidates(
        candidates,
        max_context_chars=1_000,
        max_batches=9,
    )

    assert len(batches) == 9
    assert truncated is True


def test_reader_batches_keep_linked_memory_and_timeline_together() -> None:
    candidates = [
        EvidenceCandidate(
            source_id="memory:m1",
            source_type="structured_memory",
            text="derived fact",
            evidence_ids=("chunk-1",),
        ),
        EvidenceCandidate(
            source_id="memory:m2",
            source_type="structured_memory",
            text="another fact",
            evidence_ids=("chunk-2",),
        ),
        EvidenceCandidate(
            source_id="timeline:chunk-1",
            source_type="timeline",
            text="raw source one",
            evidence_ids=("chunk-1",),
        ),
        EvidenceCandidate(
            source_id="timeline:chunk-2",
            source_type="timeline",
            text="raw source two",
            evidence_ids=("chunk-2",),
        ),
    ]

    batches, truncated = batch_evidence_candidates(
        candidates,
        max_context_chars=100_000,
        max_candidates_per_batch=2,
    )

    assert truncated is False
    assert [candidate.source_id for candidate in batches[0]] == ["memory:m1", "timeline:chunk-1"]
    assert [candidate.source_id for candidate in batches[1]] == ["memory:m2", "timeline:chunk-2"]


def test_excluded_source_quantity_is_not_part_of_decimal_aggregation() -> None:
    candidates = [EvidenceCandidate(source_id="s1", source_type="timeline", text="irrelevant list")]

    validation = validate_aggregation_ledger(
        {
            "items": [{
                "canonical_key": "excluded:s1",
                "label": "out-of-scope details",
                "quantity": "many",
                "unit": "item",
                "status": "excluded",
                "source_ids": ["s1"],
            }],
            "aggregation": {"operation": "count", "value": "0", "unit": "item"},
            "final_answer": "0 items.",
        },
        candidates,
    )

    assert validation.valid is True
    assert validation.value == Decimal("0")


def test_compact_consolidation_source_lists_expand_to_standard_ledger_items() -> None:
    expanded = expand_consolidated_ledger({
        "items": [{
            "canonical_key": "item-a",
            "quantity": "1",
            "status": "included",
            "source_ids": ["s1"],
        }],
        "excluded_source_ids": ["s2"],
        "uncertain_source_ids": ["s3"],
        "aggregation": {"operation": "count", "value": "1", "unit": "item"},
        "final_answer": "1 item.",
    })

    validation = validate_aggregation_ledger(
        expanded,
        [
            EvidenceCandidate(source_id="s1", source_type="timeline", text="included"),
            EvidenceCandidate(source_id="s2", source_type="timeline", text="excluded"),
            EvidenceCandidate(source_id="s3", source_type="timeline", text="uncertain"),
        ],
    )

    assert validation.valid is True
    assert validation.accounted_source_ids == ("s1", "s2", "s3")


def test_ledger_normalizes_included_source_over_its_background_exclusion() -> None:
    candidates = [EvidenceCandidate(source_id="s1", source_type="timeline", text="fact plus background")]

    validation = validate_aggregation_ledger(
        {
            "items": [
                {
                    "canonical_key": "task-a",
                    "label": "Task A",
                    "quantity": "1",
                    "unit": "item",
                    "status": "included",
                    "source_ids": ["s1"],
                },
                {
                    "canonical_key": "background",
                    "label": "unrelated background",
                    "quantity": "0",
                    "unit": "item",
                    "status": "excluded",
                    "source_ids": ["s1"],
                },
            ],
            "aggregation": {"operation": "count", "value": "1", "unit": "item"},
            "final_answer": "1 item.",
        },
        candidates,
    )

    assert validation.valid is True
    assert validation.accounted_source_ids == ("s1",)


def test_ledger_rejects_one_source_assigned_to_two_included_canonical_items() -> None:
    candidates = [EvidenceCandidate(source_id="s1", source_type="timeline", text="ambiguous list")]

    validation = validate_aggregation_ledger(
        {
            "items": [
                {
                    "canonical_key": "item-a",
                    "quantity": "1",
                    "status": "included",
                    "source_ids": ["s1"],
                },
                {
                    "canonical_key": "item-b",
                    "quantity": "1",
                    "status": "included",
                    "source_ids": ["s1"],
                },
            ],
            "aggregation": {"operation": "count", "value": "2", "unit": "item"},
            "final_answer": "2 items.",
        },
        candidates,
    )

    assert validation.valid is False
    assert "item_1_duplicate_source_ids" in validation.errors

    provisional = validate_aggregation_ledger(
        {
            "items": [
                {
                    "canonical_key": "item-a",
                    "quantity": "1",
                    "status": "included",
                    "source_ids": ["s1"],
                },
                {
                    "canonical_key": "item-b",
                    "quantity": "1",
                    "status": "included",
                    "source_ids": ["s1"],
                },
            ],
            "aggregation": {"operation": "count", "value": "2", "unit": "item"},
            "final_answer": "2 provisional items.",
        },
        candidates,
        allow_duplicate_source_ids=True,
    )
    assert provisional.valid is True
