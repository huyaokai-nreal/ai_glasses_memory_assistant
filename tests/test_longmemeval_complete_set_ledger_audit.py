from __future__ import annotations

from ai_glasses_memory_assistant.evals import longmemeval_complete_set_ledger_audit as audit


def _detail(*, source_ids: list[str], memories: list[dict[str, object]], timeline: list[dict[str, object]], items: list[dict[str, object]], accounted: int) -> dict[str, object]:
    return {
        "question_type": "multi-session",
        "recalled_memories": memories,
        "recalled_timeline_chunks": timeline,
        "reader_debug": {
            "reader_status": "answered",
            "ledger_valid": True,
            "ledger_value": "2",
            "ledger_operation": "count",
            "ledger_accounted_source_count": accounted,
            "answer_task": {"coverage_requirement": "complete_set", "source_ids": source_ids},
            "ledger_items": items,
        },
    }


def test_audit_detail_proves_multiline_source_input_is_reversible() -> None:
    detail = _detail(
        source_ids=["memory:m1", "timeline:t1"],
        memories=[{"id": "m1", "content": "first line\nsecond line"}],
        timeline=[{"id": "t1", "text": "timeline fact"}],
        items=[{"source_ids": ["memory:m1", "timeline:t1"]}],
        accounted=2,
    )

    row = audit.audit_detail("case-1", detail)

    assert row is not None
    assert row["reader_input_reversible"] is True
    assert row["source_inventory"][0]["multiline"] is True
    assert len(row["source_inventory"][0]["text_sha256"]) == 64


def test_audit_detail_rejects_missing_or_unknown_source_provenance() -> None:
    detail = _detail(
        source_ids=["memory:m1", "timeline:t1"],
        memories=[{"id": "m1", "content": "known fact"}],
        timeline=[],
        items=[{"source_ids": ["memory:m1", "timeline:unknown"]}],
        accounted=2,
    )

    row = audit.audit_detail("case-2", detail)

    assert row is not None
    assert row["reader_input_reversible"] is False
    assert row["missing_declared_sources"] == ["timeline:t1"]
    assert row["unknown_ledger_item_sources"] == ["timeline:unknown"]


def test_audit_detail_ignores_non_complete_or_invalid_ledgers() -> None:
    detail = _detail(
        source_ids=["memory:m1"],
        memories=[{"id": "m1", "content": "fact"}],
        timeline=[],
        items=[{"source_ids": ["memory:m1"]}],
        accounted=1,
    )
    detail["reader_debug"]["ledger_valid"] = False  # type: ignore[index]

    assert audit.audit_detail("case-3", detail) is None
