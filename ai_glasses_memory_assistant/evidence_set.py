from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class PagedSourceResult(Generic[T]):
    items: list[T]
    next_cursor: str | None
    scanned_count: int
    exhausted: bool


def encode_page_cursor(sort_value: float, source_id: str) -> str:
    return json.dumps([float(sort_value), str(source_id)], separators=(",", ":"))


def decode_page_cursor(cursor: str | None) -> tuple[float, str] | None:
    if not cursor:
        return None
    try:
        payload = json.loads(cursor)
        if not isinstance(payload, list) or len(payload) != 2:
            return None
        return float(payload[0]), str(payload[1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class EvidenceCandidate:
    source_id: str
    source_type: str
    text: str
    occurred_at: float | None = None
    recorded_at: float | None = None
    memory_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    status: str = "active"
    superseded_by: str = ""
    ranking_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceSourceStats:
    scanned: int = 0
    candidate: int = 0
    selected: int = 0
    source_exhausted: bool = False
    truncated: bool = False

    def debug_payload(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "candidate": self.candidate,
            "selected": self.selected,
            "source_exhausted": self.source_exhausted,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class EvidenceSet:
    candidates: list[EvidenceCandidate] = field(default_factory=list)
    source_stats: dict[str, EvidenceSourceStats] = field(default_factory=dict)
    coverage_complete: bool = False
    truncated: bool = False
    truncation_reason: str = ""

    def debug_payload(self) -> dict[str, object]:
        return {
            "candidate_count": len(self.candidates),
            "source_ids": [candidate.source_id for candidate in self.candidates],
            "source_stats": {
                name: stats.debug_payload()
                for name, stats in self.source_stats.items()
            },
            "coverage_complete": self.coverage_complete,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }


@dataclass(frozen=True)
class LedgerValidation:
    valid: bool
    items: list[dict[str, object]] = field(default_factory=list)
    operation: str = ""
    value: Decimal | None = None
    unit: str = ""
    final_answer: str = ""
    accounted_source_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def debug_payload(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "item_count": len(self.items),
            "operation": self.operation,
            "value": str(self.value) if self.value is not None else "",
            "unit": self.unit,
            "final_answer": self.final_answer,
            "accounted_source_ids": list(self.accounted_source_ids),
            "errors": list(self.errors),
        }


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    *,
    rank_constant: int = 60,
) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for ranking_name, source_ids in rankings.items():
        for rank, source_id in enumerate(dict.fromkeys(source_ids), start=1):
            scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (rank_constant + rank)
            sources.setdefault(source_id, []).append(ranking_name)
    ordered = sorted(scores, key=lambda source_id: (-scores[source_id], source_id))
    return ordered, {source_id: tuple(names) for source_id, names in sources.items()}


def build_evidence_set(
    candidates: Iterable[EvidenceCandidate],
    *,
    source_stats: dict[str, EvidenceSourceStats],
    rankings: dict[str, list[str]] | None = None,
    truncation_reason: str = "",
) -> EvidenceSet:
    by_id: dict[str, EvidenceCandidate] = {}
    for candidate in candidates:
        if not candidate.source_id or candidate.source_id in by_id:
            continue
        by_id[candidate.source_id] = candidate

    ordered_ids, ranking_sources = reciprocal_rank_fusion(rankings or {})
    ordered_ids.extend(source_id for source_id in by_id if source_id not in ranking_sources)
    ordered_candidates = []
    for source_id in ordered_ids:
        candidate = by_id.get(source_id)
        if candidate is None:
            continue
        merged_sources = tuple(dict.fromkeys((*candidate.ranking_sources, *ranking_sources.get(source_id, ()))))
        ordered_candidates.append(EvidenceCandidate(
            source_id=candidate.source_id,
            source_type=candidate.source_type,
            text=candidate.text,
            occurred_at=candidate.occurred_at,
            recorded_at=candidate.recorded_at,
            memory_id=candidate.memory_id,
            evidence_ids=candidate.evidence_ids,
            status=candidate.status,
            superseded_by=candidate.superseded_by,
            ranking_sources=merged_sources,
        ))

    truncated = bool(truncation_reason) or any(stats.truncated for stats in source_stats.values())
    coverage_complete = bool(source_stats) and not truncated and all(
        stats.source_exhausted for stats in source_stats.values()
    )
    return EvidenceSet(
        candidates=ordered_candidates,
        source_stats=dict(source_stats),
        coverage_complete=coverage_complete,
        truncated=truncated,
        truncation_reason=truncation_reason,
    )


def validate_aggregation_ledger(
    payload: object,
    candidates: Iterable[EvidenceCandidate],
    *,
    require_final_answer: bool = True,
    allow_duplicate_source_ids: bool = False,
) -> LedgerValidation:
    source_map = {
        candidate.source_id: candidate
        for candidate in candidates
        if candidate.source_id
    }
    errors: list[str] = []
    if not isinstance(payload, dict):
        return LedgerValidation(valid=False, errors=("ledger_not_object",))
    payload = normalize_ledger_source_assignments(payload)
    raw_items = payload.get("items")
    aggregation = payload.get("aggregation")
    if not isinstance(raw_items, list):
        errors.append("items_not_list")
        raw_items = []
    if not isinstance(aggregation, dict):
        errors.append("aggregation_not_object")
        aggregation = {}

    accounted: set[str] = set()
    normalized_items: list[dict[str, object]] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            errors.append(f"item_{index}_not_object")
            continue
        canonical_key = str(raw_item.get("canonical_key") or raw_item.get("canonical_item") or "").strip()
        label = str(raw_item.get("label") or canonical_key).strip()
        status = str(raw_item.get("status") or "").strip().lower()
        unit = str(raw_item.get("unit") or "item").strip() or "item"
        source_ids = list(dict.fromkeys(
            str(source_id).strip()
            for source_id in (raw_item.get("source_ids") or [])
            if str(source_id).strip()
        ))
        if not canonical_key:
            errors.append(f"item_{index}_missing_canonical_key")
        if status not in {"included", "excluded", "uncertain"}:
            errors.append(f"item_{index}_invalid_status")
        if not source_ids:
            errors.append(f"item_{index}_missing_source_ids")
        unknown_ids = [source_id for source_id in source_ids if source_id not in source_map]
        if unknown_ids:
            errors.append(f"item_{index}_unknown_source_ids")
        duplicate_ids = [source_id for source_id in source_ids if source_id in accounted]
        if duplicate_ids and not allow_duplicate_source_ids:
            errors.append(f"item_{index}_duplicate_source_ids")
        accounted.update(source_ids)
        try:
            quantity = Decimal(str(raw_item.get("quantity", "1")).replace(",", ""))
        except (InvalidOperation, ValueError):
            quantity = Decimal("0")
            if status == "included":
                errors.append(f"item_{index}_invalid_quantity")
        if status == "included" and quantity < 0:
            errors.append(f"item_{index}_negative_quantity")
        if status == "included" and any(
            source_map[source_id].superseded_by
            for source_id in source_ids
            if source_id in source_map
        ):
            errors.append(f"item_{index}_includes_superseded_source")
        normalized_items.append({
            "canonical_key": canonical_key,
            "label": label,
            "quantity": quantity,
            "unit": unit,
            "status": status,
            "source_ids": source_ids,
        })

    missing_ids = sorted(set(source_map) - accounted)
    if missing_ids:
        errors.append("unaccounted_source_ids")

    latest_by_key: dict[str, dict[str, object]] = {}
    for item in normalized_items:
        key = str(item["canonical_key"]).strip().casefold()
        if not key:
            continue
        source_ids = list(item["source_ids"])
        item_time = max(
            (
                source_map[source_id].occurred_at
                or source_map[source_id].recorded_at
                or 0.0
                for source_id in source_ids
                if source_id in source_map
            ),
            default=0.0,
        )
        existing = latest_by_key.get(key)
        if existing is None:
            latest_by_key[key] = {**item, "_time": item_time}
            continue
        merged_source_ids = list(dict.fromkeys([
            *list(existing["source_ids"]),
            *source_ids,
        ]))
        if item_time >= float(existing["_time"]):
            latest_by_key[key] = {**item, "source_ids": merged_source_ids, "_time": item_time}
        else:
            existing["source_ids"] = merged_source_ids

    included = [item for item in latest_by_key.values() if item["status"] == "included"]
    operation = str(aggregation.get("operation") or "").strip().lower()
    if operation not in {"count", "sum"}:
        errors.append("invalid_aggregation_operation")
    units = {str(item["unit"]) for item in included}
    if len(units) > 1:
        errors.append("mixed_included_units")
    unit = str(aggregation.get("unit") or (next(iter(units)) if units else "item")).strip() or "item"
    computed = sum((item["quantity"] for item in included), Decimal("0"))
    try:
        claimed = Decimal(str(aggregation.get("value", "")).replace(",", ""))
    except (InvalidOperation, ValueError):
        claimed = None
        errors.append("invalid_aggregation_value")
    if claimed is not None and claimed != computed:
        errors.append("aggregation_value_mismatch")
    final_answer = str(payload.get("final_answer") or "").strip()
    if require_final_answer:
        if not final_answer:
            errors.append("missing_final_answer")
        elif not _answer_contains_decimal(final_answer, computed):
            errors.append("final_answer_missing_verified_value")
    return LedgerValidation(
        valid=not errors,
        items=[{key: value for key, value in item.items() if key != "_time"} for item in latest_by_key.values()],
        operation=operation,
        value=computed,
        unit=unit,
        final_answer=final_answer,
        accounted_source_ids=tuple(sorted(accounted)),
        errors=tuple(errors),
    )


def normalize_ledger_source_assignments(payload: dict[str, object]) -> dict[str, object]:
    """Resolve only unambiguous duplicate source assignments.

    A source containing an in-scope fact can also contain irrelevant background;
    keeping its included assignment and dropping the background exclusion cannot
    change the fact contribution. Two different included canonical keys remain
    untouched so deterministic validation still fails instead of guessing.
    """

    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return dict(payload)
    items = [dict(item) if isinstance(item, dict) else item for item in raw_items]
    occurrences: dict[str, list[tuple[int, str, str]]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip().lower()
        canonical_key = str(item.get("canonical_key") or item.get("canonical_item") or "").strip().casefold()
        for source_id in dict.fromkeys(
            str(value).strip()
            for value in (item.get("source_ids") or [])
            if str(value).strip()
        ):
            occurrences.setdefault(source_id, []).append((index, status, canonical_key))

    keep_at: dict[str, int] = {}
    for source_id, entries in occurrences.items():
        if len(entries) < 2:
            continue
        included_keys = {
            canonical_key
            for _index, status, canonical_key in entries
            if status == "included" and canonical_key
        }
        if len(included_keys) > 1:
            continue
        priorities = {"included": 3, "uncertain": 2, "excluded": 1}
        keep_at[source_id] = max(
            entries,
            key=lambda entry: (priorities.get(entry[1], 0), -entry[0]),
        )[0]

    normalized_items: list[object] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            normalized_items.append(item)
            continue
        source_ids = list(dict.fromkeys(
            str(value).strip()
            for value in (item.get("source_ids") or [])
            if str(value).strip()
        ))
        item["source_ids"] = [
            source_id
            for source_id in source_ids
            if source_id not in keep_at or keep_at[source_id] == index
        ]
        if item["source_ids"]:
            normalized_items.append(item)
    return {**payload, "items": normalized_items}


def expand_consolidated_ledger(payload: object) -> object:
    """Expand compact final source decisions into the standard items ledger."""

    if not isinstance(payload, dict):
        return payload
    items = list(payload.get("items") or []) if isinstance(payload.get("items"), list) else []
    for field_name, status in (
        ("excluded_source_ids", "excluded"),
        ("uncertain_source_ids", "uncertain"),
    ):
        values = payload.get(field_name)
        if not isinstance(values, list):
            continue
        for source_id in dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value).strip()
        ):
            items.append({
                "canonical_key": f"{status}:{source_id}",
                "label": status,
                "quantity": "0",
                "unit": "item",
                "status": status,
                "source_ids": [source_id],
            })
    return {**payload, "items": items}


def batch_evidence_candidates(
    candidates: list[EvidenceCandidate],
    *,
    max_context_chars: int,
    utilization: float = 0.75,
    max_batches: int = 9,
    max_candidates_per_batch: int | None = None,
) -> tuple[list[list[EvidenceCandidate]], bool]:
    base_limit = max_context_chars if max_context_chars > 0 else 120_000
    char_budget = max(1_000, int(base_limit * max(0.1, min(float(utilization), 1.0))))
    candidate_budget = (
        max(1, int(max_candidates_per_batch))
        if max_candidates_per_batch is not None
        else None
    )

    # Keep a structured memory beside the raw Timeline chunks named by its
    # evidence_ids. Otherwise the two representations of one event can land in
    # different Reader calls and acquire different canonical keys.
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    evidence_owner: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        for evidence_id in candidate.evidence_ids:
            normalized = str(evidence_id).removeprefix("timeline:").strip()
            if not normalized:
                continue
            owner = evidence_owner.setdefault(normalized, index)
            union(index, owner)
    grouped: dict[int, list[EvidenceCandidate]] = {}
    group_order: list[int] = []
    for index, candidate in enumerate(candidates):
        root = find(index)
        if root not in grouped:
            grouped[root] = []
            group_order.append(root)
        grouped[root].append(candidate)

    batches: list[list[EvidenceCandidate]] = []
    current: list[EvidenceCandidate] = []
    current_chars = 0
    for group_id in group_order:
        group = grouped[group_id]
        rendered_chars = sum(
            len(json.dumps({
                "source_id": candidate.source_id,
                "source_type": candidate.source_type,
                "text": candidate.text,
                "occurred_at": candidate.occurred_at,
                "recorded_at": candidate.recorded_at,
                "status": candidate.status,
                "superseded_by": candidate.superseded_by,
            }, ensure_ascii=False))
            for candidate in group
        )
        exceeds_chars = current_chars + rendered_chars > char_budget
        exceeds_candidates = (
            candidate_budget is not None
            and len(current) + len(group) > candidate_budget
        )
        if current and (exceeds_chars or exceeds_candidates):
            batches.append(current)
            current = []
            current_chars = 0
        current.extend(group)
        current_chars += rendered_chars
    if current:
        batches.append(current)
    if len(batches) > max_batches:
        return batches[:max_batches], True
    return batches, False


def _answer_contains_decimal(answer: str, value: Decimal) -> bool:
    normalized_value = value.normalize()
    accepted = {str(value), format(normalized_value, "f")}
    if value == value.to_integral_value():
        accepted.add(str(value.to_integral_value()))
    numbers = {
        token.replace(",", "")
        for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer)
    }
    return bool(accepted & numbers)
