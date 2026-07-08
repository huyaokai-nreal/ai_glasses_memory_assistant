from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LONGMEMEVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "longmemeval"
DEFAULT_ORACLE_PATH = DEFAULT_LONGMEMEVAL_DIR / "longmemeval_oracle.json"
DEFAULT_SMALL_PATH = DEFAULT_LONGMEMEVAL_DIR / "longmemeval_s_cleaned.json"


@dataclass(frozen=True)
class LongMemEvalTurn:
    role: str
    content: str
    has_answer: bool = False


@dataclass(frozen=True)
class LongMemEvalSession:
    session_id: str
    date: str
    turns: list[LongMemEvalTurn]


@dataclass(frozen=True)
class LongMemEvalItem:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: list[LongMemEvalSession]
    answer_session_ids: list[str]
    is_abstention: bool


def load_longmemeval_items(
    path: Path,
    *,
    limit: int = 0,
    question_types: set[str] | None = None,
    include_abstention: bool = True,
) -> list[LongMemEvalItem]:
    if not path.exists():
        raise FileNotFoundError(
            f"LongMemEval dataset not found: {path}. "
            "Put the downloaded JSON files under data/benchmarks/longmemeval/."
        )
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise ValueError(f"LongMemEval dataset must be a JSON list: {path}")

    items: list[LongMemEvalItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item = parse_longmemeval_item(raw)
        if question_types and item.question_type not in question_types:
            continue
        if item.is_abstention and not include_abstention:
            continue
        items.append(item)
        if limit > 0 and len(items) >= limit:
            break
    return items


def parse_longmemeval_item(raw: dict[str, Any]) -> LongMemEvalItem:
    question_id = str(raw.get("question_id") or "").strip()
    if not question_id:
        raise ValueError("LongMemEval item missing question_id")
    question_type = str(raw.get("question_type") or "unknown").strip() or "unknown"
    session_ids = [str(item) for item in raw.get("haystack_session_ids") or []]
    dates = [str(item) for item in raw.get("haystack_dates") or []]
    raw_sessions = raw.get("haystack_sessions") or []
    sessions: list[LongMemEvalSession] = []
    for index, raw_session in enumerate(raw_sessions):
        turns: list[LongMemEvalTurn] = []
        for raw_turn in raw_session or []:
            if not isinstance(raw_turn, dict):
                continue
            content = str(raw_turn.get("content") or "").strip()
            if not content:
                continue
            turns.append(
                LongMemEvalTurn(
                    role=str(raw_turn.get("role") or "unknown").strip() or "unknown",
                    content=content,
                    has_answer=bool(raw_turn.get("has_answer")),
                )
            )
        sessions.append(
            LongMemEvalSession(
                session_id=session_ids[index] if index < len(session_ids) else f"{question_id}-session-{index + 1}",
                date=dates[index] if index < len(dates) else "",
                turns=turns,
            )
        )
    return LongMemEvalItem(
        question_id=question_id,
        question_type=question_type,
        question=str(raw.get("question") or "").strip(),
        answer=str(raw.get("answer") or "").strip(),
        question_date=str(raw.get("question_date") or "").strip(),
        sessions=sessions,
        answer_session_ids=[str(item) for item in raw.get("answer_session_ids") or []],
        is_abstention=question_id.endswith("_abs"),
    )


def answer_terms(answer: str) -> list[str]:
    terms: list[str] = []
    for part in str(answer or "").replace(";", ",").split(","):
        cleaned = part.strip().strip(".。")
        if cleaned:
            terms.append(cleaned)
    return terms or ([str(answer).strip()] if str(answer).strip() else [])
