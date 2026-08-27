"""Lossless, model-neutral transport for selected evidence sources.

This module is intentionally not wired into a Reader yet.  It establishes the
source boundary and reversibility invariants that a later, separately approved
Reader integration must preserve.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Iterable, Mapping


SOURCE_ENVELOPE_FORMAT = "source-envelope/v1"


class SourceEnvelopeError(ValueError):
    """Raised when a source envelope cannot be safely represented."""


class SourceEnvelopeIntegrityError(SourceEnvelopeError):
    """Raised when serialized source text no longer matches its recorded hash."""


@dataclass(frozen=True)
class SourceEnvelopeInput:
    """The minimum source fields required for lossless evidence transport."""

    source_id: str
    source_type: str
    text: str


@dataclass(frozen=True)
class SourceEnvelope:
    """One complete evidence source with a deterministic visible alias."""

    alias: str
    source_id: str
    source_type: str
    text: str
    text_sha256: str


@dataclass(frozen=True)
class SourceEnvelopeSelection:
    """A whole-source selection made under a decoded-text budget."""

    included: tuple[SourceEnvelope, ...]
    omitted: tuple[SourceEnvelope, ...]


def build_source_envelopes(sources: Iterable[SourceEnvelopeInput]) -> tuple[SourceEnvelope, ...]:
    """Assign stable ``S1...Sn`` aliases in the caller's already-ranked order."""

    envelopes: list[SourceEnvelope] = []
    source_ids: set[str] = set()
    for index, source in enumerate(sources, start=1):
        source_id = _required_text(source.source_id, "source_id")
        if source_id in source_ids:
            raise SourceEnvelopeError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        source_type = _required_text(source.source_type, "source_type")
        if not isinstance(source.text, str):
            raise SourceEnvelopeError("text must be a string")
        envelopes.append(SourceEnvelope(
            alias=f"S{index}",
            source_id=source_id,
            source_type=source_type,
            text=source.text,
            text_sha256=_text_sha256(source.text),
        ))
    return tuple(envelopes)


def serialize_source_envelopes(envelopes: Iterable[SourceEnvelope]) -> str:
    """Render complete source records as one JSON document, never line labels."""

    checked = _validate_envelopes(envelopes)
    payload = {
        "format": SOURCE_ENVELOPE_FORMAT,
        "sources": [
            {
                "alias": envelope.alias,
                "source_id": envelope.source_id,
                "source_type": envelope.source_type,
                "text": envelope.text,
                "text_sha256": envelope.text_sha256,
            }
            for envelope in checked
        ],
    }
    # JSON escaping keeps embedded newlines inside the source field, rather than
    # turning them into accidental record boundaries.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def deserialize_source_envelopes(serialized: str) -> tuple[SourceEnvelope, ...]:
    """Parse and verify source records before a later caller uses their text."""

    try:
        payload = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SourceEnvelopeError("invalid source envelope JSON") from exc
    if not isinstance(payload, dict) or payload.get("format") != SOURCE_ENVELOPE_FORMAT:
        raise SourceEnvelopeError("unsupported source envelope format")
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise SourceEnvelopeError("source envelope sources must be a list")

    envelopes: list[SourceEnvelope] = []
    for record in sources:
        if not isinstance(record, Mapping):
            raise SourceEnvelopeError("source envelope record must be an object")
        try:
            envelope = SourceEnvelope(
                alias=_required_text(record.get("alias"), "alias"),
                source_id=_required_text(record.get("source_id"), "source_id"),
                source_type=_required_text(record.get("source_type"), "source_type"),
                text=record["text"],
                text_sha256=_required_text(record.get("text_sha256"), "text_sha256"),
            )
        except KeyError as exc:
            raise SourceEnvelopeError("source envelope record is missing text") from exc
        if not isinstance(envelope.text, str):
            raise SourceEnvelopeError("text must be a string")
        if envelope.text_sha256 != _text_sha256(envelope.text):
            raise SourceEnvelopeIntegrityError(
                f"source text hash mismatch for {envelope.source_id}"
            )
        envelopes.append(envelope)
    return _validate_envelopes(envelopes)


def source_ids_by_alias(envelopes: Iterable[SourceEnvelope]) -> dict[str, str]:
    """Return the reversible alias-to-real-source-ID mapping for verified records."""

    return {envelope.alias: envelope.source_id for envelope in _validate_envelopes(envelopes)}


def select_complete_envelopes_for_text_budget(
    envelopes: Iterable[SourceEnvelope],
    *,
    text_char_budget: int,
) -> SourceEnvelopeSelection:
    """Keep a priority-ordered prefix without splitting any source's decoded text.

    The caller's input order is the source priority.  This helper counts decoded
    source-text characters only; a future renderer must separately reserve its
    own prompt framing budget.
    """

    if text_char_budget < 0:
        raise SourceEnvelopeError("text_char_budget must be non-negative")
    checked = _validate_envelopes(envelopes)
    used = 0
    for index, envelope in enumerate(checked):
        source_size = len(envelope.text)
        if used + source_size > text_char_budget:
            return SourceEnvelopeSelection(
                included=checked[:index],
                omitted=checked[index:],
            )
        used += source_size
    return SourceEnvelopeSelection(included=checked, omitted=())


def _validate_envelopes(envelopes: Iterable[SourceEnvelope]) -> tuple[SourceEnvelope, ...]:
    checked = tuple(envelopes)
    aliases: set[str] = set()
    source_ids: set[str] = set()
    for index, envelope in enumerate(checked, start=1):
        if not isinstance(envelope, SourceEnvelope):
            raise SourceEnvelopeError("envelope must be a SourceEnvelope")
        expected_alias = f"S{index}"
        if envelope.alias != expected_alias or envelope.alias in aliases:
            raise SourceEnvelopeError("source aliases must be sequential S1...Sn")
        if envelope.source_id in source_ids:
            raise SourceEnvelopeError(f"duplicate source_id: {envelope.source_id}")
        if envelope.text_sha256 != _text_sha256(envelope.text):
            raise SourceEnvelopeIntegrityError(
                f"source text hash mismatch for {envelope.source_id}"
            )
        aliases.add(envelope.alias)
        source_ids.add(envelope.source_id)
    return checked


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceEnvelopeError(f"{field_name} must be a non-empty string")
    return value


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
