"""Protect source boundaries; Fake decisions are plumbing, not classifier accuracy."""
from types import SimpleNamespace

from ai_glasses_memory_assistant.memory_recall_arbitration import arbitrate_recall_sources


def test_factual_artifact_question_keeps_structured_source_but_quote_requires_raw():
    event = SimpleNamespace(id="archive-location", kind="event", memory_type="fact")
    kwargs = dict(message="Where is the warranty document?", profile_memories=[], event_memories=[event],
                  timeline_chunks=[], document_mode="none", document_count=0)
    fact = arbitrate_recall_sources(recall_goal="specific_fact", **kwargs)
    assert fact.event_memories == [event] and not fact.empty_evidence_guard["triggered"]
    raw = arbitrate_recall_sources(recall_goal="raw_evidence", **kwargs)
    assert raw.event_memories == [] and raw.empty_evidence_guard["triggered"]
    assert raw.primary_source == "none"
