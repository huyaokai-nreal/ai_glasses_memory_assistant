from ai_glasses_memory_assistant.temporal_parser import TemporalResolution
from tests.helpers import CoreChatService


def test_time_fields_do_not_certify_event_dates_or_drop_later_records(tmp_path):
    service = CoreChatService(str(tmp_path))
    store = service.memory_store
    later = store.add_memory("u", "Later record describes an earlier event.", created_at=1000,
                             start_at=100, end_at=200)
    unknown = store.add_memory("u", "Two events have unresolved dates.", created_at=1100)
    temporal = TemporalResolution(has_temporal_expression=True, start_at=100, end_at=200,
                                  confidence=1, timezone="UTC")
    capsule = service._build_recent_context_capsule(user_id="u")
    scoped = service._qualify_recent_context_capsule(capsule, subject_ids=None, temporal=temporal)
    main = service._message_with_recall("that day", event_memories=[later, unknown],
                                       profile_memories=[], query_temporal=temporal,
                                       recent_context_capsule=scoped["text"])
    assert later.content in main and unknown.content in main
    assert "field relation: overlaps" in scoped["text"]
    assert "event binding: unknown" in scoped["text"] and "event binding: unknown" in main
    assert "recorded_at:" in main and "not certified event dates" in main
    assert "unknown/conflicting" in main
    service.close()
