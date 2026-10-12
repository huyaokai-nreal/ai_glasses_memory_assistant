import json
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from ai_glasses_memory_assistant.temporal_parser import resolve_temporal_expression
from ai_glasses_memory_assistant.turn_semantic_classifier import classify_pre_reply_decision


class FakeParser:
    def __init__(self, start, end, phrase=""):
        self.payload = dict(has_temporal_expression=True, kind="date_range", start_at_iso=start,
                            end_at_iso=end, temporal_text=phrase, granularity="day", confidence=.99)
        self.prompts = []

    def run_conversation(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return {"final_response": json.dumps(self.payload)}


@pytest.fixture
def host_zone():
    original = os.environ.get("TZ")
    def set_zone(value):
        os.environ["TZ"] = value
        time.tzset()
    yield set_zone
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


@pytest.mark.parametrize("host,service", [("UTC", "Asia/Shanghai"), ("Asia/Shanghai", "UTC"),
                                           ("UTC", "+08:00"), ("UTC", "America/New_York")])
@pytest.mark.parametrize("offset", [False, True])
def test_explicit_zone_controls_reference_and_naive_boundaries(host_zone, host, service, offset):
    host_zone(host)
    from ai_glasses_memory_assistant.temporal_parser import _ppd_timezone_info
    zone = _ppd_timezone_info(service)
    ref = datetime(2026, 10, 7, 16, 30, tzinfo=timezone.utc).timestamp()
    start, end = datetime(2026, 10, 8, tzinfo=zone), datetime(2026, 10, 9, tzinfo=zone)
    fake = FakeParser(start.isoformat() if offset else start.replace(tzinfo=None).isoformat(),
                      end.isoformat() if offset else end.replace(tzinfo=None).isoformat())
    result = resolve_temporal_expression(fake, "该日安排", reference_time=ref, timezone=service)
    assert (result.start_at, result.end_at) == (start.timestamp(), end.timestamp())
    assert f"Reference time: {datetime.fromtimestamp(ref, zone).isoformat()}" in fake.prompts[0]
    assert len(fake.prompts) == 1


@pytest.mark.parametrize("service", ["", "invalid/zone"])
def test_missing_invalid_zone_preserves_host_fallback(host_zone, service):
    host_zone("Asia/Shanghai")
    fake = FakeParser("2026-10-08T00:00:00", "2026-10-09T00:00:00")
    result = resolve_temporal_expression(fake, "安排", reference_time=0, timezone=service)
    assert result.start_at == datetime(2026, 10, 8, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()


def test_evening_uses_service_local_day(host_zone):
    host_zone("UTC")
    fake = FakeParser("2026-10-08T18:00:00+08:00", "2026-10-08T19:00:00+08:00", "今晚")
    result = resolve_temporal_expression(fake, "今晚安排", reference_time=0, timezone="Asia/Shanghai")
    zone = ZoneInfo("Asia/Shanghai")
    assert result.start_at == datetime(2026, 10, 8, 18, tzinfo=zone).timestamp()
    assert result.end_at == datetime(2026, 10, 9, tzinfo=zone).timestamp()


def test_offset_is_authoritative_and_dst_day_is_not_24_hours(host_zone):
    host_zone("Asia/Shanghai")
    fake = FakeParser("2026-03-08T00:00:00-05:00", "2026-03-09T00:00:00-04:00")
    result = resolve_temporal_expression(fake, "that day", reference_time=0, timezone="UTC")
    assert result.end_at - result.start_at == 23 * 3600


def test_ppd_optional_clock_context_does_not_add_calls():
    fake = FakeParser(None, None)
    classify_pre_reply_decision(fake, "安排", reference_time=0, timezone="Asia/Shanghai")
    assert "1970-01-01T08:00:00+08:00" in fake.prompts[0]
    assert len(fake.prompts) == 1


@pytest.mark.parametrize("service", ["", "invalid/zone"])
def test_ppd_fallback_context_labels_actual_host_zone(host_zone,service):
    host_zone("Asia/Shanghai")
    fake=FakeParser(None,None)
    classify_pre_reply_decision(fake,"安排",reference_time=0,timezone=service)
    assert "1970-01-01T08:00:00+08:00" in fake.prompts[0]
    assert "Timezone: invalid/zone" not in fake.prompts[0]
    assert len(fake.prompts)==1
