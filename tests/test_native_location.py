from __future__ import annotations

import json
import tempfile
from unittest.mock import Mock, patch

import pytest

from ai_glasses_memory_assistant.agent_bridge import (
    DEVICE_TURN_LOCATION_TTL_SECONDS,
    LocationContext,
)
from ai_glasses_memory_assistant.android_runtime import location_preflight
from ai_glasses_memory_assistant.memory_candidate import IntentDecision
from ai_glasses_memory_assistant.turn_planner import TurnPlan, native_location_preflight
from tests.helpers import CoreChatService, FakeAgent, isolated_app_home
from tests.test_audio_engine import assistant_final_event


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("今天的天气怎么样", "device_location_weather"),
        ("附近有什么咖啡店", "current_location_query"),
        ("我现在在哪", "current_location_query"),
        ("导航到故宫", "device_location_navigation_origin"),
    ],
)
def test_native_location_preflight_requests_only_location_dependent_queries(
    message: str,
    reason: str,
) -> None:
    assert native_location_preflight(message) == {"needed": True, "reason": reason}


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("讲个笑话", "no_location_intent"),
        ("我明天要做什么", "no_location_intent"),
        ("北京天气怎么样", "explicit_weather_place"),
        ("从北京导航到上海", "explicit_navigation_origin"),
    ],
)
def test_native_location_preflight_skips_queries_with_no_device_location_dependency(
    message: str,
    reason: str,
) -> None:
    assert native_location_preflight(message) == {"needed": False, "reason": reason}


def test_android_runtime_preflight_uses_the_shared_planner_policy() -> None:
    assert json.loads(location_preflight("今天的天气怎么样"))["needed"] is True
    assert json.loads(location_preflight("北京天气怎么样")) == {
        "needed": False,
        "reason": "explicit_weather_place",
    }


def test_final_explicit_place_weather_decision_drops_conservative_device_location() -> None:
    location = LocationContext(
        status="available",
        latitude=39.987654,
        longitude=116.123456,
        source="android_location_manager",
    )
    intent = IntentDecision(
        needs_web_search=True,
        is_weather_query=True,
        weather_location_source="explicit_place",
        weather_place_text="北京",
    )

    assert CoreChatService._location_context_for_response(
        intent=intent,
        location_context=location,
        location_needed=False,
    ) is None


def test_device_turn_location_is_event_scoped_consumed_once_and_not_persisted() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service.set_device_network_state(online=False)
        service.chat = Mock(return_value={"reply": "天气回复"})
        location = {
            "status": "available",
            "latitude": 39.987654,
            "longitude": 116.123456,
            "accuracy": 12.0,
            "timestamp": 1778131200.0,
            "source": "android_location_manager",
        }
        first = assistant_final_event("device-session", "location-event", "今天的天气怎么样")
        queued = service.ingest_device_audio_event(
            user_id="u1",
            event_payload=first.to_dict(),
            turn_context={"location": location},
        )

        durable = service.timeline_store.get_device_audio_event("u1", first.event_id)
        serialized_durable = json.dumps(durable, ensure_ascii=False)
        assert queued["status"] == "pending"
        assert "39.987654" not in serialized_durable
        assert "116.123456" not in serialized_durable
        assert "location" not in durable["private"]

        service.set_device_network_state(online=True)
        completed = service.wait_device_audio_event(user_id="u1", event_id=first.event_id, timeout=2.0)
        assert completed["status"] == "completed"
        first_location = service.chat.call_args.kwargs["location"]
        assert isinstance(first_location, LocationContext)
        assert first_location.latitude == 39.987654
        assert first_location.longitude == 116.123456
        assert service._device_turn_locations == {}

        second = assistant_final_event("device-session", "plain-event", "讲个笑话")
        service.ingest_device_audio_event(user_id="u1", event_payload=second.to_dict())
        service.wait_device_audio_event(user_id="u1", event_id=second.event_id, timeout=2.0)
        assert service.chat.call_args.kwargs["location"] is None
        service.close()


def test_expired_or_duplicate_device_turn_location_is_not_reused() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        now = [1778131200.0]
        service._clock = lambda: now[0]
        service.set_device_network_state(online=False)
        service.chat = Mock(return_value={"reply": "回复"})
        event = assistant_final_event("device-session", "expiring-event", "今天的天气怎么样")
        context = {
            "location": {
                "status": "available",
                "latitude": 39.9,
                "longitude": 116.4,
                "source": "android_location_manager",
            }
        }
        service.ingest_device_audio_event(
            user_id="u1",
            event_payload=event.to_dict(),
            turn_context=context,
        )
        now[0] += DEVICE_TURN_LOCATION_TTL_SECONDS + 1
        service.set_device_network_state(online=True)
        service.wait_device_audio_event(user_id="u1", event_id=event.event_id, timeout=2.0)
        assert service.chat.call_args.kwargs["location"] is None

        duplicate = service.ingest_device_audio_event(
            user_id="u1",
            event_payload=event.to_dict(),
            turn_context=context,
        )
        assert duplicate["created"] is False
        assert service._device_turn_locations == {}
        service.close()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("denied", "没有定位权限"),
        ("disabled", "系统定位目前已关闭"),
        ("timeout", "获取当前位置超时"),
        ("unavailable", "没有获取到可用位置"),
        ("missing", "没有收到当前位置"),
    ],
)
def test_location_failure_reply_matches_actual_status(status: str, expected: str) -> None:
    reply = CoreChatService._local_reply_for_plan(
        Mock(),
        TurnPlan(),
        "今天的天气怎么样",
        reference_time=1778131200.0,
        profile_memories=[],
        event_memories=[],
        timeline_chunks=[],
        location_context=LocationContext(status=status, source="android_location_manager"),
        location_needed=True,
    )
    assert expected in reply


def test_valid_location_with_weather_timeout_reports_weather_service_failure() -> None:
    pre_reply = dict(FakeAgent().pre_reply)
    pre_reply.update({
        "needs_location": True,
        "location_text": "当前位置",
        "needs_web_search": True,
        "web_query": "今天的天气",
        "web_reason": "current_weather",
        "reason": "test_current_weather",
    })
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir, agent=FakeAgent(pre_reply=pre_reply))
        with patch("ai_glasses_memory_assistant.web_search.search_web", side_effect=TimeoutError("timed out")):
            response = service.chat(
                "今天的天气怎么样",
                user_id="u1",
                location={
                    "status": "available",
                    "latitude": 39.987654,
                    "longitude": 116.123456,
                    "source": "android_location_manager",
                },
            )

        assert response["reply"] == "天气服务暂时不可用，请稍后再试。"
        assert "定位" not in response["reply"]
        tool = next(item for item in response["debug"]["tools"] if item["name"] == "web_search")
        assert tool["error_type"] == "TimeoutError"
        audit = service.read_audit_records(user_id="u1", limit=1)[0]
        serialized_audit = json.dumps(audit, ensure_ascii=False)
        assert "39.987654" not in serialized_audit
        assert "116.123456" not in serialized_audit
        assert audit["debug"]["location"]["latitude"] is None
        assert audit["debug"]["location"]["longitude"] is None
        service.close()


def test_native_location_is_removed_from_device_dispatch_timeline_and_audit() -> None:
    pre_reply = dict(FakeAgent().pre_reply)
    pre_reply.update({
        "needs_location": True,
        "location_text": "当前位置",
        "reason": "test_location_context",
    })
    coordinate_reply = "当前位置坐标是 39.987654,116.123456"
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(
            tmpdir,
            agent=FakeAgent(pre_reply=pre_reply, reply=coordinate_reply),
        )
        event = assistant_final_event("device-session", "persisted-location-event", "我现在在哪")
        queued = service.ingest_device_audio_event(
            user_id="u1",
            event_payload=event.to_dict(),
            turn_context={
                "location": {
                    "status": "available",
                    "latitude": 39.987654,
                    "longitude": 116.123456,
                    "source": "android_location_manager",
                }
            },
        )
        completed = service.wait_device_audio_event(user_id="u1", event_id=event.event_id, timeout=2.0)

        assert queued["created"] is True
        persisted = service.timeline_store.get_device_audio_event("u1", event.event_id)
        persisted_json = json.dumps(persisted, ensure_ascii=False)
        assert "39.987654" not in persisted_json
        assert "116.123456" not in persisted_json
        assert completed["dispatch"]["result"]["debug"]["location"]["latitude"] is None
        assert completed["dispatch"]["result"]["debug"]["location"]["longitude"] is None

        turn_id = completed["dispatch"]["result"]["debug"]["timeline"]["turn_id"]
        timeline_turn = service.timeline_store.get_turn("u1", turn_id)
        assert timeline_turn is not None
        assert "39.987654" not in timeline_turn.assistant_reply
        assert "116.123456" not in timeline_turn.assistant_reply
        audit_json = json.dumps(service.read_audit_records(user_id="u1"), ensure_ascii=False)
        assert "39.987654" not in audit_json
        assert "116.123456" not in audit_json
        service.close()
