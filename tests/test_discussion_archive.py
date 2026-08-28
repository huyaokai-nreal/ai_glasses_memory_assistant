from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ai_glasses_memory_assistant.discussion_archive import (
    DiscussionArchiveSettings,
    local_day_key,
    should_close_before_append,
    summarize_discussion_slice,
)
from ai_glasses_memory_assistant.temporal_parser import TemporalResolution
from ai_glasses_memory_assistant.timeline_store import TimelineStore
from ai_glasses_memory_assistant.turn_planner import plan_turn
from tests.helpers import CoreChatService, FakeAgent, isolated_app_home


def _append(
    service: CoreChatService,
    *,
    user_id: str,
    capture_id: str,
    text: str,
    timestamp: float,
) -> str:
    result = service.append_capture_chunk(
        user_id=user_id,
        capture_id=capture_id,
        text=text,
        timestamp=timestamp,
        metadata={"speaker_label": "用户", "speaker_hint": "user"},
    )
    return str(result["chunk_id"])


def test_discussion_settings_and_slice_boundaries_are_centralized() -> None:
    settings = DiscussionArchiveSettings(
        idle_gap_seconds=180,
        max_slice_seconds=900,
        max_slice_segments=40,
    )
    current = [{"timestamp": 1000.0, "chunk_id": "c1"}]

    assert should_close_before_append(current, {"timestamp": 1179.0}, settings=settings) is False
    assert should_close_before_append(current, {"timestamp": 1180.0}, settings=settings) is True
    assert should_close_before_append(
        [{"timestamp": 1000.0, "chunk_id": f"c{index}"} for index in range(40)],
        {"timestamp": 1010.0},
        settings=settings,
    ) is True
    assert plan_turn("今天讨论了什么", reference_time=1000).needs_discussion_recall is False
    assert plan_turn("刚才说了什么", reference_time=1000).needs_discussion_recall is False
    # Discussion recall is now owned by PreReplyDecision;
    # the deterministic preflight (plan_turn) no longer infers it.


def test_structured_summary_supports_multiple_topics_and_rejects_unknown_ids() -> None:
    class SummaryAgent:
        def __init__(self) -> None:
            self.system_message = ""

        def run_conversation(self, *_args, **_kwargs):
            self.system_message = str(_kwargs.get("system_message") or "")
            return {"final_response": json.dumps({
                "topics": [
                    {
                        "title": "音频方案",
                        "topic_key": "audio",
                        "summary": "确认继续流式 PCM",
                        "key_points": ["流式输入"],
                        "decisions": ["继续流式 PCM"],
                        "tasks": [],
                        "open_questions": [],
                        "participant_labels": ["用户"],
                        "source_chunk_ids": ["c1", "forged"],
                        "merge_topic_id": "topic-allowed",
                    },
                    {
                        "title": "电池测试",
                        "topic_key": "battery",
                        "summary": "安排续航测试",
                        "key_points": [],
                        "decisions": [],
                        "tasks": ["周五测试"],
                        "open_questions": [],
                        "participant_labels": ["Alex"],
                        "source_chunk_ids": ["c2"],
                        "merge_topic_id": "topic-forged",
                    },
                ],
            }, ensure_ascii=False)}

    agent = SummaryAgent()
    topics, backend = summarize_discussion_slice(
        agent,
        chunks=[
            {"chunk_id": "c1", "text": "讨论音频方案", "timestamp": 1, "metadata": {}},
            {"chunk_id": "c2", "text": "讨论电池测试", "timestamp": 2, "metadata": {}},
            {"chunk_id": "c3", "text": "补充测试时间", "timestamp": 3, "metadata": {}},
        ],
        existing_topics=[{"id": "topic-allowed", "title": "音频方案", "summary": "", "topic_key": "audio"}],
    )

    assert backend == "llm"
    assert [topic["title"] for topic in topics] == ["音频方案", "电池测试"]
    assert topics[0]["source_chunk_ids"] == ["c1"]
    assert topics[0]["merge_topic_id"] == "topic-allowed"
    assert topics[1]["source_chunk_ids"] == ["c2", "c3"]
    assert topics[1]["merge_topic_id"] == ""
    assert "Write every user-facing text field" in agent.system_message
    assert "Simplified Chinese" in agent.system_message


def test_running_capture_builds_one_topic_across_multiple_time_spans() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        chunk_ids = [
            _append(
                service,
                user_id="u1",
                capture_id=capture["capture_id"],
                text=f"音频方案。第 {index + 1} 个时间段继续讨论流式输入。",
                timestamp=base + index * 200,
            )
            for index in range(3)
        ]

        day = service.discussion_day(user_id="u1", day=local_day_key(base))["day"]

        assert day is not None and day["topic_count"] == 1
        assert len(day["topics"]) == 1
        assert day["topics"][0]["title"] == "音频方案"
        assert day["topics"][0]["evidence_ids"] == chunk_ids
        assert len(day["topics"][0]["time_spans"]) == 3
        assert service.memory_store.list_memories("u1") == []
        service.close()


def test_archive_falls_back_when_discussion_session_cannot_be_created() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        service._new_session = lambda **_kwargs: (_ for _ in ()).throw(ValueError("LLM unavailable"))
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        _append(
            service,
            user_id="u1",
            capture_id=capture["capture_id"],
            text="离线归档。模型不可用时仍保留摘要。",
            timestamp=base,
        )

        result = service.discussion_day(user_id="u1", day=local_day_key(base))

        assert result["day"] is not None
        assert result["day"]["topics"][0]["title"] == "离线归档"
        assert service.timeline_store.list_discussion_slices("u1")[0]["backend"] == "deterministic_fallback"
        service.close()


def test_three_hundred_segments_are_archived_beyond_recent_six() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        for index in range(300):
            _append(
                service,
                user_id="u1",
                capture_id=capture["capture_id"],
                text=f"全天讨论。第 {index + 1} 段内容。",
                timestamp=base + index * 10,
            )

        result = service.discussion_day(user_id="u1", day=local_day_key(base))
        topic = result["day"]["topics"][0]

        assert result["archive"]["timed_out"] is False
        assert len(topic["evidence_ids"]) == 300
        assert len(topic["available_evidence_ids"]) == 300
        assert topic["evidence_status"] == "available"
        assert len(topic["time_spans"]) >= 8
        managed = service.timeline_chunks_for_management(user_id="u1", chunk_ids=topic["evidence_ids"])
        assert len(managed["chunks"]) == 300
        assert managed["not_found_count"] == 0
        service.close()


def test_day_recall_flushes_running_capture_and_becomes_primary_source() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        # Discussion recall is now owned by PreReplyDecision.
        decision = {
            "turn_intent": "chat",
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "needs_location": False,
            "needs_web_search": False,
            "needs_profile_memory": False,
            "needs_event_memory": False,
            "needs_timeline_recall": False,
            "needs_discussion_recall": True,
            "discussion_query": None,
            "memory_recall_type": "none",
            "event_recall_strategy": "skipped",
            "recall_goal": "none",
            "confidence": 0.95,
        }
        agent = FakeAgent(pre_reply=decision)
        service = CoreChatService(tmpdir, agent=agent)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        _append(
            service,
            user_id="u1",
            capture_id=capture["capture_id"],
            text="产品会议。上午决定继续使用流式 PCM。",
            timestamp=base,
        )

        response = service.chat("今天讨论了什么？", user_id="u1")

        assert response["discussion_recall"]["status"] == "ready"
        assert response["discussion_recall"]["topics"][0]["title"] == "产品会议"
        assert response["source_summary"]["primary_source"] == "discussion_archive"
        assert any(
            "Archived discussion summaries" in str(call.get("message") or "")
            for call in agent.calls
        )
        service.close()


def test_ready_archive_topic_is_visible_to_pre_reply_routing_catalog() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = {
            "turn_intent": "memory_recall",
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "needs_location": False,
            "needs_web_search": False,
            "needs_profile_memory": False,
            "needs_event_memory": False,
            "needs_timeline_recall": False,
            "needs_discussion_recall": True,
            "discussion_query": "产品会议",
            "memory_recall_type": "none",
            "event_recall_strategy": "skipped",
            "recall_goal": "summary",
            "confidence": 0.95,
        }
        agent = FakeAgent(pre_reply=decision)
        service = CoreChatService(tmpdir, agent=agent)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        _append(
            service,
            user_id="u1",
            capture_id=capture["capture_id"],
            text="产品会议。确认继续使用流式 PCM。",
            timestamp=base,
        )
        service.discussion_day(user_id="u1", day=local_day_key(base))

        response = service.chat("产品会议讲了什么？", user_id="u1")

        classifier_call = next(
            call for call in agent.calls
            if "unified pre-reply decision classifier" in str(call.get("system_message") or "")
        )
        assert "Available local discussion archive catalog" in str(classifier_call["message"])
        assert "产品会议" in str(classifier_call["message"])
        assert response["debug"]["discussion_archive_catalog"]["topic_count"] == 1
        assert response["discussion_recall"]["topics"][0]["title"] == "产品会议"
        service.close()


def test_archive_catalog_does_not_override_a_no_recall_decision() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent()
        service = CoreChatService(tmpdir, agent=agent)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        _append(
            service,
            user_id="u1",
            capture_id=capture["capture_id"],
            text="产品会议。确认继续使用流式 PCM。",
            timestamp=base,
        )
        service.discussion_day(user_id="u1", day=local_day_key(base))

        response = service.chat("你好", user_id="u1")

        assert response["debug"]["discussion_archive_catalog"]["topic_count"] == 1
        assert response["discussion_recall"]["status"] == "not_requested"
        service.close()


def test_chinese_question_recalls_legacy_english_catalog_title_exactly() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = {
            "turn_intent": "memory_recall",
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "needs_location": False,
            "needs_web_search": False,
            "needs_profile_memory": False,
            "needs_event_memory": False,
            "needs_timeline_recall": False,
            "needs_discussion_recall": True,
            "discussion_query": "Walmart Executive Meeting on Retail Strategy",
            "memory_recall_type": "none",
            "event_recall_strategy": "skipped",
            "recall_goal": "summary",
            "confidence": 0.95,
        }
        agent = FakeAgent(pre_reply=decision)
        service = CoreChatService(tmpdir, agent=agent)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        chunk_id = _append(
            service,
            user_id="u1",
            capture_id=capture["capture_id"],
            text="沃尔玛会议。讨论零售战略。",
            timestamp=base,
        )
        day_key = local_day_key(base)
        day = service.discussion_day(user_id="u1", day=day_key)["day"]
        topic = day["topics"][0]
        service.timeline_store.upsert_discussion_topic(
            user_id="u1",
            day_key=day_key,
            slice_id=topic["slice_ids"][0],
            contribution={
                "title": "Walmart Executive Meeting on Retail Strategy",
                "topic_key": "walmart-retail-strategy",
                "summary": "Discussed retail strategy.",
                "key_points": [],
                "decisions": [],
                "tasks": [],
                "open_questions": [],
                "participant_labels": [],
                "source_chunk_ids": [chunk_id],
                "merge_topic_id": topic["id"],
            },
            start_at=base,
            end_at=base,
        )
        service.timeline_store.rebuild_discussion_day("u1", day_key)

        response = service.chat("沃尔玛高层会议讲了什么？", user_id="u1")

        assert [topic["title"] for topic in response["discussion_recall"]["topics"]] == [
            "Walmart Executive Meeting on Retail Strategy"
        ]
        classifier_call = next(
            call for call in agent.calls
            if "unified pre-reply decision classifier" in str(call.get("system_message") or "")
        )
        assert "Walmart Executive Meeting on Retail Strategy" in str(classifier_call["message"])
        service.close()


def test_expired_raw_is_purged_while_summary_remains() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        now = [service._clock()]
        service._clock = lambda: now[0]
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        chunk_id = _append(
            service,
            user_id="u1",
            capture_id=capture["capture_id"],
            text="发布讨论。确认下周一灰度。",
            timestamp=now[0],
        )
        day_key = local_day_key(now[0])
        service.discussion_day(user_id="u1", day=day_key)

        now[0] += 31 * 24 * 60 * 60
        purged = service._purge_expired_discussion_raw()
        retained = service.timeline_store.get_discussion_day("u1", day_key)

        assert purged["purged_chunk_count"] == 1
        assert service.timeline_store.list_chunks_by_ids("u1", [chunk_id]) == []
        assert retained is not None and retained["topics"][0]["evidence_status"] == "expired"
        service.close()


def test_sensitive_unknown_speaker_text_is_redacted_before_archive_without_memory_write() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        result = service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="安全检查。验证码是 482931，token 是 abcdefghijklmnopqrstuvwxyz。",
            timestamp=base,
            metadata={"speaker_label": "PRED_SPK0002", "speaker_hint": "unknown"},
        )
        day = service.discussion_day(user_id="u1", day=local_day_key(base))["day"]
        chunk = service.timeline_store.list_chunks_by_ids("u1", [result["chunk_id"]])[0]

        assert result["redacted"] is True
        assert "482931" not in chunk.text
        assert "abcdefghijklmnopqrstuvwxyz" not in chunk.text
        assert "482931" not in day["overview"]
        assert "abcdefghijklmnopqrstuvwxyz" not in day["overview"]
        assert day["topics"][0]["participant_labels"] == ["PRED_SPK0002"]
        assert service.memory_store.list_memories("u1") == []
        service.close()


def test_manual_day_deletion_scopes_raw_and_summary_independently() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        _append(
            service,
            user_id="u1",
            capture_id=capture["capture_id"],
            text="空间清理。保留摘要测试。",
            timestamp=base,
        )
        day_key = local_day_key(base)
        service.discussion_day(user_id="u1", day=day_key)

        raw = service.delete_discussion_day(user_id="u1", day=day_key, scope="raw")
        assert raw["purged_raw_count"] == 1
        assert service.timeline_store.get_discussion_day("u1", day_key) is not None
        appended = _append(
            service,
            user_id="u1",
            capture_id=capture["capture_id"],
            text="删除原文后继续记录的新 final。",
            timestamp=base + 60,
        )
        assert service.timeline_store.list_chunks_by_ids("u1", [appended])

        summary = service.delete_discussion_day(user_id="u1", day=day_key, scope="summary")
        assert summary["deleted_summary_record_count"] >= 3
        assert service.timeline_store.get_discussion_day("u1", day_key) is None
        service.close()


def test_restart_recovers_stale_slices_and_uncovered_ambient_chunks() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        base = 1778131200.0
        store = TimelineStore(db_path=Path(tmpdir) / "timeline.db")
        for index, status in enumerate(("pending", "running", "failed")):
            capture_id = store.add_capture(
                "u1",
                source="ambient_audio_text",
                started_at=base + index * 60,
            )["capture_id"]
            chunk = store.add_capture_chunk(
                "u1",
                capture_id,
                f"{status} 恢复讨论。",
                timestamp=base + index * 60,
                source="ambient_audio_text",
            )
            item = store.create_discussion_slice(
                user_id="u1",
                capture_id=capture_id,
                day_key=local_day_key(base),
                chunks=[{"chunk_id": chunk.id, "timestamp": chunk.timestamp}],
            )
            assert item is not None
            store.update_discussion_slice(
                user_id="u1",
                slice_id=item["id"],
                status=status,
            )
        uncovered_capture_id = store.add_capture(
            "u1",
            source="ambient_audio_text",
            started_at=base + 240,
        )["capture_id"]
        store.add_capture_chunk(
            "u1",
            uncovered_capture_id,
            "异常退出后未覆盖的讨论。",
            timestamp=base + 240,
            source="ambient_audio_text",
        )
        store._conn.close()

        service = CoreChatService(tmpdir)
        slices = service.timeline_store.list_discussion_slices("u1")
        day = service.timeline_store.get_discussion_day("u1", local_day_key(base))

        assert len(slices) == 4
        assert {item["status"] for item in slices} == {"ready"}
        assert day is not None
        assert len(day["evidence_ids"]) == 4
        assert service.memory_store.list_memories("u1") == []
        service.close()


def test_retry_after_topic_write_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        _append(
            service,
            user_id="u1",
            capture_id=capture["capture_id"],
            text="幂等讨论。确认恢复不能重复摘要。",
            timestamp=base,
        )
        service.discussion_day(user_id="u1", day=local_day_key(base))
        before = service.timeline_store.get_discussion_day("u1", local_day_key(base))
        item = service.timeline_store.list_discussion_slices("u1")[0]
        service.timeline_store.update_discussion_slice(
            user_id="u1",
            slice_id=item["id"],
            status="failed",
        )

        service._process_discussion_slice(user_id="u1", slice_id=item["id"])
        after = service.timeline_store.get_discussion_day("u1", local_day_key(base))

        assert after["topic_count"] == 1
        assert after["topics"][0]["summary"] == before["topics"][0]["summary"]
        assert after["topics"][0]["time_spans"] == before["topics"][0]["time_spans"]
        assert after["topics"][0]["evidence_ids"] == before["topics"][0]["evidence_ids"]
        assert service.timeline_store.get_discussion_slice("u1", item["id"])["status"] == "ready"
        service.close()


def test_multi_day_recall_combines_days_and_specific_topic_filter_is_precise() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        base = service._clock()
        _, day_start, _ = service._discussion_day_bounds(base)
        fixtures = [
            ("u1", "音频方案。下午决定继续流式 PCM。", day_start + 14 * 60 * 60),
            ("u1", "招聘计划。下午讨论候选人面试。", day_start + 15 * 60 * 60),
            ("u1", "电池方案。次日决定增加续航测试。", day_start + 24 * 60 * 60 + 10 * 60 * 60),
            ("u2", "私有话题。另一个用户的讨论。", day_start + 14 * 60 * 60),
        ]
        for user_id, text, timestamp in fixtures:
            capture = service.start_capture(user_id=user_id, source="ambient_audio_text")
            _append(
                service,
                user_id=user_id,
                capture_id=capture["capture_id"],
                text=text,
                timestamp=timestamp,
            )

        temporal = TemporalResolution(
            has_temporal_expression=True,
            start_at=day_start,
            end_at=day_start + 2 * 24 * 60 * 60,
            confidence=0.99,
        )
        broad = service._recall_discussions(
            user_id="u1",
            temporal=temporal,
            reference_time=base,
            query="这两天讨论了什么",
        )
        specific = service._recall_discussions(
            user_id="u1",
            temporal=temporal,
            reference_time=base,
            query="下午音频方案定了什么",
        )

        assert len(broad["daily_overviews"]) == 2
        assert {topic["title"] for topic in broad["topics"]} == {"音频方案", "招聘计划", "电池方案"}
        assert [topic["title"] for topic in specific["topics"]] == ["音频方案"]
        assert specific["raw_available"] is True
        assert specific["raw_evidence_status"] == "available"
        assert all("私有话题" not in topic["summary"] for topic in broad["topics"])
        service.close()


def test_capture_closure_and_provenance_keep_environment_separate_from_self() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        first = service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="脱口秀片段。主持人开始表演。",
            timestamp=base,
            metadata={"speaker_label": "spk_01", "speaker_state": "other", "speaker_track_scope": "capture"},
        )
        second = service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="发音讨论。有人纠正了名字读音。",
            timestamp=base + 200,
            metadata={"speaker_label": "spk_02", "speaker_state": "unknown", "speaker_track_scope": "capture"},
        )
        service.discussion_day(user_id="u1", day=local_day_key(base))

        recalled = service._recall_discussions(
            user_id="u1",
            temporal=TemporalResolution(),
            reference_time=base,
            query="脱口秀片段",
            evidence_scope="personal",
            relation_scope="capture",
            coverage_requirement="complete_set",
        )

        assert {topic["title"] for topic in recalled["topics"]} == {"脱口秀片段", "发音讨论"}
        assert recalled["evidence_provenance"]["speaker_counts"] == {
            "self": 0,
            "non_self": 1,
            "uncertain": 1,
        }
        assert recalled["evidence_provenance"]["anonymous_tracks"] == ["spk_01", "spk_02"]
        assert recalled["coverage"]["requirement"] == "complete_set"
        assert recalled["coverage"]["coverage_complete"] is True
        assert first["chunk_id"] in recalled["evidence_ids"]
        assert second["chunk_id"] in recalled["evidence_ids"]
        assert "no personal activity is confirmed" in service._discussion_context_text(recalled)
        service.close()


def test_complete_set_discussion_boundary_is_ppd_owned_and_never_infers_people() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = {
            "turn_intent": "memory_recall",
            "memory_action": "recall",
            "reply_mode": "llm",
            "answer_source": "llm",
            "needs_discussion_recall": True,
            "discussion_query": "开场笑话",
            "memory_recall_type": "none",
            "recall_goal": "summary",
            "event_recall_strategy": "skipped",
            "evidence_scope": "environment",
            "discussion_relation_scope": "capture",
            "coverage_requirement": "complete_set",
            "answer_obligations": ["speaker_attribution"],
            "confidence": 0.95,
        }
        agent = FakeAgent(pre_reply=decision, reply="不应调用自由回答")
        service = CoreChatService(tmpdir, agent=agent)
        base = service._clock()
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="开场笑话。第一段环境内容。",
            timestamp=base,
            metadata={"speaker_state": "other", "speaker_label": "嘉宾甲"},
        )
        service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="观众互动。第二段环境内容。",
            timestamp=base + 200,
            metadata={"speaker_state": "unknown", "speaker_label": "嘉宾乙"},
        )
        service.discussion_day(user_id="u1", day=local_day_key(base))

        first = service.chat("请从这段经历说起", user_id="u1")
        second = service.chat("换一种说法也请总结", user_id="u1")

        assert first["discussion_recall"]["evidence_ids"] == second["discussion_recall"]["evidence_ids"]
        assert len(first["discussion_recall"]["topics"]) == 2
        assert "环境中发生的讨论" in first["reply"]
        assert "不能用来确认参与人数或逐人发言" in first["reply"]
        assert "嘉宾甲" not in first["reply"]
        assert first["debug"]["complete_set_answer"]["reader_status"] == "deterministic_evidence_boundary"
        assert first["debug"]["local_reply_policy"]["role"] == "discussion_evidence_boundary"
        assert not any(
            "evidence-accounting Reader" in str(call.get("system_message") or "")
            for call in agent.calls
        )
        service.close()


def test_complete_set_personal_discussion_reports_unassigned_environment_topics() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        decision = {
            "turn_intent": "memory_recall",
            "memory_action": "recall",
            "reply_mode": "llm",
            "answer_source": "llm",
            "needs_discussion_recall": True,
            "discussion_query": None,
            "memory_recall_type": "none",
            "recall_goal": "summary",
            "event_recall_strategy": "skipped",
            "evidence_scope": "personal",
            "discussion_relation_scope": "time_range",
            "coverage_requirement": "complete_set",
            "confidence": 0.95,
        }
        service = CoreChatService(tmpdir, agent=FakeAgent(pre_reply=decision))
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="环境讨论，不是用户个人活动。",
            timestamp=service._clock(),
            metadata={"speaker_state": "other"},
        )
        service.discussion_day(user_id="u1", day=local_day_key(service._clock()))

        response = service.chat("任何等价的个人回顾问法", user_id="u1")

        assert "没有可确认属于你的个人活动" in response["reply"]
        assert "未归属环境主题" in response["reply"]
        assert response["discussion_recall"]["evidence_provenance"]["speaker_counts"]["self"] == 0
        service.close()


def test_unlabeled_environment_chunks_do_not_become_fake_speakers_or_memories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        capture = service.start_capture(user_id="u1", source="ambient_audio_text")
        service.append_capture_chunk(
            user_id="u1",
            capture_id=capture["capture_id"],
            text="环境里有人在聊天。",
            timestamp=service._clock(),
            metadata={"speaker_state": "unknown"},
        )

        result = service.stop_capture(user_id="u1", capture_id=capture["capture_id"])

        assert result["import_result"]["saved_count"] == 0
        assert all(
            "PRED_SPK_UNLABELED" not in str(item)
            for item in result["import_result"]["conversation_session"].get("voice_identity") or []
        )
        assert service.memory_store.list_memories("u1") == []
        service.close()
