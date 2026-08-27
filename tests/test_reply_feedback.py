from __future__ import annotations

import json
import tempfile

import pytest

from tests.helpers import CoreChatService, isolated_app_home


def test_reply_feedback_is_updatable_and_links_to_the_original_turn() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        response = service.chat("帮我总结一下这个系统", user_id="u1")
        turn_id = response["debug"]["timeline"]["turn_id"]

        first = service.record_reply_feedback(
            user_id="u1",
            turn_id=turn_id,
            rating="needs_improvement",
            note="没有说明召回依据。",
        )["feedback"]
        queued = service.list_reply_feedback(
            user_id="u1",
            rating="needs_improvement",
        )["feedback"]

        assert first["rating"] == "needs_improvement"
        assert first["note"] == "没有说明召回依据。"
        assert first["turn"]["id"] == turn_id
        assert first["turn"]["message"] == "帮我总结一下这个系统"
        assert first["turn"]["reply"] == "主回复"
        assert first["audit_lookup"] == {"timeline_turn_id": turn_id}
        assert queued == [first]

        updated = service.record_reply_feedback(
            user_id="u1",
            turn_id=turn_id,
            rating="satisfied",
            note="这次解释清楚了。",
        )["feedback"]
        assert updated["id"] == first["id"]
        assert updated["note"] == "这次解释清楚了。"
        assert service.list_reply_feedback(user_id="u1", rating="needs_improvement")["feedback"] == []
        assert service.list_reply_feedback(user_id="u1", rating="satisfied")["feedback"][0]["id"] == first["id"]

        audit_records = [
            json.loads(line)
            for line in service.audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        feedback_records = [record for record in audit_records if record.get("record_type") == "reply_feedback"]
        assert [record["rating"] for record in feedback_records] == ["needs_improvement", "satisfied"]
        assert [record["note"] for record in feedback_records] == ["没有说明召回依据。", "这次解释清楚了。"]
        assert all(record["timeline_turn_id"] == turn_id for record in feedback_records)
        service.close()


def test_reply_feedback_rejects_missing_or_unanswered_turns() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        with pytest.raises(ValueError, match="timeline turn not found"):
            service.record_reply_feedback(user_id="u1", turn_id="missing", rating="satisfied")

        pending = service.timeline_store.add_turn("u1", "尚未回复的问题").turn
        with pytest.raises(ValueError, match="no assistant reply"):
            service.record_reply_feedback(user_id="u1", turn_id=pending.id, rating="satisfied")

        response = service.chat("这次测试备注脱敏", user_id="u1")
        redacted = service.record_reply_feedback(
            user_id="u1",
            turn_id=response["debug"]["timeline"]["turn_id"],
            rating="needs_improvement",
            note="验证码是 482931，请不要泄露。",
        )["feedback"]
        assert "482931" not in redacted["note"]
        service.close()


def test_reply_feedback_ui_contract_exposes_both_choices_and_turn_binding() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    app = (root / "static" / "app.js").read_text(encoding="utf-8")
    styles = (root / "static" / "styles.css").read_text(encoding="utf-8")

    assert "appendReplyFeedbackControls(replyNode, payload.debug?.timeline?.turn_id)" in app
    assert '"satisfied", "满意"' in app
    assert '"needs_improvement", "需改进"' in app
    assert 'requestJSON("/api/reply-feedback"' in app
    assert "添加备注（可选）" in app
    assert "note: noteInput.value.trim()" in app
    assert ".reply-feedback-controls" in styles
    assert ".reply-feedback-note" in styles
