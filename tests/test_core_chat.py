from __future__ import annotations

import tempfile

from ai_glasses_memory_assistant import capture_helpers
from tests.helpers import CoreChatService, FakeAgent, isolated_app_home, pre_reply_recall, pre_reply_write


def test_capture_helpers_preserve_service_capture_text_contract() -> None:
    text = " 第一段   内容\n\n第二段\n第三段\n第四段\n第五段\n第六段"

    assert capture_helpers.summarize_capture_text(text) == "第一段 内容；第二段；第三段；第四段；第五段"
    assert CoreChatService._summarize_capture_text(text) == capture_helpers.summarize_capture_text(text)
    assert CoreChatService._continuous_capture_reply(["a"]) == "收到，我先把这段长输入整理到时间线里，有价值的内容会后台沉淀。"
    assert CoreChatService._continuous_capture_reply(["a", "b"]) == "收到，我先把这段长输入按 2 段整理，有价值的内容会后台沉淀。"


def test_chat_returns_reply_and_persists_timeline_audit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        response = service.chat("帮我总结一下这个系统", user_id="u1")

        assert response["reply"] == "主回复"
        assert response["debug"]["timeline"]["persisted"] is True
        assert service.audit_path.exists()
        assert service.fake_agent.calls


def test_chat_saves_memory_candidate_after_reply() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=pre_reply_write("周五检查 demo", memory_type="task"))
        service = CoreChatService(tmpdir, agent=agent)
        response = service.chat("记一下周五检查 demo", user_id="u1")
        memories = service.memory_store.list_memories("u1")

        assert response["reply"]
        assert "记住" in response["reply"]
        assert response["saved_memories"]
        assert memories[0].content == "周五检查 demo"
        assert memories[0].evidence_ids


def test_chat_rejects_sensitive_memory_but_keeps_reply() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=pre_reply_write("api_key=sk-abcdefghijklmnopqrstuvwxyz123456", memory_type="fact"))
        service = CoreChatService(tmpdir, agent=agent)
        response = service.chat("记一下 api_key=sk-abcdefghijklmnopqrstuvwxyz123456", user_id="u1")

        assert response["reply"]
        assert response["saved_memories"] == []
        assert "保存" in response["reply"] or response["debug"]["fast_path"] is True
        assert service.memory_store.list_memories("u1") == []


def test_chat_recalls_user_scoped_memory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        agent = FakeAgent(pre_reply=pre_reply_recall())
        service = CoreChatService(tmpdir, agent=agent)
        mine = service.memory_store.add_memory("u1", "周五检查 demo", kind="event", memory_type="task", created_at=1.0)
        service.memory_store.add_memory("u2", "周五检查 demo", kind="event", memory_type="task", created_at=1.0)

        response = service.chat("我周五要做什么", user_id="u1")

        assert response["recalled_memories"]
        assert response["recalled_memories"][0]["id"] == mine.id
        assert "周五检查 demo" in response["reply"]


def test_markdown_document_import_and_recall_use_document_helpers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        result = service.import_memory_events(
            user_id="u1",
            text="# 南太行自驾攻略\n## 费用\n红旗渠门票 80 元",
            source="markdown_upload",
            context="南太行自驾攻略.md",
        )

        recall = service._recall_documents_for_query("u1", "南太行自驾攻略里红旗渠门票多少钱？")

        assert result["document"]["title"] == "南太行自驾攻略"
        assert recall.mode == "full_document"
        assert recall.reason == "document_title_match"
        assert "红旗渠门票 80 元" in recall.context


def test_text_import_uses_import_helpers_without_bypassing_memory_gate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        result = service.import_memory_events(
            user_id="u1",
            text="- 周五检查 demo\n- 周六整理复盘",
            source="manual_import",
        )

        saved = service.memory_store.list_memories("u1")

        assert result["candidate_count"] == 2
        assert result["saved_count"] == 2
        assert {memory.content for memory in saved} == {"检查 demo", "整理复盘"}
        assert all(memory.evidence_ids for memory in saved)


def test_memory_job_payload_helpers_preserve_polling_contract() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
        service = CoreChatService(tmpdir)
        job = service._create_memory_job(
            user_id="u1",
            session_id="s1",
            mode="reply_first_background",
            candidate_count=1,
            created_at=1.0,
            evidence_ids=["chunk1"],
        )
        saved = service.memory_store.add_memory(
            "u1",
            "周五检查 demo",
            kind="event",
            memory_type="task",
            evidence_ids=["chunk1"],
        )

        updated = service._update_memory_job(
            user_id="u1",
            job_id=job["job_id"],
            status="saved",
            saved_memories=[saved],
            completed=True,
            extraction_backend="core_test",
        )
        polled = service.read_memory_job(user_id="u1", job_id=job["job_id"])

        assert updated is not None
        assert polled is not None
        assert polled["status"] == "saved"
        assert polled["saved_count"] == 1
        assert polled["saved_memory_ids"] == [saved.id]
        assert polled["memory_processing"]["status"] == "saved"
        assert polled["memory_processing"]["stage_reason"] == "memory_saved"
        assert service.read_memory_job(user_id="u2", job_id=job["job_id"]) is None
