from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from unittest.mock import patch

from ai_glasses_memory_assistant.agent_bridge import ChatSession, GlassesChatService
from ai_glasses_memory_assistant.memory_store import EventMemoryStore
from ai_glasses_memory_assistant.timeline_store import TimelineStore


@contextmanager
def isolated_app_home(tmpdir: str, extra_env: Mapping[str, str] | None = None, *, clear: bool = False) -> Iterator[None]:
    """Use the app-owned home path for core tests by default."""
    env = {**dict(extra_env or {}), "AI_GLASSES_HOME": tmpdir}
    with patch.dict("os.environ", env, clear=clear):
        yield


class FakeAgent:
    def __init__(self, pre_reply: dict[str, Any] | None = None, reply: str = "主回复") -> None:
        self.pre_reply = pre_reply or _pre_reply_chat()
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.api_mode = "fake-mode"
        self.enabled_toolsets: list[str] = []

    def run_conversation(self, message, system_message=None, conversation_history=None, persist_user_message=None):
        self.calls.append({
            "message": message,
            "system_message": system_message,
            "persist_user_message": persist_user_message,
        })
        if system_message and "unified pre-reply decision classifier" in system_message:
            return {"final_response": json.dumps(self.pre_reply, ensure_ascii=False)}
        if system_message and "answer synthesis planner" in system_message:
            return {"final_response": json.dumps({"answer_intent": "direct_answer", "confidence": 0.9}, ensure_ascii=False)}
        if system_message and "memory dedupe classifier" in system_message:
            return {"final_response": json.dumps({"action": "new", "memory_id": "", "confidence": 0.0}, ensure_ascii=False)}
        if system_message and "memory correction classifier" in system_message:
            return {"final_response": json.dumps({"is_correction": False, "confidence": 0.0}, ensure_ascii=False)}
        if system_message and "memory observation update classifier" in system_message:
            return {"final_response": json.dumps({"action": "new", "observation_id": "", "confidence": 0.0}, ensure_ascii=False)}
        reply = "周五检查 demo" if "周五检查 demo" in str(message or "") else self.reply
        return {
            "final_response": reply,
            "messages": [
                {"role": "user", "content": persist_user_message or message},
                {"role": "assistant", "content": reply},
            ],
            "api_calls": 1,
            "completed": True,
        }


class CoreChatService(GlassesChatService):
    def __init__(self, tmpdir: str, agent: FakeAgent | None = None) -> None:
        root = Path(tmpdir)
        memory_store = EventMemoryStore(db_path=root / "events.db")
        timeline_store = TimelineStore(db_path=root / "timeline.db")
        super().__init__(memory_store=memory_store, timeline_store=timeline_store, clock=lambda: 1778131200.0)
        self.fake_agent = agent or FakeAgent()

    def _new_session(self, *, user_id: str, session_id: str | None = None) -> ChatSession:
        return ChatSession(id=session_id or "test-session", agent=self.fake_agent)

    def _start_background_candidate_write(self, **kwargs) -> None:
        self._process_candidates_background(**kwargs)

    def _start_background_llm_memory_extraction(self, **kwargs) -> None:
        self._process_llm_memory_extraction_background(**kwargs)


def pre_reply_write(content: str, *, kind: str = "event", memory_type: str = "task") -> dict[str, Any]:
    return {
        **_pre_reply_chat(),
        "turn_intent": "memory_write",
        "memory_action": "write",
        "memory_kind": kind,
        "memory_type": memory_type,
        "candidate_content": content,
        "confidence": 0.95,
        "reason": "core_test_write",
    }


def pre_reply_recall(*, recall_type: str = "event", goal: str = "specific_fact") -> dict[str, Any]:
    return {
        **_pre_reply_chat(),
        "turn_intent": "memory_recall",
        "memory_action": "recall",
        "memory_recall_type": recall_type,
        "recall_type": recall_type,
        "needs_event_memory": recall_type in {"event", "observation"},
        "needs_profile_memory": recall_type == "profile",
        "recall_goal": goal,
        "event_recall_strategy": "text_search" if recall_type == "event" else "skipped",
        "confidence": 0.95,
        "reason": "core_test_recall",
    }


def _pre_reply_chat() -> dict[str, Any]:
    return {
        "turn_intent": "chat",
        "reply_mode": "llm",
        "answer_source": "llm",
        "scope": "unknown",
        "needs_location": False,
        "location_text": "",
        "needs_web_search": False,
        "web_query": None,
        "web_reason": "",
        "needs_profile_memory": False,
        "needs_event_memory": False,
        "needs_timeline_recall": False,
        "memory_recall_type": "none",
        "recall_goal": "none",
        "timeline_query": None,
        "conversation_action": "",
        "event_recall_strategy": "skipped",
        "memory_action": "none",
        "memory_kind": "none",
        "memory_type": "none",
        "recall_type": "none",
        "reply_mode_hint": "llm",
        "flags": {
            "transient": False,
            "do_not_remember": False,
            "correction": False,
            "explanation_query": False,
        },
        "candidate_content": "",
        "reason": "core_test_chat",
        "confidence": 0.95,
    }
