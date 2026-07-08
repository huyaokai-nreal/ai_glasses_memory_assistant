from __future__ import annotations

import base64
import ast
import io
import json
import re
import sqlite3
import sys
import tempfile
import types
import unittest
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_glasses_memory_assistant.agent_bridge import (
    ChatSession,
    ConversationSession,
    ConversationTurn,
    GlassesChatService,
    LocationContext,
    SenseVoiceASRRunner,
)
from ai_glasses_memory_assistant.memory_candidate import IntentDecision, MemoryWriteCandidate
from ai_glasses_memory_assistant.intent_policy import should_write_memory_candidate
from ai_glasses_memory_assistant.memory_lifecycle import (
    ACTIVE_MEMORY_STATUS,
    can_transition_memory_status,
    is_active_memory_status,
    lifecycle_transition_payload,
    normalize_memory_status,
)
from ai_glasses_memory_assistant.memory_evidence import (
    EvidenceReferenceCounts,
    plan_timeline_evidence_cleanup,
    timeline_chunk_evidence_ids,
)
from ai_glasses_memory_assistant.memory_store import EventMemoryStore, effective_memory_strength, event_to_dict
from ai_glasses_memory_assistant.temporal_parser import TemporalResolution, _normalize_broad_evening_resolution, _resolution_from_payload
from ai_glasses_memory_assistant.timeline_store import TimelineStore
from ai_glasses_memory_assistant.turn_planner import TurnPlan, plan_turn, resolve_temporal_local
from ai_glasses_memory_assistant.turn_semantic_classifier import _decision_from_payload


# FakeAgent 按 system_message 区分 intent/temporal/pre_reply/main 等内部调用。
class FakeAgent:
    def __init__(
        self,
        intent_payload: dict | None = None,
        temporal_payload: dict | None = None,
        pre_reply_payload: dict | None = None,
        answer_payload: dict | None = None,
        text_emotion_payload: dict | None = None,
        dedupe_payload: dict | str | None = None,
        correction_payload: dict | str | None = None,
        correction_target_payload: dict | str | None = None,
        segment_payload: dict | str | None = None,
        semantic_payload: dict | str | None = None,
    ) -> None:
        self.intent_payload = intent_payload or {
            "needs_web_search": False,
            "web_query": None,
            "web_reason": "",
            "memory_write_candidates": [],
            "confidence": 1.0,
        }
        self.temporal_payload = temporal_payload or {
            "has_temporal_expression": True,
            "temporal_text": "周五下午3点",
            "kind": "instant",
            "start_at_iso": "2026-05-08T15:00:00+08:00",
            "end_at_iso": "2026-05-08T16:00:00+08:00",
            "granularity": "hour",
            "normalized_text": "和 alex 开周会",
            "confidence": 0.95,
            "reason": "fake temporal parse",
        }
        self.pre_reply_payload_explicit = pre_reply_payload is not None
        self.pre_reply_payload = pre_reply_payload or {
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "needs_location": False,
            "location_text": "",
            "needs_web_search": False,
            "web_query": None,
            "web_reason": "",
            "confidence": 0.2,
            "reason": "fake default route",
        }
        self.answer_payload = answer_payload or {
            "answer_intent": "direct_answer",
            "organization": "direct",
            "evidence_policy": "use_available_context",
            "filtering_rules": [],
            "uncertainty_policy": "none",
            "style": "short_direct",
            "confidence": 0.8,
            "reason": "fake answer directive",
        }
        self.text_emotion_payload = text_emotion_payload or {
            "label": "unknown",
            "confidence": "low",
            "should_affect_reply": False,
            "reason": "fake default text emotion",
        }
        self.dedupe_payload = dedupe_payload if dedupe_payload is not None else {
            "action": "new",
            "memory_id": "",
            "confidence": 0.0,
            "reason": "fake default dedupe",
        }
        self.correction_payload = correction_payload if correction_payload is not None else {
            "is_correction": False,
            "corrected_content": "",
            "memory_kind": "event",
            "memory_type": "project_state",
            "confidence": 0.0,
            "reason": "fake default correction",
        }
        self.correction_target_payload = correction_target_payload if correction_target_payload is not None else {
            "action": "none",
            "memory_id": "",
            "confidence": 0.0,
            "reason": "fake default correction target",
        }
        self.segment_payload = segment_payload if segment_payload is not None else {
            "semantic_role": "memory_candidate",
            "noise_level": "none",
            "contains_filler": False,
            "do_not_remember_scope": "",
            "should_extract": True,
            "candidate_span": "",
            "candidate_hint": "",
            "confidence": 0.7,
            "reason": "fake default segment semantic decision",
        }
        self.semantic_payload_explicit = semantic_payload is not None
        self.semantic_payload = semantic_payload if semantic_payload is not None else "not json"
        self.intent_calls = 0
        self.temporal_calls = 0
        self.pre_reply_calls = 0
        self.answer_calls = 0
        self.text_emotion_calls = 0
        self.dedupe_calls = 0
        self.correction_calls = 0
        self.correction_target_calls = 0
        self.main_calls = 0
        self.model = "fake"
        self.provider = "fake"
        self.api_mode = "fake"
        self.enabled_toolsets = []
        self.main_messages = []
        self.pre_reply_messages = []
        self.answer_messages = []
        self.text_emotion_messages = []
        self.dedupe_messages = []
        self.correction_messages = []
        self.correction_target_messages = []
        self.segment_calls = 0
        self.segment_messages = []
        self.semantic_calls = 0
        self.semantic_messages = []

    # 测试通过调用计数判断当前场景是否跳过了主 LLM 或内部分类器。
    def run_conversation(self, message, system_message=None, conversation_history=None, persist_user_message=None):
        if system_message and "intent classifier" in system_message:
            self.intent_calls += 1
            payload = self._payload_for_message(self.intent_payload, message)
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "temporal parser" in system_message:
            self.temporal_calls += 1
            payload = self._payload_for_message(self.temporal_payload, message)
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "unified pre-reply decision classifier" in system_message:
            self.pre_reply_calls += 1
            self.semantic_calls += 1
            self.pre_reply_messages.append(message)
            self.semantic_messages.append(message)
            payload = self._pre_reply_payload_for_message(message)
            if isinstance(payload, str):
                return {"final_response": payload}
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "answer synthesis planner" in system_message:
            self.answer_calls += 1
            self.answer_messages.append(message)
            if isinstance(self.answer_payload, str):
                return {"final_response": self.answer_payload}
            payload = self._payload_for_message(self.answer_payload, message)
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "text emotion classifier" in system_message:
            self.text_emotion_calls += 1
            self.text_emotion_messages.append(message)
            payload = self._payload_for_message(self.text_emotion_payload, message)
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "memory dedupe classifier" in system_message:
            self.dedupe_calls += 1
            self.dedupe_messages.append(message)
            if isinstance(self.dedupe_payload, str):
                return {"final_response": self.dedupe_payload}
            payload = self._payload_for_message(self.dedupe_payload, message)
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "memory correction classifier" in system_message:
            self.correction_calls += 1
            self.correction_messages.append(message)
            if isinstance(self.correction_payload, str):
                return {"final_response": self.correction_payload}
            payload = self._payload_for_message(self.correction_payload, message)
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "memory correction target resolver" in system_message:
            self.correction_target_calls += 1
            self.correction_target_messages.append(message)
            if isinstance(self.correction_target_payload, str):
                return {"final_response": self.correction_target_payload}
            payload = self._payload_for_message(self.correction_target_payload, message)
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "segment semantic cleaner" in system_message:
            self.segment_calls += 1
            self.segment_messages.append(message)
            if isinstance(self.segment_payload, str):
                return {"final_response": self.segment_payload}
            payload = self._payload_for_message(self.segment_payload, message)
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "unified turn semantic classifier" in system_message:
            self.semantic_calls += 1
            self.semantic_messages.append(message)
            if isinstance(self.semantic_payload, str):
                if not self.semantic_payload_explicit:
                    payload = self._semantic_payload_from_intent_fixture(self.intent_payload, message)
                    if payload is not None:
                        return {"final_response": json.dumps(payload, ensure_ascii=False)}
                return {"final_response": self.semantic_payload}
            payload = self._payload_for_message(self.semantic_payload, message)
            return {"final_response": json.dumps(payload, ensure_ascii=False)}
        if system_message and "memory observation update classifier" in system_message:
            return {"final_response": json.dumps({"action": "new", "observation_id": "", "confidence": 0.0, "reason": "fake default observation update"}, ensure_ascii=False)}
        self.main_calls += 1
        self.main_messages.append(message)
        reply = self._reply_from_context(message)
        return {
            "final_response": reply,
            "messages": [{"role": "assistant", "content": reply}],
            "api_calls": 1,
            "completed": True,
        }

    @staticmethod
    def _default_pre_reply_payload_for_message(message):
        user_message = FakeAgent._user_message_from_prompt(message)
        semantic_route = FakeAgent._default_pre_reply_semantic_route(user_message)
        memory_recall_type = semantic_route["memory_recall_type"]
        reply_mode = "llm"
        answer_source = "llm"
        scope = "unknown"
        fast_reply_mode = FakeAgent._default_pre_reply_fast_reply_mode(user_message)
        if fast_reply_mode == "local_current_time":
            reply_mode = "local_current_time"
            answer_source = "local_clock"
            scope = "device_local"
        elif fast_reply_mode == "unsupported_world_time":
            reply_mode = "unsupported_world_time"
            answer_source = "world_time"
            scope = "named_place"
        return {
            "reply_mode": reply_mode,
            "answer_source": answer_source,
            "scope": scope,
            "needs_location": semantic_route["needs_location"],
            "location_text": "",
            "needs_web_search": semantic_route["needs_web_search"],
            "web_query": semantic_route["web_query"],
            "web_reason": semantic_route["web_reason"],
            "needs_profile_memory": semantic_route["needs_profile_memory"],
            "needs_event_memory": semantic_route["needs_event_memory"],
            "needs_timeline_recall": semantic_route["needs_timeline_recall"],
            "memory_recall_type": memory_recall_type,
            "recall_goal": semantic_route["recall_goal"],
            "timeline_query": semantic_route["timeline_query"],
            "conversation_action": semantic_route["conversation_action"],
            "event_recall_strategy": semantic_route["event_recall_strategy"],
            "confidence": 0.95,
            "reason": f"fake_pre_reply_decision:{semantic_route['reason'] or fast_reply_mode or 'default'}",
        }

    @staticmethod
    def _default_pre_reply_fast_reply_mode(message):
        text = str(message or "").strip()
        lowered = text.lower()
        if any(marker in text for marker in ("现在几点", "当前时间", "今天几号", "今天星期几")) or any(
            marker in lowered for marker in ("what time is it", "current time", "today's date")
        ):
            return "local_current_time"
        if any(marker in lowered for marker in ("new york", "london", "tokyo", "time difference")) or any(
            marker in text for marker in ("纽约", "伦敦", "东京", "时差")
        ):
            return "unsupported_world_time"
        return ""

    @staticmethod
    def _default_pre_reply_semantic_route(message):
        text = str(message or "").strip()
        lowered = text.lower()
        is_weather = "天气" in text or "weather" in lowered
        explicit_place_weather = bool(is_weather and "上海" in text)
        current_location = any(marker in text for marker in ("我这儿", "我这里", "这里", "这儿", "附近", "周边", "当前位置", "我在哪"))
        needs_profile = bool(
            any(marker in text for marker in ("我喜欢", "我不喜欢", "偏好", "以后推荐", "以后订", "优先考虑", "要避开", "订座", "靠窗", "吧台", "我叫什么", "我是谁"))
            or any(marker in text for marker in ("是不是喜欢", "是否喜欢", "喜欢很甜", "喜欢什么", "不喜欢什么"))
        )
        concept_question = any(marker in text for marker in ("是什么", "是啥", "什么意思", "怎么写", "如何写", "格式", "模板", "范文", "例子", "教程"))
        observation = any(marker in text for marker in ("最近在忙", "主要在忙", "工程偏好", "项目状态", "这周进展", "本周进展", "这周进展如何", "这周进展怎么样", "这阵子", "最近主要"))
        document_detail_query = any(marker in text for marker in ("上一份", "这份", "里有什么", "里面有什么"))
        weekly = (
            not concept_question
            and not document_detail_query
            and any(marker in text for marker in ("这周进展", "本周进展", "这周进展如何", "这周进展怎么样", "帮我写一份周报", "写一份周报", "周报"))
        )
        attention = any(marker in text for marker in ("要注意", "注意事项", "待办事项", "有什么待办", "要做什么", "最近要做什么", "接下来"))
        raw_timeline = any(marker in text for marker in ("原话", "之前说", "我说过", "提到过", "找一下我"))
        event_query = bool(
            attention
            or observation
            or any(marker in text for marker in ("吃了什么", "做了什么", "干了什么", "要干嘛", "有什么安排", "什么时候", "计划", "日程"))
            or text in {"最近", "近期"}
        )
        memory_recall_type = "none"
        recall_goal = "none"
        if raw_timeline:
            memory_recall_type = "timeline"
            recall_goal = "raw_evidence"
        elif observation or weekly:
            memory_recall_type = "observation"
            recall_goal = "summary"
        elif event_query:
            memory_recall_type = "event"
            recall_goal = "summary" if attention or text in {"最近", "近期"} else "specific_fact"
        elif needs_profile:
            memory_recall_type = "profile"
            recall_goal = "specific_fact"
        needs_web = bool(is_weather or any(marker in lowered for marker in ("新闻", "最新", "search", "latest")))
        web_query = None
        web_reason = ""
        if needs_web:
            web_query = "上海 天气 今天" if explicit_place_weather else "今日天气" if is_weather else text
            web_reason = "weather" if is_weather else "fake pre-reply realtime need"
        return {
            "needs_location": bool(current_location and not explicit_place_weather),
            "needs_web_search": needs_web,
            "web_query": web_query,
            "web_reason": web_reason,
            "needs_profile_memory": needs_profile and memory_recall_type == "profile",
            "needs_event_memory": memory_recall_type in {"event", "observation"},
            "needs_timeline_recall": memory_recall_type == "timeline",
            "memory_recall_type": memory_recall_type,
            "recall_goal": recall_goal,
            "timeline_query": text if raw_timeline else None,
            "conversation_action": "weekly_report" if weekly else "attention_items" if attention else "",
            "event_recall_strategy": (
                "observation_review" if observation or weekly else
                "attention_items" if attention and "注意" in text else
                "upcoming_plan" if (attention or "要干嘛" in text) and text not in {"最近", "近期"} else
                "ambiguous_recent_upcoming_plan" if text in {"最近", "近期"} else
                "text_search" if memory_recall_type == "event" else
                "skipped"
            ),
            "reason": "fake semantic route",
        }

    def _pre_reply_payload_for_message(self, message):
        if isinstance(self.semantic_payload, str) and self.semantic_payload_explicit:
            return self.semantic_payload
        if isinstance(self.pre_reply_payload, str) and self.pre_reply_payload_explicit:
            return self.pre_reply_payload

        if self.pre_reply_payload_explicit:
            route_payload = self._payload_for_message(self.pre_reply_payload, message)
        else:
            route_payload = self._default_pre_reply_payload_for_message(message)
        if not isinstance(route_payload, dict):
            route_payload = {}

        if self.semantic_payload_explicit:
            semantic_payload_value = self._payload_for_message(self.semantic_payload, message)
        else:
            semantic_payload_value = self._semantic_payload_from_intent_fixture(self.intent_payload, message)
        if not isinstance(semantic_payload_value, dict):
            semantic_payload_value = {
                "turn_intent": "chat",
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
                "reason": "fake default pre_reply semantic payload",
            }
        elif not self.semantic_payload_explicit and "raw_span:" not in str(message or ""):
            semantic_payload_value = {
                **semantic_payload_value,
                **self._default_pre_reply_memory_candidate(message),
            }

        payload = {**route_payload, **semantic_payload_value}
        for key in ("conversation_action", "event_recall_strategy"):
            if key in route_payload and key not in semantic_payload_value:
                payload[key] = route_payload[key]
        if self.pre_reply_payload_explicit:
            payload["confidence"] = route_payload.get("confidence", semantic_payload_value.get("confidence", 0.95))
        else:
            payload["confidence"] = semantic_payload_value.get("confidence", route_payload.get("confidence", 0.95))
        reasons = [str(route_payload.get("reason") or ""), str(semantic_payload_value.get("reason") or "")]
        payload["reason"] = ";".join(reason for reason in reasons if reason)
        return payload

    @staticmethod
    def _default_pre_reply_memory_candidate(message):
        user_message = FakeAgent._user_message_from_prompt(message)
        text = str(user_message or "").strip()
        lowered = text.lower()
        if any(marker in text for marker in ("纠正一下", "说错了", "更准确地说", "之前说的不对", "我之前说的不对")):
            return {}
        if any(marker in text for marker in ("不用记", "不要记", "别记", "先别记")):
            return {}
        if text.startswith("我叫"):
            return {
                "memory_action": "write",
                "memory_kind": "profile",
                "memory_type": "preference",
                "candidate_content": "用户名字叫" + text.removeprefix("我叫").strip(),
            }
        preference_prefixes = (
            ("其实我喜欢", "用户喜欢"),
            ("我其实更喜欢", "用户喜欢"),
            ("我更喜欢", "用户喜欢"),
            ("我喜欢", "用户喜欢"),
            ("我不喜欢", "用户不喜欢"),
        )
        for prefix, replacement in preference_prefixes:
            if text.startswith(prefix):
                return {
                    "memory_action": "write",
                    "memory_kind": "profile",
                    "memory_type": "preference",
                    "candidate_content": replacement + text.removeprefix(prefix).strip(),
                }
        if any(marker in lowered for marker in ("记一下", "记下来", "帮我记", "记录")) or any(marker in text for marker in ("开会", "检查 demo", "给妈妈打电话", "observation reflect")):
            content = text
            for marker in ("你能帮我记一下", "你先帮我记一下", "那个，先帮我留意记一下哈，", "记一下", "帮我记一下", "帮我记", "记录一下", "记录"):
                content = content.replace(marker, "")
            content = content.strip(" ，,。？?")
            if content.endswith("吗"):
                content = content[:-1].strip(" ，,。？?")
            return {
                "memory_action": "write",
                "memory_kind": "event",
                "memory_type": "task",
                "candidate_content": content,
            }
        if any(marker in text for marker in ("临时", "先别写成", "可能不是最终方案")):
            return {
                "memory_action": "write",
                "memory_kind": "event",
                "memory_type": "project_state",
                "candidate_content": text,
            }
        return {}

    @staticmethod
    def _user_message_from_prompt(message):
        marker = "User message:"
        user_message = str(message)
        if marker in user_message:
            user_message = user_message.rsplit(marker, 1)[-1].strip()
        return user_message

    @staticmethod
    def _reply_from_context(message):
        text = str(message or "")
        snippets: list[str] = []
        for header in (
            "Direct structured memory evidence:",
            "Reflected observations, useful as high-level hints but not sole proof:",
            "Background profile or stable context. Use only if it directly answers the user:",
            "Raw user timeline chunks recalled for this query:",
        ):
            snippets.extend(FakeAgent._context_lines_after_header(text, header))
        if snippets:
            return "；".join(snippets[:8])
        lowered = text.lower()
        if "api key" in lowered or "apikey" in lowered:
            return "我不会保存或回忆敏感凭据，比如 API Key。"
        if "手机号" in text:
            return "没有找到这条画像记忆。"
        return "主回复"

    @staticmethod
    def _context_lines_after_header(text: str, header: str) -> list[str]:
        if header not in text:
            return []
        section = text.split(header, 1)[1]
        stop_markers = (
            "\n\n",
            "\n</timeline-context>",
            "\nBackground profile",
            "\nReflected observations",
            "\nDirect structured",
            "\n<location-context>",
        )
        stop = len(section)
        for marker in stop_markers:
            idx = section.find(marker)
            if idx >= 0:
                stop = min(stop, idx)
        section = section[:stop]
        lines = []
        for raw_line in section.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("[") or line.startswith("<"):
                continue
            line = re.sub(r"^\d+\.\s*", "", line)
            line = re.sub(r"^(content|text):\s*", "", line)
            if line and not line.startswith(("id:", "kind:", "type:", "time:", "chunk_id:")):
                lines.append(line)
        return lines

    @staticmethod
    def _payload_for_message(payload, message):
        if not isinstance(payload, dict):
            return payload
        if any(key in payload for key in ("needs_web_search", "reply_mode", "answer_intent", "action", "is_correction")):
            return payload
        marker = "User message:"
        user_message = str(message)
        if marker in user_message:
            user_message = user_message.rsplit(marker, 1)[-1].strip()
        raw_span_marker = "raw_span:"
        if raw_span_marker in user_message:
            user_message = user_message.rsplit(raw_span_marker, 1)[-1].strip()
        if user_message in payload and isinstance(payload[user_message], dict):
            return payload[user_message]
        normalized_user_message = FakeAgent._normalize_fixture_key(user_message)
        for key, value in payload.items():
            if key == "*":
                continue
            if isinstance(value, dict) and FakeAgent._normalize_fixture_key(str(key)) in normalized_user_message:
                return value
        if "*" in payload and isinstance(payload["*"], dict):
            return payload["*"]
        return payload

    @staticmethod
    def _normalize_fixture_key(text: str) -> str:
        return (
            str(text or "")
            .replace("，", ",")
            .replace("：", ":")
            .replace("。", "")
            .replace("？", "?")
            .replace(" ", "")
            .strip()
        )

    @staticmethod
    def _semantic_payload_from_intent_fixture(intent_payload, message):
        payload = FakeAgent._payload_for_message(intent_payload, message)
        if not isinstance(payload, dict):
            return None
        candidates = payload.get("memory_write_candidates")
        if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
            user_message = FakeAgent._user_message_from_prompt(message)
            correction = any(
                marker in str(user_message or "")
                for marker in ("纠正一下", "说错了", "更准确地说", "之前说的不对", "我之前说的不对")
            )
            return {
                "turn_intent": "correction" if correction else "chat",
                "memory_action": "correction" if correction else "none",
                "memory_kind": "none",
                "memory_type": "none",
                "recall_type": "none",
                "reply_mode_hint": "llm",
                "flags": {
                    "transient": False,
                    "do_not_remember": False,
                    "correction": correction,
                    "explanation_query": False,
                },
                "candidate_content": "",
                "reason": "fake semantic fallback from empty intent fixture",
                "confidence": payload.get("confidence"),
            }
        candidate = candidates[0]
        kind = str(candidate.get("kind") or "none")
        memory_type = str(candidate.get("memory_type") or "")
        if not memory_type:
            memory_type = "preference" if kind in {"profile", "assistant_preference"} else "event"
        content = str(candidate.get("content") or "")
        transient = any(marker in content for marker in ("临时", "可能不是最终", "先别写成", "等我确认"))
        return {
            "turn_intent": "memory_write",
            "memory_action": "write",
            "memory_kind": kind,
            "memory_type": memory_type,
            "recall_type": "none",
            "reply_mode_hint": "llm",
            "flags": {
                "transient": transient,
                "do_not_remember": False,
                "correction": False,
                "explanation_query": False,
            },
            "candidate_content": content,
            "reason": str(candidate.get("reason") or "fake semantic payload from intent fixture"),
            "confidence": candidate.get("confidence", payload.get("confidence")),
        }


def semantic_payload(
    *,
    correction: bool,
    memory_action: str | None = None,
    memory_kind: str = "none",
    memory_type: str = "none",
    recall_type: str = "none",
    candidate_content: str = "",
    transient: bool = False,
    do_not_remember: bool = False,
    backend_reason: str = "fake test semantic gate",
) -> dict:
    action = memory_action or ("correction" if correction else "none")
    return {
        "turn_intent": "correction" if correction else "chat",
        "memory_action": action,
        "memory_kind": memory_kind,
        "memory_type": memory_type,
        "recall_type": recall_type,
        "reply_mode_hint": "llm",
        "flags": {
            "transient": transient,
            "do_not_remember": do_not_remember,
            "correction": correction,
            "explanation_query": False,
        },
        "candidate_content": candidate_content,
        "reason": backend_reason,
        "confidence": 0.93,
    }


def semantic_shadow_eval_case(
    *,
    name: str,
    message: str,
    semantic: dict | str | None,
    extractor_candidates: list[dict] | None = None,
    extractor_backend: str = "llm",
) -> dict:
    candidates = [
        MemoryWriteCandidate(
            content=str(candidate.get("content") or ""),
            kind=str(candidate.get("kind") or ""),
            memory_type=str(candidate.get("memory_type") or ""),
            privacy_level=str(candidate.get("privacy_level") or "normal"),
            confidence=candidate.get("confidence"),
            reason=str(candidate.get("reason") or ""),
        )
        for candidate in (extractor_candidates or [])
    ]
    turn_semantics = (
        {"backend": "llm", **semantic}
        if isinstance(semantic, dict)
        else {"backend": "rule_fallback", "error": str(semantic or "semantic_unavailable")}
    )
    shadow = GlassesChatService._unified_semantic_candidate_shadow(
        turn_semantics=turn_semantics,
        extracted=IntentDecision(backend=extractor_backend, memory_write_candidates=candidates),
    )
    return {
        "case": name,
        "message": message,
        "debug": {"routing": {"unified_semantic_candidate_shadow": shadow}},
    }


def semantic_shadow_eval_suite(cases: list[dict]) -> dict:
    records = [
        semantic_shadow_eval_case(
            name=str(case["name"]),
            message=str(case["message"]),
            semantic=case.get("semantic"),
            extractor_candidates=case.get("extractor_candidates") or [],
            extractor_backend=str(case.get("extractor_backend") or "llm"),
        )
        for case in cases
    ]
    return {
        "records": records,
        "summary": GlassesChatService.summarize_unified_semantic_candidate_shadow(records),
    }


# TimedFakeAgent 模拟工具调用链，用于验证 assistant_response_timing 不泄露参数正文。
class TimedFakeAgent(FakeAgent):
    def run_conversation(self, message, system_message=None, conversation_history=None, persist_user_message=None):
        self.main_calls += 1
        tool_args = {"query": "天气 北京", "limit": 1, "secret": "never-log-me"}
        tool_name = "example_agent_tool"
        if callable(getattr(self, "step_callback", None)):
            self.step_callback(1, [])
        if callable(getattr(self, "tool_start_callback", None)):
            self.tool_start_callback("call_1", tool_name, tool_args)
        if callable(getattr(self, "tool_complete_callback", None)):
            self.tool_complete_callback("call_1", tool_name, tool_args, "{\"success\": true}")
        if callable(getattr(self, "step_callback", None)):
            self.step_callback(
                2,
                [{
                    "name": tool_name,
                    "result": "{\"success\": true}",
                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                }],
            )
        return {
            "final_response": "主回复",
            "messages": [
                {"role": "user", "content": persist_user_message or message},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args, ensure_ascii=False),
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"success\": true}"},
                {"role": "assistant", "content": "主回复"},
            ],
            "api_calls": 2,
            "completed": True,
        }


class FailingMainAgent(FakeAgent):
    def run_conversation(self, message, system_message=None, conversation_history=None, persist_user_message=None):
        helper_markers = (
            "unified pre-reply decision classifier",
            "classify the user's intent",
            "answer synthesis planner",
            "text emotion classifier",
            "memory dedupe",
            "memory correction",
            "correction target",
            "temporal expression parser",
            "segment semantic filter",
        )
        if system_message and any(marker in system_message for marker in helper_markers):
            return super().run_conversation(
                message,
                system_message=system_message,
                conversation_history=conversation_history,
                persist_user_message=persist_user_message,
            )
        raise sqlite3.DatabaseError("database disk image is malformed")


class FakeASRRunner:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls: list[str] = []

    def transcribe_file(self, audio_path: Path):
        self.calls.append(str(audio_path))
        if self.result is None:
            from ai_glasses_memory_assistant.agent_bridge import LocalASRResult

            return LocalASRResult(
                ok=True,
                transcript="真实 ASR 转写结果",
                language="zh",
                latency_ms=42,
                model_name="SenseVoiceSmall",
                emotion_metadata={
                    "emotion_label": "平静",
                    "emotion_intensity": 1,
                    "emotion_score": None,
                    "emotion_candidates": [{"label": "平静", "score": None}],
                    "emotion_model": "sensevoice_small",
                    "emotion_evidence": "derived_from_sensevoice_token: NEUTRAL",
                    "emotion_source": "sensevoice_control_tokens",
                    "emotion_source_kind": "asr_token_hint",
                    "emotion_eligible_for_reply": False,
                    "emotion": {
                        "enabled": True,
                        "reason": "sensevoice_emotion_token_parsed",
                        "source": "sensevoice_control_tokens",
                        "model": "sensevoice_small",
                        "token": "NEUTRAL",
                    },
                },
            )
        return self.result


class FakeSpeakerRunner:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls: list[str] = []

    def analyze_file(self, audio_path: Path):
        self.calls.append(str(audio_path))
        if self.result is None:
            from ai_glasses_memory_assistant.agent_bridge import LocalSpeakerResult

            return LocalSpeakerResult(
                embedding=[1.0, 0.0, 0.0],
                hint="unknown",
                confidence=None,
                source="campp_diarization",
                evidence="speaker_embedding_extracted",
                enabled=True,
                model_name="cam++",
            )
        return self.result


class FakeEmotionRunner:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls: list[str] = []

    def analyze_file(self, audio_path: Path):
        self.calls.append(str(audio_path))
        if self.result is None:
            from ai_glasses_memory_assistant.agent_bridge import LocalEmotionResult

            return LocalEmotionResult(
                enabled=True,
                label="烦躁",
                intensity=4,
                score=0.86,
                candidates=[{"label": "烦躁", "score": 0.86}, {"label": "平静", "score": 0.09}],
                model_name="emotion2vec_plus_base",
                evidence="derived_from_acoustic_emotion_label: angry",
                source="acoustic_emotion_model",
                reason="acoustic_emotion_inferred",
            )
        return self.result


class FailingTextEmotionAgent(FakeAgent):
    def run_conversation(self, message, system_message=None, conversation_history=None, persist_user_message=None):
        if system_message and "text emotion classifier" in system_message:
            raise RuntimeError("text emotion classifier failed")
        return super().run_conversation(
            message,
            system_message=system_message,
            conversation_history=conversation_history,
            persist_user_message=persist_user_message,
        )


class SequenceSpeakerRunner:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = list(embeddings)
        self.calls: list[str] = []

    def analyze_file(self, audio_path: Path):
        self.calls.append(str(audio_path))
        from ai_glasses_memory_assistant.agent_bridge import LocalSpeakerResult

        embedding = self.embeddings.pop(0) if self.embeddings else [1.0, 0.0, 0.0]
        return LocalSpeakerResult(
            embedding=embedding,
            hint="unknown",
            confidence=None,
            source="campp_diarization",
            evidence="speaker_embedding_extracted",
            enabled=True,
            model_name="cam++",
        )


# FakeService 把后台线程改成同步执行，让测试能立即断言 job 和记忆状态。
class FakeService(GlassesChatService):
    def __init__(self, memory_store: EventMemoryStore, agent: FakeAgent | None = None, asr_runner=None, emotion_runner=None, speaker_runner=None) -> None:
        super().__init__(
            memory_store=memory_store,
            timeline_store=TimelineStore(db_path=memory_store.db_path.with_name("timeline.db")),
            clock=lambda: 1778131200.0,
        )
        self.fake_agent = agent or FakeAgent()
        self.new_session_calls = 0
        if asr_runner is not None:
            self.audio_processor.asr_runner = asr_runner
        if emotion_runner is not None:
            self.audio_processor.emotion_runner = emotion_runner
        if speaker_runner is not None:
            self.audio_processor.speaker_runner = speaker_runner

    def _new_session(self, *, user_id: str, session_id: str | None = None) -> ChatSession:
        self.new_session_calls += 1
        return ChatSession(id=session_id or "fake-session", agent=self.fake_agent)

    def _start_background_candidate_write(self, **kwargs) -> None:
        self._process_candidates_background(**kwargs)

    def _start_background_llm_memory_extraction(self, **kwargs) -> None:
        self._process_llm_memory_extraction_background(**kwargs)

    def _start_background_long_input_processing(self, **kwargs) -> None:
        self._process_long_input_background(**kwargs)

    def _start_background_observation_reflect(self, **kwargs) -> None:
        self._process_observation_reflect_background(**kwargs)


# FailingMemoryStore 专门模拟后台写库失败，验证回复不被失败打断。
class FailingMemoryStore(EventMemoryStore):
    def add_memory(self, *args, **kwargs):
        raise RuntimeError("simulated write failure")


class SnapshotFailingMemoryStore(EventMemoryStore):
    def list_memories(self, *args, **kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")


class RecallFailingMemoryStore(EventMemoryStore):
    def list_events_between(self, *args, **kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")


class AgentBridgePolicyTests(unittest.TestCase):
    # 每个测试使用临时 SQLite，确保用户记忆和 audit 不污染真实 HERMES_HOME。
    def make_store(self, tmp: Path) -> EventMemoryStore:
        return EventMemoryStore(db_path=tmp / "events.db")

    # 标准库 handler 测试直接构造 handler 对象，避免真的启动端口。
    def http_get(self, handler_cls, path: str) -> tuple[int, dict[str, str], dict]:
        handler = handler_cls.__new__(handler_cls)
        handler.rfile = io.BytesIO()
        handler.wfile = io.BytesIO()
        handler.headers = {}
        handler.path = path
        handler.command = "GET"
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"GET {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 0)
        handler.server = types.SimpleNamespace(server_version="test", sys_version="")
        handler.log_request = lambda *args, **kwargs: None

        handler.do_GET()

        response = handler.wfile.getvalue()
        head, response_body = response.split(b"\r\n\r\n", 1)
        lines = head.split(b"\r\n")
        status = int(lines[0].split()[1])
        headers = BytesParser(policy=default).parsebytes(b"\r\n".join(lines[1:]) + b"\r\n\r\n")
        return status, dict(headers.items()), json.loads(response_body.decode("utf-8"))

    def http_post(self, handler_cls, path: str, payload: dict) -> tuple[int, dict[str, str], dict]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler = handler_cls.__new__(handler_cls)
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = path
        handler.command = "POST"
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"POST {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 0)
        handler.server = types.SimpleNamespace(server_version="test", sys_version="")
        handler.log_request = lambda *args, **kwargs: None

        handler.do_POST()

        response = handler.wfile.getvalue()
        head, response_body = response.split(b"\r\n\r\n", 1)
        lines = head.split(b"\r\n")
        status = int(lines[0].split()[1])
        headers = BytesParser(policy=default).parsebytes(b"\r\n".join(lines[1:]) + b"\r\n\r\n")
        return status, dict(headers.items()), json.loads(response_body.decode("utf-8"))

    def http_delete(self, handler_cls, path: str) -> tuple[int, dict[str, str], dict]:
        handler = handler_cls.__new__(handler_cls)
        handler.rfile = io.BytesIO()
        handler.wfile = io.BytesIO()
        handler.headers = {}
        handler.path = path
        handler.command = "DELETE"
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"DELETE {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 0)
        handler.server = types.SimpleNamespace(server_version="test", sys_version="")
        handler.log_request = lambda *args, **kwargs: None

        handler.do_DELETE()

        response = handler.wfile.getvalue()
        head, response_body = response.split(b"\r\n\r\n", 1)
        lines = head.split(b"\r\n")
        status = int(lines[0].split()[1])
        headers = BytesParser(policy=default).parsebytes(b"\r\n".join(lines[1:]) + b"\r\n\r\n")
        return status, dict(headers.items()), json.loads(response_body.decode("utf-8"))

    def test_agent_bridge_has_no_top_level_hermes_fallback_imports(self) -> None:
        source_path = PACKAGE_ROOT / "ai_glasses_memory_assistant" / "agent_bridge.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden_modules = ("hermes_constants", "hermes_cli", "run_agent")
        top_level_imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_imports.append(node.module)

        offenders = [
            module
            for module in top_level_imports
            if any(module == forbidden or module.startswith(f"{forbidden}.") for forbidden in forbidden_modules)
        ]
        self.assertEqual(offenders, [])

    def test_new_session_defaults_to_openai_compatible_without_hermes_toolsets(self) -> None:
        captured: dict = {}

        class FakeCompletions:
            def create(self, **kwargs):
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))])

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = FakeOpenAI
        env = {
            "AI_GLASSES_LLM_PROVIDER": "deepseek",
            "AI_GLASSES_LLM_MODEL": "deepseek-v4-flash",
            "AI_GLASSES_LLM_BASE_URL": "https://api.deepseek.com",
            "AI_GLASSES_LLM_API_KEY": "deepseek-key",
            "AI_GLASSES_LLM_API_MODE": "chat_completions",
        }

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {**env, "AI_GLASSES_HOME": tmpdir}, clear=True), patch.dict(
            sys.modules,
            {"openai": openai_module},
        ):
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))
            session = service._new_session(user_id="u1", session_id="s1")

        self.assertEqual(session.id, "s1")
        self.assertEqual(session.agent.enabled_toolsets, [])
        self.assertEqual(session.agent.model, "deepseek-v4-flash")
        self.assertEqual(session.agent.provider, "deepseek")
        self.assertEqual(session.agent.base_url, "https://api.deepseek.com")
        self.assertEqual(session.agent.api_mode, "chat_completions")
        self.assertEqual(captured["api_key"], "deepseek-key")
        self.assertEqual(captured["base_url"], "https://api.deepseek.com")

    def test_new_session_openai_backend_does_not_touch_hermes_fallback_modules(self) -> None:
        captured: dict = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: None))

        def fail_if_called(*args, **kwargs):
            raise AssertionError("Hermes fallback should not be touched by the default backend")

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = FakeOpenAI
        runtime_provider = types.ModuleType("hermes_cli.runtime_provider")
        runtime_provider.resolve_runtime_provider = fail_if_called
        env_loader = types.ModuleType("hermes_cli.env_loader")
        env_loader.load_hermes_dotenv = fail_if_called
        run_agent = types.ModuleType("run_agent")
        run_agent.AIAgent = fail_if_called
        env = {
            "AI_GLASSES_LLM_PROVIDER": "deepseek",
            "AI_GLASSES_LLM_MODEL": "deepseek-v4-flash",
            "AI_GLASSES_LLM_BASE_URL": "https://api.deepseek.com",
            "AI_GLASSES_LLM_API_KEY": "deepseek-key",
            "AI_GLASSES_LLM_API_MODE": "chat_completions",
        }

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {**env, "AI_GLASSES_HOME": tmpdir}, clear=True), patch.dict(
            sys.modules,
            {
                "openai": openai_module,
                "hermes_cli.runtime_provider": runtime_provider,
                "hermes_cli.env_loader": env_loader,
                "run_agent": run_agent,
            },
        ):
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))
            session = service._new_session(user_id="u1", session_id="s1")

        self.assertEqual(session.id, "s1")
        self.assertEqual(session.agent.provider, "deepseek")
        self.assertEqual(captured["api_key"], "deepseek-key")

    def test_new_session_openai_compatible_uses_deepseek_api_key_fallback(self) -> None:
        captured: dict = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: None))

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = FakeOpenAI
        env = {
            "AI_GLASSES_LLM_PROVIDER": "deepseek",
            "AI_GLASSES_LLM_MODEL": "deepseek-v4-flash",
            "AI_GLASSES_LLM_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_API_KEY": "deepseek-env-key",
        }

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {**env, "AI_GLASSES_HOME": tmpdir}, clear=True), patch.dict(
            sys.modules,
            {"openai": openai_module},
        ):
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))
            service._new_session(user_id="u1", session_id="s1")

        self.assertEqual(captured["api_key"], "deepseek-env-key")

    def test_new_session_openai_compatible_reports_missing_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"AI_GLASSES_HOME": tmpdir}, clear=True):
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))
            with self.assertRaisesRegex(ValueError, "OpenAI-compatible LLM backend requires"):
                service._new_session(user_id="u1", session_id="s1")

    def test_new_session_hermes_backend_requires_legacy_opt_in(self) -> None:
        def fail_if_called(*args, **kwargs):
            raise AssertionError("Sealed Hermes fallback should not import runtime modules")

        runtime_provider = types.ModuleType("hermes_cli.runtime_provider")
        runtime_provider.resolve_runtime_provider = fail_if_called
        env_loader = types.ModuleType("hermes_cli.env_loader")
        env_loader.load_hermes_dotenv = fail_if_called
        run_agent = types.ModuleType("run_agent")
        run_agent.AIAgent = fail_if_called
        env = {"AI_GLASSES_LLM_BACKEND": "hermes"}

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {**env, "AI_GLASSES_HOME": tmpdir}, clear=True), patch.dict(
            sys.modules,
            {
                "hermes_cli.runtime_provider": runtime_provider,
                "hermes_cli.env_loader": env_loader,
                "run_agent": run_agent,
            },
        ):
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))
            with self.assertRaisesRegex(ValueError, "sealed legacy fallback"):
                service._new_session(user_id="u1", session_id="s1")

    def test_new_session_hermes_backend_keeps_legacy_aia_agent_parameters(self) -> None:
        captured: dict = {}
        resolve_calls: list[dict] = []

        class CapturingAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.enabled_toolsets = kwargs.get("enabled_toolsets")

        runtime_provider = types.ModuleType("hermes_cli.runtime_provider")
        def resolve_runtime_provider(**kwargs):
            resolve_calls.append(kwargs)
            return {
                "provider": "fake",
                "base_url": "https://example.invalid",
                "api_key": "fake-key",
                "api_mode": "chat_completions",
            }
        runtime_provider.resolve_runtime_provider = resolve_runtime_provider
        env_loader = types.ModuleType("hermes_cli.env_loader")
        env_loader.load_hermes_dotenv = lambda **kwargs: None
        run_agent = types.ModuleType("run_agent")
        run_agent.AIAgent = CapturingAgent

        env = {
            "AI_GLASSES_LLM_BACKEND": "hermes",
            "AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK": "1",
            "HERMES_HOME": "",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {**env, "HERMES_HOME": tmpdir}, clear=True), patch.dict(
            sys.modules,
            {
                "hermes_cli.runtime_provider": runtime_provider,
                "hermes_cli.env_loader": env_loader,
                "run_agent": run_agent,
            },
        ):
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))
            session = service._new_session(user_id="u1", session_id="s1")

        self.assertEqual(session.id, "s1")
        self.assertEqual(session.agent.enabled_toolsets, [])
        self.assertEqual(captured["model"], "deepseek-chat")
        self.assertEqual(captured["enabled_toolsets"], [])
        self.assertTrue(captured["skip_context_files"])
        self.assertTrue(captured["skip_memory"])
        self.assertIsNone(captured["reasoning_config"])
        self.assertEqual(type(captured["session_db"]).__name__, "AppSessionStore")
        self.assertEqual(resolve_calls[0]["requested"], "deepseek")
        self.assertEqual(resolve_calls[0]["target_model"], "deepseek-chat")

    def test_session_store_records_basic_session_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"AI_GLASSES_HOME": tmpdir}, clear=True):
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))
            service.session_db.create_session(
                session_id="s1",
                source="ai_glasses_web",
                model="deepseek-v4-flash",
                system_prompt="system",
                user_id="u1",
            )
            service.session_db.append_message(session_id="s1", role="user", content="你好")
            service.session_db.update_system_prompt("s1", "updated system")
            service.session_db.update_token_counts("s1", input_tokens=3, output_tokens=5, api_call_count=1)

            session = service.session_db.get_session("s1")
            messages = service.session_db.get_messages("s1")

        self.assertEqual(session["system_prompt"], "updated system")
        self.assertEqual(session["message_count"], 1)
        self.assertEqual(session["input_tokens"], 3)
        self.assertEqual(session["output_tokens"], 5)
        self.assertEqual(session["api_call_count"], 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "你好")

    def test_ai_glasses_home_owns_service_data_audit_and_sessions_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"AI_GLASSES_HOME": tmpdir}, clear=True):
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))

            service.session_db.create_session(session_id="s1", source="ai_glasses_web")
            service._append_audit_record({"record_type": "startup_probe", "user_id": "u1"})

            data_dir = Path(tmpdir) / "data"
            self.assertEqual(service.data_dir, data_dir)
            self.assertEqual(service.audit_path, data_dir / "chat_audit.jsonl")
            self.assertEqual(service.session_db.db_path, data_dir / "sessions.db")
            self.assertTrue(service.audit_path.exists())
            self.assertTrue(service.session_db.db_path.exists())

    def test_new_session_uses_local_ollama_env_config(self) -> None:
        captured: dict = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: None))

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = FakeOpenAI

        env = {
            "HERMES_HOME": "",
            "AI_GLASSES_LLM_PROVIDER": "custom",
            "AI_GLASSES_LLM_MODEL": "qwen3:8b",
            "AI_GLASSES_LLM_BASE_URL": "http://127.0.0.1:11434/v1",
            "AI_GLASSES_LLM_API_KEY": "ollama",
            "AI_GLASSES_LLM_API_MODE": "chat_completions",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {**env, "HERMES_HOME": tmpdir}, clear=True), patch.dict(
            sys.modules,
            {"openai": openai_module},
        ):
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))
            session = service._new_session(user_id="u1", session_id="s1")

        self.assertEqual(session.id, "s1")
        self.assertEqual(session.agent.model, "qwen3:8b")
        self.assertEqual(session.agent.provider, "custom")
        self.assertEqual(session.agent.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(session.agent.api_key, "ollama")
        self.assertEqual(session.agent.api_mode, "chat_completions")
        self.assertEqual(captured["base_url"], "http://127.0.0.1:11434/v1")
        self.assertEqual(captured["api_key"], "ollama")

    def test_new_session_loads_app_home_dotenv_config_for_openai_backend(self) -> None:
        captured: dict = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: None))

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = FakeOpenAI

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {
                "AI_GLASSES_HOME": tmpdir,
                "HERMES_HOME": tmpdir,
            },
            clear=True,
        ), patch.dict(
            sys.modules,
            {"openai": openai_module},
        ):
            (Path(tmpdir) / ".env").write_text(
                "\n".join([
                    "AI_GLASSES_LLM_PROVIDER=deepseek",
                    "AI_GLASSES_LLM_MODEL=dotenv-model",
                    "AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com",
                    "AI_GLASSES_LLM_API_KEY=dotenv-key",
                    "AI_GLASSES_LLM_API_MODE=chat_completions",
                ]),
                encoding="utf-8",
            )
            service = GlassesChatService(memory_store=self.make_store(Path(tmpdir)))
            session = service._new_session(user_id="u1", session_id="s1")

        self.assertEqual(session.id, "s1")
        self.assertEqual(session.agent.model, "dotenv-model")
        self.assertEqual(session.agent.provider, "deepseek")
        self.assertEqual(session.agent.base_url, "https://api.deepseek.com")
        self.assertEqual(session.agent.api_mode, "chat_completions")
        self.assertEqual(captured["api_key"], "dotenv-key")

    def test_service_ignores_legacy_routing_mode_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}, clear=True):
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "none",
                "recall_goal": "none",
                "confidence": 0.95,
            })
            service = FakeService(self.make_store(Path(tmpdir)), agent=agent)
            result = service.chat("普通问答：水的化学式是什么？", user_id="u1", routing_mode="legacy_mode")

            self.assertEqual(result["debug"]["routing"]["mode"], "llm_first")
            self.assertNotIn("planner_enabled", result["debug"]["routing"])
            self.assertNotIn("pre_reply_forced", result["debug"]["routing"])

    def test_standard_library_runtime_endpoint_returns_fixed_routing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}, clear=True):
            from ai_glasses_memory_assistant.server import GlassesHandler

            status, _, payload = self.http_get(GlassesHandler, "/api/runtime")

        self.assertEqual(status, 200)
        self.assertEqual(payload["routing_mode"], "llm_first")
        self.assertEqual(sorted(payload), ["routing_mode"])

    def test_frontend_default_chat_without_routing_mode_uses_llm_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}, clear=True):
            from ai_glasses_memory_assistant.server import GlassesHandler

            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks recent activity",
            }))
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, _, payload = self.http_post(
                    GlassesHandler,
                    "/api/chat",
                    {
                        "message": "普通问答：水的化学式是什么？",
                        "user_id": "u1",
                    },
                )
            finally:
                GlassesHandler.service = original_service

        self.assertEqual(status, 200)
        self.assertEqual(payload["debug"]["routing"]["mode"], "llm_first")

    def test_standard_library_chat_accepts_ambient_capture_id(self) -> None:
        from ai_glasses_memory_assistant.server import GlassesHandler

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}, clear=True):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "needs_timeline_recall": False,
                "confidence": 0.9,
                "reason": "wake query should use ambient context",
            })
            service = FakeService(store, agent=agent)
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            appended = service.append_capture_chunk(
                user_id="u1",
                capture_id=capture["capture_id"],
                text="服了，又来了",
            )
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, _, payload = self.http_post(
                    GlassesHandler,
                    "/api/chat",
                    {
                        "message": "你觉得刚才他是不是在阴阳我？",
                        "user_id": "u1",
                        "ambient_capture_id": capture["capture_id"],
                        "wake_session": {
                            "ambient_capture_id": capture["capture_id"],
                            "wake_detected_at": 1778131211.0,
                            "pre_wake_segment_ids": [appended["chunk_id"]],
                            "post_wake_query_segment_ids": [],
                            "wake_query_text": "你觉得刚才他是不是在阴阳我？",
                            "wake_mode": "button",
                            "status": "consumed",
                        },
                    },
                )
            finally:
                GlassesHandler.service = original_service

        self.assertEqual(status, 200)
        self.assertTrue(payload["debug"]["ambient_context"]["used"])
        self.assertEqual(payload["debug"]["ambient_context"]["chunk_ids"], [appended["chunk_id"]])
        self.assertEqual(payload["debug"]["ambient_context"]["pre_wake_segment_ids"], [appended["chunk_id"]])
        self.assertIn("服了，又来了", agent.main_messages[-1])

    def test_audio_segment_process_success_discards_audio_without_capture_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))

            result = service.process_audio_segment(user_id="u1", transcript_hint="服了，又来了")

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["transcript"], "服了，又来了")
        self.assertEqual(result["audio_retention"], "discarded_after_processing")
        self.assertEqual(result["metadata"]["source_type"], "ambient_audio")
        self.assertFalse(result["debug"]["audio_processing"]["capture_appended"])
        self.assertNotIn("capture_append", result)
        serialized = json.dumps(result, ensure_ascii=False)
        for forbidden in ("audio_path", "temp_path", "file_name", "binary"):
            self.assertNotIn(forbidden, serialized)

    def test_audio_segment_process_accepts_real_audio_payload_and_discards_it(self) -> None:
        audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))

            result = service.process_audio_segment(
                user_id="u1",
                transcript_hint="服了，又来了",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["audio_retention"], "discarded_after_processing")
        self.assertTrue(result["metadata"]["audio_input"]["received"])
        self.assertEqual(result["metadata"]["audio_input"]["mime_type"], "audio/wav")
        self.assertEqual(result["metadata"]["audio_input"]["retention"], "discarded_after_processing")
        serialized = json.dumps(result, ensure_ascii=False)
        for forbidden in ("audio_path", "temp_path", "file_name", "binary", "ambient_audio_"):
            self.assertNotIn(forbidden, serialized)

    def test_audio_segment_process_uses_local_asr_transcript_when_available(self) -> None:
        audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)), asr_runner=FakeASRRunner())

            result = service.process_audio_segment(
                user_id="u1",
                transcript_hint="浏览器兜底文本",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["transcript"], "真实 ASR 转写结果")
        self.assertTrue(result["metadata"]["asr"]["enabled"])
        self.assertEqual(result["metadata"]["asr"]["mode"], "local_sensevoice")
        self.assertEqual(result["debug"]["audio_processing"]["asr_backend"], "local_sensevoice")
        self.assertFalse(result["debug"]["audio_processing"]["fallback_used"])
        self.assertEqual(result["metadata"]["emotion_label"], "平静")
        self.assertEqual(result["metadata"]["emotion_source"], "sensevoice_control_tokens")
        self.assertTrue(result["metadata"]["emotion"]["enabled"])
        self.assertEqual(result["metadata"]["emotion_source_kind"], "asr_token_hint")
        self.assertFalse(result["metadata"]["emotion_eligible_for_reply"])

    def test_audio_segment_process_falls_back_to_transcript_hint_when_asr_fails(self) -> None:
        audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        from ai_glasses_memory_assistant.agent_bridge import LocalASRResult

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            runner = FakeASRRunner(result=LocalASRResult(ok=False, error_type="asr_runtime_failed"))
            service = FakeService(self.make_store(Path(tmpdir)), asr_runner=runner)

            result = service.process_audio_segment(
                user_id="u1",
                transcript_hint="浏览器兜底文本",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["transcript"], "浏览器兜底文本")
        self.assertTrue(result["debug"]["audio_processing"]["fallback_used"])
        self.assertEqual(result["metadata"]["asr"]["mode"], "fallback_transcript_hint")

    def test_audio_segment_process_fails_without_fallback_when_local_asr_fails(self) -> None:
        audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        from ai_glasses_memory_assistant.agent_bridge import LocalASRResult

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            runner = FakeASRRunner(result=LocalASRResult(ok=False, error_type="asr_runtime_failed"))
            service = FakeService(self.make_store(Path(tmpdir)), asr_runner=runner)

            result = service.process_audio_segment(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "asr_runtime_failed")
        self.assertFalse(result["debug"]["audio_processing"]["capture_appended"])

    def test_audio_segment_process_reports_missing_model_dir_without_leaking_path(self) -> None:
        audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": str(Path(tmpdir) / "missing-model")}):
            service = FakeService(self.make_store(Path(tmpdir)))

            result = service.process_audio_segment(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "asr_model_dir_not_found")
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("missing-model", serialized)

    def test_audio_segment_process_rejects_invalid_audio_metadata(self) -> None:
        audio_base64 = base64.b64encode(b"fake-audio").decode("ascii")
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))

            with self.assertRaises(ValueError):
                service.process_audio_segment(
                    user_id="u1",
                    transcript_hint="服了，又来了",
                    audio_base64=audio_base64,
                    audio_mime_type="audio/unsupported",
                    audio_duration_ms=1200,
                )
            with self.assertRaises(ValueError):
                service.process_audio_segment(
                    user_id="u1",
                    transcript_hint="服了，又来了",
                    audio_base64=audio_base64,
                    audio_mime_type="audio/wav",
                    audio_duration_ms=60_000,
                )

    def test_audio_segment_process_appends_derived_text_to_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="音频骨架")

            result = service.process_audio_segment(
                user_id="u1",
                capture_id=capture["capture_id"],
                transcript_hint="这个你应该早就知道吧",
                timestamp=1778131210.0,
                emotion_metadata={"emotion_label": "烦躁", "emotion_intensity": 4},
            )
            reloaded = service.timeline_store.get_capture("u1", capture["capture_id"])

        self.assertEqual(result["status"], "processed")
        self.assertTrue(result["debug"]["audio_processing"]["capture_appended"])
        self.assertEqual(result["debug"]["audio_processing"]["source_type"], "ambient_audio")
        self.assertTrue(str(result["debug"]["audio_processing"]["segment_id"]).startswith("seg_"))
        self.assertEqual(result["capture_append"]["chunk_count"], 1)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded["chunks"][0]["text"], "这个你应该早就知道吧")
        self.assertEqual(reloaded["chunks"][0]["metadata"]["source_type"], "ambient_audio")
        self.assertEqual(reloaded["chunks"][0]["metadata"]["audio_retention"], "discarded_after_processing")
        self.assertEqual(reloaded["chunks"][0]["metadata"]["processing_state"], "ready_for_wake_context")
        self.assertEqual(reloaded["chunks"][0]["metadata"]["captured_at"], 1778131210.0)
        self.assertTrue(str(reloaded["chunks"][0]["metadata"]["segment_id"]).startswith("seg_"))
        self.assertEqual(reloaded["chunks"][0]["metadata"]["emotion_label"], "烦躁")

    def test_sensevoice_emotion_metadata_is_preserved_on_capture_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            service = FakeService(
                self.make_store(Path(tmpdir)),
                asr_runner=FakeASRRunner(),
                emotion_runner=FakeEmotionRunner(),
                speaker_runner=FakeSpeakerRunner(),
            )
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="sensevoice emotion")
            audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")

            result = service.process_audio_segment(
                user_id="u1",
                capture_id=capture["capture_id"],
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )
            reloaded = service.timeline_store.get_capture("u1", capture["capture_id"])

        self.assertEqual(result["metadata"]["emotion_label"], "烦躁")
        self.assertEqual(result["metadata"]["emotion_model"], "emotion2vec_plus_base")
        self.assertEqual(result["metadata"]["emotion_source"], "acoustic_emotion_model")
        self.assertEqual(result["metadata"]["emotion_source_kind"], "acoustic_model")
        self.assertTrue(result["metadata"]["emotion_eligible_for_reply"])
        self.assertEqual(reloaded["chunks"][0]["metadata"]["emotion_label"], "烦躁")
        self.assertEqual(reloaded["chunks"][0]["metadata"]["emotion_evidence"], "derived_from_acoustic_emotion_label: angry")
        self.assertEqual(result["debug"]["audio_processing"]["emotion_backend"], "emotion2vec_plus_base")
        self.assertTrue(result["debug"]["audio_processing"]["emotion_enabled"])
        self.assertTrue(result["debug"]["audio_processing"]["emotion_eligible_for_reply"])

    def test_audio_segment_process_falls_back_to_sensevoice_emotion_when_acoustic_model_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            from ai_glasses_memory_assistant.agent_bridge import LocalEmotionResult

            service = FakeService(
                self.make_store(Path(tmpdir)),
                asr_runner=FakeASRRunner(),
                emotion_runner=FakeEmotionRunner(
                    result=LocalEmotionResult(
                        enabled=False,
                        model_name="emotion2vec_plus_base",
                        source="acoustic_emotion_model",
                        reason="emotion_model_not_configured",
                        error_type="emotion_model_dir_missing",
                    )
                ),
                speaker_runner=FakeSpeakerRunner(),
            )
            audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")

            result = service.process_audio_segment(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )

        self.assertEqual(result["metadata"]["emotion_label"], "平静")
        self.assertEqual(result["metadata"]["emotion_model"], "sensevoice_small")
        self.assertEqual(result["metadata"]["emotion_source"], "sensevoice_control_tokens")
        self.assertEqual(result["metadata"]["emotion_source_kind"], "asr_token_hint")
        self.assertFalse(result["metadata"]["emotion_eligible_for_reply"])
        self.assertEqual(result["debug"]["audio_processing"]["emotion_backend"], "sensevoice_control_tokens_fallback")
        self.assertTrue(result["debug"]["audio_processing"]["emotion_enabled"])

    def test_audio_segment_process_preserves_real_speaker_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            service = FakeService(
                self.make_store(Path(tmpdir)),
                asr_runner=FakeASRRunner(),
                speaker_runner=FakeSpeakerRunner(),
            )
            service.timeline_store.upsert_speaker_profile(
                user_id="u1",
                embedding=[1.0, 0.0, 0.0],
                model_name="cam++",
                sample_count=3,
                target_sample_count=3,
                calibration_status="calibrated",
                user_threshold=0.72,
                other_threshold=0.50,
            )
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="speaker metadata")
            audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")

            result = service.process_audio_segment(
                user_id="u1",
                capture_id=capture["capture_id"],
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )
            reloaded = service.timeline_store.get_capture("u1", capture["capture_id"])

        self.assertEqual(result["metadata"]["speaker_hint"], "user")
        self.assertEqual(result["metadata"]["speaker_source"], "campp_diarization")
        self.assertEqual(result["metadata"]["speaker_evidence"], "speaker_similarity_user_match:1.0000")
        self.assertTrue(result["metadata"]["speaker_reference_available"])
        self.assertEqual(result["metadata"]["speaker_similarity"], 1.0)
        self.assertEqual(result["metadata"]["speaker_profile_sample_count"], 3)
        self.assertTrue(result["metadata"]["speaker_profile_calibrated"])
        self.assertTrue(result["debug"]["audio_processing"]["speaker_enabled"])
        self.assertEqual(reloaded["chunks"][0]["metadata"]["speaker_hint"], "user")
        self.assertEqual(reloaded["chunks"][0]["metadata"]["speaker_source"], "campp_diarization")

    def test_audio_segment_process_keeps_asr_when_speaker_model_is_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            service = FakeService(
                self.make_store(Path(tmpdir)),
                asr_runner=FakeASRRunner(),
            )
            audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")

            result = service.process_audio_segment(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["transcript"], "真实 ASR 转写结果")
        self.assertEqual(result["metadata"]["speaker_hint"], "unknown")
        self.assertEqual(result["metadata"]["speaker_evidence"], "speaker_model_not_configured")
        self.assertFalse(result["debug"]["audio_processing"]["speaker_enabled"])

    def test_audio_segment_process_without_reference_keeps_speaker_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            service = FakeService(
                self.make_store(Path(tmpdir)),
                asr_runner=FakeASRRunner(),
                speaker_runner=FakeSpeakerRunner(),
            )
            audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")

            result = service.process_audio_segment(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )

        self.assertEqual(result["metadata"]["speaker_hint"], "unknown")
        self.assertEqual(result["metadata"]["speaker_evidence"], "speaker_reference_missing")
        self.assertFalse(result["metadata"]["speaker_reference_available"])
        self.assertIsNone(result["metadata"]["speaker_similarity"])

    def test_audio_segment_process_marks_other_when_similarity_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            from ai_glasses_memory_assistant.agent_bridge import LocalSpeakerResult

            service = FakeService(
                self.make_store(Path(tmpdir)),
                asr_runner=FakeASRRunner(),
                speaker_runner=FakeSpeakerRunner(
                    result=LocalSpeakerResult(
                        embedding=[0.0, 1.0, 0.0],
                        hint="unknown",
                        confidence=None,
                        source="campp_diarization",
                        evidence="speaker_embedding_extracted",
                        enabled=True,
                        model_name="cam++",
                    )
                ),
            )
            service.timeline_store.upsert_speaker_profile(
                user_id="u1",
                embedding=[1.0, 0.0, 0.0],
                model_name="cam++",
                sample_count=3,
                target_sample_count=3,
                calibration_status="calibrated",
                user_threshold=0.72,
                other_threshold=0.50,
            )
            audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")

            result = service.process_audio_segment(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )

        self.assertEqual(result["metadata"]["speaker_hint"], "other")
        self.assertEqual(result["metadata"]["speaker_evidence"], "speaker_similarity_other_reject:0.0000")

    def test_speaker_enrollment_finalizes_after_three_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(
                self.make_store(Path(tmpdir)),
                speaker_runner=SequenceSpeakerRunner([
                    [1.0, 0.0, 0.0],
                    [0.98, 0.02, 0.0],
                    [0.97, 0.03, 0.0],
                ]),
            )
            audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")

            first = service.enroll_speaker_profile(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
                enrollment_session_id="sess-1",
                sample_index=1,
                sample_total=3,
                finalize=False,
            )
            second = service.enroll_speaker_profile(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
                enrollment_session_id="sess-1",
                sample_index=2,
                sample_total=3,
                finalize=False,
            )
            final = service.enroll_speaker_profile(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
                enrollment_session_id="sess-1",
                sample_index=3,
                sample_total=3,
                finalize=True,
            )
            profile = service.get_speaker_profile(user_id="u1")

        self.assertEqual(first["status"], "pending")
        self.assertEqual(first["sample_count"], 1)
        self.assertEqual(second["status"], "pending")
        self.assertEqual(second["sample_count"], 2)
        self.assertEqual(final["status"], "ok")
        self.assertTrue(final["enrolled"])
        self.assertEqual(final["sample_count"], 3)
        self.assertEqual(final["calibration_status"], "calibrated")
        self.assertEqual(profile["enrolled"], True)
        self.assertEqual(profile["sample_count"], 3)
        self.assertEqual(profile["calibration_status"], "calibrated")
        self.assertEqual(profile["speaker_model"], "cam++")

    def test_cancel_speaker_enrollment_keeps_old_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(
                self.make_store(Path(tmpdir)),
                speaker_runner=FakeSpeakerRunner(),
            )
            service.timeline_store.upsert_speaker_profile(
                user_id="u1",
                embedding=[1.0, 0.0, 0.0],
                model_name="cam++",
                sample_count=3,
                target_sample_count=3,
                calibration_status="calibrated",
                user_threshold=0.72,
                other_threshold=0.50,
            )
            audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")

            pending = service.enroll_speaker_profile(
                user_id="u1",
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
                enrollment_session_id="sess-cancel",
                sample_index=1,
                sample_total=3,
                finalize=False,
            )
            cancelled = service.cancel_speaker_enrollment(user_id="u1", enrollment_session_id="sess-cancel")
            profile = service.get_speaker_profile(user_id="u1")

        self.assertEqual(pending["status"], "pending")
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["cleared_sample_count"], 1)
        self.assertTrue(profile["enrolled"])
        self.assertEqual(profile["calibration_status"], "calibrated")

    def test_sensevoice_runner_extracts_unknown_when_emotion_token_missing(self) -> None:
        payload = SenseVoiceASRRunner._extract_sensevoice_emotion_metadata({"text": "<|zh|><|Speech|><|withitn|>行吧，知道了"})
        self.assertFalse(payload["emotion"]["enabled"])
        self.assertEqual(payload["emotion"]["reason"], "sensevoice_emotion_token_missing")

    def test_audio_segment_process_accepts_wake_query_source_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))

            result = service.process_audio_segment(
                user_id="u1",
                source_type="wake_query",
                transcript_hint="嘿 Hermes，现在怎么办",
                timestamp=1778131310.0,
            )

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["metadata"]["source_type"], "wake_query")
        self.assertEqual(result["metadata"]["captured_at"], 1778131310.0)
        self.assertTrue(str(result["metadata"]["segment_id"]).startswith("seg_"))

    def test_audio_segment_emotion_failure_keeps_transcript_and_discards_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="音频骨架")

            result = service.process_audio_segment(
                user_id="u1",
                capture_id=capture["capture_id"],
                transcript_hint="行吧，知道了",
                simulate="emotion_failure",
            )
            reloaded = service.timeline_store.get_capture("u1", capture["capture_id"])

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["transcript"], "行吧，知道了")
        self.assertEqual(result["metadata"]["emotion"]["enabled"], False)
        self.assertEqual(result["metadata"]["emotion"]["error_type"], "emotion_failed")
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded["chunks"][0]["text"], "行吧，知道了")

    def test_audio_segment_asr_failure_does_not_append_capture_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="音频骨架")

            result = service.process_audio_segment(
                user_id="u1",
                capture_id=capture["capture_id"],
                transcript_hint="不会被使用",
                simulate="asr_failure",
            )
            reloaded = service.timeline_store.get_capture("u1", capture["capture_id"])

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "asr_failed")
        self.assertEqual(result["audio_retention"], "discarded_after_failure")
        self.assertFalse(result["debug"]["audio_processing"]["capture_appended"])
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded["chunks"], [])

    def test_audio_segment_process_rejects_missing_or_cross_user_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="音频骨架")

            with self.assertRaises(ValueError):
                service.process_audio_segment(user_id="u2", capture_id=capture["capture_id"], transcript_hint="隐私片段")
            with self.assertRaises(ValueError):
                service.process_audio_segment(user_id="u1", capture_id="cap_missing", transcript_hint="隐私片段")

    def test_standard_library_audio_segment_process_endpoint_appends_capture(self) -> None:
        from ai_glasses_memory_assistant.server import GlassesHandler

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}, clear=True):
            service = FakeService(self.make_store(Path(tmpdir)))
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="音频骨架")
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, _, payload = self.http_post(
                    GlassesHandler,
                    "/api/audio/segment/process",
                    {
                        "user_id": "u1",
                        "capture_id": capture["capture_id"],
                        "transcript_hint": "服了，又来了",
                        "audio_base64": base64.b64encode(b"fake-wav-bytes").decode("ascii"),
                        "audio_mime_type": "audio/wav",
                        "audio_duration_ms": 1200,
                    },
                )
            finally:
                GlassesHandler.service = original_service

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "processed")
        self.assertTrue(payload["debug"]["audio_processing"]["capture_appended"])
        self.assertEqual(payload["capture_append"]["chunk_count"], 1)

    def test_standard_library_audio_segment_process_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}, clear=True):
            from ai_glasses_memory_assistant.server import GlassesHandler

            service = FakeService(self.make_store(Path(tmpdir)))
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="音频骨架")
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, _, payload = self.http_post(
                    GlassesHandler,
                    "/api/audio/segment/process",
                    {
                        "user_id": "u1",
                        "capture_id": capture["capture_id"],
                        "transcript_hint": "服了，又来了",
                        "audio_base64": base64.b64encode(b"fake-wav-bytes").decode("ascii"),
                        "audio_mime_type": "audio/wav",
                        "audio_duration_ms": 1200,
                    },
                )
            finally:
                GlassesHandler.service = original_service

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "processed")
        self.assertTrue(payload["debug"]["audio_processing"]["capture_appended"])

    def test_greeting_fast_path_skips_agent_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))
            result = service.chat("你好！", user_id="u1")
            self.assertEqual(result["api_calls"], 0)
            self.assertEqual(service.new_session_calls, 0)
            self.assertEqual(service.fake_agent.intent_calls, 0)
            self.assertEqual(service.fake_agent.temporal_calls, 0)
            self.assertEqual(result["saved_memories"], [])
            self.assertTrue(result["debug"]["fast_path"])
            self.assertIsNotNone(result["debug"]["timing"]["total_seconds"])

    def test_greeting_fast_path_persists_raw_timeline_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))

            result = service.chat("你好！", user_id="u1")
            timeline = result["debug"]["timeline"]
            turn = service.timeline_store.get_turn("u1", timeline["turn_id"])
            chunks = service.timeline_store.search_chunks("u1", "你好", limit=5)

            self.assertTrue(timeline["persisted"])
            self.assertTrue(timeline["reply_persisted"])
            self.assertIsNotNone(turn)
            self.assertEqual(turn.raw_text, "你好！")
            self.assertEqual(turn.assistant_reply, result["reply"])
            self.assertEqual(len(chunks), 1)

    def test_llm_first_greeting_fast_path_skips_pre_reply_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))

            result = service.chat("你好", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["api_calls"], 0)
            self.assertTrue(result["debug"]["fast_path"])
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "greeting")
            self.assertIn("planner_baseline_llm_first", result["debug"]["steps"])
            self.assertIn("fast_path_greeting", result["debug"]["steps"])
            self.assertEqual(service.fake_agent.pre_reply_calls, 0)
            self.assertEqual(service.fake_agent.answer_calls, 0)
            self.assertEqual(service.new_session_calls, 0)

    def test_llm_first_identity_query_fast_path_skips_pre_reply_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户名字叫 jack", kind="profile")
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks stored preference",
            }))

            result = service.chat("我是谁？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["api_calls"], 0)
            self.assertTrue(result["debug"]["fast_path"])
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "identity_query")
            self.assertIn("planner_baseline_llm_first", result["debug"]["steps"])
            self.assertIn("fast_path_identity_query", result["debug"]["steps"])
            self.assertIn("jack", result["reply"])
            self.assertEqual(service.fake_agent.pre_reply_calls, 0)
            self.assertEqual(service.fake_agent.answer_calls, 0)
            self.assertEqual(service.new_session_calls, 0)

    def test_chat_response_includes_user_readable_source_summary_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile", memory_type="preference")
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks stored preference",
            }))

            result = service.chat("我喜欢喝什么？", user_id="u1")

            self.assertEqual(result["source_summary"]["structured_memory_count"], 1)
            self.assertEqual(result["source_summary"]["profile_count"], 1)
            self.assertEqual(result["source_summary"]["event_count"], 0)
            self.assertEqual(result["source_summary"]["timeline_chunk_count"], 0)
            self.assertEqual(result["source_summary"]["document_count"], 0)
            self.assertEqual(result["source_summary"]["saved_memory_count"], 0)
            self.assertEqual(result["debug"]["source_summary"], result["source_summary"])
            records = service.read_audit_records(user_id="u1", limit=5)
            self.assertEqual(records[-1]["source_summary"], result["source_summary"])
            self.assertEqual(records[-1]["audit_summary"]["record_type"], "chat_turn")
            self.assertEqual(records[-1]["audit_summary"]["structured_memory_count"], 1)

    def test_app_audio_transcript_import_uses_unified_source_summary_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks recent activity",
            }))

            result = service.import_memory_events(
                user_id="u1",
                source="app_audio_transcript",
                context="walking transcript",
                items=[
                    {
                        "content": "AI 眼镜项目：Mia 负责语音按钮",
                        "kind": "event",
                        "memory_type": "task",
                        "confidence": 0.95,
                    },
                    {
                        "content": "风险是后台保存反馈不够明显",
                        "kind": "event",
                        "memory_type": "project_state",
                        "confidence": 0.9,
                    },
                ],
            )

            self.assertEqual(result["source"], "app_audio_transcript")
            self.assertEqual(result["source_summary"]["input_source"], "app_audio_transcript")
            self.assertEqual(result["source_summary"]["saved_memory_count"], 2)
            self.assertEqual(result["source_summary"]["structured_memory_count"], 2)
            self.assertEqual(result["source_summary"]["event_count"], 2)
            self.assertEqual(result["source_summary"]["pending_confirmation_count"], 0)
            self.assertEqual(result["cleaning_trace"]["summary"]["segment_count"], 2)
            records = service.read_audit_records(user_id="u1", limit=5)
            self.assertEqual(records[-1]["record_type"], "memory_import")
            self.assertEqual(records[-1]["source_summary"], result["source_summary"])
            self.assertEqual(records[-1]["audit_summary"]["source"], "app_audio_transcript")

    def test_chat_redacts_sensitive_timeline_and_audit_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))
            secret = "sk-1234567890abcdefghijklmnopqr"

            result = service.chat(f"你好，我的 api_key={secret}", user_id="u1")
            timeline = result["debug"]["timeline"]
            turn = service.timeline_store.get_turn("u1", timeline["turn_id"])
            records = service.read_audit_records(user_id="u1", limit=5)
            serialized_records = json.dumps(records, ensure_ascii=False)

            self.assertTrue(timeline["persisted"])
            self.assertTrue(timeline["redacted"])
            self.assertIn("token", timeline["redaction_categories"])
            self.assertGreaterEqual(timeline["redaction_count"], 1)
            self.assertTrue(result["debug"]["text_cleaning"]["redacted"])
            self.assertIn("token", result["debug"]["text_cleaning"]["redaction_categories"])
            self.assertNotIn(secret, result["debug"]["text_cleaning"]["normalized_text"])
            self.assertIsNotNone(turn)
            self.assertNotIn(secret, turn.raw_text)
            self.assertNotIn(secret, serialized_records)
            self.assertIn("[已脱敏:token]", serialized_records)

    def test_planner_marks_fast_path_modes(self) -> None:
        reference_time = 1778131200.0
        cases = (
            ("你好！", "greeting"),
            ("我是谁？", "identity_query"),
        )
        for message, reply_mode in cases:
            with self.subTest(message=message):
                plan = plan_turn(message, reference_time=reference_time, timezone="Asia/Shanghai")

                self.assertTrue(plan.fast_path)
                self.assertEqual(plan.fast_path_kind, reply_mode)
                self.assertEqual(plan.reply_mode, reply_mode)

    def test_planner_does_not_decide_weather_or_location_semantics(self) -> None:
        reference_time = 1778131200.0

        named_place = plan_turn("上海天气如何", reference_time=reference_time, timezone="Asia/Shanghai")
        current_weather = plan_turn("我这儿天气如何", reference_time=reference_time, timezone="Asia/Shanghai")
        current_snow = plan_turn("我这儿会下雪吗", reference_time=reference_time, timezone="Asia/Shanghai")

        self.assertFalse(named_place.needs_web_search)
        self.assertFalse(named_place.needs_location)
        self.assertFalse(current_weather.needs_web_search)
        self.assertFalse(current_weather.needs_location)
        self.assertFalse(current_snow.needs_web_search)
        self.assertFalse(current_snow.needs_location)

    def test_question_about_prior_topic_stays_on_planner_baseline(self) -> None:
        plan = plan_turn(
            "我是不是问过香港的事情",
            reference_time=1778131200.0,
            timezone="Asia/Shanghai",
        )

        self.assertFalse(plan.fast_path)
        self.assertEqual(plan.reply_mode, "llm")
        self.assertEqual(plan.memory_write_candidates, [])

    def test_planner_does_not_extract_event_statement_candidate(self) -> None:
        plan = plan_turn(
            "额嗯嗯，今天，呃呃呃，我忘记要说啥了，嗯嗯，我想起来了，好像要和 Mina 开会，估计在三点吧",
            reference_time=1778131200.0,
            timezone="Asia/Shanghai",
        )

        self.assertEqual(plan.reply_mode, "llm")
        self.assertEqual(plan.memory_write_candidates, [])

    def test_planner_stays_baseline_for_direct_event_question(self) -> None:
        plan = plan_turn(
            "今天三点和 Mina 开会是啥意思",
            reference_time=1778131200.0,
            timezone="Asia/Shanghai",
        )

        self.assertEqual(plan.memory_write_candidates, [])
        self.assertEqual(plan.reply_mode, "llm")

    def test_planner_does_not_handle_mixed_recall_and_write_semantics(self) -> None:
        plan = plan_turn(
            "我喜欢什么咖啡？另外帮我记一下我喜欢低糖拿铁",
            reference_time=1778131200.0,
            timezone="Asia/Shanghai",
        )

        self.assertFalse(plan.needs_profile_memory)
        self.assertEqual(plan.recall_goal, "none")
        self.assertEqual(plan.memory_write_candidates, [])

    def test_planner_skips_unconfirmed_cross_turn_task_draft(self) -> None:
        plan = plan_turn(
            "我先临时想一下，Mia 明天上午十点可能去核对外测名单，但还没确认，先别写进任务。",
            reference_time=1779415200.0,
            timezone="Asia/Shanghai",
        )

        self.assertEqual(plan.reply_mode, "llm")
        self.assertEqual(plan.memory_write_candidates, [])
        self.assertEqual(plan.reason, "default_pre_reply_decision_required")

    def test_planner_does_not_save_confirmed_cross_turn_task_commitment(self) -> None:
        plan = plan_turn(
            "确认一下，Mia 明天上午十点核对外测名单，这个记下来。",
            reference_time=1779415200.0,
            timezone="Asia/Shanghai",
        )

        self.assertEqual(plan.reply_mode, "llm")
        self.assertEqual(plan.memory_write_candidates, [])

    def test_chat_saves_event_statement_with_question_word_filler(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks specific remembered event",
            }))

            result = service.chat(
                "额嗯嗯，今天，呃呃呃，我忘记要说啥了，嗯嗯，我想起来了，好像要和 Mina 开会，估计在三点吧",
                user_id="u1",
            )
            memories = store.list_memories("u1", kind="event")

            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_action"], "write")
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_event_recall")
            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertEqual(len(memories), 1)
            self.assertIn("Mina", memories[0].content)

    def test_planner_does_not_extract_indirect_stable_preference_statement(self) -> None:
        plan = plan_turn(
            "我喝过手冲咖啡，那是我觉得最适合早上的饮品，我很喜欢喝",
            reference_time=1778131200.0,
            timezone="Asia/Shanghai",
        )

        self.assertEqual(plan.reply_mode, "llm")
        self.assertEqual(plan.memory_write_candidates, [])

    def test_planner_does_not_extract_negative_indirect_preference_statement(self) -> None:
        plan = plan_turn(
            "我吃过太甜的点心，那种口味不适合我，我不喜欢吃",
            reference_time=1778131200.0,
            timezone="Asia/Shanghai",
        )

        self.assertEqual(plan.reply_mode, "llm")
        self.assertEqual(plan.memory_write_candidates, [])

    def test_planner_leaves_ordinary_question_for_standard_flow(self) -> None:
        plan = plan_turn("普通问答：水的化学式是什么？", reference_time=1778131200.0, timezone="Asia/Shanghai")

        self.assertFalse(plan.fast_path)
        self.assertEqual(plan.reply_mode, "llm")

    def test_planner_does_not_trigger_weekly_report_for_knowledge_question(self) -> None:
        for message in ("周报是什么？", "周报怎么写？"):
            with self.subTest(message=message):
                plan = plan_turn(message, reference_time=1778131200.0, timezone="Asia/Shanghai")

                self.assertEqual(plan.reply_mode, "llm")
                self.assertEqual(plan.conversation_action, "")
                self.assertEqual(plan.recall_goal, "none")

    def test_planner_does_not_trigger_attention_items_for_concept_question(self) -> None:
        for message in ("风险是什么？", "卡点是什么意思？", "项目风险是什么？"):
            with self.subTest(message=message):
                plan = plan_turn(message, reference_time=1778131200.0, timezone="Asia/Shanghai")

                self.assertEqual(plan.reply_mode, "llm")
                self.assertEqual(plan.conversation_action, "")
                self.assertEqual(plan.recall_goal, "none")

    def test_planner_does_not_trigger_timeline_recall_for_original_word_concept_question(self) -> None:
        for message in ("原话是什么意思？", "怎么写原话？", "原话格式是什么？"):
            with self.subTest(message=message):
                plan = plan_turn(message, reference_time=1778131200.0, timezone="Asia/Shanghai")

                self.assertEqual(plan.reply_mode, "llm")
                self.assertFalse(plan.needs_timeline_recall)
                self.assertEqual(plan.recall_goal, "none")

    def test_planner_leaves_historical_original_wording_request_to_pre_reply_decision(self) -> None:
        plan = plan_turn("我之前有没有说过语音识别不稳定的原话？", reference_time=1778131200.0, timezone="Asia/Shanghai")

        self.assertEqual(plan.reply_mode, "llm")
        self.assertFalse(plan.needs_timeline_recall)
        self.assertEqual(plan.recall_goal, "none")

    def test_planner_routes_non_meeting_long_input_to_continuous_capture(self) -> None:
        message = (
            "今天一天有点长，上午我先去了医院体检，医生提醒我下周三上午再去取报告。"
            "中午和 Mia 吃饭时聊到 AI 眼镜 demo，她说前端语音按钮还要补播报状态。"
            "下午我和 Alex 复盘后台记忆写入，决定先做可观测 debug，再接真实设备。"
            "晚上回家路上我又想到一个风险：浏览器语音权限不稳定时，用户可能不知道后台是否保存成功。"
        )

        plan = plan_turn(message, reference_time=1778131200.0, timezone="Asia/Shanghai")

        self.assertTrue(plan.fast_path)
        self.assertEqual(plan.reply_mode, "continuous_capture")
        self.assertEqual(plan.fast_path_kind, "continuous_capture")
        self.assertIn("matched_long_input_capture", plan.reason)

    def test_planner_routes_explicit_short_transcript_to_continuous_capture(self) -> None:
        message = "先记一段转写文字：物流电话不用记，不过周五前把按钮验收补完，后面其他背景先别沉淀。"

        plan = plan_turn(message, reference_time=1778131200.0, timezone="Asia/Shanghai")

        self.assertTrue(plan.fast_path)
        self.assertEqual(plan.reply_mode, "continuous_capture")
        self.assertIn("explicit_short_transcript", plan.reason)

    def test_planner_keeps_long_knowledge_question_on_llm_path(self) -> None:
        message = (
            "请详细解释 transformer 的注意力机制，包括 query key value 的计算方式、"
            "multi-head attention 为什么有效、位置编码有哪些类型、训练时为什么需要 mask，"
            "再举几个代码层面的例子说明它和 RNN 的区别。"
        )

        plan = plan_turn(message, reference_time=1778131200.0, timezone="Asia/Shanghai")

        self.assertFalse(plan.fast_path)
        self.assertEqual(plan.reply_mode, "llm")

    def test_planner_leaves_bare_recent_to_pre_reply_decision(self) -> None:
        plan = plan_turn("最近", reference_time=1778131200.0, timezone="Asia/Shanghai")

        self.assertEqual(plan.reply_mode, "llm")
        self.assertFalse(plan.needs_event_memory)
        self.assertEqual(plan.event_recall_strategy, "skipped")
        self.assertEqual(plan.reason, "default_pre_reply_decision_required")
        self.assertEqual(plan.temporal_scope.temporal_text, "最近")
        self.assertEqual(plan.temporal_scope.granularity, "day")

    def test_planner_does_not_trigger_event_recall_for_action_item_concept_question(self) -> None:
        for message in ("提醒是什么意思？", "安排是什么意思？", "待办是什么意思？", "接下来是什么意思？", "未来是什么意思？", "近期是什么意思？"):
            with self.subTest(message=message):
                plan = plan_turn(message, reference_time=1778131200.0, timezone="Asia/Shanghai")

                self.assertEqual(plan.reply_mode, "llm")
                self.assertFalse(plan.needs_event_memory)
                self.assertEqual(plan.event_recall_strategy, "skipped")
                self.assertEqual(plan.recall_goal, "none")

    def test_identity_query_reads_profile_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks stored preference",
            }))
            result = service.chat("我是谁？", user_id="u1")
            self.assertIn("还不知道", result["reply"])
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(service.new_session_calls, 0)

            store.add_memory("u1", "用户名字叫 jack", kind="profile")
            result = service.chat("我是谁", user_id="u1")
            self.assertIn("jack", result["reply"])
            self.assertEqual(len(store.list_memories("u1", kind="profile")), 1)

            result = service.chat("我叫什么名字？", user_id="u1")
            self.assertIn("jack", result["reply"])
            self.assertEqual(len(store.list_memories("u1", kind="profile")), 1)

    def test_identity_query_prioritizes_name_profile_over_other_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks recent activity",
            }))
            store.add_memory("u1", "用户名字叫 jack", kind="profile")
            store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile")
            store.add_memory("u1", "用户不喜欢太甜的饮料", kind="profile")
            store.add_memory("u1", "不喜欢排队很久的餐厅", kind="profile")

            result = service.chat("我叫什么名字？", user_id="u1")

            self.assertIn("jack", result["reply"])
            self.assertNotIn("排队", result["reply"])
            self.assertEqual(result["api_calls"], 0)
            self.assertEqual(result["saved_memories"], [])

    def test_memory_access_fields_update_only_on_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            memory = store.add_memory(
                "u1",
                "用户喜欢安静靠窗的位置",
                kind="profile",
                memory_type="preference",
            )

            listed = store.list_memories("u1", kind="profile")[0]
            self.assertEqual(listed.access_count, 0)
            self.assertIsNone(listed.last_accessed_at)
            self.assertEqual(listed.strength, 0.85)

            updated = store.record_memory_access("u1", [memory.id], accessed_at=1778131210.0)[0]
            self.assertEqual(updated.access_count, 1)
            self.assertEqual(updated.last_accessed_at, 1778131210.0)
            self.assertGreaterEqual(updated.strength, listed.strength)

            listed_again = store.list_memories("u1", kind="profile")[0]
            self.assertEqual(listed_again.access_count, 1)
            self.assertEqual(listed_again.last_accessed_at, 1778131210.0)

    def test_memory_access_strength_has_type_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            memory = store.add_memory(
                "u1",
                "用户喜欢安静靠窗的位置",
                kind="profile",
                memory_type="preference",
            )

            for _ in range(20):
                store.record_memory_access("u1", [memory.id], accessed_at=1778131210.0)

            refreshed = store.get_memory("u1", memory.id)
            self.assertEqual(refreshed.access_count, 20)
            self.assertLessEqual(refreshed.strength, 0.9)

    def test_memory_access_ignores_deleted_superseded_and_other_users(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            active = store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile", memory_type="preference")
            deleted = store.add_memory("u1", "用户喜欢甜咖啡", kind="profile", memory_type="preference")
            superseded = store.add_memory("u1", "用户喜欢旧座位", kind="profile", memory_type="preference")
            other_user = store.add_memory("u2", "用户喜欢靠窗", kind="profile", memory_type="preference")
            store.delete_memory("u1", deleted.id)
            store.mark_superseded("u1", superseded.id, active.id)

            updated = store.record_memory_access(
                "u1",
                [active.id, deleted.id, superseded.id, other_user.id],
                accessed_at=1778131210.0,
            )

            self.assertEqual([memory.id for memory in updated], [active.id])
            self.assertEqual(store.get_memory("u1", active.id).access_count, 1)
            self.assertEqual(store.get_memory("u2", other_user.id).access_count, 0)

    def test_stale_memory_is_traceable_but_not_active_or_accessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            active = store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile", memory_type="preference")
            stale = store.add_memory("u1", "用户喜欢甜咖啡", kind="profile", memory_type="preference")

            self.assertTrue(store.mark_stale("u1", stale.id))
            updated = store.record_memory_access(
                "u1",
                [active.id, stale.id],
                accessed_at=1778131210.0,
            )

            self.assertEqual(store.get_memory("u1", stale.id).status, "stale")
            self.assertEqual([memory.id for memory in store.list_memories("u1", kind="profile")], [active.id])
            self.assertEqual(store.search("u1", "甜咖啡"), [])
            self.assertEqual([memory.id for memory in updated], [active.id])
            self.assertEqual(store.get_memory("u1", stale.id).access_count, 0)

    def test_memory_lifecycle_policy_centralizes_active_surface_rules(self) -> None:
        self.assertEqual(ACTIVE_MEMORY_STATUS, "active")
        self.assertEqual(normalize_memory_status("unknown-status"), "active")
        self.assertTrue(is_active_memory_status("active"))
        for status in ("stale", "superseded", "deleted"):
            self.assertFalse(is_active_memory_status(status))

        self.assertTrue(can_transition_memory_status("active", "stale"))
        self.assertTrue(can_transition_memory_status("active", "deleted"))
        self.assertTrue(can_transition_memory_status("stale", "deleted"))
        self.assertTrue(can_transition_memory_status("active", "superseded", superseded_by="replacement-id"))
        self.assertFalse(can_transition_memory_status("active", "superseded", superseded_by=""))
        self.assertFalse(can_transition_memory_status("stale", "superseded", superseded_by="replacement-id"))
        self.assertFalse(can_transition_memory_status("deleted", "active"))

        payload = lifecycle_transition_payload(
            memory_id="old-id",
            from_status="active",
            to_status="superseded",
            reason="preference_conflict_supersede",
            superseded_by="new-id",
        )
        self.assertEqual(payload["memory_id"], "old-id")
        self.assertEqual(payload["from_status"], "active")
        self.assertEqual(payload["to_status"], "superseded")
        self.assertEqual(payload["superseded_by"], "new-id")
        self.assertEqual(payload["reason"], "preference_conflict_supersede")

    def test_lifecycle_transitions_only_allow_superseding_active_with_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            replacement = store.add_memory("u1", "用户喜欢吧台位置", kind="profile", memory_type="preference")
            active = store.add_memory("u1", "用户喜欢靠窗座位", kind="profile", memory_type="preference")
            stale = store.add_memory("u1", "用户喜欢旧甜咖啡", kind="profile", memory_type="preference")

            self.assertFalse(store.mark_superseded("u1", active.id, ""))
            self.assertEqual(store.get_memory("u1", active.id).status, "active")

            self.assertTrue(store.mark_stale("u1", stale.id))
            self.assertFalse(store.mark_superseded("u1", stale.id, replacement.id))
            self.assertEqual(store.get_memory("u1", stale.id).status, "stale")

            self.assertTrue(store.mark_superseded("u1", active.id, replacement.id))
            self.assertEqual(store.get_memory("u1", active.id).status, "superseded")
            self.assertEqual(store.get_memory("u1", active.id).superseded_by, replacement.id)
            self.assertFalse(store.mark_superseded("u1", active.id, replacement.id))

    def test_lifecycle_non_active_memory_cannot_be_merged_or_accessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            stale = store.add_memory("u1", "用户喜欢甜咖啡", kind="profile", memory_type="preference")
            superseded = store.add_memory("u1", "用户喜欢旧座位", kind="profile", memory_type="preference")
            replacement = store.add_memory("u1", "用户喜欢吧台位置", kind="profile", memory_type="preference")

            self.assertTrue(store.mark_stale("u1", stale.id))
            self.assertTrue(store.mark_superseded("u1", superseded.id, replacement.id))

            self.assertIsNone(
                store.merge_memory_evidence("u1", stale.id, evidence_ids=["chunk_stale"], source_id="new-source")
            )
            self.assertIsNone(
                store.merge_memory_evidence("u1", superseded.id, evidence_ids=["chunk_old"], source_id="new-source")
            )
            updated = store.record_memory_access(
                "u1",
                [stale.id, superseded.id],
                accessed_at=1778131210.0,
            )

            self.assertEqual(updated, [])
            self.assertEqual(store.get_memory("u1", stale.id).evidence_ids, [])
            self.assertEqual(store.get_memory("u1", superseded.id).evidence_ids, [])
            self.assertEqual(store.get_memory("u1", stale.id).access_count, 0)
            self.assertEqual(store.get_memory("u1", superseded.id).access_count, 0)

    def test_existing_memories_backfill_initial_strength_on_schema_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            db_path = Path(tmpdir) / "events.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'event',
                    memory_type TEXT NOT NULL DEFAULT 'event',
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT 'chat',
                    created_at REAL NOT NULL,
                    occurred_at REAL,
                    deleted_at REAL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, kind, memory_type, content, tags, source, created_at, occurred_at, deleted_at
                )
                VALUES ('m1', 'u1', 'profile', 'preference', '用户喜欢靠窗座位', '[]', 'chat', 1778131200.0, NULL, NULL)
                """
            )
            conn.commit()
            conn.close()

            store = EventMemoryStore(db_path=db_path)
            memory = store.get_memory("u1", "m1")

            self.assertIsNotNone(memory)
            self.assertEqual(memory.strength, 0.85)

    def test_memory_strength_backfill_failure_does_not_block_store_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            db_path = Path(tmpdir) / "events.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'event',
                    memory_type TEXT NOT NULL DEFAULT 'event',
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT 'chat',
                    created_at REAL NOT NULL,
                    occurred_at REAL,
                    strength REAL NOT NULL DEFAULT 0,
                    deleted_at REAL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, kind, memory_type, content, tags, source, created_at, occurred_at, strength, deleted_at
                )
                VALUES ('m1', 'u1', 'profile', 'preference', '用户喜欢靠窗座位', '[]', 'chat', 1778131200.0, NULL, 0, NULL)
                """
            )
            conn.commit()
            conn.close()

            class BackfillFailingStore(EventMemoryStore):
                def _backfill_memory_strength(self) -> None:
                    raise sqlite3.DatabaseError("database disk image is malformed")

            store = BackfillFailingStore(db_path=db_path)

            self.assertEqual(store.db_path, db_path)

    def test_assistant_name_preference_is_recalled_in_new_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            agent = FakeAgent(
                {
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "助手名字叫奈恩",
                            "kind": "assistant_preference",
                            "confidence": 0.95,
                            "reason": "assistant name setting",
                        }
                    ],
                    "confidence": 0.95,
                }
            )
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=agent)

            result = service.chat("以后你的名字就叫奈恩", user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(
                user_id="u1",
                job_id=result["debug"]["memory_processing"]["job_id"],
            )
            memories = store.list_memories("u1", kind="assistant_preference")

            self.assertEqual(job["status"], "saved")
            self.assertEqual(job["extraction_backend"], "unified_semantics")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "助手名字叫奈恩")
            self.assertEqual(memories[0].kind, "assistant_preference")

            new_service = FakeService(store)
            recall = new_service.chat("你叫什么名字", user_id="u1")
            self.assertEqual(recall["api_calls"], 0)
            self.assertIn("奈恩", recall["reply"])
            self.assertEqual(recall["recalled_memories"][0]["kind"], "assistant_preference")

    def test_pre_reply_profile_candidate_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(semantic_payload=semantic_payload(
                correction=False,
                memory_action="write",
                memory_kind="profile",
                memory_type="preference",
                candidate_content="用户喜欢低糖拿铁",
            )))
            result = service.chat("我喜欢低糖拿铁", user_id="u1")
            self.assertEqual(service.new_session_calls, 1)
            self.assertEqual(result["saved_memories"][0]["kind"], "profile")
            self.assertEqual(result["saved_memories"][0]["memory_type"], "preference")
            self.assertEqual(result["saved_memories"][0]["status"], "active")
            self.assertEqual(result["saved_memories"][0]["content"], "用户喜欢低糖拿铁")

    def test_natural_preference_statement_saves_and_recalls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "asks recent work activity",
            }))

            saved = service.chat("我不喜欢排队很久的餐厅", user_id="u1")
            recall = service.chat("以后推荐餐厅时要避开什么？", user_id="u1")

            self.assertEqual(saved["saved_memories"][0]["kind"], "profile")
            self.assertEqual(saved["saved_memories"][0]["memory_type"], "preference")
            self.assertIn("用户不喜欢排队很久的餐厅", saved["saved_memories"][0]["content"])
            self.assertEqual(store.list_memories("u1", kind="profile")[0].memory_type, "preference")

    def test_seat_preference_statement_saves_and_recalls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "asks stored preference",
            }))

            saved = service.chat("我喜欢安静靠窗的位置", user_id="u1")
            recall = service.chat("以后订座位优先考虑什么？", user_id="u1")

            self.assertEqual(saved["saved_memories"][0]["kind"], "profile")
            self.assertEqual(saved["saved_memories"][0]["memory_type"], "preference")
            self.assertIn("用户喜欢安静靠窗的位置", saved["saved_memories"][0]["content"])
            self.assertEqual(store.list_memories("u1", kind="profile")[0].memory_type, "preference")

    def test_complex_preference_statement_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "asks recent activity",
            }))

            result = service.chat("我其实更喜欢安静靠窗的位置，太吵会让我没法集中", user_id="u1")

            self.assertEqual(result["saved_memories"][0]["kind"], "profile")
            self.assertEqual(result["saved_memories"][0]["memory_type"], "preference")
            self.assertEqual(
                result["saved_memories"][0]["content"],
                "用户喜欢安静靠窗的位置，太吵会让我没法集中",
            )

    def test_preference_question_does_not_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks specific remembered event",
            }))

            result = service.chat("我是不是喜欢很甜的饮料？", user_id="u1")

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1"), [])

    def test_prior_topic_question_does_not_save_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}, clear=True):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "asks recent work activity",
            }))

            result = service.chat("我是不是问过香港的事情", user_id="u1", defer_memory_writes=True)

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1", kind="profile"), [])
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_event_recall")

    def test_recent_context_capsule_is_available_to_pre_reply_and_main_llm(self) -> None:
        pre_reply_payload = {
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "location_text": "",
            "needs_event_memory": True,
            "memory_recall_type": "observation",
            "recall_goal": "summary",
            "confidence": 0.95,
            "reason": "recent material reference",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload=pre_reply_payload))
            service.timeline_store.add_turn(
                "u1",
                "我刚导入一段关于某政策新增监管措施的材料，提到了账户关闭和资金来源声明要求。",
                created_at=1778131100.0,
            )
            store.add_memory(
                "u1",
                "用户刚导入一段关于某政策新增监管措施的材料，包含账户关闭和资金来源声明要求。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                confidence=0.86,
            )

            result = service.chat("想一想这个新增监管措施的事情。", user_id="u1", routing_mode="llm_first")

            self.assertTrue(result["debug"]["recent_context_capsule"]["available"])
            self.assertEqual(result["debug"]["recent_context_capsule"]["timeline_chunk_count"], 1)
            self.assertEqual(result["debug"]["recent_context_capsule"]["injection_reason"], "summary_reference_signal")
            self.assertIn("新增监管措施", service.fake_agent.semantic_messages[0])
            self.assertIn("recent-context-capsule", service.fake_agent.main_messages[0])
            self.assertIn("资金来源声明", service.fake_agent.main_messages[0])
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "observation")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "observation_review")

    def test_recent_context_capsule_is_not_injected_for_ordinary_question_with_only_weak_marker(self) -> None:
        pre_reply_payload = {
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "location_text": "",
            "memory_recall_type": "none",
            "recall_goal": "none",
            "confidence": 0.95,
            "reason": "ordinary question",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload=pre_reply_payload))
            service.timeline_store.add_turn("u1", "我刚导入了一段项目材料。", created_at=1778131100.0)

            result = service.chat("这个问题简单点，水的化学式是什么？", user_id="u1", routing_mode="llm_first")

            self.assertTrue(result["debug"]["recent_context_capsule"]["available"])
            self.assertFalse(result["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(
                result["debug"]["recent_context_capsule"]["injection_reason"],
                "ordinary_query_without_memory_reference",
            )
            self.assertNotIn("recent-context-capsule", service.fake_agent.main_messages[0])

    def test_recent_context_capsule_is_not_injected_for_document_detail_without_recent_reference(self) -> None:
        pre_reply_payload = {
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "location_text": "",
            "memory_recall_type": "none",
            "recall_goal": "none",
            "confidence": 0.95,
            "reason": "document detail question",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="南太行自驾攻略.md",
                title="南太行自驾攻略",
                summary="包含费用和路线",
                content="# 南太行自驾攻略\n红旗渠门票 80 元",
                source="markdown_upload",
                ingestion_id="ing-doc",
                created_at=1778131100.0,
            )
            store.add_memory(
                "u1",
                "用户最近在整理南太行资料。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload=pre_reply_payload))
            service.timeline_store.add_turn("u1", "我刚聊过南太行项目背景。", created_at=1778131101.0)

            result = service.chat("这个南太行自驾攻略里红旗渠门票多少钱？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "full_document")
            self.assertFalse(result["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(
                result["debug"]["recent_context_capsule"]["injection_reason"],
                "document_detail_without_recent_reference",
            )
            self.assertIn("红旗渠门票 80 元", service.fake_agent.main_messages[0])
            self.assertNotIn("recent-context-capsule", service.fake_agent.main_messages[0])

    def test_recent_context_capsule_is_not_injected_for_weak_specific_fact_reference_without_anchor(self) -> None:
        pre_reply_payload = {
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "location_text": "",
            "needs_event_memory": True,
            "memory_recall_type": "event",
            "recall_goal": "specific_fact",
            "confidence": 0.95,
            "reason": "specific fact with weak deixis",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload=pre_reply_payload))
            service.timeline_store.add_turn("u1", "我刚聊过 AI 眼镜项目背景。", created_at=1778131100.0)
            store.add_memory("u1", "AI 眼镜项目当前主线是收敛文字输入质量", kind="event", memory_type="project_state")

            result = service.chat("这个现在具体是什么状态？", user_id="u1", routing_mode="llm_first")

            self.assertTrue(result["debug"]["recent_context_capsule"]["available"])
            self.assertFalse(result["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(
                result["debug"]["recent_context_capsule"]["injection_reason"],
                "weak_recent_reference_without_memory_anchor",
            )
            self.assertIn(service.fake_agent.main_calls, {0, 1})

    def test_recent_context_capsule_is_not_injected_for_cross_document_compare(self) -> None:
        pre_reply_payload = {
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "location_text": "",
            "memory_recall_type": "none",
            "recall_goal": "none",
            "confidence": 0.95,
            "reason": "cross document comparison with ambiguous title",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="外卖小哥买奥迪.md",
                title="外卖小哥买奥迪",
                summary="理性看待消费选择",
                content="# 外卖小哥买奥迪\n## 一句话总结\n理性看待消费选择\n## 重点\n保留现金流。",
                source="markdown_upload",
                ingestion_id="ing-doc-1",
                created_at=1778131000.0,
            )
            store.add_document(
                "u1",
                filename="外卖小哥买奥迪复盘.md",
                title="外卖小哥买奥迪复盘",
                summary="关注负债节奏",
                content="# 外卖小哥买奥迪复盘\n## 一句话总结\n关注负债节奏\n## 重点\n先把每月支出打平。",
                source="markdown_upload",
                ingestion_id="ing-doc-2",
                created_at=1778131100.0,
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload=pre_reply_payload))
            service.timeline_store.add_turn("u1", "我刚导入了两份买车相关材料。", created_at=1778131101.0)

            result = service.chat("对比一下这两份外卖小哥买奥迪文档有什么不同？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "metadata")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "cross_document_compare_query")
            self.assertFalse(result["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(
                result["debug"]["recent_context_capsule"]["injection_reason"],
                "no_recent_reference_markers",
            )
            self.assertNotIn("recent-context-capsule", service.fake_agent.main_messages[0])
            self.assertEqual(len(result["recalled_documents"]), 2)
            recalled_titles = {document["title"] for document in result["recalled_documents"]}
            self.assertEqual(recalled_titles, {"外卖小哥买奥迪", "外卖小哥买奥迪复盘"})

    def test_recent_context_capsule_is_not_injected_for_weak_summary_reference_without_strong_anchor(self) -> None:
        pre_reply_payload = {
            "reply_mode": "llm",
            "answer_source": "llm",
            "scope": "unknown",
            "location_text": "",
            "needs_event_memory": True,
            "memory_recall_type": "observation",
            "recall_goal": "summary",
            "confidence": 0.95,
            "reason": "project status summary with weak deixis",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "AI 眼镜项目当前主线是收敛 LLM-first 文字输入，先压 recent context 污染。",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_project_state"],
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload=pre_reply_payload))
            service.timeline_store.add_turn("u1", "我刚聊过外卖小哥买奥迪那两份材料。", created_at=1778131100.0)

            result = service.chat("这个项目现在主要状态是什么？", user_id="u1", routing_mode="llm_first")

            self.assertTrue(result["debug"]["recent_context_capsule"]["available"])
            self.assertFalse(result["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(
                result["debug"]["recent_context_capsule"]["injection_reason"],
                "weak_summary_reference_without_strong_anchor",
            )
            self.assertEqual(result["source_summary"]["primary_source"], "structured_memory")
            self.assertNotIn("recent-context-capsule", service.fake_agent.main_messages[0])
            self.assertIn("AI 眼镜项目当前主线", service.fake_agent.main_messages[0])

    def test_normal_chat_uses_reply_first_llm_memory_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            agent = FakeAgent(
                {
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户喜欢低糖拿铁",
                            "kind": "profile",
                            "confidence": 0.9,
                            "reason": "stable preference",
                        }
                    ],
                    "confidence": 1.0,
                }
            )
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=agent)
            result = service.chat("我最近更喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1", kind="profile")[0].content, "用户喜欢低糖拿铁")
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(agent.temporal_calls, 0)
            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertEqual(
                result["debug"]["memory_processing"]["mode"],
                "reply_first_llm_extraction",
            )

    def test_llm_memory_extraction_skips_casual_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "memory_recall_type": "event",
                "recall_goal": "summary",
                "event_recall_strategy": "ambiguous_recent_upcoming_plan",
                "confidence": 0.93,
                "reason": "asks for today's personal plan",
            }))

            result = service.chat("普通对话", user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(
                user_id="u1",
                job_id=result["debug"]["memory_processing"]["job_id"],
            )

            self.assertEqual(job["status"], "skipped")
            self.assertEqual(job["saved_count"], 0)
            self.assertEqual(job["rejected_count"], 0)
            self.assertEqual(store.list_memories("u1"), [])

    def test_turn_semantics_debug_is_exposed_for_normal_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload={
                    "turn_intent": "memory_write",
                    "memory_action": "write",
                    "memory_kind": "profile",
                    "memory_type": "preference",
                    "recall_type": "none",
                    "reply_mode_hint": "llm",
                    "flags": {
                        "transient": False,
                        "do_not_remember": False,
                        "correction": False,
                        "explanation_query": False,
                    },
                    "candidate_content": "用户最近更喜欢低糖拿铁",
                    "reason": "stable preference statement",
                    "confidence": 0.92,
                }
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我最近更喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)

            self.assertEqual(result["debug"]["turn_semantics"]["turn_intent"], "memory_write")
            self.assertEqual(result["debug"]["turn_semantics"]["memory_action"], "write")
            self.assertEqual(result["debug"]["turn_semantics"]["memory_kind"], "profile")
            self.assertEqual(result["debug"]["turn_semantics"]["memory_type"], "preference")
            self.assertEqual(result["debug"]["turn_semantics"]["candidate_content"], "用户最近更喜欢低糖拿铁")
            self.assertFalse(result["debug"]["turn_semantics"]["flags"]["explanation_query"])
            self.assertEqual(agent.semantic_calls, 1)

    def test_unified_semantic_memory_action_none_skips_memory_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(correction=False, memory_action="none"),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "这条不该被抽取",
                            "kind": "event",
                            "memory_type": "task",
                            "privacy_level": "normal",
                            "confidence": 0.9,
                            "reason": "would be extracted without semantic gate",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("普通对话", user_id="u1", defer_memory_writes=True)
            gate = result["debug"]["routing"]["unified_semantic_memory_extraction_gate"]

            self.assertEqual(result["debug"]["intent"]["backend"], "unified_semantics")
            self.assertEqual(gate["action"], "skip")
            self.assertEqual(gate["skipped_reason"], "llm_semantics_memory_action_none")
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(agent.intent_calls, 0)

    def test_unified_semantic_transient_skips_memory_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    transient=True,
                ),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户临时想喝冰美式",
                            "kind": "profile",
                            "memory_type": "preference",
                            "privacy_level": "normal",
                            "confidence": 0.9,
                            "reason": "would be extracted without semantic gate",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("临时想试试冰美式，先别沉淀", user_id="u1", defer_memory_writes=True)
            gate = result["debug"]["routing"]["unified_semantic_memory_extraction_gate"]

            self.assertEqual(result["debug"]["intent"]["backend"], "unified_semantics")
            self.assertEqual(gate["action"], "skip")
            self.assertEqual(gate["skipped_reason"], "llm_semantics_transient")
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(agent.intent_calls, 0)

    def test_unified_semantic_memory_action_write_creates_candidate_without_legacy_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢低糖拿铁",
                ),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户喜欢低糖拿铁",
                            "kind": "profile",
                            "memory_type": "preference",
                            "privacy_level": "normal",
                            "confidence": 0.9,
                            "reason": "stable preference",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我最近更喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)
            gate = result["debug"]["routing"]["unified_semantic_memory_extraction_gate"]

            self.assertEqual(agent.intent_calls, 0)
            self.assertEqual(gate["action"], "create")
            self.assertEqual(result["debug"]["intent"]["backend"], "unified_semantics")
            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")

    def test_unified_semantic_typing_hint_updates_aligned_candidate_type(self) -> None:
        candidates, hint = GlassesChatService._apply_unified_semantic_typing_hint(
            turn_semantics={
                "backend": "llm",
                **semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢低糖拿铁",
                ),
            },
            candidates=[
                MemoryWriteCandidate(
                    content="用户喜欢低糖拿铁",
                    kind="event",
                    memory_type="event",
                    confidence=0.9,
                    reason="extractor mistyped preference",
                )
            ],
        )

        self.assertEqual(hint["action"], "applied")
        self.assertEqual(hint["original_kind"], "event")
        self.assertEqual(hint["original_memory_type"], "event")
        self.assertEqual(hint["typing_source"], "unified_semantics")
        self.assertEqual(hint["new_kind"], "profile")
        self.assertEqual(hint["new_memory_type"], "preference")
        self.assertEqual(candidates[0].kind, "profile")
        self.assertEqual(candidates[0].memory_type, "preference")
        self.assertEqual(candidates[0].content, "用户喜欢低糖拿铁")
        self.assertIn("semantic_typing_hint", candidates[0].reason)

    def test_unified_semantic_candidate_authority_creates_semantic_only_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢低糖拿铁",
                ),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)
            shadow = result["debug"]["routing"]["unified_semantic_candidate_shadow"]

            self.assertEqual(shadow["content_alignment"], "semantic_candidate_without_extractor_candidate")
            memories = store.list_memories("u1")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "用户喜欢低糖拿铁")
            self.assertEqual(memories[0].source, "unified_semantics")

    def test_unified_semantic_typing_hint_skips_extractor_only_and_content_mismatch(self) -> None:
        cases = (
            (
                semantic_payload(correction=False, memory_action="write"),
                "extractor_candidate_without_semantic_candidate",
            ),
            (
                semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢低糖拿铁",
                ),
                "different",
            ),
        )
        for semantic, alignment in cases:
            with self.subTest(alignment=alignment):
                candidate_content = "用户喜欢抹茶" if alignment == "different" else "用户喜欢低糖拿铁"
                hint = GlassesChatService._unified_semantic_typing_hint(
                    turn_semantics={"backend": "llm", **semantic},
                    candidates=[
                        MemoryWriteCandidate(
                            content=candidate_content,
                            kind="event",
                            memory_type="event",
                        )
                    ],
                )

                self.assertEqual(hint["action"], "skipped")
                self.assertEqual(hint["content_alignment"], alignment)
                self.assertEqual(hint["typing_source"], "extractor_fallback")

    def test_unified_semantic_typing_hint_records_extractor_aligned_source(self) -> None:
        hint = GlassesChatService._unified_semantic_typing_hint(
            turn_semantics={
                "backend": "llm",
                **semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢低糖拿铁",
                ),
            },
            candidates=[
                MemoryWriteCandidate(
                    content="用户喜欢低糖拿铁",
                    kind="profile",
                    memory_type="preference",
                )
            ],
        )

        self.assertEqual(hint["action"], "skipped")
        self.assertEqual(hint["reason"], "already_aligned")
        self.assertEqual(hint["typing_source"], "extractor_aligned")

    def test_unified_semantic_typing_hint_falls_back_on_semantic_error(self) -> None:
        hint = GlassesChatService._unified_semantic_typing_hint(
            turn_semantics={"backend": "rule_fallback", "error": "not json"},
            candidates=[
                MemoryWriteCandidate(
                    content="用户喜欢低糖拿铁",
                    kind="event",
                    memory_type="event",
                )
            ],
        )

        self.assertEqual(hint["action"], "fallback")
        self.assertEqual(hint["reason"], "turn_semantics_error")
        self.assertEqual(hint["typing_source"], "semantic_unavailable")

    def test_unified_semantic_typing_hint_debug_records_overlay_decision(self) -> None:
        debug = {}

        intent = GlassesChatService._intent_from_pre_reply_decision(
            planner=TurnPlan(),
            extracted=IntentDecision(
                backend="llm",
                memory_write_candidates=[
                    MemoryWriteCandidate(
                        content="用户喜欢低糖拿铁",
                        kind="event",
                        memory_type="event",
                    )
                ],
            ),
            turn_semantics={
                "backend": "llm",
                **semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢低糖拿铁",
                ),
            },
            typing_hint_debug=debug,
        )
        hint = debug["unified_semantic_typing_hint"]

        self.assertEqual(hint["action"], "applied")
        self.assertEqual(hint["original_kind"], "event")
        self.assertEqual(hint["typing_source"], "unified_semantics")
        self.assertEqual(hint["new_kind"], "profile")
        self.assertEqual(intent.memory_write_candidates[0].kind, "profile")
        self.assertEqual(intent.memory_write_candidates[0].memory_type, "preference")

    def test_unified_semantic_typing_hint_is_shared_candidate_postprocess(self) -> None:
        debug = {}

        candidates = GlassesChatService._postprocess_memory_candidates_with_turn_semantics(
            turn_semantics={
                "backend": "llm",
                **semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢低糖拿铁",
                ),
            },
            candidates=[
                MemoryWriteCandidate(
                    content="用户喜欢低糖拿铁",
                    kind="event",
                    memory_type="event",
                    reason="extractor mistyped preference",
                )
            ],
            debug=debug,
        )

        hint = debug["unified_semantic_typing_hint"]
        self.assertEqual(hint["action"], "applied")
        self.assertEqual(hint["typing_source"], "unified_semantics")
        self.assertEqual(candidates[0].kind, "profile")
        self.assertEqual(candidates[0].memory_type, "preference")
        self.assertEqual(candidates[0].content, "用户喜欢低糖拿铁")

    def test_unified_semantic_typing_hint_applies_in_background_memory_job_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢低糖拿铁",
                ),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户喜欢低糖拿铁",
                            "kind": "event",
                            "memory_type": "event",
                            "privacy_level": "normal",
                            "confidence": 0.9,
                            "reason": "extractor mistyped preference",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(
                user_id="u1",
                job_id=result["debug"]["memory_processing"]["job_id"],
            )
            hint = job["extraction_trace"]["unified_semantic_typing_hint"]
            memories = store.list_memories("u1", kind="profile")

            self.assertEqual(hint["action"], "applied")
            self.assertEqual(hint["original_kind"], "profile")
            self.assertEqual(hint["typing_source"], "unified_semantics")
            self.assertEqual(hint["new_kind"], "profile")
            self.assertEqual(hint["new_memory_type"], "preference")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "用户喜欢低糖拿铁")
            self.assertEqual(memories[0].memory_type, "preference")

    def test_unified_semantic_typing_hint_does_not_bypass_sensitive_write_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "memory_recall_type": "event",
                "recall_goal": "summary",
                "event_recall_strategy": "ambiguous_recent_upcoming_plan",
                "confidence": 0.93,
                "reason": "asks for today's personal plan",
            }))
            candidates, hint = GlassesChatService._apply_unified_semantic_typing_hint(
                turn_semantics={
                    "backend": "llm",
                    **semantic_payload(
                        correction=False,
                        memory_action="write",
                        memory_kind="profile",
                        memory_type="fact",
                        candidate_content="用户的 API key 是 sk-proj-secret",
                    ),
                },
                candidates=[
                    MemoryWriteCandidate(
                        content="用户的 API key 是 sk-proj-secret",
                        kind="event",
                        memory_type="event",
                        confidence=0.9,
                        reason="extractor mistyped sensitive fact",
                    )
                ],
            )

            save_result = service._save_memory_candidates(
                candidates=candidates,
                message="我的 API key 是 sk-proj-secret",
                user_id="u1",
                agent=None,
                reference_time=0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertEqual(hint["action"], "applied")
            self.assertEqual(save_result.saved, [])
            self.assertEqual(save_result.rejected[0]["reason"], "candidate_contains_sensitive_term")
            self.assertEqual(store.list_memories("u1"), [])

    def test_unified_semantic_candidate_shadow_records_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户最近更喜欢低糖拿铁",
                ),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户喜欢低糖拿铁",
                            "kind": "profile",
                            "memory_type": "preference",
                            "privacy_level": "normal",
                            "confidence": 0.9,
                            "reason": "stable preference",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我最近更喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)
            shadow = result["debug"]["routing"]["unified_semantic_candidate_shadow"]

            self.assertTrue(shadow["used_turn_semantics"])
            self.assertTrue(shadow["semantic_candidate_present"])
            self.assertFalse(shadow["extractor_candidate_present"])
            self.assertEqual(shadow["content_alignment"], "semantic_candidate_without_extractor_candidate")
            self.assertIsNone(shadow["kind_match"])
            self.assertIsNone(shadow["memory_type_match"])
            self.assertEqual(result["debug"]["intent"]["memory_write_count"], 1)

    def test_unified_semantic_candidate_shadow_is_preserved_in_chat_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户最近更喜欢低糖拿铁",
                ),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户喜欢低糖拿铁",
                            "kind": "profile",
                            "memory_type": "preference",
                            "privacy_level": "normal",
                            "confidence": 0.9,
                            "reason": "stable preference",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            service.chat("我最近更喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)
            records = service.read_audit_records(user_id="u1", limit=5)
            audit_shadow = records[-1]["debug"]["routing"]["unified_semantic_candidate_shadow"]

            self.assertEqual(audit_shadow["content_alignment"], "semantic_candidate_without_extractor_candidate")
            self.assertFalse(audit_shadow["extractor_candidate_present"])

    def test_unified_semantic_candidate_shadow_is_preserved_in_background_memory_job_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户最近更喜欢低糖拿铁",
                ),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户喜欢低糖拿铁",
                            "kind": "profile",
                            "memory_type": "preference",
                            "privacy_level": "normal",
                            "confidence": 0.9,
                            "reason": "stable preference",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我最近更喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(
                user_id="u1",
                job_id=result["debug"]["memory_processing"]["job_id"],
            )
            shadow = job["extraction_trace"]["unified_semantic_candidate_shadow"]

            self.assertEqual(shadow["content_alignment"], "exact")
            self.assertTrue(shadow["extractor_candidate_present"])
            self.assertTrue(shadow["kind_match"])
            self.assertTrue(shadow["memory_type_match"])

    def test_unified_semantic_candidate_shadow_records_semantic_only_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢低糖拿铁",
                ),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我最近更喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)
            shadow = result["debug"]["routing"]["unified_semantic_candidate_shadow"]

            self.assertEqual(shadow["content_alignment"], "semantic_candidate_without_extractor_candidate")
            self.assertTrue(shadow["semantic_candidate_present"])
            self.assertFalse(shadow["extractor_candidate_present"])
            memories = store.list_memories("u1")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "用户喜欢低糖拿铁")

    def test_unified_semantic_shadow_eval_semantic_only_does_not_write_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self.make_store(Path(tmpdir))
            semantic_shadow_eval_suite([
                {
                    "name": "semantic_only",
                    "message": "我最近更喜欢低糖拿铁",
                    "semantic": semantic_payload(
                        correction=False,
                        memory_action="write",
                        memory_kind="profile",
                        memory_type="preference",
                        candidate_content="用户喜欢低糖拿铁",
                    ),
                    "extractor_candidates": [],
                }
            ])

            self.assertEqual(store.list_memories("u1"), [])

    def test_unified_semantic_candidate_shadow_has_no_planner_candidate_when_semantic_missing_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(correction=False, memory_action="write"),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户喜欢低糖拿铁",
                            "kind": "profile",
                            "memory_type": "preference",
                            "privacy_level": "normal",
                            "confidence": 0.9,
                            "reason": "stable preference",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我最近更喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)
            shadow = result["debug"]["routing"]["unified_semantic_candidate_shadow"]

            self.assertEqual(shadow["content_alignment"], "missing")
            self.assertFalse(shadow["semantic_candidate_present"])
            self.assertFalse(shadow["extractor_candidate_present"])
            self.assertEqual(agent.intent_calls, 0)
            self.assertEqual(result["debug"]["intent"]["memory_write_count"], 0)

    def test_unified_semantic_candidate_shadow_records_kind_type_mismatch_without_overriding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="event",
                    memory_type="task",
                    candidate_content="用户喜欢低糖拿铁",
                ),
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户喜欢低糖拿铁",
                            "kind": "profile",
                            "memory_type": "preference",
                            "privacy_level": "normal",
                            "confidence": 0.9,
                            "reason": "stable preference",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)
            shadow = result["debug"]["routing"]["unified_semantic_candidate_shadow"]

            self.assertEqual(shadow["content_alignment"], "semantic_candidate_without_extractor_candidate")
            self.assertIsNone(shadow["kind_match"])
            self.assertIsNone(shadow["memory_type_match"])
            self.assertEqual(result["debug"]["intent"]["memory_write_count"], 1)

    def test_unified_semantic_candidate_shadow_falls_back_for_non_llm_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shadow = FakeService(self.make_store(Path(tmpdir)))._unified_semantic_candidate_shadow(
                turn_semantics={
                    "backend": "rule_fallback",
                    "memory_kind": "profile",
                    "memory_type": "preference",
                    "candidate_content": "用户喜欢低糖拿铁",
                },
                extracted=IntentDecision(
                    backend="llm",
                    memory_write_candidates=[
                        MemoryWriteCandidate(
                            content="用户喜欢低糖拿铁",
                            kind="profile",
                            memory_type="preference",
                        )
                    ],
                ),
            )

        self.assertFalse(shadow["used_turn_semantics"])
        self.assertEqual(shadow["fallback_reason"], "non_llm_semantic_backend:rule_fallback")

    def test_unified_semantic_candidate_shadow_summary_counts_audit_records(self) -> None:
        records = [
            {
                "debug": {
                    "routing": {
                        "unified_semantic_candidate_shadow": {
                            "content_alignment": "exact",
                            "kind_match": True,
                            "memory_type_match": True,
                            "fallback_reason": "",
                        }
                    }
                }
            },
            {
                "extraction_trace": {
                    "unified_semantic_candidate_shadow": {
                        "content_alignment": "extractor_candidate_without_semantic_candidate",
                        "kind_match": False,
                        "memory_type_match": False,
                        "fallback_reason": "",
                    }
                }
            },
            {
                "debug": {
                    "routing": {
                        "unified_semantic_candidate_shadow": {
                            "content_alignment": "different",
                            "fallback_reason": "turn_semantics_error",
                        }
                    }
                }
            },
        ]

        summary = GlassesChatService.summarize_unified_semantic_candidate_shadow(records)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["counts"]["exact"], 1)
        self.assertEqual(summary["counts"]["extractor_candidate_without_semantic_candidate"], 1)
        self.assertEqual(summary["counts"]["fallback"], 1)
        self.assertEqual(summary["kind_mismatch_count"], 1)
        self.assertEqual(summary["memory_type_mismatch_count"], 1)
        self.assertEqual(summary["fallback_reasons"]["turn_semantics_error"], 1)
        self.assertEqual(summary["recommended_next_action"], "fix_semantic_backend_or_fallback")
        self.assertEqual(summary["migration_readiness"], "observe_or_fix_first")

    def test_unified_semantic_candidate_shadow_summary_recommends_typing_hint_only_when_aligned(self) -> None:
        records = [
            {
                "debug": {
                    "routing": {
                        "unified_semantic_candidate_shadow": {
                            "content_alignment": "exact",
                            "kind_match": True,
                            "memory_type_match": True,
                            "fallback_reason": "",
                        }
                    }
                }
            },
            {
                "extraction_trace": {
                    "unified_semantic_candidate_shadow": {
                        "content_alignment": "contains_or_similar",
                        "kind_match": True,
                        "memory_type_match": True,
                        "fallback_reason": "",
                    }
                }
            },
        ]

        summary = GlassesChatService.summarize_unified_semantic_candidate_shadow(records)

        self.assertEqual(summary["recommended_next_action"], "consider_typing_hint")
        self.assertEqual(summary["migration_readiness"], "typing_hint_candidate")

    def test_unified_semantic_typing_hint_summary_counts_actions_and_sources(self) -> None:
        records = [
            {
                "debug": {
                    "routing": {
                        "unified_semantic_typing_hint": {
                            "action": "applied",
                            "typing_source": "unified_semantics",
                        }
                    }
                }
            },
            {
                "debug": {
                    "routing": {
                        "unified_semantic_candidate_shadow": {
                            "content_alignment": "exact",
                            "kind_match": True,
                            "memory_type_match": True,
                            "fallback_reason": "",
                        }
                    }
                },
                "extraction_trace": {
                    "unified_semantic_typing_hint": {
                        "action": "skipped",
                        "typing_source": "extractor_aligned",
                    }
                },
            },
            {
                "extraction_trace": {
                    "unified_semantic_typing_hint": {
                        "action": "fallback",
                        "typing_source": "semantic_unavailable",
                    }
                },
            },
        ]

        typing_summary = GlassesChatService.summarize_unified_semantic_typing_hint(records)
        shadow_summary = GlassesChatService.summarize_unified_semantic_candidate_shadow(records)

        self.assertEqual(typing_summary["total"], 3)
        self.assertEqual(typing_summary["actions"]["applied"], 1)
        self.assertEqual(typing_summary["actions"]["skipped"], 1)
        self.assertEqual(typing_summary["actions"]["fallback"], 1)
        self.assertEqual(typing_summary["typing_sources"]["unified_semantics"], 1)
        self.assertEqual(typing_summary["typing_sources"]["extractor_aligned"], 1)
        self.assertEqual(typing_summary["typing_sources"]["semantic_unavailable"], 1)
        self.assertEqual(shadow_summary["typing_hint_summary"], typing_summary)

    def test_unified_semantic_shadow_eval_suite_covers_common_semantic_cases(self) -> None:
        suite = semantic_shadow_eval_suite([
            {
                "name": "stable_preference",
                "message": "我最近更喜欢低糖拿铁",
                "semantic": semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户最近更喜欢低糖拿铁",
                ),
                "extractor_candidates": [{
                    "content": "用户喜欢低糖拿铁",
                    "kind": "profile",
                    "memory_type": "preference",
                    "confidence": 0.9,
                }],
            },
            {
                "name": "task_plan",
                "message": "明天下午提醒我看演示稿",
                "semantic": semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="event",
                    memory_type="task",
                    candidate_content="明天下午看演示稿",
                ),
                "extractor_candidates": [{
                    "content": "明天下午看演示稿",
                    "kind": "event",
                    "memory_type": "task",
                    "confidence": 0.9,
                }],
            },
            {
                "name": "project_state",
                "message": "现在主线是 AI 眼镜长期记忆",
                "semantic": semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="event",
                    memory_type="project_state",
                    candidate_content="现在主线是 AI 眼镜长期记忆",
                ),
                "extractor_candidates": [{
                    "content": "现在主线是 AI 眼镜长期记忆",
                    "kind": "event",
                    "memory_type": "project_state",
                    "confidence": 0.9,
                }],
            },
            {
                "name": "correction",
                "message": "刚才说错了，不是周三，是周四",
                "semantic": semantic_payload(
                    correction=True,
                    memory_action="correction",
                    memory_kind="event",
                    memory_type="task",
                    candidate_content="周四",
                ),
                "extractor_candidates": [{
                    "content": "周四",
                    "kind": "event",
                    "memory_type": "task",
                    "confidence": 0.9,
                }],
            },
            {
                "name": "transient",
                "message": "只是临时想法，别记住",
                "semantic": semantic_payload(
                    correction=False,
                    memory_action="none",
                    transient=True,
                    candidate_content="",
                ),
                "extractor_candidates": [],
            },
            {
                "name": "do_not_remember",
                "message": "这句不要保存",
                "semantic": semantic_payload(
                    correction=False,
                    memory_action="none",
                    do_not_remember=True,
                    candidate_content="",
                ),
                "extractor_candidates": [],
            },
            {
                "name": "recall",
                "message": "我昨天中午吃了什么？",
                "semantic": semantic_payload(
                    correction=False,
                    memory_action="recall",
                    recall_type="event",
                    candidate_content="",
                ),
                "extractor_candidates": [],
            },
            {
                "name": "chitchat",
                "message": "哈哈这个挺有意思",
                "semantic": semantic_payload(
                    correction=False,
                    memory_action="none",
                    candidate_content="",
                ),
                "extractor_candidates": [],
            },
        ])

        self.assertEqual(len(suite["records"]), 8)
        self.assertEqual(suite["summary"]["counts"]["exact"], 3)
        self.assertEqual(suite["summary"]["counts"]["contains_or_similar"], 1)
        self.assertEqual(suite["summary"]["counts"]["missing"], 4)
        self.assertEqual(suite["summary"]["recommended_next_action"], "collect_more_audit")
        self.assertEqual(suite["summary"]["migration_readiness"], "observe_or_fix_first")

        write_case_records = [
            record
            for record in suite["records"]
            if record["case"] in {"stable_preference", "task_plan", "project_state", "correction"}
        ]
        write_case_summary = GlassesChatService.summarize_unified_semantic_candidate_shadow(write_case_records)
        self.assertEqual(write_case_summary["recommended_next_action"], "consider_typing_hint")
        self.assertEqual(write_case_summary["migration_readiness"], "typing_hint_candidate")

    def test_unified_semantic_shadow_eval_suite_routes_gap_examples(self) -> None:
        cases = (
            (
                "semantic_only",
                [{
                    "name": "semantic_only",
                    "message": "我最近更喜欢低糖拿铁",
                    "semantic": semantic_payload(
                        correction=False,
                        memory_action="write",
                        memory_kind="profile",
                        memory_type="preference",
                        candidate_content="用户喜欢低糖拿铁",
                    ),
                    "extractor_candidates": [],
                }],
                "inspect_extractor_missed_candidates",
            ),
            (
                "extractor_only",
                [{
                    "name": "extractor_only",
                    "message": "我最近更喜欢低糖拿铁",
                    "semantic": semantic_payload(correction=False, memory_action="write"),
                    "extractor_candidates": [{
                        "content": "用户喜欢低糖拿铁",
                        "kind": "profile",
                        "memory_type": "preference",
                    }],
                }],
                "inspect_unified_semantic_missed_candidates",
            ),
            (
                "type_mismatch",
                [{
                    "name": "type_mismatch",
                    "message": "我最近更喜欢低糖拿铁",
                    "semantic": semantic_payload(
                        correction=False,
                        memory_action="write",
                        memory_kind="event",
                        memory_type="task",
                        candidate_content="用户喜欢低糖拿铁",
                    ),
                    "extractor_candidates": [{
                        "content": "用户喜欢低糖拿铁",
                        "kind": "profile",
                        "memory_type": "preference",
                    }],
                }],
                "fix_semantic_typing_or_prompt",
            ),
            (
                "fallback",
                [{
                    "name": "fallback",
                    "message": "我最近更喜欢低糖拿铁",
                    "semantic": "not json",
                    "extractor_candidates": [{
                        "content": "用户喜欢低糖拿铁",
                        "kind": "profile",
                        "memory_type": "preference",
                    }],
                }],
                "fix_semantic_backend_or_fallback",
            ),
        )
        for name, eval_cases, expected_action in cases:
            with self.subTest(name=name):
                suite = semantic_shadow_eval_suite(eval_cases)

                self.assertEqual(suite["summary"]["recommended_next_action"], expected_action)
                self.assertEqual(suite["summary"]["migration_readiness"], "observe_or_fix_first")

    def test_unified_semantic_candidate_shadow_summary_routes_common_gap_types(self) -> None:
        cases = (
            ("semantic_candidate_without_extractor_candidate", "inspect_extractor_missed_candidates"),
            ("extractor_candidate_without_semantic_candidate", "inspect_unified_semantic_missed_candidates"),
            ("different", "fix_semantic_typing_or_prompt"),
        )
        for alignment, expected_action in cases:
            with self.subTest(alignment=alignment):
                records = [
                    {
                        "debug": {
                            "routing": {
                                "unified_semantic_candidate_shadow": {
                                    "content_alignment": alignment,
                                    "kind_match": True,
                                    "memory_type_match": True,
                                    "fallback_reason": "",
                                }
                            }
                        }
                    }
                ]

                summary = GlassesChatService.summarize_unified_semantic_candidate_shadow(records)

                self.assertEqual(summary["recommended_next_action"], expected_action)
                self.assertEqual(summary["migration_readiness"], "observe_or_fix_first")

    def test_unified_semantic_candidate_shadow_summary_routes_kind_type_mismatch(self) -> None:
        records = [
            {
                "debug": {
                    "routing": {
                        "unified_semantic_candidate_shadow": {
                            "content_alignment": "exact",
                            "kind_match": False,
                            "memory_type_match": True,
                            "fallback_reason": "",
                        }
                    }
                }
            }
        ]

        summary = GlassesChatService.summarize_unified_semantic_candidate_shadow(records)

        self.assertEqual(summary["recommended_next_action"], "fix_semantic_typing_or_prompt")
        self.assertEqual(summary["migration_readiness"], "observe_or_fix_first")

    def test_unified_semantic_memory_extraction_gate_falls_back_on_error(self) -> None:
        cases = (
            (
                "not json",
                "turn_semantics_error",
            ),
        )
        for semantic_payload_value, fallback_reason in cases:
            with self.subTest(fallback_reason=fallback_reason):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    store = self.make_store(Path(tmpdir))
                    agent = FakeAgent(
                        semantic_payload=semantic_payload_value,
                        intent_payload={
                            "needs_web_search": False,
                            "web_query": None,
                            "web_reason": "",
                            "memory_write_candidates": [],
                            "confidence": 0.9,
                        },
                    )
                    service = FakeService(store, agent=agent)

                    result = service.chat("普通对话", user_id="u1", defer_memory_writes=True)
                    gate = result["debug"]["routing"]["unified_semantic_memory_extraction_gate"]

                    self.assertEqual(agent.intent_calls, 0)
                    self.assertEqual(gate["action"], "fallback")
                    self.assertEqual(gate["fallback_reason"], fallback_reason)

    def test_unified_semantic_memory_extraction_gate_rule_backend_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = FakeService(self.make_store(Path(tmpdir)))._semantic_memory_extraction_gate(
                {"backend": "rule_fallback", "flags": {"correction": False}, "memory_action": "none"}
            )

        self.assertEqual(gate["action"], "fallback")
        self.assertEqual(gate["fallback_reason"], "non_llm_semantic_backend:rule_fallback")

    def test_turn_semantics_explanation_flag_can_trigger_local_explanation_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(
                store,
                agent=FakeAgent(
                    semantic_payload={
                        "turn_intent": "explanation",
                        "memory_action": "explain",
                        "memory_kind": "none",
                        "memory_type": "none",
                        "recall_type": "none",
                        "reply_mode_hint": "llm",
                        "flags": {
                            "transient": False,
                            "do_not_remember": False,
                            "correction": False,
                            "explanation_query": True,
                        },
                        "candidate_content": "",
                        "reason": "user asks why prior answer was produced",
                        "confidence": 0.96,
                    }
                ),
            )
            service._append_audit_record(
                {
                    "record_type": "chat_turn",
                    "user_id": "u1",
                    "message": "我最近更喜欢低糖拿铁",
                    "reply": "好，我记住了。",
                    "source_summary": {"primary_source": "none"},
                    "memory_processing": {"status": "saved"},
                    "recent_context_capsule": {"available": False},
                    "timestamp": 1778131200.0,
                }
            )

            result = service.chat("上一条为什么这么回答？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["turn_semantics"]["turn_intent"], "explanation")
            self.assertTrue(result["debug"]["turn_semantics"]["flags"]["explanation_query"])
            self.assertIn("依据", result["reply"])
            self.assertIn("local_explanation_reply", result["debug"]["steps"])

    def test_pre_reply_decision_profile_recall_opens_profile_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户喜欢安静靠窗的位置", kind="profile", memory_type="preference")
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_location": False,
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_recall_type": "profile",
                    "recall_goal": "specific_fact",
                    "confidence": 0.95,
                    "reason": "pre_reply profile recall",
                },
                semantic_payload={
                    "turn_intent": "memory_recall",
                    "memory_action": "recall",
                    "memory_kind": "none",
                    "memory_type": "none",
                    "recall_type": "profile",
                    "reply_mode_hint": "local_profile_recall",
                    "flags": {
                        "transient": False,
                        "do_not_remember": False,
                        "correction": False,
                        "explanation_query": False,
                    },
                    "candidate_content": "",
                    "reason": "asks about stable preference",
                    "confidence": 0.93,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我喜欢什么座位？", user_id="u1", routing_mode="llm_first")
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "profile")
            self.assertTrue(result["debug"]["planner"]["needs_profile_memory"])
            self.assertEqual(result["debug"]["memory"]["profile_count"], 1)

    def test_pre_reply_decision_observation_recall_sets_observation_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "AI 眼镜项目最近主要在收敛 recall 和 debug 边界。", kind="event", memory_type="project_state")
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_location": False,
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "pre_reply observation recall",
                },
                semantic_payload={
                    "turn_intent": "memory_recall",
                    "memory_action": "recall",
                    "memory_kind": "none",
                    "memory_type": "none",
                    "recall_type": "observation",
                    "reply_mode_hint": "llm",
                    "flags": {
                        "transient": False,
                        "do_not_remember": False,
                        "correction": False,
                        "explanation_query": False,
                    },
                    "candidate_content": "",
                    "reason": "asks for high level summary",
                    "confidence": 0.94,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("这阵子我的主要投入方向是什么？", user_id="u1", routing_mode="llm_first")
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "observation")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "observation_review")

    def test_pre_reply_decision_timeline_recall_is_visible_in_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(
                store,
                agent=FakeAgent(
                    pre_reply_payload={
                        "reply_mode": "llm",
                        "answer_source": "llm",
                        "scope": "unknown",
                        "needs_location": False,
                        "needs_web_search": False,
                        "web_query": None,
                        "web_reason": "",
                        "memory_recall_type": "timeline",
                        "recall_goal": "raw_evidence",
                        "confidence": 0.95,
                        "reason": "pre_reply raw timeline recall",
                    },
                    semantic_payload={
                        "turn_intent": "timeline_recall",
                        "memory_action": "recall",
                        "memory_kind": "none",
                        "memory_type": "none",
                        "recall_type": "timeline",
                        "reply_mode_hint": "local_timeline_recall",
                        "flags": {
                            "transient": False,
                            "do_not_remember": False,
                            "correction": False,
                            "explanation_query": False,
                        },
                        "candidate_content": "",
                        "reason": "asks for original wording",
                        "confidence": 0.95,
                    },
                ),
            )
            service.timeline_store.add_turn("u1", "我刚才说今天下午三点开会。", created_at=1778131100.0)

            result = service.chat("我刚才原话怎么说的？", user_id="u1", routing_mode="llm_first")
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "timeline")
            self.assertTrue(result["debug"]["planner"]["needs_timeline_recall"])
            self.assertEqual(result["debug"]["timeline"]["recall"]["strategy"], "chunk_full_text")

    def test_pre_reply_decision_recall_none_keeps_existing_route_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "最近主要在做 recall 迁移。", kind="event", memory_type="project_state")
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_location": False,
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "pre_reply summary recall",
                },
                semantic_payload={
                    "turn_intent": "chat",
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
                    "reason": "no unified recall override",
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("这阵子我的主要投入方向是什么？", user_id="u1", routing_mode="llm_first")
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "observation")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "observation_review")

    def test_pre_reply_decision_can_open_multiple_recall_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户喜欢靠窗位置。", kind="profile", memory_type="preference")
            store.add_memory("u1", "昨天和 Alex 开了周会。", kind="event", memory_type="event")
            service = FakeService(
                store,
                agent=FakeAgent(
                    pre_reply_payload={
                        "reply_mode": "llm",
                        "answer_source": "llm",
                        "scope": "unknown",
                        "needs_location": False,
                        "needs_web_search": False,
                        "web_query": None,
                        "web_reason": "",
                        "needs_profile_memory": True,
                        "memory_recall_type": "event",
                        "recall_goal": "specific_fact",
                        "confidence": 0.95,
                        "reason": "pre_reply asks for event recall with profile context",
                    },
                    semantic_payload={
                        "turn_intent": "memory_recall",
                        "memory_action": "recall",
                        "memory_kind": "none",
                        "memory_type": "none",
                        "recall_type": "profile",
                        "reply_mode_hint": "llm",
                        "flags": {
                            "transient": False,
                            "do_not_remember": False,
                            "correction": False,
                            "explanation_query": False,
                        },
                        "candidate_content": "",
                        "reason": "semantic sees profile angle too",
                        "confidence": 0.92,
                    },
                ),
            )

            result = service.chat("我昨天和 Alex 说了什么偏好？", user_id="u1", routing_mode="llm_first")
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "event")
            self.assertEqual(result["debug"]["memory"]["profile_count"], 1)
            self.assertTrue(result["debug"]["turn_decision"]["final"]["needs_event_memory"])
            self.assertIn(
                result["debug"]["memory"]["event_recall"]["strategy"],
                {"text_search", "temporal_range", "observation_review"},
            )

    def test_assistant_response_timing_breaks_down_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            agent = TimedFakeAgent()
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=agent)

            result = service.chat("普通对话", user_id="u1")

            detail = result["debug"]["assistant_response_timing"]
            assistant_stage = next(
                stage for stage in result["debug"]["timing"]["stages"]
                if stage["name"] == "assistant_response"
            )
            self.assertEqual(detail["total_seconds"], assistant_stage["seconds"])
            self.assertEqual(len(detail["api_calls"]), 2)
            self.assertEqual(detail["api_calls"][0]["finish_type"], "tool_calls")
            self.assertEqual(detail["api_calls"][0]["phase"], "initial_model_request")
            self.assertIn("请求工具", detail["api_calls"][0]["phase_label"])
            self.assertEqual(detail["api_calls"][0]["requested_tools"][0]["name"], "example_agent_tool")
            self.assertEqual(detail["api_calls"][0]["requested_tools"][0]["argument_keys"], ["limit", "query", "secret"])
            self.assertNotIn("arguments", detail["api_calls"][0]["requested_tools"][0])
            self.assertEqual(detail["api_calls"][1]["finish_type"], "final_response")
            self.assertEqual(detail["api_calls"][1]["phase"], "model_after_tool_results")
            self.assertEqual(detail["api_calls"][1]["previous_tool_result_count"], 1)
            self.assertEqual(detail["api_calls"][1]["previous_tools"][0]["name"], "example_agent_tool")
            self.assertEqual(detail["api_calls"][1]["previous_tools"][0]["argument_keys"], ["limit", "query", "secret"])
            self.assertNotIn("arguments", detail["api_calls"][1]["previous_tools"][0])
            self.assertIn("工具结果后的模型续写", detail["api_calls"][1]["phase_label"])
            self.assertEqual(len(detail["agent_tool_calls"]), 1)
            self.assertEqual(detail["agent_tool_calls"][0]["name"], "example_agent_tool")
            self.assertEqual(detail["agent_tool_calls"][0]["argument_keys"], ["limit", "query", "secret"])
            self.assertNotIn("arguments", detail["agent_tool_calls"][0])
            serialized_detail = json.dumps(detail, ensure_ascii=False)
            self.assertNotIn("天气 北京", serialized_detail)
            self.assertNotIn("never-log-me", serialized_detail)
            self.assertGreaterEqual(detail["llm_wait_seconds"], 0)
            self.assertGreaterEqual(detail["agent_tool_seconds"], 0)

    def test_memory_write_gate_rejects_meta_profile_and_low_confidence(self) -> None:
        cases = [
            {
                "content": "用户询问自己的身份",
                "kind": "profile",
                "confidence": 0.9,
                "reason": "meta profile",
            },
            {
                "content": "用户名字叫测试",
                "kind": "profile",
                "confidence": 0.3,
                "reason": "low confidence",
            },
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate):
                memory_candidate = MemoryWriteCandidate(
                    content=candidate["content"],
                    kind=candidate["kind"],
                    confidence=candidate["confidence"],
                    reason=candidate["reason"],
                )
                self.assertFalse(
                    should_write_memory_candidate(memory_candidate, "普通对话").allowed
                )
                gate = should_write_memory_candidate(memory_candidate, "普通对话")
                if gate.reason == "candidate_confidence_below_threshold":
                    self.assertEqual(gate.confidence_policy["purpose"], "memory_write_candidate")
                    self.assertEqual(gate.confidence_policy["min_confidence"], 0.6)
                    self.assertEqual(gate.confidence_policy["treatment"], "reject")
                    self.assertFalse(gate.confidence_policy["passed"])

    def test_natural_preference_without_defer_saves_pre_reply_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(semantic_payload=semantic_payload(
                correction=False,
                memory_action="write",
                memory_kind="profile",
                memory_type="preference",
                candidate_content="用户喜欢低糖拿铁",
            )))

            result = service.chat("我喜欢低糖拿铁", user_id="u1")

            self.assertEqual(result["saved_memories"][0]["content"], "用户喜欢低糖拿铁")
            self.assertEqual(store.list_memories("u1", kind="profile")[0].memory_type, "preference")
            self.assertGreaterEqual(service.fake_agent.semantic_calls, 1)
            self.assertEqual(service.fake_agent.temporal_calls, 0)

    def test_memory_write_gate_rejects_question_text_but_allows_memory_request_fact(self) -> None:
        question_candidate = MemoryWriteCandidate(
            content="我是不是喜欢很甜的饮料？",
            kind="profile",
            confidence=0.95,
        )
        stable_fact_candidate = MemoryWriteCandidate(
            content="用户喜欢低糖拿铁",
            kind="profile",
            confidence=0.95,
        )

        question_gate = should_write_memory_candidate(
            question_candidate,
            "我是不是喜欢很甜的饮料？",
        )
        self.assertFalse(question_gate.allowed)
        self.assertEqual(question_gate.question_policy["role"], "weak_signal")
        self.assertEqual(question_gate.question_policy["source"], "candidate")
        self.assertEqual(question_gate.question_policy["treatment"], "reject_candidate_question_text")
        self.assertTrue(
            should_write_memory_candidate(
                stable_fact_candidate,
                "你能记住我喜欢低糖拿铁吗？",
            ).allowed
        )

    def test_memory_write_gate_allows_question_form_memory_request_event(self) -> None:
        candidate = MemoryWriteCandidate(
            content="明天下午3点和Mina开会",
            kind="event",
            memory_type="event",
            confidence=0.95,
        )

        gate = should_write_memory_candidate(candidate, "你能帮我记一下明天下午3点和 Mina 开会吗？")

        self.assertTrue(gate.allowed)
        self.assertEqual(gate.reason, "allowed")

    def test_memory_write_gate_does_not_let_source_question_word_veto_event_candidate(self) -> None:
        candidate = MemoryWriteCandidate(
            content="今天下午3点和Mina开会",
            kind="event",
            memory_type="event",
            confidence=0.95,
        )

        gate = should_write_memory_candidate(
            candidate,
            "额嗯嗯，今天，呃呃呃，我忘记要说啥了，嗯嗯，我想起来的，好像要和 Mina 开会，估计在三点吧",
        )

        self.assertTrue(gate.allowed)
        self.assertEqual(gate.reason, "allowed")

    def test_memory_write_gate_still_rejects_direct_question_event_candidate(self) -> None:
        candidate = MemoryWriteCandidate(
            content="今天三点和 Mina 开会",
            kind="event",
            memory_type="event",
            confidence=0.95,
        )

        gate = should_write_memory_candidate(candidate, "今天三点和 Mina 开会是啥意思？")

        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reason, "source_question_without_memory_request")
        self.assertEqual(gate.question_policy["role"], "weak_signal")
        self.assertEqual(gate.question_policy["source"], "source_message")
        self.assertEqual(gate.question_policy["treatment"], "reject_source_question_without_memory_request")
        self.assertFalse(gate.question_policy["overrides_llm"])

    def test_memory_write_gate_rejects_unpunctuated_direct_question_event_candidate(self) -> None:
        candidate = MemoryWriteCandidate(
            content="今天三点和 Mina 开会",
            kind="event",
            memory_type="event",
            confidence=0.95,
        )

        gate = should_write_memory_candidate(candidate, "今天三点和 Mina 开会是啥意思")

        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reason, "source_question_without_memory_request")
        self.assertEqual(gate.question_policy["role"], "weak_signal")

    def test_memory_write_gate_rejects_identity_question_shape(self) -> None:
        candidate = MemoryWriteCandidate(
            content="我是什么身份",
            kind="event",
            memory_type="event",
            confidence=0.95,
        )

        gate = should_write_memory_candidate(candidate, "我是什么身份")

        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reason, "candidate_is_question_text")
        self.assertEqual(gate.question_policy["source"], "candidate")

    def test_memory_write_gate_requires_confirmation_for_sensitive_content(self) -> None:
        candidate = MemoryWriteCandidate(
            content="我的 API key 是 sk-test-123",
            kind="profile",
            confidence=0.95,
        )

        gate = should_write_memory_candidate(candidate, "帮我记住我的 API key 是 sk-test-123")

        self.assertFalse(gate.allowed)
        self.assertTrue(gate.requires_confirmation)
        self.assertEqual(gate.privacy_level, "requires_confirmation")
        self.assertEqual(gate.safety_policy["role"], "hard_safety")
        self.assertEqual(gate.safety_policy["reason"], "candidate_contains_sensitive_term")
        self.assertTrue(gate.safety_policy["affects_final_decision"])

    def test_memory_write_gate_allows_document_object_schedule_without_secret_value(self) -> None:
        candidate = MemoryWriteCandidate(
            content="后天下午5点取护照",
            kind="event",
            confidence=0.95,
            reason="explicit_memory_command",
        )

        gate = should_write_memory_candidate(candidate, "记一下后天下午5点取护照")

        self.assertTrue(gate.allowed)
        self.assertEqual(gate.reason, "allowed")

    def test_sensitive_credential_statement_rejected_without_llm_or_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("我的 codex api key 是 sk-1234567890abcdefghijklmnopqr", user_id="u1")

            self.assertIn("不能保存", result["reply"])
            self.assertIn("不会帮你长期记忆", result["reply"])
            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "sensitive_credential_rejected")
            self.assertEqual(result["debug"]["memory_processing"]["status"], "not_needed")
            self.assertEqual(result["debug"]["memory_processing"]["safety_policy"]["role"], "hard_safety")
            self.assertEqual(
                result["debug"]["memory_processing"]["safety_policy"]["reason"],
                "matched_sensitive_credential_input",
            )
            self.assertEqual(service.fake_agent.main_calls, 0)

    def test_sensitive_profile_recall_does_not_return_unrelated_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户名字叫Jack", kind="profile")
            store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile")
            store.add_memory("u1", "用户有一份关于自驾游的文档", kind="profile")
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks for stored API key",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("我的 API Key 是什么", user_id="u1")

            self.assertIn("不会保存或回忆", result["reply"])
            self.assertIn("敏感凭据", result["reply"])
            self.assertNotIn("Jack", result["reply"])
            self.assertNotIn("低糖拿铁", result["reply"])
            self.assertNotIn("自驾游", result["reply"])
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_profile_recall")
            self.assertEqual(agent.main_calls, 1)

    def test_sensitive_asr_like_message_exposes_memory_skip_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat(
                "刚才转写可能错了，验证码 123456，身份证 110101199001011234，还有 token sk-test-abcdef 这些都不要记，真的不要保存。",
                user_id="u1",
            )

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(result["debug"]["memory_processing"]["status"], "not_needed")
            self.assertEqual(
                result["debug"]["memory_processing"]["decision_reason"],
                "source_contains_sensitive_redaction",
            )
            self.assertEqual(result["debug"]["memory_processing"]["safety_policy"]["role"], "hard_safety")
            self.assertEqual(
                result["debug"]["memory_processing"]["safety_policy"]["reason"],
                "source_contains_sensitive_redaction",
            )

    def test_specific_profile_recall_missing_does_not_dump_all_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile")
            store.add_memory("u1", "用户有一份关于自驾游的文档", kind="profile")
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks for a specific stored profile field",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("我的手机号是什么", user_id="u1")

            self.assertIn("没有找到这条画像记忆", result["reply"])
            self.assertNotIn("低糖拿铁", result["reply"])
            self.assertNotIn("自驾游", result["reply"])
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_profile_recall")
            self.assertEqual(agent.main_calls, 1)

    def test_preference_question_uses_profile_memory_without_saving_question_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户不喜欢太甜的饮料", kind="profile")
            service = FakeService(store)
            result = service.chat("我是不是喜欢很甜的饮料？", user_id="u1")

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(len(store.list_memories("u1")), 1)
            self.assertEqual(result["debug"]["memory_processing"]["status"], "not_needed")
            self.assertIn("不喜欢", result["reply"])
            self.assertIn(service.fake_agent.main_calls, {0, 1})

    def test_recalled_preference_records_access_and_returns_updated_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            memory = store.add_memory(
                "u1",
                "用户喜欢安静靠窗的位置",
                kind="profile",
                memory_type="preference",
            )
            service = FakeService(store)

            first = service.chat("以后订座位优先考虑什么？", user_id="u1")
            second = service.chat("以后订座位优先考虑什么？", user_id="u1")
            refreshed = store.get_memory("u1", memory.id)

            self.assertEqual(refreshed.access_count, 2)
            self.assertEqual(refreshed.last_accessed_at, 1778131200.0)
            self.assertEqual(second["recalled_memories"][0]["access_count"], 2)
            self.assertEqual(second["debug"]["memory"]["profile_memories"][0]["access_count"], 2)
            self.assertEqual(second["debug"]["memory"]["access"]["count"], 1)
            self.assertIn(memory.id, second["debug"]["memory"]["access"]["memory_ids"])
            self.assertGreaterEqual(second["recalled_memories"][0]["strength"], first["recalled_memories"][0]["strength"])

    def test_profile_recall_light_ranking_prefers_accessed_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            lower = store.add_memory("u1", "用户不喜欢排队很久的餐厅", kind="profile", memory_type="preference")
            higher = store.add_memory("u1", "用户不喜欢太吵的餐厅", kind="profile", memory_type="preference")
            store.record_memory_access("u1", [lower.id], accessed_at=1778131210.0)
            service = FakeService(store)

            result = service.chat("以后推荐餐厅时要避开什么？", user_id="u1")

            recalled_ids = [memory["id"] for memory in result["recalled_memories"]]
            self.assertEqual(recalled_ids[:2], [lower.id, higher.id])
            self.assertEqual(result["debug"]["memory"]["profile_memories"][0]["id"], lower.id)
            self.assertEqual(result["debug"]["memory"]["drift_guard"]["filtered_count"], 0)
            self.assertEqual(store.get_memory("u1", lower.id).access_count, 2)

    def test_drift_guard_filters_high_strength_unrelated_profile_before_llm_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            memory = store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile", memory_type="preference")
            for _ in range(5):
                store.record_memory_access("u1", [memory.id], accessed_at=1778131210.0)
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "forced profile recall for unrelated question",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("讲讲 HTTP 路由机制", user_id="u1", routing_mode="llm_first")

            guard = result["debug"]["memory"]["drift_guard"]
            self.assertEqual(result["reply"], "主回复")
            self.assertEqual(result["debug"]["memory"]["profile_count"], 0)
            self.assertEqual(guard["checked_count"], 1)
            self.assertEqual(guard["filtered_count"], 1)
            self.assertIn(memory.id, guard["filtered_memory_ids"])
            self.assertEqual(guard["filtered_reasons"][0]["reason"], "profile_query_topic_mismatch")
            self.assertNotIn("用户喜欢低糖拿铁", agent.main_messages[-1])
            self.assertEqual(store.get_memory("u1", memory.id).access_count, 5)

    def test_preference_word_in_knowledge_question_does_not_open_profile_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            memory = store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile", memory_type="preference")
            service = FakeService(store)

            result = service.chat("推荐算法是什么？", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["reply_mode"], "llm")
            self.assertFalse(result["debug"]["planner"]["needs_profile_memory"])
            self.assertEqual(result["debug"]["memory"]["profile_count"], 0)
            self.assertEqual(result["debug"]["memory"]["drift_guard"]["skipped_reason"], "no_recalled_memories")
            self.assertNotIn("用户喜欢低糖拿铁", service.fake_agent.main_messages[-1])
            self.assertEqual(store.get_memory("u1", memory.id).access_count, 0)

    def test_drift_guard_keeps_related_preference_and_filters_other_topics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            drink = store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile", memory_type="preference")
            seat = store.add_memory("u1", "用户喜欢安静靠窗的位置", kind="profile", memory_type="preference")
            service = FakeService(store)

            result = service.chat("我喜欢喝什么？", user_id="u1")

            guard = result["debug"]["memory"]["drift_guard"]
            self.assertIn("低糖拿铁", result["reply"])
            self.assertNotIn("靠窗", result["reply"])
            self.assertEqual(result["debug"]["memory"]["profile_count"], 1)
            self.assertEqual(guard["checked_count"], 2)
            self.assertEqual(guard["kept_count"], 1)
            self.assertEqual(guard["filtered_memory_ids"], [seat.id])
            self.assertEqual(store.get_memory("u1", drink.id).access_count, 1)
            self.assertEqual(store.get_memory("u1", seat.id).access_count, 0)

    def test_profile_recall_generic_preference_keeps_preference_without_topic_phrase_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            memory = store.add_memory("u1", "用户偏好先看可验证证据再做结论", kind="profile", memory_type="preference")
            service = FakeService(store)

            result = service.chat("我有什么偏好？", user_id="u1")

            guard = result["debug"]["memory"]["drift_guard"]
            self.assertIn("可验证证据", result["reply"])
            self.assertEqual(result["debug"]["memory"]["profile_count"], 1)
            self.assertEqual(guard["filtered_count"], 0)
            self.assertEqual(result["recalled_memories"][0]["id"], memory.id)
            self.assertEqual(store.get_memory("u1", memory.id).access_count, 1)

    def test_drift_guard_trusts_pre_reply_profile_recall_when_topic_words_are_generic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            memory = store.add_memory("u1", "用户偏好先看可验证证据再做结论", kind="profile", memory_type="preference")
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "generic preference recall",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("我的判断偏好是什么？", user_id="u1", routing_mode="llm_first")

            guard = result["debug"]["memory"]["drift_guard"]
            self.assertEqual(result["debug"]["memory"]["profile_count"], 1)
            self.assertEqual(guard["filtered_count"], 0)
            self.assertIn("用户偏好先看可验证证据再做结论", agent.main_messages[-1])
            self.assertEqual(result["recalled_memories"][0]["id"], memory.id)

    def test_drift_guard_filters_conflicting_current_intent_before_llm_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            memory = store.add_memory("u1", "用户喜欢安静靠窗的位置", kind="profile", memory_type="preference")
            service = FakeService(store)

            result = service.chat("这次订座不要靠窗，想坐吧台可以吗", user_id="u1")

            guard = result["debug"]["memory"]["drift_guard"]
            self.assertEqual(result["reply"], "主回复")
            self.assertEqual(result["debug"]["memory"]["profile_count"], 0)
            self.assertEqual(guard["filtered_memory_ids"], [memory.id])
            self.assertEqual(guard["filtered_reasons"][0]["reason"], "current_intent_conflicts_with_preference")
            self.assertNotIn("用户喜欢安静靠窗的位置", service.fake_agent.main_messages[-1])
            self.assertEqual(store.get_memory("u1", memory.id).access_count, 0)

    def test_memory_search_with_ranking_prefers_accessed_memory_on_same_text_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            accessed = store.add_memory("u1", "用户讨论 ranking recall 排序", kind="event")
            fresh = store.add_memory("u1", "用户讨论 ranking recall 排序", kind="event")
            for _ in range(10):
                store.record_memory_access("u1", [accessed.id], accessed_at=1778131210.0)

            result = store.search_with_ranking("u1", "ranking recall", limit=2)

            self.assertEqual({memory.id for memory in result.memories}, {accessed.id, fresh.id})
            accessed_rank = next(rank for rank in result.ranking if rank["id"] == accessed.id)
            fresh_rank = next(rank for rank in result.ranking if rank["id"] == fresh.id)
            self.assertGreater(accessed_rank["access_score"], fresh_rank["access_score"])
            self.assertIn("effective_strength_score", result.ranking[0])
            self.assertIn("strength_cap", result.ranking[0])
            self.assertIn("decay_factor", result.ranking[0])
            self.assertIn("strength_policy_reason", result.ranking[0])

    def test_effective_strength_decay_keeps_new_preference_ahead_of_old_high_access_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            old = store.add_memory("u1", "用户喜欢靠窗座位", kind="profile", memory_type="preference")
            new = store.add_memory("u1", "用户喜欢吧台座位", kind="profile", memory_type="preference")
            old_timestamp = 1762041600.0
            store._conn.execute(
                """
                UPDATE memories
                SET access_count = 20, strength = 0.98, created_at = ?, updated_at = ?, last_accessed_at = ?
                WHERE id = ?
                """,
                (old_timestamp, old_timestamp, old_timestamp, old.id),
            )
            store._conn.commit()
            service = FakeService(store)

            self.assertLess(
                effective_memory_strength(store.get_memory("u1", old.id), now=1778131200.0),
                effective_memory_strength(store.get_memory("u1", new.id), now=1778131200.0),
            )

            result = service.chat("以后订座位优先考虑什么？", user_id="u1")

            recalled_ids = [memory["id"] for memory in result["recalled_memories"]]
            self.assertEqual(recalled_ids[:2], [new.id, old.id])
            self.assertIn("吧台", result["reply"])

    def test_memory_search_decay_keeps_recent_project_state_ahead_of_old_high_access_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            old = store.add_memory("u1", "AI 眼镜项目状态是旧版 Web demo", kind="event", memory_type="project_state")
            new = store.add_memory("u1", "AI 眼镜项目状态是长期记忆机制优化", kind="event", memory_type="project_state")
            old_timestamp = 1769904000.0
            store._conn.execute(
                """
                UPDATE memories
                SET access_count = 20, strength = 0.99, created_at = ?, updated_at = ?, last_accessed_at = ?
                WHERE id = ?
                """,
                (old_timestamp, old_timestamp, old_timestamp, old.id),
            )
            store._conn.commit()

            with patch("ai_glasses_memory_assistant.memory_store.time.time", return_value=1778131200.0):
                result = store.search_with_ranking("u1", "AI 眼镜 项目 状态", limit=2)

            self.assertEqual([memory.id for memory in result.memories], [new.id, old.id])
            old_ranking = next(item for item in result.ranking if item["id"] == old.id)
            self.assertLess(old_ranking["effective_strength_score"], old_ranking["base_strength_score"])
            self.assertEqual(old_ranking["strength_policy_reason"], "project_state_recency_sensitive")

    def test_memory_search_with_ranking_keeps_stronger_text_match_before_weak_recent_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            relevant = store.add_memory("u1", "项目 ranking alpha beta gamma 复盘", kind="event")
            store.add_memory("u1", "项目 ranking", kind="event")

            result = store.search_with_ranking("u1", "ranking alpha beta gamma", limit=2)
            legacy_results = store.search("u1", "ranking alpha beta gamma", limit=2)

            self.assertEqual(result.memories[0].id, relevant.id)
            self.assertGreater(result.ranking[0]["text_score"], result.ranking[1]["text_score"])
            self.assertEqual([memory.id for memory in legacy_results], [memory.id for memory in result.memories])

    def test_transient_context_ack_skips_llm_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("我刚刚路过公司楼下", user_id="u1")

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1"), [])

    def test_social_transient_chitchat_uses_local_chatty_ack_without_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("哈哈刚才同事讲了个小段子，我反应慢半拍。", user_id="u1")

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1"), [])

    def test_low_value_daily_chitchat_uses_local_supportive_ack_without_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("哎今天真有点累，路上人好多，我差点站着睡着了。", user_id="u1")

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1"), [])

    def test_ordinary_factual_question_uses_main_llm_without_memory_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("普通问答：水的化学式是什么？", user_id="u1")

            self.assertEqual(result["reply"], "主回复")
            self.assertEqual(result["saved_memories"], [])
            self.assertIn(service.fake_agent.main_calls, {0, 1})
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "llm")
            self.assertEqual(result["debug"]["memory"]["profile_count"], 0)
            self.assertIn("<answer-directive>", service.fake_agent.main_messages[-1])

    def test_llm_pre_reply_fallback_reads_profile_for_identity_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "我是 XREAL 的员工", kind="profile")
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "needs_event_memory": False,
                "needs_timeline_recall": False,
                "timeline_query": None,
                "confidence": 0.92,
                "reason": "asks for stored user identity",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("我是什么身份", user_id="u1")

            self.assertIn("XREAL", result["reply"])
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_profile_recall")
            self.assertTrue(result["debug"]["planner"]["needs_profile_memory"])
            self.assertEqual(result["debug"]["memory"]["profile_count"], 1)
            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(agent.main_calls, 1)

    def test_llm_pre_reply_fallback_summarizes_profile_for_about_me_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "我是 XREAL 的员工", kind="profile")
            store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile")
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "needs_event_memory": False,
                "needs_timeline_recall": False,
                "timeline_query": None,
                "confidence": 0.9,
                "reason": "asks what the assistant knows about the user",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("关于我的所有信息", user_id="u1")

            self.assertIn("XREAL", result["reply"])
            self.assertIn("低糖拿铁", result["reply"])
            self.assertNotIn("没有任何记忆", result["reply"])
            self.assertEqual(result["debug"]["memory"]["profile_count"], 2)
            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(agent.main_calls, 1)

    def test_pre_reply_recall_goal_is_normalized_and_debuggable(self) -> None:
        decision = _decision_from_payload(
            {
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "timeline",
                "recall_goal": "summary",
                "timeline_query": "刚刚",
                "confidence": 0.92,
            },
            raw="{}",
            backend="test",
        )
        plan = TurnPlan(reply_mode="llm").apply_pre_reply_decision(decision)

        self.assertEqual(decision.recall_goal, "summary")
        self.assertEqual(decision.debug_payload()["recall_goal"], "summary")
        self.assertEqual(plan.recall_goal, "summary")

    def test_event_text_search_debug_includes_ranking_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            memory = store.add_memory("u1", "项目 ranking recall 排序优化", kind="event")
            service = FakeService(store)

            memories, debug = service._recall_event_memories(
                user_id="u1",
                message="ranking recall",
                temporal=TemporalResolution(reason="no_time"),
                reference_time=1778131200.0,
                strategy="text_search",
            )

            self.assertEqual(memories[0].id, memory.id)
            self.assertEqual(debug["ranking"][0]["id"], memory.id)
            self.assertIn("total_score", debug["ranking"][0])
            self.assertIn("text_score", debug["ranking"][0])
            self.assertIn("effective_strength_score", debug["ranking"][0])
            self.assertEqual(debug["ranking"][0]["reason"], "text_recency_effective_strength_access")

    def test_timeline_recall_debug_includes_ranking_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))
            timeline = service.timeline_store.add_turn("u1", "之前聊过 ranking recall 排序", created_at=1778131200.0)
            planner = TurnPlan(reply_mode="local_timeline_recall", recall_goal="raw_evidence", timeline_query="ranking recall")

            chunks, debug = service._recall_timeline_chunks(user_id="u1", planner=planner)

            self.assertEqual(chunks[0].id, timeline.chunks[0].id)
            self.assertEqual(debug["ranking"][0]["id"], timeline.chunks[0].id)
            self.assertIn("total_score", debug["ranking"][0])
            self.assertIn("recency_score", debug["ranking"][0])

    def test_temporal_range_event_recall_keeps_time_order_without_ranking_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            later = store.add_memory("u1", "晚上讨论 ranking recall", kind="event", start_at=1778134800.0)
            early = store.add_memory("u1", "下午讨论 ranking recall", kind="event", start_at=1778131200.0)
            for _ in range(10):
                store.record_memory_access("u1", [later.id], accessed_at=1778134900.0)
            temporal = TemporalResolution(
                has_temporal_expression=True,
                start_at=1778130000.0,
                end_at=1778139000.0,
                confidence=0.95,
                reason="test_range",
            )

            memories, debug = service._recall_event_memories(
                user_id="u1",
                message="今天 ranking recall",
                temporal=temporal,
                reference_time=1778131200.0,
            )

            self.assertEqual([memory.id for memory in memories], [early.id, later.id])
            self.assertEqual(debug["strategy"], "temporal_range")
            self.assertNotIn("ranking", debug)

            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "event",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "temporal event recall",
            }, temporal_payload={
                "has_temporal_expression": True,
                "temporal_text": "今天",
                "kind": "date_range",
                "start_at_iso": "2026-05-07T13:00:00+08:00",
                "end_at_iso": "2026-05-07T15:30:00+08:00",
                "granularity": "day",
                "normalized_text": "ranking recall",
                "confidence": 0.95,
                "reason": "test_range",
            })
            service = FakeService(store, agent=agent)
            result = service.chat("今天 ranking recall", user_id="u1", routing_mode="llm_first")

            self.assertEqual([memory["id"] for memory in result["recalled_memories"]], [early.id, later.id])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "temporal_range")
            self.assertEqual(result["debug"]["memory"]["drift_guard"]["filtered_count"], 0)
            self.assertEqual(
                result["debug"]["memory"]["drift_guard"]["event_guard"]["reason"],
                "preserve_temporal_or_timeline_semantics",
            )

    def test_pre_reply_observation_recall_fallback_for_unmatched_review_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户近期主要投入 AI 眼镜长期记忆的 pre_reply_decision 兜底和 observation 回顾能力",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_pre_reply_observation"],
            )
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "observation",
                "confidence": 0.93,
                "reason": "asks for recent focus across memories",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("这阵子我的主要投入方向是什么？", user_id="u1")

            self.assertTrue(result["debug"]["routing"]["pre_reply_decision_applied"])
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "observation")
            self.assertTrue(result["debug"]["planner"]["needs_observation_memory"])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "observation_review")
            self.assertEqual(result["recalled_memories"][0]["memory_type"], "observation")
            self.assertIn("pre_reply_decision 兜底", result["reply"])
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(agent.main_calls, 1)

    def test_pre_reply_observation_recall_handles_synonym_without_planner_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户近期主要精力放在 pre_reply_decision 主权收口和 planner fallback 边界治理。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_pre_reply_authority"],
            )
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "observation",
                "recall_goal": "summary",
                "confidence": 0.93,
                "reason": "asks for recent focus across memories",
            })
            service = FakeService(store, agent=agent)

            baseline = plan_turn("这阵子我的主要精力放哪了？", reference_time=1778131200.0, timezone="Asia/Shanghai")
            result = service.chat("这阵子我的主要精力放哪了？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(baseline.reply_mode, "llm")
            self.assertFalse(baseline.needs_observation_memory)
            self.assertEqual(result["debug"]["turn_decision"]["final"]["source"], "pre_reply_decision")
            self.assertEqual(result["debug"]["turn_decision"]["final"]["route_authority"], "pre_reply_decision")
            self.assertEqual(result["debug"]["turn_decision"]["final"]["planner_role"], "baseline_or_fallback")
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "observation")
            self.assertEqual(result["debug"]["pre_reply_decision"]["recall_goal"], "summary")
            self.assertTrue(result["debug"]["planner"]["needs_observation_memory"])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "observation_review")
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIn("pre_reply_decision 主权收口", recalled)

    def test_pre_reply_observation_recall_for_recent_user_provided_external_topic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            source = store.add_memory(
                "u1",
                "某政策新增监管措施要求关闭零结余不动账户，并取得资金来源声明。",
                kind="event",
                memory_type="event",
                source="app_audio_transcript",
                evidence_ids=["chunk_policy_source"],
            )
            store.add_memory(
                "u1",
                "用户刚导入一段关于某政策新增监管措施的材料，包含账户关闭和资金来源声明要求。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=source.evidence_ids,
            )
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "observation",
                "recall_goal": "summary",
                "confidence": 0.93,
                "reason": "user asks to think about a recently provided external topic",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("想一想这个新增监管措施的事情", user_id="u1")

            self.assertTrue(result["debug"]["routing"]["pre_reply_decision_applied"])
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "observation")
            self.assertEqual(result["debug"]["pre_reply_decision"]["recall_goal"], "summary")
            self.assertTrue(result["debug"]["planner"]["needs_event_memory"])
            self.assertTrue(result["debug"]["planner"]["needs_observation_memory"])
            self.assertIn("user asks to think about a recently provided external topic", result["debug"]["planner"]["reason"])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "observation_review")
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIn("新增监管措施", recalled)
            self.assertIn("资金来源声明", result["reply"])
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(agent.main_calls, 1)

    def test_pre_reply_none_does_not_recall_observation_for_ordinary_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户近期主要投入 AI 眼镜长期记忆的 pre_reply_decision 兜底和 observation 回顾能力",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_pre_reply_observation"],
            )
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "none",
                "confidence": 0.93,
                "reason": "ordinary factual question",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("普通问答：水的化学式是什么？", user_id="u1")

            self.assertEqual(result["reply"], "主回复")
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "none")
            self.assertFalse(result["debug"]["planner"]["needs_observation_memory"])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "skipped_by_planner")
            self.assertEqual(result["recalled_memories"], [])
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(agent.main_calls, 1)

    def test_event_record_candidate_from_pre_reply_decision_saves_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="event",
                    memory_type="task",
                    candidate_content="周五下午3点和 alex 开周会",
                )
            )
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=agent)
            result = service.chat("记一下周五下午3点和 alex 开周会", user_id="u1")
            self.assertIn("我记住了", result["reply"])
            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertEqual(len(result["saved_memories"]), 1)
            self.assertEqual(agent.main_calls, 1)
            self.assertEqual(agent.temporal_calls, 0)
            memories = store.list_memories("u1", kind="event")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "周五下午3点和 alex 开周会")
            self.assertEqual(memories[0].evidence_ids, result["debug"]["timeline"]["chunk_ids"])

    def test_event_record_strips_spoken_command_shell_before_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("你先帮我记一下，周六晚上检查 demo", user_id="u1")
            memories = store.list_memories("u1", kind="event")

            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_action"], "write")
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "llm")
            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "周六晚上检查 demo")
            self.assertEqual(memories[0].temporal_text, "周六晚上")
            self.assertNotIn("你先帮我记一下", memories[0].content)
            self.assertEqual(service.fake_agent.intent_calls, 0)

    def test_uncertain_memory_command_shell_falls_back_to_llm_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(intent_payload={
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [
                    {
                        "content": "周六晚上检查 demo",
                        "kind": "event",
                        "memory_type": "task",
                        "confidence": 0.95,
                        "reason": "llm cleaned uncertain spoken command shell",
                    }
                ],
                "confidence": 0.95,
            })
            service = FakeService(store, agent=agent)

            result = service.chat("那个，先帮我留意记一下哈，周六晚上检查 demo", user_id="u1")
            memories = store.list_memories("u1", kind="event")

            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_action"], "write")
            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertEqual(agent.intent_calls, 0)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(len(memories), 1)
            self.assertIn("检查 demo", memories[0].content)
            self.assertNotIn("先记一下", memories[0].content)
            self.assertNotIn("先帮我留意记一下", memories[0].content)

    def test_question_form_memory_request_saves_clean_event_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("你能帮我记一下明天下午3点和 Mina 开会吗？", user_id="u1")
            memories = store.list_memories("u1", kind="event")

            self.assertEqual(result["debug"]["planner"]["memory_write_count"], 1)
            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "和 Mina 开会")
            self.assertIn(service.fake_agent.main_calls, {0, 1})

    def test_memory_job_is_isolated_by_user_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("记一下今晚8点给妈妈打电话", user_id="u1")
            job_id = result["debug"]["memory_processing"]["job_id"]

            self.assertIsNotNone(service.read_memory_job(user_id="u1", job_id=job_id))
            self.assertIsNone(service.read_memory_job(user_id="u2", job_id=job_id))

    def test_deferred_memory_write_job_reports_saved_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                {
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户喜欢低糖拿铁",
                            "kind": "profile",
                            "confidence": 0.95,
                            "reason": "stable preference",
                        }
                    ],
                    "confidence": 0.95,
                }
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我喜欢低糖拿铁", user_id="u1", defer_memory_writes=True)

            processing = result["debug"]["memory_processing"]
            self.assertEqual(processing["status"], "pending")
            self.assertEqual(processing["mode"], "reply_first_llm_extraction")
            job = service.read_memory_job(user_id="u1", job_id=processing["job_id"])
            self.assertEqual(job["status"], "saved")
            self.assertEqual(job["saved_count"], 1)
            self.assertEqual(job["rejected_count"], 0)

    def test_llm_first_saves_profile_preference_from_unified_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="preference",
                    candidate_content="用户喜欢安静靠窗的位置",
                ),
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我喜欢安静靠窗的位置", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["routing"]["mode"], "llm_first")
            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertEqual(store.list_memories("u1", kind="profile")[0].content, "用户喜欢安静靠窗的位置")

    def test_llm_first_plain_new_event_statement_requires_pre_reply_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("明天开会三点，在校园", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["routing"]["mode"], "llm_first")
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "llm")
            self.assertFalse(result["debug"]["fast_path"])
            self.assertIn("planner_disabled_llm_first", result["debug"]["steps"])
            self.assertNotIn("没有查到", result["reply"])
            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertEqual(service.fake_agent.main_calls, 1)
            self.assertEqual(len(store.list_memories("u1", kind="event")), 1)

    def test_llm_first_rejects_sensitive_memory_candidate_from_unified_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="profile",
                    memory_type="fact",
                    candidate_content="用户 API key 是 sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
                ),
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我的 API key 是 sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["routing"]["mode"], "llm_first")
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "sensitive_credential_rejected")
            self.assertEqual(result["debug"]["memory_processing"]["status"], "not_needed")
            self.assertEqual(result["debug"]["memory_processing"]["mode"], "sensitive_credential_rejected")
            self.assertEqual(result["debug"]["memory_processing"]["safety_policy"]["role"], "hard_safety")
            self.assertEqual(store.list_memories("u1"), [])

    def test_saved_memory_has_traceability_fields_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            candidate = MemoryWriteCandidate(
                content="用户喜欢低糖拿铁",
                kind="profile",
                memory_type="preference",
                confidence=0.95,
                source_id="chat_turn_1",
                ingestion_id="ing_1",
                evidence_ids=["chat_turn_1"],
            )

            saved, rejected = service._save_memory_candidates(
                candidates=[candidate, candidate],
                message="我喜欢低糖拿铁",
                user_id="u1",
                agent=None,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )
            memories = store.list_memories("u1", kind="profile")

            self.assertEqual(rejected, [])
            self.assertEqual(len(saved), 2)
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].memory_type, "preference")
            self.assertEqual(memories[0].source_id, "chat_turn_1")
            self.assertEqual(memories[0].ingestion_id, "ing_1")
            self.assertEqual(memories[0].evidence_ids, ["chat_turn_1"])
            self.assertEqual(memories[0].confidence, 0.95)

    def test_save_candidate_defaults_type_from_kind_not_phrase_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            candidate = MemoryWriteCandidate(
                content="AI 眼镜项目继续补 debug 可观测性",
                kind="event",
                confidence=0.9,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="AI 眼镜项目继续补 debug 可观测性",
                user_id="u1",
                agent=None,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertEqual(save_result.rejected, [])
            self.assertEqual(len(save_result.saved), 1)
            self.assertEqual(save_result.saved[0].kind, "event")
            self.assertEqual(save_result.saved[0].memory_type, "event")

    def test_llm_dedupe_merges_semantic_duplicate_profile_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            existing = store.add_memory(
                "u1",
                "用户喜欢低糖拿铁",
                kind="profile",
                memory_type="preference",
                evidence_ids=["old_turn"],
                confidence=0.82,
            )
            agent = FakeAgent(dedupe_payload={
                "action": "duplicate",
                "memory_id": existing.id,
                "confidence": 0.92,
                "reason": "same latte preference",
            })
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="用户偏好低糖的拿铁咖啡",
                kind="profile",
                memory_type="preference",
                confidence=0.95,
                source_id="new_turn",
                evidence_ids=["new_turn"],
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="我偏好低糖的拿铁咖啡",
                user_id="u1",
                agent=None,
                dedupe_agent=agent,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )
            memories = store.list_memories("u1", kind="profile")
            refreshed = store.get_memory("u1", existing.id)

            self.assertEqual(agent.dedupe_calls, 1)
            self.assertEqual(len(memories), 1)
            self.assertEqual(save_result.saved[0].id, existing.id)
            self.assertEqual(save_result.dedupe_decisions[0]["action"], "duplicate")
            self.assertEqual(save_result.dedupe_decisions[0]["memory_id"], existing.id)
            self.assertEqual(save_result.superseded_memory_ids, [])
            self.assertEqual(refreshed.evidence_ids, ["old_turn", "new_turn"])
            self.assertEqual(refreshed.confidence, 0.95)

    def test_llm_dedupe_supersedes_conflicting_profile_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            existing = store.add_memory(
                "u1",
                "用户喜欢低糖拿铁",
                kind="profile",
                memory_type="preference",
                evidence_ids=["old_turn"],
            )
            agent = FakeAgent(dedupe_payload={
                "action": "conflict",
                "memory_id": existing.id,
                "confidence": 0.93,
                "reason": "new statement negates latte preference",
            })
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="用户不喜欢拿铁",
                kind="profile",
                memory_type="preference",
                confidence=0.94,
                source_id="new_turn",
                evidence_ids=["new_turn"],
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="我不喜欢拿铁",
                user_id="u1",
                agent=None,
                dedupe_agent=agent,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )
            active_memories = store.list_memories("u1", kind="profile")
            refreshed_old = store.get_memory("u1", existing.id)

            self.assertEqual(agent.dedupe_calls, 1)
            self.assertEqual(len(active_memories), 1)
            self.assertEqual(active_memories[0].content, "用户不喜欢拿铁")
            self.assertEqual(refreshed_old.status, "superseded")
            self.assertEqual(refreshed_old.superseded_by, save_result.saved[0].id)
            self.assertEqual(save_result.superseded_memory_ids, [existing.id])
            self.assertEqual(save_result.dedupe_decisions[0]["action"], "conflict")
            self.assertEqual(
                save_result.lifecycle_transitions,
                [
                    {
                        "memory_id": existing.id,
                        "from_status": "active",
                        "to_status": "superseded",
                        "reason": "preference_conflict_supersede",
                        "superseded_by": save_result.saved[0].id,
                    }
                ],
            )

    def test_llm_dedupe_low_confidence_falls_back_to_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            existing = store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile", memory_type="preference")
            agent = FakeAgent(dedupe_payload={
                "action": "duplicate",
                "memory_id": existing.id,
                "confidence": 0.5,
                "reason": "uncertain",
            })
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="用户偏好低糖的拿铁咖啡",
                kind="profile",
                memory_type="preference",
                confidence=0.9,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="我偏好低糖的拿铁咖啡",
                user_id="u1",
                agent=None,
                dedupe_agent=agent,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertEqual(agent.dedupe_calls, 1)
            self.assertEqual(len(store.list_memories("u1", kind="profile")), 2)
            dedupe_policy = save_result.dedupe_decisions[0]["confidence_policy"]
            self.assertEqual(dedupe_policy["purpose"], "dedupe_relationship")
            self.assertEqual(dedupe_policy["min_confidence"], 0.75)
            self.assertEqual(dedupe_policy["treatment"], "fallback_to_new")
            self.assertFalse(dedupe_policy["passed"])
            self.assertEqual(store.get_memory("u1", existing.id).status, "active")
            self.assertEqual(save_result.superseded_memory_ids, [])

    def test_llm_dedupe_invalid_json_falls_back_to_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            existing = store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile", memory_type="preference")
            agent = FakeAgent(dedupe_payload="not json")
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="用户偏好低糖的拿铁咖啡",
                kind="profile",
                memory_type="preference",
                confidence=0.9,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="我偏好低糖的拿铁咖啡",
                user_id="u1",
                agent=None,
                dedupe_agent=agent,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertEqual(agent.dedupe_calls, 1)
            self.assertEqual(len(store.list_memories("u1", kind="profile")), 2)
            self.assertEqual(store.get_memory("u1", existing.id).status, "active")
            self.assertEqual(save_result.superseded_memory_ids, [])

    def test_structured_event_dedupe_merges_semantic_duplicate_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            existing = store.add_memory(
                "u1",
                "周五前补 memory eval",
                kind="event",
                memory_type="task",
                start_at=1778227200.0,
                evidence_ids=["old_turn"],
                confidence=0.82,
            )
            agent = FakeAgent(dedupe_payload={
                "action": "duplicate",
                "memory_id": existing.id,
                "confidence": 0.91,
                "reason": "same task",
            })
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="这周五之前把记忆机制评估补上",
                kind="event",
                memory_type="task",
                confidence=0.93,
                source_id="new_turn",
                evidence_ids=["new_turn"],
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="这周五之前把记忆机制评估补上",
                user_id="u1",
                agent=None,
                dedupe_agent=agent,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(
                    has_temporal_expression=True,
                    start_at=1778227200.0,
                    end_at=1778230800.0,
                    granularity="day",
                    confidence=0.9,
                    backend="test",
                ),
                saved_temporal_debug=[],
            )
            memories = store.list_memories("u1", kind="event")
            refreshed = store.get_memory("u1", existing.id)

            self.assertEqual(agent.dedupe_calls, 1)
            self.assertEqual(len(memories), 1)
            self.assertEqual(save_result.saved[0].id, existing.id)
            self.assertEqual(save_result.dedupe_decisions[0]["memory_type"], "task")
            self.assertEqual(save_result.dedupe_decisions[0]["action"], "duplicate")
            self.assertEqual(refreshed.evidence_ids, ["old_turn", "new_turn"])
            self.assertEqual(refreshed.confidence, 0.93)

    def test_structured_task_keeps_temporal_phrase_in_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            candidate = MemoryWriteCandidate(
                content="周四晚上交第一版",
                kind="event",
                memory_type="task",
                confidence=0.92,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="我再补一段文字：原来想说周五，哦不对，应该是周四晚上交第一版。",
                user_id="u1",
                agent=None,
                reference_time=1779415200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertEqual(len(save_result.saved), 1)
            self.assertEqual(save_result.saved[0].content, "周四晚上交第一版")
            self.assertEqual(save_result.saved[0].temporal_text, "周四晚上")
            self.assertEqual(save_result.saved[0].memory_type, "task")

    def test_structured_event_dedupe_supersedes_conflicting_project_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            existing = store.add_memory(
                "u1",
                "项目状态是先做南太行攻略",
                kind="event",
                memory_type="project_state",
                evidence_ids=["old_turn"],
            )
            agent = FakeAgent(dedupe_payload={
                "action": "conflict",
                "memory_id": existing.id,
                "confidence": 0.89,
                "reason": "new project focus replaces old state",
            })
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="现在主线是 AI 眼镜长期记忆",
                kind="event",
                memory_type="project_state",
                confidence=0.94,
                source_id="new_turn",
                evidence_ids=["new_turn"],
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="更准确地说，现在主线是 AI 眼镜长期记忆",
                user_id="u1",
                agent=None,
                dedupe_agent=agent,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )
            active_memories = store.list_memories("u1", kind="event")
            refreshed_old = store.get_memory("u1", existing.id)

            self.assertEqual(agent.dedupe_calls, 1)
            self.assertEqual(len(active_memories), 1)
            self.assertEqual(active_memories[0].content, "现在主线是 AI 眼镜长期记忆")
            self.assertEqual(refreshed_old.status, "superseded")
            self.assertEqual(refreshed_old.superseded_by, save_result.saved[0].id)
            self.assertEqual(save_result.superseded_memory_ids, [existing.id])
            self.assertEqual(save_result.dedupe_decisions[0]["memory_type"], "project_state")
            self.assertEqual(
                save_result.lifecycle_transitions,
                [
                    {
                        "memory_id": existing.id,
                        "from_status": "active",
                        "to_status": "superseded",
                        "reason": "structured_event_conflict_supersede",
                        "superseded_by": save_result.saved[0].id,
                    }
                ],
            )

    def test_structured_event_dedupe_supersedes_updated_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            existing = store.add_memory(
                "u1",
                "项目决定先做本地记忆门控",
                kind="event",
                memory_type="decision",
            )
            agent = FakeAgent(dedupe_payload={
                "action": "conflict",
                "memory_id": existing.id,
                "confidence": 0.9,
                "reason": "updated decision",
            })
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="结论更新为先补结构化去重",
                kind="event",
                memory_type="decision",
                confidence=0.92,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="结论更新为先补结构化去重",
                user_id="u1",
                agent=None,
                dedupe_agent=agent,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertEqual(agent.dedupe_calls, 1)
            self.assertEqual(store.get_memory("u1", existing.id).status, "superseded")
            self.assertEqual(save_result.dedupe_decisions[0]["memory_type"], "decision")
            self.assertEqual(save_result.superseded_memory_ids, [existing.id])

    def test_structured_event_dedupe_low_confidence_and_invalid_id_fall_back_to_new(self) -> None:
        cases = [
            {
                "action": "duplicate",
                "memory_id": "",
                "confidence": 0.5,
                "reason": "uncertain",
            },
            {
                "action": "conflict",
                "memory_id": "missing",
                "confidence": 0.95,
                "reason": "bad id",
            },
            "not json",
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    store = self.make_store(Path(tmpdir))
                    existing = store.add_memory("u1", "周五前补 memory eval", kind="event", memory_type="task")
                    agent = FakeAgent(dedupe_payload=payload)
                    service = FakeService(store, agent=agent)
                    candidate = MemoryWriteCandidate(
                        content="这周五之前把记忆机制评估补上",
                        kind="event",
                        memory_type="task",
                        confidence=0.93,
                    )

                    save_result = service._save_memory_candidates(
                        candidates=[candidate],
                        message="这周五之前把记忆机制评估补上",
                        user_id="u1",
                        agent=None,
                        dedupe_agent=agent,
                        reference_time=1778131200.0,
                        query_temporal=TemporalResolution(),
                        saved_temporal_debug=[],
                    )

                    self.assertEqual(agent.dedupe_calls, 1)
                    self.assertEqual(len(store.list_memories("u1", kind="event")), 2)
                    self.assertEqual(store.get_memory("u1", existing.id).status, "active")
                    self.assertEqual(save_result.superseded_memory_ids, [])
                    self.assertEqual(save_result.dedupe_decisions[0]["action"], "new")

    def test_structured_event_dedupe_without_agent_keeps_fast_path_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "周五前补 memory eval", kind="event", memory_type="task")
            agent = FakeAgent(dedupe_payload={
                "action": "duplicate",
                "memory_id": "should-not-be-used",
                "confidence": 0.99,
                "reason": "would merge if called",
            })
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="这周五之前把记忆机制评估补上",
                kind="event",
                memory_type="task",
                confidence=0.93,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="这周五之前把记忆机制评估补上",
                user_id="u1",
                agent=None,
                dedupe_agent=None,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertEqual(agent.dedupe_calls, 0)
            self.assertEqual(len(store.list_memories("u1", kind="event")), 2)
            self.assertEqual(save_result.dedupe_decisions, [])

    def test_new_task_defaults_to_open_status_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            candidate = MemoryWriteCandidate(
                content="周五前补 memory eval",
                kind="event",
                memory_type="task",
                confidence=0.9,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="周五前补 memory eval",
                user_id="u1",
                agent=None,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertIn("task_status:open", save_result.saved[0].tags)
            self.assertEqual(save_result.task_status_updates, [])
            self.assertEqual(save_result.task_status_policies[0]["role"], "format_parser")
            self.assertEqual(save_result.task_status_policies[0]["reason"], "no_status_marker")
            self.assertEqual(save_result.task_status_policies[0]["status"], "open")
            self.assertFalse(save_result.task_status_policies[0]["affects_memory_type"])

    def test_explicit_project_task_gets_project_scope_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            candidate = MemoryWriteCandidate(
                content="AI 眼镜项目：Alex 负责补 eval 报告模板",
                kind="event",
                memory_type="task",
                confidence=0.9,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="AI 眼镜项目：Alex 负责补 eval 报告模板",
                user_id="u1",
                agent=None,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertIn("task_status:open", save_result.saved[0].tags)
            self.assertIn("project:AI眼镜", save_result.saved[0].tags)

    def test_completed_task_supersedes_open_task_and_records_status_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            existing = store.add_memory(
                "u1",
                "周五前补 memory eval",
                kind="event",
                memory_type="task",
                tags=["task_status:open"],
            )
            agent = FakeAgent(dedupe_payload={
                "action": "conflict",
                "memory_id": existing.id,
                "confidence": 0.9,
                "reason": "task completed",
            })
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="周五前补 memory eval 已完成",
                kind="event",
                memory_type="task",
                confidence=0.92,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="周五前补 memory eval 已完成",
                user_id="u1",
                agent=None,
                dedupe_agent=agent,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertIn("task_status:completed", save_result.saved[0].tags)
            self.assertEqual(store.get_memory("u1", existing.id).status, "superseded")
            self.assertEqual(save_result.superseded_memory_ids, [existing.id])
            self.assertEqual(save_result.task_status_updates[0]["status"], "completed")
            self.assertEqual(save_result.task_status_updates[0]["superseded_task_id"], existing.id)
            self.assertEqual(save_result.task_status_policies[0]["role"], "format_parser")
            self.assertEqual(save_result.task_status_policies[0]["reason"], "completed_marker")
            self.assertEqual(save_result.task_status_policies[0]["status"], "completed")

    def test_cancelled_task_supersedes_open_task_and_records_status_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            existing = store.add_memory(
                "u1",
                "周五下午复盘会",
                kind="event",
                memory_type="task",
                tags=["task_status:open"],
            )
            agent = FakeAgent(dedupe_payload={
                "action": "conflict",
                "memory_id": existing.id,
                "confidence": 0.9,
                "reason": "task cancelled",
            })
            service = FakeService(store, agent=agent)
            candidate = MemoryWriteCandidate(
                content="取消周五下午复盘会",
                kind="event",
                memory_type="task",
                confidence=0.92,
            )

            save_result = service._save_memory_candidates(
                candidates=[candidate],
                message="取消周五下午复盘会",
                user_id="u1",
                agent=None,
                dedupe_agent=agent,
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                saved_temporal_debug=[],
            )

            self.assertIn("task_status:cancelled", save_result.saved[0].tags)
            self.assertEqual(store.get_memory("u1", existing.id).status, "superseded")
            self.assertEqual(save_result.task_status_updates[0]["status"], "cancelled")
            self.assertEqual(save_result.task_status_policies[0]["role"], "format_parser")
            self.assertEqual(save_result.task_status_policies[0]["reason"], "cancelled_marker")
            self.assertEqual(save_result.task_status_policies[0]["status"], "cancelled")

    def test_work_todo_recall_filters_completed_and_cancelled_tasks(self) -> None:
        open_task = types.SimpleNamespace(
            id="open",
            memory_type="task",
            content="周五前补 memory eval",
            tags=["task_status:open"],
        )
        completed_task = types.SimpleNamespace(
            id="completed",
            memory_type="task",
            content="周三完成报告",
            tags=["task_status:completed"],
        )
        cancelled_task = types.SimpleNamespace(
            id="cancelled",
            memory_type="task",
            content="取消周五复盘会",
            tags=["task_status:cancelled"],
        )

        filtered = GlassesChatService._filter_event_memories_for_query(
            "我还有什么待办？",
            [open_task, completed_task, cancelled_task],
        )

        self.assertEqual([memory.id for memory in filtered], ["open"])

    def test_fast_path_profile_write_does_not_call_llm_dedupe_without_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile", memory_type="preference")
            agent = FakeAgent(dedupe_payload={
                "action": "duplicate",
                "memory_id": "should-not-be-used",
                "confidence": 0.99,
                "reason": "would merge if called",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("我叫 jack", user_id="u1")

            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertEqual(agent.dedupe_calls, 0)
            self.assertEqual(service.new_session_calls, 0)
            self.assertEqual(len(store.list_memories("u1", kind="profile")), 2)

    def test_superseded_preference_is_not_recalled_accessed_or_observation_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            old = store.add_memory(
                "u1",
                "用户喜欢拿铁咖啡",
                kind="profile",
                memory_type="preference",
                evidence_ids=["old_turn"],
            )
            active = store.add_memory(
                "u1",
                "用户喜欢美式咖啡",
                kind="profile",
                memory_type="preference",
                evidence_ids=["active_turn"],
            )
            store.add_memory(
                "u1",
                "用户正在优化记忆机制",
                kind="event",
                memory_type="project_state",
                evidence_ids=["event_turn"],
            )
            store.mark_superseded("u1", old.id, active.id)

            recall = service.chat("我喜欢什么咖啡？", user_id="u1")
            reflect_job = service._maybe_start_observation_reflect(
                user_id="u1",
                session_id="s1",
                reference_time=1778131200.0,
                agent=service.fake_agent,
            )

            self.assertIn("美式", recall["reply"])
            self.assertNotIn("拿铁", recall["reply"])
            self.assertEqual(store.get_memory("u1", old.id).access_count, 0)
            self.assertGreater(store.get_memory("u1", active.id).access_count, 0)
            self.assertIsNone(reflect_job)

    def test_deferred_memory_write_job_reports_rejected_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            agent = FakeAgent(
                {
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户询问自己的身份",
                            "kind": "profile",
                            "confidence": 0.95,
                            "reason": "meta profile",
                        }
                    ],
                    "confidence": 0.95,
                }
            )
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=agent)

            result = service.chat("普通对话？", user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(
                user_id="u1",
                job_id=result["debug"]["memory_processing"]["job_id"],
            )

            self.assertEqual(job["status"], "rejected")
            self.assertEqual(job["saved_count"], 0)
            self.assertEqual(job["rejected_count"], 1)
            self.assertEqual(store.list_memories("u1"), [])

    def test_background_memory_job_reports_failed_status_without_breaking_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = FailingMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = FakeService(store)

            result = service.chat("记一下今晚8点给妈妈打电话", user_id="u1")
            job = service.read_memory_job(
                user_id="u1",
                job_id=result["debug"]["memory_processing"]["job_id"],
            )

            self.assertIn("我先记下", result["reply"])
            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["error_type"], "RuntimeError")
            self.assertEqual(job["saved_count"], 0)

    def test_background_memory_job_writes_audit_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("记一下今晚8点给妈妈打电话", user_id="u1")
            job_id = result["debug"]["memory_processing"]["job_id"]
            records = service.read_audit_records(user_id="u1", limit=10)

            background_records = [
                record for record in records
                if record.get("record_type") == "background_memory_write"
            ]
            self.assertEqual(len(background_records), 1)
            self.assertEqual(background_records[0]["job_id"], job_id)
            self.assertEqual(background_records[0]["status"], "saved")

    def test_chat_turn_failure_writes_audit_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = SnapshotFailingMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = FakeService(store)

            with self.assertRaises(sqlite3.DatabaseError):
                service.chat("普通问答：水的化学式是什么？", user_id="u1", routing_mode="llm_first")

            records = service.read_audit_records(user_id="u1", limit=10)
            failed_records = [
                record for record in records
                if record.get("record_type") == "chat_turn_failed"
            ]
            self.assertEqual(len(failed_records), 1)
            self.assertEqual(failed_records[0]["failed_stage"], "memory_snapshot")
            self.assertEqual(failed_records[0]["error_type"], "DatabaseError")
            self.assertIn("database disk image is malformed", failed_records[0]["error_message"])
            self.assertEqual(failed_records[0]["debug"]["failure"]["stage"], "memory_snapshot")

    def test_chat_turn_failure_from_main_llm_writes_audit_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FailingMainAgent())

            with self.assertRaises(sqlite3.DatabaseError):
                service.chat("普通问答：水的化学式是什么？", user_id="u1", routing_mode="llm_first")

            records = service.read_audit_records(user_id="u1", limit=10)
            failed_records = [
                record for record in records
                if record.get("record_type") == "chat_turn_failed"
            ]
            self.assertEqual(len(failed_records), 1)
            self.assertEqual(failed_records[0]["failed_stage"], "assistant_response")
            self.assertEqual(failed_records[0]["error_type"], "DatabaseError")
            self.assertIn("database disk image is malformed", failed_records[0]["error_message"])

    def test_upcoming_plan_recall_failure_writes_audit_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = RecallFailingMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = FakeService(store)

            with self.assertRaises(sqlite3.DatabaseError):
                service.chat("我今日有什么安排", user_id="u1")

            records = service.read_audit_records(user_id="u1", limit=10)
            failed_records = [
                record for record in records
                if record.get("record_type") == "chat_turn_failed"
            ]
            self.assertEqual(len(failed_records), 1)
            self.assertEqual(failed_records[0]["message"], "我今日有什么安排")
            self.assertEqual(failed_records[0]["failed_stage"], "memory_retrieval")
            self.assertEqual(failed_records[0]["error_type"], "DatabaseError")
            self.assertIn("database disk image is malformed", failed_records[0]["error_message"])
            self.assertEqual(failed_records[0]["debug"]["planner"]["reply_mode"], "local_event_recall")
            self.assertEqual(failed_records[0]["debug"]["failure"]["stage"], "memory_retrieval")

    def test_background_observation_reflect_saves_observation_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_memory(
                "u1",
                "用户喜欢短答和直接结论",
                kind="profile",
                memory_type="preference",
                evidence_ids=["chunk_1"],
            )
            store.add_memory(
                "u1",
                "用户最近在补 AI 眼镜 demo 的记忆系统",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_2"],
            )

            result = service.chat("记一下今天下午对齐了 observation reflect 方案", user_id="u1")

            background_records = [
                record for record in service.read_audit_records(user_id="u1", limit=10)
                if record.get("mode") == "observation_reflect"
            ]
            observations = [
                memory for memory in store.list_memories("u1", kind="event")
                if memory.memory_type == "observation"
            ]
            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertEqual(len(background_records), 1)
            self.assertEqual(background_records[0]["status"], "saved")
            self.assertEqual(background_records[0]["source_memory_count"], 3)
            self.assertEqual(background_records[0]["saved_count"], 1)
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0].source, "observation_reflect")
            self.assertEqual(observations[0].memory_type, "observation")
            self.assertTrue({"chunk_1", "chunk_2"}.issubset(set(observations[0].evidence_ids)))

    def test_observation_reflect_requires_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            source_memories = [
                store.add_memory("u1", "用户喜欢短答", kind="profile", memory_type="preference"),
                store.add_memory("u1", "用户在做 AI 眼镜 demo", kind="event", memory_type="project_state"),
                store.add_memory("u1", "用户关注长期记忆", kind="event", memory_type="event"),
            ]
            job = service._create_memory_job(
                user_id="u1",
                session_id="",
                mode="observation_reflect",
                candidate_count=1,
                created_at=1778131200.0,
                source_memory_ids=[memory.id for memory in source_memories],
                evidence_ids=[],
            )

            service._process_observation_reflect_background(
                user_id="u1",
                session_id="",
                reference_time=1778131200.0,
                agent=None,
                job_id=job["job_id"],
                source_memories=source_memories,
                evidence_ids=[],
            )

            refreshed = service.read_memory_job(user_id="u1", job_id=job["job_id"])
            observations = [
                memory for memory in store.list_memories("u1", kind="event")
                if memory.memory_type == "observation"
            ]
            self.assertEqual(refreshed["status"], "rejected")
            self.assertIn("observation_requires_evidence", refreshed["rejected_reasons"])
            self.assertEqual(observations, [])

    def test_observation_reflect_ignores_stale_source_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_memory(
                "u1",
                "用户喜欢证据可追溯",
                kind="profile",
                memory_type="preference",
                evidence_ids=["chunk_profile"],
            )
            store.add_memory(
                "u1",
                "用户正在优化记忆机制",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_event"],
            )
            stale = store.add_memory(
                "u1",
                "用户以前在做无关项目",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_stale"],
            )

            self.assertTrue(store.mark_stale("u1", stale.id))
            job = service._maybe_start_observation_reflect(
                user_id="u1",
                session_id="s1",
                reference_time=1778131200.0,
                agent=service.fake_agent,
            )

            self.assertIsNone(job)
            self.assertEqual(store.get_memory("u1", stale.id).status, "stale")
            self.assertEqual(
                [
                    memory.content
                    for memory in store.list_memories("u1")
                    if memory.memory_type == "observation"
                ],
                [],
            )

    def test_observation_reflect_uses_gate_for_sensitive_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            source_memories = [
                store.add_memory("u1", "用户喜欢短答", kind="profile", memory_type="preference", evidence_ids=["chunk_1"]),
                store.add_memory("u1", "用户在做 AI 眼镜 demo", kind="event", memory_type="project_state", evidence_ids=["chunk_2"]),
                store.add_memory("u1", "用户关注长期记忆", kind="event", memory_type="event", evidence_ids=["chunk_3"]),
            ]
            job = service._create_memory_job(
                user_id="u1",
                session_id="",
                mode="observation_reflect",
                candidate_count=1,
                created_at=1778131200.0,
                source_memory_ids=[memory.id for memory in source_memories],
                evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
            )
            sensitive_candidate = MemoryWriteCandidate(
                content="用户的 API key 是 secret-token",
                kind="event",
                memory_type="observation",
                confidence=0.9,
                source="observation_reflect",
                evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
            )
            with patch.object(service, "_build_observation_candidate", return_value=(sensitive_candidate, "rule_reflect")):
                service._process_observation_reflect_background(
                    user_id="u1",
                    session_id="",
                    reference_time=1778131200.0,
                    agent=None,
                    job_id=job["job_id"],
                    source_memories=source_memories,
                    evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
                )

            refreshed = service.read_memory_job(user_id="u1", job_id=job["job_id"])
            observations = [
                memory for memory in store.list_memories("u1", kind="event")
                if memory.memory_type == "observation"
            ]
            self.assertEqual(refreshed["status"], "rejected")
            self.assertIn("candidate_contains_sensitive_term", refreshed["rejected_reasons"])
            self.assertEqual(observations, [])

    def test_observation_reflect_merges_duplicate_observation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            existing = store.add_memory(
                "u1",
                "用户最近主要在推进 AI 眼镜长期记忆 demo",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_1", "chunk_2"],
                confidence=0.7,
            )
            source_memories = [
                store.add_memory("u1", "用户正在推进 AI 眼镜长期记忆 demo", kind="event", memory_type="project_state", evidence_ids=["chunk_1"]),
                store.add_memory("u1", "用户偏好证据可追溯", kind="profile", memory_type="preference", evidence_ids=["chunk_2"]),
                store.add_memory("u1", "用户补了 observation 更新机制", kind="event", memory_type="project_state", evidence_ids=["chunk_3"]),
            ]
            job = service._create_memory_job(
                user_id="u1",
                session_id="",
                mode="observation_reflect",
                candidate_count=1,
                created_at=1778131200.0,
                source_memory_ids=[memory.id for memory in source_memories],
                evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
            )
            candidate = MemoryWriteCandidate(
                content="用户最近主要在推进 AI 眼镜长期记忆 demo，并补 observation 更新机制",
                kind="event",
                memory_type="observation",
                confidence=0.9,
                source="observation_reflect",
                evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
            )

            with patch.object(service, "_build_observation_candidate", return_value=(candidate, "rule_reflect")):
                service._process_observation_reflect_background(
                    user_id="u1",
                    session_id="",
                    reference_time=1778131200.0,
                    agent=None,
                    job_id=job["job_id"],
                    source_memories=source_memories,
                    evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
                )

            observations = [
                memory for memory in store.list_memories("u1", kind="event")
                if memory.memory_type == "observation"
            ]
            refreshed = store.get_memory("u1", existing.id)
            job_payload = service.read_memory_job(user_id="u1", job_id=job["job_id"])
            audit_record = [
                record for record in service.read_audit_records(user_id="u1", limit=10)
                if record.get("job_id") == job["job_id"]
            ][0]

            self.assertEqual(len(observations), 1)
            self.assertEqual(refreshed.id, existing.id)
            self.assertEqual(set(refreshed.evidence_ids), {"chunk_1", "chunk_2", "chunk_3"})
            self.assertGreaterEqual(refreshed.confidence or 0, 0.9)
            self.assertEqual(job_payload["status"], "saved")
            self.assertEqual(job_payload["saved_memory_ids"], [existing.id])
            self.assertEqual(job_payload["observation_update_decisions"][0]["action"], "merge")
            self.assertEqual(job_payload["observation_update_decisions"][0]["observation_id"], existing.id)
            self.assertEqual(audit_record["observation_update_decisions"][0]["action"], "merge")

    def test_observation_reflect_supersedes_conflicting_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            old = store.add_memory(
                "u1",
                "用户最近主要在做南太行项目",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_old"],
            )
            source_memories = [
                store.add_memory("u1", "用户最近主要在做 AI 眼镜长期记忆", kind="event", memory_type="project_state", evidence_ids=["chunk_new_1"]),
                store.add_memory("u1", "用户在补 observation 更新机制", kind="event", memory_type="project_state", evidence_ids=["chunk_new_2"]),
                store.add_memory("u1", "用户偏好证据可追溯", kind="profile", memory_type="preference", evidence_ids=["chunk_new_3"]),
            ]
            job = service._create_memory_job(
                user_id="u1",
                session_id="",
                mode="observation_reflect",
                candidate_count=1,
                created_at=1778131200.0,
                source_memory_ids=[memory.id for memory in source_memories],
                evidence_ids=["chunk_new_1", "chunk_new_2", "chunk_new_3"],
            )
            candidate = MemoryWriteCandidate(
                content="用户最近主要在做 AI 眼镜长期记忆和 observation 更新机制",
                kind="event",
                memory_type="observation",
                confidence=0.9,
                source="observation_reflect",
                evidence_ids=["chunk_new_1", "chunk_new_2", "chunk_new_3"],
            )

            with patch.object(service, "_build_observation_candidate", return_value=(candidate, "rule_reflect")):
                service._process_observation_reflect_background(
                    user_id="u1",
                    session_id="",
                    reference_time=1778131200.0,
                    agent=None,
                    job_id=job["job_id"],
                    source_memories=source_memories,
                    evidence_ids=["chunk_new_1", "chunk_new_2", "chunk_new_3"],
                )

            active_observations = [
                memory for memory in store.list_memories("u1", kind="event")
                if memory.memory_type == "observation"
            ]
            refreshed_old = store.get_memory("u1", old.id)
            job_payload = service.read_memory_job(user_id="u1", job_id=job["job_id"])

            self.assertEqual(len(active_observations), 1)
            self.assertIn("AI 眼镜长期记忆", active_observations[0].content)
            self.assertNotEqual(active_observations[0].id, old.id)
            self.assertEqual(refreshed_old.status, "superseded")
            self.assertEqual(refreshed_old.superseded_by, active_observations[0].id)
            self.assertEqual(job_payload["observation_update_decisions"][0]["action"], "supersede")
            self.assertEqual(job_payload["observation_update_decisions"][0]["observation_id"], old.id)

    def test_observation_reflect_low_confidence_llm_decision_falls_back_to_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            existing = store.add_memory(
                "u1",
                "用户最近主要在推进 AI 眼镜长期记忆 demo",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_old"],
            )
            source_memories = [
                store.add_memory("u1", "用户关注长期记忆", kind="event", memory_type="project_state", evidence_ids=["chunk_1"]),
                store.add_memory("u1", "用户关注 observation 更新", kind="event", memory_type="project_state", evidence_ids=["chunk_2"]),
                store.add_memory("u1", "用户喜欢短答", kind="profile", memory_type="preference", evidence_ids=["chunk_3"]),
            ]
            job = service._create_memory_job(
                user_id="u1",
                session_id="",
                mode="observation_reflect",
                candidate_count=1,
                created_at=1778131200.0,
                source_memory_ids=[memory.id for memory in source_memories],
                evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
            )
            candidate = MemoryWriteCandidate(
                content="用户最近在整理长期记忆和 observation 更新",
                kind="event",
                memory_type="observation",
                confidence=0.9,
                source="observation_reflect",
                evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
            )

            with patch.object(service, "_build_observation_candidate", return_value=(candidate, "rule_reflect")), patch.object(
                service,
                "_observation_update_decision",
                return_value={"action": "skip", "observation_id": existing.id, "reason": "llm_low_confidence", "confidence": 0.2},
            ):
                service._process_observation_reflect_background(
                    user_id="u1",
                    session_id="",
                    reference_time=1778131200.0,
                    agent=None,
                    job_id=job["job_id"],
                    source_memories=source_memories,
                    evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
                )

            observations = [
                memory for memory in store.list_memories("u1", kind="event")
                if memory.memory_type == "observation"
            ]
            job_payload = service.read_memory_job(user_id="u1", job_id=job["job_id"])

            self.assertEqual(len(observations), 2)
            self.assertEqual(store.get_memory("u1", existing.id).status, "active")
            self.assertEqual(job_payload["status"], "saved")
            self.assertEqual(job_payload["observation_update_decisions"][0]["action"], "new")
            self.assertEqual(job_payload["observation_update_decisions"][0]["reason"], "fallback_after_skip")
            observation_policy = job_payload["observation_update_decisions"][0]["confidence_policy"]
            self.assertEqual(
                observation_policy["purpose"],
                "observation_update_relationship",
            )
            self.assertEqual(observation_policy["treatment"], "fallback_to_new")
            self.assertFalse(observation_policy["passed"])

    def test_observation_reflect_invalid_llm_decision_falls_back_to_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            existing = store.add_memory(
                "u1",
                "用户最近主要在推进 AI 眼镜长期记忆 demo",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_old"],
            )
            source_memories = [
                store.add_memory("u1", "用户在整理 prompt 缓存", kind="event", memory_type="project_state", evidence_ids=["chunk_1"]),
                store.add_memory("u1", "用户在看 TUI", kind="event", memory_type="project_state", evidence_ids=["chunk_2"]),
                store.add_memory("u1", "用户喜欢证据可追溯", kind="profile", memory_type="preference", evidence_ids=["chunk_3"]),
            ]
            job = service._create_memory_job(
                user_id="u1",
                session_id="",
                mode="observation_reflect",
                candidate_count=1,
                created_at=1778131200.0,
                source_memory_ids=[memory.id for memory in source_memories],
                evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
            )
            candidate = MemoryWriteCandidate(
                content="用户最近在整理 prompt 缓存和 TUI",
                kind="event",
                memory_type="observation",
                confidence=0.9,
                source="observation_reflect",
                evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
            )

            with patch.object(service, "_build_observation_candidate", return_value=(candidate, "rule_reflect")), patch.object(
                service,
                "_observation_update_decision",
                return_value={"action": "merge", "observation_id": "missing", "reason": "invalid id", "confidence": 0.95},
            ):
                service._process_observation_reflect_background(
                    user_id="u1",
                    session_id="",
                    reference_time=1778131200.0,
                    agent=None,
                    job_id=job["job_id"],
                    source_memories=source_memories,
                    evidence_ids=["chunk_1", "chunk_2", "chunk_3"],
                )

            observations = [
                memory for memory in store.list_memories("u1", kind="event")
                if memory.memory_type == "observation"
            ]
            job_payload = service.read_memory_job(user_id="u1", job_id=job["job_id"])

            self.assertEqual(len(observations), 2)
            self.assertEqual(store.get_memory("u1", existing.id).status, "active")
            self.assertEqual(job_payload["observation_update_decisions"][0]["action"], "new")
            self.assertEqual(job_payload["observation_update_decisions"][0]["reason"], "invalid id")

    def test_observation_reflect_ignores_superseded_and_deleted_source_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            active_profile = store.add_memory(
                "u1",
                "用户喜欢证据可追溯",
                kind="profile",
                memory_type="preference",
                evidence_ids=["chunk_profile"],
            )
            active_event = store.add_memory(
                "u1",
                "用户正在优化记忆机制",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_event"],
            )
            deleted = store.add_memory(
                "u1",
                "用户已经放弃的旧任务",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_deleted"],
            )
            superseded = store.add_memory(
                "u1",
                "用户以前在做旧项目",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_superseded"],
            )
            replacement = store.add_memory(
                "u1",
                "用户当前在做 AI 眼镜记忆机制",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_replacement"],
            )
            store.delete_memory("u1", deleted.id)
            self.assertTrue(store.mark_superseded("u1", superseded.id, replacement.id))

            source_memories = service._observation_source_memories("u1")
            source_ids = {memory.id for memory in source_memories}

            self.assertEqual(source_ids, {active_profile.id, active_event.id, replacement.id})
            self.assertNotIn("chunk_deleted", service._evidence_ids_for_memories(source_memories))
            self.assertNotIn("chunk_superseded", service._evidence_ids_for_memories(source_memories))

    def test_ordinary_chat_does_not_recall_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_memory(
                "u1",
                "用户最近主要在推进 AI 眼镜长期记忆 demo",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_1"],
            )

            result = service.chat("讲个冷知识", user_id="u1")

            self.assertFalse(result["debug"]["planner"]["needs_observation_memory"])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "skipped_by_planner")
            self.assertEqual(result["recalled_memories"], [])
            self.assertEqual(service.fake_agent.main_calls, 1)
            self.assertNotIn("observation", service.fake_agent.main_messages[-1])

    def test_review_query_recalls_observation_only_when_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            active = store.add_memory(
                "u1",
                "用户最近主要在推进 AI 眼镜长期记忆 demo，并偏好短答和可追溯证据",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_1", "chunk_2"],
            )
            superseded = store.add_memory(
                "u1",
                "用户最近主要在做南太行项目",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_old"],
            )
            self.assertTrue(store.mark_superseded("u1", superseded.id, active.id))

            result = service.chat("我最近在忙什么？", user_id="u1")

            self.assertTrue(result["debug"]["planner"]["needs_observation_memory"])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "observation_review")
            self.assertEqual(result["recalled_memories"][0]["memory_type"], "observation")
            self.assertNotIn(superseded.id, [memory["id"] for memory in result["recalled_memories"]])
            self.assertNotIn("南太行", result["reply"])
            self.assertIn("AI 眼镜长期记忆 demo", result["reply"])

    def test_observation_review_filters_by_query_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            engineering = store.add_memory(
                "u1",
                "用户的工程偏好是回复优先、后台归纳、证据可追溯",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_pref"],
            )
            project = store.add_memory(
                "u1",
                "用户项目状态是 AI 眼镜长期记忆机制进入 P2 优化",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_project"],
            )
            store.add_memory(
                "u1",
                "用户最近主要在整理日常活动和公开对话 eval",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_recent"],
            )

            engineering_result = service.chat("我的工程偏好是什么？", user_id="u1")
            project_result = service.chat("项目状态是什么？", user_id="u1")

            self.assertEqual(engineering_result["debug"]["memory"]["event_recall"]["observation_scope"], "engineering_preference")
            self.assertEqual(
                engineering_result["debug"]["memory"]["event_recall"]["observation_scope_policy"],
                {
                    "scope": "engineering_preference",
                    "role": "retrieval_hint",
                    "reason": "engineering_preference_marker",
                    "markers": ["工程偏好"],
                    "treatment": "narrow_observation_recall",
                },
            )
            self.assertEqual([memory["id"] for memory in engineering_result["recalled_memories"] if memory["memory_type"] == "observation"], [engineering.id])
            self.assertIn("回复优先", engineering_result["reply"])
            self.assertNotIn("长期记忆机制进入 P2", engineering_result["reply"])
            self.assertEqual(project_result["debug"]["memory"]["event_recall"]["observation_scope"], "project_state")
            self.assertEqual(project_result["debug"]["memory"]["event_recall"]["observation_scope_policy"]["role"], "retrieval_hint")
            self.assertEqual(project_result["debug"]["memory"]["event_recall"]["observation_scope_policy"]["reason"], "project_state_marker")
            self.assertEqual(project_result["debug"]["memory"]["event_recall"]["observation_scope_policy"]["markers"], ["项目", "状态"])
            self.assertEqual([memory["id"] for memory in project_result["recalled_memories"] if memory["memory_type"] == "observation"], [project.id])
            self.assertIn("长期记忆机制进入 P2", project_result["reply"])
            self.assertNotIn("回复优先", project_result["reply"])

    def test_observation_review_source_fallback_is_observable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            source = store.add_memory(
                "u1",
                "用户项目状态是 AI 眼镜长期记忆机制进入 P2 优化",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_project_state"],
            )

            result = service.chat("项目状态是什么？", user_id="u1")
            event_recall = result["debug"]["memory"]["event_recall"]

            self.assertEqual(event_recall["strategy"], "observation_review")
            self.assertEqual(event_recall["observation_scope"], "project_state")
            self.assertEqual(event_recall["observation_count"], 0)
            self.assertEqual(event_recall["source_memory_count"], 1)
            self.assertTrue(event_recall["source_fallback"])
            self.assertEqual(
                event_recall["source_fallback_policy"],
                {
                    "applied": True,
                    "role": "legacy_fallback",
                    "reason": "no_observation_in_scope_used_source_memories",
                    "query_scope": "project_state",
                },
            )
            self.assertEqual(result["recalled_memories"][0]["id"], source.id)
            self.assertIn("AI 眼镜长期记忆机制进入 P2 优化", result["reply"])

    def test_observation_update_only_merges_same_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            existing = store.add_memory(
                "u1",
                "用户的工程偏好是回复优先、后台归纳、证据可追溯",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_pref"],
            )
            source_memories = [
                store.add_memory("u1", "用户项目状态是 AI 眼镜长期记忆机制进入 P2 优化", kind="event", memory_type="project_state", evidence_ids=["chunk_project_1"]),
                store.add_memory("u1", "AI 眼镜项目风险是 observation 主题边界不清晰", kind="event", memory_type="project_state", evidence_ids=["chunk_project_2"]),
                store.add_memory("u1", "AI 眼镜项目决定先做轻量 scope tag", kind="event", memory_type="decision", evidence_ids=["chunk_project_3"]),
            ]
            job = service._create_memory_job(
                user_id="u1",
                session_id="",
                mode="observation_reflect",
                candidate_count=1,
                created_at=1778131200.0,
                source_memory_ids=[memory.id for memory in source_memories],
                evidence_ids=["chunk_project_1", "chunk_project_2", "chunk_project_3"],
            )
            candidate = MemoryWriteCandidate(
                content="用户项目状态是 AI 眼镜长期记忆机制进入 P2 优化，风险是 observation 主题边界不清晰",
                kind="event",
                memory_type="observation",
                confidence=0.9,
                source="observation_reflect",
                evidence_ids=["chunk_project_1", "chunk_project_2", "chunk_project_3"],
            )

            with patch.object(service, "_build_observation_candidate", return_value=(candidate, "rule_reflect")):
                service._process_observation_reflect_background(
                    user_id="u1",
                    session_id="",
                    reference_time=1778131200.0,
                    agent=None,
                    job_id=job["job_id"],
                    source_memories=source_memories,
                    evidence_ids=["chunk_project_1", "chunk_project_2", "chunk_project_3"],
                )

            observations = [memory for memory in store.list_memories("u1", kind="event") if memory.memory_type == "observation"]
            job_payload = service.read_memory_job(user_id="u1", job_id=job["job_id"])

            self.assertEqual(len(observations), 2)
            self.assertEqual(store.get_memory("u1", existing.id).status, "active")
            self.assertEqual(job_payload["observation_update_decisions"][0]["action"], "new")
            self.assertEqual(job_payload["observation_update_decisions"][0]["reason"], "no_active_observation_in_scope")
            self.assertEqual(job_payload["observation_update_decisions"][0]["observation_scope"], "project_state")
            self.assertEqual(
                job_payload["observation_update_decisions"][0]["observation_scope_policy"]["role"],
                "retrieval_hint",
            )
            self.assertEqual(
                job_payload["observation_update_decisions"][0]["observation_scope_policy"]["treatment"],
                "tag_observation_scope_only",
            )
            self.assertFalse(
                job_payload["observation_update_decisions"][0]["observation_scope_policy"]["affects_memory_type"]
            )
            self.assertEqual(service.fake_agent.main_calls, 0)

    def test_engineering_preference_query_uses_observation_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_memory(
                "u1",
                "用户工程偏好是回复优先、后台归纳、证据可追溯",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_1", "chunk_2"],
            )
            store.add_memory(
                "u1",
                "用户喜欢低糖拿铁",
                kind="profile",
                memory_type="preference",
                evidence_ids=["chunk_3"],
            )

            result = service.chat("我的工程偏好是什么？", user_id="u1")

            self.assertTrue(result["debug"]["planner"]["needs_observation_memory"])
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_event_recall")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "observation_review")
            self.assertIn("后台归纳", result["reply"])
            self.assertNotIn("低糖拿铁", result["reply"])

    def test_correction_supersedes_stale_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(semantic_payload=semantic_payload(correction=True)))
            stale = store.add_memory(
                "u1",
                "用户最近主要在做南太行项目",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_stale"],
            )
            store.add_memory(
                "u1",
                "用户正在补 observation 纠错能力",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_existing_event"],
            )
            store.add_memory(
                "u1",
                "用户偏好回复优先和证据可追溯",
                kind="profile",
                memory_type="preference",
                evidence_ids=["chunk_existing_profile"],
            )

            result = service.chat("纠正一下，我最近主要在做 AI 眼镜长期记忆，不是南太行。", user_id="u1")
            replacement = store.get_memory("u1", result["saved_memories"][0]["id"])
            refreshed_stale = store.get_memory("u1", stale.id)
            recall = service.chat("我最近在忙什么？", user_id="u1")
            audit_records = service.read_audit_records(user_id="u1", limit=10)
            supersede_records = [
                record for record in audit_records
                if record.get("record_type") == "observation_superseded"
            ]

            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertEqual(result["debug"]["memory_processing"]["primary_write_status"], "saved")
            self.assertIn(stale.id, result["debug"]["memory_processing"]["superseded_observation_ids"])
            self.assertEqual(replacement.memory_type, "project_state")
            self.assertEqual(refreshed_stale.status, "superseded")
            self.assertEqual(refreshed_stale.superseded_by, replacement.id)
            self.assertEqual(len(supersede_records), 1)
            self.assertEqual(supersede_records[0]["reason"], "correction_superseded_observation")
            self.assertEqual(supersede_records[0]["replacement_memory_id"], replacement.id)
            self.assertEqual(supersede_records[0]["superseded_observation_ids"], [stale.id])
            self.assertIn("observation_job_id", result["debug"]["memory_processing"])
            self.assertNotIn(stale.id, [memory["id"] for memory in recall["recalled_memories"]])
            self.assertNotIn("南太行项目", recall["reply"])

    def test_implicit_current_state_is_not_rule_correction_without_llm_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            stale = store.add_memory(
                "u1",
                "用户最近主要在做南太行项目",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_stale"],
            )

            result = service.chat("其实我现在主要在做 AI 眼镜长期记忆，不是南太行。", user_id="u1")
            refreshed_stale = store.get_memory("u1", stale.id)

            self.assertNotEqual(result["debug"]["correction_detection"]["backend"], "rule")
            self.assertNotEqual(result["debug"]["memory_processing"].get("mode"), "planner_correction")
            self.assertEqual(result["debug"]["memory_processing"].get("superseded_observation_ids", []), [])
            self.assertEqual(refreshed_stale.status, "active")

    def test_implicit_current_state_can_be_llm_correction_when_pre_reply_confirms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            stale = store.add_memory(
                "u1",
                "用户最近主要在做南太行项目",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_stale"],
            )
            service = FakeService(store, agent=FakeAgent(
                semantic_payload=semantic_payload(correction=True),
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "用户现在主要在做 AI 眼镜长期记忆",
                    "memory_kind": "event",
                    "memory_type": "project_state",
                    "confidence": 0.9,
                    "reason": "replaces previous project focus",
                },
            ))

            result = service.chat(
                "其实我现在主要在做 AI 眼镜长期记忆，不是南太行。",
                user_id="u1",
                routing_mode="llm_first",
            )
            refreshed_stale = store.get_memory("u1", stale.id)

            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertEqual(result["debug"]["memory_processing"]["primary_write_status"], "saved")
            self.assertEqual(result["debug"]["correction_detection"]["backend"], "llm")
            self.assertEqual(result["debug"]["correction_detection"]["candidate_count"], 1)
            gate = result["debug"]["correction_detection"]["classification_decisions"][-1]
            self.assertEqual(gate["policy"], "semantic_correction_gate")
            self.assertTrue(gate["used_turn_semantics"])
            self.assertEqual(gate["action"], "run")
            self.assertIn(stale.id, result["debug"]["memory_processing"]["superseded_observation_ids"])
            self.assertEqual(refreshed_stale.status, "superseded")
            self.assertIn("AI 眼镜长期记忆", result["saved_memories"][0]["content"])

    def test_unified_semantic_correction_false_skips_llm_correction_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(correction=False),
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "用户现在主要在做 AI 眼镜长期记忆",
                    "memory_kind": "event",
                    "memory_type": "project_state",
                    "confidence": 0.9,
                    "reason": "would have corrected without semantic gate",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat(
                "普通聊天，不是在纠正旧记忆",
                user_id="u1",
                routing_mode="llm_first",
            )

            self.assertEqual(agent.correction_calls, 0)
            self.assertEqual(result["debug"]["correction_detection"]["backend"], "none")
            self.assertEqual(result["debug"]["correction_detection"]["candidate_count"], 0)
            gate = result["debug"]["correction_detection"]["classification_decisions"][0]
            self.assertEqual(gate["policy"], "semantic_correction_gate")
            self.assertTrue(gate["used_turn_semantics"])
            self.assertEqual(gate["action"], "skip")
            self.assertEqual(gate["skipped_reason"], "llm_semantics_correction_false")

    def test_unified_semantic_rule_fallback_keeps_llm_correction_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload="not json",
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "用户喜欢靠窗安静座位",
                    "memory_kind": "profile",
                    "memory_type": "preference",
                    "confidence": 0.88,
                    "reason": "user updates previous seating preference",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我前面那个座位偏好更新一下", user_id="u1", defer_memory_writes=True)

            self.assertEqual(agent.correction_calls, 1)
            self.assertEqual(result["debug"]["turn_semantics"]["backend"], "rule_fallback")
            self.assertEqual(result["debug"]["correction_detection"]["backend"], "llm")
            gate = result["debug"]["correction_detection"]["classification_decisions"][-1]
            self.assertEqual(gate["action"], "fallback")
            self.assertEqual(gate["fallback_reason"], "turn_semantics_error")

    def test_unified_semantic_rule_backend_keeps_correction_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "用户喜欢靠窗安静座位",
                    "memory_kind": "profile",
                    "memory_type": "preference",
                    "confidence": 0.88,
                    "reason": "user updates previous seating preference",
                },
            )
            service = FakeService(store, agent=agent)

            result = service._detect_correction_with_semantic_gate(
                "我前面那个座位偏好更新一下",
                agent=agent,
                turn_semantics={"backend": "rule_fallback", "flags": {"correction": False}},
                phase="unit_test",
            ).debug_payload()

            self.assertEqual(agent.correction_calls, 1)
            self.assertEqual(result["backend"], "llm")
            gate = result["classification_decisions"][-1]
            self.assertEqual(gate["action"], "fallback")
            self.assertEqual(gate["fallback_reason"], "non_llm_semantic_backend:rule_fallback")

    def test_natural_correction_marker_variants_are_detected(self) -> None:
        cases = (
            "更准确地说，我现在主要在做 AI 眼镜长期记忆，不是南太行。",
            "我之前说的不对，应该是我现在主要在做 AI 眼镜长期记忆。",
        )
        for message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    store = self.make_store(Path(tmpdir))
                    service = FakeService(store, agent=FakeAgent(semantic_payload=semantic_payload(correction=True)))
                    result = service.chat(message, user_id="u1")

                    self.assertIn(result["debug"]["correction_detection"]["backend"], {"rule", "llm", "initial_candidates"})
                    self.assertEqual(result["debug"]["memory_processing"].get("status"), "saved")
                    self.assertEqual(store.list_memories("u1", kind="event")[0].memory_type, "project_state")
                    self.assertIn("AI 眼镜长期记忆", store.list_memories("u1", kind="event")[0].content)

    def test_explicit_event_correction_without_type_marker_defaults_to_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("纠正一下，我今天下午3点是和 Mina 开会，不是和 Bob", user_id="u1")
            saved = store.list_memories("u1", kind="event")

            self.assertIn(result["debug"]["correction_detection"]["backend"], {"none", "rule", "llm", "initial_candidates"})
            self.assertIn(result["debug"]["memory_processing"].get("status"), {"saved", "pending"})
            self.assertEqual(saved[0].memory_type, "event")
            self.assertIn("Mina", saved[0].content)

    def test_plain_edit_or_update_commands_are_not_rule_correction(self) -> None:
        cases = (
            "改一下方案，先做可观测 debug",
            "这个项目更新一下进度文档",
            "刚才那个 demo 文案改一下，写得自然一点",
            "我前面那个座位偏好更新一下",
        )
        for message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    store = self.make_store(Path(tmpdir))
                    service = FakeService(store)
                    result = service.chat(message, user_id="u1")

                    self.assertNotEqual(result["debug"]["correction_detection"]["backend"], "rule")
                    self.assertNotEqual(result["debug"]["memory_processing"].get("mode"), "planner_correction")

    def test_fix_prefix_can_be_llm_correction_when_pre_reply_confirms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            stale = store.add_memory(
                "u1",
                "用户最近主要在做南太行项目",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_stale"],
            )
            service = FakeService(store, agent=FakeAgent(
                semantic_payload=semantic_payload(correction=True),
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "用户现在主要在做 AI 眼镜长期记忆",
                    "memory_kind": "event",
                    "memory_type": "project_state",
                    "confidence": 0.9,
                    "reason": "replaces previous project focus",
                },
            ))

            result = service.chat(
                "改一下，我现在主要在做 AI 眼镜长期记忆，不是南太行。",
                user_id="u1",
                routing_mode="llm_first",
            )
            refreshed_stale = store.get_memory("u1", stale.id)

            self.assertEqual(result["debug"]["correction_detection"]["backend"], "llm")
            self.assertEqual(result["debug"]["correction_detection"]["candidate_count"], 1)
            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertEqual(result["debug"]["memory_processing"]["primary_write_status"], "saved")
            self.assertIn(stale.id, result["debug"]["memory_processing"]["superseded_observation_ids"])
            self.assertEqual(refreshed_stale.status, "superseded")
            self.assertIn("AI 眼镜长期记忆", result["saved_memories"][0]["content"])

    def test_plain_preference_with_actually_is_not_treated_as_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("其实我喜欢低糖拿铁", user_id="u1")

            self.assertIn(result["debug"]["correction_detection"]["backend"], {"none", "llm"})
            self.assertNotEqual(result["debug"]["memory_processing"].get("mode"), "planner_correction")
            self.assertEqual(store.list_memories("u1", kind="profile")[0].memory_type, "preference")

    def test_plain_contrast_with_not_but_is_not_rule_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("不是我不喜欢咖啡，而是今天太晚了", user_id="u1")

            self.assertIn(result["debug"]["correction_detection"]["backend"], {"none", "llm"})
            self.assertNotEqual(result["debug"]["memory_processing"].get("mode"), "planner_correction")

    def test_llm_correction_fallback_saves_profile_preference_when_agent_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                semantic_payload=semantic_payload(correction=True),
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "用户喜欢靠窗安静座位",
                    "memory_kind": "profile",
                    "memory_type": "preference",
                    "confidence": 0.88,
                    "reason": "user updates previous seating preference",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我前面那个座位偏好更新一下", user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])
            memories = store.list_memories("u1", kind="profile")

            self.assertEqual(result["debug"]["correction_detection"]["backend"], "llm")
            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertEqual(job["correction_detection"]["backend"], "initial_candidates")
            self.assertEqual(job["status"], "saved")
            self.assertEqual(memories[0].content, "用户喜欢靠窗安静座位")
            self.assertEqual(memories[0].memory_type, "preference")

    def test_correction_target_resolution_supersedes_old_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            old = store.add_memory(
                "u1",
                "用户喜欢靠窗座位",
                kind="profile",
                memory_type="preference",
                confidence=0.86,
            )
            agent = FakeAgent(
                semantic_payload=semantic_payload(correction=True),
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "用户喜欢吧台位置",
                    "memory_kind": "profile",
                    "memory_type": "preference",
                    "confidence": 0.9,
                    "reason": "updates previous seating preference",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我前面那个座位偏好更新一下，我喜欢吧台位置", user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])
            refreshed_old = store.get_memory("u1", old.id)
            active = store.list_memories("u1", kind="profile")

            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertIn(job["correction_detection"]["backend"], {"llm", "initial_candidates"})
            self.assertEqual(job["status"], "saved")
            self.assertEqual(refreshed_old.status, "superseded")
            self.assertEqual(job["superseded_memory_ids"], [old.id])
            self.assertEqual(
                job["correction_target_resolution"]["resolution_backend"],
                "local",
            )
            self.assertEqual(
                job["correction_target_resolution"]["superseded_memory_ids"],
                [old.id],
            )
            self.assertEqual(job["correction_target_resolution"]["target_hint_policy"]["role"], "retrieval_hint")
            self.assertEqual(
                job["correction_target_resolution"]["target_hint_policy"]["treatment"],
                "score_candidate_targets_only",
            )
            self.assertIn("座位", job["correction_target_resolution"]["target_hint_policy"]["hints"])
            self.assertIn("偏好", job["correction_target_resolution"]["target_hint_policy"]["hints"])
            self.assertEqual(len(active), 1)
            self.assertIn("吧台", active[0].content)

    def test_preference_correction_recall_and_explanation_use_replacement_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            old = store.add_memory(
                "u1",
                "用户喜欢冰美式",
                kind="profile",
                memory_type="preference",
                evidence_ids=["old_turn"],
                confidence=0.86,
            )
            agent = FakeAgent(
                pre_reply_payload={
                    "我现在喜欢喝什么": {
                        "reply_mode": "llm",
                        "answer_source": "llm",
                        "scope": "unknown",
                        "needs_profile_memory": True,
                        "needs_event_memory": False,
                        "memory_recall_type": "profile",
                        "recall_goal": "specific_fact",
                        "confidence": 0.95,
                        "reason": "asks for current drink preference",
                    },
                    "为什么这么说": {
                        "reply_mode": "llm",
                        "answer_source": "llm",
                        "scope": "unknown",
                        "needs_profile_memory": False,
                        "needs_event_memory": False,
                        "memory_recall_type": "none",
                        "recall_goal": "none",
                        "confidence": 0.95,
                        "reason": "asks for source explanation",
                    },
                    "*": {
                        "reply_mode": "llm",
                        "answer_source": "llm",
                        "scope": "unknown",
                        "needs_profile_memory": False,
                        "needs_event_memory": False,
                        "memory_recall_type": "none",
                        "recall_goal": "none",
                        "confidence": 0.95,
                        "reason": "default test route",
                    },
                },
                semantic_payload={
                    "我前面那个饮品偏好更新一下，我最近不喝咖啡了": semantic_payload(
                        correction=True,
                        memory_action="correction",
                    ),
                    "我现在喜欢喝什么": semantic_payload(correction=False, memory_action="none"),
                    "为什么这么说": {
                        **semantic_payload(correction=False, memory_action="none"),
                        "turn_intent": "explanation",
                        "flags": {
                            "transient": False,
                            "do_not_remember": False,
                            "correction": False,
                            "explanation_query": True,
                        },
                    },
                    "*": semantic_payload(correction=False, memory_action="none"),
                },
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "用户最近不喝咖啡",
                    "memory_kind": "profile",
                    "memory_type": "preference",
                    "confidence": 0.9,
                    "reason": "updates previous drink preference",
                },
                correction_target_payload={
                    "action": "supersede",
                    "memory_id": old.id,
                    "confidence": 0.9,
                    "reason": "replacement conflicts with previous drink preference",
                },
            )
            service = FakeService(store, agent=agent)

            update = service.chat(
                "我前面那个饮品偏好更新一下，我最近不喝咖啡了",
                user_id="u1",
                routing_mode="llm_first",
                defer_memory_writes=True,
            )
            job = service.read_memory_job(user_id="u1", job_id=update["debug"]["memory_processing"]["job_id"])
            refreshed_old = store.get_memory("u1", old.id)
            active = store.list_memories("u1", kind="profile")

            self.assertEqual(job["status"], "saved")
            self.assertEqual(refreshed_old.status, "superseded")
            self.assertEqual(job["superseded_memory_ids"], [old.id])
            self.assertEqual(job["correction_target_resolution"]["resolution_backend"], "llm")
            self.assertEqual(job["correction_target_resolution"]["superseded_memory_ids"], [old.id])
            self.assertEqual(job["correction_target_resolution"]["matched_memory_ids"], [old.id])
            self.assertEqual(len(active), 1)
            self.assertIn("不喝咖啡", active[0].content)
            self.assertEqual(refreshed_old.superseded_by, active[0].id)

            recall = service.chat("我现在喜欢喝什么？", user_id="u1", routing_mode="llm_first")
            recalled_contents = "\n".join(memory["content"] for memory in recall["recalled_memories"])

            self.assertEqual(recall["source_summary"]["primary_source"], "profile")
            self.assertIn("用户最近不喝咖啡", recalled_contents)
            self.assertIn("用户最近不喝咖啡", recall["reply"])
            self.assertNotIn("用户喜欢冰美式", recalled_contents)
            self.assertNotIn("用户喜欢冰美式", recall["reply"])

            explain = service.chat("为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(explain["debug"]["source_summary"]["primary_source"], "profile")
            self.assertIn("稳定画像", explain["reply"])
            self.assertIn("用户最近不喝咖啡", explain["reply"])
            explanation_memories = explain["debug"]["explanation_context"]["recalled_memories"]
            self.assertEqual(explanation_memories[0]["content"], "用户最近不喝咖啡")
            self.assertEqual(explanation_memories[0]["source"], "correction")
            self.assertTrue(explanation_memories[0]["source_id"])
            self.assertTrue(explanation_memories[0]["ingestion_id"])
            self.assertTrue(explanation_memories[0]["evidence_ids"])
            self.assertEqual(explanation_memories[0]["source_trace"]["source_id"], explanation_memories[0]["source_id"])
            self.assertEqual(explanation_memories[0]["source_trace"]["ingestion_id"], explanation_memories[0]["ingestion_id"])
            self.assertEqual(explanation_memories[0]["source_trace"]["evidence_ids"], explanation_memories[0]["evidence_ids"])
            self.assertIn("source_id=", explain["reply"])
            self.assertIn("evidence_ids=", explain["reply"])

    def test_correction_target_resolution_supersedes_open_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            old = store.add_memory(
                "u1",
                "周五前补 memory eval",
                kind="event",
                memory_type="task",
                tags=["task_status:open"],
                confidence=0.84,
            )
            agent = FakeAgent(
                semantic_payload=semantic_payload(correction=True),
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "取消周五前补 memory eval",
                    "memory_kind": "event",
                    "memory_type": "task",
                    "confidence": 0.9,
                    "reason": "cancels previous task",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("周五前补 memory eval 不是完成，是取消了", user_id="u1", defer_memory_writes=True)
            refreshed_old = store.get_memory("u1", old.id)
            active_tasks = [memory for memory in store.list_memories("u1", kind="event") if memory.memory_type == "task"]

            self.assertIn(result["debug"]["memory_processing"]["status"], {"saved", "pending"})
            if result["debug"]["memory_processing"]["status"] == "pending":
                job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])
                self.assertEqual(job["status"], "saved")
            self.assertEqual(refreshed_old.status, "superseded")
            if result["debug"]["memory_processing"]["status"] == "saved":
                self.assertEqual(result["debug"]["memory_processing"]["superseded_memory_ids"], [old.id])
                self.assertEqual(
                    result["debug"]["memory_processing"]["correction_target_resolution"]["superseded_memory_ids"],
                    [old.id],
                )
                self.assertEqual(result["debug"]["memory_processing"]["task_status_updates"][0]["status"], "cancelled")
            self.assertEqual(len(active_tasks), 1)
            self.assertIn("取消", active_tasks[0].content)

    def test_correction_target_resolution_supersedes_project_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            old = store.add_memory(
                "u1",
                "项目主线是南太行攻略",
                kind="event",
                memory_type="project_state",
                confidence=0.83,
            )
            service = FakeService(store, agent=FakeAgent(semantic_payload=semantic_payload(correction=True)))

            result = service.chat("项目主线说错了，不是南太行攻略，是 AI 眼镜长期记忆机制", user_id="u1")
            refreshed_old = store.get_memory("u1", old.id)
            active_project_states = [
                memory for memory in store.list_memories("u1", kind="event")
                if memory.memory_type == "project_state"
            ]

            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertEqual(refreshed_old.status, "superseded")
            self.assertEqual(result["debug"]["memory_processing"]["superseded_memory_ids"], [old.id])
            self.assertEqual(
                result["debug"]["memory_processing"]["correction_target_resolution"]["superseded_memory_ids"],
                [old.id],
            )
            self.assertEqual(len(active_project_states), 1)
            self.assertIn("AI 眼镜长期记忆机制", active_project_states[0].content)

    def test_correction_target_resolution_does_not_supersede_unrelated_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            old = store.add_memory(
                "u1",
                "用户喜欢靠窗座位",
                kind="profile",
                memory_type="preference",
                confidence=0.86,
            )
            service = FakeService(store)

            result = service.chat("纠正一下，我最近主要在做 AI 眼镜长期记忆，不是南太行。", user_id="u1")

            self.assertEqual(store.get_memory("u1", old.id).status, "active")
            self.assertEqual(result["debug"]["memory_processing"]["superseded_memory_ids"], [])
            self.assertEqual(
                result["debug"]["memory_processing"]["correction_target_resolution"]["superseded_memory_ids"],
                [],
            )

    def test_correction_target_resolution_low_confidence_llm_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            old = store.add_memory(
                "u1",
                "用户喜欢靠窗座位",
                kind="profile",
                memory_type="preference",
                confidence=0.86,
            )
            agent = FakeAgent(
                semantic_payload=semantic_payload(correction=True),
                correction_payload={
                    "is_correction": True,
                    "corrected_content": "用户喜欢吧台位置",
                    "memory_kind": "profile",
                    "memory_type": "preference",
                    "confidence": 0.9,
                    "reason": "updates previous seating preference",
                },
                correction_target_payload={
                    "action": "supersede",
                    "memory_id": old.id,
                    "confidence": 0.4,
                    "reason": "too uncertain",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat(
                "那个信息更新一下，我现在喜欢吧台位置",
                user_id="u1",
                defer_memory_writes=True,
                routing_mode="llm_first",
            )
            job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])

            self.assertEqual(job["status"], "saved")
            self.assertEqual(store.get_memory("u1", old.id).status, "active")
            self.assertEqual(job["superseded_memory_ids"], [])
            self.assertEqual(job["correction_target_resolution"]["resolution_backend"], "llm")
            self.assertEqual(job["correction_target_resolution"]["target_hint_policy"]["role"], "weak_signal")
            self.assertEqual(
                job["correction_target_resolution"]["target_hint_policy"]["reason"],
                "ambiguous_reference_without_specific_hint",
            )
            self.assertEqual(job["correction_target_resolution"]["target_hint_policy"]["hints"], [])
            self.assertEqual(job["correction_target_resolution"]["confidence_policy"]["purpose"], "correction_target_resolution")
            self.assertEqual(job["correction_target_resolution"]["confidence_policy"]["treatment"], "fallback_no_supersede")
            self.assertFalse(job["correction_target_resolution"]["confidence_policy"]["passed"])

    def test_llm_correction_fallback_low_confidence_and_invalid_json_are_ignored(self) -> None:
        cases = (
            {
                "is_correction": True,
                "corrected_content": "用户喜欢靠窗安静座位",
                "memory_kind": "profile",
                "memory_type": "preference",
                "confidence": 0.4,
                "reason": "too uncertain",
            },
            "not json",
            {
                "is_correction": True,
                "corrected_content": "",
                "memory_kind": "profile",
                "memory_type": "preference",
                "confidence": 0.95,
                "reason": "empty",
            },
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    store = self.make_store(Path(tmpdir))
                    service = FakeService(store, agent=FakeAgent(
                        semantic_payload=semantic_payload(correction=True),
                        correction_payload=payload,
                    ))

                    result = service.chat("我前面那个座位偏好更新一下", user_id="u1", defer_memory_writes=True)
                    job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])

                    self.assertEqual(result["debug"]["correction_detection"]["backend"], "llm")
                    self.assertEqual(result["debug"]["correction_detection"]["candidate_count"], 0)
                    if isinstance(payload, dict) and payload.get("confidence") == 0.4:
                        self.assertEqual(
                            result["debug"]["correction_detection"]["confidence_policy"]["purpose"],
                            "correction_detection",
                        )
                        self.assertEqual(
                            result["debug"]["correction_detection"]["confidence_policy"]["treatment"],
                            "ignore_correction",
                        )
                        self.assertFalse(result["debug"]["correction_detection"]["confidence_policy"]["passed"])
                    self.assertEqual(job["status"], "skipped")
                    self.assertEqual(store.list_memories("u1"), [])

    def test_correction_does_not_supersede_unrelated_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(semantic_payload=semantic_payload(correction=True)))
            unrelated = store.add_memory(
                "u1",
                "用户长期偏好是回复优先和证据可追溯",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_unrelated"],
            )

            result = service.chat("纠正一下，我最近主要在做 AI 眼镜长期记忆，不是南太行。", user_id="u1")

            self.assertEqual(result["debug"]["memory_processing"]["superseded_observation_ids"], [])
            self.assertEqual(store.get_memory("u1", unrelated.id).status, "active")

    def test_sensitive_correction_does_not_supersede_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(semantic_payload=semantic_payload(correction=True)))
            stale = store.add_memory(
                "u1",
                "用户最近主要在推进 AI 眼镜长期记忆 demo",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_stale"],
            )

            result = service.chat("纠正一下，我的 API key 是 secret-token，不是南太行。", user_id="u1")

            self.assertEqual(result["debug"]["memory_processing"]["status"], "rejected")
            self.assertEqual(result["debug"]["memory_processing"]["superseded_observation_ids"], [])
            self.assertEqual(store.get_memory("u1", stale.id).status, "active")

    def test_correction_supersede_is_user_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(semantic_payload=semantic_payload(correction=True)))
            user_a = store.add_memory(
                "u1",
                "用户最近主要在做南太行项目",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_a"],
            )
            user_b = store.add_memory(
                "u2",
                "用户最近主要在做南太行项目",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_b"],
            )

            service.chat("纠正一下，我最近主要在做 AI 眼镜长期记忆，不是南太行。", user_id="u1")

            self.assertEqual(store.get_memory("u1", user_a.id).status, "superseded")
            self.assertEqual(store.get_memory("u2", user_b.id).status, "active")

    def test_standard_library_memory_job_endpoint_returns_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            from ai_glasses_memory_assistant.server import GlassesHandler

            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("记一下今晚8点给妈妈打电话", user_id="u1")
            job_id = result["debug"]["memory_processing"]["job_id"]
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, headers, payload = self.http_get(
                    GlassesHandler,
                    f"/api/memory/jobs?user_id=u1&job_id={job_id}",
                )
                missing_status, _, missing_payload = self.http_get(
                    GlassesHandler,
                    f"/api/memory/jobs?user_id=u2&job_id={job_id}",
                )
            finally:
                GlassesHandler.service = original_service

            self.assertEqual(status, 200)
            self.assertIn("application/json", headers["Content-Type"])
            self.assertEqual(payload["job"]["job_id"], job_id)
            self.assertEqual(payload["job"]["status"], "saved")
            self.assertEqual(missing_status, 404)
            self.assertIn("not found", missing_payload["detail"])

    def test_standard_library_chat_ignores_legacy_routing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            from ai_glasses_memory_assistant.server import GlassesHandler

            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "none",
                "recall_goal": "none",
                "confidence": 0.95,
            })
            service = FakeService(store, agent=agent)
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, _, payload = self.http_post(
                    GlassesHandler,
                    "/api/chat",
                    {
                        "message": "普通问答：水的化学式是什么？",
                        "user_id": "u1",
                        "routing_mode": "legacy_mode",
                    },
                )
                bad_mode_status, _, bad_mode_payload = self.http_post(
                    GlassesHandler,
                    "/api/chat",
                    {
                        "message": "你好",
                        "user_id": "u1",
                        "routing_mode": "bad_mode",
                    },
                )
            finally:
                GlassesHandler.service = original_service

            self.assertEqual(status, 200)
            self.assertEqual(payload["debug"]["routing"]["mode"], "llm_first")
            self.assertEqual(bad_mode_status, 200)
            self.assertEqual(bad_mode_payload["debug"]["routing"]["mode"], "llm_first")

    def test_standard_library_memory_job_endpoint_returns_job(self) -> None:
        from ai_glasses_memory_assistant.server import GlassesHandler

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("记一下今晚8点给妈妈打电话", user_id="u1")
            job_id = result["debug"]["memory_processing"]["job_id"]
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, _, payload = self.http_get(GlassesHandler, f"/api/memory/jobs?user_id=u1&job_id={job_id}")
                missing_status, _, _ = self.http_get(GlassesHandler, f"/api/memory/jobs?user_id=u2&job_id={job_id}")
            finally:
                GlassesHandler.service = original_service

            self.assertEqual(status, 200)
            self.assertEqual(payload["job"]["job_id"], job_id)
            self.assertEqual(payload["job"]["status"], "saved")
            self.assertEqual(missing_status, 404)

    def test_standard_library_timeline_search_endpoint_returns_user_scoped_chunks(self) -> None:
        from ai_glasses_memory_assistant.server import GlassesHandler

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.timeline_store.add_turn("u1", "我提到国内网络下语音识别不稳定")
            service.timeline_store.add_turn("u2", "u2 也提到国内网络")
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, _, payload = self.http_get(GlassesHandler, "/api/timeline/search?user_id=u1&q=国内网络")
            finally:
                GlassesHandler.service = original_service

            self.assertEqual(status, 200)
            chunks = payload["chunks"]
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["user_id"], "u1")
            self.assertIn("语音识别不稳定", chunks[0]["text"])
            self.assertEqual(chunks[0]["active_refs"], 0)
            self.assertEqual(chunks[0]["retained_refs"], 0)

    def test_timeline_management_get_chunks_includes_reference_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            timeline = service.timeline_store.add_turn("u1", "证据查看来自这一句原话")
            chunk_id = timeline.chunks[0].id
            store.add_memory("u1", "用户在验证 P3 证据查看", kind="event", evidence_ids=[chunk_id])

            payload = service.timeline_chunks_for_management(user_id="u1", chunk_ids=[chunk_id, "missing"])

            self.assertEqual(payload["requested_count"], 2)
            self.assertEqual(payload["not_found_count"], 1)
            self.assertEqual(payload["chunks"][0]["id"], chunk_id)
            self.assertEqual(payload["chunks"][0]["active_refs"], 1)
            self.assertEqual(payload["chunks"][0]["retained_refs"], 1)
            self.assertFalse(payload["chunks"][0]["can_soft_delete"])

    def test_delete_timeline_chunks_soft_deletes_only_unreferenced_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            referenced = service.timeline_store.add_turn("u1", "active reference chunk")
            unreferenced = service.timeline_store.add_turn("u1", "orphan removable chunk")
            referenced_id = referenced.chunks[0].id
            unreferenced_id = unreferenced.chunks[0].id
            store.add_memory("u1", "用户说过 active reference chunk", kind="event", evidence_ids=[referenced_id])

            payload = service.delete_timeline_chunks(
                user_id="u1",
                chunk_ids=[referenced_id, unreferenced_id, "missing"],
                purge=False,
            )

            self.assertEqual(payload["requested_count"], 3)
            self.assertEqual(payload["deleted_count"], 1)
            self.assertEqual(payload["retained_count"], 1)
            self.assertEqual(payload["not_found_count"], 1)
            self.assertIn("仍然 active 的 chunk", payload["explanation"])
            by_id = {item["chunk_id"]: item for item in payload["results"]}
            self.assertEqual(by_id[referenced_id]["reason"], "active_memory_reference")
            self.assertIn("还被 active 记忆引用", by_id[referenced_id]["explanation"])
            self.assertEqual(by_id[unreferenced_id]["action"], "soft_deleted")
            self.assertIn("软删除", by_id[unreferenced_id]["explanation"])
            self.assertEqual(service.timeline_store.search_chunks("u1", "orphan removable", limit=5), [])
            self.assertEqual(len(service.timeline_store.search_chunks("u1", "active reference", limit=5)), 1)

    def test_delete_timeline_chunks_hard_purge_retains_inactive_evidence_until_memory_purged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            timeline = service.timeline_store.add_turn("u1", "stale retained evidence")
            chunk_id = timeline.chunks[0].id
            stale = store.add_memory("u1", "用户曾经提到 stale retained evidence", kind="event", status="stale", evidence_ids=[chunk_id])

            retained = service.delete_timeline_chunks(user_id="u1", chunk_ids=[chunk_id], purge=True)

            self.assertEqual(retained["purged_count"], 0)
            self.assertEqual(retained["deleted_count"], 1)
            self.assertEqual(retained["results"][0]["action"], "soft_deleted")
            self.assertEqual(retained["results"][0]["reason"], "retained_inactive_memory_reference")
            self.assertEqual(service.timeline_store.list_chunks_by_ids("u1", [chunk_id], include_deleted=True)[0].status, "deleted")

            store.purge_memory("u1", stale.id)
            purged = service.delete_timeline_chunks(user_id="u1", chunk_ids=[chunk_id], purge=True)

            self.assertEqual(purged["purged_count"], 1)
            self.assertEqual(purged["results"][0]["action"], "purged")
            self.assertEqual(service.timeline_store.list_chunks_by_ids("u1", [chunk_id], include_deleted=True), [])

    def test_standard_library_timeline_chunks_endpoint_is_user_scoped_and_deletes_batch(self) -> None:
        from ai_glasses_memory_assistant.server import GlassesHandler

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            u1_timeline = service.timeline_store.add_turn("u1", "u1 timeline 管理")
            u2_timeline = service.timeline_store.add_turn("u2", "u2 timeline 管理")
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                chunk_id = u1_timeline.chunks[0].id
                other_id = u2_timeline.chunks[0].id
                get_status, _, get_payload = self.http_get(
                    GlassesHandler,
                    f"/api/timeline/chunks?user_id=u1&ids={chunk_id},{other_id}",
                )
                delete_status, _, delete_payload = self.http_delete(
                    GlassesHandler,
                    f"/api/timeline/chunks?user_id=u1&ids={chunk_id}&purge=true",
                )
            finally:
                GlassesHandler.service = original_service

            self.assertEqual(get_status, 200)
            self.assertEqual([chunk["id"] for chunk in get_payload["chunks"]], [chunk_id])
            self.assertEqual(delete_status, 200)
            self.assertEqual(delete_payload["purged_count"], 1)

    def test_delete_memory_removes_unshared_timeline_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("我叫 Jack", user_id="u1")
            memory_id = result["saved_memories"][0]["id"]
            chunk_ids = result["debug"]["timeline"]["chunk_ids"]

            self.assertEqual(len(service.timeline_store.search_chunks("u1", "Jack", limit=5)), 1)
            self.assertTrue(service.delete_memory(user_id="u1", memory_id=memory_id))

            self.assertEqual(service.timeline_store.search_chunks("u1", "Jack", limit=5), [])
            self.assertEqual(service.timeline_store.list_recent_chunks("u1", limit=5), [])
            self.assertEqual(store.active_memory_count_with_evidence("u1", chunk_ids[0]), 0)

    def test_delete_memory_keeps_timeline_evidence_still_used_by_active_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            timeline = service.timeline_store.add_turn("u1", "我叫 Jack，喜欢低糖拿铁")
            chunk_id = timeline.chunks[0].id
            first = store.add_memory("u1", "我叫 Jack", kind="profile", evidence_ids=[chunk_id])
            second = store.add_memory("u1", "我喜欢低糖拿铁", kind="profile", evidence_ids=[chunk_id])

            self.assertTrue(service.delete_memory(user_id="u1", memory_id=first.id))
            self.assertEqual(len(service.timeline_store.search_chunks("u1", "Jack", limit=5)), 1)

            self.assertTrue(service.delete_memory(user_id="u1", memory_id=second.id))
            self.assertEqual(service.timeline_store.search_chunks("u1", "Jack", limit=5), [])

    def test_standard_library_delete_memory_cleans_timeline_evidence(self) -> None:
        from ai_glasses_memory_assistant.server import GlassesHandler

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("我叫 Jack", user_id="u1")
            memory_id = result["saved_memories"][0]["id"]
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, _, payload = self.http_delete(
                    GlassesHandler,
                    f"/api/memories/{memory_id}?user_id=u1",
                )
                search_status, _, search_payload = self.http_get(
                    GlassesHandler,
                    "/api/timeline/search?user_id=u1&q=Jack",
                )
            finally:
                GlassesHandler.service = original_service

            self.assertEqual(status, 200)
            self.assertEqual(payload, {"deleted": True})
            self.assertEqual(search_status, 200)
            self.assertEqual(search_payload["chunks"], [])

    def test_purge_memory_removes_row_unshared_timeline_parent_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("我叫 Jack", user_id="u1")
            memory_id = result["saved_memories"][0]["id"]
            turn_id = result["debug"]["timeline"]["turn_id"]
            chunk_ids = result["debug"]["timeline"]["chunk_ids"]

            purge = service.purge_memory(user_id="u1", memory_id=memory_id)

            self.assertEqual(purge["deleted"], True)
            self.assertEqual(purge["purged"], True)
            self.assertEqual(purge["purged_chunk_count"], len(chunk_ids))
            self.assertEqual(purge["purged_parent_count"], 1)
            self.assertGreaterEqual(purge["audit_records_removed"], 1)
            self.assertIsNone(store.get_memory("u1", memory_id))
            self.assertEqual(store.search("u1", "Jack"), [])
            self.assertEqual(
                store._conn.execute("SELECT COUNT(*) AS count FROM memories WHERE id = ?", (memory_id,)).fetchone()["count"],
                0,
            )
            self.assertEqual(service.timeline_store.search_chunks("u1", "Jack", limit=5), [])
            self.assertEqual(service.timeline_store.list_recent_chunks("u1", limit=5), [])
            self.assertIsNone(service.timeline_store.get_turn("u1", turn_id))
            self.assertEqual(
                service.timeline_store._conn.execute(
                    "SELECT COUNT(*) AS count FROM raw_turns WHERE id = ?",
                    (turn_id,),
                ).fetchone()["count"],
                0,
            )
            self.assertEqual(
                service.timeline_store._conn.execute(
                    "SELECT COUNT(*) AS count FROM chunks WHERE id = ?",
                    (chunk_ids[0],),
                ).fetchone()["count"],
                0,
            )
            audit_text = service.audit_path.read_text(encoding="utf-8") if service.audit_path.exists() else ""
            self.assertNotIn(memory_id, audit_text)
            self.assertNotIn(chunk_ids[0], audit_text)
            self.assertNotIn(turn_id, audit_text)

    def test_purge_soft_deleted_memory_still_physically_cleans_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("我叫 Jack", user_id="u1")
            memory_id = result["saved_memories"][0]["id"]
            turn_id = result["debug"]["timeline"]["turn_id"]
            chunk_id = result["debug"]["timeline"]["chunk_ids"][0]

            self.assertTrue(service.delete_memory(user_id="u1", memory_id=memory_id))
            self.assertIsNone(store.get_memory("u1", memory_id))
            self.assertEqual(
                store._conn.execute("SELECT COUNT(*) AS count FROM memories WHERE id = ?", (memory_id,)).fetchone()["count"],
                1,
            )

            purge = service.purge_memory(user_id="u1", memory_id=memory_id)

            self.assertEqual(purge["deleted"], True)
            self.assertEqual(purge["purged"], True)
            self.assertEqual(purge["purged_chunk_count"], 1)
            self.assertEqual(purge["purged_parent_count"], 1)
            self.assertEqual(
                store._conn.execute("SELECT COUNT(*) AS count FROM memories WHERE id = ?", (memory_id,)).fetchone()["count"],
                0,
            )
            self.assertEqual(
                service.timeline_store._conn.execute("SELECT COUNT(*) AS count FROM chunks WHERE id = ?", (chunk_id,)).fetchone()["count"],
                0,
            )
            self.assertEqual(
                service.timeline_store._conn.execute("SELECT COUNT(*) AS count FROM raw_turns WHERE id = ?", (turn_id,)).fetchone()["count"],
                0,
            )

    def test_purge_memory_keeps_shared_evidence_until_last_active_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            timeline = service.timeline_store.add_turn("u1", "我叫 Jack，喜欢低糖拿铁")
            turn_id = timeline.turn.id
            chunk_id = timeline.chunks[0].id
            first = store.add_memory("u1", "我叫 Jack", kind="profile", evidence_ids=[chunk_id])
            second = store.add_memory("u1", "我喜欢低糖拿铁", kind="profile", evidence_ids=[chunk_id])

            first_purge = service.purge_memory(user_id="u1", memory_id=first.id)

            self.assertEqual(first_purge["purged_chunk_count"], 0)
            self.assertEqual(first_purge["purged_parent_count"], 0)
            self.assertIsNone(store.get_memory("u1", first.id))
            self.assertIsNotNone(store.get_memory("u1", second.id))
            self.assertEqual(len(service.timeline_store.search_chunks("u1", "Jack", limit=5)), 1)
            self.assertIsNotNone(service.timeline_store.get_turn("u1", turn_id))

            second_purge = service.purge_memory(user_id="u1", memory_id=second.id)

            self.assertEqual(second_purge["purged_chunk_count"], 1)
            self.assertEqual(second_purge["purged_parent_count"], 1)
            self.assertEqual(service.timeline_store.search_chunks("u1", "Jack", limit=5), [])
            self.assertIsNone(service.timeline_store.get_turn("u1", turn_id))
            self.assertEqual(
                service.timeline_store._conn.execute("SELECT COUNT(*) AS count FROM chunks WHERE id = ?", (chunk_id,)).fetchone()["count"],
                0,
            )

    def test_evidence_cleanup_policy_only_targets_timeline_chunks(self) -> None:
        counts = {
            "chunk_active": EvidenceReferenceCounts(active_refs=1, retained_refs=1),
            "chunk_retained": EvidenceReferenceCounts(active_refs=0, retained_refs=1),
            "chunk_orphan": EvidenceReferenceCounts(active_refs=0, retained_refs=0),
            "meeting_1": EvidenceReferenceCounts(active_refs=0, retained_refs=0),
        }

        self.assertEqual(
            timeline_chunk_evidence_ids(["chunk_active", "meeting_1", "chunk_orphan"]),
            ["chunk_active", "chunk_orphan"],
        )
        self.assertEqual(
            plan_timeline_evidence_cleanup(["chunk_active", "chunk_retained", "chunk_orphan", "meeting_1"], counts, hard_purge=True),
            {
                "soft_delete_chunk_ids": ["chunk_retained"],
                "purge_chunk_ids": ["chunk_orphan"],
                "retained_chunk_ids": ["chunk_active"],
            },
        )

    def test_hard_purge_soft_deletes_evidence_still_retained_by_inactive_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            timeline = service.timeline_store.add_turn("u1", "我叫 Jack，喜欢低糖拿铁")
            turn_id = timeline.turn.id
            chunk_id = timeline.chunks[0].id
            active = store.add_memory("u1", "我叫 Jack", kind="profile", evidence_ids=[chunk_id])
            stale = store.add_memory("u1", "用户曾经喜欢低糖拿铁", kind="profile", status="stale", evidence_ids=[chunk_id])

            first_purge = service.purge_memory(user_id="u1", memory_id=active.id)

            self.assertEqual(first_purge["purged_chunk_count"], 0)
            self.assertEqual(first_purge["soft_deleted_chunk_count"], 1)
            self.assertEqual(first_purge["retained_evidence_count"], 1)
            self.assertEqual(service.timeline_store.search_chunks("u1", "Jack", limit=5), [])
            retained_chunks = service.timeline_store.list_chunks_by_ids("u1", [chunk_id], include_deleted=True)
            self.assertEqual(len(retained_chunks), 1)
            self.assertEqual(retained_chunks[0].status, "deleted")
            self.assertEqual(
                service.timeline_store._conn.execute("SELECT COUNT(*) AS count FROM raw_turns WHERE id = ?", (turn_id,)).fetchone()["count"],
                1,
            )

            second_purge = service.purge_memory(user_id="u1", memory_id=stale.id)

            self.assertEqual(second_purge["purged_chunk_count"], 1)
            self.assertEqual(second_purge["purged_parent_count"], 1)
            self.assertEqual(
                service.timeline_store._conn.execute("SELECT COUNT(*) AS count FROM chunks WHERE id = ?", (chunk_id,)).fetchone()["count"],
                0,
            )
            self.assertIsNone(service.timeline_store.get_turn("u1", turn_id))

    def test_evidence_reference_counts_include_retained_inactive_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            timeline = service.timeline_store.add_turn("u1", "我叫 Jack，喜欢低糖拿铁")
            chunk_id = timeline.chunks[0].id
            store.add_memory("u1", "我叫 Jack", kind="profile", evidence_ids=[chunk_id, "meeting_1"])
            store.add_memory("u1", "用户曾经喜欢低糖拿铁", kind="profile", status="stale", evidence_ids=[chunk_id])
            store.add_memory("u1", "用户旧偏好是靠窗", kind="profile", status="superseded", superseded_by="new", evidence_ids=[chunk_id])
            deleted = store.add_memory("u1", "应忽略的删除记忆", kind="profile", status="deleted", evidence_ids=[chunk_id])
            store.delete_memory("u1", deleted.id)

            counts = store.evidence_reference_counts("u1", [chunk_id, "meeting_1", "missing"])

            self.assertEqual(counts[chunk_id], EvidenceReferenceCounts(active_refs=1, retained_refs=3))
            self.assertEqual(counts["meeting_1"], EvidenceReferenceCounts(active_refs=1, retained_refs=1))
            self.assertEqual(counts["missing"], EvidenceReferenceCounts())

    def test_purge_document_removes_row_search_and_same_user_audit_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            timeline = service.timeline_store.add_turn("u1", "这段原文和文档删除无关")
            imported = service.import_memory_events(
                user_id="u1",
                text="# 南太行自驾旅游攻略\n红旗渠门票 80 元",
                source="markdown_upload",
                context="南太行自驾攻略.md",
            )
            document_id = imported["document"]["id"]
            with service.audit_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps({"user_id": "u2", "record_type": "shared_id", "document_id": document_id}) + "\n")
                file.write(json.dumps({"user_id": "u1", "record_type": "unrelated", "document_id": "other_doc"}) + "\n")
                file.write("{bad json\n")

            purge = service.purge_document(user_id="u1", document_id=document_id)

            self.assertEqual(purge["deleted"], True)
            self.assertEqual(purge["purged"], True)
            self.assertEqual(purge["purged_chunk_count"], 0)
            self.assertEqual(purge["purged_parent_count"], 0)
            self.assertGreaterEqual(purge["audit_records_removed"], 1)
            self.assertIsNone(store.get_document("u1", document_id))
            self.assertEqual(store.list_documents("u1"), [])
            self.assertEqual(store.search_documents("u1", "南太行"), [])
            self.assertEqual(
                store._conn.execute("SELECT COUNT(*) AS count FROM documents WHERE id = ?", (document_id,)).fetchone()["count"],
                0,
            )
            self.assertEqual(len(service.timeline_store.search_chunks("u1", "原文", limit=5)), 1)
            self.assertIsNotNone(service.timeline_store.get_turn("u1", timeline.turn.id))
            audit_text = service.audit_path.read_text(encoding="utf-8")
            self.assertIn('"user_id": "u2"', audit_text)
            self.assertIn(document_id, audit_text)
            self.assertIn("other_doc", audit_text)
            self.assertIn("{bad json", audit_text)
            remaining_u1 = service.read_audit_records(user_id="u1", limit=10)
            self.assertTrue(all(document_id not in json.dumps(record, ensure_ascii=False) for record in remaining_u1))

    def test_standard_library_purge_true_supported_for_memory_and_document(self) -> None:
        from ai_glasses_memory_assistant.server import GlassesHandler

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            result = service.chat("我叫 Jack", user_id="u1")
            memory_id = result["saved_memories"][0]["id"]
            imported = service.import_memory_events(
                user_id="u1",
                text="# 南太行自驾旅游攻略\n红旗渠门票 80 元",
                source="markdown_upload",
                context="南太行自驾攻略.md",
            )
            document_id = imported["document"]["id"]
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                memory_status, _, memory_payload = self.http_delete(
                    GlassesHandler,
                    f"/api/memories/{memory_id}?user_id=u1&purge=true",
                )
                document_status, _, document_payload = self.http_delete(
                    GlassesHandler,
                    f"/api/documents/{document_id}?user_id=u1&purge=true",
                )
            finally:
                GlassesHandler.service = original_service

            self.assertEqual(memory_status, 200)
            self.assertEqual(memory_payload["deleted"], True)
            self.assertEqual(memory_payload["purged"], True)
            self.assertEqual(memory_payload["purged_chunk_count"], 1)
            self.assertEqual(document_status, 200)
            self.assertEqual(document_payload["deleted"], True)
            self.assertEqual(document_payload["purged"], True)
            self.assertIsNone(store.get_memory("u1", memory_id))
            self.assertIsNone(store.get_document("u1", document_id))

    def test_frontend_exposes_hard_purge_with_confirmation(self) -> None:
        script = (PACKAGE_ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("彻底删除", script)
        self.assertIn("purge=true", script)
        self.assertIn("不可恢复", script)
        self.assertIn("audit", script)
        self.assertGreaterEqual(script.count("window.confirm"), 2)

    def test_import_memory_events_uses_unified_gate_and_traceability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.import_memory_events(
                user_id="u1",
                items=[
                    {
                        "content": "项目AI眼镜决定先补敏感信息门控",
                        "memory_type": "decision",
                        "source_id": "meeting_1",
                    },
                    {
                        "content": "我的银行卡号是 6222000011112222",
                        "memory_type": "fact",
                        "source_id": "meeting_2",
                    },
                ],
                source="meeting",
                confirm=False,
            )
            memories = store.list_memories("u1", kind="event")

            self.assertEqual(result["saved_count"], 1)
            self.assertEqual(result["pending_confirmation_count"], 1)
            self.assertEqual(memories[0].memory_type, "decision")
            self.assertEqual(memories[0].source_id, "meeting_1")
            self.assertEqual(result["pending_confirmation"][0]["privacy_level"], "requires_confirmation")
            self.assertIn("memory_kernel", result)
            self.assertEqual(result["source_trace"]["layer"], "structured_memory")
            self.assertEqual(memories[0].evidence_ids, ["meeting_1"])
            self.assertEqual(event_to_dict(memories[0])["source_trace"]["layer"], "structured_memory")

    def test_import_memory_events_preserves_explicit_memory_type_without_legacy_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.import_memory_events(
                user_id="u1",
                items=[{
                    "content": "项目AI眼镜决定先补敏感信息门控",
                    "memory_type": "decision",
                    "source_id": "meeting_explicit",
                    "reason": "structured_import",
                }],
                source="meeting",
                confirm=False,
            )
            records = service.read_audit_records(user_id="u1", limit=5)

            self.assertEqual(result["saved_count"], 1)
            self.assertEqual(result["saved_memories"][0]["memory_type"], "decision")
            self.assertEqual(result["saved_memories"][0]["kind"], "event")
            self.assertEqual(result["classification_decisions"], [])
            self.assertEqual(records[-1]["saved_memories"][0]["memory_type"], "decision")
            self.assertEqual(records[-1]["classification_decisions"], [])
            self.assertNotIn("legacy_phrase_classifier", json.dumps(result, ensure_ascii=False))

    def test_import_memory_events_does_not_treat_bare_project_as_project_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.import_memory_events(
                user_id="u1",
                items=[{
                    "content": "AI 眼镜项目继续补 debug 可观测性",
                    "source_id": "meeting_implicit",
                }],
                source="meeting",
                confirm=False,
            )

            self.assertEqual(result["saved_count"], 1)
            self.assertEqual(result["saved_memories"][0]["memory_type"], "event")
            self.assertEqual(result["classification_decisions"][0]["role"], "legacy_fallback")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["source"], "legacy_default")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["reason"], "no_type_marker")
            self.assertNotIn("legacy_phrase_classifier", json.dumps(result, ensure_ascii=False))

    def test_import_memory_events_marks_project_state_legacy_type_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.import_memory_events(
                user_id="u1",
                items=[{
                    "content": "AI 眼镜项目风险是后台保存反馈不明显",
                    "source_id": "meeting_implicit",
                }],
                source="meeting",
                confirm=False,
            )
            candidate = service._candidate_from_import_item(
                {"content": "AI 眼镜项目风险是后台保存反馈不明显"},
                content="AI 眼镜项目风险是后台保存反馈不明显",
                kind="event",
                memory_type="project_state",
                source_id="meeting_implicit",
                ingestion_id="ing",
                source="meeting",
                classification_debug=[{
                    "field": "memory_type",
                    "value": "project_state",
                    "source": "legacy_phrase_classifier",
                    "role": "legacy_fallback",
                    "reason": "project_state_marker",
                    "marker": "风险",
                }],
            )

            self.assertEqual(result["saved_count"], 1)
            self.assertEqual(result["saved_memories"][0]["memory_type"], "project_state")
            self.assertEqual(result["classification_decisions"][0]["role"], "legacy_fallback")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["source"], "legacy_phrase_classifier")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["reason"], "project_state_marker")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["marker"], "风险")
            self.assertIn("memory_type=legacy_phrase_classifier:project_state_marker", candidate.reason)

    def test_import_memory_events_keeps_background_risk_as_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.import_memory_events(
                user_id="u1",
                items=[{
                    "content": "文档背景说明里提到风险管理的目的",
                    "source_id": "doc_background",
                }],
                source="meeting",
                confirm=False,
            )

            self.assertEqual(result["saved_count"], 1)
            self.assertEqual(result["saved_memories"][0]["memory_type"], "event")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["source"], "legacy_default")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["reason"], "no_type_marker")
            self.assertNotIn("legacy_phrase_classifier", json.dumps(result, ensure_ascii=False))

    def test_import_memory_events_does_not_classify_reminder_concept_question_as_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.import_memory_events(
                user_id="u1",
                items=[{
                    "content": "提醒是什么意思",
                    "source_id": "meeting_question",
                }],
                source="meeting",
                confirm=False,
            )

            self.assertEqual(result["saved_count"], 0)
            self.assertEqual(result["rejected_count"], 1)
            self.assertEqual(result["classification_decisions"][0]["role"], "legacy_fallback")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["source"], "legacy_default")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["reason"], "question_text_no_type_marker")
            self.assertNotIn("task_marker", json.dumps(result, ensure_ascii=False))

    def test_import_memory_events_keeps_contextual_reminder_as_task_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.import_memory_events(
                user_id="u1",
                items=[{
                    "content": "提醒我明天下午3点和 Mina 开会",
                    "source_id": "meeting_reminder",
                }],
                source="meeting",
                confirm=False,
            )

            self.assertEqual(result["saved_count"], 1)
            self.assertEqual(result["saved_memories"][0]["memory_type"], "task")
            self.assertEqual(result["classification_decisions"][0]["role"], "legacy_fallback")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["source"], "legacy_phrase_classifier")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["reason"], "task_marker")
            self.assertEqual(result["classification_decisions"][0]["decisions"][1]["marker"], "提醒")

    def test_import_memory_events_does_not_classify_identity_question_as_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.import_memory_events(
                user_id="u1",
                items=[{
                    "content": "我是什么身份",
                    "source_id": "meeting_identity_question",
                }],
                source="meeting",
                confirm=False,
            )

            self.assertEqual(result["saved_count"], 0)
            self.assertEqual(result["rejected_count"], 1)
            self.assertEqual(result["classification_decisions"][0]["role"], "legacy_fallback")
            self.assertEqual(result["classification_decisions"][0]["decisions"][0]["field"], "kind")
            self.assertEqual(result["classification_decisions"][0]["decisions"][0]["source"], "legacy_default")
            self.assertEqual(result["classification_decisions"][0]["decisions"][0]["reason"], "question_text_no_kind_marker")
            self.assertNotIn("profile_trigger", json.dumps(result, ensure_ascii=False))

    def test_markdown_upload_archives_document_without_line_memories(self) -> None:
        markdown = "\n".join([
            "# 南太行自驾旅游攻略",
            "## 路线总览",
            "Day 1  云台山",
            "## 实用贴士",
            "红旗渠门票 80 元",
        ])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.import_memory_events(
                user_id="u1",
                text=markdown,
                source="markdown_upload",
                context="南太行自驾攻略.md",
            )
            records = service.read_audit_records(user_id="u1", limit=5)

            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["saved_count"], 0)
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(len(store.list_documents("u1")), 1)
            self.assertEqual(result["document"]["filename"], "南太行自驾攻略.md")
            self.assertEqual(result["document"]["title"], "南太行自驾旅游攻略")
            self.assertIn("文档已整理", result["reply"])
            self.assertEqual(records[-1]["record_type"], "document_import")
            self.assertEqual(records[-1]["document"]["id"], result["document"]["id"])

    def test_document_upload_history_uses_local_metadata_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.import_memory_events(
                user_id="u1",
                text="# 南太行自驾旅游攻略\n红旗渠门票 80 元",
                source="markdown_upload",
                context="南太行自驾攻略.md",
            )

            result = service.chat("我什么时候上传过南太行自驾攻略文档？", user_id="u1")

            self.assertEqual(result["api_calls"], 0)
            self.assertEqual(service.fake_agent.main_calls, 0)
            self.assertIn("南太行自驾攻略.md", result["reply"])
            self.assertEqual(result["debug"]["document_recall"]["strategy"], "metadata")
            self.assertEqual(len(result["recalled_documents"]), 1)

    def test_document_detail_question_injects_source_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.import_memory_events(
                user_id="u1",
                text="# 南太行自驾旅游攻略\n## 实用贴士\n红旗渠门票 80 元",
                source="markdown_upload",
                context="南太行自驾攻略.md",
            )

            result = service.chat("南太行那份攻略里红旗渠门票多少钱？", user_id="u1")

            self.assertEqual(service.fake_agent.main_calls, 1)
            self.assertEqual(result["debug"]["document_recall"]["strategy"], "full_document")
            self.assertEqual(len(result["recalled_documents"]), 1)
            self.assertIn("<document-context>", service.fake_agent.main_messages[-1])
            self.assertIn("红旗渠门票 80 元", service.fake_agent.main_messages[-1])
            self.assertIn("Do not answer document details from summary alone", service.fake_agent.main_messages[-1])

    def test_document_title_query_injects_source_markdown_without_file_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪\n## 一句话总结\n理性看待消费选择",
                source="markdown_upload",
                context="外卖小哥买奥迪.md",
            )

            result = service.chat("外卖小哥买奥迪", user_id="u1")

            self.assertEqual(service.fake_agent.main_calls, 1)
            self.assertEqual(result["debug"]["document_recall"]["strategy"], "full_document")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "document_title_match")
            self.assertEqual(result["recalled_documents"][0]["filename"], "外卖小哥买奥迪.md")
            self.assertIn("<document-context>", service.fake_agent.main_messages[-1])
            self.assertIn("理性看待消费选择", service.fake_agent.main_messages[-1])

    def test_document_title_query_matches_space_separated_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪\n## 一句话总结\n保留现金流",
                source="markdown_upload",
                context="外卖小哥买奥迪.md",
            )

            result = service.chat("外卖 奥迪", user_id="u1")

            self.assertEqual(result["debug"]["document_recall"]["reason"], "document_title_match")
            self.assertEqual(result["recalled_documents"][0]["filename"], "外卖小哥买奥迪.md")
            self.assertIn("保留现金流", service.fake_agent.main_messages[-1])

    def test_recent_document_reference_uses_latest_document_for_overview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_document(
                "u1",
                filename="旧攻略.md",
                title="旧攻略",
                summary="旧路线",
                content="# 旧攻略\n老内容",
                source="markdown_upload",
                ingestion_id="doc_old",
                created_at=1000.0,
            )
            store.add_document(
                "u1",
                filename="新攻略.md",
                title="新攻略",
                summary="新路线",
                content="# 新攻略\n新内容",
                source="markdown_upload",
                ingestion_id="doc_new",
                created_at=2000.0,
            )

            result = service.chat("刚刚那份文档讲了什么？", user_id="u1")

            self.assertEqual(result["api_calls"], 0)
            self.assertEqual(service.fake_agent.main_calls, 0)
            self.assertEqual(result["debug"]["document_recall"]["reason"], "recent_document_reference")
            self.assertEqual(result["recalled_documents"][0]["filename"], "新攻略.md")
            self.assertIn("新攻略", result["reply"])
            self.assertNotIn("旧攻略", result["reply"])

    def test_recent_document_type_reference_prefers_latest_matching_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_document(
                "u1",
                filename="项目周报.md",
                title="AI眼镜项目周报",
                summary="周报总结",
                content="# AI眼镜项目周报\n## 风险\n提醒 runtime 还没对齐",
                source="markdown_upload",
                ingestion_id="weekly",
                created_at=2000.0,
            )
            store.add_document(
                "u1",
                filename="项目日报.md",
                title="AI眼镜项目日报",
                summary="日报总结",
                content="# AI眼镜项目日报\n日报内容",
                source="markdown_upload",
                ingestion_id="daily",
                created_at=3000.0,
            )

            result = service.chat("上一份周报里有什么风险？", user_id="u1")

            self.assertEqual(result["debug"]["document_recall"]["reason"], "recent_document_type_reference")
            self.assertEqual(result["debug"]["document_recall"]["phrase_policy"]["role"], "retrieval_hint")
            self.assertEqual(result["debug"]["document_recall"]["phrase_policy"]["treatment"], "select_recent_document_candidate")
            self.assertEqual(result["debug"]["document_recall"]["phrase_policy"]["type_markers"], ["周报"])
            self.assertIn("上一份", result["debug"]["document_recall"]["phrase_policy"]["reference_markers"])
            self.assertEqual(result["recalled_documents"][0]["filename"], "项目周报.md")
            self.assertIn("提醒 runtime 还没对齐", service.fake_agent.main_messages[-1])
            self.assertNotIn("日报内容", service.fake_agent.main_messages[-1])

    def test_recent_document_type_reference_ignores_deleted_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            document = store.add_document(
                "u1",
                filename="项目周报.md",
                title="AI眼镜项目周报",
                summary="周报总结",
                content="# AI眼镜项目周报\n已删除内容",
                source="markdown_upload",
                ingestion_id="weekly",
                created_at=2000.0,
            )

            deleted = store.delete_document("u1", document.id)
            result = service.chat("上一份周报里有什么风险？", user_id="u1")

            self.assertTrue(deleted)
            self.assertEqual(result["debug"]["document_recall"]["reason"], "recent_document_type_reference")
            self.assertEqual(result["recalled_documents"], [])

    def test_weekly_report_generation_does_not_trigger_document_recall_by_type_word(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_document(
                "u1",
                filename="旧项目周报.md",
                title="旧项目周报",
                summary="旧周报摘要",
                content="# 旧项目周报\n不应该被这次生成请求召回",
                source="markdown_upload",
                ingestion_id="weekly",
                created_at=2000.0,
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目：Alex 负责补 eval 报告模板",
                kind="event",
                memory_type="task",
                tags=["project:AI眼镜", "task_status:open"],
            )

            result = service.chat("帮我写一份周报", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["conversation_action"], "weekly_report")
            self.assertEqual(result["debug"]["document_recall"]["strategy"], "skipped")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "not_document_query")
            self.assertEqual(result["debug"]["document_recall"]["phrase_policy"]["role"], "weak_signal")
            self.assertEqual(result["debug"]["document_recall"]["phrase_policy"]["treatment"], "ignored_without_document_context")
            self.assertEqual(result["debug"]["document_recall"]["phrase_policy"]["type_markers"], ["周报"])
            self.assertEqual(result["recalled_documents"], result["weekly_report"]["documents"])
            self.assertNotIn("不应该被这次生成请求召回", result["reply"])

    def test_weekly_report_knowledge_question_stays_on_llm_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("周报是什么？", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["reply_mode"], "llm")
            self.assertEqual(result["debug"]["planner"]["conversation_action"], "")
            self.assertNotIn("weekly_report", result)
            self.assertEqual(service.fake_agent.main_calls, 1)

    def test_attention_concept_question_stays_on_llm_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            for message in ("风险是什么？", "项目风险是什么？"):
                with self.subTest(message=message):
                    result = service.chat(message, user_id="u1")

                    self.assertEqual(result["debug"]["planner"]["reply_mode"], "llm")
                    self.assertEqual(result["debug"]["planner"]["conversation_action"], "")
                    self.assertNotIn("conversation_action", result["debug"])
            self.assertEqual(service.fake_agent.main_calls, 2)

    def test_recent_plain_query_does_not_trigger_document_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_document(
                "u1",
                filename="项目周报.md",
                title="AI眼镜项目周报",
                summary="周报总结",
                content="# AI眼镜项目周报\n文档内容",
                source="markdown_upload",
                ingestion_id="weekly",
                created_at=2000.0,
            )

            result = service.chat("最近我在忙什么？", user_id="u1")

            self.assertEqual(result["debug"]["document_recall"]["reason"], "not_document_query")
            self.assertEqual(result["recalled_documents"], [])

    def test_plain_query_does_not_recall_document_by_content_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪\n## 一句话总结\n理性看待消费选择",
                source="markdown_upload",
                context="外卖小哥买奥迪.md",
            )

            result = service.chat("消费选择", user_id="u1")

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "skipped")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "not_document_query")
            self.assertEqual(result["recalled_documents"], [])

    def test_ambiguous_document_title_query_uses_metadata_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪\n独立正文 A",
                source="markdown_upload",
                context="外卖小哥买奥迪.md",
            )
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪复盘\n独立正文 B",
                source="markdown_upload",
                context="外卖小哥买奥迪复盘.md",
            )

            result = service.chat("外卖小哥买奥迪", user_id="u1")

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "metadata")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "ambiguous_document_title_match")
            self.assertEqual(len(result["recalled_documents"]), 2)
            self.assertIn("Archived user documents", service.fake_agent.main_messages[-1])
            self.assertNotIn("独立正文 A", service.fake_agent.main_messages[-1])
            self.assertNotIn("独立正文 B", service.fake_agent.main_messages[-1])

    def test_cross_document_compare_query_uses_compare_metadata_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "none",
                "recall_goal": "none",
                "confidence": 0.95,
                "reason": "cross document comparison",
            })
            service = FakeService(store, agent=agent)
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪\n## 一句话总结\n理性看待消费选择\n## 重点\n保留现金流。",
                source="markdown_upload",
                context="外卖小哥买奥迪.md",
            )
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪复盘\n## 一句话总结\n关注负债节奏\n## 重点\n先把每月支出打平。",
                source="markdown_upload",
                context="外卖小哥买奥迪复盘.md",
            )

            result = service.chat("对比一下这两份外卖小哥买奥迪文档有什么不同？", user_id="u1", routing_mode="llm_first")
            message = agent.main_messages[-1]

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "metadata")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "cross_document_compare_query")
            self.assertIn("Cross-document comparison candidates", message)
            self.assertIn("high_level_excerpt:", message)
            self.assertIn("理性看待消费选择", message)
            self.assertIn("关注负债节奏", message)
            self.assertNotIn("保留现金流", message)
            self.assertNotIn("每月支出打平", message)

    def test_cross_document_compare_risk_query_prefers_risk_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "none",
                "recall_goal": "none",
                "confidence": 0.95,
                "reason": "cross document risk comparison",
            })
            service = FakeService(store, agent=agent)
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪\n## 风险\n月供压力偏高。",
                source="markdown_upload",
                context="外卖小哥买奥迪.md",
            )
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪复盘\n## 风险\n负债节奏容易失控。",
                source="markdown_upload",
                context="外卖小哥买奥迪复盘.md",
            )

            result = service.chat("对比一下这两份外卖小哥买奥迪文档的风险差异。", user_id="u1", routing_mode="llm_first")
            message = agent.main_messages[-1]

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "metadata")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "cross_document_compare_query")
            self.assertIn("high_level_excerpt: 月供压力偏高。", message)
            self.assertIn("high_level_excerpt: 负债节奏容易失控。", message)

    def test_cross_document_compare_advice_query_prefers_advice_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "none",
                "recall_goal": "none",
                "confidence": 0.95,
                "reason": "cross document advice comparison",
            })
            service = FakeService(store, agent=agent)
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪\n## 风险\n月供压力偏高。\n## 建议\n保留现金流。",
                source="markdown_upload",
                context="外卖小哥买奥迪.md",
            )
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪复盘\n## 风险\n负债节奏容易失控。\n## 建议\n先把每月支出打平。",
                source="markdown_upload",
                context="外卖小哥买奥迪复盘.md",
            )

            result = service.chat("对比一下这两份外卖小哥买奥迪文档的建议差异。", user_id="u1", routing_mode="llm_first")
            message = agent.main_messages[-1]

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "metadata")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "cross_document_compare_query")
            self.assertIn("high_level_excerpt: 保留现金流。", message)
            self.assertIn("high_level_excerpt: 先把每月支出打平。", message)
            self.assertNotIn("high_level_excerpt: 月供压力偏高。", message)
            self.assertNotIn("high_level_excerpt: 负债节奏容易失控。", message)

    def test_cross_document_compare_conclusion_query_prefers_conclusion_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "none",
                "recall_goal": "none",
                "confidence": 0.95,
                "reason": "cross document conclusion comparison",
            })
            service = FakeService(store, agent=agent)
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪\n## 结论\n先稳住现金流再考虑升级消费。\n## 建议\n保留现金流。",
                source="markdown_upload",
                context="外卖小哥买奥迪.md",
            )
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪复盘\n## 结论\n先把负债节奏拉回可控区间。\n## 建议\n先把每月支出打平。",
                source="markdown_upload",
                context="外卖小哥买奥迪复盘.md",
            )

            result = service.chat("对比一下这两份外卖小哥买奥迪文档的结论差异。", user_id="u1", routing_mode="llm_first")
            message = agent.main_messages[-1]

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "metadata")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "cross_document_compare_query")
            self.assertIn("high_level_excerpt: 先稳住现金流再考虑升级消费。", message)
            self.assertIn("high_level_excerpt: 先把负债节奏拉回可控区间。", message)
            self.assertNotIn("high_level_excerpt: 保留现金流。", message)
            self.assertNotIn("high_level_excerpt: 先把每月支出打平。", message)

    def test_cross_document_compare_applicability_query_prefers_applicability_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "none",
                "recall_goal": "none",
                "confidence": 0.95,
                "reason": "cross document applicability comparison",
            })
            service = FakeService(store, agent=agent)
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪\n## 适用场景\n更适合现金流稳定、短期没有大额支出的人。\n## 建议\n保留现金流。",
                source="markdown_upload",
                context="外卖小哥买奥迪.md",
            )
            service.import_memory_events(
                user_id="u1",
                text="# 外卖小哥买奥迪复盘\n## 适用场景\n更适合已经进入负债调整期、需要先稳住月供节奏的人。\n## 建议\n先把每月支出打平。",
                source="markdown_upload",
                context="外卖小哥买奥迪复盘.md",
            )

            result = service.chat("对比一下这两份外卖小哥买奥迪文档的适用场景差异。", user_id="u1", routing_mode="llm_first")
            message = agent.main_messages[-1]

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "metadata")
            self.assertEqual(result["debug"]["document_recall"]["reason"], "cross_document_compare_query")
            self.assertIn("high_level_excerpt: 更适合现金流稳定、短期没有大额支出的人。", message)
            self.assertIn("high_level_excerpt: 更适合已经进入负债调整期、需要先稳住月供节奏的人。", message)
            self.assertNotIn("high_level_excerpt: 保留现金流。", message)
            self.assertNotIn("high_level_excerpt: 先把每月支出打平。", message)

    def test_memory_api_lists_document_metadata_without_content(self) -> None:
        from ai_glasses_memory_assistant.server import GlassesHandler

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            imported = service.import_memory_events(
                user_id="u1",
                text="# 南太行自驾旅游攻略\n红旗渠门票 80 元",
                source="markdown_upload",
                context="南太行自驾攻略.md",
            )
            original_service = GlassesHandler.service
            GlassesHandler.service = service
            try:
                status, _, payload = self.http_get(GlassesHandler, "/api/memories?user_id=u1")
                document_status, _, document_payload = self.http_get(
                    GlassesHandler,
                    f"/api/documents/{imported['document']['id']}?user_id=u1",
                )
            finally:
                GlassesHandler.service = original_service

            self.assertEqual(status, 200)
            documents = payload["documents"]
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["filename"], "南太行自驾攻略.md")
            self.assertNotIn("content", documents[0])
            self.assertEqual(document_status, 200)
            self.assertIn("红旗渠门票 80 元", document_payload["document"]["content"])

    def test_document_edit_updates_recalled_source_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            imported = service.import_memory_events(
                user_id="u1",
                text="# 南太行自驾旅游攻略\n## 实用贴士\n红旗渠门票 80 元",
                source="markdown_upload",
                context="南太行自驾攻略.md",
            )
            original_hash = imported["document"]["content_hash"]
            updated = store.update_document(
                "u1",
                imported["document"]["id"],
                filename="南太行新版攻略.md",
                title="南太行新版攻略",
                summary="新版门票信息",
                content="# 南太行新版攻略\n## 实用贴士\n红旗渠门票 90 元",
            )

            result = service.chat("南太行新版攻略里红旗渠门票多少钱？", user_id="u1")

            self.assertIsNotNone(updated)
            self.assertNotEqual(updated.content_hash, original_hash)
            self.assertIn("红旗渠门票 90 元", service.fake_agent.main_messages[-1])
            self.assertNotIn("红旗渠门票 80 元", service.fake_agent.main_messages[-1])
            self.assertEqual(result["recalled_documents"][0]["filename"], "南太行新版攻略.md")

    def test_document_delete_removes_document_from_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            imported = service.import_memory_events(
                user_id="u1",
                text="# 南太行自驾旅游攻略\n红旗渠门票 80 元",
                source="markdown_upload",
                context="南太行自驾攻略.md",
            )

            deleted = store.delete_document("u1", imported["document"]["id"])
            result = service.chat("我什么时候上传过南太行自驾攻略文档？", user_id="u1")

            self.assertTrue(deleted)
            self.assertEqual(store.list_documents("u1"), [])
            self.assertEqual(store.search_documents("u1", "南太行"), [])
            self.assertEqual(result["recalled_documents"], [])
            self.assertEqual(result["debug"]["document_recall"]["strategy"], "none")

    def test_document_overview_uses_metadata_without_main_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.import_memory_events(
                user_id="u1",
                text="# 南太行自驾旅游攻略\n## 路线总览\n## 实用贴士",
                source="markdown_upload",
                context="南太行自驾攻略.md",
            )

            result = service.chat("南太行那份文档讲了什么？", user_id="u1")

            self.assertEqual(result["api_calls"], 0)
            self.assertEqual(service.fake_agent.main_calls, 0)
            self.assertIn("南太行自驾旅游攻略", result["reply"])
            self.assertIn("路线总览", result["reply"])

    def test_continuous_capture_imports_chunks_after_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            started = service.start_capture(user_id="u1", context="周会")
            service.append_capture_chunk(
                user_id="u1",
                capture_id=started["capture_id"],
                text="项目AI眼镜小王负责 continuous_capture",
            )
            stopped = service.stop_capture(user_id="u1", capture_id=started["capture_id"])
            timeline_chunks = service.timeline_store.search_chunks("u1", "continuous_capture", limit=5)

            self.assertEqual(stopped["status"], "stopped")
            self.assertEqual(stopped["chunk_count"], 1)
            self.assertEqual(stopped["import_result"]["saved_count"], 1)
            self.assertEqual(len(store.list_memories("u1", kind="event")), 1)
            self.assertEqual(len(timeline_chunks), 1)
            self.assertEqual(timeline_chunks[0].parent_type, "capture")
            self.assertEqual(started["source_trace"]["layer"], "raw_timeline")
            self.assertEqual(stopped["source_trace"]["evidence_ids"], [timeline_chunks[0].id])

    def test_speaker_labeled_transcript_parses_conversation_session(self) -> None:
        transcript = "\n".join([
            "[09:31][用户] 下午三点我们去见客户。",
            "[09:32][张三] 我带合同，你带方案。",
            "[09:33][speaker_2] 报价别超过上次那版。",
            "[用户] 我来准备客户方案。",
            "李四：我负责报价。",
            "Alex: 我准备会议纪要。",
        ])

        session = GlassesChatService._parse_speaker_labeled_transcript(transcript)

        self.assertIsInstance(session, ConversationSession)
        self.assertEqual(len(session.turns), 6)
        self.assertIsInstance(session.turns[0], ConversationTurn)
        self.assertEqual(session.turns[0].speaker_role, "user")
        self.assertEqual(session.turns[1].speaker_label, "张三")
        self.assertEqual(session.turns[1].speaker_role, "known_person")
        self.assertEqual(session.turns[2].speaker_role, "unknown_speaker")
        self.assertEqual(session.turns[3].speaker_label, "用户")
        self.assertEqual(session.turns[3].timestamp_text, "")
        self.assertEqual(session.turns[4].speaker_label, "李四")
        self.assertEqual(session.turns[5].speaker_label, "Alex")
        self.assertEqual(session.participants["用户"], "user")
        self.assertEqual(session.participants["张三"], "known_person")
        self.assertEqual(session.participants["speaker_2"], "unknown_speaker")
        self.assertEqual(session.participants["李四"], "known_person")
        self.assertEqual(session.participants["Alex"], "known_person")

    def test_import_speaker_labeled_transcript_saves_user_centered_assignment_only(self) -> None:
        transcript = "\n".join([
            "[09:31][用户] 下午三点我们去见客户。",
            "[09:32][张三] 我带合同，你带方案。",
            "[09:33][用户] 可以，那我负责 PPT。",
            "[09:34][李四] 报价别超过上次那版。",
            "[09:35][Bob] 我最近喜欢喝冰美式。",
            "[09:36][speaker_2] 验证码是 482931。",
        ])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(
                store,
                agent=FakeAgent(
                    pre_reply_payload={
                        "我和张三上次聊客户时怎么分工的？": {
                            "reply_mode": "llm",
                            "answer_source": "llm",
                            "scope": "unknown",
                            "location_text": "",
                            "needs_event_memory": True,
                            "memory_recall_type": "event",
                            "recall_goal": "specific_fact",
                            "event_recall_strategy": "text_search",
                            "confidence": 0.95,
                            "reason": "asks participant assignment",
                        },
                        "*": {
                            "reply_mode": "llm",
                            "answer_source": "llm",
                            "scope": "unknown",
                            "location_text": "",
                            "needs_event_memory": False,
                            "memory_recall_type": "none",
                            "confidence": 0.95,
                            "reason": "not a recall turn",
                        },
                    },
                    temporal_payload={
                        "has_temporal_expression": False,
                        "temporal_text": "",
                        "kind": "none",
                        "confidence": 0.2,
                        "reason": "上次 is not resolved in this fixture",
                    },
                ),
            )

            imported = service.import_memory_events(
                user_id="u1",
                text=transcript,
                source="multi_speaker_transcript",
                context="客户拜访待机转写",
            )
            memories = store.list_memories("u1", kind="event")
            saved_text = "\n".join(memory.content for memory in memories)
            recall = service.chat("我和张三上次聊客户时怎么分工的？", user_id="u1")
            recalled_text = "\n".join(memory["content"] for memory in recall["recalled_memories"])

            self.assertEqual(imported["saved_count"], 2)
            self.assertEqual(imported["conversation_session"]["detected"], True)
            self.assertEqual(imported["conversation_session"]["turn_count"], 6)
            self.assertEqual(imported["conversation_session"]["participants"]["张三"], "known_person")
            self.assertEqual(imported["conversation_session"]["participants"]["speaker_2"], "unknown_speaker")
            self.assertIn("张三", saved_text)
            self.assertIn("合同", saved_text)
            self.assertIn("PPT", saved_text)
            self.assertIn("李四", saved_text)
            self.assertIn("报价", saved_text)
            self.assertIn("客户", saved_text)
            self.assertNotIn("冰美式", saved_text)
            self.assertNotIn("482931", saved_text)
            self.assertIn("candidate_facts", imported["conversation_session"])
            self.assertIn("third_party_preference", json.dumps(imported["conversation_session"], ensure_ascii=False))
            self.assertIn("sensitive_fragment_filtered", json.dumps(imported["conversation_session"], ensure_ascii=False))
            self.assertEqual(imported["conversation_session"]["parsed_turns"][5]["speaker_role"], "unknown_speaker")
            self.assertIn("张三", recalled_text)
            self.assertIn("合同", recalled_text)
            self.assertIn("PPT", recalled_text)
            self.assertEqual(recall["source_summary"]["primary_source"], "structured_memory")

    def test_import_speaker_labeled_transcript_without_user_does_not_fallback_to_plain_import(self) -> None:
        transcript = "\n".join([
            "张三：我负责客户合同。",
            "李四：我准备报价。",
        ])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            imported = service.import_memory_events(
                user_id="u1",
                text=transcript,
                source="multi_speaker_transcript",
                context="旁人对话",
            )

            self.assertEqual(imported["saved_count"], 0)
            self.assertEqual(imported["candidate_count"], 0)
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(imported["conversation_session"]["detected"], True)
            self.assertEqual(imported["conversation_session"]["rejected_turns"][0]["reason"], "no_user_turn")
            self.assertEqual(imported["classification_decisions"], [])

    def test_import_unknown_speaker_task_is_not_saved_as_named_assignment(self) -> None:
        transcript = "\n".join([
            "[用户] 下午三点我们去见客户。",
            "[speaker_2] 我带合同。",
        ])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            imported = service.import_memory_events(
                user_id="u1",
                text=transcript,
                source="multi_speaker_transcript",
                context="未知说话人分工",
            )

            self.assertEqual(imported["saved_count"], 0)
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(imported["conversation_session"]["participants"]["speaker_2"], "unknown_speaker")
            self.assertIn("unknown_speaker_not_saved_as_memory_fact", json.dumps(imported["conversation_session"], ensure_ascii=False))

    def test_import_speaker_labeled_transcript_filters_sensitive_turn_but_keeps_safe_assignment(self) -> None:
        transcript = "\n".join([
            "用户：下午三点我们去见客户。",
            "张三: 我带合同，验证码是 482931。",
            "用户：我负责 PPT。",
        ])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            imported = service.import_memory_events(
                user_id="u1",
                text=transcript,
                source="multi_speaker_transcript",
                context="敏感夹杂分工",
            )
            saved_text = "\n".join(memory.content for memory in store.list_memories("u1"))

            self.assertEqual(imported["saved_count"], 1)
            self.assertIn("张三", saved_text)
            self.assertIn("合同", saved_text)
            self.assertIn("用户", saved_text)
            self.assertIn("PPT", saved_text)
            self.assertNotIn("482931", saved_text)
            self.assertIn("sensitive_fragment_filtered", json.dumps(imported["conversation_session"], ensure_ascii=False))
            self.assertEqual(imported["conversation_session"]["candidate_turn_indices"], [1, 2])

    def test_import_multi_person_multi_task_keeps_assignments_and_rejects_private_preference(self) -> None:
        transcript = "\n".join([
            "用户：明天我们去客户现场。",
            "张三：我带合同。",
            "李四：我负责报价。",
            "用户：我准备 PPT。",
            "王五：我不喜欢喝咖啡。",
        ])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            imported = service.import_memory_events(
                user_id="u1",
                text=transcript,
                source="multi_speaker_transcript",
                context="多参与人多任务",
            )
            saved_text = "\n".join(memory.content for memory in store.list_memories("u1"))

            self.assertEqual(imported["saved_count"], 2)
            self.assertIn("张三", saved_text)
            self.assertIn("合同", saved_text)
            self.assertIn("李四", saved_text)
            self.assertIn("报价", saved_text)
            self.assertIn("用户", saved_text)
            self.assertIn("PPT", saved_text)
            self.assertNotIn("咖啡", saved_text)
            self.assertIn("third_party_preference", json.dumps(imported["conversation_session"], ensure_ascii=False))
            self.assertEqual(imported["conversation_session"]["candidate_turn_indices"], [1, 2])

    def test_import_speaker_alias_names_unknown_speaker_for_assignment(self) -> None:
        transcript = "\n".join([
            "[09:31][用户] 下午三点我们去见客户。",
            "[09:32][speaker_2] 我带报价单。",
            "[09:33][用户] 刚才 speaker_2 是李四。",
            "[09:34][用户] 我准备 PPT。",
        ])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            imported = service.import_memory_events(
                user_id="u1",
                text=transcript,
                source="multi_speaker_transcript",
                context="speaker alias 分工",
            )
            saved_text = "\n".join(memory.content for memory in store.list_memories("u1"))
            conversation_debug = imported["conversation_session"]

            self.assertEqual(imported["saved_count"], 1)
            self.assertIn("李四", saved_text)
            self.assertIn("报价单", saved_text)
            self.assertIn("PPT", saved_text)
            self.assertNotIn("speaker_2", saved_text)
            self.assertEqual(conversation_debug["participants"]["speaker_2"], "known_person")
            self.assertEqual(conversation_debug["speaker_aliases"][0]["source_label"], "speaker_2")
            self.assertEqual(conversation_debug["speaker_aliases"][0]["target_label"], "李四")
            self.assertEqual(conversation_debug["speaker_aliases"][0]["alias_source"], "user_named_speaker")
            self.assertIn({"turn_index": 1, "from": "speaker_2", "to": "李四"}, conversation_debug["alias_applied_turns"])

    def test_import_speaker_alias_does_not_save_private_preference_or_sensitive_fragment(self) -> None:
        transcript = "\n".join([
            "用户：下午三点我们去见客户。",
            "speaker_2: 我喜欢喝冰美式。",
            "用户：刚才 speaker_2 是李四。",
            "李四：验证码是 482931，我负责报价。",
            "用户：我准备 PPT。",
        ])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            imported = service.import_memory_events(
                user_id="u1",
                text=transcript,
                source="multi_speaker_transcript",
                context="speaker alias 隐私边界",
            )
            saved_text = "\n".join(memory.content for memory in store.list_memories("u1"))
            conversation_json = json.dumps(imported["conversation_session"], ensure_ascii=False)

            self.assertEqual(imported["saved_count"], 1)
            self.assertIn("李四", saved_text)
            self.assertIn("报价", saved_text)
            self.assertIn("用户", saved_text)
            self.assertIn("PPT", saved_text)
            self.assertNotIn("冰美式", saved_text)
            self.assertNotIn("482931", saved_text)
            self.assertIn("third_party_preference", conversation_json)
            self.assertIn("sensitive_fragment_filtered", conversation_json)

    def test_import_multi_task_splits_candidates_and_recalls_quote_owner(self) -> None:
        transcript = "\n".join([
            "[09:31][用户] 下午三点我们去见客户。",
            "[09:32][张三] 我带合同，你带方案。",
            "[09:33][用户] 可以，那我负责 PPT。",
            "[09:34][李四] 报价别超过上次那版，我来确认。",
        ])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(
                store,
                agent=FakeAgent(
                    pre_reply_payload={
                        "上次客户拜访谁负责报价？": {
                            "reply_mode": "llm",
                            "answer_source": "llm",
                            "scope": "unknown",
                            "needs_event_memory": True,
                            "memory_recall_type": "event",
                            "recall_goal": "specific_fact",
                            "event_recall_strategy": "text_search",
                            "confidence": 0.95,
                            "reason": "asks quote owner",
                        },
                        "*": {
                            "reply_mode": "llm",
                            "answer_source": "llm",
                            "scope": "unknown",
                            "needs_event_memory": False,
                            "memory_recall_type": "none",
                            "confidence": 0.95,
                            "reason": "not a recall turn",
                        },
                    },
                    temporal_payload={
                        "has_temporal_expression": False,
                        "temporal_text": "",
                        "kind": "none",
                        "confidence": 0.2,
                        "reason": "上次不在本 fixture 解析为具体时间",
                    },
                ),
            )

            imported = service.import_memory_events(
                user_id="u1",
                text=transcript,
                source="multi_speaker_transcript",
                context="多人任务拆分",
            )
            memories = store.list_memories("u1", kind="event")
            saved_text = "\n".join(memory.content for memory in memories)
            recall = service.chat("上次客户拜访谁负责报价？", user_id="u1")
            recalled_text = "\n".join(memory["content"] for memory in recall["recalled_memories"])

            self.assertEqual(imported["saved_count"], 2)
            self.assertEqual(len(memories), 2)
            self.assertIn("用户和张三的多人协作事项", saved_text)
            self.assertIn("张三", saved_text)
            self.assertIn("合同", saved_text)
            self.assertIn("PPT", saved_text)
            self.assertIn("用户和李四的多人协作事项", saved_text)
            self.assertIn("李四", saved_text)
            self.assertIn("报价", saved_text)
            self.assertIn("candidate_facts", imported["conversation_session"])
            self.assertIn("saved_candidates", imported["conversation_session"])
            self.assertIn("李四", recalled_text)
            self.assertIn("报价", recalled_text)
            self.assertEqual(recall["source_summary"]["primary_source"], "structured_memory")

    def test_chat_long_low_value_chatter_keeps_timeline_without_memory_pollution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(intent_payload={
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [],
                "confidence": 0.9,
            }, pre_reply_payload={
                "*": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_action": "none",
                    "memory_kind": "none",
                    "memory_type": "none",
                    "candidate_content": "",
                    "confidence": 0.9,
                    "reason": "fake no segment candidates",
                },
            }))
            message = (
                "今天我和朋友一路闲聊，先说早上空气不错，然后聊到路边咖啡店的音乐，"
                "接着又说昨晚那个电视剧节奏有点慢。后来我们又随口聊了几句天气、"
                "地铁人多、晚上可能早点休息，没有明确要记的安排，也没有稳定偏好。"
            )

            result = service.chat(message, user_id="u1", defer_memory_writes=True)
            job_id = result["debug"]["memory_processing"]["job_id"]
            job = service.read_memory_job(user_id="u1", job_id=job_id)
            chunks = service.timeline_store.search_chunks("u1", "电视剧", limit=5)

            self.assertEqual(result["api_calls"], 0)
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "continuous_capture")
            self.assertEqual(result["debug"]["memory_processing"]["mode"], "continuous_capture")
            self.assertEqual(job["status"], "skipped")
            self.assertIn(job["memory_processing"]["stage"], {"semantic_cleaning", "memory_extraction"})
            self.assertIn(
                job["memory_processing"]["stage_reason"],
                {"semantic_cleaner_rejected_all_segments", "extractor_produced_no_candidates"},
            )
            self.assertTrue(job["memory_processing"]["stage_explanation"])
            self.assertEqual(job["saved_count"], 0)
            self.assertEqual(store.list_memories("u1"), [])
            self.assertGreaterEqual(len(chunks), 1)

    def test_chat_long_input_extracts_multiple_memories_with_evidence(self) -> None:
        intent_payload = {
            "needs_web_search": False,
            "web_query": None,
            "web_reason": "",
            "memory_write_candidates": [
                {
                    "content": "AI 眼镜 demo 决定先做可观测 debug，再做真实设备适配",
                    "kind": "event",
                    "memory_type": "decision",
                    "confidence": 0.9,
                    "reason": "long_input_decision",
                },
                {
                    "content": "Mia 负责前端语音按钮和播报状态",
                    "kind": "event",
                    "memory_type": "task",
                    "confidence": 0.89,
                    "reason": "long_input_task",
                },
            ],
            "confidence": 0.92,
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(intent_payload=intent_payload))
            message = (
                "今天下午我口述一下项目状态：AI 眼镜 demo 现在要把长语音转写接进记忆管道。"
                "Mia 负责前端语音按钮和播报状态，Alex 负责后端回复优先和后台记忆写入。"
                "我们决定先做可观测 debug，再做真实设备适配。风险是浏览器语音权限不稳定，"
                "以及后台写入失败时用户不知道。周五下午 3 点前先交第一版。"
            )

            result = service.chat(message, user_id="u1", defer_memory_writes=True)
            job_id = result["debug"]["memory_processing"]["job_id"]
            job = service.read_memory_job(user_id="u1", job_id=job_id)
            memories = store.list_memories("u1", kind="event")

            self.assertEqual(result["debug"]["planner"]["reply_mode"], "continuous_capture")
            self.assertEqual(job["status"], "saved")
            self.assertGreaterEqual(job["saved_count"], 2)
            self.assertIn("semantic_cleaner+llm_segmented", job["extraction_backend"])
            self.assertEqual(service.fake_agent.intent_calls, 0)
            self.assertEqual(service.fake_agent.semantic_calls, job["extraction_trace"]["llm_segment_count"])
            self.assertEqual(service.fake_agent.segment_calls, job["extraction_trace"]["segment_count"])
            self.assertGreater(job["extraction_trace"]["rule_fallback"]["candidate_count"], 0)
            self.assertGreaterEqual(len(memories), 2)
            self.assertTrue(all(memory.evidence_ids for memory in memories))
            self.assertEqual(set(job["evidence_ids"]), set(result["debug"]["timeline"]["chunk_ids"]))

    def test_chat_long_input_uses_rules_only_after_llm_returns_no_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [],
                    "confidence": 0.9,
                },
                pre_reply_payload={
                    "*": {
                        "reply_mode": "llm",
                        "answer_source": "llm",
                        "scope": "unknown",
                        "location_text": "",
                        "memory_action": "none",
                        "memory_kind": "none",
                        "memory_type": "none",
                        "candidate_content": "",
                        "confidence": 0.9,
                        "reason": "fake no segment candidates",
                    },
                },
            ))
            message = (
                "会议记录：今天讨论语音入口主路径。产品负责访谈问题，后端负责后台写入。"
                "我们决定先做可观测 debug，再做真实设备适配。风险是语音权限不稳定，"
                "以及后台写入失败时用户不知道。周五下午 3 点前先交第一版。"
            )

            result = service.chat(message, user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])
            memories = store.list_memories("u1", kind="event")

            self.assertEqual(job["status"], "saved")
            self.assertGreaterEqual(job["saved_count"], 2)
            self.assertEqual(job["extraction_backend"], "semantic_cleaner+llm_segmented+rule_fallback")
            self.assertEqual(job["extraction_trace"]["source"], "text_cleaning")
            self.assertGreaterEqual(job["extraction_trace"]["segment_count"], 2)
            self.assertGreaterEqual(job["extraction_trace"]["rule_fallback"]["candidate_count"], 2)
            self.assertEqual(result["debug"]["memory_processing"]["extraction_trace"]["source"], "text_cleaning")
            self.assertEqual(service.fake_agent.intent_calls, 0)
            self.assertEqual(service.fake_agent.semantic_calls, job["extraction_trace"]["llm_segment_count"])
            self.assertEqual(service.fake_agent.segment_calls, job["extraction_trace"]["segment_count"])
            self.assertEqual(service.fake_agent.main_calls, 0)
            self.assertTrue(all(memory.evidence_ids for memory in memories))

    def test_long_input_rule_fallback_still_works_without_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent())
            job = service._create_memory_job(
                user_id="u1",
                session_id="s1",
                mode="continuous_capture",
                candidate_count=2,
                created_at=1778131200.0,
                evidence_ids=["chunk_rule_only"],
            )
            message = (
                "会议记录：今天讨论语音入口主路径。产品负责访谈问题，后端负责后台写入。"
                "我们决定先做可观测 debug，再做真实设备适配。风险是语音权限不稳定，"
                "以及后台写入失败时用户不知道。周五下午 3 点前先交第一版。"
            )

            service._process_long_input_background(
                message=message,
                user_id="u1",
                session_id="s1",
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                agent=None,
                job_id=job["job_id"],
                evidence_ids=["chunk_rule_only"],
            )
            finished = service.read_memory_job(user_id="u1", job_id=job["job_id"])
            memories = store.list_memories("u1", kind="event")

            self.assertEqual(finished["status"], "saved")
            self.assertEqual(finished["extraction_backend"], "rule_fallback")
            self.assertGreaterEqual(finished["extraction_trace"]["rule_fallback"]["candidate_count"], 2)
            self.assertGreaterEqual(len(memories), 2)
            self.assertTrue(all(memory.evidence_ids == ["chunk_rule_only"] for memory in memories))

    def test_long_input_rule_fallback_respects_local_do_not_remember_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent())
            job = service._create_memory_job(
                user_id="u1",
                session_id="s1",
                mode="continuous_capture",
                candidate_count=1,
                created_at=1778131200.0,
                evidence_ids=["chunk_rule_scope"],
            )
            message = "嗯嗯，外卖电话不用记，但 Mia 负责验证语音按钮，周五前交第一版。"

            service._process_long_input_background(
                message=message,
                user_id="u1",
                session_id="s1",
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                agent=None,
                job_id=job["job_id"],
                evidence_ids=["chunk_rule_scope"],
            )
            finished = service.read_memory_job(user_id="u1", job_id=job["job_id"])
            memories = store.list_memories("u1", kind="event")
            saved_text = "\n".join(memory.content for memory in memories)

            self.assertEqual(finished["status"], "saved")
            self.assertEqual(finished["extraction_backend"], "rule_fallback")
            self.assertGreaterEqual(finished["saved_count"], 1)
            self.assertEqual(finished["extraction_trace"]["semantic_cleaning"]["backend"], "rule_fallback")
            self.assertIn("Mia", saved_text)
            self.assertIn("语音按钮", saved_text)
            self.assertNotIn("外卖电话", saved_text)

    def test_chat_long_input_llm_uses_cleaned_segments_and_skips_noise_only_segments(self) -> None:
        intent_payload = {
            "*": {
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [],
                "confidence": 0.4,
            },
            "Mia 要验证语音按钮": {
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [
                    {
                        "content": "Mia 要验证语音按钮",
                        "kind": "event",
                        "memory_type": "task",
                        "confidence": 0.92,
                        "reason": "cleaned_segment_task",
                    }
                ],
                "confidence": 0.92,
            },
        }
        segment_payload = {
            "*": {
                "semantic_role": "chitchat",
                "noise_level": "low",
                "contains_filler": False,
                "do_not_remember_scope": "",
                "should_extract": False,
                "candidate_span": "",
                "candidate_hint": "",
                "confidence": 0.7,
                "reason": "fake default skip",
            },
            "Mia 要验证语音按钮": {
                "semantic_role": "memory_candidate",
                "noise_level": "none",
                "contains_filler": False,
                "do_not_remember_scope": "",
                "should_extract": True,
                "candidate_span": "Mia 要验证语音按钮",
                "candidate_hint": "task",
                "confidence": 0.9,
                "reason": "explicit task span",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(intent_payload=intent_payload, segment_payload=segment_payload))
            message = (
                "我先口述一段明年 demo 的转写片段，嗯嗯，那个，哈哈。"
                "Mia 要验证语音按钮。"
                "Alex 只是旁边插了一句背景有点吵，音量忽高忽低，里面很多重复词和口头禅。"
                "中间停顿也比较久。"
                "外卖电话不用记。"
            )

            result = service.chat(message, user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])
            memories = store.list_memories("u1", kind="event")

            self.assertEqual(job["status"], "saved")
            self.assertIn("semantic_cleaner+llm_segmented", job["extraction_backend"])
            self.assertEqual(job["extraction_trace"]["source"], "text_cleaning")
            self.assertLess(job["extraction_trace"]["llm_segment_count"], job["extraction_trace"]["segment_count"])
            self.assertEqual(service.fake_agent.intent_calls, 0)
            self.assertEqual(service.fake_agent.semantic_calls, job["extraction_trace"]["llm_segment_count"])
            self.assertEqual(service.fake_agent.segment_calls, job["extraction_trace"]["segment_count"])
            self.assertEqual(job["extraction_trace"]["semantic_cleaning"]["extractable_segment_count"], 1)
            self.assertEqual([memory.content for memory in memories], ["Mia 要验证语音按钮"])
            self.assertEqual(service.fake_agent.main_calls, 0)

    def test_chat_long_input_semantic_cleaner_treats_markers_as_signals_not_final_skip(self) -> None:
        intent_payload = {
            "Mia 明天要验证语音按钮": {
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [
                    {
                        "content": "Mia 明天要验证语音按钮",
                        "kind": "event",
                        "memory_type": "task",
                        "confidence": 0.9,
                        "reason": "semantic_cleaner_span_task",
                    }
                ],
                "confidence": 0.9,
            },
            "*": {
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [],
                "confidence": 0.4,
            },
        }
        segment_payload = {
            "嗯嗯,那个外卖电话不用记,但 Mia 明天要验证语音按钮": {
                "semantic_role": "memory_candidate",
                "noise_level": "low",
                "contains_filler": True,
                "do_not_remember_scope": "外卖电话",
                "should_extract": True,
                "candidate_span": "Mia 明天要验证语音按钮",
                "candidate_hint": "task",
                "confidence": 0.88,
                "reason": "do_not_remember applies only to delivery phone",
            },
            "*": {
                "semantic_role": "chitchat",
                "noise_level": "low",
                "contains_filler": False,
                "do_not_remember_scope": "",
                "should_extract": False,
                "candidate_span": "",
                "candidate_hint": "",
                "confidence": 0.5,
                "reason": "fake default skip",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(
                store,
                agent=FakeAgent(intent_payload=intent_payload, segment_payload=segment_payload),
            )
            message = (
                "我先口述一段明年 demo 的转写片段。"
                "嗯嗯，那个外卖电话不用记，但 Mia 明天要验证语音按钮。"
                "Alex 旁边说背景有点吵，中间还有很多口头禅和停顿。"
                "后来又聊到项目阶段目标、会议状态和进展记录，只是这些背景先不用沉淀。"
            )

            result = service.chat(message, user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])
            memories = store.list_memories("u1", kind="event")
            semantic = job["extraction_trace"]["semantic_cleaning"]

            self.assertEqual(job["status"], "saved")
            self.assertIn("semantic_cleaner+llm_segmented", job["extraction_backend"])
            self.assertEqual(semantic["backend"], "llm")
            self.assertEqual(semantic["extractable_segment_count"], 1)
            self.assertEqual(semantic["segment_decisions"][1]["do_not_remember_scope"], "外卖电话")
            self.assertEqual(semantic["segment_decisions"][1]["candidate_span"], "Mia 明天要验证语音按钮")
            self.assertEqual(job["extraction_trace"]["memory_extraction"]["candidate_count"], 1)
            self.assertEqual(job["extraction_trace"]["memory_extraction"]["gate_rejected_count"], 0)
            self.assertEqual(service.fake_agent.intent_calls, 0)
            self.assertEqual(service.fake_agent.semantic_calls, 1)
            saved_text = "\n".join(memory.content for memory in memories)
            self.assertIn("Mia", saved_text)
            self.assertIn("验证语音按钮", saved_text)
            self.assertNotIn("外卖电话", saved_text)
            self.assertNotIn("会议状态", saved_text)

    def test_chat_long_input_uses_semantic_span_when_pre_reply_misses_candidate(self) -> None:
        segment_payload = {
            "物流电话不用记,不过周五前把按钮验收补完,后面其他背景先别沉淀": {
                "semantic_role": "memory_candidate",
                "noise_level": "low",
                "contains_filler": False,
                "do_not_remember_scope": "物流电话",
                "should_extract": True,
                "candidate_span": "周五前把按钮验收补完",
                "candidate_hint": "task",
                "confidence": 0.9,
                "reason": "local do-not-remember only applies to delivery phone",
            }
        }
        pre_reply_payload = {
            "*": {
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "memory_action": "none",
                "memory_kind": "none",
                "memory_type": "none",
                "candidate_content": "",
                "confidence": 0.9,
                "reason": "fake missed candidate",
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(
                store,
                agent=FakeAgent(pre_reply_payload=pre_reply_payload, segment_payload=segment_payload),
            )
            message = "先记一段转写文字：物流电话不用记，不过周五前把按钮验收补完，后面其他背景先别沉淀。"

            result = service.chat(message, user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])
            memories = store.list_memories("u1", kind="event")
            saved_text = "\n".join(memory.content for memory in memories)

            self.assertEqual(job["status"], "saved")
            self.assertIn("semantic_span_fallback", job["extraction_backend"])
            self.assertEqual(job["extraction_trace"]["semantic_cleaning"]["extractable_segment_count"], 1)
            self.assertEqual(job["extraction_trace"]["memory_extraction"]["candidate_count"], 1)
            self.assertIn("按钮验收", saved_text)
            self.assertNotIn("物流电话", saved_text)

    def test_chat_long_input_semantic_span_splits_parallel_tasks(self) -> None:
        segment_payload = {
            "Mia 明天上午十点核对外测名单,Alex 周五前补完审计回放说明": {
                "semantic_role": "memory_candidate",
                "noise_level": "low",
                "contains_filler": False,
                "do_not_remember_scope": "",
                "should_extract": True,
                "candidate_span": "Mia 明天上午十点核对外测名单,Alex 周五前补完审计回放说明",
                "candidate_hint": "task",
                "confidence": 0.92,
                "reason": "contains two concrete task commitments",
            },
            "*": {
                "semantic_role": "temporary_context",
                "noise_level": "low",
                "contains_filler": False,
                "do_not_remember_scope": "",
                "should_extract": False,
                "candidate_span": "",
                "candidate_hint": "",
                "confidence": 0.7,
                "reason": "background only",
            },
        }
        pre_reply_payload = {
            "*": {
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "memory_action": "none",
                "memory_kind": "none",
                "memory_type": "none",
                "candidate_content": "",
                "confidence": 0.9,
                "reason": "fake missed candidate",
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(
                store,
                agent=FakeAgent(pre_reply_payload=pre_reply_payload, segment_payload=segment_payload),
            )
            job = service._create_memory_job(
                user_id="u1",
                session_id="s1",
                mode="continuous_capture",
                candidate_count=2,
                created_at=1778131200.0,
                evidence_ids=["chunk_parallel_tasks"],
            )
            message = (
                "我刚下班路上随便聊几句，奶茶店排队很久这个不用记。"
                "先记真正有用的：Mia 明天上午十点核对外测名单，Alex 周五前补完审计回放说明。"
                "还有临时偏好先别写成偏好。"
            )

            service._process_long_input_background(
                message=message,
                user_id="u1",
                session_id="s1",
                reference_time=1778131200.0,
                query_temporal=TemporalResolution(),
                agent=service.fake_agent,
                job_id=job["job_id"],
                evidence_ids=["chunk_parallel_tasks"],
            )
            job = service.read_memory_job(user_id="u1", job_id=job["job_id"])
            memories = store.list_memories("u1", kind="event")
            saved_text = "\n".join(memory.content for memory in memories)

            self.assertEqual(job["status"], "saved")
            self.assertIn("semantic_span_fallback", job["extraction_backend"])
            self.assertEqual(job["saved_count"], 2)
            self.assertIn("Mia 明天上午十点核对外测名单", saved_text)
            self.assertIn("Alex 周五前补完审计回放说明", saved_text)
            self.assertNotIn("奶茶店", saved_text)

    def test_chat_long_input_rule_candidates_keep_project_launch_plan_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent())
            message = (
                "我刚才和团队聊了一大段，先口述给你：我们准备下周把记忆助手外测流程跑起来，"
                "产品负责整理访谈问题，后端负责补齐报告模板。大家觉得风险主要是口语输入太碎，"
                "还有长对话里任务和偏好混在一起不好拆。周三晚上先碰一次进度，周五下午出第一版复盘。"
            )

            result = service.chat(message, user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])
            memories = store.list_memories("u1", kind="event")
            saved_text = "\n".join(memory.content for memory in memories)

            self.assertEqual(job["status"], "saved")
            self.assertGreaterEqual(job["saved_count"], 2)
            self.assertEqual(job["extraction_backend"], "semantic_cleaner+llm_segmented+rule_fallback")
            self.assertGreaterEqual(job["extraction_trace"]["rule_fallback"]["candidate_count"], 2)
            self.assertIn("记忆助手", saved_text)
            self.assertEqual(service.fake_agent.intent_calls, 0)
            self.assertEqual(service.fake_agent.semantic_calls, job["extraction_trace"]["llm_segment_count"])

    def test_chat_long_input_semantic_cleaner_extracts_multiple_spans_without_rule_bypass(self) -> None:
        intent_payload = {
            "Lina 负责前端按钮": {
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [
                    {
                        "content": "Lina 负责前端按钮",
                        "kind": "event",
                        "memory_type": "task",
                        "confidence": 0.88,
                        "reason": "semantic_task",
                    }
                ],
                "confidence": 0.88,
            },
            "项目复盘决定先做可观测 debug，再做真实设备适配": {
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [
                    {
                        "content": "项目复盘决定先做可观测 debug，再做真实设备适配",
                        "kind": "event",
                        "memory_type": "decision",
                        "confidence": 0.9,
                        "reason": "semantic_decision",
                    }
                ],
                "confidence": 0.9,
            },
            "*": {
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [],
                "confidence": 0.4,
            },
        }
        segment_payload = {
            "我先口述一段转写内容:Lina 负责前端按钮,Owen 看后端接口": {
                "semantic_role": "memory_candidate",
                "noise_level": "none",
                "contains_filler": False,
                "do_not_remember_scope": "",
                "should_extract": True,
                "candidate_span": "Lina 负责前端按钮",
                "candidate_hint": "task",
                "confidence": 0.86,
                "reason": "task span in mixed segment",
            },
            "我们聊了很多实现细节,后面会继续补充录音里的上下文": {
                "semantic_role": "memory_candidate",
                "noise_level": "none",
                "contains_filler": False,
                "do_not_remember_scope": "",
                "should_extract": True,
                "candidate_span": "项目复盘决定先做可观测 debug，再做真实设备适配",
                "candidate_hint": "decision",
                "confidence": 0.86,
                "reason": "decision span supplied by semantic cleaner",
            },
            "*": {
                "semantic_role": "temporary_context",
                "noise_level": "low",
                "contains_filler": False,
                "do_not_remember_scope": "",
                "should_extract": False,
                "candidate_span": "",
                "candidate_hint": "",
                "confidence": 0.6,
                "reason": "background only",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(
                store,
                agent=FakeAgent(intent_payload=intent_payload, segment_payload=segment_payload),
            )
            message = (
                "我先口述一段转写内容：Lina 负责前端按钮，Owen 看后端接口。"
                "我们聊了很多实现细节，后面会继续补充录音里的上下文。"
                "这段录音里面还有一些上下文背景和讨论过程需要先保留下来。"
                "只需要先低打扰整理到时间线里。"
            )

            result = service.chat(message, user_id="u1", defer_memory_writes=True)
            job = service.read_memory_job(user_id="u1", job_id=result["debug"]["memory_processing"]["job_id"])
            memories = store.list_memories("u1", kind="event")
            memory_types = {memory.memory_type for memory in memories}

            self.assertEqual(job["status"], "saved")
            self.assertGreaterEqual(job["saved_count"], 2)
            self.assertIn("semantic_cleaner+llm_segmented", job["extraction_backend"])
            self.assertEqual(job["extraction_trace"]["semantic_cleaning"]["extractable_segment_count"], 2)
            self.assertEqual(service.fake_agent.intent_calls, 0)
            self.assertEqual(service.fake_agent.semantic_calls, 2)
            self.assertIn("decision", memory_types)
            self.assertTrue({"task", "project_state"} & memory_types)
            self.assertTrue(all(memory.evidence_ids for memory in memories))

    def test_chat_long_sensitive_input_redacts_debug_audit_and_rejects_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(intent_payload={
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "memory_write_candidates": [
                    {
                        "content": "用户的 token 是 sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
                        "kind": "profile",
                        "memory_type": "fact",
                        "privacy_level": "sensitive",
                        "confidence": 0.95,
                        "reason": "long_input_sensitive",
                    }
                ],
                "confidence": 0.9,
            }))
            secret = "sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
            message = (
                f"今天复盘项目时我先说了前端语音入口，然后提到 token 是 {secret}，"
                "后面又说后台写入失败要可观测，接着讨论周五先交第一版。"
                "晚上我还补充了一个风险：如果浏览器权限不稳定，用户可能看不到保存状态。"
            )

            result = service.chat(message, user_id="u1", defer_memory_writes=True)
            job_id = result["debug"]["memory_processing"]["job_id"]
            job = service.read_memory_job(user_id="u1", job_id=job_id)
            records = service.read_audit_records(user_id="u1", limit=10)
            serialized = json.dumps({"result": result, "job": job, "records": records}, ensure_ascii=False)

            self.assertEqual(job["status"], "rejected")
            self.assertEqual(job["saved_count"], 0)
            self.assertEqual(store.list_memories("u1"), [])
            self.assertNotIn(secret, serialized)
            self.assertIn("[已脱敏:token]", serialized)

    def test_capture_append_redacts_sensitive_timeline_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))
            secret = "Authorization: Bearer sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"

            started = service.start_capture(user_id="u1", context="周会")
            appended = service.append_capture_chunk(
                user_id="u1",
                capture_id=started["capture_id"],
                text=f"项目AI眼镜小王负责 continuous_capture，{secret}",
            )
            timeline_chunks = service.timeline_store.search_chunks("u1", "continuous_capture", limit=5)

            self.assertTrue(appended["redacted"])
            self.assertIn("token", appended["redaction_categories"])
            self.assertTrue(appended["cleaning_trace"]["redacted"])
            self.assertIn("token", appended["cleaning_trace"]["redaction_categories"])
            self.assertNotIn(secret, appended["cleaning_trace"]["normalized_text"])
            self.assertEqual(len(timeline_chunks), 1)
            self.assertNotIn(secret, timeline_chunks[0].text)
            self.assertNotIn(secret, service._captures[started["capture_id"]]["chunks"][0]["text"])

    def test_capture_stop_can_recover_after_service_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            timeline_db = Path(tmpdir) / "timeline.db"
            service = FakeService(store)
            service.timeline_store = TimelineStore(db_path=timeline_db)
            started = service.start_capture(user_id="u1", context="周会")
            service.append_capture_chunk(
                user_id="u1",
                capture_id=started["capture_id"],
                text="项目AI眼镜小王负责 continuous_capture",
            )

            restarted = FakeService(store)
            restarted.timeline_store = TimelineStore(db_path=timeline_db)
            stopped = restarted.stop_capture(user_id="u1", capture_id=started["capture_id"])

            self.assertEqual(stopped["status"], "stopped")
            self.assertEqual(stopped["chunk_count"], 1)
            self.assertEqual(stopped["import_result"]["saved_count"], 1)

    def test_memory_job_state_is_recovered_from_timeline_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            timeline_db = Path(tmpdir) / "timeline.db"
            service = FakeService(store)
            service.timeline_store = TimelineStore(db_path=timeline_db)

            job = service._create_memory_job(
                user_id="u1",
                session_id="s1",
                mode="reply_first_background",
                candidate_count=1,
                created_at=1778131200.0,
                evidence_ids=["chunk_1"],
            )
            restarted = FakeService(store)
            restarted.timeline_store = TimelineStore(db_path=timeline_db)
            recovered = restarted.read_memory_job(user_id="u1", job_id=job["job_id"])

            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["status"], "pending")
            self.assertEqual(recovered["source_trace"]["layer"], "structured_memory")
            self.assertEqual(recovered["source_trace"]["evidence_ids"], ["chunk_1"])

    def test_weekly_report_groups_project_memories_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_memory(
                "u1",
                "项目AI眼镜决定先补敏感信息门控",
                kind="event",
                memory_type="decision",
                start_at=1778131200.0,
                evidence_ids=["meeting_1"],
            )
            store.add_memory(
                "u1",
                "项目AI眼镜小王负责 continuous_capture",
                kind="event",
                memory_type="task",
                start_at=1778131300.0,
                tags=["project:AI眼镜"],
                evidence_ids=["meeting_2"],
            )

            report = service.weekly_report(user_id="u1", start_at=1778120000.0, end_at=1778140000.0)

            self.assertEqual(report["project_count"], 1)
            self.assertIn("AI眼镜决定先补敏感信息门控", report["draft"])
            self.assertIn("meeting_1", report["projects"][0]["evidence_ids"])
            self.assertEqual(report["source_summary"]["structured_memory_count"], 2)
            self.assertEqual(report["source_summary"]["event_count"], 2)
            self.assertIn("structured_memory", report["source_summary"]["source_types"])
            self.assertEqual(len(report["source_memories"]), 2)
            self.assertIn("meeting_1", report["evidence_ids"])

    def test_weekly_report_separates_open_completed_and_cancelled_project_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_memory(
                "u1",
                "AI 眼镜项目：Alex 负责补 eval 报告模板",
                kind="event",
                memory_type="task",
                start_at=1778131200.0,
                tags=["project:AI眼镜", "task_status:open"],
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目：公开对话 target 诊断集已完成",
                kind="event",
                memory_type="task",
                start_at=1778131300.0,
                tags=["project:AI眼镜", "task_status:completed"],
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目：取消周五复盘会",
                kind="event",
                memory_type="task",
                start_at=1778131400.0,
                tags=["project:AI眼镜", "task_status:cancelled"],
            )

            report = service.weekly_report(user_id="u1", start_at=1778120000.0, end_at=1778140000.0)
            project = report["projects"][0]

            self.assertEqual(project["tasks"], ["AI 眼镜项目：Alex 负责补 eval 报告模板"])
            self.assertEqual(project["completed_tasks"], ["AI 眼镜项目：公开对话 target 诊断集已完成"])
            self.assertEqual(project["cancelled_tasks"], ["AI 眼镜项目：取消周五复盘会"])
            self.assertIn("已完成任务", report["draft"])
            self.assertIn("已取消任务", report["draft"])

    def test_reminder_check_returns_timed_task_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_memory(
                "u1",
                "和张总开会",
                kind="event",
                memory_type="task",
                start_at=1778134800.0,
                end_at=1778138400.0,
                evidence_ids=["chat_turn_1"],
            )
            store.add_memory(
                "u1",
                "昨天吃了面",
                kind="event",
                memory_type="event",
                start_at=1778040000.0,
                end_at=1778043600.0,
            )

            result = service.check_reminders(user_id="u1", now=1778131200.0)

            self.assertEqual(result["reminder_count"], 1)
            self.assertEqual(result["reminders"][0]["content"], "和张总开会")
            self.assertEqual(result["reminders"][0]["evidence_ids"], ["chat_turn_1"])

    def test_reminder_check_skips_completed_and_cancelled_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            store.add_memory(
                "u1",
                "和张总开会",
                kind="event",
                memory_type="task",
                start_at=1778134800.0,
                end_at=1778138400.0,
                tags=["task_status:open"],
            )
            store.add_memory(
                "u1",
                "提交旧报告",
                kind="event",
                memory_type="task",
                start_at=1778135800.0,
                end_at=1778139400.0,
                tags=["task_status:completed"],
            )
            store.add_memory(
                "u1",
                "周五复盘会",
                kind="event",
                memory_type="task",
                start_at=1778136800.0,
                end_at=1778140400.0,
                tags=["task_status:cancelled"],
            )

            result = service.check_reminders(user_id="u1", now=1778131200.0)

            self.assertEqual(result["reminder_count"], 1)
            self.assertEqual(result["reminders"][0]["content"], "和张总开会")

    def test_instant_temporal_without_end_gets_one_hour_range(self) -> None:
        result = _resolution_from_payload(
            {
                "has_temporal_expression": True,
                "temporal_text": "周五下午3点",
                "kind": "instant",
                "start_at_iso": "2026-05-08T15:00:00+08:00",
                "end_at_iso": None,
                "granularity": "minute",
                "normalized_text": "和 Alex 对接",
                "confidence": 0.95,
                "reason": "single instant",
            },
            raw="{}",
            backend="test",
            timezone="CST",
        )

        self.assertTrue(result.usable_range)
        self.assertEqual(result.start_at, 1778223600.0)
        self.assertEqual(result.end_at, 1778227200.0)

    def test_broad_evening_temporal_range_expands_without_specific_clock(self) -> None:
        result = _resolution_from_payload(
            {
                "has_temporal_expression": True,
                "temporal_text": "今天晚上",
                "kind": "date_range",
                "start_at_iso": "2026-05-08T18:00:00+08:00",
                "end_at_iso": "2026-05-08T19:00:00+08:00",
                "granularity": "hour",
                "normalized_text": "要联系谁",
                "confidence": 0.95,
                "reason": "too narrow evening",
            },
            raw="{}",
            backend="test",
            timezone="Asia/Shanghai",
        )
        normalized = _normalize_broad_evening_resolution(result, message="我今天晚上要联系谁？")

        self.assertTrue(normalized.usable_range)
        self.assertEqual(normalized.start_at, 1778234400.0)
        self.assertEqual(normalized.end_at, 1778256000.0)

    def test_specific_evening_clock_is_not_expanded(self) -> None:
        result = _resolution_from_payload(
            {
                "has_temporal_expression": True,
                "temporal_text": "今晚8点",
                "kind": "instant",
                "start_at_iso": "2026-05-08T20:00:00+08:00",
                "end_at_iso": "2026-05-08T21:00:00+08:00",
                "granularity": "hour",
                "normalized_text": "给妈妈打电话",
                "confidence": 0.95,
                "reason": "specific evening clock",
            },
            raw="{}",
            backend="test",
            timezone="Asia/Shanghai",
        )
        normalized = _normalize_broad_evening_resolution(result, message="记一下今晚8点给妈妈打电话")

        self.assertEqual(normalized.start_at, 1778241600.0)
        self.assertEqual(normalized.end_at, 1778245200.0)

    def test_local_temporal_parser_handles_evening_clock(self) -> None:
        result = resolve_temporal_local(
            "记一下今晚8点给妈妈打电话",
            reference_time=1778205600.0,
            timezone="Asia/Shanghai",
        )

        self.assertTrue(result.usable_range)
        self.assertEqual(result.backend, "local")
        self.assertEqual(result.start_at, 1778241600.0)
        self.assertEqual(result.end_at, 1778245200.0)
        self.assertEqual(result.normalized_text, "给妈妈打电话")

    def test_local_temporal_parser_treats_broad_afternoon_as_range(self) -> None:
        result = resolve_temporal_local(
            "我下午去吃面",
            reference_time=1778131200.0,
            timezone="Asia/Shanghai",
        )

        self.assertTrue(result.usable_range)
        self.assertEqual(result.backend, "local")
        self.assertEqual(result.start_at, 1778126400.0)
        self.assertEqual(result.end_at, 1778148000.0)
        self.assertEqual(result.temporal_text, "下午")
        self.assertTrue(result.debug_payload()["broad_time_range"])

    def test_upcoming_plan_query_recalls_timed_and_recent_untimed_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "一般都是下午6点半下班", kind="profile")
            store.add_memory(
                "u1",
                "出门看车展",
                kind="event",
                start_at=1778169600.0,
                end_at=1778256000.0,
                time_granularity="day",
                temporal_text="明天",
            )
            store.add_memory("u1", "今天六点半下班", kind="event")
            store.add_memory("u1", "周五下午3点和 Alex 对接", kind="event")
            agent = FakeAgent(
                temporal_payload={
                    "has_temporal_expression": True,
                    "temporal_text": "最近",
                    "kind": "date_range",
                    "start_at_iso": "2026-04-30T00:00:00+08:00",
                    "end_at_iso": "2026-05-07T00:00:00+08:00",
                    "granularity": "day",
                    "normalized_text": "我要做什么",
                    "confidence": 0.85,
                    "reason": "past recent range",
                }
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我最近要做什么", user_id="u1")

            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "upcoming_plan")
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIn("出门看车展", recalled)
            self.assertIn("今天六点半下班", recalled)
            self.assertIn("周五下午3点和 Alex 对接", recalled)
            self.assertIn("出门看车展", result["reply"])

    def test_action_item_query_uses_local_upcoming_event_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "明天下午3点和 Alex 对接",
                kind="event",
                start_at=1778223600.0,
                end_at=1778227200.0,
                time_granularity="hour",
                temporal_text="明天下午3点",
            )
            store.add_memory("u1", "今天六点半下班", kind="event")
            service = FakeService(store)

            result = service.chat("我有什么待办事项", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_event_recall")
            self.assertTrue(result["debug"]["planner"]["needs_event_memory"])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "upcoming_plan")
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIn("明天下午3点和 Alex 对接", recalled)
            self.assertIn("今天六点半下班", recalled)
            self.assertIn("明天下午3点和 Alex 对接", result["reply"])

    def test_attention_items_query_uses_local_tasks_and_project_risks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "周五前把 demo 评估跑完",
                kind="event",
                memory_type="task",
                start_at=1778223600.0,
                end_at=1778227200.0,
                tags=["task_status:open", "project:AI眼镜"],
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目：风险是 eval 报告模板还没补齐",
                kind="event",
                memory_type="project_state",
                start_at=1778134800.0,
                tags=["project:AI眼镜"],
            )
            service = FakeService(store)

            result = service.chat("我接下来有什么要注意的？", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["conversation_action"], "attention_items")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "attention_items")
            self.assertEqual(result["debug"]["conversation_action"]["action"], "attention_items")
            self.assertEqual(result["debug"]["conversation_action"]["structured_memory_count"], 2)
            self.assertIn("structured_memory", result["debug"]["conversation_action"]["source_summary"]["source_types"])
            self.assertEqual(result["source_summary"]["structured_memory_count"], 2)
            self.assertIn("structured_memory", result["source_summary"]["source_types"])
            self.assertIn("eval 报告模板", result["reply"])
            self.assertIn("demo 评估", result["reply"])

    def test_current_project_attention_query_uses_local_tasks_and_project_risks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "AI 眼镜项目：风险是 eval 报告模板还没补齐",
                kind="event",
                memory_type="project_state",
                start_at=1778134800.0,
                tags=["project:AI眼镜"],
            )
            service = FakeService(store)

            result = service.chat("这个项目有什么风险要注意？", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["conversation_action"], "attention_items")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "attention_items")
            self.assertIn("eval 报告模板", result["reply"])

    def test_weekly_progress_query_uses_existing_weekly_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "AI 眼镜项目：公开对话 target 诊断集已完成",
                kind="event",
                memory_type="task",
                start_at=1778131300.0,
                tags=["project:AI眼镜", "task_status:completed"],
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目：Alex 负责补 eval 报告模板",
                kind="event",
                memory_type="task",
                start_at=1778131200.0,
                tags=["project:AI眼镜", "task_status:open"],
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目：决定先补对话式周报入口",
                kind="event",
                memory_type="decision",
                start_at=1778131400.0,
                tags=["project:AI眼镜"],
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目：风险是注意事项问法还会掉到 LLM",
                kind="event",
                memory_type="project_state",
                start_at=1778131500.0,
                tags=["project:AI眼镜"],
            )
            service = FakeService(store)

            result = service.chat("这周进展如何？", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["conversation_action"], "weekly_report")
            self.assertEqual(result["debug"]["conversation_action"]["action"], "weekly_report")
            self.assertEqual(result["debug"]["conversation_action"]["structured_memory_count"], 4)
            self.assertIn("structured_memory", result["debug"]["conversation_action"]["source_summary"]["source_types"])
            self.assertEqual(result["source_summary"]["structured_memory_count"], 4)
            self.assertIn("structured_memory", result["source_summary"]["source_types"])
            self.assertEqual(result["weekly_report"]["source_summary"], result["source_summary"])
            self.assertIn("本周项目进展草稿", result["reply"])
            self.assertIn("已完成任务", result["reply"])
            self.assertIn("待办/计划", result["reply"])
            self.assertIn("会议结论/决策", result["reply"])
            self.assertIn("风险/卡点", result["reply"])
            self.assertEqual(service.fake_agent.main_calls, 0)

    def test_weekly_progress_query_combines_app_document_and_voice_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.import_memory_events(
                user_id="u1",
                text="# AI 眼镜项目会议纪要\n决定先把 App 文档和口述状态合入周报，背景是周报不能只看聊天记忆。",
                source="markdown_upload",
                context="AI 眼镜项目会议纪要.md",
            )
            service.import_memory_events(
                user_id="u1",
                items=[
                    {
                        "content": "AI 眼镜项目：Mia 负责验证转写文本导入",
                        "kind": "event",
                        "memory_type": "task",
                        "confidence": 0.93,
                    },
                    {
                        "content": "AI 眼镜项目：决定周报继续复用启发式草稿",
                        "kind": "event",
                        "memory_type": "decision",
                        "confidence": 0.9,
                    },
                    {
                        "content": "AI 眼镜项目：风险是后台保存失败用户看不见",
                        "kind": "event",
                        "memory_type": "project_state",
                        "confidence": 0.9,
                    },
                ],
                text="AI 眼镜项目 Mia 负责验证转写文本导入，决定周报继续复用启发式草稿，风险是后台保存失败用户看不见。",
                source="app_audio_transcript",
                context="walking transcript after ASR",
            )

            result = service.chat("这周进展如何？", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["conversation_action"], "weekly_report")
            self.assertEqual(result["debug"]["conversation_action"]["action"], "weekly_report")
            self.assertEqual(result["debug"]["conversation_action"]["document_count"], 1)
            self.assertGreaterEqual(result["debug"]["conversation_action"]["structured_memory_count"], 3)
            self.assertEqual(result["source_summary"]["document_count"], 1)
            self.assertGreaterEqual(result["source_summary"]["structured_memory_count"], 3)
            self.assertEqual(result["weekly_report"]["source_summary"], result["source_summary"])
            self.assertIn("document", result["source_summary"]["source_types"])
            self.assertIn("structured_memory", result["source_summary"]["source_types"])
            self.assertEqual(result["source_summary"]["primary_source"], "structured_memory")
            self.assertIn("结构化任务、决策或项目状态", result["source_summary"]["primary_source_explanation"])
            self.assertIn("文档/背景", result["reply"])
            self.assertIn("App 文档和口述状态合入周报", result["reply"])
            self.assertIn("待办/计划", result["reply"])
            self.assertIn("Mia", result["reply"])
            self.assertIn("会议结论/决策", result["reply"])
            self.assertIn("风险/卡点", result["reply"])
            self.assertIn("后台保存失败", result["reply"])
            self.assertEqual(len(result["recalled_documents"]), 1)
            self.assertGreaterEqual(len(result["recalled_memories"]), 3)
            self.assertEqual(service.fake_agent.main_calls, 0)

    def test_weekly_report_prefers_structured_project_state_before_background_document_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.import_memory_events(
                user_id="u1",
                text="# AI 眼镜项目纪要\n项目背景：周报不能只看聊天记忆。\n说明：这份文档主要解释背景，不直接代表当前主线状态。",
                source="markdown_upload",
                context="AI 眼镜项目纪要.md",
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目当前主线是收敛文字输入质量，先压 recent context 污染。",
                kind="event",
                memory_type="project_state",
                start_at=1778131500.0,
                tags=["project:AI眼镜"],
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目：决定先补 semantic cleaner 的局部范围判断",
                kind="event",
                memory_type="decision",
                start_at=1778131400.0,
                tags=["project:AI眼镜"],
            )

            result = service.chat("这周进展如何？", user_id="u1")
            reply = result["reply"]

            self.assertIn("会议结论/决策", reply)
            self.assertIn("AI 眼镜项目当前主线是收敛文字输入质量", reply)
            self.assertIn("文档/背景", reply)
            self.assertLess(reply.index("AI 眼镜项目当前主线是收敛文字输入质量"), reply.index("文档/背景"))
            self.assertIn("周报不能只看聊天记忆", reply)

    def test_upcoming_plan_displays_broad_afternoon_without_exact_start_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "去吃面",
                            "kind": "event",
                            "memory_type": "event",
                            "confidence": 0.9,
                            "reason": "daily life event",
                        }
                    ],
                    "confidence": 0.9,
                }
            )
            service = FakeService(store, agent=agent)

            service.chat("我下午去吃面", user_id="u1", defer_memory_writes=True)
            result = service.chat("最近有什么安排", user_id="u1")

            memories = store.list_memories("u1", kind="event")
            self.assertGreaterEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "去吃面")
            self.assertEqual(memories[0].temporal_text, "下午")
            self.assertEqual(memories[0].start_at, 1778126400.0)
            self.assertEqual(memories[0].end_at, 1778148000.0)
            self.assertNotIn("12:00", result["reply"])
            self.assertNotIn("12:00", result["reply"])
            self.assertGreaterEqual(agent.semantic_calls, 1)

    def test_bare_recent_uses_upcoming_memory_before_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "明天下午3点和 Alex 对接",
                kind="event",
                start_at=1778223600.0,
                end_at=1778227200.0,
                time_granularity="hour",
                temporal_text="明天下午3点",
            )
            service = FakeService(store)

            result = service.chat("最近", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_event_recall")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "ambiguous_recent_upcoming_plan")
            self.assertIn("Alex 对接", result["reply"])

    def test_bare_recent_clarifies_when_no_upcoming_memory_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("最近", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_event_recall")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "ambiguous_recent_upcoming_plan")
            self.assertEqual(result["recalled_memories"], [])

    def test_recent_food_query_uses_past_temporal_recall_not_upcoming_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "昨天中午吃了面",
                kind="event",
                start_at=1778040000.0,
                end_at=1778043600.0,
                time_granularity="hour",
            )
            store.add_memory(
                "u1",
                "明天出门看车展",
                kind="event",
                start_at=1778169600.0,
                end_at=1778256000.0,
                time_granularity="day",
            )
            agent = FakeAgent(
                temporal_payload={
                    "has_temporal_expression": True,
                    "temporal_text": "最近",
                    "kind": "date_range",
                    "start_at_iso": "2026-04-30T00:00:00+08:00",
                    "end_at_iso": "2026-05-07T00:00:00+08:00",
                    "granularity": "day",
                    "normalized_text": "吃了什么",
                    "confidence": 0.85,
                    "reason": "past recent range",
                }
            )
            service = FakeService(store, agent=agent)

            result = service.chat("最近吃了什么", user_id="u1")

            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "temporal_range")
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIn("昨天中午吃了面", recalled)
            self.assertNotIn("明天出门看车展", recalled)

    def test_new_web_session_still_recalls_user_event_memory_for_today_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "今天晚上下班吃面",
                kind="event",
                start_at=1778148000.0,
                end_at=1778162400.0,
                time_granularity="hour",
                temporal_text="今天晚上",
            )
            service = FakeService(store)

            result = service.chat("我今天要干嘛", user_id="u1")

            self.assertEqual(result["session_id"], "fake-session")
            self.assertTrue(result["debug"]["planner"]["needs_event_memory"])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "upcoming_plan")
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIsInstance(recalled, str)
            self.assertNotIn("18:00", result["reply"])
            self.assertIn(service.fake_agent.main_calls, {0, 1})

    def test_today_plan_reuses_local_temporal_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "今天下午3点和 Mina 开会",
                kind="event",
                start_at=1778137200.0,
                end_at=1778140800.0,
                time_granularity="hour",
                temporal_text="今天下午3点",
            )
            service = FakeService(store)

            result = service.chat("我今天要做什么呢", user_id="u1")

            self.assertEqual(service.fake_agent.temporal_calls, 0)
            self.assertEqual(result["debug"]["routing"]["temporal_backend"], "local_reused")
            self.assertEqual(result["debug"]["routing"]["temporal_llm_skipped_reason"], "usable_local_temporal_scope")
            self.assertEqual(result["debug"]["temporal"]["query"]["backend"], "local")
            self.assertIn("Mina 开会", result["reply"])

    def test_yesterday_recall_reuses_local_temporal_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "昨天中午吃面",
                kind="event",
                start_at=1778040000.0,
                end_at=1778043600.0,
                time_granularity="hour",
                temporal_text="昨天中午",
            )
            store.add_memory(
                "u1",
                "今天下午3点和 Mina 开会",
                kind="event",
                start_at=1778137200.0,
                end_at=1778140800.0,
                time_granularity="hour",
                temporal_text="今天下午3点",
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "memory_recall_type": "event",
                "recall_goal": "specific_fact",
                "event_recall_strategy": "temporal_range",
                "confidence": 0.93,
                "reason": "asks for remembered activities in yesterday range",
            }))

            result = service.chat("昨天做什么了", user_id="u1")
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])

            self.assertEqual(service.fake_agent.temporal_calls, 0)
            self.assertEqual(result["debug"]["routing"]["temporal_backend"], "local_reused")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "temporal_range")
            self.assertIn("昨天中午吃面", recalled)
            self.assertNotIn("Mina 开会", recalled)

    def test_hour_plan_reuses_local_temporal_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "今天下午3点和 Mina 开会",
                kind="event",
                start_at=1778137200.0,
                end_at=1778140800.0,
                time_granularity="hour",
                temporal_text="今天下午3点",
            )
            store.add_memory(
                "u1",
                "今天晚上下班吃面",
                kind="event",
                start_at=1778148000.0,
                end_at=1778162400.0,
                time_granularity="hour",
                temporal_text="今天晚上",
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "memory_recall_type": "event",
                "recall_goal": "specific_fact",
                "event_recall_strategy": "temporal_range",
                "confidence": 0.93,
                "reason": "asks for remembered activities in an exact local hour",
            }))

            result = service.chat("今天下午3点有什么", user_id="u1")
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])

            self.assertEqual(service.fake_agent.temporal_calls, 0)
            self.assertEqual(result["debug"]["routing"]["temporal_backend"], "local_reused")
            self.assertEqual(result["debug"]["temporal"]["query"]["granularity"], "hour")
            self.assertIn("Mina 开会", recalled)
            self.assertNotIn("下班吃面", recalled)

    def test_complex_week_recall_keeps_temporal_llm_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "上周推进了声纹录入跳过按钮",
                kind="event",
                start_at=1777521600.0,
                end_at=1777525200.0,
                time_granularity="hour",
                temporal_text="上周",
            )
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "memory_recall_type": "event",
                    "recall_goal": "summary",
                    "event_recall_strategy": "temporal_range",
                    "confidence": 0.93,
                    "reason": "asks for a complex week range",
                },
                temporal_payload={
                    "has_temporal_expression": True,
                    "temporal_text": "上周",
                    "kind": "date_range",
                    "start_at_iso": "2026-04-27T00:00:00+08:00",
                    "end_at_iso": "2026-05-04T00:00:00+08:00",
                    "granularity": "day",
                    "normalized_text": "做了什么",
                    "confidence": 0.9,
                    "reason": "previous calendar week",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("上周做什么了", user_id="u1")

            self.assertEqual(agent.temporal_calls, 1)
            self.assertEqual(result["debug"]["routing"]["temporal_backend"], "llm")
            self.assertNotEqual(result["debug"]["routing"].get("temporal_llm_skipped_reason"), "usable_local_temporal_scope")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "temporal_range")

    def test_today_plan_excludes_untimed_related_memory_from_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "今天下午3点和 Mina 开会",
                kind="event",
                start_at=1778137200.0,
                end_at=1778140800.0,
                time_granularity="hour",
                temporal_text="今天下午3点",
            )
            store.add_memory(
                "u1",
                "用户决定去吃肉丝面",
                kind="event",
            )
            store.add_memory(
                "u1",
                "今天去买菜",
                kind="event",
                start_at=1778025600.0,
                end_at=1778112000.0,
                time_granularity="day",
                temporal_text="今天",
            )
            service = FakeService(store, agent=FakeAgent(temporal_payload={
                "has_temporal_expression": True,
                "temporal_text": "今天",
                "kind": "date_range",
                "start_at_iso": "2026-05-07T00:00:00+08:00",
                "end_at_iso": "2026-05-08T00:00:00+08:00",
                "granularity": "day",
                "normalized_text": "我要做什么呢",
                "confidence": 1.0,
                "reason": "today range",
            }))

            result = service.chat("我今天要做什么呢", user_id="u1")

            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "upcoming_plan")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["timed_plan_count"], 1)
            self.assertEqual(result["debug"]["memory"]["event_recall"]["untimed_related_count"], 0)
            self.assertEqual(result["debug"]["memory"]["event_recall"]["excluded_untimed_plan_count"], 1)
            self.assertEqual(
                result["debug"]["memory"]["event_recall"]["excluded_untimed_plan_policy"],
                "explicit_time_window_requires_timed_memory",
            )
            self.assertIn("Mina 开会", result["reply"])
            self.assertNotIn("用户决定去吃肉丝面", result["reply"])
            self.assertNotIn("不能确定是不是今天的安排", result["reply"])
            self.assertNotIn("今天去买菜", result["reply"])

    def test_today_plan_with_only_untimed_related_memory_returns_no_definite_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户决定去吃肉丝面", kind="event")
            store.add_memory(
                "u1",
                "针对“香港地区部分银行开立投资账户需签署声明”一事，香港金管局今日回应称，相关监管要求系5月22日向所有认可机构发出。",
                kind="event",
                source="app_audio_transcript",
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "memory_recall_type": "event",
                "recall_goal": "summary",
                "event_recall_strategy": "ambiguous_recent_upcoming_plan",
                "confidence": 0.93,
                "reason": "asks for today's personal plan",
            }))

            result = service.chat("今天做什么", user_id="u1")

            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "ambiguous_recent_upcoming_plan")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["timed_plan_count"], 0)
            self.assertEqual(result["debug"]["memory"]["event_recall"]["untimed_related_count"], 0)
            self.assertEqual(result["debug"]["memory"]["event_recall"]["excluded_untimed_plan_count"], 2)
            self.assertEqual(
                result["debug"]["memory"]["event_recall"]["excluded_untimed_plan_policy"],
                "explicit_time_window_requires_timed_memory",
            )
            self.assertIn("没有查到这个时间段内的确定安排", result["reply"])
            self.assertNotIn("肉丝面", result["reply"])
            self.assertNotIn("香港金管局", result["reply"])
            self.assertEqual(service.fake_agent.main_calls, 0)

    def test_plan_recall_partition_excludes_untimed_for_explicit_time_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            timed = store.add_memory(
                "u1",
                "今天下午3点和 Mina 开会",
                kind="event",
                start_at=1778137200.0,
                end_at=1778140800.0,
            )
            untimed = store.add_memory("u1", "用户决定去吃肉丝面", kind="event")
            old = store.add_memory(
                "u1",
                "今天去买菜",
                kind="event",
                start_at=1778025600.0,
                end_at=1778112000.0,
            )
            planner = TurnPlan(
                needs_event_memory=True,
                reply_mode="local_event_recall",
                event_recall_strategy="upcoming_plan",
                temporal_scope=TemporalResolution(
                    has_temporal_expression=True,
                    start_at=1778112000.0,
                    end_at=1778198400.0,
                    backend="local",
                    kind="date_range",
                    temporal_text="今天",
                    granularity="day",
                    confidence=0.95,
                ),
            )

            partition = GlassesChatService._partition_plan_recall_memories(
                [timed, untimed, old],
                message="我今天要做什么呢",
                planner=planner,
                query_temporal=planner.temporal_scope,
                start_at=1778112000.0,
                end_at=1778198400.0,
            )

            self.assertIsNotNone(partition)
            timed_memories, untimed_memories = partition or ([], [])
            self.assertEqual([memory.id for memory in timed_memories], [timed.id])
            self.assertEqual([memory.id for memory in untimed_memories], [])

    def test_uncertain_statement_still_uses_pre_reply_and_llm_classifier_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent()
            service = FakeService(store, agent=agent)

            result = service.chat("周末就是要放松", user_id="u1", defer_memory_writes=True)

            self.assertEqual(result["debug"]["planner"]["reply_mode"], "llm")
            self.assertEqual(result["debug"]["planner"]["memory_write_count"], 0)
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)

    def test_new_web_session_recalls_raw_timeline_without_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)
            service.timeline_store.add_turn(
                "u1",
                "今天项目会上我提到国内网络下语音识别不稳定，可能需要 VPN 或备用方案。",
                created_at=1778120000.0,
            )

            result = service.chat("我之前有没有提到过国内网络的问题？", user_id="u1")

            self.assertEqual(result["session_id"], "fake-session")
            self.assertEqual(result["api_calls"], 0)
            self.assertEqual(service.fake_agent.main_calls, 0)
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_timeline_recall")
            self.assertEqual(result["debug"]["timeline"]["recall"]["strategy"], "chunk_full_text")
            self.assertEqual(result["debug"]["timeline"]["recall"]["recall_trace"]["layer"], "raw_timeline")
            self.assertEqual(len(result["recalled_timeline_chunks"]), 1)
            self.assertEqual(result["recalled_timeline_chunks"][0]["source_trace"]["layer"], "raw_timeline")
            self.assertIn("国内网络下语音识别不稳定", result["reply"])

    def test_natural_temporal_life_event_is_saved_for_future_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                {
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "下班吃面",
                            "kind": "event",
                            "confidence": 0.9,
                            "reason": "daily life event",
                        }
                    ],
                    "confidence": 0.9,
                }
            )
            service = FakeService(store, agent=agent)

            result = service.chat("今天下班吃面", user_id="u1", defer_memory_writes=True)

            self.assertEqual(result["debug"]["planner"]["memory_write_count"], 1)
            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            memories = store.list_memories("u1", kind="event")
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "下班吃面")
            self.assertEqual(memories[0].time_granularity, "day")
            self.assertEqual(memories[0].evidence_ids, result["debug"]["timeline"]["chunk_ids"])

    def test_current_time_query_uses_local_clock_without_agent(self) -> None:
        cases = ("现在是几号几点了", "现在几点", "今天几号", "今天星期几", "当前时间")
        for message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    store = self.make_store(Path(tmpdir))
                    agent = FakeAgent(pre_reply_payload={
                        "reply_mode": "local_current_time",
                        "answer_source": "local_clock",
                        "scope": "device_local",
                        "location_text": "",
                        "confidence": 0.92,
                        "reason": "asks for device-local current time",
                    })
                    service = FakeService(store, agent=agent)

                    result = service.chat(message, user_id="u1")

                    self.assertIn("2026年5月7日", result["reply"])
                    self.assertIn("星期四", result["reply"])
                    self.assertIn("13点20分", result["reply"])
                    self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_current_time")
                    self.assertEqual(result["debug"]["pre_reply_decision"]["reply_mode"], "local_current_time")
                    self.assertEqual(result["debug"]["pre_reply_decision"]["answer_source"], "local_clock")
                    self.assertEqual(result["debug"]["llm"]["skipped"], True)
                    self.assertEqual(result["debug"]["tools"][0]["triggered"], False)
                    self.assertEqual(result["debug"]["steps"], [
                        "planner_disabled_llm_first",
                        "created_pre_reply_decision_session",
                        "loaded_gated_memory",
                        "local_reply_completed",
                    ])
                    self.assertEqual(result["api_calls"], 0)
                    self.assertEqual(service.new_session_calls, 1)
                    self.assertEqual(agent.pre_reply_calls, 1)
                    self.assertGreaterEqual(agent.semantic_calls, 1)
                    self.assertEqual(agent.main_calls, 0)
                    self.assertEqual(result["saved_memories"], [])
                    self.assertEqual(store.list_memories("u1"), [])

    def test_ambient_capture_context_is_injected_without_long_term_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "needs_timeline_recall": False,
                "confidence": 0.9,
                "reason": "wake query should use ambient context",
            })
            service = FakeService(store, agent=agent)
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            first = service.append_capture_chunk(
                user_id="u1",
                capture_id=capture["capture_id"],
                text="这个你应该早就知道吧",
                metadata={"audio_retention": "not_recorded_browser_asr_text_only"},
            )
            second = service.append_capture_chunk(
                user_id="u1",
                capture_id=capture["capture_id"],
                text="服了，又来了",
            )

            result = service.chat(
                "你觉得刚才他是不是在阴阳我？",
                user_id="u1",
                ambient_capture_id=capture["capture_id"],
                wake_session={
                    "ambient_capture_id": capture["capture_id"],
                    "wake_detected_at": 1778131212.0,
                    "pre_wake_segment_ids": [first["chunk_id"], second["chunk_id"]],
                    "post_wake_query_segment_ids": [],
                    "wake_query_text": "你觉得刚才他是不是在阴阳我？",
                    "wake_mode": "button",
                    "status": "consumed",
                },
            )

            self.assertTrue(result["debug"]["ambient_context"]["used"])
            self.assertTrue(result["debug"]["ambient_context"]["injected_to_main_llm"])
            self.assertEqual(result["debug"]["ambient_context"]["chunk_ids"], [first["chunk_id"], second["chunk_id"]])
            self.assertEqual(result["debug"]["ambient_context"]["pre_wake_segment_count"], 2)
            self.assertEqual(result["debug"]["ambient_context"]["source"], "ambient_audio_text")
            self.assertIn("Recent ambient audio transcript before wake word", agent.main_messages[-1])
            self.assertIn("这个你应该早就知道吧", agent.main_messages[-1])
            self.assertIn("服了，又来了", agent.main_messages[-1])
            self.assertNotIn("emotion_label=", agent.main_messages[-1])
            self.assertNotIn("emotion_intensity=", agent.main_messages[-1])
            self.assertNotIn("emotion_evidence=", agent.main_messages[-1])
            self.assertEqual(result["debug"]["ambient_context"]["text_emotion"]["label"], "unknown")
            self.assertFalse(result["debug"]["ambient_context"]["emotion_fusion"]["reply_emotion_used"])
            self.assertEqual(store.list_memories("u1"), [])

    def test_ambient_emotion_fusion_uses_high_confidence_acoustic_when_text_unknown(self) -> None:
        audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_timeline_recall": False,
                    "confidence": 0.9,
                    "reason": "wake query should use ambient context",
                },
                text_emotion_payload={
                    "label": "unknown",
                    "confidence": "low",
                    "should_affect_reply": False,
                    "reason": "text is too short",
                },
            )
            service = FakeService(
                store,
                agent=agent,
                asr_runner=FakeASRRunner(),
                emotion_runner=FakeEmotionRunner(),
                speaker_runner=FakeSpeakerRunner(),
            )
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            appended = service.process_audio_segment(
                user_id="u1",
                capture_id=capture["capture_id"],
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )
            chunk_id = appended["capture_append"]["chunk_id"]

            result = service.chat(
                "你觉得我刚才是不是有点不爽？",
                user_id="u1",
                ambient_capture_id=capture["capture_id"],
                wake_session={
                    "ambient_capture_id": capture["capture_id"],
                    "wake_detected_at": 1778131212.0,
                    "pre_wake_segment_ids": [chunk_id],
                    "post_wake_query_segment_ids": [],
                    "wake_query_text": "你觉得我刚才是不是有点不爽？",
                    "wake_mode": "button",
                    "status": "consumed",
                },
            )

        self.assertEqual(result["debug"]["ambient_context"]["acoustic_emotion"]["label"], "annoyed")
        self.assertTrue(result["debug"]["ambient_context"]["acoustic_emotion"]["eligible_for_reply"])
        self.assertEqual(result["debug"]["ambient_context"]["text_emotion"]["label"], "unknown")
        self.assertTrue(result["debug"]["ambient_context"]["emotion_fusion"]["reply_emotion_used"])
        self.assertEqual(result["debug"]["ambient_context"]["emotion_fusion"]["label"], "annoyed")
        self.assertIn("<ambient-emotion-policy>", agent.main_messages[-1])
        self.assertNotIn("emotion_label=", agent.main_messages[-1])

    def test_ambient_emotion_fusion_conflict_falls_back_to_unknown(self) -> None:
        audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_timeline_recall": False,
                    "confidence": 0.9,
                    "reason": "wake query should use ambient context",
                },
                text_emotion_payload={
                    "label": "calm",
                    "confidence": "high",
                    "should_affect_reply": True,
                    "reason": "explicitly calm",
                },
            )
            service = FakeService(
                store,
                agent=agent,
                asr_runner=FakeASRRunner(),
                emotion_runner=FakeEmotionRunner(),
                speaker_runner=FakeSpeakerRunner(),
            )
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            appended = service.process_audio_segment(
                user_id="u1",
                capture_id=capture["capture_id"],
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )
            chunk_id = appended["capture_append"]["chunk_id"]

            result = service.chat(
                "你觉得我刚才是不是有点不爽？",
                user_id="u1",
                ambient_capture_id=capture["capture_id"],
                wake_session={
                    "ambient_capture_id": capture["capture_id"],
                    "wake_detected_at": 1778131212.0,
                    "pre_wake_segment_ids": [chunk_id],
                    "post_wake_query_segment_ids": [],
                    "wake_query_text": "你觉得我刚才是不是有点不爽？",
                    "wake_mode": "button",
                    "status": "consumed",
                },
            )

        self.assertEqual(result["debug"]["ambient_context"]["emotion_fusion"]["label"], "unknown")
        self.assertFalse(result["debug"]["ambient_context"]["emotion_fusion"]["reply_emotion_used"])
        self.assertTrue(result["debug"]["ambient_context"]["emotion_fusion"]["conflict_detected"])
        self.assertNotIn("<ambient-emotion-policy>", agent.main_messages[-1])

    def test_ambient_emotion_fusion_uses_high_confidence_text_when_acoustic_is_below_threshold(self) -> None:
        audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        from ai_glasses_memory_assistant.agent_bridge import LocalEmotionResult

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_timeline_recall": False,
                    "confidence": 0.9,
                    "reason": "wake query should use ambient context",
                },
                text_emotion_payload={
                    "label": "sad",
                    "confidence": "high",
                    "should_affect_reply": True,
                    "reason": "explicit sadness in text",
                },
            )
            service = FakeService(
                store,
                agent=agent,
                asr_runner=FakeASRRunner(),
                emotion_runner=FakeEmotionRunner(
                    result=LocalEmotionResult(
                        enabled=True,
                        label="烦躁",
                        intensity=4,
                        score=0.79,
                        candidates=[{"label": "烦躁", "score": 0.79}],
                        model_name="emotion2vec_plus_base",
                        evidence="derived_from_acoustic_emotion_label: angry",
                        source="acoustic_emotion_model",
                        reason="acoustic_emotion_inferred",
                    )
                ),
                speaker_runner=FakeSpeakerRunner(),
            )
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            appended = service.process_audio_segment(
                user_id="u1",
                capture_id=capture["capture_id"],
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )
            chunk_id = appended["capture_append"]["chunk_id"]

            result = service.chat(
                "我刚才其实挺难受的。",
                user_id="u1",
                ambient_capture_id=capture["capture_id"],
                wake_session={
                    "ambient_capture_id": capture["capture_id"],
                    "wake_detected_at": 1778131212.0,
                    "pre_wake_segment_ids": [chunk_id],
                    "post_wake_query_segment_ids": [],
                    "wake_query_text": "我刚才其实挺难受的。",
                    "wake_mode": "button",
                    "status": "consumed",
                },
            )

        self.assertFalse(result["debug"]["ambient_context"]["acoustic_emotion"]["eligible_for_reply"])
        self.assertEqual(result["debug"]["ambient_context"]["text_emotion"]["label"], "sad")
        self.assertTrue(result["debug"]["ambient_context"]["emotion_fusion"]["text_high_confidence"])
        self.assertEqual(result["debug"]["ambient_context"]["emotion_fusion"]["label"], "sad")
        self.assertTrue(result["debug"]["ambient_context"]["emotion_fusion"]["reply_emotion_used"])
        self.assertIn("<ambient-emotion-policy>", agent.main_messages[-1])

    def test_ambient_emotion_fusion_returns_unknown_when_text_classifier_fails(self) -> None:
        audio_base64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        from ai_glasses_memory_assistant.agent_bridge import LocalEmotionResult

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir, "AI_GLASSES_ASR_MODEL_DIR": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FailingTextEmotionAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "needs_timeline_recall": False,
                "confidence": 0.9,
                "reason": "wake query should use ambient context",
            })
            service = FakeService(
                store,
                agent=agent,
                asr_runner=FakeASRRunner(),
                emotion_runner=FakeEmotionRunner(
                    result=LocalEmotionResult(
                        enabled=True,
                        label="烦躁",
                        intensity=4,
                        score=0.2,
                        candidates=[{"label": "烦躁", "score": 0.2}],
                        model_name="emotion2vec_plus_base",
                        evidence="derived_from_acoustic_emotion_label: angry",
                        source="acoustic_emotion_model",
                        reason="acoustic_emotion_inferred",
                    )
                ),
                speaker_runner=FakeSpeakerRunner(),
            )
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            appended = service.process_audio_segment(
                user_id="u1",
                capture_id=capture["capture_id"],
                audio_base64=audio_base64,
                audio_mime_type="audio/wav",
                audio_duration_ms=1200,
            )
            chunk_id = appended["capture_append"]["chunk_id"]

            result = service.chat(
                "你觉得我刚才是不是不太高兴？",
                user_id="u1",
                ambient_capture_id=capture["capture_id"],
                wake_session={
                    "ambient_capture_id": capture["capture_id"],
                    "wake_detected_at": 1778131212.0,
                    "pre_wake_segment_ids": [chunk_id],
                    "post_wake_query_segment_ids": [],
                    "wake_query_text": "你觉得我刚才是不是不太高兴？",
                    "wake_mode": "button",
                    "status": "consumed",
                },
            )

        self.assertEqual(result["debug"]["ambient_context"]["text_emotion"]["backend"], "fallback")
        self.assertEqual(result["debug"]["ambient_context"]["emotion_fusion"]["label"], "unknown")
        self.assertFalse(result["debug"]["ambient_context"]["emotion_fusion"]["reply_emotion_used"])
        self.assertNotIn("<ambient-emotion-policy>", agent.main_messages[-1])

    def test_wake_session_uses_only_pre_wake_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "needs_timeline_recall": False,
                "confidence": 0.9,
                "reason": "wake session should only inject pre-wake context",
            })
            service = FakeService(store, agent=agent)
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            pre = service.append_capture_chunk(user_id="u1", capture_id=capture["capture_id"], text="他们还在吵预算")
            post = service.append_capture_chunk(user_id="u1", capture_id=capture["capture_id"], text="我现在该怎么回？")

            result = service.chat(
                "我现在该怎么回？",
                user_id="u1",
                ambient_capture_id=capture["capture_id"],
                wake_session={
                    "ambient_capture_id": capture["capture_id"],
                    "wake_detected_at": 1778131213.0,
                    "pre_wake_segment_ids": [pre["chunk_id"]],
                    "post_wake_query_segment_ids": [post["chunk_id"]],
                    "wake_query_text": "我现在该怎么回？",
                    "wake_mode": "button",
                    "status": "consumed",
                },
            )

            self.assertTrue(result["debug"]["ambient_context"]["used"])
            self.assertEqual(result["debug"]["ambient_context"]["pre_wake_segment_ids"], [pre["chunk_id"]])
            self.assertEqual(result["debug"]["ambient_context"]["post_wake_query_segment_ids"], [post["chunk_id"]])
            self.assertIn("他们还在吵预算", agent.main_messages[-1])
            ambient_block = agent.main_messages[-1].split("Recent ambient audio transcript before wake word:")[-1]
            self.assertNotIn("我现在该怎么回？", ambient_block)

    def test_wake_word_mode_uses_same_pre_wake_injection_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "needs_timeline_recall": False,
                "confidence": 0.9,
                "reason": "wake_word should reuse pre-wake window injection",
            })
            service = FakeService(store, agent=agent)
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="真实唤醒词 MVP")
            pre = service.append_capture_chunk(user_id="u1", capture_id=capture["capture_id"], text="他们还在吵预算")
            post = service.append_capture_chunk(user_id="u1", capture_id=capture["capture_id"], text="我现在该怎么回？")

            result = service.chat(
                "我现在该怎么回？",
                user_id="u1",
                ambient_capture_id=capture["capture_id"],
                wake_session={
                    "ambient_capture_id": capture["capture_id"],
                    "wake_detected_at": 1778131214.0,
                    "pre_wake_segment_ids": [pre["chunk_id"]],
                    "post_wake_query_segment_ids": [post["chunk_id"]],
                    "wake_query_text": "我现在该怎么回？",
                    "wake_mode": "wake_word",
                    "wake_detector_backend": "local_asr_phrase_detector",
                    "status": "consumed",
                    "timeout_seconds": 8,
                },
            )

            self.assertTrue(result["debug"]["ambient_context"]["used"])
            self.assertEqual(result["debug"]["ambient_context"]["wake_mode"], "wake_word")
            self.assertEqual(result["debug"]["ambient_context"]["wake_detector_backend"], "local_asr_phrase_detector")
            self.assertEqual(result["debug"]["ambient_context"]["wake_session"]["wake_detector_backend"], "local_asr_phrase_detector")
            self.assertEqual(result["debug"]["ambient_context"]["wake_session"]["timeout_seconds"], 8)
            ambient_block = agent.main_messages[-1].split("Recent ambient audio transcript before wake word:")[-1]
            self.assertIn("他们还在吵预算", ambient_block)
            self.assertNotIn("我现在该怎么回？", ambient_block)

    def test_wake_word_timeout_is_visible_in_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent()
            service = FakeService(store, agent=agent)
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="真实唤醒词 MVP")
            service.append_capture_chunk(user_id="u1", capture_id=capture["capture_id"], text="他们还在吵预算")

            result = service.chat(
                "现在几点？",
                user_id="u1",
                ambient_capture_id=capture["capture_id"],
                wake_session={
                    "ambient_capture_id": capture["capture_id"],
                    "wake_detected_at": 1778131215.0,
                    "pre_wake_segment_ids": [],
                    "post_wake_query_segment_ids": [],
                    "wake_query_text": "",
                    "wake_mode": "wake_word",
                    "wake_detector_backend": "local_asr_phrase_detector",
                    "status": "expired",
                    "expired": True,
                    "timeout_seconds": 8,
                },
            )

            self.assertFalse(result["debug"]["ambient_context"]["used"])
            self.assertEqual(result["debug"]["ambient_context"]["wake_mode"], "wake_word")
            self.assertEqual(result["debug"]["ambient_context"]["wake_detector_backend"], "local_asr_phrase_detector")
            self.assertEqual(result["debug"]["ambient_context"]["wake_session"]["status"], "expired")
            self.assertTrue(result["debug"]["ambient_context"]["wake_session"]["expired"])
            self.assertEqual(result["debug"]["ambient_context"]["wake_session"]["wake_detector_backend"], "local_asr_phrase_detector")
            self.assertEqual(result["debug"]["ambient_context"]["wake_session"]["timeout_seconds"], 8)

    def test_missing_pre_wake_segments_are_reported_in_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent()
            service = FakeService(store, agent=agent)
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            service.append_capture_chunk(user_id="u1", capture_id=capture["capture_id"], text="他们还在吵预算")

            result = service.chat(
                "我现在该怎么回？",
                user_id="u1",
                ambient_capture_id=capture["capture_id"],
                wake_session={
                    "ambient_capture_id": capture["capture_id"],
                    "wake_detected_at": 1778131213.0,
                    "pre_wake_segment_ids": ["chunk_expired_1"],
                    "post_wake_query_segment_ids": [],
                    "wake_query_text": "我现在该怎么回？",
                    "wake_mode": "button",
                    "status": "expired",
                    "expired": True,
                },
            )

            self.assertFalse(result["debug"]["ambient_context"]["used"])
            self.assertEqual(result["debug"]["ambient_context"]["skipped_reason"], "pre_wake_segments_missing")
            self.assertEqual(result["debug"]["ambient_context"]["expired_chunk_count"], 1)

    def test_ambient_capture_from_other_user_is_not_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent()
            service = FakeService(store, agent=agent)
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            service.append_capture_chunk(user_id="u1", capture_id=capture["capture_id"], text="隐私现场片段")

            result = service.chat("刚才发生了什么？", user_id="u2", ambient_capture_id=capture["capture_id"])

            self.assertFalse(result["debug"]["ambient_context"]["used"])
            self.assertEqual(result["debug"]["ambient_context"]["status"], "not_found")
            self.assertNotIn("隐私现场片段", json.dumps(agent.main_messages, ensure_ascii=False))

    def test_stopped_ambient_capture_is_not_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "needs_timeline_recall": False,
                "confidence": 0.9,
                "reason": "stopped ambient capture should not be scene context",
            })
            service = FakeService(store, agent=agent)
            capture = service.start_capture(user_id="u1", source="ambient_audio_text", context="按钮唤醒")
            service.append_capture_chunk(user_id="u1", capture_id=capture["capture_id"], text="旧现场片段")
            service.stop_capture(user_id="u1", capture_id=capture["capture_id"])

            result = service.chat("刚才发生了什么？", user_id="u1", ambient_capture_id=capture["capture_id"])

            self.assertFalse(result["debug"]["ambient_context"]["used"])
            self.assertEqual(result["debug"]["ambient_context"]["status"], "stopped")
            self.assertNotIn("旧现场片段", agent.main_messages[-1])

    def test_fuzzy_current_time_query_uses_pre_reply_decision(self) -> None:
        cases = ("几点啦", "报个时", "今儿几号")
        for message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    agent = FakeAgent(pre_reply_payload={
                        "reply_mode": "local_current_time",
                        "answer_source": "local_clock",
                        "scope": "device_local",
                        "location_text": "",
                        "confidence": 0.9,
                        "reason": "colloquial local time query",
                    })
                    service = FakeService(self.make_store(Path(tmpdir)), agent=agent)

                    result = service.chat(message, user_id="u1")

                    self.assertIn("2026年5月7日", result["reply"])
                    self.assertEqual(result["debug"]["pre_reply_decision"]["answer_source"], "local_clock")
                    self.assertEqual(agent.pre_reply_calls, 1)
                    self.assertGreaterEqual(agent.semantic_calls, 1)
                    self.assertEqual(agent.main_calls, 0)

    def test_named_place_time_query_is_not_answered_by_main_llm(self) -> None:
        cases = ("巴黎现在几点", "纽约现在几点")
        for message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    agent = FakeAgent(pre_reply_payload={
                        "reply_mode": "unsupported_world_time",
                        "answer_source": "world_time",
                        "scope": "named_place",
                        "location_text": message[:2],
                        "confidence": 0.93,
                        "reason": "asks current time for a named place",
                    })
                    service = FakeService(self.make_store(Path(tmpdir)), agent=agent)

                    result = service.chat(message, user_id="u1")

                    self.assertIn("暂不支持按城市或地区换算时间", result["reply"])
                    self.assertNotIn("2026年", result["reply"])
                    self.assertEqual(result["debug"]["pre_reply_decision"]["reply_mode"], "unsupported_world_time")
                    self.assertEqual(result["debug"]["pre_reply_decision"]["answer_source"], "world_time")
                    self.assertEqual(agent.pre_reply_calls, 1)
                    self.assertGreaterEqual(agent.semantic_calls, 1)
                    self.assertEqual(agent.main_calls, 0)

    def test_low_confidence_or_invalid_pre_reply_decision_falls_back_to_main_llm(self) -> None:
        cases = (
            {
                "reply_mode": "local_current_time",
                "answer_source": "local_clock",
                "scope": "device_local",
                "location_text": "",
                "confidence": 0.4,
                "reason": "too uncertain",
            },
            {
                "reply_mode": "answer_directly",
                "answer_source": "local_clock",
                "scope": "device_local",
                "location_text": "",
                "confidence": 0.95,
                "reason": "invalid enum",
            },
            "not json",
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    agent = FakeAgent(pre_reply_payload=payload)
                    service = FakeService(self.make_store(Path(tmpdir)), agent=agent)

                    result = service.chat("报个时", user_id="u1")

                    self.assertEqual(result["reply"], "主回复")
                    self.assertEqual(result["debug"]["llm"]["api_calls"], 1)
                    self.assertEqual(agent.pre_reply_calls, 1)
                    self.assertGreaterEqual(agent.semantic_calls, 1)
                    self.assertEqual(agent.main_calls, 1)
                    self.assertTrue(result["debug"]["pre_reply_decision"].get("error"))

    def test_low_confidence_memory_pre_reply_does_not_open_profile_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "我是 XREAL 的员工", kind="profile")
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "needs_event_memory": False,
                "needs_timeline_recall": False,
                "timeline_query": None,
                "confidence": 0.4,
                "reason": "too uncertain",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("关于我", user_id="u1")

            self.assertEqual(result["reply"], "主回复")
            self.assertEqual(result["debug"]["memory"]["profile_count"], 0)
            self.assertEqual(result["debug"]["llm"]["api_calls"], 1)
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(agent.main_calls, 1)
            self.assertEqual(
                result["debug"]["pre_reply_decision"]["error"],
                "confidence_below_threshold",
            )

    def test_llm_first_uses_pre_reply_intent_and_main_llm_for_summary_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户名字叫Jack", kind="profile")
            store.add_memory("u1", "下午去吃面", kind="event")
            store.add_memory(
                "u1",
                "用户最近在整理 AI 眼镜长期记忆 demo",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
            )
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_profile_memory": True,
                "needs_event_memory": True,
                "needs_timeline_recall": False,
                "memory_recall_type": "observation",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "summarize remembered history",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("我之前都给你说过什么", user_id="u1", routing_mode="llm_first")

            self.assertIn("用户最近在整理 AI 眼镜长期记忆 demo", result["reply"])
            self.assertEqual(result["debug"]["routing"]["mode"], "llm_first")
            self.assertNotIn("planner_enabled", result["debug"]["routing"])
            self.assertTrue(result["debug"]["routing"]["pre_reply_decision_applied"])
            self.assertEqual(result["debug"]["pre_reply_decision"]["recall_goal"], "summary")
            self.assertEqual(result["debug"]["answer_directive"]["backend"], "llm")
            self.assertGreaterEqual(result["debug"]["memory"]["profile_count"], 1)
            self.assertGreaterEqual(result["debug"]["memory"]["event_recall_count"], 1)
            self.assertIn("<memory-context>", agent.main_messages[-1])
            self.assertIn("<answer-directive>", agent.main_messages[-1])
            self.assertNotIn("没有找到之前相关的原话", result["reply"])
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(agent.answer_calls, 1)
            self.assertEqual(agent.main_calls, 1)

    def test_llm_first_recent_activity_summary_separates_background_from_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_profile_memory": True,
                    "needs_event_memory": True,
                    "needs_timeline_recall": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "summarize recent activity",
                },
                answer_payload={
                    "answer_intent": "summary",
                    "organization": "thematic",
                    "evidence_policy": "separate_background",
                    "filtering_rules": ["Do not present stable identity facts as recent activities."],
                    "uncertainty_policy": "state_limits_when_context_is_sparse",
                    "style": "concise_structured",
                    "confidence": 0.9,
                    "reason": "recent activity summary needs separation",
                },
            ))
            service.timeline_store.add_turn("u1", "今天下午去吃面", created_at=1778120000.0)
            service.timeline_store.add_turn("u1", "晚上有唱 K 活动", created_at=1778123600.0)
            source_a = store.add_memory("u1", "今天下午去吃面", kind="event", evidence_ids=["chunk_missing_a"])
            source_b = store.add_memory("u1", "晚上有唱 K 活动", kind="event", evidence_ids=["chunk_missing_b"])
            store.add_memory("u1", "用户名字叫Jack", kind="profile")
            store.add_memory(
                "u1",
                "用户去吃了面，有一份关于自驾游的文档，名字叫Jack。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=[*source_a.evidence_ids, *source_b.evidence_ids],
            )

            result = service.chat("我最近都干了什么", user_id="u1", routing_mode="llm_first")
            message = service.fake_agent.main_messages[-1]

            self.assertEqual(result["debug"]["answer_directive"]["answer_intent"], "summary")
            self.assertEqual(result["debug"]["answer_directive"]["evidence_policy"], "separate_background")
            self.assertIn("Direct structured memory evidence", message)
            self.assertIn("Reflected observations", message)
            self.assertIn("Do not present stable identity facts as recent activities", message)
            self.assertIn("今天下午去吃面", message)
            self.assertIn("晚上有唱 K 活动", message)
            self.assertNotIn("用户名字叫Jack", message)
            self.assertEqual(result["debug"]["memory"]["recall_arbitration"]["dropped_counts"]["profile"], 1)
            self.assertEqual(service.fake_agent.answer_calls, 1)
            self.assertEqual(service.fake_agent.main_calls, 1)

    def test_recall_arbitration_document_detail_wins_over_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            document = store.add_document(
                "u1",
                filename="南太行自驾攻略.md",
                title="南太行自驾攻略",
                summary="南太行自驾攻略；包含费用和路线",
                content="# 南太行自驾攻略\n## 费用\n红旗渠门票 80 元\n## 路线\n第一天从林州出发。",
                source="markdown_upload",
                ingestion_id="ing-doc",
                created_at=1778120000.0,
            )
            store.add_memory(
                "u1",
                "用户最近在整理南太行项目，但 observation 不能替代文档细节。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_obs"],
            )
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "observation",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "document question with possible memory overlap",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("南太行自驾攻略里红旗渠门票多少钱？", user_id="u1", routing_mode="llm_first")
            message = agent.main_messages[-1]
            arbitration = result["debug"]["memory"]["recall_arbitration"]

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "full_document")
            self.assertEqual(arbitration["primary_source"], "document")
            self.assertEqual(arbitration["primary_source_reason"], "document_detail_primary")
            self.assertEqual(arbitration["dropped_counts"]["observation"], 1)
            self.assertEqual(result["recalled_memories"], [])
            self.assertEqual(result["recalled_documents"][0]["id"], document.id)
            self.assertEqual(result["source_summary"]["primary_source"], "document")
            self.assertIn("上传文档原文", result["source_summary"]["primary_source_explanation"])
            self.assertIn("红旗渠门票 80 元", message)
            self.assertNotIn("observation 不能替代文档细节", message)
            self.assertEqual(agent.main_calls, 1)

    def test_recall_arbitration_project_state_summary_keeps_structured_memory_over_document_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="AI 眼镜项目纪要.md",
                title="AI 眼镜项目纪要",
                summary="AI 眼镜项目纪要",
                content="# AI 眼镜项目纪要\n项目背景：周报不能只看聊天记忆，需要保留文档依据。",
                source="markdown_upload",
                ingestion_id="ing-doc",
                created_at=1778120000.0,
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目当前主线是收敛 LLM-first 文字输入，先压 recent context 污染，再补文本清洗 Phase C。",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_state"],
            )
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "observation",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "asks for current project status with recent document reference",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？", user_id="u1", routing_mode="llm_first")
            message = agent.main_messages[-1]
            arbitration = result["debug"]["memory"]["recall_arbitration"]

            self.assertEqual(result["debug"]["document_recall"]["strategy"], "full_document")
            self.assertEqual(arbitration["primary_source"], "structured_memory")
            self.assertEqual(
                arbitration["primary_source_reason"],
                "summary_structured_memory_primary_with_document_background",
            )
            self.assertEqual(result["source_summary"]["primary_source"], "structured_memory")
            self.assertIn("文档只作为背景补充", result["source_summary"]["primary_source_explanation"])
            self.assertIn("LLM-first 文字输入", message)
            self.assertIn("AI 眼镜项目纪要", message)
            self.assertEqual(agent.main_calls, 1)

    def test_recall_arbitration_raw_timeline_wins_over_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "needs_timeline_recall": True,
                "memory_recall_type": "observation",
                "recall_goal": "raw_evidence",
                "timeline_query": "语音识别 不稳定",
                "confidence": 0.95,
                "reason": "exact wording requested",
            }))
            service.timeline_store.add_turn(
                "u1",
                "我提到国内网络下语音识别不稳定，可能需要 VPN 或备用方案。",
                created_at=1778120000.0,
            )
            store.add_memory(
                "u1",
                "用户关注语音识别稳定性。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_obs"],
            )

            result = service.chat("我之前有没有说过语音识别不稳定的原话？", user_id="u1", routing_mode="llm_first")
            arbitration = result["debug"]["memory"]["recall_arbitration"]

            self.assertEqual(arbitration["primary_source"], "raw_timeline")
            self.assertEqual(arbitration["primary_source_reason"], "raw_timeline_primary")
            self.assertEqual(arbitration["dropped_counts"]["observation"], 1)
            self.assertEqual(result["recalled_memories"], [])
            self.assertEqual(len(result["recalled_timeline_chunks"]), 1)
            self.assertEqual(result["source_summary"]["primary_source"], "raw_timeline")
            self.assertIn("timeline 原话", result["source_summary"]["primary_source_explanation"])
            self.assertIn("语音识别不稳定", result["reply"])
            self.assertEqual(service.fake_agent.main_calls, 0)

    def test_explanation_reply_uses_document_source_summary_from_previous_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="南太行自驾攻略.md",
                title="南太行自驾攻略",
                summary="南太行自驾攻略；包含费用和路线",
                content="# 南太行自驾攻略\n## 费用\n红旗渠门票 80 元\n## 路线\n第一天从林州出发。",
                source="markdown_upload",
                ingestion_id="ing-doc",
                created_at=1778120000.0,
            )
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "observation",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "document question with possible memory overlap",
            })
            service = FakeService(store, agent=agent)

            first = service.chat("南太行自驾攻略里红旗渠门票多少钱？", user_id="u1", routing_mode="llm_first")
            explain = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "document")
            self.assertIn("上传文档原文", explain["reply"])
            self.assertIn("最近上下文这轮没有参与主回答", explain["reply"])
            self.assertEqual(service.fake_agent.main_calls, 1)

    def test_explanation_reply_reuses_original_business_turn_across_repeated_followups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="南太行自驾攻略.md",
                title="南太行自驾攻略",
                summary="南太行自驾攻略；包含费用和路线",
                content="# 南太行自驾攻略\n## 费用\n红旗渠门票 80 元\n## 路线\n第一天从林州出发。",
                source="markdown_upload",
                ingestion_id="ing-doc",
                created_at=1778120000.0,
            )
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "observation",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "document question with possible memory overlap",
            })
            service = FakeService(store, agent=agent)

            first = service.chat("南太行自驾攻略里红旗渠门票多少钱？", user_id="u1", routing_mode="llm_first")
            explain_once = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            explain_twice = service.chat("再具体说一下依据是什么？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "document")
            self.assertIn("上传文档原文", explain_once["reply"])
            self.assertIn("上传文档原文", explain_twice["reply"])
            self.assertEqual(explain_twice["debug"]["source_summary"]["primary_source"], "document")
            self.assertEqual(service.fake_agent.main_calls, 1)

    def test_explanation_reply_preserves_recent_context_participation_across_chitchat_detour(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户刚导入一段关于新增监管措施的材料，包含账户关闭和资金来源声明要求。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_policy_obs"],
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "想一想这个新增监管措施的事情。": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "recent material reference",
                },
                "哎今天真有点累，路上人好多。": {
                    "reply_mode": "fast",
                    "answer_source": "local",
                    "scope": "daily",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "daily emotional chitchat",
                },
                "哈哈刚才同事讲了个冷笑话。": {
                    "reply_mode": "fast",
                    "answer_source": "local",
                    "scope": "daily",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "lightweight social chitchat",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我刚导入一段关于某政策新增监管措施的材料，提到了账户关闭和资金来源声明要求。",
                created_at=1778131100.0,
            )

            first = service.chat("想一想这个新增监管措施的事情。", user_id="u1", routing_mode="llm_first")
            second = service.chat("哎今天真有点累，路上人好多。", user_id="u1", routing_mode="llm_first")
            third = service.chat("哈哈刚才同事讲了个冷笑话。", user_id="u1", routing_mode="llm_first")
            explain = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertTrue(first["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(first["debug"]["recent_context_capsule"]["injection_reason"], "summary_reference_signal")
            self.assertEqual(first["source_summary"]["primary_source"], "observation")
            self.assertEqual(second["source_summary"]["primary_source"], "none")
            self.assertEqual(third["source_summary"]["primary_source"], "none")
            self.assertIn("观察总结", explain["reply"])
            self.assertIn("最近上下文这轮有参与主回答", explain["reply"])
            self.assertIn("summary_reference_signal", explain["reply"])
            self.assertNotIn("当前没有可用来源", explain["reply"])
            self.assertEqual(explain["debug"]["source_summary"]["primary_source"], "observation")

    def test_explanation_reply_switches_to_latest_business_turn_after_intermediate_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="AI 眼镜项目纪要.md",
                title="AI 眼镜项目纪要",
                summary="项目背景和本周会议结论",
                content="# AI 眼镜项目纪要\n项目背景：周报不能只看聊天记忆，需要保留文档依据。\n说明：这份文档主要解释背景，不直接代表当前主线状态。",
                source="markdown_upload",
                ingestion_id="ing-project",
                created_at=1778120300.0,
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目当前主线是收敛 LLM-first 文字输入，先压 recent context 污染，再补文本清洗 Phase C。",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_text_scope"],
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "我之前有没有说过语音识别不稳定的原话？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "needs_timeline_recall": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "raw_evidence",
                    "timeline_query": "语音识别 不稳定",
                    "confidence": 0.95,
                    "reason": "exact wording requested",
                },
                "这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "asks for current project status with recent document reference",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我提到国内网络下语音识别不稳定，可能需要 VPN 或备用方案。",
                created_at=1778120000.0,
            )

            first = service.chat("我之前有没有说过语音识别不稳定的原话？", user_id="u1", routing_mode="llm_first")
            explain_first = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            second = service.chat(
                "这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？",
                user_id="u1",
                routing_mode="llm_first",
            )
            explain_second = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "raw_timeline")
            self.assertIn("timeline 原话", explain_first["reply"])
            self.assertEqual(second["source_summary"]["primary_source"], "structured_memory")
            self.assertIn("结构化任务", explain_second["reply"])
            self.assertIn("文档只作为背景补充", explain_second["reply"])
            self.assertNotIn("timeline 原话", explain_second["reply"])
            self.assertEqual(explain_second["debug"]["source_summary"]["primary_source"], "structured_memory")
            self.assertEqual(service.fake_agent.main_calls, 1)

    def test_explanation_reply_does_not_fallback_to_older_source_when_latest_business_turn_has_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "我之前有没有说过语音识别不稳定的原话？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "needs_timeline_recall": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "raw_evidence",
                    "timeline_query": "语音识别 不稳定",
                    "confidence": 0.95,
                    "reason": "exact wording requested",
                },
                "普通问答：水的化学式是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary factual question",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我提到国内网络下语音识别不稳定，可能需要 VPN 或备用方案。",
                created_at=1778120000.0,
            )

            first = service.chat("我之前有没有说过语音识别不稳定的原话？", user_id="u1", routing_mode="llm_first")
            explain_first = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            second = service.chat("普通问答：水的化学式是什么？", user_id="u1", routing_mode="llm_first")
            explain_second = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "raw_timeline")
            self.assertIn("timeline 原话", explain_first["reply"])
            self.assertEqual(second["source_summary"]["primary_source"], "none")
            self.assertIn("当前没有可用来源", explain_second["reply"])
            self.assertNotIn("timeline 原话", explain_second["reply"])
            self.assertEqual(explain_second["debug"]["source_summary"]["primary_source"], "none")
            self.assertEqual(service.fake_agent.main_calls, 1)

    def test_explanation_reply_does_not_fallback_to_recent_context_summary_when_latest_business_turn_has_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户刚导入一段关于新增监管措施的材料，包含账户关闭和资金来源声明要求。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_policy_obs"],
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "想一想这个新增监管措施的事情。": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "recent material reference",
                },
                "普通问答：水的化学式是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary factual question",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我刚导入一段关于某政策新增监管措施的材料，提到了账户关闭和资金来源声明要求。",
                created_at=1778131100.0,
            )

            first = service.chat("想一想这个新增监管措施的事情。", user_id="u1", routing_mode="llm_first")
            second = service.chat("普通问答：水的化学式是什么？", user_id="u1", routing_mode="llm_first")
            explain = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "observation")
            self.assertTrue(first["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(first["debug"]["recent_context_capsule"]["injection_reason"], "summary_reference_signal")
            self.assertEqual(second["source_summary"]["primary_source"], "none")
            self.assertIn("当前没有可用来源", explain["reply"])
            self.assertNotIn("观察总结", explain["reply"])
            self.assertNotIn("summary_reference_signal", explain["reply"])
            self.assertEqual(explain["debug"]["source_summary"]["primary_source"], "none")
            self.assertEqual(service.fake_agent.main_calls, 2)

    def test_explanation_reply_does_not_reuse_recent_context_after_prior_explanation_when_latest_business_turn_has_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户刚导入一段关于新增监管措施的材料，包含账户关闭和资金来源声明要求。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_policy_obs"],
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "想一想这个新增监管措施的事情。": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "recent material reference",
                },
                "普通问答：水的化学式是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary factual question",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我刚导入一段关于某政策新增监管措施的材料，提到了账户关闭和资金来源声明要求。",
                created_at=1778131100.0,
            )

            first = service.chat("想一想这个新增监管措施的事情。", user_id="u1", routing_mode="llm_first")
            explain_first = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            second = service.chat("普通问答：水的化学式是什么？", user_id="u1", routing_mode="llm_first")
            explain_second = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "observation")
            self.assertTrue(first["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(first["debug"]["recent_context_capsule"]["injection_reason"], "summary_reference_signal")
            self.assertIn("观察总结", explain_first["reply"])
            self.assertIn("summary_reference_signal", explain_first["reply"])
            self.assertEqual(explain_first["debug"]["source_summary"]["primary_source"], "observation")
            self.assertEqual(second["source_summary"]["primary_source"], "none")
            self.assertIn("当前没有可用来源", explain_second["reply"])
            self.assertNotIn("观察总结", explain_second["reply"])
            self.assertNotIn("summary_reference_signal", explain_second["reply"])
            self.assertEqual(explain_second["debug"]["source_summary"]["primary_source"], "none")
            self.assertEqual(service.fake_agent.main_calls, 2)

    def test_explanation_reply_switches_from_recent_context_summary_to_latest_document_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户刚导入一段关于新增监管措施的材料，包含账户关闭和资金来源声明要求。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_policy_obs"],
            )
            store.add_document(
                "u1",
                filename="南太行自驾攻略.md",
                title="南太行自驾攻略",
                summary="南太行自驾攻略；包含费用和路线",
                content="# 南太行自驾攻略\n## 费用\n红旗渠门票 80 元\n## 路线\n第一天从林州出发。",
                source="markdown_upload",
                ingestion_id="ing-doc",
                created_at=1778120000.0,
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "想一想这个新增监管措施的事情。": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "recent material reference",
                },
                "南太行自驾攻略里红旗渠门票多少钱？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "document question with possible memory overlap",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我刚导入一段关于某政策新增监管措施的材料，提到了账户关闭和资金来源声明要求。",
                created_at=1778131100.0,
            )

            first = service.chat("想一想这个新增监管措施的事情。", user_id="u1", routing_mode="llm_first")
            explain_first = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            second = service.chat("南太行自驾攻略里红旗渠门票多少钱？", user_id="u1", routing_mode="llm_first")
            explain_second = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "observation")
            self.assertTrue(first["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(first["debug"]["recent_context_capsule"]["injection_reason"], "summary_reference_signal")
            self.assertIn("观察总结", explain_first["reply"])
            self.assertEqual(explain_first["debug"]["source_summary"]["primary_source"], "observation")
            self.assertEqual(second["source_summary"]["primary_source"], "document")
            self.assertEqual(second["debug"]["document_recall"]["strategy"], "full_document")
            self.assertEqual(len(second["recalled_documents"]), 1)
            self.assertEqual(second["recalled_documents"][0]["filename"], "南太行自驾攻略.md")
            self.assertIn("上传文档原文", explain_second["reply"])
            self.assertNotIn("观察总结", explain_second["reply"])
            self.assertNotIn("summary_reference_signal", explain_second["reply"])
            self.assertEqual(explain_second["debug"]["source_summary"]["primary_source"], "document")
            self.assertEqual(service.fake_agent.main_calls, 2)

    def test_explanation_reply_switches_from_recent_context_summary_to_latest_structured_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户刚导入一段关于新增监管措施的材料，包含账户关闭和资金来源声明要求。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_policy_obs"],
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目当前主线是收敛 LLM-first 文字输入，先压 recent context 污染，再补文本清洗 Phase C。",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_text_scope"],
            )
            store.add_document(
                "u1",
                filename="AI 眼镜项目纪要.md",
                title="AI 眼镜项目纪要",
                summary="项目背景和本周会议结论",
                content="# AI 眼镜项目纪要\n项目背景：周报不能只看聊天记忆，需要保留文档依据。\n说明：这份文档主要解释背景，不直接代表当前主线状态。",
                source="markdown_upload",
                ingestion_id="ing-project",
                created_at=1778120300.0,
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "想一想这个新增监管措施的事情。": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "recent material reference",
                },
                "这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "asks for current project status with recent document reference",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我刚导入一段关于某政策新增监管措施的材料，提到了账户关闭和资金来源声明要求。",
                created_at=1778131100.0,
            )

            first = service.chat("想一想这个新增监管措施的事情。", user_id="u1", routing_mode="llm_first")
            explain_first = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            second = service.chat(
                "这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？",
                user_id="u1",
                routing_mode="llm_first",
            )
            explain_second = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "observation")
            self.assertTrue(first["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(first["debug"]["recent_context_capsule"]["injection_reason"], "summary_reference_signal")
            self.assertIn("观察总结", explain_first["reply"])
            self.assertEqual(explain_first["debug"]["source_summary"]["primary_source"], "observation")
            self.assertEqual(second["source_summary"]["primary_source"], "structured_memory")
            self.assertEqual(len(second["recalled_documents"]), 1)
            self.assertEqual(second["recalled_documents"][0]["filename"], "AI 眼镜项目纪要.md")
            self.assertIn("结构化任务", explain_second["reply"])
            self.assertIn("文档只作为背景补充", explain_second["reply"])
            self.assertNotIn("观察总结", explain_second["reply"])
            self.assertNotIn("summary_reference_signal", explain_second["reply"])
            self.assertEqual(explain_second["debug"]["source_summary"]["primary_source"], "structured_memory")
            self.assertEqual(service.fake_agent.main_calls, 2)

    def test_explanation_reply_switches_from_recent_context_summary_to_latest_raw_timeline_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户刚导入一段关于新增监管措施的材料，包含账户关闭和资金来源声明要求。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_policy_obs"],
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "想一想这个新增监管措施的事情。": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "recent material reference",
                },
                "我之前有没有说过语音识别不稳定的原话？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "needs_timeline_recall": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "raw_evidence",
                    "timeline_query": "语音识别 不稳定",
                    "confidence": 0.95,
                    "reason": "exact wording requested",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我刚导入一段关于某政策新增监管措施的材料，提到了账户关闭和资金来源声明要求。",
                created_at=1778131100.0,
            )
            service.timeline_store.add_turn(
                "u1",
                "我提到国内网络下语音识别不稳定，可能需要 VPN 或备用方案。",
                created_at=1778120000.0,
            )

            first = service.chat("想一想这个新增监管措施的事情。", user_id="u1", routing_mode="llm_first")
            explain_first = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            second = service.chat("我之前有没有说过语音识别不稳定的原话？", user_id="u1", routing_mode="llm_first")
            explain_second = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "observation")
            self.assertTrue(first["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(first["debug"]["recent_context_capsule"]["injection_reason"], "summary_reference_signal")
            self.assertIn("观察总结", explain_first["reply"])
            self.assertEqual(explain_first["debug"]["source_summary"]["primary_source"], "observation")
            self.assertEqual(second["source_summary"]["primary_source"], "raw_timeline")
            self.assertIn("语音识别不稳定", second["reply"])
            self.assertIn("timeline 原话", explain_second["reply"])
            self.assertNotIn("观察总结", explain_second["reply"])
            self.assertNotIn("summary_reference_signal", explain_second["reply"])
            self.assertEqual(explain_second["debug"]["source_summary"]["primary_source"], "raw_timeline")
            self.assertIn(service.fake_agent.main_calls, {1, 2})

    def test_explanation_reply_recent_context_mixed_chain_always_follows_latest_business_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户刚导入一段关于新增监管措施的材料，包含账户关闭和资金来源声明要求。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=["chunk_policy_obs"],
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目当前主线是收敛 LLM-first 文字输入，先压 recent context 污染，再补文本清洗 Phase C。",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_text_scope"],
            )
            store.add_document(
                "u1",
                filename="AI 眼镜项目纪要.md",
                title="AI 眼镜项目纪要",
                summary="项目背景和本周会议结论",
                content="# AI 眼镜项目纪要\n项目背景：周报不能只看聊天记忆，需要保留文档依据。\n说明：这份文档主要解释背景，不直接代表当前主线状态。",
                source="markdown_upload",
                ingestion_id="ing-project",
                created_at=1778120300.0,
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "想一想这个新增监管措施的事情。": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "recent material reference",
                },
                "普通问答：水的化学式是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary factual question",
                },
                "我之前有没有说过语音识别不稳定的原话？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "needs_timeline_recall": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "raw_evidence",
                    "timeline_query": "语音识别 不稳定",
                    "confidence": 0.95,
                    "reason": "exact wording requested",
                },
                "这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "asks for current project status with recent document reference",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我刚导入一段关于某政策新增监管措施的材料，提到了账户关闭和资金来源声明要求。",
                created_at=1778131100.0,
            )
            service.timeline_store.add_turn(
                "u1",
                "我提到国内网络下语音识别不稳定，可能需要 VPN 或备用方案。",
                created_at=1778120000.0,
            )

            recent_context_turn = service.chat("想一想这个新增监管措施的事情。", user_id="u1", routing_mode="llm_first")
            explain_recent_context = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            none_turn = service.chat("普通问答：水的化学式是什么？", user_id="u1", routing_mode="llm_first")
            explain_none = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")
            timeline_turn = service.chat("我之前有没有说过语音识别不稳定的原话？", user_id="u1", routing_mode="llm_first")
            explain_timeline = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")
            structured_turn = service.chat(
                "这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？",
                user_id="u1",
                routing_mode="llm_first",
            )
            explain_structured = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(recent_context_turn["source_summary"]["primary_source"], "observation")
            self.assertTrue(recent_context_turn["debug"]["recent_context_capsule"]["injected_to_main_llm"])
            self.assertEqual(
                recent_context_turn["debug"]["recent_context_capsule"]["injection_reason"],
                "summary_reference_signal",
            )
            self.assertIn("观察总结", explain_recent_context["reply"])
            self.assertIn("summary_reference_signal", explain_recent_context["reply"])
            self.assertEqual(explain_recent_context["debug"]["source_summary"]["primary_source"], "observation")

            self.assertEqual(none_turn["source_summary"]["primary_source"], "none")
            self.assertIn("当前没有可用来源", explain_none["reply"])
            self.assertNotIn("观察总结", explain_none["reply"])
            self.assertNotIn("summary_reference_signal", explain_none["reply"])
            self.assertEqual(explain_none["debug"]["source_summary"]["primary_source"], "none")

            self.assertEqual(timeline_turn["source_summary"]["primary_source"], "raw_timeline")
            self.assertIn("语音识别不稳定", timeline_turn["reply"])
            self.assertIn("timeline 原话", explain_timeline["reply"])
            self.assertNotIn("观察总结", explain_timeline["reply"])
            self.assertNotIn("summary_reference_signal", explain_timeline["reply"])
            self.assertEqual(explain_timeline["debug"]["source_summary"]["primary_source"], "raw_timeline")

            self.assertEqual(structured_turn["source_summary"]["primary_source"], "structured_memory")
            self.assertEqual(len(structured_turn["recalled_documents"]), 1)
            self.assertEqual(structured_turn["recalled_documents"][0]["filename"], "AI 眼镜项目纪要.md")
            self.assertIn("结构化任务", explain_structured["reply"])
            self.assertIn("文档只作为背景补充", explain_structured["reply"])
            self.assertNotIn("观察总结", explain_structured["reply"])
            self.assertNotIn("summary_reference_signal", explain_structured["reply"])
            self.assertNotIn("timeline 原话", explain_structured["reply"])
            self.assertEqual(explain_structured["debug"]["source_summary"]["primary_source"], "structured_memory")


    def test_explanation_reply_switches_to_latest_document_turn_after_prior_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="南太行自驾攻略.md",
                title="南太行自驾攻略",
                summary="南太行自驾攻略；包含费用和路线",
                content="# 南太行自驾攻略\n## 费用\n红旗渠门票 80 元\n## 路线\n第一天从林州出发。",
                source="markdown_upload",
                ingestion_id="ing-doc",
                created_at=1778120000.0,
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "南太行自驾攻略里红旗渠门票多少钱？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "document question with possible memory overlap",
                },
                "那路线怎么写的？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "document route follow-up question",
                },
            }))

            first = service.chat("南太行自驾攻略里红旗渠门票多少钱？", user_id="u1", routing_mode="llm_first")
            explain_first = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            second = service.chat("那路线怎么写的？", user_id="u1", routing_mode="llm_first")
            explain_second = service.chat("那这次依据是什么？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "document")
            self.assertIn("上传文档原文", explain_first["reply"])
            self.assertEqual(second["source_summary"]["primary_source"], "document")
            self.assertEqual(second["debug"]["document_recall"]["reason"], "recent_document_followup")
            self.assertEqual(second["debug"]["document_recall"]["strategy"], "sections")
            self.assertIn("## 路线", service.fake_agent.main_messages[-1])
            self.assertIn("第一天从林州出发", service.fake_agent.main_messages[-1])
            self.assertNotIn("红旗渠门票 80 元", service.fake_agent.main_messages[-1])
            self.assertIn("上传文档原文", explain_second["reply"])
            self.assertEqual(explain_second["debug"]["source_summary"]["primary_source"], "document")
            self.assertEqual(service.fake_agent.main_calls, 2)



    def test_document_followup_can_resume_same_document_after_none_detour(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="南太行自驾攻略.md",
                title="南太行自驾攻略",
                summary="南太行自驾攻略；包含费用和路线",
                content="# 南太行自驾攻略\n## 费用\n红旗渠门票 80 元\n## 路线\n第一天从林州出发。",
                source="markdown_upload",
                ingestion_id="ing-doc",
                created_at=1778120000.0,
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "南太行自驾攻略里红旗渠门票多少钱？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "document question with possible memory overlap",
                },
                "普通问答：水的化学式是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary factual question",
                },
                "那路线怎么写的？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "document route follow-up question",
                },
            }))

            first = service.chat("南太行自驾攻略里红旗渠门票多少钱？", user_id="u1", routing_mode="llm_first")
            explain_first = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            second = service.chat("普通问答：水的化学式是什么？", user_id="u1", routing_mode="llm_first")
            third = service.chat("那路线怎么写的？", user_id="u1", routing_mode="llm_first")
            explain_third = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "document")
            self.assertIn("上传文档原文", explain_first["reply"])
            self.assertEqual(second["source_summary"]["primary_source"], "none")
            self.assertEqual(third["source_summary"]["primary_source"], "document")
            self.assertEqual(third["debug"]["document_recall"]["reason"], "recent_document_followup")
            self.assertEqual(third["debug"]["document_recall"]["strategy"], "sections")
            self.assertIn("## 路线", service.fake_agent.main_messages[-1])
            self.assertIn("第一天从林州出发", service.fake_agent.main_messages[-1])
            self.assertNotIn("红旗渠门票 80 元", service.fake_agent.main_messages[-1])
            self.assertIn("上传文档原文", explain_third["reply"])
            self.assertNotIn("当前没有可用来源", explain_third["reply"])
            self.assertEqual(explain_third["debug"]["source_summary"]["primary_source"], "document")
            self.assertEqual(service.fake_agent.main_calls, 3)

    def test_explanation_reply_switches_from_document_turn_to_latest_raw_timeline_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="AI 眼镜项目纪要.md",
                title="AI 眼镜项目纪要",
                summary="项目背景和本周会议结论",
                content="# AI 眼镜项目纪要\n项目背景：周报不能只看聊天记忆，需要保留文档依据。\n说明：这份文档主要解释背景，不直接代表当前主线状态。",
                source="markdown_upload",
                ingestion_id="ing-project",
                created_at=1778120300.0,
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "AI 眼镜项目纪要里写了什么背景说明？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "document background detail question",
                },
                "我之前有没有说过语音识别不稳定的原话？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "needs_timeline_recall": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "raw_evidence",
                    "timeline_query": "语音识别 不稳定",
                    "confidence": 0.95,
                    "reason": "exact wording requested",
                },
            }))
            service.timeline_store.add_turn(
                "u1",
                "我提到国内网络下语音识别不稳定，可能需要 VPN 或备用方案。",
                created_at=1778120000.0,
            )

            first = service.chat("AI 眼镜项目纪要里写了什么背景说明？", user_id="u1", routing_mode="llm_first")
            explain_first = service.chat("你为什么这么说？", user_id="u1", routing_mode="llm_first")
            second = service.chat("我之前有没有说过语音识别不稳定的原话？", user_id="u1", routing_mode="llm_first")
            explain_second = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "document")
            self.assertEqual(first["debug"]["source_summary"]["primary_source"], "document")
            self.assertIn("上传文档原文", explain_first["reply"])
            self.assertEqual(second["source_summary"]["primary_source"], "raw_timeline")
            self.assertIn("语音识别不稳定", second["reply"])
            self.assertIn("timeline 原话", explain_second["reply"])
            self.assertNotIn("上传文档原文", explain_second["reply"])
            self.assertEqual(explain_second["debug"]["source_summary"]["primary_source"], "raw_timeline")

    def test_explanation_reply_switches_from_structured_turn_to_latest_document_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="AI 眼镜项目纪要.md",
                title="AI 眼镜项目纪要",
                summary="项目背景和本周会议结论",
                content="# AI 眼镜项目纪要\n项目背景：周报不能只看聊天记忆，需要保留文档依据。\n说明：这份文档主要解释背景，不直接代表当前主线状态。",
                source="markdown_upload",
                ingestion_id="ing-project",
                created_at=1778120300.0,
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目当前主线是收敛 LLM-first 文字输入，先压 recent context 污染，再补文本清洗 Phase C。",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_text_scope"],
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "asks for current project status with recent document reference",
                },
                "AI 眼镜项目纪要里写了什么背景说明？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "document background detail question",
                },
            }))

            first = service.chat(
                "这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？",
                user_id="u1",
                routing_mode="llm_first",
            )
            explain_first = service.chat("你的依据是什么？", user_id="u1", routing_mode="llm_first")
            second = service.chat("AI 眼镜项目纪要里写了什么背景说明？", user_id="u1", routing_mode="llm_first")
            explain_second = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "structured_memory")
            self.assertIn("结构化任务", explain_first["reply"])
            self.assertEqual(second["source_summary"]["primary_source"], "document")
            self.assertEqual(second["debug"]["source_summary"]["primary_source"], "document")
            self.assertIn("上传文档原文", explain_second["reply"])
            self.assertNotIn("结构化任务", explain_second["reply"])
            self.assertEqual(explain_second["debug"]["source_summary"]["primary_source"], "document")










    def test_explanation_reply_skips_emotional_chitchat_after_structured_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_document(
                "u1",
                filename="AI 眼镜项目纪要.md",
                title="AI 眼镜项目纪要",
                summary="项目背景和本周会议结论",
                content="# AI 眼镜项目纪要\n项目背景：周报不能只看聊天记忆，需要保留文档依据。\n说明：这份文档主要解释背景，不直接代表当前主线状态。",
                source="markdown_upload",
                ingestion_id="ing-project",
                created_at=1778120300.0,
            )
            store.add_memory(
                "u1",
                "AI 眼镜项目当前主线是收敛 LLM-first 文字输入，先压 recent context 污染，再补文本清洗 Phase C。",
                kind="event",
                memory_type="project_state",
                evidence_ids=["chunk_text_scope"],
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？": {
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "asks for current project status with recent document reference",
                },
                "哎今天真有点累，路上人好多，我差点站着睡着了。": {
                    "reply_mode": "fast",
                    "answer_source": "local",
                    "scope": "daily",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "daily emotional chitchat",
                },
                "哈哈刚才同事讲了个冷笑话，我反应慢半拍。": {
                    "reply_mode": "fast",
                    "answer_source": "local",
                    "scope": "daily",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "lightweight social chitchat",
                },
            }))

            first = service.chat("这个 AI 眼镜项目纪要对应的项目现在主要状态是什么？", user_id="u1", routing_mode="llm_first")
            second = service.chat("哎今天真有点累，路上人好多，我差点站着睡着了。", user_id="u1", routing_mode="llm_first")
            third = service.chat("哈哈刚才同事讲了个冷笑话，我反应慢半拍。", user_id="u1", routing_mode="llm_first")
            explain = service.chat("那你这次为什么这么说？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "structured_memory")
            self.assertEqual(second["source_summary"]["primary_source"], "none")
            self.assertEqual(third["source_summary"]["primary_source"], "none")
            self.assertIn("结构化任务", explain["reply"])
            self.assertIn("文档只作为背景补充", explain["reply"])
            self.assertNotIn("当前没有可用来源", explain["reply"])
            self.assertEqual(explain["debug"]["source_summary"]["primary_source"], "structured_memory")




    def test_explanation_reply_uses_raw_timeline_source_summary_from_previous_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "needs_timeline_recall": True,
                "memory_recall_type": "observation",
                "recall_goal": "raw_evidence",
                "timeline_query": "语音识别 不稳定",
                "confidence": 0.95,
                "reason": "exact wording requested",
            }))
            service.timeline_store.add_turn(
                "u1",
                "我提到国内网络下语音识别不稳定，可能需要 VPN 或备用方案。",
                created_at=1778120000.0,
            )

            first = service.chat("我之前有没有说过语音识别不稳定的原话？", user_id="u1", routing_mode="llm_first")
            explain = service.chat("依据是什么？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(first["source_summary"]["primary_source"], "raw_timeline")
            self.assertIn("timeline 原话", explain["reply"])
            self.assertEqual(service.fake_agent.main_calls, 0)

    def test_explanation_reply_uses_structured_memory_for_weekly_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "推进 memory eval 收口", kind="event", memory_type="task", evidence_ids=["chunk_task"])
            store.add_memory("u1", "当前风险是语音识别稳定性不足", kind="event", memory_type="project_state", evidence_ids=["chunk_risk"])
            store.add_document(
                "u1",
                filename="周会纪要.md",
                title="周会纪要",
                summary="项目背景和本周会议结论",
                content="这里主要是背景说明，不该盖过当前 task/risk。",
                source="markdown_upload",
                ingestion_id="ing-weekly",
                created_at=1778120000.0,
            )
            service = FakeService(store)

            first = service.chat("这周进展如何？", user_id="u1")
            explain = service.chat("你的依据是什么", user_id="u1")

            self.assertEqual(first["source_summary"]["primary_source"], "structured_memory")
            self.assertIn("结构化任务", explain["reply"])
            self.assertIn("文档只作为背景", explain["reply"])

    def test_explanation_reply_includes_active_evidence_quote_for_profile_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "local_profile_recall",
                "answer_source": "memory",
                "scope": "profile",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks current drink preference",
            }))
            timeline_result = service.timeline_store.add_turn(
                "u1",
                "我叫Jack。",
                created_at=1778120000.0,
            )
            chunk_id = timeline_result.chunks[0].id
            store.add_memory(
                "u1",
                "用户名字叫Jack",
                kind="profile",
                memory_type="profile",
                evidence_ids=[chunk_id],
            )

            first = service.chat("我叫什么名字？", user_id="u1")
            explain = service.chat("依据是什么？", user_id="u1")

            self.assertEqual(first["source_summary"]["primary_source"], "structured_memory")
            self.assertIn("用户名字叫Jack", explain["reply"])
            self.assertIn("我叫Jack", explain["reply"])
            self.assertIn("你当时这句原话", explain["reply"])
            self.assertIn("我叫Jack。", explain["debug"]["explanation_context"]["evidence_quotes"])

    def test_explanation_reply_does_not_quote_deleted_evidence_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "local_profile_recall",
                "answer_source": "memory",
                "scope": "profile",
                "needs_profile_memory": True,
                "memory_recall_type": "profile",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks current drink preference",
            }))
            timeline_result = service.timeline_store.add_turn(
                "u1",
                "我叫Jack。",
                created_at=1778120000.0,
            )
            chunk_id = timeline_result.chunks[0].id
            service.timeline_store.delete_chunks("u1", [chunk_id])
            store.add_memory(
                "u1",
                "用户名字叫Jack",
                kind="profile",
                memory_type="profile",
                evidence_ids=[chunk_id],
            )

            first = service.chat("我叫什么名字？", user_id="u1")
            explain = service.chat("依据是什么？", user_id="u1")

            self.assertEqual(first["source_summary"]["primary_source"], "structured_memory")
            self.assertIn("具体依据是这条记忆", explain["reply"])
            self.assertIn("用户名字叫Jack", explain["reply"])
            self.assertIn(f"evidence_ids={chunk_id}", explain["reply"])
            self.assertNotIn("你当时这句原话", explain["reply"])
            self.assertNotIn("我叫Jack。", explain["reply"])
            self.assertEqual(explain["debug"]["explanation_context"]["evidence_quotes"], [])

    def test_explanation_reply_reports_no_available_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))

            result = service.chat("你为什么这么说", user_id="u1", routing_mode="llm_first")

            self.assertIn("当前没有可用来源", result["reply"])
            self.assertEqual(service.fake_agent.main_calls, 0)

    def test_explanation_reply_reports_memory_stage_when_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户的 API key 是 sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
                            "kind": "profile",
                            "memory_type": "fact",
                            "confidence": 0.95,
                            "reason": "sensitive fact",
                        }
                    ],
                    "confidence": 0.95,
                },
            )
            service = FakeService(store, agent=agent)

            service.chat("我的 API key 是 sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ", user_id="u1", routing_mode="llm_first")
            explain = service.chat("为什么没记住？", user_id="u1", routing_mode="llm_first")

            self.assertIn("安全门控", explain["reply"])
            self.assertIn("拒绝", explain["reply"])

    def test_explanation_reply_reports_transient_context_rejection_when_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户想临时试试每天喝冰美式",
                            "kind": "profile",
                            "memory_type": "preference",
                            "confidence": 0.95,
                            "reason": "temporary preference draft",
                        }
                    ],
                    "confidence": 0.95,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat(
                "我最近想临时试试每天喝冰美式，先别写成偏好，等我确认了再说。",
                user_id="u1",
                routing_mode="llm_first",
            )
            explain = service.chat("为什么没记住？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["memory_processing"]["status"], "not_needed")
            self.assertEqual(result["debug"]["memory_processing"]["decision_reason"], "source_transient_context_only")
            self.assertEqual(result["debug"]["memory_processing"]["skip_policy"]["role"], "ephemeral_context")
            self.assertIn("不应写入长期记忆", explain["reply"])
            self.assertIn("临时上下文", explain["reply"])
            self.assertNotIn("安全门控", explain["reply"])
            self.assertEqual(explain["debug"]["memory_processing"]["status"], "not_needed")

    def test_explanation_reply_reports_memory_already_saved_when_user_asks_why_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("记一下今晚8点给妈妈打电话", user_id="u1", routing_mode="llm_first")
            explain = service.chat("为什么没保存？", user_id="u1", routing_mode="llm_first")

            job = service.read_memory_job(
                user_id="u1",
                job_id=result["debug"]["memory_processing"]["job_id"],
            )
            self.assertEqual(job["status"], "saved")
            self.assertEqual(job["saved_count"], 1)
            self.assertIn("其实已经保存了 1 条记忆", explain["reply"])
            self.assertEqual(explain["debug"]["memory_processing"]["status"], "saved")

    def test_explanation_reply_reports_local_do_not_remember_scope_when_saved_partially(self) -> None:
        segment_payload = {
            "物流电话不用记,不过周五前把按钮验收补完,后面其他背景先别沉淀": {
                "semantic_role": "memory_candidate",
                "noise_level": "low",
                "contains_filler": False,
                "do_not_remember_scope": "物流电话",
                "should_extract": True,
                "candidate_span": "周五前把按钮验收补完",
                "candidate_hint": "task",
                "confidence": 0.9,
                "reason": "local do-not-remember only applies to delivery phone",
            }
        }
        pre_reply_payload = {
            "*": {
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "memory_action": "none",
                "memory_kind": "none",
                "memory_type": "none",
                "candidate_content": "",
                "confidence": 0.9,
                "reason": "fake missed candidate",
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(
                store,
                agent=FakeAgent(pre_reply_payload=pre_reply_payload, segment_payload=segment_payload),
            )

            result = service.chat(
                "先记一段转写文字：物流电话不用记，不过周五前把按钮验收补完，后面其他背景先别沉淀。",
                user_id="u1",
                defer_memory_writes=True,
            )
            explain = service.chat("为什么没保存物流电话？", user_id="u1", routing_mode="llm_first")
            saved_text = "\n".join(memory.content for memory in store.list_memories("u1", kind="event"))

            self.assertIn("按钮验收", saved_text)
            self.assertNotIn("物流电话", saved_text)
            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertIn("物流电话", explain["reply"])
            self.assertIn("局部", explain["reply"])
            self.assertIn("不要记", explain["reply"])
            self.assertNotIn("安全门控", explain["reply"])
            self.assertNotIn("写入或后续流程失败", explain["reply"])
            self.assertEqual(explain["debug"]["memory_processing"]["status"], "saved")
            self.assertIn("物流电话", explain["debug"]["memory_processing"]["local_do_not_remember_scopes"])

    def test_explanation_reply_reports_background_write_failure_when_user_asks_why_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = FailingMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = FakeService(store)

            result = service.chat("记一下今晚8点给妈妈打电话", user_id="u1", routing_mode="llm_first")
            explain = service.chat("为什么没保存？", user_id="u1", routing_mode="llm_first")

            job = service.read_memory_job(
                user_id="u1",
                job_id=result["debug"]["memory_processing"]["job_id"],
            )
            self.assertEqual(result["debug"]["memory_processing"]["status"], "pending")
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["error_type"], "RuntimeError")
            self.assertIn("写入或后续流程失败", explain["reply"])
            self.assertIn("RuntimeError", explain["reply"])
            self.assertEqual(explain["debug"]["memory_processing"]["status"], "failed")

    def test_recall_arbitration_recent_activity_summary_filters_stable_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            source = store.add_memory("u1", "今天下午去吃面", kind="event", evidence_ids=["chunk_noodle"])
            store.add_memory("u1", "用户名字叫Jack", kind="profile", evidence_ids=["chunk_name"])
            store.add_memory(
                "u1",
                "用户最近主要有吃面和唱 K 活动。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
                evidence_ids=source.evidence_ids,
            )
            service = FakeService(store, agent=FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_profile_memory": True,
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "recent activity summary",
                },
                answer_payload={
                    "answer_intent": "summary",
                    "organization": "thematic",
                    "evidence_policy": "separate_background",
                    "filtering_rules": [],
                    "uncertainty_policy": "state_limits_when_context_is_sparse",
                    "style": "concise_structured",
                    "confidence": 0.9,
                    "reason": "summary",
                },
            ))

            result = service.chat("我最近都干了什么", user_id="u1", routing_mode="llm_first")
            message = service.fake_agent.main_messages[-1]
            arbitration = result["debug"]["memory"]["recall_arbitration"]

            self.assertEqual(arbitration["primary_source"], "observation")
            self.assertEqual(arbitration["dropped_counts"]["profile"], 1)
            self.assertEqual(
                arbitration["summary_profile_policy"],
                {
                    "scope": "recent_activity",
                    "dynamic_evidence": True,
                    "profile_treatment": "dropped_due_to_dynamic_evidence",
                },
            )
            self.assertIn("summary_dynamic_evidence_primary_over_stable_profile", json.dumps(arbitration, ensure_ascii=False))
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIn("今天下午去吃面", recalled)
            self.assertIn("用户最近主要有吃面", recalled)
            self.assertNotIn("Jack", recalled)
            self.assertNotIn("用户名字叫Jack", message)
            self.assertEqual(service.fake_agent.main_calls, 1)

    def test_recall_arbitration_recent_summary_keeps_profile_when_no_dynamic_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory("u1", "用户最近主要在做 AI 眼镜长期记忆项目", kind="profile", memory_type="project")
            service = FakeService(store, agent=FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_profile_memory": True,
                    "needs_event_memory": True,
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.95,
                    "reason": "recent activity summary",
                },
                answer_payload={
                    "answer_intent": "summary",
                    "organization": "thematic",
                    "evidence_policy": "separate_background",
                    "filtering_rules": [],
                    "uncertainty_policy": "state_limits_when_context_is_sparse",
                    "style": "concise_structured",
                    "confidence": 0.9,
                    "reason": "summary",
                },
            ))

            result = service.chat("我最近主要在做什么", user_id="u1", routing_mode="llm_first")
            message = service.fake_agent.main_messages[-1]
            arbitration = result["debug"]["memory"]["recall_arbitration"]

            self.assertEqual(arbitration["primary_source"], "profile")
            self.assertEqual(arbitration["dropped_counts"]["profile"], 0)
            self.assertEqual(
                arbitration["summary_profile_policy"],
                {
                    "scope": "recent_activity",
                    "dynamic_evidence": False,
                    "profile_treatment": "kept_no_dynamic_evidence",
                },
            )
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIn("用户最近主要在做 AI 眼镜长期记忆项目", recalled)
            self.assertIn("Background profile or stable context", message)
            self.assertIn("用户最近主要在做 AI 眼镜长期记忆项目", message)
            self.assertEqual(service.fake_agent.main_calls, 1)

    def test_recall_arbitration_specific_fact_empty_evidence_skips_main_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "specific remembered event",
            })
            service = FakeService(store, agent=agent)

            result = service.chat("我什么时候吃了面？", user_id="u1", routing_mode="llm_first")
            arbitration = result["debug"]["memory"]["recall_arbitration"]

            self.assertEqual(arbitration["primary_source"], "none")
            self.assertTrue(arbitration["empty_evidence_guard"]["triggered"])
            self.assertEqual(arbitration["empty_evidence_guard"]["reason"], "specific_fact_requested_but_no_direct_evidence")
            self.assertEqual(result["reply"], "我没有查到这件事的具体记录。")
            self.assertEqual(result["debug"]["local_reply_policy"]["role"], "retrieval_guard")
            self.assertEqual(result["debug"]["local_reply_policy"]["reason"], "specific_fact_requested_but_no_direct_evidence")
            self.assertEqual(result["debug"]["local_reply_policy"]["phrase_match_role"], "none")
            self.assertEqual(result["debug"]["llm"]["skipped"], True)
            self.assertEqual(agent.main_calls, 0)

    def test_llm_first_new_event_statement_does_not_fall_into_empty_evidence_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "device_local",
                    "location_text": "",
                    "needs_event_memory": False,
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "future event should be stored",
                },
                semantic_payload=semantic_payload(
                    correction=False,
                    memory_action="write",
                    memory_kind="event",
                    memory_type="task",
                    candidate_content="明天开会三点，在校园",
                ),
            )
            service = FakeService(store, agent=agent)

            result = service.chat("明天开会三点，在校园", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_action"], "write")
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "llm")
            self.assertEqual(result["debug"]["memory_processing"]["status"], "saved")
            self.assertFalse(result["debug"]["memory"]["recall_arbitration"]["empty_evidence_guard"]["triggered"])
            self.assertEqual(len(store.list_memories("u1", kind="event")), 1)

    def test_llm_first_summary_with_empty_timeline_query_does_not_recall_recent_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "needs_timeline_recall": True,
                "memory_recall_type": "observation",
                "recall_goal": "summary",
                "timeline_query": None,
                "confidence": 0.95,
                "reason": "summary with no specific timeline query",
            })
            service = FakeService(store, agent=agent)
            service.timeline_store.add_turn("u1", "你好", created_at=1778110000.0)
            service.timeline_store.add_turn("u1", "最近有什么安排", created_at=1778113600.0)

            result = service.chat("刚刚都做了什么", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["timeline"]["recall"]["strategy"], "skipped_empty_summary_query")
            self.assertEqual(result["recalled_timeline_chunks"], [])
            self.assertNotIn("Raw user timeline chunks recalled", agent.main_messages[-1])

    def test_llm_first_ordinary_question_uses_direct_answer_directive_without_memory_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary question",
                },
                answer_payload={
                    "answer_intent": "direct_answer",
                    "organization": "direct",
                    "evidence_policy": "use_available_context",
                    "filtering_rules": [],
                    "uncertainty_policy": "none",
                    "style": "short_direct",
                    "confidence": 0.9,
                    "reason": "ordinary chat",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("水的化学式是什么？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["answer_directive"]["answer_intent"], "direct_answer")
            self.assertIn("<answer-directive>", agent.main_messages[-1])
            self.assertNotIn("Direct structured memory evidence", agent.main_messages[-1])
            self.assertEqual(agent.answer_calls, 1)
            self.assertEqual(agent.main_calls, 1)
            self.assertEqual(
                result["debug"]["turn_decision"]["final"]["source"],
                "pre_reply_decision",
            )
            self.assertFalse(result["debug"]["turn_decision"]["final"]["needs_location"])
            self.assertFalse(result["debug"]["turn_decision"]["final"]["needs_web_search"])
            self.assertEqual(result["debug"]["turn_decision"]["final"]["recall_goal"], "none")

    def test_llm_first_pre_reply_authority_overrides_classifier_route_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_location": False,
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary question",
                },
                intent_payload={
                    "needs_web_search": True,
                    "web_query": "今日天气",
                    "web_reason": "classifier route noise",
                    "is_weather_query": True,
                    "weather_location_source": "device_location",
                    "weather_place_text": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户偏好用中文简短回答",
                            "kind": "assistant_preference",
                            "memory_type": "preference",
                            "confidence": 0.9,
                            "reason": "valid extraction",
                        }
                    ],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("水的化学式是什么？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["turn_decision"]["final"]["source"], "pre_reply_decision")
            self.assertFalse(result["debug"]["turn_decision"]["final"]["needs_web_search"])
            self.assertFalse(result["debug"]["turn_decision"]["final"]["needs_location"])
            self.assertFalse(result["debug"]["intent"]["needs_web_search"])
            self.assertEqual(result["debug"]["intent"]["web_query"], None)
            self.assertEqual(result["debug"]["intent"]["authority"], "memory_extraction_only")
            self.assertEqual(result["debug"]["intent"]["route_authority"], "pre_reply_decision")
            self.assertEqual(result["debug"]["weather"]["mode"], "none")
            self.assertEqual(result["debug"]["tools"][0]["triggered"], False)
            self.assertEqual(result["debug"]["tools"][0]["reason"], "classifier_no_realtime_intent")
            self.assertNotIn("reason", result["debug"]["location"])
            self.assertEqual(agent.intent_calls, 0)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertEqual(agent.main_calls, 1)

    def test_pre_reply_prompt_uses_orthogonal_route_fields_not_turn_type_enum(self) -> None:
        agent = FakeAgent()

        from ai_glasses_memory_assistant.turn_semantic_classifier import classify_pre_reply_decision

        classify_pre_reply_decision(agent, "这阵子我的主要投入方向是什么？")
        prompt = agent.semantic_messages[-1]

        self.assertIn("needs_web_search", prompt)
        self.assertIn("memory_recall_type", prompt)
        self.assertIn("recall_goal", prompt)
        self.assertIn("candidate_content", prompt)
        self.assertNotIn('"turn_type"', prompt)
        self.assertNotIn("ordinary|greeting|weather", prompt)
        self.assertEqual(agent.pre_reply_calls, 1)
        self.assertEqual(agent.semantic_calls, 1)

    def test_llm_first_pre_reply_controls_location_context_for_web_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_location": False,
                    "needs_web_search": True,
                    "web_query": "OpenAI 最新消息",
                    "web_reason": "current external information",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary realtime search",
                },
                intent_payload={
                    "needs_web_search": True,
                    "web_query": "我这里 OpenAI 最新消息",
                    "web_reason": "classifier route noise",
                    "is_weather_query": False,
                    "weather_location_source": "device_location",
                    "weather_place_text": "",
                    "memory_write_candidates": [],
                    "confidence": 0.9,
                },
            )
            service = FakeService(store, agent=agent)
            location = LocationContext(
                status="available",
                latitude=39.9042,
                longitude=116.4074,
                accuracy=25.0,
                timestamp=1778131200.0,
            )

            with patch("ai_glasses_memory_assistant.web_search.duckduckgo_search", return_value=[]):
                result = service.chat("查一下 OpenAI 最新消息", user_id="u1", location=location, routing_mode="llm_first")

            self.assertTrue(result["debug"]["turn_decision"]["final"]["needs_web_search"])
            self.assertFalse(result["debug"]["turn_decision"]["final"]["needs_location"])
            self.assertEqual(result["debug"]["tools"][0]["query"], "OpenAI 最新消息")
            self.assertNotIn("latitude", result["debug"]["tools"][0]["query"])
            self.assertNotIn("<location-context>", agent.main_messages[-1])
            self.assertEqual(result["debug"]["intent"]["authority"], "memory_extraction_only")

    def test_llm_first_web_search_uses_project_web_search_boundary(self) -> None:
        from ai_glasses_memory_assistant.web_search import WebSearchResponse

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_location": False,
                    "needs_web_search": True,
                    "web_query": "OpenAI 最新消息",
                    "web_reason": "current external information",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary realtime search",
                }
            )
            service = FakeService(store, agent=agent)
            search_response = WebSearchResponse(
                query="OpenAI 最新消息",
                results=[{
                    "title": "OpenAI update",
                    "url": "https://example.com/openai",
                    "description": "A current OpenAI news summary.",
                }],
            )

            with patch("ai_glasses_memory_assistant.web_search.search_web", return_value=search_response) as search_web:
                result = service.chat("查一下 OpenAI 最新消息", user_id="u1", routing_mode="llm_first")

            search_web.assert_called_once_with("OpenAI 最新消息", limit=5)
            tool_debug = result["debug"]["tools"][0]
            self.assertEqual(tool_debug["backend"], "duckduckgo_html_fallback")
            self.assertEqual(tool_debug["query"], "OpenAI 最新消息")
            self.assertEqual(tool_debug["reason"], "current external information")
            self.assertEqual(tool_debug["results_count"], 1)
            self.assertEqual(tool_debug["results"][0]["title"], "OpenAI update")
            self.assertIn("Web/tool context:", agent.main_messages[-1])
            self.assertIn("A current OpenAI news summary.", agent.main_messages[-1])
            self.assertIn("<tool-state>", agent.main_messages[-1])
            self.assertIn("web_search.status: completed_with_results", agent.main_messages[-1])
            self.assertIn("Ground realtime claims in the Web/tool context", agent.main_messages[-1])

    def test_llm_first_web_search_empty_results_injects_completed_tool_state(self) -> None:
        from ai_glasses_memory_assistant.web_search import WebSearchResponse

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_location": False,
                    "needs_web_search": True,
                    "web_query": "realtime external query",
                    "web_reason": "current external information",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary realtime search",
                }
            )
            service = FakeService(store, agent=agent)
            search_response = WebSearchResponse(
                query="realtime external query",
                results=[],
                backend="ddgs",
            )

            with patch("ai_glasses_memory_assistant.web_search.search_web", return_value=search_response):
                result = service.chat("realtime external query", user_id="u1", routing_mode="llm_first")

            tool_debug = result["debug"]["tools"][0]
            self.assertTrue(tool_debug["triggered"])
            self.assertFalse(tool_debug["available"])
            self.assertEqual(tool_debug["backend"], "ddgs")
            self.assertEqual(tool_debug["results_count"], 0)
            self.assertNotIn("Web/tool context:", agent.main_messages[-1])
            self.assertIn("<tool-state>", agent.main_messages[-1])
            self.assertIn("web_search.status: completed_without_results", agent.main_messages[-1])
            self.assertIn("do not invent realtime facts", agent.main_messages[-1])

    def test_llm_first_web_search_not_triggered_injects_not_triggered_tool_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_location": False,
                    "needs_web_search": False,
                    "web_query": "",
                    "web_reason": "",
                    "memory_recall_type": "none",
                    "recall_goal": "none",
                    "confidence": 0.95,
                    "reason": "ordinary non-web question",
                }
            )
            service = FakeService(store, agent=agent)

            result = service.chat("ordinary question", user_id="u1", routing_mode="llm_first")

            tool_debug = result["debug"]["tools"][0]
            self.assertFalse(tool_debug["triggered"])
            self.assertNotIn("Web/tool context:", agent.main_messages[-1])
            self.assertIn("<tool-state>", agent.main_messages[-1])
            self.assertIn("web_search.status: not_triggered", agent.main_messages[-1])
            self.assertIn("Do not claim that a web search is in progress or pending", agent.main_messages[-1])

    def test_llm_first_uses_temporal_parser_for_yesterday_same_time_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "昨天这个时间在开产品会",
                kind="event",
                start_at=1778119200.0,
                end_at=1778122800.0,
                time_granularity="hour",
                temporal_text="昨天这个时间",
            )
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "event",
                    "recall_goal": "specific_fact",
                    "confidence": 0.93,
                    "reason": "asks for remembered activity at a relative time",
                },
                temporal_payload={
                    "has_temporal_expression": True,
                    "temporal_text": "昨天的现在",
                    "kind": "instant",
                    "start_at_iso": "2026-05-07T10:00:00+08:00",
                    "end_at_iso": "2026-05-07T11:00:00+08:00",
                    "granularity": "hour",
                    "normalized_text": "我在干什么",
                    "confidence": 0.95,
                    "reason": "yesterday same local hour",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("昨天的现在我在干什么", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["routing"]["mode"], "llm_first")
            self.assertEqual(result["debug"]["temporal"]["query"]["backend"], "llm")
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "temporal_range")
            self.assertIn("产品会", "\n".join(memory["content"] for memory in result["recalled_memories"]))
            self.assertIn("<memory-context>", agent.main_messages[-1])
            self.assertEqual(result["debug"]["answer_directive"]["backend"], "llm")
            self.assertEqual(agent.temporal_calls, 1)
            self.assertEqual(agent.answer_calls, 1)
            self.assertEqual(agent.main_calls, 1)

    def test_llm_first_keeps_raw_evidence_timeline_reply_for_explicit_original_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "memory_recall_type": "timeline",
                "recall_goal": "raw_evidence",
                "timeline_query": "国内网络 语音识别",
                "confidence": 0.95,
                "reason": "asks for exact wording",
            })
            service = FakeService(store, agent=agent)
            service.timeline_store.add_turn(
                "u1",
                "我提到国内网络下语音识别不稳定，可能需要 VPN 或备用方案。",
                created_at=1778120000.0,
            )

            result = service.chat("我之前有没有说过国内网络语音识别的原话？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["pre_reply_decision"]["recall_goal"], "raw_evidence")
            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_timeline_recall")
            self.assertEqual(result["debug"]["turn_decision"]["final"]["source"], "pre_reply_decision")
            self.assertEqual(result["debug"]["answer_directive"]["backend"], "not_used")
            self.assertEqual(result["debug"]["memory"]["drift_guard"]["skipped_reason"], "no_recalled_memories")
            self.assertEqual(agent.main_calls, 0)
            self.assertEqual(agent.answer_calls, 0)
            self.assertIn("国内网络下语音识别不稳定", result["reply"])

    def test_llm_first_unified_pre_reply_drives_observation_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "用户最近主要在推进 AI 眼镜长期记忆 demo 的 pre_reply_decision 收口。",
                kind="event",
                memory_type="observation",
                source="observation_reflect",
            )
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "needs_location": False,
                    "location_text": "",
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_recall_type": "observation",
                    "recall_goal": "summary",
                    "confidence": 0.96,
                    "reason": "recent focus summary across memories",
                },
                answer_payload={
                    "answer_intent": "summary",
                    "organization": "thematic",
                    "evidence_policy": "separate_background",
                    "filtering_rules": [],
                    "uncertainty_policy": "state_limits_when_context_is_sparse",
                    "style": "concise_structured",
                    "confidence": 0.92,
                    "reason": "recent focus summary",
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我最近主要在做什么？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["turn_decision"]["final"]["source"], "pre_reply_decision")
            self.assertEqual(result["debug"]["turn_decision"]["final"]["event_recall_strategy"], "observation_review")
            self.assertTrue(result["debug"]["planner"]["needs_observation_memory"])
            self.assertEqual(result["debug"]["pre_reply_decision"]["memory_recall_type"], "observation")
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)

    def test_legacy_invalid_routing_mode_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            service = FakeService(self.make_store(Path(tmpdir)))

            result = service.chat("你好", user_id="u1", routing_mode="unknown")

            self.assertEqual(result["debug"]["routing"]["mode"], "llm_first")

    def test_temporal_status_statement_is_not_saved_as_life_event(self) -> None:
        cases = ("今天看起来不错", "明天看情况")
        for message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    store = self.make_store(Path(tmpdir))
                    service = FakeService(store)

                    result = service.chat(message, user_id="u1")

                    self.assertEqual(result["saved_memories"], [])
                    self.assertEqual(store.list_memories("u1", kind="event"), [])
                    self.assertEqual(result["debug"]["memory_processing"]["status"], "not_needed")

    def test_transient_context_message_exposes_skip_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            service = FakeService(store)

            result = service.chat("我刚刚路过公司楼下", user_id="u1")

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1", kind="event"), [])
            self.assertEqual(result["debug"]["memory_processing"]["status"], "not_needed")
            self.assertEqual(
                result["debug"]["memory_processing"]["decision_reason"],
                "source_transient_context_only",
            )
            self.assertEqual(
                result["debug"]["memory_processing"]["skip_policy"]["role"],
                "ephemeral_context",
            )

    def test_tentative_draft_context_is_not_saved_as_project_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "AI 眼镜项目临时草稿是先把风险列表写进周报，可能不是最终方案",
                            "kind": "event",
                            "memory_type": "project_state",
                            "confidence": 0.95,
                            "reason": "tentative project state draft",
                        }
                    ],
                    "confidence": 0.95,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat(
                "我先临时想一下，AI 眼镜项目草稿是先把风险列表写进周报，可能不是最终方案",
                user_id="u1",
                routing_mode="llm_first",
            )

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1", kind="event"), [])
            self.assertEqual(result["debug"]["memory_processing"]["status"], "not_needed")
            self.assertEqual(result["debug"]["memory_processing"]["decision_reason"], "source_transient_context_only")
            self.assertEqual(result["debug"]["memory_processing"]["skip_policy"]["role"], "ephemeral_context")

    def test_temporary_preference_context_is_not_saved_as_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "memory_write_candidates": [
                        {
                            "content": "用户想临时试试每天喝冰美式",
                            "kind": "profile",
                            "memory_type": "preference",
                            "confidence": 0.95,
                            "reason": "temporary preference draft",
                        }
                    ],
                    "confidence": 0.95,
                },
            )
            service = FakeService(store, agent=agent)

            result = service.chat(
                "我最近想临时试试每天喝冰美式，先别写成偏好，等我确认了再说。",
                user_id="u1",
                routing_mode="llm_first",
            )

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1", kind="profile"), [])
            self.assertEqual(result["debug"]["memory_processing"]["status"], "not_needed")
            self.assertEqual(result["debug"]["memory_processing"]["decision_reason"], "source_transient_context_only")
            self.assertEqual(result["debug"]["memory_processing"]["skip_policy"]["role"], "ephemeral_context")

    def test_ambient_audio_candidate_is_blocked_from_long_term_memory(self) -> None:
        candidate = MemoryWriteCandidate(
            content="他们还在吵预算",
            kind="event",
            memory_type="project_state",
            confidence=0.95,
            reason="ambient background project state",
            source_type="ambient_audio",
        )
        gate = should_write_memory_candidate(candidate, "他们还在吵预算")
        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reason, "ambient_only")

    def test_wake_query_other_speaker_is_not_saved(self) -> None:
        candidate = MemoryWriteCandidate(
            content="明天我要找 Alex 对预算",
            kind="event",
            memory_type="task",
            confidence=0.95,
            reason="wake query candidate",
            source_type="wake_query",
            speaker_hint="other",
        )
        gate = should_write_memory_candidate(candidate, "记一下，明天我要找 Alex 对预算")
        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reason, "third_party_speech_blocked")

    def test_wake_query_do_not_remember_is_not_saved(self) -> None:
        candidate = MemoryWriteCandidate(
            content="这个只是我随口吐槽",
            kind="event",
            memory_type="event",
            confidence=0.95,
            reason="wake query do not remember",
            source_type="wake_query",
            speaker_hint="user",
            do_not_remember_scope="这个只是我随口吐槽",
        )
        gate = should_write_memory_candidate(candidate, "你就按刚才那段帮我判断一下，不用保存。")
        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reason, "explicit_do_not_remember")

    def test_wake_query_sensitive_audio_is_not_saved(self) -> None:
        candidate = MemoryWriteCandidate(
            content="验证码是 482931",
            kind="event",
            memory_type="fact",
            confidence=0.95,
            reason="wake query sensitive audio",
            source_type="wake_query",
            speaker_hint="user",
        )
        gate = should_write_memory_candidate(candidate, "你刚才听到了什么？")
        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reason, "do_not_memorize_sensitive_audio")

    def test_wake_query_user_task_can_still_be_saved(self) -> None:
        candidate = MemoryWriteCandidate(
            content="明天我要找 Alex 对预算",
            kind="event",
            memory_type="task",
            confidence=0.95,
            reason="wake query task",
            source_type="wake_query",
            speaker_hint="user",
        )
        gate = should_write_memory_candidate(candidate, "记一下，明天我要找 Alex 对预算")
        self.assertTrue(gate.allowed)

    def test_colloquial_questions_stay_on_planner_baseline(self) -> None:
        cases = ("今天干嘛去", "明天干嘛去", "今天去哪里吃饭？")
        for message in cases:
            with self.subTest(message=message):
                planner = plan_turn(message, reference_time=1778131200.0, timezone="Asia/Shanghai")
                self.assertFalse(planner.fast_path)
                self.assertEqual(planner.reply_mode, "llm")

                with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
                    store = self.make_store(Path(tmpdir))
                    service = FakeService(store)

                    result = service.chat(message, user_id="u1")

                    self.assertEqual(result["saved_memories"], [])
                    self.assertEqual(store.list_memories("u1", kind="event"), [])

    def test_recent_activity_query_recalls_past_event_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "昨天中午在公司楼下吃了沙拉",
                kind="event",
                start_at=1778126400.0,
                end_at=1778130000.0,
                time_granularity="hour",
                temporal_text="昨天中午",
            )
            store.add_memory(
                "u1",
                "明天出门看车展",
                kind="event",
                start_at=1778256000.0,
                end_at=1778342400.0,
                time_granularity="day",
                temporal_text="明天",
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "asks recent activity",
            }))

            result = service.chat("最近干了什么", user_id="u1")

            self.assertTrue(result["debug"]["planner"]["needs_event_memory"])
            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "temporal_range")
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIsInstance(recalled, str)
            self.assertNotIn("明天出门看车展", recalled)
            self.assertNotIn("明天出门看车展", result["reply"])
            self.assertIn(service.fake_agent.main_calls, {0, 1})

    def test_specific_event_query_without_evidence_does_not_fall_through_to_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "memory_recall_type": "event",
                    "recall_goal": "specific_fact",
                    "confidence": 0.9,
                    "reason": "asks for a remembered event",
                }
            )
            service = FakeService(store, agent=agent)

            result = service.chat("我什么时候吃了面", user_id="u1")

            self.assertEqual(result["debug"]["planner"]["reply_mode"], "local_event_recall")
            self.assertIn(result["debug"]["memory"]["event_recall"]["strategy"], {"text_search", "temporal_range"})
            self.assertEqual(result["debug"]["memory"]["event_recall_count"], 0)
            self.assertNotIn("小四川", result["reply"])
            self.assertNotIn("小四川", result["reply"])
            self.assertNotIn("牛肉面", result["reply"])
            self.assertNotIn("12:35", result["reply"])
            self.assertIn(agent.main_calls, {0, 1})

    def test_specific_event_query_recalls_chinese_action_object_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "下午去吃面",
                kind="event",
                start_at=1778126400.0,
                end_at=1778148000.0,
                time_granularity="hour",
                temporal_text="下午",
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "specific_fact",
                "confidence": 0.95,
                "reason": "asks specific remembered event",
            }))

            result = service.chat("我什么时候吃了面", user_id="u1")

            self.assertIn(result["debug"]["memory"]["event_recall"]["strategy"], {"text_search", "temporal_range"})
            recalled = "\n".join(memory["content"] for memory in result["recalled_memories"])
            self.assertIsInstance(recalled, str)
            self.assertIn(result["debug"]["local_reply_policy"]["role"], {"retrieval_hint", "retrieval_guard"})
            self.assertNotIn("小四川", result["reply"])
            self.assertNotIn("牛肉面", result["reply"])
            self.assertIn(service.fake_agent.main_calls, {0, 1})

    def test_recent_work_query_does_not_return_life_event_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            store.add_memory(
                "u1",
                "去吃面",
                kind="event",
                start_at=1778126400.0,
                end_at=1778148000.0,
                time_granularity="hour",
                temporal_text="下午",
            )
            service = FakeService(store, agent=FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "unknown",
                "location_text": "",
                "needs_event_memory": True,
                "memory_recall_type": "event",
                "recall_goal": "summary",
                "confidence": 0.95,
                "reason": "asks recent work activity",
            }))

            result = service.chat("我最近有什么工作", user_id="u1")

            self.assertEqual(result["debug"]["memory"]["event_recall"]["strategy"], "temporal_range")
            self.assertEqual(result["debug"]["memory"]["event_recall_count"], 0)
            self.assertIn(result["debug"]["memory"]["event_recall"]["unfiltered_count"], {0, 1})
            self.assertEqual(
                result["debug"]["memory"]["event_recall"]["filter_policy"],
                {
                    "role": "retrieval_narrowing",
                    "reason": "work_memory_query_filter",
                    "markers": ["工作"],
                    "treatment": "filter_to_work_memory_types_and_markers",
                    "input_count": result["debug"]["memory"]["event_recall"]["filter_policy"]["input_count"],
                    "output_count": 0,
                    "filtered_count": result["debug"]["memory"]["event_recall"]["filter_policy"]["filtered_count"],
                    "include_closed_tasks": False,
                },
            )
            self.assertNotIn("吃面", result["reply"])
            self.assertIn(service.fake_agent.main_calls, {0, 1})

    def test_location_context_is_injected_without_memory_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent()
            service = FakeService(store, agent=agent)
            location = LocationContext(
                status="available",
                latitude=39.9042,
                longitude=116.4074,
                accuracy=25.0,
                timestamp=1778131200.0,
            )
            result = service.chat("我在哪？", user_id="u1", location=location)

            self.assertEqual(result["saved_memories"], [])
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(result["debug"]["location"]["status"], "available")
            self.assertTrue(result["debug"]["location"]["needed"])
            self.assertIn("<location-context>", agent.main_messages[-1])
            self.assertIn("latitude: 39.9042", agent.main_messages[-1])
            self.assertIn("privacy: ephemeral session context", agent.main_messages[-1])

    def test_location_query_without_location_marks_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(pre_reply_payload={
                "reply_mode": "llm",
                "answer_source": "llm",
                "scope": "device_local",
                "needs_location": True,
                "location_text": "",
                "needs_web_search": False,
                "web_query": None,
                "web_reason": "",
                "confidence": 0.95,
                "reason": "nearby places require current location",
            })
            service = FakeService(store, agent=agent)
            result = service.chat("附近有什么咖啡店？", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["location"]["status"], "missing")
            self.assertTrue(result["debug"]["location"]["needed"])
            self.assertTrue(result["debug"]["routing"]["pre_reply_decision_applied"])
            self.assertTrue(result["debug"]["turn_decision"]["final"]["needs_location"])
            self.assertEqual(
                result["debug"]["location"]["reason"],
                "location_dependent_query_without_current_location",
            )
            self.assertEqual(agent.main_calls, 0)
            self.assertEqual(agent.pre_reply_calls, 1)
            self.assertGreaterEqual(agent.semantic_calls, 1)
            self.assertIn("定位", result["reply"])

    def test_explicit_place_weather_ignores_device_location_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "named_place",
                    "needs_location": False,
                    "location_text": "上海",
                    "needs_web_search": True,
                    "web_query": "上海 天气 今天",
                    "web_reason": "weather",
                    "confidence": 0.96,
                    "reason": "explicit place weather query",
                },
                intent_payload={
                "needs_web_search": True,
                "web_query": "上海 天气 今天",
                "web_reason": "用户询问当前天气，需要实时数据",
                "is_weather_query": True,
                "weather_location_source": "explicit_place",
                "weather_place_text": "上海",
                "memory_write_candidates": [],
                "confidence": 1.0,
            })
            service = FakeService(store, agent=agent)
            location = LocationContext(
                status="available",
                latitude=39.9042,
                longitude=116.4074,
                accuracy=25.0,
                timestamp=1778131200.0,
            )
            with patch("ai_glasses_memory_assistant.web_search.duckduckgo_search", return_value=[]):
                result = service.chat("上海天气如何", user_id="u1", location=location, routing_mode="llm_first")

            self.assertEqual(result["debug"]["weather"]["mode"], "explicit_place_weather")
            self.assertEqual(result["debug"]["weather"]["location_source"], "explicit_place")
            self.assertEqual(result["debug"]["weather"]["place_text"], "上海")
            self.assertEqual(result["debug"]["weather"]["query_strategy"], "explicit_place_query")
            self.assertFalse(result["debug"]["location"]["needed"])
            self.assertTrue(result["debug"]["turn_decision"]["final"]["needs_web_search"])
            self.assertEqual(result["debug"]["tools"][0]["query"], "上海 天气 今天")
            self.assertNotIn("latitude", result["debug"]["tools"][0]["query"])
            self.assertNotIn("<location-context>", agent.main_messages[-1])

    def test_current_location_weather_fallback_query_avoids_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "device_local",
                    "needs_location": True,
                    "location_text": "",
                    "needs_web_search": True,
                    "web_query": "今日天气",
                    "web_reason": "weather",
                    "confidence": 0.96,
                    "reason": "current location weather query",
                },
                intent_payload={
                "needs_web_search": True,
                "web_query": "今日天气",
                "web_reason": "用户询问当前天气，需要实时数据",
                "is_weather_query": True,
                "weather_location_source": "device_location",
                "weather_place_text": "",
                "memory_write_candidates": [],
                "confidence": 1.0,
            })
            service = FakeService(store, agent=agent)
            location = LocationContext(
                status="available",
                latitude=39.9042,
                longitude=116.4074,
                accuracy=25.0,
                timestamp=1778131200.0,
            )
            with patch("ai_glasses_memory_assistant.web_search.duckduckgo_search", return_value=[]):
                result = service.chat("今日天气", user_id="u1", location=location, routing_mode="llm_first")

            self.assertEqual(result["debug"]["weather"]["mode"], "current_location_weather")
            self.assertEqual(result["debug"]["weather"]["query_strategy"], "device_weather_query")
            self.assertTrue(result["debug"]["location"]["needed"])
            self.assertTrue(result["debug"]["turn_decision"]["final"]["needs_web_search"])
            self.assertEqual(result["debug"]["tools"][0]["query"], "今日天气")
            self.assertNotIn("latitude", result["debug"]["tools"][0]["query"])
            self.assertIn("<location-context>", agent.main_messages[-1])

    def test_current_location_weather_without_location_uses_missing_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "device_local",
                    "needs_location": True,
                    "location_text": "",
                    "needs_web_search": True,
                    "web_query": "今日天气",
                    "web_reason": "weather",
                    "confidence": 0.96,
                    "reason": "current location weather query",
                },
                intent_payload={
                "needs_web_search": True,
                "web_query": "今日天气",
                "web_reason": "用户询问当前天气，需要实时数据",
                "is_weather_query": True,
                "weather_location_source": "device_location",
                "weather_place_text": "",
                "memory_write_candidates": [],
                "confidence": 1.0,
            })
            service = FakeService(store, agent=agent)
            result = service.chat("今日天气", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["weather"]["mode"], "current_location_weather")
            self.assertEqual(result["debug"]["weather"]["query_strategy"], "missing_location_guard")
            self.assertTrue(result["debug"]["location"]["needed"])
            self.assertTrue(result["debug"]["routing"]["pre_reply_decision_applied"])
            self.assertEqual(result["debug"]["tools"][0]["reason"], "location_required_but_missing")

    def test_llm_first_classifier_weather_hint_does_not_promote_explicit_place_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "is_weather_query": True,
                    "weather_location_source": "explicit_place",
                    "weather_place_text": "上海",
                    "memory_write_candidates": [],
                    "confidence": 1.0,
                },
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "confidence": 0.95,
                    "reason": "pre_reply does not request realtime weather",
                },
            )
            service = FakeService(store, agent=agent)
            with patch("ai_glasses_memory_assistant.web_search.duckduckgo_search", return_value=[]):
                result = service.chat("帮我查一下上海天气", user_id="u1", routing_mode="llm_first")

            self.assertEqual(result["debug"]["intent"]["authority"], "memory_extraction_only")
            self.assertEqual(result["debug"]["intent"]["route_authority"], "pre_reply_decision")
            self.assertFalse(result["debug"]["turn_decision"]["final"]["needs_web_search"])
            self.assertEqual(result["debug"]["weather"]["mode"], "none")
            self.assertFalse(result["debug"]["tools"][0]["triggered"])
            self.assertEqual(result["debug"]["tools"][0]["reason"], "classifier_no_realtime_intent")

    def test_llm_first_classifier_weather_hint_does_not_promote_device_location_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = self.make_store(Path(tmpdir))
            agent = FakeAgent(
                intent_payload={
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "is_weather_query": True,
                    "weather_location_source": "device_location",
                    "weather_place_text": "",
                    "memory_write_candidates": [],
                    "confidence": 1.0,
                },
                pre_reply_payload={
                    "reply_mode": "llm",
                    "answer_source": "llm",
                    "scope": "unknown",
                    "location_text": "",
                    "needs_web_search": False,
                    "web_query": None,
                    "web_reason": "",
                    "confidence": 0.95,
                    "reason": "pre_reply does not request realtime weather",
                },
            )
            service = FakeService(store, agent=agent)
            location = LocationContext(
                status="available",
                latitude=39.9042,
                longitude=116.4074,
                accuracy=25.0,
                timestamp=1778131200.0,
            )
            with patch("ai_glasses_memory_assistant.web_search.duckduckgo_search", return_value=[]):
                result = service.chat("我这里天气如何", user_id="u1", location=location, routing_mode="llm_first")

            self.assertEqual(result["debug"]["intent"]["authority"], "memory_extraction_only")
            self.assertEqual(result["debug"]["intent"]["route_authority"], "pre_reply_decision")
            self.assertFalse(result["debug"]["turn_decision"]["final"]["needs_web_search"])
            self.assertEqual(result["debug"]["weather"]["mode"], "none")
            self.assertFalse(result["debug"]["tools"][0]["triggered"])
            self.assertEqual(result["debug"]["tools"][0]["reason"], "classifier_no_realtime_intent")
            self.assertNotIn("<location-context>", agent.main_messages[-1])


if __name__ == "__main__":
    unittest.main()
