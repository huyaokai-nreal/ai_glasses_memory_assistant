from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_glasses_memory_assistant.evals.metrics import evaluate_turn, summarize_runs
from ai_glasses_memory_assistant.evals.longmemeval_adapter import (
    answer_terms,
    load_longmemeval_items,
    parse_longmemeval_item,
)
from ai_glasses_memory_assistant.evals.longmemeval_runner import summarize_longmemeval_runs
from ai_glasses_memory_assistant.evals.runner import (
    _EvalPreReplyDecisionAgent,
    filter_scenarios,
    _install_eval_pre_reply_decision_agent,
    _preload_documents,
    _preload_memories,
    run_turn,
)
from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
from ai_glasses_memory_assistant.memory_store import EventMemoryStore


# eval 指标测试聚焦硬判逻辑，避免报告结果依赖 LLM 自评。
class EvalMetricsTests(unittest.TestCase):
    def test_eval_runner_has_no_top_level_hermes_fallback_imports(self) -> None:
        source_path = Path(__file__).resolve().parents[1] / "ai_glasses_memory_assistant" / "evals" / "runner.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden_modules = ("hermes_constants", "hermes_cli")
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

    def test_longmemeval_adapter_parses_official_shape(self) -> None:
        item = parse_longmemeval_item(
            {
                "question_id": "q1",
                "question_type": "single-session-preference",
                "question": "What drink does the user like?",
                "answer": "latte",
                "question_date": "2026-01-02",
                "haystack_session_ids": ["s1"],
                "haystack_dates": ["2026-01-01"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "I like latte.", "has_answer": True},
                        {"role": "assistant", "content": "Noted."},
                    ]
                ],
                "answer_session_ids": ["s1"],
            }
        )

        self.assertEqual(item.question_id, "q1")
        self.assertEqual(item.question_type, "single-session-preference")
        self.assertFalse(item.is_abstention)
        self.assertEqual(item.sessions[0].session_id, "s1")
        self.assertTrue(item.sessions[0].turns[0].has_answer)

    def test_longmemeval_loader_filters_type_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "longmemeval.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "question_id": "q1",
                            "question_type": "single-session-user",
                            "question": "Q1",
                            "answer": "A1",
                            "haystack_sessions": [],
                        },
                        {
                            "question_id": "q2_abs",
                            "question_type": "multi-session",
                            "question": "Q2",
                            "answer": "A2",
                            "haystack_sessions": [],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            items = load_longmemeval_items(path, limit=1, question_types={"multi-session"})

        self.assertEqual([item.question_id for item in items], ["q2_abs"])
        self.assertTrue(items[0].is_abstention)

    def test_longmemeval_summary_groups_question_types(self) -> None:
        summary = summarize_longmemeval_runs(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "answer_hit": True,
                    "recall_hit": True,
                    "measured_seconds": 1.0,
                },
                {
                    "question_id": "q2_abs",
                    "question_type": "knowledge-update",
                    "is_abstention": True,
                    "answer_hit": False,
                    "recall_hit": False,
                    "measured_seconds": 3.0,
                },
            ]
        )

        self.assertEqual(summary["overall"]["total"], 2)
        self.assertEqual(summary["overall"]["answer_hit_rate"], 0.5)
        self.assertEqual(summary["by_question_type"]["single-session-user"]["answer_hit_rate"], 1.0)
        self.assertEqual(summary["by_question_type"]["abstention"]["total"], 1)

    def test_longmemeval_answer_terms_splits_comma_answers(self) -> None:
        self.assertEqual(answer_terms("latte, espresso."), ["latte", "espresso"])

    def test_evaluate_turn_hard_checks_reply_recall_and_memory(self) -> None:
        result = evaluate_turn(
            response={
                "reply": "我记得你昨天中午吃了牛肉饭。",
                "recalled_memories": [{"content": "昨天中午吃了牛肉饭"}],
                "debug": {"memory": {"event_recall": {"strategy": "temporal_range"}}},
                "completed": True,
            },
            exception="",
            expect={
                "reply_contains": ["牛肉饭"],
                "recalled_contains": ["牛肉饭"],
                "saved_contains": ["牛肉饭"],
                "debug_equals": {"debug.memory.event_recall.strategy": "temporal_range"},
            },
            new_memories=[{"id": "m1", "kind": "event", "content": "昨天中午吃了牛肉饭"}],
            all_memories=[{"id": "m1", "kind": "event", "content": "昨天中午吃了牛肉饭"}],
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_evaluate_turn_normalizes_case_unicode_and_spacing(self) -> None:
        result = evaluate_turn(
            response={
                "reply": "H₂O。",
                "recalled_memories": [{"content": "用户名字叫jack"}],
                "debug": {},
                "completed": True,
            },
            exception="",
            expect={
                "reply_contains_any": ["H2O"],
                "recalled_contains": ["用户名字叫 jack"],
                "saved_contains": ["coco", "杨枝甘露"],
            },
            new_memories=[{"id": "m1", "kind": "profile", "content": "最喜欢喝的饮品是 CoCo 的杨枝甘露"}],
            all_memories=[],
        )

        self.assertTrue(result["passed"])

    def test_evaluate_turn_checks_observation_document_and_timeline_recall(self) -> None:
        result = evaluate_turn(
            response={
                "reply": "你最近在推进 AI 眼镜长期记忆 demo。",
                "recalled_memories": [
                    {
                        "content": "用户最近主要在推进 AI 眼镜长期记忆 demo",
                        "memory_type": "observation",
                    }
                ],
                "recalled_documents": [
                    {
                        "filename": "南太行自驾攻略.md",
                        "title": "南太行自驾攻略",
                        "summary": "红旗渠门票信息",
                    }
                ],
                "recalled_timeline_chunks": [{"text": "国内网络下语音识别不稳定，可能需要 VPN。"}],
                "debug": {},
                "completed": True,
            },
            exception="",
            expect={
                "recalled_memory_type": "observation",
                "recalled_not_memory_type": "task",
                "recalled_document_contains": ["南太行自驾攻略"],
                "recalled_timeline_contains": ["国内网络下语音识别不稳定"],
            },
            new_memories=[],
            all_memories=[],
        )

        self.assertTrue(result["passed"])

    def test_evaluate_turn_supports_target_eval_checks(self) -> None:
        result = evaluate_turn(
            response={
                "reply": "收到，我会保持安静记录。",
                "triggered": False,
                "debug": {"reminder_check": {"reason": "user_not_allowed"}},
                "completed": True,
            },
            exception="",
            expect={
                "reply_not_contains_any": ["请问", "需要我回答"],
                "saved_count_min": 2,
                "not_triggered": True,
                "debug_equals": {"debug.reminder_check.reason": "user_not_allowed"},
            },
            new_memories=[
                {"id": "m1", "kind": "event", "content": "会议目标是语音主路径"},
                {"id": "m2", "kind": "event", "content": "风险点是语音权限不稳定"},
            ],
            all_memories=[],
        )

        self.assertTrue(result["passed"])

    def test_saved_kind_ignores_derived_observation_memories(self) -> None:
        result = evaluate_turn(
            response={
                "reply": "好的，我记住了。",
                "recalled_memories": [],
                "debug": {},
                "completed": True,
            },
            exception="",
            expect={
                "saved_contains": ["安静", "靠窗"],
                "saved_kind": "profile",
            },
            new_memories=[
                {
                    "id": "m1",
                    "kind": "event",
                    "memory_type": "observation",
                    "source": "observation_reflect",
                    "content": "最近稳定偏好包括用户喜欢安静靠窗的位置。",
                },
                {
                    "id": "m2",
                    "kind": "profile",
                    "memory_type": "preference",
                    "source": "chat-auto",
                    "content": "用户喜欢安静靠窗的位置",
                },
            ],
            all_memories=[],
        )

        self.assertTrue(result["passed"])

    def test_evaluate_turn_checks_triggered_true(self) -> None:
        result = evaluate_turn(
            response={
                "reply": "Alex 周会还有 10 分钟开始。",
                "triggered": True,
                "debug": {},
                "completed": True,
            },
            exception="",
            expect={"triggered": True, "reply_contains_any": ["Alex", "周会"]},
            new_memories=[],
            all_memories=[],
        )

        self.assertTrue(result["passed"])

    def test_evaluate_turn_checks_active_memory_and_debug_payload(self) -> None:
        result = evaluate_turn(
            response={
                "reply": "好的，这次不按靠窗偏好。",
                "debug": {
                    "memory": {
                        "drift_guard": {
                            "filtered_reasons": ["current_intent_conflicts_with_preference"],
                        }
                    }
                },
                "completed": True,
            },
            exception="",
            expect={
                "active_memory_contains": ["用户更喜欢吧台位置"],
                "active_memory_not_contains": ["用户喜欢靠窗位置"],
                "active_memory_count": 1,
                "debug_contains": ["current_intent_conflicts_with_preference"],
            },
            new_memories=[],
            all_memories=[
                {"id": "m1", "kind": "profile", "content": "用户更喜欢吧台位置"},
            ],
        )

        self.assertTrue(result["passed"])

    def test_evaluate_turn_fails_missing_active_memory_and_debug_payload(self) -> None:
        result = evaluate_turn(
            response={
                "reply": "好的。",
                "debug": {"memory": {"drift_guard": {"filtered_reasons": []}}},
                "completed": True,
            },
            exception="",
            expect={
                "active_memory_contains": ["用户更喜欢吧台位置"],
                "debug_contains": ["current_intent_conflicts_with_preference"],
            },
            new_memories=[],
            all_memories=[],
        )

        self.assertFalse(result["passed"])
        self.assertTrue(any("active_memory_contains" in failure for failure in result["failures"]))
        self.assertTrue(any("debug_contains" in failure for failure in result["failures"]))

    def test_evaluate_turn_checks_debug_path_contains_values_inside_lists(self) -> None:
        result = evaluate_turn(
            response={
                "reply": "好的。",
                "debug": {
                    "memory_processing": {
                        "observation_update_decisions": [
                            {
                                "action": "new",
                                "confidence_policy": {
                                    "purpose": "observation_update_relationship",
                                    "treatment": "fallback_to_new",
                                    "passed": False,
                                },
                            }
                        ]
                    }
                },
                "completed": True,
            },
            exception="",
            expect={
                "debug_path_contains": {
                    "debug.memory_processing.observation_update_decisions.*.confidence_policy.purpose": "observation_update_relationship",
                    "debug.memory_processing.observation_update_decisions.*.confidence_policy.treatment": "fallback_to_new",
                    "debug.memory_processing.observation_update_decisions.*.confidence_policy.passed": False,
                }
            },
            new_memories=[],
            all_memories=[],
        )

        self.assertTrue(result["passed"])
        missing = evaluate_turn(
            response={
                "reply": "好的。",
                "debug": {
                    "memory_processing": {
                        "observation_update_decisions": [
                            {"confidence_policy": {"treatment": "apply_decision"}}
                        ]
                    }
                },
                "completed": True,
            },
            exception="",
            expect={
                "debug_path_contains": {
                    "debug.memory_processing.observation_update_decisions.*.confidence_policy.treatment": "fallback_to_new"
                }
            },
            new_memories=[],
            all_memories=[],
        )

        self.assertFalse(missing["passed"])
        self.assertTrue(any("debug_path_contains" in failure for failure in missing["failures"]))

    def test_summarize_runs_reports_failures_and_latency(self) -> None:
        runs = [
            {
                "scenario_id": "s1",
                "category": "isolation",
                "passed": False,
                "turns": [
                    {
                        "scenario_id": "s1",
                        "repeat_index": 1,
                        "turn_index": 1,
                        "message": "我喜欢喝什么？",
                        "passed": False,
                        "checks": [{"name": "reply_contains", "passed": False}],
                        "response": {
                            "completed": True,
                            "api_calls": 2,
                            "debug": {
                                "timing": {
                                    "total_seconds": 1.2,
                                    "stages": [{"name": "assistant_response", "seconds": 0.9}],
                                }
                            },
                        },
                    }
                ],
            }
        ]

        summary = summarize_runs(runs)

        self.assertEqual(summary["total_turns"], 1)
        self.assertEqual(summary["failed_turns"], 1)
        self.assertEqual(summary["active_failed_turns"], 1)
        self.assertEqual(summary["reply_fact_hit_rate"], 0.0)
        self.assertEqual(summary["isolation_pass_rate"], 0.0)
        self.assertEqual(summary["latency_seconds"]["max"], 1.2)

    def test_summarize_runs_counts_new_recall_checks(self) -> None:
        runs = [
            {
                "scenario_id": "observation",
                "category": "observation",
                "passed": False,
                "turns": [
                    {
                        "scenario_id": "observation",
                        "repeat_index": 1,
                        "turn_index": 1,
                        "message": "我最近在忙什么？",
                        "passed": False,
                        "checks": [
                            {"name": "recalled_memory_type", "passed": True},
                            {"name": "recalled_not_memory_type", "passed": True},
                            {"name": "recalled_document_contains", "passed": False},
                            {"name": "reply_contains", "passed": True},
                        ],
                        "response": {"completed": True, "debug": {"timing": {"total_seconds": 0.1}}},
                    }
                ],
            }
        ]

        summary = summarize_runs(runs)

        self.assertEqual(summary["memory_recall_hit_rate"], 0.6667)
        self.assertEqual(summary["reply_fact_hit_rate"], 1.0)

    def test_summarize_runs_keeps_target_failures_out_of_active_failures(self) -> None:
        runs = [
            {
                "scenario_id": "target_gap",
                "category": "proactive_reminder",
                "status": "target",
                "passed": False,
                "turns": [
                    {
                        "scenario_id": "target_gap",
                        "status": "target",
                        "repeat_index": 1,
                        "turn_index": 1,
                        "message": None,
                        "passed": False,
                        "checks": [{"name": "triggered", "passed": False}],
                        "response": {
                            "completed": True,
                            "triggered": False,
                            "debug": {"timing": {"total_seconds": 0.0, "stages": []}},
                        },
                    }
                ],
            }
        ]

        summary = summarize_runs(runs)

        self.assertEqual(summary["failed_turns"], 1)
        self.assertEqual(summary["active_failed_turns"], 0)
        self.assertEqual(summary["target_failed_turns"], 1)
        self.assertEqual(summary["active_pass_rate"], 1.0)

    def test_filter_scenarios_by_category_and_id(self) -> None:
        scenarios = [
            {"id": "identity", "category": "identity"},
            {"id": "pressure", "category": "pressure"},
            {"id": "negative", "category": "negative"},
            {"id": "continuous", "category": "continuous_input", "status": "target"},
        ]

        self.assertEqual(
            filter_scenarios(scenarios, scenario_ids=[], categories=["pressure"]),
            [{"id": "pressure", "category": "pressure"}],
        )
        self.assertEqual(
            filter_scenarios(scenarios, scenario_ids=["negative"], categories=[]),
            [{"id": "negative", "category": "negative"}],
        )
        self.assertEqual(
            filter_scenarios(scenarios, scenario_ids=[], categories=["continuous_input"]),
            [{"id": "continuous", "category": "continuous_input", "status": "target"}],
        )

    def test_run_turn_waits_for_memory_job_terminal_after_saved_terms(self) -> None:
        class SlowJobService:
            def __init__(self, store: EventMemoryStore) -> None:
                self.store = store
                self.reads = 0

            def chat(self, *args, **kwargs) -> dict:
                self.store.add_memory(
                    "u1",
                    "用户喜欢安静靠窗的位置",
                    kind="profile",
                    source="test",
                )
                return {
                    "reply": "我先记下。",
                    "completed": True,
                    "debug": {"memory_processing": {"job_id": "memjob_eval_wait", "status": "pending"}},
                }

            def read_memory_job(self, *, user_id: str, job_id: str) -> dict:
                self.reads += 1
                status = "saved" if self.reads >= 3 else "running"
                return {"user_id": user_id, "job_id": job_id, "status": status}

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = SlowJobService(store)

            result = run_turn(
                service=service,
                store=store,
                scenario={"id": "s1", "category": "memory", "user_id": "u1"},
                turn={
                    "message": "我喜欢安静靠窗的位置",
                    "expect": {"saved_contains": ["安静", "靠窗"]},
                },
                repeat_index=1,
                turn_index=1,
                background_wait=1.0,
            )

        self.assertTrue(result["passed"])
        self.assertGreaterEqual(service.reads, 3)

    def test_run_turn_reads_new_memories_after_background_job_saves(self) -> None:
        class DelayedSaveJobService:
            def __init__(self, store: EventMemoryStore) -> None:
                self.store = store
                self.reads = 0

            def chat(self, *args, **kwargs) -> dict:
                return {
                    "reply": "收到，我先整理。",
                    "completed": True,
                    "debug": {"memory_processing": {"job_id": "memjob_delayed_save", "status": "pending"}},
                }

            def read_memory_job(self, *, user_id: str, job_id: str) -> dict:
                self.reads += 1
                if self.reads >= 2 and not self.store.list_memories(user_id):
                    self.store.add_memory(
                        user_id,
                        "上午十点去园区见 Mia 看语音按钮",
                        kind="event",
                        source="test",
                    )
                    self.store.add_memory(
                        user_id,
                        "下午三点和 Alex 过后台记忆写入",
                        kind="event",
                        source="test",
                    )
                status = "saved" if self.reads >= 2 else "running"
                return {"user_id": user_id, "job_id": job_id, "status": status}

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = DelayedSaveJobService(store)

            result = run_turn(
                service=service,
                store=store,
                scenario={"id": "s1", "category": "memory", "user_id": "u1"},
                turn={
                    "message": "上午见 Mia，下午和 Alex 过后台写入。",
                    "expect": {
                        "saved_count_min": 2,
                        "saved_contains": ["Mia", "Alex"],
                    },
                },
                repeat_index=1,
                turn_index=1,
                background_wait=1.0,
            )

        self.assertTrue(result["passed"], result["failures"])
        self.assertGreaterEqual(service.reads, 2)

    def test_run_turn_waits_for_saved_count_min_after_job_terminal(self) -> None:
        class DelayedCountJobService:
            def __init__(self, store: EventMemoryStore) -> None:
                self.store = store
                self.reads = 0

            def chat(self, *args, **kwargs) -> dict:
                return {
                    "reply": "收到，我先整理。",
                    "completed": True,
                    "debug": {"memory_processing": {"job_id": "memjob_saved_count_wait", "status": "pending"}},
                }

            def read_memory_job(self, *, user_id: str, job_id: str) -> dict:
                self.reads += 1
                return {
                    "user_id": user_id,
                    "job_id": job_id,
                    "status": "saved" if self.reads >= 2 else "running",
                }

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = DelayedCountJobService(store)
            writes = {"count": 0}

            def delayed_sleep(seconds: float) -> None:
                writes["count"] += 1
                if writes["count"] == 2:
                    store.add_memory("u1", "上午十点去园区见 Mia 看语音按钮", kind="event", source="test")
                    store.add_memory("u1", "下午三点和 Alex 过后台记忆写入", kind="event", source="test")

            with patch("ai_glasses_memory_assistant.evals.runner.time.sleep", side_effect=delayed_sleep):
                result = run_turn(
                    service=service,
                    store=store,
                    scenario={"id": "s1", "category": "memory", "user_id": "u1"},
                    turn={
                        "message": "上午见 Mia，下午和 Alex 过后台写入。",
                        "expect": {
                            "saved_count_min": 2,
                            "saved_contains": ["Mia", "Alex"],
                        },
                    },
                    repeat_index=1,
                    turn_index=1,
                    background_wait=1.0,
                )

        self.assertTrue(result["passed"], result["failures"])
        self.assertGreaterEqual(service.reads, 2)
        self.assertGreaterEqual(writes["count"], 2)

    def test_run_turn_waits_for_observation_memory_job_terminal(self) -> None:
        class ObservationJobService:
            def __init__(self, store: EventMemoryStore) -> None:
                self.store = store
                self.read_jobs: list[str] = []

            def chat(self, *args, **kwargs) -> dict:
                self.store.add_memory(
                    "u1",
                    "用户喜欢安静靠窗的位置",
                    kind="profile",
                    source="test",
                )
                return {
                    "reply": "我先记下。",
                    "completed": True,
                    "debug": {
                        "memory_processing": {
                            "status": "saved",
                            "observation_job_id": "memjob_observation_wait",
                        }
                    },
                }

            def read_memory_job(self, *, user_id: str, job_id: str) -> dict:
                self.read_jobs.append(job_id)
                status = "saved" if len(self.read_jobs) >= 2 else "running"
                return {"user_id": user_id, "job_id": job_id, "status": status}

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = ObservationJobService(store)

            result = run_turn(
                service=service,
                store=store,
                scenario={"id": "s1", "category": "memory", "user_id": "u1"},
                turn={
                    "message": "我喜欢安静靠窗的位置",
                    "expect": {"saved_contains": ["安静", "靠窗"]},
                },
                repeat_index=1,
                turn_index=1,
                background_wait=1.0,
            )

        self.assertTrue(result["passed"])
        self.assertIn("memjob_observation_wait", service.read_jobs)
        self.assertGreaterEqual(len(service.read_jobs), 2)

    def test_run_turn_attaches_terminal_memory_job_debug_for_eval_assertions(self) -> None:
        class ObservationJobService:
            def __init__(self, store: EventMemoryStore) -> None:
                self.store = store

            def chat(self, *args, **kwargs) -> dict:
                return {
                    "reply": "我先记下。",
                    "completed": True,
                    "debug": {
                        "memory_processing": {
                            "status": "saved",
                            "observation_job_id": "memjob_observation_policy",
                        }
                    },
                }

            def read_memory_job(self, *, user_id: str, job_id: str) -> dict:
                return {
                    "user_id": user_id,
                    "job_id": job_id,
                    "status": "saved",
                    "observation_update_decisions": [
                        {
                            "action": "new",
                            "confidence_policy": {
                                "purpose": "observation_update_relationship",
                                "treatment": "fallback_to_new",
                                "passed": False,
                            },
                        }
                    ],
                }

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = ObservationJobService(store)

            result = run_turn(
                service=service,
                store=store,
                scenario={"id": "s1", "category": "memory", "user_id": "u1"},
                turn={
                    "message": "我最近在做 AI 眼镜长期记忆",
                    "expect": {
                        "debug_path_contains": {
                            "debug.memory_jobs.*.observation_update_decisions.*.confidence_policy.treatment": "fallback_to_new"
                        }
                    },
                },
                repeat_index=1,
                turn_index=1,
                background_wait=1.0,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["response"]["debug"]["memory_jobs"][0]["job_id"],
            "memjob_observation_policy",
        )

    def test_run_turn_waits_for_known_memory_jobs_not_in_response_debug(self) -> None:
        class HiddenJobService:
            def __init__(self, store: EventMemoryStore) -> None:
                self.store = store
                self._memory_jobs = {
                    "memjob_hidden": {
                        "job_id": "memjob_hidden",
                        "user_id": "u1",
                        "status": "running",
                    }
                }

            def chat(self, *args, **kwargs) -> dict:
                self.store.add_memory(
                    "u1",
                    "用户喜欢安静靠窗的位置",
                    kind="profile",
                    source="test",
                )
                return {
                    "reply": "我先记下。",
                    "completed": True,
                    "debug": {"memory_processing": {"status": "saved"}},
                }

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = HiddenJobService(store)
            checks = {"count": 0}

            def complete_hidden_job(seconds: float) -> None:
                checks["count"] += 1
                if checks["count"] >= 2:
                    service._memory_jobs["memjob_hidden"]["status"] = "saved"

            with patch("ai_glasses_memory_assistant.evals.runner.time.sleep", side_effect=complete_hidden_job):
                result = run_turn(
                    service=service,
                    store=store,
                    scenario={"id": "s1", "category": "memory", "user_id": "u1"},
                    turn={
                        "message": "我喜欢安静靠窗的位置",
                        "expect": {"saved_contains": ["安静", "靠窗"]},
                    },
                    repeat_index=1,
                    turn_index=1,
                    background_wait=1.0,
                )

        self.assertTrue(result["passed"])
        self.assertGreaterEqual(checks["count"], 2)
        self.assertEqual(service._memory_jobs["memjob_hidden"]["status"], "saved")

    def test_preload_memories_preserves_observation_trace_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")

            _preload_memories(
                store,
                {
                    "preload": [
                        {
                            "kind": "event",
                            "memory_type": "observation",
                            "content": "用户最近在推进 AI 眼镜长期记忆 demo",
                            "source": "observation_reflect",
                            "source_id": "reflect-job-1",
                            "ingestion_id": "ing-1",
                            "evidence_ids": ["chunk_1", "chunk_2"],
                            "privacy_level": "normal",
                            "confidence": 0.82,
                        }
                    ]
                },
                default_user_id="u1",
            )

            memories = store.list_memories("u1", limit=10)
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].memory_type, "observation")
            self.assertEqual(memories[0].source, "observation_reflect")
            self.assertEqual(memories[0].source_id, "reflect-job-1")
            self.assertEqual(memories[0].ingestion_id, "ing-1")
            self.assertEqual(memories[0].evidence_ids, ["chunk_1", "chunk_2"])
            self.assertEqual(memories[0].privacy_level, "normal")
            self.assertEqual(memories[0].confidence, 0.82)

    def test_preload_documents_imports_markdown_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = GlassesChatService(memory_store=store, clock=lambda: 1778131200.0)

            _preload_documents(
                service,
                {
                    "preload_documents": [
                        {
                            "filename": "南太行自驾攻略.md",
                            "text": "# 南太行自驾攻略\n红旗渠门票 80 元",
                        }
                    ]
                },
                default_user_id="u1",
            )

            documents = store.list_documents("u1")
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].filename, "南太行自驾攻略.md")
            self.assertEqual(documents[0].source, "markdown_upload")
            self.assertIn("红旗渠门票 80 元", documents[0].content)

    def test_run_turn_supports_memory_import_action_for_app_audio_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = GlassesChatService(memory_store=store, clock=lambda: 1778131200.0)

            result = run_turn(
                service=service,
                store=store,
                scenario={"id": "s1", "category": "app_audio_transcript", "user_id": "u1"},
                turn={
                    "action": "memory_import",
                    "source": "app_audio_transcript",
                    "context": "walking transcript",
                    "items": [
                        {
                            "content": "AI 眼镜项目：Mia 负责语音按钮",
                            "kind": "event",
                            "memory_type": "task",
                            "confidence": 0.95,
                        }
                    ],
                    "expect": {
                        "saved_contains": ["Mia", "语音按钮"],
                        "debug_equals": {"debug.eval_action.action": "memory_import"},
                        "debug_path_contains": {
                            "debug.eval_action.source_summary.source_types.*": ["memory_write"],
                            "debug.eval_action.cleaning_trace.summary.segment_count": 1,
                        },
                    },
                },
                repeat_index=1,
                turn_index=1,
                background_wait=0.0,
            )

        self.assertTrue(result["passed"], result["failures"])

    def test_run_turn_supports_delete_timeline_chunks_action_by_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = GlassesChatService(memory_store=store, clock=lambda: 1778131200.0)
            service.timeline_store.add_turn("u1", "刚才说过语音识别不稳定")

            result = run_turn(
                service=service,
                store=store,
                scenario={"id": "s1", "category": "source_control", "user_id": "u1"},
                turn={
                    "action": "delete_timeline_chunks",
                    "query": "语音识别不稳定",
                    "expect": {
                        "api_calls": 0,
                        "debug_equals": {"debug.eval_action.deleted_count": 1},
                    },
                },
                repeat_index=1,
                turn_index=1,
                background_wait=0.0,
            )

            chunks = service.timeline_store.search_chunks("u1", "语音识别不稳定", limit=5)

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(chunks, [])

    def test_run_turn_supports_ambient_wake_query_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = GlassesChatService(memory_store=store, clock=lambda: 1778131200.0)
            _install_eval_pre_reply_decision_agent(service, {"pre_reply_payloads": {"*": {"reply_mode": "llm", "answer_source": "llm"}}})

            result = run_turn(
                service=service,
                store=store,
                scenario={
                    "id": "ambient",
                    "category": "ambient_emotion_metadata",
                    "user_id": "u1",
                    "reference_time": "2026-05-22T10:00:00+08:00",
                },
                turn={
                    "action": "ambient_wake_query",
                    "ambient_chunks": [
                        {
                            "text": "服了，又来了",
                            "metadata": {
                                "emotion_label": "烦躁",
                                "emotion_intensity": 4,
                            },
                        }
                    ],
                    "message": "你觉得刚才他是不是在阴阳我？",
                    "expect": {
                        "reply_contains": ["服了，又来了"],
                        "no_saved": True,
                        "debug_equals": {
                            "debug.ambient_context.used": True,
                            "debug.ambient_context.injected_to_main_llm": True,
                            "debug.eval_action.action": "ambient_wake_query",
                        },
                    },
                },
                repeat_index=1,
                turn_index=1,
                background_wait=0.0,
            )

        self.assertTrue(result["passed"], result["failures"])

    def test_eval_pre_reply_payloads_install_fake_pre_reply_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"HERMES_HOME": tmpdir}):
            store = EventMemoryStore(db_path=Path(tmpdir) / "events.db")
            service = GlassesChatService(memory_store=store, clock=lambda: 1778131200.0)
            _install_eval_pre_reply_decision_agent(
                service,
                {
                    "pre_reply_payloads": {
                        "这阵子我的主要投入方向是什么？": {
                            "reply_mode": "llm",
                            "answer_source": "llm",
                            "scope": "unknown",
                            "location_text": "",
                            "memory_recall_type": "observation",
                            "confidence": 0.93,
                            "reason": "eval pre_reply fallback",
                        }
                    }
                },
            )

            session = service._new_session(user_id="u1", session_id="eval-session")
            response = session.agent.run_conversation(
                "Route...\nUser message:\n这阵子我的主要投入方向是什么？",
                system_message="You are an internal unified pre-reply decision classifier.",
            )

            self.assertIn('"memory_recall_type": "observation"', response["final_response"])

    def test_eval_pre_reply_decision_agent_main_reply_uses_memory_context_evidence(self) -> None:
        agent = _EvalPreReplyDecisionAgent({})
        message = "\n".join([
            "<memory-context>",
            "Direct structured memory evidence:",
            "1. 今天下午去吃面",
            "2. 晚上有唱 K 活动",
            "",
            "Background profile or stable context. Use only if it directly answers the user:",
            "1. 用户名字叫Jack",
            "</memory-context>",
            "User message: 我最近都干了什么",
        ])

        response = agent.run_conversation(message)

        self.assertIn("吃面", response["final_response"])
        self.assertIn("唱 K", response["final_response"])
        self.assertNotIn("Jack", response["final_response"])

    def test_eval_agent_returns_dedupe_payload_with_resolved_memory_id(self) -> None:
        agent = _EvalPreReplyDecisionAgent(
            {},
            dedupe_payloads={
                "用户喜欢安静靠窗的位置": {
                    "action": "duplicate",
                    "memory_content_contains": ["靠窗"],
                    "confidence": 0.92,
                    "reason": "same preference",
                }
            },
        )
        prompt = "\n\n".join([
            'New candidate:\n{"content": "用户喜欢安静靠窗的位置", "confidence": 0.9}',
            'Current active profile preferences:\n[\n  {"memory_id": "pref_1", "content": "用户喜欢靠窗座位", "confidence": 0.8}\n]',
            "Return one JSON object only.",
        ])

        response = agent.run_conversation(
            prompt,
            system_message="You are a memory dedupe classifier for profile preference memories.",
        )

        self.assertIn('"action": "duplicate"', response["final_response"])
        self.assertIn('"memory_id": "pref_1"', response["final_response"])

    def test_eval_agent_returns_structured_dedupe_payload_with_resolved_memory_id(self) -> None:
        agent = _EvalPreReplyDecisionAgent(
            {},
            dedupe_payloads={
                "这周五之前把记忆机制评估补上": {
                    "action": "duplicate",
                    "memory_content_contains": ["memory eval"],
                    "confidence": 0.9,
                    "reason": "same task",
                }
            },
        )
        prompt = "\n\n".join([
            'New candidate:\n{"content": "这周五之前把记忆机制评估补上", "memory_type": "task", "confidence": 0.9}',
            'Current active structured memories:\n[\n  {"memory_id": "task_1", "content": "周五前补 memory eval", "memory_type": "task", "confidence": 0.8}\n]',
            "Return one JSON object only.",
        ])

        response = agent.run_conversation(
            prompt,
            system_message="You are a memory dedupe classifier for structured event memories.",
        )

        self.assertIn('"action": "duplicate"', response["final_response"])
        self.assertIn('"memory_id": "task_1"', response["final_response"])

    def test_eval_agent_returns_correction_and_observation_update_payloads(self) -> None:
        agent = _EvalPreReplyDecisionAgent(
            {},
            correction_payloads={
                "其实我现在主要在做 AI 眼镜长期记忆": {
                    "is_correction": True,
                    "corrected_content": "用户现在主要在做 AI 眼镜长期记忆",
                    "memory_kind": "event",
                    "memory_type": "project_state",
                    "confidence": 0.9,
                    "reason": "natural correction",
                }
            },
            observation_update_payloads={
                "用户最近主要在做 AI 眼镜长期记忆": {
                    "action": "supersede",
                    "observation_content_contains": ["南太行"],
                    "confidence": 0.91,
                    "reason": "new summary replaces old one",
                }
            },
        )

        correction = agent.run_conversation(
            "User message:\n其实我现在主要在做 AI 眼镜长期记忆\n\nReturn one JSON object only.",
            system_message="You are a memory correction classifier.",
        )
        observation = agent.run_conversation(
            "\n\n".join([
                "New observation candidate:\n用户最近主要在做 AI 眼镜长期记忆",
                "Current active observations:\n[\n  {\"memory_id\": \"obs_1\", \"content\": \"用户最近主要在做南太行项目\"}\n]",
                "Return one JSON object only.",
            ]),
            system_message="You are an observation memory update classifier.",
        )

        self.assertIn('"is_correction": true', correction["final_response"])
        self.assertIn('"action": "supersede"', observation["final_response"])
        self.assertIn('"observation_id": "obs_1"', observation["final_response"])

    def test_eval_agent_routes_correction_target_to_dedicated_payload_or_live_agent(self) -> None:
        class LiveTargetAgent:
            def __init__(self) -> None:
                self.calls = 0

            def run_conversation(
                self,
                message: str,
                system_message: str | None = None,
                conversation_history: list[dict[str, object]] | None = None,
                persist_user_message: str | None = None,
            ) -> dict[str, object]:
                self.calls += 1
                return {"final_response": json.dumps({"action": "none", "confidence": 0.4})}

        live_agent = LiveTargetAgent()
        agent = _EvalPreReplyDecisionAgent(
            {},
            dedupe_payloads={
                "我前面那个饮品偏好更新一下，我最近不喝咖啡了": {
                    "action": "duplicate",
                    "memory_content_contains": ["靠窗"],
                    "confidence": 0.9,
                }
            },
            correction_target_payloads={
                "我前面那个饮品偏好更新一下，我最近不喝咖啡了": {
                    "action": "supersede",
                    "memory_content_contains": ["冰美式"],
                    "confidence": 0.9,
                    "reason": "target payload should be independent from dedupe payloads",
                }
            },
            live_correction_target_agent=live_agent,
        )
        prompt = "\n\n".join([
            "User message:\n我前面那个饮品偏好更新一下，我最近不喝咖啡了",
            'Replacement memory:\n{"content": "用户最近不喝咖啡", "memory_type": "preference"}',
            'Candidate active memories:\n[\n  {"memory_id": "pref_drink", "content": "用户喜欢冰美式"},\n  {"memory_id": "pref_seat", "content": "用户喜欢靠窗座位"}\n]',
            "Return one JSON object only.",
        ])

        response = agent.run_conversation(
            prompt,
            system_message="You are a memory correction target resolver.",
        )

        self.assertIn('"action": "supersede"', response["final_response"])
        self.assertIn('"memory_id": "pref_drink"', response["final_response"])
        self.assertEqual(live_agent.calls, 0)

        live_only = _EvalPreReplyDecisionAgent(
            {},
            dedupe_payloads={
                "我前面那个饮品偏好更新一下，我最近不喝咖啡了": {
                    "action": "duplicate",
                    "memory_content_contains": ["靠窗"],
                    "confidence": 0.9,
                }
            },
            live_correction_target_agent=live_agent,
        )
        live_response = live_only.run_conversation(
            prompt,
            system_message="You are a memory correction target resolver.",
        )

        self.assertIn('"action": "none"', live_response["final_response"])
        self.assertEqual(live_agent.calls, 1)


if __name__ == "__main__":
    unittest.main()
