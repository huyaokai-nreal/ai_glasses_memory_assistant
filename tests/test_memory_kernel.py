from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_glasses_memory_assistant.memory_kernel import memory_kernel_contract, recall_trace, source_trace
from ai_glasses_memory_assistant.memory_store import (
    EventMemoryStore,
    classify_memory_kind_with_reason,
    classify_memory_type,
    classify_memory_type_with_reason,
    extract_memory_candidates,
)


class MemoryKernelTests(unittest.TestCase):
    def test_contract_names_required_layers(self) -> None:
        contract = memory_kernel_contract()

        names = [layer["name"] for layer in contract["layers"]]

        self.assertEqual(
            names,
            ["raw_timeline", "structured_memory", "reflection", "runtime_recall"],
        )
        self.assertEqual(contract["write_contract"]["gate"], "should_write_memory_candidate")
        self.assertIn("evidence_ids", contract["write_contract"]["required_fields"])

    def test_source_trace_dedupes_evidence_and_keeps_delete_policy(self) -> None:
        trace = source_trace(
            layer="raw_timeline",
            user_id="u1",
            source="chat",
            source_id="turn_1",
            evidence_ids=["chunk_1", "chunk_1", ""],
            status="active",
        )

        self.assertEqual(trace["evidence_ids"], ["chunk_1"])
        self.assertEqual(trace["delete_policy"], "soft_delete_chunk; parent turn/capture becomes partial_deleted or deleted")

    def test_recall_trace_marks_runtime_strategy(self) -> None:
        trace = recall_trace(
            layer="structured_memory",
            strategy="temporal_range",
            count=2,
            reason="yesterday",
            evidence_ids=["chunk_a", "chunk_b"],
        )

        self.assertEqual(trace["strategy"], "temporal_range")
        self.assertEqual(trace["count"], 2)
        self.assertEqual(trace["evidence_ids"], ["chunk_a", "chunk_b"])

    def test_legacy_memory_extractor_allows_question_form_memory_request(self) -> None:
        candidates = extract_memory_candidates("你能帮我记一下明天下午3点和 Mina 开会吗？")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["content"], "明天下午3点和 Mina 开会")
        self.assertEqual(candidates[0]["kind"], "event")
        self.assertEqual(candidates[0]["memory_type"], "event")
        self.assertEqual(candidates[0]["source"], "rule_fallback")
        self.assertIn("legacy_memory_extractor", candidates[0]["reason"])
        self.assertEqual(candidates[0]["classification"]["memory_type"]["source"], "legacy_default")
        self.assertEqual(extract_memory_candidates("明天下午3点和 Mina 开会是啥意思？"), [])

    def test_legacy_memory_type_classifier_does_not_treat_bare_project_as_project_state(self) -> None:
        classification = classify_memory_type_with_reason("AI 眼镜项目继续补 debug")

        self.assertEqual(classify_memory_type("AI 眼镜项目继续补 debug"), "event")
        self.assertEqual(classification.value, "event")
        self.assertEqual(classification.source, "legacy_default")
        self.assertEqual(classification.reason, "no_type_marker")

    def test_legacy_memory_type_classifier_exposes_project_state_phrase_source(self) -> None:
        classification = classify_memory_type_with_reason("AI 眼镜项目风险是后台保存反馈不明显")

        self.assertEqual(classify_memory_type("AI 眼镜项目风险是后台保存反馈不明显"), "project_state")
        self.assertEqual(classification.value, "project_state")
        self.assertEqual(classification.source, "legacy_phrase_classifier")
        self.assertEqual(classification.reason, "project_state_marker")
        self.assertEqual(classification.marker, "风险")

    def test_legacy_memory_type_classifier_requires_context_for_preference_and_reminder(self) -> None:
        reminder_question = classify_memory_type_with_reason("提醒是什么意思")
        preference_question = classify_memory_type_with_reason("喜欢是什么意思")
        reminder_task = classify_memory_type_with_reason("提醒我明天下午3点和 Mina 开会")
        preference = classify_memory_type_with_reason("我喜欢低糖拿铁")

        self.assertEqual(reminder_question.value, "event")
        self.assertEqual(reminder_question.source, "legacy_default")
        self.assertEqual(reminder_question.reason, "question_text_no_type_marker")
        self.assertEqual(preference_question.value, "event")
        self.assertEqual(preference_question.source, "legacy_default")
        self.assertEqual(preference_question.reason, "question_text_no_type_marker")
        self.assertEqual(reminder_task.value, "task")
        self.assertEqual(reminder_task.source, "legacy_phrase_classifier")
        self.assertEqual(reminder_task.marker, "提醒")
        self.assertEqual(preference.value, "preference")
        self.assertEqual(preference.source, "legacy_phrase_classifier")
        self.assertEqual(preference.marker, "我喜欢")

    def test_legacy_memory_kind_classifier_does_not_treat_identity_question_as_profile(self) -> None:
        question = classify_memory_kind_with_reason("我是什么身份")
        profile = classify_memory_kind_with_reason("我是 XREAL 的员工")

        self.assertEqual(question.value, "event")
        self.assertEqual(question.source, "legacy_default")
        self.assertEqual(question.reason, "question_text_no_kind_marker")
        self.assertEqual(profile.value, "profile")
        self.assertEqual(profile.source, "legacy_phrase_classifier")
        self.assertEqual(profile.marker, "我是")

    def test_add_memory_defaults_type_from_structured_kind_not_phrase_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventMemoryStore(Path(tmpdir) / "events.db")

            event = store.add_memory("u1", "AI 眼镜项目继续补 debug", kind="event")
            profile = store.add_memory("u1", "用户喜欢低糖拿铁", kind="profile")

            self.assertEqual(event.kind, "event")
            self.assertEqual(event.memory_type, "event")
            self.assertEqual(profile.kind, "profile")
            self.assertEqual(profile.memory_type, "preference")


if __name__ == "__main__":
    unittest.main()
