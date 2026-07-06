from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_glasses_memory_assistant.text_cleaning import clean_text_for_memory
from ai_glasses_memory_assistant.segment_semantic_cleaner import (
    _decision_from_payload,
    fallback_segment_semantic_decision,
)


class TextCleaningTests(unittest.TestCase):
    def test_normalizes_unicode_whitespace_and_duplicate_punctuation(self) -> None:
        trace = clean_text_for_memory("  ＡＩ   眼镜！！！\n\nMia\t负责  验收  ")

        self.assertEqual(trace.normalized_text, "AI 眼镜!\nMia 负责 验收")
        self.assertEqual([segment.text for segment in trace.segments], ["AI 眼镜", "Mia 负责 验收"])

    def test_redacts_sensitive_text_before_trace_output(self) -> None:
        secret = "sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
        trace = clean_text_for_memory(f"验证码是123456，token={secret}")
        payload = trace.debug_payload()

        self.assertTrue(payload["redacted"])
        self.assertIn("code", payload["redaction_categories"])
        self.assertIn("token", payload["redaction_categories"])
        self.assertNotIn(secret, payload["normalized_text"])
        self.assertIn("[已脱敏:token]", payload["normalized_text"])

    def test_marks_fillers_noise_corrections_and_negations(self) -> None:
        trace = clean_text_for_memory("嗯刚才有点吵，那个 Alex 周五交第一版，哦不对，应该是周四晚上，不是 coco 是喜茶，哈哈")
        payload = trace.debug_payload()

        self.assertGreaterEqual(payload["summary"]["noise_marker_count"], 3)
        self.assertGreaterEqual(payload["summary"]["correction_marker_count"], 2)
        self.assertGreaterEqual(payload["summary"]["negation_marker_count"], 1)
        marker_kinds = {
            kind
            for segment in payload["segments"]
            for kind in segment["marker_kinds"]
        }
        self.assertIn("correction:negate_previous", marker_kinds)
        self.assertIn("negation:not_preference", marker_kinds)

    def test_sensitive_reject_role_does_not_extract_memory_span(self) -> None:
        decision = _decision_from_payload(
            {
                "semantic_role": "sensitive_reject",
                "should_extract": True,
                "candidate_span": "token sk-test-abcdef",
                "candidate_hint": "sensitive",
                "confidence": 0.9,
            },
            segment_index=0,
            raw_span="token sk-test-abcdef 不要保存",
            raw="{}",
            backend="llm",
        )

        self.assertEqual(decision.semantic_role, "sensitive_reject")
        self.assertFalse(decision.should_extract)
        self.assertEqual(decision.candidate_span, "token sk-test-abcdef")

    def test_fallback_segment_semantic_decision_keeps_task_after_local_do_not_remember(self) -> None:
        decision = fallback_segment_semantic_decision(
            segment="嗯嗯，外卖电话不用记，但 Mia 明天要验证语音按钮",
            segment_index=0,
            marker_kinds=["filler:um", "correction:do_not_remember"],
        )

        self.assertEqual(decision.semantic_role, "memory_candidate")
        self.assertTrue(decision.should_extract)
        self.assertEqual(decision.do_not_remember_scope, "嗯嗯，外卖电话不用记")
        self.assertEqual(decision.candidate_span, "Mia 明天要验证语音按钮")

    def test_fallback_segment_semantic_decision_marks_correction_segments_extractable(self) -> None:
        decision = fallback_segment_semantic_decision(
            segment="哦不对，应该是周四晚上交第一版",
            segment_index=0,
            marker_kinds=["correction:negate_previous", "correction:replace_with"],
        )

        self.assertEqual(decision.semantic_role, "correction")
        self.assertTrue(decision.should_extract)
        self.assertEqual(decision.candidate_span, "周四晚上交第一版")

    def test_fallback_segment_semantic_decision_extracts_high_value_span_from_mixed_background(self) -> None:
        decision = fallback_segment_semantic_decision(
            segment="Alex 只是旁边插了一句背景有点吵，但 Mia 周五前交第一版，后面很多口头禅和停顿。",
            segment_index=0,
            marker_kinds=["noise:background"],
        )

        self.assertEqual(decision.semantic_role, "memory_candidate")
        self.assertTrue(decision.should_extract)
        self.assertEqual(decision.candidate_span, "Mia 周五前交第一版")
        self.assertEqual(decision.candidate_hint, "task")


if __name__ == "__main__":
    unittest.main()
