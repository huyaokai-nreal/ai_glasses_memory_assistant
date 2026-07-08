from __future__ import annotations

import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tests.helpers import isolated_app_home

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_glasses_memory_assistant.memory_evidence import EvidenceReferenceCounts
from ai_glasses_memory_assistant.timeline_store import TimelineStore


class TimelineStoreTests(unittest.TestCase):
    def make_store(self, tmp: Path) -> TimelineStore:
        return TimelineStore(db_path=tmp / "timeline.db")

    def test_add_turn_persists_raw_text_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))

            result = store.add_turn(
                "u1",
                "今天项目会上 Mia 负责前端交互，Alex 提到国内网络不稳定。",
                assistant_reply="收到",
                legacy_session_id="legacy-session",
                created_at=1778131200.0,
            )

            turn = store.get_turn("u1", result.turn.id)
            self.assertIsNotNone(turn)
            self.assertEqual(turn.raw_text, "今天项目会上 Mia 负责前端交互，Alex 提到国内网络不稳定。")
            self.assertEqual(turn.assistant_reply, "收到")
            self.assertEqual(turn.legacy_session_id, "legacy-session")
            self.assertEqual(len(result.chunks), 1)
            self.assertEqual(result.chunks[0].parent_id, result.turn.id)

    def test_add_turn_redacts_sensitive_raw_text_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            secret = "sk-1234567890abcdefghijklmnopqr"
            jwt = "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0NTY3ODkw.SflKxwRJSMeKKF2QT4fwpM"

            result = store.add_turn(
                "u1",
                f"我的 token 是 {secret}，Authorization: Bearer {jwt}",
                created_at=1778131200.0,
            )
            turn = store.get_turn("u1", result.turn.id)
            chunks = store.search_chunks("u1", "脱敏", limit=5)

            self.assertIsNotNone(turn)
            self.assertNotIn(secret, turn.raw_text)
            self.assertNotIn(jwt, turn.raw_text)
            self.assertIn("[已脱敏:token]", turn.raw_text)
            self.assertTrue(result.redaction.redacted)
            self.assertIn("token", result.redaction.categories)
            self.assertGreaterEqual(result.redaction.count, 2)
            self.assertEqual(len(chunks), 1)
            self.assertNotIn(secret, chunks[0].text)
            self.assertTrue(chunks[0].metadata["redacted"])
            self.assertIn("token", chunks[0].metadata["redaction_categories"])

    def test_search_chunks_uses_user_scope_and_chinese_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            store.add_turn("u1", "我之前提到国内网络下语音识别不稳定", created_at=1778131200.0)
            store.add_turn("u2", "u2 也提到国内网络", created_at=1778131300.0)

            results = store.search_chunks("u1", "国内网络", limit=5)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].user_id, "u1")
            self.assertIn("语音识别不稳定", results[0].text)

    def test_update_turn_reply_does_not_change_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            result = store.add_turn("u1", "找 Alex 评审 demo", created_at=1778131200.0)

            updated = store.update_turn_reply(
                "u1",
                result.turn.id,
                "我记住了",
                legacy_session_id="s1",
                updated_at=1778131300.0,
            )
            chunks = store.search_chunks("u1", "Alex", limit=5)

            self.assertIsNotNone(updated)
            self.assertEqual(updated.assistant_reply, "我记住了")
            self.assertEqual(updated.legacy_session_id, "s1")
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].id, result.chunks[0].id)

    def test_capture_chunks_are_searchable_before_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            capture = store.add_capture("u1", context="周会", started_at=1778131200.0)

            chunk = store.add_capture_chunk(
                "u1",
                capture["capture_id"],
                "Mia 负责前端语音按钮",
                timestamp=1778131210.0,
            )
            stopped = store.finish_capture("u1", capture["capture_id"], summary="周会摘要", ended_at=1778131300.0)
            results = store.search_chunks("u1", "语音按钮", limit=5)

            self.assertTrue(stopped)
            self.assertEqual(results[0].id, chunk.id)
            self.assertEqual(results[0].parent_type, "capture")

    def test_capture_chunk_preserves_metadata_for_ambient_mvp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            capture = store.add_capture("u1", source="ambient_audio_text", context="按钮唤醒", started_at=1778131200.0)

            chunk = store.add_capture_chunk(
                "u1",
                capture["capture_id"],
                "服了，又来了",
                timestamp=1778131210.0,
                source="ambient_audio_text",
                metadata={
                    "audio_retention": "not_recorded_browser_asr_text_only",
                    "emotion_enabled": False,
                },
            )

            self.assertEqual(chunk.source, "ambient_audio_text")
            self.assertEqual(chunk.metadata["audio_retention"], "not_recorded_browser_asr_text_only")
            self.assertEqual(chunk.metadata["emotion_enabled"], False)
            reloaded = store.get_capture("u1", capture["capture_id"])
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded["chunks"][0]["metadata"]["audio_retention"], "not_recorded_browser_asr_text_only")

    def test_capture_chunk_preserves_vad_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            capture = store.add_capture("u1", source="ambient_audio", context="VAD 待机", started_at=1778131200.0)

            chunk = store.add_capture_chunk(
                "u1",
                capture["capture_id"],
                "他们还在吵预算",
                timestamp=1778131210.0,
                source="ambient_audio",
                metadata={
                    "source_type": "ambient_audio",
                    "audio_retention": "discarded_after_processing",
                    "processing_state": "ready_for_wake_context",
                    "captured_at": 1778131210.0,
                    "segment_id": "seg_demo001",
                },
            )

            self.assertEqual(chunk.metadata["source_type"], "ambient_audio")
            self.assertEqual(chunk.metadata["processing_state"], "ready_for_wake_context")
            self.assertEqual(chunk.metadata["captured_at"], 1778131210.0)
            self.assertEqual(chunk.metadata["segment_id"], "seg_demo001")
            reloaded = store.get_capture("u1", capture["capture_id"])
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded["chunks"][0]["metadata"]["segment_id"], "seg_demo001")

    def test_capture_chunk_redacts_sensitive_text_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            db_path = Path(tmpdir) / "timeline.db"
            store = TimelineStore(db_path=db_path)
            capture = store.add_capture("u1", context="周会", started_at=1778131200.0)
            secret = "Authorization: Bearer sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"

            chunk = store.add_capture_chunk(
                "u1",
                capture["capture_id"],
                f"会议提到语音按钮，{secret}",
                timestamp=1778131210.0,
            )
            reloaded = TimelineStore(db_path=db_path).get_capture("u1", capture["capture_id"])
            results = store.search_chunks("u1", "语音按钮", limit=5)

            self.assertNotIn(secret, chunk.text)
            self.assertIn("[已脱敏:token]", chunk.text)
            self.assertTrue(chunk.metadata["redacted"])
            self.assertIsNotNone(reloaded)
            self.assertNotIn(secret, reloaded["chunks"][0]["text"])
            self.assertNotIn(secret, results[0].text)

    def test_capture_state_can_be_reloaded_from_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            db_path = Path(tmpdir) / "timeline.db"
            store = TimelineStore(db_path=db_path)
            capture = store.add_capture("u1", context="周会", started_at=1778131200.0)
            chunk = store.add_capture_chunk(
                "u1",
                capture["capture_id"],
                "Mia 负责前端语音按钮",
                timestamp=1778131210.0,
            )

            reloaded = TimelineStore(db_path=db_path).get_capture("u1", capture["capture_id"])

            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded["capture_id"], capture["capture_id"])
            self.assertEqual(reloaded["status"], "running")
            self.assertEqual(reloaded["chunks"][0]["chunk_id"], chunk.id)
            self.assertEqual(reloaded["chunks"][0]["text"], "Mia 负责前端语音按钮")

    def test_memory_jobs_are_persisted_and_user_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            db_path = Path(tmpdir) / "timeline.db"
            store = TimelineStore(db_path=db_path)
            payload = {
                "job_id": "memjob_1",
                "user_id": "u1",
                "status": "pending",
                "mode": "reply_first_background",
                "created_at": 1778131200.0,
                "updated_at": 1778131200.0,
            }

            store.upsert_memory_job("u1", "memjob_1", payload)
            loaded = TimelineStore(db_path=db_path).get_memory_job("u1", "memjob_1")
            wrong_user = TimelineStore(db_path=db_path).get_memory_job("u2", "memjob_1")

            self.assertEqual(loaded["status"], "pending")
            self.assertEqual(loaded["mode"], "reply_first_background")
            self.assertIsNone(wrong_user)

    def test_delete_chunks_removes_chunks_from_search_and_recent_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            result = store.add_turn("u1", "我叫 Jack，喜欢低糖拿铁", created_at=1778131200.0)

            deleted = store.delete_chunks("u1", [result.chunks[0].id])
            search_results = store.search_chunks("u1", "Jack", limit=5)
            recent_results = store.list_recent_chunks("u1", limit=5)

            self.assertEqual(deleted, 1)
            self.assertEqual(search_results, [])
            self.assertEqual(recent_results, [])

    def test_list_chunk_references_returns_chunks_with_reference_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            result = store.add_turn("u1", "P3 evidence 管理入口", created_at=1778131200.0)
            chunk_id = result.chunks[0].id

            references = store.list_chunk_references(
                "u1",
                [chunk_id],
                {chunk_id: EvidenceReferenceCounts(active_refs=1, retained_refs=2)},
            )

            self.assertEqual(len(references), 1)
            self.assertEqual(references[0].chunk.id, chunk_id)
            self.assertEqual(references[0].active_refs, 1)
            self.assertEqual(references[0].retained_refs, 2)

    def test_search_chunks_like_ranks_specific_matches_before_generic_problem_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            store.add_turn("u1", "我提到国内网络下语音识别不稳定", created_at=1778131200.0)
            store.add_turn("u1", "今天提到另一个完全不同的问题", created_at=1778131300.0)

            results = store.search_chunks("u1", "国内网络的问题", limit=5)

            self.assertGreaterEqual(len(results), 2)
            self.assertIn("国内网络", results[0].text)
            self.assertIn("语音识别不稳定", results[0].text)

    def test_search_chunks_with_ranking_exposes_factors_and_keeps_legacy_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            first = store.add_turn("u1", "ranking recall 排序调试", created_at=1778131200.0)
            store.add_turn("u1", "ranking 普通问题", created_at=1778131300.0)

            result = store.search_chunks_with_ranking("u1", "ranking recall 排序", limit=2)
            legacy_chunks = store.search_chunks("u1", "ranking recall 排序", limit=2)

            self.assertEqual(result.chunks[0].id, first.chunks[0].id)
            self.assertEqual(result.ranking[0]["id"], first.chunks[0].id)
            self.assertGreater(result.ranking[0]["text_score"], result.ranking[1]["text_score"])
            self.assertEqual([chunk.id for chunk in legacy_chunks], [chunk.id for chunk in result.chunks])

    def test_search_chunks_with_ranking_prefers_newer_chunk_on_equal_text_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            store.add_turn("u1", "ranking recall 同等命中", created_at=1778131200.0)
            newer = store.add_turn("u1", "ranking recall 同等命中", created_at=1778131300.0)

            result = store.search_chunks_with_ranking("u1", "ranking recall", limit=2)

            self.assertEqual(result.chunks[0].id, newer.chunks[0].id)
            self.assertGreaterEqual(result.ranking[0]["recency_score"], result.ranking[1]["recency_score"])

    def test_add_turn_is_safe_under_parallel_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, isolated_app_home(tmpdir):
            store = self.make_store(Path(tmpdir))
            errors = []

            def write_turn(index: int) -> str:
                result = store.add_turn(
                    "u1",
                    f"并发写入 timeline 第 {index} 条 国内网络 语音识别",
                    created_at=1778131200.0 + index,
                )
                return result.turn.id

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(write_turn, index) for index in range(200)]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:  # pragma: no cover - failure detail for assertion message
                        errors.append(f"{type(exc).__name__}: {exc}")

            results = store.search_chunks("u1", "国内网络", limit=20)
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 20)


if __name__ == "__main__":
    unittest.main()
