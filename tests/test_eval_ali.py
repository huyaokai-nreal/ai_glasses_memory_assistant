from __future__ import annotations

import importlib.util
import json
import os
import sys
import wave
from pathlib import Path

from ai_glasses_memory_assistant.evals.eval_ali import (
    AMBIENT_MEMORY_GOLD_SCHEMA,
    AmbientMemoryEvidence,
    AmbientMemoryGoldCase,
    AmbientMemoryQuestion,
    EvalAliCase,
    ReferenceInterval,
    build_manifest,
    load_ambient_memory_gold,
    normalize_text,
    score_cer,
    score_der,
    score_evidence_asr,
    score_events,
    score_required_forbidden,
    score_vad,
    select_case_window,
)


def _offline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_eval_ali_offline.py"
    spec = importlib.util.spec_from_file_location("run_eval_ali_offline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _textgrid(tiers: list[tuple[str, list[tuple[float, float, str]]]]) -> str:
    parts = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "item []:"]
    for index, (name, intervals) in enumerate(tiers, start=1):
        parts += [f"    item [{index}]:", '        class = "IntervalTier"', f'        name = "{name}"']
        for interval_index, (start, end, text) in enumerate(intervals, start=1):
            parts += [f"        intervals [{interval_index}]:", f"            xmin = {start}", f"            xmax = {end}", f'            text = "{text}"']
    return "\n".join(parts) + "\n"


def _wav(path: Path, seconds: int = 1) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(8)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 8 * 16_000 * seconds)


def _case(case_id: str = "R0000_M0000-smoke") -> EvalAliCase:
    return EvalAliCase(case_id, "R0000_M0000", "audio", "grid", "a", "b", 10.0, 0.0, 2.0, (ReferenceInterval("N_SPK1", 0.0, 2.0, "环境讨论平台推广"),))


def _gold_case() -> AmbientMemoryGoldCase:
    evidence = AmbientMemoryEvidence("promotion", 0.0, 2.0, "环境讨论平台推广", ("平台", "推广"), ("声纹匹配",))
    questions = (AmbientMemoryQuestion("q1", "推广方式？", ("平台",), (), ("promotion",)), AmbientMemoryQuestion("q2", "推广内容？", ("推广",), (), ("promotion",)))
    return AmbientMemoryGoldCase("R0000_M0000-smoke", "R0000_M0000", 0.0, 2.0, (evidence,), ("平台",), (), ("推广",), (), questions)


def test_parse_textgrid_and_manifest_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "Eval_Ali"
    audio = root / "Eval_Ali_far" / "audio_dir"
    grids = root / "Eval_Ali_far" / "textgrid_dir"
    audio.mkdir(parents=True)
    grids.mkdir()
    for number in range(8):
        session = f"R{number:04d}_M{number:04d}"
        _wav(audio / f"{session}_MS{number:03d}.wav")
        grids.joinpath(f"{session}.TextGrid").write_text(_textgrid([("N_SPK1", [(0.0, 0.5, "你好，世界")]), ("N_SPK2", [(0.2, 0.8, "讨论")])]), encoding="utf-8")
    manifest = build_manifest(root, "smoke")
    assert len(manifest["cases"]) == 8
    assert manifest["channel"] == 0


def test_metrics_and_gold_evidence_score_are_deterministic() -> None:
    reference = [ReferenceInterval("N_SPK1", 0.0, 1.0, "你好，世界"), ReferenceInterval("N_SPK2", 0.5, 1.5, "讨论")]
    assert normalize_text("你 好，世界！") == "你好世界"
    assert score_cer("你好，世界", "你好世界")["normalized"]["cer"] == 0.0
    assert score_vad(reference, [(0.0, 1.5)], start_s=0.0, end_s=1.5)["recall"] == 1.0
    assert score_der(reference, [{"start_s": 0.3, "end_s": 0.7, "voice_group": "spk_01"}], start_s=0.0, end_s=1.5)["overlap_frames"] > 0
    score, mapping = score_evidence_asr(_gold_case(), [{"event_id": "e", "start_s": 0.0, "end_s": 1.0, "type": "transcript_final", "text": "环境讨论平台推广"}])
    assert score["passed"] is True
    assert mapping == {"e": {"promotion"}}
    assert score_required_forbidden("平台推广", required_terms=["平台"], forbidden_terms=["声纹匹配"])["passed"] is True
    assert score_required_forbidden("平台声纹匹配", required_terms=["平台"], forbidden_terms=["声纹匹配"])["passed"] is False


def test_gold_loader_requires_all_cases_and_validates_textgrid_span(tmp_path: Path) -> None:
    cases = [_case(f"R{index:04d}_M{index:04d}-smoke") for index in range(8)]
    rows = []
    for case in cases:
        rows.append({"case_id": case.case_id, "window": {"start_s": 0.0, "end_s": 2.0}, "evidence": [{"id": "e", "start_s": 0.0, "end_s": 2.0, "reference_text": "环境讨论平台推广", "required_terms": ["平台"], "forbidden_terms": []}], "summary": {"topic": {"required_terms": ["平台"]}, "overview": {"required_terms": ["推广"]}}, "questions": [{"id": "q1", "question": "问题一", "required_terms": ["平台"], "forbidden_terms": [], "evidence_ids": ["e"]}, {"id": "q2", "question": "问题二", "required_terms": ["推广"], "forbidden_terms": [], "evidence_ids": ["e"]}]})
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"schema": AMBIENT_MEMORY_GOLD_SCHEMA, "cases": rows}, ensure_ascii=False), encoding="utf-8")
    assert len(load_ambient_memory_gold(path, smoke_cases=cases)) == 8
    rows[0]["evidence"][0]["reference_text"] = "不存在"
    path.write_text(json.dumps({"schema": AMBIENT_MEMORY_GOLD_SCHEMA, "cases": rows}, ensure_ascii=False), encoding="utf-8")
    try:
        load_ambient_memory_gold(path, smoke_cases=cases)
    except ValueError as exc:
        assert "does not match TextGrid" in str(exc)
    else:
        raise AssertionError("invalid TextGrid evidence was accepted")


def test_window_and_event_offset_stay_available_for_health_scoring() -> None:
    intervals = [ReferenceInterval("N_SPK1", 30.0, 45.0, "有语音")]
    assert select_case_window(intervals, 60.0, 15.0) == (30.0, 45.0)
    case = EvalAliCase("x", "x", "audio", "grid", "a", "b", 20.0, 10.0, 12.0, (ReferenceInterval("N_SPK1", 10.0, 12.0, "你好"),))
    score = score_events(case, [{"event": {"event_id": "e", "start_ms": 5000, "end_ms": 7000, "type": "transcript_final", "text": "你好", "speaker": {"voice_group": "spk_01"}}}], playback_offset_ms=5000)
    assert score["cer"]["normalized"]["cer"] == 0.0


def test_await_archives_distinguishes_ready_late_and_incomplete() -> None:
    module = _offline_module()

    class Store:
        def __init__(self, values): self.values, self.calls = values, 0
        def list_discussion_slices(self, user_id, *, capture_id):
            value = self.values[min(self.calls, len(self.values) - 1)]
            self.calls += 1
            return [{"id": "slice", "status": value}]
    class Service:
        def __init__(self, values): self.timeline_store = Store(values)

    assert module.await_archives(Service(["ready"]), user_id="u", capture_id="c", timeout_seconds=1)["status"] == "ready"
    assert module.await_archives(Service(["failed"]), user_id="u", capture_id="c", timeout_seconds=1)["status"] == "failed"
    assert module.await_archives(Service(["pending"]), user_id="u", capture_id="c", timeout_seconds=0.01)["status"] == "incomplete"
    assert module.await_archives(Service(["pending", "ready"]), user_id="u", capture_id="c", timeout_seconds=1)["status"] == "ready"
    moments = iter([0.0, 2.0])
    assert module.await_archives(Service(["pending", "ready"]), user_id="u", capture_id="c", timeout_seconds=1, monotonic=lambda: next(moments), sleep=lambda _: None)["status"] == "ready_late"


def test_daily_summary_provenance_and_chat_question_scoring_use_real_output_shape() -> None:
    module = _offline_module()
    gold = _gold_case()
    expected = {"promotion": {"chunk_1"}}
    archive = module.score_archive(gold, day_payload={"overview": "推广安排", "topics": [{"title": "平台推广", "summary": "平台推广", "evidence_ids": ["chunk_1"]}]}, expected_chunks=expected)
    assert archive["passed"] is True

    class Service:
        def chat(self, question, *, user_id, memory_writes_allowed):
            return {"reply": "平台推广", "source_summary": {"primary_source": "discussion_archive"}, "discussion_recall": {"status": "ready", "evidence_ids": ["chunk_1"]}}

    questions = module.score_questions(Service(), gold=gold, user_id="u", expected_chunks=expected)
    assert questions["passed"] is True


def test_preflight_error_is_saved_under_local_run_directory(tmp_path: Path, monkeypatch) -> None:
    module = _offline_module()
    for name in module.LOCAL_LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["run_eval_ali_offline.py", "--out", str(tmp_path), "--run-id", "missing-llm"])
    try:
        module.main()
    except SystemExit:
        pass
    error = json.loads((tmp_path / "missing-llm" / "run-error.json").read_text(encoding="utf-8"))
    assert error["stage"] == "preflight"
    assert "explicit local LLM configuration" in error["error"]


def test_runner_loads_repo_audio_paths_before_isolating_app_home(tmp_path: Path, monkeypatch) -> None:
    module = _offline_module()
    for name in ("AI_GLASSES_ASR_MODEL_DIR", "AI_GLASSES_STREAMING_ASR_MODEL_DIR"):
        monkeypatch.delenv(name, raising=False)
    tmp_path.joinpath(".env").write_text("AI_GLASSES_ASR_MODEL_DIR=/models/sensevoice\nAI_GLASSES_STREAMING_ASR_MODEL_DIR=/models/streaming\n", encoding="utf-8")
    module.load_repo_audio_model_config(tmp_path)
    assert os.environ["AI_GLASSES_ASR_MODEL_DIR"] == "/models/sensevoice"
    assert os.environ["AI_GLASSES_STREAMING_ASR_MODEL_DIR"] == "/models/streaming"


def test_resume_and_baseline_reject_incompatible_v2_runs(tmp_path: Path) -> None:
    module = _offline_module()
    run = tmp_path / "run"
    (run / "case").mkdir(parents=True)
    (run / "case" / "case.json").write_text("{}", encoding="utf-8")
    (run / "cases.jsonl").write_text(json.dumps({"status": "completed", "case_id": "case"}) + "\n", encoding="utf-8")
    assert module.existing_completed(run / "cases.jsonl") == {"case"}
    manifest = {"schema": "ambient_audio_memory_v2_manifest.v1", "preset": "smoke", "gold_sha256": "a", "delivery_mode": "fast_virtual_time", "runtime": {}, "cases": []}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"compatibility": module.compatibility(manifest), "scores": {}}), encoding="utf-8")
    scores = {"status": "complete", "health": {"memory_saved": 0, "normalized_cer": 0.1}, "closure": {"end_to_end_passed": True}}
    assert module.compare_baseline(scores, {**manifest, "gold_sha256": "b"}, baseline)["comparable"] is False


def test_offline_baseline_locks_three_full_v2_runs(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "lock_eval_ali_offline_baseline.py"
    spec = importlib.util.spec_from_file_location("lock_eval_ali_offline_baseline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = []
    for index in range(3):
        run = tmp_path / f"run-{index}"
        run.mkdir()
        run.joinpath("run-manifest.json").write_text(json.dumps({"schema": "ambient_audio_memory_v2_manifest.v1", "preset": "full", "gold_sha256": "g", "delivery_mode": "fast_virtual_time", "runtime": {}, "cases": []}), encoding="utf-8")
        run.joinpath("scores.json").write_text(json.dumps({"status": "complete", "completed_cases": 8, "health": {"memory_saved": 0, "normalized_cer": 0.1 + index * 0.01, "vad": {"f1": 0.8}}, "closure": {"passed_cases": 8, "passed_questions": 16, "end_to_end_passed": True}}), encoding="utf-8")
        runs.append(run)
    output = tmp_path / "baseline.json"
    assert module.main([str(run) for run in runs] + ["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["scores"]["health"]["normalized_cer"] == 0.11
