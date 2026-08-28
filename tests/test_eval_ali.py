from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

from ai_glasses_memory_assistant.evals.eval_ali import (
    EvalAliCase,
    ReferenceInterval,
    build_manifest,
    diagnose_case,
    normalize_text,
    score_cer,
    score_der,
    score_events,
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


def test_parse_textgrid_and_manifest_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "Eval_Ali"
    audio = root / "Eval_Ali_far" / "audio_dir"
    grids = root / "Eval_Ali_far" / "textgrid_dir"
    audio.mkdir(parents=True)
    grids.mkdir()
    for number in range(8):
        session = f"R{number:04d}_M{number:04d}"
        _wav(audio / f"{session}_MS{number:03d}.wav")
        grids.joinpath(f"{session}.TextGrid").write_text(
            _textgrid([("N_SPK1", [(0.0, 0.5, "你好，世界")]), ("N_SPK2", [(0.2, 0.8, "讨论")])]), encoding="utf-8"
        )
    manifest = build_manifest(root, "smoke")
    assert len(manifest["cases"]) == 8
    assert manifest["channel"] == 0
    assert manifest["cases"][0]["reference_intervals"][0]["text"] == "你好，世界"


def test_metrics_normalize_chinese_and_find_overlap() -> None:
    reference = [ReferenceInterval("N_SPK1", 0.0, 1.0, "你好，世界"), ReferenceInterval("N_SPK2", 0.5, 1.5, "讨论")]
    assert normalize_text("你 好，世界！") == "你好世界"
    assert score_cer("你好，世界", "你好世界")["normalized"]["cer"] == 0.0
    assert score_vad(reference, [(0.0, 1.5)], start_s=0.0, end_s=1.5)["recall"] == 1.0
    assert score_der(reference, [{"start_s": 0.3, "end_s": 0.7, "voice_group": "spk_01"}], start_s=0.0, end_s=1.5)["overlap_frames"] > 0


def test_window_diagnosis_and_capture_offset() -> None:
    intervals = [ReferenceInterval("N_SPK1", 30.0, 45.0, "有语音")]
    assert select_case_window(intervals, 60.0, 15.0) == (30.0, 45.0)
    assert diagnose_case({"event_count": 1, "vad": {"recall": 1.0}, "cer": {"normalized": {"cer": 0.0}}}, preflight_ok=True, queue_failed=0, memory_saved=1)["primary_stage"] == "memory_safety"
    case = EvalAliCase("x", "x", "audio", "grid", "a", "b", 20.0, 10.0, 12.0, (ReferenceInterval("N_SPK1", 10.0, 12.0, "你好"),))
    score = score_events(case, [{"event": {"event_id": "e", "start_ms": 5000, "end_ms": 7000, "type": "transcript_final", "text": "你好", "speaker": {"voice_group": "spk_01"}}}], playback_offset_ms=5000)
    assert score["cer"]["normalized"]["cer"] == 0.0


class _FakeMemoryStore:
    def list_memories(self, *, user_id: str, limit: int):
        return []


class _FakeTimelineStore:
    def list_discussion_slices(self, user_id: str, *, capture_id: str):
        return [{"id": "slice", "status": "completed"}]


class _FakeService:
    def __init__(self) -> None:
        self.memory_store = _FakeMemoryStore()
        self.timeline_store = _FakeTimelineStore()

    def start_audio_session(self, *, user_id: str, mode: str):
        return {"user_id": user_id, "audio_session_id": "session", "session_token": "token", "capture_id": "capture", "format": {"recommended_push_samples": 4096}}

    def push_audio_session(self, **kwargs):
        return {"events": [{"event_id": "event", "start_ms": 0, "end_ms": 500, "type": "transcript_final", "text": "你好", "speaker": {"voice_group": "spk_01"}}], "dispatches": []}

    def stop_audio_session(self, **kwargs):
        return {"status": "stopped"}


def test_offline_runner_streams_pcm_and_keeps_reports_audio_free(tmp_path: Path) -> None:
    module = _offline_module()
    wav = tmp_path / "source.wav"
    _wav(wav)
    case = EvalAliCase("case", "session", str(wav), "grid", "audio", "grid", 1.0, 0.0, 1.0, (ReferenceInterval("N_SPK1", 0.0, 0.5, "你好"),))
    clock = module.VirtualClock(100.0)
    result = module.run_case(_FakeService(), case=case, user_id="isolated", clock=clock, realtime=False, archive_timeout=1.0)
    assert result["status"] == "complete"
    assert result["stream"]["push_count"] >= 4
    assert result["memory_saved"] == 0
    assert clock() > 101.9
    assert "pcm" not in json.dumps(result, ensure_ascii=False).lower()


def test_offline_resume_requires_complete_report_and_baseline_rejects_mismatch(tmp_path: Path) -> None:
    module = _offline_module()
    run = tmp_path / "run"
    case = run / "case"
    case.mkdir(parents=True)
    case.joinpath("score.json").write_text("{}", encoding="utf-8")
    case.joinpath("diagnosis.json").write_text("{}", encoding="utf-8")
    case.joinpath("case.json").write_text("{}", encoding="utf-8")
    run.joinpath("cases.jsonl").write_text(json.dumps({"status": "complete", "case_id": "case", "archive": {"status": "completed"}, "memory_saved": 0}) + "\n", encoding="utf-8")
    assert module.existing_complete_cases(run / "cases.jsonl") == {"case"}
    case.joinpath("case.json").unlink()
    assert module.existing_complete_cases(run / "cases.jsonl") == set()
    manifest = {"schema": "eval_ali.v1", "preset": "smoke", "channel": 0, "delivery_mode": "fast_virtual_time", "frame_samples": 4096, "safety_policy": "safe", "runtime": {"model": "x"}, "cases": []}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"compatibility": module.offline_compatibility(manifest), "scores": {"normalized_cer": 0.1, "vad": {"f1": 0.8}, "throughput": {"wall_seconds_per_audio_second": 0.2}}}), encoding="utf-8")
    comparison = module.compare_baseline({"status": "complete", "normalized_cer": 0.1, "vad": {"f1": 0.8}, "throughput": {"wall_seconds_per_audio_second": 0.2}, "archive_failed": 0, "memory_saved": 0}, {**manifest, "delivery_mode": "realtime"}, baseline)
    assert comparison["comparable"] is False


def test_offline_baseline_locks_three_comparable_runs(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "lock_eval_ali_offline_baseline.py"
    spec = importlib.util.spec_from_file_location("lock_eval_ali_offline_baseline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = []
    for index in range(3):
        run = tmp_path / f"run-{index}"
        run.mkdir()
        run.joinpath("run-manifest.json").write_text(json.dumps({"schema": "eval_ali.v1", "preset": "smoke", "channel": 0, "delivery_mode": "fast_virtual_time", "frame_samples": 4096, "safety_policy": "safe", "runtime": {}, "cases": []}), encoding="utf-8")
        run.joinpath("scores.json").write_text(json.dumps({"status": "complete", "completed_cases": 8, "normalized_cer": 0.1 + index * 0.01, "vad": {"f1": 0.8}, "throughput": {"wall_seconds_per_audio_second": 0.2}}), encoding="utf-8")
        runs.append(run)
    output = tmp_path / "baseline.json"
    assert module.main([str(run) for run in runs] + ["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["scores"]["normalized_cer"] == 0.11
