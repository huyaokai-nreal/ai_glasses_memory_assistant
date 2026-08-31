#!/usr/bin/env python3
"""Run the V2 ambient-audio memory closure benchmark through the shared pipeline.

Eval_Ali WAV is pushed as PCM16 into the same AudioSession API used after
microphone capture. The benchmark measures software-chain behaviour only; it
does not measure Android hardware or real-world acoustics.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_glasses_memory_assistant.audio_engine.runtime import AudioSessionManager
from ai_glasses_memory_assistant.env_loader import load_app_dotenv
from ai_glasses_memory_assistant.evals.eval_ali import (
    AmbientMemoryGoldCase,
    EvalAliCase,
    ReferenceInterval,
    build_manifest,
    load_ambient_memory_gold,
    score_evidence_asr,
    score_events,
    score_required_forbidden,
    sha256_json_file,
    write_json,
)

LOCAL_LLM_ENV = ("AI_GLASSES_LLM_PROVIDER", "AI_GLASSES_LLM_MODEL", "AI_GLASSES_LLM_BASE_URL", "AI_GLASSES_LLM_API_KEY")
SAMPLE_RATE = 16_000
BOUNDARY_SILENCE_SECONDS = 1.0
ARCHIVE_TIMEOUT_SECONDS = 180.0
GOLD_PATH = ROOT / "data" / "benchmarks" / "eval_ali" / "ambient_memory_v2_gold.json"
VIRTUAL_DAY_EPOCH = 1_704_067_200.0  # 2024-01-01T00:00:00Z


@dataclass
class VirtualClock:
    now: float

    def __post_init__(self) -> None:
        self._lock = Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += max(0.0, float(seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "Eval_Ali")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "eval_ali")
    parser.add_argument("--run-id", default="", help="Existing run id is required with --resume.")
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--limit-cases", type=int, default=0, help="Debug only; not baseline-comparable.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Pace PCM delivery at 1x.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--archive-timeout-seconds", type=float, default=ARCHIVE_TIMEOUT_SECONDS)
    parser.add_argument("--baseline", type=Path, help="A V2 locked baseline JSON.")
    return parser.parse_args()


def require_local_llm() -> dict[str, str]:
    values = {name: str(os.getenv(name) or "").strip() for name in LOCAL_LLM_ENV}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit("offline archive requires explicit local LLM configuration: " + ", ".join(missing))
    if values["AI_GLASSES_LLM_PROVIDER"].lower() not in {"llama_cpp", "ollama"}:
        raise SystemExit("offline archive only permits a local llama_cpp or ollama provider")
    base_url = values["AI_GLASSES_LLM_BASE_URL"].lower()
    if "deepseek" in base_url or base_url.startswith("https://"):
        raise SystemExit("offline archive refuses cloud/DeepSeek configuration")
    return values


def load_repo_audio_model_config(repo_root: Path = ROOT) -> None:
    load_app_dotenv(paths=[repo_root / ".env"])


def case_from_dict(payload: dict[str, Any]) -> EvalAliCase:
    return EvalAliCase(**{key: value for key, value in payload.items() if key != "reference_intervals"}, reference_intervals=tuple(ReferenceInterval(**item) for item in payload["reference_intervals"]))


def pcm16_base64(samples: np.ndarray) -> str:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return base64.b64encode(np.rint(clipped * 32767.0).astype("<i2", copy=False).tobytes()).decode("ascii")


def runtime_fingerprint(llm: dict[str, str]) -> dict[str, Any]:
    def command(*parts: str) -> str:
        result = subprocess.run(parts, cwd=ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip() or result.stderr.strip()

    return {"git_head": command("git", "rev-parse", "HEAD"), "git_dirty": bool(command("git", "status", "--porcelain")),
            "llm_provider": llm["AI_GLASSES_LLM_PROVIDER"], "llm_model": llm["AI_GLASSES_LLM_MODEL"],
            "llm_base_url": llm["AI_GLASSES_LLM_BASE_URL"], "asr_model_dir": str(os.getenv("AI_GLASSES_ASR_MODEL_DIR") or "")}


def await_archives(
    service: Any,
    *,
    user_id: str,
    capture_id: str,
    timeout_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait for actual slice states and perform one terminal re-read at deadline."""

    active = {"pending", "running"}
    deadline = monotonic() + max(1.0, timeout_seconds)
    while True:
        slices = service.timeline_store.list_discussion_slices(user_id, capture_id=capture_id)
        statuses = [str(item.get("status") or "") for item in slices]
        if not slices:
            return {"status": "incomplete", "slices": [], "reason": "no_discussion_slices"}
        if all(status not in active for status in statuses):
            return {"status": "failed" if any(status == "failed" for status in statuses) else "ready", "slices": slices, "reason": "terminal_slice_state"}
        if monotonic() >= deadline:
            final_slices = service.timeline_store.list_discussion_slices(user_id, capture_id=capture_id)
            final_statuses = [str(item.get("status") or "") for item in final_slices]
            if final_slices and all(status not in active for status in final_statuses):
                return {"status": "failed" if any(status == "failed" for status in final_statuses) else "ready_late", "slices": final_slices, "reason": "terminal_after_deadline"}
            return {"status": "incomplete", "slices": final_slices, "reason": "deadline_with_active_slice"}
        sleep(0.05)


def push_case_audio(service: Any, *, session: dict[str, Any], case: EvalAliCase, clock: VirtualClock, realtime: bool) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_chunks: dict[str, str] = {}
    frame_samples = int(session["format"]["recommended_push_samples"])
    sequence, sent_seconds, wall_start, max_lag, max_push = 1, 0.0, time.monotonic(), 0.0, 0.0

    def push(samples: np.ndarray) -> None:
        nonlocal sequence, sent_seconds, max_lag, max_push
        if realtime:
            lag = time.monotonic() - (wall_start + sent_seconds)
            max_lag = max(max_lag, lag)
            if lag < 0:
                time.sleep(-lag)
        duration = len(samples) / SAMPLE_RATE
        clock.advance(duration)
        started = time.perf_counter()
        response = service.push_audio_session(user_id=session["user_id"], audio_session_id=session["audio_session_id"], session_token=session["session_token"], sequence=sequence, pcm16_base64=pcm16_base64(samples))
        max_push = max(max_push, time.perf_counter() - started)
        events.extend({"event": event} for event in response.get("events") or [])
        for dispatch in response.get("dispatches") or []:
            result = dispatch.get("result") if isinstance(dispatch, dict) else None
            event_id = str(dispatch.get("event_id") or "") if isinstance(dispatch, dict) else ""
            chunk_id = str((result or {}).get("chunk_id") or "") if isinstance(result, dict) else ""
            if event_id and chunk_id:
                event_chunks[event_id] = chunk_id
        sequence += 1
        sent_seconds += duration

    with sf.SoundFile(case.audio_path) as source:
        if source.samplerate != SAMPLE_RATE or source.channels != 8:
            raise ValueError(f"unexpected Eval_Ali_far source format: {case.audio_path}")
        source.seek(round(case.start_s * SAMPLE_RATE))
        remaining = round((case.end_s - case.start_s) * SAMPLE_RATE)
        while remaining:
            frames = min(frame_samples, remaining)
            audio = source.read(frames, dtype="float32", always_2d=True)
            if len(audio) != frames:
                raise ValueError(f"truncated Eval_Ali_far source: {case.audio_path}")
            push(audio[:, 0])
            remaining -= frames
    for _ in range(round(BOUNDARY_SILENCE_SECONDS * SAMPLE_RATE / frame_samples)):
        push(np.zeros(frame_samples, dtype=np.float32))
    return events, event_chunks, {"frame_samples": frame_samples, "audio_seconds": sent_seconds, "wall_seconds": time.monotonic() - wall_start, "max_delivery_lag_seconds": max_lag, "max_push_seconds": max_push}


def _topic_text(topic: dict[str, Any]) -> str:
    values = [str(topic.get(key) or "") for key in ("title", "summary")]
    for key in ("key_points", "decisions", "tasks", "open_questions"):
        values.extend(str(item) for item in topic.get(key) or [])
    return "\n".join(values)


def _topic_evidence_ids(topics: Iterable[dict[str, Any]]) -> set[str]:
    return {str(item).removeprefix("timeline:") for topic in topics for key in ("evidence_ids", "available_evidence_ids") for item in topic.get(key) or [] if str(item).strip()}


def score_archive(gold: AmbientMemoryGoldCase, *, day_payload: dict[str, Any] | None, expected_chunks: dict[str, set[str]]) -> dict[str, Any]:
    if not day_payload:
        return {"passed": False, "reason": "discussion_day_missing", "provenance": {"passed": False}}
    topics = list(day_payload.get("topics") or [])
    checks = [score_required_forbidden(_topic_text(topic), required_terms=gold.topic_required_terms, forbidden_terms=gold.topic_forbidden_terms) for topic in topics]
    topic = next((item for item in checks if item["passed"]), checks[0] if checks else score_required_forbidden("", required_terms=gold.topic_required_terms, forbidden_terms=gold.topic_forbidden_terms))
    overview = score_required_forbidden(str(day_payload.get("overview") or ""), required_terms=gold.overview_required_terms, forbidden_terms=gold.overview_forbidden_terms)
    archived_ids = _topic_evidence_ids(topics)
    linked = {evidence_id: sorted(chunk_ids) for evidence_id, chunk_ids in expected_chunks.items()}
    missing_source = sorted(evidence_id for evidence_id, chunk_ids in linked.items() if not chunk_ids)
    missing_archive = sorted(evidence_id for evidence_id, chunk_ids in linked.items() if chunk_ids and not (set(chunk_ids) & archived_ids))
    provenance = {"passed": not missing_source and not missing_archive, "expected_chunks": linked, "archived_evidence_ids": sorted(archived_ids), "missing_source_evidence": missing_source, "missing_archive_evidence": missing_archive}
    return {"passed": bool(topic["passed"] and overview["passed"] and provenance["passed"]), "topic": topic, "overview": overview, "provenance": provenance}


def judge_answer(service: Any, *, user_id: str, question: str, required_terms: list[str], answer: str, recalled_ids: list[str]) -> dict[str, Any]:
    """Diagnostic-only local-Qwen verdict; it never changes hard scores."""

    try:
        session = service._new_session(user_id=user_id, session_id=f"eval-judge:{user_id}")
        result = session.agent.run_conversation(json.dumps({"question": question, "required_terms": required_terms, "answer": answer, "retrieved_evidence_ids": recalled_ids}, ensure_ascii=False), system_message="Judge only whether the answer addresses the question using the listed retrieved evidence. Return JSON with verdict (pass/fail) and concise reason. Do not infer facts not shown.", max_tokens=160, disable_thinking=True)
        return {"status": "completed", "raw": str((result or {}).get("final_response") or "")}
    except Exception as exc:
        return {"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc)}


def score_questions(service: Any, *, gold: AmbientMemoryGoldCase, user_id: str, expected_chunks: dict[str, set[str]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for question in gold.questions:
        try:
            response = service.chat(question.question, user_id=user_id, memory_writes_allowed=False)
            recall = response.get("discussion_recall") or {}
            source = (response.get("source_summary") or {}).get("primary_source")
            recalled_ids = [str(item).removeprefix("timeline:") for item in recall.get("evidence_ids") or []]
            expected = set().union(*(expected_chunks.get(evidence_id, set()) for evidence_id in question.evidence_ids))
            terms = score_required_forbidden(str(response.get("reply") or ""), required_terms=question.required_terms, forbidden_terms=question.forbidden_terms)
            source_ok = source == "discussion_archive" and str(recall.get("status") or "") == "ready"
            provenance_ok = bool(expected and expected & set(recalled_ids))
            passed = bool(terms["passed"] and source_ok and provenance_ok)
            rows.append({"question_id": question.id, "question": question.question, "reply": str(response.get("reply") or ""), "term_check": terms, "source": source, "discussion_status": recall.get("status"), "expected_evidence_ids": sorted(expected), "recalled_evidence_ids": recalled_ids, "source_ok": source_ok, "provenance_ok": provenance_ok, "passed": passed, "judge": judge_answer(service, user_id=user_id, question=question.question, required_terms=list(question.required_terms), answer=str(response.get("reply") or ""), recalled_ids=recalled_ids)})
        except Exception as exc:
            rows.append({"question_id": question.id, "question": question.question, "passed": False, "stage": "answer", "error_type": type(exc).__name__, "error": str(exc)})
    return {"passed": len(rows) == 2 and all(row["passed"] for row in rows), "questions": rows}


def run_case(service: Any, *, case: EvalAliCase, gold: AmbientMemoryGoldCase, user_id: str, clock: VirtualClock, realtime: bool, archive_timeout: float) -> dict[str, Any]:
    session = service.start_audio_session(user_id=user_id, mode="ambient")
    events, event_chunks, stream = push_case_audio(service, session=session, case=case, clock=clock, realtime=realtime)
    stopped = service.stop_audio_session(user_id=user_id, audio_session_id=session["audio_session_id"], session_token=session["session_token"])
    capture_id = str(session["capture_id"])
    archive = await_archives(service, user_id=user_id, capture_id=capture_id, timeout_seconds=archive_timeout)
    memory_saved = len(service.memory_store.list_memories(user_id=user_id, limit=10_000))
    health = score_events(case, events, playback_offset_ms=0.0)
    asr, event_evidence = score_evidence_asr(gold, health["events"])
    expected_chunks: dict[str, set[str]] = {evidence.id: set() for evidence in gold.evidence}
    for event_id, evidence_ids in event_evidence.items():
        if chunk_id := event_chunks.get(event_id):
            for evidence_id in evidence_ids:
                expected_chunks[evidence_id].add(chunk_id)
    day_key = datetime.datetime.fromtimestamp(clock(), tz=datetime.timezone.utc).date().isoformat()
    day_result = service.discussion_day(user_id=user_id, day=day_key) if archive["status"] == "ready" else {"day": None}
    archive_score = score_archive(gold, day_payload=day_result.get("day"), expected_chunks=expected_chunks) if archive["status"] == "ready" else {"passed": False, "reason": f"archive_{archive['status']}"}
    qa = score_questions(service, gold=gold, user_id=user_id, expected_chunks=expected_chunks) if archive_score["passed"] and memory_saved == 0 else {"passed": False, "questions": [], "reason": "archive_or_privacy_gate"}
    # The questions explicitly disable writes, but score the post-QA store state as
    # well so a future routing regression cannot hide behind a pre-QA snapshot.
    memory_saved = len(service.memory_store.list_memories(user_id=user_id, limit=10_000))
    closure = {"asr": asr, "archive": archive_score, "qa": qa, "privacy": {"passed": memory_saved == 0, "memory_saved": memory_saved}}
    closure["passed"] = bool(asr["passed"] and archive_score["passed"] and qa["passed"] and memory_saved == 0 and archive["status"] == "ready")
    return {"status": "completed" if archive["status"] == "ready" and memory_saved == 0 else "incomplete", "case_id": case.case_id, "session_id": case.session_id, "user_id": user_id, "capture_id": capture_id, "day_key": day_key, "stream": stream, "stop": stopped, "archive": archive, "memory_saved": memory_saved, "health": health, "closure": closure}


def existing_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "completed" and (path.parent / str(row.get("case_id") or "") / "case.json").is_file():
            completed.add(str(row["case_id"]))
    return completed


def summarize(run_dir: Path, cases: list[EvalAliCase]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    path = run_dir / "cases.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("case_id") or "") in {case.case_id for case in cases}:
            rows[str(row["case_id"])] = row
    complete = len(rows) == len(cases) and all(row.get("status") == "completed" for row in rows.values())
    health_rows = [row.get("health") or {} for row in rows.values()]
    ref_chars = sum(int(((row.get("cer") or {}).get("normalized") or {}).get("reference_chars") or 0) for row in health_rows)
    errors = sum(int(((row.get("cer") or {}).get("normalized") or {}).get("errors") or 0) for row in health_rows)
    vad_tp = sum(int((row.get("vad") or {}).get("true_positive_frames") or 0) for row in health_rows)
    vad_fp = sum(int((row.get("vad") or {}).get("false_positive_frames") or 0) for row in health_rows)
    vad_fn = sum(int((row.get("vad") or {}).get("false_negative_frames") or 0) for row in health_rows)
    precision = vad_tp / (vad_tp + vad_fp) if vad_tp + vad_fp else 0.0
    recall = vad_tp / (vad_tp + vad_fn) if vad_tp + vad_fn else 0.0
    qa_rows = [question for row in rows.values() for question in ((row.get("closure") or {}).get("qa") or {}).get("questions") or []]
    closure_rows = [row.get("closure") or {} for row in rows.values()]
    diagnostics: dict[str, int] = {}
    for row in rows.values():
        closure = row.get("closure") or {}
        stage = "none" if closure.get("passed") else ("privacy" if row.get("memory_saved") else "archive" if (row.get("archive") or {}).get("status") != "ready" else "asr" if not (closure.get("asr") or {}).get("passed") else "summary" if not (closure.get("archive") or {}).get("passed") else "answer")
        diagnostics[stage] = diagnostics.get(stage, 0) + 1
    return {"schema": "ambient_audio_memory_v2_scores.v1", "status": "complete" if complete else "incomplete", "expected_cases": len(cases), "completed_cases": sum(row.get("status") == "completed" for row in rows.values()), "health": {"normalized_cer": errors / ref_chars if complete and ref_chars else None, "vad": {"precision": precision if complete else None, "recall": recall if complete else None, "f1": 2 * precision * recall / (precision + recall) if complete and precision + recall else None}, "throughput": {"audio_seconds": sum(float((row.get("stream") or {}).get("audio_seconds") or 0) for row in rows.values()), "wall_seconds": sum(float((row.get("stream") or {}).get("wall_seconds") or 0) for row in rows.values())}, "archive_statuses": {status: sum((row.get("archive") or {}).get("status") == status for row in rows.values()) for status in ("ready", "ready_late", "failed", "incomplete")}, "memory_saved": max((int(row.get("memory_saved") or 0) for row in rows.values()), default=0)}, "closure": {"passed_cases": sum(bool(row.get("passed")) for row in closure_rows), "total_cases": len(closure_rows), "passed_questions": sum(bool(row.get("passed")) for row in qa_rows), "total_questions": len(qa_rows), "end_to_end_passed": bool(complete and closure_rows and all(row.get("passed") for row in closure_rows))}, "diagnosis": {"by_primary_stage": diagnostics, "missing_cases": sorted({case.case_id for case in cases} - set(rows))}}


def compatibility(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: manifest.get(key) for key in ("schema", "preset", "gold_sha256", "delivery_mode", "runtime", "cases")}


def compare_baseline(scores: dict[str, Any], manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("compatibility") != compatibility(manifest):
        return {"comparable": False, "pass": None, "reason": "manifest_or_gold_differs"}
    base = baseline.get("scores") or {}
    checks = [{"metric": "run_complete", "pass": scores["status"] == "complete"}, {"metric": "privacy", "pass": scores["health"]["memory_saved"] == 0}, {"metric": "closure", "pass": scores["closure"]["end_to_end_passed"]}]
    if isinstance(base.get("health", {}).get("normalized_cer"), (int, float)) and isinstance(scores["health"]["normalized_cer"], (int, float)):
        limit = base["health"]["normalized_cer"] + max(0.02, base["health"]["normalized_cer"] * 0.10)
        checks.append({"metric": "normalized_cer", "pass": scores["health"]["normalized_cer"] <= limit, "limit": limit})
    if isinstance(base.get("health", {}).get("vad_f1"), (int, float)) and isinstance(scores["health"]["vad"]["f1"], (int, float)):
        limit = base["health"]["vad_f1"] - 0.05
        checks.append({"metric": "vad_f1", "pass": scores["health"]["vad"]["f1"] >= limit, "limit": limit})
    return {"comparable": True, "checks": checks, "pass": all(item["pass"] for item in checks)}


def write_summary(run_dir: Path, scores: dict[str, Any], comparison: dict[str, Any] | None) -> None:
    health, closure = scores["health"], scores["closure"]
    lines = ["# 通用背景音频记忆闭环评测 V2", "", "- 本结果只验证共享软件链路，不代表 Android 麦克风或真实环境声学效果。", f"- 状态：{scores['status']}", f"- 健康 case：{scores['completed_cases']}/{scores['expected_cases']}", f"- 全量规范化 CER：{health['normalized_cer']}", f"- 全量 VAD F1：{health['vad']['f1']}", f"- 归档状态：{health['archive_statuses']}", f"- 环境音错误写长期个人记忆：{health['memory_saved']}", f"- 闭环 case：{closure['passed_cases']}/{closure['total_cases']}", f"- 闭环问答：{closure['passed_questions']}/{closure['total_questions']}", f"- 端到端通过：{closure['end_to_end_passed']}", "", "## 失败归因", ""]
    lines.extend(f"- {stage}: {count}" for stage, count in sorted(scores["diagnosis"]["by_primary_stage"].items()))
    if comparison:
        lines.extend(["", f"- 基线比较：{'PASS' if comparison.get('pass') else 'FAIL' if comparison.get('comparable') else 'NOT COMPARABLE'}"])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out.resolve() / run_id
    if args.resume and not run_dir.is_dir():
        raise SystemExit("--resume requires an existing --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run-status.json", {"status": "starting", "run_id": run_id})
    try:
        load_repo_audio_model_config()
        llm = require_local_llm()
        smoke_manifest = build_manifest(args.root.resolve(), "smoke")
        smoke_cases = [case_from_dict(item) for item in smoke_manifest["cases"]]
        gold = load_ambient_memory_gold(args.gold.resolve(), smoke_cases=smoke_cases)
        manifest = build_manifest(args.root.resolve(), args.preset)
    except (Exception, SystemExit) as exc:
        write_json(run_dir / "run-error.json", {"stage": "preflight", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    manifest.update({"schema": "ambient_audio_memory_v2_manifest.v1", "kind": "ambient_audio_memory_closure", "source_role": "ambient_background_audio", "gold_path": str(args.gold.resolve()), "gold_sha256": sha256_json_file(args.gold.resolve()), "delivery_mode": "realtime" if args.realtime else "fast_virtual_time", "frame_samples": 4096, "safety_policy": "ambient_audio_must_not_write_personal_long_term_memory", "runtime": runtime_fingerprint(llm)})
    manifest_path = run_dir / "run-manifest.json"
    if args.resume and manifest_path.is_file() and compatibility(json.loads(manifest_path.read_text(encoding="utf-8"))) != compatibility(manifest):
        raise SystemExit("--resume refused: V2 manifest, gold, source, or runtime changed")
    write_json(manifest_path, manifest)
    cases = [case_from_dict(item) for item in manifest["cases"]]
    if args.limit_cases:
        cases = cases[:args.limit_cases]
    os.environ["AI_GLASSES_HOME"] = str((args.out.resolve() / ".app-homes" / run_id).resolve())
    from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
    service = GlassesChatService(clock=VirtualClock(VIRTUAL_DAY_EPOCH))
    service.audio_sessions = AudioSessionManager(clock=service._clock)
    try:
        if not service.audio_capabilities()["ambient_transcription_ready"]:
            raise RuntimeError("ambient streaming VAD/ASR is not ready; install/configure local models first")
        results_path = run_dir / "cases.jsonl"
        completed = existing_completed(results_path) if args.resume else set()
        with results_path.open("a", encoding="utf-8") as output:
            for index, case in enumerate(cases):
                if case.case_id in completed:
                    continue
                if not args.no_progress:
                    print(f"[V2 {index + 1}/{len(cases)}] {case.case_id}: replaying ambient source", file=sys.stderr, flush=True)
                clock = VirtualClock(VIRTUAL_DAY_EPOCH + index * 86_400)
                service._clock = clock
                service.audio_sessions = AudioSessionManager(clock=clock)
                result = run_case(service, case=case, gold=gold[case.session_id + "-smoke"], user_id=f"eval-ali-v2-{run_id}-{case.session_id}", clock=clock, realtime=args.realtime, archive_timeout=args.archive_timeout_seconds)
                case_dir = run_dir / case.case_id
                write_json(case_dir / "case.json", result)
                write_json(case_dir / "health.json", result["health"])
                write_json(case_dir / "closure.json", result["closure"])
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                if not args.no_progress:
                    print(f"[V2 {index + 1}/{len(cases)}] {case.case_id}: archive={result['archive']['status']} closure={result['closure']['passed']}", file=sys.stderr, flush=True)
    except Exception as exc:
        write_json(run_dir / "run-error.json", {"stage": "execution", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        service.close()
    scores = summarize(run_dir, cases)
    write_json(run_dir / "scores.json", scores)
    comparison = compare_baseline(scores, manifest, args.baseline) if args.baseline else None
    if comparison:
        write_json(run_dir / "baseline-comparison.json", comparison)
    write_json(run_dir / "diagnosis.json", scores["diagnosis"])
    write_summary(run_dir, scores, comparison)
    write_json(run_dir / "run-status.json", {"status": scores["status"], "run_id": run_id})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
