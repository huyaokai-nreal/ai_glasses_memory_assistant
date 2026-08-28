#!/usr/bin/env python3
"""Run Eval_Ali_far through the shared streaming audio and memory pipeline.

This is an offline software benchmark: Eval_Ali WAV is pushed directly into
the same PCM session API that the app uses after microphone capture. It does
not exercise Android hardware, speakers, or acoustic propagation.
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
from typing import Any, Callable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_glasses_memory_assistant.audio_engine.runtime import AudioSessionManager  # noqa: E402
from ai_glasses_memory_assistant.evals.eval_ali import (  # noqa: E402
    EvalAliCase,
    ReferenceInterval,
    build_manifest,
    diagnose_case,
    score_events,
    write_json,
)


LOCAL_LLM_ENV = (
    "AI_GLASSES_LLM_PROVIDER",
    "AI_GLASSES_LLM_MODEL",
    "AI_GLASSES_LLM_BASE_URL",
    "AI_GLASSES_LLM_API_KEY",
)
SAMPLE_RATE = 16_000
BOUNDARY_SILENCE_SECONDS = 1.0
ARCHIVE_TIMEOUT_SECONDS = 180.0


@dataclass
class VirtualClock:
    """Thread-safe audio time used only by this deterministic offline runner."""

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
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "eval_ali")
    parser.add_argument("--run-id", default="", help="Existing run id is required with --resume.")
    parser.add_argument("--preset", choices=("smoke", "regression", "full"), default="smoke")
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Pace PCM delivery at 1x instead of processing as fast as possible.")
    parser.add_argument("--continuous-full", action="store_true", help="Use one unbroken session for every full-meeting case.")
    parser.add_argument("--archive-timeout-seconds", type=float, default=ARCHIVE_TIMEOUT_SECONDS)
    parser.add_argument("--baseline", type=Path, help="Locked offline baseline JSON to compare after scoring.")
    return parser.parse_args()


def require_local_llm() -> dict[str, str]:
    values = {name: str(os.getenv(name) or "").strip() for name in LOCAL_LLM_ENV}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit("offline archive requires explicit local LLM configuration: " + ", ".join(missing))
    if values["AI_GLASSES_LLM_PROVIDER"].lower() not in {"llama_cpp", "ollama"}:
        raise SystemExit("offline archive only permits a local llama_cpp or ollama provider")
    if "deepseek" in values["AI_GLASSES_LLM_BASE_URL"].lower() or values["AI_GLASSES_LLM_BASE_URL"].startswith("https://"):
        raise SystemExit("offline archive refuses cloud/DeepSeek configuration")
    return values


def case_from_dict(payload: dict[str, Any]) -> EvalAliCase:
    return EvalAliCase(
        **{key: value for key, value in payload.items() if key != "reference_intervals"},
        reference_intervals=tuple(ReferenceInterval(**item) for item in payload["reference_intervals"]),
    )


def pcm16_base64(samples: np.ndarray) -> str:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2", copy=False).tobytes()
    return base64.b64encode(pcm).decode("ascii")


def offline_compatibility(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest.get("schema"),
        "preset": manifest.get("preset"),
        "channel": manifest.get("channel"),
        "delivery_mode": manifest.get("delivery_mode"),
        "frame_samples": manifest.get("frame_samples"),
        "safety_policy": manifest.get("safety_policy"),
        "runtime": manifest.get("runtime"),
        "cases": [{key: item.get(key) for key in ("case_id", "audio_sha256", "textgrid_sha256", "start_s", "end_s")}
                  for item in manifest.get("cases") or []],
    }


def runtime_fingerprint(llm: dict[str, str]) -> dict[str, Any]:
    def command(*parts: str) -> str:
        result = subprocess.run(parts, cwd=ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip() or result.stderr.strip()

    return {
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "llm_provider": llm["AI_GLASSES_LLM_PROVIDER"],
        "llm_model": llm["AI_GLASSES_LLM_MODEL"],
        "llm_base_url": llm["AI_GLASSES_LLM_BASE_URL"],
        "asr_model_dir": str(os.getenv("AI_GLASSES_ASR_MODEL_DIR") or ""),
    }


def await_archives(service: Any, *, user_id: str, capture_id: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    while True:
        slices = service.timeline_store.list_discussion_slices(user_id, capture_id=capture_id)
        statuses = [str(item.get("status") or "") for item in slices]
        if not slices:
            return {"status": "no_slices", "slices": []}
        if all(status in {"completed", "failed"} for status in statuses):
            return {"status": "completed" if all(status == "completed" for status in statuses) else "failed", "slices": slices}
        if time.monotonic() >= deadline:
            return {"status": "timeout", "slices": slices}
        time.sleep(0.05)


def audit_summary(service: Any, *, user_id: str) -> dict[str, int]:
    """Keep only record-type counts; the full audit remains isolated app state."""

    counts: dict[str, int] = {}
    path = getattr(service, "audit_path", None)
    if not isinstance(path, Path) or not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get("user_id") or "") != user_id:
            continue
        kind = str(record.get("record_type") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _pace_realtime(*, realtime: bool, wall_start: float, audio_seconds_sent: float, sleep: Callable[[float], None]) -> float:
    if not realtime:
        return 0.0
    lag = time.monotonic() - (wall_start + audio_seconds_sent)
    if lag < 0.0:
        sleep(-lag)
        return 0.0
    return lag


def push_case_audio(
    service: Any,
    *,
    session: dict[str, Any],
    case: EvalAliCase,
    clock: VirtualClock,
    sequence: int,
    realtime: bool,
    wall_start: float,
    audio_seconds_sent: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, float, list[dict[str, Any]], dict[str, float | int]]:
    """Feed one selected source window and flush it with deterministic silence."""

    events: list[dict[str, Any]] = []
    push_count = 0
    max_lag = 0.0
    max_push_seconds = 0.0
    frame_samples = int(session["format"]["recommended_push_samples"])

    def push(samples: np.ndarray) -> None:
        nonlocal sequence, audio_seconds_sent, push_count, max_lag, max_push_seconds
        max_lag = max(max_lag, _pace_realtime(
            realtime=realtime, wall_start=wall_start, audio_seconds_sent=audio_seconds_sent, sleep=sleep,
        ))
        duration = len(samples) / SAMPLE_RATE
        clock.advance(duration)
        started = time.perf_counter()
        response = service.push_audio_session(
            user_id=session["user_id"],
            audio_session_id=session["audio_session_id"],
            session_token=session["session_token"],
            sequence=sequence,
            pcm16_base64=pcm16_base64(samples),
        )
        max_push_seconds = max(max_push_seconds, time.perf_counter() - started)
        events.extend({"event": event, "dispatches": response.get("dispatches") or []} for event in response.get("events") or [])
        sequence += 1
        push_count += 1
        audio_seconds_sent += duration

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
    silence = np.zeros(frame_samples, dtype=np.float32)
    for _ in range(round(BOUNDARY_SILENCE_SECONDS * SAMPLE_RATE / frame_samples)):
        push(silence)
    return sequence, audio_seconds_sent, events, {
        "push_count": push_count,
        "max_delivery_lag_seconds": max_lag,
        "max_push_seconds": max_push_seconds,
    }


def score_case(case: EvalAliCase, events: list[dict[str, Any]], *, offset_ms: float, archive: dict[str, Any], memory_saved: int) -> tuple[dict[str, Any], dict[str, Any]]:
    score = score_events(case, events, playback_offset_ms=offset_ms)
    if archive["status"] in {"failed", "timeout"}:
        diagnosis = {"primary_stage": "archive", "reason": f"discussion_archive_{archive['status']}"}
    else:
        diagnosis = diagnose_case(score, preflight_ok=True, queue_failed=0, memory_saved=memory_saved)
    return score, diagnosis


def run_case(service: Any, *, case: EvalAliCase, user_id: str, clock: VirtualClock, realtime: bool, archive_timeout: float) -> dict[str, Any]:
    session = service.start_audio_session(user_id=user_id, mode="ambient")
    wall_start = time.monotonic()
    sequence, audio_seconds, events, delivery = push_case_audio(
        service, session=session, case=case, clock=clock, sequence=1, realtime=realtime,
        wall_start=wall_start, audio_seconds_sent=0.0,
    )
    stopped = service.stop_audio_session(
        user_id=user_id,
        audio_session_id=session["audio_session_id"],
        session_token=session["session_token"],
    )
    capture_id = str(session["capture_id"])
    archive = await_archives(service, user_id=user_id, capture_id=capture_id, timeout_seconds=archive_timeout)
    memory_saved = len(service.memory_store.list_memories(user_id=user_id, limit=10_000))
    score, diagnosis = score_case(case, events, offset_ms=0.0, archive=archive, memory_saved=memory_saved)
    return {
        "status": "complete" if archive["status"] not in {"failed", "timeout"} and memory_saved == 0 else "failed",
        "case_id": case.case_id,
        "user_id": user_id,
        "capture_id": capture_id,
        "stream": {"frame_samples": session["format"]["recommended_push_samples"], "audio_seconds": audio_seconds,
                   "wall_seconds": time.monotonic() - wall_start, **delivery},
        "stop": stopped,
        "archive": archive,
        "audit_record_type_counts": audit_summary(service, user_id=user_id),
        "memory_saved": memory_saved,
        "score": score,
        "diagnosis": diagnosis,
    }


def run_continuous_full(service: Any, *, cases: list[EvalAliCase], run_id: str, clock: VirtualClock, realtime: bool, archive_timeout: float) -> list[dict[str, Any]]:
    user_id = f"eval-ali-offline-{run_id}-continuous"
    session = service.start_audio_session(user_id=user_id, mode="ambient")
    wall_start = time.monotonic()
    sequence = 1
    audio_seconds = 0.0
    results: list[dict[str, Any]] = []
    for case in cases:
        case_wall_start = time.monotonic()
        offset_ms = round(audio_seconds * 1000.0)
        sequence, audio_seconds, events, delivery = push_case_audio(
            service, session=session, case=case, clock=clock, sequence=sequence, realtime=realtime,
            wall_start=wall_start, audio_seconds_sent=audio_seconds,
        )
        results.append({"case": case, "events": events, "offset_ms": offset_ms, "delivery": delivery,
                        "wall_seconds": time.monotonic() - case_wall_start})
    stopped = service.stop_audio_session(
        user_id=user_id, audio_session_id=session["audio_session_id"], session_token=session["session_token"],
    )
    archive = await_archives(service, user_id=user_id, capture_id=str(session["capture_id"]), timeout_seconds=archive_timeout)
    memory_saved = len(service.memory_store.list_memories(user_id=user_id, limit=10_000))
    output = []
    for item in results:
        score, diagnosis = score_case(item["case"], item["events"], offset_ms=item["offset_ms"], archive=archive, memory_saved=memory_saved)
        output.append({
            "status": "complete" if archive["status"] not in {"failed", "timeout"} and memory_saved == 0 else "failed",
            "case_id": item["case"].case_id, "user_id": user_id, "capture_id": session["capture_id"],
            "stream": {"frame_samples": session["format"]["recommended_push_samples"], "audio_seconds": item["case"].end_s - item["case"].start_s,
                       "wall_seconds": item["wall_seconds"], **item["delivery"]},
            "stop": stopped, "archive": archive, "audit_record_type_counts": audit_summary(service, user_id=user_id),
            "memory_saved": memory_saved, "score": score, "diagnosis": diagnosis,
        })
    return output


def existing_complete_cases(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        case_id = str(row.get("case_id") or "")
        case_dir = path.parent / case_id
        if (row.get("status") == "complete" and row.get("archive", {}).get("status") in {"completed", "no_slices"}
                and int(row.get("memory_saved") or 0) == 0 and (case_dir / "score.json").is_file()
                and (case_dir / "diagnosis.json").is_file() and (case_dir / "case.json").is_file()):
            completed.add(case_id)
    return completed


def summarize(run_dir: Path, cases: list[EvalAliCase]) -> tuple[dict[str, Any], dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    cases_path = run_dir / "cases.jsonl"
    for line in cases_path.read_text(encoding="utf-8").splitlines() if cases_path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("case_id") or "") in {case.case_id for case in cases}:
            records[str(row["case_id"])] = row
    score_rows = [json.loads((run_dir / case.case_id / "score.json").read_text(encoding="utf-8"))
                  for case in cases if (run_dir / case.case_id / "score.json").is_file()]
    complete = len(records) == len(cases) and len(score_rows) == len(cases) and all(row.get("status") == "complete" for row in records.values())
    ref_chars = sum(int(row["cer"]["normalized"]["reference_chars"]) for row in score_rows)
    errors = sum(int(row["cer"]["normalized"]["errors"]) for row in score_rows)
    vad_tp = sum(int(row["vad"]["true_positive_frames"]) for row in score_rows)
    vad_fp = sum(int(row["vad"]["false_positive_frames"]) for row in score_rows)
    vad_fn = sum(int(row["vad"]["false_negative_frames"]) for row in score_rows)
    precision = vad_tp / (vad_tp + vad_fp) if vad_tp + vad_fp else 0.0
    recall = vad_tp / (vad_tp + vad_fn) if vad_tp + vad_fn else 0.0
    wall_seconds = sum(float((row.get("stream") or {}).get("wall_seconds") or 0.0) for row in records.values())
    audio_seconds = sum(float((row.get("stream") or {}).get("audio_seconds") or 0.0) for row in records.values())
    diagnosis_counts: dict[str, int] = {}
    for row in records.values():
        stage = str((row.get("diagnosis") or {}).get("primary_stage") or "unknown")
        diagnosis_counts[stage] = diagnosis_counts.get(stage, 0) + 1
    scores = {
        "schema": "eval_ali_offline_scores.v1", "status": "complete" if complete else "incomplete",
        "expected_cases": len(cases), "completed_cases": len(score_rows),
        "normalized_cer": errors / ref_chars if complete and ref_chars else None,
        "vad": {"precision": precision if complete else None, "recall": recall if complete else None,
                "f1": 2 * precision * recall / (precision + recall) if complete and precision + recall else None},
        "throughput": {"audio_seconds": audio_seconds if complete else None, "wall_seconds": wall_seconds if complete else None,
                       "wall_seconds_per_audio_second": wall_seconds / audio_seconds if complete and audio_seconds else None},
        "archive_failed": sum((row.get("archive") or {}).get("status") in {"failed", "timeout"} for row in records.values()),
        "memory_saved": max((int(row.get("memory_saved") or 0) for row in records.values()), default=0),
    }
    diagnosis = {"schema": "eval_ali_offline_diagnosis.v1", "by_primary_stage": diagnosis_counts,
                 "missing_cases": sorted({case.case_id for case in cases} - set(records))}
    return scores, diagnosis


def compare_baseline(scores: dict[str, Any], manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("compatibility") != offline_compatibility(manifest):
        return {"schema": "eval_ali_offline_baseline_comparison.v1", "comparable": False,
                "reason": "configuration_or_frozen_cases_differ", "pass": None, "checks": []}
    base = baseline["scores"]
    checks = [{"metric": "run_complete", "pass": scores["status"] == "complete", "actual": scores["status"], "limit": "complete"}]
    if isinstance(scores["normalized_cer"], (int, float)) and isinstance(base.get("normalized_cer"), (int, float)):
        limit = base["normalized_cer"] + max(0.02, base["normalized_cer"] * 0.10)
        checks.append({"metric": "normalized_cer", "pass": scores["normalized_cer"] <= limit, "actual": scores["normalized_cer"], "limit": limit})
    if isinstance(scores["vad"]["f1"], (int, float)) and isinstance(base.get("vad", {}).get("f1"), (int, float)):
        limit = base["vad"]["f1"] - 0.05
        checks.append({"metric": "vad_f1", "pass": scores["vad"]["f1"] >= limit, "actual": scores["vad"]["f1"], "limit": limit})
    current = scores["throughput"].get("wall_seconds_per_audio_second")
    previous = base.get("throughput", {}).get("wall_seconds_per_audio_second")
    if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
        limit = previous * 1.25 + 0.02
        checks.append({"metric": "wall_seconds_per_audio_second", "pass": current <= limit, "actual": current, "limit": limit})
    checks.extend([
        {"metric": "archive_failed", "pass": scores["archive_failed"] == 0, "actual": scores["archive_failed"], "limit": 0},
        {"metric": "memory_saved", "pass": scores["memory_saved"] == 0, "actual": scores["memory_saved"], "limit": 0},
    ])
    return {"schema": "eval_ali_offline_baseline_comparison.v1", "comparable": True, "checks": checks,
            "pass": all(check["pass"] for check in checks)}


def write_summary(run_dir: Path, scores: dict[str, Any], diagnosis: dict[str, Any], comparison: dict[str, Any] | None) -> None:
    lines = [
        "# Eval_Ali_far 离线持续收音基准", "",
        "- 此结果只验证共享软件链路，不代表手机麦克风或真实环境声学效果。",
        f"- 状态：{scores['status']}", f"- 完成 case：{scores['completed_cases']}/{scores['expected_cases']}",
        f"- 规范化 CER：{scores['normalized_cer']}", f"- VAD F1：{scores['vad']['f1']}",
        f"- 每音频秒墙钟耗时：{scores['throughput']['wall_seconds_per_audio_second']}",
        f"- 归档失败：{scores['archive_failed']}", f"- 环境音错误写长期记忆：{scores['memory_saved']}", "", "## 首要归因", "",
    ]
    lines.extend(f"- {stage}: {count}" for stage, count in sorted(diagnosis["by_primary_stage"].items()))
    if comparison:
        outcome = "NOT COMPARABLE" if not comparison["comparable"] else ("PASS" if comparison["pass"] else "FAIL")
        lines.extend(["", f"- 基线比较：{outcome}"])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.continuous_full and args.preset != "full":
        raise SystemExit("--continuous-full requires --preset full")
    if args.continuous_full and args.resume:
        raise SystemExit("--continuous-full is deliberately not resumable")
    llm = require_local_llm()
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out.resolve() / run_id
    if args.resume and not run_dir.is_dir():
        raise SystemExit("--resume requires --run-id for an existing run")
    os.environ["AI_GLASSES_HOME"] = str((args.out.resolve() / ".app-homes" / run_id).resolve())
    manifest = build_manifest(args.root.resolve(), args.preset)
    manifest.update({
        "kind": "eval_ali_far_offline_streaming", "delivery_mode": "realtime" if args.realtime else "fast_virtual_time",
        "frame_samples": 4096, "boundary_silence_seconds": BOUNDARY_SILENCE_SECONDS,
        "safety_policy": "ambient_external_audio_must_not_write_long_term_memory",
        "runtime": runtime_fingerprint(llm), "command_sha256": hashlib.sha256("\0".join(sys.argv).encode()).hexdigest(),
    })
    manifest_path = run_dir / "run-manifest.json"
    if args.resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if offline_compatibility(previous) != offline_compatibility(manifest):
            raise SystemExit("--resume refused: frozen data or offline stream configuration changed")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    cases = [case_from_dict(item) for item in manifest["cases"]]
    if args.limit_cases:
        cases = cases[:args.limit_cases]
    from ai_glasses_memory_assistant.agent_bridge import GlassesChatService

    clock = VirtualClock(1_704_067_200.0)  # fixed 2024-01-01T00:00:00Z for deterministic archives
    service = GlassesChatService(clock=clock)
    service.audio_sessions = AudioSessionManager(clock=clock)
    try:
        capabilities = service.audio_capabilities()
        if not capabilities["ambient_transcription_ready"]:
            raise SystemExit("ambient streaming VAD/ASR is not ready; install/configure local models first")
        results_path = run_dir / "cases.jsonl"
        completed = existing_complete_cases(results_path) if args.resume else set()
        with results_path.open("a", encoding="utf-8") as output:
            if args.continuous_full:
                results = run_continuous_full(service, cases=cases, run_id=run_id, clock=clock,
                                              realtime=args.realtime, archive_timeout=args.archive_timeout_seconds)
                for result in results:
                    case_dir = run_dir / result["case_id"]
                    write_json(case_dir / "score.json", result.pop("score"))
                    write_json(case_dir / "diagnosis.json", result["diagnosis"])
                    write_json(case_dir / "case.json", result)
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
            else:
                for case in cases:
                    if case.case_id in completed:
                        continue
                    user_id = f"eval-ali-offline-{run_id}-{case.case_id}"
                    try:
                        result = run_case(service, case=case, user_id=user_id, clock=clock, realtime=args.realtime,
                                          archive_timeout=args.archive_timeout_seconds)
                        case_dir = run_dir / case.case_id
                        write_json(case_dir / "score.json", result.pop("score"))
                        write_json(case_dir / "diagnosis.json", result["diagnosis"])
                        write_json(case_dir / "case.json", result)
                    except Exception as error:  # noqa: BLE001
                        result = {"status": "failed", "case_id": case.case_id, "error_type": type(error).__name__, "error": str(error)}
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
    finally:
        service.close()
    scores, diagnosis = summarize(run_dir, cases)
    write_json(run_dir / "scores.json", scores)
    write_json(run_dir / "diagnosis.json", diagnosis)
    comparison = compare_baseline(scores, manifest, args.baseline) if args.baseline else None
    if comparison:
        write_json(run_dir / "baseline-comparison.json", comparison)
    write_summary(run_dir, scores, diagnosis, comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
