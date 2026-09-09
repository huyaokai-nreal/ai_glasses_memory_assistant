#!/usr/bin/env python3
"""Score one continuous multichannel frontend run and replay it into an isolated service.

This is the evaluation-only half of the continuous pipeline. It is the only place
allowed to read gold data: the TextGrid text, the evaluation-only speaker mapping
and the TextGrid interval boundaries. Nothing it computes can flow back into the
frontend; the frontend run directory is read-only input here.

Outputs:
    asr.jsonl            per-segment transcription (one row per segment_id/variant)
    audio-events.jsonl   public ``audio_event.v1`` payloads (no embedding, no PCM)
    scores.json          cpCER, component CER, DER, overlap recall, gates, RTF
    service-replay.json  isolated-home replay counters (only with --replay-service)
    REPORT.md            facts, diagnosis and open items, kept separate
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ai_glasses_memory_assistant.audio_engine import AudioEvent  # noqa: E402
from ai_glasses_memory_assistant.evals.eval_ali import (  # noqa: E402
    ReferenceInterval,
    edit_distance,
    normalize_text,
    parse_textgrid,
    score_cer,
    sha256_file,
)
from ai_glasses_memory_assistant.audio_engine.backends import OfflineAsrBackend  # noqa: E402
from diarize_sortformer import MODEL_SHA256 as SORTFORMER_MODEL_SHA256  # noqa: E402
from p0_oracle_interval_ablation import (  # noqa: E402
    IntervalResult,
    aggregate,
    load_source_run,
    overlaps_other_speaker,
    summarize_cer_counts,
)
from run_continuous_frontend import SCHEMA as FRONTEND_SCHEMA  # noqa: E402
from score_diarization import score_case  # noqa: E402

SCHEMA = "eval_ali_continuous_e2e_score.v1"
PREP_SCHEMA = "eval_ali_continuous_prep.v1"
SAMPLE_RATE = 16_000
# Locked by reports/p2_gss_prep/20260904-real-smoke/prep-manifest.json and the
# eval-ali-e-candidate-fix2 source run; any drift fails closed.
ASR_MODEL_SHA256 = "12ca1a2ae7ecf3e0019ef2822307ee0b5cadc9196569e379b4c4026f8205276d"


class ContinuousScoreError(Exception):
    """Any input drift, duplicate identity or contract violation."""


# --------------------------------------------------------------------------- #
# Input loading and fail-closed checks
# --------------------------------------------------------------------------- #


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_frontend_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "frontend-manifest.json"
    if not manifest_path.is_file():
        raise ContinuousScoreError(f"missing frontend manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FRONTEND_SCHEMA:
        raise ContinuousScoreError(f"unexpected frontend schema: {manifest.get('schema')}")
    rows = load_jsonl(run_dir / "segments.jsonl")
    if not rows:
        raise ContinuousScoreError("frontend produced no segments")
    # Lock the segment file with its own content hash so an edited or swapped
    # segments.jsonl cannot silently pass scoring (P1-5 second layer).
    expected_segments_sha = manifest.get("segments_jsonl_sha256")
    if not expected_segments_sha:
        raise ContinuousScoreError("frontend manifest missing segments_jsonl_sha256")
    actual_segments_sha = sha256_file(run_dir / "segments.jsonl")
    if actual_segments_sha != expected_segments_sha:
        raise ContinuousScoreError(
            f"segments.jsonl drift: manifest {expected_segments_sha}, file {actual_segments_sha}"
        )
    return manifest, rows


def load_prep_case(prep_manifest: Path, case_id: str) -> dict[str, Any]:
    prep = json.loads(Path(prep_manifest).read_text(encoding="utf-8"))
    if prep.get("schema") != PREP_SCHEMA:
        raise ContinuousScoreError(f"unexpected prep schema: {prep.get('schema')}")
    for case in prep.get("cases") or []:
        if case.get("case_id") == case_id:
            return case
    raise ContinuousScoreError(f"case {case_id!r} absent from the prep manifest")


def verify_frontend_against_prep(manifest: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on any input, model or threshold drift."""
    problems: list[str] = []
    frontend_case = manifest.get("case") or {}
    if frontend_case.get("cut_wav_sha256") != case.get("cut_wav_sha256"):
        problems.append("input WAV hash drift between the frontend run and the prep manifest")
    if frontend_case.get("case_id") != case.get("case_id"):
        problems.append("case mismatch between the frontend run and the prep manifest")
    if (manifest.get("model") or {}).get("sha256") != SORTFORMER_MODEL_SHA256:
        problems.append("Sortformer model hash drift")
    inputs = manifest.get("inputs") or {}
    run_manifest_path = inputs.get("sortformer_run_manifest")
    if not run_manifest_path or not Path(run_manifest_path).is_file():
        problems.append("Sortformer run manifest referenced by the frontend run is missing")
    elif inputs.get("sortformer_run_manifest_sha256") != sha256_file(Path(run_manifest_path)):
        problems.append("Sortformer run manifest drifted after the frontend run")
    probabilities_path = inputs.get("probabilities_path")
    if not probabilities_path or not Path(probabilities_path).is_file():
        problems.append("Sortformer probability file referenced by the frontend run is missing")
    elif inputs.get("probabilities_sha256") != sha256_file(Path(probabilities_path)):
        problems.append("Sortformer probability file drifted after the frontend run")
    if not (manifest.get("integrity") or {}).get("passed"):
        problems.append(f"frontend integrity gate failed: {manifest.get('integrity')}")
    if manifest.get("external_push_state_validated"):
        problems.append("frontend must not claim external push() state validation")
    return {"problems": problems, "passed": not problems}


def check_segment_rows(rows: list[dict[str, Any]], processed_seconds: float) -> dict[str, Any]:
    """Duplicate ids, duplicate intervals and out-of-range intervals fail closed."""
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    duplicate_intervals: list[str] = []
    out_of_range: list[str] = []
    per_variant_track: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        if row["segment_id"] in seen_ids:
            duplicate_ids.append(row["segment_id"])
        seen_ids.add(row["segment_id"])
        if row["start_s"] < -1e-9 or row["end_s"] > processed_seconds + 1e-9 or row["end_s"] <= row["start_s"]:
            out_of_range.append(row["segment_id"])
        per_variant_track.setdefault((row["variant"], row["track"]), []).append((row["start_s"], row["end_s"]))
    for spans in per_variant_track.values():
        ordered = sorted(spans)
        for (_, previous_end), (next_start, _) in zip(ordered, ordered[1:]):
            if next_start < previous_end - 1e-9:
                duplicate_intervals.append(f"{previous_end}..{next_start}")
    return {
        "duplicate_segment_ids": duplicate_ids,
        "duplicate_track_intervals": duplicate_intervals,
        "out_of_range_intervals": out_of_range,
        "total_rows": len(rows),
        "passed": not duplicate_ids and not duplicate_intervals and not out_of_range,
    }


# --------------------------------------------------------------------------- #
# Gold loading (scorer only)
# --------------------------------------------------------------------------- #


def load_gold_intervals(case: dict[str, Any], processed_seconds: float) -> list[ReferenceInterval]:
    """TextGrid intervals clipped to the processed window. Scorer-only."""
    textgrid = Path(case["textgrid_path"])
    if sha256_file(textgrid) != case["textgrid_sha256"]:
        raise ContinuousScoreError("TextGrid drifted after the prep manifest was written")
    intervals: list[ReferenceInterval] = []
    for item in parse_textgrid(textgrid):
        if not item.text.strip() or item.end_s <= item.start_s:
            continue
        start = max(0.0, float(item.start_s))
        end = min(processed_seconds, float(item.end_s))
        if end <= start:
            continue
        intervals.append(ReferenceInterval(speaker=item.speaker, start_s=start, end_s=end, text=item.text))
    if not intervals:
        raise ContinuousScoreError("no gold intervals inside the processed window")
    return intervals


# --------------------------------------------------------------------------- #
# cpCER
# --------------------------------------------------------------------------- #


def concatenated_cer(
    reference_by_speaker: dict[str, str],
    hypothesis_by_track: dict[str, str],
) -> dict[str, Any]:
    """Permutation-invariant concatenated CER over up to four anonymous tracks.

    Missing, extra and empty tracks are all scored: an unmatched reference
    contributes deletions, an unmatched hypothesis contributes insertions.

    The normalized edit distance for each unique (speaker, track) pair is
    computed exactly once and reused across permutations; each permutation only
    sums the pre-computed costs. ``score_cer`` also computes a ``strict``
    (un-normalized) distance that this function never read, so we call
    ``edit_distance`` on the normalized text directly and skip that work. The
    permutation order, tie handling, unmatched/empty semantics and returned
    fields are unchanged.
    """
    ref_keys = sorted(reference_by_speaker)
    hyp_keys = sorted(hypothesis_by_track)
    best: dict[str, Any] | None = None
    if len(hyp_keys) <= len(ref_keys):
        combinations = [
            list(zip(hyp_keys, assignment)) for assignment in itertools.permutations(ref_keys, len(hyp_keys))
        ]
    else:
        combinations = [
            list(zip(assignment, ref_keys)) for assignment in itertools.permutations(hyp_keys, len(ref_keys))
        ]
    if not combinations:
        combinations = [[]]

    def _normalized(reference: str, hypothesis: str) -> tuple[int, int]:
        distance = edit_distance(normalize_text(reference), normalize_text(hypothesis))
        return int(distance["errors"]), int(distance["reference_chars"])

    pair_cost: dict[tuple[str, str], tuple[int, int]] = {
        (speaker, track): _normalized(reference_by_speaker[speaker], hypothesis_by_track[track])
        for speaker in ref_keys
        for track in hyp_keys
    }
    deletion_cost: dict[str, tuple[int, int]] = {
        speaker: _normalized(reference_by_speaker[speaker], "") for speaker in ref_keys
    }
    insertion_errors: dict[str, int] = {
        track: _normalized("", hypothesis_by_track[track])[0] for track in hyp_keys
    }

    for pairs in combinations:
        errors = 0
        characters = 0
        used_ref = set()
        used_hyp = set()
        for track, speaker in pairs:
            pair_errors, pair_characters = pair_cost[(speaker, track)]
            errors += pair_errors
            characters += pair_characters
            used_ref.add(speaker)
            used_hyp.add(track)
        for speaker in ref_keys:
            if speaker in used_ref:
                continue
            deletion_errors, deletion_characters = deletion_cost[speaker]
            errors += deletion_errors
            characters += deletion_characters
        for track in hyp_keys:
            if track in used_hyp:
                continue
            errors += insertion_errors[track]
        candidate = {
            "errors": errors,
            "reference_chars": characters,
            "cer": (errors / characters) if characters else None,
            "mapping": {track: speaker for track, speaker in pairs},
        }
        if best is None or (candidate["cer"] is not None and (best["cer"] is None or candidate["cer"] < best["cer"])):
            best = candidate
    assert best is not None
    best["reference_speakers"] = len(ref_keys)
    best["hypothesis_tracks"] = len(hyp_keys)
    return best


# --------------------------------------------------------------------------- #
# Audio slicing
# --------------------------------------------------------------------------- #


def slice_track_audio(
    audio_by_id: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """Rebuild a fixed-timeline [start_s, end_s) range of one track.

    Each segment is placed at its declared offset; gaps (predicted holes) stay
    silent instead of being collapsed, so overlap/non-overlap CER follow the
    original time axis (P2-2 fix).
    """
    total = int(round((end_s - start_s) * SAMPLE_RATE))
    result = np.zeros(total, dtype=np.float32)
    for row in rows:
        if row["end_s"] <= start_s + 1e-9 or row["start_s"] >= end_s - 1e-9:
            continue
        segment_start = max(start_s, row["start_s"])
        segment_end = min(end_s, row["end_s"])
        if segment_end <= segment_start:
            continue
        waveform = audio_by_id.get(row["segment_id"])
        if waveform is None or waveform.size == 0:
            continue
        seg_start = max(start_s, row["start_s"])
        seg_end = min(end_s, row["end_s"])
        offset_in_seg = int(round((seg_start - row["start_s"]) * SAMPLE_RATE))
        length = int(round((seg_end - seg_start) * SAMPLE_RATE))
        if length <= 0:
            continue
        dest_start = int(round((seg_start - start_s) * SAMPLE_RATE))
        dest_end = dest_start + length
        if dest_end > total:
            length = total - dest_start
            dest_end = total
        if length <= 0:
            continue
        result[dest_start:dest_end] = waveform[offset_in_seg:offset_in_seg + length]
    return result


def load_segment_audio(rows: list[dict[str, Any]], run_dir: Path) -> dict[str, np.ndarray]:
    audio: dict[str, np.ndarray] = {}
    for row in rows:
        path = run_dir / row["wav_path"]
        with sf.SoundFile(str(path)) as source:
            if source.samplerate != SAMPLE_RATE:
                raise ContinuousScoreError(f"{path}: expected {SAMPLE_RATE} Hz")
            data = source.read(dtype="float32", always_2d=True)
        expected = row.get("wav_sha256")
        if not expected:
            raise ContinuousScoreError(f"{row.get('segment_id')}: missing wav_sha256")
        actual = sha256_file(path)
        if actual != expected:
            raise ContinuousScoreError(f"{path}: WAV hash drift (expected {expected}, got {actual})")
        expected_samples = row.get("sample_count")
        if expected_samples is None:
            raise ContinuousScoreError(f"{row.get('segment_id')}: missing sample_count")
        if int(expected_samples) != int(data.shape[0]):
            raise ContinuousScoreError(
                f"{row.get('segment_id')}: sample_count mismatch (manifest {expected_samples}, WAV {data.shape[0]})"
            )
        audio[row["segment_id"]] = data[:, 0]
    return audio


# --------------------------------------------------------------------------- #
# Event construction
# --------------------------------------------------------------------------- #


def build_audio_event(row: dict[str, Any], text: str, *, variant: str, asr_backend: str, asr_sha: str, threshold: float) -> dict[str, Any]:
    """Public ``audio_event.v1`` payload: anonymous, text-only, never PCM."""
    rejected_reason = ""
    if not text.strip():
        rejected_reason = "empty_asr_text"
    elif row.get("fallback_reason"):
        rejected_reason = "track_enhancement_failed"
    elif float(row.get("confidence") or 0.0) < threshold:
        rejected_reason = "low_track_confidence"
    overlap_state = "suspected" if bool(row.get("overlap")) else "not_observed"
    payload = {
        "schema_version": "audio_event.v1",
        "event_id": f"{row['segment_id']}:{variant}",
        "audio_session_id": f"continuous-{row['session_id']}",
        "segment_id": row["segment_id"],
        "type": "speech_rejected" if rejected_reason else "transcript_final",
        "lane": "ambient",
        "source_type": "ambient_audio",
        "start_ms": int(round(float(row["start_s"]) * 1000)),
        "end_ms": int(round(float(row["end_s"]) * 1000)),
        "text": "" if rejected_reason else text,
        "final": True,
        "vad": {
            "backend": "sortformer_activity",
            "threshold": threshold,
            **({"reason": rejected_reason} if rejected_reason else {}),
        },
        "wake": {},
        "asr": {"backend": asr_backend, "model_sha256": asr_sha},
        "speaker": {
            "state": "unknown",
            "reason": "anonymous_diarization_track",
            "voice_group": row["track"],
            "track_confidence": row.get("confidence"),
            "track_scope": "capture",
        },
        "overlap": {"state": overlap_state},
        "audio_retention": "discarded_after_processing",
    }
    # Round-trip through the product contract so an invalid payload fails here.
    AudioEvent.from_dict(payload)
    return payload


# --------------------------------------------------------------------------- #
# Phase 4: isolated service replay
# --------------------------------------------------------------------------- #


class _NetworkGuard:
    """Count and block outbound network attempts during the isolated replay."""

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._connect = socket.socket.connect
        self._create = socket.create_connection

    def __enter__(self) -> "_NetworkGuard":
        guard = self

        def guarded_connect(sock: socket.socket, address: Any) -> None:
            if getattr(sock, "family", None) == socket.AF_UNIX:
                return guard._connect(sock, address)
            guard.attempts.append(f"connect:{address}")
            raise RuntimeError("network access is not allowed during the isolated replay")

        def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
            guard.attempts.append(f"create_connection:{address}")
            raise RuntimeError("network access is not allowed during the isolated replay")

        socket.socket.connect = guarded_connect  # type: ignore[assignment]
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]
        return self

    def __exit__(self, *exc_info: Any) -> None:
        socket.socket.connect = self._connect  # type: ignore[assignment]
        socket.create_connection = self._create  # type: ignore[assignment]


def replay_events(events: list[dict[str, Any]], *, timeout: float = 30.0) -> dict[str, Any]:
    """Replay generated events through the real service in a throwaway home."""
    from ai_glasses_memory_assistant.agent_bridge import GlassesChatService

    previous_home = os.environ.get("AI_GLASSES_HOME")
    home = Path(tempfile.mkdtemp(prefix="continuous-e2e-home-"))
    os.environ["AI_GLASSES_HOME"] = str(home)
    user_id = "continuous-e2e"
    result: dict[str, Any] = {
        "isolated_home": str(home),
        "events_replayed": len(events),
        "network_attempts": [],
        "transcript_final_events": 0,
        "speech_rejected_events": 0,
        "partial_events": 0,
        "ui_only_events": 0,
        "capture_chunk_count": 0,
        "timeline_entries_added": 0,
        "memory_before": 0,
        "memory_after": 0,
        "capture_status": "",
    }
    guard = _NetworkGuard()
    service = None
    try:
        with guard:
            service = GlassesChatService()
            service.set_device_network_state(online=False)
            memory_before = len(service.memory_store.list_memories(user_id, limit=500))
            capture = service.start_device_capture(user_id=user_id)
            capture_id = str(capture["capture_id"])
            for event in events:
                queued = service.ingest_device_audio_event(
                    user_id=user_id,
                    event_payload=event,
                    capture_id=capture_id,
                )
                if not queued.get("queued"):
                    result["ui_only_events"] += 1
                    continue
                if event["type"] == "transcript_final":
                    result["transcript_final_events"] += 1
                else:
                    result["speech_rejected_events"] += 1
                service.wait_device_audio_event(
                    user_id=user_id, event_id=event["event_id"], timeout=timeout
                )
            persisted = service.timeline_store.get_capture(user_id, capture_id) or {}
            chunks = persisted.get("chunks") or []
            result["capture_chunk_count"] = len(chunks)
            # Each persisted timeline chunk stores the originating event id in
            # metadata.audio_event_id (set by agent_bridge._consume_audio_event).
            # Read from there, never the chunk top level (P3-1).
            result["persisted_audio_event_ids"] = [
                (
                    (chunk.get("metadata") or {}).get("audio_event_id")
                    or chunk.get("audio_event_id")
                    or chunk.get("event_id")
                )
                for chunk in chunks
            ]
            result["memory_before"] = memory_before
            result["memory_after"] = len(service.memory_store.list_memories(user_id, limit=500))
            result["capture_status"] = str(persisted.get("status") or "")
            # Interrupt instead of stop_capture(): stopping schedules the
            # discussion archive, which would need an LLM call.
            service.timeline_store.finish_capture(
                user_id,
                capture_id,
                summary="continuous e2e replay interrupted",
                ended_at=time.time(),
                status="interrupted",
            )
            result["capture_status_after_finish"] = str(
                (service.timeline_store.get_capture(user_id, capture_id) or {}).get("status") or ""
            )
            audit_text = service.audit_path.read_text(encoding="utf-8") if service.audit_path.is_file() else ""
            result["audit_bytes"] = len(audit_text)
            result["audit_contains_embedding"] = "speaker_embedding" in audit_text
            result["audit_contains_pcm"] = "pcm16" in audit_text.lower()
            result["chunk_texts_are_from_rejected"] = any(not str(chunk.get("text") or "").strip() for chunk in chunks)
            service.close()
    finally:
        result["network_attempts"] = list(guard.attempts)
        if previous_home is None:
            os.environ.pop("AI_GLASSES_HOME", None)
        else:
            os.environ["AI_GLASSES_HOME"] = previous_home
    result["network_calls"] = len(result["network_attempts"])
    return result


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def transcribe_segments(
    asr: OfflineAsrBackend,
    rows: list[dict[str, Any]],
    audio_by_id: dict[str, np.ndarray],
    *,
    asr_sha: str,
) -> list[dict[str, Any]]:
    transcripts: list[dict[str, Any]] = []
    for row in rows:
        waveform = audio_by_id[row["segment_id"]]
        started = time.perf_counter()
        text = str(asr.transcribe(np.asarray(waveform, dtype=np.float32)) or "").strip()
        transcripts.append({
            "segment_id": row["segment_id"],
            "variant": row["variant"],
            "track": row["track"],
            "start_s": row["start_s"],
            "end_s": row["end_s"],
            "text": text,
            "asr_model_sha256": asr_sha,
            "seconds": round(time.perf_counter() - started, 4),
        })
    return transcripts


def score_intervals(
    asr: OfflineAsrBackend,
    *,
    gold: list[ReferenceInterval],
    rows_by_track: dict[str, list[dict[str, Any]]],
    audio_by_id: dict[str, np.ndarray],
    mapping: dict[str, str],
) -> list[IntervalResult]:
    results: list[IntervalResult] = []
    gold_speakers = sorted({item.speaker for item in gold})
    for interval in gold:
        speaker_index = gold_speakers.index(interval.speaker)
        track = None
        for candidate, mapped in mapping.items():
            if mapped == interval.speaker or mapped == f"speaker_{speaker_index}":
                track = candidate
                break
        waveform = (
            slice_track_audio(audio_by_id, rows_by_track.get(track, []), interval.start_s, interval.end_s)
            if track
            else np.zeros(0, dtype=np.float32)
        )
        hypothesis = str(asr.transcribe(np.asarray(waveform, dtype=np.float32)) or "").strip() if waveform.size else ""
        normalized = score_cer(interval.text, hypothesis)["normalized"]
        results.append(
            IntervalResult(
                case_id="continuous",
                speaker=interval.speaker,
                start_s=interval.start_s,
                end_s=interval.end_s,
                duration_s=interval.end_s - interval.start_s,
                overlap_exposed=overlaps_other_speaker(interval, gold),
                reference_chars=int(normalized["reference_chars"]),
                errors=int(normalized["errors"]),
                substitutions=int(normalized["substitutions"]),
                deletions=int(normalized["deletions"]),
                insertions=int(normalized["insertions"]),
            )
        )
    return results


#: Diarization quality gates. These only make sense at full-meeting scale: a
#: 75-second smoke slice carries ~80 overlap reference frames, far too few to
#: judge recall, and all8/ch0 share one Sortformer prediction so the metric has
#: zero discriminative power for the enhancement comparison. They are therefore
#: *diagnostic only* in the smoke scope and *enforced* in the full scope.
DIAGNOSTIC_ONLY_IN_SMOKE: tuple[str, ...] = ("der_le_25", "overlap_recall_ge_70")
GATE_SCOPES: tuple[str, ...] = ("smoke", "full")
#: A run only counts as truncated when the processed window is strictly shorter
#: than the meeting it came from, by more than rounding noise (1 ms).
TRUNCATION_EPSILON_S = 1e-3


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def explain_gate_scope(manifest: dict[str, Any]) -> dict[str, Any]:
    """Collect the evidence needed to claim "this run is a truncated smoke slice".

    The smoke scope relaxes the diarization gates, so it must be *earned*: a run
    may only be called truncated when the frontend manifest proves, with numbers,
    that the processed window is shorter than the meeting it came from. Anything
    missing, unreadable or not strictly shorter falls back to the full scope.
    """
    case = manifest.get("case") or {}
    limit = _as_float(case.get("limit_seconds"))
    processed = _as_float(case.get("processed_seconds"))
    window_start = _as_float(case.get("window_start_s"))
    window_end = _as_float(case.get("window_end_s"))
    source_duration = _as_float(case.get("source_duration_s"))
    meeting_seconds: float | None = None
    if window_start is not None and window_end is not None and window_end > window_start:
        meeting_seconds = window_end - window_start
    if meeting_seconds is None:
        meeting_seconds = source_duration

    reasons: list[str] = []
    if limit is None or limit <= 0:
        reasons.append("no positive --limit-seconds on the frontend run")
    if processed is None:
        reasons.append("processed_seconds missing or unreadable")
    if meeting_seconds is None or meeting_seconds <= 0:
        reasons.append("meeting duration missing or unreadable (window_end_s/window_start_s/source_duration_s)")
    truncated = not reasons
    if truncated and not (processed < meeting_seconds - TRUNCATION_EPSILON_S):  # type: ignore[operator]
        reasons.append(
            f"processed_seconds {processed} is not shorter than the meeting "
            f"{meeting_seconds} (not truncated)"
        )
        truncated = False
    return {
        "truncated": truncated,
        "limit_seconds": limit,
        "processed_seconds": processed,
        "meeting_seconds": meeting_seconds,
        "reasons": reasons,
    }


def resolve_gate_scope(manifest: dict[str, Any], override: str | None = None) -> str:
    """Pick the gate scope. ``auto`` follows the proven truncation evidence.

    A run may only take the relaxed ``smoke`` scope when the manifest proves the
    processed window is shorter than the meeting (see :func:`explain_gate_scope`).
    Missing, invalid or unprovable metadata falls back to ``full``, so a complete
    meeting can never silently downgrade itself past the diarization gates.

    An explicit ``full`` override is always honoured (it can only strengthen the
    checks). An explicit ``smoke`` override on a run that is not provably
    truncated is *refused*: this is the failure mode the scope exists to prevent.
    """
    if override and override != "auto":
        if override not in GATE_SCOPES:
            raise ContinuousScoreError(f"unknown gate scope {override!r}; known: {list(GATE_SCOPES)}")
        if override == "full":
            return "full"
        evidence = explain_gate_scope(manifest)
        if not evidence["truncated"]:
            raise ContinuousScoreError(
                "refusing --gate-scope smoke: the run is not provably truncated "
                f"({'; '.join(evidence['reasons']) or 'no evidence'}); "
                "a full meeting must keep the diarization gates enforced"
            )
        return "smoke"
    return "smoke" if explain_gate_scope(manifest)["truncated"] else "full"


def evaluate_gates(
    scores: dict[str, dict[str, Any]],
    *,
    baseline: str,
    enhance: str,
    rtf: float,
    replay: dict[str, Any] | None,
    diarization: dict[str, Any],
    expected_final_event_ids: Iterable[str] | None = None,
    gate_scope: str = "full",
) -> dict[str, Any]:
    """Hard gates for one run. Any enforced failure => exit non-zero.

    Scope decides which checks actually count:

    * ``smoke`` (75-second preflight): the diarization gates are reported as
      **diagnostics** with their real values and never counted as passed, so an
      inapplicable metric is never written down as a pass.
    * ``full`` (complete meeting): DER <= 25% and overlap frame recall >= 70%
      are enforced, and a missing diarization metric counts as a failure rather
      than being skipped.

    Replay gates require a real isolated-service replay; when none was performed
    they are not-evaluated (False) and the full E2E gate FAILS, because "not
    executed" must never be counted as "passed" (P1-2).
    """
    if gate_scope not in GATE_SCOPES:
        raise ContinuousScoreError(f"unknown gate scope {gate_scope!r}; known: {list(GATE_SCOPES)}")
    base = scores[baseline]
    candidate = scores[enhance]
    acoustic_keys = (
        "duplicate_identities",
        "at_least_one_qualified_final",
        "cpcer_strictly_better",
        "overlap_cer_strictly_better",
        "rtf",
        "der_le_25",
        "overlap_recall_ge_70",
    )
    replay_keys = (
        "network_calls_zero",
        "memory_unchanged",
        "timeline_chunk_equals_finals",
        "partial_not_persisted",
    )
    if gate_scope == "smoke":
        enforced_acoustic_keys = tuple(k for k in acoustic_keys if k not in DIAGNOSTIC_ONLY_IN_SMOKE)
    else:
        enforced_acoustic_keys = acoustic_keys
    if replay is None:
        # Not evaluated => counted as failed for the full E2E claim.
        net_zero = mem_zero = chunk_eq = partial_zero = False
        replay_evaluated = False
    else:
        net_zero = replay.get("network_calls", 1) == 0
        mem_zero = replay.get("memory_before", 0) == 0 and replay.get("memory_after", 0) == 0
        # Prove the replay persisted exactly the qualified finals and nothing else.
        # The persisted chunk audio_event_ids must equal the qualified-final event
        # IDs, with no None/empty/duplicate IDs and exact count agreement between
        # the timeline chunk count and the replayed-final count (P1-6 / P3-2).
        persisted_raw = list(replay.get("persisted_audio_event_ids") or [])
        expected_ids = list(expected_final_event_ids or [])
        persisted_clean = [pid for pid in persisted_raw if pid and str(pid).strip()]
        persisted_set = set(persisted_clean)
        has_invalid = any(pid is None or not str(pid).strip() for pid in persisted_raw)
        has_duplicates = len(persisted_clean) != len(persisted_set)
        expected_set = set(expected_ids)
        count_match = (
            replay.get("capture_chunk_count") == replay.get("transcript_final_events") == len(expected_ids)
        )
        chunk_eq = (
            not has_invalid
            and not has_duplicates
            and persisted_set == expected_set
            and count_match
        )
        partial_zero = replay.get("partial_events", 1) == 0 and replay.get("ui_only_events", 1) == 0
        replay_evaluated = True
    der_value = diarization.get("der")
    recall_value = diarization.get("overlap_frame_recall")
    checks = {
        "duplicate_identities": bool(base["integrity"]["passed"] and candidate["integrity"]["passed"]),
        "at_least_one_qualified_final": int(candidate["qualified_finals"]) >= 1,
        "cpcer_strictly_better": (
            candidate["cpcer"]["cer"] is not None
            and base["cpcer"]["cer"] is not None
            and candidate["cpcer"]["cer"] < base["cpcer"]["cer"]
        ),
        "overlap_cer_strictly_better": (
            candidate["intervals"]["overlap"]["normalized_cer"] is not None
            and base["intervals"]["overlap"]["normalized_cer"] is not None
            and candidate["intervals"]["overlap"]["normalized_cer"]
            < base["intervals"]["overlap"]["normalized_cer"]
        ),
        "rtf": rtf <= 1.0,
        "network_calls_zero": net_zero,
        "memory_unchanged": mem_zero,
        "timeline_chunk_equals_finals": chunk_eq,
        "partial_not_persisted": partial_zero,
        # A missing metric is never a pass: in the full scope that fails the run
        # outright, and in the smoke scope it is reported as non-evaluable
        # instead of being silently counted as satisfied.
        "der_le_25": der_value is not None and float(der_value) <= 0.25,
        "overlap_recall_ge_70": recall_value is not None and float(recall_value) >= 0.70,
    }
    acoustic_passed = all(checks[k] for k in enforced_acoustic_keys)
    if replay is None:
        full_e2e_passed = False
    else:
        full_e2e_passed = acoustic_passed and all(checks[k] for k in replay_keys)
    diagnostics: dict[str, Any] = {}
    if gate_scope == "smoke":
        diagnostics = {
            "der": {
                "value": der_value,
                "threshold_max": 0.25,
                "meets": checks["der_le_25"],
                "evaluable": der_value is not None,
            },
            "overlap_frame_recall": {
                "value": recall_value,
                "threshold_min": 0.70,
                "meets": checks["overlap_recall_ge_70"],
                "evaluable": recall_value is not None,
            },
        }
    return {
        "baseline_variant": baseline,
        "enhance_variant": enhance,
        "gate_scope": gate_scope,
        "thresholds": {"rtf_max": 1.0, "der_max": 0.25, "overlap_recall_min": 0.70},
        "replay_evaluated": replay_evaluated,
        "enforced_checks": list(enforced_acoustic_keys + replay_keys),
        "diagnostic_checks": list(DIAGNOSTIC_ONLY_IN_SMOKE) if gate_scope == "smoke" else [],
        "acoustic_passed": acoustic_passed,
        "full_e2e_passed": full_e2e_passed,
        "checks": checks,
        "diagnostics": diagnostics,
        "passed": full_e2e_passed,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    scores = payload["variants"]
    lines = [
        "# 连续多通道端到端评分报告",
        "",
        f"- 会议：`{payload['case_id']}`；处理时长：{payload['processed_seconds']} 秒；变体：{', '.join(sorted(scores))}",
        f"- ASR：`{payload['asr_model_sha256']}`；Sortformer：`{payload['sortformer_model_sha256']}`",
        f"- 完整闭环（处理）RTF：{payload['full_loop_rtf']}（frontend {payload['frontend_rtf']} / scoring {payload['scoring_rtf']} / replay {payload['replay_rtf']}，口径：`{payload.get('rtf_basis', 'stage_sum')}`，不含进程启动/导入/输入校验/产物写出）",
    ]
    scope = payload.get("gate_scope", "full")
    lines.append(
        f"- 门禁范围（gate scope）：**{scope}**"
        + ("（75 秒 smoke：DER / 重叠帧召回仅作诊断，不计入通过）" if scope == "smoke" else "（完整单场：DER ≤25% 与重叠帧召回 ≥70% 强制生效）")
    )
    basis = payload.get("gate_scope_basis") or {}
    if basis:
        lines.append(
            "- 截短证据：processed_seconds `{proc}` vs 会议时长 `{meet}`（limit `{limit}`）→ truncated=`{trunc}`".format(
                proc=basis.get("processed_seconds"),
                meet=basis.get("meeting_seconds"),
                limit=basis.get("limit_seconds"),
                trunc=basis.get("truncated"),
            )
        )
    diagnostics = payload.get("gates", {}).get("diagnostics") or {}
    if diagnostics:
        der = diagnostics.get("der") or {}
        rec = diagnostics.get("overlap_frame_recall") or {}
        lines.append(
            "- 诊断项（不计入通过）：DER `{der}`（阈值 ≤0.25，{der_ok}）；重叠帧召回 `{rec}`（阈值 ≥0.70，{rec_ok}）".format(
                der="n/a" if der.get("value") is None else f"{float(der['value']):.4f}",
                rec="n/a" if rec.get("value") is None else f"{float(rec['value']):.4f}",
                der_ok="达标" if der.get("meets") else ("不可评估" if not der.get("evaluable") else "未达标"),
                rec_ok="达标" if rec.get("meets") else ("不可评估" if not rec.get("evaluable") else "未达标"),
            )
        )
    lines.extend([
        "",
        "## 事实（实测）",
        "",
        "| 变体 | cpCER | 区间 CER(全部) | 非重叠 CER | 重叠 CER | 合格 final | rejected | fallback |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for name in sorted(scores):
        item = scores[name]
        cp = item["cpcer"]["cer"]
        lines.append(
            "| {name} | {cp} | {all} | {non} | {ovl} | {ok} | {rej} | {fb} |".format(
                name=name,
                cp="n/a" if cp is None else f"{cp:.4f}",
                all=_fmt(item["intervals"]["all"]["normalized_cer"]),
                non=_fmt(item["intervals"]["non_overlap"]["normalized_cer"]),
                ovl=_fmt(item["intervals"]["overlap"]["normalized_cer"]),
                ok=item["qualified_finals"],
                rej=item["rejected_events"],
                fb=item["fallback_segments"],
            )
        )
    lines.extend([
        "",
        "## 诊断与边界（不得写成结论）",
        "",
        "- 本结果是**单场**离线连续软件链路 smoke，不代表全 8 场汇总，也不代表 Android 真机多麦验收。",
        "- overlap CER ≤30% 与非重叠 CER ≤11.5% 只在单场作为诊断显示；正式结论需要全 8 场聚合。",
        "- cpCER 是拼接级置换不变指标，不能与 75 秒窗口的 oracle 片段 CER 直接比较。",
        "- 概率来自 Sortformer 全会议运行；外部 `push()` 跨调用状态保持**未验证**。",
        "",
        "## 未验证项",
        "",
        "- 跨 block 的说话人身份连续性只继承了全会议概率列，未做增量推理验证。",
        "- 未做阵列几何校准；all8 为固定 8 通道顺序，非眼镜真实阵列。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-manifest", type=Path, required=True)
    parser.add_argument("--frontend-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay-service", action="store_true")
    parser.add_argument("--replay-timeout", type=float, default=30.0)
    parser.add_argument(
        "--gate-scope",
        choices=("auto", "smoke", "full"),
        default="auto",
        help="smoke = diarization gates are diagnostics only; full = enforced. "
        "auto follows the frontend run's proven truncation and falls back to full. "
        "Explicit smoke on a run that is not provably shorter than the meeting is refused.",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    frontend_run = args.frontend_run.resolve()
    manifest, rows = load_frontend_run(frontend_run)
    case_id = manifest["case"]["case_id"]
    processed_seconds = float(manifest["case"]["processed_seconds"])
    enhance_variant = str(manifest["config"]["enhance_variant"])
    baseline_variant = str(manifest["config"]["baseline_variant"])
    threshold = float(manifest["config"]["activity_threshold"])
    # Decided up-front and stamped into the artefacts so a full meeting cannot
    # quietly borrow the smoke scope to skip the diarization gates.
    gate_scope = resolve_gate_scope(manifest, args.gate_scope)
    gate_scope_basis = explain_gate_scope(manifest)

    prep_case = load_prep_case(args.prep_manifest.resolve(), case_id)
    drift = verify_frontend_against_prep(manifest, prep_case)
    if not drift["passed"]:
        raise ContinuousScoreError(f"input drift: {drift['problems']}")
    integrity = check_segment_rows(rows, processed_seconds)
    if not integrity["passed"]:
        raise ContinuousScoreError(f"segment identity check failed: {integrity}")

    profile, _ = load_source_run(args.source_run.resolve())
    if profile.get("asr_model_sha256") != ASR_MODEL_SHA256:
        raise ContinuousScoreError(
            f"ASR model drift: expected {ASR_MODEL_SHA256}, source run has {profile.get('asr_model_sha256')}"
        )
    asr = OfflineAsrBackend()
    capability = asr.capability()
    if capability.status != "ready":
        raise ContinuousScoreError(f"ASR is not ready: {capability.reason}")
    # Close the ASR lock: the event must declare the backend the scorer actually
    # loaded, and the model file on disk must match the locked hash (P1-2).
    asr_backend = asr.backend
    if asr_backend != "sherpa_sensevoice":
        raise ContinuousScoreError(
            f"ASR backend lock failed: expected sherpa_sensevoice, loaded {asr_backend}"
        )
    loaded_model_path = Path(str(asr.model_dir)) / "model.int8.onnx"
    if not loaded_model_path.is_file():
        raise ContinuousScoreError(f"ASR model file missing: {loaded_model_path}")
    loaded_model_hash = sha256_file(loaded_model_path)
    if loaded_model_hash != ASR_MODEL_SHA256:
        raise ContinuousScoreError(
            f"ASR model hash mismatch: expected {ASR_MODEL_SHA256}, loaded {loaded_model_hash}"
        )

    gold = load_gold_intervals(prep_case, processed_seconds)
    gold_by_speaker: dict[str, str] = {}
    for interval in sorted(gold, key=lambda item: (item.start_s, item.end_s)):
        gold_by_speaker[interval.speaker] = gold_by_speaker.get(interval.speaker, "") + interval.text

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # The frontend already created this directory; refuse to overwrite a scored run.
    for name in ("scores.json", "asr.jsonl", "audio-events.jsonl", "REPORT.md"):
        if (out_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite scored run: {out_dir / name}")

    variants = sorted({row["variant"] for row in rows})
    scores: dict[str, dict[str, Any]] = {}
    all_transcripts: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    asr_seconds = 0.0
    scoring_seconds = 0.0
    cpcer_seconds = 0.0

    for variant in variants:
        variant_rows = sorted(
            [row for row in rows if row["variant"] == variant], key=lambda row: (row["track"], row["start_s"])
        )
        audio_by_id = load_segment_audio(variant_rows, frontend_run)
        variant_started = time.perf_counter()
        transcripts = transcribe_segments(asr, variant_rows, audio_by_id, asr_sha=ASR_MODEL_SHA256)
        variant_asr_seconds = time.perf_counter() - variant_started
        asr_seconds += variant_asr_seconds
        all_transcripts.extend(transcripts)

        by_track: dict[str, list[dict[str, Any]]] = {}
        text_by_track: dict[str, str] = {}
        for row, transcript in zip(variant_rows, transcripts):
            by_track.setdefault(row["track"], []).append(row)
            text_by_track[row["track"]] = text_by_track.get(row["track"], "") + transcript["text"]

        cpcer_started = time.perf_counter()
        cpcer = concatenated_cer(gold_by_speaker, text_by_track)
        variant_cpcer_seconds = time.perf_counter() - cpcer_started
        cpcer_seconds += variant_cpcer_seconds
        interval_started = time.perf_counter()
        interval_results = score_intervals(
            asr,
            gold=gold,
            rows_by_track=by_track,
            audio_by_id=audio_by_id,
            mapping=cpcer["mapping"],
        )
        variant_interval_seconds = time.perf_counter() - interval_started
        scoring_seconds += variant_interval_seconds

        if variant == enhance_variant:
            for row, transcript in zip(variant_rows, transcripts):
                events.append(
                    build_audio_event(
                        row,
                        transcript["text"],
                        variant=variant,
                        asr_backend=asr_backend,
                        asr_sha=ASR_MODEL_SHA256,
                        threshold=threshold,
                    )
                )

        diarization = score_case(
            [{"speaker": item.speaker, "start_s": item.start_s, "end_s": item.end_s} for item in gold],
            [{"speaker": row["track"], "start_s": row["start_s"], "end_s": row["end_s"]} for row in variant_rows],
            processed_seconds,
        )
        scores[variant] = {
            "variant": variant,
            "channels": variant_rows[0]["channels"],
            "segments": len(variant_rows),
            "cpcer": cpcer,
            "intervals": {
                "all": aggregate(interval_results),
                "non_overlap": aggregate(row for row in interval_results if not row.overlap_exposed),
                "overlap": aggregate(row for row in interval_results if row.overlap_exposed),
            },
            "diarization": diarization,
            "integrity": integrity,
            "fallback_segments": sum(1 for row in variant_rows if row.get("fallback_reason")),
            "qualified_finals": sum(1 for item in events if item["type"] == "transcript_final") if variant == enhance_variant else 0,
            "rejected_events": sum(1 for item in events if item["type"] == "speech_rejected") if variant == enhance_variant else 0,
            "asr_seconds": round(variant_asr_seconds, 3),
            "interval_scoring_seconds": round(variant_interval_seconds, 3),
            "cpcer_seconds": round(variant_cpcer_seconds, 3),
        }

    expected_final_event_ids = {
        e["event_id"] for e in events if e.get("type") == "transcript_final"
    }
    scoring_wall_seconds = time.perf_counter() - started
    frontend_rtf = float(manifest.get("rtf") or 0.0)
    scoring_rtf = scoring_wall_seconds / processed_seconds if processed_seconds else 0.0
    replay_started = time.perf_counter()
    replay = replay_events(events, timeout=args.replay_timeout) if args.replay_service else None
    replay_seconds = time.perf_counter() - replay_started
    replay_rtf = replay_seconds / processed_seconds if processed_seconds else 0.0
    # Full loop is the sequential sum of the three stage RTFs. This is the
    # *processing* RTF: it deliberately excludes interpreter start-up, imports,
    # input verification and artefact writing. The wall-clock RTF that covers
    # the whole process is measured from outside by run_full_loop_timed.py and
    # reported separately as full_process_rtf.
    full_loop_rtf = frontend_rtf + scoring_rtf + replay_rtf
    gates = evaluate_gates(
        scores,
        baseline=baseline_variant,
        enhance=enhance_variant,
        rtf=full_loop_rtf,
        replay=replay,
        diarization=scores[enhance_variant]["diarization"],
        expected_final_event_ids=expected_final_event_ids,
        gate_scope=gate_scope,
    )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "case_id": case_id,
        "processed_seconds": processed_seconds,
        "gate_scope": gate_scope,
        "gate_scope_basis": gate_scope_basis,
        "activity_threshold": threshold,
        "asr_model_sha256": ASR_MODEL_SHA256,
        "sortformer_model_sha256": SORTFORMER_MODEL_SHA256,
        "frontend_manifest": str(frontend_run / "frontend-manifest.json"),
        "frontend_rtf": round(frontend_rtf, 6),
        "asr_seconds": round(asr_seconds, 3),
        "interval_scoring_seconds": round(scoring_seconds, 3),
        "cpcer_seconds": round(cpcer_seconds, 3),
        "scoring_wall_seconds": round(scoring_wall_seconds, 3),
        "scoring_rtf": round(scoring_rtf, 6),
        "replay_wall_seconds": round(replay_seconds, 3),
        "replay_rtf": round(replay_rtf, 6),
        "full_loop_rtf": round(full_loop_rtf, 6),
        # Which clock decided the "rtf" gate. "stage_sum" is the processing-only
        # sum of the three stages and excludes interpreter start-up, imports,
        # input verification and artefact writing. The outer wall-clock figure
        # that does cover those is measured by run_full_loop_timed.py and lands
        # in full-loop-timing.json; it stays None when the run was not wrapped.
        "rtf_basis": "stage_sum",
        "full_process_seconds": None,
        "full_process_rtf": None,
        "input_drift": drift,
        "segment_integrity": integrity,
        "gold_intervals": len(gold),
        "gold_speakers": sorted(gold_by_speaker),
        "variants": scores,
        "gates": gates,
        "service_replay": replay,
    }

    (out_dir / "asr.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_transcripts), encoding="utf-8"
    )
    (out_dir / "audio-events.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events), encoding="utf-8"
    )
    (out_dir / "scores.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if replay is not None:
        (out_dir / "service-replay.json").write_text(
            json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    write_report(out_dir / "REPORT.md", payload)
    print(json.dumps({
        "scores_json": str(out_dir / "scores.json"),
        "variants": {name: {"cpcer": scores[name]["cpcer"]["cer"]} for name in scores},
        "full_loop_rtf": payload["full_loop_rtf"],
        "gate_scope": gate_scope,
        "gates": gates["checks"],
        "enforced_checks": gates["enforced_checks"],
        "diagnostic_checks": gates["diagnostic_checks"],
        "gates_passed": gates["passed"],
        "service_replay": {
            "chunks": replay["capture_chunk_count"],
            "network_calls": replay["network_calls"],
            "memory_added": replay["memory_after"] - replay["memory_before"],
        }
        if replay
        else None,
    }, ensure_ascii=False))
    return 0 if gates["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContinuousScoreError as exc:
        # Fail closed with a readable message instead of a traceback; the exit
        # code stays non-zero so the wrapper never reads it as a pass.
        print(f"score_continuous_e2e: {exc}", file=sys.stderr)
        raise SystemExit(2)
