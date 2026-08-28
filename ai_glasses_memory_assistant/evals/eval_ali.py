"""Deterministic, local-only scoring primitives for Eval_Ali_far.

The official runner feeds selected WAV samples into the shared streaming audio
core. This is not an official M2MeT score and it never persists source PCM:
callers keep only frozen case metadata, reference text, and sanitized events.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "eval_ali.v1"
LICENSE = "AliMeeting (SLR119), CC BY-SA 4.0"
PRESET_SECONDS = {"smoke": 75.0, "regression": 225.0, "full": None}
SELECTION_STEP_SECONDS = 15.0
FRAME_SECONDS = 0.1


@dataclass(frozen=True)
class ReferenceInterval:
    speaker: str
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class EvalAliCase:
    case_id: str
    session_id: str
    audio_path: str
    textgrid_path: str
    audio_sha256: str
    textgrid_sha256: str
    source_duration_s: float
    start_s: float
    end_s: float
    reference_intervals: tuple[ReferenceInterval, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reference_intervals"] = [asdict(item) for item in self.reference_intervals]
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_textgrid(path: Path) -> list[ReferenceInterval]:
    """Parse the long text form emitted by Praat for the AliMeeting labels."""

    intervals: list[ReferenceInterval] = []
    tier = ""
    item_depth = False
    interval: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if re.fullmatch(r"item \[\d+\]:", line):
            item_depth = True
            tier = ""
            interval = None
            continue
        if item_depth and line.startswith("name = "):
            tier = _textgrid_string(line.removeprefix("name = "))
            continue
        if line.startswith("intervals ["):
            interval = {"speaker": tier}
            continue
        if interval is None:
            continue
        if line.startswith("xmin = "):
            interval["start_s"] = float(line.removeprefix("xmin = "))
        elif line.startswith("xmax = "):
            interval["end_s"] = float(line.removeprefix("xmax = "))
        elif line.startswith("text = "):
            interval["text"] = _textgrid_string(line.removeprefix("text = "))
            if interval.get("speaker") and "start_s" in interval and "end_s" in interval:
                intervals.append(ReferenceInterval(**interval))
            interval = None
    if not intervals:
        raise ValueError(f"no intervals found in TextGrid: {path}")
    return intervals


def _textgrid_string(value: str) -> str:
    value = value.strip()
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        raise ValueError(f"invalid TextGrid string: {value[:80]}")
    return value[1:-1].replace('""', '"')


def wav_duration_seconds(path: Path) -> float:
    # AliMeeting uses WAVE_FORMAT_EXTENSIBLE, which Python's stdlib wave
    # reader intentionally rejects on Python 3.11. soundfile is already a
    # project dependency and exposes the actual channel/sample metadata.
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate != 16_000:
        raise ValueError(f"Eval_Ali_far must be 16 kHz: {path}")
    if info.channels != 8:
        raise ValueError(f"Eval_Ali_far must be 8-channel: {path}")
    return info.frames / float(info.samplerate)


def discover_cases(root: Path, preset: str) -> list[EvalAliCase]:
    if preset not in PRESET_SECONDS:
        raise ValueError(f"unsupported preset: {preset}")
    far = root / "Eval_Ali_far"
    audio_dir = far / "audio_dir"
    textgrid_dir = far / "textgrid_dir"
    if not audio_dir.is_dir() or not textgrid_dir.is_dir():
        raise ValueError("expected Eval_Ali_far/audio_dir and textgrid_dir")
    cases: list[EvalAliCase] = []
    for textgrid_path in sorted(textgrid_dir.glob("*.TextGrid")):
        session_id = textgrid_path.stem
        matches = sorted(audio_dir.glob(f"{session_id}_*.wav"))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one far WAV for {session_id}, found {len(matches)}")
        audio_path = matches[0]
        duration_s = wav_duration_seconds(audio_path)
        intervals = tuple(item for item in parse_textgrid(textgrid_path) if item.text.strip())
        start_s, end_s = select_case_window(intervals, duration_s, PRESET_SECONDS[preset])
        selected = tuple(
            item for item in intervals if item.end_s > start_s and item.start_s < end_s
        )
        cases.append(EvalAliCase(
            case_id=f"{session_id}-{preset}",
            session_id=session_id,
            audio_path=str(audio_path.resolve()),
            textgrid_path=str(textgrid_path.resolve()),
            audio_sha256=sha256_file(audio_path),
            textgrid_sha256=sha256_file(textgrid_path),
            source_duration_s=duration_s,
            start_s=start_s,
            end_s=end_s,
            reference_intervals=selected,
        ))
    if len(cases) != 8:
        raise ValueError(f"expected 8 Eval_Ali_far sessions, found {len(cases)}")
    return cases


def select_case_window(
    intervals: Iterable[ReferenceInterval], duration_s: float, window_s: float | None,
) -> tuple[float, float]:
    if window_s is None or window_s >= duration_s:
        return 0.0, duration_s
    if window_s <= 0:
        raise ValueError("window duration must be positive")
    materialized = tuple(intervals)
    best: tuple[float, float] | None = None
    last_start = max(0.0, duration_s - window_s)
    candidates = [round(index * SELECTION_STEP_SECONDS, 6) for index in range(int(last_start // SELECTION_STEP_SECONDS) + 1)]
    if not candidates or candidates[-1] != last_start:
        candidates.append(last_start)
    for start_s in candidates:
        end_s = min(duration_s, start_s + window_s)
        score = _window_score(materialized, start_s, end_s)
        candidate = (score, -start_s)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    start_s = -best[1]
    return start_s, min(duration_s, start_s + window_s)


def _window_score(intervals: tuple[ReferenceInterval, ...], start_s: float, end_s: float) -> float:
    frames = _frame_count(start_s, end_s)
    speech = 0
    overlap = 0
    for index in range(frames):
        point = start_s + (index + 0.5) * FRAME_SECONDS
        active = sum(item.start_s <= point < item.end_s for item in intervals)
        speech += active > 0
        overlap += active > 1
    return speech * FRAME_SECONDS + overlap * FRAME_SECONDS * 0.5


def build_manifest(root: Path, preset: str) -> dict[str, Any]:
    cases = discover_cases(root, preset)
    return {
        "schema": SCHEMA_VERSION,
        "kind": "eval_ali_far_selection",
        "license": LICENSE,
        "preset": preset,
        "channel": 0,
        "selection_step_seconds": SELECTION_STEP_SECONDS,
        "cases": [item.to_dict() for item in cases],
    }


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(char for char in normalized if char.isalnum())


def edit_distance(reference: str, hypothesis: str) -> dict[str, int | float]:
    previous = [(index, 0, index, 0) for index in range(len(hypothesis) + 1)]
    for ref_index, reference_char in enumerate(reference, start=1):
        current = [(ref_index, 0, 0, ref_index)]
        for hyp_index, hypothesis_char in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1]
            deletion = previous[hyp_index]
            insertion = current[hyp_index - 1]
            choices = [
                (substitution[0] + (reference_char != hypothesis_char), substitution[1] + (reference_char != hypothesis_char), substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            current.append(min(choices, key=lambda item: (item[0], item[1], item[2], item[3])))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_chars": len(reference),
        "cer": errors / len(reference) if reference else (0.0 if not hypothesis else 1.0),
    }


def score_cer(reference: str, hypothesis: str) -> dict[str, Any]:
    return {
        "strict": edit_distance(reference, hypothesis),
        "normalized": edit_distance(normalize_text(reference), normalize_text(hypothesis)),
    }


def score_vad(
    reference: Iterable[ReferenceInterval], detected: Iterable[tuple[float, float]], *, start_s: float, end_s: float,
) -> dict[str, float | int]:
    reference_items = tuple(reference)
    detected_items = tuple(detected)
    tp = fp = fn = 0
    for index in range(_frame_count(start_s, end_s)):
        point = start_s + (index + 0.5) * FRAME_SECONDS
        expected = any(item.start_s <= point < item.end_s for item in reference_items)
        actual = any(start <= point < end for start, end in detected_items)
        if expected and actual:
            tp += 1
        elif actual:
            fp += 1
        elif expected:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "frame_seconds": FRAME_SECONDS,
        "true_positive_frames": tp,
        "false_positive_frames": fp,
        "false_negative_frames": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "missed_seconds": fn * FRAME_SECONDS,
        "false_trigger_seconds": fp * FRAME_SECONDS,
    }


def score_der(
    reference: Iterable[ReferenceInterval], events: Iterable[dict[str, Any]], *, start_s: float, end_s: float,
    collar_s: float = 0.25,
) -> dict[str, float | int | dict[str, str]]:
    reference_items = tuple(reference)
    predicted = tuple(events)
    references = sorted({item.speaker for item in reference_items})
    groups = sorted({str(item.get("voice_group") or "") for item in predicted if item.get("voice_group")})
    frame_rows: list[tuple[str, str]] = []
    overlap_frames = overlap_missed = 0
    for index in range(_frame_count(start_s, end_s)):
        point = start_s + (index + 0.5) * FRAME_SECONDS
        raw_active = [item.speaker for item in reference_items if item.start_s <= point < item.end_s]
        active = [item.speaker for item in reference_items if _inside_with_collar(item, point, collar_s)]
        event = next((item for item in predicted if float(item["start_s"]) <= point < float(item["end_s"])), None)
        group = str(event.get("voice_group") or "") if event else ""
        if len(raw_active) > 1:
            overlap_frames += 1
            overlap_missed += not bool(group)
        if not active:
            continue
        for speaker in active:
            frame_rows.append((speaker, group))
    mapping = _best_mapping(references, groups, frame_rows)
    misses = false_alarms = confusions = 0
    for speaker, group in frame_rows:
        if not group:
            misses += 1
        elif mapping.get(group) != speaker:
            confusions += 1
    for index in range(_frame_count(start_s, end_s)):
        point = start_s + (index + 0.5) * FRAME_SECONDS
        has_reference = any(_inside_with_collar(item, point, collar_s) for item in reference_items)
        has_prediction = any(float(item["start_s"]) <= point < float(item["end_s"]) and item.get("voice_group") for item in predicted)
        false_alarms += has_prediction and not has_reference
    denominator = len(frame_rows)
    return {
        "collar_seconds": collar_s,
        "reference_speaker_frames": denominator,
        "miss_frames": misses,
        "false_alarm_frames": false_alarms,
        "confusion_frames": confusions,
        "der": (misses + false_alarms + confusions) / denominator if denominator else 0.0,
        "overlap_frames": overlap_frames,
        "overlap_miss_rate": overlap_missed / overlap_frames if overlap_frames else 0.0,
        "mapping": mapping,
    }


def _inside_with_collar(item: ReferenceInterval, point: float, collar_s: float) -> bool:
    return item.start_s + collar_s <= point < item.end_s - collar_s


def _best_mapping(references: list[str], groups: list[str], rows: list[tuple[str, str]]) -> dict[str, str]:
    if not references or not groups:
        return {}
    score = {(group, speaker): sum(row_speaker == speaker and row_group == group for row_speaker, row_group in rows)
             for group in groups for speaker in references}
    best_total = -1
    best: dict[str, str] = {}
    for speakers in itertools.permutations(references, min(len(groups), len(references))):
        mapping = dict(zip(groups, speakers))
        total = sum(score[(group, speaker)] for group, speaker in mapping.items())
        if total > best_total:
            best_total, best = total, mapping
    return best


def score_events(case: EvalAliCase, events: Iterable[dict[str, Any]], *, playback_offset_ms: float) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for record in events:
        event = dict(record.get("event") or record)
        start_s = (float(event.get("start_ms") or 0.0) - playback_offset_ms) / 1000.0 + case.start_s
        end_s = (float(event.get("end_ms") or 0.0) - playback_offset_ms) / 1000.0 + case.start_s
        if end_s <= case.start_s or start_s >= case.end_s:
            continue
        normalized.append({
            "event_id": str(event.get("event_id") or record.get("event_id") or ""),
            "start_s": max(case.start_s, start_s),
            "end_s": min(case.end_s, end_s),
            "text": str(event.get("text") or ""),
            "type": str(event.get("type") or ""),
            "voice_group": str((event.get("speaker") or {}).get("voice_group") or ""),
            "overlap": str((event.get("overlap") or {}).get("state") or ""),
            "status": str(record.get("status") or ""),
            "completed_at": record.get("completed_at"),
        })
    normalized.sort(key=lambda item: (item["start_s"], item["end_s"], item["event_id"]))
    reference_text = "".join(item.text for item in sorted(case.reference_intervals, key=lambda item: (item.start_s, item.end_s, item.speaker)))
    hypothesis = "".join(item["text"] for item in normalized if item["type"] == "transcript_final")
    detected = [(item["start_s"], item["end_s"]) for item in normalized]
    return {
        "schema": SCHEMA_VERSION,
        "case_id": case.case_id,
        "reference_text": reference_text,
        "hypothesis_text": hypothesis,
        "event_count": len(normalized),
        "rejected_event_count": sum(item["type"] == "speech_rejected" for item in normalized),
        "cer": score_cer(reference_text, hypothesis),
        "vad": score_vad(case.reference_intervals, detected, start_s=case.start_s, end_s=case.end_s),
        "speaker": score_der(case.reference_intervals, normalized, start_s=case.start_s, end_s=case.end_s),
        "events": normalized,
    }


def diagnose_case(score: dict[str, Any], *, preflight_ok: bool, queue_failed: int, memory_saved: int) -> dict[str, Any]:
    if not preflight_ok:
        stage, reason = "preflight", "route_or_model_not_ready"
    elif queue_failed:
        stage, reason = "event_queue", "device_audio_event_failed"
    elif memory_saved:
        stage, reason = "memory_safety", "environment_audio_saved_to_long_term_memory"
    elif not score["event_count"]:
        stage, reason = "vad", "no_final_event_for_reference_speech"
    elif score["vad"]["recall"] < 0.5:
        stage, reason = "vad", "reference_speech_mostly_not_detected"
    elif score["cer"]["normalized"]["cer"] > 0.5:
        stage, reason = "asr", "high_normalized_cer"
    else:
        stage, reason = "none", "no_primary_failure_detected"
    return {"primary_stage": stage, "reason": reason}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _frame_count(start_s: float, end_s: float) -> int:
    return max(0, int(math.ceil((end_s - start_s) / FRAME_SECONDS)))
