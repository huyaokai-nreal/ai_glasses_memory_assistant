#!/usr/bin/env python3
"""Continuous multichannel frontend for one AliMeeting session.

This runner turns a full-meeting Sortformer probability matrix plus the matching
8-channel WAV into per-anonymous-track enhanced audio segments. It is an offline
evaluation pipeline: it proves the *software* chain (block watermark -> activity
-> MVDR -> segments) works on a continuous meeting. It does not prove on-device
multichannel capture and it does not validate Sortformer state across external
``push()`` calls.

Gold isolation (hard rule): this runner never reads TextGrid text, gold speaker
mapping or gold segmentation. Only probability columns, the audio itself and the
whitelisted prep-manifest fields are consumed.

Streaming model:
    * The Sortformer probability matrix comes from one full-meeting NeMo run, so
      column k is the same anonymous track for the whole session (the model's
      internal speaker cache owns that continuity). ``external_push_state_validated``
      stays false because nothing here re-drives the model incrementally.
    * Audio is read in core blocks of 30 s plus 3.2 s of right context. Every
      STFT frame is owned by exactly one core block; the final block flushes.

Raw PCM is written only under the git-ignored output directory.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from diarize_sortformer import (  # noqa: E402
    FRAME_SECONDS,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SHA256,
    NEMO_COMMIT,
    sha256_file,
)
from enhance_mvdr_oracle import frame_centers, istft, mvdr_filter, stft, N_FFT, HOP  # noqa: E402
from prepare_ali_multichannel import CHANNEL_VARIANTS, SAMPLE_RATE  # noqa: E402

SCHEMA = "eval_ali_continuous_frontend.v1"
PREP_SCHEMA = "eval_ali_continuous_prep.v1"
SORTFORMER_SCHEMA = "eval_ali_sortformer_predictions.v1"

CORE_BLOCK_SECONDS = 30.0
# 40 Sortformer frames x 80 ms: the model's own right context. Here it only
# provides audio look-ahead for the beamformer STFT; those frames are never owned.
RIGHT_CONTEXT_SECONDS = 3.2
REG = 1e-6

# Fields the frontend is allowed to consume from the prep manifest. Everything
# else (textgrid_path, diarization_reference_intervals) is gold and stays out.
CASE_FIELD_WHITELIST: tuple[str, ...] = (
    "case_id",
    "session_id",
    "cut_wav_path",
    "cut_wav_sha256",
    "window_start_s",
    "window_end_s",
)
GOLD_FORBIDDEN_FIELDS: tuple[str, ...] = (
    "textgrid_path",
    "textgrid_sha256",
    "diarization_reference_intervals",
    "reference_intervals",
    "segments",
)

VARIANT_CHANNELS: dict[str, list[int]] = {item["name"]: list(item["channels"]) for item in CHANNEL_VARIANTS}


class FrontendInputError(Exception):
    """Raised when required inputs are missing, drifted or inconsistent."""


@dataclass(frozen=True)
class Block:
    """One core block plus its audio read window."""

    index: int
    core_start_s: float
    core_end_s: float
    read_end_s: float


def plan_blocks(
    duration_s: float,
    *,
    core_seconds: float = CORE_BLOCK_SECONDS,
    right_context_seconds: float = RIGHT_CONTEXT_SECONDS,
) -> list[Block]:
    """Advance the watermark so every frame is owned by exactly one block."""
    if duration_s <= 0:
        raise ValueError(f"duration must be positive, got {duration_s}")
    if core_seconds <= 0 or right_context_seconds < 0:
        raise ValueError("core_seconds must be positive and right context non-negative")
    blocks: list[Block] = []
    index = 0
    start = 0.0
    while start < duration_s - 1e-9:
        core_end = min(duration_s, start + core_seconds)
        # The last block has no future audio to peek at; it must flush.
        read_end = min(duration_s, core_end + right_context_seconds)
        blocks.append(Block(index, start, core_end, read_end))
        index += 1
        start = core_end
    return blocks


def probability_frame_count(duration_s: float) -> int:
    """Number of 80 ms Sortformer frames covering ``duration_s``."""
    return max(0, int(np.ceil(duration_s / FRAME_SECONDS - 1e-9)))


def activity_mask(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold the probability columns into a (frames, tracks) boolean mask."""
    array = np.asarray(probabilities)
    if array.ndim != 2:
        raise ValueError(f"expected a (frames, tracks) probability matrix, got {array.shape}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"activity threshold must be within [0, 1], got {threshold}")
    return array >= float(threshold)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index runs of True in a 1-D boolean array."""
    flags = np.asarray(mask, dtype=np.int8)
    padded = np.concatenate((np.zeros(1, dtype=np.int8), flags, np.zeros(1, dtype=np.int8)))
    delta = np.diff(padded)
    starts = np.where(delta == 1)[0]
    ends = np.where(delta == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends) if int(end) > int(start)]


def plan_block_segments(
    activity: np.ndarray,
    frame_lo: int,
    frame_hi: int,
    duration_s: float,
) -> list[tuple[int, float, float]]:
    """Owned activity runs -> (track index, start_s, end_s).

    A run touching a block boundary is cut there, so a long utterance spanning
    several blocks yields one deterministic segment per block (<= 30 s each).
    """
    segments: list[tuple[int, float, float]] = []
    for track in range(activity.shape[1]):
        window = activity[frame_lo:frame_hi, track]
        for start, end in contiguous_runs(window):
            start_s = (frame_lo + start) * FRAME_SECONDS
            end_s = min(duration_s, (frame_lo + end) * FRAME_SECONDS)
            if end_s <= start_s + 1e-9:
                continue
            segments.append((track, float(start_s), float(end_s)))
    segments.sort(key=lambda item: (item[1], item[2], item[0]))
    return segments


def covariance_sum(Zxx: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Unnormalized per-frequency covariance sum over the selected frames."""
    selected = Zxx[:, frames, :]              # (C, M, F)
    if selected.shape[1] == 0:
        raise ValueError("covariance_sum requires at least one frame")
    reordered = selected.transpose(2, 0, 1)   # (F, C, M)
    return np.einsum("fcm,fdm->fcd", reordered, reordered.conj())


def max_energy_channel(Zxx: np.ndarray, frames: np.ndarray) -> int:
    """Crude but stable fallback: the channel with the most segment energy."""
    if frames.sum() == 0:
        return 0
    selected = Zxx[:, frames, :]
    power = np.sum(np.abs(selected) ** 2, axis=(1, 2))
    return int(np.argmax(power))


def segment_id(session_id: str, track: str, start_s: float, end_s: float, variant: str) -> str:
    """Deterministic identity: same inputs always produce the same ID."""
    start_ms = int(round(start_s * 1000))
    end_ms = int(round(end_s * 1000))
    return f"{session_id}__{track}__{start_ms:09d}-{end_ms:09d}__{variant}"


def track_name(track_index: int) -> str:
    """Capture-local anonymous label. Never a real identity."""
    return f"spk_{track_index + 1:02d}"


def select_case_fields(case: dict[str, Any]) -> dict[str, Any]:
    """Whitelist prep-manifest fields so gold data cannot leak into inference."""
    missing = [field for field in CASE_FIELD_WHITELIST if field not in case]
    if missing:
        raise FrontendInputError(f"prep manifest case is missing fields: {missing}")
    leaked = [field for field in GOLD_FORBIDDEN_FIELDS if field in case and field in CASE_FIELD_WHITELIST]
    if leaked:
        raise FrontendInputError(f"gold fields must never be whitelisted: {leaked}")
    return {field: case[field] for field in CASE_FIELD_WHITELIST}


def load_case(prep_manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    schema = prep_manifest.get("schema")
    if schema != PREP_SCHEMA:
        raise FrontendInputError(f"unexpected prep schema: {schema}")
    for case in prep_manifest.get("cases") or []:
        if case.get("case_id") == case_id:
            return select_case_fields(case)
    available = [case.get("case_id") for case in prep_manifest.get("cases") or []]
    raise FrontendInputError(f"case {case_id!r} is absent from the prep manifest (available: {available})")


def load_probabilities(run_dir: Path, case_id: str, diar_variant: str, threshold: float) -> dict[str, Any]:
    """Load and verify the pinned Sortformer probability matrix for one case."""
    run_manifest_path = run_dir / "run-manifest.json"
    if not run_manifest_path.is_file():
        raise FrontendInputError(f"missing Sortformer run manifest: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("schema") != SORTFORMER_SCHEMA:
        raise FrontendInputError(f"unexpected Sortformer schema: {run_manifest.get('schema')}")

    model = run_manifest.get("model") or {}
    if model.get("sha256") != MODEL_SHA256:
        raise FrontendInputError(
            f"Sortformer model drift: expected {MODEL_SHA256}, run recorded {model.get('sha256')}"
        )
    if (run_manifest.get("runtime") or {}).get("nemo_commit") != NEMO_COMMIT:
        raise FrontendInputError("NeMo commit drift between the pinned value and the Sortformer run")

    run_threshold = run_manifest.get("postprocessing_threshold")
    if run_threshold is None or abs(float(run_threshold) - float(threshold)) > 1e-9:
        raise FrontendInputError(
            f"threshold mismatch: run manifest used {run_threshold}, runner was asked for {threshold}"
        )

    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FrontendInputError(f"missing predictions: {predictions_path}")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in rows if row.get("case_id") == case_id and row.get("input_variant") == diar_variant]
    if len(matches) != 1:
        raise FrontendInputError(
            f"expected exactly one prediction row for {case_id}/{diar_variant}, got {len(matches)}"
        )
    row = matches[0]

    probabilities_path = Path(row["probabilities_path"])
    if not probabilities_path.is_file():
        raise FrontendInputError(f"missing probability file: {probabilities_path}")
    actual = sha256_file(probabilities_path)
    if actual != row.get("probabilities_sha256"):
        raise FrontendInputError(
            f"probability drift for {case_id}/{diar_variant}: {actual} != {row.get('probabilities_sha256')}"
        )
    probabilities = np.load(probabilities_path)
    if probabilities.ndim != 2:
        raise FrontendInputError(f"unexpected probability shape: {probabilities.shape}")
    return {
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "probabilities_path": probabilities_path,
        "probabilities_sha256": actual,
        "run_manifest": run_manifest,
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "prediction_row": row,
        "emitted_segment_count": len(row.get("segments") or []),
    }


def check_segment_integrity(rows: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    """Duplicate ids, same-track time overlaps and out-of-range intervals."""
    seen: set[str] = set()
    duplicate_ids = 0
    out_of_range = 0
    overlaps = 0
    per_track: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        if row["segment_id"] in seen:
            duplicate_ids += 1
        seen.add(row["segment_id"])
        if row["start_s"] < -1e-9 or row["end_s"] > duration_s + 1e-9 or row["end_s"] <= row["start_s"]:
            out_of_range += 1
        per_track.setdefault((row["variant"], row["track"]), []).append((row["start_s"], row["end_s"]))
    for spans in per_track.values():
        ordered = sorted(spans)
        for (_, previous_end), (next_start, _) in zip(ordered, ordered[1:]):
            if next_start < previous_end - 1e-9:
                overlaps += 1
    return {
        "duplicate_segment_ids": duplicate_ids,
        "same_track_overlaps": overlaps,
        "out_of_range_intervals": out_of_range,
        "total_segments": len(rows),
        "passed": duplicate_ids == 0 and overlaps == 0 and out_of_range == 0,
    }


def _frame_bounds(block: Block, duration_s: float, valid_frames: int) -> tuple[int, int]:
    frame_lo = int(np.floor(block.core_start_s / FRAME_SECONDS + 1e-9))
    frame_hi = min(valid_frames, int(np.ceil(block.core_end_s / FRAME_SECONDS - 1e-9)))
    return max(0, frame_lo), max(0, frame_hi)


def process_variant(
    *,
    variant: str,
    channels: list[int],
    wav_path: Path,
    activity: np.ndarray,
    probabilities: np.ndarray,
    blocks: list[Block],
    duration_s: float,
    valid_frames: int,
    out_dir: Path,
    session_id: str,
    case_id: str,
    reg: float = REG,
) -> dict[str, Any]:
    """Stream one channel variant and write its per-track segments."""
    channels = list(channels)
    n_tracks = activity.shape[1]
    variant_dir = out_dir / "segments" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    cov_target: dict[int, np.ndarray | None] = {track: None for track in range(n_tracks)}
    cov_noise: dict[int, np.ndarray | None] = {track: None for track in range(n_tracks)}
    count_target: dict[int, int] = {track: 0 for track in range(n_tracks)}
    count_noise: dict[int, int] = {track: 0 for track in range(n_tracks)}

    rows: list[dict[str, Any]] = []
    fallback_reasons: dict[str, int] = {}
    beamform_seconds = 0.0
    read_seconds = 0.0
    write_seconds = 0.0
    max_pcm_samples = 0
    max_stft_values = 0
    audio_seconds = 0.0

    with sf.SoundFile(str(wav_path)) as source:
        if source.samplerate != SAMPLE_RATE:
            raise FrontendInputError(f"{wav_path}: expected {SAMPLE_RATE} Hz, got {source.samplerate}")
        if max(channels) >= source.channels:
            raise FrontendInputError(f"{wav_path}: channels {channels} exceed {source.channels}")
        for block in blocks:
            read_start = int(round(block.core_start_s * SAMPLE_RATE))
            read_frames = int(round((block.read_end_s - block.core_start_s) * SAMPLE_RATE))
            if read_frames <= 0:
                continue
            read_t0 = time.perf_counter()
            source.seek(read_start)
            raw = source.read(read_frames, dtype="float32", always_2d=True)
            read_seconds += time.perf_counter() - read_t0
            max_pcm_samples = max(max_pcm_samples, int(raw.shape[0]) * int(raw.shape[1]))
            block_audio = raw[:, channels].T                      # (C, N)
            Zxx = stft(block_audio)                               # (C, T, F)
            max_stft_values = max(max_stft_values, int(Zxx.size))
            n_channels = Zxx.shape[0]
            centers = frame_centers(block_audio.shape[1])
            global_times = block.core_start_s + centers
            prob_idx = np.floor(global_times / FRAME_SECONDS).astype(np.int64)
            frame_lo, frame_hi = _frame_bounds(block, duration_s, valid_frames)
            owned = (prob_idx >= frame_lo) & (prob_idx < frame_hi)
            if not owned.any():
                continue
            owned_activity = activity[np.clip(prob_idx[owned], 0, activity.shape[0] - 1)]  # (M, K)

            beam_t0 = time.perf_counter()
            # Causal covariance update: only owned frames, so nothing is counted twice.
            for track in range(n_tracks):
                track_active = owned_activity[:, track]
                # "Solo" must mean the target speaks while no OTHER track is
                # active. Using "any active and not the target" would fold
                # overlap frames into the target covariance.
                others_active = (
                    owned_activity.astype(np.int16).sum(axis=1) - track_active.astype(np.int16)
                ) > 0
                solo = track_active & ~others_active
                noise = ~track_active
                if solo.any():
                    contribution = covariance_sum(Zxx[:, owned, :], solo)
                    cov_target[track] = contribution if cov_target[track] is None else cov_target[track] + contribution
                    count_target[track] += int(solo.sum())
                if noise.any():
                    contribution = covariance_sum(Zxx[:, owned, :], noise)
                    cov_noise[track] = contribution if cov_noise[track] is None else cov_noise[track] + contribution
                    count_noise[track] += int(noise.sum())

            planned = plan_block_segments(activity, frame_lo, frame_hi, duration_s)
            for track, start_s, end_s in planned:
                track_label = track_name(track)
                in_segment = owned & (global_times >= start_s - 1e-6) & (global_times < end_s - 1e-6)
                if not in_segment.any():
                    continue
                selected_frames = np.where(in_segment)[0]
                first_center = float(global_times[selected_frames[0]])
                # Map the segment back onto the probability grid for confidence/overlap.
                seg_lo = int(np.floor(start_s / FRAME_SECONDS + 1e-9))
                seg_hi = min(activity.shape[0], int(np.ceil(end_s / FRAME_SECONDS - 1e-9)))
                seg_probabilities = probabilities[seg_lo:seg_hi, track]
                confidence = float(np.mean(seg_probabilities)) if seg_probabilities.size else 0.0
                overlapping = bool(np.any(activity[seg_lo:seg_hi].sum(axis=1) > 1)) if seg_hi > seg_lo else False

                reason: str | None = None
                if n_channels == 1:
                    enhancement = "pass_through"
                    spectrum = Zxx[0, in_segment, :].T
                elif (
                    cov_target[track] is None
                    or cov_noise[track] is None
                    or count_target[track] < n_channels
                    or count_noise[track] < n_channels
                ):
                    enhancement = "fallback_max_energy_channel"
                    if cov_target[track] is None or count_target[track] < n_channels:
                        reason = "insufficient_target_solo_frames"
                    else:
                        reason = "insufficient_non_target_frames"
                    fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
                    channel = max_energy_channel(Zxx, in_segment)
                    spectrum = Zxx[channel, in_segment, :].T
                else:
                    enhancement = "mvdr"
                    target_cov = cov_target[track] / float(count_target[track])
                    noise_cov = cov_noise[track] / float(count_noise[track])
                    weights = mvdr_filter(target_cov, noise_cov, reg=reg)
                    selected = Zxx[:, in_segment, :].transpose(2, 0, 1)   # (F, C, N)
                    spectrum = np.einsum("fc,fct->ft", weights.conj(), selected)

                write_t0 = time.perf_counter()
                waveform = istft(spectrum.T)
                identifier = segment_id(session_id, track_label, start_s, end_s, variant)
                wav_path_out = variant_dir / f"{identifier}.wav"
                # The production write path: align to the fixed timeline and force the
                # on-disk frame count to exactly round((end_s-start_s)*SAMPLE_RATE).
                # Tested directly in tests/test_audio_frontend_continuous_frontend.py (P3-4).
                sample_count, wav_sha256 = write_segment_wav(
                    waveform, start_s=start_s, end_s=end_s, first_center=first_center, wav_path_out=wav_path_out
                )
                write_seconds += time.perf_counter() - write_t0
                audio_seconds += sample_count / SAMPLE_RATE
                rows.append({
                    "segment_id": identifier,
                    "case_id": case_id,
                    "session_id": session_id,
                    "variant": variant,
                    "channels": channels,
                    "track": track_label,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "duration_s": round(end_s - start_s, 3),
                    "confidence": round(confidence, 6),
                    "overlap": overlapping,
                    "block_index": block.index,
                    "enhancement": enhancement,
                    "fallback_reason": reason,
                    "sample_count": sample_count,
                    "wav_path": str(wav_path_out.relative_to(out_dir)),
                    "wav_sha256": wav_sha256,
                })
            beamform_seconds += time.perf_counter() - beam_t0

    if not rows:
        raise FrontendInputError(f"variant {variant}: produced no segments; refusing to write an empty run")
    return {
        "rows": rows,
        "stats": {
            "channels": channels,
            "segments": len(rows),
            "audio_seconds": round(audio_seconds, 3),
            "fallback_segments": sum(1 for row in rows if row["fallback_reason"]),
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "beamform_seconds": round(beamform_seconds, 3),
            "read_seconds": round(read_seconds, 3),
            "write_seconds": round(write_seconds, 3),
            "max_pcm_buffer_samples": max_pcm_samples,
            "max_stft_values": max_stft_values,
        },
    }


def write_segment_wav(
    waveform: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    first_center: float,
    wav_path_out: Path,
) -> tuple[int, str]:
    """Write one segment WAV aligned to the fixed timeline.

    The on-disk frame count is forced to exactly round((end_s - start_s) * SAMPLE_RATE):
    the ISTFT output is sliced to the declared interval, then padded with silence or
    truncated so downstream consumers (scorer, timeline) see a precise, gap-free sample
    count (P2-1 / P3-4). Returns (sample_count, wav_sha256).
    """
    target_samples = int(round((end_s - start_s) * SAMPLE_RATE))
    local0_time = first_center - N_FFT / (2.0 * SAMPLE_RATE)
    start_sample = int(round((start_s - local0_time) * SAMPLE_RATE))
    end_sample = int(round((end_s - local0_time) * SAMPLE_RATE))
    start_sample = max(0, min(start_sample, len(waveform)))
    end_sample = max(start_sample, min(end_sample, len(waveform)))
    waveform = waveform[start_sample:end_sample]
    if len(waveform) < target_samples:
        waveform = np.concatenate([waveform, np.zeros(target_samples - len(waveform), dtype=waveform.dtype)])
    elif len(waveform) > target_samples:
        waveform = waveform[:target_samples]
    sf.write(str(wav_path_out), waveform, SAMPLE_RATE, subtype="PCM_16")
    return int(len(waveform)), sha256_file(wav_path_out)


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    segments_path = out_dir / "segments.jsonl"
    ordered = sorted(rows, key=lambda row: (row["variant"], row["track"], row["start_s"]))
    segments_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8"
    )
    # Lock the segment file with its own content hash so the scorer can fail-closed
    # if segments.jsonl is edited or swapped after the frontend wrote it (P1-5).
    manifest["segments_jsonl_sha256"] = sha256_file(segments_path)
    (out_dir / "frontend-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    prep_dir = args.prep_dir.resolve()
    prep_path = prep_dir / "prep-manifest.json"
    if not prep_path.is_file():
        raise FrontendInputError(f"missing prep manifest: {prep_path}")
    prep_manifest = json.loads(prep_path.read_text(encoding="utf-8"))
    case = load_case(prep_manifest, args.case_id)

    out_dir = args.out.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite frontend run: {out_dir}")
    out_dir.mkdir(parents=True)

    wav_path = Path(case["cut_wav_path"])
    if not wav_path.is_file():
        raise FrontendInputError(f"missing input WAV: {wav_path}")
    actual_wav_sha = sha256_file(wav_path)
    if actual_wav_sha != case["cut_wav_sha256"]:
        raise FrontendInputError(
            f"input WAV drift for {case['case_id']}: {actual_wav_sha} != {case['cut_wav_sha256']}"
        )

    loaded = load_probabilities(args.predictions.resolve(), args.case_id, args.diar_variant, args.activity_threshold)
    probabilities = loaded["probabilities"]

    with sf.SoundFile(str(wav_path)) as source:
        if source.samplerate != SAMPLE_RATE or source.channels != 8:
            raise FrontendInputError(f"unexpected input format: {source.samplerate}Hz/{source.channels}ch")
        source_duration = source.frames / SAMPLE_RATE
    duration_s = float(case["window_end_s"]) - float(case["window_start_s"])
    if duration_s > source_duration + 1e-6:
        raise FrontendInputError(f"case window {duration_s}s exceeds source {source_duration}s")
    if args.limit_seconds and args.limit_seconds > 0:
        duration_s = min(duration_s, float(args.limit_seconds))

    activity = activity_mask(probabilities, args.activity_threshold)
    valid_frames = min(activity.shape[0], probability_frame_count(duration_s))
    blocks = plan_blocks(duration_s)

    variants: dict[str, list[int]] = {}
    for name in (args.enhance_variant, args.baseline_variant):
        if name not in VARIANT_CHANNELS:
            raise FrontendInputError(f"unknown channel variant {name!r}; known: {sorted(VARIANT_CHANNELS)}")
        variants[name] = VARIANT_CHANNELS[name]

    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    variant_stats: dict[str, Any] = {}
    for name, channels in variants.items():
        result = process_variant(
            variant=name,
            channels=channels,
            wav_path=wav_path,
            activity=activity,
            probabilities=probabilities,
            blocks=blocks,
            duration_s=duration_s,
            valid_frames=valid_frames,
            out_dir=out_dir,
            session_id=case["session_id"],
            case_id=case["case_id"],
            reg=args.reg,
        )
        all_rows.extend(result["rows"])
        variant_stats[name] = result["stats"]
    total_seconds = time.perf_counter() - started

    integrity = check_segment_integrity(all_rows, duration_s)
    if not integrity["passed"]:
        raise FrontendInputError(f"segment integrity gate failed: {integrity}")

    derived_segments = len({(row["track"], row["start_s"], row["end_s"]) for row in all_rows})
    # Cross-check: re-thresholding the probability columns must reproduce the
    # segments the full-meeting Sortformer run emitted inside the same window.
    emitted_within_window = sum(
        1
        for segment in (loaded["prediction_row"].get("segments") or [])
        if float(segment["start_s"]) < duration_s - 1e-9
    )
    emitted_active_seconds = sum(
        min(duration_s, float(segment["end_s"])) - float(segment["start_s"])
        for segment in (loaded["prediction_row"].get("segments") or [])
        if float(segment["start_s"]) < duration_s - 1e-9
    )
    derived_active_seconds = sum(
        row["duration_s"] for row in all_rows if row["variant"] == args.enhance_variant
    )
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "case": {
            "case_id": case["case_id"],
            "session_id": case["session_id"],
            "cut_wav_path": str(wav_path),
            "cut_wav_sha256": actual_wav_sha,
            "source_duration_s": round(source_duration, 3),
            "window_start_s": case["window_start_s"],
            "window_end_s": case["window_end_s"],
            "processed_seconds": round(duration_s, 3),
            "limit_seconds": args.limit_seconds or None,
        },
        "inputs": {
            "prep_manifest": str(prep_path),
            "prep_manifest_sha256": sha256_file(prep_path),
            "sortformer_run_manifest": str(args.predictions.resolve() / "run-manifest.json"),
            "sortformer_run_manifest_sha256": loaded["run_manifest_sha256"],
            "diar_variant": args.diar_variant,
            "probabilities_path": str(loaded["probabilities_path"]),
            "probabilities_sha256": loaded["probabilities_sha256"],
            "probability_shape": list(probabilities.shape),
            "emitted_segment_count": loaded["emitted_segment_count"],
            "emitted_segments_within_window": emitted_within_window,
            "emitted_active_seconds": round(emitted_active_seconds, 3),
            "derived_segment_count": derived_segments,
            "derived_active_seconds": round(derived_active_seconds, 3),
            "segment_count_delta_from_30s_cuts": derived_segments - emitted_within_window,
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "sha256": MODEL_SHA256,
            "nemo_commit": NEMO_COMMIT,
            "streaming_config_80ms_frames": (loaded["run_manifest"].get("streaming_config_80ms_frames") or {}),
            "postprocessing_threshold": loaded["run_manifest"].get("postprocessing_threshold"),
        },
        "config": {
            "activity_threshold": args.activity_threshold,
            "core_block_seconds": CORE_BLOCK_SECONDS,
            "right_context_seconds": RIGHT_CONTEXT_SECONDS,
            "frame_seconds": FRAME_SECONDS,
            "stft_n_fft": 512,
            "stft_hop": 128,
            "reg": args.reg,
            "variants": {name: channels for name, channels in variants.items()},
            "enhance_variant": args.enhance_variant,
            "baseline_variant": args.baseline_variant,
        },
        # The probability matrix is the output of one full-meeting NeMo run; this
        # pipeline does not re-drive the model chunk by chunk.
        "external_push_state_validated": False,
        "blocks": len(blocks),
        "segments": len(all_rows),
        "stages_seconds": {
            "read": round(sum(item["read_seconds"] for item in variant_stats.values()), 3),
            "beamform": round(sum(item["beamform_seconds"] for item in variant_stats.values()), 3),
            "write": round(sum(item["write_seconds"] for item in variant_stats.values()), 3),
            "total": round(total_seconds, 3),
        },
        "rtf": round(total_seconds / duration_s, 6) if duration_s else None,
        "max_pcm_buffer_samples": max(item["max_pcm_buffer_samples"] for item in variant_stats.values()),
        "max_pcm_buffer_bytes": 4 * max(item["max_pcm_buffer_samples"] for item in variant_stats.values()),
        "fallback_segments": sum(item["fallback_segments"] for item in variant_stats.values()),
        "fallback_reasons": {
            reason: sum(item["fallback_reasons"].get(reason, 0) for item in variant_stats.values())
            for reason in sorted({r for item in variant_stats.values() for r in item["fallback_reasons"]})
        },
        "integrity": integrity,
        "variants": {name: {key: value for key, value in item.items()} for name, item in variant_stats.items()},
        "gold_isolation": {
            "prep_fields_used": list(CASE_FIELD_WHITELIST),
            "gold_fields_excluded": list(GOLD_FORBIDDEN_FIELDS),
            "frontend_reads_textgrid": False,
        },
    }
    write_outputs(out_dir, all_rows, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="Sortformer run dir (run-manifest.json + predictions.jsonl).")
    parser.add_argument("--case-id", required=True, help="Explicit case selection; the frontend never guesses.")
    parser.add_argument("--diar-variant", default="micA_mean")
    parser.add_argument("--activity-threshold", type=float, default=0.30)
    parser.add_argument("--enhance-variant", default="all8")
    parser.add_argument("--baseline-variant", default="ch0")
    parser.add_argument("--limit-seconds", type=float, default=0.0, help="Process only the first N seconds (0 = full).")
    parser.add_argument("--reg", type=float, default=REG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.case_id is None:
        parser.error("--case-id is required; the frontend never guesses which case to process")
    if not 0.0 <= args.activity_threshold <= 1.0:
        parser.error("--activity-threshold must be within [0, 1]")
    manifest = run(args)
    print(json.dumps({
        "frontend_manifest": str(Path(args.out).resolve() / "frontend-manifest.json"),
        "case_id": manifest["case"]["case_id"],
        "processed_seconds": manifest["case"]["processed_seconds"],
        "blocks": manifest["blocks"],
        "segments": manifest["segments"],
        "fallback_segments": manifest["fallback_segments"],
        "rtf": manifest["rtf"],
        "integrity": manifest["integrity"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
