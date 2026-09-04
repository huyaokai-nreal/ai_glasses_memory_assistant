#!/usr/bin/env python3
"""Run a fixed two-speaker AliMeeting pilot for MossFormer2 separation.

The pilot selects short, high-overlap intervals from the three two-speaker Eval
meetings, so the upstream model can run in one pass without stream stitching.
It compares one mixed ASR stream, two separated streams, and the aligned two
near-field oracle streams with permutation-invariant character error rate.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ai_glasses_memory_assistant.audio_engine.backends import OfflineAsrBackend
from ai_glasses_memory_assistant.evals.eval_ali import ReferenceInterval, normalize_text, parse_textgrid, score_cer, sha256_file
from p0_oracle_interval_ablation import load_source_run

MODEL_ID = "alibabasglab/MossFormer2_SS_16K"
MODEL_LICENSE = "Apache-2.0"
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class OverlapClip:
    session_id: str
    speaker_a: str
    speaker_b: str
    start_s: float
    end_s: float
    overlap_s: float
    overlap_ratio: float
    reference_a: str
    reference_b: str


def select_overlap_clips(
    eval_root: Path,
    *,
    clips_per_session: int,
    min_overlap_seconds: float,
    max_clip_seconds: float,
    min_reference_chars: int,
) -> list[OverlapClip]:
    selected: list[OverlapClip] = []
    for textgrid_path in sorted((eval_root / "Eval_Ali_far" / "textgrid_dir").glob("*.TextGrid")):
        intervals = [item for item in parse_textgrid(textgrid_path) if item.text.strip()]
        speakers = sorted({item.speaker for item in intervals})
        if len(speakers) != 2:
            continue
        candidates: list[OverlapClip] = []
        for first_index, first in enumerate(intervals):
            for second in intervals[first_index + 1 :]:
                if first.speaker == second.speaker:
                    continue
                overlap_s = min(first.end_s, second.end_s) - max(first.start_s, second.start_s)
                start_s = min(first.start_s, second.start_s)
                end_s = max(first.end_s, second.end_s)
                duration_s = end_s - start_s
                if overlap_s < min_overlap_seconds or duration_s > max_clip_seconds:
                    continue
                if min(len(normalize_text(first.text)), len(normalize_text(second.text))) < min_reference_chars:
                    continue
                ordered = sorted((first, second), key=lambda item: item.speaker)
                candidates.append(OverlapClip(
                    session_id=textgrid_path.stem,
                    speaker_a=ordered[0].speaker,
                    speaker_b=ordered[1].speaker,
                    start_s=start_s,
                    end_s=end_s,
                    overlap_s=overlap_s,
                    overlap_ratio=overlap_s / duration_s,
                    reference_a=ordered[0].text,
                    reference_b=ordered[1].text,
                ))
        chosen: list[OverlapClip] = []
        for candidate in sorted(candidates, key=lambda item: (-item.overlap_s, -item.overlap_ratio, item.start_s)):
            if any(min(candidate.end_s, old.end_s) > max(candidate.start_s, old.start_s) for old in chosen):
                continue
            chosen.append(candidate)
            if len(chosen) == clips_per_session:
                break
        if len(chosen) != clips_per_session:
            raise ValueError(f"not enough qualifying overlap clips in {textgrid_path.stem}")
        selected.extend(sorted(chosen, key=lambda item: item.start_s))
    if not selected:
        raise ValueError("no two-speaker AliMeeting sessions found")
    return selected


def permutation_cer(references: tuple[str, str], hypotheses: tuple[str, str]) -> dict[str, Any]:
    best: tuple[int, tuple[str, str], list[dict[str, Any]]] | None = None
    for permuted in permutations(hypotheses):
        scores = [score_cer(reference, hypothesis)["normalized"] for reference, hypothesis in zip(references, permuted)]
        errors = sum(int(score["errors"]) for score in scores)
        candidate = (errors, permuted, scores)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    reference_chars = sum(int(score["reference_chars"]) for score in best[2])
    return {
        "errors": best[0],
        "reference_chars": reference_chars,
        "cer": best[0] / reference_chars if reference_chars else None,
        "hypotheses": list(best[1]),
        "per_speaker": best[2],
    }


def load_separator(package_dir: Path, checkpoint_path: Path, *, torch_threads: int) -> tuple[Any, Any, float]:
    sys.path.insert(0, str(package_dir.resolve()))
    import torch
    from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K

    torch.set_num_threads(torch_threads)
    args = Namespace(
        num_spks=2,
        encoder_kernel_size=16,
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
    )
    started = time.perf_counter()
    model = MossFormer2_SS_16K(args).model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return torch, model, time.perf_counter() - started


def separate_one_pass(torch: Any, model: Any, mixture: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    started = time.perf_counter()
    with torch.inference_mode():
        raw = model(torch.from_numpy(np.asarray(mixture[None, :], dtype=np.float32)))
    elapsed = time.perf_counter() - started
    input_rms = float(np.sqrt(np.mean(np.square(mixture))))
    outputs: list[np.ndarray] = []
    for tensor in raw:
        audio = tensor[0, : len(mixture)].detach().cpu().numpy().astype(np.float32)
        output_rms = float(np.sqrt(np.mean(np.square(audio))))
        if output_rms > 1e-8:
            audio = audio * (input_rms / output_rms)
        outputs.append(audio)
    if len(outputs) != 2:
        raise ValueError(f"separator returned {len(outputs)} streams instead of 2")
    return (outputs[0], outputs[1]), elapsed


def _read_range(path: Path, start_s: float, end_s: float, *, channel: int) -> np.ndarray:
    audio, sample_rate = sf.read(
        path,
        start=round(start_s * SAMPLE_RATE),
        stop=round(end_s * SAMPLE_RATE),
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != SAMPLE_RATE or channel >= audio.shape[1]:
        raise ValueError(f"unexpected audio format: {path}")
    return np.asarray(audio[:, channel], dtype=np.float32)


def _far_audio_path(eval_root: Path, session_id: str) -> Path:
    matches = sorted((eval_root / "Eval_Ali_far" / "audio_dir").glob(f"{session_id}_*.wav"))
    if len(matches) != 1:
        raise ValueError(f"expected one far track for {session_id}, found {len(matches)}")
    return matches[0]


def run_clip(
    clip: OverlapClip,
    *,
    eval_root: Path,
    asr: OfflineAsrBackend,
    torch: Any,
    separator: Any,
) -> dict[str, Any]:
    mixture = _read_range(_far_audio_path(eval_root, clip.session_id), clip.start_s, clip.end_s, channel=0)
    separated, inference_seconds = separate_one_pass(torch, separator, mixture)
    raw_hypothesis = asr.transcribe(mixture)
    separated_hypotheses = tuple(asr.transcribe(audio) for audio in separated)
    near_hypotheses = tuple(
        asr.transcribe(_read_range(
            eval_root / "Eval_Ali_near" / "audio_dir" / f"{clip.session_id}_{speaker}.wav",
            clip.start_s,
            clip.end_s,
            channel=0,
        ))
        for speaker in (clip.speaker_a, clip.speaker_b)
    )
    references = (clip.reference_a, clip.reference_b)
    return {
        **asdict(clip),
        "duration_s": clip.end_s - clip.start_s,
        "inference_seconds": inference_seconds,
        "realtime_factor": inference_seconds / (clip.end_s - clip.start_s),
        "raw_single_stream": permutation_cer(references, (raw_hypothesis, "")),
        "mossformer2_two_stream": permutation_cer(references, separated_hypotheses),
        "near_oracle_two_stream": permutation_cer(references, near_hypotheses),
    }


def aggregate_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)

    def score(name: str) -> dict[str, Any]:
        errors = sum(int(row[name]["errors"]) for row in materialized)
        reference_chars = sum(int(row[name]["reference_chars"]) for row in materialized)
        return {"errors": errors, "reference_chars": reference_chars, "cer": errors / reference_chars if reference_chars else None}

    rtfs = [float(row["realtime_factor"]) for row in materialized]
    return {
        "clips": len(materialized),
        "audio_seconds": sum(float(row["duration_s"]) for row in materialized),
        "raw_single_stream": score("raw_single_stream"),
        "mossformer2_two_stream": score("mossformer2_two_stream"),
        "near_oracle_two_stream": score("near_oracle_two_stream"),
        "clip_outcomes_vs_raw": {
            "improved": sum(row["mossformer2_two_stream"]["errors"] < row["raw_single_stream"]["errors"] for row in materialized),
            "equal": sum(row["mossformer2_two_stream"]["errors"] == row["raw_single_stream"]["errors"] for row in materialized),
            "worse": sum(row["mossformer2_two_stream"]["errors"] > row["raw_single_stream"]["errors"] for row in materialized),
        },
        "realtime_factor": {
            "mean": statistics.fmean(rtfs),
            "median": statistics.median(rtfs),
            "max": max(rtfs),
        },
    }


def write_summary(path: Path, scores: dict[str, Any]) -> None:
    aggregate = scores["aggregate"]
    lines = [
        "# MossFormer2 中文会议重叠语音 pilot",
        "",
        "- 输入来自 AliMeeting Eval 的 3 场双人会议；按固定规则每场选择短高重叠片段。",
        "- raw 用 1 路 ASR 加 1 路空结果评分；MossFormer2 和 near oracle 均以两路置换不变 CER 评分。",
        "- 这是离线候选筛选，不是连续分离、说话人轨道拼接或 Android 验收。",
        "",
        "## 结果",
        "",
        f"- raw 单流 CER：{aggregate['raw_single_stream']['cer']}",
        f"- MossFormer2 双流 CER：{aggregate['mossformer2_two_stream']['cer']}",
        f"- near oracle 双流 CER：{aggregate['near_oracle_two_stream']['cer']}",
        f"- clip 改善/持平/变差：{aggregate['clip_outcomes_vs_raw']}",
        f"- CPU 实时率 mean/median/max：{aggregate['realtime_factor']}",
        "",
        "MossFormer2 只有在双流 CER 明显低于 raw 且输出稳定时才值得继续；near oracle 给出当前 ASR 的可达上限。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True, help="Eval_Ali run that pins the ASR checkpoint.")
    parser.add_argument("--eval-root", type=Path, default=ROOT / "data" / "Eval_Ali")
    parser.add_argument("--clearvoice-package-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "p1_mossformer_overlap")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--clips-per-session", type=int, default=2)
    parser.add_argument("--min-overlap-seconds", type=float, default=0.5)
    parser.add_argument("--max-clip-seconds", type=float, default=2.0)
    parser.add_argument("--min-reference-chars", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=6)
    args = parser.parse_args(argv)
    if args.clips_per_session <= 0 or args.min_overlap_seconds <= 0 or args.max_clip_seconds <= 0:
        parser.error("clip counts and durations must be positive")
    if args.max_clip_seconds > 2.0:
        parser.error("this one-pass pilot requires --max-clip-seconds <= 2.0")

    profile, _ = load_source_run(args.source_run.resolve())
    asr = OfflineAsrBackend()
    capability = asr.capability()
    if capability.status != "ready":
        raise ValueError(f"ASR is not ready: {capability.reason}")
    checkpoint = args.checkpoint.resolve()
    torch, separator, model_load_seconds = load_separator(
        args.clearvoice_package_dir,
        checkpoint,
        torch_threads=args.torch_threads,
    )
    clips = select_overlap_clips(
        args.eval_root.resolve(),
        clips_per_session=args.clips_per_session,
        min_overlap_seconds=args.min_overlap_seconds,
        max_clip_seconds=args.max_clip_seconds,
        min_reference_chars=args.min_reference_chars,
    )
    rows = [run_clip(clip, eval_root=args.eval_root.resolve(), asr=asr, torch=torch, separator=separator) for clip in clips]
    scores = {
        "schema": "eval_ali_mossformer_overlap_pilot.v1",
        "model": {"id": MODEL_ID, "license": MODEL_LICENSE, "checkpoint_sha256": sha256_file(checkpoint)},
        "asr": {"profile": profile.get("name"), "model_sha256": profile.get("asr_model_sha256")},
        "selection": {
            "clips_per_session": args.clips_per_session,
            "min_overlap_seconds": args.min_overlap_seconds,
            "max_clip_seconds": args.max_clip_seconds,
            "min_reference_chars": args.min_reference_chars,
        },
        "model_load_seconds": model_load_seconds,
        "aggregate": aggregate_results(rows),
        "limitations": [
            "Only the three two-speaker Eval meetings are included.",
            "Each clip is shorter than two seconds and uses one-pass separation.",
            "Permutation is optimized independently per clip; continuous stream assignment is not tested.",
            "The model was not trained or officially benchmarked on AliMeeting by its publisher.",
            "This offline pilot does not establish Android deployability.",
        ],
    }
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "per_clip.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(run_dir / "summary.md", scores)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
