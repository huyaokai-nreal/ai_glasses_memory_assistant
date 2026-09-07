#!/usr/bin/env python3
"""Drive four-mic MVDR with Sortformer activity instead of oracle activity.

This is an attribution experiment. Sortformer probabilities create anonymous
activity masks; evaluation-only Hungarian mapping aligns those tracks to gold
speaker labels, while the 255 gold CER intervals still define output cut times.
Thus it measures how much spatial-enhancement gain survives automatic activity
estimation, but it is not yet fully automatic continuous transcription.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from enhance_mvdr_oracle import SAMPLE_RATE, enhance_variant, istft  # noqa: E402
from score_diarization import load_complete_references, score_case  # noqa: E402
from tune_sortformer_threshold import load_probability_index, probabilities_to_segments  # noqa: E402

SCHEMA = "eval_ali_mvdr_auto_mask.v1"
CHANNEL_VARIANTS = {"micA_mean": ("micA", [0, 2, 4, 6])}


def write_activity_rttm(path: Path, session_id: str, segments: list[dict[str, Any]]) -> None:
    lines = [
        f"SPEAKER {session_id} 1 {segment['start_s']:.3f} "
        f"{segment['end_s'] - segment['start_s']:.3f} <NA> <NA> {segment['speaker']} <NA>"
        for segment in segments
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--prediction-run", type=Path, required=True)
    parser.add_argument("--threshold-scan", type=Path, required=True)
    parser.add_argument("--diar-variant", choices=tuple(CHANNEL_VARIANTS), default="micA_mean")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reg", type=float, default=1e-6)
    args = parser.parse_args(argv)

    prep_dir = args.prep_dir.resolve()
    prep_path = prep_dir / "prep-manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    references, durations = load_complete_references(prep_path)
    probability_paths = load_probability_index(args.prediction_run)
    scan = json.loads(args.threshold_scan.read_text(encoding="utf-8"))
    selected = scan["variants"][args.diar_variant]
    if not selected.get("holdout_gate_passed"):
        raise ValueError(f"selected diarization variant did not pass holdout gate: {args.diar_variant}")
    threshold = float(selected["selected_threshold"])
    enhance_variant_name, channels = CHANNEL_VARIANTS[args.diar_variant]
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite auto-mask run: {args.out}")
    (args.out / enhance_variant_name).mkdir(parents=True)
    activity_dir = args.out / "predicted_activity"
    activity_dir.mkdir()

    total_cpu_seconds = 0.0
    total_audio_seconds = 0.0
    total_segments = 0
    mappings: dict[str, Any] = {}
    prep_cases = {case["case_id"]: case for case in prep["cases"]}
    for case_id in sorted(prep_cases):
        case = prep_cases[case_id]
        probabilities = np.load(probability_paths[args.diar_variant][case_id], allow_pickle=False)
        activity_segments = probabilities_to_segments(probabilities, threshold)
        diar_score = score_case(references[case_id], activity_segments, durations[case_id])
        hyp_to_ref = diar_score["speaker_mapping"]
        ref_to_hyp = {ref: hyp for hyp, ref in hyp_to_ref.items() if ref is not None}
        write_activity_rttm(activity_dir / f"{case_id}.rttm", case["session_id"], activity_segments)

        t0 = time.perf_counter()
        enhanced = enhance_variant(
            cut_wav=prep_dir / case["cut_wav_path"],
            segments=case["segments"],
            channels=channels,
            reg=args.reg,
            activity_segments=activity_segments,
            output_speaker_map=ref_to_hyp,
        )
        cpu_seconds = time.perf_counter() - t0
        case_audio_seconds = 0.0
        for segment_id, spectrum in enhanced.items():
            audio = istft(spectrum.T)
            case_audio_seconds += audio.shape[0] / SAMPLE_RATE
            sf.write(
                str(args.out / enhance_variant_name / f"{segment_id}.wav"),
                audio,
                SAMPLE_RATE,
                subtype="PCM_16",
            )
        gold_speakers = {segment["speaker"] for segment in case["segments"]}
        mappings[case_id] = {
            "hypothesis_to_reference": hyp_to_ref,
            "reference_to_hypothesis": ref_to_hyp,
            "unmapped_reference_speakers": sorted(gold_speakers - set(ref_to_hyp)),
            "diarization": diar_score,
        }
        total_cpu_seconds += cpu_seconds
        total_audio_seconds += case_audio_seconds
        total_segments += len(enhanced)
        print(f"{case_id}: {len(enhanced)} segments, cpu_rtf={cpu_seconds / case_audio_seconds:.4f}")

    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "Sortformer activity + evaluation speaker mapping + NumPy MVDR",
        "oracle_segment_boundaries": True,
        "evaluation_speaker_mapping": True,
        "fully_automatic_continuous_asr": False,
        "prep_manifest": str(prep_path),
        "prediction_run": str(args.prediction_run.resolve()),
        "threshold_scan": str(args.threshold_scan.resolve()),
        "diar_variant": args.diar_variant,
        "activity_threshold": threshold,
        "variants": {
            enhance_variant_name: {
                "channels": channels,
                "segments": total_segments,
                "audio_seconds": total_audio_seconds,
                "cpu_seconds": total_cpu_seconds,
                "cpu_rtf": total_cpu_seconds / total_audio_seconds,
            }
        },
        "speaker_mappings": mappings,
    }
    (args.out / "enhance-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(args.out / "enhance-manifest.json"),
        "variant": enhance_variant_name,
        "segments": total_segments,
        "cpu_rtf": total_cpu_seconds / total_audio_seconds,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
