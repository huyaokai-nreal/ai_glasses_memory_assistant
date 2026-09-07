#!/usr/bin/env python3
"""Run pinned NVIDIA Streaming Sortformer on AliMeeting 75-second windows.

Sortformer accepts mono audio. For each 8-channel window this runner evaluates:
``ch0`` and the two approved four-channel proxies (``micA`` and ``micB``) mixed
with a fixed zero-delay average. The average is gold-free and reproducible; it
is only a diagnostic reference because the AliMeeting array geometry is not
encoded here and no steering direction is estimated.

The model revision and NeMo source commit are pinned. Inputs, model SHA-256,
runtime, streaming parameters and predictions are recorded in the run manifest.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "eval_ali_sortformer_predictions.v1"
MODEL_REPO = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"
MODEL_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
MODEL_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
NEMO_COMMIT = "ca4daa1470f6c01068c4e6a9a73b19b9a91dc366"
SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.08
INPUT_VARIANTS = {
    "ch0": [0],
    "micA_mean": [0, 2, 4, 6],
    "micB_mean": [1, 3, 5, 7],
}
STREAMING_CONFIG = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    "spkcache_update_period": 300,
    "spkcache_len": 188,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_average(data: np.ndarray, channels: list[int]) -> np.ndarray:
    """Return a mono zero-delay channel average without reference leakage."""
    if data.ndim != 2:
        raise ValueError(f"expected [frames, channels], got {data.shape}")
    if not channels or min(channels) < 0 or max(channels) >= data.shape[1]:
        raise ValueError(f"invalid channels {channels} for shape {data.shape}")
    return np.mean(data[:, channels], axis=1, dtype=np.float32)


def parse_segment(value: Any) -> tuple[float, float, str]:
    """Normalize NeMo's string/tuple/dict segment forms."""
    if isinstance(value, str):
        fields = re.split(r"[\s,]+", value.strip())
        if len(fields) < 3:
            raise ValueError(f"unexpected Sortformer segment: {value!r}")
        return float(fields[0]), float(fields[1]), str(fields[2])
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return float(value[0]), float(value[1]), str(value[2])
    if isinstance(value, dict):
        start = value.get("start_s", value.get("start", value.get("begin")))
        end = value.get("end_s", value.get("end"))
        speaker = value.get("speaker", value.get("label"))
        if start is not None and end is not None and speaker is not None:
            return float(start), float(end), str(speaker)
    raise ValueError(f"unexpected Sortformer segment type: {value!r}")


def segment_confidence(probabilities: Any, start_s: float, end_s: float, speaker: str) -> float | None:
    """Average the matching speaker activity probability over one segment."""
    match = re.search(r"(\d+)$", speaker)
    if probabilities is None or match is None:
        return None
    if hasattr(probabilities, "detach"):
        probabilities = probabilities.detach().cpu().numpy()
    probs = np.asarray(probabilities)
    if probs.ndim == 3 and probs.shape[0] == 1:
        probs = probs[0]
    speaker_index = int(match.group(1))
    if probs.ndim != 2 or speaker_index >= probs.shape[1]:
        return None
    start_frame = max(0, int(np.floor(start_s / FRAME_SECONDS)))
    end_frame = min(probs.shape[0], max(start_frame + 1, int(np.ceil(end_s / FRAME_SECONDS))))
    return float(np.mean(probs[start_frame:end_frame, speaker_index]))


def download_pinned_model(model_dir: Path) -> Path:
    """Download the exact model revision and fail closed on weight drift."""
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        revision=MODEL_REVISION,
        local_dir=str(model_dir),
    ))
    actual = sha256_file(path)
    if actual != MODEL_SHA256:
        raise ValueError(f"Sortformer model SHA-256 drift: expected {MODEL_SHA256}, got {actual}")
    return path


def load_model(model_path: Path, device: str):
    """Import NeMo lazily so unit tests do not require the evaluation runtime."""
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(
        restore_path=str(model_path), map_location=torch.device(device), strict=False
    )
    model.eval()
    for name, value in STREAMING_CONFIG.items():
        setattr(model.sortformer_modules, name, value)
    model.sortformer_modules._check_streaming_parameters()
    return model


def normalize_prediction_batch(raw: Any) -> list[Any]:
    """Extract one audio item's segment list from NeMo's batch output."""
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        return raw[0]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"unexpected Sortformer prediction batch: {type(raw).__name__}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import nemo
    import torch

    prep_dir = args.prep_dir.resolve()
    prep = json.loads((prep_dir / "prep-manifest.json").read_text(encoding="utf-8"))
    if prep.get("schema") not in {"eval_ali_gss_prep.v1", "eval_ali_continuous_prep.v1"}:
        raise ValueError(f"unexpected prep schema: {prep.get('schema')}")
    out_dir = args.out.resolve() / args.run_id if args.run_id else args.out.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite Sortformer run: {out_dir}")
    out_dir.mkdir(parents=True)
    input_dir = out_dir / "inputs"
    input_dir.mkdir()
    probs_dir = out_dir / "probabilities"
    probs_dir.mkdir()
    postprocessing_path: Path | None = args.postprocessing_yaml.resolve() if args.postprocessing_yaml else None
    if args.postprocessing_threshold is not None:
        if postprocessing_path is not None:
            raise ValueError("use either --postprocessing-yaml or --postprocessing-threshold")
        postprocessing_path = out_dir / "postprocessing.yaml"
        postprocessing_path.write_text(
            "parameters:\n"
            "  onset: {0}\n  offset: {0}\n  pad_onset: 0.0\n  pad_offset: 0.0\n"
            "  min_duration_on: 0.0\n  min_duration_off: 0.0\n".format(args.postprocessing_threshold),
            encoding="utf-8",
        )

    model_path = args.model_path.resolve() if args.model_path else download_pinned_model(
        ROOT / "reports" / "models" / "sortformer-v2.1"
    )
    actual_model_sha = sha256_file(model_path)
    if actual_model_sha != MODEL_SHA256:
        raise ValueError(f"Sortformer model SHA-256 drift: expected {MODEL_SHA256}, got {actual_model_sha}")
    model = load_model(model_path, args.device)

    prediction_rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    cases = prep["cases"][: args.limit_cases or None]
    selected_variants = list(INPUT_VARIANTS)
    if args.variants:
        selected_variants = [name.strip() for name in args.variants.split(",") if name.strip()]
        unknown = set(selected_variants) - set(INPUT_VARIANTS)
        if unknown:
            raise ValueError(f"unknown input variants: {sorted(unknown)}")
    for case in cases:
        input_wav = Path(case["cut_wav_path"])
        if not input_wav.is_absolute():
            input_wav = prep_dir / input_wav
        with sf.SoundFile(str(input_wav)) as source:
            if source.samplerate != SAMPLE_RATE or source.channels != 8:
                raise ValueError(f"unexpected input format: {source.samplerate}Hz/{source.channels}ch")
            data = source.read(dtype="float32", always_2d=True)
        duration_s = data.shape[0] / SAMPLE_RATE
        for variant in selected_variants:
            channels = INPUT_VARIANTS[variant]
            mono = fixed_average(data, channels)
            audio_path = input_dir / f"{case['case_id']}__{variant}.wav"
            sf.write(str(audio_path), mono, SAMPLE_RATE, subtype="PCM_16")
            t0 = time.perf_counter()
            raw_segments, raw_probs = model.diarize(
                audio=[str(audio_path)],
                batch_size=1,
                include_tensor_outputs=True,
                postprocessing_yaml=str(postprocessing_path) if postprocessing_path else None,
                verbose=False,
            )
            runtime_s = time.perf_counter() - t0
            segments = normalize_prediction_batch(raw_segments)
            probs = raw_probs[0] if isinstance(raw_probs, (list, tuple)) and len(raw_probs) == 1 else raw_probs
            probs_array = probs.detach().cpu().numpy() if hasattr(probs, "detach") else np.asarray(probs)
            if probs_array.ndim == 3 and probs_array.shape[0] == 1:
                probs_array = probs_array[0]
            probs_path = probs_dir / f"{case['case_id']}__{variant}.npy"
            np.save(probs_path, np.asarray(probs_array, dtype=np.float32), allow_pickle=False)
            normalized: list[dict[str, Any]] = []
            for raw_segment in segments:
                start_s, end_s, speaker = parse_segment(raw_segment)
                start_s = max(0.0, min(duration_s, start_s))
                end_s = max(start_s, min(duration_s, end_s))
                if end_s <= start_s:
                    continue
                confidence = segment_confidence(probs, start_s, end_s, speaker)
                normalized.append({
                    "start_s": start_s,
                    "end_s": end_s,
                    "speaker": speaker,
                    "confidence": confidence,
                })
            audio_sha = sha256_file(audio_path)
            prediction_rows.append({
                "case_id": case["case_id"],
                "session_id": case["session_id"],
                "input_variant": variant,
                "channels": channels,
                "mix": "fixed_zero_delay_average",
                "audio_path": str(audio_path),
                "audio_sha256": audio_sha,
                "audio_seconds": duration_s,
                "runtime_seconds": runtime_s,
                "rtf": runtime_s / duration_s,
                "probabilities_path": str(probs_path),
                "probabilities_sha256": sha256_file(probs_path),
                "segments": normalized,
            })
            input_records.append({
                "case_id": case["case_id"], "input_variant": variant,
                "channels": channels, "audio_sha256": audio_sha,
            })
            print(f"{case['case_id']} {variant}: {len(normalized)} segments, rtf={runtime_s / duration_s:.3f}")

    predictions_path = out_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows), encoding="utf-8"
    )
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prep_manifest": str(prep_dir / "prep-manifest.json"),
        "model": {
            "repo": MODEL_REPO,
            "filename": MODEL_FILENAME,
            "revision": MODEL_REVISION,
            "sha256": actual_model_sha,
            "license": "NVIDIA Open Model License",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": args.device,
            "torch": torch.__version__,
            "nemo": getattr(nemo, "__version__", "unknown"),
            "nemo_commit": NEMO_COMMIT,
        },
        "streaming_config_80ms_frames": STREAMING_CONFIG,
        "postprocessing_yaml": str(postprocessing_path) if postprocessing_path else None,
        "postprocessing_threshold": args.postprocessing_threshold,
        "input_variants": {name: INPUT_VARIANTS[name] for name in selected_variants},
        "inputs": input_records,
        "cases": len(cases),
        "prediction_rows": len(prediction_rows),
        "confidence": "mean speaker probability over emitted segment",
    }
    (out_dir / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "p3_sortformer")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--postprocessing-yaml", type=Path, default=None)
    parser.add_argument("--postprocessing-threshold", type=float, default=None)
    parser.add_argument("--variants", default="", help="Comma-separated subset of ch0,micA_mean,micB_mean.")
    parser.add_argument("--limit-cases", type=int, default=0)
    args = parser.parse_args(argv)
    if args.postprocessing_threshold is not None and not 0.0 <= args.postprocessing_threshold <= 1.0:
        parser.error("--postprocessing-threshold must be within [0, 1]")
    manifest = run(args)
    print(json.dumps({
        "run_manifest": str((args.out.resolve() / args.run_id if args.run_id else args.out.resolve()) / "run-manifest.json"),
        "cases": manifest["cases"],
        "prediction_rows": manifest["prediction_rows"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
