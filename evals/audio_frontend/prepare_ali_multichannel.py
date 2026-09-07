#!/usr/bin/env python3
"""Build a reproducible GSS input package from the existing Eval_Ali oracle intervals.

This tool isolates stage 12 (four-microphone oracle GSS) from diarization, the
memory system, and the Android product chain. It only prepares inputs:

1. Reuses the 255 fully-contained, >=0.5s reference intervals already scored by
   the P0 oracle-interval run (``per_interval.jsonl``). It does NOT invent new
   intervals or re-add boundary-truncated / short ones.
2. Cuts the matching 75-second, 8-channel far-field window from each AliMeeting
   session and keeps all 8 channels on disk (the GSS CLI selects the subset).
3. Writes one RTTM per session with times zeroed to the window start. The RTTM
   carries only speaker activity and anonymous speaker labels -- never gold text.
4. Writes a reference manifest that records, per segment: case, original and
   relative time, anonymous speaker, gold text, overlap label, the four preset
   channel variants, and the input SHA-256. The manifest also records the GSS
   code commit, license, source run manifest hash, and the locked ASR model hash
   so a later scoring run can fail closed on drift.

Raw PCM is written only under the output directory (which is git-ignored:
``reports/`` / ``data/``), never into the memory store, Timeline, or git.

The script does NOT require the ASR model or any GPU. Local smoke is enough to
verify the package before requesting A100 execution.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ai_glasses_memory_assistant.evals.eval_ali import sha256_file  # noqa: E402

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 75.0
DEFAULT_MIN_INTERVAL_SECONDS = 0.5

# Four fixed channel variants. micA / micB are the interleaved four-microphone
# proxies; ch0 is the single-channel baseline; all8 is the full array reference.
CHANNEL_VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "ch0", "channels": [0]},
    {"name": "micA", "channels": [0, 2, 4, 6]},
    {"name": "micB", "channels": [1, 3, 5, 7]},
    {"name": "all8", "channels": [0, 1, 2, 3, 4, 5, 6, 7]},
)

DEFAULT_GSS_LICENSE = "MIT"  # desh2608/gss
SCHEMA = "eval_ali_gss_prep.v1"


@dataclass(frozen=True)
class SegmentRef:
    segment_id: str
    case_id: str
    speaker: str
    original_start_s: float
    original_end_s: float
    rel_start_s: float
    rel_end_s: float
    text: str
    overlap_exposed: bool


def segment_id(case_id: str, speaker: str, rel_start_s: float, rel_end_s: float) -> str:
    """Stable, filesystem-safe identity for one (case, speaker, interval).

    The scorer matches GSS-enhanced audio to this id, so it must be fully
    determined by the interval and free of speaker/text content beyond the label.
    """
    start_ms = int(round(rel_start_s * 1000))
    end_ms = int(round(rel_end_s * 1000))
    return f"{case_id}__{speaker}__{start_ms:07d}__{end_ms:07d}"


def _interval_key(case_id: str, speaker: str, start_s: float, end_s: float) -> tuple[str, str, float, float]:
    return (case_id, speaker, round(start_s, 6), round(end_s, 6))


def load_source_cases(source_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read the source run manifest and index its reference intervals by key.

    Unlike ``p0_oracle_interval_ablation.load_source_run`` this does NOT require
    the ASR model files to be present, because package preparation only needs the
    time boundaries, gold text, and the locked ASR model hash -- not inference.
    """
    manifest_path = source_run / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = ((manifest.get("runtime") or {}).get("ambient_audio_profile") or {})
    if profile.get("asr_backend") != "sherpa_sensevoice":
        raise ValueError("source run must use sherpa_sensevoice to lock the GSS scoring ASR")
    asr_model_sha256 = profile.get("asr_model_sha256")
    if not asr_model_sha256:
        raise ValueError("source run manifest is missing asr_model_sha256")

    cases: dict[str, dict[str, Any]] = {}
    for item in manifest.get("cases") or []:
        interval_index = {
            _interval_key(item["case_id"], ri["speaker"], ri["start_s"], ri["end_s"]): ri
            for ri in item.get("reference_intervals") or []
        }
        cases[item["case_id"]] = {
            "case_id": item["case_id"],
            "session_id": item["session_id"],
            "audio_path": Path(item["audio_path"]),
            "window_start_s": float(item["start_s"]),
            "window_end_s": float(item["end_s"]),
            "intervals": interval_index,
        }
    if not cases:
        raise ValueError("source run manifest contains no cases")
    return manifest, cases


def cut_8ch_window(far_wav: Path, start_s: float, end_s: float, out_wav: Path) -> str:
    """Slice the [start_s, end_s) region of an 8-channel far WAV, bit-exact PCM16.

    Returns the SHA-256 of the written file (the GSS input fingerprint).
    """
    with sf.SoundFile(str(far_wav)) as source:
        if source.samplerate != SAMPLE_RATE:
            raise ValueError(f"{far_wav}: expected {SAMPLE_RATE} Hz, got {source.samplerate}")
        if source.channels != 8:
            raise ValueError(f"{far_wav}: expected 8 channels, got {source.channels}")
        start_frame = int(round(start_s * SAMPLE_RATE))
        width_frames = int(round((end_s - start_s) * SAMPLE_RATE))
        if start_frame < 0 or start_frame + width_frames > source.frames:
            raise ValueError(f"{far_wav}: window [{start_frame}, {start_frame + width_frames}) exceeds {source.frames} frames")
        source.seek(start_frame)
        data = source.read(width_frames, dtype="int16", always_2d=True)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), data, SAMPLE_RATE, subtype="PCM_16")
    return sha256_file(out_wav)


def write_rttm(rttm_path: Path, session_id: str, segments: Iterable[SegmentRef]) -> None:
    """Write an RTTM with times zeroed to the window start.

    Each line: SPEAKER <session_id> 1 <start> <dur> <NA> <NA> <speaker> <NA>.
    No gold text is written -- only activity and anonymous speaker labels.
    """
    lines = [
        "SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {} <NA>".format(
            session_id, seg.rel_start_s, seg.rel_end_s - seg.rel_start_s, seg.speaker
        )
        for seg in segments
    ]
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    rttm_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_package(
    *,
    source_run: Path,
    per_interval_path: Path,
    out_dir: Path,
    gss_commit: str,
    gss_license: str,
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Build the GSS input package and return the reference manifest dict."""
    source_run = source_run.resolve()
    per_interval_path = per_interval_path.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, cases = load_source_cases(source_run)
    source_manifest_sha256 = sha256_file(source_run / "run-manifest.json")
    asr_model_sha256 = ((manifest.get("runtime") or {}).get("ambient_audio_profile") or {}).get("asr_model_sha256")
    per_interval_sha256 = sha256_file(per_interval_path)

    rows = [json.loads(line) for line in per_interval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Join every P0 interval to its gold reference interval; fail closed on drift.
    segments_by_case: dict[str, list[SegmentRef]] = {}
    unmatched = 0
    for row in rows:
        case = cases.get(row["case_id"])
        if case is None:
            unmatched += 1
            continue
        key = _interval_key(row["case_id"], row["speaker"], row["start_s"], row["end_s"])
        ref = case["intervals"].get(key)
        if ref is None:
            unmatched += 1
            continue
        rel_start = float(row["start_s"]) - case["window_start_s"]
        rel_end = float(row["end_s"]) - case["window_start_s"]
        if rel_start < -1e-6 or rel_end > (case["window_end_s"] - case["window_start_s"]) + 1e-6:
            unmatched += 1
            continue
        seg = SegmentRef(
            segment_id=segment_id(row["case_id"], row["speaker"], rel_start, rel_end),
            case_id=row["case_id"],
            speaker=row["speaker"],
            original_start_s=float(row["start_s"]),
            original_end_s=float(row["end_s"]),
            rel_start_s=rel_start,
            rel_end_s=rel_end,
            text=ref["text"],
            overlap_exposed=bool(row.get("overlap_exposed", False)),
        )
        segments_by_case.setdefault(row["case_id"], []).append(seg)
    if unmatched:
        raise ValueError(f"{unmatched} per_interval rows did not match a source reference interval")

    case_manifests: list[dict[str, Any]] = []
    total_segments = 0
    for case_id, case in cases.items():
        segs = sorted(segments_by_case.get(case_id, []), key=lambda s: s.rel_start_s)
        cut_wav = out_dir / f"{case['session_id']}.wav"
        rttm = out_dir / f"{case['session_id']}.rttm"
        cut_sha = cut_8ch_window(case["audio_path"], case["window_start_s"], case["window_end_s"], cut_wav)
        write_rttm(rttm, case["session_id"], segs)
        case_manifests.append({
            "case_id": case_id,
            "session_id": case["session_id"],
            "far_audio_path": str(case["audio_path"]),
            "window_start_s": case["window_start_s"],
            "window_end_s": case["window_end_s"],
            "cut_wav_path": cut_wav.name,
            "cut_wav_sha256": cut_sha,
            "rttm_path": rttm.name,
            "segment_count": len(segs),
            "segments": [asdict(s) for s in segs],
        })
        total_segments += len(segs)

    if total_segments == 0:
        raise ValueError("no oracle intervals produced any GSS segment")

    out_manifest = {
        "schema": SCHEMA,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_run": str(source_run),
        "source_manifest_sha256": source_manifest_sha256,
        "per_interval_path": str(per_interval_path),
        "per_interval_sha256": per_interval_sha256,
        "asr_model_sha256": asr_model_sha256,
        "gss_commit": gss_commit,
        "gss_license": gss_license,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "min_interval_seconds": min_interval_seconds,
        "rttm_zeroed_to": "window_start",
        "channel_variants": [dict(v) for v in CHANNEL_VARIANTS],
        "cases": case_manifests,
        "total_segments": total_segments,
    }
    (out_dir / "prep-manifest.json").write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True, help="Completed Eval_Ali run dir containing run-manifest.json.")
    parser.add_argument("--per-interval", type=Path, required=True, help="P0 per_interval.jsonl with the 255 oracle intervals.")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "p2_gss_prep", help="Output dir (git-ignored). Must not exist.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--gss-commit", default="pending-verify", help="GSS repo commit; set after clone is pinned.")
    parser.add_argument("--gss-license", default=DEFAULT_GSS_LICENSE, help="GSS license identifier.")
    parser.add_argument("--limit-cases", type=int, default=0, help="Debug-only: prepare only the first N cases.")
    parser.add_argument("--min-interval-seconds", type=float, default=DEFAULT_MIN_INTERVAL_SECONDS)
    args = parser.parse_args(argv)
    if args.min_interval_seconds <= 0:
        parser.error("--min-interval-seconds must be positive")

    out_dir = args.out.resolve()
    if args.run_id:
        out_dir = out_dir / args.run_id
    manifest = prepare_package(
        source_run=args.source_run,
        per_interval_path=args.per_interval,
        out_dir=out_dir,
        gss_commit=args.gss_commit,
        gss_license=args.gss_license,
        min_interval_seconds=args.min_interval_seconds,
    )
    # --limit-cases is only a debug aid and must not silently shrink a real package.
    if args.limit_cases and args.limit_cases < len(manifest["cases"]):
        raise SystemExit(
            f"--limit-cases={args.limit_cases} would shrink a real package of {len(manifest['cases'])} cases; "
            "use a separate debug output dir instead."
        )
    print(json.dumps({
        "prep_manifest": str(out_dir / "prep-manifest.json"),
        "cases": len(manifest["cases"]),
        "total_segments": manifest["total_segments"],
        "asr_model_sha256": manifest["asr_model_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
