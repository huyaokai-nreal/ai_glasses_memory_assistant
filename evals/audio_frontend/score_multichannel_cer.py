#!/usr/bin/env python3
"""Score stage-12 enhanced audio against the P2 oracle reference manifest.

This scorer is the second half of stage 12. It pairs each enhanced segment
with its gold reference text using the ``segment_id`` recorded in the prep
manifest, transcribes with the SAME SenseVoice checkpoint locked by the source
Eval_Ali run, and reports CER split by all / non-overlap / overlap-exposed, plus
per-case counts and deletion/substitution/insertion breakdown.

Fail-closed rules (raise, never silently score):
- A GSS output whose ``segment_id`` is not in the prep manifest (unknown segment).
- A ``segment_id`` produced more than once in a variant directory (duplicate).
- Any prep-manifest segment missing from a variant directory (missing).
- The ASR model hash derived from the source run differs from the hash recorded
  in the prep manifest (model drift between prepare and score).

Output naming convention:
    <gss_root>/<variant_name>/<segment_id>.wav

The raw PCM / enhanced audio never enters the memory store, Timeline, or git.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ai_glasses_memory_assistant.audio_engine.backends import OfflineAsrBackend  # noqa: E402
from ai_glasses_memory_assistant.evals.eval_ali import score_cer  # noqa: E402
from p0_oracle_interval_ablation import (  # noqa: E402
    IntervalResult,
    aggregate,
    load_source_run,
    summarize_cer_counts,
)

SCHEMA = "eval_ali_gss_score.v1"


class GssDiscoveryError(Exception):
    """Base class for fail-closed mapping errors."""


class UnknownSegmentError(GssDiscoveryError):
    """A GSS output file does not correspond to any prep-manifest segment."""


class DuplicateSegmentError(GssDiscoveryError):
    """A segment_id appears more than once in a variant directory."""


class MissingSegmentsError(GssDiscoveryError):
    """Prep-manifest segments are absent from a variant directory."""


class AsrHashMismatchError(GssDiscoveryError):
    """ASR checkpoint drifted between prepare and score."""


def load_prep_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_segments(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten all cases into segment_id -> segment record (with case_id)."""
    idx: dict[str, dict[str, Any]] = {}
    for case in manifest.get("cases") or []:
        for seg in case.get("segments") or []:
            sid = seg["segment_id"]
            if sid in idx:
                raise DuplicateSegmentError(f"{sid}: duplicate segment_id in prep manifest")
            idx[sid] = {**seg, "case_id": case["case_id"]}
    if not idx:
        raise MissingSegmentsError("prep manifest contains no segments")
    return idx


def discover_gss_outputs(
    gss_root: Path,
    manifest: dict[str, Any],
    variant_names: set[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Map variant_name -> segment_id -> wav path, fail-closed on any anomaly."""
    gss_root = Path(gss_root)
    index = index_segments(manifest)
    result: dict[str, dict[str, Path]] = {}
    for variant in manifest.get("channel_variants") or []:
        vname = variant["name"]
        if variant_names is not None and vname not in variant_names:
            continue
        vdir = gss_root / vname
        if not vdir.is_dir():
            raise MissingSegmentsError(f"variant directory missing: {vdir}")
        found: dict[str, Path] = {}
        for wav in sorted(vdir.glob("*.wav")):
            sid = wav.stem
            if sid not in index:
                raise UnknownSegmentError(f"{wav}: segment_id not present in prep manifest")
            if sid in found:
                raise DuplicateSegmentError(f"{sid}: duplicate output in {vdir}")
            found[sid] = wav
        missing = set(index) - set(found)
        if missing:
            example = sorted(missing)[0]
            raise MissingSegmentsError(f"variant {vname}: {len(missing)} segment(s) missing (e.g. {example})")
        result[vname] = found
    return result


def transcribe_file(asr: OfflineAsrBackend, path: Path) -> str:
    with sf.SoundFile(str(path)) as src:
        if src.channels > 1:
            audio = src.read(dtype="float32", always_2d=True)[:, 0]
        else:
            audio = src.read(dtype="float32").reshape(-1)
    return str(asr.transcribe(np.asarray(audio, dtype=np.float32)) or "").strip()


def score_segment(asr: OfflineAsrBackend, seg: dict[str, Any], wav: Path) -> IntervalResult:
    hypothesis = transcribe_file(asr, wav)
    normalized = score_cer(seg["text"], hypothesis)["normalized"]
    return IntervalResult(
        case_id=seg["case_id"],
        speaker=seg["speaker"],
        start_s=seg["original_start_s"],
        end_s=seg["original_end_s"],
        duration_s=seg["rel_end_s"] - seg["rel_start_s"],
        overlap_exposed=bool(seg["overlap_exposed"]),
        reference_chars=int(normalized["reference_chars"]),
        errors=int(normalized["errors"]),
        substitutions=int(normalized["substitutions"]),
        deletions=int(normalized["deletions"]),
        insertions=int(normalized["insertions"]),
    )


def aggregate_per_case(rows: Iterable[IntervalResult]) -> dict[str, dict[str, Any]]:
    by_case: dict[str, list[IntervalResult]] = {}
    for row in rows:
        by_case.setdefault(row.case_id, []).append(row)
    return {case_id: aggregate(items) for case_id, items in by_case.items()}


def score_variant(
    asr: OfflineAsrBackend,
    manifest: dict[str, Any],
    variant_name: str,
    discovered: dict[str, dict[str, Path]],
    *,
    frontend_rtf: float | None = None,
) -> dict[str, Any]:
    """Transcribe and score one channel variant. ``discovered`` must already be fail-closed."""
    index = index_segments(manifest)
    rows: list[IntervalResult] = []
    for seg_id, wav in discovered[variant_name].items():
        rows.append(score_segment(asr, index[seg_id], wav))
    audio_seconds = sum(row.duration_s for row in rows)
    variant = next(v for v in manifest["channel_variants"] if v["name"] == variant_name)
    result: dict[str, Any] = {
        "variant": variant_name,
        "channels": variant["channels"],
        "schema": SCHEMA,
        "asr_model_sha256": manifest.get("asr_model_sha256"),
        "audio_seconds": audio_seconds,
        "segments": len(rows),
        "all": aggregate(rows),
        "non_overlap": aggregate(row for row in rows if not row.overlap_exposed),
        "overlap_exposed": aggregate(row for row in rows if row.overlap_exposed),
        "per_case": aggregate_per_case(rows),
    }
    if frontend_rtf is not None:
        result["frontend_rtf"] = float(frontend_rtf)
    return result


def load_frontend_rtfs(path: Path | None) -> dict[str, float]:
    """Read either an enhance manifest or the legacy {variant: seconds} file."""
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    variants = payload.get("variants") if isinstance(payload, dict) else None
    if isinstance(variants, dict):
        result: dict[str, float] = {}
        for name, record in variants.items():
            if isinstance(record, dict) and record.get("cpu_rtf") is not None:
                result[name] = float(record["cpu_rtf"])
            elif isinstance(record, dict) and record.get("gpu_rtf") is not None:
                result[name] = float(record["gpu_rtf"])
        return result
    if isinstance(payload, dict):
        return {str(name): float(value) for name, value in payload.items()}
    raise ValueError(f"unexpected frontend metadata format: {path}")


def evaluate_gates(
    scores: dict[str, dict[str, Any]],
    *,
    baseline_name: str = "ch0",
    overlap_cer_max: float = 0.30,
    non_overlap_cer_max: float = 0.115,
    rtf_max: float = 1.0,
    cases_improved_min: int = 7,
) -> dict[str, Any]:
    """Apply the approved stage-12 gates using the scored ch0 as baseline."""
    if baseline_name not in scores:
        raise ValueError(f"gate baseline is absent: {baseline_name}")
    baseline_cases = scores[baseline_name]["per_case"]
    variants: dict[str, Any] = {}
    for name, score in scores.items():
        improved = sum(
            1
            for case_id, case_score in score["per_case"].items()
            if case_id in baseline_cases
            and case_score["normalized_cer"] < baseline_cases[case_id]["normalized_cer"]
        )
        checks = {
            "overlap_cer": score["overlap_exposed"]["normalized_cer"] <= overlap_cer_max,
            "non_overlap_cer": score["non_overlap"]["normalized_cer"] <= non_overlap_cer_max,
            "cases_improved": improved >= cases_improved_min,
            "rtf": score.get("frontend_rtf") is not None and score["frontend_rtf"] <= rtf_max,
        }
        variants[name] = {
            "is_baseline": name == baseline_name,
            "cases_improved_vs_baseline": improved,
            "checks": checks,
            "passed": False if name == baseline_name else all(checks.values()),
        }
    return {
        "baseline_variant": baseline_name,
        "thresholds": {
            "overlap_exposed_cer_max": overlap_cer_max,
            "non_overlap_cer_max": non_overlap_cer_max,
            "rtf_max": rtf_max,
            "cases_improved_min": cases_improved_min,
        },
        "variants": variants,
        "passing_candidates": [
            name for name, result in variants.items() if result["passed"]
        ],
    }


def _assert_asr_hash(profile: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = manifest.get("asr_model_sha256")
    actual = profile.get("asr_model_sha256")
    if not actual or actual != expected:
        raise AsrHashMismatchError(
            f"ASR model hash drift: prep manifest has {expected}, source run derived {actual}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True, help="Eval_Ali run dir (run-manifest.json) locking the ASR.")
    parser.add_argument("--prep-manifest", type=Path, required=True, help="P2 prep-manifest.json.")
    parser.add_argument("--gss-root", type=Path, required=True, help="Root dir holding <variant>/<segment_id>.wav.")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "p2_gss_score")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--variant", default="", help="Score only this variant name (debug).")
    parser.add_argument(
        "--gss-meta",
        type=Path,
        default=None,
        help="Optional enhance manifest (preferred) or legacy JSON {variant: rtf}.",
    )
    args = parser.parse_args(argv)

    prep_path = args.prep_manifest.resolve()
    manifest = load_prep_manifest(prep_path)
    if manifest.get("schema") != "eval_ali_gss_prep.v1":
        raise ValueError(f"unexpected prep manifest schema: {manifest.get('schema')}")

    profile, _ = load_source_run(args.source_run.resolve())
    _assert_asr_hash(profile, manifest)

    asr = OfflineAsrBackend()
    capability = asr.capability()
    if capability.status != "ready":
        raise ValueError(f"ASR is not ready: {capability.reason}")

    requested_variants = {args.variant} if args.variant else None
    discovered = discover_gss_outputs(args.gss_root, manifest, requested_variants)
    frontend_rtfs = load_frontend_rtfs(args.gss_meta)

    variant_names = [args.variant] if args.variant else [v["name"] for v in manifest["channel_variants"]]
    scores: dict[str, dict[str, Any]] = {}
    for vname in variant_names:
        if vname not in discovered:
            raise MissingSegmentsError(f"variant {vname} not discovered in gss-root")
        scores[vname] = score_variant(
            asr, manifest, vname, discovered, frontend_rtf=frontend_rtfs.get(vname)
        )

    gate_result = evaluate_gates(scores) if "ch0" in scores else {
        "status": "not_evaluated",
        "reason": "ch0 baseline was not part of this scoped scoring run",
        "passing_candidates": [],
    }

    out_dir = args.out.resolve()
    if args.run_id:
        out_dir = out_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "prep_manifest": str(prep_path),
        "asr_model_sha256": manifest.get("asr_model_sha256"),
        "variants": scores,
        "gate": gate_result,
    }
    (out_dir / "scores.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "scores_json": str(out_dir / "scores.json"),
        "variants": list(scores.keys()),
        "passing_candidates": gate_result["passing_candidates"],
        "asr_model_sha256": manifest.get("asr_model_sha256"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
