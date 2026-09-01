#!/usr/bin/env python3
"""Score on-device Eval_Ali offline-audio exports against the desktop eval's gold.

The Android Settings "offline audio test" runs the real device-side VAD/ASR
stack (Ten VAD + 2025 SenseVoice int8) over a pushed WAV and exports a debug
JSON with schema `android_eval_ali_events.v1`. That export only carries final
segment boundaries + transcript text (never PCM or raw scores), so we cannot
recompute the desktop pipeline end-to-end on-device. But we CAN verify the one
thing that matters for the candidate pack flip:

  1. The v2 pack actually loaded on-device (model_pack_version + vad_backend).
  2. The on-device VAD segments, scored with the SAME frame-level logic and the
     SAME gold TextGrids as the desktop eval, produce a recall/precision/F1 in
     the same ballpark as the desktop E-fix run (~0.92 far recall). A large
     drop means the device is NOT applying the tuned threshold (still on 0.5).

This reuses eval_ali.discover_cases (gold + smoke-window offsets) and
eval_ali.score_vad verbatim, so the number is directly comparable to the
desktop gate.

Usage:
  # Score one or more on-device exports copied from the phone:
  python scripts/score_android_eval_ali.py export-R8001_M8004-smoke.json [...]
  python scripts/score_android_eval_ali.py --dir ./android-exports

  # Validate the gold/offset plumbing without a device (self-check):
  python scripts/score_android_eval_ali.py --selftest
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_glasses_memory_assistant.evals import eval_ali as eval_ali_mod  # noqa: E402
from ai_glasses_memory_assistant.evals.eval_ali import (  # noqa: E402
    discover_cases,
    score_vad,
)

# discover_cases() needs wav_duration_seconds(), which eval_ali imports lazily
# from soundfile. soundfile may be absent in some envs; these are plain PCM16
# WAVs, so the stdlib `wave` module is a drop-in replacement (no network/conda).
def _wav_duration_seconds(path):  # type: ignore[no-untyped-def]
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            raise ValueError(f"invalid WAV framerate in {path}")
        return n_frames / float(rate)


eval_ali_mod.wav_duration_seconds = _wav_duration_seconds

SCHEMA = "android_eval_ali_events.v1"
PRESET = "smoke"
GOLD_ROOT = ROOT / "data" / "Eval_Ali"


def _case_map() -> dict[str, Any]:
    cases = discover_cases(GOLD_ROOT, PRESET)
    if len(cases) != 8:
        raise RuntimeError(f"discover_cases returned {len(cases)} smoke cases, expected 8")
    return {c.case_id: c for c in cases}


def load_export(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path.name}: unexpected schema {payload.get('schema')!r} (want {SCHEMA})")
    return payload


def score_export(payload: dict[str, Any], cases: dict[str, Any]) -> dict[str, Any]:
    case_id = payload["case_id"]
    case = cases.get(case_id)
    if case is None:
        raise KeyError(f"export case_id {case_id!r} not found in smoke manifest")
    detected = [
        (float(e["start_ms"]) / 1000.0 + case.start_s, float(e["end_ms"]) / 1000.0 + case.start_s)
        for e in payload.get("events", [])
    ]
    sc = score_vad(case.reference_intervals, detected, start_s=case.start_s, end_s=case.end_s)
    profile = payload.get("model_profile", {})
    return {
        "case_id": case_id,
        "model_pack_version": profile.get("model_pack_version", "?"),
        "vad_backend": profile.get("vad_backend", "?"),
        "n_detected_segments": len(detected),
        "recall": sc["recall"],
        "precision": sc["precision"],
        "f1": sc["f1"],
        "missed_seconds": sc["missed_seconds"],
        "false_trigger_seconds": sc["false_trigger_seconds"],
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = ["case_id", "pack", "vad", "segs", "recall", "prec", "f1"]
    widths = [22, 26, 9, 6, 8, 8, 8]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    recalls, precs, f1s = [], [], []
    for r in rows:
        pack = str(r["model_pack_version"])
        pack = pack if len(pack) <= 26 else pack[:23] + "..."
        vals = [
            r["case_id"],
            pack,
            str(r["vad_backend"]),
            str(r["n_detected_segments"]),
            f"{r['recall']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['f1']:.3f}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))
        recalls.append(r["recall"])
        precs.append(r["precision"])
        f1s.append(r["f1"])
    if recalls:
        print("-" * len(line))
        macro = [
            "MACRO-AVG",
            "",
            "",
            "",
            f"{statistics.fmean(recalls):.3f}",
            f"{statistics.fmean(precs):.3f}",
            f"{statistics.fmean(f1s):.3f}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(macro, widths)))
        print()
        print("Interpretation:")
        print(f"  far-field recall target (desktop E-fix) ~0.92; if on-device recall")
        print(f"  is materially lower (e.g. <0.85) the device is likely NOT applying")
        print(f"  the tuned threshold=0.35 (still on 0.5) -> re-check the installed pack.")


def _selftest() -> int:
    """Validate gold load + window offset + score_vad plumbing without a device.

    Builds a synthetic export per case where detected == gold (expect ~1.0/1.0)
    and one where detected is empty (expect recall 0). Proves the absolute-time
    alignment is correct before trusting any real on-device number.
    """
    cases = _case_map()
    print(f"[selftest] loaded {len(cases)} smoke cases from {GOLD_ROOT}")
    ok = True
    for case_id, case in cases.items():
        gold = [(i.start_s, i.end_s) for i in case.reference_intervals]
        # gold-as-detected should score ~1.0 recall/precision
        sc_gold = score_vad(case.reference_intervals, gold, start_s=case.start_s, end_s=case.end_s)
        sc_empty = score_vad(case.reference_intervals, [], start_s=case.start_s, end_s=case.end_s)
        good = abs(sc_gold["recall"] - 1.0) < 1e-6 and abs(sc_gold["precision"] - 1.0) < 1e-6 and sc_empty["recall"] == 0.0
        ok = ok and good
        print(f"  {case_id:24s} gold-as-detected recall={sc_gold['recall']:.3f} "
              f"prec={sc_gold['precision']:.3f} | empty recall={sc_empty['recall']:.3f} "
              f"{'OK' if good else 'FAIL'}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exports", nargs="*", help="android_eval_ali_events.v1 JSON files")
    parser.add_argument("--dir", type=Path, help="directory to glob for *.json exports")
    parser.add_argument("--selftest", action="store_true", help="validate gold/offset plumbing, no device needed")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    paths: list[Path] = []
    for p in args.exports:
        paths.append(Path(p))
    if args.dir:
        paths.extend(sorted(args.dir.glob("*.json")))
    if not paths:
        parser.error("provide export JSON paths, --dir, or --selftest")

    cases = _case_map()
    rows: list[dict[str, Any]] = []
    for p in paths:
        payload = load_export(p)
        try:
            rows.append(score_export(payload, cases))
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR scoring {p.name}: {exc}", file=sys.stderr)

    if not rows:
        print("No exports scored.", file=sys.stderr)
        return 2
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
