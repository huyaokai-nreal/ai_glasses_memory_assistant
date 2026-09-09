"""Post-run artifact verification for continuous frontend outputs.

This is NOT a replacement for pytest. It verifies the *actual files on disk*
produced by a real run:

1. every segment: wav_frames == sample_count == round((end_s-start_s)*16000)
2. every segment: wav_sha256 matches the file content
3. per-track interval integrity: no overlap, no out-of-range, ascending start
4. block boundaries of the whole run: for every 30 s block edge (and the last
   block end) show the neighbouring same-track segments on both sides and assert
   they are disjoint. A full meeting must not only check 30 s / 60 s.
5. last block: no segment may end past the processed window; the trailing
   segment of every (variant, track) is reported.

Usage:
    python verify_frontend_wavs.py <frontend-run-dir> [limit-seconds]

Exit codes: 0 all checks passed; 1 at least one problem; 2 usage/input error.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf

SAMPLE_RATE = 16000
FRAME_TOL = 0  # exact equality required
BLOCK_SECONDS = 30.0


def block_boundaries(window_seconds: float | None) -> list[float]:
    """Every 30 s block edge inside the window, plus the last block end.

    A 75 s smoke run has 30 / 60 / 75; a full 1573.85 s meeting has 30, 60, ...
    1560 and 1573.85. Checking only the first two would let a late-block defect
    through, so the whole run is enumerated.
    """
    if not window_seconds:
        return []
    edges = []
    index = 1
    while BLOCK_SECONDS * index < window_seconds - 1e-9:
        edges.append(round(BLOCK_SECONDS * index, 6))
        index += 1
    edges.append(round(float(window_seconds), 6))
    return edges


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(run_dir: Path, limit_seconds: float | None) -> int:
    segments_path = run_dir / "segments.jsonl"
    manifest_path = run_dir / "frontend-manifest.json"
    if not segments_path.exists():
        print(f"FAIL: missing {segments_path}")
        return 2

    rows = [json.loads(line) for line in segments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    # The processed window from the manifest is authoritative; the CLI argument
    # is only a fallback. Without it a full meeting would skip every range and
    # last-block check.
    case = manifest.get("case") or {}
    window_seconds = limit_seconds
    if window_seconds is None:
        try:
            window_seconds = float(case["processed_seconds"])
        except (KeyError, TypeError, ValueError):
            window_seconds = None

    print(f"run_dir            : {run_dir}")
    print(f"segments           : {len(rows)}")
    print(f"window_seconds     : {window_seconds} (source: {'--limit-seconds' if limit_seconds is not None else 'manifest case.processed_seconds'})")
    print(f"manifest segments_jsonl_sha256: {manifest.get('segments_jsonl_sha256', 'MISSING')}")
    print()

    failures: list[str] = []

    # ---- 1 + 2: exact frame count and WAV hash -------------------------------
    for row in rows:
        sid = row["segment_id"]
        wav = run_dir / row["wav_path"]
        if not wav.exists():
            failures.append(f"{sid}: missing wav {wav}")
            continue
        info = sf.info(str(wav))
        frames = int(info.frames)
        declared = int(row.get("sample_count", -1))
        expected = int(round((float(row["end_s"]) - float(row["start_s"])) * SAMPLE_RATE))
        if not (frames == declared == expected):
            failures.append(
                f"{sid}: frames={frames} sample_count={declared} expected={expected} "
                f"(start={row['start_s']} end={row['end_s']})"
            )
        if info.samplerate != SAMPLE_RATE:
            failures.append(f"{sid}: samplerate {info.samplerate} != {SAMPLE_RATE}")

    bad_frames = len([f for f in failures if "frames=" in f or "missing wav" in f])
    print(f"[1/5] exact frame count: {len(rows) - bad_frames}/{len(rows)} ok")

    hash_bad = 0
    for row in rows:
        wav = run_dir / row["wav_path"]
        declared_hash = row.get("wav_sha256")
        if not declared_hash:
            failures.append(f"{row['segment_id']}: missing wav_sha256 in segments.jsonl")
            continue
        if not wav.exists():
            continue
        if sha256_file(wav) != declared_hash:
            hash_bad += 1
            failures.append(f"{row['segment_id']}: wav_sha256 drift")
    print(f"[2/5] wav sha256 match : {len(rows) - hash_bad}/{len(rows)} ok")

    # ---- 3: per-track interval integrity ------------------------------------
    by_track: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_track.setdefault((row["variant"], row["track"]), []).append(row)

    overlap_bad = 0
    range_bad = 0
    for key, items in by_track.items():
        items.sort(key=lambda r: (float(r["start_s"]), float(r["end_s"])))
        prev_end = None
        for row in items:
            start = float(row["start_s"])
            end = float(row["end_s"])
            if window_seconds is not None and (start < -1e-9 or end > window_seconds + 1e-9):
                range_bad += 1
                failures.append(f"{row['segment_id']}: out of range [{start}, {end}] vs window {window_seconds}")
            if start >= end:
                range_bad += 1
                failures.append(f"{row['segment_id']}: non-positive duration [{start}, {end}]")
            if prev_end is not None and start < prev_end - 1e-9:
                overlap_bad += 1
                failures.append(
                    f"{row['segment_id']}: overlaps previous same-track segment "
                    f"(prev_end={prev_end}, start={start})"
                )
            prev_end = max(prev_end or end, end)
    total_pairs = sum(len(v) for v in by_track.values())
    print(
        f"[3/5] track intervals  : overlap_bad={overlap_bad} range_bad={range_bad} "
        f"across {total_pairs} segments / {len(by_track)} (variant,track) groups"
    )

    # ---- 4: every block boundary of the whole run ---------------------------
    edges = block_boundaries(window_seconds)
    print(f"[4/5] block boundary inspection: {len(edges)} edges "
          f"({'all 30s edges + last block end' if edges else 'window unknown'})")
    total_shown = 0
    for position, boundary in enumerate(edges):
        shown = 0
        detail: list[str] = []
        for key, items in sorted(by_track.items()):
            items.sort(key=lambda r: (float(r["start_s"]), float(r["end_s"])))
            for idx, row in enumerate(items):
                start = float(row["start_s"])
                end = float(row["end_s"])
                crosses = start < boundary < end
                touches = abs(end - boundary) < 1e-6 or abs(start - boundary) < 1e-6
                if not (crosses or touches):
                    continue
                prev_row = items[idx - 1] if idx > 0 else None
                next_row = items[idx + 1] if idx + 1 < len(items) else None
                head = f"    {key[0]}/{key[1]}: [{start:.3f}, {end:.3f})"
                head += f" prev_end={float(prev_row['end_s']):.3f}" if prev_row else " prev_end=None"
                head += f" next_start={float(next_row['start_s']):.3f}" if next_row else " next_start=None"
                detail.append(head)
                if prev_row and float(prev_row["end_s"]) > start + 1e-9:
                    failures.append(f"{row['segment_id']}: boundary overlap with previous segment at {boundary}s")
                if next_row and end > float(next_row["start_s"]) + 1e-9:
                    failures.append(f"{row['segment_id']}: boundary overlap with next segment at {boundary}s")
                shown += 1
        total_shown += shown
        # Keep the log readable for a full meeting: full detail for the first and
        # last two edges, a one-line count for everything in between.
        if position < 2 or position >= len(edges) - 2:
            print(f"  --- boundary {boundary:.3f}s ({shown} adjacent segments) ---")
            for line in detail:
                print(line)
            if shown == 0:
                print("    (no segment crosses or touches this edge)")
        else:
            print(f"  boundary {boundary:.3f}s: {shown} adjacent segment(s), disjointness checked")
    print(f"  total adjacent segments inspected across all edges: {total_shown}")

    # ---- 5: last block ------------------------------------------------------
    print("[5/5] last block / window tail:")
    if window_seconds is None:
        print("    SKIPPED: window length unknown")
    else:
        max_end = 0.0
        tail_bad = 0
        for key, items in sorted(by_track.items()):
            items.sort(key=lambda r: (float(r["start_s"]), float(r["end_s"])))
            last_end = float(items[-1]["end_s"])
            max_end = max(max_end, last_end)
            tail_state = "ok" if last_end <= window_seconds + 1e-9 else "OVERRUN"
            if tail_state != "ok":
                tail_bad += 1
                failures.append(f"{key[0]}/{key[1]}: last segment ends at {last_end} > window {window_seconds}")
            print(f"    {key[0]}/{key[1]}: last segment [{float(items[-1]['start_s']):.3f}, {last_end:.3f}) -> {tail_state}")
        print(f"    window={window_seconds:.3f}s max_segment_end={max_end:.3f}s overrun_groups={tail_bad}")
        print(f"    segments ending exactly at the window: "
              f"{sum(1 for r in rows if abs(float(r['end_s']) - window_seconds) < 1e-6)}")

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} problems)")
        for item in failures[:40]:
            print(f"  - {item}")
        if len(failures) > 40:
            print(f"  ... {len(failures) - 40} more")
        return 1
    print("RESULT: PASS (WAV frames/counts/hashes, per-track intervals, all block boundaries and last block verified)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: verify_frontend_wavs.py <frontend-run-dir> [limit-seconds]")
        raise SystemExit(2)
    run_dir = Path(sys.argv[1]).resolve()
    limit = float(sys.argv[2]) if len(sys.argv) > 2 else None
    raise SystemExit(main(run_dir, limit))
