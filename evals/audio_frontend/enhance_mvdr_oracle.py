#!/usr/bin/env python3
"""CPU oracle-guided MVDR proxy for stage 12 four-mic spatial enhancement.

This deliberately does not claim to implement ``desh2608/gss``. Official GSS
adds WPE dereverberation and guided CACGMM mask estimation before beamforming.
Here the gold RTTM supplies binary speaker activity, and a compact NumPy MVDR
beamformer estimates one spatial filter per speaker. The experiment therefore
answers the narrower oracle question: "with correct speaker activity, can the
available microphone channels improve recognition?"

The result is an attribution checkpoint and upper-bound proxy. It cannot prove
that automatic masks, continuous speaker tracking, or a future glasses array
will obtain the same gain.

Output layout matches the existing scorer exactly so no scoring code changes:
    <enhance_root>/<variant_name>/<segment_id>.wav   (16k mono PCM16)

Raw PCM never enters the memory store, Timeline, or git.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from prepare_ali_multichannel import CHANNEL_VARIANTS, SAMPLE_RATE, segment_id  # noqa: E402

N_FFT = 512
HOP = 128
METHOD = "oracle-mvdr"
SCHEMA = "eval_ali_mvdr_enhance.v1"


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Manual STFT. x: (C, N) -> (C, T, F) complex (F = n_fft//2 + 1)."""
    C, N = x.shape
    if N < n_fft:
        raise ValueError(f"signal too short for STFT: {N} < {n_fft}")
    n_frames = 1 + (N - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    window = np.hanning(n_fft)
    frames = x[:, idx] * window[None, None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=2)
    return spec  # (C, T, F)


def istft(spec: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Inverse STFT. spec: (T, F) complex -> (N,) float time signal."""
    T, F = spec.shape
    N = n_fft + hop * (T - 1)
    window = np.hanning(n_fft)
    y = np.zeros(N, dtype=np.float64)
    wsum = np.zeros(N, dtype=np.float64)
    for t in range(T):
        frame = np.fft.irfft(spec[t], n=n_fft) * window
        start = hop * t
        y[start:start + n_fft] += frame
        wsum[start:start + n_fft] += window * window
    eps = 1e-8
    y = y / np.where(wsum > eps, wsum, eps)
    return y.astype(np.float32)


def frame_centers(n_samples: int, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Center time (seconds, relative to window start) of each STFT frame."""
    n_frames = 1 + (n_samples - n_fft) // hop
    return (np.arange(n_frames) * hop + n_fft / 2) / SAMPLE_RATE


def build_global_activity(
    segments: list[dict[str, Any]], centers: np.ndarray
) -> dict[str, np.ndarray]:
    """speaker -> bool frame-active array over the whole window."""
    speakers = sorted({s["speaker"] for s in segments})
    active: dict[str, np.ndarray] = {sp: np.zeros(centers.shape[0], dtype=bool) for sp in speakers}
    for seg in segments:
        start = seg["rel_start_s"] if "rel_start_s" in seg else seg["start_s"]
        end = seg["rel_end_s"] if "rel_end_s" in seg else seg["end_s"]
        mask = (centers >= start - 1e-6) & (centers < end - 1e-6)
        active[seg["speaker"]] |= mask
    return active


def spatial_cov(Y: np.ndarray) -> np.ndarray:
    """Y: (C, M, F) -> per-frequency spatial covariance (F, C, C)."""
    Yc = Y.transpose(2, 0, 1)
    M = Yc.shape[1]
    return np.einsum("fcm,fdm->fcd", Yc, Yc.conj()) / M


def mvdr_filter(Rs: np.ndarray, Rn: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Mask-based MVDR filter per frequency. Rs,Rn: (F, C, C) -> w: (F, C)."""
    F, C, _ = Rs.shape
    trace_rn = np.einsum("fcc->f", Rn).real
    Rn = Rn + (reg * trace_rn / C)[:, None, None] * np.eye(C)[None]
    Rn_inv = np.linalg.inv(Rn)
    one = np.ones(C)
    tmp = np.einsum("fcd,fde->fce", Rn_inv, Rs)
    num = np.einsum("fce,e->fc", tmp, one)
    den = np.einsum("c,fc->f", one, num)
    den = np.where(np.abs(den) < 1e-12, 1.0, den)
    return num / den[:, None]


def enhance_segment(
    Zxx: np.ndarray,
    rs_frames: np.ndarray,
    rn_frames: np.ndarray,
    apply_frames: np.ndarray,
    reg: float = 1e-6,
) -> np.ndarray:
    """Enhance one speaker segment. Zxx: (C, T, F). Returns (F, Nt) complex.

    rs_frames: global frames where the target speaks ALONE (solo) -> target
        spatial covariance R_s.
    rn_frames: global frames where the target is NOT active -> noise covariance R_n.
    apply_frames: the segment's own frames to beamform (may be overlapping).
    """
    C = Zxx.shape[0]
    if C == 1:
        # Single-channel variant: pass-through (this IS the single-mic baseline).
        return Zxx[0, apply_frames, :].T
    Ys = Zxx[:, rs_frames, :]    # (C, Ns, F)
    Yn = Zxx[:, rn_frames, :]    # (C, Nn, F)
    if Ys.shape[1] < C or Yn.shape[1] < C:
        # Too few frames for a stable covariance: fall back to the channel with
        # the most energy over the segment (crude but stable; never silently garbage).
        Ya = Zxx[:, apply_frames, :]
        pw = np.sum(np.abs(Ya) ** 2, axis=(1, 2)) if Ya.shape[1] > 0 else np.zeros(C)
        k = int(np.argmax(pw)) if Ya.shape[1] > 0 else 0
        return Zxx[k, apply_frames, :].T
    Rs = spatial_cov(Ys)
    Rn = spatial_cov(Yn)
    w = mvdr_filter(Rs, Rn, reg=reg)
    Ya = Zxx[:, apply_frames, :].transpose(2, 0, 1)  # (F, C, Nt)
    out = np.einsum("fc,fct->ft", w.conj(), Ya)  # (F, Nt)
    return out


def _apply_beamformer(w: np.ndarray, Zxx: np.ndarray, apply_frames: np.ndarray) -> np.ndarray:
    """Apply a precomputed per-frequency MVDR filter to the segment frames."""
    Ya = Zxx[:, apply_frames, :].transpose(2, 0, 1)  # (F, C, Nt)
    return np.einsum("fc,fct->ft", w.conj(), Ya)      # (F, Nt)


# Sentinel for "insufficient frames for a stable covariance -> max-energy channel".
_FALLBACK = object()


def enhance_variant(
    *,
    cut_wav: Path,
    segments: list[dict[str, Any]],
    channels: list[int],
    reg: float,
    activity_segments: list[dict[str, Any]] | None = None,
    output_speaker_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Enhance every segment of one channel variant. Returns segment_id -> (F, Nt).

    The target/noise spatial covariances are estimated once per speaker (they
    depend only on global activity, not on the individual segment), then every
    segment of that speaker reuses the same MVDR filter. This avoids recomputing
    the full-window covariance for all 255 segments.
    """
    with sf.SoundFile(str(cut_wav)) as src:
        if src.samplerate != SAMPLE_RATE:
            raise ValueError(f"{cut_wav}: expected {SAMPLE_RATE} Hz")
        data = src.read(dtype="float32", always_2d=True)  # (N, 8)
    audio = data[:, channels].T  # (C, N)
    Zxx = stft(audio)  # (C, T, F)
    C = Zxx.shape[0]
    centers = frame_centers(audio.shape[1])
    mask_segments = segments if activity_segments is None else activity_segments
    activity = build_global_activity(mask_segments, centers)
    speakers = list(activity.keys())

    # Precompute one MVDR filter per speaker (or mark fallback / pass-through).
    speaker_w: dict[str, Any] = {}
    for sp in speakers:
        other_active = np.zeros(centers.shape[0], dtype=bool)
        for o in speakers:
            if o != sp:
                other_active |= activity[o]
        solo = activity[sp] & ~other_active
        non_target = ~activity[sp]
        if C == 1:
            speaker_w[sp] = None  # single-mic pass-through
            continue
        Ys = Zxx[:, solo, :]
        Yn = Zxx[:, non_target, :]
        if Ys.shape[1] < C or Yn.shape[1] < C:
            speaker_w[sp] = _FALLBACK
        else:
            speaker_w[sp] = mvdr_filter(spatial_cov(Ys), spatial_cov(Yn), reg=reg)

    out: dict[str, np.ndarray] = {}
    for seg in segments:
        sid = segment_id(seg["case_id"], seg["speaker"], seg["rel_start_s"], seg["rel_end_s"])
        sp = output_speaker_map.get(seg["speaker"], "") if output_speaker_map is not None else seg["speaker"]
        apply_frames = (centers >= seg["rel_start_s"] - 1e-6) & (centers < seg["rel_end_s"] - 1e-6)
        if C == 1:
            out[sid] = Zxx[0, apply_frames, :].T
        elif sp not in speaker_w or speaker_w[sp] is _FALLBACK:
            Ya = Zxx[:, apply_frames, :]
            pw = np.sum(np.abs(Ya) ** 2, axis=(1, 2)) if Ya.shape[1] > 0 else np.zeros(C)
            k = int(np.argmax(pw)) if Ya.shape[1] > 0 else 0
            out[sid] = Zxx[k, apply_frames, :].T
        else:
            out[sid] = _apply_beamformer(speaker_w[sp], Zxx, apply_frames)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True,
                        help="P2 prep dir: prep-manifest.json + <session>.wav + <session>.rttm")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "p2_mvdr_enhance")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--variant", default="", help="Enhance only this variant (debug).")
    parser.add_argument("--limit-cases", type=int, default=0, help="Debug: first N cases only.")
    parser.add_argument("--reg", type=float, default=1e-6, help="MVDR diagonal loading.")
    args = parser.parse_args(argv)

    prep_dir = args.prep_dir.resolve()
    manifest = json.loads((prep_dir / "prep-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "eval_ali_gss_prep.v1":
        raise ValueError(f"unexpected prep schema: {manifest.get('schema')}")

    out_root = args.out.resolve()
    if args.run_id:
        out_root = out_root / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    variants = [v for v in CHANNEL_VARIANTS if (not args.variant or v["name"] == args.variant)]
    cases = manifest["cases"]
    if args.limit_cases:
        cases = cases[: args.limit_cases]

    meta: dict[str, Any] = {
        "schema": SCHEMA,
        "method": METHOD,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prep_dir": str(prep_dir),
        "prep_manifest_sha256": manifest.get("prep_manifest_sha256") or manifest.get("per_interval_sha256"),
        "n_fft": N_FFT,
        "hop": HOP,
        "reg": args.reg,
        "external_model": None,
        "license": "algorithm-only (NumPy BSD); no external weights",
        "variants": {},
        "total_segments": 0,
    }

    # Initialize per-variant accumulators once; the case loop below accumulates
    # across all meetings (the old code overwrote per case -> only last case shown).
    for v in variants:
        meta["variants"][v["name"]] = {
            "channels": v["channels"],
            "segments": 0,
            "audio_seconds": 0.0,
            "cpu_seconds": 0.0,
        }

    for case in cases:
        cut_wav = prep_dir / case["cut_wav_path"]
        segs = case["segments"]
        for v in variants:
            t0 = time.perf_counter()
            enhanced = enhance_variant(cut_wav=cut_wav, segments=segs, channels=v["channels"], reg=args.reg)
            cpu_s = time.perf_counter() - t0
            vdir = out_root / v["name"]
            vdir.mkdir(parents=True, exist_ok=True)
            audio_seconds = 0.0
            for sid, spec in enhanced.items():
                audio = istft(spec.T)  # spec (F,Nt) -> (Nt,F).T
                audio_seconds += audio.shape[0] / SAMPLE_RATE
                sf.write(str(vdir / f"{sid}.wav"), audio, SAMPLE_RATE, subtype="PCM_16")
            # Accumulate per-variant stats across all cases (not overwrite).
            vstat = meta["variants"][v["name"]]
            vstat["segments"] += len(enhanced)
            vstat["audio_seconds"] += audio_seconds
            vstat["cpu_seconds"] += cpu_s
            meta["total_segments"] += len(enhanced)
            rtf_c = cpu_s / audio_seconds if audio_seconds > 0 else 0.0
            print(f"variant {v['name']}: {len(enhanced)} segs, cpu_rtf={rtf_c:.3f}")

    # Finalize per-variant RTF after accumulating every case.
    for vstat in meta["variants"].values():
        as_ = vstat["audio_seconds"]
        vstat["cpu_rtf"] = round(vstat["cpu_seconds"] / as_, 4) if as_ > 0 else 0.0
        vstat["audio_seconds"] = round(as_, 2)
        vstat["cpu_seconds"] = round(vstat["cpu_seconds"], 2)

    (out_root / "enhance-manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"enhance_manifest": str(out_root / "enhance-manifest.json"),
                      "total_segments": meta["total_segments"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
