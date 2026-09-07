"""Unit tests for the CPU oracle-MVDR beamformer (stage 12, GPU-free).

Pure synthetic signals -- no AliMeeting data, no ASR model. Verifies the
spatial separation actually attenuates an off-target directional interferer
compared with the single-channel baseline, plus the single-channel pass-through
and the insufficient-frames fallback.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "evals" / "audio_frontend"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from enhance_mvdr_oracle import (  # noqa: E402
    SAMPLE_RATE,
    enhance_segment,
    frame_centers,
    istft,
    stft,
)


def _make_directional_mix(fs=SAMPLE_RATE, tau_samples=4):
    """2-ch, 1.5s mix with two directional tones.

    Speaker A (f0=2000) active [0,1.0)s; speaker B (f1=3000) active [0.5,1.5)s.
    So [0,0.5)s is A solo, [0.5,1.0)s is overlap, [1.0,1.5)s is B solo.
    Both tones come from different directions (B delayed by tau_samples on ch1).
    Returns (audio (2,N), f0, f1, centers, activity, any_active).
    """
    total = int(1.5 * fs)
    t = np.arange(total) / fs
    f0, f1 = 2000.0, 3000.0
    tone_tgt = np.sin(2 * np.pi * f0 * t)
    tone_inf = np.sin(2 * np.pi * f1 * t)
    ch0 = np.zeros(total, dtype=np.float32)
    ch1 = np.zeros(total, dtype=np.float32)
    # A solo + overlap (A active [0,1.0)s, angle 0 -> no delay)
    ch0[:2 * total // 3] += tone_tgt[:2 * total // 3]
    ch1[:2 * total // 3] += tone_tgt[:2 * total // 3]
    # B overlap + B solo (B active [0.5,1.5)s, delayed on ch1)
    b = total // 3
    ch0[b:] += tone_inf[b:]
    ch1[b:] += np.sin(2 * np.pi * f1 * (t[b:] - tau_samples / fs))
    audio = np.stack([ch0, ch1], axis=0)  # (2, N)
    centers = frame_centers(total)
    activity = {
        "A": (centers >= 0.0) & (centers < 1.0),
        "B": (centers >= 0.5) & (centers < 1.5),
    }
    any_active = activity["A"] | activity["B"]
    return audio, f0, f1, centers, activity, any_active


def _band_power(sig: np.ndarray, freq: float, fs=SAMPLE_RATE) -> float:
    m = max(200, len(sig) // 8)
    sig = sig[m:len(sig) - m] if len(sig) > 2 * m else sig
    spec = np.abs(np.fft.rfft(sig))
    k = int(round(freq / fs * sig.shape[0]))
    return float(spec[k])


def test_mvdr_suppresses_offaxis_interferer():
    audio, f0, f1, centers, activity, any_active = _make_directional_mix()
    Zxx = stft(audio)  # (2, T, F)
    # Segment = overlap region of speaker A.
    apply = (centers >= 0.5) & (centers < 1.0)
    rs_frames = activity["A"] & ~activity["B"]   # A solo (other speaker B silent)
    rn_frames = ~activity["A"]                   # A silent (B solo / silence)
    enhanced = enhance_segment(Zxx, rs_frames, rn_frames, apply).T  # (Nt, F)
    enhanced_t = istft(enhanced)

    n_frames = Zxx.shape[1]
    ov_idx = np.where(apply)[0]
    start = ov_idx[0] * 128
    end = (ov_idx[-1] + 1) * 128
    single = audio[0, start:end]

    enh_f1 = _band_power(enhanced_t, f1)
    single_f1 = _band_power(single, f1)
    enh_f0 = _band_power(enhanced_t, f0)
    single_f0 = _band_power(single, f0)

    # Interferer must be strongly attenuated vs single channel.
    assert enh_f1 < 0.5 * single_f1, f"interferer not suppressed: {enh_f1} vs {single_f1}"
    # Target must be preserved (not killed along with the interferer).
    assert enh_f0 > 0.5 * single_f0, f"target destroyed: {enh_f0} vs {single_f0}"


def test_single_channel_passthrough():
    # C=1 -> enhance_segment returns the channel unchanged (single-mic baseline).
    audio, f0, f1, centers, activity, any_active = _make_directional_mix()
    Zxx = stft(audio[:1, :])  # (1, T, F)
    apply = (centers >= 0.5) & (centers < 1.0)
    rs_frames = activity["A"] & ~activity["B"]
    rn_frames = ~activity["A"]
    out = enhance_segment(Zxx, rs_frames, rn_frames, apply).T  # (Nt, F)
    rec = istft(out)

    ov_idx = np.where(apply)[0]
    start = ov_idx[0] * 128
    end = (ov_idx[-1] + 1) * 128
    expected = audio[0, start:end]
    rec_p = _band_power(rec, f0)
    exp_p = _band_power(expected, f0)
    assert rec_p > 0.7 * exp_p, f"pass-through changed target power: {rec_p} vs {exp_p}"


def test_insufficient_frames_fallback():
    # Degenerate masks -> no crash, returns a channel-shaped spectrum.
    rng = np.random.default_rng(0)
    Zxx = rng.standard_normal((2, 20, 257)) + 1j * rng.standard_normal((2, 20, 257))
    rs = np.zeros(20, dtype=bool); rs[:1] = True
    rn = np.zeros(20, dtype=bool); rn[:1] = True
    apply = np.zeros(20, dtype=bool); apply[:2] = True
    out = enhance_segment(Zxx, rs, rn, apply)
    assert out.shape == (257, 2)
