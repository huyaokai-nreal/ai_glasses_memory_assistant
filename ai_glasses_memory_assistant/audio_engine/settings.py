from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioEngineSettings:
    sample_rate: int = 16_000
    frame_samples: int = 512
    recommended_push_samples: int = 4_096
    pre_roll_frames: int = 16
    partial_samples: int = 8_192
    speaker_window_samples: int = 64_000
    ambient_segment_samples: int = 480_000
    wake_query_start_timeout_seconds: float = 10.0
    idle_timeout_seconds: float = 30.0
    playback_timeout_seconds: float = 300.0
    reaper_interval_seconds: float = 5.0
    worklet_flush_timeout_seconds: float = 2.0
    sequence_cache_limit: int = 8
    wake_ack_text: str = "我在"

    def __post_init__(self) -> None:
        if self.sequence_cache_limit < 1:
            raise ValueError("sequence_cache_limit must be at least 1")
        if self.playback_timeout_seconds <= 0:
            raise ValueError("playback_timeout_seconds must be positive")

    @classmethod
    def from_env(cls) -> "AudioEngineSettings":
        wake_ack_text = str(os.getenv("AI_GLASSES_WAKE_ACK_TEXT") or "").strip()
        return cls(wake_ack_text=wake_ack_text or cls.wake_ack_text)


DEFAULT_AUDIO_SETTINGS = AudioEngineSettings()
