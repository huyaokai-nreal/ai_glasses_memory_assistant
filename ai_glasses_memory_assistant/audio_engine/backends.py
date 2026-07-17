from __future__ import annotations

import importlib.util
import os
import tempfile
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .settings import AudioEngineSettings, DEFAULT_AUDIO_SETTINGS


SAMPLE_RATE = DEFAULT_AUDIO_SETTINGS.sample_rate
FRAME_SAMPLES = DEFAULT_AUDIO_SETTINGS.frame_samples


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("text") or "").strip()
    if isinstance(value, list):
        return "".join(_extract_text(item) for item in value).strip()
    return ""


def _extract_embedding(value: Any) -> tuple[float, ...]:
    if isinstance(value, dict):
        for key in ("spk_embedding", "embedding", "value"):
            if key in value:
                return _extract_embedding(value[key])
    if isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], dict):
            for item in value:
                result = _extract_embedding(item)
                if result:
                    return result
        if isinstance(value[0], (list, tuple)):
            return _extract_embedding(value[0])
        try:
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return ()
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0 or not np.isfinite(norm):
            return ()
        return tuple(float(item) for item in vector / norm)
    if hasattr(value, "detach"):
        try:
            return _extract_embedding(value.detach().cpu().numpy().reshape(-1).tolist())
        except Exception:
            return ()
    return ()


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if not left or len(left) != len(right):
        return None
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= 0.0:
        return None
    return float(np.dot(left_array, right_array) / denominator)


@dataclass(frozen=True)
class BackendCapability:
    status: str
    backend: str
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "backend": self.backend, "reason": self.reason}


@dataclass(frozen=True)
class VadTransition:
    active: bool
    state: str
    start_sample: int
    end_sample: int

    def __bool__(self) -> bool:
        return self.active


class StableVad:
    """Session-scoped Silero VAD with an explicit energy fallback."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        model: Any = None,
        lock: threading.Lock | None = None,
        load_model: bool = True,
    ) -> None:
        self.threshold = threshold
        self._recent: deque[bool] = deque(maxlen=5)
        self._stable = False
        self.backend = "energy_fallback"
        self.reason = "silero_vad_unavailable"
        self._iterator: Any = None
        self._lock = lock or threading.Lock()
        self._processed_samples = 0
        if model is None and not load_model:
            return
        try:
            from silero_vad import VADIterator, load_silero_vad

            self._iterator = VADIterator(
                model if model is not None else load_silero_vad() if load_model else None,
                threshold=threshold,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=100,
                speech_pad_ms=30,
            )
            self.backend = "silero_vad"
            self.reason = "ready"
        except Exception:
            self._iterator = None

    def accept(self, frame: np.ndarray) -> VadTransition:
        was_stable = self._stable
        if self._iterator is not None:
            import torch

            with self._lock:
                self._iterator(torch.tensor(frame, dtype=torch.float32), return_seconds=True)
                raw = bool(self._iterator.triggered)
        else:
            rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64))) if frame.size else 0.0
            raw = rms >= 0.012
        self._recent.append(raw)
        ratio = sum(self._recent) / len(self._recent)
        if not self._stable and ratio >= 0.6:
            self._stable = True
        elif self._stable and ratio <= 0.4:
            self._stable = False
        start_sample = self._processed_samples
        self._processed_samples += int(frame.size)
        state = "speech_frame"
        if self._stable and not was_stable:
            state = "speech_start"
        elif was_stable and not self._stable:
            state = "speech_end"
        elif not self._stable:
            state = "silence"
        return VadTransition(
            active=self._stable,
            state=state,
            start_sample=start_sample,
            end_sample=self._processed_samples,
        )

    def reset(self) -> None:
        self._recent.clear()
        self._stable = False
        self._processed_samples = 0
        if self._iterator is not None and hasattr(self._iterator, "reset_states"):
            with self._lock:
                self._iterator.reset_states()


class VadFactory:
    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()

    def capability(self) -> BackendCapability:
        if importlib.util.find_spec("silero_vad") is None:
            return BackendCapability("degraded", "silero_vad", "energy_fallback")
        return BackendCapability("ready", "silero_vad", "configured")

    def create(self) -> StableVad:
        if self.capability().status != "ready":
            return StableVad(load_model=False)
        with self._lock:
            if self._model is None:
                from silero_vad import load_silero_vad

                self._model = load_silero_vad()
        return StableVad(model=self._model, lock=self._lock, load_model=False)


class StreamingAsrBackend:
    def __init__(self) -> None:
        self.model_dir = str(os.getenv("AI_GLASSES_STREAMING_ASR_MODEL_DIR") or "").strip()
        self._model: Any = None
        self._lock = threading.Lock()

    def capability(self) -> BackendCapability:
        if not self.model_dir:
            return BackendCapability("unavailable", "funasr_paraformer_streaming", "model_dir_missing")
        if not Path(self.model_dir).exists():
            return BackendCapability("unavailable", "funasr_paraformer_streaming", "model_dir_not_found")
        if importlib.util.find_spec("funasr") is None:
            return BackendCapability("unavailable", "funasr_paraformer_streaming", "funasr_not_installed")
        return BackendCapability("ready", "funasr_paraformer_streaming", "configured")

    def transcribe(self, audio: np.ndarray, *, cache: dict[str, Any], is_final: bool) -> str:
        if audio.size == 0 or self.capability().status != "ready":
            return ""
        model = self._get_model()
        with self._lock:
            result = model.generate(
                input=np.asarray(audio, dtype=np.float32),
                cache=cache,
                is_final=is_final,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
            )
        return _extract_text(result)

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is None:
                from funasr import AutoModel

                self._model = AutoModel(
                    model=self.model_dir,
                    trust_remote_code=False,
                    disable_update=True,
                    device="cpu",
                )
            return self._model


class OfflineAsrBackend:
    def __init__(self, *, runner: Any = None, lock: Any = None) -> None:
        self.model_dir = str(os.getenv("AI_GLASSES_ASR_MODEL_DIR") or "").strip()
        self._model: Any = None
        self._runner = runner
        self._lock = lock or threading.RLock()

    def capability(self) -> BackendCapability:
        if not self.model_dir:
            return BackendCapability("unavailable", "sensevoice", "model_dir_missing")
        if not Path(self.model_dir).exists():
            return BackendCapability("unavailable", "sensevoice", "model_dir_not_found")
        if importlib.util.find_spec("funasr") is None:
            return BackendCapability("unavailable", "sensevoice", "funasr_not_installed")
        return BackendCapability("ready", "sensevoice", "configured")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0 or self.capability().status != "ready":
            return ""
        if self._runner is not None:
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file, self._lock:
                sf.write(temp_file.name, np.asarray(audio, dtype=np.float32), SAMPLE_RATE)
                result = self._runner.transcribe_file(Path(temp_file.name))
            return str(result.transcript if result.ok else "").strip()
        if self._model is None:
            from funasr import AutoModel

            self._model = AutoModel(
                model=self.model_dir,
                trust_remote_code=False,
                disable_update=True,
                device="cpu",
            )
        with self._lock:
            result = self._model.generate(
                input=np.asarray(audio, dtype=np.float32),
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
        text = _extract_text(result)
        import re

        return " ".join(re.sub(r"<\|[^|>]+\|>", " ", text).split()).strip()


@dataclass(frozen=True)
class SpeakerAnalysis:
    embedding: tuple[float, ...] = ()
    model_name: str = ""
    reason: str = "speaker_unavailable"


class SpeakerBackend:
    def __init__(self, *, runner: Any = None, lock: Any = None) -> None:
        self.model_dir = str(os.getenv("AI_GLASSES_SPEAKER_MODEL_DIR") or "").strip()
        self._model: Any = None
        self._runner = runner
        self._lock = lock or threading.RLock()

    def capability(self) -> BackendCapability:
        if not self.model_dir:
            return BackendCapability("unavailable", "campp", "model_dir_missing")
        if not Path(self.model_dir).exists():
            return BackendCapability("unavailable", "campp", "model_dir_not_found")
        if importlib.util.find_spec("funasr") is None:
            return BackendCapability("unavailable", "campp", "funasr_not_installed")
        return BackendCapability("ready", "campp", "configured")

    def analyze(self, audio: np.ndarray) -> SpeakerAnalysis:
        if audio.size == 0 or self.capability().status != "ready":
            return SpeakerAnalysis(reason=self.capability().reason)
        if self._runner is not None:
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file, self._lock:
                sf.write(temp_file.name, np.asarray(audio, dtype=np.float32), SAMPLE_RATE)
                result = self._runner.analyze_file(Path(temp_file.name))
            return SpeakerAnalysis(
                embedding=tuple(float(item) for item in result.embedding),
                model_name=str(result.model_name or ""),
                reason=("speaker_embedding_extracted" if result.embedding else result.error_type or result.evidence),
            )
        if self._model is None:
            from funasr import AutoModel

            self._model = AutoModel(
                model=self.model_dir,
                trust_remote_code=False,
                disable_update=True,
                device="cpu",
            )
        with self._lock:
            result = self._model.generate(input=np.asarray(audio, dtype=np.float32), batch_size_s=60)
        embedding = _extract_embedding(result)
        return SpeakerAnalysis(
            embedding=embedding,
            model_name=Path(self.model_dir).name,
            reason="speaker_embedding_extracted" if embedding else "speaker_embedding_missing",
        )


class KeywordSpotterSession:
    def __init__(
        self,
        model: Any = None,
        stream: Any = None,
        *,
        lock: threading.Lock | None = None,
        reason: str = "kws_unavailable",
    ) -> None:
        self.model = model
        self.stream = stream
        self._lock = lock or threading.Lock()
        self.reason = reason

    def accept(self, audio: np.ndarray) -> str:
        if self.model is None or self.stream is None or audio.size == 0:
            return ""
        with self._lock:
            self.stream.accept_waveform(SAMPLE_RATE, np.asarray(audio, dtype=np.float32))
            while self.model.is_ready(self.stream):
                self.model.decode_stream(self.stream)
                keyword = str(self.model.get_result(self.stream) or "").strip()
                if keyword:
                    self._reset_unlocked()
                    return keyword
        return ""

    def reset(self) -> None:
        with self._lock:
            self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        if self.model is None:
            self.stream = None
            return
        if self.stream is not None and hasattr(self.model, "reset_stream"):
            self.model.reset_stream(self.stream)
        else:
            self.stream = self.model.create_stream()


class KeywordSpotterFactory:
    def __init__(self, *, settings: AudioEngineSettings | None = None) -> None:
        self.settings = settings or DEFAULT_AUDIO_SETTINGS
        self.model_dir = str(os.getenv("AI_GLASSES_KWS_MODEL_DIR") or "").strip()
        self.keywords_file = str(os.getenv("AI_GLASSES_KWS_KEYWORDS_FILE") or "").strip()
        self._model: Any = None
        self._lock = threading.Lock()

    def _paths(self) -> dict[str, Path]:
        root = Path(self.model_dir) if self.model_dir else Path()
        patterns = {
            "tokens": "tokens.txt",
            "encoder": "encoder*.onnx",
            "decoder": "decoder*.onnx",
            "joiner": "joiner*.onnx",
        }
        paths: dict[str, Path] = {}
        if not self.model_dir or not root.exists():
            return paths
        for key, pattern in patterns.items():
            matches = sorted(root.rglob(pattern))
            if matches:
                paths[key] = matches[0]
        keyword_path = Path(self.keywords_file) if self.keywords_file else None
        if keyword_path is not None and keyword_path.exists():
            paths["keywords"] = keyword_path
        return paths

    def capability(self) -> BackendCapability:
        if importlib.util.find_spec("sherpa_onnx") is None:
            return BackendCapability("degraded", "sherpa_onnx_kws", "sherpa_onnx_not_installed")
        paths = self._paths()
        if set(paths) != {"tokens", "encoder", "decoder", "joiner", "keywords"}:
            return BackendCapability("degraded", "sherpa_onnx_kws", "model_or_keywords_missing")
        return BackendCapability("ready", "sherpa_onnx_kws", "configured")

    def create(self) -> KeywordSpotterSession:
        if self.capability().status != "ready":
            return KeywordSpotterSession(reason=self.capability().reason)
        paths = self._paths()
        with self._lock:
            if self._model is None:
                import sherpa_onnx

                self._model = sherpa_onnx.KeywordSpotter(
                    tokens=str(paths["tokens"]),
                    encoder=str(paths["encoder"]),
                    decoder=str(paths["decoder"]),
                    joiner=str(paths["joiner"]),
                    keywords_file=str(paths["keywords"]),
                    num_threads=2,
                    sample_rate=self.settings.sample_rate,
                    feature_dim=80,
                    max_active_paths=4,
                    keywords_score=1.0,
                    keywords_threshold=0.25,
                    num_trailing_blanks=1,
                    provider="cpu",
                )
        return KeywordSpotterSession(
            self._model,
            self._model.create_stream(),
            lock=self._lock,
            reason="ready",
        )


@dataclass
class AudioBackendRegistry:
    settings: AudioEngineSettings = field(default_factory=AudioEngineSettings.from_env)
    streaming_asr: Any = None
    offline_asr: Any = None
    speaker: Any = None
    kws: Any = None
    vad: Any = None
    offline_adapter: Any = field(default=None, repr=False)
    _offline_asr_lock: Any = field(default_factory=threading.RLock, repr=False)
    _speaker_lock: Any = field(default_factory=threading.RLock, repr=False)
    _emotion_lock: Any = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if self.offline_adapter is None:
            from .offline import AudioSegmentProcessor

            self.offline_adapter = AudioSegmentProcessor()
        self.streaming_asr = self.streaming_asr or StreamingAsrBackend()
        self.offline_asr = self.offline_asr or OfflineAsrBackend(
            runner=self.offline_adapter.asr_runner,
            lock=self._offline_asr_lock,
        )
        self.speaker = self.speaker or SpeakerBackend(
            runner=self.offline_adapter.speaker_runner,
            lock=self._speaker_lock,
        )
        self.kws = self.kws or KeywordSpotterFactory(settings=self.settings)
        self.vad = self.vad or VadFactory()

    def process_offline(self, **kwargs: Any) -> Any:
        with self._offline_asr_lock, self._speaker_lock, self._emotion_lock:
            return self.offline_adapter.process(**kwargs)

    def prepare_offline_audio(self, **kwargs: Any) -> Any:
        return self.offline_adapter._prepare_audio_segment(**kwargs)

    def analyze_speaker_file(self, path: Path) -> Any:
        with self._speaker_lock:
            return self.offline_adapter.speaker_runner.analyze_file(path)

    def capabilities(self) -> dict[str, dict[str, str]]:
        return {
            "vad": self.vad.capability().to_dict(),
            "kws": self.kws.capability().to_dict(),
            "streaming_asr": self.streaming_asr.capability().to_dict(),
            "utterance_asr": self.offline_asr.capability().to_dict(),
            "speaker": self.speaker.capability().to_dict(),
        }
