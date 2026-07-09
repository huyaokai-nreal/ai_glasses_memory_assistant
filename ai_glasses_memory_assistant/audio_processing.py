from __future__ import annotations

import base64
import binascii
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


ACOUSTIC_EMOTION_REPLY_MIN_SCORE = 0.8
DISCARDED_AFTER_PROCESSING = "discarded_after_processing"
DISCARDED_AFTER_FAILURE = "discarded_after_failure"
ASR_MODEL_DIR_ENV = "AI_GLASSES_ASR_MODEL_DIR"
SPEAKER_TARGET_SAMPLE_COUNT = 3
EMOTION_MODEL_DIR_ENV = "AI_GLASSES_EMOTION_MODEL_DIR"


@dataclass(frozen=True)
class AudioSegmentProcessResult:
    status: str
    transcript: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    audio_retention: str = DISCARDED_AFTER_FAILURE
    error_type: str = ""
    capture_append: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "transcript": self.transcript,
            "metadata": dict(self.metadata),
            "audio_retention": self.audio_retention,
            "error_type": self.error_type,
        }
        if self.capture_append is not None:
            payload["capture_append"] = dict(self.capture_append)
        return payload


@dataclass(frozen=True)
class AudioPreparationResult:
    debug: dict[str, Any]
    temp_path: Path | None = None


@dataclass(frozen=True)
class LocalASRResult:
    ok: bool
    transcript: str = ""
    language: str = ""
    latency_ms: int = 0
    error_type: str = ""
    model_name: str = ""
    emotion_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalEmotionResult:
    enabled: bool
    label: str = "unknown"
    intensity: int | None = None
    score: float | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    model_name: str = ""
    evidence: str = ""
    source: str = "acoustic_emotion_model"
    reason: str = ""
    error_type: str = ""

    def to_metadata(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "emotion": {
                    "enabled": False,
                    "reason": self.reason or self.error_type or "emotion_model_unavailable",
                    "source": self.source or "acoustic_emotion_model",
                    "model": self.model_name or "",
                }
            }
        return {
            "emotion_label": self.label,
            "emotion_intensity": self.intensity,
            "emotion_score": self.score,
            "emotion_candidates": list(self.candidates),
            "emotion_model": self.model_name,
            "emotion_evidence": self.evidence,
            "emotion_source": self.source,
            "emotion_source_kind": "acoustic_model",
            "emotion_eligible_for_reply": bool(self.score is not None and self.score >= ACOUSTIC_EMOTION_REPLY_MIN_SCORE),
            "emotion": {
                "enabled": True,
                "reason": self.reason or "acoustic_emotion_inferred",
                "source": self.source,
                "model": self.model_name,
            },
        }


@dataclass(slots=True)
class LocalSpeakerResult:
    embedding: list[float] = field(default_factory=list)
    hint: str = "unknown"
    confidence: float | None = None
    source: str = "campp_diarization"
    evidence: str = "speaker_analysis_not_run"
    enabled: bool = False
    error_type: str = ""
    model_name: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "speaker_hint": self.hint or "unknown",
            "speaker_confidence": self.confidence,
            "speaker_source": self.source or "campp_diarization",
            "speaker_evidence": self.evidence or "speaker_analysis_not_run",
        }


class SenseVoiceASRRunner:
    def __init__(self, model_dir_env: str = ASR_MODEL_DIR_ENV) -> None:
        self.model_dir_env = model_dir_env
        self._model: Any = None
        self._model_key: str = ""

    def transcribe_file(self, audio_path: Path) -> LocalASRResult:
        started = time.time()
        model_dir = str(os.getenv(self.model_dir_env) or "").strip()
        if not model_dir:
            return LocalASRResult(ok=False, error_type="asr_model_dir_missing")
        model_path = Path(model_dir)
        if not model_path.exists():
            return LocalASRResult(ok=False, error_type="asr_model_dir_not_found")
        try:
            auto_model_cls = self._load_sensevoice_model()
        except ModuleNotFoundError:
            return LocalASRResult(ok=False, error_type="sensevoice_not_installed")
        except Exception:
            return LocalASRResult(ok=False, error_type="sensevoice_import_failed")
        try:
            model = self._get_or_create_model(auto_model_cls, model_path)
            result = model.generate(
                input=str(audio_path),
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
            first = result[0] if isinstance(result, list) and result else {}
            transcript = self._normalize_sensevoice_text(str(first.get("text") or ""))
            if not transcript:
                return LocalASRResult(ok=False, error_type="asr_empty_transcript")
            return LocalASRResult(
                ok=True,
                transcript=transcript,
                language=self._normalize_sensevoice_language(first),
                latency_ms=int((time.time() - started) * 1000),
                model_name=model_path.name,
                emotion_metadata=self._extract_sensevoice_emotion_metadata(first),
            )
        except Exception:
            return LocalASRResult(ok=False, error_type="asr_runtime_failed")

    @staticmethod
    def _load_sensevoice_model() -> Any:
        from funasr import AutoModel

        return AutoModel

    def _get_or_create_model(self, auto_model_cls: Any, model_path: Path) -> Any:
        model_key = str(model_path.resolve())
        if self._model is not None and self._model_key == model_key:
            return self._model
        self._model = auto_model_cls(
            model=str(model_key),
            trust_remote_code=False,
            disable_update=True,
            device="cpu",
        )
        self._model_key = model_key
        return self._model

    @staticmethod
    def _normalize_sensevoice_text(text: str) -> str:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"<\|[^|>]+\|>", " ", cleaned)
        return " ".join(cleaned.split()).strip()

    @staticmethod
    def _normalize_sensevoice_language(payload: dict[str, Any]) -> str:
        explicit = str(payload.get("language") or "").strip()
        if explicit:
            return explicit
        text = str(payload.get("text") or "")
        match = re.match(r"<\|([^|>]+)\|>", text)
        return str(match.group(1) if match else "")

    @staticmethod
    def _extract_sensevoice_emotion_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text") or "")
        tokens = re.findall(r"<\|([^|>]+)\|>", text)
        emotion_token = ""
        for token in tokens:
            upper = str(token or "").strip().upper()
            if upper in {"NEUTRAL", "HAPPY", "ANGRY", "SAD"}:
                emotion_token = upper
                break
        if not emotion_token:
            return {
                "emotion": {
                    "enabled": False,
                    "reason": "sensevoice_emotion_token_missing",
                    "source": "sensevoice_control_tokens",
                    "model": "sensevoice_small",
                }
            }
        label_map = {
            "NEUTRAL": "平静",
            "HAPPY": "开心",
            "ANGRY": "烦躁",
            "SAD": "委屈",
        }
        intensity_map = {
            "NEUTRAL": 1,
            "HAPPY": 3,
            "ANGRY": 4,
            "SAD": 3,
        }
        label = label_map.get(emotion_token, "unknown")
        return {
            "emotion_label": label,
            "emotion_intensity": intensity_map.get(emotion_token, 1),
            "emotion_score": None,
            "emotion_candidates": [{"label": label, "score": None}],
            "emotion_model": "sensevoice_small",
            "emotion_evidence": f"derived_from_sensevoice_token: {emotion_token}",
            "emotion_source": "sensevoice_control_tokens",
            "emotion_source_kind": "asr_token_hint",
            "emotion_eligible_for_reply": False,
            "emotion": {
                "enabled": label != "unknown",
                "reason": "sensevoice_emotion_token_parsed" if label != "unknown" else "sensevoice_emotion_unknown",
                "source": "sensevoice_control_tokens",
                "model": "sensevoice_small",
                "token": emotion_token,
            },
        }


class EmotionAudioRunner:
    LABEL_MAP = {
        "neutral": ("平静", 1),
        "happy": ("开心", 3),
        "angry": ("烦躁", 4),
        "sad": ("难过", 3),
        "surprised": ("紧张", 3),
        "unknown": ("unknown", None),
    }

    def __init__(self, model_dir_env: str = EMOTION_MODEL_DIR_ENV) -> None:
        self.model_dir_env = model_dir_env
        self._model: Any = None
        self._model_key: str = ""

    def analyze_file(self, audio_path: Path) -> LocalEmotionResult:
        model_dir = str(os.getenv(self.model_dir_env) or "").strip()
        if not model_dir:
            return LocalEmotionResult(enabled=False, source="acoustic_emotion_model", reason="emotion_model_not_configured", error_type="emotion_model_dir_missing")
        model_path = Path(model_dir)
        if not model_path.exists():
            return LocalEmotionResult(enabled=False, source="acoustic_emotion_model", reason="emotion_model_not_found", error_type="emotion_model_dir_not_found")
        try:
            auto_model_cls = self._load_emotion_model()
        except ModuleNotFoundError:
            return LocalEmotionResult(enabled=False, source="acoustic_emotion_model", reason="emotion_model_not_installed", error_type="emotion_model_not_installed")
        except Exception:
            return LocalEmotionResult(enabled=False, source="acoustic_emotion_model", reason="emotion_model_import_failed", error_type="emotion_model_import_failed")
        try:
            model = self._get_or_create_model(auto_model_cls, model_path)
            result = model.generate(input=str(audio_path), batch_size_s=60)
            return self._normalize_emotion_result(result, model_path.name)
        except Exception:
            return LocalEmotionResult(enabled=False, source="acoustic_emotion_model", reason="emotion_runtime_failed", error_type="emotion_runtime_failed")

    @staticmethod
    def _load_emotion_model() -> Any:
        from funasr import AutoModel

        return AutoModel

    def _get_or_create_model(self, auto_model_cls: Any, model_path: Path) -> Any:
        model_key = str(model_path.resolve())
        if self._model is not None and self._model_key == model_key:
            return self._model
        self._model = auto_model_cls(
            model=str(model_key),
            trust_remote_code=False,
            disable_update=True,
            device="cpu",
        )
        self._model_key = model_key
        return self._model

    @classmethod
    def _normalize_emotion_result(cls, result: Any, model_name: str) -> LocalEmotionResult:
        if not isinstance(result, list) or not result:
            return LocalEmotionResult(enabled=False, source="acoustic_emotion_model", model_name=model_name, reason="emotion_result_empty", error_type="emotion_result_empty")
        first = result[0] if isinstance(result[0], dict) else {}
        candidates = cls._emotion_candidates_from_payload(first)
        if not candidates:
            return LocalEmotionResult(enabled=False, source="acoustic_emotion_model", model_name=model_name, reason="emotion_label_missing", error_type="emotion_label_missing")
        top = candidates[0]
        normalized_label = str(top.get("raw_label") or "unknown").lower()
        mapped_label, mapped_intensity = cls.LABEL_MAP.get(normalized_label, ("unknown", None))
        return LocalEmotionResult(
            enabled=mapped_label != "unknown",
            label=mapped_label,
            intensity=mapped_intensity,
            score=top.get("score"),
            candidates=[{"label": item["label"], "score": item["score"]} for item in candidates],
            model_name=model_name,
            evidence=f"derived_from_acoustic_emotion_label: {normalized_label}",
            source="acoustic_emotion_model",
            reason="acoustic_emotion_inferred" if mapped_label != "unknown" else "acoustic_emotion_unknown",
        )

    @classmethod
    def _emotion_candidates_from_payload(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        raw_labels = payload.get("labels")
        raw_scores = payload.get("scores")
        if isinstance(raw_labels, list):
            for index, raw_label in enumerate(raw_labels):
                label = str(raw_label or "").strip().lower()
                if not label:
                    continue
                score = None
                if isinstance(raw_scores, list) and index < len(raw_scores):
                    try:
                        score = float(raw_scores[index])
                    except (TypeError, ValueError):
                        score = None
                mapped_label, _ = cls.LABEL_MAP.get(label, ("unknown", None))
                candidates.append({"raw_label": label, "label": mapped_label, "score": score})
        if candidates:
            return candidates
        single = str(payload.get("label") or payload.get("text") or payload.get("value") or "").strip().lower()
        if single:
            mapped_label, _ = cls.LABEL_MAP.get(single, ("unknown", None))
            return [{"raw_label": single, "label": mapped_label, "score": None}]
        return []


class CamppSpeakerRunner:
    USER_MATCH_THRESHOLD = 0.72
    OTHER_REJECT_THRESHOLD = 0.45

    def __init__(self, model_dir_env: str = "AI_GLASSES_SPEAKER_MODEL_DIR") -> None:
        self.model_dir_env = model_dir_env
        self._model: Any = None
        self._model_key: str = ""

    def analyze_file(self, audio_path: Path) -> LocalSpeakerResult:
        model_dir = str(os.getenv(self.model_dir_env) or "").strip()
        if not model_dir:
            return LocalSpeakerResult(evidence="speaker_model_not_configured", error_type="speaker_model_dir_missing")
        model_path = Path(model_dir)
        if not model_path.exists():
            return LocalSpeakerResult(evidence="speaker_model_dir_not_found", error_type="speaker_model_dir_not_found")
        try:
            auto_model_cls = self._load_speaker_model()
        except ModuleNotFoundError:
            return LocalSpeakerResult(evidence="campp_not_installed", error_type="speaker_model_not_installed")
        except Exception:
            return LocalSpeakerResult(evidence="campp_import_failed", error_type="speaker_model_import_failed")
        try:
            model = self._get_or_create_model(auto_model_cls, model_path)
            result = model.generate(input=str(audio_path), batch_size_s=60)
            return self._normalize_speaker_result(result, model_path.name)
        except Exception:
            return LocalSpeakerResult(evidence="speaker_runtime_failed", error_type="speaker_runtime_failed")

    @staticmethod
    def _load_speaker_model() -> Any:
        from funasr import AutoModel

        return AutoModel

    def _get_or_create_model(self, auto_model_cls: Any, model_path: Path) -> Any:
        model_key = str(model_path.resolve())
        if self._model is not None and self._model_key == model_key:
            return self._model
        self._model = auto_model_cls(
            model=str(model_key),
            trust_remote_code=False,
            disable_update=True,
            device="cpu",
        )
        self._model_key = model_key
        return self._model

    @staticmethod
    def _normalize_speaker_result(result: Any, model_name: str) -> LocalSpeakerResult:
        if not isinstance(result, list) or not result:
            return LocalSpeakerResult(model_name=model_name, evidence="speaker_result_empty", error_type="speaker_result_empty")
        first = result[0] if isinstance(result[0], dict) else {}
        embedding = CamppSpeakerRunner._extract_embedding(first)
        if embedding:
            return LocalSpeakerResult(
                embedding=embedding,
                hint="unknown",
                confidence=None,
                source="campp_diarization",
                evidence="speaker_embedding_extracted",
                enabled=True,
                model_name=model_name,
            )
        return LocalSpeakerResult(
            hint="unknown",
            confidence=None,
            source="campp_diarization",
            evidence="speaker_embedding_missing",
            enabled=False,
            error_type="speaker_embedding_missing",
            model_name=model_name,
        )

    @staticmethod
    def _extract_embedding(payload: dict[str, Any]) -> list[float]:
        value = payload.get("spk_embedding")
        if value is None:
            return []
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            if hasattr(value, "tolist"):
                value = value.tolist()
        except Exception:
            return []
        if not isinstance(value, list):
            return []
        if value and isinstance(value[0], list):
            value = value[0]
        embedding: list[float] = []
        for item in value:
            try:
                embedding.append(float(item))
            except (TypeError, ValueError):
                continue
        return embedding

    @classmethod
    def compare_with_reference(
        cls,
        current_embedding: list[float],
        reference_embedding: list[float],
        *,
        model_name: str = "",
        user_threshold: float | None = None,
        other_threshold: float | None = None,
    ) -> LocalSpeakerResult:
        similarity = cls._cosine_similarity(current_embedding, reference_embedding)
        resolved_user_threshold = float(user_threshold) if user_threshold is not None else cls.USER_MATCH_THRESHOLD
        resolved_other_threshold = float(other_threshold) if other_threshold is not None else cls.OTHER_REJECT_THRESHOLD
        if similarity is None:
            return LocalSpeakerResult(
                embedding=list(current_embedding),
                hint="unknown",
                confidence=None,
                source="campp_diarization",
                evidence="speaker_similarity_unavailable",
                enabled=False,
                error_type="speaker_similarity_unavailable",
                model_name=model_name,
            )
        if similarity >= resolved_user_threshold:
            return LocalSpeakerResult(
                embedding=list(current_embedding),
                hint="user",
                confidence=round(similarity, 4),
                source="campp_diarization",
                evidence=f"speaker_similarity_user_match:{similarity:.4f}",
                enabled=True,
                model_name=model_name,
            )
        if similarity <= resolved_other_threshold:
            return LocalSpeakerResult(
                embedding=list(current_embedding),
                hint="other",
                confidence=round(1.0 - similarity, 4),
                source="campp_diarization",
                evidence=f"speaker_similarity_other_reject:{similarity:.4f}",
                enabled=True,
                model_name=model_name,
            )
        return LocalSpeakerResult(
            embedding=list(current_embedding),
            hint="unknown",
            confidence=round(similarity, 4),
            source="campp_diarization",
            evidence=f"speaker_similarity_gray_zone:{similarity:.4f}",
            enabled=False,
            model_name=model_name,
        )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
        if not left or not right:
            return None
        size = min(len(left), len(right))
        if size <= 0:
            return None
        lhs = left[:size]
        rhs = right[:size]
        dot = sum(a * b for a, b in zip(lhs, rhs))
        left_norm = sum(a * a for a in lhs) ** 0.5
        right_norm = sum(b * b for b in rhs) ** 0.5
        if left_norm <= 0 or right_norm <= 0:
            return None
        similarity = dot / (left_norm * right_norm)
        return max(-1.0, min(1.0, similarity))


def speaker_thresholds_from_samples(embeddings: list[list[float]]) -> tuple[float, float, float | None]:
    pairwise: list[float] = []
    for index, left in enumerate(embeddings):
        for right in embeddings[index + 1:]:
            similarity = CamppSpeakerRunner._cosine_similarity(left, right)
            if similarity is not None:
                pairwise.append(similarity)
    self_min_similarity = min(pairwise) if pairwise else None
    if self_min_similarity is None:
        return CamppSpeakerRunner.USER_MATCH_THRESHOLD, CamppSpeakerRunner.OTHER_REJECT_THRESHOLD, None
    user_threshold = max(0.68, min(0.86, self_min_similarity - 0.08))
    other_threshold = min(0.50, user_threshold - 0.18)
    return user_threshold, other_threshold, self_min_similarity


def speaker_centroid_embedding(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    size = min(len(embedding) for embedding in embeddings if embedding)
    if size <= 0:
        return []
    centroid: list[float] = []
    for index in range(size):
        centroid.append(sum(embedding[index] for embedding in embeddings) / len(embeddings))
    return centroid


class AudioSegmentProcessor:
    """Validates temporary audio input and only keeps derived text/metadata."""

    MAX_AUDIO_BYTES = 2 * 1024 * 1024
    MAX_AUDIO_DURATION_MS = 30_000
    ALLOWED_AUDIO_MIME_TYPES = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
    }

    def __init__(
        self,
        asr_runner: SenseVoiceASRRunner | None = None,
        emotion_runner: EmotionAudioRunner | None = None,
        speaker_runner: CamppSpeakerRunner | None = None,
    ) -> None:
        self.asr_runner = asr_runner or SenseVoiceASRRunner()
        self.emotion_runner = emotion_runner or EmotionAudioRunner()
        self.speaker_runner = speaker_runner or CamppSpeakerRunner()

    def process(
        self,
        *,
        transcript_hint: str = "",
        simulate: str = "success",
        emotion_metadata: dict[str, Any] | None = None,
        audio_base64: str = "",
        audio_mime_type: str = "",
        audio_duration_ms: int | None = None,
        reference_speaker_embedding: list[float] | None = None,
        speaker_profile_sample_count: int = 0,
        speaker_profile_calibrated: bool = False,
        speaker_match_threshold_user: float | None = None,
        speaker_match_threshold_other: float | None = None,
    ) -> AudioSegmentProcessResult:
        mode = str(simulate or "success").strip() or "success"
        fallback_transcript = str(transcript_hint or "").strip()
        prepared = self._prepare_audio_segment(
            audio_base64=str(audio_base64 or ""),
            audio_mime_type=str(audio_mime_type or ""),
            audio_duration_ms=audio_duration_ms,
        )
        try:
            transcript = fallback_transcript
            asr_metadata: dict[str, Any] = {"enabled": False, "mode": "simulated_transcript_placeholder"}
            speaker_metadata: dict[str, Any] = {
                "speaker_hint": "unknown",
                "speaker_confidence": None,
                "speaker_source": "campp_diarization",
                "speaker_evidence": "speaker_analysis_not_run",
                "speaker": {
                    "enabled": False,
                    "reason": "speaker_analysis_not_run",
                    "source": "campp_diarization",
                },
            }
            emotion_debug: dict[str, Any] = {
                "emotion_backend": "sensevoice_control_tokens_fallback",
                "emotion_enabled": False,
                "emotion_label": "",
                "emotion_score": None,
                "emotion_source": "",
                "emotion_source_kind": "unknown",
                "eligible_for_reply": False,
                "emotion_decision_reason": "emotion_model_not_run",
            }
            fallback_used = False
            if prepared.temp_path is not None and mode == "success":
                asr_result = self.asr_runner.transcribe_file(prepared.temp_path)
                if asr_result.ok:
                    transcript = asr_result.transcript
                    asr_metadata = {
                        "enabled": True,
                        "mode": "local_sensevoice",
                        "model_name": asr_result.model_name,
                        "language": asr_result.language,
                        "latency_ms": asr_result.latency_ms,
                    }
                    if asr_result.emotion_metadata:
                        emotion_metadata = {
                            **(emotion_metadata if isinstance(emotion_metadata, dict) else {}),
                            **asr_result.emotion_metadata,
                        }
                        emotion_debug = {
                            "emotion_backend": "sensevoice_control_tokens_fallback",
                            "emotion_enabled": bool(asr_result.emotion_metadata.get("emotion", {}).get("enabled")),
                            "emotion_label": str(asr_result.emotion_metadata.get("emotion_label") or ""),
                            "emotion_score": asr_result.emotion_metadata.get("emotion_score"),
                            "emotion_source": str(asr_result.emotion_metadata.get("emotion_source") or "sensevoice_control_tokens"),
                            "emotion_source_kind": str(asr_result.emotion_metadata.get("emotion_source_kind") or "asr_token_hint"),
                            "eligible_for_reply": bool(asr_result.emotion_metadata.get("emotion_eligible_for_reply")),
                            "emotion_decision_reason": str(asr_result.emotion_metadata.get("emotion", {}).get("reason") or ""),
                        }
                    acoustic_emotion = self.emotion_runner.analyze_file(prepared.temp_path)
                    if acoustic_emotion.enabled:
                        emotion_metadata = {
                            **(emotion_metadata if isinstance(emotion_metadata, dict) else {}),
                            **acoustic_emotion.to_metadata(),
                        }
                        emotion_debug = {
                            "emotion_backend": acoustic_emotion.model_name or "acoustic_emotion_model",
                            "emotion_enabled": True,
                            "emotion_label": acoustic_emotion.label,
                            "emotion_score": acoustic_emotion.score,
                            "emotion_source": acoustic_emotion.source,
                            "emotion_source_kind": "acoustic_model",
                            "eligible_for_reply": bool(acoustic_emotion.score is not None and acoustic_emotion.score >= ACOUSTIC_EMOTION_REPLY_MIN_SCORE),
                            "emotion_decision_reason": acoustic_emotion.reason or "acoustic_emotion_inferred",
                        }
                    elif not emotion_metadata:
                        emotion_metadata = acoustic_emotion.to_metadata()
                        emotion_debug = {
                            "emotion_backend": acoustic_emotion.model_name or "acoustic_emotion_model",
                            "emotion_enabled": False,
                            "emotion_label": str(acoustic_emotion.label or ""),
                            "emotion_score": acoustic_emotion.score,
                            "emotion_source": acoustic_emotion.source,
                            "emotion_source_kind": "acoustic_model",
                            "eligible_for_reply": False,
                            "emotion_decision_reason": acoustic_emotion.reason or acoustic_emotion.error_type or "emotion_model_unavailable",
                        }
                    speaker_result = self.speaker_runner.analyze_file(prepared.temp_path)
                    speaker_reference = list(reference_speaker_embedding or [])
                    if speaker_result.embedding and speaker_reference:
                        speaker_result = CamppSpeakerRunner.compare_with_reference(
                            speaker_result.embedding,
                            speaker_reference,
                            model_name=speaker_result.model_name,
                            user_threshold=speaker_match_threshold_user,
                            other_threshold=speaker_match_threshold_other,
                        )
                    elif speaker_result.embedding and not speaker_reference:
                        speaker_result = replace(
                            speaker_result,
                            hint="unknown",
                            confidence=None,
                            evidence="speaker_reference_missing",
                            enabled=False,
                        )
                    speaker_metadata = {
                        **speaker_result.to_metadata(),
                        "speaker_reference_available": bool(speaker_reference),
                        "speaker_similarity": speaker_similarity_from_evidence(speaker_result.evidence),
                        "speaker_match_policy": {
                            "user_threshold": speaker_match_threshold_user if speaker_match_threshold_user is not None else CamppSpeakerRunner.USER_MATCH_THRESHOLD,
                            "other_threshold": speaker_match_threshold_other if speaker_match_threshold_other is not None else CamppSpeakerRunner.OTHER_REJECT_THRESHOLD,
                            "mode": "reference_embedding_similarity",
                        },
                        "speaker_profile_sample_count": int(speaker_profile_sample_count or 0),
                        "speaker_profile_calibrated": bool(speaker_profile_calibrated),
                        "speaker_match_threshold_user": speaker_match_threshold_user if speaker_match_threshold_user is not None else CamppSpeakerRunner.USER_MATCH_THRESHOLD,
                        "speaker_match_threshold_other": speaker_match_threshold_other if speaker_match_threshold_other is not None else CamppSpeakerRunner.OTHER_REJECT_THRESHOLD,
                        "speaker_decision_reason": speaker_result.evidence or "speaker_analysis_not_run",
                        "speaker": {
                            "enabled": speaker_result.enabled,
                            "reason": speaker_result.evidence,
                            "source": speaker_result.source,
                            "model": speaker_result.model_name or "",
                            "error_type": speaker_result.error_type or "",
                        },
                    }
                elif fallback_transcript:
                    fallback_used = True
                    asr_metadata = {
                        "enabled": False,
                        "mode": "fallback_transcript_hint",
                        "error_type": asr_result.error_type,
                        "fallback_used": True,
                    }
                else:
                    return AudioSegmentProcessResult(
                        status="failed",
                        audio_retention=DISCARDED_AFTER_FAILURE,
                        error_type=asr_result.error_type or "asr_failed",
                        metadata={
                            "source_type": "ambient_audio",
                            "audio_retention": DISCARDED_AFTER_FAILURE,
                            "processing_state": "discarded",
                            "audio_input": dict(prepared.debug, retention=DISCARDED_AFTER_FAILURE),
                            "asr": {
                                "enabled": False,
                                "mode": "local_sensevoice",
                                "error_type": asr_result.error_type or "asr_failed",
                            },
                        },
                    )
            if mode == "asr_failure":
                return AudioSegmentProcessResult(
                    status="failed",
                    audio_retention=DISCARDED_AFTER_FAILURE,
                    error_type="asr_failed",
                    metadata={
                        "source_type": "ambient_audio",
                        "audio_retention": DISCARDED_AFTER_FAILURE,
                        "processing_state": "discarded",
                        "audio_input": dict(prepared.debug, retention=DISCARDED_AFTER_FAILURE),
                    },
                )
            if not transcript:
                return AudioSegmentProcessResult(
                    status="failed",
                    audio_retention=DISCARDED_AFTER_FAILURE,
                    error_type="empty_transcript",
                    metadata={
                        "source_type": "ambient_audio",
                        "audio_retention": DISCARDED_AFTER_FAILURE,
                        "processing_state": "discarded",
                        "audio_input": dict(prepared.debug, retention=DISCARDED_AFTER_FAILURE),
                    },
                )
            emotion = emotion_metadata if isinstance(emotion_metadata, dict) else {}
            metadata: dict[str, Any] = {
                "source_type": "ambient_audio",
                "audio_retention": DISCARDED_AFTER_PROCESSING,
                "processing_state": "discarded",
                "asr": asr_metadata,
                "emotion": {
                    "enabled": False,
                    "reason": "emotion_model_not_enabled_for_mvp",
                    "source": "none",
                },
                "audio_input": dict(prepared.debug, retention=DISCARDED_AFTER_PROCESSING),
            }
            metadata.update(speaker_metadata)
            if fallback_used:
                metadata["fallback_used"] = True
            if mode == "emotion_failure":
                metadata["emotion"] = {
                    "enabled": False,
                    "error_type": "emotion_failed",
                    "reason": "emotion_processing_failed",
                }
            elif emotion:
                metadata.update(emotion)
            metadata["emotion_processing"] = emotion_debug
            return AudioSegmentProcessResult(
                status="processed",
                transcript=transcript,
                metadata=metadata,
                audio_retention=DISCARDED_AFTER_PROCESSING,
            )
        finally:
            if prepared.temp_path is not None and prepared.temp_path.exists():
                prepared.temp_path.unlink()

    def _prepare_audio_segment(
        self,
        *,
        audio_base64: str,
        audio_mime_type: str,
        audio_duration_ms: int | None,
    ) -> AudioPreparationResult:
        has_audio = bool(audio_base64.strip())
        mime_type = audio_mime_type.strip()
        duration_ms = int(audio_duration_ms) if audio_duration_ms is not None else None
        debug: dict[str, Any] = {
            "received": has_audio,
            "mime_type": mime_type,
            "duration_ms": duration_ms,
            "bytes_received": 0,
            "storage": "none",
            "temp_file_created": False,
            "retention": "not_provided",
        }
        if not has_audio:
            return AudioPreparationResult(debug=debug)
        if mime_type not in self.ALLOWED_AUDIO_MIME_TYPES:
            raise ValueError("unsupported audio mime type")
        if duration_ms is None or duration_ms <= 0:
            raise ValueError("audio_duration_ms is required for audio segments")
        if duration_ms > self.MAX_AUDIO_DURATION_MS:
            raise ValueError("audio segment duration exceeds limit")
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid audio_base64 payload") from exc
        if not audio_bytes:
            raise ValueError("audio segment is empty")
        if len(audio_bytes) > self.MAX_AUDIO_BYTES:
            raise ValueError("audio segment exceeds size limit")
        debug["bytes_received"] = len(audio_bytes)
        suffix = self.ALLOWED_AUDIO_MIME_TYPES[mime_type]
        with NamedTemporaryFile(prefix="ambient_audio_", suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = Path(tmp.name)
        debug["storage"] = "temp_file"
        debug["temp_file_created"] = True
        debug["retention"] = "pending_processing"
        return AudioPreparationResult(debug=debug, temp_path=temp_path)


def speaker_similarity_from_evidence(evidence: str) -> float | None:
    match = re.search(r":(-?\d+(?:\.\d+)?)$", str(evidence or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def speaker_enrollment_error_detail(reason: str) -> str:
    normalized = str(reason or "").strip() or "speaker_runtime_failed"
    public_messages = {
        "speaker_model_dir_missing": "speaker_model_not_configured",
        "speaker_model_dir_not_found": "speaker_model_not_available",
        "speaker_model_not_installed": "speaker_model_not_installed",
        "speaker_model_import_failed": "speaker_model_import_failed",
        "speaker_runtime_failed": "speaker_runtime_failed",
        "speaker_result_empty": "speaker_result_empty",
        "speaker_embedding_missing": "speaker_embedding_missing",
    }
    return public_messages.get(normalized, normalized)