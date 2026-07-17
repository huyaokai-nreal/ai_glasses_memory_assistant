"""Backward-compatible exports for the unified audio engine.

Model construction and inference live in ``audio_engine.offline``. Existing
imports keep working while all runtime paths share ``AudioBackendRegistry``.
"""

from .audio_engine.offline import (
    ACOUSTIC_EMOTION_REPLY_MIN_SCORE,
    ASR_MODEL_DIR_ENV,
    DISCARDED_AFTER_FAILURE,
    DISCARDED_AFTER_PROCESSING,
    EMOTION_MODEL_DIR_ENV,
    SPEAKER_TARGET_SAMPLE_COUNT,
    AudioPreparationResult,
    AudioSegmentProcessResult,
    AudioSegmentProcessor,
    CamppSpeakerRunner,
    EmotionAudioRunner,
    LocalASRResult,
    LocalEmotionResult,
    LocalSpeakerResult,
    SenseVoiceASRRunner,
    speaker_centroid_embedding,
    speaker_enrollment_error_detail,
    speaker_similarity_from_evidence,
    speaker_thresholds_from_samples,
)

__all__ = [
    "ACOUSTIC_EMOTION_REPLY_MIN_SCORE",
    "ASR_MODEL_DIR_ENV",
    "DISCARDED_AFTER_FAILURE",
    "DISCARDED_AFTER_PROCESSING",
    "EMOTION_MODEL_DIR_ENV",
    "SPEAKER_TARGET_SAMPLE_COUNT",
    "AudioPreparationResult",
    "AudioSegmentProcessResult",
    "AudioSegmentProcessor",
    "CamppSpeakerRunner",
    "EmotionAudioRunner",
    "LocalASRResult",
    "LocalEmotionResult",
    "LocalSpeakerResult",
    "SenseVoiceASRRunner",
    "speaker_centroid_embedding",
    "speaker_enrollment_error_detail",
    "speaker_similarity_from_evidence",
    "speaker_thresholds_from_samples",
]
