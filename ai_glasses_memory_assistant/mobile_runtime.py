"""Platform-neutral entry point for the embedded mobile Python service.

Android keeps importing ``android_runtime`` for compatibility. iOS imports this
module through its CPython bridge, while the shared Python service remains the
only owner of planner, memory, SQLite, privacy, and audit behavior.
"""

import json

from .android_runtime import (
    cancel_speaker_enrollment,
    capture_status,
    classify_speaker,
    create_diagnostic_bundle,
    ingest_audio_event,
    location_preflight,
    queue_status,
    retry_failed,
    set_device_state,
    set_network_state,
    speaker_profile,
    start as _start,
    start_capture,
    status,
    stop,
    stop_capture,
    wait_audio_event,
)


def start(config_json: str) -> str:
    """Start the shared runtime with the iOS public runtime marker."""

    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime config must be valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("runtime config must be a JSON object")
    config["platform"] = "ios"
    return _start(json.dumps(config, ensure_ascii=False))

__all__ = [
    "cancel_speaker_enrollment", "capture_status", "classify_speaker",
    "create_diagnostic_bundle", "ingest_audio_event", "location_preflight",
    "queue_status", "retry_failed", "set_device_state", "set_network_state",
    "speaker_profile", "start", "start_capture", "status", "stop",
    "stop_capture", "wait_audio_event",
]
