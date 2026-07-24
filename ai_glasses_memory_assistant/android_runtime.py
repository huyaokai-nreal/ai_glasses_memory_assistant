from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_bridge import GlassesChatService
from .server import ExclusiveThreadingHTTPServer, GlassesHandler
from .text_cleaning import redact_sensitive_text
from .turn_planner import native_location_preflight


_CONFIG_ENV = {
    "app_home": "AI_GLASSES_HOME",
    "static_dir": "AI_GLASSES_STATIC_DIR",
    "provider": "AI_GLASSES_LLM_PROVIDER",
    "model": "AI_GLASSES_LLM_MODEL",
    "base_url": "AI_GLASSES_LLM_BASE_URL",
    "api_key": "AI_GLASSES_LLM_API_KEY",
}


@dataclass
class _AndroidRuntime:
    server: ExclusiveThreadingHTTPServer
    service: GlassesChatService
    thread: threading.Thread
    handler: type[GlassesHandler]
    token: str
    owner_id: str
    previous_env: dict[str, str | None]


_lock = threading.RLock()
_runtime: _AndroidRuntime | None = None
_device_state: dict[str, Any] = {}


def start(config_json: str) -> str:
    """Start the shared local service for the Android WebView."""

    config = _parse_config(config_json)
    global _runtime
    with _lock:
        if _runtime is not None:
            return json.dumps(_public_payload(_runtime), ensure_ascii=False)
        previous_env: dict[str, str | None] = {}
        for key, env_name in _CONFIG_ENV.items():
            previous_env[env_name] = os.environ.get(env_name)
            os.environ[env_name] = config[key]
        previous_env["AI_GLASSES_LLM_TRANSPORT"] = os.environ.get("AI_GLASSES_LLM_TRANSPORT")
        os.environ["AI_GLASSES_LLM_TRANSPORT"] = "stdlib_http"
        service: GlassesChatService | None = None
        server: ExclusiveThreadingHTTPServer | None = None
        try:
            service = GlassesChatService()
            service.start_audio_reaper()
            token = secrets.token_urlsafe(32)
            owner_id = config["owner_id"]
            handler = type(
                "AndroidGlassesHandler",
                (GlassesHandler,),
                {
                    "service": service,
                    "local_auth_token": token,
                    "runtime_info": {
                        "routing_mode": "llm_first",
                        "platform": "android",
                        "audio_input_owner": "native",
                        "network_state": "unknown",
                        "owner_id": owner_id,
                    },
                    "runtime_info_provider": staticmethod(
                        lambda: _runtime_info(service, owner_id)
                    ),
                },
            )
            server = ExclusiveThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, name="android-local-http", daemon=True)
            thread.start()
            _runtime = _AndroidRuntime(
                server=server,
                service=service,
                thread=thread,
                handler=handler,
                token=token,
                owner_id=owner_id,
                previous_env=previous_env,
            )
            service.recover_device_audio_events()
            return json.dumps(_public_payload(_runtime), ensure_ascii=False)
        except Exception:
            if server is not None:
                server.server_close()
            if service is not None:
                service.close()
            _restore_env(previous_env)
            raise


def status() -> str:
    with _lock:
        payload = {"running": False}
        if _runtime is not None:
            payload = _public_payload(_runtime)
        return json.dumps(payload, ensure_ascii=False)


def stop() -> str:
    global _runtime
    with _lock:
        runtime = _runtime
        _runtime = None
    if runtime is None:
        return json.dumps({"running": False}, ensure_ascii=False)
    runtime.server.shutdown()
    runtime.server.server_close()
    runtime.thread.join(timeout=5.0)
    runtime.service.close()
    runtime.handler.service = None
    _restore_env(runtime.previous_env)
    return json.dumps({"running": False}, ensure_ascii=False)


def start_capture(user_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    return json.dumps(runtime.service.start_device_capture(user_id=user_id), ensure_ascii=False)


def stop_capture(user_id: str, capture_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    result = runtime.service.stop_capture(user_id=user_id, capture_id=str(capture_id or "").strip())
    return json.dumps(result, ensure_ascii=False)


def capture_status(user_id: str, capture_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    result = runtime.service.device_capture_status(
        user_id=user_id,
        capture_id=str(capture_id or "").strip(),
    )
    return json.dumps(result, ensure_ascii=False)


def ingest_audio_event(
    user_id: str,
    capture_id: str,
    event_json: str,
    private_json: str = "{}",
    turn_context_json: str = "{}",
) -> str:
    runtime = _runtime_for_owner(user_id)
    event = _json_object(event_json, "audio event")
    private = _json_object(private_json, "private audio event")
    turn_context = _json_object(turn_context_json, "audio turn context")
    result = runtime.service.ingest_device_audio_event(
        user_id=user_id,
        capture_id=str(capture_id or "").strip(),
        event_payload=event,
        private_payload=private,
        turn_context=turn_context,
    )
    return json.dumps(result, ensure_ascii=False)


def location_preflight(message: str) -> str:
    return json.dumps(native_location_preflight(message), ensure_ascii=False)


def set_network_state(online: bool) -> str:
    runtime = _require_runtime()
    result = runtime.service.set_device_network_state(online=bool(online))
    return json.dumps(result, ensure_ascii=False)


def set_device_state(state_json: str) -> str:
    state = _json_object(state_json, "android device state")
    allowed = {
        "state",
        "running",
        "capture_id",
        "started_at_ms",
        "duration_seconds",
        "captured_samples",
        "audio_rms_dbfs",
        "audio_peak_dbfs",
        "audio_level_at_ms",
        "vad_segment_count",
        "ambient_final_count",
        "speech_rejected_count",
        "last_final_at_ms",
        "sample_rate",
        "channels",
        "encoding",
        "network_online",
        "latest_partial",
        "partial_sequence",
        "inference_queue_depth",
        "enrollment_state",
        "enrollment_session_id",
        "enrollment_sample_count",
        "enrollment_sample_total",
        "model_state",
        "model_version",
        "model_self_test",
        "transcription_ready",
        "pss_kb",
        "last_error",
        "device_event_queue",
    }
    public = {key: state[key] for key in allowed if key in state}
    with _lock:
        _device_state.clear()
        _device_state.update(public)
    return json.dumps(public, ensure_ascii=False)


def queue_status(user_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    return json.dumps(runtime.service.device_audio_event_queue(user_id=user_id), ensure_ascii=False)


def wait_audio_event(user_id: str, event_id: str, timeout: float = 10.0) -> str:
    runtime = _runtime_for_owner(user_id)
    result = runtime.service.wait_device_audio_event(
        user_id=user_id,
        event_id=str(event_id or "").strip(),
        timeout=max(0.0, min(float(timeout), 30.0)),
    )
    return json.dumps(result or {}, ensure_ascii=False)


def speaker_profile(user_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    return json.dumps(runtime.service.get_speaker_profile(user_id=user_id), ensure_ascii=False)


def cancel_speaker_enrollment(user_id: str, enrollment_session_id: str = "") -> str:
    runtime = _runtime_for_owner(user_id)
    result = runtime.service.cancel_speaker_enrollment(
        user_id=user_id,
        enrollment_session_id=str(enrollment_session_id or "").strip(),
    )
    return json.dumps(result, ensure_ascii=False)


def classify_speaker(user_id: str, embedding_json: str, model_name: str) -> str:
    runtime = _runtime_for_owner(user_id)
    try:
        embedding = json.loads(embedding_json)
    except json.JSONDecodeError as exc:
        raise ValueError("speaker embedding must be valid JSON") from exc
    if not isinstance(embedding, list):
        raise ValueError("speaker embedding must be an array")
    result = runtime.service.classify_device_speaker(
        user_id=user_id,
        embedding=embedding,
        model_name=str(model_name or "").strip(),
    )
    return json.dumps(result, ensure_ascii=False)


def retry_failed(user_id: str, event_id: str = "") -> str:
    runtime = _runtime_for_owner(user_id)
    result = runtime.service.retry_device_audio_events(user_id=user_id, event_id=event_id)
    return json.dumps(result, ensure_ascii=False)


def create_diagnostic_bundle(app_home: str, output_path: str, device_json: str = "{}") -> str:
    """Create a private ZIP snapshot which Android encrypts before export."""

    root = Path(str(app_home or "")).expanduser().resolve()
    destination = Path(str(output_path or "")).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("android app home does not exist")
    if not destination.is_absolute() or not destination.parent.is_dir():
        raise ValueError("diagnostic output directory does not exist")
    device = _json_object(device_json, "android diagnostic device state")
    temporary_zip = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    sanitized_paths: list[Path] = []
    exported_files: list[str] = []
    skipped_files: dict[str, str] = {}
    redaction_summary: dict[str, Any] = {
        "voice_profile_rows_removed": 0,
        "private_audio_payloads_cleared": 0,
        "json_values_scrubbed": 0,
    }
    data_dir = root / "data"
    try:
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ("events.db", "timeline.db", "sessions.db"):
                source = data_dir / name
                if not source.is_file():
                    continue
                sanitized = destination.parent / f".{uuid.uuid4().hex}-{name}"
                sanitized_paths.append(sanitized)
                try:
                    _backup_sqlite(source, sanitized)
                    summary = _sanitize_diagnostic_database(sanitized)
                    for key, value in summary.items():
                        redaction_summary[key] = int(redaction_summary.get(key, 0)) + int(value)
                    archive.write(sanitized, f"database/{name}")
                    exported_files.append(f"database/{name}")
                except (OSError, sqlite3.Error, ValueError) as exc:
                    skipped_files[name] = type(exc).__name__

            audit_path = data_dir / "chat_audit.jsonl"
            if audit_path.is_file():
                audit_lines: list[str] = []
                for raw_line in audit_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    sanitized_record = _sanitize_diagnostic_json(record)
                    audit_lines.append(json.dumps(sanitized_record, ensure_ascii=False, separators=(",", ":")))
                archive.writestr("audit/chat_audit.redacted.jsonl", "\n".join(audit_lines) + ("\n" if audit_lines else ""))
                exported_files.append("audit/chat_audit.redacted.jsonl")

            metadata = {
                "schema": "ai_glasses_diagnostic.v1",
                "platform": "android",
                "device": _sanitize_diagnostic_json(device),
                "exported_files": exported_files,
                "skipped_files": skipped_files,
                "redaction": redaction_summary,
                "excludes": ["api_key", "raw_pcm", "speaker_embeddings", "enrollment_embeddings"],
            }
            archive.writestr(
                "diagnostics.json",
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            )
            exported_files.append("diagnostics.json")
        os.replace(temporary_zip, destination)
        return json.dumps(
            {
                "schema": "ai_glasses_diagnostic.v1",
                "files": exported_files,
                "skipped_files": skipped_files,
                "size_bytes": destination.stat().st_size,
                "redaction": redaction_summary,
            },
            ensure_ascii=False,
        )
    finally:
        temporary_zip.unlink(missing_ok=True)
        for path in sanitized_paths:
            path.unlink(missing_ok=True)


def _backup_sqlite(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(str(source))
    destination_connection = sqlite3.connect(str(destination))
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def _sanitize_diagnostic_database(path: Path) -> dict[str, int]:
    summary = {
        "voice_profile_rows_removed": 0,
        "private_audio_payloads_cleared": 0,
        "json_values_scrubbed": 0,
    }
    connection = sqlite3.connect(str(path))
    try:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        for table in ("memory_voice_profiles", "speaker_profiles", "speaker_enrollment_samples"):
            if table not in tables:
                continue
            count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            connection.execute(f'DELETE FROM "{table}"')
            summary["voice_profile_rows_removed"] += count
        if "device_audio_events" in tables:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM device_audio_events WHERE private_payload != '{}'"
                ).fetchone()[0]
            )
            connection.execute("UPDATE device_audio_events SET private_payload = '{}'")
            summary["private_audio_payloads_cleared"] += count
        json_columns = {
            "chunks": ("metadata",),
            "device_audio_events": ("event_payload", "dispatch_payload"),
            "memory_jobs": ("payload",),
            "discussion_slices": ("summary_payload",),
        }
        for table, columns in json_columns.items():
            if table not in tables:
                continue
            available = {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}
            for column in columns:
                if column not in available:
                    continue
                rows = connection.execute(
                    f'SELECT rowid, "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL'
                ).fetchall()
                for row_id, raw in rows:
                    try:
                        value = json.loads(str(raw))
                    except (TypeError, json.JSONDecodeError):
                        continue
                    sanitized = _sanitize_diagnostic_json(value)
                    encoded = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
                    if encoded == str(raw):
                        continue
                    connection.execute(
                        f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?',
                        (encoded, row_id),
                    )
                    summary["json_values_scrubbed"] += 1
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    return summary


def _sanitize_diagnostic_json(value: Any, key: str = "") -> Any:
    normalized_key = str(key).casefold()
    if normalized_key in {
        "api_key",
        "api_key_ciphertext",
        "api_key_iv",
        "authorization",
        "embedding",
        "embedding_json",
        "speaker_embedding",
    }:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {item_key: _sanitize_diagnostic_json(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_diagnostic_json(item, key) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_diagnostic_json(item, key) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value).text
    return value


def _parse_config(config_json: str) -> dict[str, str]:
    try:
        raw = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("android runtime config must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("android runtime config must be a JSON object")
    config = {key: str(raw.get(key) or "").strip() for key in {*_CONFIG_ENV, "owner_id"}}
    missing = [key for key, value in config.items() if not value]
    if missing:
        raise ValueError("android runtime config missing: " + ", ".join(sorted(missing)))
    if not config["base_url"].startswith("https://"):
        raise ValueError("android runtime base_url must use https")
    return config


def _json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_runtime() -> _AndroidRuntime:
    with _lock:
        if _runtime is None:
            raise RuntimeError("android runtime is not running")
        return _runtime


def _runtime_for_owner(user_id: str) -> _AndroidRuntime:
    runtime = _require_runtime()
    if str(user_id or "").strip() != runtime.owner_id:
        raise ValueError("user_id does not match this device owner")
    return runtime


def _runtime_info(service: GlassesChatService, owner_id: str) -> dict[str, Any]:
    queue = service.device_audio_event_queue(user_id=owner_id, limit=20)
    capture_id = str(_device_state.get("capture_id") or "")
    ambient_context = service.device_capture_status(user_id=owner_id, capture_id=capture_id)
    return {
        "routing_mode": "llm_first",
        "platform": "android",
        "audio_input_owner": "native",
        "network_state": "online" if queue["network_online"] else "offline",
        "owner_id": owner_id,
        "device_event_queue": queue["summary"],
        "ambient_context": ambient_context,
        "device": dict(_device_state),
    }


def _public_payload(runtime: _AndroidRuntime) -> dict[str, Any]:
    _, port = runtime.server.server_address
    return {
        "running": True,
        "base_url": f"http://127.0.0.1:{port}",
        "local_token": runtime.token,
        "owner_id": runtime.owner_id,
        "platform": "android",
    }


def _restore_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
