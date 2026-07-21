from __future__ import annotations

import json
import os
import secrets
import threading
from dataclasses import dataclass
from typing import Any

from .agent_bridge import GlassesChatService
from .server import ExclusiveThreadingHTTPServer, GlassesHandler


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


def ingest_audio_event(
    user_id: str,
    capture_id: str,
    event_json: str,
    private_json: str = "{}",
) -> str:
    runtime = _runtime_for_owner(user_id)
    event = _json_object(event_json, "audio event")
    private = _json_object(private_json, "private audio event")
    result = runtime.service.ingest_device_audio_event(
        user_id=user_id,
        capture_id=str(capture_id or "").strip(),
        event_payload=event,
        private_payload=private,
    )
    return json.dumps(result, ensure_ascii=False)


def set_network_state(online: bool) -> str:
    runtime = _require_runtime()
    result = runtime.service.set_device_network_state(online=bool(online))
    return json.dumps(result, ensure_ascii=False)


def queue_status(user_id: str) -> str:
    runtime = _runtime_for_owner(user_id)
    return json.dumps(runtime.service.device_audio_event_queue(user_id=user_id), ensure_ascii=False)


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
    return {
        "routing_mode": "llm_first",
        "platform": "android",
        "audio_input_owner": "native",
        "network_state": "online" if queue["network_online"] else "offline",
        "owner_id": owner_id,
        "device_event_queue": queue["summary"],
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
