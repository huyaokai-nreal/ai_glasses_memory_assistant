"""Opt-in isolated acceptance transport and synthetic fixture controls.

Never enable this in the original application or against personal databases.
The authority reserves before each actual model POST; a failed reservation is
fail-closed. Normal factory/HTTP routes are unchanged when it is not enabled.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from urllib.request import Request, ProxyHandler, build_opener
from zoneinfo import ZoneInfo

from .llm_client import StdlibOpenAICompatibleLLMClient

MODEL = "qwen3.8-27b-32k"
BASE_URL = "http://10.252.17.5:11438/v1"


def authority(path, payload):
    url = os.environ["AI_GLASSES_ACCEPTANCE_LEDGER_URL"]
    if url not in {"http://127.0.0.1:27901"}:
        raise ValueError("Acceptance authority must be the isolated loopback endpoint")
    request = Request(url + path, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json",
        "X-Acceptance-Token": os.environ["AI_GLASSES_ACCEPTANCE_TOKEN"],
    })
    with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
        return json.load(response)


class GuardedAcceptanceClient(StdlibOpenAICompatibleLLMClient):
    def _send_request(self, request):
        if (self.provider, self.model, self.base_url, self.api_key) != (
            "llama_cpp", MODEL, BASE_URL, "ollama"
        ) or request.full_url != BASE_URL + "/chat/completions":
            raise ValueError("Acceptance model configuration mismatch; no fallback")
        if any(os.environ.get(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")):
            # The generation destination must remain direct, without inherited proxies.
            os.environ["NO_PROXY"] = "*"
        reservation = authority("/reserve", {"request": json.loads(request.data),
            "client": {"provider": self.provider, "model": self.model, "base_url": self.base_url}})
        started = time.perf_counter()
        result = None
        error = None
        try:
            result = super()._send_request(request)
            return result
        except Exception as exc:
            error = {"type": type(exc).__name__, "detail": str(exc)}
            raise
        finally:
            # Completion failure must not cause a generation retry; its reservation
            # stays pending and prevents the authority from advancing the variant.
            authority("/finish", {"request_id": reservation["request_id"],
                "response": result, "error": error, "seconds": time.perf_counter() - started})


def prepare(service, body, owner_id=None):
    """Seed only a new synthetic user; never delete or overwrite existing data."""
    user_id = owner_id or str(body["user_id"])
    if not user_id.startswith(("acceptance_", "android_")):
        raise ValueError("Synthetic acceptance user required")
    if service.memory_store.list_memories(user_id, limit=1):
        raise ValueError("Fixture user already has data; refuse overwrite")
    condition = body.get("condition") or {}
    reference = datetime.fromisoformat(condition.get("reference_iso", "2026-10-12T12:00:00+08:00"))
    service._clock = lambda: reference.timestamp()
    service.timezone = condition.get("service_timezone", "Asia/Shanghai")
    if condition.get("host_timezone"):
        os.environ["TZ"] = condition["host_timezone"]
        if hasattr(time, "tzset"):
            time.tzset()
    ids = []
    content = condition.get("content")
    if content:
        iso = condition.get("stored_start_iso")
        start = datetime.fromisoformat(iso) if iso else None
        if start and start.tzinfo is None:
            start = start.replace(tzinfo=ZoneInfo(service.timezone))
        created = condition.get("created_at", reference.timestamp() - 100 * 86400)
        recorded = condition.get("expected", {}).get("recorded_at")
        if recorded:
            created = datetime.fromisoformat(recorded).timestamp()
        memory = service.memory_store.add_memory(user_id, content,
            kind=condition.get("kind", "event"), memory_type=condition.get("memory_type", "event"),
            source="acceptance_synthetic", created_at=created,
            start_at=start.timestamp() if start else None,
            end_at=start.timestamp() + 3600 if start else None,
            time_granularity="hour" if start else "unknown", tags=[])
        ids.append(memory.id)
    for i in range(int(condition.get("noise_rows", 0))):
        service.memory_store.add_memory(user_id, f"Synthetic unrelated warehouse log {i}.",
            kind=condition.get("kind", "event"), created_at=reference.timestamp() - i - 1)
    if body.get("projection"):
        service.memory_store.install_logical_tag_projection(with_fts=True)
    return {"user_id": user_id, "target_ids": ids}


def acceptance_handler(base):
    """Routes exist only in explicitly bootstrapped acceptance processes."""
    class AcceptanceHandler(base):
        def do_POST(self):
            if self.path == "/api/acceptance/prepare":
                try:
                    body = self._read_json()
                    result = prepare(self._service(), body)
                    self._send_json(result)
                except Exception as exc:
                    self._send_json({"detail": str(exc)}, status=400)
                return
            super().do_POST()

        def do_GET(self):
            if self.path == "/api/acceptance/status":
                service = self._service()
                with service._lock:
                    pending = [key for key, value in service._background_workers.items() if value.is_alive()]
                audit = []
                if service.audit_path.exists():
                    audit = [json.loads(line) for line in service.audit_path.read_text().splitlines() if line]
                self._send_json({"pending": pending, "audit": audit})
                return
            if self.path == "/api/acceptance/models":
                try:
                    with build_opener(ProxyHandler({})).open(BASE_URL + "/models", timeout=6) as response:
                        self._send_json(json.load(response))
                except Exception as exc:
                    self._send_json({"detail": str(exc)}, status=503)
                return
            super().do_GET()
    return AcceptanceHandler


def configure_android(token):
    if not token or len(token) < 32:
        raise ValueError("Acceptance authority token required")
    os.environ["AI_GLASSES_ACCEPTANCE_LEDGER_URL"] = "http://127.0.0.1:27901"
    os.environ["AI_GLASSES_ACCEPTANCE_TOKEN"] = token
