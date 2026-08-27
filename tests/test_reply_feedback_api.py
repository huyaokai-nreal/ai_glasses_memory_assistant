from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai_glasses_memory_assistant.server import GlassesHandler
from tests.helpers import CoreChatService, isolated_app_home


@contextmanager
def reply_feedback_api(tmp_path):
    with isolated_app_home(str(tmp_path)):
        service = CoreChatService(str(tmp_path))

        class FeedbackAPIHandler(GlassesHandler):
            pass

        FeedbackAPIHandler.service = service
        server = ThreadingHTTPServer(("127.0.0.1", 0), FeedbackAPIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield service, f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            FeedbackAPIHandler.service = None
            service.close()


def request_json(base_url: str, path: str, *, method: str = "GET", payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_reply_feedback_http_api_records_and_filters_needs_improvement(tmp_path) -> None:
    with reply_feedback_api(tmp_path) as (service, base_url):
        response = service.chat("为什么刚才回答不好", user_id="u1")
        turn_id = response["debug"]["timeline"]["turn_id"]

        status, saved = request_json(
            base_url,
            "/api/reply-feedback",
            method="POST",
            payload={
                "user_id": "u1",
                "turn_id": turn_id,
                "rating": "needs_improvement",
                "note": "回答没有引用当天的讨论归档。",
            },
        )
        assert status == 200
        assert saved["feedback"]["turn"]["id"] == turn_id
        assert saved["feedback"]["note"] == "回答没有引用当天的讨论归档。"

        status, queued = request_json(
            base_url,
            "/api/reply-feedback?user_id=u1&rating=needs_improvement",
        )
        assert status == 200
        assert [item["id"] for item in queued["feedback"]] == [saved["feedback"]["id"]]

        status, invalid = request_json(
            base_url,
            "/api/reply-feedback",
            method="POST",
            payload={"user_id": "u1", "turn_id": turn_id, "rating": "bad"},
        )
        assert status == 400
        assert "rating" in invalid["detail"]
