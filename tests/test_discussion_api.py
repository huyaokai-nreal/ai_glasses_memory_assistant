from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai_glasses_memory_assistant.discussion_archive import local_day_key
from ai_glasses_memory_assistant.server import GlassesHandler
from tests.helpers import CoreChatService, isolated_app_home


@contextmanager
def discussion_api(tmp_path):
    with isolated_app_home(str(tmp_path)):
        service = CoreChatService(str(tmp_path))

        class DiscussionAPIHandler(GlassesHandler):
            pass

        DiscussionAPIHandler.service = service
        server = ThreadingHTTPServer(("127.0.0.1", 0), DiscussionAPIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            yield f"http://{host}:{port}", service
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            service.close()


def request_json(base_url: str, path: str, *, method: str = "GET") -> tuple[int, dict]:
    request = Request(f"{base_url}{path}", method=method)
    try:
        with urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_discussion_management_routes_validate_and_isolate_users(tmp_path) -> None:
    with discussion_api(tmp_path) as (base_url, service):
        base = service._clock()
        day = local_day_key(base)
        for user_id, text in (("u1", "音频方案。确认继续流式输入。"), ("u2", "私有计划。只属于二号用户。")):
            capture = service.start_capture(user_id=user_id, source="ambient_audio_text")
            service.append_capture_chunk(
                user_id=user_id,
                capture_id=capture["capture_id"],
                text=text,
                timestamp=base,
            )

        status, days = request_json(base_url, "/api/discussions/days?user_id=u1&limit=10")
        assert status == 200
        assert len(days["days"]) == 1
        assert days["archive"]["pending_slice_ids"] == []
        assert "私有计划" not in days["days"][0]["overview"]

        status, other_days = request_json(base_url, "/api/discussions/days?user_id=u2&limit=10")
        assert status == 200
        assert len(other_days["days"]) == 1

        status, detail = request_json(base_url, f"/api/discussions/day?user_id=u1&date={day}")
        assert status == 200
        assert [topic["title"] for topic in detail["day"]["topics"]] == ["音频方案"]

        status, invalid_day = request_json(base_url, "/api/discussions/day?user_id=u1&date=20-07-2026")
        assert status == 400
        assert "YYYY-MM-DD" in invalid_day["detail"]

        status, invalid_scope = request_json(
            base_url,
            f"/api/discussions/day?user_id=u1&date={day}&scope=invalid",
            method="DELETE",
        )
        assert status == 400
        assert "raw, summary, or all" in invalid_scope["detail"]

        status, deleted = request_json(
            base_url,
            f"/api/discussions/day?user_id=u1&date={day}&scope=all",
            method="DELETE",
        )
        assert status == 200
        assert deleted["purged_raw_count"] == 1
        assert service.timeline_store.get_discussion_day("u1", day) is None
        assert service.timeline_store.get_discussion_day("u2", day) is not None
