from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ai_glasses_memory_assistant.memory_store import EventMemoryStore
from ai_glasses_memory_assistant.server import GlassesHandler


@contextmanager
def subject_api(tmp_path):
    store = EventMemoryStore(tmp_path / "subject-api.db")

    class SubjectAPIHandler(GlassesHandler):
        service = SimpleNamespace(memory_store=store)

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), SubjectAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        store._conn.close()


def request_json(base_url: str, path: str, *, method: str = "GET", body: dict | None = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=2) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_memories_api_exposes_subjects_and_filters_by_subject(tmp_path) -> None:
    with subject_api(tmp_path) as base_url:
        status, initial = request_json(base_url, "/api/memories?user_id=u1")
        assert status == 200
        assert initial["memories"] == []
        assert [(subject["subject_type"], subject["display_name"]) for subject in initial["subjects"]] == [
            ("self", "我"),
        ]

        _, named_memory = request_json(
            base_url,
            "/api/memories",
            method="POST",
            body={
                "user_id": "u1",
                "content": "张三喜欢喝苏打水",
                "kind": "profile",
                "subject_name": "张三",
                "subject_type": "named",
            },
        )
        named_subject_id = named_memory["memory"]["subject_id"]
        assert named_memory["memory"]["subject_name"] == "张三"
        assert named_memory["memory"]["subject_type"] == "named"

        _, reused_subject = request_json(
            base_url,
            "/api/memories",
            method="POST",
            body={
                "user_id": "u1",
                "content": "张三周五参加评审",
                "subject_id": named_subject_id,
            },
        )
        assert reused_subject["memory"]["subject_id"] == named_subject_id

        request_json(
            base_url,
            "/api/memories",
            method="POST",
            body={"user_id": "u1", "content": "我喜欢喝温水"},
        )

        _, filtered = request_json(
            base_url,
            f"/api/memories?user_id=u1&subject_id={named_subject_id}",
        )
        assert [memory["content"] for memory in filtered["memories"]] == [
            "张三周五参加评审",
            "张三喜欢喝苏打水",
        ]
        assert {subject["display_name"] for subject in filtered["subjects"]} == {"我", "张三"}

        _, subjects_payload = request_json(base_url, "/api/subjects?user_id=u1")
        assert [(subject["subject_type"], subject["display_name"]) for subject in subjects_payload["subjects"]] == [
            ("self", "我"),
            ("named", "张三"),
        ]

        _, zhang_search_memory = request_json(
            base_url,
            "/api/memories",
            method="POST",
            body={
                "user_id": "u1",
                "content": "shared marker for first person",
                "subject_id": named_subject_id,
            },
        )
        _, li_search_memory = request_json(
            base_url,
            "/api/memories",
            method="POST",
            body={
                "user_id": "u1",
                "content": "shared marker for second person",
                "subject_name": "李四",
                "subject_type": "named",
            },
        )
        _, search_payload = request_json(
            base_url,
            f"/api/memory/search?user_id=u1&q=shared&subject_id={named_subject_id}",
        )
        assert [memory["id"] for memory in search_payload["memories"]] == [
            zhang_search_memory["memory"]["id"]
        ]
        assert li_search_memory["memory"]["id"] not in {
            memory["id"] for memory in search_payload["memories"]
        }


def test_manual_memory_subject_id_cannot_cross_user_boundary(tmp_path) -> None:
    with subject_api(tmp_path) as base_url:
        _, memory_payload = request_json(
            base_url,
            "/api/memories",
            method="POST",
            body={
                "user_id": "u1",
                "content": "李四喜欢喝矿泉水",
                "subject_name": "李四",
                "subject_type": "named",
            },
        )
        u1_subject_id = memory_payload["memory"]["subject_id"]

        with pytest.raises(HTTPError) as exc_info:
            request_json(
                base_url,
                "/api/memories",
                method="POST",
                body={
                    "user_id": "u2",
                    "content": "不应写入李四",
                    "subject_id": u1_subject_id,
                },
            )

        assert exc_info.value.code == 400
        error_payload = json.loads(exc_info.value.read().decode("utf-8"))
        assert error_payload["detail"] == "subject does not exist for user"

        _, u2_memories = request_json(base_url, "/api/memories?user_id=u2")
        assert u2_memories["memories"] == []
        assert [(subject["subject_type"], subject["display_name"]) for subject in u2_memories["subjects"]] == [
            ("self", "我"),
        ]
