from __future__ import annotations

import json
import ssl
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .agent_bridge import GlassesChatService, LocationContext, static_dir
from .memory_store import document_to_dict, event_to_dict
from .server_config import parse_server_bind, startup_message
from .tts_service import TTSServiceError, synthesize_speech


class ThreadingHTTPSServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, ssl_context: ssl.SSLContext):
        super().__init__(server_address, RequestHandlerClass)
        self._ssl_context = ssl_context

    # TLS handshake must happen inside the worker thread; otherwise one slow
    # Chrome/client handshake can block the main accept loop for every request.
    def process_request_thread(self, request, client_address):
        try:
            request.settimeout(10)
            tls_request = self._ssl_context.wrap_socket(
                request,
                server_side=True,
                do_handshake_on_connect=False,
            )
            tls_request.do_handshake()
        except OSError:
            self.shutdown_request(request)
            return
        super().process_request_thread(tls_request, client_address)


class GlassesHandler(SimpleHTTPRequestHandler):
    service: GlassesChatService | None = None

    # 标准库 server 直接服务 static 目录，保持 demo 不依赖 FastAPI。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(static_dir()), **kwargs)

    # /static/* 强制映射到包内静态资源，其他路径交给 SimpleHTTPRequestHandler。
    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        static_prefix = "/static/"
        if parsed.path.startswith(static_prefix):
            rel = parsed.path[len(static_prefix):]
            return str(static_dir() / rel)
        return super().translate_path(path)

    # GET 路由主要负责记忆查询、job 查询、周报、提醒检查和 audit 查看。
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(static_dir() / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/runtime":
            self._send_json({"routing_mode": "llm_first"})
            return
        if parsed.path == "/api/speaker/profile":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            self._send_json(self._service().get_speaker_profile(user_id=user_id))
            return
        if parsed.path == "/api/memories":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            limit = self._int_query(qs, "limit", 100)
            memories = self._service().memory_store.list_memories(user_id, limit=limit)
            documents = self._service().memory_store.list_documents(user_id, limit=limit)
            self._send_json({
                "memories": [event_to_dict(m) for m in memories],
                "documents": [document_to_dict(document) for document in documents],
            })
            return
        document_id = self._document_id(parsed.path)
        if document_id:
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            document = self._service().memory_store.get_document(user_id, document_id)
            if document is None:
                self._send_json({"detail": "document not found"}, status=HTTPStatus.NOT_FOUND)
            else:
                self._send_json({"document": document_to_dict(document, include_content=True)})
            return
        if parsed.path == "/api/memory/search":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            query = qs.get("q", [""])[0]
            limit = self._int_query(qs, "limit", 5)
            memories = self._service().memory_store.search(user_id, query, limit=limit)
            self._send_json({"memories": [event_to_dict(m) for m in memories]})
            return
        if parsed.path == "/api/timeline/search":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            query = qs.get("q", [""])[0]
            limit = self._int_query(qs, "limit", 5)
            self._send_json(self._service().search_timeline_for_management(user_id=user_id, query=query, limit=limit))
            return
        if parsed.path == "/api/timeline/chunks":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            chunk_ids = self._ids_query(qs)
            self._send_json(self._service().timeline_chunks_for_management(user_id=user_id, chunk_ids=chunk_ids))
            return
        if parsed.path == "/api/memory/jobs":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            job_id = qs.get("job_id", [""])[0]
            job = self._service().read_memory_job(user_id=user_id, job_id=job_id)
            if job is None:
                self._send_json({"detail": "memory job not found"}, status=HTTPStatus.NOT_FOUND)
            else:
                self._send_json({"job": job})
            return
        if parsed.path == "/api/weekly-report":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            self._send_json(self._service().weekly_report(user_id=user_id))
            return
        if parsed.path == "/api/reminders/check":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            self._send_json(self._service().check_reminders(user_id=user_id))
            return
        if parsed.path == "/api/debug/audit":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            limit = self._int_query(qs, "limit", 50)
            records = self._service().read_audit_records(user_id=user_id, limit=limit)
            self._send_json({"records": records})
            return
        return super().do_GET()

    # POST 路由承接聊天、手动记忆、导入、capture 和 TTS。
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        # 聊天请求进入 GlassesChatService.chat 主链路。
        if parsed.path == "/api/chat":
            body = self._read_json()
            try:
                result = self._service().chat(
                    str(body.get("message") or ""),
                    user_id=str(body.get("user_id") or "local-user"),
                    session_id=body.get("session_id") or None,
                    location=LocationContext(**body["location"]) if isinstance(body.get("location"), dict) else None,
                    defer_memory_writes=bool(body.get("defer_memory_writes")),
                    ambient_capture_id=str(body.get("ambient_capture_id") or ""),
                    wake_session=body.get("wake_session") if isinstance(body.get("wake_session"), dict) else None,
                )
            except ValueError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._log_exception("/api/chat", exc)
                self._send_json({"detail": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/memories":
            body = self._read_json()
            try:
                memory = self._service().memory_store.add_memory(
                    str(body.get("user_id") or "local-user"),
                    str(body.get("content") or ""),
                    kind=str(body.get("kind") or "event"),
                    memory_type=str(body.get("memory_type") or ""),
                    tags=body.get("tags") if isinstance(body.get("tags"), list) else [],
                    source="manual",
                    source_id=str(body.get("source_id") or ""),
                    ingestion_id=str(body.get("ingestion_id") or ""),
                    evidence_ids=body.get("evidence_ids") if isinstance(body.get("evidence_ids"), list) else [],
                    occurred_at=self._optional_float(body.get("occurred_at")),
                    start_at=self._optional_float(body.get("start_at")),
                    end_at=self._optional_float(body.get("end_at")),
                    time_granularity=str(body.get("time_granularity") or "unknown"),
                    temporal_text=str(body.get("temporal_text") or ""),
                    temporal_confidence=self._optional_float(body.get("temporal_confidence")),
                    privacy_level=str(body.get("privacy_level") or "normal"),
                    status=str(body.get("status") or "active"),
                    confidence=self._optional_float(body.get("confidence")),
                )
            except ValueError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"memory": event_to_dict(memory)})
            return
        # 文本/JSON 导入复用 service 的统一记忆候选和门控。
        if parsed.path == "/api/memory/import":
            body = self._read_json()
            try:
                result = self._service().import_memory_events(
                    user_id=str(body.get("user_id") or "local-user"),
                    items=body.get("items") if isinstance(body.get("items"), list) else None,
                    text=str(body.get("text") or ""),
                    source=str(body.get("source") or "manual_import"),
                    context=str(body.get("context") or ""),
                    confirm=bool(body.get("confirm", False)),
                    occurred_at=self._optional_float(body.get("occurred_at")),
                )
            except ValueError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/capture/start":
            body = self._read_json()
            self._send_json(self._service().start_capture(
                user_id=str(body.get("user_id") or "local-user"),
                source=str(body.get("source") or "continuous_capture"),
                context=str(body.get("context") or ""),
            ))
            return
        if parsed.path == "/api/capture/append":
            body = self._read_json()
            try:
                result = self._service().append_capture_chunk(
                    user_id=str(body.get("user_id") or "local-user"),
                    capture_id=str(body.get("capture_id") or ""),
                    text=str(body.get("text") or ""),
                    timestamp=self._optional_float(body.get("timestamp")),
                    metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                )
            except ValueError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return
        # 停止 capture 时会把片段汇总后导入记忆管道。
        if parsed.path == "/api/capture/stop":
            body = self._read_json()
            try:
                result = self._service().stop_capture(
                    user_id=str(body.get("user_id") or "local-user"),
                    capture_id=str(body.get("capture_id") or ""),
                    confirm=bool(body.get("confirm", False)),
                )
            except ValueError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/audio/segment/process":
            body = self._read_json()
            try:
                result = self._service().process_audio_segment(
                    user_id=str(body.get("user_id") or "local-user"),
                    transcript_hint=str(body.get("transcript_hint") or ""),
                    capture_id=str(body.get("capture_id") or ""),
                    source_type=str(body.get("source_type") or "ambient_audio"),
                    simulate=str(body.get("simulate") or "success"),
                    emotion_metadata=body.get("emotion_metadata") if isinstance(body.get("emotion_metadata"), dict) else None,
                    timestamp=self._optional_float(body.get("timestamp")),
                    audio_base64=str(body.get("audio_base64") or ""),
                    audio_mime_type=str(body.get("audio_mime_type") or ""),
                    audio_duration_ms=self._optional_int(body.get("audio_duration_ms")),
                )
            except ValueError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/speaker/enroll":
            body = self._read_json()
            try:
                result = self._service().enroll_speaker_profile(
                    user_id=str(body.get("user_id") or "local-user"),
                    audio_base64=str(body.get("audio_base64") or ""),
                    audio_mime_type=str(body.get("audio_mime_type") or ""),
                    audio_duration_ms=self._optional_int(body.get("audio_duration_ms")),
                    enrollment_session_id=str(body.get("enrollment_session_id") or ""),
                    sample_index=self._optional_int(body.get("sample_index")) or 1,
                    sample_total=self._optional_int(body.get("sample_total")) or 3,
                    finalize=bool(body.get("finalize", False)),
                )
            except ValueError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return
        # TTS 返回二进制音频，不走 JSON 响应封装。
        if parsed.path == "/api/tts":
            body = self._read_json()
            try:
                speech = synthesize_speech(
                    str(body.get("text") or ""),
                    voice=body.get("voice") if isinstance(body.get("voice"), str) else None,
                    rate=body.get("rate") if isinstance(body.get("rate"), str) else None,
                )
            except ValueError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except TTSServiceError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self._send_bytes(
                speech.audio,
                speech.media_type,
                headers={"X-TTS-Provider": speech.provider, "X-TTS-Voice": speech.voice},
            )
            return
        self._send_json({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)

    # PATCH 只用于文档管理，允许用户修正文档 metadata 和原文。
    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        document_id = self._document_id(parsed.path)
        if document_id:
            body = self._read_json()
            try:
                document = self._service().memory_store.update_document(
                    str(body.get("user_id") or "local-user"),
                    document_id,
                    filename=self._optional_string(body, "filename"),
                    title=self._optional_string(body, "title"),
                    summary=self._optional_string(body, "summary"),
                    content=self._optional_string(body, "content"),
                )
            except ValueError as exc:
                self._send_json({"detail": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if document is None:
                self._send_json({"detail": "document not found"}, status=HTTPStatus.NOT_FOUND)
            else:
                self._send_json({"document": document_to_dict(document, include_content=True)})
            return
        self._send_json({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)

    # 默认删除仍是软删除；只有显式 purge=true 才进入不可恢复清理。
    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/speaker/profile":
            qs = parse_qs(parsed.query)
            self._send_json(self._service().cancel_speaker_enrollment(
                user_id=qs.get("user_id", ["local-user"])[0],
                enrollment_session_id=qs.get("enrollment_session_id", [""])[0],
            ))
            return
        if parsed.path == "/api/timeline/chunks":
            qs = parse_qs(parsed.query)
            body = self._read_json()
            body_ids = body.get("chunk_ids") if isinstance(body.get("chunk_ids"), list) else []
            chunk_ids = [str(item) for item in body_ids] or self._ids_query(qs)
            self._send_json(self._service().delete_timeline_chunks(
                user_id=qs.get("user_id", ["local-user"])[0],
                chunk_ids=chunk_ids,
                purge=self._bool_query(qs, "purge", False),
            ))
            return
        document_id = self._document_id(parsed.path)
        if document_id:
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            if self._bool_query(qs, "purge", False):
                result = self._service().purge_document(user_id=user_id, document_id=document_id)
                if result["deleted"]:
                    self._send_json(result)
                else:
                    self._send_json({"detail": "document not found"}, status=HTTPStatus.NOT_FOUND)
                return
            if self._service().memory_store.delete_document(user_id, document_id):
                self._send_json({"deleted": True})
            else:
                self._send_json({"detail": "document not found"}, status=HTTPStatus.NOT_FOUND)
            return
        prefix = "/api/memories/"
        if parsed.path.startswith(prefix):
            memory_id = parsed.path[len(prefix):]
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", ["local-user"])[0]
            if self._bool_query(qs, "purge", False):
                result = self._service().purge_memory(user_id=user_id, memory_id=memory_id)
                if result["deleted"]:
                    self._send_json(result)
                else:
                    self._send_json({"detail": "memory not found"}, status=HTTPStatus.NOT_FOUND)
                return
            if self._service().delete_memory(user_id=user_id, memory_id=memory_id):
                self._send_json({"deleted": True})
            else:
                self._send_json({"detail": "memory not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)

    # demo API 对坏 JSON 宽容处理为空对象，由各路由自行校验必填字段。
    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    # 所有 JSON 响应统一使用 UTF-8，方便前端直接展示中文 debug。
    def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str, *, headers: dict[str, str] | None = None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _log_exception(self, route: str, exc: Exception) -> None:
        self.log_error(
            "Unhandled exception in %s: %s: %s\n%s",
            route,
            type(exc).__name__,
            exc,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
        )

    # 标准库 server 复用单例 service，保留 session/job/capture 进程内状态。
    @classmethod
    def _service(cls) -> GlassesChatService:
        if cls.service is None:
            cls.service = GlassesChatService()
        return cls.service

    @staticmethod
    def _int_query(qs: dict[str, list[str]], key: str, default: int) -> int:
        try:
            return int(qs.get(key, [str(default)])[0])
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool_query(qs: dict[str, list[str]], key: str, default: bool) -> bool:
        value = str(qs.get(key, [str(default).lower()])[0]).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _ids_query(qs: dict[str, list[str]]) -> list[str]:
        values: list[str] = []
        values.extend(qs.get("id", []))
        values.extend(qs.get("ids", []))
        return [
            item.strip()
            for raw_value in values
            for item in str(raw_value or "").split(",")
            if item.strip()
        ]

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_string(body: dict, key: str) -> str | None:
        if key not in body or body.get(key) is None:
            return None
        return str(body.get(key))

    @staticmethod
    def _document_id(path: str) -> str:
        prefix = "/api/documents/"
        return path[len(prefix):] if path.startswith(prefix) else ""


# 命令行入口：默认绑定 0.0.0.0，方便同局域网手机访问 demo。
def main() -> None:
    bind = parse_server_bind()
    if bind.certfile and bind.keyfile:
        # 同局域网端侧测试需要 HTTPS，浏览器才允许定位和部分语音能力。
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=bind.certfile, keyfile=bind.keyfile)
        httpd = ThreadingHTTPSServer((bind.host, bind.port), GlassesHandler, context)
    else:
        httpd = ThreadingHTTPServer((bind.host, bind.port), GlassesHandler)
    print(startup_message(bind), flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
