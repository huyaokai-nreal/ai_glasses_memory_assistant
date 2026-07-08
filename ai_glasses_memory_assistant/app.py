from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent_bridge import GlassesChatService, LocationContext, static_dir
from .memory_store import document_to_dict, event_to_dict
from .server_config import parse_server_bind
from .tts_service import TTSServiceError, synthesize_speech_async


# FastAPI 入口与标准库 server 保持同一套 service 行为。
app = FastAPI(title="AI Glasses Memory Assistant", version="0.1.0")
service = GlassesChatService()


class LocationContextRequest(BaseModel):
    status: str = "missing"
    latitude: float | None = None
    longitude: float | None = None
    accuracy: float | None = None
    timestamp: float | None = None
    source: str = "browser_geolocation"
    error: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = "local-user"
    session_id: str | None = None
    location: LocationContextRequest | None = None
    defer_memory_writes: bool = False
    ambient_capture_id: str | None = None
    wake_session: dict[str, Any] | None = None


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1)
    user_id: str = "local-user"
    kind: str = "event"
    memory_type: str = ""
    tags: list[str] = []
    source_id: str = ""
    ingestion_id: str = ""
    evidence_ids: list[str] = []
    occurred_at: float | None = None
    start_at: float | None = None
    end_at: float | None = None
    time_granularity: str = "unknown"
    temporal_text: str = ""
    temporal_confidence: float | None = None
    privacy_level: str = "normal"
    status: str = "active"
    confidence: float | None = None


# 导入请求兼容纯文本和结构化 items，最终都走同一套记忆门控。
class MemoryImportRequest(BaseModel):
    user_id: str = "local-user"
    text: str = ""
    items: list[dict[str, Any]] | None = None
    source: str = "manual_import"
    context: str = ""
    confirm: bool = False
    occurred_at: float | None = None


class DocumentUpdateRequest(BaseModel):
    user_id: str = "local-user"
    filename: str | None = None
    title: str | None = None
    summary: str | None = None
    content: str | None = None


class TimelineChunksDeleteRequest(BaseModel):
    chunk_ids: list[str] = []


class CaptureStartRequest(BaseModel):
    user_id: str = "local-user"
    source: str = "continuous_capture"
    context: str = ""


class CaptureAppendRequest(BaseModel):
    user_id: str = "local-user"
    capture_id: str
    text: str = Field(min_length=1)
    timestamp: float | None = None
    metadata: dict[str, Any] | None = None


class CaptureStopRequest(BaseModel):
    user_id: str = "local-user"
    capture_id: str
    confirm: bool = False


class AudioSegmentProcessRequest(BaseModel):
    user_id: str = "local-user"
    transcript_hint: str = ""
    capture_id: str | None = None
    source_type: str = "ambient_audio"
    simulate: str = "success"
    emotion_metadata: dict[str, Any] | None = None
    timestamp: float | None = None
    audio_base64: str | None = None
    audio_mime_type: str | None = None
    audio_duration_ms: int | None = None


class SpeakerEnrollRequest(BaseModel):
    user_id: str = "local-user"
    audio_base64: str | None = None
    audio_mime_type: str | None = None
    audio_duration_ms: int | None = None
    enrollment_session_id: str = ""
    sample_index: int = 1
    sample_total: int = 3
    finalize: bool = False


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None
    rate: str | None = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir() / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/api/runtime")
async def runtime() -> dict[str, Any]:
    return {"routing_mode": "llm_first"}


# 主聊天 API 只是薄封装，业务流程全部在 GlassesChatService.chat。
@app.post("/api/chat")
async def chat(body: ChatRequest) -> dict[str, Any]:
    try:
        return service.chat(
            body.message,
            user_id=body.user_id,
            session_id=body.session_id,
            location=LocationContext(**body.location.dict()) if body.location else None,
            defer_memory_writes=body.defer_memory_writes,
            ambient_capture_id=body.ambient_capture_id or "",
            wake_session=body.wake_session if isinstance(body.wake_session, dict) else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/memories")
async def list_memories(user_id: str = "local-user", limit: int = 100) -> dict[str, Any]:
    memories = service.memory_store.list_memories(user_id, limit=limit)
    documents = service.memory_store.list_documents(user_id, limit=limit)
    return {
        "memories": [event_to_dict(m) for m in memories],
        "documents": [document_to_dict(document) for document in documents],
    }


@app.post("/api/memories")
async def create_memory(body: MemoryCreateRequest) -> dict[str, Any]:
    try:
        memory = service.memory_store.add_memory(
            body.user_id,
            body.content,
            kind=body.kind,
            memory_type=body.memory_type,
            tags=body.tags,
            source="manual",
            source_id=body.source_id,
            ingestion_id=body.ingestion_id,
            evidence_ids=body.evidence_ids,
            occurred_at=body.occurred_at,
            start_at=body.start_at,
            end_at=body.end_at,
            time_granularity=body.time_granularity,
            temporal_text=body.temporal_text,
            temporal_confidence=body.temporal_confidence,
            privacy_level=body.privacy_level,
            status=body.status,
            confidence=body.confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"memory": event_to_dict(memory)}


# 文本/JSON 导入 API 复用 service 的 import_memory_events。
@app.post("/api/memory/import")
async def import_memory(body: MemoryImportRequest) -> dict[str, Any]:
    try:
        return service.import_memory_events(
            user_id=body.user_id,
            items=body.items,
            text=body.text,
            source=body.source,
            context=body.context,
            confirm=body.confirm,
            occurred_at=body.occurred_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/capture/start")
async def start_capture(body: CaptureStartRequest) -> dict[str, Any]:
    return service.start_capture(user_id=body.user_id, source=body.source, context=body.context)


@app.post("/api/capture/append")
async def append_capture(body: CaptureAppendRequest) -> dict[str, Any]:
    try:
        return service.append_capture_chunk(
            user_id=body.user_id,
            capture_id=body.capture_id,
            text=body.text,
            timestamp=body.timestamp,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# stop 会触发 capture 文本汇总和导入，不只是改状态。
@app.post("/api/capture/stop")
async def stop_capture(body: CaptureStopRequest) -> dict[str, Any]:
    try:
        return service.stop_capture(
            user_id=body.user_id,
            capture_id=body.capture_id,
            confirm=body.confirm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/audio/segment/process")
async def process_audio_segment(body: AudioSegmentProcessRequest) -> dict[str, Any]:
    try:
        return service.process_audio_segment(
            user_id=body.user_id,
            transcript_hint=body.transcript_hint,
            capture_id=body.capture_id or "",
            source_type=body.source_type,
            simulate=body.simulate,
            emotion_metadata=body.emotion_metadata,
            timestamp=body.timestamp,
            audio_base64=body.audio_base64,
            audio_mime_type=body.audio_mime_type,
            audio_duration_ms=body.audio_duration_ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/speaker/enroll")
async def speaker_enroll(body: SpeakerEnrollRequest) -> dict[str, Any]:
    try:
        return service.enroll_speaker_profile(
            user_id=body.user_id,
            audio_base64=str(body.audio_base64 or ""),
            audio_mime_type=str(body.audio_mime_type or ""),
            audio_duration_ms=body.audio_duration_ms,
            enrollment_session_id=body.enrollment_session_id,
            sample_index=body.sample_index,
            sample_total=body.sample_total,
            finalize=body.finalize,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/speaker/profile")
async def speaker_profile(user_id: str = "local-user") -> dict[str, Any]:
    return service.get_speaker_profile(user_id=user_id)


@app.delete("/api/speaker/profile")
async def cancel_speaker_profile_enrollment(user_id: str = "local-user", enrollment_session_id: str = "") -> dict[str, Any]:
    return service.cancel_speaker_enrollment(user_id=user_id, enrollment_session_id=enrollment_session_id)


# TTS 返回音频 Response，供前端语音播报 fallback 使用。
@app.post("/api/tts")
async def tts(body: TTSRequest) -> Response:
    try:
        speech = await synthesize_speech_async(body.text, voice=body.voice, rate=body.rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TTSServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(
        content=speech.audio,
        media_type=speech.media_type,
        headers={
            "Cache-Control": "no-store",
            "X-TTS-Provider": speech.provider,
            "X-TTS-Voice": speech.voice,
        },
    )


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str, user_id: str = "local-user", purge: bool = False) -> dict[str, Any]:
    if purge:
        result = service.purge_memory(user_id=user_id, memory_id=memory_id)
        if not result["deleted"]:
            raise HTTPException(status_code=404, detail="memory not found")
        return result
    deleted = service.delete_memory(user_id=user_id, memory_id=memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": True}


@app.get("/api/documents/{document_id}")
async def get_document(document_id: str, user_id: str = "local-user") -> dict[str, Any]:
    document = service.memory_store.get_document(user_id, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return {"document": document_to_dict(document, include_content=True)}


# 文档编辑允许修正文档原文，召回和哈希都会随之更新。
@app.patch("/api/documents/{document_id}")
async def update_document(document_id: str, body: DocumentUpdateRequest) -> dict[str, Any]:
    try:
        document = service.memory_store.update_document(
            body.user_id,
            document_id,
            filename=body.filename,
            title=body.title,
            summary=body.summary,
            content=body.content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return {"document": document_to_dict(document, include_content=True)}


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, user_id: str = "local-user", purge: bool = False) -> dict[str, Any]:
    if purge:
        result = service.purge_document(user_id=user_id, document_id=document_id)
        if not result["deleted"]:
            raise HTTPException(status_code=404, detail="document not found")
        return result
    deleted = service.memory_store.delete_document(user_id, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="document not found")
    return {"deleted": True}


@app.get("/api/memory/search")
async def search_memories(q: str, user_id: str = "local-user", limit: int = 5) -> dict[str, Any]:
    memories = service.memory_store.search(user_id, q, limit=limit)
    return {"memories": [event_to_dict(m) for m in memories]}


@app.get("/api/timeline/search")
async def search_timeline(q: str, user_id: str = "local-user", limit: int = 5) -> dict[str, Any]:
    return service.search_timeline_for_management(user_id=user_id, query=q, limit=limit)


@app.get("/api/timeline/chunks")
async def get_timeline_chunks(ids: str = "", user_id: str = "local-user") -> dict[str, Any]:
    chunk_ids = [item.strip() for item in ids.split(",") if item.strip()]
    return service.timeline_chunks_for_management(user_id=user_id, chunk_ids=chunk_ids)


@app.delete("/api/timeline/chunks")
async def delete_timeline_chunks(
    user_id: str = "local-user",
    ids: str = "",
    purge: bool = False,
    body: TimelineChunksDeleteRequest | None = Body(default=None),
) -> dict[str, Any]:
    chunk_ids = body.chunk_ids if body and body.chunk_ids else [item.strip() for item in ids.split(",") if item.strip()]
    return service.delete_timeline_chunks(user_id=user_id, chunk_ids=chunk_ids, purge=purge)


# 后台 job 是进程内 demo 状态，按 user_id + job_id 查询。
@app.get("/api/memory/jobs")
async def read_memory_job(user_id: str = "local-user", job_id: str = "") -> dict[str, Any]:
    job = service.read_memory_job(user_id=user_id, job_id=job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="memory job not found")
    return {"job": job}


@app.get("/api/weekly-report")
async def weekly_report(user_id: str = "local-user") -> dict[str, Any]:
    return service.weekly_report(user_id=user_id)


# 当前提醒只是手动检查候选，不是主动推送 runtime。
@app.get("/api/reminders/check")
async def check_reminders(user_id: str = "local-user") -> dict[str, Any]:
    return service.check_reminders(user_id=user_id)


@app.get("/api/debug/audit")
async def read_audit(user_id: str = "local-user", limit: int = 50) -> dict[str, Any]:
    return {"records": service.read_audit_records(user_id=user_id, limit=limit)}


app.mount("/static", StaticFiles(directory=static_dir()), name="static")


# 可选 FastAPI 启动入口，默认命令仍可用标准库 server。
def main() -> None:
    import uvicorn

    bind = parse_server_bind()
    uvicorn.run(
        app,
        host=bind.host,
        port=bind.port,
        log_level="info",
        ssl_certfile=bind.certfile,
        ssl_keyfile=bind.keyfile,
    )


if __name__ == "__main__":
    main()
