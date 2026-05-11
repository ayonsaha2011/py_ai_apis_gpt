from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .config import settings
from .rag import close_rag, ensure_collection, ingest, retrieve, start_rag
from .scheduler import GenerationScheduler
from .schemas import ChatMessage, ChatRequest, RagCollectionRequest, RagIngestRequest

logger = logging.getLogger(__name__)
scheduler = GenerationScheduler()


def _check_service_key(x_service_key: str | None) -> None:
    if settings.service_api_key and x_service_key != settings.service_api_key:
        raise HTTPException(status_code=401, detail="invalid service key")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await start_rag()
    await scheduler.start()
    yield
    await scheduler.stop()
    await close_rag()


app = FastAPI(title="text-worker", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code >= 500:
        logger.error(
            "text-worker HTTP error method=%s path=%s status=%s detail=%s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        logger.warning(
            "text-worker request rejected method=%s path=%s status=%s detail=%s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "text-worker unhandled error method=%s path=%s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(status_code=500, content={"detail": "internal text-worker error; see server logs"})


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_id": scheduler.model_id,
        "model_loaded": scheduler.model_loaded(),
        "text_model_dir": str(settings.text_model_dir),
    }


@app.post("/admin/models/start")
async def start_model(body: dict, x_service_key: str | None = Header(default=None)) -> dict:
    _check_service_key(x_service_key)
    model_id = body.get("model_id")
    if not model_id:
        raise HTTPException(status_code=422, detail="model_id required")
    await scheduler.switch_model(str(model_id))
    return {"status": "ready", "model_id": scheduler.model_id}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_service_key: str | None = Header(default=None),
    x_rag_qdrant_collection: str | None = Header(default=None),
):
    _check_service_key(x_service_key)
    body = await request.json()
    try:
        req = ChatRequest.model_validate(body)
    except ValidationError as exc:
        logger.warning("invalid chat request detail=%s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        scheduler.assert_serves_model(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rag_audit: dict | None = None
    if req.use_rag and x_rag_qdrant_collection:
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        if last_user:
            hits = await retrieve(x_rag_qdrant_collection, last_user)
            if hits:
                chunks = [hit["text"] for hit in hits if hit.get("text")]
                rag_audit = {
                    "collection": x_rag_qdrant_collection,
                    "document_ids": sorted({hit["document_id"] for hit in hits if hit.get("document_id")}),
                    "hits": [
                        {
                            "document_id": hit.get("document_id"),
                            "source_name": hit.get("source_name"),
                            "score": hit.get("score"),
                        }
                        for hit in hits
                    ],
                }
                req.messages.insert(
                    0,
                    ChatMessage(
                        role="system",
                        content="Use the retrieved context when it is relevant.\n\n" + "\n\n---\n\n".join(chunks),
                    ),
                )

    model_name = req.model or scheduler.model_id
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    headers = {"X-RAG-Audit": json.dumps(rag_audit)} if rag_audit else {}
    if req.stream:
        async def events():
            try:
                async for token in scheduler.stream(req):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(
                    "chat stream failed completion_id=%s model=%s session_id=%s",
                    completion_id,
                    model_name,
                    req.session_id,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                error_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "error": {"message": "stream generation failed; see text-worker logs"},
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", **headers},
        )

    text = await scheduler.submit(req)
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        },
        headers=headers,
    )


@app.post("/rag/collections")
async def create_collection(body: RagCollectionRequest, x_service_key: str | None = Header(default=None)) -> dict:
    _check_service_key(x_service_key)
    await ensure_collection(body.collection)
    return {"collection": body.collection, "status": "ready"}


@app.post("/rag/ingest")
async def rag_ingest(
    body: RagIngestRequest,
    x_service_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> dict:
    _check_service_key(x_service_key)
    if not x_user_id:
        raise HTTPException(status_code=401, detail="x-user-id required")
    count = await ingest(body.collection, body.document_id, body.texts, body.source_name, x_user_id, body.metadata)
    return {"collection": body.collection, "document_id": body.document_id, "ingested_chunks": count}


@app.delete("/rag/documents/{document_id}")
async def rag_delete(
    document_id: str,
    x_service_key: str | None = Header(default=None),
    x_rag_qdrant_collection: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> dict:
    _check_service_key(x_service_key)
    if not x_rag_qdrant_collection:
        raise HTTPException(status_code=422, detail="x-rag-qdrant-collection required")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="x-user-id required")
    from .rag import delete_document

    await delete_document(x_rag_qdrant_collection, document_id, x_user_id)
    return {"document_id": document_id, "status": "deleted"}
