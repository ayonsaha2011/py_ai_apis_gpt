from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .rag import close_rag, ensure_collection, ingest, retrieve, start_rag
from .scheduler import GenerationScheduler
from .schemas import ChatMessage, ChatRequest, RagCollectionRequest, RagIngestRequest

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


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_id": scheduler.model_id,
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
    req = ChatRequest.model_validate(body)
    if req.use_rag and x_rag_qdrant_collection:
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        if last_user:
            chunks = await retrieve(x_rag_qdrant_collection, last_user)
            if chunks:
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
    if req.stream:
        async def events():
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

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    text = await scheduler.submit(req)
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        }
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
    count = await ingest(body.collection, body.texts, body.source_name, x_user_id, body.metadata)
    return {"collection": body.collection, "ingested_chunks": count}


@app.delete("/rag/documents/{document_id}")
async def rag_delete(
    document_id: str,
    x_service_key: str | None = Header(default=None),
    x_rag_qdrant_collection: str | None = Header(default=None),
) -> dict:
    _check_service_key(x_service_key)
    if not x_rag_qdrant_collection:
        raise HTTPException(status_code=422, detail="x-rag-qdrant-collection required")
    from .rag import delete_document

    await delete_document(x_rag_qdrant_collection, document_id)
    return {"document_id": document_id, "status": "deleted"}
