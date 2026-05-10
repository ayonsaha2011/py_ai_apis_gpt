from __future__ import annotations

import hashlib
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from .config import settings

_client: AsyncQdrantClient | None = None
_embedder: SentenceTransformer | None = None


async def start_rag() -> None:
    global _client, _embedder
    _client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    _embedder = SentenceTransformer(settings.rag_embed_model, device="cpu")


async def close_rag() -> None:
    if _client is not None:
        await _client.close()


async def ensure_collection(collection: str) -> None:
    client = _require_client()
    exists = await client.collection_exists(collection)
    if not exists:
        await client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=settings.rag_vector_size, distance=qmodels.Distance.COSINE),
        )


async def ingest(collection: str, texts: list[str], source_name: str, user_id: str, metadata: dict[str, Any]) -> int:
    await ensure_collection(collection)
    chunks = []
    for text in texts:
        chunks.extend(_chunk(text, settings.rag_chunk_size, settings.rag_chunk_overlap))
    if not chunks:
        return 0
    vectors = _embed(chunks)
    points = []
    doc_id = str(uuid.uuid7()) if hasattr(uuid, "uuid7") else str(uuid.uuid4())
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        stable = hashlib.sha256(f"{user_id}:{source_name}:{idx}:{chunk}".encode()).hexdigest()
        points.append(
            qmodels.PointStruct(
                id=stable,
                vector=vector,
                payload={
                    "user_id": user_id,
                    "document_id": doc_id,
                    "source_name": source_name,
                    "text": chunk,
                    "metadata": metadata,
                },
            )
        )
    await _require_client().upsert(collection_name=collection, points=points)
    return len(points)


async def retrieve(collection: str, query: str, top_k: int | None = None) -> list[str]:
    await ensure_collection(collection)
    vector = _embed([query])[0]
    hits = await _require_client().search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k or settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )
    return [str(hit.payload.get("text", "")) for hit in hits if hit.payload]


async def delete_document(collection: str, document_id: str) -> None:
    await _require_client().delete(
        collection_name=collection,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
            )
        ),
    )


def _embed(texts: list[str]) -> list[list[float]]:
    if _embedder is None:
        raise RuntimeError("RAG embedder not loaded")
    vectors = _embedder.encode(texts, normalize_embeddings=True)
    return vectors.astype("float32").tolist()


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def _require_client() -> AsyncQdrantClient:
    if _client is None:
        raise RuntimeError("Qdrant client not initialized")
    return _client

