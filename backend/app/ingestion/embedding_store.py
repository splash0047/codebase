"""
Embedding persistence layer.

After EmbeddingService produces vectors, this module:
 1. Saves Embedding rows to Postgres (chunk_hash, vector_id, intent_meta)
 2. Upserts vectors into the vector store (FAISS/Pinecone)
 3. Tracks existing chunk_hashes to enable incremental re-embedding
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.vector_store import get_vector_store
from app.ingestion.chunker import CodeChunk
from app.ingestion.embedder import EmbeddingService
from app.models.db_models import Embedding, File, ChunkType

logger = get_logger(__name__)


async def load_known_hashes(db: AsyncSession, file_id: str) -> dict[str, str]:
    """Load existing chunk_hashes from Postgres for a file (incremental re-embed)."""
    result = await db.execute(
        select(Embedding.vector_id, Embedding.chunk_hash, Embedding.chunk_type,
               Embedding.symbol_name, Embedding.file_id)
        .where(Embedding.file_id == file_id)   # type: ignore[arg-type]
    )
    rows = result.all()
    return {
        f"{r.symbol_name}::{r.chunk_type}": r.chunk_hash
        for r in rows if r.chunk_hash
    }


async def persist_embeddings(
    db: AsyncSession,
    file_obj: File,
    chunks: list[CodeChunk],
    repo_id: str,
) -> dict[str, Any]:
    """
    Full Phase 2 pipeline for a single file:
      load hashes → embed changed chunks → upsert vectors → save to Postgres
    """
    file_id   = str(file_obj.id)
    known     = await load_known_hashes(db, file_id)

    service   = EmbeddingService(known_hashes=known)
    results   = await service.embed_chunks(chunks)

    vector_store = get_vector_store(repo_id)
    to_upsert: list[dict[str, Any]] = []

    for r in results:
        if not r["success"]:
            continue
        chunk: CodeChunk = r["chunk"]
        to_upsert.append({
            "vector":      r["vector"],
            "chunk_hash":  chunk.chunk_hash,
            "file_path":   chunk.file_path,
            "symbol_name": chunk.symbol_name,
            "chunk_type":  chunk.chunk_type.value,
            "start_line":  chunk.start_line,
            "end_line":    chunk.end_line,
            "chunk_text":  chunk.text,
            "intent_meta": chunk.intent_meta,
        })

    vector_ids = await vector_store.upsert(to_upsert)

    # ── Persist to Postgres ─────────────────────────────────────────────────
    saved = 0
    for i, r in enumerate([r for r in results if r["success"]]):
        chunk: CodeChunk = r["chunk"]
        vid = vector_ids[i] if i < len(vector_ids) else None

        # Upsert Embedding row (idempotent via chunk_hash match)
        existing = await db.execute(
            select(Embedding).where(
                Embedding.file_id == file_obj.id,
                Embedding.symbol_name == chunk.symbol_name,
                Embedding.chunk_type == chunk.chunk_type,
            )
        )
        emb = existing.scalar_one_or_none()
        if emb:
            emb.chunk_hash  = chunk.chunk_hash
            emb.vector_id   = vid
            emb.intent_meta = chunk.intent_meta
            emb.chunk_text  = chunk.text
            emb.commit_id   = chunk.commit_id
        else:
            emb = Embedding(
                file_id      = file_obj.id,
                chunk_type   = chunk.chunk_type,
                chunk_index  = i,
                symbol_name  = chunk.symbol_name,
                start_line   = chunk.start_line,
                end_line     = chunk.end_line,
                commit_id    = chunk.commit_id,
                chunk_hash   = chunk.chunk_hash,
                vector_id    = vid,
                chunk_text   = chunk.text,
                intent_meta  = chunk.intent_meta,
            )
            db.add(emb)
        saved += 1

    await db.commit()
    logger.info(
        "embeddings_persisted",
        file_id=file_id,
        saved=saved,
        failed=len([r for r in results if not r["success"]]),
    )
    return {"saved": saved, "failed": len(results) - saved}
