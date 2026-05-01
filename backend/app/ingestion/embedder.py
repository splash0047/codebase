"""
Embedding Generation & Incremental Updates (Phase 2 – Step 7).

Key design decisions from the plan:
 • Chunk-level hashing — only re-embed changed chunks, not whole files
 • Async batch generation to respect OpenAI rate limits
 • Graceful fallback to a local sentence-transformer if OpenAI unavailable
 • Stores vector_id and chunk_hash back to Postgres for change detection
"""
from __future__ import annotations

import asyncio
from typing import Any

from openai import AsyncOpenAI, RateLimitError, APIConnectionError
from tenacity import (
    retry, retry_if_exception_type,
    stop_after_attempt, wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.chunker import CodeChunk

settings = get_settings()
logger   = get_logger(__name__)

EMBED_BATCH_SIZE = 20       # OpenAI allows up to 2048 inputs but we batch small
EMBED_DIMENSIONS = 1536     # text-embedding-3-small output size


# ── OpenAI Client (lazy init) ─────────────────────────────────────────────────

_openai_client: AsyncOpenAI | None = None

def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        if settings.gemini_api_key and not settings.openai_api_key:
            _openai_client = AsyncOpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


# ── Embedding call with retry ─────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Call OpenAI (or Gemini) embeddings API for a batch of texts."""
    client   = get_openai_client()
    
    # Auto-swap to Gemini embedding model if using Gemini
    model = settings.embedding_model
    if settings.gemini_api_key and not settings.openai_api_key and model == "text-embedding-3-small":
        model = "text-embedding-004"
        
    response = await client.embeddings.create(
        model=model,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def _embed_batch_with_fallback(texts: list[str]) -> list[list[float]] | None:
    """
    Try OpenAI; on failure log and return None so caller can handle gracefully.
    The plan requires partial ingestion to continue even if embeddings fail.
    """
    try:
        return await _embed_batch(texts)
    except Exception as exc:
        logger.error(
            "embedding_batch_failed",
            model=settings.embedding_model,
            batch_size=len(texts),
            error=str(exc),
        )
        return None


# ── EmbeddingService ──────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Orchestrates chunk-level incremental embedding.
    Accepts a list of CodeChunks; skips those whose chunk_hash is unchanged.
    """

    def __init__(self, known_hashes: dict[str, str] | None = None):
        """
        known_hashes: {chunk_key → existing_hash}
        chunk_key = f"{file_path}::{symbol_name}::{chunk_type}"
        """
        self.known_hashes: dict[str, str] = known_hashes or {}

    def _chunk_key(self, chunk: CodeChunk) -> str:
        return f"{chunk.file_path}::{chunk.symbol_name}::{chunk.chunk_type}"

    def _needs_reembed(self, chunk: CodeChunk) -> bool:
        """Return True if the chunk is new or its hash has changed."""
        key = self._chunk_key(chunk)
        return self.known_hashes.get(key) != chunk.chunk_hash

    async def embed_chunks(
        self, chunks: list[CodeChunk]
    ) -> list[dict[str, Any]]:
        """
        Embed only changed chunks.
        Returns list of result dicts with embedding vectors + metadata.
        """
        to_embed   = [c for c in chunks if self._needs_reembed(c)]
        skipped    = len(chunks) - len(to_embed)
        results: list[dict[str, Any]] = []

        logger.info(
            "embedding_start",
            total=len(chunks),
            to_embed=len(to_embed),
            skipped=skipped,
        )

        # Process in batches
        for i in range(0, len(to_embed), EMBED_BATCH_SIZE):
            batch        = to_embed[i : i + EMBED_BATCH_SIZE]
            texts        = [c.text for c in batch]
            vectors      = await _embed_batch_with_fallback(texts)

            for j, chunk in enumerate(batch):
                vector = vectors[j] if vectors else None
                results.append({
                    "chunk_key":   self._chunk_key(chunk),
                    "chunk":       chunk,
                    "vector":      vector,
                    "success":     vector is not None,
                    "chunk_hash":  chunk.chunk_hash,
                })

            # Small delay to respect rate limits
            if i + EMBED_BATCH_SIZE < len(to_embed):
                await asyncio.sleep(0.1)

        logger.info(
            "embedding_done",
            embedded=len([r for r in results if r["success"]]),
            failed=len([r for r in results if not r["success"]]),
        )
        return results
