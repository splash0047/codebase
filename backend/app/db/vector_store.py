"""
Vector Store — FAISS (MVP) with Pinecone upgrade path (Phase 2 – Step 8).

Architecture:
  • FAISSVectorStore  — local, fast, zero-cost for development/MVP
  • PineconeVectorStore — production-grade managed index (Phase 5 upgrade)
  • VectorStoreRouter  — selects the right backend based on config

Each stored record includes:
  chunk_hash, file_path, symbol_name, chunk_type, start_line, end_line
so retrieval results can be matched back to their source exactly.
"""
from __future__ import annotations

import json
import os
import pickle
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger   = get_logger(__name__)

VECTOR_DIM = 1536   # text-embedding-3-small

# ── Base interface ─────────────────────────────────────────────────────────────

class BaseVectorStore(ABC):
    @abstractmethod
    async def upsert(self, records: list[dict[str, Any]]) -> list[str]:
        """Upsert vectors. Returns list of vector_ids."""

    @abstractmethod
    async def search(
        self, query_vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[dict[str, Any]]:
        """Search for similar vectors. Returns ranked results."""

    @abstractmethod
    async def delete_by_file(self, file_path: str) -> int:
        """Delete all vectors for a file. Returns count deleted."""


# ── FAISS Vector Store ─────────────────────────────────────────────────────────

class FAISSVectorStore(BaseVectorStore):
    """
    Local FAISS index with metadata sidecar (JSON).
    Suitable for MVP (<1M vectors). For production switch to PineconeVectorStore.
    """

    INDEX_DIR = Path("./faiss_index")

    def __init__(self, repo_id: str):
        self.repo_id   = repo_id
        self.index_dir = self.INDEX_DIR / repo_id
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index    = None
        self._metadata: dict[str, dict[str, Any]] = {}   # vector_id → metadata
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    def _meta_path(self) -> Path:
        return self.index_dir / "metadata.json"

    def _load(self) -> None:
        try:
            import faiss
            if self._index_path().exists():
                self._index = faiss.read_index(str(self._index_path()))
                logger.info("faiss_index_loaded", repo_id=self.repo_id)
            else:
                self._index = faiss.IndexIDMap(faiss.IndexFlatIP(VECTOR_DIM))
                logger.info("faiss_index_created", repo_id=self.repo_id)

            if self._meta_path().exists():
                with open(self._meta_path()) as f:
                    self._metadata = json.load(f)
        except ImportError:
            logger.error("faiss_not_installed")
            raise

    def _save(self) -> None:
        import faiss
        faiss.write_index(self._index, str(self._index_path()))
        with open(self._meta_path(), "w") as f:
            json.dump(self._metadata, f)

    # ── Upsert ─────────────────────────────────────────────────────────────────

    async def upsert(self, records: list[dict[str, Any]]) -> list[str]:
        """
        Each record must have: vector, chunk_hash, file_path, symbol_name,
        chunk_type, start_line, end_line.
        """
        import faiss

        vectors: list[list[float]] = []
        ids:     list[int]         = []
        vid_map: list[str]         = []

        for rec in records:
            vec = rec.get("vector")
            if vec is None:
                continue
            vector_id = str(uuid.uuid4())
            # Use first 8 bytes of UUID as FAISS integer ID
            int_id = int(uuid.UUID(vector_id).int >> 96)
            ids.append(int_id)
            vectors.append(vec)
            self._metadata[vector_id] = {
                "int_id":      int_id,
                "chunk_hash":  rec.get("chunk_hash"),
                "file_path":   rec.get("file_path"),
                "symbol_name": rec.get("symbol_name"),
                "chunk_type":  rec.get("chunk_type"),
                "start_line":  rec.get("start_line"),
                "end_line":    rec.get("end_line"),
                "chunk_text":  rec.get("chunk_text", ""),
                "intent_meta": rec.get("intent_meta", {}),
            }
            vid_map.append(vector_id)

        if vectors:
            mat = np.array(vectors, dtype=np.float32)
            # Normalise for cosine similarity (Inner Product on normalised = cosine)
            faiss.normalize_L2(mat)
            self._index.add_with_ids(mat, np.array(ids, dtype=np.int64))
            self._save()
            logger.info("faiss_upserted", count=len(vectors), repo_id=self.repo_id)

        return vid_map

    # ── Search ─────────────────────────────────────────────────────────────────

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        import faiss

        # Hard ceiling from plan
        top_k = min(top_k, settings.max_top_k)

        mat = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(mat)

        scores, int_ids = self._index.search(mat, top_k * 3)  # over-fetch for filtering

        # Build int_id → vector_id reverse map (lazy)
        int_to_vid = {meta["int_id"]: vid for vid, meta in self._metadata.items()}

        results: list[dict[str, Any]] = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id == -1:
                continue
            vid  = int_to_vid.get(int(int_id))
            if not vid:
                continue
            meta = self._metadata[vid]

            # Apply file_path filter if provided
            if filters and filters.get("file_path"):
                if meta.get("file_path") != filters["file_path"]:
                    continue

            results.append({
                "vector_id":   vid,
                "score":       float(score),
                "file_path":   meta.get("file_path"),
                "symbol_name": meta.get("symbol_name"),
                "chunk_type":  meta.get("chunk_type"),
                "start_line":  meta.get("start_line"),
                "end_line":    meta.get("end_line"),
                "chunk_text":  meta.get("chunk_text"),
                "intent_meta": meta.get("intent_meta", {}),
            })
            if len(results) >= top_k:
                break

        return results

    # ── Delete ─────────────────────────────────────────────────────────────────

    async def delete_by_file(self, file_path: str) -> int:
        to_remove = [
            (vid, meta["int_id"])
            for vid, meta in self._metadata.items()
            if meta.get("file_path") == file_path
        ]
        if not to_remove:
            return 0
        import numpy as np
        int_ids = np.array([t[1] for t in to_remove], dtype=np.int64)
        self._index.remove_ids(int_ids)
        for vid, _ in to_remove:
            del self._metadata[vid]
        self._save()
        logger.info("faiss_deleted", file=file_path, count=len(to_remove))
        return len(to_remove)


# ── Pinecone Vector Store (Production upgrade path) ────────────────────────────

class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone-backed vector store for production.
    Uses the pinecone-mcp-server tooling already available.
    Activated by setting VECTOR_STORE=pinecone in env.
    """

    def __init__(self, repo_id: str, namespace: str | None = None):
        self.repo_id   = repo_id
        self.namespace = namespace or repo_id
        self._index_name = os.getenv("PINECONE_INDEX_NAME", "codebase-knowledge")
        # Client is initialised lazily via mcp tooling
        logger.info("pinecone_store_init", namespace=self.namespace)

    async def upsert(self, records: list[dict[str, Any]]) -> list[str]:
        # Implemented in Phase 5 using pinecone-mcp-server upsert-records
        logger.info("pinecone_upsert_stub", count=len(records))
        return [str(uuid.uuid4()) for _ in records]

    async def search(
        self, query_vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[dict[str, Any]]:
        # Implemented in Phase 5 using pinecone-mcp-server search-records
        logger.info("pinecone_search_stub", top_k=top_k)
        return []

    async def delete_by_file(self, file_path: str) -> int:
        logger.info("pinecone_delete_stub", file=file_path)
        return 0


# ── Router ─────────────────────────────────────────────────────────────────────

_store_cache: dict[str, BaseVectorStore] = {}

def get_vector_store(repo_id: str) -> BaseVectorStore:
    """
    Returns the correct vector store backend based on VECTOR_STORE env var.
    Caches instances per repo_id.
    """
    if repo_id in _store_cache:
        return _store_cache[repo_id]

    backend = os.getenv("VECTOR_STORE", "faiss").lower()
    if backend == "pinecone":
        store = PineconeVectorStore(repo_id)
    else:
        store = FAISSVectorStore(repo_id)

    _store_cache[repo_id] = store
    return store
