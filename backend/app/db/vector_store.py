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


# ── Pinecone Vector Store (Production) ─────────────────────────────────────────

class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone-backed vector store for production.
    Uses the official pinecone-client SDK.
    Activated by setting VECTOR_STORE=pinecone in env.

    Required env vars:
      PINECONE_API_KEY    — Pinecone API key
      PINECONE_INDEX_NAME — Index name (default: codebase-knowledge)
    """

    def __init__(self, repo_id: str, namespace: str | None = None):
        self.repo_id   = repo_id
        self.namespace = namespace or repo_id
        self._index_name = os.getenv("PINECONE_INDEX_NAME", "codebase-knowledge")
        self._client = None
        self._index  = None
        logger.info("pinecone_store_init", namespace=self.namespace, index=self._index_name)

    def _ensure_client(self):
        """Lazy-initialise the Pinecone client and index."""
        if self._index is not None:
            return
        try:
            from pinecone import Pinecone

            api_key = os.getenv("PINECONE_API_KEY", "")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not set")

            self._client = Pinecone(api_key=api_key)
            self._index = self._client.Index(self._index_name)
            logger.info("pinecone_client_ready", index=self._index_name)
        except ImportError:
            logger.error("pinecone_sdk_not_installed")
            raise
        except Exception as e:
            logger.error("pinecone_init_failed", error=str(e))
            raise

    async def upsert(self, records: list[dict[str, Any]]) -> list[str]:
        """
        Upsert vectors with metadata to Pinecone.
        Each record must have: vector, chunk_hash, file_path, symbol_name,
        chunk_type, start_line, end_line.
        """
        self._ensure_client()

        vectors = []
        vid_map = []

        for rec in records:
            vec = rec.get("vector")
            if vec is None:
                continue
            vector_id = str(uuid.uuid4())
            metadata = {
                "chunk_hash":  rec.get("chunk_hash", ""),
                "file_path":   rec.get("file_path", ""),
                "symbol_name": rec.get("symbol_name", ""),
                "chunk_type":  rec.get("chunk_type", ""),
                "start_line":  rec.get("start_line", 0),
                "end_line":    rec.get("end_line", 0),
                "chunk_text":  rec.get("chunk_text", "")[:1000],  # Pinecone metadata limit
                "repo_id":     self.repo_id,
            }
            vectors.append({
                "id": vector_id,
                "values": vec,
                "metadata": metadata,
            })
            vid_map.append(vector_id)

        # Batch upsert in chunks of 100 (Pinecone recommended batch size)
        BATCH_SIZE = 100
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            try:
                self._index.upsert(vectors=batch, namespace=self.namespace)
            except Exception as e:
                logger.error("pinecone_upsert_failed", batch=i, error=str(e))
                raise

        logger.info("pinecone_upserted", count=len(vectors), namespace=self.namespace)
        return vid_map

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search Pinecone with optional metadata filtering."""
        self._ensure_client()

        top_k = min(top_k, settings.max_top_k)

        # Build Pinecone filter
        pc_filter: dict[str, Any] = {"repo_id": {"$eq": self.repo_id}}
        if filters and filters.get("file_path"):
            pc_filter["file_path"] = {"$eq": filters["file_path"]}

        try:
            response = self._index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=self.namespace,
                filter=pc_filter,
                include_metadata=True,
            )
        except Exception as e:
            logger.error("pinecone_search_failed", error=str(e))
            return []

        results: list[dict[str, Any]] = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append({
                "vector_id":   match["id"],
                "score":       float(match.get("score", 0.0)),
                "file_path":   meta.get("file_path"),
                "symbol_name": meta.get("symbol_name"),
                "chunk_type":  meta.get("chunk_type"),
                "start_line":  meta.get("start_line"),
                "end_line":    meta.get("end_line"),
                "chunk_text":  meta.get("chunk_text"),
                "intent_meta": {},
            })

        return results

    async def delete_by_file(self, file_path: str) -> int:
        """Delete all vectors for a given file path in the namespace."""
        self._ensure_client()

        try:
            # Pinecone supports delete by filter
            self._index.delete(
                filter={"file_path": {"$eq": file_path}},
                namespace=self.namespace,
            )
            logger.info("pinecone_deleted", file=file_path, namespace=self.namespace)
            return 1  # Pinecone doesn't return count for filter deletes
        except Exception as e:
            logger.error("pinecone_delete_failed", file=file_path, error=str(e))
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
