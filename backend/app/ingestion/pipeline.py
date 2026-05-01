"""
Repository Ingestion Pipeline (Phase 1 – Step 4).

Implements:
 - Cold-start optimization: indexes entry points / exported modules first
 - Eventual consistency: tracks ingestion_status (pending→partial→complete|failed)
 - Idempotent ingestion (content-hash based change detection)
 - Failure Recovery with DLQ logging
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

import xxhash
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.redis_client import preload_entry_point
from app.graph.builder import GraphBuilder
from app.ingestion.chunker import chunk_parsed_file
from app.ingestion.embedding_store import persist_embeddings
from app.models.db_models import (
    Embedding,
    File,
    IngestionStatus,
    Repository,
)
from app.parsing.ast_parser import parse_file

settings = get_settings()
logger = get_logger(__name__)

# ── Entry-point file patterns (Cold Start Optimization) ───────────────────────
ENTRY_POINT_PATTERNS = {
    "main.py", "app.py", "server.py", "index.js", "index.ts",
    "app.js", "app.ts", "main.go", "main.rs", "Main.java",
    "__init__.py",
}
EXPORTED_PATTERNS = {"api", "routes", "controllers", "handlers", "gateway"}


def _is_entry_point(path: str) -> bool:
    name = Path(path).name
    parts = set(Path(path).parts)
    return name in ENTRY_POINT_PATTERNS or bool(parts & EXPORTED_PATTERNS)


def _compute_hash(content: str) -> str:
    return xxhash.xxh64(content.encode()).hexdigest()


# ── Main Ingestion Pipeline ───────────────────────────────────────────────────

class IngestionPipeline:
    def __init__(self, db: AsyncSession, repo_id: str, commit_id: str | None = None):
        self.db        = db
        self.repo_id   = repo_id
        self.commit_id = commit_id
        self.graph     = GraphBuilder(repo_id, commit_id)

    async def _get_repo(self) -> Repository | None:
        result = await self.db.execute(
            select(Repository).where(Repository.id == uuid.UUID(self.repo_id))
        )
        return result.scalar_one_or_none()

    async def _mark_repo(self, status: IngestionStatus, coverage: float | None = None) -> None:
        vals: dict[str, Any] = {"ingestion_status": status}
        if coverage is not None:
            vals["coverage_pct"] = coverage
        await self.db.execute(
            update(Repository)
            .where(Repository.id == uuid.UUID(self.repo_id))
            .values(**vals)
        )
        await self.db.commit()

    async def _should_skip_file(self, path: str, content_hash: str) -> bool:
        """Skip re-ingestion if file hash hasn't changed (idempotent)."""
        result = await self.db.execute(
            select(File).where(
                File.repository_id == uuid.UUID(self.repo_id),
                File.path == path,
                File.content_hash == content_hash,
                File.ingestion_status == IngestionStatus.COMPLETE,
            )
        )
        return result.scalar_one_or_none() is not None

    async def _get_or_create_file(self, path: str, language: str, size: int) -> File:
        result = await self.db.execute(
            select(File).where(
                File.repository_id == uuid.UUID(self.repo_id),
                File.path == path,
            )
        )
        file_obj = result.scalar_one_or_none()
        if not file_obj:
            file_obj = File(
                repository_id=uuid.UUID(self.repo_id),
                path=path,
                language=language,
                size_bytes=size,
                ingestion_status=IngestionStatus.PENDING,
                commit_id=self.commit_id,
            )
            self.db.add(file_obj)
            await self.db.flush()
        return file_obj

    async def ingest_file(
        self, path: str, content: str, priority: bool = False
    ) -> dict[str, Any]:
        """
        Ingest a single file through parse → graph → embedding slots.
        Returns a result dict for status tracking.
        """
        content_hash = _compute_hash(content)
        if await self._should_skip_file(path, content_hash):
            logger.debug("file_skipped_unchanged", path=path)
            return {"path": path, "status": "skipped"}

        parsed = parse_file(path, content)
        language = parsed.language or "unknown"

        file_obj = await self._get_or_create_file(path, language, len(content.encode()))
        file_id  = str(file_obj.id)

        # Mark as partial immediately so the system is eventually consistent
        file_obj.ingestion_status = IngestionStatus.PARTIAL
        file_obj.content_hash     = content_hash
        await self.db.commit()

        # ── Phase 1: Build knowledge graph ──────────────────────────────────
        graph_summary = await self.graph.build_from_parsed_file(parsed, file_id)

        # Store AST summary in Postgres for quick access
        file_obj.ast_summary = {
            "functions": [f.name for f in parsed.functions],
            "classes":   [c.name for c in parsed.classes],
            "imports":   [i.module for i in parsed.imports],
            "quality":   parsed.parse_quality,
        }

        # ── Phase 2: Chunk + embed ───────────────────────────────────────────
        embed_summary: dict[str, Any] = {"saved": 0, "failed": 0}
        try:
            chunks = chunk_parsed_file(parsed, content, commit_id=self.commit_id)
            if chunks:
                embed_summary = await persist_embeddings(
                    self.db, file_obj, chunks, self.repo_id
                )
        except Exception as exc:
            embed_summary["failed"] = 1
            logger.error("embedding_phase_failed", path=path, error=str(exc))

        # Hot cache preload for entry points
        if _is_entry_point(path):
            await preload_entry_point(
                repo_id=self.repo_id,
                node_id=file_id,
                data={"path": path, "ast": file_obj.ast_summary, "language": language},
            )

        # Mark complete only if no critical errors in either phase
        has_errors = bool(graph_summary.get("errors")) or embed_summary.get("failed", 0) > 0
        file_obj.ingestion_status = (
            IngestionStatus.PARTIAL if has_errors else IngestionStatus.COMPLETE
        )
        await self.db.commit()

        logger.info(
            "file_ingested",
            path=path,
            status=file_obj.ingestion_status,
            parse_quality=parsed.parse_quality,
            graph_errors=graph_summary.get("errors"),
            embeddings_saved=embed_summary.get("saved"),
        )
        return {
            "path": path,
            "status": file_obj.ingestion_status,
            "graph": graph_summary,
            "embeddings": embed_summary,
        }

    async def ingest_files(self, files: list[dict[str, str]]) -> dict[str, Any]:
        """
        Ingest a list of files with cold-start prioritization.
        files: [{"path": str, "content": str}]

        Cold-start: entry points are processed first so the system
        answers queries at partial coverage before full indexing completes.
        """
        await self._mark_repo(IngestionStatus.PARTIAL)

        entry_points = [f for f in files if _is_entry_point(f["path"])]
        remaining    = [f for f in files if not _is_entry_point(f["path"])]
        ordered      = entry_points + remaining

        total   = len(ordered)
        done    = 0
        errors  = 0
        results = []

        for file_data in ordered:
            path    = file_data["path"]
            content = file_data.get("content", "")
            try:
                r = await self.ingest_file(path, content, priority=_is_entry_point(path))
                results.append(r)
                if r.get("status") != "skipped":
                    done += 1
            except Exception as exc:
                errors += 1
                logger.error(
                    "file_ingestion_failed",
                    path=path,
                    error=str(exc),
                    repo_id=self.repo_id,
                )
                results.append({"path": path, "status": "failed", "error": str(exc)})

            # Update coverage after every file (shows "Coverage: N% ⚠️" in UI)
            repo = await self._get_repo()
            if repo:
                coverage = round((done / total) * 100, 1) if total > 0 else 0.0
                await self._mark_repo(IngestionStatus.PARTIAL, coverage)

        # Final status
        final_status = (
            IngestionStatus.COMPLETE if errors == 0
            else IngestionStatus.PARTIAL
        )
        await self._mark_repo(final_status, 100.0 if errors == 0 else None)

        logger.info(
            "ingestion_complete",
            repo_id=self.repo_id,
            total=total,
            done=done,
            errors=errors,
            status=final_status,
        )
        return {
            "repo_id": self.repo_id,
            "total":   total,
            "done":    done,
            "errors":  errors,
            "status":  final_status,
            "results": results,
        }
