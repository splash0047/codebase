"""
Celery tasks for background ingestion (Phase 1 – Steps 1 & 4).
Priority queues: high_priority | default | low_priority
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from celery import Task
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.postgres import AsyncSessionLocal
from app.ingestion.pipeline import IngestionPipeline
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Ingest Repository Task ────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.workers.tasks.ingest_repository",
    queue="default",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def ingest_repository(
    self: Task,
    repo_id: str,
    files: list[dict[str, str]],
    commit_id: str | None = None,
) -> dict[str, Any]:
    """
    Main ingestion task. Dispatched by the API when a repo is submitted.
    files: [{"path": str, "content": str}]
    """
    logger.info("task_ingest_repository_started", repo_id=repo_id, file_count=len(files))
    try:
        async def _run():
            async with AsyncSessionLocal() as db:
                pipeline = IngestionPipeline(db, repo_id, commit_id)
                return await pipeline.ingest_files(files)
        return _run_async(_run())
    except Exception as exc:
        logger.error("task_ingest_repository_failed", repo_id=repo_id, error=str(exc))
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 10)


# ── Re-index Single File (high priority – webhook triggered) ──────────────────

@celery_app.task(
    bind=True,
    name="app.workers.tasks.reindex_file",
    queue="high_priority",
    max_retries=3,
    acks_late=True,
)
def reindex_file(
    self: Task,
    repo_id: str,
    file_path: str,
    content: str,
    commit_id: str | None = None,
) -> dict[str, Any]:
    logger.info("task_reindex_file_started", repo_id=repo_id, path=file_path)
    try:
        async def _run():
            async with AsyncSessionLocal() as db:
                pipeline = IngestionPipeline(db, repo_id, commit_id)
                return await pipeline.ingest_file(file_path, content, priority=True)
        return _run_async(_run())
    except Exception as exc:
        logger.error("task_reindex_file_failed", path=file_path, error=str(exc))
        raise self.retry(exc=exc, countdown=5)


# ── Process GitHub/GitLab Webhook ─────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.workers.tasks.process_webhook",
    queue="high_priority",
    max_retries=3,
    acks_late=True,
)
def process_webhook(self: Task, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Handle push event webhook. Extracts changed files and dispatches reindex tasks.
    """
    repo_id   = payload.get("repo_id")
    commits   = payload.get("commits", [])
    commit_id = payload.get("head_commit", {}).get("id")

    changed_files: list[str] = []
    for commit in commits:
        changed_files.extend(commit.get("added", []))
        changed_files.extend(commit.get("modified", []))

    logger.info(
        "webhook_processing",
        repo_id=repo_id,
        changed_count=len(changed_files),
        commit_id=commit_id,
    )
    # Individual file reindex tasks (high priority)
    for path in changed_files:
        reindex_file.apply_async(
            args=[repo_id, path, "", commit_id],
            queue="high_priority",
        )
    return {"dispatched": len(changed_files)}


# ── Low-Priority Background Tasks ─────────────────────────────────────────────

@celery_app.task(
    name="app.workers.tasks.recompute_embeddings",
    queue="low_priority",
    rate_limit="10/m",
)
def recompute_embeddings(repo_id: str) -> dict[str, Any]:
    """Triggered by scheduler to refresh stale embeddings."""
    logger.info("task_recompute_embeddings", repo_id=repo_id)
    # Phase 2 implementation placeholder
    return {"status": "queued", "repo_id": repo_id}


@celery_app.task(
    name="app.workers.tasks.recalculate_stats",
    queue="low_priority",
    rate_limit="5/m",
)
def recalculate_stats(repo_id: str) -> dict[str, Any]:
    """Recalculates Neo4j centrality scores and updates importance_score."""
    logger.info("task_recalculate_stats", repo_id=repo_id)
    # Phase 5 full implementation; stub for now
    return {"status": "queued", "repo_id": repo_id}
