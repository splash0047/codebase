"""
Ingestion API routes — repository registration and webhook handler.
"""
from __future__ import annotations

import hmac
import hashlib
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status
from pydantic import BaseModel, HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.postgres import get_db
from app.models.db_models import IngestionStatus, Repository, RepoProvider
from app.workers.tasks import ingest_repository, process_webhook

settings = get_settings()
logger   = get_logger(__name__)
router   = APIRouter(prefix="/ingest", tags=["Ingestion"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class FilePayload(BaseModel):
    path: str
    content: str


class IngestRequest(BaseModel):
    full_name: str                      # e.g. "org/repo-name"
    provider: RepoProvider = RepoProvider.LOCAL
    clone_url: str | None = None
    default_branch: str = "main"
    commit_id: str | None = None
    files: list[FilePayload]


class IngestResponse(BaseModel):
    repo_id: str
    status: IngestionStatus
    file_count: int
    task_id: str
    coverage_pct: float


class WebhookResponse(BaseModel):
    dispatched: int


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/repository",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a repository for ingestion",
)
async def submit_repository(
    payload: IngestRequest,
):
    """
    Register a repository and dispatch a background ingestion task.
    Returns immediately with 202 Accepted; coverage updates are tracked async.
    """
    async with get_db() as db:
        # Check if repo already exists
        from sqlalchemy import select
        existing = await db.execute(
            select(Repository).where(Repository.full_name == payload.full_name)
        )
        repo = existing.scalar_one_or_none()

        if not repo:
            repo = Repository(
                name=payload.full_name.split("/")[-1],
                full_name=payload.full_name,
                provider=payload.provider,
                clone_url=payload.clone_url,
                default_branch=payload.default_branch,
                latest_commit_id=payload.commit_id,
                ingestion_status=IngestionStatus.PENDING,
                total_files=len(payload.files),
            )
            db.add(repo)
            await db.flush()
        else:
            repo.ingestion_status = IngestionStatus.PENDING
            repo.total_files = len(payload.files)
            repo.latest_commit_id = payload.commit_id
            await db.flush()

        repo_id = str(repo.id)

    # Dispatch Celery task (default queue)
    task = ingest_repository.apply_async(
        args=[
            repo_id,
            [f.model_dump() for f in payload.files],
            payload.commit_id,
        ],
        queue="default",
    )

    logger.info(
        "ingestion_dispatched",
        repo_id=repo_id,
        file_count=len(payload.files),
        task_id=task.id,
    )
    return IngestResponse(
        repo_id=repo_id,
        status=IngestionStatus.PENDING,
        file_count=len(payload.files),
        task_id=task.id,
        coverage_pct=0.0,
    )


@router.get(
    "/repository/{repo_id}/status",
    summary="Get ingestion status and coverage for a repository",
)
async def get_ingestion_status(repo_id: str):
    """
    Returns the current ingestion status and coverage percentage.
    The UI uses this to show 'Coverage: 23% ⚠️ Results may be incomplete'.
    """
    from sqlalchemy import select
    async with get_db() as db:
        result = await db.execute(
            select(Repository).where(Repository.id == uuid.UUID(repo_id))
        )
        repo = result.scalar_one_or_none()
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")

        return {
            "repo_id":          repo_id,
            "status":           repo.ingestion_status,
            "coverage_pct":     repo.coverage_pct,
            "total_files":      repo.total_files,
            "indexed_files":    repo.indexed_files,
            "latest_commit_id": repo.latest_commit_id,
            "warning":          repo.coverage_pct < 100.0,
        }


@router.post(
    "/webhook/github",
    response_model=WebhookResponse,
    status_code=status.HTTP_200_OK,
    summary="GitHub push webhook handler",
)
async def github_webhook(
    payload: dict,
    x_hub_signature_256: Annotated[str | None, Header()] = None,
):
    """
    Receives GitHub push events and dispatches high-priority reindex tasks
    for changed files only.
    """
    # Verify webhook signature
    if settings.github_webhook_secret:
        if not x_hub_signature_256:
            raise HTTPException(status_code=401, detail="Missing signature header")
        import json
        body = json.dumps(payload).encode()
        expected = "sha256=" + hmac.new(
            settings.github_webhook_secret.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    task = process_webhook.apply_async(args=[payload], queue="high_priority")
    logger.info("webhook_received", task_id=task.id)
    return WebhookResponse(dispatched=0)   # actual count returned async
