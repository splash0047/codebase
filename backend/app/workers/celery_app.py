"""
Celery application with three priority queues:
  high_priority  → webhook-triggered re-indexing, user-initiated
  default        → new repository ingestion
  low_priority   → background re-embedding, stats recalculation

Backpressure: task_annotations rate-limit low_priority to prevent I/O flooding.
"""
from celery import Celery
from kombu import Exchange, Queue

from app.core.config import get_settings

settings = get_settings()

# ── App ───────────────────────────────────────────────────────────────────────
celery_app = Celery("codebase_knowledge_ai")

celery_app.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,

    # ── Priority Queues ───────────────────────────────────────────────────────
    task_queues=(
        Queue("high_priority", Exchange("high_priority"), routing_key="high_priority"),
        Queue("default",       Exchange("default"),       routing_key="default"),
        Queue("low_priority",  Exchange("low_priority"),  routing_key="low_priority"),
    ),
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",

    # ── Backpressure (rate limit low-priority workers) ────────────────────────
    task_annotations={
        "app.workers.tasks.recompute_embeddings": {"rate_limit": "10/m"},
        "app.workers.tasks.recalculate_stats":    {"rate_limit": "5/m"},
    },

    # ── Reliability ───────────────────────────────────────────────────────────
    task_acks_late=True,                      # ACK only after success
    task_reject_on_worker_lost=True,          # Re-queue on worker crash
    task_max_retries=settings.max_retries,    # Capped at 3 per plan
    task_soft_time_limit=600,                 # 10 min soft limit
    task_time_limit=900,                      # 15 min hard kill

    # ── Result Expiry ─────────────────────────────────────────────────────────
    result_expires=3600,

    # ── Worker Settings ───────────────────────────────────────────────────────
    worker_prefetch_multiplier=1,             # One task at a time per worker slot
    worker_max_tasks_per_child=500,           # Recycle to prevent memory leaks
)

# ── Task routing ──────────────────────────────────────────────────────────────
celery_app.conf.task_routes = {
    "app.workers.tasks.ingest_repository":    {"queue": "default"},
    "app.workers.tasks.reindex_file":         {"queue": "high_priority"},
    "app.workers.tasks.recompute_embeddings": {"queue": "low_priority"},
    "app.workers.tasks.recalculate_stats":    {"queue": "low_priority"},
    "app.workers.tasks.process_webhook":      {"queue": "high_priority"},
}

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.workers"])
