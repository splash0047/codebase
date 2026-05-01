"""
Prometheus Metrics — counters, histograms, and gauges for full observability.

Exposed at /metrics via the FastAPI route in main.py.

Metric categories:
  • Query pipeline — latency histograms, intent counters, confidence distribution
  • Ingestion pipeline — files processed, embedding batches, errors
  • Infrastructure — cache hit/miss, vector store operations, graph queries
"""
from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ── Application Info ──────────────────────────────────────────────────────────

APP_INFO = Info("ckai", "Codebase Knowledge AI service info")
APP_INFO.info({
    "version": "0.1.0",
    "component": "backend",
})

# ── Query Pipeline ────────────────────────────────────────────────────────────

QUERY_TOTAL = Counter(
    "ckai_query_total",
    "Total queries processed",
    ["intent", "zero_llm", "cached"],
)

QUERY_LATENCY = Histogram(
    "ckai_query_latency_seconds",
    "Query pipeline end-to-end latency",
    ["intent"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

QUERY_CONFIDENCE = Histogram(
    "ckai_query_confidence",
    "Top-result confidence distribution",
    ["intent"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PIPELINE_STAGE_LATENCY = Histogram(
    "ckai_pipeline_stage_seconds",
    "Latency of individual pipeline stages",
    ["stage"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

LLM_SKIPPED_TOTAL = Counter(
    "ckai_llm_skipped_total",
    "Times LLM was skipped (timeout or zero-LLM mode)",
    ["reason"],
)

FEEDBACK_TOTAL = Counter(
    "ckai_feedback_total",
    "User feedback submissions",
    ["helpful"],
)

# ── Ingestion Pipeline ────────────────────────────────────────────────────────

INGESTION_FILES_TOTAL = Counter(
    "ckai_ingestion_files_total",
    "Files processed during ingestion",
    ["status"],
)

INGESTION_LATENCY = Histogram(
    "ckai_ingestion_latency_seconds",
    "Per-file ingestion latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

EMBEDDING_BATCH_TOTAL = Counter(
    "ckai_embedding_batch_total",
    "Embedding API batches processed",
    ["provider"],
)

EMBEDDING_ERRORS = Counter(
    "ckai_embedding_errors_total",
    "Embedding API errors",
    ["error_type"],
)

# ── Infrastructure ────────────────────────────────────────────────────────────

CACHE_OPS = Counter(
    "ckai_cache_ops_total",
    "Cache operations (hit/miss/set)",
    ["operation"],
)

VECTOR_OPS = Counter(
    "ckai_vector_ops_total",
    "Vector store operations",
    ["operation", "backend"],
)

VECTOR_SEARCH_LATENCY = Histogram(
    "ckai_vector_search_seconds",
    "Vector search latency",
    ["backend"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

GRAPH_QUERY_LATENCY = Histogram(
    "ckai_graph_query_seconds",
    "Neo4j graph query latency",
    ["query_type"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

ACTIVE_SESSIONS = Gauge(
    "ckai_active_sessions",
    "Currently active query sessions",
)

INDEXED_REPOS = Gauge(
    "ckai_indexed_repos",
    "Number of indexed repositories",
)

# ── Helper functions ──────────────────────────────────────────────────────────

def record_query_metrics(
    intent: str,
    confidence: float,
    zero_llm: bool,
    cached: bool,
    total_seconds: float,
    pipeline_ms: dict[str, float] | None = None,
) -> None:
    """Record all query-related metrics in one call."""
    QUERY_TOTAL.labels(
        intent=intent,
        zero_llm=str(zero_llm).lower(),
        cached=str(cached).lower(),
    ).inc()

    QUERY_LATENCY.labels(intent=intent).observe(total_seconds)
    QUERY_CONFIDENCE.labels(intent=intent).observe(confidence)

    if pipeline_ms:
        for stage, ms in pipeline_ms.items():
            PIPELINE_STAGE_LATENCY.labels(stage=stage).observe(ms / 1000.0)

    if zero_llm:
        LLM_SKIPPED_TOTAL.labels(reason="zero_llm").inc()


def get_metrics_text() -> tuple[bytes, str]:
    """Generate Prometheus text exposition."""
    return generate_latest(), CONTENT_TYPE_LATEST
