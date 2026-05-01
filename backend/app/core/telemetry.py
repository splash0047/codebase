"""
OpenTelemetry Setup — distributed tracing across the full query pipeline.

Architecture:
  • SpanKind.SERVER  — API endpoints (auto-instrumented by FastAPI middleware)
  • SpanKind.INTERNAL — pipeline stages (classify, embed, retrieve, rank, llm)
  • SpanKind.CLIENT  — outbound calls (Neo4j, FAISS, OpenAI, Redis)

Exporters:
  • Development: ConsoleSpanExporter (human-readable)
  • Production:  OTLP gRPC to Grafana Tempo / Jaeger

Key metrics attached to spans:
  • query.intent, query.confidence, query.zero_llm
  • retrieve.vector_count, retrieve.graph_count
  • rank.top_score, rank.confidence_gap
  • context.total_tokens, context.truncated
  • pipeline.total_ms per stage
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode, Span

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ── Service resource ──────────────────────────────────────────────────────────

_RESOURCE = Resource.create({
    "service.name": "codebase-knowledge-ai",
    "service.version": "0.1.0",
    "deployment.environment": settings.app_env,
})


def _build_exporter() -> SpanProcessor:
    """Select span exporter based on environment."""
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            logger.info("otel_exporter_otlp", endpoint=otlp_endpoint)
            return BatchSpanProcessor(exporter)
        except ImportError:
            logger.warning("otel_otlp_not_installed_falling_back_to_console")

    # Default: console exporter for development
    return SimpleSpanProcessor(ConsoleSpanExporter())


def init_telemetry() -> None:
    """Initialise the global tracer provider. Call once at app startup."""
    provider = TracerProvider(resource=_RESOURCE)
    provider.add_span_processor(_build_exporter())
    trace.set_tracer_provider(provider)
    logger.info("otel_initialised", env=settings.app_env)


def shutdown_telemetry() -> None:
    """Flush and shut down the tracer provider. Call at app shutdown."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()  # type: ignore[attr-defined]


# ── Tracer singleton ──────────────────────────────────────────────────────────

_tracer: trace.Tracer | None = None


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("codebase-knowledge-ai", "0.1.0")
    return _tracer


# ── Convenience helpers ───────────────────────────────────────────────────────

@contextmanager
def pipeline_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """
    Context manager for instrumenting pipeline stages.

    Usage:
        with pipeline_span("retrieve.vector", {"top_k": 10}) as span:
            results = await vector_store.search(...)
            span.set_attribute("result.count", len(results))
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def record_pipeline_metrics(span: Span, result: dict[str, Any]) -> None:
    """Attach standard pipeline metrics to a span."""
    mappings = {
        "query.intent":           result.get("intent"),
        "query.confidence":       result.get("confidence"),
        "query.zero_llm":         result.get("zero_llm_mode"),
        "query.cached":           result.get("cached"),
        "context.total_tokens":   result.get("context_tokens"),
        "context.truncated":      result.get("coverage_warning"),
        "rank.confidence_gap":    result.get("confidence_gap"),
    }
    for key, value in mappings.items():
        if value is not None:
            span.set_attribute(key, value)

    # Pipeline timing breakdown
    pipeline_ms = result.get("pipeline_ms", {})
    for stage, ms in pipeline_ms.items():
        span.set_attribute(f"pipeline.{stage}_ms", ms)
