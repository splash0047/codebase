"""
FastAPI application entry point.
Initialises all DB connections on startup and tears them down on shutdown.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.telemetry import init_telemetry, shutdown_telemetry
from app.core.metrics import get_metrics_text
from app.db.neo4j_client import close_neo4j, init_neo4j, run_schema_setup
from app.db.postgres import create_tables, dispose_engine
from app.db.redis_client import close_redis, init_redis
from app.api.routes import ingest, query as query_router, analysis

settings = get_settings()
setup_logging()
logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown sequence."""
    logger.info("app_starting", env=settings.app_env)

    # Initialise telemetry
    init_telemetry()

    # Initialise all database connections
    await init_redis()
    await init_neo4j()
    await run_schema_setup()

    if settings.app_env == "development":
        await create_tables()   # Auto-create tables in dev (use Alembic in prod)

    # Auto-instrument FastAPI with OpenTelemetry
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("otel_fastapi_instrumented")
    except ImportError:
        logger.warning("otel_fastapi_instrumentation_unavailable")

    logger.info("app_ready")
    yield

    # Graceful shutdown
    logger.info("app_shutting_down")
    shutdown_telemetry()
    await close_redis()
    await close_neo4j()
    await dispose_engine()
    logger.info("app_stopped")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Codebase Knowledge AI",
    description=(
        "AI-powered developer intelligence system that converts codebases "
        "into queryable knowledge graphs with semantic search."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(ingest.router, prefix="/api/v1")
app.include_router(query_router.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe."""
    return JSONResponse({"status": "ok", "version": "0.2.0"})


@app.get("/metrics", tags=["Observability"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    body, content_type = get_metrics_text()
    return Response(content=body, media_type=content_type)


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness probe — checks all downstream dependencies.
    Returns 503 if any service is unhealthy.
    """
    from app.db.redis_client import get_redis
    from app.db.neo4j_client import get_driver
    checks: dict[str, str] = {}
    healthy = True

    try:
        await get_redis().ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        healthy = False

    try:
        await get_driver().verify_connectivity()
        checks["neo4j"] = "ok"
    except Exception as e:
        checks["neo4j"] = f"error: {e}"
        healthy = False

    status_code = 200 if healthy else 503
    return JSONResponse({"status": "ready" if healthy else "degraded", "checks": checks}, status_code=status_code)
