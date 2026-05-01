"""
Structured logging setup using structlog + standard logging.
All pipeline failures are logged at 100%. Normal traffic uses 10-20% sampling.
"""
import logging
import random
import sys
from typing import Any

import structlog
from app.core.config import get_settings

settings = get_settings()

# ── Sampling filter ──────────────────────────────────────────────────────────
class SamplingFilter(logging.Filter):
    """Pass all ERROR/CRITICAL; sample INFO/DEBUG at SAMPLE_RATE."""
    SAMPLE_RATE = 0.15  # 15% for normal traffic

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True  # 100% of failures logged
        return random.random() < self.SAMPLE_RATE


def setup_logging() -> None:
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure structlog processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.app_env == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(SamplingFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)

    # Silence noisy third-party loggers in production
    if settings.app_env != "development":
        for noisy in ("uvicorn.access", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
