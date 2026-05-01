"""
Redis client with LRU eviction and per-repo cache budget enforcement.
All cache budget logic lives here so no other module needs to know about it.
"""
import json
from typing import Any

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

_redis: aioredis.Redis | None = None

# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def init_redis() -> None:
    global _redis
    _redis = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
    )
    await _redis.ping()
    logger.info("redis_connected", url=settings.redis_url)


async def close_redis() -> None:
    if _redis:
        await _redis.aclose()
        logger.info("redis_connection_closed")


def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis client not initialised. Call init_redis() first.")
    return _redis


# ── Cache Budget Keys ─────────────────────────────────────────────────────────
def _repo_budget_key(repo_id: str) -> str:
    return f"budget:repo:{repo_id}:bytes"

def _repo_lru_key(repo_id: str) -> str:
    return f"lru:repo:{repo_id}"


# ── Query Result Cache ────────────────────────────────────────────────────────

async def get_cached_result(cache_key: str) -> dict | None:
    """Retrieve a cached query result. Returns None on miss or error."""
    r = get_redis()
    try:
        raw = await r.get(f"qcache:{cache_key}")
        if raw:
            logger.debug("cache_hit", key=cache_key)
            return json.loads(raw)
    except Exception as exc:
        logger.error("cache_get_failed", key=cache_key, error=str(exc))
    return None


async def set_cached_result(
    cache_key: str,
    result: dict[str, Any],
    ttl_seconds: int = 3600,
) -> None:
    """Store a query result in Redis with TTL."""
    r = get_redis()
    try:
        await r.setex(f"qcache:{cache_key}", ttl_seconds, json.dumps(result))
        logger.debug("cache_set", key=cache_key, ttl=ttl_seconds)
    except Exception as exc:
        logger.error("cache_set_failed", key=cache_key, error=str(exc))


async def invalidate_repo_cache(repo_id: str) -> None:
    """Invalidate all query cache entries for a repository."""
    r = get_redis()
    pattern = f"qcache:repo:{repo_id}:*"
    try:
        async for key in r.scan_iter(pattern):
            await r.delete(key)
        logger.info("cache_invalidated", repo_id=repo_id)
    except Exception as exc:
        logger.error("cache_invalidation_failed", repo_id=repo_id, error=str(exc))


# ── Hot Cache Preload (with budget) ──────────────────────────────────────────

BUDGET_BYTES = settings.cache_budget_per_repo_mb * 1024 * 1024


async def preload_entry_point(
    repo_id: str,
    node_id: str,
    data: dict[str, Any],
    ttl_seconds: int = 7200,
) -> bool:
    """
    Load a critical entry-point node into Redis only if within budget.
    Returns True if stored, False if budget exceeded (LRU will handle eviction).
    Redis itself is configured with allkeys-lru so overflow is safe.
    """
    r = get_redis()
    payload = json.dumps(data)
    payload_bytes = len(payload.encode())

    budget_key = _repo_budget_key(repo_id)
    current = int(await r.get(budget_key) or 0)

    if current + payload_bytes > BUDGET_BYTES:
        logger.warning(
            "hot_cache_budget_exceeded",
            repo_id=repo_id,
            current_bytes=current,
            payload_bytes=payload_bytes,
            budget_bytes=BUDGET_BYTES,
        )
        return False

    cache_key = f"hot:{repo_id}:{node_id}"
    async with r.pipeline() as pipe:
        pipe.setex(cache_key, ttl_seconds, payload)
        pipe.incrby(budget_key, payload_bytes)
        pipe.expire(budget_key, ttl_seconds)
        await pipe.execute()

    logger.debug("hot_cache_stored", repo_id=repo_id, node_id=node_id)
    return True


async def get_hot_node(repo_id: str, node_id: str) -> dict | None:
    r = get_redis()
    raw = await r.get(f"hot:{repo_id}:{node_id}")
    return json.loads(raw) if raw else None


# ── Query Frequency Tracking ──────────────────────────────────────────────────

async def increment_query_frequency(fingerprint: str) -> int:
    """Track how many times a query fingerprint has been seen."""
    r = get_redis()
    key = f"qfreq:{fingerprint}"
    count = await r.incr(key)
    await r.expire(key, 86400 * 30)   # 30-day window
    return count


async def get_query_frequency(fingerprint: str) -> int:
    r = get_redis()
    val = await r.get(f"qfreq:{fingerprint}")
    return int(val) if val else 0
