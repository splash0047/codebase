"""
Neo4j async driver wrapper.
Enforces hard limits on traversal depth and node counts from the plan.
"""
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

_driver: AsyncDriver | None = None


async def init_neo4j() -> None:
    global _driver
    _driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_pool_size=50,
        connection_timeout=10.0,
    )
    # Verify connectivity
    await _driver.verify_connectivity()
    logger.info("neo4j_connected", uri=settings.neo4j_uri)


async def close_neo4j() -> None:
    global _driver
    if _driver:
        await _driver.close()
        logger.info("neo4j_connection_closed")


def get_driver() -> AsyncDriver:
    if _driver is None:
        raise RuntimeError("Neo4j driver not initialised. Call init_neo4j() first.")
    return _driver


@asynccontextmanager
async def get_neo4j_session() -> AsyncGenerator[AsyncSession, None]:
    driver = get_driver()
    async with driver.session(database="neo4j") as session:
        yield session


async def run_schema_setup() -> None:
    """
    Create uniqueness constraints and indexes on first boot.
    Idempotent — safe to call multiple times.
    """
    constraints = [
        # Repository
        "CREATE CONSTRAINT repo_id IF NOT EXISTS FOR (r:Repository) REQUIRE r.id IS UNIQUE",
        # File
        "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
        # Function
        "CREATE CONSTRAINT func_id IF NOT EXISTS FOR (fn:Function) REQUIRE fn.id IS UNIQUE",
        # Class
        "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
        # Module
        "CREATE CONSTRAINT module_id IF NOT EXISTS FOR (m:Module) REQUIRE m.id IS UNIQUE",
        # Variable (only exported/critical ones)
        "CREATE CONSTRAINT var_id IF NOT EXISTS FOR (v:Variable) REQUIRE v.id IS UNIQUE",
    ]
    indexes = [
        # Full-text search on function names and docstrings
        "CREATE TEXT INDEX func_name_idx IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
        "CREATE TEXT INDEX class_name_idx IF NOT EXISTS FOR (c:Class) ON (c.name)",
        # importance_score index for fast pruning
        "CREATE INDEX func_importance_idx IF NOT EXISTS FOR (fn:Function) ON (fn.importance_score)",
        # Ingestion status
        "CREATE INDEX file_status_idx IF NOT EXISTS FOR (f:File) ON (f.ingestion_status)",
    ]
    async with get_neo4j_session() as session:
        for stmt in constraints + indexes:
            await session.run(stmt)
    logger.info("neo4j_schema_ready")


def build_traversal_query(
    start_node_id: str,
    depth: int | None = None,
    node_limit: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build a Cypher traversal query with hard limits enforced from config.
    depth and node_limit are clamped to the plan's absolute ceilings.
    """
    safe_depth = min(
        depth if depth is not None else settings.max_traversal_depth,
        settings.max_traversal_depth,
    )
    safe_limit = min(
        node_limit if node_limit is not None else settings.max_nodes_absolute,
        settings.max_nodes_absolute,
    )
    query = (
        "MATCH path = (start {id: $start_id})-[*1..$depth]->(neighbor) "
        "RETURN nodes(path) AS nodes, relationships(path) AS rels "
        "LIMIT $limit"
    )
    params = {"start_id": start_node_id, "depth": safe_depth, "limit": safe_limit}
    return query, params
