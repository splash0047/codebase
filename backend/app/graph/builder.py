"""
Neo4j Graph Builder (Phase 1 – Step 5).

Populates the graph with nodes and relationships from ParsedFile data.
Implements:
 - Deterministic importance_score formula from the plan
 - Time-decayed usage_frequency scoring
 - Low-importance node visibility flag (never deleted)
 - Idempotent MERGE operations
 - Failure Recovery via retry with max_retries cap
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.neo4j_client import get_neo4j_session
from app.parsing.ast_parser import ParsedFile, ParsedFunction, ParsedClass

settings = get_settings()
logger = get_logger(__name__)


# ── Importance Score Formula ──────────────────────────────────────────────────
# importance = w_degree*(in+out) + w_central*centrality + w_exposure*is_exported
# + w_usage*(usage_freq * time_decay)
# Weights from the plan's deterministic formula
W_DEGREE   = 0.35
W_CENTRAL  = 0.25
W_EXPOSURE = 0.25
W_USAGE    = 0.15   # capped at MEMORY_BOOST_MAX


def compute_importance(
    in_degree: int,
    out_degree: int,
    centrality: float,
    is_exported: bool,
    usage_freq: int = 0,
    last_used_days_ago: float = 0.0,
) -> float:
    """Deterministic node importance formula (plan Step 5)."""
    degree_score   = min((in_degree + out_degree) / 20.0, 1.0)
    exposure_score = 1.0 if is_exported else 0.0
    half_life      = settings.usage_half_life_days
    decay          = math.exp(-0.693 * last_used_days_ago / half_life) if half_life > 0 else 1.0
    usage_score    = min(usage_freq / 100.0, 1.0) * decay
    # Cap memory/usage boost at plan's ceiling
    capped_usage = min(usage_score * W_USAGE, settings.memory_boost_max_pct)

    raw = (
        W_DEGREE   * degree_score
        + W_CENTRAL  * min(centrality, 1.0)
        + W_EXPOSURE * exposure_score
        + capped_usage
    )
    return round(min(raw, 1.0), 4)


def is_low_importance(importance: float) -> bool:
    """
    Mark nodes as low_importance for UI visibility toggling.
    We NEVER delete them — structure is always preserved in the graph.
    """
    return importance < 0.25


# ── Graph Builder ─────────────────────────────────────────────────────────────

class GraphBuilder:
    def __init__(self, repo_id: str, commit_id: str | None = None):
        self.repo_id  = repo_id
        self.commit_id = commit_id or "unknown"

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def upsert_file_node(self, file_id: str, path: str, language: str) -> None:
        async with get_neo4j_session() as session:
            await session.run(
                """
                MERGE (f:File {id: $id})
                SET  f.path       = $path,
                     f.language   = $language,
                     f.repo_id    = $repo_id,
                     f.commit_id  = $commit_id,
                     f.updated_at = $now
                """,
                id=file_id, path=path, language=language,
                repo_id=self.repo_id, commit_id=self.commit_id,
                now=datetime.now(timezone.utc).isoformat(),
            )

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def upsert_function_node(
        self,
        func: ParsedFunction,
        file_id: str,
        in_degree: int = 0,
        out_degree: int = 0,
    ) -> str:
        """Returns the Neo4j node id for further relationship creation."""
        node_id = f"{file_id}::{func.qualified_name}"
        importance = compute_importance(
            in_degree=in_degree,
            out_degree=out_degree + len(func.calls),
            centrality=0.0,   # Updated in background recalculation job
            is_exported=func.is_exported,
        )
        low_imp = is_low_importance(importance)

        async with get_neo4j_session() as session:
            await session.run(
                """
                MERGE (fn:Function {id: $id})
                SET  fn.name            = $name,
                     fn.qualified_name  = $qname,
                     fn.file_id         = $file_id,
                     fn.repo_id         = $repo_id,
                     fn.start_line      = $start_line,
                     fn.end_line        = $end_line,
                     fn.is_public       = $is_public,
                     fn.is_exported     = $is_exported,
                     fn.importance_score= $importance,
                     fn.low_importance  = $low_imp,
                     fn.commit_id       = $commit_id,
                     fn.updated_at      = $now
                """,
                id=node_id, name=func.name, qname=func.qualified_name,
                file_id=file_id, repo_id=self.repo_id,
                start_line=func.start_line, end_line=func.end_line,
                is_public=func.is_public, is_exported=func.is_exported,
                importance=importance, low_imp=low_imp,
                commit_id=self.commit_id,
                now=datetime.now(timezone.utc).isoformat(),
            )
            # DEFINED_IN edge
            await session.run(
                """
                MATCH (fn:Function {id: $fn_id}), (f:File {id: $file_id})
                MERGE (fn)-[:DEFINED_IN]->(f)
                """,
                fn_id=node_id, file_id=file_id,
            )
        return node_id

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def upsert_class_node(self, cls: ParsedClass, file_id: str) -> str:
        node_id = f"{file_id}::class::{cls.name}"
        importance = compute_importance(
            in_degree=0, out_degree=len(cls.methods),
            centrality=0.0, is_exported=cls.is_public,
        )
        async with get_neo4j_session() as session:
            await session.run(
                """
                MERGE (c:Class {id: $id})
                SET  c.name             = $name,
                     c.file_id          = $file_id,
                     c.repo_id          = $repo_id,
                     c.base_classes     = $bases,
                     c.importance_score = $importance,
                     c.low_importance   = $low_imp,
                     c.commit_id        = $commit_id
                """,
                id=node_id, name=cls.name, file_id=file_id,
                repo_id=self.repo_id,
                bases=cls.base_classes,
                importance=importance,
                low_imp=is_low_importance(importance),
                commit_id=self.commit_id,
            )
            await session.run(
                """
                MATCH (c:Class {id: $cls_id}), (f:File {id: $file_id})
                MERGE (c)-[:DEFINED_IN]->(f)
                """,
                cls_id=node_id, file_id=file_id,
            )
            for base in cls.base_classes:
                await session.run(
                    """
                    MERGE (b:Class {name: $base_name, repo_id: $repo_id})
                    ON CREATE SET b.id = $base_id, b.importance_score = 0.1
                    WITH b
                    MATCH (c:Class {id: $cls_id})
                    MERGE (c)-[:EXTENDS]->(b)
                    """,
                    base_name=base, repo_id=self.repo_id,
                    base_id=f"{self.repo_id}::class::{base}",
                    cls_id=node_id,
                )
        return node_id

    async def build_call_edges(
        self, func_node_id: str, called_names: list[str], file_id: str
    ) -> None:
        """Create CALLS edges between functions. Best-effort — no retry needed."""
        async with get_neo4j_session() as session:
            for callee_name in called_names:
                await session.run(
                    """
                    MATCH (caller:Function {id: $caller_id})
                    MERGE (callee:Function {name: $callee_name, repo_id: $repo_id})
                    ON CREATE SET callee.id = $callee_id,
                                  callee.importance_score = 0.1,
                                  callee.low_importance = true
                    MERGE (caller)-[:CALLS]->(callee)
                    """,
                    caller_id=func_node_id,
                    callee_name=callee_name,
                    repo_id=self.repo_id,
                    callee_id=f"{file_id}::fn::{callee_name}",
                )

    async def build_from_parsed_file(
        self, parsed: ParsedFile, file_id: str
    ) -> dict[str, Any]:
        """
        Full graph population for one file.
        Returns summary for ingestion status tracking.
        """
        summary: dict[str, Any] = {
            "file_id": file_id,
            "functions_added": 0,
            "classes_added": 0,
            "errors": [],
        }
        try:
            await self.upsert_file_node(file_id, parsed.path, parsed.language)

            for func in parsed.functions:
                try:
                    fn_id = await self.upsert_function_node(
                        func, file_id, out_degree=len(func.calls)
                    )
                    await self.build_call_edges(fn_id, func.calls, file_id)
                    summary["functions_added"] += 1
                except Exception as e:
                    summary["errors"].append(f"Function {func.name}: {e}")
                    logger.error("graph_function_upsert_failed", func=func.name, error=str(e))

            for cls in parsed.classes:
                try:
                    await self.upsert_class_node(cls, file_id)
                    for method in cls.methods:
                        fn_id = await self.upsert_function_node(method, file_id)
                        await self.build_call_edges(fn_id, method.calls, file_id)
                    summary["classes_added"] += 1
                except Exception as e:
                    summary["errors"].append(f"Class {cls.name}: {e}")
                    logger.error("graph_class_upsert_failed", cls=cls.name, error=str(e))

        except Exception as exc:
            summary["errors"].append(str(exc))
            logger.error("graph_build_failed", file_id=file_id, error=str(exc))

        return summary
