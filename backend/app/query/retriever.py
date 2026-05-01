"""
Hybrid Retrieval Engine (Phase 3 – Step 10).

Implements the full retrieval pipeline:
  Vector Search → Symbol Filter → Graph Expansion → Fallback Hierarchy → Deduplication

Plan decisions:
  - Fallback hierarchy: exact symbol match → graph neighbours → semantic matches
  - Context-preserving node deduplication (keeps source[] metadata)
  - Soft result diversity control (max_per_file is soft, not hard)
  - Noise detection: auto-prune if low-confidence node ratio > threshold
  - Strict pipeline timeout budgets (100ms vector / 300ms graph)
  - Hard ceiling: max_nodes_absolute = 50
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.neo4j_client import get_neo4j_session, build_traversal_query
from app.db.redis_client import get_cached_result, set_cached_result
from app.db.vector_store import get_vector_store
from app.ingestion.embedder import _embed_batch_with_fallback
from app.query.classifier import ClassificationResult, QueryIntent

settings = get_settings()
logger   = get_logger(__name__)

# Noise threshold: if >40% of retrieved nodes score below this, prune the batch
NOISE_SCORE_THRESHOLD = 0.3
NOISE_RATIO_MAX       = 0.4
SOFT_MAX_PER_FILE     = 3   # Soft diversity limit — relaxed if best result is here


@dataclass
class RetrievedNode:
    node_id:     str
    node_type:   str                    # "vector", "graph", "symbol"
    score:       float
    file_path:   str | None = None
    symbol_name: str | None = None
    chunk_type:  str | None = None
    start_line:  int | None = None
    end_line:    int | None = None
    chunk_text:  str | None = None
    intent_meta: dict = field(default_factory=dict)
    # Context preservation: which retrievers surfaced this node
    sources:     list[str] = field(default_factory=list)
    importance:  float = 0.0


# ── Vector Retrieval (100ms budget) ───────────────────────────────────────────

async def _vector_search(
    repo_id: str,
    query_vector: list[float],
    top_k: int,
) -> list[RetrievedNode]:
    """Timed vector search — returns empty list on timeout."""
    t0 = time.monotonic()
    try:
        store   = get_vector_store(repo_id)
        results = await asyncio.wait_for(
            store.search(query_vector, top_k),
            timeout=settings.timeout_vector_ms / 1000,
        )
        elapsed = (time.monotonic() - t0) * 1000
        logger.debug("vector_search_done", ms=round(elapsed, 1), count=len(results))

        return [
            RetrievedNode(
                node_id=r["vector_id"],
                node_type="vector",
                score=r["score"],
                file_path=r.get("file_path"),
                symbol_name=r.get("symbol_name"),
                chunk_type=r.get("chunk_type"),
                start_line=r.get("start_line"),
                end_line=r.get("end_line"),
                chunk_text=r.get("chunk_text"),
                intent_meta=r.get("intent_meta", {}),
                sources=["vector"],
            )
            for r in results
        ]
    except asyncio.TimeoutError:
        logger.warning("vector_search_timeout", budget_ms=settings.timeout_vector_ms)
        return []
    except Exception as exc:
        logger.error("vector_search_failed", error=str(exc))
        return []


# ── Graph Traversal (300ms budget) ────────────────────────────────────────────

async def _graph_search(
    repo_id: str,
    symbol_name: str | None,
    depth: int | None = None,
    node_limit: int | None = None,
) -> list[RetrievedNode]:
    """Timed graph traversal — returns empty list on timeout or Neo4j down."""
    if not symbol_name:
        return []

    t0 = time.monotonic()
    try:
        async with get_neo4j_session() as session:
            # First: exact symbol lookup
            result = await asyncio.wait_for(
                session.run(
                    """
                    MATCH (n {name: $name, repo_id: $repo_id})
                    RETURN n.id AS id, n.name AS name, labels(n)[0] AS type,
                           n.importance_score AS importance, n.low_importance AS low_imp
                    LIMIT 1
                    """,
                    name=symbol_name, repo_id=repo_id,
                ),
                timeout=0.05,
            )
            start_record = await result.single()

        if not start_record:
            return []

        start_id   = start_record["id"]
        query, params = build_traversal_query(start_id, depth, node_limit)

        async with get_neo4j_session() as session:
            trav_result = await asyncio.wait_for(
                session.run(query, **params),
                timeout=(settings.timeout_graph_ms - 50) / 1000,
            )
            records = await trav_result.data()

        elapsed = (time.monotonic() - t0) * 1000
        logger.debug("graph_search_done", ms=round(elapsed, 1), nodes=len(records))

        nodes: list[RetrievedNode] = []
        for rec in records:
            for node in rec.get("nodes", []):
                imp = node.get("importance_score", 0.0) or 0.0
                nodes.append(RetrievedNode(
                    node_id=node.get("id", ""),
                    node_type="graph",
                    score=imp,
                    symbol_name=node.get("name"),
                    file_path=node.get("file_id"),
                    importance=imp,
                    sources=["graph"],
                ))
        return nodes

    except asyncio.TimeoutError:
        logger.warning("graph_search_timeout", budget_ms=settings.timeout_graph_ms)
        return []
    except Exception as exc:
        logger.error("graph_search_failed", error=str(exc))
        return []  # Fallback to vector-only on Neo4j downtime


# ── Exact Symbol Search ───────────────────────────────────────────────────────

async def _symbol_search(repo_id: str, query: str) -> list[RetrievedNode]:
    """Fast exact symbol match via Neo4j full-text index."""
    try:
        async with get_neo4j_session() as session:
            result = await asyncio.wait_for(
                session.run(
                    """
                    CALL db.index.fulltext.queryNodes('func_name_idx', $q)
                    YIELD node, score
                    WHERE node.repo_id = $repo_id
                    RETURN node.id AS id, node.name AS name, node.importance_score AS importance
                    LIMIT 5
                    """,
                    q=query, repo_id=repo_id,
                ),
                timeout=0.1,
            )
            records = await result.data()
        return [
            RetrievedNode(
                node_id=r["id"],
                node_type="symbol",
                score=r.get("importance", 0.5),
                symbol_name=r["name"],
                importance=r.get("importance", 0.5),
                sources=["symbol"],
            )
            for r in records
        ]
    except Exception as exc:
        logger.error("symbol_search_failed", error=str(exc))
        return []


# ── Fallback Hierarchy ────────────────────────────────────────────────────────

async def _fallback_hierarchy(
    repo_id: str,
    query: str,
    query_vector: list[float] | None,
    top_k: int,
) -> list[RetrievedNode]:
    """
    Plan fallback order:
      1. Exact symbol match
      2. Graph neighbours
      3. Semantic vector matches
    """
    # 1. Symbol match
    results = await _symbol_search(repo_id, query)
    if results:
        logger.info("fallback_resolved_symbol", count=len(results))
        return results

    # 2. Graph neighbours (breadth=1 for semantic fallback)
    symbol_hint = query.split()[-1] if query else None
    results = await _graph_search(repo_id, symbol_hint, depth=1, node_limit=10)
    if results:
        logger.info("fallback_resolved_graph", count=len(results))
        return results

    # 3. Semantic fallback
    if query_vector:
        results = await _vector_search(repo_id, query_vector, top_k=min(top_k, 5))
        if results:
            logger.info("fallback_resolved_vector", count=len(results))
            return results

    logger.warning("fallback_exhausted", query=query[:60])
    return []


# ── Deduplication (context-preserving) ────────────────────────────────────────

def _deduplicate(nodes: list[RetrievedNode]) -> list[RetrievedNode]:
    """
    Merge duplicate node_ids but preserve all source[] metadata.
    Never loses path context.
    """
    seen: dict[str, RetrievedNode] = {}
    for node in nodes:
        key = node.node_id or f"{node.file_path}::{node.symbol_name}"
        if key in seen:
            # Merge: keep highest score, union sources
            existing = seen[key]
            existing.score   = max(existing.score, node.score)
            existing.sources = list(set(existing.sources + node.sources))
        else:
            seen[key] = node
    return list(seen.values())


# ── Noise Detection ───────────────────────────────────────────────────────────

def _prune_noise(nodes: list[RetrievedNode]) -> list[RetrievedNode]:
    """Auto-prune if low-confidence node ratio > NOISE_RATIO_MAX."""
    if not nodes:
        return nodes
    low_conf = [n for n in nodes if n.score < NOISE_SCORE_THRESHOLD]
    ratio    = len(low_conf) / len(nodes)
    if ratio > NOISE_RATIO_MAX:
        pruned = [n for n in nodes if n.score >= NOISE_SCORE_THRESHOLD]
        logger.info("noise_pruned", removed=len(low_conf), kept=len(pruned), ratio=round(ratio, 2))
        return pruned
    return nodes


# ── Soft Diversity Control ────────────────────────────────────────────────────

def _apply_diversity(nodes: list[RetrievedNode], best_score: float) -> list[RetrievedNode]:
    """
    Soft per-file diversity: max SOFT_MAX_PER_FILE results per file.
    Exception: if the best result lives in a file that would be capped, keep it.
    """
    file_counts: dict[str, int] = {}
    kept: list[RetrievedNode]   = []
    for node in nodes:
        fp = node.file_path or "__unknown__"
        count = file_counts.get(fp, 0)
        # Always keep if it's the top scorer from any file
        if count < SOFT_MAX_PER_FILE or node.score >= best_score * 0.95:
            kept.append(node)
            file_counts[fp] = count + 1
    return kept


# ── Main Retriever ────────────────────────────────────────────────────────────

class HybridRetriever:
    async def retrieve(
        self,
        repo_id: str,
        query: str,
        classification: ClassificationResult,
        top_k: int | None = None,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedNode]:
        """
        Full hybrid retrieval with fallback, deduplication, noise pruning,
        and soft diversity control.
        """
        effective_k = min(
            top_k if top_k else settings.max_top_k,
            settings.max_top_k,
        )
        intent    = classification.intent
        use_graph = intent in (QueryIntent.DEPENDENCY, QueryIntent.DEFINITION, QueryIntent.EXPLANATION)
        use_vec   = intent in (QueryIntent.SEMANTIC, QueryIntent.EXPLANATION) or classification.use_hybrid

        nodes: list[RetrievedNode] = []

        # ── Vector retrieval
        if use_vec and query_vector:
            vec_nodes = await _vector_search(repo_id, query_vector, effective_k)
            nodes.extend(vec_nodes)

        # ── Graph retrieval (extract likely symbol name from query)
        if use_graph:
            symbol_hint = self._extract_symbol(query)
            # Multi-hop reasoning for complex intents (Phase 6)
            graph_depth = 5 if intent in (QueryIntent.EXPLANATION, QueryIntent.DEPENDENCY) else 2
            graph_nodes = await _graph_search(repo_id, symbol_hint, depth=graph_depth)
            nodes.extend(graph_nodes)

        # ── Fallback if still empty
        if not nodes:
            nodes = await _fallback_hierarchy(repo_id, query, query_vector, effective_k)

        # ── Post-processing pipeline
        nodes = _deduplicate(nodes)
        nodes = _prune_noise(nodes)
        nodes.sort(key=lambda n: n.score, reverse=True)
        nodes = nodes[:settings.max_nodes_absolute]   # Hard ceiling

        if nodes:
            nodes = _apply_diversity(nodes, nodes[0].score)

        logger.info("retrieval_complete", count=len(nodes), intent=intent, repo_id=repo_id)
        return nodes

    def _extract_symbol(self, query: str) -> str | None:
        """Best-effort symbol extraction from query text."""
        # Look for backtick-wrapped or camelCase/snake_case tokens
        bt = re.findall(r"`(\w+)`", query)
        if bt:
            return bt[0]
        words = query.split()
        for w in reversed(words):
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", w) and len(w) > 2:
                return w
        return None


import re  # noqa: E402 (needed by _extract_symbol)
