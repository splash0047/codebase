"""
Dynamic Ranking & Re-ranking (Phase 3 – Step 11).

Implements the plan's hybrid scoring formula:
  final_score = 0.5 * embedding_similarity
              + 0.3 * graph_distance_score
              + 0.2 * symbol_match_score
              + capped_memory_boost (≤ 15%)

Weights shift dynamically based on query intent.
Memory boost uses time-decayed usage_frequency (plan Step 5 formula).
Re-ranking retry is non-blocking (expands top-K from existing pool, no re-query).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.query.classifier import ClassificationResult, QueryIntent
from app.query.retriever import RetrievedNode

settings = get_settings()
logger   = get_logger(__name__)

# ── Default intent weights ────────────────────────────────────────────────────
# Weights shift by intent as per the plan's context-aware ranking.
INTENT_WEIGHTS: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.SEMANTIC: {
        "embedding": 0.65, "graph": 0.15, "symbol": 0.20,
    },
    QueryIntent.DEFINITION: {
        "embedding": 0.20, "graph": 0.30, "symbol": 0.50,
    },
    QueryIntent.DEPENDENCY: {
        "embedding": 0.20, "graph": 0.60, "symbol": 0.20,
    },
    QueryIntent.EXPLANATION: {
        "embedding": 0.50, "graph": 0.35, "symbol": 0.15,
    },
    QueryIntent.UNKNOWN: {
        "embedding": 0.50, "graph": 0.30, "symbol": 0.20,
    },
}


@dataclass
class RankedResult:
    node: RetrievedNode
    final_score: float
    embedding_contribution: float
    graph_contribution: float
    symbol_contribution: float
    memory_boost: float
    confidence: float


# ── Time-decayed usage boost ──────────────────────────────────────────────────

def _usage_boost(usage_freq: int, last_used_days: float) -> float:
    """
    Time-decayed memory boost — capped at settings.memory_boost_max_pct.
    Only applied AFTER base ranking is computed (plan requirement).
    half_life = settings.usage_half_life_days
    """
    if usage_freq <= 0:
        return 0.0
    half_life = settings.usage_half_life_days
    decay     = math.exp(-0.693 * last_used_days / half_life) if half_life > 0 else 1.0
    raw_boost = min(usage_freq / 200.0, 1.0) * decay
    return round(min(raw_boost, settings.memory_boost_max_pct), 4)


# ── Confidence threshold auto-adjustment ─────────────────────────────────────

def _auto_adjust_threshold(
    base_threshold: float,
    intent: QueryIntent,
    retrieval_strength: float,
) -> float:
    """
    Shift confidence threshold based on query type and retrieval quality.
    Strong retrieval (high scores) → be more selective.
    Weak retrieval → relax threshold slightly.
    """
    if retrieval_strength > 0.8:
        return min(base_threshold + 0.05, 0.95)
    elif retrieval_strength < 0.4:
        return max(base_threshold - 0.10, 0.20)
    return base_threshold


# ── Ranker ────────────────────────────────────────────────────────────────────

class ResultRanker:
    def rank(
        self,
        nodes: list[RetrievedNode],
        classification: ClassificationResult,
        usage_stats: dict[str, tuple[int, float]] | None = None,
    ) -> list[RankedResult]:
        """
        Score and rank retrieved nodes using intent-weighted hybrid formula.
        usage_stats: {node_id → (freq, days_since_last_use)}
        """
        if not nodes:
            return []

        intent  = classification.intent
        weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS[QueryIntent.UNKNOWN])
        usage_stats = usage_stats or {}

        # Retrieval strength = mean of top-5 raw scores (for threshold adjustment)
        top_scores = sorted([n.score for n in nodes], reverse=True)[:5]
        retrieval_strength = sum(top_scores) / len(top_scores) if top_scores else 0.0

        scored: list[RankedResult] = []
        for node in nodes:
            # ── Base components
            emb_score = node.score if "vector" in node.sources else 0.0
            gph_score = node.importance if "graph" in node.sources else 0.0
            sym_score = node.score if "symbol" in node.sources else 0.0

            base = (
                weights["embedding"] * emb_score
                + weights["graph"]   * gph_score
                + weights["symbol"]  * sym_score
            )

            # ── Memory boost (applied after base, capped at 15%)
            freq, days = usage_stats.get(node.node_id, (0, 0.0))
            boost      = _usage_boost(freq, days)

            final = round(min(base + boost, 1.0), 4)

            scored.append(RankedResult(
                node=node,
                final_score=final,
                embedding_contribution=weights["embedding"] * emb_score,
                graph_contribution=weights["graph"] * gph_score,
                symbol_contribution=weights["symbol"] * sym_score,
                memory_boost=boost,
                confidence=retrieval_strength,
            ))

        scored.sort(key=lambda r: r.final_score, reverse=True)

        # ── Confidence Action Routing (with cooldown managed by caller)
        top_confidence = scored[0].final_score if scored else 0.0
        if top_confidence < settings.confidence_refine_threshold:
            logger.info("low_confidence_flag", score=top_confidence, action="suggest_refine")
        elif top_confidence < settings.confidence_warn_threshold:
            logger.info("medium_confidence_flag", score=top_confidence, action="warn_user")

        return scored

    def non_blocking_rerank(
        self,
        ranked: list[RankedResult],
        expand_from: list[RetrievedNode],
        classification: ClassificationResult,
    ) -> list[RankedResult]:
        """
        Non-blocking top-K retry (plan Step 11):
        If confidence is low, expand from existing retrieved pool — NO new query.
        Reuses previous results rather than doubling latency.
        """
        if not expand_from:
            return ranked

        # Add extra candidates from pool and re-score
        existing_ids = {r.node.node_id for r in ranked}
        new_nodes    = [n for n in expand_from if n.node_id not in existing_ids]

        if not new_nodes:
            return ranked

        expanded = [r.node for r in ranked] + new_nodes[: settings.max_top_k]
        return self.rank(expanded, classification)

    def get_confidence_gap(self, ranked: list[RankedResult]) -> float:
        """
        Returns gap between top result and second — used by UI to highlight
        when the best result is mathematically dominant.
        """
        if len(ranked) < 2:
            return 1.0
        return round(ranked[0].final_score - ranked[1].final_score, 4)
