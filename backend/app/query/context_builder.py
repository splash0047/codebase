"""
Context Builder & Session Memory (Phase 3 – Steps 12 & 13).

Responsibilities:
  - Context pruning to stay within MAX_TOKENS_ABSOLUTE
  - Adaptive limits based on query complexity (with absolute ceiling)
  - Session memory with precise reset (embedding similarity + keyword overlap)
  - Answer versioning (query + answer + retrieval snapshot)
  - Pipeline timeout budgeting
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.query.classifier import ClassificationResult, QueryIntent
from app.query.ranker import RankedResult

settings = get_settings()
logger   = get_logger(__name__)

# ── Rough token estimation ────────────────────────────────────────────────────
# 1 token ≈ 4 characters (conservative)
CHARS_PER_TOKEN = 4

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


# ── Adaptive token budget ─────────────────────────────────────────────────────

def _get_token_budget(classification: ClassificationResult) -> int:
    """
    Adaptive budget: complex explanation queries get more tokens.
    HARD CEILING = settings.max_tokens_absolute (4000). No exceptions.
    """
    base = settings.max_tokens_absolute
    if classification.intent == QueryIntent.EXPLANATION:
        budget = int(base * 1.0)   # Full budget for explanations
    elif classification.intent == QueryIntent.DEPENDENCY:
        budget = int(base * 0.8)
    else:
        budget = int(base * 0.6)
    # Absolute ceiling — never exceeded
    return min(budget, settings.max_tokens_absolute)


# ── Context Builder ────────────────────────────────────────────────────────────

@dataclass
class BuiltContext:
    chunks:        list[str]
    total_tokens:  int
    node_count:    int
    budget_used_pct: float
    truncated:     bool = False


def build_context(
    ranked: list[RankedResult],
    classification: ClassificationResult,
) -> BuiltContext:
    """
    Prune and assemble LLM context from ranked results.
    Respects adaptive budget with hard ceiling.
    """
    budget = _get_token_budget(classification)
    chunks: list[str] = []
    used   = 0

    for result in ranked:
        text = result.node.chunk_text or ""
        if not text:
            continue
        tokens = _estimate_tokens(text)
        if used + tokens > budget:
            logger.debug("context_budget_hit", used=used, budget=budget)
            return BuiltContext(
                chunks=chunks,
                total_tokens=used,
                node_count=len(chunks),
                budget_used_pct=round(used / budget * 100, 1),
                truncated=True,
            )
        chunks.append(text)
        used += tokens

    return BuiltContext(
        chunks=chunks,
        total_tokens=used,
        node_count=len(chunks),
        budget_used_pct=round(used / budget * 100, 1) if budget > 0 else 0.0,
        truncated=False,
    )


# ── Session Memory ────────────────────────────────────────────────────────────

@dataclass
class SessionTurn:
    query:         str
    normalised:    str
    intent:        QueryIntent
    answer:        str | None = None
    vector:        list[float] | None = None   # Embedding of the query

@dataclass
class SessionMemory:
    """
    Stores recent query turns for context continuity.
    Resets if topic changes abruptly (plan: embedding sim + keyword overlap).
    """
    turns:     list[SessionTurn] = field(default_factory=list)
    max_turns: int = 5   # Keep last 5 turns

    # Thresholds for session reset (plan precision)
    SIM_RESET_THRESHOLD     = 0.4   # Below this cosine sim → reset
    KEYWORD_OVERLAP_MIN     = 0.15  # Require at least 15% keyword overlap to keep context

    def add(self, turn: SessionTurn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def should_reset(self, new_query: str, new_vector: list[float] | None) -> bool:
        """
        Decide if session context should be cleared.
        Uses both embedding similarity AND keyword overlap (plan requirement).
        """
        if not self.turns or new_vector is None:
            return False

        last = self.turns[-1]
        if last.vector is None:
            return False

        # ── Embedding cosine similarity
        sim = _cosine_similarity(last.vector, new_vector)

        # ── Keyword overlap
        old_words = set(last.normalised.split())
        new_words = set(new_query.lower().split())
        if not old_words or not new_words:
            overlap = 0.0
        else:
            overlap = len(old_words & new_words) / len(old_words | new_words)

        if sim < self.SIM_RESET_THRESHOLD and overlap < self.KEYWORD_OVERLAP_MIN:
            logger.info(
                "session_context_reset",
                sim=round(sim, 3),
                overlap=round(overlap, 3),
            )
            return True
        return False

    def reset(self) -> None:
        self.turns.clear()
        logger.info("session_reset")

    def get_context_window(self) -> list[dict[str, str]]:
        """Return recent turns formatted for LLM context."""
        return [
            {"role": "user", "content": t.query,  "intent": t.intent}
            for t in self.turns[-3:]   # Only last 3 turns for LLM context
        ]


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot    = sum(a * b for a, b in zip(v1, v2))
    mag1   = math.sqrt(sum(a * a for a in v1))
    mag2   = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ── Answer Versioning ─────────────────────────────────────────────────────────

@dataclass
class AnswerVersion:
    """
    Snapshot of query + answer + retrieval metadata.
    Used to track pipeline improvements over time.
    """
    query:          str
    intent:         str
    answer:         str
    top_node_ids:   list[str]
    confidence:     float
    timestamp:      float = field(default_factory=time.time)
    pipeline_ms:    float = 0.0
    version_hash:   str = field(init=False)

    def __post_init__(self) -> None:
        raw = f"{self.query}::{self.answer}::{self.confidence}"
        self.version_hash = hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Pipeline Budget Tracker ────────────────────────────────────────────────────

class PipelineTimer:
    """
    Tracks time spent in each pipeline stage.
    Enforces the priority timeout strategy: skip LLM if buffer is threatened.
    """
    BUDGETS_MS = {
        "vector": settings.timeout_vector_ms,
        "graph":  settings.timeout_graph_ms,
        "llm":    settings.timeout_llm_ms,
        "buffer": settings.timeout_buffer_ms,
    }
    TOTAL_BUDGET_MS = sum(BUDGETS_MS.values())

    def __init__(self) -> None:
        self._start   = time.monotonic()
        self._stages: dict[str, float] = {}

    def mark(self, stage: str) -> float:
        elapsed = (time.monotonic() - self._start) * 1000
        self._stages[stage] = elapsed
        return elapsed

    def remaining_ms(self) -> float:
        elapsed = (time.monotonic() - self._start) * 1000
        return max(0.0, self.TOTAL_BUDGET_MS - elapsed)

    def should_skip_llm(self) -> bool:
        """Return True if LLM budget would be violated."""
        remaining = self.remaining_ms()
        return remaining < (settings.timeout_llm_ms + settings.timeout_buffer_ms)

    def summary(self) -> dict[str, float]:
        return {k: round(v, 1) for k, v in self._stages.items()}
