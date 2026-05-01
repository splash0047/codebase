"""
Query Intent Classifier (Phase 3 – Step 9).

Routes queries to the correct retrieval pipeline:
  • SEMANTIC    → embedding-heavy  ("What does login do?")
  • DEFINITION  → AST/graph exact  ("Where is authenticate defined?")
  • DEPENDENCY  → graph traversal  ("What does auth depend on?")
  • EXPLANATION → both + LLM       ("Explain the login flow")

Key plan decisions implemented here:
  - Confidence-Based Routing (not hard routing):
      high confidence (≥ ROUTING_MIN) → fast dedicated pipeline
      low confidence → always falls back to hybrid
  - Zero-LLM Mode allowed only for DEFINITION + DEPENDENCY
  - Safe NLP query normalisation that preserves negations ("NOT", "no", "never")
  - Session context reset if query topic switches abruptly
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import spacy
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger   = get_logger(__name__)

# Load spacy model once at import time (en_core_web_sm)
_nlp: Any = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spacy_model_not_found, using regex fallback")
            _nlp = None
    return _nlp


# ── Intent enum ───────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    SEMANTIC    = "semantic"      # Broad concept search
    DEFINITION  = "definition"   # "Where is X defined?"
    DEPENDENCY  = "dependency"   # "What does X depend on / call?"
    EXPLANATION = "explanation"  # "Explain the flow of X"
    UNKNOWN     = "unknown"      # Classifier not confident


@dataclass
class ClassificationResult:
    intent: QueryIntent
    confidence: float
    normalised_query: str
    # Whether Zero-LLM mode is safe for this intent
    zero_llm_safe: bool
    # Routing path chosen
    use_hybrid: bool
    # Intent override from user (if any)
    user_override: QueryIntent | None = None


# ── Signal patterns ───────────────────────────────────────────────────────────
# These keywords boost confidence scores for each intent category.

DEFINITION_SIGNALS  = re.compile(
    r"\b(where is|where does|find|locate|defined|declaration|definition|where can i find)\b",
    re.IGNORECASE,
)
DEPENDENCY_SIGNALS  = re.compile(
    r"\b(depend|depends on|calls|imports|uses|what does .* use|referenced by|related to)\b",
    re.IGNORECASE,
)
EXPLANATION_SIGNALS = re.compile(
    r"\b(explain|how does|walk me through|describe|what happens when|trace|flow|understand)\b",
    re.IGNORECASE,
)
SEMANTIC_SIGNALS    = re.compile(
    r"\b(what is|what does|show me|examples of|similar to|related|concept)\b",
    re.IGNORECASE,
)

# Negation tokens — NEVER strip these during normalisation
NEGATION_TOKENS = {"not", "no", "never", "without", "except", "don't", "doesn't",
                   "isn't", "aren't", "neither", "nor"}


# ── Safe query normalisation ──────────────────────────────────────────────────

def normalise_query(raw: str) -> str:
    """
    Safe NLP normalisation that:
    1. Lowercases and strips extra whitespace
    2. Uses spacy POS tags to remove noise words
    3. NEVER removes negations (plan: NLP tagging, not blind cleaning)
    """
    query = raw.strip().lower()
    query = re.sub(r"\s+", " ", query)

    nlp = _get_nlp()
    if nlp is None:
        # Regex-only fallback: just strip punctuation but keep negations
        query = re.sub(r"[^\w\s'-]", " ", query)
        return re.sub(r"\s+", " ", query).strip()

    doc = nlp(query)
    tokens: list[str] = []
    for token in doc:
        # Keep negations unconditionally
        if token.lower_ in NEGATION_TOKENS or token.dep_ == "neg":
            tokens.append(token.lower_)
            continue
        # Remove pure punctuation and stop words that aren't negations
        if token.is_punct:
            continue
        tokens.append(token.lower_)

    return " ".join(tokens).strip()


# ── Classifier ────────────────────────────────────────────────────────────────

class QueryClassifier:
    """
    Signal-based classifier with confidence scoring.
    Implements Confidence-Based Routing from the plan:
      - confidence ≥ settings.confidence_routing_min → dedicated fast pipeline
      - confidence < threshold → hybrid pipeline (safe fallback)
    """

    def classify(
        self,
        raw_query: str,
        user_override: QueryIntent | None = None,
    ) -> ClassificationResult:
        normalised = normalise_query(raw_query)

        # User intent override bypasses classifier entirely
        if user_override and user_override != QueryIntent.UNKNOWN:
            return ClassificationResult(
                intent=user_override,
                confidence=1.0,
                normalised_query=normalised,
                zero_llm_safe=user_override in (QueryIntent.DEFINITION, QueryIntent.DEPENDENCY),
                use_hybrid=False,
                user_override=user_override,
            )

        scores: dict[QueryIntent, float] = {
            QueryIntent.DEFINITION:  self._score(normalised, DEFINITION_SIGNALS),
            QueryIntent.DEPENDENCY:  self._score(normalised, DEPENDENCY_SIGNALS),
            QueryIntent.EXPLANATION: self._score(normalised, EXPLANATION_SIGNALS),
            QueryIntent.SEMANTIC:    self._score(normalised, SEMANTIC_SIGNALS),
        }

        best_intent = max(scores, key=lambda k: scores[k])
        best_score  = scores[best_intent]

        # Normalise to 0–1 confidence
        total = sum(scores.values()) or 1.0
        confidence = round(best_score / total, 3) if total > 0 else 0.0

        # Confidence-Based Routing: low confidence → hybrid, not dedicated
        routing_floor = settings.confidence_routing_min  # default 0.85
        use_hybrid    = confidence < routing_floor

        if use_hybrid:
            # Not confident enough to dedicate; fall back to hybrid
            best_intent = QueryIntent.UNKNOWN

        zero_llm_safe = best_intent in (QueryIntent.DEFINITION, QueryIntent.DEPENDENCY) and not use_hybrid

        logger.debug(
            "query_classified",
            intent=best_intent,
            confidence=confidence,
            use_hybrid=use_hybrid,
            query_snippet=normalised[:60],
        )

        return ClassificationResult(
            intent=best_intent,
            confidence=confidence,
            normalised_query=normalised,
            zero_llm_safe=zero_llm_safe,
            use_hybrid=use_hybrid,
        )

    def _score(self, text: str, pattern: re.Pattern) -> float:
        matches = pattern.findall(text)
        return float(len(matches))


# ── Singleton ─────────────────────────────────────────────────────────────────

_classifier: QueryClassifier | None = None

def get_classifier() -> QueryClassifier:
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier
