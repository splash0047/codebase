"""
Phase 3 Test Suite — Query Classifier, Ranker, Context Builder.
Run with: pytest backend/tests/test_phase3.py -v
"""
import pytest
import math
from app.query.classifier import (
    QueryClassifier, QueryIntent, normalise_query, NEGATION_TOKENS
)
from app.query.ranker import ResultRanker, _usage_boost
from app.query.context_builder import (
    SessionMemory, SessionTurn, build_context, PipelineTimer,
    _cosine_similarity,
)
from app.query.retriever import (
    _deduplicate, _prune_noise, RetrievedNode,
)


# ── Classifier Tests ──────────────────────────────────────────────────────────

def test_classify_definition_intent():
    clf = QueryClassifier()
    r   = clf.classify("Where is the authenticate function defined?")
    assert r.intent == QueryIntent.DEFINITION

def test_classify_dependency_intent():
    clf = QueryClassifier()
    r   = clf.classify("What does the auth module depend on?")
    assert r.intent == QueryIntent.DEPENDENCY

def test_classify_explanation_intent():
    clf = QueryClassifier()
    r   = clf.classify("Explain how the login flow works")
    assert r.intent == QueryIntent.EXPLANATION

def test_classify_semantic_intent():
    clf = QueryClassifier()
    r   = clf.classify("What is the user authentication concept?")
    # Should be semantic or explanation
    assert r.intent in (QueryIntent.SEMANTIC, QueryIntent.EXPLANATION, QueryIntent.UNKNOWN)

def test_classify_low_confidence_uses_hybrid():
    clf = QueryClassifier()
    r   = clf.classify("foo bar baz")   # No strong signals
    assert r.use_hybrid or r.intent == QueryIntent.UNKNOWN

def test_user_override_respected():
    clf = QueryClassifier()
    r   = clf.classify("Where is X?", user_override=QueryIntent.SEMANTIC)
    assert r.intent == QueryIntent.SEMANTIC
    assert r.confidence == 1.0

def test_zero_llm_safe_for_definition():
    clf = QueryClassifier()
    r   = clf.classify("Where is login defined?")
    if not r.use_hybrid:
        assert r.zero_llm_safe

def test_zero_llm_not_safe_for_explanation():
    clf = QueryClassifier()
    r   = clf.classify("Explain the entire login flow in detail")
    if r.intent == QueryIntent.EXPLANATION:
        assert not r.zero_llm_safe


# ── Normalisation Tests ───────────────────────────────────────────────────────

def test_normalise_preserves_negation():
    result = normalise_query("how is auth NOT handled here")
    assert "not" in result.lower(), "Negation 'NOT' must be preserved"

def test_normalise_lowercases():
    result = normalise_query("WHERE IS LoginService DEFINED")
    assert result == result.lower()

def test_normalise_strips_extra_whitespace():
    result = normalise_query("where   is   login   defined")
    assert "  " not in result


# ── Ranker Tests ──────────────────────────────────────────────────────────────

def _make_node(node_id, score, file_path="a.py", sources=None):
    return RetrievedNode(
        node_id=node_id, node_type="vector", score=score,
        file_path=file_path, symbol_name=node_id,
        sources=sources or ["vector"],
    )

def test_ranker_sorts_descending():
    from app.query.classifier import ClassificationResult
    nodes = [_make_node("b", 0.3), _make_node("a", 0.9), _make_node("c", 0.6)]
    clf   = QueryClassifier()
    cr    = clf.classify("What is login?")
    ranked = ResultRanker().rank(nodes, cr)
    scores = [r.final_score for r in ranked]
    assert scores == sorted(scores, reverse=True)

def test_memory_boost_capped():
    boost = _usage_boost(usage_freq=10000, last_used_days=0.0)
    from app.core.config import get_settings
    assert boost <= get_settings().memory_boost_max_pct

def test_memory_boost_decays_with_time():
    fresh = _usage_boost(100, 0.0)
    stale = _usage_boost(100, 60.0)
    assert fresh > stale

def test_confidence_gap():
    from app.query.classifier import ClassificationResult
    nodes  = [_make_node("top", 0.9), _make_node("mid", 0.4)]
    cr     = QueryClassifier().classify("locate function")
    ranked = ResultRanker().rank(nodes, cr)
    gap    = ResultRanker().get_confidence_gap(ranked)
    assert 0.0 < gap <= 1.0


# ── Deduplication Tests ───────────────────────────────────────────────────────

def test_dedup_merges_sources():
    n1 = RetrievedNode("id1", "vector", 0.8, sources=["vector"])
    n2 = RetrievedNode("id1", "graph",  0.6, sources=["graph"])
    result = _deduplicate([n1, n2])
    assert len(result) == 1
    assert "vector" in result[0].sources
    assert "graph"  in result[0].sources

def test_dedup_keeps_max_score():
    n1 = RetrievedNode("id1", "vector", 0.8, sources=["vector"])
    n2 = RetrievedNode("id1", "graph",  0.5, sources=["graph"])
    result = _deduplicate([n1, n2])
    assert result[0].score == 0.8

def test_dedup_preserves_unique():
    n1 = RetrievedNode("id1", "vector", 0.8)
    n2 = RetrievedNode("id2", "vector", 0.7)
    assert len(_deduplicate([n1, n2])) == 2


# ── Noise Pruning Tests ───────────────────────────────────────────────────────

def test_noise_prune_removes_low_confidence():
    nodes = [
        RetrievedNode("a", "vector", 0.9),
        RetrievedNode("b", "vector", 0.1),
        RetrievedNode("c", "vector", 0.1),
        RetrievedNode("d", "vector", 0.1),
        RetrievedNode("e", "vector", 0.1),
    ]
    pruned = _prune_noise(nodes)
    # 4/5 = 80% low confidence → should prune
    assert len(pruned) < len(nodes)

def test_noise_prune_keeps_good_results():
    nodes = [RetrievedNode(f"id{i}", "vector", 0.8) for i in range(5)]
    assert len(_prune_noise(nodes)) == 5


# ── Session Memory Tests ──────────────────────────────────────────────────────

def test_session_reset_on_topic_change():
    session = SessionMemory()
    old_vec = [1.0, 0.0, 0.0] + [0.0] * 1533
    new_vec = [0.0, 1.0, 0.0] + [0.0] * 1533
    session.add(SessionTurn("auth query", "auth query", QueryIntent.DEFINITION, vector=old_vec))
    assert session.should_reset("database schema", new_vec)

def test_session_no_reset_on_same_topic():
    session = SessionMemory()
    vec = [1.0] * 1536
    session.add(SessionTurn("auth query", "auth query", QueryIntent.DEFINITION, vector=vec))
    assert not session.should_reset("auth function", vec)


# ── Pipeline Timer Tests ──────────────────────────────────────────────────────

def test_pipeline_timer_marks():
    timer = PipelineTimer()
    import time; time.sleep(0.01)
    elapsed = timer.mark("test")
    assert elapsed > 0

def test_pipeline_timer_remaining():
    timer = PipelineTimer()
    rem   = timer.remaining_ms()
    assert rem > 0

def test_cosine_similarity_identical():
    v = [1.0, 0.5, 0.25]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-5

def test_cosine_similarity_orthogonal():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert abs(_cosine_similarity(v1, v2)) < 1e-5
