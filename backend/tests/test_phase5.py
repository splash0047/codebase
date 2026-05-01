"""
Phase 5 Tests — Observability, Pinecone, and Benchmarks.

Test categories:
  1. Telemetry — OTel initialisation, span creation, metric recording
  2. Metrics — Prometheus counters/histograms increment correctly
  3. Pinecone — Full implementation (mocked SDK)
  4. Benchmarks — Latency, throughput, and memory regression checks
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Telemetry Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTelemetrySetup(unittest.TestCase):
    """Verify OTel initialisation and span creation."""

    def test_init_and_shutdown_no_crash(self):
        """init_telemetry + shutdown_telemetry must not raise."""
        from app.core.telemetry import init_telemetry, shutdown_telemetry
        init_telemetry()
        shutdown_telemetry()

    def test_get_tracer_returns_tracer(self):
        from app.core.telemetry import init_telemetry, get_tracer
        init_telemetry()
        tracer = get_tracer()
        self.assertIsNotNone(tracer)

    def test_pipeline_span_creates_span(self):
        from app.core.telemetry import init_telemetry, pipeline_span
        init_telemetry()
        with pipeline_span("test_span", {"key": "value"}) as span:
            span.set_attribute("custom", 42)
            self.assertIsNotNone(span)

    def test_pipeline_span_records_exception(self):
        from app.core.telemetry import init_telemetry, pipeline_span
        init_telemetry()
        with self.assertRaises(ValueError):
            with pipeline_span("error_span") as span:
                raise ValueError("test error")

    def test_record_pipeline_metrics_on_span(self):
        from app.core.telemetry import init_telemetry, pipeline_span, record_pipeline_metrics
        init_telemetry()
        with pipeline_span("metrics_test") as span:
            result_data = {
                "intent": "definition",
                "confidence": 0.92,
                "zero_llm_mode": True,
                "cached": False,
                "context_tokens": 1500,
                "coverage_warning": False,
                "confidence_gap": 0.15,
                "pipeline_ms": {"classify": 5.2, "retrieve": 85.0, "rank": 12.3},
            }
            record_pipeline_metrics(span, result_data)
            # No exception = pass


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrometheusMetrics(unittest.TestCase):
    """Verify Prometheus metric operations."""

    def test_query_metrics_recording(self):
        from app.core.metrics import record_query_metrics, QUERY_TOTAL
        before = QUERY_TOTAL._metrics.copy()
        record_query_metrics(
            intent="semantic",
            confidence=0.88,
            zero_llm=False,
            cached=False,
            total_seconds=0.234,
            pipeline_ms={"classify": 5, "retrieve": 100, "rank": 20, "llm": 109},
        )
        # Counter should have incremented
        sample = QUERY_TOTAL.labels(intent="semantic", zero_llm="false", cached="false")
        self.assertGreater(sample._value.get(), 0)

    def test_llm_skipped_counter(self):
        from app.core.metrics import LLM_SKIPPED_TOTAL
        LLM_SKIPPED_TOTAL.labels(reason="zero_llm").inc()
        val = LLM_SKIPPED_TOTAL.labels(reason="zero_llm")._value.get()
        self.assertGreater(val, 0)

    def test_cache_ops_counter(self):
        from app.core.metrics import CACHE_OPS
        CACHE_OPS.labels(operation="hit").inc()
        CACHE_OPS.labels(operation="miss").inc()
        CACHE_OPS.labels(operation="set").inc()
        self.assertGreater(CACHE_OPS.labels(operation="hit")._value.get(), 0)

    def test_metrics_text_generation(self):
        from app.core.metrics import get_metrics_text
        body, content_type = get_metrics_text()
        self.assertIn(b"ckai_query_total", body)
        self.assertIn("text/plain", content_type)

    def test_feedback_counter(self):
        from app.core.metrics import FEEDBACK_TOTAL
        FEEDBACK_TOTAL.labels(helpful="true").inc()
        self.assertGreater(FEEDBACK_TOTAL.labels(helpful="true")._value.get(), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Pinecone Vector Store Tests (mocked SDK)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPineconeVectorStore(unittest.TestCase):
    """Test the production Pinecone implementation with a mocked SDK."""

    def _make_store(self):
        from app.db.vector_store import PineconeVectorStore
        store = PineconeVectorStore("test-repo")
        # Mock the Pinecone client
        mock_index = MagicMock()
        store._index = mock_index
        store._client = MagicMock()
        return store, mock_index

    def test_upsert_batching(self):
        """Upsert should batch at 100 records."""
        import asyncio
        store, mock_index = self._make_store()
        records = [
            {"vector": [0.1] * 1536, "chunk_hash": f"h{i}", "file_path": "a.py",
             "symbol_name": f"fn{i}", "chunk_type": "function", "start_line": i, "end_line": i+5}
            for i in range(150)
        ]
        result = asyncio.get_event_loop().run_until_complete(store.upsert(records))
        self.assertEqual(len(result), 150)
        # Should have been called twice (100 + 50)
        self.assertEqual(mock_index.upsert.call_count, 2)

    def test_upsert_metadata_truncation(self):
        """chunk_text should be truncated to 1000 chars."""
        import asyncio
        store, mock_index = self._make_store()
        records = [
            {"vector": [0.1] * 1536, "chunk_hash": "h1", "file_path": "a.py",
             "symbol_name": "fn1", "chunk_type": "function", "start_line": 1, "end_line": 10,
             "chunk_text": "x" * 5000}
        ]
        asyncio.get_event_loop().run_until_complete(store.upsert(records))
        call_args = mock_index.upsert.call_args
        vectors = call_args[1]["vectors"] if "vectors" in call_args[1] else call_args[0][0]
        meta_text = vectors[0]["metadata"]["chunk_text"]
        self.assertLessEqual(len(meta_text), 1000)

    def test_search_with_filters(self):
        """Search should apply repo_id and file_path filters."""
        import asyncio
        store, mock_index = self._make_store()
        mock_index.query.return_value = {
            "matches": [
                {"id": "v1", "score": 0.95, "metadata": {
                    "file_path": "utils.py", "symbol_name": "helper",
                    "chunk_type": "function", "start_line": 10, "end_line": 20,
                    "chunk_text": "def helper(): ...",
                }},
            ]
        }
        results = asyncio.get_event_loop().run_until_complete(
            store.search([0.1] * 1536, top_k=5, filters={"file_path": "utils.py"})
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["symbol_name"], "helper")
        # Verify filter was passed
        call_filter = mock_index.query.call_args[1]["filter"]
        self.assertIn("file_path", call_filter)

    def test_search_error_returns_empty(self):
        """Search errors should return empty list, not crash."""
        import asyncio
        store, mock_index = self._make_store()
        mock_index.query.side_effect = Exception("connection error")
        results = asyncio.get_event_loop().run_until_complete(
            store.search([0.1] * 1536, top_k=5)
        )
        self.assertEqual(results, [])

    def test_delete_by_file(self):
        """Delete should call Pinecone delete with filter."""
        import asyncio
        store, mock_index = self._make_store()
        count = asyncio.get_event_loop().run_until_complete(
            store.delete_by_file("old_file.py")
        )
        self.assertEqual(count, 1)
        mock_index.delete.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Benchmark Tests — Latency Regression Guards
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarks(unittest.TestCase):
    """
    Performance regression guards.
    These tests ensure critical operations stay within latency budgets.
    """

    def test_classifier_latency_under_10ms(self):
        """Intent classifier must run under 10ms for simple queries."""
        from app.query.classifier import get_classifier
        classifier = get_classifier()
        queries = [
            "Where is the auth handler defined?",
            "What does the UserService class do?",
            "Show me the imports of utils.py",
            "How does caching work in this project?",
        ]
        for q in queries:
            start = time.perf_counter()
            classifier.classify(q)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.assertLess(elapsed_ms, 10, f"Classifier took {elapsed_ms:.1f}ms for: {q}")

    def test_ranker_latency_under_5ms_for_50_results(self):
        """Ranker must handle 50 results under 5ms."""
        from app.query.ranker import ResultRanker
        from app.query.retriever import RetrievedNode
        from app.query.classifier import ClassificationResult

        ranker = ResultRanker()
        nodes = [
            RetrievedNode(
                node_id=f"n{i}", file_path="a.py", symbol_name=f"fn{i}",
                chunk_type="function", start_line=i, end_line=i+5,
                score=0.9 - i * 0.015, sources=["vector"],
            )
            for i in range(50)
        ]
        classification = ClassificationResult(
            intent="semantic", normalised_query="test query",
            confidence=0.8, zero_llm_safe=False,
        )

        start = time.perf_counter()
        ranked = ranker.rank(nodes, classification)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 5, f"Ranker took {elapsed_ms:.1f}ms for 50 results")
        self.assertEqual(len(ranked), 50)

    def test_context_builder_latency_under_10ms(self):
        """Context builder must process 20 ranked results under 10ms."""
        from app.query.context_builder import build_context
        from app.query.ranker import RankedResult
        from app.query.retriever import RetrievedNode
        from app.query.classifier import ClassificationResult

        ranked = [
            RankedResult(
                node=RetrievedNode(
                    node_id=f"n{i}", file_path="a.py", symbol_name=f"fn{i}",
                    chunk_type="function", start_line=i, end_line=i+10,
                    score=0.9, sources=["vector"], chunk_text=f"def fn{i}(): pass\n" * 5,
                ),
                final_score=0.9 - i * 0.02,
            )
            for i in range(20)
        ]
        classification = ClassificationResult(
            intent="explanation", normalised_query="explain the pipeline",
            confidence=0.75, zero_llm_safe=False,
        )

        start = time.perf_counter()
        ctx = build_context(ranked, classification)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 10, f"Context builder took {elapsed_ms:.1f}ms")
        self.assertGreater(ctx.total_tokens, 0)

    def test_fingerprint_latency_under_1ms(self):
        """Cache fingerprinting must be sub-millisecond."""
        import asyncio
        from app.query.pipeline import _fingerprint

        vector = [0.123] * 1536
        start = time.perf_counter()
        fp = asyncio.get_event_loop().run_until_complete(
            _fingerprint(vector, "test query")
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 1, f"Fingerprint took {elapsed_ms:.2f}ms")
        self.assertTrue(fp.startswith("sem:"))

    def test_zero_llm_answer_latency_under_1ms(self):
        """Zero-LLM answer generation must be sub-millisecond."""
        from app.query.pipeline import _zero_llm_answer
        from app.query.ranker import RankedResult
        from app.query.retriever import RetrievedNode

        ranked = [
            RankedResult(
                node=RetrievedNode(
                    node_id=f"n{i}", file_path="a.py", symbol_name=f"fn{i}",
                    chunk_type="function", start_line=i, end_line=i+5,
                    score=0.9, sources=["vector", "graph"],
                ),
                final_score=0.9 - i * 0.05,
            )
            for i in range(5)
        ]

        start = time.perf_counter()
        answer = _zero_llm_answer(ranked, "find auth handler")
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 1, f"Zero-LLM answer took {elapsed_ms:.2f}ms")
        self.assertIn("fn0", answer)

    def test_metrics_text_generation_under_50ms(self):
        """Prometheus metrics endpoint must respond under 50ms."""
        from app.core.metrics import get_metrics_text
        start = time.perf_counter()
        body, _ = get_metrics_text()
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 50, f"Metrics generation took {elapsed_ms:.1f}ms")
        self.assertGreater(len(body), 100)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Vector Store Router Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreRouter(unittest.TestCase):
    """Verify the router selects the correct backend."""

    @patch.dict("os.environ", {"VECTOR_STORE": "faiss"})
    def test_faiss_selection(self):
        from app.db.vector_store import get_vector_store, FAISSVectorStore, _store_cache
        _store_cache.clear()
        store = get_vector_store("test-faiss")
        self.assertIsInstance(store, FAISSVectorStore)
        _store_cache.clear()

    @patch.dict("os.environ", {"VECTOR_STORE": "pinecone"})
    def test_pinecone_selection(self):
        from app.db.vector_store import get_vector_store, PineconeVectorStore, _store_cache
        _store_cache.clear()
        store = get_vector_store("test-pinecone")
        self.assertIsInstance(store, PineconeVectorStore)
        _store_cache.clear()

    def test_store_caching(self):
        from app.db.vector_store import get_vector_store, _store_cache
        _store_cache.clear()
        s1 = get_vector_store("cache-test")
        s2 = get_vector_store("cache-test")
        self.assertIs(s1, s2)
        _store_cache.clear()


if __name__ == "__main__":
    unittest.main()
