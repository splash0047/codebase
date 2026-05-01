"""
Main Query Pipeline Orchestrator (Phase 3 – Steps 9–13).

Full pipeline:
  normalise → classify → cold-query check → embed query →
  hybrid retrieve → rank → context prune → LLM (or Zero-LLM) → stream

Plan guarantees:
  - Strict pipeline timeout budget (Vector/Graph/LLM/Buffer)
  - Priority timeout: skip LLM if buffer is threatened
  - Zero-LLM mode for DEFINITION + DEPENDENCY (returns graph/AST directly)
  - Partial result streaming (structured: Searching → Found → Answer)
  - Semantic query fingerprinting for cache (not just string hash)
  - Session reset when topic drifts
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncGenerator, Any

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import pipeline_span, record_pipeline_metrics, get_tracer
from app.core.metrics import (
    record_query_metrics,
    CACHE_OPS,
    LLM_SKIPPED_TOTAL,
)
from app.db.redis_client import (
    get_cached_result, set_cached_result,
    get_query_frequency, increment_query_frequency,
)
from app.ingestion.embedder import _embed_batch_with_fallback
from app.query.classifier import ClassificationResult, QueryIntent, get_classifier
from app.query.context_builder import (
    AnswerVersion, BuiltContext, PipelineTimer,
    SessionMemory, SessionTurn, build_context,
)
from app.query.ranker import RankedResult, ResultRanker
from app.query.retriever import HybridRetriever, RetrievedNode

settings = get_settings()
logger   = get_logger(__name__)


# ── Streaming event types ─────────────────────────────────────────────────────

STREAM_EVENTS = {
    "searching":  "🔍 Searching codebase…",
    "found":      "📂 Found relevant code…",
    "analysing":  "🧠 Analysing with AI…",
    "complete":   "✅ Done",
    "fallback":   "⚠️ Using structural results (AI unavailable)",
    "partial":    "⚠️ Partial results — coverage may be incomplete",
}


def _stream_event(event: str, data: Any = None) -> str:
    payload = {"event": event, "message": STREAM_EVENTS.get(event, event), "data": data}
    return f"data: {json.dumps(payload)}\n\n"


# ── Cache fingerprinting ──────────────────────────────────────────────────────

async def _fingerprint(query_vector: list[float] | None, normalised: str) -> str:
    """
    Semantic fingerprinting using embedding vector (not just string hash).
    Falls back to string hash if vector unavailable.
    """
    if query_vector and len(query_vector) > 0:
        # Use first 8 floats as a compact fingerprint signature
        sig = "_".join(f"{x:.3f}" for x in query_vector[:8])
        return f"sem:{sig}"
    import hashlib
    return f"str:{hashlib.md5(normalised.encode()).hexdigest()[:16]}"


async def _is_cold_query(fingerprint: str) -> bool:
    """Skip cache if query frequency < threshold (cold query detection)."""
    freq = await get_query_frequency(fingerprint)
    return freq < settings.cold_query_frequency_min


# ── LLM Answer Generation ─────────────────────────────────────────────────────

async def _call_llm(
    query: str,
    context: BuiltContext,
    classification: ClassificationResult,
    session_turns: list[dict],
) -> str:
    """Call OpenAI LLM with context. Respects timeout budget."""
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    context_text = "\n\n---\n\n".join(context.chunks)
    system_prompt = (
        "You are a developer intelligence assistant. "
        "Answer ONLY based on the provided code context. "
        "If you cannot answer from the context, say so explicitly. "
        "Always cite the file and function name where relevant."
    )

    messages = [{"role": "system", "content": system_prompt}]
    # Add session context (last 3 turns)
    for turn in session_turns[-3:]:
        messages.append({"role": "user", "content": turn.get("content", "")})

    messages.append({
        "role": "user",
        "content": f"Code Context:\n{context_text}\n\nQuestion: {query}",
    })

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=min(1000, settings.max_tokens_absolute // 4),
            temperature=0.2,
        ),
        timeout=settings.timeout_llm_ms / 1000,
    )
    return response.choices[0].message.content or ""


def _zero_llm_answer(ranked: list[RankedResult], query: str) -> str:
    """
    Pure graph + AST answer for DEFINITION and DEPENDENCY queries.
    No LLM — deterministic, no hallucination risk.
    Scope limited to symbol lookup + dependency tracing (plan requirement).
    """
    lines = [f"**Query:** {query}\n"]
    for i, r in enumerate(ranked[:5], 1):
        node = r.node
        lines.append(
            f"{i}. **{node.symbol_name or 'Unknown'}** "
            f"in `{node.file_path or 'unknown'}` "
            f"(lines {node.start_line}–{node.end_line})"
            f"\n   Score: {r.final_score:.3f} | Sources: {', '.join(node.sources)}"
        )
    return "\n".join(lines)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    answer:          str
    intent:          str
    confidence:      float
    ranked_results:  list[dict]
    context_tokens:  int
    pipeline_ms:     dict[str, float]
    coverage_warning: bool = False
    zero_llm_mode:   bool = False
    cached:          bool = False
    version_hash:    str  = ""
    confidence_gap:  float = 0.0


class QueryPipeline:
    def __init__(self, repo_id: str, session: SessionMemory | None = None):
        self.repo_id   = repo_id
        self.session   = session or SessionMemory()
        self.retriever = HybridRetriever()
        self.ranker    = ResultRanker()

    async def run(
        self,
        raw_query: str,
        user_intent_override: QueryIntent | None = None,
        reproducible: bool = False,
    ) -> QueryResult:
        """
        Full synchronous pipeline — use stream() for SSE streaming.
        Fully instrumented with OpenTelemetry spans and Prometheus metrics.
        """
        import time as _time
        _start = _time.monotonic()

        tracer = get_tracer()
        with tracer.start_as_current_span("query_pipeline") as root_span:
            root_span.set_attribute("query.raw", raw_query[:200])
            root_span.set_attribute("query.repo_id", self.repo_id)

            await self.session.load()
            
            timer = PipelineTimer()

            # ── 1. Classify & normalise
            with pipeline_span("classify", {"query": raw_query[:100]}) as span:
                classifier     = get_classifier()
                classification = classifier.classify(raw_query, user_intent_override)
                normalised     = classification.normalised_query
                span.set_attribute("intent", classification.intent)
                span.set_attribute("confidence", classification.confidence)
                span.set_attribute("zero_llm_safe", classification.zero_llm_safe)
            timer.mark("classify")

            # ── 2. Embed query
            query_vector: list[float] | None = None
            with pipeline_span("embed_query") as span:
                if not classification.zero_llm_safe:
                    vecs = await _embed_batch_with_fallback([normalised])
                    query_vector = vecs[0] if vecs else None
                    span.set_attribute("has_vector", query_vector is not None)
                else:
                    span.set_attribute("skipped", True)
            timer.mark("embed")

            # ── 3. Session reset check
            if self.session.should_reset(normalised, query_vector):
                self.session.reset()

            # ── 4. Cold-query + cache check
            fingerprint  = await _fingerprint(query_vector, normalised)
            is_cold      = await _is_cold_query(fingerprint)
            await increment_query_frequency(fingerprint)

            if not is_cold and not reproducible:
                cached = await get_cached_result(f"repo:{self.repo_id}:{fingerprint}")
                if cached:
                    CACHE_OPS.labels(operation="hit").inc()
                    logger.info("query_cache_hit", fingerprint=fingerprint)
                    return QueryResult(**{**cached, "cached": True})
                CACHE_OPS.labels(operation="miss").inc()

            # ── 5. Retrieve
            with pipeline_span("retrieve", {"repo_id": self.repo_id}) as span:
                timer.mark("retrieve_start")
                ranked_results = await self.retriever.retrieve(
                    repo_id=self.repo_id,
                    query=normalised,
                    classification=classification,
                    query_vector=query_vector,
                )
                span.set_attribute("result_count", len(ranked_results))
            timer.mark("retrieve_done")

            # ── 6. Rank
            with pipeline_span("rank") as span:
                ranked = self.ranker.rank(ranked_results, classification)

                # ── 7. Non-blocking rerank if confidence is low
                if ranked and ranked[0].final_score < settings.confidence_refine_threshold:
                    ranked = self.ranker.non_blocking_rerank(ranked, ranked_results, classification)

                confidence_gap = self.ranker.get_confidence_gap(ranked)
                top_confidence = ranked[0].final_score if ranked else 0.0
                span.set_attribute("top_score", top_confidence)
                span.set_attribute("confidence_gap", confidence_gap)
                span.set_attribute("ranked_count", len(ranked))
            timer.mark("rank")

            # ── 8. Build context
            with pipeline_span("build_context") as span:
                context = build_context(ranked, classification)
                span.set_attribute("total_tokens", context.total_tokens)
                span.set_attribute("truncated", context.truncated)

            # ── 9. Answer generation
            answer       = ""
            zero_llm     = False

            if classification.zero_llm_safe and ranked:
                with pipeline_span("zero_llm_answer"):
                    answer   = _zero_llm_answer(ranked, raw_query)
                    zero_llm = True
                    LLM_SKIPPED_TOTAL.labels(reason="zero_llm").inc()
                timer.mark("zero_llm")
            elif timer.should_skip_llm():
                with pipeline_span("llm_skipped_timeout"):
                    answer   = _zero_llm_answer(ranked, raw_query)
                    zero_llm = True
                    LLM_SKIPPED_TOTAL.labels(reason="timeout").inc()
                logger.warning("llm_skipped_timeout", remaining_ms=timer.remaining_ms())
                timer.mark("llm_skipped")
            elif not settings.openai_api_key or not context.chunks:
                answer = _zero_llm_answer(ranked, raw_query)
                zero_llm = True
                LLM_SKIPPED_TOTAL.labels(reason="no_api_key").inc()
            else:
                try:
                    with pipeline_span("llm_call", {"model": "gpt-4o-mini"}):
                        answer = await _call_llm(
                            raw_query, context, classification,
                            self.session.get_context_window(),
                        )
                    timer.mark("llm")
                except asyncio.TimeoutError:
                    answer   = _zero_llm_answer(ranked, raw_query)
                    zero_llm = True
                    LLM_SKIPPED_TOTAL.labels(reason="llm_timeout").inc()
                    logger.warning("llm_timeout")

            # ── 10. Session update
            self.session.add(SessionTurn(
                query=raw_query, normalised=normalised,
                intent=classification.intent, answer=answer,
                vector=query_vector,
            ))
            await self.session.save()

            # ── 11. Answer versioning
            version = AnswerVersion(
                query=raw_query, intent=classification.intent,
                answer=answer,
                top_node_ids=[r.node.node_id for r in ranked[:5]],
                confidence=top_confidence,
                pipeline_ms=timer.summary().get("llm", 0.0),
            )

            result = QueryResult(
                answer=answer,
                intent=classification.intent,
                confidence=top_confidence,
                ranked_results=[
                    {
                        "file":     r.node.file_path,
                        "symbol":   r.node.symbol_name,
                        "score":    r.final_score,
                        "lines":    f"{r.node.start_line}–{r.node.end_line}",
                        "sources":  r.node.sources,
                        "gap":      confidence_gap,
                    }
                    for r in ranked[:10]
                ],
                context_tokens=context.total_tokens,
                pipeline_ms=timer.summary(),
                coverage_warning=context.truncated,
                zero_llm_mode=zero_llm,
                version_hash=version.version_hash,
                confidence_gap=confidence_gap,
            )

            # ── Record telemetry & metrics
            import dataclasses
            record_pipeline_metrics(root_span, dataclasses.asdict(result))

            total_seconds = _time.monotonic() - _start
            record_query_metrics(
                intent=classification.intent,
                confidence=top_confidence,
                zero_llm=zero_llm,
                cached=False,
                total_seconds=total_seconds,
                pipeline_ms=timer.summary(),
            )

            # ── Cache (skip for cold / reproducible queries)
            if not is_cold and not reproducible:
                await set_cached_result(
                    f"repo:{self.repo_id}:{fingerprint}",
                    dataclasses.asdict(result),
                )
                CACHE_OPS.labels(operation="set").inc()

            return result

    async def stream(
        self,
        raw_query: str,
        user_intent_override: QueryIntent | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        SSE streaming in structured order (plan Step 13):
          1. "Searching…"
          2. "Found relevant code…"
          3. Final answer
        Never dumps raw nodes first.
        """
        yield _stream_event("searching")

        try:
            # Classify + retrieve phase
            classifier     = get_classifier()
            classification = classifier.classify(raw_query, user_intent_override)
            vecs           = await _embed_batch_with_fallback([classification.normalised_query])
            query_vector   = vecs[0] if vecs else None
            
            await self.session.load()

            ranked_nodes = await self.retriever.retrieve(
                repo_id=self.repo_id,
                query=classification.normalised_query,
                classification=classification,
                query_vector=query_vector,
            )

            yield _stream_event("found", {"count": len(ranked_nodes)})

            ranked  = self.ranker.rank(ranked_nodes, classification)
            context = build_context(ranked, classification)

            # Generate answer
            timer = PipelineTimer()
            if classification.zero_llm_safe or timer.should_skip_llm():
                answer = _zero_llm_answer(ranked, raw_query)
                yield _stream_event("fallback")
            else:
                yield _stream_event("analysing")
                try:
                    answer = await _call_llm(
                        raw_query, context, classification,
                        self.session.get_context_window(),
                    )
                except Exception:
                    answer = _zero_llm_answer(ranked, raw_query)
                    yield _stream_event("fallback")

            payload = {
                "answer":     answer,
                "intent":     classification.intent,
                "confidence": ranked[0].final_score if ranked else 0.0,
                "results":    [
                    {"file": r.node.file_path, "symbol": r.node.symbol_name, "score": r.final_score}
                    for r in ranked[:5]
                ],
            }
            
            # Session update
            self.session.add(SessionTurn(
                query=raw_query, normalised=classification.normalised_query,
                intent=classification.intent, answer=answer,
                vector=query_vector,
            ))
            await self.session.save()

            yield _stream_event("complete", payload)

        except Exception as exc:
            logger.error("query_pipeline_failed", error=str(exc))
            yield _stream_event("partial", {"error": str(exc)})
