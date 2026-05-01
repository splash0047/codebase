"""
Query API Routes (Phase 3 – Step 13).

Endpoints:
  POST /api/v1/query          — synchronous query
  GET  /api/v1/query/stream   — SSE streaming query
  POST /api/v1/query/feedback — result quality feedback
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.logging import get_logger
from app.query.classifier import QueryIntent
from app.query.context_builder import SessionMemory
from app.query.pipeline import QueryPipeline, QueryResult

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

def _get_session(repo_id: str, session_id: str | None = None) -> SessionMemory:
    """Returns a SessionMemory instance backed by Redis."""
    sid = session_id or f"default_{repo_id}"
    return SessionMemory(session_id=sid)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:               str
    repo_id:             str
    session_id:          str | None = None
    intent_override:     QueryIntent | None = None
    reproducible:        bool = False    # Reproducibility Mode
    zero_llm:            bool = False    # Force Zero-LLM mode


class FeedbackRequest(BaseModel):
    repo_id:     str
    query:       str
    version_hash: str
    helpful:     bool
    comment:     str | None = None


# ── Synchronous Query ─────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=None,
    summary="Submit a synchronous query against an indexed repository",
)
async def query(payload: QueryRequest) -> dict:
    session  = _get_session(payload.repo_id, payload.session_id)
    pipeline = QueryPipeline(repo_id=payload.repo_id, session=session)

    override = payload.intent_override
    if payload.zero_llm:
        # Force Zero-LLM by overriding intent to DEFINITION
        override = QueryIntent.DEFINITION

    result = await pipeline.run(
        raw_query=payload.query,
        user_intent_override=override,
        reproducible=payload.reproducible,
    )

    # Confidence Action Routing — show warning/suggestion to user
    response = {
        "answer":           result.answer,
        "intent":           result.intent,
        "confidence":       result.confidence,
        "confidence_gap":   result.confidence_gap,
        "results":          result.ranked_results,
        "context_tokens":   result.context_tokens,
        "pipeline_ms":      result.pipeline_ms,
        "zero_llm_mode":    result.zero_llm_mode,
        "cached":           result.cached,
        "version_hash":     result.version_hash,
        "coverage_warning": result.coverage_warning,
    }

    # Confidence-based UI signals
    if result.confidence < 0.3:
        response["ui_action"] = "suggest_refine"
        response["ui_message"] = "⚠️ Results may not fully answer your question. Try rephrasing."
    elif result.confidence < 0.5:
        response["ui_action"] = "warn"
        response["ui_message"] = "ℹ️ Results are uncertain. Review carefully."

    return response


# ── SSE Streaming Query ────────────────────────────────────────────────────────

@router.get(
    "/stream",
    summary="Stream query results via Server-Sent Events",
)
async def stream_query(
    repo_id:         str = Query(...),
    query:           str = Query(...),
    session_id:      str = Query(None),
    intent_override: str = Query(None),
) -> StreamingResponse:
    override = QueryIntent(intent_override) if intent_override else None
    session  = _get_session(repo_id, session_id)
    pipeline = QueryPipeline(repo_id=repo_id, session=session)

    return StreamingResponse(
        pipeline.stream(query, user_intent_override=override),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post("/feedback", summary="Submit implicit trust feedback on a result")
async def submit_feedback(payload: FeedbackRequest) -> dict:
    """
    Implicit trust filtering (plan Step 15):
    Feedback is logged but weighted — clicks alone don't boost scores.
    Dwell time is tracked separately by the frontend.
    """
    logger.info(
        "query_feedback",
        repo_id=payload.repo_id,
        version_hash=payload.version_hash,
        helpful=payload.helpful,
        has_comment=bool(payload.comment),
    )
    return {"status": "received", "version_hash": payload.version_hash}
