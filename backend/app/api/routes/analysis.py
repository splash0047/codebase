"""
Advanced Analysis API Routes (Phase 6).

Endopints:
  POST /api/v1/analysis/diff-impact — Compute blast radius and code smells for a git diff.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.logging import get_logger
from app.db.neo4j_client import get_neo4j_session

logger = get_logger(__name__)
router = APIRouter(prefix="/analysis", tags=["Analysis"])


class DiffPayload(BaseModel):
    file_path: str
    changed_symbols: list[str]


class DiffAnalysisRequest(BaseModel):
    repo_id: str
    changes: list[DiffPayload]


@router.post(
    "/diff-impact",
    summary="Compute blast radius and potential code smells for changed files",
)
async def analyze_diff_impact(payload: DiffAnalysisRequest) -> dict:
    """
    Diff-graph analysis (Phase 6).
    Given a list of changed files and symbols, calculates the downstream
    impact (blast radius) by traversing inbound dependency edges in Neo4j.
    Also flags potential code smells (e.g. tightly coupled God classes).
    """
    repo_id = payload.repo_id
    impact_report = {
        "affected_files": set(),
        "affected_symbols": [],
        "smells": [],
    }

    try:
        async with get_neo4j_session() as session:
            for change in payload.changes:
                # 1. Blast Radius: Find everything that depends on this file/symbol
                for symbol in change.changed_symbols:
                    # Query inbound edges (CALLS, IMPORTS, EXTENDS) up to depth 3
                    result = await session.run(
                        """
                        MATCH path = (dependent)-[*1..3]->(target)
                        WHERE target.repo_id = $repo_id 
                          AND (target.name = $symbol OR target.id ENDS WITH $file_path)
                        RETURN dependent.id AS id, dependent.name AS name, 
                               labels(dependent)[0] AS type, dependent.file_id AS file_path
                        """,
                        repo_id=repo_id, symbol=symbol, file_path=change.file_path,
                    )
                    records = await result.data()
                    
                    for rec in records:
                        impact_report["affected_files"].add(rec.get("file_path"))
                        impact_report["affected_symbols"].append({
                            "symbol": rec.get("name"),
                            "type": rec.get("type"),
                            "file": rec.get("file_path"),
                            "dependency_of": symbol,
                        })

                # 2. Code Smell Detection: High inbound/outbound coupling
                # Find if the changed file is a "God Class" (high fan-out/fan-in)
                smell_result = await session.run(
                    """
                    MATCH (n:File {id: $file_id})-[r]-()
                    WITH n, count(r) AS degree
                    WHERE degree > 20
                    RETURN n.id AS file_id, degree
                    """,
                    file_id=f"{repo_id}_{change.file_path}",
                )
                smell_records = await smell_result.data()
                for rec in smell_records:
                    impact_report["smells"].append({
                        "type": "High Coupling",
                        "severity": "high",
                        "message": f"File {change.file_path} has {rec['degree']} dependencies. Changes here are risky.",
                    })

    except Exception as exc:
        logger.error("diff_analysis_failed", repo_id=repo_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to analyze diff impact")

    # Clean up sets for JSON serialization
    impact_report["affected_files"] = list(filter(bool, impact_report["affected_files"]))
    
    # Deduplicate symbols
    seen_symbols = set()
    unique_symbols = []
    for sym in impact_report["affected_symbols"]:
        key = f"{sym['file']}::{sym['symbol']}"
        if key not in seen_symbols:
            seen_symbols.add(key)
            unique_symbols.append(sym)
            
    impact_report["affected_symbols"] = unique_symbols

    logger.info(
        "diff_analysis_complete",
        repo_id=repo_id,
        affected_count=len(unique_symbols),
        smell_count=len(impact_report["smells"]),
    )
    
    return {"status": "success", "report": impact_report}
