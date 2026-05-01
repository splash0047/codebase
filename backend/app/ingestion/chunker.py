"""
Smart Code Chunking Strategy (Phase 2 – Step 6).

Produces three chunk types from a ParsedFile:
  • function-level  — one chunk per function/method
  • class-level     — class signature + all method stubs
  • intent-level    — natural-language description for semantic search

Each chunk carries rich metadata (file path, lines, commit_id, symbol name)
so retrieval results can be precisely anchored back to source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import xxhash

from app.core.logging import get_logger
from app.models.db_models import ChunkType
from app.parsing.ast_parser import ParsedFile, ParsedFunction, ParsedClass

logger = get_logger(__name__)

# Maximum characters per chunk (rough token guard — 1 char ≈ 0.3 tokens)
MAX_CHUNK_CHARS = 6_000   # ~2 000 tokens, well inside embedding model limits


@dataclass
class CodeChunk:
    chunk_type:   ChunkType
    text:         str
    symbol_name:  str
    file_path:    str
    start_line:   int
    end_line:     int
    commit_id:    str | None = None
    chunk_hash:   str = field(init=False)
    intent_meta:  dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chunk_hash = xxhash.xxh64(self.text.encode()).hexdigest()

    def is_empty(self) -> bool:
        return len(self.text.strip()) < 10


# ── Chunk builders ─────────────────────────────────────────────────────────────

def _function_chunk(func: ParsedFunction, file_path: str, source_lines: list[str], commit_id: str | None) -> CodeChunk:
    """Extract raw source lines for a function and build a chunk."""
    # Extract source text from line numbers (1-indexed)
    start = max(0, func.start_line - 1)
    end   = min(len(source_lines), func.end_line)
    raw   = "\n".join(source_lines[start:end])

    # Trim if exceeding limit
    if len(raw) > MAX_CHUNK_CHARS:
        raw = raw[:MAX_CHUNK_CHARS] + "\n# [truncated]"
        logger.warning("chunk_truncated", symbol=func.qualified_name, file=file_path)

    # Intent metadata (used by LLM for summarization in Phase 3)
    intent = {
        "purpose":       func.docstring or "",
        "params":        func.params,
        "return_type":   func.return_type,
        "calls":         func.calls[:20],   # Cap to avoid noise
        "is_exported":   func.is_exported,
    }

    return CodeChunk(
        chunk_type=ChunkType.FUNCTION,
        text=raw,
        symbol_name=func.qualified_name,
        file_path=file_path,
        start_line=func.start_line,
        end_line=func.end_line,
        commit_id=commit_id,
        intent_meta=intent,
    )


def _class_chunk(cls: ParsedClass, file_path: str, commit_id: str | None) -> CodeChunk:
    """Build a synthetic class-level chunk: signature + method stubs."""
    bases = f"({', '.join(cls.base_classes)})" if cls.base_classes else ""
    stubs = "\n".join(
        f"    def {m.name}({', '.join(m.params)}): ..." for m in cls.methods
    )
    text = f"class {cls.name}{bases}:\n{stubs or '    pass'}"

    return CodeChunk(
        chunk_type=ChunkType.CLASS,
        text=text[:MAX_CHUNK_CHARS],
        symbol_name=cls.name,
        file_path=file_path,
        start_line=cls.start_line,
        end_line=cls.end_line,
        commit_id=commit_id,
        intent_meta={"base_classes": cls.base_classes, "method_count": len(cls.methods)},
    )


def _intent_chunk(symbol_name: str, file_path: str, start_line: int, end_line: int,
                  intent_meta: dict, commit_id: str | None) -> CodeChunk:
    """
    Natural-language intent chunk for semantic search.
    Template filled from parsed metadata — LLM summary added later in Phase 3.
    """
    parts: list[str] = [f"Symbol: {symbol_name}", f"File: {file_path}"]

    if intent_meta.get("purpose"):
        parts.append(f"Purpose: {intent_meta['purpose']}")
    if intent_meta.get("params"):
        parts.append(f"Parameters: {', '.join(intent_meta['params'])}")
    if intent_meta.get("return_type"):
        parts.append(f"Returns: {intent_meta['return_type']}")
    if intent_meta.get("calls"):
        parts.append(f"Calls: {', '.join(intent_meta['calls'][:10])}")

    text = "\n".join(parts)
    return CodeChunk(
        chunk_type=ChunkType.INTENT,
        text=text,
        symbol_name=symbol_name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        commit_id=commit_id,
        intent_meta=intent_meta,
    )


# ── Main chunker ──────────────────────────────────────────────────────────────

def chunk_parsed_file(
    parsed: ParsedFile,
    raw_source: str,
    commit_id: str | None = None,
) -> list[CodeChunk]:
    """
    Produce all chunks for a parsed file.
    Order: function → class → intent (for each symbol).
    """
    source_lines = raw_source.splitlines()
    chunks: list[CodeChunk] = []

    # ── Top-level functions
    for func in parsed.functions:
        fc = _function_chunk(func, parsed.path, source_lines, commit_id)
        if not fc.is_empty():
            chunks.append(fc)
            # Intent chunk alongside each function
            ic = _intent_chunk(
                func.qualified_name, parsed.path,
                func.start_line, func.end_line,
                fc.intent_meta, commit_id,
            )
            chunks.append(ic)

    # ── Classes (class-level chunk + per-method function chunks)
    for cls in parsed.classes:
        cc = _class_chunk(cls, parsed.path, commit_id)
        if not cc.is_empty():
            chunks.append(cc)

        for method in cls.methods:
            mc = _function_chunk(method, parsed.path, source_lines, commit_id)
            if not mc.is_empty():
                chunks.append(mc)
                ic = _intent_chunk(
                    method.qualified_name, parsed.path,
                    method.start_line, method.end_line,
                    mc.intent_meta, commit_id,
                )
                chunks.append(ic)

    logger.info(
        "file_chunked",
        file=parsed.path,
        chunk_count=len(chunks),
        functions=len(parsed.functions),
        classes=len(parsed.classes),
    )
    return chunks
