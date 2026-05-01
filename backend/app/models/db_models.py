"""
SQLAlchemy ORM models — Repository, File, Embedding.
Tracks ingestion_status for eventual consistency (Phase 1 – Step 2).
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.postgres import Base


# ── Enums ─────────────────────────────────────────────────────────────────────

class IngestionStatus(str, enum.Enum):
    """
    Eventual consistency model from the plan.
    pending → partial → complete | failed
    """
    PENDING  = "pending"
    PARTIAL  = "partial"
    COMPLETE = "complete"
    FAILED   = "failed"


class RepoProvider(str, enum.Enum):
    GITHUB = "github"
    GITLAB = "gitlab"
    LOCAL  = "local"


# ── Repository ────────────────────────────────────────────────────────────────

class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    provider: Mapped[RepoProvider] = mapped_column(
        Enum(RepoProvider), default=RepoProvider.GITHUB
    )
    clone_url: Mapped[str | None] = mapped_column(String(1024))
    default_branch: Mapped[str] = mapped_column(String(255), default="main")

    # Lightweight version tagging (replaces full snapshots)
    latest_commit_id: Mapped[str | None] = mapped_column(String(64))

    ingestion_status: Mapped[IngestionStatus] = mapped_column(
        Enum(IngestionStatus), default=IngestionStatus.PENDING, index=True
    )
    # How much of the repo is indexed (used for Coverage UI: "Coverage: 23% ⚠️")
    coverage_pct: Mapped[float] = mapped_column(Float, default=0.0)
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    indexed_files: Mapped[int] = mapped_column(Integer, default=0)

    extra_meta: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    files: Mapped[list["File"]] = relationship(
        "File", back_populates="repository", cascade="all, delete-orphan"
    )


# ── File ──────────────────────────────────────────────────────────────────────

class File(Base):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("repository_id", "path", name="uq_file_repo_path"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE"), index=True
    )
    path: Mapped[str] = mapped_column(String(2048), nullable=False)
    language: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)

    # Content hash for incremental re-embedding (xxhash)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    # Commit when this file was last indexed
    commit_id: Mapped[str | None] = mapped_column(String(64))

    ingestion_status: Mapped[IngestionStatus] = mapped_column(
        Enum(IngestionStatus), default=IngestionStatus.PENDING, index=True
    )
    # Neo4j node ID for this file (cross-store reference)
    neo4j_node_id: Mapped[str | None] = mapped_column(String(255))

    # AST parse results stored as JSONB for quick access without graph traversal
    ast_summary: Mapped[dict | None] = mapped_column(JSONB)

    # Graph importance score for this file (deterministic formula from plan)
    importance_score: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    repository: Mapped["Repository"] = relationship("Repository", back_populates="files")
    embeddings: Mapped[list["Embedding"]] = relationship(
        "Embedding", back_populates="file", cascade="all, delete-orphan"
    )


# ── Embedding ─────────────────────────────────────────────────────────────────

class ChunkType(str, enum.Enum):
    FUNCTION  = "function"
    CLASS     = "class"
    MODULE    = "module"
    INTENT    = "intent"   # Synthesized natural-language chunk
    DOCSTRING = "docstring"


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), index=True
    )
    chunk_type: Mapped[ChunkType] = mapped_column(Enum(ChunkType))
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)

    # Symbol name (function name, class name, etc.)
    symbol_name: Mapped[str | None] = mapped_column(String(512))

    # Source location
    start_line: Mapped[int | None] = mapped_column(Integer)
    end_line: Mapped[int | None] = mapped_column(Integer)
    commit_id: Mapped[str | None] = mapped_column(String(64))

    # Hash of the chunk text for incremental re-embedding
    chunk_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    # FAISS/PGVector vector ID (external reference)
    vector_id: Mapped[str | None] = mapped_column(String(255), index=True)

    # Pinecone namespace (if using Pinecone)
    pinecone_namespace: Mapped[str | None] = mapped_column(String(255))

    # Raw chunk text (stored for LLM context reconstruction)
    chunk_text: Mapped[str | None] = mapped_column(Text)

    # Rich metadata: purpose, inputs, outputs, side_effects (plan: intent layer)
    intent_meta: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    file: Mapped["File"] = relationship("File", back_populates="embeddings")
