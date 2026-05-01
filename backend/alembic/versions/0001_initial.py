"""Initial schema — repositories, files, embeddings

Revision ID: 0001_initial
Revises: 
Create Date: 2026-05-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── repositories ────────────────────────────────────────────────────────
    op.create_table(
        'repositories',
        sa.Column('id',                 postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name',               sa.String(255),  nullable=False),
        sa.Column('full_name',          sa.String(512),  nullable=False, unique=True),
        sa.Column('provider',           sa.String(32),   nullable=False, server_default='github'),
        sa.Column('clone_url',          sa.String(1024), nullable=True),
        sa.Column('default_branch',     sa.String(255),  nullable=False, server_default='main'),
        sa.Column('latest_commit_id',   sa.String(64),   nullable=True),
        sa.Column('ingestion_status',   sa.String(32),   nullable=False, server_default='pending'),
        sa.Column('coverage_pct',       sa.Float(),      nullable=False, server_default='0.0'),
        sa.Column('total_files',        sa.Integer(),    nullable=False, server_default='0'),
        sa.Column('indexed_files',      sa.Integer(),    nullable=False, server_default='0'),
        sa.Column('extra_meta',         postgresql.JSONB(), nullable=True),
        sa.Column('created_at',         sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at',         sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_repos_status', 'repositories', ['ingestion_status'])

    # ── files ────────────────────────────────────────────────────────────────
    op.create_table(
        'files',
        sa.Column('id',               postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('repository_id',    postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False),
        sa.Column('path',             sa.String(2048), nullable=False),
        sa.Column('language',         sa.String(64),   nullable=True),
        sa.Column('size_bytes',       sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('content_hash',     sa.String(64),   nullable=True),
        sa.Column('commit_id',        sa.String(64),   nullable=True),
        sa.Column('ingestion_status', sa.String(32),   nullable=False, server_default='pending'),
        sa.Column('neo4j_node_id',    sa.String(255),  nullable=True),
        sa.Column('ast_summary',      postgresql.JSONB(), nullable=True),
        sa.Column('importance_score', sa.Float(),      nullable=False, server_default='0.0'),
        sa.Column('created_at',       sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at',       sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('repository_id', 'path', name='uq_file_repo_path'),
    )
    op.create_index('ix_files_repo',    'files', ['repository_id'])
    op.create_index('ix_files_status',  'files', ['ingestion_status'])

    # ── embeddings ────────────────────────────────────────────────────────────
    op.create_table(
        'embeddings',
        sa.Column('id',                  postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('file_id',             postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('files.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_type',          sa.String(32),   nullable=False),
        sa.Column('chunk_index',         sa.Integer(),    nullable=False, server_default='0'),
        sa.Column('symbol_name',         sa.String(512),  nullable=True),
        sa.Column('start_line',          sa.Integer(),    nullable=True),
        sa.Column('end_line',            sa.Integer(),    nullable=True),
        sa.Column('commit_id',           sa.String(64),   nullable=True),
        sa.Column('chunk_hash',          sa.String(64),   nullable=True),
        sa.Column('vector_id',           sa.String(255),  nullable=True),
        sa.Column('pinecone_namespace',  sa.String(255),  nullable=True),
        sa.Column('chunk_text',          sa.Text(),       nullable=True),
        sa.Column('intent_meta',         postgresql.JSONB(), nullable=True),
        sa.Column('created_at',          sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_embeddings_file',   'embeddings', ['file_id'])
    op.create_index('ix_embeddings_hash',   'embeddings', ['chunk_hash'])
    op.create_index('ix_embeddings_vector', 'embeddings', ['vector_id'])


def downgrade() -> None:
    op.drop_table('embeddings')
    op.drop_table('files')
    op.drop_table('repositories')
