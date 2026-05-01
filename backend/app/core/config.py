from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = "change_me"
    log_level: str = "INFO"

    # PostgreSQL
    postgres_user: str = "ckai"
    postgres_password: str = "ckai_secret"
    postgres_db: str = "codebase_knowledge"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ckai_secret"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"

    # GitHub
    github_token: str = ""
    github_webhook_secret: str = ""

    # ── Adaptive Hard Limits (Phase 1 – Step 1) ─────────────────────────────
    # These are ABSOLUTE CEILINGS. Adaptive logic may flex within these bounds
    # but may NEVER exceed them regardless of query complexity.
    max_nodes_absolute: int = 50
    max_tokens_absolute: int = 4000
    max_retries: int = 3
    max_traversal_depth: int = 2   # Expanded to depth>2 only in Phase 6
    max_top_k: int = 20            # Hard ceiling on dynamic top-K

    # Cache Budget
    cache_budget_per_repo_mb: int = 50
    cache_lru_max_items: int = 1000

    # Pipeline Timeout Budgets (ms)
    timeout_vector_ms: int = 100
    timeout_graph_ms: int = 300
    timeout_llm_ms: int = 800
    timeout_buffer_ms: int = 200

    # Scoring Decay
    usage_half_life_days: int = 10   # Within 7–14 day range

    # Confidence Thresholds
    confidence_warn_threshold: float = 0.5
    confidence_refine_threshold: float = 0.3
    confidence_routing_min: float = 0.85   # Confidence-Based Routing safety floor
    cold_query_frequency_min: int = 3      # Skip cache if seen < N times
    cold_query_similarity_min: float = 0.7

    # Memory Boosting Cap
    memory_boost_max_pct: float = 0.15    # ≤ 15% of total score

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
