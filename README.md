# Codebase Knowledge AI

**AI-powered developer intelligence system** that converts a codebase into a structured, searchable knowledge graph with semantic understanding.

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend — Phase 4)

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY and GitHub tokens
```

### 2. Start Infrastructure
```bash
docker-compose up -d postgres neo4j redis
```

### 3. Run the Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4. Start the Celery Worker
```bash
cd backend
celery -A app.workers.celery_app worker --loglevel=info -Q high_priority,default,low_priority
```

### 5. Full Stack (Docker)
```bash
docker-compose up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest/repository` | Submit a repository for ingestion |
| `GET`  | `/api/v1/ingest/repository/{id}/status` | Get coverage & ingestion status |
| `POST` | `/api/v1/ingest/webhook/github` | GitHub push webhook |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/ready` | Readiness probe (checks all deps) |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🏗 Architecture

```
Repository → Webhook / API
      ↓
  [Celery Worker]
      ↓
  AST Parser (Tree-sitter)
      ↓
  ┌─────────────────────────────┐
  │  Neo4j Graph Builder        │  ← CALLS, EXTENDS, DEFINED_IN edges
  │  FAISS Vector Store         │  ← function/class/intent embeddings
  │  PostgreSQL                 │  ← metadata, ingestion_status, coverage
  └─────────────────────────────┘
      ↓
  Query Engine (Phase 3)
      ↓
  React UI (Phase 4)
```

---

## 🔬 Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Data Ingestion, AST Parsing, Neo4j Graph |
| **Phase 2** | ✅ Complete | Chunking, Embeddings, Vector Store |
| **Phase 3** | 🔄 Next | Query Engine (RAG + Graph), Intent Classification |
| **Phase 4** | ⏳ Planned | React UI, Graph Visualization |
| **Phase 5** | ⏳ Planned | Observability, Pinecone upgrade, Benchmarks |
| **Phase 6** | ⏳ Future | Multi-hop reasoning, Diff Graph, Code Smell Detection |

---

## ⚙️ Configuration Reference

All limits are configurable via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_NODES_ABSOLUTE` | `50` | Hard ceiling on graph node retrieval |
| `MAX_TOKENS_ABSOLUTE` | `4000` | Hard token ceiling for LLM context |
| `MAX_TRAVERSAL_DEPTH` | `2` | Max Neo4j traversal hops |
| `MAX_TOP_K` | `20` | Hard ceiling on vector search results |
| `CACHE_BUDGET_PER_REPO_MB` | `50` | Hot cache budget per repo |
| `USAGE_HALF_LIFE_DAYS` | `10` | Time decay for usage-based scoring |
| `VECTOR_STORE` | `faiss` | `faiss` (MVP) or `pinecone` (production) |

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## 📐 Key Design Decisions

- **Eventual Consistency**: `ingestion_status` tracks `pending → partial → complete | failed`
- **Incremental Embedding**: Only re-embeds changed chunks (xxhash comparison)
- **Cold Start**: Entry points (`main.py`, `index.ts`, API routes) indexed first
- **Hard Limits**: All traversal/node/token ceilings are absolute — no exceptions
- **Low Importance**: Helper nodes are never deleted — just flagged for UI filtering
- **Time Decay**: `usage_weight *= exp(-0.693 * days / half_life)` prevents popular code from permanently dominating
