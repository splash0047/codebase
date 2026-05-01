"""
Phase 2 Test Suite — Chunker & Embedder.
Run with: pytest backend/tests/test_phase2.py -v
"""
import pytest
from app.ingestion.chunker import chunk_parsed_file, CodeChunk, MAX_CHUNK_CHARS
from app.models.db_models import ChunkType
from app.parsing.ast_parser import parse_file
from app.ingestion.embedder import EmbeddingService

# ── Sample source ─────────────────────────────────────────────────────────────

SAMPLE = '''
def login(username: str, password: str) -> dict:
    """Handles user login."""
    token = generate_token(username)
    return {"token": token}

def generate_token(username: str) -> str:
    return "tok_" + username

class AuthService:
    def __init__(self, db):
        self._db = db

    def verify(self, token: str) -> bool:
        return bool(token)
'''


# ── Chunker tests ─────────────────────────────────────────────────────────────

def test_chunk_count_reasonable():
    parsed = parse_file("auth.py", SAMPLE)
    chunks = chunk_parsed_file(parsed, SAMPLE, commit_id="abc123")
    # Expect at least one chunk per top-level function + class + methods
    assert len(chunks) >= 3

def test_function_chunks_exist():
    parsed = parse_file("auth.py", SAMPLE)
    chunks = chunk_parsed_file(parsed, SAMPLE)
    func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
    assert len(func_chunks) >= 1

def test_intent_chunks_exist():
    parsed = parse_file("auth.py", SAMPLE)
    chunks = chunk_parsed_file(parsed, SAMPLE)
    intent_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTENT]
    assert len(intent_chunks) >= 1

def test_class_chunks_exist():
    parsed = parse_file("auth.py", SAMPLE)
    chunks = chunk_parsed_file(parsed, SAMPLE)
    class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
    assert len(class_chunks) >= 1

def test_chunk_has_hash():
    parsed = parse_file("auth.py", SAMPLE)
    chunks = chunk_parsed_file(parsed, SAMPLE)
    for c in chunks:
        assert c.chunk_hash and len(c.chunk_hash) == 16

def test_chunk_commit_id_propagated():
    parsed = parse_file("auth.py", SAMPLE)
    chunks = chunk_parsed_file(parsed, SAMPLE, commit_id="deadbeef")
    for c in chunks:
        assert c.commit_id == "deadbeef"

def test_chunk_no_empty_text():
    parsed = parse_file("auth.py", SAMPLE)
    chunks = chunk_parsed_file(parsed, SAMPLE)
    for c in chunks:
        assert not c.is_empty()

def test_chunk_respects_max_size():
    large_source = "def big_func():\n    " + "x = 1\n    " * 5000
    parsed = parse_file("big.py", large_source)
    chunks = chunk_parsed_file(parsed, large_source)
    for c in chunks:
        assert len(c.text) <= MAX_CHUNK_CHARS + 20  # small buffer for truncation marker

def test_chunk_metadata_file_path():
    parsed = parse_file("app/auth.py", SAMPLE)
    chunks = chunk_parsed_file(parsed, SAMPLE)
    for c in chunks:
        assert c.file_path == "app/auth.py"

def test_chunk_hash_deterministic():
    parsed = parse_file("auth.py", SAMPLE)
    c1 = chunk_parsed_file(parsed, SAMPLE)
    c2 = chunk_parsed_file(parsed, SAMPLE)
    hashes1 = {c.chunk_hash for c in c1}
    hashes2 = {c.chunk_hash for c in c2}
    assert hashes1 == hashes2


# ── EmbeddingService incremental logic tests ──────────────────────────────────

def _make_chunk(symbol: str, text: str) -> CodeChunk:
    parsed = parse_file("test.py", f"def {symbol}(): pass")
    chunks = chunk_parsed_file(parsed, f"def {symbol}(): pass")
    # Return a fresh chunk with controlled text
    c = chunks[0] if chunks else CodeChunk(
        chunk_type=ChunkType.FUNCTION,
        text=text,
        symbol_name=symbol,
        file_path="test.py",
        start_line=1,
        end_line=2,
    )
    c.text = text
    import xxhash
    c.chunk_hash = xxhash.xxh64(text.encode()).hexdigest()
    return c

def test_embedding_service_skips_unchanged():
    chunk = _make_chunk("unchanged_func", "def unchanged_func(): pass")
    known = {f"test.py::unchanged_func::function": chunk.chunk_hash}
    svc   = EmbeddingService(known_hashes=known)
    assert not svc._needs_reembed(chunk)

def test_embedding_service_embeds_changed():
    chunk = _make_chunk("changed_func", "def changed_func(): return 42")
    known = {f"test.py::changed_func::function": "old_hash_xyz"}
    svc   = EmbeddingService(known_hashes=known)
    assert svc._needs_reembed(chunk)

def test_embedding_service_embeds_new():
    chunk = _make_chunk("new_func", "def new_func(): pass")
    svc   = EmbeddingService(known_hashes={})
    assert svc._needs_reembed(chunk)
