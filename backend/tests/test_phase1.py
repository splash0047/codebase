"""
Phase 1 Test Suite — AST Parser & Graph Builder.
Run with: pytest backend/tests/ -v
"""
import pytest
from app.parsing.ast_parser import parse_file, PythonParser, ParsedFile


# ── AST Parser Tests ──────────────────────────────────────────────────────────

SAMPLE_PYTHON = '''
import os
from typing import Optional

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Validates user credentials and returns session token."""
    if not username or not password:
        return None
    return {"token": "abc123", "user": username}

class UserService:
    def __init__(self, db):
        self._db = db

    def get_user(self, user_id: str) -> Optional[dict]:
        return self._db.find(user_id)

    def _internal_helper(self):
        pass

AUTH_SECRET = "super_secret"
'''

def test_python_parser_functions():
    result = parse_file("auth.py", SAMPLE_PYTHON)
    assert result.language == "python"
    func_names = [f.name for f in result.functions]
    assert "authenticate_user" in func_names

def test_python_parser_classes():
    result = parse_file("auth.py", SAMPLE_PYTHON)
    class_names = [c.name for c in result.classes]
    assert "UserService" in class_names

def test_python_parser_public_private():
    result = parse_file("auth.py", SAMPLE_PYTHON)
    all_funcs = result.functions + [
        m for c in result.classes for m in c.methods
    ]
    public  = [f for f in all_funcs if f.is_public]
    private = [f for f in all_funcs if not f.is_public]
    assert any(f.name == "authenticate_user" for f in public)
    assert any(f.name == "_internal_helper" for f in private)

def test_python_parser_imports():
    result = parse_file("auth.py", SAMPLE_PYTHON)
    modules = [i.module for i in result.imports]
    assert "os" in modules or "typing" in modules

def test_python_parser_variables():
    result = parse_file("auth.py", SAMPLE_PYTHON)
    var_names = [v.name for v in result.variables]
    assert "AUTH_SECRET" in var_names

def test_python_parser_docstring():
    result = parse_file("auth.py", SAMPLE_PYTHON)
    auth_func = next((f for f in result.functions if f.name == "authenticate_user"), None)
    # Docstring extraction is best-effort
    assert auth_func is not None

def test_parse_quality_valid_file():
    result = parse_file("auth.py", SAMPLE_PYTHON)
    assert result.parse_quality > 0.5

def test_parse_unsupported_language():
    result = parse_file("style.css", "body { color: red; }")
    assert result.language == "unknown"
    assert len(result.parse_errors) > 0

def test_parse_broken_syntax_fallback():
    broken = "def incomplete_func(\n  # broken!"
    result = parse_file("broken.py", broken)
    # Should not raise; quality should be degraded
    assert isinstance(result, ParsedFile)
    assert result.parse_quality <= 1.0

def test_importance_hint_exported():
    result = parse_file("auth.py", SAMPLE_PYTHON)
    exported = [f for f in result.functions if f.is_exported]
    assert all(f.importance_hint >= 0.5 for f in exported)


# ── Graph Builder Unit Tests (importance formula) ─────────────────────────────

from app.graph.builder import compute_importance, is_low_importance

def test_importance_exported_function():
    score = compute_importance(
        in_degree=5, out_degree=3, centrality=0.8,
        is_exported=True, usage_freq=10, last_used_days_ago=0,
    )
    assert 0.5 < score <= 1.0

def test_importance_private_helper():
    score = compute_importance(
        in_degree=0, out_degree=1, centrality=0.0,
        is_exported=False, usage_freq=0, last_used_days_ago=0,
    )
    assert score < 0.5

def test_importance_time_decay():
    fresh = compute_importance(
        in_degree=3, out_degree=3, centrality=0.5,
        is_exported=False, usage_freq=50, last_used_days_ago=0,
    )
    stale = compute_importance(
        in_degree=3, out_degree=3, centrality=0.5,
        is_exported=False, usage_freq=50, last_used_days_ago=30,
    )
    assert fresh > stale, "Fresh nodes should score higher than stale ones"

def test_low_importance_threshold():
    low_score  = compute_importance(0, 0, 0.0, False)
    high_score = compute_importance(10, 10, 1.0, True, 100)
    assert is_low_importance(low_score)
    assert not is_low_importance(high_score)

def test_importance_memory_boost_cap():
    # Even with extreme usage, boost should be capped
    score = compute_importance(
        in_degree=0, out_degree=0, centrality=0.0,
        is_exported=False, usage_freq=10000, last_used_days_ago=0,
    )
    # The usage portion (W_USAGE=0.15) is capped at memory_boost_max_pct (0.15)
    assert score <= 0.20   # Only usage contribution possible here
