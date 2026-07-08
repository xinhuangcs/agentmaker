"""Regression for agentmaker's public API surface (hermetic: no key / no network).

Locks the top-level exports: PEP 562 lazy exports (import agentmaker doesn't eagerly pull heavy submodules, both
import styles work, _LAZY matches __all__, the whole table resolves), runtime / retrieval-backend symbols present, py.typed in place.
Note: tests/ is gitignored, so new files need `git add -f`.
"""

import os
import subprocess
import sys

import agentmaker


def test_all_and_lazy_map_agree():
    """_LAZY (name -> source module) matches __all__ (the runtime contract) exactly -- a dual-list drift guard (an import-time assert already checks it; this backs it up)."""
    assert set(agentmaker._LAZY) == set(agentmaker.__all__)
    assert sorted(agentmaker.__dir__()) == sorted(agentmaker.__all__)


def test_every_public_symbol_resolves():
    """getattr over every name in __all__ raises no AttributeError (the whole lazy table resolves to real symbols)."""
    for name in agentmaker.__all__:
        assert getattr(agentmaker, name) is not None, name


def test_new_runtime_and_tool_exports_present():
    """current_scope/current_step/governed_chat + ToolRetriever/ToolSearchTool are part of the top-level public surface."""
    from agentmaker import (ToolRetriever, ToolSearchTool, current_run_id, current_scope,
                        current_step, governed_chat)
    for sym in ("current_run_id", "current_scope", "current_step", "governed_chat",
                "ToolRetriever", "ToolSearchTool"):
        assert sym in agentmaker.__all__, sym
    assert all(callable(f) for f in (current_run_id, current_scope, current_step, governed_chat))
    assert isinstance(ToolRetriever, type) and isinstance(ToolSearchTool, type)


def test_retrieval_backend_author_symbols():
    """Retrieval-backend author-extension symbols are importable at the agentmaker.retrieval subpackage level (not promoted to top level)."""
    from agentmaker.retrieval import (RetrievalError, SqliteHybridRetriever, require_explicit_scope,
                                  require_valid_top_k, scope_exact_where, scope_is_empty, scope_where)
    import agentmaker.retrieval as r
    for sym in ("SqliteHybridRetriever", "require_valid_top_k", "require_explicit_scope",
                "scope_is_empty", "RetrievalError", "scope_where", "scope_exact_where"):
        assert sym in r.__all__, sym
    assert isinstance(SqliteHybridRetriever, type) and issubclass(RetrievalError, Exception)
    assert all(callable(f) for f in (scope_where, scope_exact_where, require_valid_top_k,
                                     require_explicit_scope, scope_is_empty))


def test_deep_path_import_still_works_and_is_same_object():
    """Laziness only adds top-level aliases without moving sources: deep-path imports still work and are the same object as the top-level one."""
    from agentmaker import current_run_id as top_rid
    from agentmaker import current_scope as top
    from agentmaker.runtime import current_scope as rt
    from agentmaker.runtime.execution.run_context import current_scope as deep
    from agentmaker.runtime.observability import current_run_id as observ_rid   # trace side re-exports the related API
    assert top is rt is deep                                                # current_scope is one object all the way through
    assert top_rid is observ_rid                                            # current_run_id re-exported via observability is the same object too


def test_import_agentmaker_is_lazy():
    """import agentmaker doesn't eagerly load heavy submodules (rag/agents) or heavy deps (jieba/pydantic); they load only on symbol access. A subprocess isolates against pollution."""
    code = (
        "import agentmaker, sys;"
        "assert 'agentmaker.rag' not in sys.modules and 'agentmaker.agents' not in sys.modules;"
        "assert 'jieba' not in sys.modules and 'pydantic' not in sys.modules;"
        "_ = agentmaker.Agent;"
        "assert 'agentmaker.agents' in sys.modules"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_py_typed_marker_present():
    """PEP 561: the package root carries a py.typed marker (declares to downstream mypy/pyright that this package ships type annotations)."""
    assert os.path.exists(os.path.join(os.path.dirname(agentmaker.__file__), "py.typed"))


def test_unknown_attribute_raises():
    """Accessing a nonexistent symbol raises AttributeError (the PEP 562 __getattr__ fallback)."""
    import pytest
    with pytest.raises(AttributeError):
        agentmaker.NoSuchSymbol
