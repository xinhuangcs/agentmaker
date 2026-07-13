"""Hermetic tests for agentmaker's public API surface.

Locks the top-level exports: PEP 562 lazy exports (import agentmaker doesn't eagerly pull heavy submodules, both
import styles work, _LAZY matches __all__, the whole table resolves), runtime / retrieval-backend symbols present, py.typed in place.
"""

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import get_args, get_type_hints

import agentmaker
from pydantic import TypeAdapter


def test_all_and_lazy_map_agree():
    """_LAZY (name -> source module) matches __all__ (the runtime contract) exactly -- a dual-list drift guard (an import-time assert already checks it; this backs it up)."""
    assert set(agentmaker._LAZY) == set(agentmaker.__all__)
    assert sorted(agentmaker.__dir__()) == sorted(agentmaker.__all__)


def test_type_checking_exports_cover_public_api():
    """Every lazy runtime export also has a static import for IDEs and downstream type checkers."""
    tree = ast.parse(Path(agentmaker.__file__).read_text(encoding="utf-8"))
    typed = set()
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom):
                    typed.update(alias.asname or alias.name for alias in child.names)
    assert typed == set(agentmaker.__all__)


def test_tool_permissions_annotations_support_runtime_reflection():
    assert get_type_hints(agentmaker.ToolPermissions)["prompts"] is object
    assert TypeAdapter(agentmaker.ToolPermissions).json_schema()["type"] == "object"


def test_critical_public_callables_declare_return_types():
    """Async, streaming, and run-context entry points expose return annotations."""
    from agentmaker.agents.agent import Agent
    from agentmaker.core.aio import iter_sync, run_sync

    functions = (
        agentmaker.current_scope,
        agentmaker.Harness.astream_llm,
        agentmaker.Harness.areduce,
        Agent.astream_run,
        Agent.stream_run,
        run_sync,
        iter_sync,
    )
    assert all(inspect.signature(function).return_annotation is not inspect.Signature.empty
               for function in functions)


def test_message_content_excludes_absent_values():
    """Conversation Message content remains text or a multimodal part list."""
    from agentmaker.core.multimodal import MessageContent

    assert type(None) not in get_args(MessageContent)


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
