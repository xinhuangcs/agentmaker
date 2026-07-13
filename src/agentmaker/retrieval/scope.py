"""agentmaker.retrieval.scope: the ownership label for memory / data (multi-dimensional isolation and filtering).

Tags each piece of data with labels like user / agent / session / app, and isolates reads and writes by dimension.
Each dimension occupies its own column in the underlying store, so any combination of dimensions can be filtered on precisely.

Filter semantics (B semantics, filters only non-empty dimensions): search adds a WHERE only for dimensions given a
value in the scope; dimensions left unset are unrestricted. For example, Scope(user="alice") -> returns all of alice's
memories, regardless of agent / session. All dimensions (including base) are treated equally: empty means unrestricted.
base distinguishes subsystems such as memory / rag, and by convention is passed explicitly by each upper layer (e.g.
Memory defaults to Scope(base="memory")).

This module holds only the general "ownership" concept and its guardrails; mapping Scope to the columns and WHERE
fragments of a relational / SQL store lives in `scope_sql.py`.

Scope dimensions form the persistence schema contract for every SQLite table. Each dimension occupies a column and
participates in primary-key or uniqueness constraints. Schema changes require coordinated table migration and index
rebuilds. Stores validate their scope columns and constraints when opened.
"""

from dataclasses import dataclass, fields, replace
from typing import Iterable, Optional

from ..core.exceptions import RetrievalError


@dataclass(frozen=True)
class Scope:
    """The ownership label for data. Every dimension is optional; omitting one means "do not restrict this dimension" (B semantics).

    Fields:
        base: Subsystem distinction (memory / rag, etc.); leave empty to not restrict.
        user: User identifier (the key to multi-user isolation, the minimal security boundary).
        agent: Agent identifier (in a multi-agent system each agent keeps its own records).
        session: Session identifier (= run_id, the transient context of a single conversation).
        app: Application / organization identifier (shared context).
    """
    base: Optional[str] = None
    user: Optional[str] = None
    agent: Optional[str] = None
    session: Optional[str] = None
    app: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate dimensions and normalize empty strings to the unset value."""
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"Scope.{field.name} must be a string or None, got {type(value).__name__}")
            if value == "":
                object.__setattr__(self, field.name, None)


_INHERITABLE_DIMENSIONS = ("user", "agent", "session", "app")


def canonical_scope(scope: Optional[Scope], base: str, action: str) -> Scope:
    """Return a scope with the subsystem base enforced.

    A per-call scope replaces the manager's other dimensions, but it must never broaden a
    memory/RAG operation across subsystem boundaries by dropping ``base``. An explicitly
    conflicting base is a configuration error rather than something to silently rewrite.

    Args:
        scope: Caller-provided scope, or None.
        base: Canonical subsystem base.
        action: Operation name used in errors.

    Returns:
        Scope: The caller scope with the canonical base filled in.

    Raises:
        RetrievalError: If scope explicitly names another base.
    """
    current = scope or Scope()
    if current.base is not None and current.base != base:
        raise RetrievalError(
            f"{action} requires scope.base={base!r}, got {current.base!r}; "
            "a manager cannot cross subsystem boundaries.")
    return replace(current, base=base)


def merge_run_scope(fixed: Scope, runtime: Optional[Scope], dimensions: Iterable[str]) -> Scope:
    """Overlay selected runtime ownership dimensions onto a fixed tool scope.

    ``base`` is deliberately not inheritable; the owning Memory/RAG manager enforces it.

    Args:
        fixed: The tool's configured fixed scope.
        runtime: The current Agent run scope, if any.
        dimensions: Selected dimensions to inherit from the run.

    Returns:
        Scope: The merged scope.

    Raises:
        ValueError: If dimensions contains an unsupported or base dimension.
    """
    dims = tuple(dimensions)
    bad = [d for d in dims if d not in _INHERITABLE_DIMENSIONS]
    if bad:
        raise ValueError(
            f"scope inheritance dimensions must be chosen from {_INHERITABLE_DIMENSIONS}, got {bad}")
    if runtime is None:
        return fixed
    updates = {}
    for dimension in dims:
        fixed_value = getattr(fixed, dimension)
        runtime_value = getattr(runtime, dimension)
        if fixed_value is not None and runtime_value is not None and fixed_value != runtime_value:
            raise RetrievalError(
                f"run scope conflicts with the tool's fixed {dimension}: "
                f"{runtime_value!r} != {fixed_value!r}")
        if fixed_value is None and runtime_value is not None:
            updates[dimension] = runtime_value
    return replace(fixed, **updates)


def scope_is_empty(scope: Scope) -> bool:
    """True if no dimension has a value (i.e. no dimension is restricted, so the operation applies to the whole database). Scope fields are the source of truth for the dimensions."""
    return all(getattr(scope, f.name) is None for f in fields(scope))


def require_explicit_scope(scope: Scope, all_scopes: bool, action: str) -> None:
    """Guardrail for destructive / global operations: reject when scope is fully empty and all_scopes=True was not passed explicitly, to avoid accidentally acting on the whole database.

    The mechanism provides the guardrail, the caller provides the rule: by default it forbids slip-ups like a bare
    Scope() that deletes / searches across the whole database, and if global scope is truly needed the caller must
    explicitly opt in. The framework's built-in memory / rag always carry a (non-empty) base and are unaffected: this
    only blocks a fully-unrestricted Scope().

    Args:
        scope: The ownership range to check.
        all_scopes: Whether the caller explicitly declares "yes, this should apply to all scopes".
        action: The operation name, used in the error message (e.g. "delete" / "search").
    """
    if not all_scopes and scope_is_empty(scope):
        raise RetrievalError(
            f"Refusing to {action} across all scopes: scope restricts no dimension. "
            f"If you really need to apply this to all scopes, pass all_scopes=True explicitly.")
