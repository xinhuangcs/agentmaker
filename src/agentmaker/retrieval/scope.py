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

The set of Scope dimensions is the persistence schema contract for every SQLite table (essential to know): each
dimension occupies a column in all persistent tables (memories / chunks / docs / kv / session_messages / checkpoints /
vector / full-text / bookkeeping) and is embedded into the primary key / UNIQUE constraint. Adding or removing a
dimension = a whole-database schema change: existing old databases must be migrated (ALTER to add columns + backfill +
rebuild constraints + rebuild virtual tables) to stay compatible, it is not just editing `scope_sql._SCOPE_COLS` in one
place (that only guarantees newly-created databases align with the code). Each store self-checks with PRAGMA on open:
if the dimensions do not match, it fails loud (source-of-truth) or auto-rebuilds (derived), turning "no such column" and
"leaking data across scopes" into clear up-front failures. See doc/retrieval/scope.md for details.
"""

from dataclasses import dataclass, fields
from typing import Optional

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


def scope_is_empty(scope: Scope) -> bool:
    """True if no dimension has a value (i.e. no dimension is restricted, so the operation applies to the whole database). Scope fields are the source of truth for the dimensions."""
    return not any(getattr(scope, f.name) for f in fields(scope))


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
