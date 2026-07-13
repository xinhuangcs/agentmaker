"""agentmaker.retrieval.scope_sql: maps Scope to the columns and filter fragments of a relational / SQL store.

The Scope in `scope.py` is just the general "data ownership" concept; this module is its landing on SQLite and other
relational backends: dimension -> column name, Scope -> stored values / WHERE fragment. It is shared by retrieval's two
stores as well as the SQLite storage across memory / rag / sessions / execution, keeping isolation behavior consistent.
Switching to a non-SQL backend (Qdrant / pgvector payload filter, etc.) = writing a separate mapping, leaving the
concept and guardrails in `scope.py` untouched.
"""

from dataclasses import fields
from typing import List, Tuple

from .scope import Scope

# scope dimension -> underlying SQL column name (with an sc_ prefix to avoid clashing with SQL keywords / reserved words)
_SCOPE_COLS = {
    "base": "base",
    "user": "sc_user",
    "agent": "sc_agent",
    "session": "sc_session",
    "app": "sc_app",
}

# Dimension-drift guardrail: the column mapping must cover every field of Scope, otherwise a dimension has no SQL column
# to store into (silently leaking data across scopes). Scope is the source of truth for the dimensions.
assert set(_SCOPE_COLS) == {f.name for f in fields(Scope)}, "scope_sql._SCOPE_COLS is out of sync with Scope fields"


def scope_column_names() -> List[str]:
    """The list of scope's underlying column names (used when creating tables / on INSERT)."""
    return list(_SCOPE_COLS.values())


def scope_column_for(dimension: str) -> str:
    """Map a scope dimension name (base/user/agent/session/app) to its underlying SQL column name; unknown dimensions raise ValueError.

    Used by queries that "enumerate / group by a dimension" to get the column name. The validation doubles as a SQL
    injection guardrail (the return value can only be one of the fixed _SCOPE_COLS column names, never a caller string
    passed through verbatim).

    Args:
        dimension: A Scope dimension name.

    Returns:
        str: The corresponding SQL column name.
    """
    try:
        return _SCOPE_COLS[dimension]
    except KeyError:
        raise ValueError(f"unknown scope dimension {dimension!r}, must be one of {list(_SCOPE_COLS)}") from None


def scope_store_values(scope: Scope) -> List[str]:
    """The list of values for storage; None -> empty string, to make column filtering easy.

    Values and column names both follow _SCOPE_COLS, keeping inserts aligned with the persistence schema.
    Changing the dimension set requires a coordinated database migration.
    """
    return [(getattr(scope, name) or "") for name in _SCOPE_COLS]


def scope_from_store_values(values) -> Scope:
    """Restore the exact stored ownership footprint from scope-column values."""
    return Scope(**{
        name: value or None
        for name, value in zip(_SCOPE_COLS, values)
    })


def scope_where(scope: Scope) -> Tuple[str, List[str]]:
    """The AND fragment and params for queries (B semantics: only non-empty dimensions), for appending after an existing WHERE.

    Returns:
        (sql, params): sql looks like " AND base = ? AND sc_user = ?" (with a leading space, ready to concatenate);
        returns ("", []) when there is no non-empty dimension.
    """
    parts, params = [], []
    for field_name, col in _SCOPE_COLS.items():
        value = getattr(scope, field_name)
        if value:
            parts.append(f"AND {col} = ?")
            params.append(value)
    return ((" " + " ".join(parts)) if parts else ""), params


def scope_where_clause(scope: Scope) -> Tuple[str, List[str]]:
    """The full WHERE clause and params (B semantics), for queries with no other filter conditions.

    Returns:
        (sql, params): sql looks like " WHERE base = ? AND sc_user = ?" (with a leading space);
        returns ("", []) when there is no non-empty dimension.
    """
    where, params = scope_where(scope)
    return ((" WHERE" + where[len(" AND"):]) if where else ""), params


def scope_exact_where(scope: Scope) -> Tuple[str, List[str]]:
    """The AND fragment and params that exactly match all dimensions (including empty ones, compared as empty string), for appending after an existing WHERE.

    Unlike scope_where (B semantics, filters only non-empty dimensions), this brings all five dimensions into the
    equality match, comparing empty dimensions against their stored value "" (see scope_store_values), so it only hits
    rows whose scope footprint matches exactly. The pre-write upsert (delete-old-then-insert by (id, scope)) uses this
    to avoid wrongly deleting sibling rows with the same id but a different scope.

    Returns:
        (sql, params): sql looks like " AND base = ? AND sc_user = ? AND ... (all five dimensions)", with params
        aligned to the column order.
    """
    sql = "".join(f" AND {col} = ?" for col in _SCOPE_COLS.values())
    return sql, scope_store_values(scope)


def scope_exact_where_clause(scope: Scope) -> Tuple[str, List[str]]:
    """The full WHERE clause and params that exactly match all dimensions (including empty ones, compared as empty string), for queries with no other filter conditions.

    This is the "own WHERE" version of scope_exact_where (as scope_where_clause is to scope_where): it brings all five
    dimensions into the equality match, so it only hits rows whose scope footprint matches exactly. Storage that does
    "point-access by exact scope", such as sessions / checkpoint, uses this: an empty scope (Scope()) only hits the
    all-empty bucket, not the whole table (avoiding wrongly reading / deleting all sessions when scope is not passed).
    It always carries a WHERE (all five dimensions, matching against empty strings even when no dimension is non-empty),
    and unlike B semantics does not degrade to an empty string.

    Returns:
        (sql, params): sql looks like " WHERE base = ? AND sc_user = ? AND ... (all five dimensions)", with params
        aligned to the column order.
    """
    where, params = scope_exact_where(scope)
    return " WHERE" + where[len(" AND"):], params
