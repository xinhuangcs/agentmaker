"""Unified SQLite connection factory plus open-time schema self-check primitives.

Every store in the framework (retrieval backend, bookkeeping, memory, kv, RAG source-of-truth,
sessions, checkpoints, trace exporter) opens its connection through open_sqlite, which centrally
sets concurrency-friendly PRAGMAs. This avoids per-call bare-connect drift and the random
"database is locked" errors that arise when multiple connections write the same database.

Open-time schema self-check (structural contract): the framework does not keep a version-number
table. Multiple tables share one database file, so the per-file PRAGMA user_version cannot record
separate versions for tables that evolve independently within the same file; and since the Scope
dimensions are fixed and tables are mostly append-only or rebuildable, a version number plus a
migration engine would be over-engineering. Instead we treat "structure is the contract": at open
time we compare the actual schema against what the code expects via PRAGMA, and handle it by the
table's role. Source-of-truth tables that can safely gain a column are auto-ALTERed
(`ensure_columns`); primary-key or unique-constraint drift raises loudly
(`require_primary_key` / `require_unique_columns`, never a silent cross-scope data leak); virtual
tables whose creation parameters are locked in raise on change with a rebuild hint
(`require_ddl_contains`); derived tables that have drifted can be dropped and rebuilt (the caller
decides using primitives like `primary_key_columns`).

This module performs pure SQLite introspection only and does not know about Scope (dependency
direction: core <- retrieval): the expected scope columns are passed in by the caller.
"""

import sqlite3
from typing import Dict, List, Optional, Set, Type


def open_sqlite(db_path: str = ":memory:") -> sqlite3.Connection:
    """Open a SQLite connection with concurrency-friendly PRAGMAs set.

    - check_same_thread=False: async paths reuse the connection across threads via to_thread in a
      thread pool, so each store adds its own locking.
    - WAL (write-ahead logging: writes go to a side journal first, so reads are not blocked by
      writes) is only set for file databases, which makes concurrent multi-connection read/write on
      the same database friendly. The default rollback-journal mode holds an exclusive write
      transaction over the whole database and makes reads and writes mutually exclusive, so multiple
      connections on the same database easily hit lock contention.
    - busy_timeout=5000: on lock contention, wait up to 5 seconds and retry rather than immediately
      raising "database is locked".
    - synchronous=NORMAL: the standard WAL pairing (safety/performance balance).

    An in-memory database (:memory:) has no use for WAL, so it is skipped.

    Args:
        db_path: Database file path; ":memory:" for an in-memory database.

    Returns:
        sqlite3.Connection: A connection with PRAGMAs configured.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    if db_path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# Open-time schema self-check primitives (pure SQLite introspection, unaware of Scope; the expected
# columns are passed in by the caller).
# Note: table names come from internal framework constants, not external input, so building the name
# into a PRAGMA f-string is safe (PRAGMA does not support ? binding for table names).

def column_names(conn: sqlite3.Connection, table: str) -> Set[str]:
    """Return the set of column names for a table (PRAGMA table_info); empty set if the table does not exist."""
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def primary_key_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    """Return the set of primary-key columns for a table (PRAGMA table_info column 6 pk flag > 0 marks a pk column); empty set if the table is missing or has no primary key."""
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})") if r[5]}


def unique_column_sets(conn: sqlite3.Connection, table: str) -> List[Set[str]]:
    """Return the list of column sets covered by each UNIQUE index on the table (including primary key, UNIQUE constraints, and CREATE UNIQUE INDEX); empty list if the table does not exist.

    Used to verify that a "unique across scope" constraint was not swallowed by an old database's
    old definition (constraint drift lets upserts across a new dimension overwrite each other and
    silently leak data across scopes).
    """
    out: List[Set[str]] = []
    for idx in conn.execute(f"PRAGMA index_list({table})"):
        if idx[2]:                                   # idx[2] = unique flag
            out.append({r[2] for r in conn.execute(f"PRAGMA index_info({idx[1]})")})
    return out


def table_ddl(conn: sqlite3.Connection, table: str) -> Optional[str]:
    """Return the raw CREATE DDL text (sqlite_master.sql) of a table or virtual table; None if it does not exist. Used to compare locked-in vec0/fts5 parameters."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()
    return row[0] if row else None


def ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    """Auto-add columns: for each missing business column, ALTER ADD it (columns maps column name -> SQL type declaration). Idempotent and does not commit (the caller does).

    Only for business columns that can be safely added (such as updated_at / superseded_by): adding
    a column gives existing rows a NULL default and is harmless.

    Scope columns must never be added this way: existing rows would get NULL for the new scope
    column, breaking the empty-string exact-match semantics and making old rows permanently
    unfindable. That is a persistence-contract change and should raise loudly via
    require_primary_key.
    """
    existing = column_names(conn, table)
    for col, decl in columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def require_columns(conn: sqlite3.Connection, table: str, expected: Set[str], *,
                    error_cls: Type[Exception]) -> None:
    """Verify the table contains all expected columns, raising error_cls if any are missing.

    Used by append-only tables without a primary key (such as session_messages): when an old
    database is missing new scope-dimension columns, INSERT or exact-match would raise the obscure
    "no such column"; checking at open time turns that into a clear error with migration guidance.
    Scope columns cannot be auto-added (NULL on old rows breaks empty-string exact-match semantics),
    so a missing column raises rather than calling ensure_columns.
    """
    missing = expected - column_names(conn, table)
    if missing:
        raise error_cls(
            f"Table {table} is missing columns {sorted(missing)}: detected an old database with a "
            "mismatched schema (most likely the set of Scope dimensions changed). "
            "The Scope dimensions are this table's persistence contract; a missing column makes old "
            "rows permanently unfindable under empty-string semantics. Rebuild the table and reimport "
            "the data, or use a new database file.")


def require_primary_key(conn: sqlite3.Connection, table: str, expected_pk: Set[str], *,
                        error_cls: Type[Exception]) -> None:
    """Verify the table's primary-key column set == expected_pk, raising error_cls (with migration guidance) otherwise. A missing table (empty set) also triggers this, but creation happens first, so it will not occur.

    Used by source-of-truth tables: the primary key must include all scope dimensions so rows do not
    overwrite each other across scopes. If an old database has primary-key drift (for example an
    early single-column primary key, or an old database whose primary key lacks a newly added scope
    dimension), a composite primary key will not migrate automatically and the same id would leak
    data across scopes, so this raises loudly rather than silently. expected_pk is assembled by the
    caller (business key + scope_column_names()); this module does not know about Scope.
    """
    actual = primary_key_columns(conn, table)
    if actual != expected_pk:
        raise error_cls(
            f"Table {table} has primary key {sorted(actual)}, expected {sorted(expected_pk)}: detected "
            "an old database with a mismatched schema. "
            "The set of Scope dimensions is this table's persistence contract; a primary key that does "
            "not include all scope dimensions leaks data across scopes, and a composite primary key "
            "cannot migrate automatically. Rebuild the table and reimport the data, or use a new "
            "database file.")


def require_unique_columns(conn: sqlite3.Connection, table: str, expected: Set[str], *,
                           error_cls: Type[Exception]) -> None:
    """Verify a UNIQUE index exists on the table whose column set == expected, raising error_cls otherwise.

    Used by kv (UNIQUE(scope,key)) and checkpoints (UNIQUE INDEX(scope)): if an old database has
    unique-constraint drift (for example not including a newly added scope dimension), upserts across
    the new dimension would overwrite each other and silently leak data across scopes, so this raises
    loudly at open time.
    """
    if expected not in unique_column_sets(conn, table):
        raise error_cls(
            f"Table {table} is missing a UNIQUE constraint covering {sorted(expected)} (existing unique "
            f"constraints: {[sorted(s) for s in unique_column_sets(conn, table)]}): detected an old "
            "database with a mismatched schema. "
            "The Scope dimensions are the persistence contract; a unique constraint that does not include "
            "all scope dimensions makes upserts across scopes overwrite each other and leak data. Rebuild "
            "the table and reimport the data, or use a new database file.")


def require_ddl_contains(conn: sqlite3.Connection, table: str, fragments: List[str], *,
                         error_cls: Type[Exception], hint: str = "") -> None:
    """Verify the CREATE DDL of a table or virtual table contains all fragments, raising error_cls (with hint) otherwise.

    Used for locked-in vec0/fts5 parameters: virtual-table parameters (dimension float[N], metadata
    columns, tokenize, scope columns) are locked in at creation, and IF NOT EXISTS silently reuses
    the old parameters and cannot be ALTERed. So we compare the existing DDL against the parameter
    fragments expected this time and raise with a rebuild hint if any are missing (rather than
    silently running with the wrong parameters). A nonexistent table (DDL is None) is skipped: the
    correct DDL will be created this run.
    """
    ddl = table_ddl(conn, table)
    if ddl is None:
        return
    missing = [f for f in fragments if f not in ddl]
    if missing:
        raise error_cls(
            f"The creation parameters of virtual table {table} do not match the current expectation "
            f"(missing fragments {missing}): virtual-table parameters are locked in at creation and "
            f"cannot be ALTERed. {hint}")
