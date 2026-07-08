"""agentmaker.memory.store: the source-of-truth store for memories (the authoritative copy).

The retrieval base stores only id + body for fast search; here we store the full MemoryItem with type / importance /
timestamps, which is the authoritative copy: if an index (vector / keyword) breaks, it can be rebuilt from this.
Persisted in SQLite, accessed by id.
"""

import json
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional

from ..core.clock import now_utc
from ..core.exceptions import RetrievalError
from ..core.sqlite_util import ensure_columns, open_sqlite, require_primary_key
from ..retrieval.scope import Scope
from ..retrieval.scope_sql import scope_column_names, scope_store_values, scope_where, scope_where_clause
from .types import MemoryItem


class MemoryStore:
    """Memory source-of-truth store: full MemoryItems persisted by id in SQLite."""

    def __init__(self, db_path: str = ":memory:", *, lock: Optional[threading.RLock] = None):
        """Open the connection and create the table if needed.

        Args:
            db_path: SQLite file path; defaults to ":memory:" for self-tests only. For real use pass a file path to
                persist (pass the same file path as the vector store / keyword index when sharing a database).
            lock: a reentrant lock serializing the connection; pass the same one when sharing a connection with another
                store, otherwise a new one is created.
        """
        scope_cols = ", ".join(f"{c} TEXT" for c in scope_column_names())
        # Composite primary key (id, each scope dimension): the same id under different scopes occupies its own row
        # without overwriting the other (aligned with the index layer's "(id, scope) is what's unique"), otherwise
        # data leaks across users.
        pk_cols = ", ".join(["id", *scope_column_names()])
        self._lock = lock or threading.RLock()  # check_same_thread=False reuses the connection across threads, so one lock must serialize access to avoid corrupting the connection under concurrency
        try:
            self._db = open_sqlite(db_path)
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS memories("
                f"id TEXT, {scope_cols}, content TEXT, type TEXT, "
                f"importance REAL, created_at TEXT, updated_at TEXT, last_accessed_at TEXT, "
                f"invalid_at TEXT, superseded_by TEXT, metadata TEXT, "
                f"PRIMARY KEY ({pk_cols}))")
            # Open-time self-check (structure is the contract): CREATE TABLE IF NOT EXISTS does not change an existing
            # table's primary key. If an old database has an early single-column primary key (id), or the primary key is
            # missing a new dimension after Scope gained one, the composite primary key won't migrate automatically
            # (same id leaks data across scopes), so fail loud and prompt a rebuild.
            require_primary_key(self._db, "memories", {"id", *scope_column_names()}, error_cls=RetrievalError)
            # Add-column migration (non-scope business columns can be added safely; old rows take a harmless NULL
            # default): if an old database lacks the time-validity / usage-feedback columns, ALTER them in place.
            ensure_columns(self._db, "memories",
                           {"updated_at": "TEXT", "last_accessed_at": "TEXT", "invalid_at": "TEXT", "superseded_by": "TEXT"})
            self._db.commit()
        except sqlite3.Error as e:
            raise RetrievalError(f"failed to open / initialize the memory source-of-truth store: {e}") from e

    _COLS = "id, content, type, importance, created_at, updated_at, last_accessed_at, invalid_at, superseded_by, metadata"

    def save(self, item: MemoryItem, *, scope: Optional[Scope] = None) -> None:
        """Store a memory (overwrites if the same id + same scope already exists; same id under different scopes stays independent). metadata is stored as JSON, times as isoformat."""
        sv = scope_store_values(scope or Scope())
        cols = ", ".join(scope_column_names())
        placeholders = ", ".join("?" for _ in scope_column_names())
        opt = [item.updated_at, item.last_accessed_at, item.invalid_at]
        try:
            with self._lock:
                self._db.execute(
                    f"INSERT OR REPLACE INTO memories(id, {cols}, content, type, importance, created_at, "
                    f"updated_at, last_accessed_at, invalid_at, superseded_by, metadata) "
                    f"VALUES (?, {placeholders}, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (item.id, *sv, item.content, item.type, item.importance, item.created_at.isoformat(),
                     *[t.isoformat() if t else None for t in opt], item.superseded_by,
                     json.dumps(item.metadata, ensure_ascii=False)))
                self._db.commit()
        except sqlite3.Error as e:
            raise RetrievalError(f"failed to write memory: {e}") from e

    def replace(self, id: str, item: MemoryItem, *, scope: Optional[Scope] = None) -> None:
        """Atomically replace within a single transaction: delete the old row matching (id, scope) first, then insert item.

        Used by update / invalidate: it folds the two "delete then write" steps into one transaction (`with self._db:`
        commits on success, rolls back on exception), avoiding a crash between deleting the old and writing the new that
        loses the row. Previously the source-of-truth delete and write were two independent commits, and a crash in
        between permanently lost the memory.
        """
        sc = scope or Scope()
        del_where, del_params = scope_where(sc)
        sv = scope_store_values(sc)
        cols = ", ".join(scope_column_names())
        placeholders = ", ".join("?" for _ in scope_column_names())
        opt = [item.updated_at, item.last_accessed_at, item.invalid_at]
        try:
            with self._lock:
                with self._db:      # the connection as a context manager: multiple DMLs in the block form one transaction, committing normally and rolling back on exception
                    self._db.execute(f"DELETE FROM memories WHERE id = ?{del_where}", (id, *del_params))
                    self._db.execute(
                        f"INSERT OR REPLACE INTO memories(id, {cols}, content, type, importance, created_at, "
                        f"updated_at, last_accessed_at, invalid_at, superseded_by, metadata) "
                        f"VALUES (?, {placeholders}, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (item.id, *sv, item.content, item.type, item.importance, item.created_at.isoformat(),
                         *[t.isoformat() if t else None for t in opt], item.superseded_by,
                         json.dumps(item.metadata, ensure_ascii=False)))
        except sqlite3.Error as e:
            raise RetrievalError(f"failed to replace memory: {e}") from e

    def get(self, id: str, *, scope: Optional[Scope] = None) -> Optional[MemoryItem]:
        """Fetch the full memory by id (including soft-invalidated ones; whether to filter is up to the caller); returns None if not found.

        With a scope, the fetch is limited to that ownership (B semantics: only filter on non-empty dimensions), to
        avoid reading someone else's memory across scopes. Without a scope (None / empty Scope), fetch by id across the
        whole database (raw low-level use; the higher-level Memory always passes a scope).
        """
        where, params = scope_where(scope or Scope())
        with self._lock:
            cur = self._db.execute(
                f"SELECT {self._COLS} FROM memories WHERE id = ?{where}", (id, *params))
            row = cur.fetchone()
        return self._row_to_item(row) if row else None

    def touch(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Batch-write last_accessed_at = now (a retrieval hit means "got used", Generative Agents semantics); one commit."""
        if not ids:
            return
        now = now_utc().isoformat()
        where, params = scope_where(scope or Scope())
        with self._lock:
            try:
                self._db.executemany(
                    f"UPDATE memories SET last_accessed_at = ? WHERE id = ?{where}",
                    [(now, i, *params) for i in ids])
                self._db.commit()
            except sqlite3.Error as e:
                raise RetrievalError(f"failed to write back memory access time: {e}") from e

    def delete(self, id: str, *, scope: Optional[Scope] = None) -> None:
        """Delete a memory (within the scope-limited range, to avoid cross-ownership deletion)."""
        self.delete_many([id], scope=scope)

    def delete_many(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Batch-delete memories (within the scope-limited range); one commit."""
        if not ids:
            return
        where, params = scope_where(scope or Scope())
        with self._lock:
            self._db.executemany(f"DELETE FROM memories WHERE id = ?{where}", [(i, *params) for i in ids])
            self._db.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def all(self, *, scope: Optional[Scope] = None, include_invalid: bool = False) -> List[MemoryItem]:
        """List all valid memories within the scope-limited range (in reverse time order; B semantics: only filter on non-empty dimensions).

        With include_invalid=True, soft-invalidated ones are returned too (for auditing / inspecting the fact evolution
        chain). By default only valid ones are returned: rebuild_index / forget / consolidate / summary all build on
        this, so invalidated memories won't be re-indexed or revived by cleanup.
        """
        where, params = scope_where_clause(scope or Scope())
        valid = "" if include_invalid else (
            " AND invalid_at IS NULL" if where else " WHERE invalid_at IS NULL")
        with self._lock:
            cur = self._db.execute(
                f"SELECT {self._COLS} FROM memories{where}{valid} ORDER BY created_at DESC", params)
            rows = cur.fetchall()
        return [self._row_to_item(row) for row in rows]

    @staticmethod
    def _row_to_item(row) -> MemoryItem:
        """Restore a database row into a MemoryItem."""
        id_, content, type_, importance, created_at, updated_at, last_accessed_at, invalid_at, superseded_by, metadata = row
        opt = [datetime.fromisoformat(t) if t else None for t in (updated_at, last_accessed_at, invalid_at)]
        return MemoryItem(content=content, id=id_, type=type_, importance=importance,
                          created_at=datetime.fromisoformat(created_at),
                          updated_at=opt[0], last_accessed_at=opt[1], invalid_at=opt[2],
                          superseded_by=superseded_by, metadata=json.loads(metadata))
