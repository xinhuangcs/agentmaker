"""agentmaker.memory.store: the source-of-truth store for memories (the authoritative copy).

The retrieval base stores only id + body for fast search; here we store the full MemoryItem with type / importance /
timestamps, which is the authoritative copy: if an index (vector / keyword) breaks, it can be rebuilt from this.
Persisted in SQLite, accessed by id.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional

from ..core.clock import now_utc
from ..core.exceptions import RetrievalError
from ..core.sqlite_util import ensure_columns, open_sqlite, require_primary_key
from ..retrieval.scope import Scope
from ..retrieval.scope_sql import (scope_column_names, scope_from_store_values,
                                   scope_store_values, scope_where, scope_where_clause)
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
        self._coordination_key = (
            None if db_path == ":memory:" else f"memory:{os.path.realpath(db_path)}")
        self._lock = lock or threading.RLock()  # check_same_thread=False reuses the connection across threads, so one lock must serialize access to avoid corrupting the connection under concurrency
        try:
            self._db = open_sqlite(db_path)
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS memories("
                f"id TEXT, {scope_cols}, content TEXT, type TEXT, "
                f"importance REAL, created_at TEXT, updated_at TEXT, last_accessed_at TEXT, "
                f"invalid_at TEXT, superseded_by TEXT, metadata TEXT, "
                f"PRIMARY KEY ({pk_cols}))")
            require_primary_key(self._db, "memories", {"id", *scope_column_names()}, error_cls=RetrievalError)
            ensure_columns(self._db, "memories",
                           {"updated_at": "TEXT", "last_accessed_at": "TEXT", "invalid_at": "TEXT", "superseded_by": "TEXT"})
            self._db.commit()
        except sqlite3.Error as e:
            db = getattr(self, "_db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
            raise RetrievalError(f"failed to open / initialize the memory source-of-truth store: {e}") from e
        except BaseException:
            db = getattr(self, "_db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
            raise

    _COLS = "id, content, type, importance, created_at, updated_at, last_accessed_at, invalid_at, superseded_by, metadata"
    _SCOPE_COLS = ", ".join(scope_column_names())

    def save(self, item: MemoryItem, *, scope: Optional[Scope] = None) -> None:
        """Store a memory (overwrites if the same id + same scope already exists; same id under different scopes stays independent). metadata is stored as JSON, times as isoformat."""
        sv = scope_store_values(scope or Scope())
        cols = ", ".join(scope_column_names())
        placeholders = ", ".join("?" for _ in scope_column_names())
        opt = [item.updated_at, item.last_accessed_at, item.invalid_at]
        try:
            with self._lock:
                with self._db:
                    self._db.execute(
                        f"INSERT OR REPLACE INTO memories(id, {cols}, content, type, importance, created_at, "
                        f"updated_at, last_accessed_at, invalid_at, superseded_by, metadata) "
                        f"VALUES (?, {placeholders}, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (item.id, *sv, item.content, item.type, item.importance, item.created_at.isoformat(),
                         *[t.isoformat() if t else None for t in opt], item.superseded_by,
                         json.dumps(item.metadata, ensure_ascii=False)))
        except sqlite3.Error as e:
            raise RetrievalError(f"failed to write memory: {e}") from e

    def replace(self, id: str, item: MemoryItem, *, scope: Optional[Scope] = None) -> None:
        """Atomically replace within a single transaction: delete the old row matching (id, scope) first, then insert item.

        When a range scope resolves to one finer ownership footprint, that exact footprint is
        retained. Multiple matches fail without modifying the store.
        """
        sc = scope or Scope()
        del_where, del_params = scope_where(sc)
        cols = ", ".join(scope_column_names())
        placeholders = ", ".join("?" for _ in scope_column_names())
        opt = [item.updated_at, item.last_accessed_at, item.invalid_at]
        try:
            with self._lock:
                with self._db:
                    matches = self._db.execute(
                        f"SELECT {self._SCOPE_COLS} FROM memories WHERE id = ?{del_where} LIMIT 2",
                        (id, *del_params)).fetchall()
                    if len(matches) > 1:
                        raise RetrievalError(
                            f"memory.replace('{id}') matched multiple rows under the given scope; "
                            "use a scope narrowed to one ownership footprint")
                    sv = list(matches[0]) if matches else scope_store_values(sc)
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
        resolved = self.get_with_scope(id, scope=scope)
        return resolved[0] if resolved is not None else None

    def get_with_scope(self, id: str, *, scope: Optional[Scope] = None) -> Optional[tuple[MemoryItem, Scope]]:
        """Fetch one memory together with its exact stored ownership footprint."""
        where, params = scope_where(scope or Scope())
        with self._lock:
            cur = self._db.execute(
                f"SELECT {self._SCOPE_COLS}, {self._COLS} FROM memories WHERE id = ?{where}",
                (id, *params))
            rows = cur.fetchmany(2)
        if len(rows) > 1:
            raise RetrievalError(
                f"memory.get('{id}') matched multiple rows under the given scope; "
                "use a scope narrowed to one ownership footprint")
        if not rows:
            return None
        width = len(scope_column_names())
        return self._row_to_item(rows[0][width:]), scope_from_store_values(rows[0][:width])

    def touch(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Batch-write last_accessed_at = now (a retrieval hit means "got used", Generative Agents semantics); one commit."""
        if not ids:
            return
        now = now_utc().isoformat()
        where, params = scope_where(scope or Scope())
        with self._lock:
            try:
                with self._db:
                    self._db.executemany(
                        f"UPDATE memories SET last_accessed_at = ? WHERE id = ?{where}",
                        [(now, i, *params) for i in ids])
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
            try:
                with self._db:
                    self._db.executemany(
                        f"DELETE FROM memories WHERE id = ?{where}",
                        [(i, *params) for i in ids])
            except sqlite3.Error as e:
                raise RetrievalError(f"failed to delete memories: {e}") from e

    def scopes_for_ids(self, ids: List[str], *, scope: Optional[Scope] = None) -> List[tuple[str, Scope]]:
        """Resolve ids to every exact stored footprint within a scope range."""
        if not ids:
            return []
        where, params = scope_where(scope or Scope())
        with self._lock:
            rows = []
            for id_ in ids:
                rows.extend(self._db.execute(
                    f"SELECT id, {self._SCOPE_COLS} FROM memories WHERE id = ?{where}",
                    (id_, *params)).fetchall())
        return [(row[0], scope_from_store_values(row[1:])) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, exc, _tb):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"MemoryStore cleanup also failed: {cleanup_error}")
        return False

    def all(self, *, scope: Optional[Scope] = None, include_invalid: bool = False) -> List[MemoryItem]:
        """List all valid memories within the scope-limited range (in reverse time order; B semantics: only filter on non-empty dimensions).

        With include_invalid=True, soft-invalidated ones are returned too (for auditing / inspecting the fact evolution
        chain). By default only valid ones are returned: rebuild_index / forget / consolidate / summary all build on
        this, so invalidated memories won't be re-indexed or revived by cleanup.
        """
        return [item for item, _ in self.all_with_scopes(
            scope=scope, include_invalid=include_invalid)]

    def all_with_scopes(self, *, scope: Optional[Scope] = None,
                        include_invalid: bool = False) -> List[tuple[MemoryItem, Scope]]:
        """List memories together with their exact stored ownership footprints."""
        where, params = scope_where_clause(scope or Scope())
        valid = "" if include_invalid else (
            " AND invalid_at IS NULL" if where else " WHERE invalid_at IS NULL")
        with self._lock:
            cur = self._db.execute(
                f"SELECT {self._SCOPE_COLS}, {self._COLS} FROM memories{where}{valid} "
                "ORDER BY created_at DESC", params)
            rows = cur.fetchall()
        width = len(scope_column_names())
        return [
            (self._row_to_item(row[width:]), scope_from_store_values(row[:width]))
            for row in rows
        ]

    @staticmethod
    def _row_to_item(row) -> MemoryItem:
        """Restore a database row into a MemoryItem."""
        id_, content, type_, importance, created_at, updated_at, last_accessed_at, invalid_at, superseded_by, metadata = row
        opt = [datetime.fromisoformat(t) if t else None for t in (updated_at, last_accessed_at, invalid_at)]
        return MemoryItem(content=content, id=id_, type=type_, importance=importance,
                          created_at=datetime.fromisoformat(created_at),
                          updated_at=opt[0], last_accessed_at=opt[1], invalid_at=opt[2],
                          superseded_by=superseded_by, metadata=json.loads(metadata))
