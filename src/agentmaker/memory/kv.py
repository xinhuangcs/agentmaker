"""agentmaker.memory.kv: key-value memory (structured facts stored and read by exact key).

Complements semantic memory (Memory, fuzzy retrieval): KV stores and reads by exact "key", one value per key, direct
overwrite, for definite structured facts (e.g. location=Shanghai, allergies=[peanuts]), without guessing via retrieval.
Modeled on the KV layer of Mem0's three-tier storage; implemented on SQLite with scope isolation (reusing retrieval.scope).

    KVStore: the low-level key-value table (values are strings).
    KVMemory: a facade adding JSON encode/decode on top of KVStore, supporting str / number / list / dict values.
"""

import json
import sqlite3
import threading
from typing import Any, Optional

from ..core.exceptions import RetrievalError
from ..core.sqlite_util import open_sqlite, require_unique_columns
from ..retrieval.scope import Scope, require_explicit_scope
from ..retrieval.scope_sql import scope_column_names, scope_store_values, scope_where, scope_where_clause


class KVStore:
    """Key-value store: (scope, key) is unique, set writes by overwrite, get reads exactly. Values are strings."""

    def __init__(self, db_path: str = ":memory:", *, lock: Optional[threading.RLock] = None):
        """Open the connection and create the table if needed. (each scope column + key) is the unique key, guaranteeing "one value per key under a given ownership".

        Args:
            db_path: SQLite file path; defaults to ":memory:" for self-tests only.
            lock: a reentrant lock serializing the connection; pass the same one when sharing a connection with another
                store, otherwise a new one is created.
        """
        cols = scope_column_names()
        scope_cols = ", ".join(f"{c} TEXT" for c in cols)
        unique_cols = ", ".join(cols + ["key"])
        self._lock = lock or threading.RLock()  # check_same_thread=False reuses the connection across threads, so one lock must serialize access to avoid corrupting the connection under concurrency
        try:
            self._db = open_sqlite(db_path)
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS kv("
                f"{scope_cols}, key TEXT, value TEXT, UNIQUE({unique_cols}))")
            # Open-time self-check: the unique constraint must cover exactly (each scope dimension + key). If an old
            # database (e.g. built before Scope gained a dimension) has a UNIQUE missing the new dimension, an
            # INSERT OR REPLACE across the new dimension would overwrite and silently leak data (kv stores
            # profile-like authoritative facts), so fail loud.
            require_unique_columns(self._db, "kv", {*cols, "key"}, error_cls=RetrievalError)
            self._db.commit()
        except sqlite3.Error as e:
            db = getattr(self, "_db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
            raise RetrievalError(f"failed to open / initialize the key-value store: {e}") from e
        except BaseException:
            db = getattr(self, "_db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
            raise

    def set(self, key: str, value: str, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> None:
        """Overwrite-write: update value if (scope, key) already exists, otherwise insert.

        Rejects an empty scope (unless all_scopes=True is explicit): an empty scope would write into an "ownerless" row
        mixed in with all users, which is a slip.
        """
        sc = scope or Scope()
        require_explicit_scope(sc, all_scopes, "kv.set")
        sv = scope_store_values(sc)
        cols = ", ".join(scope_column_names())
        placeholders = ", ".join("?" for _ in scope_column_names())
        try:
            with self._lock:
                self._db.execute(
                    f"INSERT OR REPLACE INTO kv({cols}, key, value) VALUES ({placeholders}, ?, ?)",
                    (*sv, key, value))
                self._db.commit()
        except sqlite3.Error as e:
            raise RetrievalError(f"failed to write key-value: {e}") from e

    def get(self, key: str, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> Optional[str]:
        """Read exactly by key; returns None if not found.

        Rejects an empty scope (unless all_scopes=True is explicit): an empty scope adds no ownership filter and would
        read another user's value. When the scope spans multiple sub-ownerships (a coarse scope hits several rows for
        the same key), fail loud: KV semantics are "one value per key under a scope" and require a unique footprint.
        """
        sc = scope or Scope()
        require_explicit_scope(sc, all_scopes, "kv.get")
        where, params = scope_where(sc)
        with self._lock:
            rows = self._db.execute(f"SELECT value FROM kv WHERE key = ?{where}", (key, *params)).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            raise RetrievalError(
                f"kv.get('{key}') matched {len(rows)} rows under the given scope (the scope spans multiple sub-ownerships): use a scope narrowed to a unique ownership")
        return rows[0][0]

    def delete(self, key: str, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> None:
        """Delete a key.

        Rejects an empty scope (unless all_scopes=True is explicit): an empty scope would delete this key across all
        users, which is a dangerous operation.
        """
        sc = scope or Scope()
        require_explicit_scope(sc, all_scopes, "kv.delete")
        where, params = scope_where(sc)
        with self._lock:
            self._db.execute(f"DELETE FROM kv WHERE key = ?{where}", (key, *params))
            self._db.commit()

    def all(self, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> dict:
        """Return all key-values under this scope (as a dict).

        Rejects an empty scope (unless all_scopes=True is explicit): an empty scope would read across users. When the
        same key appears in multiple matched sub-ownerships (a coarse scope / all_scopes spanning users), fail loud: a
        `dict` cannot represent duplicate keys, so ambiguity requires an exact scope or per-scope reads.
        """
        sc = scope or Scope()
        require_explicit_scope(sc, all_scopes, "kv.all")
        where, params = scope_where_clause(sc)
        with self._lock:
            rows = self._db.execute(f"SELECT key, value FROM kv{where}", params).fetchall()
        result: dict = {}
        for k, v in rows:
            if k in result:
                raise RetrievalError(
                    f"kv.all matched multiple rows for key '{k}' (the scope spans multiple sub-ownerships; a dict can't represent this): use an exact scope or read per-scope")
            result[k] = v
        return result

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
            exc.add_note(f"KVStore cleanup also failed: {cleanup_error}")
        return False


class KVMemory:
    """Key-value memory facade: adds JSON encode/decode on top of KVStore, supporting str / number / list / dict values; carries a fixed scope."""

    def __init__(self, kv: KVStore, *, scope: Optional[Scope] = None):
        """Initialize the facade.

        Args:
            kv: the underlying key-value store.
            scope: the ownership of this facade (e.g. a given user); defaults to Scope(base="kv").
        """
        self.kv = kv
        self.scope = scope or Scope(base="kv")

    def set(self, key: str, value: Any) -> None:
        """Write a key; value may be str / number / list / dict, JSON-encoded internally before storing."""
        self.kv.set(key, json.dumps(value, ensure_ascii=False), scope=self.scope)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a key and JSON-decode it; returns default if not found."""
        raw = self.kv.get(key, scope=self.scope)
        return default if raw is None else self._decode(raw)

    def delete(self, key: str) -> None:
        """Delete a key."""
        self.kv.delete(key, scope=self.scope)

    def close(self) -> None:
        """Close the underlying KVStore's database connection."""
        self.kv.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, exc, _tb):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"KVMemory cleanup also failed: {cleanup_error}")
        return False

    def as_dict(self) -> dict:
        """Return the entire key-value set (values already JSON-decoded)."""
        return {k: self._decode(v) for k, v in self.kv.all(scope=self.scope).items()}

    @staticmethod
    def _decode(raw: str) -> Any:
        """JSON-decode; on failure return an externally written non-JSON value as-is."""
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
