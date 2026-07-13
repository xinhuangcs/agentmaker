"""agentmaker.runtime.execution.checkpoint: execution-state persistence (CheckpointStore: one point per scope, overwrite, cleared after resume).

Persists a serialized `ExecutionState` (messages + pending calls + decisions + remaining iterations + optional pending)
by scope, shared across three uses (aligned with LangGraph's checkpointer):
    - HITL: save at the suspend point (with pending), then resume(decisions) to continue.
    - Crash recovery: save every step (no pending), then resume() after the process restarts.
    - Long-task resume: same as crash recovery.
The semantics differ from SessionStore (append-only conversation history): a checkpoint is the "single current
restorable state", so save overwrites and clear runs once the resume succeeds. It reuses the same Scope isolation and
can share a database with sessions / memory (in a different table). Single point, overwrite, no history kept (so
time-travel is not supported).
"""

import asyncio
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import Optional

from ...core.clock import now_utc
from ...core.exceptions import SessionError
from ...core.sqlite_util import open_sqlite, require_columns, require_unique_columns
from ...retrieval.scope import Scope
from ...retrieval.scope_sql import scope_column_names, scope_exact_where_clause, scope_store_values


class CheckpointStore(ABC):
    """Checkpoint storage interface: save / load / clear the current execution-state checkpoint by Scope."""

    @abstractmethod
    def save(self, state_json: str, *, scope: Optional[Scope] = None) -> None:
        """Save (overwrite) the current checkpoint for scope; state_json is a serialized ExecutionState JSON string."""

    @abstractmethod
    def load(self, *, scope: Optional[Scope] = None) -> Optional[str]:
        """Read the current checkpoint for scope; return None if absent."""

    @abstractmethod
    def clear(self, *, scope: Optional[Scope] = None) -> None:
        """Clear the checkpoint for scope (called after a run completes normally or a resume succeeds)."""

    # a* async pair: the framework's async execution layer (BaseAgent finalization chain) calls through a*; the
    # default wraps the sync version in to_thread (DB IO, where paying for a thread hop is reasonable), so a sync
    # implementation gets async for free. Async / distributed backends may override with a native await.
    async def asave(self, state_json: str, *, scope: Optional[Scope] = None) -> None:
        """Async version of save (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.save(state_json, scope=scope))

    async def aload(self, *, scope: Optional[Scope] = None) -> Optional[str]:
        """Async version of load (defaults to to_thread)."""
        return await asyncio.to_thread(lambda: self.load(scope=scope))

    async def aclear(self, *, scope: Optional[Scope] = None) -> None:
        """Async version of clear (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.clear(scope=scope))


class SqliteCheckpointStore(CheckpointStore):
    """Checkpoints persisted to SQLite: at most one row per scope (full-dimension unique index on scope + upsert overwrite, always keeping only the latest)."""

    def __init__(self, db_path: str = ":memory:"):
        """Open the connection and create the table if needed (with a full-dimension unique index on scope so the DB enforces "one point per scope").

        Args:
            db_path: SQLite file path; defaults to ":memory:" for self-tests only. In production pass a file path to
                persist (can share a database with session / memory by passing the same file path).
        """
        scope_cols = ", ".join(f"{c} TEXT" for c in scope_column_names())
        cols = ", ".join(scope_column_names())
        self._lock = threading.Lock()  # Serialize cross-thread access (single connection with check_same_thread=False; low concurrency of a personal daemon is enough).
        try:
            self._db = open_sqlite(db_path)
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS checkpoints({scope_cols}, state TEXT, created_at TEXT)")
            scope_set = set(scope_column_names())
            require_columns(self._db, "checkpoints", scope_set, error_cls=SessionError)
            self._db.execute(
                f"DELETE FROM checkpoints WHERE rowid NOT IN "
                f"(SELECT MAX(rowid) FROM checkpoints GROUP BY {cols})")
            self._db.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS idx_checkpoint_scope ON checkpoints({cols})")
            require_unique_columns(self._db, "checkpoints", scope_set, error_cls=SessionError)
            self._db.commit()
        except sqlite3.Error as e:
            db = getattr(self, "_db", None)
            if db is not None:
                try:
                    db.close()  # Table / index creation failed: close the already-opened connection before raising to avoid a connection leak.
                except Exception:
                    pass
            raise SessionError(f"Failed to open / initialize checkpoint store: {e}") from e
        except BaseException:
            db = getattr(self, "_db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
            raise

    def save(self, state_json: str, *, scope: Optional[Scope] = None) -> None:
        """Overwrite-save: use the full-dimension unique index on scope to upsert (`INSERT OR REPLACE`): exactly one row per scope, a single atomic statement.

        A single statement has no delete/insert intermediate state, and the unique index ensures concurrent
        saves of the same scope cannot produce duplicate rows.
        """
        sv = scope_store_values(scope or Scope())
        cols = ", ".join(scope_column_names())
        placeholders = ", ".join("?" for _ in scope_column_names())
        with self._lock:
            try:
                self._db.execute(
                    f"INSERT OR REPLACE INTO checkpoints({cols}, state, created_at) "
                    f"VALUES ({placeholders}, ?, ?)",
                    (*sv, state_json, now_utc().isoformat()))
                self._db.commit()
            except sqlite3.Error as e:
                self._db.rollback()
                raise SessionError(f"Failed to save checkpoint: {e}") from e

    def load(self, *, scope: Optional[Scope] = None) -> Optional[str]:
        """Read the current checkpoint for scope (exact match on all scope dimensions; the unique index guarantees at most one row); return None if absent."""
        where, params = self._where(scope)
        with self._lock:
            try:
                cur = self._db.execute(f"SELECT state FROM checkpoints{where} LIMIT 1", params)
                row = cur.fetchone()
                return row[0] if row else None
            except sqlite3.Error as e:
                raise SessionError(f"Failed to load checkpoint: {e}") from e

    def clear(self, *, scope: Optional[Scope] = None) -> None:
        """Delete the checkpoint for scope (exact match on all scope dimensions)."""
        where, params = self._where(scope)
        with self._lock:
            try:
                self._db.execute(f"DELETE FROM checkpoints{where}", params)
                self._db.commit()
            except sqlite3.Error as e:
                raise SessionError(f"Failed to clear checkpoint: {e}") from e

    @staticmethod
    def _where(scope: Optional[Scope]):
        """Build the WHERE + params for an exact match on all scope dimensions (empty dimension = empty string), via the shared scope_exact_where_clause.

        CheckpointStore is "one checkpoint per exact scope", so load/clear/overwrite must all match ALL dimensions of
        that scope EXACTLY, and must NOT use prefix semantics (filtering only non-empty dimensions). Otherwise a parent
        scope's delete (e.g. Plan: session=p1) would, via `WHERE session=p1`, wrongly hit a child scope's row (e.g.
        executor: session=p1, agent=::plan_exec). Nested suspend (Plan) exposed this pitfall.
        """
        return scope_exact_where_clause(scope or Scope())

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()
