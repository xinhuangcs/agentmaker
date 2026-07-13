"""agentmaker.retrieval.index_sync: the derived-index sync seam (IndexSync), shared by memory and rag.

Both Memory and RAG have "a source-of-truth store plus a derived retrieval index" that must stay in sync:
Memory = MemoryStore + HybridRetriever, RAG = SourceStore + HybridRetriever. How to keep the two consistent
is a cross-cutting concern collapsed into this one pluggable seam: each subsystem's write path only calls it,
rather than each writing its own compensation. Placing it in the shared retrieval layer (rather than one copy
each in memory / rag) means the invariants and the "sync vs async" pluggable point are defined once for the
whole framework.

Invariant: the source-of-truth store is the sole authority, the retrieval index is a derivation rebuildable
from it, and must converge to it.

Two failure semantics (chosen per each subsystem's read-path robustness):
    - best-effort (`index` / default `drop`): on write failure it does not raise, marks the ids pending, and relies on
      read-time self-heal + reconcile to converge (eventual consistency). Memory uses this: its read path
      filters orphans and self-heals at read time, so it tolerates a brief index lag.
    - fail-loud replacement (`replace`): failure raises so the caller can compensate its source-of-truth side.
      Transactional visibility is a backend capability: the shared-connection SQLite backend swaps the batch
      atomically, while generic backends may expose a short overlap window and rely on compensation.

Bookkeeping (fingerprints + pending set) is swappable through `SyncBookkeeping`. The default is shared
in-process state; `SqliteBookkeeping` provides persistent cross-process state. Distributed delivery can
provide another `IndexSync` implementation without changing subsystem write paths.
Write coordination in this module is process-local and is not a distributed lock.
"""

import hashlib
import inspect
import itertools
import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from ..core.clock import now_utc
from ..core.exceptions import RetrievalError
from ..core.sqlite_util import open_sqlite
from ._coordination import shared_coordinator, shared_value
from .scope import Scope
from .scope_sql import (scope_column_names, scope_exact_where, scope_exact_where_clause,
                        scope_from_store_values, scope_store_values, scope_where)
from ..core.trace_events import EVENT_INDEX_SYNC_PENDING, EVENT_INDEX_SYNC_RECONCILE


def _digest(content: str, metadata: Optional[dict] = None) -> str:
    """Content fingerprint (sha1): used for idempotent upsert, so an unchanged content / metadata skips re-writing the index / embedding.

    When metadata is empty (None / {}) it falls back to a pure content hash, keeping the fingerprint unchanged
    for metadata-free paths like memory; when metadata is
    present it hashes the canonicalized JSON alongside, so a metadata change (e.g. the filterable column
    doc_id) triggers a rewrite instead of being short-circuited as "content unchanged"."""
    if not metadata:
        return hashlib.sha1(content.encode("utf-8")).hexdigest()
    raw = content + "\x00" + json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _accepts_strict(method) -> bool:
    """Whether a drop-family callable accepts the strict keyword (unknowable signatures count as yes)."""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return True
    return "strict" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values())


def call_drop(sync: "IndexSync", ids: List[str], *, scope=None, strict: bool = False) -> None:
    """Call sync.drop, forwarding strict only when the implementation accepts it.

    An implementation whose drop signature lacks the strict flag keeps its
    best-effort contract: physical failures surface via pending(), not raising.
    """
    if not strict:
        sync.drop(ids, scope=scope)
        return
    drop = getattr(type(sync), "drop", None)
    if drop is None or _accepts_strict(drop):
        sync.drop(ids, scope=scope, strict=True)
    else:
        sync.drop(ids, scope=scope)


def call_drop_exact(sync: "IndexSync", ids: List[str], *, scope=None, strict: bool = False) -> None:
    """Call sync.drop_exact when present (forwarding strict only when accepted), else a ranged drop.

    The strict downgrade rule matches call_drop, so legacy implementations keep
    their best-effort contract on every drop-family entry point.
    """
    drop_exact = getattr(sync, "drop_exact", None)
    if drop_exact is None:
        call_drop(sync, ids, scope=scope, strict=strict)
        return
    if not strict:
        drop_exact(ids, scope=scope)
        return
    method = getattr(type(sync), "drop_exact", None) or drop_exact
    if _accepts_strict(method):
        drop_exact(ids, scope=scope, strict=True)
    else:
        drop_exact(ids, scope=scope)


class IndexSync(ABC):
    """The derived-index sync seam: a subsystem's write path propagates changes to the retrieval index and reconciles through it.

    Subclasses decide sync / async and whether to persist (the default SyncIndexSync writes through
    synchronously; production may switch to outbox + worker).
    """

    @abstractmethod
    def index(self, ids: List[str], contents: List[str], *, scope=None, metadatas=None) -> None:
        """Upsert several entries into the index (idempotent: unchanged content is skipped). Best-effort: does not raise on failure, marks pending.

        metadatas (optional, same length as ids): a metadata dict per entry, passed through to the retrieval
        foundation to be stored as filterable columns (see MetadataFilter in retrieval/types.py).
        """

    @abstractmethod
    def replace(self, old_ids: List[str], new_ids: List[str], contents: List[str], *, scope=None,
                metadatas=None) -> None:
        """Replace old ids with a new batch and raise on failure.

        Transactional visibility depends on the retrieval backend. Callers compensate a failed
        non-transactional backend; SQLite's shared-connection backend performs one transaction.
        """

    @abstractmethod
    def drop(self, ids: List[str], *, scope=None, strict: bool = False) -> None:
        """Delete entries, raising physical failures only when strict is true."""

    def drop_exact(self, ids: List[str], *, scope=None, strict: bool = False) -> None:
        """Delete entries from one exact ownership footprint.

        The default falls back to a ranged ``drop`` over the footprint scope, which
        equals the footprint unless sibling footprints share an id within that range;
        exact-capable implementations (such as SyncIndexSync) override this.
        """
        call_drop(self, ids, scope=scope, strict=strict)

    def drop_range(self, entries, *, scope=None) -> None:
        """Delete a scope range while receiving its exact bookkeeping groups."""
        for exact_scope, ids in entries.items():
            exact_ids = sorted(set(ids))
            if exact_ids:
                self.drop_exact(exact_ids, scope=exact_scope)

    @abstractmethod
    def reconcile(self, items: Sequence, *, scope=None, batch_size: int = 256) -> int:
        """Make the index converge with items (a source-of-truth snapshot; elements must have .id / .content) as the authority: drop orphans + force-reingest the source. Returns the number reingested."""

    @abstractmethod
    def pending(self, *, scope=None) -> set:
        """The currently pending ids (recent best-effort write failures not yet converged by reconcile). For apps to monitor / trigger reconciliation."""

    def tracked_ids(self, *, scope=None) -> set:
        """All ids currently indexed (fingerprint registered) in this scope, for consistency cross-checks (e.g. IngestionPipeline.verify).

        Optional capability: SyncIndexSync delegates to bookkeeping. Implementations that cannot enumerate
        raise NotImplementedError, which verifiers treat as an unavailable consistency check."""
        raise NotImplementedError

    def exact_scopes(self, *, scope=None) -> set[Scope]:
        """Enumerate bookkeeping footprints within a scope range when supported."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources held by this seam; resource-free implementations need not override."""


# -- Bookkeeping seam: where the fingerprints (idempotency) and pending set (self-heal) live and how they are stored --

class SyncBookkeeping(ABC):
    """The bookkeeping-storage seam for SyncIndexSync: content fingerprints + pending set. Default in-process, switchable to SQLite (cross-process persistence)."""

    @abstractmethod
    def get_hash(self, scope, id: str) -> Optional[str]:
        """Read one registered content fingerprint; returns None if absent."""

    @abstractmethod
    def set_hashes(self, scope, pairs: List[tuple]) -> None:
        """Register (id, fingerprint) in batch; overwrites for the same (id, scope)."""

    @abstractmethod
    def delete_hashes(self, scope, ids: List[str]) -> None:
        """Remove fingerprint registrations in batch (after entries are deleted from the index). Idempotent."""

    @abstractmethod
    def tracked_ids(self, scope) -> set:
        """All ids with a registered fingerprint in this scope (used for reconcile's orphan detection)."""

    @abstractmethod
    def mark_pending(self, scope, ids: List[str]) -> None:
        """Mark several ids as pending (on best-effort write failure); repeated marking keeps the earliest time."""

    @abstractmethod
    def clear_pending(self, scope, ids: List[str]) -> None:
        """Remove several ids from the pending set after a successful write. Idempotent."""

    @abstractmethod
    def pending_ids(self, scope) -> set:
        """The set of currently pending ids in this scope (a copy)."""

    def exact_scopes(self, scope) -> set[Scope]:
        """Enumerate non-empty bookkeeping footprints within a scope range."""
        raise NotImplementedError

    def pending_ancestor_ids(self, scope) -> set:
        """Return pending ids marked on this scope or any broader ancestor scope."""
        requested = scope or Scope()
        names = ("base", "user", "agent", "session", "app")
        choices = [
            (None, value) if value is not None else (None,)
            for value in (getattr(requested, name) for name in names)
        ]
        result = set()
        for values in itertools.product(*choices):
            result.update(self.pending_ids(Scope(**dict(zip(names, values)))))
        return result

    def close(self) -> None:
        """Release resources held by bookkeeping (e.g. the SQLite connection). Default is a no-op: the in-process implementation (InMemoryBookkeeping) need not override."""


class InMemoryBookkeeping(SyncBookkeeping):
    """The default bookkeeping: an in-process dict (zero dependency, zero overhead). Cleared on restart, so the
    idempotent-skip and pending set are lost with it, but the source-of-truth data remains available
    for reconciliation."""

    def __init__(self):
        self._lock = threading.RLock()
        self._hashes: dict = {}    # scope -> {id: fingerprint}
        self._pending: dict = {}   # scope -> {id: mark time} (the time aligns the semantics with the persistent implementation; here it only keeps the earliest value)

    def get_hash(self, scope, id):
        scope = scope or Scope()          # normalize None -> Scope(), aligned with SqliteBookkeeping (otherwise the two use different buckets and behavior drifts across backends)
        with self._lock:
            return self._hashes.get(scope, {}).get(id)

    def set_hashes(self, scope, pairs):
        scope = scope or Scope()
        with self._lock:
            h = self._hashes.setdefault(scope, {})
            for i, d in pairs:
                h[i] = d

    def delete_hashes(self, scope, ids):
        scope = scope or Scope()
        with self._lock:
            h = self._hashes.get(scope)
            if h:
                for i in ids:
                    h.pop(i, None)

    def tracked_ids(self, scope):
        scope = scope or Scope()
        with self._lock:
            return set(self._hashes.get(scope, {}))

    def mark_pending(self, scope, ids):
        scope = scope or Scope()
        with self._lock:
            pend = self._pending.setdefault(scope, {})
            now = now_utc().isoformat()
            for i in ids:
                pend.setdefault(i, now)

    def clear_pending(self, scope, ids):
        scope = scope or Scope()
        with self._lock:
            pend = self._pending.get(scope)
            if pend:
                for i in ids:
                    pend.pop(i, None)

    def pending_ids(self, scope):
        scope = scope or Scope()
        with self._lock:
            return set(self._pending.get(scope, {}))

    def exact_scopes(self, scope):
        requested = scope or Scope()
        with self._lock:
            candidates = set(self._hashes) | set(self._pending)
            return {
                candidate for candidate in candidates
                if (self._hashes.get(candidate) or self._pending.get(candidate))
                and all(
                    getattr(requested, name) is None
                    or getattr(requested, name) == getattr(candidate, name)
                    for name in ("base", "user", "agent", "session", "app")
                )
            }

    def pending_ancestor_ids(self, scope):
        requested = scope or Scope()
        with self._lock:
            return {
                id_
                for candidate, pending in self._pending.items()
                if pending and all(
                    getattr(candidate, name) is None
                    or getattr(candidate, name) == getattr(requested, name)
                    for name in ("base", "user", "agent", "session", "app")
                )
                for id_ in pending
            }


class SqliteBookkeeping(SyncBookkeeping):
    """Persistent bookkeeping (aligned with the de-facto standard form of LangChain's RecordManager): one SQLite table storing (scope, id, fingerprint, timestamp).

    It preserves idempotent fingerprints and pending repair state across process restarts, and its
    pending timestamp supports age-based monitoring.
    """

    def __init__(self, db_path: str, *, table: str = "index_sync_bookkeeping"):
        """db_path: the SQLite file path (co-locating with the source-of-truth store is fine, since bookkeeping is companion state of the ingestion pipeline)."""
        from .backends.sqlite import ensure_safe_table   # lazy import: reuse table-name sanitization, avoiding a top-level retrieval<->backends cycle
        from ..core.sqlite_util import primary_key_columns
        self.table = ensure_safe_table(table)
        self._coordination_key = (
            None if db_path == ":memory:"
            else f"index-bookkeeping:{os.path.realpath(db_path)}:{self.table}")
        scope_cols = ", ".join(f"{c} TEXT" for c in scope_column_names())
        pk = ", ".join(["id", *scope_column_names()])
        body = (f"id TEXT, {scope_cols}, content_hash TEXT, updated_at TEXT, pending_since TEXT, "
                f"PRIMARY KEY ({pk})")
        self._lock = threading.RLock()  # serialize across threads (the async path reuses the connection via to_thread)
        try:
            self._db = open_sqlite(db_path)
            self._db.execute(f"CREATE TABLE IF NOT EXISTS {self.table}({body})")
            # Rebuild derived bookkeeping when its ownership key changes.
            if primary_key_columns(self._db, self.table) != {"id", *scope_column_names()}:
                self._db.execute(f"DROP TABLE {self.table}")
                self._db.execute(f"CREATE TABLE {self.table}({body})")
            self._db.commit()
        except sqlite3.Error as e:
            db = getattr(self, "_db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
            raise RetrievalError(f"failed to open / initialize the index bookkeeping table: {e}") from e
        except BaseException:
            db = getattr(self, "_db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
            raise

    def _exec(self, fn_desc: str, sql: str, rows) -> None:
        """executemany + commit under the lock; sqlite errors are normalized to RetrievalError."""
        with self._lock:
            try:
                self._db.executemany(sql, rows)
                self._db.commit()
            except sqlite3.Error as e:
                self._db.rollback()
                raise RetrievalError(f"{fn_desc} failed: {e}") from e

    def get_hash(self, scope, id):
        where, params = scope_exact_where(scope or Scope())
        with self._lock:
            row = self._db.execute(
                f"SELECT content_hash FROM {self.table} WHERE id = ?{where}", (id, *params)).fetchone()
        return row[0] if row and row[0] else None

    def set_hashes(self, scope, pairs):
        sv = scope_store_values(scope or Scope())
        cols = ", ".join(scope_column_names())
        ph = ", ".join("?" for _ in scope_column_names())
        now = now_utc().isoformat()
        # upsert without overwriting pending_since (fingerprint registration and pending marking do not interfere; the successful-write path calls clear_pending afterward)
        self._exec("register index fingerprint", f"INSERT INTO {self.table}(id, {cols}, content_hash, updated_at, pending_since) "
                                  f"VALUES (?, {ph}, ?, ?, NULL) "
                                  f"ON CONFLICT(id, {cols}) DO UPDATE SET content_hash=excluded.content_hash, "
                                  f"updated_at=excluded.updated_at",
                   [(i, *sv, d, now) for i, d in pairs])

    def delete_hashes(self, scope, ids):
        where, params = scope_exact_where(scope or Scope())
        self._exec("remove index fingerprint", f"DELETE FROM {self.table} WHERE id = ?{where}", [(i, *params) for i in ids])

    def tracked_ids(self, scope):
        clause, params = scope_exact_where_clause(scope or Scope())
        with self._lock:
            rows = self._db.execute(
                f"SELECT id FROM {self.table}{clause} AND content_hash IS NOT NULL", params).fetchall()
        return {r[0] for r in rows}

    def mark_pending(self, scope, ids):
        sv = scope_store_values(scope or Scope())
        cols = ", ".join(scope_column_names())
        ph = ", ".join("?" for _ in scope_column_names())
        now = now_utc().isoformat()
        # insert the row if absent (hash set to NULL); if it exists, overwrite the time only when pending_since is empty (keep the earliest mark, the "how long stuck" measure)
        self._exec("mark pending", f"INSERT INTO {self.table}(id, {cols}, content_hash, updated_at, pending_since) "
                              f"VALUES (?, {ph}, NULL, ?, ?) "
                              f"ON CONFLICT(id, {cols}) DO UPDATE SET "
                              f"pending_since=COALESCE({self.table}.pending_since, excluded.pending_since)",
                   [(i, *sv, now, now) for i in ids])

    def clear_pending(self, scope, ids):
        where, params = scope_exact_where(scope or Scope())
        self._exec("clear pending", f"UPDATE {self.table} SET pending_since = NULL WHERE id = ?{where}",
                   [(i, *params) for i in ids])

    def pending_ids(self, scope):
        clause, params = scope_exact_where_clause(scope or Scope())
        with self._lock:
            rows = self._db.execute(
                f"SELECT id FROM {self.table}{clause} AND pending_since IS NOT NULL", params).fetchall()
        return {r[0] for r in rows}

    def exact_scopes(self, scope):
        where, params = scope_where(scope or Scope())
        cols = ", ".join(scope_column_names())
        with self._lock:
            rows = self._db.execute(
                f"SELECT DISTINCT {cols} FROM {self.table} "
                f"WHERE (content_hash IS NOT NULL OR pending_since IS NOT NULL){where}",
                params).fetchall()
        return {scope_from_store_values(row) for row in rows}

    def pending_ancestor_ids(self, scope):
        requested = scope or Scope()
        clauses = []
        params = []
        for column, value in zip(scope_column_names(), scope_store_values(requested)):
            if value:
                clauses.append(f"({column} = '' OR {column} = ?)")
                params.append(value)
            else:
                clauses.append(f"{column} = ''")
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._db.execute(
                f"SELECT id FROM {self.table} WHERE pending_since IS NOT NULL AND {where}",
                params).fetchall()
        return {row[0] for row in rows}

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()


class SyncIndexSync(IndexSync):
    """The default implementation: write through synchronously to the retrieval foundation + bookkeeping (content fingerprint for idempotency, pending set for self-heal).

    Bookkeeping defaults to in-process (InMemoryBookkeeping, cleared on restart, rebuildable by reconcile from
    the source-of-truth store); pass `bookkeeping=SqliteBookkeeping(db_path)` for cross-process persistence
    (from_config installs it by default). For fully async / distributed (outbox + worker), switch to a
    different IndexSync implementation without touching each subsystem's write path.
    """

    def __init__(self, retriever, *, bookkeeping: Optional[SyncBookkeeping] = None, tracer=None):
        """retriever: the retrieval foundation (HybridRetriever, needs add / replace / delete, signatures in hybrid.py).
        bookkeeping: optional bookkeeping storage; defaults to shared in-process state for the retriever.
        tracer: optional tracer (duck-typed emit); once attached, write failures (marked pending) and completed reconciliations emit index_sync events. Zero overhead when not attached."""
        self.retriever = retriever
        self.bookkeeping = bookkeeping if bookkeeping is not None else shared_value(
            retriever, "index_sync_bookkeeping", InMemoryBookkeeping)
        self._state_owner = shared_coordinator(self.bookkeeping)
        self._state = shared_value(
            self._state_owner, "index_sync_state", lambda: {"fingerprints_trusted": True})
        self.tracer = tracer
        self._coordinator = shared_coordinator(retriever)

    def _locked_ids(self, scope, ids):
        """Serialize writes for the same ids without deadlocking batches."""
        return self._coordinator.hold(ids)

    @staticmethod
    def _validate_batch(ids, contents, metadatas, operation: str) -> None:
        """Reject misaligned batches before zip can silently truncate them."""
        if len(ids) != len(contents):
            raise RetrievalError(f"{operation}: ids and contents must have the same length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise RetrievalError(f"{operation}: metadatas and ids must have the same length")

    def _trace(self, event: dict) -> None:
        """Emit one index-sync event without affecting data operations."""
        if self.tracer is not None:
            try:
                self.tracer.emit(event)
            except Exception:  # noqa: BLE001
                pass

    def _mark_pending(self, scope, ids, operation: str) -> None:
        """Record drift when bookkeeping is available."""
        if not ids:
            return
        try:
            self.bookkeeping.mark_pending(scope, ids)
        except Exception:  # noqa: BLE001
            return
        self._trace({"type": EVENT_INDEX_SYNC_PENDING, "op": operation, "count": len(ids)})

    def index(self, ids, contents, *, scope=None, metadatas=None) -> None:
        """Idempotent upsert: skip unchanged entries by content fingerprint; write to the foundation, and on failure mark this batch of ids pending, without raising or rolling back the source-of-truth store."""
        self._validate_batch(ids, contents, metadatas, "index")
        with self._locked_ids(scope, ids):
            self._index_locked(ids, contents, scope=scope, metadatas=metadatas)

    def _index_locked(self, ids, contents, *, scope=None, metadatas=None) -> None:
        """Index a validated batch while its scope/id stripes are held."""
        bk = self.bookkeeping
        mds = metadatas if metadatas is not None else [None] * len(ids)
        trusted = self._state["fingerprints_trusted"]
        try:
            ancestor_pending = bk.pending_ancestor_ids(scope) if trusted else set()
        except Exception:  # noqa: BLE001
            trusted = False
            ancestor_pending = set()
        todo_ids, todo_contents, todo_mds, digests = [], [], [], []
        for i, c, m in zip(ids, contents, mds):
            d = _digest(c, m)
            try:
                unchanged = (
                    trusted
                    and i not in ancestor_pending
                    and bk.get_hash(scope, i) == d
                )
            except Exception:  # noqa: BLE001
                unchanged = False
            if unchanged:
                continue
            todo_ids.append(i)
            todo_contents.append(c)
            todo_mds.append(m)
            digests.append(d)
        if not todo_ids:
            return
        try:
            mkw = {"metadatas": todo_mds} if any(m for m in todo_mds) else {}   # pass only when there is metadata (do not force stubs to grow the parameter)
            self.retriever.add(todo_ids, todo_contents, scope=scope, **mkw)
        except Exception:  # noqa: BLE001
            self._mark_pending(scope, todo_ids, "index")
            return
        try:
            bk.set_hashes(scope, list(zip(todo_ids, digests)))
            bk.clear_pending(scope, todo_ids)
        except Exception:  # noqa: BLE001
            self._state["fingerprints_trusted"] = False
            self._mark_pending(scope, todo_ids, "index")

    def replace(self, old_ids, new_ids, contents, *, scope=None, metadatas=None) -> None:
        """Delegate old-to-new replacement and update bookkeeping only after success.

        A transactional backend makes the replacement atomic. A compensating backend may expose a
        short overlap window but still raises so the ingestion caller can remove the new batch.
        """
        self._validate_batch(new_ids, contents, metadatas, "replace")
        all_ids = list(old_ids) + list(new_ids)
        with self._locked_ids(scope, all_ids):
            mkw = {"metadatas": metadatas} if metadatas is not None else {}
            self.retriever.replace(old_ids, new_ids, contents, scope=scope, **mkw)
            bk = self.bookkeeping
            new_set = set(new_ids)
            stale = [i for i in old_ids if i not in new_set]
            mds = metadatas if metadatas is not None else [None] * len(new_ids)
            try:
                if stale:
                    bk.delete_hashes(scope, stale)
                bk.set_hashes(scope, [
                    (i, _digest(c, m)) for i, c, m in zip(new_ids, contents, mds)
                ])
                bk.clear_pending(scope, all_ids)
            except Exception:  # noqa: BLE001
                self._state["fingerprints_trusted"] = False
                self._mark_pending(scope, all_ids, "replace")

    def drop(self, ids, *, scope=None, strict: bool = False) -> None:
        """Delete entries and optionally raise physical failures after marking them pending."""
        self._drop(ids, scope=scope, exact=False, strict=strict)

    def drop_exact(self, ids, *, scope=None, strict: bool = False) -> None:
        """Delete from one exact ownership footprint when the backend supports it."""
        self._drop(ids, scope=scope, exact=True, strict=strict)

    def _drop(self, ids, *, scope=None, exact: bool, strict: bool) -> None:
        """Apply an index delete and update bookkeeping."""
        ids = list(ids)
        if not ids:
            return
        delete = getattr(self.retriever, "delete_exact", None) if exact else None
        with self._locked_ids(scope, ids):
            try:
                if delete is None:
                    self.retriever.delete(ids, scope=scope)
                else:
                    delete(ids, scope=scope)
            except NotImplementedError:
                if strict:
                    raise
                self._mark_pending(scope, ids, "drop")
                return
            except Exception:  # noqa: BLE001
                self._mark_pending(scope, ids, "drop")
                if strict:
                    raise
                return
            bookkeeping_scopes = [scope]
            if not exact:
                try:
                    exact_scopes = self.bookkeeping.exact_scopes(scope)
                except Exception:  # noqa: BLE001
                    self._state["fingerprints_trusted"] = False
                    self._mark_pending(scope, ids, "drop")
                    return
                if exact_scopes:
                    bookkeeping_scopes = list(exact_scopes)
            for bookkeeping_scope in bookkeeping_scopes:
                try:
                    self.bookkeeping.delete_hashes(bookkeeping_scope, ids)
                    self.bookkeeping.clear_pending(bookkeeping_scope, ids)
                except Exception:  # noqa: BLE001
                    self._state["fingerprints_trusted"] = False
                    self._mark_pending(bookkeeping_scope, ids, "drop")

    def reconcile(self, items, *, scope=None, batch_size: int = 256) -> int:
        """Reconcile one exact ownership footprint from an authoritative item snapshot.

        Tracked or pending ids absent from the snapshot are dropped, then every source item is
        reingested in batches. Physical orphans unknown to bookkeeping remain a read-time repair.
        """
        if batch_size < 1:
            raise RetrievalError(f"reconcile batch_size must be >= 1, got {batch_size}")
        items = list(items)
        source_ids = {it.id for it in items}
        try:
            known_ids = self.bookkeeping.tracked_ids(scope) | self.bookkeeping.pending_ids(scope)
        except Exception:  # noqa: BLE001
            known_ids = set()
        stale = list(known_ids - source_ids)
        if stale:
            self.drop_exact(stale, scope=scope)
        succeeded = 0
        for k in range(0, len(items), batch_size):
            batch = items[k:k + batch_size]
            ids = [it.id for it in batch]
            mds = [getattr(it, "metadata", None) for it in batch]   # reingest with metadata (otherwise RAG's filterable columns like md_doc_id are wiped and filters silently return zero hits)
            with self._locked_ids(scope, ids):
                try:
                    mkw = {"metadatas": mds} if any(m for m in mds) else {}
                    self.retriever.add(ids, [it.content for it in batch], scope=scope, **mkw)
                except Exception:  # noqa: BLE001
                    self._mark_pending(scope, ids, "reconcile")
                    continue
                succeeded += len(batch)
                try:
                    self.bookkeeping.set_hashes(
                        scope, [(it.id, _digest(it.content, m)) for it, m in zip(batch, mds)])
                    self.bookkeeping.clear_pending(scope, ids)
                except Exception:  # noqa: BLE001
                    self._state["fingerprints_trusted"] = False
                    self._mark_pending(scope, ids, "reconcile")
        try:
            pending_after = len(self.pending(scope=scope))
        except Exception:  # noqa: BLE001
            pending_after = None
        self._trace({"type": EVENT_INDEX_SYNC_RECONCILE, "items": succeeded,
                     "pending_after": pending_after})
        return succeeded

    def pending(self, *, scope=None) -> set:
        """Return pending ids across every exact footprint in a scope range."""
        try:
            exact_scopes = self.bookkeeping.exact_scopes(scope)
        except NotImplementedError:
            return self.bookkeeping.pending_ids(scope)
        if not exact_scopes:
            return self.bookkeeping.pending_ids(scope)
        result = set()
        for exact_scope in exact_scopes:
            result.update(self.bookkeeping.pending_ids(exact_scope))
        return result

    def tracked_ids(self, *, scope=None) -> set:
        """Delegate to bookkeeping: all ids with a registered fingerprint (already indexed) in this scope."""
        return self.bookkeeping.tracked_ids(scope)

    def exact_scopes(self, *, scope=None) -> set[Scope]:
        """Delegate exact-footprint enumeration to bookkeeping."""
        return self.bookkeeping.exact_scopes(scope)

    def close(self) -> None:
        """Close bookkeeping resources (delegates to bookkeeping.close; e.g. the SqliteBookkeeping DB connection)."""
        self.bookkeeping.close()
