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
    - best-effort (`index` / `drop`): on write failure it does not raise, marks the ids pending, and relies on
      read-time self-heal + reconcile to converge (eventual consistency). Memory uses this: its read path
      filters orphans and self-heals at read time, so it tolerates a brief index lag.
    - atomic and raise on failure (`replace`): the whole batch of old->new is replaced atomically, and on
      failure it raises so the caller can roll back its own source-of-truth side. RAG ingestion uses this: one
      document = one batch of new chunk_ids, all swapped or none (to prevent "old and new coexisting and both
      being hit"); on failure the caller must keep the old version.

Bookkeeping (fingerprints + pending set) has swappable storage via the small `SyncBookkeeping` seam: the
default is in-process (zero overhead, cleared on restart, and reconcile can fully rebuild from the
source-of-truth store); for cross-process idempotency / a pending set that survives, use `SqliteBookkeeping`
(a persistent bookkeeping table aligned with LangChain's RecordManager, installed by default by
`Memory.from_config` / `IngestionPipeline.from_config`). That table is also the landing spot for a future
async outbox implementation (a pending row = a pending event). For fully async / distributed, implement and
inject your own `IndexSync`, without changing a line of the subsystem write paths.
"""

import hashlib
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from ..core.clock import now_utc
from ..core.exceptions import RetrievalError
from ..core.sqlite_util import open_sqlite
from .scope import Scope
from .scope_sql import scope_column_names, scope_exact_where, scope_exact_where_clause, scope_store_values
from ..core.trace_events import EVENT_INDEX_SYNC_PENDING, EVENT_INDEX_SYNC_RECONCILE


def _digest(content: str, metadata: Optional[dict] = None) -> str:
    """Content fingerprint (sha1): used for idempotent upsert, so an unchanged content / metadata skips re-writing the index / embedding.

    When metadata is empty (None / {}) it falls back to a pure content hash, keeping the fingerprint unchanged
    for metadata-free paths like memory so an upgrade does not trigger a full re-embedding; when metadata is
    present it hashes the canonicalized JSON alongside, so a metadata change (e.g. the filterable column
    doc_id) triggers a rewrite instead of being short-circuited as "content unchanged"."""
    if not metadata:
        return hashlib.sha1(content.encode("utf-8")).hexdigest()
    raw = content + "\x00" + json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


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
        """Atomically replace old->new (whole batch, by-doc, with no "old and new coexisting" window). Raises on failure, so the caller rolls back its source-of-truth side and keeps the old version."""

    @abstractmethod
    def drop(self, ids: List[str], *, scope=None) -> None:
        """Delete several entries from the index. Best-effort: does not raise on failure, marks pending, and leaves it to reconcile / read-time self-heal."""

    @abstractmethod
    def reconcile(self, items: Sequence, *, scope=None, batch_size: int = 256) -> int:
        """Make the index converge with items (a source-of-truth snapshot; elements must have .id / .content) as the authority: drop orphans + force-reingest the source. Returns the number reingested."""

    @abstractmethod
    def pending(self, *, scope=None) -> set:
        """The currently pending ids (recent best-effort write failures not yet converged by reconcile). For apps to monitor / trigger reconciliation."""

    def tracked_ids(self, *, scope=None) -> set:
        """All ids currently indexed (fingerprint registered) in this scope, for consistency cross-checks (e.g. IngestionPipeline.verify).

        Optional capability: the default implementation SyncIndexSync delegates to bookkeeping; a custom
        IndexSync that cannot enumerate may leave it unoverridden (the verifier skips it on NotImplementedError).
        Non-abstract, so it does not force existing subclasses to implement it and does not break compatibility."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources held by this seam (e.g. the bookkeeping DB connection / worker). Default is a no-op: resource-free implementations need not override (non-abstract, does not break existing subclasses)."""


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

    def close(self) -> None:
        """Release resources held by bookkeeping (e.g. the SQLite connection). Default is a no-op: the in-process implementation (InMemoryBookkeeping) need not override."""


class InMemoryBookkeeping(SyncBookkeeping):
    """The default bookkeeping: an in-process dict (zero dependency, zero overhead). Cleared on restart, so the
    idempotent-skip and pending set are lost with it, but no data is lost: reconcile / rebuild can still fully
    rebuild from the source-of-truth store, only the first write after restart does one extra full embedding."""

    def __init__(self):
        self._hashes: dict = {}    # scope -> {id: fingerprint}
        self._pending: dict = {}   # scope -> {id: mark time} (the time aligns the semantics with the persistent implementation; here it only keeps the earliest value)

    def get_hash(self, scope, id):
        scope = scope or Scope()          # normalize None -> Scope(), aligned with SqliteBookkeeping (otherwise the two use different buckets and behavior drifts across backends)
        return self._hashes.get(scope, {}).get(id)

    def set_hashes(self, scope, pairs):
        scope = scope or Scope()
        h = self._hashes.setdefault(scope, {})
        for i, d in pairs:
            h[i] = d

    def delete_hashes(self, scope, ids):
        scope = scope or Scope()
        h = self._hashes.get(scope)
        if h:
            for i in ids:
                h.pop(i, None)

    def tracked_ids(self, scope):
        scope = scope or Scope()
        return set(self._hashes.get(scope, {}))

    def mark_pending(self, scope, ids):
        scope = scope or Scope()
        pend = self._pending.setdefault(scope, {})
        now = now_utc().isoformat()
        for i in ids:
            pend.setdefault(i, now)   # keep the earliest mark time (the "how long stuck" measure)

    def clear_pending(self, scope, ids):
        scope = scope or Scope()
        pend = self._pending.get(scope)
        if pend:
            for i in ids:
                pend.pop(i, None)

    def pending_ids(self, scope):
        scope = scope or Scope()
        return set(self._pending.get(scope, {}))


class SqliteBookkeeping(SyncBookkeeping):
    """Persistent bookkeeping (aligned with the de-facto standard form of LangChain's RecordManager): one SQLite table storing (scope, id, fingerprint, timestamp).

    Benefits: (1) idempotent skipping takes effect cross-process (unchanged content is no longer re-embedded
    after restart, mainly benefiting the index path of Memory's write path); (2) the pending set is not lost on
    restart (entries whose index write failed can still be converged by reconcile after restart); (3) the
    pending_since timestamp lets an app monitor "how long the oldest has been stuck". This table is naturally
    the seed of a future async outbox.
    """

    def __init__(self, db_path: str, *, table: str = "index_sync_bookkeeping"):
        """db_path: the SQLite file path (co-locating with the source-of-truth store is fine, since bookkeeping is companion state of the ingestion pipeline)."""
        from .backends.sqlite import ensure_safe_table   # lazy import: reuse table-name sanitization, avoiding a top-level retrieval<->backends cycle
        from ..core.sqlite_util import primary_key_columns
        self.table = ensure_safe_table(table)
        scope_cols = ", ".join(f"{c} TEXT" for c in scope_column_names())
        pk = ", ".join(["id", *scope_column_names()])
        body = (f"id TEXT, {scope_cols}, content_hash TEXT, updated_at TEXT, pending_since TEXT, "
                f"PRIMARY KEY ({pk})")
        self._lock = threading.RLock()  # serialize across threads (the async path reuses the connection via to_thread)
        try:
            self._db = open_sqlite(db_path)
            self._db.execute(f"CREATE TABLE IF NOT EXISTS {self.table}({body})")
            # Startup self-check (grade C: derived-data drift means drop and rebuild, the opposite of the source-of-truth store's "fail loud"): a primary-key drift (a Scope dimension added/removed) would make set_hashes's ON CONFLICT raise a cryptic "does not match any unique index" error. Bookkeeping is purely derived data (rebuild cost = one extra full embedding + loss of the pending set, the same as an InMemoryBookkeeping restart), so an old table with a mismatched primary key is dropped and rebuilt transparently to the user, with no need to raise for intervention.
            if primary_key_columns(self._db, self.table) != {"id", *scope_column_names()}:
                self._db.execute(f"DROP TABLE {self.table}")
                self._db.execute(f"CREATE TABLE {self.table}({body})")
            self._db.commit()
        except sqlite3.Error as e:
            raise RetrievalError(f"failed to open / initialize the index bookkeeping table: {e}") from e

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
        bookkeeping: optional bookkeeping storage; defaults to the in-process default if omitted (behaves as before).
        tracer: optional tracer (duck-typed emit); once attached, write failures (marked pending) and completed reconciliations emit index_sync events. Zero overhead when not attached."""
        self.retriever = retriever
        self.bookkeeping = bookkeeping if bookkeeping is not None else InMemoryBookkeeping()
        self.tracer = tracer

    def _trace(self, event: dict) -> None:
        """Emit one index_sync event (zero overhead when no tracer is attached)."""
        if self.tracer is not None:
            self.tracer.emit(event)

    def index(self, ids, contents, *, scope=None, metadatas=None) -> None:
        """Idempotent upsert: skip unchanged entries by content fingerprint; write to the foundation, and on failure mark this batch of ids pending, without raising or rolling back the source-of-truth store."""
        bk = self.bookkeeping
        mds = metadatas or [None] * len(ids)
        todo_ids, todo_contents, todo_mds, digests = [], [], [], []
        for i, c, m in zip(ids, contents, mds):
            d = _digest(c, m)
            if bk.get_hash(scope, i) == d:          # content / metadata unchanged: skip, avoiding a repeat embedding
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
        except Exception:                           # noqa: BLE001  source-of-truth is authoritative: an index failure does not roll back, marks pending for later convergence
            bk.mark_pending(scope, todo_ids)
            self._trace({"type": EVENT_INDEX_SYNC_PENDING, "op": "index", "count": len(todo_ids)})
            return
        bk.set_hashes(scope, list(zip(todo_ids, digests)))
        bk.clear_pending(scope, todo_ids)

    def replace(self, old_ids, new_ids, contents, *, scope=None, metadatas=None) -> None:
        """Atomically replace old->new (delegated to retriever.replace, with atomicity guaranteed by the foundation); raises on failure, does not touch bookkeeping, and the caller rolls back its source-of-truth side.

        On success, update bookkeeping: register the new chunks' fingerprints and remove the replaced-out old
        chunks (the old outside of new).
        """
        mkw = {"metadatas": metadatas} if metadatas else {}                    # pass only when given
        self.retriever.replace(old_ids, new_ids, contents, scope=scope, **mkw)   # raises here on failure, so the bookkeeping below does not run
        bk = self.bookkeeping
        new_set = set(new_ids)
        stale = [i for i in old_ids if i not in new_set]
        if stale:
            bk.delete_hashes(scope, stale)
        mds = metadatas or [None] * len(new_ids)                               # the fingerprint includes metadata, consistent with index/reconcile
        bk.set_hashes(scope, [(i, _digest(c, m)) for i, c, m in zip(new_ids, contents, mds)])
        bk.clear_pending(scope, list(old_ids) + list(new_ids))

    def drop(self, ids, *, scope=None) -> None:
        """Delete from the index; on failure mark this batch of ids pending (leaving orphans, with reconcile / read-time self-heal as backstop), without raising."""
        ids = list(ids)
        if not ids:
            return
        try:
            self.retriever.delete(ids, scope=scope)
        except Exception:                           # noqa: BLE001
            self.bookkeeping.mark_pending(scope, ids)
            self._trace({"type": EVENT_INDEX_SYNC_PENDING, "op": "drop", "count": len(ids)})
            return
        self.bookkeeping.delete_hashes(scope, ids)
        self.bookkeeping.clear_pending(scope, ids)

    def reconcile(self, items, *, scope=None, batch_size: int = 256) -> int:
        """Reconcile with the source-of-truth snapshot items as authority: first drop orphans (in bookkeeping but no longer in the source), then force-reingest the source in batches (ignoring fingerprints, because the index may be entirely lost / swapped to an empty backend, in which case fingerprints would wrongly judge "already indexed"). Returns the number reingested. Unregistered orphans are backstopped by read-time self-heal."""
        items = list(items)
        source_ids = {it.id for it in items}
        # Delete index residue "not in the source": already-indexed orphans (tracked - source) + ghost pending (ids marked pending but never successfully written, and then deleted from the source).
        # A ghost was originally neither in tracked nor touched by reconcile, so it would stay in pending() forever: fold it into drop too (a successful drop clears its fingerprint and pending).
        stale = list((self.bookkeeping.tracked_ids(scope) | self.bookkeeping.pending_ids(scope)) - source_ids)
        if stale:
            self.drop(stale, scope=scope)
        for k in range(0, len(items), batch_size):
            batch = items[k:k + batch_size]
            ids = [it.id for it in batch]
            mds = [getattr(it, "metadata", None) for it in batch]   # reingest with metadata (otherwise RAG's filterable columns like md_doc_id are wiped and filters silently return zero hits)
            try:
                mkw = {"metadatas": mds} if any(m for m in mds) else {}
                self.retriever.add(ids, [it.content for it in batch], scope=scope, **mkw)
            except Exception:                       # noqa: BLE001  a batch failed: mark pending, continue to the next batch
                self.bookkeeping.mark_pending(scope, ids)
                self._trace({"type": EVENT_INDEX_SYNC_PENDING, "op": "reconcile", "count": len(ids)})
                continue
            self.bookkeeping.set_hashes(scope, [(it.id, _digest(it.content, m)) for it, m in zip(batch, mds)])
            self.bookkeeping.clear_pending(scope, ids)
        self._trace({"type": EVENT_INDEX_SYNC_RECONCILE, "items": len(items),
                     "pending_after": len(self.bookkeeping.pending_ids(scope))})
        return len(items)

    def pending(self, *, scope=None) -> set:
        """A copy of the pending ids (read-only snapshot)."""
        return self.bookkeeping.pending_ids(scope)

    def tracked_ids(self, *, scope=None) -> set:
        """Delegate to bookkeeping: all ids with a registered fingerprint (already indexed) in this scope."""
        return self.bookkeeping.tracked_ids(scope)

    def close(self) -> None:
        """Close bookkeeping resources (delegates to bookkeeping.close; e.g. the SqliteBookkeeping DB connection)."""
        self.bookkeeping.close()
