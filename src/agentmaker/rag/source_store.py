"""agentmaker.rag.source_store: RAG's source-of-truth store (the authoritative copy of chunk text).

The retrieval base only stores chunk_id + body for the sake of fast search; here we store the full
Chunk with doc_id / heading path / index, which is the authoritative copy: it can rebuild the index if
it gets corrupted, and it is also used to "manage add/delete of a whole document by doc_id" (the basis
for upsert dedup). It lands in SQLite and reuses the scope isolation from retrieval.scope: the primary
key includes the scope columns, so the same chunk_id does not overwrite across different scopes.
"""

import json
import os
import sqlite3
import threading
from typing import List, Optional

from ..core.exceptions import RetrievalError
from ..core.sqlite_util import open_sqlite, require_primary_key
from ..retrieval.scope import Scope
from ..retrieval.scope_sql import (scope_column_names, scope_exact_where,
                                    scope_from_store_values, scope_store_values,
                                    scope_where, scope_where_clause)
from .types import Chunk


class SourceStore:
    """RAG source-of-truth store: full Chunks stored into SQLite by (chunk_id, scope), recording doc_id to make per-document add/delete easy."""

    def __init__(self, db_path: str = ":memory:"):
        """
        Open a connection and create tables as needed.

        Args:
            db_path: SQLite file path; the default ":memory:" is for self-tests only, use a file path in
                production to persist.
        """
        scope_cols = scope_column_names()
        cols_ddl = ", ".join(f"{c} TEXT" for c in scope_cols)
        # Composite primary key (chunk_id, scope columns): consistent with the retrieval base's (id, scope) upsert, so the same id under different scopes does not overwrite.
        pk = ", ".join(["chunk_id", *scope_cols])
        self._coordination_key = (
            None if db_path == ":memory:" else f"rag:{os.path.realpath(db_path)}")
        self._lock = threading.RLock()  # cross-thread serialization: async retrieval reuses this connection via to_thread (the connection object itself is not thread-safe)
        try:
            self._db = open_sqlite(db_path)
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS chunks("
                f"chunk_id TEXT, doc_id TEXT, {cols_ddl}, "
                f"content TEXT, heading_path TEXT, idx INTEGER, metadata TEXT, "
                f"PRIMARY KEY ({pk}))")
            doc_pk = ", ".join(["doc_id", *scope_cols])
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS docs("
                f"doc_id TEXT, {cols_ddl}, content_hash TEXT, "
                f"PRIMARY KEY ({doc_pk}))")
            self._db.commit()
        except sqlite3.Error as e:
            db = getattr(self, "_db", None)
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
            raise RetrievalError(f"Failed to open / initialize the RAG source-of-truth store: {e}") from e
        try:
            self._verify_schema()
        except BaseException as schema_error:
            try:
                self._db.close()
            except BaseException as cleanup_error:
                schema_error.add_note(f"SourceStore construction cleanup also failed: {cleanup_error}")
            raise

    def _verify_schema(self) -> None:
        """Require ownership-complete primary keys in both authoritative tables."""
        with self._lock:
            require_primary_key(self._db, "chunks", {"chunk_id", *scope_column_names()}, error_cls=RetrievalError)
            require_primary_key(self._db, "docs", {"doc_id", *scope_column_names()}, error_cls=RetrievalError)

    def save_chunks(self, chunks: List[Chunk], *, scope: Optional[Scope] = None) -> None:
        """Store chunks in batch (overwrites if the same (chunk_id, scope) already exists; the same chunk_id under a different scope is unaffected)."""
        if not chunks:
            return
        sv = scope_store_values(scope or Scope())
        cols = ", ".join(scope_column_names())
        ph = ", ".join("?" for _ in scope_column_names())
        rows = [(c.chunk_id, c.doc_id, *sv, c.content, c.heading_path, c.index,
                 json.dumps(c.metadata, ensure_ascii=False)) for c in chunks]
        with self._lock:
            try:
                with self._db:
                    self._db.executemany(
                        f"INSERT OR REPLACE INTO chunks(chunk_id, doc_id, {cols}, "
                        f"content, heading_path, idx, metadata) "
                        f"VALUES (?, ?, {ph}, ?, ?, ?, ?)", rows)
            except sqlite3.Error as e:
                raise RetrievalError(f"Failed to write to the RAG source-of-truth store: {e}") from e

    def get(self, chunk_id: str, *, scope: Optional[Scope] = None) -> Optional[Chunk]:
        """Fetch the full chunk by (chunk_id, scope); returns None if it does not exist.

        scope uses B semantics (only filters non-empty dimensions), consistent with the retrieval base's
        search scope filtering: after a retrieval hit, fetch back with the same scope to avoid fetching a
        sibling chunk with the same chunk_id from a different scope.
        """
        where, params = scope_where(scope or Scope())
        with self._lock:
            cur = self._db.execute(
                f"SELECT chunk_id, doc_id, content, heading_path, idx, metadata FROM chunks "
                f"WHERE chunk_id = ?{where} LIMIT 2", (chunk_id, *params))
            rows = cur.fetchall()
        if len(rows) > 1:
            raise RetrievalError(
                f"source_store.get('{chunk_id}') matched multiple rows under the given scope; "
                "use a scope narrowed to one ownership footprint")
        return self._row_to_chunk(rows[0]) if rows else None

    def chunk_ids_of_doc(self, doc_id: str, *, scope: Optional[Scope] = None) -> List[str]:
        """List all chunk_ids of a document (within scope) (read-only, no delete). Lets the layer above get the ids to delete for "delete the index first, then the source of truth"."""
        where, params = scope_where(scope or Scope())
        with self._lock:
            cur = self._db.execute(
                f"SELECT chunk_id FROM chunks WHERE doc_id = ?{where}", (doc_id, *params))
            return [r[0] for r in cur.fetchall()]

    def chunk_ids_with_scopes_of_doc(
            self, doc_id: str, *, scope: Optional[Scope] = None) -> List[tuple[str, Scope]]:
        """List a document's chunk ids with their exact ownership footprints."""
        where, params = scope_where(scope or Scope())
        scope_cols = ", ".join(scope_column_names())
        with self._lock:
            rows = self._db.execute(
                f"SELECT chunk_id, {scope_cols} FROM chunks WHERE doc_id = ?{where}",
                (doc_id, *params)).fetchall()
        return [(row[0], scope_from_store_values(row[1:])) for row in rows]

    def chunk_ids_of_doc_exact(self, doc_id: str, *, scope: Optional[Scope] = None) -> List[str]:
        """List chunk ids stored in exactly one ownership footprint."""
        where, params = scope_exact_where(scope or Scope())
        with self._lock:
            rows = self._db.execute(
                f"SELECT chunk_id FROM chunks WHERE doc_id = ?{where}",
                (doc_id, *params)).fetchall()
        return [row[0] for row in rows]

    def scopes_of_doc(self, doc_id: str, *, scope: Optional[Scope] = None) -> set[Scope]:
        """Return exact footprints containing chunks or a fingerprint for a document."""
        where, params = scope_where(scope or Scope())
        scope_cols = ", ".join(scope_column_names())
        with self._lock:
            rows = self._db.execute(
                f"SELECT {scope_cols} FROM chunks WHERE doc_id = ?{where} "
                f"UNION SELECT {scope_cols} FROM docs WHERE doc_id = ?{where}",
                (doc_id, *params, doc_id, *params)).fetchall()
        return {scope_from_store_values(row) for row in rows}

    def delete_chunks(self, chunk_ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Precisely delete the given chunks by (chunk_id, scope) (all-dimension match, does not touch sibling rows with the same id under a different scope). Idempotent."""
        if not chunk_ids:
            return
        where, params = scope_exact_where(scope or Scope())
        with self._lock:
            try:
                with self._db:
                    self._db.executemany(
                        f"DELETE FROM chunks WHERE chunk_id = ?{where}",
                        [(cid, *params) for cid in chunk_ids])
            except sqlite3.Error as e:
                raise RetrievalError(f"Failed to delete RAG source-of-truth chunks: {e}") from e

    def list_docs(self, *, scope: Optional[Scope] = None) -> dict:
        """Return {doc_id: chunk count} (within the scope-limited range), for use by stats."""
        where, params = scope_where_clause(scope or Scope())
        with self._lock:
            cur = self._db.execute(f"SELECT doc_id, COUNT(*) FROM chunks{where} GROUP BY doc_id", params)
            return dict(cur.fetchall())

    def all_chunks(self, *, scope: Optional[Scope] = None) -> List[Chunk]:
        """Fetch all chunks in the scope (ordered by doc_id, idx): the realization of "the index can be rebuilt from the authoritative copy", for rebuild_index's full re-push.

        Mirrors MemoryStore.all: load all at once (local SQLite, manageable scale); batching happens on the
        index-write side of the re-push (reconcile batch_size).
        """
        return [chunk for chunk, _ in self.all_chunks_with_scopes(scope=scope)]

    def all_chunks_with_scopes(
            self, *, scope: Optional[Scope] = None) -> List[tuple[Chunk, Scope]]:
        """Fetch chunks together with their exact stored ownership footprints."""
        where, params = scope_where_clause(scope or Scope())
        scope_cols = ", ".join(scope_column_names())
        with self._lock:
            cur = self._db.execute(
                f"SELECT {scope_cols}, chunk_id, doc_id, content, heading_path, idx, metadata "
                f"FROM chunks{where} "
                f"ORDER BY doc_id, idx", params)
            rows = cur.fetchall()
        width = len(scope_column_names())
        return [
            (self._row_to_chunk(row[width:]), scope_from_store_values(row[:width]))
            for row in rows
        ]

    def get_doc_chunks(self, doc_id: str, *, index_range: Optional[tuple] = None,
                       scope: Optional[Scope] = None) -> List[Chunk]:
        """Fetch a document's chunks (ordered by idx); with index_range=(lo, hi), fetch only those whose index is in [lo, hi]: for neighbor-chunk / parent-chunk expansion.

        Args:
            doc_id: The document id.
            index_range: Optional closed interval (lo, hi); if omitted, fetch the whole document.
            scope: Ownership filter (B semantics); defaults to Scope().

        Returns:
            List[Chunk]: Ascending by in-document index.
        """
        where, params = scope_where(scope or Scope())
        rng = ""
        if index_range is not None:
            lo, hi = index_range
            rng = " AND idx >= ? AND idx <= ?"
            params = [*params, lo, hi]
        with self._lock:
            cur = self._db.execute(
                f"SELECT chunk_id, doc_id, content, heading_path, idx, metadata FROM chunks "
                f"WHERE doc_id = ?{where}{rng} ORDER BY idx", (doc_id, *params))
            rows = cur.fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # Doc-level content fingerprint (short-circuit the whole run when re-ingesting an unchanged document; see ingest.py)

    def get_doc_hash(self, doc_id: str, *, scope: Optional[Scope] = None) -> Optional[str]:
        """Read a document's ingestion fingerprint; returns None if there is none."""
        where, params = scope_exact_where(scope or Scope())
        with self._lock:
            row = self._db.execute(
                f"SELECT content_hash FROM docs WHERE doc_id = ?{where}", (doc_id, *params)).fetchone()
        return row[0] if row else None

    def set_doc_hash(self, doc_id: str, content_hash: str, *, scope: Optional[Scope] = None) -> None:
        """Register / overwrite a document's ingestion fingerprint (called after a successful ingest)."""
        sv = scope_store_values(scope or Scope())
        cols = ", ".join(scope_column_names())
        ph = ", ".join("?" for _ in scope_column_names())
        with self._lock:
            try:
                with self._db:
                    self._db.execute(
                        f"INSERT OR REPLACE INTO docs(doc_id, {cols}, content_hash) "
                        f"VALUES (?, {ph}, ?)", (doc_id, *sv, content_hash))
            except sqlite3.Error as e:
                raise RetrievalError(f"Failed to register document fingerprint: {e}") from e

    def delete_doc_hash(self, doc_id: str, *, scope: Optional[Scope] = None) -> None:
        """Delete a document's ingestion fingerprint (called when deleting the document). Idempotent."""
        where, params = scope_exact_where(scope or Scope())
        with self._lock:
            try:
                with self._db:
                    self._db.execute(
                        f"DELETE FROM docs WHERE doc_id = ?{where}", (doc_id, *params))
            except sqlite3.Error as e:
                raise RetrievalError(f"Failed to delete document fingerprint: {e}") from e

    def delete_doc_hashes_in_scope(self, doc_ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Delete document fingerprints across the range matched by a possibly coarse scope."""
        if not doc_ids:
            return
        where, params = scope_where(scope or Scope())
        with self._lock:
            try:
                with self._db:
                    self._db.executemany(
                        f"DELETE FROM docs WHERE doc_id = ?{where}",
                        [(doc_id, *params) for doc_id in doc_ids])
            except sqlite3.Error as e:
                raise RetrievalError(f"Failed to delete document fingerprints: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()

    @staticmethod
    def _row_to_chunk(row) -> Chunk:
        """Reconstruct a Chunk from a database row."""
        chunk_id, doc_id, content, heading_path, idx, metadata = row
        return Chunk(content=content, chunk_id=chunk_id, doc_id=doc_id, heading_path=heading_path,
                     index=idx, metadata=json.loads(metadata))
