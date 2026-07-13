"""agentmaker.retrieval.backends.sqlite: complete local SQLite retrieval backend.

A single file holding the full implementation of "local zero-ops retrieval":
    - SqliteBackend     : connection / transaction pipeline shared by both indexes (connect, lock, conditional commit / rollback)
    - ensure_safe_table : table-name identifier hygiene
    - SqliteVecStore    : vector store and nearest neighbor (sqlite-vec's vec0 virtual table)
    - Fts5KeywordIndex  : keyword retrieval (jieba tokenization + FTS5 + BM25)
    - SqliteHybridRetriever / build_sqlite_hybrid : both indexes share one connection -> add/delete is atomic across indexes in a single transaction

The abstract interfaces live one level up in `../base.py`, the orchestration base class in `../hybrid.py`, and scope
plus its SQL helpers in `../scope.py` / `../scope_sql.py` (shared across all of agentmaker). To use a non-SQLite
backend (pgvector / Qdrant...), write another file under backends/ implementing the interfaces in `../base.py`; the
framework core stays untouched.
"""

import asyncio
import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Sequence

from ...core.exceptions import RetrievalError
from ...core.sqlite_util import open_sqlite, require_ddl_contains
from ..base import Embedder, FusionStrategy, KeywordIndex, Reranker, VectorStore
from ..hybrid import HybridRetriever, require_valid_top_k
from ..scope import Scope, require_explicit_scope
from ..scope_sql import scope_column_names, scope_exact_where, scope_store_values, scope_where
from ..types import RetrievalConfig, RetrievalResult

# Valid SQL identifier: starts with a letter / underscore, contains only letters / digits / underscores.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Whether a jieba token "contains a word" (letter / digit / Chinese character); used to filter out pure-punctuation / whitespace fragments.
_WORD_RE = re.compile(r"\w", re.UNICODE)
# Fixed columns of both index tables (including the five scope columns): metadata column names must not collide with these.
_RESERVED_COLS = frozenset({"id", "content", "embedding", "tokens"} | set(scope_column_names()))


def ensure_safe_table(table: str) -> str:
    """Validate that the table name is a legal SQL identifier and return it unchanged, otherwise raise RetrievalError.

    The table name is interpolated into DDL / DML via f-string (placeholders can only bind values, not identifiers), so
    this is identifier hygiene: the table name is a developer configuration set at index-creation time, not user input,
    so the validation is to catch typo'd table names, not to prevent injection (all data goes through ? placeholders).
    """
    if not _IDENT_RE.fullmatch(table):
        raise RetrievalError(f"Illegal table name '{table}': only letters / digits / underscores allowed, and must not start with a digit.")
    return table


def _segment(text):
    """Tokenize text into words with jieba, filtering out pure punctuation / whitespace. jieba is imported lazily (the first call loads its dictionary)."""
    import jieba
    return [tok for tok in jieba.lcut(text) if _WORD_RE.search(tok)]


def _require_unique_ids(ids):
    """Ids must be unique within a single add batch: a duplicate id would bypass the "delete-then-insert" upsert (delete once, insert several) and pile up duplicate rows, so reject it outright."""
    seen = set()
    for id_ in ids:
        if id_ in seen:
            raise RetrievalError(f"add() ids must not repeat within a batch (duplicate id: '{id_}').")
        seen.add(id_)


def _scope_insert_parts(scope):
    """The three pieces needed to write one scope: column-name string, placeholder string, stored-value list (aligned with the column names)."""
    names = scope_column_names()
    return ", ".join(names), ", ".join("?" for _ in names), scope_store_values(scope)


# -- metadata filterable columns (pre-filtering): fields declared via metadata_columns= at index creation, materialized as md_<field> columns --

def _require_safe_metadata_columns(columns) -> tuple:
    """Validate the declared metadata field names: legal identifiers, no collision with fixed columns, no duplicates. Return a tuple (the same declaration is shared by table creation and filtering)."""
    cols = tuple(columns or ())
    seen = set()
    for key in cols:
        if not isinstance(key, str) or not _IDENT_RE.fullmatch(key):
            raise RetrievalError(f"Illegal metadata field name '{key}': only letters / digits / underscores allowed, and must not start with a digit.")
        if key in _RESERVED_COLS or f"md_{key}" in _RESERVED_COLS:
            raise RetrievalError(f"metadata field name '{key}' collides with an index fixed column; please rename.")
        if key in seen:
            raise RetrievalError(f"metadata_columns field '{key}' is duplicated.")
        seen.add(key)
    return cols


def _metadata_store_values(metadata, declared) -> list:
    """Take the declared field values from one metadata dict (missing / None -> empty string), stored uniformly as text (matching the text-comparison convention used by filters)."""
    md = metadata or {}
    return ["" if (v := md.get(key)) is None else str(v) for key in declared]


def _filters_where(filters, declared) -> tuple:
    """Compile a list of MetadataFilter into AND fragments and parameters (AND semantics; values compared as text, see MetadataFilter in types.py).

    Filtering on an undeclared field fails loud: a silent empty result is far harder to debug than an error.
    """
    if not filters:
        return "", []
    parts, params = [], []
    for f in filters:
        if f.key not in declared:
            raise RetrievalError(
                f"filter field '{f.key}' is not declared as a metadata column (this index declared {list(declared) or '(none)'}); "
                "declare it via metadata_columns=(...) at index creation before you can filter on it.")
        if f.op == "eq":
            parts.append(f"AND md_{f.key} = ?")
            params.append(str(f.value))
        else:                                       # "in" (filters.py already validated a non-empty sequence at construction)
            vals = list(f.value)
            parts.append(f"AND md_{f.key} IN ({', '.join('?' for _ in vals)})")
            params.extend(str(v) for v in vals)
    return " " + " ".join(parts), params


def _delete_exact(db, table, ids, scope):
    """Pre-upsert delete of old rows: match by (id, scope) across **all dimensions exactly**, deleting only rows with an identical footprint, never touching sibling rows with the same id but a different scope."""
    where, params = scope_exact_where(scope)
    db.executemany(f"DELETE FROM {table} WHERE id = ?{where}", [(id_, *params) for id_ in ids])


def _delete_scoped(db, table, ids, scope):
    """Used by public delete: delete by id, with scope taking the B semantics (filter only the non-empty dimensions)."""
    where, params = scope_where(scope)
    db.executemany(f"DELETE FROM {table} WHERE id = ?{where}", [(id_, *params) for id_ in ids])


class SqliteBackend:
    """A wrapper over one SQLite connection + one lock: manages an owned / shared connection and conditional commit / rollback.

    connection given = shared (same database, same connection as the other index): this object only writes without
    committing, leaving commit / rollback / close to the orchestrator (SqliteHybridRetriever collects both indexes'
    writes into a single transaction); not given = build an owned, exclusive connection and manage transactions itself.
    When lock is shared, both indexes pass the same one, serializing the whole connection (the connection object itself
    is not thread-safe).
    """

    def __init__(self, db_path: str = ":memory:", *, connection: Optional[sqlite3.Connection] = None,
                 lock: Optional[threading.RLock] = None):
        self._owns_conn = connection is None
        self._lock = lock or threading.RLock()
        try:
            self._db = connection or open_sqlite(db_path)  # reused across async retrieval via to_thread; connection params in open_sqlite
        except sqlite3.Error as e:
            raise RetrievalError(f"Failed to open SQLite: {e}") from e

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying sqlite connection, for table creation / loading extensions."""
        return self._db

    @contextmanager
    def transaction(self, error_msg: str):
        """Context for running a block of write operations under the lock: yields the connection for the caller to executemany.

        Owned connection: commit at block end, rollback on a sqlite error inside the block, and normalize the error to
        RetrievalError (prefixed with error_msg).
        Shared connection: only write without committing (commit / rollback are managed uniformly by
        SqliteHybridRetriever); on error still normalize to RetrievalError and re-raise.
        """
        with self._lock:
            try:
                yield self._db
                if self._owns_conn:
                    self._db.commit()
            except sqlite3.Error as e:
                if self._owns_conn:
                    self._db.rollback()
                raise RetrievalError(f"{error_msg}: {e}") from e

    def execute(self, sql: str, params=()):
        """Run one query under the lock and return all rows; **does not swallow / wrap exceptions**, since each store's read path has different error semantics and handles them itself."""
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def close(self) -> None:
        """Close the connection (only if owned; a shared connection is closed by the orchestrator)."""
        with self._lock:
            if self._owns_conn:
                self._db.close()


class SqliteVecStore(VectorStore):
    """Vector store implemented with sqlite-vec's vec0 virtual table; data lives in a single SQLite file (or in memory).

    Table structure: embedding vector column + id + one column per scope dimension + declared metadata columns
    (md_*, filterable in KNN) + content auxiliary column (stored but not filtered). Distance uses vec0's default L2;
    with already-normalized vectors like OpenAI's, L2 and cosine rank identically. The dimension is fixed at table
    creation; a model with a different dimension needs a new table / new database file. Connection / transaction / lock
    are delegated to SqliteBackend; this class only handles vec0-specific table creation and KNN.
    """

    def __init__(self, dim: int, db_path: str = ":memory:", *, table: str = "vec_items",
                 connection: Optional[sqlite3.Connection] = None, lock: Optional[threading.RLock] = None,
                 metadata_columns=()):
        """metadata_columns: declares which metadata fields are filterable (fixed at table-creation time, materialized as md_<field> columns); defaults to none."""
        self.dim = dim
        self.table = ensure_safe_table(table)
        self.metadata_columns = _require_safe_metadata_columns(metadata_columns)
        self._sql = SqliteBackend(db_path, connection=connection, lock=lock)
        try:
            self._initialize(self._sql.connection)
        except BaseException as initialization_error:
            try:
                self._sql.close()
            except BaseException as cleanup_error:
                initialization_error.add_note(f"Vector-store construction cleanup also failed: {cleanup_error}")
            raise

    def _initialize(self, conn) -> None:
        """Load sqlite-vec and create the configured table."""
        try:
            conn.enable_load_extension(True)
            import sqlite_vec  # lazy import: not installing this capability does not affect the rest of the framework
            sqlite_vec.load(conn)
        except ImportError as e:
            raise RetrievalError("Using the vector store requires installing sqlite-vec: uv add sqlite-vec") from e
        except (sqlite3.Error, AttributeError) as e:
            # AttributeError: some Python builds disable enable_load_extension
            raise RetrievalError(f"Failed to initialize the vector store (SQLite may not support loading extensions): {e}") from e
        finally:
            # Turn extension loading back off whether or not the load succeeded, so the failure path also does not leave a connection with "extension loading still enabled".
            try:
                conn.enable_load_extension(False)
            except (sqlite3.Error, AttributeError):
                pass  # extension loading was never enabled (e.g. this build disables enable_load_extension), so disabling it fails too; ignore
        scope_cols = ", ".join(f"{c} text" for c in scope_column_names())
        md_cols = "".join(f", md_{k} text" for k in self.metadata_columns)   # declared filterable metadata columns
        try:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} USING vec0("
                f"embedding float[{self.dim}], id text, {scope_cols}{md_cols}, +content text)")
        except sqlite3.Error as e:
            # Normalize to RetrievalError like Fts5KeywordIndex's CREATE (e.g. an illegal dimension float[0], a bad metadata column name)
            raise RetrievalError(f"Failed to create the vector table (is dimension {self.dim} valid?): {e}") from e
        require_ddl_contains(
            conn, self.table,
            [f"embedding float[{self.dim}]", *[f"{c} text" for c in scope_column_names()], *[f"md_{k} text" for k in self.metadata_columns]],
            error_cls=RetrievalError,
            hint="After swapping the embedding model (dimension change) or changing metadata_columns, rebuild via Memory.rebuild_index / IngestionPipeline.rebuild_index, or use a new database file.")

    def add(self, ids, vectors, contents, *, scope=None, metadatas=None):
        """Batch upsert: delete old rows by (id, scope) first, then insert; ids must be unique within a batch, and rewriting the same (id, scope) across batches just overwrites. Vectors are passed as JSON array text.

        metadatas (optional, same length as ids): each item's metadata dict; only declared fields are stored
        (metadata_columns, text-ified), the rest ignored.
        """
        if not (len(ids) == len(vectors) == len(contents)):
            raise RetrievalError("add() ids / vectors / contents must all be the same length.")
        if metadatas is not None and len(metadatas) != len(ids):
            raise RetrievalError("add() metadatas must be the same length as ids.")
        _require_unique_ids(ids)
        scope = scope or Scope()
        cols, placeholders, sv = _scope_insert_parts(scope)
        mds = metadatas or [None] * len(ids)
        md_cols = "".join(f", md_{k}" for k in self.metadata_columns)
        md_ph = "".join(", ?" for _ in self.metadata_columns)
        rows = [(json.dumps(vec), id_, *sv, *_metadata_store_values(md, self.metadata_columns), content)
                for id_, vec, content, md in zip(ids, vectors, contents, mds)]
        with self._sql.transaction("Failed to write vectors") as db:
            _delete_exact(db, self.table, ids, scope)
            db.executemany(f"INSERT INTO {self.table}(embedding, id, {cols}{md_cols}, content) "
                           f"VALUES (?, ?, {placeholders}{md_ph}, ?)", rows)

    def delete(self, ids, *, scope=None):
        """Batch delete by id (within the scope's range, B semantics). Idempotent. An empty scope deletes across the whole database; the destructive guardrail is at the HybridRetriever layer."""
        scope = scope or Scope()
        with self._sql.transaction("Failed to delete vectors") as db:
            _delete_scoped(db, self.table, ids, scope)

    def delete_exact(self, ids, *, scope=None):
        """Delete by write footprint **exactly** (all-dimension match, _delete_exact): used by add's compensating undo of just-written vectors, without wrongly deleting same-id sibling rows across scopes."""
        scope = scope or Scope()
        with self._sql.transaction("Failed to exactly delete vectors") as db:
            _delete_exact(db, self.table, ids, scope)

    def search(self, query_vector, *, top_k=5, scope=None, filters=None):
        """KNN retrieval: vec0's MATCH + k + scope (+ optional metadata filters); smaller distance = nearer, converted into a "larger = more relevant" score.

        The same query uses vec_to_json(embedding) to fetch the matched items' vectors back too (no extra query),
        filling embedding for upstream MMR reuse to avoid recomputation.
        """
        require_valid_top_k(top_k)
        scope = scope or Scope()
        where, params = scope_where(scope)
        fwhere, fparams = _filters_where(filters, self.metadata_columns)
        try:
            rows = self._sql.execute(
                f"SELECT id, content, distance, vec_to_json(embedding) FROM {self.table} "
                f"WHERE embedding MATCH ? AND k = {int(top_k)}{where}{fwhere} ORDER BY distance",
                (json.dumps(query_vector), *params, *fparams))
        except sqlite3.Error as e:
            raise RetrievalError(f"Vector retrieval failed: {e}") from e
        return [RetrievalResult(content=content, score=1.0 / (1.0 + distance),
                                source="vector", id=id_, embedding=json.loads(vec),
                                metadata={"distance": distance})
                for id_, content, distance, vec in rows]

    def close(self):
        """Close the database connection (a shared connection is closed by the orchestrator)."""
        self._sql.close()


class Fts5KeywordIndex(KeywordIndex):
    """Keyword index implemented with SQLite FTS5 + jieba tokenization + bm25 ranking.

    Before ingestion, jieba splits Chinese into words, joined by spaces and stored in the indexed tokens column; the
    original text goes into the UNINDEXED content column. This way FTS5's default unicode61 tokenizer only needs to
    split on spaces to get jieba's real words (unicode61 does not tokenize raw Chinese). Queries likewise jieba-tokenize
    first, OR-ing between words. Connection / transaction / lock are delegated to SqliteBackend; this class only handles
    FTS5-specific tokenization and table creation / querying.
    """

    def __init__(self, db_path: str = ":memory:", *, table: str = "kw_items",
                 connection: Optional[sqlite3.Connection] = None, lock: Optional[threading.RLock] = None,
                 metadata_columns=()):
        """metadata_columns: declares which metadata fields are filterable (materialized as md_<field> UNINDEXED columns); defaults to none, sharing the same declaration as the vector store."""
        self.table = ensure_safe_table(table)
        self.metadata_columns = _require_safe_metadata_columns(metadata_columns)
        self._sql = SqliteBackend(db_path, connection=connection, lock=lock)
        try:
            self._initialize()
        except BaseException as initialization_error:
            try:
                self._sql.close()
            except BaseException as cleanup_error:
                initialization_error.add_note(f"Keyword-index construction cleanup also failed: {cleanup_error}")
            raise

    def _initialize(self) -> None:
        """Create and validate the configured FTS5 table."""
        scope_cols = ", ".join(f"{c} UNINDEXED" for c in scope_column_names())
        md_cols = "".join(f", md_{k} UNINDEXED" for k in self.metadata_columns)
        try:
            self._sql.connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} USING fts5("
                f"tokens, content UNINDEXED, id UNINDEXED, {scope_cols}{md_cols}, tokenize='unicode61')")
        except sqlite3.Error as e:
            raise RetrievalError(f"Failed to open / initialize the keyword index (SQLite may be compiled without FTS5): {e}") from e
        require_ddl_contains(
            self._sql.connection, self.table,
            ["tokenize='unicode61'", *[f"{c} UNINDEXED" for c in scope_column_names()], *[f"md_{k} UNINDEXED" for k in self.metadata_columns]],
            error_cls=RetrievalError,
            hint="After changing metadata_columns / the tokenizer, rebuild via IngestionPipeline.rebuild_index, or use a new database file.")

    def add(self, ids, contents, *, scope=None, metadatas=None):
        """Batch upsert: delete old rows by (id, scope) first, then insert; ids must be unique within a batch. The tokens column stores jieba tokens, the content column stores the original text.

        metadatas (optional, same length as ids): each item's metadata dict; only declared fields are stored
        (metadata_columns, text-ified), the rest ignored.
        """
        if len(ids) != len(contents):
            raise RetrievalError("add() ids / contents must be the same length.")
        if metadatas is not None and len(metadatas) != len(ids):
            raise RetrievalError("add() metadatas must be the same length as ids.")
        _require_unique_ids(ids)
        scope = scope or Scope()
        cols, placeholders, sv = _scope_insert_parts(scope)
        mds = metadatas or [None] * len(ids)
        md_cols = "".join(f", md_{k}" for k in self.metadata_columns)
        md_ph = "".join(", ?" for _ in self.metadata_columns)
        rows = [(" ".join(_segment(c)), c, id_, *sv, *_metadata_store_values(md, self.metadata_columns))
                for id_, c, md in zip(ids, contents, mds)]
        with self._sql.transaction("Failed to write the keyword index") as db:
            _delete_exact(db, self.table, ids, scope)
            db.executemany(f"INSERT INTO {self.table}(tokens, content, id, {cols}{md_cols}) "
                           f"VALUES (?, ?, ?, {placeholders}{md_ph})", rows)

    def delete(self, ids, *, scope=None):
        """Batch delete by id (within the scope's range, B semantics). Idempotent. An empty scope deletes across the whole database; the destructive guardrail is at the HybridRetriever layer."""
        scope = scope or Scope()
        with self._sql.transaction("Failed to delete from the keyword index") as db:
            _delete_scoped(db, self.table, ids, scope)

    def delete_exact(self, ids, *, scope=None):
        """Delete ids from one exact ownership footprint."""
        scope = scope or Scope()
        with self._sql.transaction("Failed to exactly delete from the keyword index") as db:
            _delete_exact(db, self.table, ids, scope)

    def search(self, query, *, top_k=5, scope=None, filters=None):
        """jieba-tokenize -> OR between words -> FTS5 + bm25 retrieval (narrowed by scope + optional metadata filters); smaller bm25 = more relevant, negated to satisfy the "larger = more relevant" convention."""
        require_valid_top_k(top_k)
        scope = scope or Scope()
        tokens = _segment(query)
        if not tokens:
            return []  # the query tokenizes to nothing, treated as no keyword hit
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)  # wrap each word as a phrase and escape it, avoiding FTS5 query syntax
        where, params = scope_where(scope)
        fwhere, fparams = _filters_where(filters, self.metadata_columns)
        try:
            rows = self._sql.execute(
                f"SELECT id, content, bm25({self.table}) AS score FROM {self.table} "
                f"WHERE {self.table} MATCH ?{where}{fwhere} ORDER BY bm25({self.table}) LIMIT {int(top_k)}",
                (match, *params, *fparams))
        except sqlite3.OperationalError as e:
            # Only treat errors like "malformed FTS5 query syntax / MATCH" as no hit and return [] (each token is already
            # escaped, so this rarely triggers). A locked database / missing table / disk corruption is also an
            # OperationalError, but must not be silently swallowed: identify by message, then normalize to RetrievalError and raise.
            msg = str(e).lower()
            if "fts5" in msg or "syntax error" in msg or "malformed match" in msg:
                return []
            raise RetrievalError(f"Keyword retrieval failed: {e}") from e
        except sqlite3.Error as e:
            raise RetrievalError(f"Keyword retrieval failed: {e}") from e
        return [RetrievalResult(content=content, score=-bm, source="keyword", id=id_, metadata={"bm25": bm})
                for id_, content, bm in rows]

    def close(self):
        """Close the database connection (a shared connection is closed by the orchestrator)."""
        self._sql.close()


class SqliteHybridRetriever(HybridRetriever):
    """Hybrid retriever sharing a single connection: add / delete collect both indexes' writes into a **single cross-index transaction** for atomic commit.

    Stronger than the base HybridRetriever's best-effort compensation: any failed step rolls back the whole thing, with
    no half-write, and a failed update does not lose the old value either (vec0 and FTS5 share a connection and a
    transaction, committing / rolling back together). Constructed by build_sqlite_hybrid (both indexes share one
    connection + one lock).
    """

    def __init__(self, embedder, vector_store, keyword_index, connection, lock, *, reranker=None, config=None,
                 fusion=None):
        super().__init__(embedder, vector_store, keyword_index, reranker=reranker, config=config, fusion=fusion)
        self._conn = connection
        self._lock = lock

    def add(self, ids, contents, *, scope=None, metadatas=None):
        """Both indexes' writes + one commit wrapped in the same transaction; any failed step rolls back the whole thing (a failed overwrite of an existing id also does not lose the old value)."""
        if len(ids) != len(contents):
            raise RetrievalError("add() ids / contents must be the same length.")
        vectors = self.embedder.embed(contents)
        with self._lock:
            try:
                self.vector_store.add(ids, vectors, contents, scope=scope, metadatas=metadatas)   # shared connection, the store does not commit internally
                self.keyword_index.add(ids, contents, scope=scope, metadatas=metadatas)
                self._conn.commit()                                          # both indexes committed at once -> atomic
            except Exception:
                self._conn.rollback()
                raise

    def replace(self, old_ids, new_ids, contents, *, scope=None, metadatas=None):
        """Ingesting new chunks + deleting old chunks (those not in new) collected into the **same transaction** for atomic commit: retrieval sees either all-old or all-new, eliminating the concurrency window where "old and new coexist and are hit together (duplicate hits)" and any leftover from a failed old-chunk delete. On failure, roll back the whole thing and keep the old version (no data loss)."""
        # Reject an empty scope early: deleting stale ids with an empty scope would delete these ids across all scopes (this class must guard even when called directly by external code); reject before any write.
        require_explicit_scope(scope or Scope(), False, "replace")
        if len(new_ids) != len(contents):
            raise RetrievalError("replace() new_ids / contents must be the same length.")
        stale = [i for i in old_ids if i not in set(new_ids)]
        vectors = self.embedder.embed(contents)                          # network call, kept outside the lock
        with self._lock:
            try:
                self.vector_store.add(new_ids, vectors, contents, scope=scope, metadatas=metadatas)   # shared connection, the store does not commit internally
                self.keyword_index.add(new_ids, contents, scope=scope, metadatas=metadatas)
                if stale:
                    self.vector_store.delete_exact(stale, scope=scope)
                    self.keyword_index.delete_exact(stale, scope=scope)
                self._conn.commit()                                      # loading new + deleting old committed at once -> atomic replace
            except Exception:
                self._conn.rollback()
                raise

    def delete(self, ids, *, scope=None, all_scopes=False):
        """Both indexes' deletes wrapped in the same transaction for atomic commit. Empty-scope guardrail same as the base class (see scope.py)."""
        require_explicit_scope(scope or Scope(), all_scopes, "delete")
        with self._lock:
            try:
                self.vector_store.delete(ids, scope=scope)
                self.keyword_index.delete(ids, scope=scope)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def delete_exact(self, ids, *, scope=None):
        """Delete one exact ownership footprint from both indexes atomically."""
        with self._lock:
            try:
                self.vector_store.delete_exact(ids, scope=scope)
                self.keyword_index.delete_exact(ids, scope=scope)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # -- The read a* methods are overridden to wrap the sync versions in a single to_thread: both indexes share one
    #    connection + one lock, so the base class's "vector path / keyword path gather concurrency" would only wait on
    #    each other under a single lock, with no gain and extra thread hopping, so running the sync search entirely under
    #    one to_thread is cheaper. The write a* methods (aadd/areplace/adelete) inherit the base class's
    #    to_thread(self.add/replace/delete): self.* is this class's cross-index single-transaction atomic version, run
    #    entirely in the thread with full commit/rollback, so atomicity is not lost.
    async def asearch(self, query, **kwargs):
        """asearch wraps the sync search in a single to_thread (two-path gather has no gain under the shared connection lock)."""
        return await asyncio.to_thread(lambda: self.search(query, **kwargs))

    async def asearch_many(self, queries, **kwargs):
        """asearch_many wraps the sync search_many in a single to_thread."""
        return await asyncio.to_thread(lambda: self.search_many(queries, **kwargs))

    def close(self):
        """Close both stores (a no-op under a shared connection), then close the shared connection uniformly."""
        super().close()
        self._conn.close()


def _check_embedder_fingerprint(conn, vec_table: str, embedder: Embedder) -> None:
    """Index <-> embedder fingerprint check: the index_meta table records "model name | dimension" per vector table, compared on open.

    Vector spaces of different embedding models are not comparable: swapping to "a different model of the same
    dimension" without a fingerprint would **silently mix writes** (new query vectors are incomparable with old stored
    vectors, retrieval silently degrades, with no error at all). Rule: if the dimension differs, or both sides have a
    model name and they differ -> reject and prompt a full rebuild; if either side's model name is unknown
    (model_id=None on a custom Embedder) -> compare only the dimension, and fill the record in to a more specific
    fingerprint.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS index_meta(key TEXT PRIMARY KEY, value TEXT)")
    meta_key = f"{vec_table}:embedder"
    cur_model = getattr(embedder, "model_id", None) or ""
    current = f"{cur_model}|{embedder.dim}"
    row = conn.execute("SELECT value FROM index_meta WHERE key = ?", (meta_key,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO index_meta(key, value) VALUES (?, ?)", (meta_key, current))
        conn.commit()
        return
    stored_model, _, stored_dim = row[0].partition("|")
    if stored_dim != str(embedder.dim) or (stored_model and cur_model and stored_model != cur_model):
        raise RetrievalError(
            f"Index '{vec_table}' was built by embedder '{stored_model or 'unknown model'}' ({stored_dim} dims), "
            f"not matching the current '{cur_model or 'unknown model'}' ({embedder.dim} dims): vector spaces of different models are not comparable, and mixing writes silently degrades retrieval. "
            "Switch back to the original model; or use a new database / new table and rebuild fully (memory uses Memory.rebuild_index, RAG uses IngestionPipeline.rebuild_index).")
    if not stored_model and cur_model:   # old record missing the model name, dimension matches: fill it in to a more specific fingerprint (so a same-dimension model swap can be caught next time)
        conn.execute("UPDATE index_meta SET value = ? WHERE key = ?", (current, meta_key))
        conn.commit()


def build_sqlite_hybrid(embedder: Embedder, *, db_path: str = ":memory:", reranker: Optional[Reranker] = None,
                        vec_table: str = "vec_items", kw_table: str = "kw_items",
                        config: Optional[RetrievalConfig] = None, metadata_columns: Sequence[str] = (),
                        fusion: Optional[FusionStrategy] = None) -> SqliteHybridRetriever:
    """Convenience constructor: the vector store + keyword index share one SQLite connection -> add / delete is atomic across both indexes in a single transaction.

    The recommended local construction: compared to two stores each with their own exclusive connection (where writes
    can only be best-effort compensated), a shared connection makes both indexes' writes either succeed together or roll
    back together, with no half-write, and a failed update does not lose the old value. On open it does an embedder
    fingerprint check (index_meta table): swapping in a mismatched embedding model errors out directly, preventing
    "silent mixed writes, degraded retrieval".

    Args:
        embedder: Text-to-vector (embedder.dim builds the vector table; model_id goes into the fingerprint).
        db_path: SQLite file path; defaults to ":memory:" for self-tests only, use a file path in production to persist.
        reranker: Optional reranker.
        vec_table: The vector index table name.
        kw_table: The keyword index table name in the same database.
        metadata_columns: Declares which metadata fields are filterable (e.g. ("doc_id", "tag")), shared by both
            indexes; defaults to none. Fixed at table-creation time; adding fields later requires a new table rebuild.
        fusion: Optional fusion strategy (FusionStrategy); if omitted, use the RRF default.

    Returns:
        SqliteHybridRetriever: A hybrid retriever wired to a shared connection with atomic add / delete.
    """
    conn = open_sqlite(db_path)
    lock = threading.RLock()
    try:
        vector_store = SqliteVecStore(dim=embedder.dim, table=vec_table, connection=conn, lock=lock,
                                      metadata_columns=metadata_columns)
        keyword_index = Fts5KeywordIndex(table=kw_table, connection=conn, lock=lock,
                                         metadata_columns=metadata_columns)
        # Write the fingerprint only after both stores construct successfully: a construction failure (missing sqlite-vec / FTS5 not compiled) must not leave "fingerprint written but an actually empty store", a false rejection.
        _check_embedder_fingerprint(conn, ensure_safe_table(vec_table), embedder)
    except BaseException as construction_error:
        try:
            conn.close()
        except BaseException as cleanup_error:
            construction_error.add_note(f"SQLite hybrid construction cleanup also failed: {cleanup_error}")
        raise
    return SqliteHybridRetriever(embedder, vector_store, keyword_index, conn, lock,
                                 reranker=reranker, config=config, fusion=fusion)
