"""Storage versioning regression (hermetic: in-memory SQLite, no key / no network).

Locks the "schema is a contract" open-time self-check primitives (agentmaker/core/sqlite_util.py) and each store's behavior on top of them: a drifted legacy primary key / unique constraint fails loudly, safe additive columns auto-ALTER, a change to a virtual table's locked-in parameters raises, and a mismatched checkpoint version is discarded.
"""

import sqlite3

import pytest

from agentmaker.core.exceptions import RetrievalError, SessionError
from agentmaker.core.sqlite_util import (
    column_names, ensure_columns, primary_key_columns, require_ddl_contains,
    require_primary_key, require_unique_columns, table_ddl, unique_column_sets,
)


def _db():
    return sqlite3.connect(":memory:")


# ---------- introspection primitives ----------

def test_introspection_primitives():
    """column_names / primary_key_columns / unique_column_sets / table_ddl read the schema correctly."""
    c = _db()
    c.execute("CREATE TABLE t(id TEXT, user TEXT, val TEXT, PRIMARY KEY(id, user))")
    c.execute("CREATE UNIQUE INDEX uq ON t(val)")
    assert column_names(c, "t") == {"id", "user", "val"}
    assert primary_key_columns(c, "t") == {"id", "user"}
    sets = unique_column_sets(c, "t")
    assert {"id", "user"} in sets and {"val"} in sets        # PK-derived unique + explicit UNIQUE INDEX
    assert "CREATE TABLE t" in table_ddl(c, "t")
    # nonexistent table: empty set / None, no raise
    assert column_names(c, "nope") == set() and primary_key_columns(c, "nope") == set()
    assert unique_column_sets(c, "nope") == [] and table_ddl(c, "nope") is None


# ---------- tier A: safe additive columns auto-ALTER ----------

def test_ensure_columns_adds_missing_only():
    """ensure_columns adds only missing business columns, idempotent, leaving existing columns untouched."""
    c = _db()
    c.execute("CREATE TABLE m(id TEXT, content TEXT)")
    ensure_columns(c, "m", {"content": "TEXT", "updated_at": "TEXT", "superseded_by": "TEXT"})
    assert column_names(c, "m") == {"id", "content", "updated_at", "superseded_by"}
    ensure_columns(c, "m", {"updated_at": "TEXT"})           # idempotent: calling again does not error or re-add
    assert column_names(c, "m") == {"id", "content", "updated_at", "superseded_by"}


# ---------- primary-key drift fails loud ----------

def test_require_primary_key_pass_and_drift():
    """Matching primary-key column set passes; drift (early single-column PK / missing scope dimension) raises with migration guidance."""
    c = _db()
    c.execute("CREATE TABLE memories(id TEXT, user TEXT, content TEXT, PRIMARY KEY(id, user))")
    require_primary_key(c, "memories", {"id", "user"}, error_cls=RetrievalError)   # match -> no raise
    # legacy DB with an early single-column PK
    c.execute("CREATE TABLE old(id TEXT PRIMARY KEY, user TEXT, content TEXT)")
    with pytest.raises(RetrievalError) as e:
        require_primary_key(c, "old", {"id", "user"}, error_cls=RetrievalError)
    assert "across scopes" in str(e.value) and "user" in str(e.value)


# ---------- unique-constraint drift fails loud ----------

def test_require_unique_columns_pass_and_drift():
    """A UNIQUE constraint covering the expected column set passes; its absence raises (cross-scope upsert risks mixing data)."""
    c = _db()
    c.execute("CREATE TABLE kv(user TEXT, key TEXT, value TEXT, UNIQUE(user, key))")
    require_unique_columns(c, "kv", {"user", "key"}, error_cls=SessionError)        # match -> no raise
    # legacy DB's unique constraint misses the new dimension (only UNIQUE(key))
    c.execute("CREATE TABLE kv_old(user TEXT, key TEXT, value TEXT, UNIQUE(key))")
    with pytest.raises(SessionError):
        require_unique_columns(c, "kv_old", {"user", "key"}, error_cls=SessionError)


# ---------- virtual-table locked-in params: DDL fragment check ----------

def test_require_ddl_contains_virtual_params():
    """Table DDL containing all expected fragments passes; a missing one (e.g. a dimension change) raises with rebuild guidance; a nonexistent table is skipped."""
    c = _db()
    # use a plain table as a stand-in (vec0 needs an extension absent in tests; the DDL-check logic is table-type agnostic)
    c.execute("CREATE TABLE vec_items(rowid INTEGER, embedding TEXT, user TEXT)")
    require_ddl_contains(c, "vec_items", ["embedding", "user"], error_cls=RetrievalError)   # contains -> no raise
    with pytest.raises(RetrievalError) as e:
        require_ddl_contains(c, "vec_items", ["float[1536]"], error_cls=RetrievalError,
                             hint="请用 Memory.rebuild_index 重建。")
    assert "locked in" in str(e.value) and "rebuild_index" in str(e.value)
    require_ddl_contains(c, "absent", ["x"], error_cls=RetrievalError)              # nonexistent table -> skipped, no raise


# ---------- per-store end to end: legacy drift -> loud error at open time ----------

def _seed(tmp_path, name, ddl):
    """Seed a file DB with a legacy-schema table and return its path (for a store to hit at open time)."""
    p = str(tmp_path / f"{name}.db")
    c = sqlite3.connect(p)
    c.execute(ddl)
    c.commit()
    c.close()
    return p


def test_memory_store_rejects_legacy_single_pk(tmp_path):
    """MemoryStore hitting a legacy DB with an early single-column PK (no scope dimension) -> RetrievalError (no silent data mixing)."""
    from agentmaker.memory.store import MemoryStore
    p = _seed(tmp_path, "mem", "CREATE TABLE memories(id TEXT PRIMARY KEY, content TEXT)")
    with pytest.raises(RetrievalError):
        MemoryStore(p)
    MemoryStore(str(tmp_path / "fresh.db"))             # a fresh DB builds fine, no error


def test_kv_store_rejects_drifted_unique(tmp_path):
    """KVStore hitting a legacy DB whose UNIQUE lacks the scope dimension -> RetrievalError (cross-scope upsert would mix data)."""
    from agentmaker.memory.kv import KVStore
    p = _seed(tmp_path, "kv", "CREATE TABLE kv(key TEXT, value TEXT, UNIQUE(key))")
    with pytest.raises(RetrievalError):
        KVStore(p)


def test_session_store_rejects_missing_scope_column(tmp_path):
    """SqliteSessionStore hitting a legacy DB missing the scope column -> SessionError (turns a later 'no such column' into a clear up-front error)."""
    from agentmaker.runtime.sessions import SqliteSessionStore
    p = _seed(tmp_path, "sess", "CREATE TABLE session_messages(role TEXT, content TEXT, created_at TEXT, metadata TEXT)")
    with pytest.raises(SessionError):
        SqliteSessionStore(p)


def test_source_store_rejects_legacy_pk(tmp_path):
    """SourceStore hitting a legacy single-column-PK chunks table -> RetrievalError (RAG can re-import from source)."""
    from agentmaker.rag.source_store import SourceStore
    p = _seed(tmp_path, "rag", "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, content TEXT)")
    with pytest.raises(RetrievalError):
        SourceStore(p)


def test_checkpoint_store_rejects_missing_scope_column(tmp_path):
    """SqliteCheckpointStore hitting a legacy DB missing the scope column -> SessionError with migration guidance (require_columns runs before the dedup DELETE)."""
    from agentmaker.runtime.execution.checkpoint import SqliteCheckpointStore
    p = _seed(tmp_path, "ckpt", "CREATE TABLE checkpoints(state TEXT, created_at TEXT)")
    with pytest.raises(SessionError, match="persistence contract"):                # clear error, not an opaque 'no such column'
        SqliteCheckpointStore(p)


def test_checkpoint_store_missing_scope_with_data_row_keeps_clear_error(tmp_path):
    """With the scope column missing **and a data row present**, the self-check still runs before the dedup DELETE (whose GROUP BY on scope would otherwise hit 'no such column' first), preserving the migration guidance."""
    from agentmaker.runtime.execution.checkpoint import SqliteCheckpointStore
    p = str(tmp_path / "ckpt2.db")
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE checkpoints(state TEXT, created_at TEXT)")
    c.execute("INSERT INTO checkpoints VALUES ('{}', '2026-01-01')")   # with a row present, the old DELETE...GROUP BY base would raise 'no such column' first
    c.commit()
    c.close()
    with pytest.raises(SessionError, match="persistence contract"):
        SqliteCheckpointStore(p)


# ---------- derived tables (tier C): drift -> drop+rebuild (no error); vec0 locked-in params raise ----------

def test_bookkeeping_self_heals_drifted_pk(tmp_path):
    """SqliteBookkeeping hitting a legacy single-column-PK table -> transparent drop+rebuild (derived data, no error); the rebuilt PK includes all scope dimensions."""
    from agentmaker.retrieval.index_sync import SqliteBookkeeping
    from agentmaker.retrieval.scope_sql import scope_column_names
    p = _seed(tmp_path, "bk", "CREATE TABLE index_sync_bookkeeping(id TEXT PRIMARY KEY, content_hash TEXT)")
    bk = SqliteBookkeeping(p)                            # no raise: derived data self-heals
    assert primary_key_columns(bk._db, "index_sync_bookkeeping") == {"id", *scope_column_names()}


def test_vec_store_rejects_dimension_drift(tmp_path):
    """SqliteVecStore hitting a legacy vec0 DB with a different dimension -> RetrievalError (dimension is locked in at first build and cannot be ALTERed)."""
    pytest.importorskip("sqlite_vec")                    # skip when the extension is absent
    from agentmaker.retrieval.backends.sqlite import SqliteVecStore
    p = str(tmp_path / "vec.db")
    SqliteVecStore(dim=8, db_path=p)                     # first build: float[8]
    with pytest.raises(RetrievalError) as e:
        SqliteVecStore(dim=16, db_path=p)               # same DB, 16 dims -> locked-in param mismatch
    assert "locked in" in str(e.value) and "rebuild_index" in str(e.value)


# ---------- session-store robustness (rollback / index / prune / circular refs) ----------

def test_session_store_has_scope_index():
    """session_messages has a scope composite index (load/clear/prune no longer full-scan once sessions grow)."""
    from agentmaker.runtime.sessions import SqliteSessionStore
    store = SqliteSessionStore()
    idx = store._db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='session_messages'").fetchall()
    assert any("idx_session_messages_scope" == r[0] for r in idx)


def test_session_store_append_rolls_back_on_failure():
    """On a failed append, the half-written transaction rolls back and the connection stays clean: later appends work and the failed row leaves no residue."""
    from agentmaker.runtime.sessions import SqliteSessionStore
    from agentmaker.core.message import Message
    from agentmaker.retrieval.scope import Scope
    store = SqliteSessionStore()
    sc = Scope(user="u")

    class _FlakyConn:
        def __init__(self, real):
            self._real = real
        def execute(self, sql, params=()):
            if sql.lstrip().upper().startswith("INSERT") and "boom" in params:
                raise sqlite3.OperationalError("模拟写失败")
            return self._real.execute(sql, params)
        def __getattr__(self, name):
            return getattr(self._real, name)

    store._db = _FlakyConn(store._db)
    with pytest.raises(SessionError):
        store.append(Message("boom", "user"), scope=sc)          # triggers the write failure -> rollback + SessionError
    store.append(Message("ok", "user"), scope=sc)                # connection is clean, later appends work
    assert [m.content for m in store.load(scope=sc)] == ["ok"]   # the failed row left no residue


def test_session_store_prune_keep_last_and_before():
    """prune: keep_last retains only the most recent N; before deletes anything earlier than a time; supplying neither raises."""
    from datetime import datetime, timezone
    from agentmaker.runtime.sessions import SqliteSessionStore
    from agentmaker.core.message import Message
    from agentmaker.retrieval.scope import Scope
    store = SqliteSessionStore()
    sc = Scope(user="u")
    for i in range(10):
        store.append(Message(f"m{i}", "user"), scope=sc)
    assert store.prune(scope=sc, keep_last=3) == 7
    assert [m.content for m in store.load(scope=sc)] == ["m7", "m8", "m9"]

    old = Scope(user="v")
    store.append(Message("old", "user", timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc)), scope=old)
    store.append(Message("new", "user", timestamp=datetime(2030, 1, 1, tzinfo=timezone.utc)), scope=old)
    assert store.prune(scope=old, before=datetime(2025, 1, 1, tzinfo=timezone.utc)) == 1
    assert [m.content for m in store.load(scope=old)] == ["new"]

    with pytest.raises(SessionError):
        store.prune(scope=sc)                                    # neither condition given -> raises (guards against accidentally wiping everything)


def test_session_store_prune_before_normalizes_timezones():
    """prune(before=) compares in canonical UTC order: after write-side ensure_utc normalization, non-UTC offsets / naive timestamps are not misjudged.

    created_at is TEXT and `< ?` is lexicographic, so without normalization there are two traps:
    1. 09:00 at +09:00 (= 00:00Z, before the cutoff) has '09..' < '05..' false lexically -> missed deletion;
    2. naive '..05:00:00' is a prefix of the cutoff '..05:00:00+00:00' -> judged smaller lexically, wrongly deleted (violating strict `<`).
    """
    from datetime import datetime, timezone, timedelta
    from agentmaker.runtime.sessions import SqliteSessionStore
    from agentmaker.core.message import Message
    from agentmaker.retrieval.scope import Scope
    store = SqliteSessionStore()

    # 1. offset timestamp: 09:00 at +09:00 = 00:00Z, before the 05:00Z cutoff -> should delete
    jst = timezone(timedelta(hours=9))
    sc1 = Scope(user="tz1")
    store.append(Message("early", "user", timestamp=datetime(2025, 1, 1, 9, 0, 0, tzinfo=jst)), scope=sc1)
    assert store.prune(scope=sc1, before=datetime(2025, 1, 1, 5, 0, 0, tzinfo=timezone.utc)) == 1
    assert store.load(scope=sc1) == []

    # 2. naive timestamp exactly equals the cutoff (naive treated as UTC) -> strict `<` does not delete
    sc2 = Scope(user="tz2")
    store.append(Message("boundary", "user", timestamp=datetime(2025, 1, 1, 5, 0, 0)), scope=sc2)
    assert store.prune(scope=sc2, before=datetime(2025, 1, 1, 5, 0, 0, tzinfo=timezone.utc)) == 0
    assert [m.content for m in store.load(scope=sc2)] == ["boundary"]


def test_session_store_circular_metadata_raises_session_error():
    """Circular reference in metadata (json.dumps raises ValueError) -> normalized to SessionError, no bare exception escapes."""
    from agentmaker.runtime.sessions import SqliteSessionStore
    from agentmaker.core.message import Message
    from agentmaker.retrieval.scope import Scope
    store = SqliteSessionStore()
    d = {}
    d["self"] = d
    with pytest.raises(SessionError):
        store.append(Message("x", "user", metadata=d), scope=Scope(user="u"))


def test_execution_state_circular_meta_raises_session_error():
    """ExecutionState.to_json on a circular reference (ValueError) is also normalized to SessionError."""
    from agentmaker.runtime.execution.state import ExecutionState
    d = {}
    d["self"] = d
    with pytest.raises(SessionError):
        ExecutionState(messages=[], input_text="x", meta={"bad": d}).to_json()
