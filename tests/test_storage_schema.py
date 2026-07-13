"""Storage schema-contract tests (hermetic: in-memory SQLite, no key or network).

Checks open-time schema validation, safe additive columns, locked virtual-table parameters, and
checkpoint format rejection.
"""

import importlib
import sqlite3

import pytest

from agentmaker.core.exceptions import RetrievalError, SessionError
from agentmaker.core.sqlite_util import (
    column_names, ensure_columns, primary_key_columns, require_ddl_contains,
    require_primary_key, require_unique_columns, table_ddl, unique_column_sets,
)


def _db():
    return sqlite3.connect(":memory:")


_INIT_CASES = [
    ("agentmaker.memory.kv", "KVStore", "require_unique_columns", RetrievalError),
    ("agentmaker.memory.store", "MemoryStore", "require_primary_key", RetrievalError),
    ("agentmaker.runtime.sessions", "SqliteSessionStore", "require_columns", SessionError),
    ("agentmaker.runtime.execution.checkpoint", "SqliteCheckpointStore", "require_columns", SessionError),
]

_SQLITE_INIT_CASES = [
    *_INIT_CASES,
    ("agentmaker.rag.source_store", "SourceStore", None, RetrievalError),
    ("agentmaker.retrieval.index_sync", "SqliteBookkeeping", None, RetrievalError),
]


class _InitConnection:
    def __init__(self, *, sqlite_failure=False):
        self.closed = False
        self.sqlite_failure = sqlite_failure

    def execute(self, *args, **kwargs):
        if self.sqlite_failure:
            raise sqlite3.OperationalError("sqlite init failed")
        return self

    def commit(self):
        pass

    def close(self):
        self.closed = True


@pytest.mark.parametrize("module_name,class_name,helper_name,error_type", _INIT_CASES)
def test_store_init_closes_connection_when_schema_validation_fails(
        monkeypatch, module_name, class_name, helper_name, error_type):
    module = importlib.import_module(module_name)
    connection = _InitConnection()

    def fail_schema(*args, **kwargs):
        raise error_type("schema failed")

    monkeypatch.setattr(module, "open_sqlite", lambda *args, **kwargs: connection)
    monkeypatch.setattr(module, helper_name, fail_schema)

    with pytest.raises(error_type, match="schema failed"):
        getattr(module, class_name)()

    assert connection.closed is True


@pytest.mark.parametrize("module_name,class_name,_helper_name,error_type", _SQLITE_INIT_CASES)
def test_store_init_normalizes_sqlite_errors_and_closes_connection(
        monkeypatch, module_name, class_name, _helper_name, error_type):
    module = importlib.import_module(module_name)
    connection = _InitConnection(sqlite_failure=True)
    monkeypatch.setattr(module, "open_sqlite", lambda *args, **kwargs: connection)

    with pytest.raises(error_type) as caught:
        getattr(module, class_name)(*(() if class_name != "SqliteBookkeeping" else (":memory:",)))

    assert isinstance(caught.value.__cause__, sqlite3.Error)
    assert connection.closed is True


def test_sqlite_bookkeeping_closes_connection_on_schema_helper_failure(monkeypatch):
    module = importlib.import_module("agentmaker.retrieval.index_sync")
    sqlite_util = importlib.import_module("agentmaker.core.sqlite_util")
    connection = _InitConnection()
    monkeypatch.setattr(module, "open_sqlite", lambda *args, **kwargs: connection)
    monkeypatch.setattr(
        sqlite_util, "primary_key_columns",
        lambda *args, **kwargs: (_ for _ in ()).throw(RetrievalError("schema failed")),
    )

    with pytest.raises(RetrievalError, match="schema failed"):
        module.SqliteBookkeeping(":memory:")

    assert connection.closed is True


def test_open_sqlite_closes_connection_when_a_pragma_fails(monkeypatch):
    from agentmaker.core.sqlite_util import open_sqlite

    connection = _InitConnection(sqlite_failure=True)
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: connection)

    with pytest.raises(sqlite3.OperationalError, match="sqlite init failed"):
        open_sqlite("database.db")

    assert connection.closed is True


@pytest.mark.parametrize(
    "class_name,args",
    [("SqliteVecStore", (3,)), ("Fts5KeywordIndex", ())],
)
def test_sqlite_index_init_failure_closes_only_owned_connections(
        monkeypatch, class_name, args):
    module = importlib.import_module("agentmaker.retrieval.backends.sqlite")
    cls = getattr(module, class_name)
    monkeypatch.setattr(
        cls, "_initialize",
        lambda *args, **kwargs: (_ for _ in ()).throw(RetrievalError("init failed")),
    )

    owned_connection = _InitConnection()
    monkeypatch.setattr(module, "open_sqlite", lambda *args, **kwargs: owned_connection)
    with pytest.raises(RetrievalError, match="init failed"):
        cls(*args)
    assert owned_connection.closed is True

    shared_connection = _InitConnection()
    with pytest.raises(RetrievalError, match="init failed"):
        cls(*args, connection=shared_connection)
    assert shared_connection.closed is False


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
    """A matching primary key passes while a scope-incomplete key raises with migration guidance."""
    c = _db()
    c.execute("CREATE TABLE memories(id TEXT, user TEXT, content TEXT, PRIMARY KEY(id, user))")
    require_primary_key(c, "memories", {"id", "user"}, error_cls=RetrievalError)   # match -> no raise
    c.execute("CREATE TABLE incompatible(id TEXT PRIMARY KEY, user TEXT, content TEXT)")
    with pytest.raises(RetrievalError) as e:
        require_primary_key(c, "incompatible", {"id", "user"}, error_cls=RetrievalError)
    assert "across scopes" in str(e.value) and "user" in str(e.value)


# ---------- unique-constraint drift fails loud ----------

def test_require_unique_columns_pass_and_drift():
    """A UNIQUE constraint covering the expected column set passes; its absence raises (cross-scope upsert risks mixing data)."""
    c = _db()
    c.execute("CREATE TABLE kv(user TEXT, key TEXT, value TEXT, UNIQUE(user, key))")
    require_unique_columns(c, "kv", {"user", "key"}, error_cls=SessionError)        # match -> no raise
    c.execute("CREATE TABLE kv_incompatible(user TEXT, key TEXT, value TEXT, UNIQUE(key))")
    with pytest.raises(SessionError):
        require_unique_columns(c, "kv_incompatible", {"user", "key"}, error_cls=SessionError)


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


# ---------- per-store end to end: incompatible schemas fail at open time ----------

def _seed(tmp_path, name, ddl):
    """Seed a file database with the supplied schema and return its path."""
    p = str(tmp_path / f"{name}.db")
    c = sqlite3.connect(p)
    c.execute(ddl)
    c.commit()
    c.close()
    return p


def test_memory_store_rejects_incomplete_primary_key(tmp_path):
    """MemoryStore rejects a primary key that omits ownership dimensions."""
    from agentmaker.memory.store import MemoryStore
    p = _seed(tmp_path, "mem", "CREATE TABLE memories(id TEXT PRIMARY KEY, content TEXT)")
    with pytest.raises(RetrievalError):
        MemoryStore(p)
    MemoryStore(str(tmp_path / "fresh.db"))             # a fresh DB builds fine, no error


def test_kv_store_rejects_drifted_unique(tmp_path):
    """KVStore rejects a UNIQUE constraint that omits ownership dimensions."""
    from agentmaker.memory.kv import KVStore
    p = _seed(tmp_path, "kv", "CREATE TABLE kv(key TEXT, value TEXT, UNIQUE(key))")
    with pytest.raises(RetrievalError):
        KVStore(p)


def test_session_store_rejects_missing_scope_column(tmp_path):
    """SqliteSessionStore reports a missing scope column during construction."""
    from agentmaker.runtime.sessions import SqliteSessionStore
    p = _seed(tmp_path, "sess", "CREATE TABLE session_messages(role TEXT, content TEXT, created_at TEXT, metadata TEXT)")
    with pytest.raises(SessionError):
        SqliteSessionStore(p)


def test_source_store_rejects_incomplete_primary_key(tmp_path):
    """SourceStore rejects a chunk primary key without ownership dimensions."""
    from agentmaker.rag.source_store import SourceStore
    p = _seed(tmp_path, "rag", "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, content TEXT)")
    with pytest.raises(RetrievalError):
        SourceStore(p)


def test_checkpoint_store_rejects_missing_scope_column(tmp_path):
    """SqliteCheckpointStore reports a missing scope column before running deduplication."""
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
    c.execute("INSERT INTO checkpoints VALUES ('{}', '2026-01-01')")
    c.commit()
    c.close()
    with pytest.raises(SessionError, match="persistence contract"):
        SqliteCheckpointStore(p)


# ---------- derived tables (tier C): drift -> drop+rebuild (no error); vec0 locked-in params raise ----------

def test_bookkeeping_self_heals_drifted_pk(tmp_path):
    """SqliteBookkeeping rebuilds derived state whose primary key omits scope dimensions."""
    from agentmaker.retrieval.index_sync import SqliteBookkeeping
    from agentmaker.retrieval.scope_sql import scope_column_names
    p = _seed(tmp_path, "bk", "CREATE TABLE index_sync_bookkeeping(id TEXT PRIMARY KEY, content_hash TEXT)")
    bk = SqliteBookkeeping(p)                            # no raise: derived data self-heals
    assert primary_key_columns(bk._db, "index_sync_bookkeeping") == {"id", *scope_column_names()}


def test_vec_store_rejects_dimension_drift(tmp_path):
    """SqliteVecStore rejects a vec0 table whose locked dimension differs."""
    pytest.importorskip("sqlite_vec")                    # skip when the extension is absent
    from agentmaker.retrieval.backends.sqlite import SqliteVecStore
    p = str(tmp_path / "vec.db")
    SqliteVecStore(dim=8, db_path=p)                     # first build: float[8]
    with pytest.raises(RetrievalError) as e:
        SqliteVecStore(dim=16, db_path=p)               # same DB, 16 dims -> locked-in param mismatch
    assert "locked in" in str(e.value) and "rebuild_index" in str(e.value)


# ---------- session-store robustness (rollback / index / prune / circular refs) ----------

def test_session_store_has_scope_index():
    """session_messages has a composite scope index for load, clear, and prune queries."""
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
