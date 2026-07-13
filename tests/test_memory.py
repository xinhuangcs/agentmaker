"""Hermetic memory subsystem tests using SQLite and fake retrieval backends."""

import sqlite3
import threading
import time

import pytest

from agentmaker.core.exceptions import RetrievalError
from agentmaker.retrieval.hybrid import require_valid_top_k
from agentmaker.memory import Memory, MemoryStore
from agentmaker.memory.kv import KVMemory, KVStore
from agentmaker.memory.smart_writer import SmartWriter
from agentmaker.memory.memory_tool import MemoryTool
from agentmaker.memory.types import MemoryItem
from agentmaker.retrieval.scope import Scope
from agentmaker.runtime.execution.run_context import reset_run, start_run

ALICE = Scope(base="memory", user="alice")
BOB = Scope(base="memory", user="bob")


class _Hit:
    """Fake retrieval hit: Memory.search only touches id / score / embedding."""

    def __init__(self, id, score=1.0):
        self.id = id
        self.score = score
        self.embedding = None


class FakeRetriever:
    """Controllable fake retriever: records (id -> scope), search filters by exact scope, delete really removes. Enough to test isolation / compensation / lazy cleanup."""

    def __init__(self):
        self.docs = {}  # id -> scope

    def add(self, ids, contents, *, scope=None):
        for i in ids:
            self.docs[i] = scope

    def delete(self, ids, *, scope=None):
        for i in ids:
            self.docs.pop(i, None)

    delete_exact = delete

    def search(self, query, *, top_k=5, candidate_pool=20, scope=None):
        require_valid_top_k(top_k, candidate_pool=candidate_pool)  # same validation as the real backend, so the fake isn't laxer and can't mask candidate-pool bugs
        return [_Hit(i) for i, sc in self.docs.items() if sc == scope][:top_k]

    def close(self):
        pass


class BoomAddRetriever(FakeRetriever):
    """add always raises; used to trigger Memory.add's index-failure pending-mark path (without rolling back the source of truth)."""

    def add(self, ids, contents, *, scope=None):
        raise RuntimeError("index add boom")


class FlakyAddRetriever(FakeRetriever):
    """The first fail_times adds raise, then succeed; verifies rebuild_index reconciles pending items back into the index."""

    def __init__(self, fail_times=1):
        super().__init__()
        self.fail_times = fail_times

    def add(self, ids, contents, *, scope=None):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("index add boom (flaky)")
        super().add(ids, contents, scope=scope)


def _memory(retriever=None):
    return Memory(retriever=retriever or FakeRetriever(), store=MemoryStore(), scope=ALICE)


# ---------- timestamps normalized to aware UTC ----------

def test_memory_item_normalizes_naive_timestamp():
    """MemoryItem normalizes naive datetimes to aware UTC (prevents a TypeError when subtracting a naive time from an aware now in recency / expiry scoring)."""
    from datetime import datetime
    item = MemoryItem(content="x", created_at=datetime(2020, 1, 1), updated_at=datetime(2020, 1, 2))
    assert item.created_at.tzinfo is not None and item.updated_at.tzinfo is not None
    assert item.last_accessed_at is None and item.invalid_at is None      # None stays None (not fabricated)
    assert MemoryItem(content="y").created_at.tzinfo is not None          # the default factory is aware too


# ---------- consolidate: soft-invalidate + mean importance ----------

def test_consolidate_soft_invalidates_and_averages_importance():
    """consolidate: the merged item's importance is the mean (not the global max, so repeated consolidation can't
    ratchet the score to the ceiling); old items are soft-invalidated for the record (not physically deleted,
    auditable), and invalid items stay out of store.all's default result (a second consolidate won't re-feed them)."""
    m = _memory()
    m.add("事实A", importance=0.2)
    m.add("事实B", importance=0.8)
    assert m._apply_consolidate(m.store.all(scope=ALICE), ["合并后的事实"]) == {"before": 2, "after": 1}
    valid = m.store.all(scope=ALICE)                                       # default holds valid items only
    assert len(valid) == 1 and valid[0].content == "合并后的事实"
    assert abs(valid[0].importance - 0.5) < 1e-9                           # mean (0.2+0.8)/2, not max 0.8
    assert len(m.store.all(scope=ALICE, include_invalid=True)) == 3        # the old 2 are soft-invalidated but retained (physical, auditable)


# ---------- resource close chain doesn't leak ----------

def test_index_sync_close_delegates_to_bookkeeping():
    """SyncIndexSync.close delegates to bookkeeping.close (plugs the connection leak from from_config's default SqliteBookkeeping); the default no-op close doesn't raise."""
    from agentmaker.retrieval.index_sync import InMemoryBookkeeping, SyncIndexSync
    closed = []
    bk = InMemoryBookkeeping()
    bk.close = lambda: closed.append(1)
    SyncIndexSync(FakeRetriever(), bookkeeping=bk).close()
    assert closed == [1]                                                   # close delegated to bookkeeping
    SyncIndexSync(FakeRetriever()).close()                                 # default InMemoryBookkeeping: no-op close, must not raise


def test_memory_close_respects_injected_resource_ownership():
    from agentmaker.retrieval.index_sync import SyncIndexSync

    retriever = FakeRetriever()
    store = MemoryStore()
    closed = []
    store_close = store.close
    retriever.close = lambda: closed.append("retriever")
    store.close = lambda: closed.append("store")
    sync = SyncIndexSync(retriever)
    sync.close = lambda: closed.append("sync")

    Memory(retriever, store, scope=ALICE, index_sync=sync).close()

    assert closed == []
    store_close()


def test_memory_close_releases_only_resources_created_by_from_config():
    from agentmaker import AgentmakerConfig

    retriever = FakeRetriever()
    store = MemoryStore()
    retriever_closed = []
    store_closed = []
    sync_closed = []
    store_close = store.close
    retriever.close = lambda: retriever_closed.append(True)
    store.close = lambda: store_closed.append(True)
    memory = Memory.from_config(AgentmakerConfig(), retriever=retriever, store=store)
    memory._sync.close = lambda: sync_closed.append(True)

    memory.close()
    memory.close()

    assert sync_closed == [True]
    assert retriever_closed == []
    assert store_closed == []
    store_close()


def test_memory_close_releases_the_default_from_config_stack():
    from agentmaker import AgentmakerConfig

    class Embedder:
        dim = 3

        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    memory = Memory.from_config(AgentmakerConfig(), embedder=Embedder())
    closed = []
    sync_close = memory._sync.close
    retriever_close = memory.retriever.close
    store_close = memory.store.close
    memory._sync.close = lambda: closed.append("sync")
    memory.retriever.close = lambda: closed.append("retriever")
    memory.store.close = lambda: closed.append("store")

    memory.close()

    assert closed == ["sync", "retriever", "store"]
    sync_close()
    retriever_close()
    store_close()


def test_memory_context_cleanup_does_not_mask_business_error():
    memory = _memory()
    memory._owns_sync = True
    memory._sync.close = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))

    with pytest.raises(ValueError, match="business failed") as caught:
        with memory:
            raise ValueError("business failed")

    assert any("Memory cleanup also failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("kind", ["kv_store", "kv_memory", "memory_store"])
def test_storage_context_cleanup_does_not_mask_business_error(kind):
    if kind == "kv_store":
        resource = KVStore()
        original_close = resource.close
        resource.close = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
    elif kind == "kv_memory":
        kv = KVStore()
        resource = KVMemory(kv)
        original_close = kv.close
        kv.close = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
    else:
        resource = MemoryStore()
        original_close = resource.close
        resource.close = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))

    with pytest.raises(ValueError, match="business failed") as caught:
        with resource:
            raise ValueError("business failed")

    assert any("close failed" in note for note in caught.value.__notes__)
    original_close()


# ---------- source-of-truth composite key: same id across scopes doesn't overwrite ----------

def test_store_same_id_across_scopes_coexist():
    st = MemoryStore()
    st.save(MemoryItem(content="alice 的", id="X"), scope=ALICE)
    st.save(MemoryItem(content="bob 的", id="X"), scope=BOB)         # same id, different scope
    assert st.get("X", scope=ALICE).content == "alice 的"
    assert st.get("X", scope=BOB).content == "bob 的"               # both coexist, neither overwrites
    assert [m.content for m in st.all(scope=ALICE)] == ["alice 的"]


def test_store_same_id_same_scope_overwrites():
    st = MemoryStore()
    st.save(MemoryItem(content="v1", id="X"), scope=ALICE)
    st.save(MemoryItem(content="v2", id="X"), scope=ALICE)          # same id + same scope -> upsert overwrites
    assert st.get("X", scope=ALICE).content == "v2"
    assert len(st.all(scope=ALICE)) == 1


def test_store_coarse_point_access_rejects_ambiguous_siblings():
    """Point access must not choose and collapse an arbitrary sibling scope."""
    st = MemoryStore()
    st.save(MemoryItem(content="a1", id="X"), scope=Scope(base="memory", user="alice", agent="a1"))
    st.save(MemoryItem(content="a2", id="X"), scope=Scope(base="memory", user="alice", agent="a2"))
    with pytest.raises(RetrievalError, match="multiple rows"):
        st.get("X", scope=ALICE)
    with pytest.raises(RetrievalError, match="multiple rows"):
        st.replace("X", MemoryItem(content="new", id="X"), scope=ALICE)


# ---------- get / update are scope-isolated, no cross-scope writes ----------

def test_store_get_scope_isolates():
    st = MemoryStore()
    st.save(MemoryItem(content="bob 私有", id="B"), scope=BOB)
    assert st.get("B", scope=BOB).content == "bob 私有"
    assert st.get("B", scope=ALICE) is None                        # alice can't read bob's


def test_memory_update_cannot_cross_scope():
    shared, store = FakeRetriever(), MemoryStore()
    alice = Memory(retriever=shared, store=store, scope=ALICE)
    bob = Memory(retriever=shared, store=store, scope=BOB)
    bob_item = bob.add("bob 住北京")
    assert alice.update(bob_item.id, "被 alice 改写") is None       # cross-scope update rejected (treated as nonexistent)
    assert store.get(bob_item.id, scope=BOB).content == "bob 住北京"  # bob's source of truth untouched


def test_memory_update_same_scope_works():
    m = _memory()
    item = m.add("旧内容")
    assert m.update(item.id, "新内容").content == "新内容"
    assert m.store.get(item.id, scope=ALICE).content == "新内容"
    assert len(m.store.all(scope=ALICE)) == 1                      # delete-old + insert-new, no duplicate rows


def test_coarse_update_preserves_the_exact_stored_scope():
    store = MemoryStore()
    fine = Scope(base="memory", user="alice", agent="writer")
    store.save(MemoryItem(content="old", id="X"), scope=fine)
    memory = Memory(retriever=FakeRetriever(), store=store,
                    scope=Scope(base="memory", user="alice"))

    memory.update("X", "new")

    assert store.get("X", scope=fine).content == "new"
    row = store._db.execute(
        "SELECT base, sc_user, sc_agent, sc_session, sc_app FROM memories WHERE id = 'X'"
    ).fetchone()
    assert row == ("memory", "alice", "writer", "", "")


def test_coarse_delete_clears_each_exact_bookkeeping_footprint():
    from agentmaker.retrieval.index_sync import InMemoryBookkeeping, SyncIndexSync

    class ScopedRetriever:
        def __init__(self):
            self.rows = set()

        def add(self, ids, contents, *, scope=None):
            self.rows.update((scope, id_) for id_ in ids)

        def delete(self, ids, *, scope=None):
            self.rows = {
                (stored_scope, id_) for stored_scope, id_ in self.rows
                if id_ not in ids or not all(
                    getattr(scope, name) is None
                    or getattr(scope, name) == getattr(stored_scope, name)
                    for name in ("base", "user", "agent", "session", "app")
                )
            }

        def delete_exact(self, ids, *, scope=None):
            self.rows.difference_update((scope, id_) for id_ in ids)

    fine_scopes = [
        Scope(base="memory", user="alice", agent="a"),
        Scope(base="memory", user="alice", agent="b"),
    ]
    store = MemoryStore()
    retriever = ScopedRetriever()
    bookkeeping = InMemoryBookkeeping()
    sync = SyncIndexSync(retriever, bookkeeping=bookkeeping)
    for exact_scope in fine_scopes:
        store.save(MemoryItem(content="value", id="X"), scope=exact_scope)
        sync.index(["X"], ["value"], scope=exact_scope)
        bookkeeping.mark_pending(exact_scope, ["X"])
    memory = Memory(retriever, store, scope=ALICE, index_sync=sync)

    memory.delete_many(["X"])

    assert store.scopes_for_ids(["X"], scope=ALICE) == []
    assert retriever.rows == set()
    for exact_scope in fine_scopes:
        assert sync.tracked_ids(scope=exact_scope) == set()
        assert sync.pending(scope=exact_scope) == set()


def test_coarse_rebuild_removes_orphan_only_fine_scope():
    from agentmaker.retrieval.index_sync import InMemoryBookkeeping, SyncIndexSync

    class ScopedRetriever:
        def __init__(self):
            self.rows = set()

        def add(self, ids, contents, *, scope=None):
            self.rows.update((scope, id_) for id_ in ids)

        def delete(self, ids, *, scope=None):
            self.rows.difference_update((scope, id_) for id_ in ids)

        delete_exact = delete

    fine = Scope(base="memory", user="alice", agent="writer")
    retriever = ScopedRetriever()
    sync = SyncIndexSync(retriever, bookkeeping=InMemoryBookkeeping())
    sync.index(["orphan"], ["stale"], scope=fine)
    memory = Memory(retriever, MemoryStore(), scope=ALICE, index_sync=sync)

    assert memory.rebuild_index() == 0
    assert retriever.rows == set()
    assert sync.exact_scopes(scope=ALICE) == set()


def test_memory_store_delete_many_rolls_back_the_whole_batch():
    store = MemoryStore()
    store.save(MemoryItem(content="a", id="A"), scope=ALICE)
    store.save(MemoryItem(content="b", id="B"), scope=ALICE)
    store._db.execute(
        "CREATE TRIGGER reject_b BEFORE DELETE ON memories "
        "WHEN OLD.id = 'B' BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    with pytest.raises(RetrievalError):
        store.delete_many(["A", "B"], scope=ALICE)
    assert store.get("A", scope=ALICE) is not None
    assert store.get("B", scope=ALICE) is not None


def test_memory_mutation_is_serialized_across_managers(tmp_path):
    class BlockingRetriever:
        def __init__(self):
            self.first_started = threading.Event()
            self.release_first = threading.Event()
            self.contents = {}

        def add(self, ids, contents, *, scope=None):
            if contents == ["first"]:
                self.first_started.set()
                self.release_first.wait(timeout=2)
            self.contents[ids[0]] = contents[0]

        def delete(self, ids, *, scope=None):
            pass

    db = str(tmp_path / "shared-memory.db")
    first_store = MemoryStore(db)
    first_store.save(MemoryItem(content="initial", id="X"), scope=ALICE)
    retriever = BlockingRetriever()
    first = Memory(retriever, first_store, scope=ALICE)
    second = Memory(retriever, MemoryStore(db), scope=ALICE)

    first_thread = threading.Thread(target=first.update, args=("X", "first"))
    second_thread = threading.Thread(target=second.update, args=("X", "second"))
    first_thread.start()
    assert retriever.first_started.wait(timeout=2)
    second_thread.start()
    time.sleep(0.02)
    retriever.release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert first.store.get("X", scope=ALICE).content == "second"
    assert retriever.contents["X"] == "second"


def test_memory_per_call_scope_keeps_canonical_base_and_rejects_conflict():
    m = Memory(retriever=FakeRetriever(), store=MemoryStore())
    item = m.add("alice", scope=Scope(user="alice"))
    assert m.store.get(item.id, scope=ALICE).content == "alice"
    with pytest.raises(RetrievalError, match="scope.base"):
        m.search("x", scope=Scope(base="rag", user="alice"))


def test_memory_tool_merge_run_scope_isolates_users_and_rejects_conflict():
    memory = Memory(retriever=FakeRetriever(), store=MemoryStore())
    tool = MemoryTool(memory, scope_policy="merge_run")
    token = start_run("alice", scope=Scope(user="alice"))
    try:
        tool.run({"action": "remember", "content": "alice secret"})
    finally:
        reset_run(token)
    token = start_run("bob", scope=Scope(user="bob"))
    try:
        assert "alice secret" not in tool.run({"action": "recall", "query": "secret"}).text
    finally:
        reset_run(token)

    fixed = MemoryTool(Memory(FakeRetriever(), MemoryStore(), scope=ALICE), scope_policy="merge_run")
    token = start_run("bob2", scope=Scope(user="bob"))
    try:
        with pytest.raises(RetrievalError, match="conflicts"):
            fixed.run({"action": "stats"})
    finally:
        reset_run(token)


def test_memory_tool_guards_only_content_returning_actions():
    tool = MemoryTool(_memory())
    assert tool.is_external_content({"action": "recall"}) is True
    assert tool.is_external_content({"action": "summary"}) is True
    for action in ("remember", "stats", "forget", "consolidate"):
        assert tool.is_external_content({"action": action}) is False


def test_summary_without_query_honors_top_k():
    m = _memory()
    m.llm = object()
    for i in range(5):
        m.add(f"m{i}")
    assert len(m._summary_items(None, 2)) == 2


# ---------- KVStore empty-scope guard ----------

@pytest.mark.parametrize("call", [
    lambda kv: kv.get("k"),
    lambda kv: kv.all(),
    lambda kv: kv.set("k", "v"),
    lambda kv: kv.delete("k"),
])
def test_kv_empty_scope_rejected(call):
    kv = KVStore()
    with pytest.raises(RetrievalError):
        call(kv)                                                    # empty scope rejected by default


def test_kv_empty_scope_opt_in_allowed():
    kv = KVStore()
    kv.set("k", "v", scope=ALICE)
    assert kv.all(all_scopes=True) == {"k": "v"}                    # only explicit all_scopes=True is allowed through


def test_kv_memory_facade_isolates():
    store = KVStore()
    alice = KVMemory(store, scope=ALICE)
    bob = KVMemory(store, scope=BOB)
    alice.set("location", "上海")
    bob.set("location", "北京")
    assert alice.get("location") == "上海"                          # the facade always carries scope; each reads its own, no collapsing
    assert bob.get("location") == "北京"


# ---------- parameter validation ----------

def test_add_rejects_bad_importance():
    m = _memory()
    with pytest.raises(RetrievalError):
        m.add("x", importance=1.5)
    with pytest.raises(RetrievalError):
        m.add("x", importance=-0.1)


@pytest.mark.parametrize("kw", [
    {"top_k": 0},
    {"recency_halflife_hours": 0},
    {"recency_halflife_hours": -1},
    {"relevance_weight": -1},
])
def test_search_validates_params(kw):
    m = _memory()
    with pytest.raises(RetrievalError):
        m.search("q", **kw)


@pytest.mark.parametrize("top_k", [6, 20])
def test_search_large_top_k_passes_candidate_pool(top_k):
    """When top_k>=6, pool = top_k*4 > 20: Memory.search must pass the enlarged candidate pool down to the backend's candidate_pool, or the backend's validation raises RetrievalError."""
    m = _memory()
    for i in range(top_k + 2):
        m.add(f"记忆{i}")
    hits = m.search("q", top_k=top_k)
    assert isinstance(hits, list)


# ---------- add index failure -> source of truth kept + marked pending (eventually consistent), no raise, no rollback ----------

def test_add_keeps_source_and_marks_pending_on_index_failure():
    m = Memory(retriever=BoomAddRetriever(), store=MemoryStore(), scope=ALICE)
    item = m.add("会失败")                                          # doesn't raise: source of truth is authoritative, the write returns normally
    assert [x.content for x in m.store.all(scope=ALICE)] == ["会失败"]  # source of truth kept (no rollback)
    assert item.id in m.pending_reindex()                          # index write failed -> marked pending, converges on rebuild


def test_pending_reconciles_on_rebuild():
    """A pending item is reconciled into the index by rebuild_index, the pending set clears, and it becomes searchable afterward (source of truth -> index eventually consistent)."""
    m = Memory(retriever=FlakyAddRetriever(fail_times=1), store=MemoryStore(), scope=ALICE)
    item = m.add("先失败后收敛")                                    # first index write fails -> pending
    assert item.id in m.pending_reindex() and m.search("q") == []  # not in the index yet
    n = m.rebuild_index()                                          # reconcile: force-reload from source of truth (this add succeeds)
    assert n == 1 and item.id not in m.pending_reindex()           # pending clears after convergence
    assert {h.id for h in m.search("q", top_k=5)} == {item.id}     # now searchable


# ---------- search lazily cleans orphans (in the index, absent from the source of truth) ----------

def test_search_lazily_cleans_orphan():
    retriever = FakeRetriever()
    m = Memory(retriever=retriever, store=MemoryStore(), scope=ALICE)
    item = m.add("会变孤儿")
    m.store.delete_many([item.id], scope=ALICE)                    # delete only the source of truth, not the index (simulates a reverse delete failure)
    assert m.search("q") == []                                     # absent from source of truth -> filtered out
    assert item.id not in retriever.docs                           # and the index orphan is cleaned up in passing


# ---------- MemoryTool per-action confirmation ----------

def test_memory_tool_action_level_confirmation():
    tool = MemoryTool(memory=_memory())
    assert tool.needs_confirmation({"action": "forget"}) is True
    assert tool.needs_confirmation({"action": "consolidate"}) is True
    assert tool.needs_confirmation({"action": "recall"}) is False
    assert tool.needs_confirmation({"action": "remember"}) is False
    assert tool.needs_confirmation({}) is False


def test_memory_tool_confirmation_gated_via_registry():
    from agentmaker.tools.registry import ToolRegistry

    asked = []

    def confirm(tool, params):
        asked.append(params.get("action"))
        return False                                               # reject everything

    reg = ToolRegistry()
    reg.register(MemoryTool(memory=_memory()))
    deny = reg.execute_tool("memory", {"action": "forget"}, confirm=confirm)
    assert deny.status == "error" and asked == ["forget"]          # destructive action went through confirmation, denied
    asked.clear()
    ok = reg.execute_tool("memory", {"action": "stats"}, confirm=confirm)
    assert ok.status != "error" and asked == []                    # safe action skips confirmation, runs directly


# ---------- SmartWriter fail_open is configurable ----------

def test_smart_writer_fail_open_keeps_raw():
    assert SmartWriter._parse_extract("不是 JSON 的闲聊", "原文", True) == ["原文"]


def test_smart_writer_fail_closed_drops():
    assert SmartWriter._parse_extract("不是 JSON 的闲聊", "原文", False) == []


def test_smart_writer_parses_json_array_regardless():
    assert SmartWriter._parse_extract('["用户现居上海"]', "原文", False) == ["用户现居上海"]


def test_smart_writer_prompts_overridable():
    """extract_prompt / reconcile_prompt can be overridden via constructor args; omitting them uses the public DEFAULT_* prompts; and they're actually sent to the LLM as the system message."""
    from agentmaker.memory import DEFAULT_EXTRACT_PROMPT, DEFAULT_RECONCILE_PROMPT

    class _RecMem:                       # stub memory: only exposes the cfg.similar_k SmartWriter's constructor reads
        cfg = type("C", (), {"similar_k": 5})()

    class _RecLLM:                       # stub LLM: records the system message it receives
        def __init__(self): self.system = None
        async def chat(self, messages):
            self.system = messages[0]["content"]
            return type("R", (), {"content": "[]"})()

    mem = _RecMem()
    assert SmartWriter(mem, object()).extract_prompt is DEFAULT_EXTRACT_PROMPT      # omitted = public default
    assert SmartWriter(mem, object()).reconcile_prompt is DEFAULT_RECONCILE_PROMPT
    llm = _RecLLM()
    # a reconcile override must keep ADD/UPDATE/DELETE/NOOP (code dispatches on these ops); the registry validates and raises if one is missing
    import asyncio
    w = SmartWriter(mem, llm, extract_prompt="EX!", reconcile_prompt="RE! ADD UPDATE DELETE NOOP")
    assert w.extract_prompt == "EX!" and "ADD" in w.reconcile_prompt
    asyncio.run(w._extract("我住北京"))   # trigger extraction (_extract is async): the custom prompt should go out as the system message
    assert llm.system == "EX!"
    # an invalid override (dropped a protocol op) is caught by the registry
    import pytest as _pytest
    from agentmaker.prompts import PromptError
    with _pytest.raises(PromptError):
        SmartWriter(mem, object(), reconcile_prompt="只输出结论，别的不管")


# ---------- store connections take a lock (a shared connection can share one) ----------

def test_stores_reuse_injected_lock():
    lock = threading.RLock()
    assert MemoryStore(lock=lock)._lock is lock
    assert KVStore(lock=lock)._lock is lock


# ---------- rebuild_index: rebuild the index from the source of truth ----------

def test_rebuild_index_repopulates_from_truth():
    retriever = FakeRetriever()
    m = Memory(retriever=retriever, store=MemoryStore(), scope=ALICE)
    a = m.add("用户现居上海")
    b = m.add("用户对花生过敏")
    retriever.docs.clear()                                       # simulate index loss / swap to an empty backend (source of truth still present)
    assert m.search("q") == []                                   # empty index -> nothing found
    n = m.rebuild_index()
    assert n == 2 and set(retriever.docs) == {a.id, b.id}        # walk the source of truth, reload into the index
    assert {r.id for r in m.search("q", top_k=5)} == {a.id, b.id}  # search recovers them


# ---------- Memory filters pass-through + async a* API ----------

def test_memory_search_passes_filters_only_when_given():
    """Memory.search only passes filters down to the backend when given (a hard filter complementing the three-way soft scoring)."""
    class _SpyRetriever(FakeRetriever):
        def __init__(self):
            super().__init__()
            self.kwargs = None
        def search(self, query, *, top_k=5, candidate_pool=20, scope=None, **kw):
            self.kwargs = kw
            return super().search(query, top_k=top_k, candidate_pool=candidate_pool, scope=scope)

    spy = _SpyRetriever()
    m = Memory(retriever=spy, store=MemoryStore(), scope=ALICE)
    m.add("x")
    m.search("q")
    assert spy.kwargs == {}                                            # not given -> absent
    sentinel = [object()]
    m.search("q", filters=sentinel)
    assert spy.kwargs == {"filters": sentinel}                         # given -> passed through verbatim


def test_memory_async_api_smoke():
    """aadd/asearch/aupdate/adelete/arebuild_index await directly inside an event loop, same semantics as the sync versions."""
    import asyncio

    async def go():
        m = _memory()
        item = await m.aadd("用户现居上海")
        assert (await m.asearch("住哪", top_k=3))[0].id == item.id
        assert (await m.aupdate(item.id, "用户现居北京")).content == "用户现居北京"
        assert await m.arebuild_index() == 1
        await m.adelete(item.id)
        assert await m.asearch("住哪", top_k=3) == []

    asyncio.run(go())


# ---------- temporal validity (soft-invalidation) / usage feedback / conversation search ----------

def test_invalidate_soft_deletes_but_keeps_history():
    """invalidate excludes the item from retrieval while retaining its auditable supersession link."""
    m = _memory()
    old = m.add("用户现居上海")
    new = m.add("用户现居北京")
    assert m.invalidate(old.id, superseded_by=new.id) is not None
    assert {r.id for r in m.search("住哪", top_k=5)} == {new.id}        # invalidated ones aren't found
    assert [x.id for x in m.store.all(scope=ALICE)] == [new.id]         # default all returns valid only
    hist = {x.id: x for x in m.store.all(scope=ALICE, include_invalid=True)}
    assert old.id in hist and hist[old.id].invalid_at is not None       # source of truth retained
    assert hist[old.id].superseded_by == new.id                         # fact-evolution chain
    assert m.invalidate("不存在") is None


def test_invalidate_drops_only_the_resolved_index_footprint():
    broad = Scope(base="memory", user="alice")
    sibling = Scope(base="memory", user="alice", session="s1")
    item = MemoryItem(content="broad", id="shared")
    store = MemoryStore()
    store.save(item, scope=broad)

    class ScopedSync:
        def __init__(self):
            self.entries = {(item.id, broad), (item.id, sibling)}

        def drop(self, ids, *, scope=None):
            fields = ("base", "user", "agent", "session", "app")
            self.entries = {
                entry for entry in self.entries
                if entry[0] not in ids or any(
                    getattr(scope, field) is not None
                    and getattr(entry[1], field) != getattr(scope, field)
                    for field in fields)
            }

        def drop_exact(self, ids, *, scope=None):
            self.entries.difference_update((id_, scope) for id_ in ids)

        def close(self):
            pass

    sync = ScopedSync()
    memory = Memory(FakeRetriever(), store, scope=broad, index_sync=sync)

    assert memory.invalidate(item.id) is not None
    assert sync.entries == {(item.id, sibling)}


def test_rebuild_does_not_resurrect_invalidated():
    """rebuild_index works off the valid source of truth: invalidated memories aren't re-indexed back to life."""
    retr = FakeRetriever()
    m = Memory(retriever=retr, store=MemoryStore(), scope=ALICE)
    a = m.add("有效记忆")
    b = m.add("将失效")
    m.invalidate(b.id)
    retr.docs.clear()                                                   # simulate index loss
    assert m.rebuild_index() == 1                                       # reload only the valid ones
    assert set(retr.docs) == {a.id}


def test_smart_writer_update_supersedes_softly():
    """SmartWriter's UPDATE/DELETE go through soft-invalidation: the old fact is retained and points at the new one, not physically deleted."""
    m = _memory()
    old = m.add("用户现居上海")

    class _OneOp(SmartWriter):                                          # skip the LLM: feed the decision directly
        pass

    w = SmartWriter.__new__(SmartWriter)
    w.memory = m
    w._execute({"op": "UPDATE", "id": old.id, "content": "用户现居北京"}, "用户现居北京")
    items = {x.content: x for x in m.store.all(scope=ALICE, include_invalid=True)}
    assert items["用户现居上海"].invalid_at is not None                  # old item soft-invalidated
    assert items["用户现居上海"].superseded_by == items["用户现居北京"].id  # points at the superseder
    w._execute({"op": "DELETE", "id": items["用户现居北京"].id}, "")
    assert m.store.all(scope=ALICE) == []                               # all invalidated (but retained)
    assert len(m.store.all(scope=ALICE, include_invalid=True)) == 2


def test_update_bumps_updated_at_and_recency_anchor():
    """update refreshes updated_at and recency decays off it; with recency_anchor=last_accessed it uses the hit time."""
    from datetime import datetime, timedelta
    from agentmaker.memory.types import MemoryConfig
    m = _memory()
    item = m.add("旧闻")
    # manually set created_at to 30 days ago (edit the source of truth directly)
    item.created_at = datetime.now() - timedelta(days=30)
    m.store.delete_many([item.id], scope=ALICE)
    m.store.save(item, scope=ALICE)
    stale_rec = m.search("旧闻", top_k=1)[0].metadata["recency"]        # 30 days ago -> recency ~0
    m.update(item.id, "刚更新的内容")
    fresh_rec = m.search("内容", top_k=1)[0].metadata["recency"]        # after update -> recency ~1
    assert fresh_rec > 0.9 > stale_rec
    # anchor knob validation
    MemoryConfig(recency_anchor="last_accessed").validate()
    with pytest.raises(ValueError):
        MemoryConfig(recency_anchor="bogus").validate()


def test_search_touches_last_accessed():
    """A retrieval hit writes back last_accessed_at (the "memories that get used stay fresh" half of Generative Agents)."""
    m = _memory()
    item = m.add("被翻出来的记忆")
    assert m.store.get(item.id, scope=ALICE).last_accessed_at is None
    m.search("记忆", top_k=3)
    assert m.store.get(item.id, scope=ALICE).last_accessed_at is not None


def test_store_adds_missing_optional_columns(tmp_path):
    """A store with the required key receives any missing optional columns on open."""
    import sqlite3
    from agentmaker.retrieval.scope_sql import scope_column_names
    db = str(tmp_path / "partial_mem.db")
    conn = sqlite3.connect(db)
    cols = ", ".join(f"{c} TEXT" for c in scope_column_names())
    pk = ", ".join(["id", *scope_column_names()])
    conn.execute(f"CREATE TABLE memories(id TEXT, {cols}, content TEXT, type TEXT, "
                 f"importance REAL, created_at TEXT, metadata TEXT, PRIMARY KEY ({pk}))")   # older schema, missing the newer columns
    conn.commit()
    conn.close()
    st = MemoryStore(db)                                               # doesn't raise: auto-adds the columns
    st.save(MemoryItem(content="迁移后可写", id="X"), scope=ALICE)
    assert st.get("X", scope=ALICE).invalid_at is None


def test_conversation_search_end_to_end():
    """ConversationSearch: wraps a SessionStore feeding the shared backend (isolated by base=conversation), searchable, and clear wipes the index too."""
    from agentmaker.core.message import Message
    from agentmaker.runtime import ConversationSearch, SqliteSessionStore
    from agentmaker.retrieval.scope import Scope as _S

    class _ConvRetriever(FakeRetriever):
        def __init__(self):
            super().__init__()
            self.contents = {}
        def add(self, ids, contents, *, scope=None, metadatas=None):
            for i, c in zip(ids, contents):
                self.docs[i] = scope
                self.contents[i] = c

    retr = _ConvRetriever()
    cs = ConversationSearch(SqliteSessionStore(), retr)
    sc = _S(user="alice", session="chat1")
    cs.append_many([Message("我们聊聊东京旅行", "user"), Message("好的，东京四月樱花最好", "assistant")], scope=sc)
    # source of truth intact (zero difference from the Agent's view)
    assert [m.role for m in cs.load(scope=sc)] == ["user", "assistant"]
    # index isolated by base="conversation" + carries a role prefix
    assert all(s.base == "conversation" and s.user == "alice" for s in retr.docs.values())
    assert any(c.startswith("user: 我们聊聊") for c in retr.contents.values())
    # searchable (FakeRetriever returns by exact scope)
    hits = cs.search("东京", top_k=5, scope=sc)
    assert len(hits) == 2
    # clear: wipes the index too
    cs.clear(scope=sc)
    assert cs.load(scope=sc) == [] and retr.docs == {}


def test_conversation_search_clear_keeps_source_when_exact_index_delete_fails():
    from agentmaker.core.message import Message
    from agentmaker.runtime import ConversationSearch, SqliteSessionStore

    class FailingDelete(FakeRetriever):
        def delete_exact(self, ids, *, scope=None):
            raise RuntimeError("index delete failed")

    retriever = FailingDelete()
    search = ConversationSearch(SqliteSessionStore(), retriever)
    scope = Scope(user="alice", session="one")
    search.append(Message("keep me", "user"), scope=scope)

    with pytest.raises(RuntimeError, match="index delete failed"):
        search.clear(scope=scope)

    assert [message.content for message in search.load(scope=scope)] == ["keep me"]


def test_conversation_search_clear_falls_back_to_ranged_delete_for_legacy_backends():
    """A retriever with only the pre-exact surface (add/delete) still supports clear: the exact drop degrades to a ranged delete over the footprint scope."""
    from agentmaker.core.message import Message
    from agentmaker.runtime import ConversationSearch, SqliteSessionStore

    class CoarseOnlyRetriever:
        def __init__(self):
            self.deleted = []

        def add(self, ids, contents, *, scope=None, metadatas=None):
            pass

        def delete(self, ids, *, scope=None):
            self.deleted.append((list(ids), scope))

    retriever = CoarseOnlyRetriever()
    search = ConversationSearch(SqliteSessionStore(), retriever)
    scope = Scope(user="alice", session="one")
    search.append(Message("bye", "user"), scope=scope)

    search.clear(scope=scope)

    assert search.load(scope=scope) == []
    assert retriever.deleted and retriever.deleted[0][1].session == "one"


def test_conversation_search_clear_supports_legacy_index_sync_signature():
    """A custom IndexSync written against the pre-strict abstract signature keeps working: strict downgrades to its best-effort drop contract."""
    from agentmaker.core.message import Message
    from agentmaker.retrieval.index_sync import IndexSync
    from agentmaker.runtime import ConversationSearch, SqliteSessionStore

    class LegacySync(IndexSync):
        def __init__(self):
            self.dropped = []

        def index(self, ids, contents, *, scope=None, metadatas=None):
            pass

        def replace(self, old_ids, new_ids, contents, *, scope=None, metadatas=None):
            pass

        def drop(self, ids, *, scope=None):
            self.dropped.extend(ids)

        def reconcile(self, items, *, scope=None, batch_size=256):
            return 0

        def pending(self, *, scope=None):
            return set()

    sync = LegacySync()
    search = ConversationSearch(SqliteSessionStore(), FakeRetriever(), index_sync=sync)
    scope = Scope(user="alice", session="one")
    search.append(Message("bye", "user"), scope=scope)

    search.clear(scope=scope)
    assert search.load(scope=scope) == [] and len(sync.dropped) == 1

    search.append(Message("all", "user"), scope=scope)
    search.clear(all_scopes=True)
    assert search.load(all_scopes=True) == [] and len(sync.dropped) == 2


def test_conversation_search_clear_tolerates_strictless_drop_exact():
    """A custom sync whose drop_exact lacks the strict flag still supports scoped clear (strict downgrades like drop)."""
    from agentmaker.core.message import Message
    from agentmaker.retrieval.index_sync import IndexSync
    from agentmaker.runtime import ConversationSearch, SqliteSessionStore

    class ExactOnlySync(IndexSync):
        def __init__(self):
            self.exact_drops = []

        def index(self, ids, contents, *, scope=None, metadatas=None):
            pass

        def replace(self, old_ids, new_ids, contents, *, scope=None, metadatas=None):
            pass

        def drop(self, ids, *, scope=None):
            pass

        def drop_exact(self, ids, *, scope=None):
            self.exact_drops.append((list(ids), scope))

        def reconcile(self, items, *, scope=None, batch_size=256):
            return 0

        def pending(self, *, scope=None):
            return set()

    sync = ExactOnlySync()
    search = ConversationSearch(SqliteSessionStore(), FakeRetriever(), index_sync=sync)
    scope = Scope(user="alice", session="one")
    search.append(Message("bye", "user"), scope=scope)
    search.clear(scope=scope)
    assert search.load(scope=scope) == [] and len(sync.exact_drops) == 1


def test_memory_mutations_support_legacy_duck_retriever():
    """invalidate / delete / rebuild_index work against a retriever without delete_exact (the pre-exact published seam)."""

    class LegacyRetriever:
        def __init__(self):
            self.docs = {}

        def add(self, ids, contents, *, scope=None, metadatas=None):
            for i in ids:
                self.docs[i] = scope

        def delete(self, ids, *, scope=None):
            for i in ids:
                self.docs.pop(i, None)

        def search(self, query, *, top_k=5, candidate_pool=20, scope=None):
            return [_Hit(i) for i, sc in self.docs.items() if sc == scope][:top_k]

        def close(self):
            pass

    assert not hasattr(LegacyRetriever, "delete_exact")

    m = Memory(retriever=LegacyRetriever(), store=MemoryStore(), scope=ALICE)
    kept = m.add("保留")
    gone = m.add("待删")
    stale = m.add("待失效")

    assert m.invalidate(stale.id) is not None
    m.delete(gone.id)
    assert {i.id for i in m.store.all(scope=ALICE)} == {kept.id}   # all() hides the invalidated audit record
    assert m.rebuild_index() >= 1


def test_memory_delete_many_drops_ids_unknown_to_store_and_bookkeeping():
    """A physical-delete batch also issues a ranged drop for ids the store and bookkeeping no longer know, so stale index rows cannot survive deletion."""
    retr = FakeRetriever()
    m = Memory(retriever=retr, store=MemoryStore(), scope=ALICE)
    live = m.add("живой")
    retr.docs["ghost"] = ALICE                    # index row with no store row and no bookkeeping entry

    m.delete_many([live.id, "ghost"])

    assert m.store.all(scope=ALICE) == []
    assert "ghost" not in retr.docs


def test_conversation_search_global_clear_cleans_exact_bookkeeping_footprints():
    from agentmaker.core.message import Message
    from agentmaker.retrieval.index_sync import InMemoryBookkeeping, SyncIndexSync
    from agentmaker.runtime import ConversationSearch, SqliteSessionStore

    class ScopedRetriever:
        def __init__(self):
            self.rows = set()

        def add(self, ids, contents, *, scope=None, metadatas=None):
            self.rows.update((scope, id_) for id_ in ids)

        def delete(self, ids, *, scope=None):
            self.rows = {
                (stored_scope, id_)
                for stored_scope, id_ in self.rows
                if id_ not in ids or not all(
                    getattr(scope, name) is None
                    or getattr(scope, name) == getattr(stored_scope, name)
                    for name in ("base", "user", "agent", "session", "app")
                )
            }

        def delete_exact(self, ids, *, scope=None):
            self.rows.difference_update((scope, id_) for id_ in ids)

    retriever = ScopedRetriever()
    sync = SyncIndexSync(retriever, bookkeeping=InMemoryBookkeeping())
    search = ConversationSearch(SqliteSessionStore(), retriever, index_sync=sync)
    first = Scope(user="alice", session="one")
    second = Scope(user="bob", session="two")
    search.append(Message("one", "user", metadata={"message_id": "same"}), scope=first)
    search.append(Message("two", "user", metadata={"message_id": "same"}), scope=second)

    search.clear(scope=first, all_scopes=True)

    assert search.load(all_scopes=True) == []
    assert retriever.rows == set()
    assert sync.exact_scopes(scope=Scope(base="conversation")) == set()


def test_conversation_search_serializes_append_with_clear():
    from agentmaker.core.message import Message
    from agentmaker.runtime import ConversationSearch, SqliteSessionStore

    deleting = threading.Event()
    release_delete = threading.Event()

    class BlockingRetriever(FakeRetriever):
        def add(self, ids, contents, *, scope=None, metadatas=None):
            super().add(ids, contents, scope=scope)

        def delete_exact(self, ids, *, scope=None):
            deleting.set()
            assert release_delete.wait(2)
            self.delete(ids, scope=scope)

    retriever = BlockingRetriever()
    search = ConversationSearch(SqliteSessionStore(), retriever)
    scope = Scope(user="alice", session="one")
    search.append(Message("old", "user"), scope=scope)
    errors = []

    def clear():
        try:
            search.clear(scope=scope)
        except Exception as error:
            errors.append(error)

    appended = threading.Event()

    def append():
        try:
            search.append(Message("new", "user"), scope=scope)
            appended.set()
        except Exception as error:
            errors.append(error)

    clear_thread = threading.Thread(target=clear)
    clear_thread.start()
    assert deleting.wait(2)
    append_thread = threading.Thread(target=append)
    append_thread.start()
    assert not appended.wait(0.05)
    release_delete.set()
    clear_thread.join(2)
    append_thread.join(2)

    assert not errors
    messages = search.load(scope=scope)
    assert [message.content for message in messages] == ["new"]
    assert set(retriever.docs) == {messages[0].metadata["message_id"]}


def test_conversation_search_close_releases_sync_bookkeeping():
    from agentmaker.retrieval.index_sync import InMemoryBookkeeping, SyncIndexSync
    from agentmaker.runtime import ConversationSearch, SqliteSessionStore

    closed = []
    bookkeeping = InMemoryBookkeeping()
    bookkeeping.close = lambda: closed.append(True)
    retriever = FakeRetriever()
    search = ConversationSearch(
        SqliteSessionStore(), retriever,
        index_sync=SyncIndexSync(retriever, bookkeeping=bookkeeping),
    )

    search.close()
    search.close()

    assert closed == [True]


def test_conversation_search_tool_runs():
    """ConversationSearchTool: wraps it as a read-only tool; run returns a readable result (empty query errors)."""
    from agentmaker.core.message import Message
    from agentmaker.runtime import ConversationSearch, ConversationSearchTool, SqliteSessionStore
    from agentmaker.retrieval.scope import Scope as _S

    retr = FakeRetriever()
    cs = ConversationSearch(SqliteSessionStore(), retr)
    sc = _S(user="alice")
    cs.append(Message("上次说到报销制度", "user"), scope=sc)
    tool = ConversationSearchTool(cs, scope=sc)
    assert tool.name == "conversation_search"
    assert tool.run({"query": ""}).status == "error"
    out = tool.run({"query": "报销"})
    assert out.status != "error"


# ---------- update / invalidate are transactional + scope-symmetric ----------

class _FailInsertConn:
    """Connection proxy: INSERT raises a sqlite error, everything else forwards to the real connection; used to verify that replace's single transaction rolls back the delete when the new write fails."""
    def __init__(self, real):
        self._real = real
    def __enter__(self):
        return self._real.__enter__()
    def __exit__(self, *exc):
        return self._real.__exit__(*exc)
    def execute(self, sql, *args, **kwargs):
        if sql.lstrip().upper().startswith("INSERT"):
            raise sqlite3.Error("insert boom")
        return self._real.execute(sql, *args, **kwargs)
    def __getattr__(self, name):
        return getattr(self._real, name)


def test_memory_update_atomic_rollback_keeps_old_value():
    """When update's new write fails, the single transaction rolls back the delete -> the old value is kept, no row lost (the lost-row hole from crashing between delete-then-write)."""
    m = _memory()
    item = m.add("原始内容")
    m.store._db = _FailInsertConn(m.store._db)                 # inject: DELETE passes, INSERT raises
    with pytest.raises(RetrievalError):
        m.update(item.id, "新内容")
    m.store._db = m.store._db._real                            # restore so we can read
    got = m.store.get(item.id, scope=ALICE)
    assert got is not None and got.content == "原始内容"        # no row lost, rolled back to the old value


def test_memory_invalidate_atomic_rollback_keeps_old_value():
    """When invalidate's write-back fails, the single transaction rolls back -> the memory stays valid, not lost to a crash between the two soft-invalidation steps."""
    m = _memory()
    item = m.add("会被尝试失效的记忆")
    m.store._db = _FailInsertConn(m.store._db)
    with pytest.raises(RetrievalError):
        m.invalidate(item.id)
    m.store._db = m.store._db._real
    got = m.store.get(item.id, scope=ALICE)
    assert got is not None and got.invalid_at is None          # still valid, no row lost


def test_memory_update_with_explicit_scope_stays_in_scope():
    """An update with an explicit scope= targets the fine scope exactly and stays there afterward (mirrors add's scope=, so a coarse instance doesn't mis-migrate it)."""
    store = MemoryStore()
    coarse = Memory(retriever=FakeRetriever(), store=store, scope=Scope(base="memory"))
    item = coarse.add("住在上海", scope=ALICE)                  # stored under the fine scope
    updated = coarse.update(item.id, "住在北京", scope=ALICE)   # update with scope=
    assert updated is not None and updated.content == "住在北京"
    got = store.get(item.id, scope=ALICE)
    assert got is not None and got.content == "住在北京"        # still in the original ALICE scope, content updated


# ---------- SmartWriter robustness + KV semantics + confirm gate ----------

def test_smart_writer_reconcile_rejects_bool_index():
    """{"op":"DELETE","index":true} must not treat the bool as index=1 and wrongly delete similar[0]; it falls back to ADD."""
    similar = [MemoryItem(content="a", id="a"), MemoryItem(content="b", id="b")]
    assert SmartWriter._parse_reconcile('{"op":"DELETE","index":true}', similar) == {"op": "ADD"}


def test_smart_writer_reconcile_accepts_numeric_string_index():
    """A numeric-string index "2" is normalized to 2 and maps to similar[1]."""
    similar = [MemoryItem(content="a", id="a"), MemoryItem(content="b", id="b")]
    d = SmartWriter._parse_reconcile('{"op":"UPDATE","index":"2"}', similar)
    assert d["op"] == "UPDATE" and d["id"] == "b"


def test_smart_writer_extract_skips_non_strings():
    """Extraction keeps only string items (taking a dict's fact field); everything else (numbers / null / blanks) is skipped, not str()'d into garbage."""
    content = '["事实一", 123, {"fact": "事实二"}, null, "  "]'
    assert SmartWriter._parse_extract(content, "原文", fail_open=True) == ["事实一", "事实二"]


def test_kv_coarse_scope_reads_fail_loud():
    """A coarse scope matching duplicate keys is ambiguous; an exact scope remains readable."""
    kv = KVStore()
    kv.set("location", "上海", scope=ALICE)
    kv.set("location", "北京", scope=BOB)
    coarse = Scope(base="memory")                                  # constrains base only, spans alice/bob
    with pytest.raises(RetrievalError):
        kv.get("location", scope=coarse)
    with pytest.raises(RetrievalError):
        kv.all(scope=coarse)
    assert kv.get("location", scope=ALICE) == "上海"               # exact scope is unambiguous, reads fine


def test_memory_tool_remember_confirm_gate_with_writer():
    """With a writer attached, remember goes through the confirmation gate by default (SmartWriter may edit / delete existing memories); turning the switch off or having no writer lets it through."""
    m = _memory()

    class _W:                                                      # minimal writer stub: needs_confirmation only checks whether a writer exists
        pass

    assert MemoryTool(memory=m, writer=_W()).needs_confirmation({"action": "remember"}) is True
    assert MemoryTool(memory=m, writer=_W(), confirm_writer_edits=False).needs_confirmation(
        {"action": "remember"}) is False
    assert MemoryTool(memory=m).needs_confirmation({"action": "remember"}) is False   # no writer: plain add, allowed through
