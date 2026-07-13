"""Hermetic retrieval backend contract tests."""

import sqlite3
import threading
import time

import pytest

from agentmaker.core.exceptions import RetrievalError
from agentmaker.retrieval import (Fts5KeywordIndex, HybridRetriever, OpenAIEmbedder, RetrievalResult, Scope,
                              SqliteVecStore, build_sqlite_hybrid, reciprocal_rank_fusion)
from agentmaker.retrieval.backends.sqlite import SqliteHybridRetriever  # internal class, not exported at the top level
from agentmaker.retrieval.index_sync import InMemoryBookkeeping, SyncBookkeeping, SyncIndexSync
from agentmaker.retrieval.scope import canonical_scope

MEM = Scope(base="memory")


def test_index_sync_rejects_misaligned_batches_and_serializes_same_id():
    """zip must not truncate writes, and overlapping scope ranges serialize the same id."""
    class Recorder:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def add(self, ids, contents, *, scope=None, metadatas=None):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1

        def delete(self, ids, *, scope=None):
            pass

    recorder = Recorder()
    sync = SyncIndexSync(recorder)
    sibling_sync = SyncIndexSync(recorder)
    with pytest.raises(RetrievalError, match="same length"):
        sync.index(["a", "b"], ["one"], scope=MEM)
    with pytest.raises(RetrievalError, match="metadatas"):
        sync.index(["a"], ["one"], scope=MEM, metadatas=[])
    fine = next(
        Scope(base="memory", user="alice", session=str(i))
        for i in range(128)
        if hash((Scope(base="memory"), "same")) % 64
        != hash((Scope(base="memory", user="alice", session=str(i)), "same")) % 64
    )
    threads = [
        threading.Thread(target=current.index, args=(["same"], [content]),
                         kwargs={"scope": scope})
        for current, content, scope in (
            (sync, "one", MEM), (sibling_sync, "two", fine))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert recorder.max_active == 1


def test_index_sync_drop_range_preserves_unlisted_sibling_footprints():
    class Recorder:
        def __init__(self):
            self.rows = set()

        def add(self, ids, contents, *, scope=None):
            self.rows.update((scope, id_) for id_ in ids)

        def delete(self, ids, *, scope=None):
            raise AssertionError("drop_range must not use coarse deletion")

        def delete_exact(self, ids, *, scope=None):
            self.rows.difference_update((scope, id_) for id_ in ids)

    target = Scope(base="memory", user="alice", agent="target")
    sibling = Scope(base="memory", user="alice", agent="sibling")
    recorder = Recorder()
    sync = SyncIndexSync(recorder)
    sync.index(["same"], ["target"], scope=target)
    sync.index(["same"], ["sibling"], scope=sibling)

    sync.drop_range({target: ["same"]}, scope=Scope(base="memory", user="alice"))

    assert recorder.rows == {(sibling, "same")}
    assert sync.tracked_ids(scope=target) == set()
    assert sync.tracked_ids(scope=sibling) == {"same"}


def test_scope_empty_strings_are_normalized_consistently():
    scope = Scope(base="", user="", agent="writer")
    assert scope == Scope(agent="writer")
    assert canonical_scope(Scope(base="", user="alice"), "memory", "test") == Scope(
        base="memory", user="alice")


@pytest.mark.parametrize("value", [0, False, 1.5, object()])
def test_scope_rejects_non_string_dimensions(value):
    with pytest.raises(TypeError, match="Scope.user must be a string or None"):
        Scope(user=value)


@pytest.mark.parametrize("dimensions", [0, -1])
def test_openai_embedder_rejects_nonpositive_dimensions(dimensions):
    with pytest.raises(RetrievalError, match="dimensions"):
        OpenAIEmbedder(api_key="test", dimensions=dimensions)


@pytest.mark.parametrize("timeout", [0, -1])
def test_openai_embedder_rejects_nonpositive_timeout(timeout):
    with pytest.raises(RetrievalError, match="timeout"):
        OpenAIEmbedder(api_key="test", timeout=timeout)


# ---------- SqliteVecStore ----------

def test_vector_upsert_dedupes_and_overwrites():
    """Writing the same (id, scope) twice leaves one row with the last content (upsert overwrites, doesn't append)."""
    vs = SqliteVecStore(dim=3)
    vs.add(["n1"], [[1.0, 0.0, 0.0]], ["hello"], scope=MEM)
    vs.add(["n1"], [[1.0, 0.0, 0.0]], ["hello v2"], scope=MEM)
    hits = vs.search([1.0, 0.0, 0.0], top_k=10, scope=MEM)
    assert len(hits) == 1                      # no duplicate rows
    assert hits[0].content == "hello v2"       # overwritten to the latest


def test_vector_upsert_does_not_clobber_sibling_scope():
    """Same id, different scopes are two independent records: upsert matches all dimensions exactly and doesn't delete the sibling scope's row."""
    vs = SqliteVecStore(dim=3)
    vs.add(["x"], [[0.0, 1.0, 0.0]], ["alice-doc"], scope=Scope(base="memory", user="alice"))
    vs.add(["x"], [[0.0, 1.0, 0.0]], ["bob-doc"], scope=Scope(base="memory", user="bob"))
    a = vs.search([0.0, 1.0, 0.0], top_k=10, scope=Scope(base="memory", user="alice"))
    b = vs.search([0.0, 1.0, 0.0], top_k=10, scope=Scope(base="memory", user="bob"))
    assert [h.content for h in a] == ["alice-doc"]
    assert [h.content for h in b] == ["bob-doc"]


def test_vector_delete_scope_boundary_is_b_semantics():
    """store.delete scope boundary (B semantics): a given dimension deletes only within it; an empty Scope() deletes by id across all scopes (the store layer sets no guard)."""
    vs = SqliteVecStore(dim=3)
    vs.add(["x"], [[1.0, 0.0, 0.0]], ["alice"], scope=Scope(base="memory", user="alice"))
    vs.add(["x"], [[1.0, 0.0, 0.0]], ["bob"], scope=Scope(base="memory", user="bob"))
    vs.delete(["x"], scope=Scope(base="memory", user="alice"))      # delete only alice
    remain = vs.search([1.0, 0.0, 0.0], top_k=10, scope=MEM)
    assert [h.content for h in remain] == ["bob"]                   # bob remains
    vs.add(["x"], [[1.0, 0.0, 0.0]], ["alice2"], scope=Scope(base="memory", user="alice"))
    vs.delete(["x"], scope=Scope())                                # empty scope -> delete across all scopes (the guard lives in the HybridRetriever layer)
    assert vs.search([1.0, 0.0, 0.0], top_k=10, scope=MEM) == []


def test_empty_scope_guarded_at_hybrid_unless_all_scopes():
    """The guard is in the HybridRetriever layer: delete / search with an empty Scope() (or scope=None) is rejected by default; only an explicit all_scopes=True allows it."""
    hr = build_sqlite_hybrid(_FakeEmbedder())
    hr.add(["x"], ["会员卡号 A8821"], scope=MEM)
    with pytest.raises(RetrievalError):
        hr.delete(["x"], scope=Scope())               # empty scope -> rejected
    with pytest.raises(RetrievalError):
        hr.search("A8821")                            # scope=None treated as empty -> rejected
    # allowed after explicit opt-in
    assert hr.search("A8821", all_scopes=True)        # searches across the whole store and finds it
    hr.delete(["x"], scope=Scope(), all_scopes=True)  # deletes across the whole store without raising
    assert hr.search("A8821", all_scopes=True) == []


def test_vector_search_ranks_nearest_first():
    """Nearest-neighbor ordering: whichever row the query vector is closest to ranks first."""
    vs = SqliteVecStore(dim=3)
    vs.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ["x-axis", "y-axis"], scope=MEM)
    hits = vs.search([0.9, 0.1, 0.0], top_k=2, scope=MEM)
    assert hits[0].content == "x-axis"


def test_vector_rejects_unsafe_table_name():
    """Table names are validated as identifiers: an illegal name (with a semicolon / space etc.) raises RetrievalError outright."""
    with pytest.raises(RetrievalError):
        SqliteVecStore(dim=3, table="vec; DROP TABLE x")


# ---------- Fts5KeywordIndex ----------

def test_keyword_upsert_dedupes():
    """The keyword index leaves one row when the same (id, scope) is written twice."""
    kw = Fts5KeywordIndex()
    kw.add(["k1"], ["会员卡号 A8821"], scope=MEM)
    kw.add(["k1"], ["会员卡号 A8821 已更新"], scope=MEM)
    hits = kw.search("A8821", top_k=10, scope=MEM)
    assert len(hits) == 1


def test_keyword_search_hits_exact_token():
    """BM25 hits the exact code literally."""
    kw = Fts5KeywordIndex()
    kw.add(["k1", "k2"], ["会员卡号 A8821", "明天上午十点开会"], scope=MEM)
    hits = kw.search("A8821", top_k=5, scope=MEM)
    assert hits and hits[0].content == "会员卡号 A8821"


def test_keyword_empty_query_returns_empty():
    """A query that tokenizes to nothing -> returns [] (no keyword hit; hybrid retrieval still falls back to the vector path)."""
    kw = Fts5KeywordIndex()
    kw.add(["k1"], ["会员卡号 A8821"], scope=MEM)
    assert kw.search("   ", top_k=5, scope=MEM) == []


def test_keyword_rejects_unsafe_table_name():
    with pytest.raises(RetrievalError):
        Fts5KeywordIndex(table="kw items")


# ---------- reciprocal_rank_fusion ----------

def test_rrf_empty_ids_are_not_merged():
    """Distinct id-less results remain separate."""
    l1 = [RetrievalResult(content="aaa", score=1.0, source="vector")]
    l2 = [RetrievalResult(content="bbb", score=1.0, source="keyword")]
    fused = reciprocal_rank_fusion([l1, l2], top_k=10)
    assert len(fused) == 2
    assert all(f.id == "" for f in fused)                # no fabricated id; the output keeps the original empty string


def test_rrf_same_id_merges_and_sums_score():
    """An id hit by multiple lists -> merged into one with summed scores (corroboration across lists scores higher)."""
    l1 = [RetrievalResult(content="c", score=9, source="vector", id="n9")]
    l2 = [RetrievalResult(content="c", score=9, source="keyword", id="n9")]
    fused = reciprocal_rank_fusion([l1, l2], top_k=10)
    assert len(fused) == 1
    # each list rank=1, RRF score = 1/(60+1) + 1/(60+1)
    assert fused[0].score == pytest.approx(2.0 / 61.0)


def test_rrf_preserves_real_id():
    """When an id is present, the output keeps the original id."""
    l1 = [RetrievalResult(content="c", score=1, source="vector", id="abc")]
    fused = reciprocal_rank_fusion([l1], top_k=10)
    assert fused[0].id == "abc"


# ---------- HybridRetriever half-write compensation ----------

class _FakeEmbedder:
    """Fake embedder: returns fixed-dimension vectors, no network."""
    dim = 3

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class _BoomKeyword:
    """Fake keyword index whose add always raises, to trigger hybrid's half-write compensation."""

    def add(self, ids, contents, *, scope=None, metadatas=None):
        raise RuntimeError("keyword index boom")

    def delete(self, ids, *, scope=None):
        pass

    def search(self, query, *, top_k=5, scope=None):
        return []

    def close(self):
        pass


def test_hybrid_add_compensates_vector_on_keyword_failure():
    """When the keyword-index write fails, hybrid rolls back the already-written vector and re-raises the original exception, leaving no "vector present, keyword missing" half-write."""
    vs = SqliteVecStore(dim=3)
    hr = HybridRetriever(embedder=_FakeEmbedder(), vector_store=vs, keyword_index=_BoomKeyword())
    with pytest.raises(RuntimeError):
        hr.add(["n1"], ["hi"], scope=MEM)
    assert vs.search([1.0, 0.0, 0.0], top_k=10, scope=MEM) == []   # the vector was compensated away


def test_hybrid_add_compensation_works_with_empty_scope():
    """Compensation still works with a separate connection and empty scope: the store sets no guard, so the compensating delete removes by id across the store cleanly, leaving no half-write."""
    vs = SqliteVecStore(dim=3)
    hr = HybridRetriever(embedder=_FakeEmbedder(), vector_store=vs, keyword_index=_BoomKeyword())
    with pytest.raises(RuntimeError):
        hr.add(["n2"], ["hi"], scope=None)                         # bare add, no scope
    assert vs.search([1.0, 0.0, 0.0], top_k=10) == []              # even the empty-scope write is fully compensated away


# ---------- SqliteHybridRetriever: cross-index single-transaction atomicity ----------

def test_atomic_update_failure_keeps_old_value():
    """SqliteHybridRetriever: on an existing-id update, a keyword-write failure rolls back the whole transaction -> the old vector survives and neither store is lost or corrupted."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    lock = threading.RLock()
    vs = SqliteVecStore(dim=3, connection=conn, lock=lock)
    hr = SqliteHybridRetriever(_FakeEmbedder(), vs, _BoomKeyword(), conn, lock)
    # seed the old value (a shared connection doesn't auto-commit, so commit the seed manually)
    vs.add(["n"], [[1.0, 0.0, 0.0]], ["old"], scope=MEM)
    conn.commit()
    # update n->new, but the keyword write fails -> the whole transaction should roll back
    with pytest.raises(RuntimeError):
        hr.add(["n"], ["new"], scope=MEM)
    hits = vs.search([1.0, 0.0, 0.0], top_k=10, scope=MEM)
    assert [h.content for h in hits] == ["old"]                    # the old value isn't lost or deleted (unlike the compensation path, which would lose it)


def test_build_sqlite_hybrid_atomic_add_search():
    """build_sqlite_hybrid builds a shared-connection retriever: add lands in both indexes, search finds the keyword hit via RRF."""
    hr = build_sqlite_hybrid(_FakeEmbedder())
    hr.add(["n1", "n2"], ["alpha apple", "beta banana"], scope=MEM)
    hits = hr.search("banana", scope=MEM)
    assert hits and hits[0].id == "n2"                             # the keyword path lifts the banana row to the top


def test_vec_delete_exact_spares_sibling_scope():
    """SqliteVecStore.delete_exact deletes precisely by write footprint and spares the sibling row with the same id but a different scope (so add-compensation doesn't delete across scopes)."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    vs = SqliteVecStore(dim=3, connection=conn, lock=threading.RLock())
    specific = Scope(base="memory", user="alice")
    vs.add(["x"], [[1.0, 0.0, 0.0]], ["broad"], scope=MEM)
    vs.add(["x"], [[0.0, 1.0, 0.0]], ["specific"], scope=specific)   # same id, more specific scope (both rows coexist)
    conn.commit()
    vs.delete_exact(["x"], scope=MEM)                                # exact-delete the MEM footprint (a B-semantics delete would wrongly take specific too)
    conn.commit()
    assert [h.content for h in vs.search([0.0, 1.0, 0.0], top_k=10, scope=specific)] == ["specific"]  # the sibling row wasn't wrongly deleted
    assert "broad" not in [h.content for h in vs.search([1.0, 0.0, 0.0], top_k=10, scope=MEM)]        # the MEM footprint is gone


# ---------- OpenAIEmbedder return validation (fake client, no network) ----------

class _FakeData:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeClient:
    """Fake OpenAI client: embeddings.create returns preset data."""

    def __init__(self, data):
        self._data = data
        self.embeddings = self

    def create(self, **kwargs):
        return _FakeResp(self._data)


def _embedder_with(data):
    emb = OpenAIEmbedder(api_key="x", dimensions=3)      # dim=3 to avoid building 1536-dim vectors
    emb._client = _FakeClient(data)                      # inject the fake client directly, bypassing _ensure_client
    return emb


def test_embedder_happy_path_aligns_by_index():
    emb = _embedder_with([_FakeData(1, [4.0, 5.0, 6.0]), _FakeData(0, [1.0, 2.0, 3.0])])
    out = emb.embed(["a", "b"])
    assert out == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]      # aligned by index, regardless of arrival order


def test_embedder_batches_large_input():
    """Inputs over max_batch are split into ordered serial batches."""
    calls = []

    class _BatchClient:
        def __init__(self):
            self.embeddings = self

        def create(self, **kwargs):
            n = len(kwargs["input"])
            calls.append(n)
            return _FakeResp([_FakeData(i, [1.0, 2.0, 3.0]) for i in range(n)])

    emb = OpenAIEmbedder(api_key="x", dimensions=3, max_batch=2)
    emb._client = _BatchClient()
    out = emb.embed(["a", "b", "c", "d", "e"])           # 5 items, max_batch=2 -> 3 batches (2+2+1)
    assert calls == [2, 2, 1] and len(out) == 5          # correct batch sizes, all concatenated in order


def test_digest_includes_metadata():
    """_digest folds in metadata (a change triggers a rewrite); empty metadata falls back to a pure-content hash (so old paths like memory keep the same fingerprint and don't trigger a full re-embed)."""
    from agentmaker.retrieval.index_sync import _digest
    assert _digest("x") == _digest("x", None) == _digest("x", {})            # empty metadata falls back to a pure-content hash
    assert _digest("x", {"doc_id": "a"}) != _digest("x", {"doc_id": "b"})    # metadata changes -> fingerprint changes
    assert _digest("x", {"doc_id": "a"}) != _digest("x")                     # with vs without metadata -> different


def test_contextualizer_fingerprint_tracks_prompt_and_model():
    """LLMContextualizer.fingerprint varies with prompt / model (a change on reimport isn't wrongly short-circuited by the fingerprint); the base class defaults to its class name."""
    from agentmaker.rag.contextualizer import HeadingContextualizer, LLMContextualizer
    stub = type("L", (), {"model": "m"})()
    assert HeadingContextualizer().fingerprint() == "HeadingContextualizer"
    assert LLMContextualizer(stub, context_prompt="A").fingerprint() != LLMContextualizer(stub, context_prompt="B").fingerprint()


def test_embedder_count_mismatch_raises():
    emb = _embedder_with([_FakeData(0, [1.0, 2.0, 3.0])])  # returns only 1, input has 2
    with pytest.raises(RetrievalError):
        emb.embed(["a", "b"])


def test_embedder_dim_mismatch_raises():
    emb = _embedder_with([_FakeData(0, [1.0, 2.0]), _FakeData(1, [3.0, 4.0])])  # dim 2 != expected 3
    with pytest.raises(RetrievalError):
        emb.embed(["a", "b"])


# ---------- HybridRetriever.search_many: batch-embed multiple queries ----------

class _SpyEmbedder:
    """Fake embedder: records each embed's input, to make the "one batch call" assertion easy."""
    dim = 3

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_search_many_batch_embeds_once():
    """search_many embeds all queries in one batch and returns a result list the same length as queries."""
    emb = _SpyEmbedder()
    hr = build_sqlite_hybrid(emb)
    hr.add(["n1", "n2"], ["花生过敏不能吃坚果", "会员卡号 A8821"], scope=MEM)
    emb.calls.clear()
    lists = hr.search_many(["坚果", "卡号"], top_k=2, scope=MEM)
    assert len(lists) == 2 and all(isinstance(lst, list) for lst in lists)   # one ranking per query
    assert emb.calls == [["坚果", "卡号"]]                                    # a single batch embeds both queries (not two calls)


# ---------- metadata filtering (pre-filtering): declared columns + filters end-to-end ----------

def test_metadata_filter_end_to_end():
    """With metadata columns declared: add ingests with metadatas, and search narrows both candidate paths by eq / in."""
    from agentmaker.retrieval import MetadataFilter
    hr = build_sqlite_hybrid(_FakeEmbedder(), metadata_columns=("doc_id", "tag"))
    hr.add(["c1", "c2", "c3"], ["报销制度 交通", "报销制度 住宿", "假勤制度 年假"], scope=MEM,
           metadatas=[{"doc_id": "D1", "tag": "policy"}, {"doc_id": "D1", "tag": "policy"},
                      {"doc_id": "D2", "tag": "hr"}])
    all_hits = hr.search("制度", top_k=10, scope=MEM)
    assert {h.id for h in all_hits} == {"c1", "c2", "c3"}
    only_d1 = hr.search("制度", top_k=10, scope=MEM, filters=[MetadataFilter("doc_id", "D1")])
    assert {h.id for h in only_d1} == {"c1", "c2"}                       # eq hard filter
    by_in = hr.search("制度", top_k=10, scope=MEM, filters=[MetadataFilter("tag", ["hr"], op="in")])
    assert {h.id for h in by_in} == {"c3"}                               # in matches one of several


def test_metadata_filter_undeclared_key_fails_loud():
    """Filtering an undeclared field fails loud (a silent empty result is hard to debug); an illegal operator is caught at construction."""
    from agentmaker.retrieval import MetadataFilter
    hr = build_sqlite_hybrid(_FakeEmbedder(), metadata_columns=("doc_id",))
    hr.add(["c1"], ["hello"], scope=MEM, metadatas=[{"doc_id": "D1"}])
    with pytest.raises(RetrievalError):
        hr.search("hello", top_k=5, scope=MEM, filters=[MetadataFilter("tag", "x")])   # tag not declared
    with pytest.raises(RetrievalError):
        MetadataFilter("doc_id", "x", op="gt")                                          # unsupported operator
    with pytest.raises(RetrievalError):
        build_sqlite_hybrid(_FakeEmbedder(), metadata_columns=("id",))                  # collides with a fixed column


# ---------- embedder fingerprint: a mismatched model swap is caught at open time ----------

class _NamedEmbedder(_FakeEmbedder):
    """Fake embedder with a model_id (for fingerprint validation)."""

    def __init__(self, name, dim=3):
        self._name = name
        self.dim = dim

    @property
    def model_id(self):
        return self._name


def test_embedder_fingerprint_blocks_model_swap(tmp_path):
    """Swapping to a "same dimension, different model" on the same DB is caught by the fingerprint (exactly the most dangerous silent-mixing case without one); reopening the same model is fine."""
    db = str(tmp_path / "fp.db")
    build_sqlite_hybrid(_NamedEmbedder("model-a"), db_path=db).close()
    build_sqlite_hybrid(_NamedEmbedder("model-a"), db_path=db).close()   # reopen same model -> passes
    with pytest.raises(RetrievalError):
        build_sqlite_hybrid(_NamedEmbedder("model-b"), db_path=db)       # same dim, different model -> rejected


def test_embedder_fingerprint_records_known_model(tmp_path):
    """A dimension-only fingerprint accepts and records a compatible named model."""
    db = str(tmp_path / "fp2.db")
    build_sqlite_hybrid(_FakeEmbedder(), db_path=db).close()             # no model_id: records "|3"
    build_sqlite_hybrid(_NamedEmbedder("model-a"), db_path=db).close()   # matching dim -> passes and backfills to model-a
    with pytest.raises(RetrievalError):
        build_sqlite_hybrid(_NamedEmbedder("model-b"), db_path=db)


# ---------- fusion strategy seam: RRF by default, injectable replacement ----------

def test_fusion_strategy_injectable():
    """With a custom FusionStrategy injected, hybrid's fusion uses it (not the hardwired RRF); the default is still RRFFusion."""
    from agentmaker.retrieval import FusionStrategy, RRFFusion

    class _TakeKeywordOnly(FusionStrategy):
        def fuse(self, result_lists, *, top_k):
            return result_lists[1][:top_k]            # by convention [vector path, keyword path]: take only the keyword path

    default = build_sqlite_hybrid(_FakeEmbedder())
    assert isinstance(default.fusion, RRFFusion)      # batteries-included default
    hr = build_sqlite_hybrid(_FakeEmbedder(), fusion=_TakeKeywordOnly())
    hr.add(["k1"], ["会员卡号 A8821"], scope=MEM)
    hits = hr.search("A8821", top_k=3, scope=MEM)
    assert [h.source for h in hits] == ["keyword"]    # all from the keyword path -> injection took effect


# ---------- a* async: awaiting in the event loop doesn't block (to_thread wrapper) ----------

def test_async_search_and_add_smoke():
    """aadd / asearch can be awaited directly in the event loop (same semantics as the sync versions)."""
    import asyncio

    async def go():
        hr = build_sqlite_hybrid(_FakeEmbedder())
        await hr.aadd(["n1"], ["会员卡号 A8821"], scope=MEM)
        hits = await hr.asearch("A8821", top_k=2, scope=MEM)
        assert [h.id for h in hits] == ["n1"]

    asyncio.run(go())


# ---------- SqliteBookkeeping persistent bookkeeping (idempotent across processes, pending survives) ----------

class _CountingRetriever:
    """Counting stub backend: records add calls and counts; failure is settable."""

    def __init__(self):
        self.add_calls = []
        self.fail = False

    def add(self, ids, contents, *, scope=None, **kw):
        if self.fail:
            raise RuntimeError("index down")
        self.add_calls.append(list(ids))

    def delete(self, ids, *, scope=None):
        pass


def test_sqlite_bookkeeping_idempotent_across_instances(tmp_path):
    """Persistent bookkeeping: rewriting the same content in a "new process" (fresh SyncIndexSync + same bookkeeping DB) is still skipped by the fingerprint, no repeat embedding."""
    from agentmaker.retrieval import SqliteBookkeeping, SyncIndexSync
    db = str(tmp_path / "bk.db")
    r1 = _CountingRetriever()
    s1 = SyncIndexSync(r1, bookkeeping=SqliteBookkeeping(db))
    s1.index(["a"], ["hello"], scope=MEM)
    assert len(r1.add_calls) == 1
    r2 = _CountingRetriever()
    s2 = SyncIndexSync(r2, bookkeeping=SqliteBookkeeping(db))   # simulate a restart: brand-new instance, same bookkeeping DB
    s2.index(["a"], ["hello"], scope=MEM)                       # content unchanged -> still skipped across processes
    assert r2.add_calls == []
    s2.index(["a"], ["hello v2"], scope=MEM)                    # changed -> writes
    assert len(r2.add_calls) == 1


def test_sqlite_bookkeeping_pending_survives_restart(tmp_path):
    """A failed write is marked pending: the pending set is persisted, still visible after a restart, and cleared once reconcile converges."""
    from agentmaker.retrieval import SqliteBookkeeping, SyncIndexSync
    from collections import namedtuple
    db = str(tmp_path / "bk2.db")
    r = _CountingRetriever()
    r.fail = True
    s1 = SyncIndexSync(r, bookkeeping=SqliteBookkeeping(db))
    s1.index(["a"], ["hello"], scope=MEM)                       # fails -> pending
    assert s1.pending(scope=MEM) == {"a"}
    r.fail = False
    s2 = SyncIndexSync(r, bookkeeping=SqliteBookkeeping(db))    # restart
    assert s2.pending(scope=MEM) == {"a"}                       # pending survives
    item = namedtuple("I", ("id", "content"))("a", "hello")
    s2.reconcile([item], scope=MEM)                             # reconcile to convergence
    assert s2.pending(scope=MEM) == set()


# ---------- retrieval-backend edges (replace scope guard / dim / config / filter None) ----------

def test_sqlite_hybrid_replace_rejects_empty_scope():
    """SqliteHybridRetriever.replace rejects an empty scope early (otherwise deleting stale would delete these ids across all scopes)."""
    hr = build_sqlite_hybrid(_FakeEmbedder())
    with pytest.raises(RetrievalError):
        hr.replace(["old"], ["new"], ["hi"])                       # scope defaults to None -> rejected


def test_base_hybrid_replace_rejects_empty_scope_before_mutation():
    """The base HybridRetriever.replace rejects an empty scope before the add (the original guard fired after the add, leaving old and new coexisting)."""
    vs = SqliteVecStore(dim=3)
    hr = HybridRetriever(embedder=_FakeEmbedder(), vector_store=vs, keyword_index=_BoomKeyword())
    with pytest.raises(RetrievalError):
        hr.replace(["old"], ["new"], ["hi"])                       # without the moved-up guard it would add first (hitting _BoomKeyword) instead of failing loud
    assert vs.search([1.0, 0.0, 0.0], top_k=10, scope=MEM) == []   # no write happened


def test_sqlite_vec_store_zero_dim_fails_loud():
    """dim=0 fails loud (RetrievalError) when building the vector table, producing no illegal table."""
    with pytest.raises(RetrievalError):
        SqliteVecStore(dim=0)


def test_hybrid_rejects_invalid_config_at_construction():
    """Invalid config (negative rrf_k / candidate_pool < top_k) is rejected at construction, not deferred to search time."""
    from agentmaker.retrieval import RetrievalConfig
    with pytest.raises(ValueError):
        HybridRetriever(_FakeEmbedder(), SqliteVecStore(dim=3), _BoomKeyword(),
                        config=RetrievalConfig(rrf_k=-1))


def test_metadata_filter_eq_rejects_none():
    """A MetadataFilter with op='eq' can't have a None value (in SQL, = NULL is always false -> a silent zero-hit)."""
    from agentmaker.retrieval import MetadataFilter
    with pytest.raises(RetrievalError):
        MetadataFilter("doc_id", None)                             # eq None -> fail loud
    with pytest.raises(RetrievalError):
        MetadataFilter("doc_id", None, op="eq")


# ---------- legacy seam compatibility (stores without exact deletes keep working) ----------

class _LegacyStore:
    """Duck store with only the pre-exact surface (add/search/delete): records ranged deletes."""

    def __init__(self):
        self.rows = {}
        self.ranged_deletes = []

    def add(self, ids, vectors_or_contents, contents=None, *, scope=None, metadatas=None):
        for i in ids:
            self.rows[i] = scope

    def search(self, *args, **kwargs):
        return []

    def delete(self, ids, *, scope=None):
        self.ranged_deletes.append((list(ids), scope))
        for i in ids:
            self.rows.pop(i, None)

    def close(self):
        pass


def test_hybrid_replace_falls_back_to_ranged_delete_for_legacy_stores():
    """replace / delete_exact degrade to ranged deletes when either store lacks delete_exact, instead of raising after the new batch was written."""
    vec, kw = _LegacyStore(), _LegacyStore()
    hr = HybridRetriever(embedder=_FakeEmbedder(), vector_store=vec, keyword_index=kw)

    hr.add(["old1", "old2"], ["旧一", "旧二"], scope=MEM)
    hr.replace(["old1", "old2"], ["new1"], ["新一"], scope=MEM)

    assert set(vec.rows) == set(kw.rows) == {"new1"}
    assert vec.ranged_deletes and kw.ranged_deletes            # stale ids went through the ranged fallback

    hr.delete_exact(["new1"], scope=MEM)
    assert vec.rows == {} and kw.rows == {}


def test_exact_delete_fallback_refuses_fully_empty_footprint():
    """The ranged fallback for an all-empty footprint would span the whole store, so it fails loud instead of deleting."""
    vec, kw = _LegacyStore(), _LegacyStore()
    hr = HybridRetriever(embedder=_FakeEmbedder(), vector_store=vec, keyword_index=kw)
    hr.add(["a"], ["内容"], scope=MEM)
    with pytest.raises(RetrievalError):
        hr.delete_exact(["a"])
    assert "a" in vec.rows and "a" in kw.rows


def test_sync_index_sync_exact_drop_falls_back_for_legacy_retriever():
    """SyncIndexSync.drop_exact on a retriever without delete_exact degrades to a ranged delete and still updates bookkeeping."""
    from agentmaker.retrieval import SyncIndexSync

    retriever = _LegacyStore()
    sync = SyncIndexSync(retriever)
    sync.index(["a"], ["内容"], scope=MEM)
    assert "a" in sync.tracked_ids(scope=MEM)

    sync.drop_exact(["a"], scope=MEM, strict=True)
    assert retriever.ranged_deletes == [(["a"], MEM)]
    assert "a" not in sync.tracked_ids(scope=MEM)


# ---------- retrieval bookkeeping (ghost-pending cleanup + InMemory scope normalization) ----------

class _NoopRetriever:
    """Minimal retriever stub: add / delete are no-ops (reconcile / drop only need these two)."""
    def add(self, ids, contents, *, scope=None, metadatas=None): pass
    def delete(self, ids, *, scope=None): pass
    delete_exact = delete


def test_reconcile_clears_ghost_pending():
    """An id marked pending on failure and then deleted from the source (a ghost): one reconcile stops it lingering in pending() forever."""
    from agentmaker.retrieval import SyncIndexSync
    sync = SyncIndexSync(_NoopRetriever())
    sync.bookkeeping.mark_pending(MEM, ["ghost"])                  # best-effort write failure marks it pending
    assert "ghost" in sync.pending(scope=MEM)
    sync.reconcile([], scope=MEM)
    assert "ghost" not in sync.pending(scope=MEM)


def test_inmemory_bookkeeping_normalizes_none_scope():
    """InMemoryBookkeeping normalizes None to Scope(), matching SqliteBookkeeping (behavior doesn't drift across backends)."""
    from agentmaker.retrieval import InMemoryBookkeeping
    bk = InMemoryBookkeeping()
    bk.set_hashes(None, [("x", "h1")])                            # write with a None scope
    assert bk.get_hash(Scope(), "x") == "h1"                      # read via Scope() (same bucket)
    assert bk.tracked_ids(None) == bk.tracked_ids(Scope()) == {"x"}


def test_pending_aggregates_exact_footprints_under_a_coarse_scope():
    from agentmaker.retrieval import InMemoryBookkeeping

    bookkeeping = InMemoryBookkeeping()
    sync = SyncIndexSync(_NoopRetriever(), bookkeeping=bookkeeping)
    first = Scope(base="memory", user="alice", agent="first")
    second = Scope(base="memory", user="alice", agent="second")
    bookkeeping.mark_pending(first, ["a"])
    bookkeeping.mark_pending(second, ["b"])

    assert sync.pending(scope=Scope(base="memory", user="alice")) == {"a", "b"}


class _FailingBookkeeping(SyncBookkeeping):
    def __init__(self, operation):
        self._delegate = InMemoryBookkeeping()
        self.operation = operation

    def _call(self, operation, *args):
        if self.operation == operation:
            raise RuntimeError(f"bookkeeping {operation} failed")
        return getattr(self._delegate, operation)(*args)

    def get_hash(self, *args): return self._call("get_hash", *args)
    def set_hashes(self, *args): return self._call("set_hashes", *args)
    def delete_hashes(self, *args): return self._call("delete_hashes", *args)
    def tracked_ids(self, *args): return self._call("tracked_ids", *args)
    def mark_pending(self, *args): return self._call("mark_pending", *args)
    def clear_pending(self, *args): return self._call("clear_pending", *args)
    def pending_ids(self, *args): return self._call("pending_ids", *args)
    def exact_scopes(self, *args): return self._call("exact_scopes", *args)
    def close(self): pass


class _StateRetriever:
    def __init__(self):
        self.rows = {}

    def add(self, ids, contents, *, scope=None, metadatas=None):
        self.rows.update(zip(ids, contents))

    def replace(self, old_ids, new_ids, contents, *, scope=None, metadatas=None):
        for id_ in old_ids:
            self.rows.pop(id_, None)
        self.rows.update(zip(new_ids, contents))

    def delete(self, ids, *, scope=None):
        for id_ in ids:
            self.rows.pop(id_, None)

    delete_exact = delete


@pytest.mark.parametrize("operation", ["get_hash", "set_hashes", "clear_pending"])
def test_index_remains_best_effort_when_bookkeeping_fails(operation):
    retriever = _StateRetriever()
    sync = SyncIndexSync(retriever, bookkeeping=_FailingBookkeeping(operation))

    sync.index(["a"], ["value"], scope=MEM)

    assert retriever.rows == {"a": "value"}


@pytest.mark.parametrize("operation", ["delete_hashes", "set_hashes", "clear_pending"])
def test_replace_does_not_report_physical_success_as_failure(operation):
    retriever = _StateRetriever()
    retriever.rows["old"] = "old"
    sync = SyncIndexSync(retriever, bookkeeping=_FailingBookkeeping(operation))

    sync.replace(["old"], ["new"], ["new"], scope=MEM)

    assert retriever.rows == {"new": "new"}


def test_replace_preserves_the_retriever_failure_when_bookkeeping_is_broken():
    class BrokenReplace(_StateRetriever):
        def replace(self, *args, **kwargs):
            raise ValueError("physical replace failed")

    sync = SyncIndexSync(BrokenReplace(), bookkeeping=_FailingBookkeeping("mark_pending"))
    with pytest.raises(ValueError, match="physical replace failed"):
        sync.replace(["old"], ["new"], ["new"], scope=MEM)


def test_drop_and_reconcile_progress_survive_bookkeeping_failures():
    from collections import namedtuple

    retriever = _StateRetriever()
    retriever.rows["old"] = "old"
    drop_sync = SyncIndexSync(retriever, bookkeeping=_FailingBookkeeping("delete_hashes"))
    drop_sync.drop(["old"], scope=MEM)
    assert retriever.rows == {}

    reconcile_sync = SyncIndexSync(retriever, bookkeeping=_FailingBookkeeping("set_hashes"))
    item = namedtuple("Item", ("id", "content"))("new", "value")
    assert reconcile_sync.reconcile([item], scope=MEM) == 1
    assert retriever.rows == {"new": "value"}


def test_drop_bookkeeping_failure_disables_stale_fingerprint_skips():
    retriever = _StateRetriever()
    bookkeeping = _FailingBookkeeping(None)
    sync = SyncIndexSync(retriever, bookkeeping=bookkeeping)
    exact = Scope(base="memory", user="alice", session="s1")

    sync.index(["same"], ["value"], scope=exact)
    bookkeeping.operation = "exact_scopes"
    sync.drop(["same"], scope=Scope(base="memory", user="alice"))
    assert retriever.rows == {}

    sibling_sync = SyncIndexSync(retriever, bookkeeping=bookkeeping)
    sibling_sync.index(["same"], ["value"], scope=exact)
    assert retriever.rows == {"same": "value"}


def test_sqlite_pending_range_prevents_cross_instance_stale_hash_skip(tmp_path):
    from agentmaker.retrieval import SqliteBookkeeping

    retriever = _StateRetriever()
    path = str(tmp_path / "bookkeeping.db")
    first_bookkeeping = SqliteBookkeeping(path)
    second_bookkeeping = SqliteBookkeeping(path)
    first = SyncIndexSync(retriever, bookkeeping=first_bookkeeping)
    second = SyncIndexSync(retriever, bookkeeping=second_bookkeeping)
    exact = Scope(base="memory", user="alice", session="s1")
    coarse = Scope(base="memory", user="alice")

    first.index(["same"], ["value"], scope=exact)
    first_bookkeeping.exact_scopes = lambda scope: (_ for _ in ()).throw(
        RuntimeError("enumeration failed"))
    first.drop(["same"], scope=coarse)
    assert retriever.rows == {}

    second._state["fingerprints_trusted"] = True
    second.index(["same"], ["value"], scope=exact)
    assert retriever.rows == {"same": "value"}
    first_bookkeeping.close()
    second_bookkeeping.close()
