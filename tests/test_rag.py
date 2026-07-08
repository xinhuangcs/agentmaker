"""Regression net for agentmaker.rag (hermetic: stub retriever, no key / no network).

Locks down: source-of-truth composite primary key (chunk_id, scope) isolation + scope-aware get, old-schema open-time
self-check, reimport failure preserving the old version, delete hitting the index before the source of truth, splitter
param validation + metadata merge, loader normalizing to RetrievalError, ingest_text title derivation, RAGTool
action-level confirmation gate, count_tokens CJK/English estimation.
"""

import sqlite3

import pytest

from agentmaker.core.exceptions import RetrievalError
from agentmaker.core.text import count_tokens
from agentmaker.rag import IngestionPipeline, RAGTool, SourceStore, load_file, split_document
from agentmaker.rag.types import Chunk, ChunkingConfig, Document
from agentmaker.rag.splitter import TextSplitter
from agentmaker.retrieval import Scope

RAG = Scope(base="rag")
MD = "# A\nalpha body one.\n\n## B\nbeta body two."


class _FakeRetriever:
    """Stub retriever: records add/delete, can simulate add failure; no embedding, no network."""

    def __init__(self, source_store=None, watch_doc=None):
        self.ids = set()
        self.calls = []           # [(op, ids[, src_remaining])]
        self.fail_add = False
        self._store = source_store
        self._watch = watch_doc

    def add(self, ids, texts, *, scope=None, metadatas=None):
        self.calls.append(("add", list(ids)))
        if self.fail_add:
            raise RuntimeError("embedding boom")
        self.ids.update(ids)

    def delete(self, ids, *, scope=None):
        # on index delete, record how many chunks of watch_doc remain in the source of truth (verifies "index before source")
        remaining = None
        if self._store is not None and self._watch is not None:
            remaining = len(self._store.chunk_ids_of_doc(self._watch, scope=scope))
        self.calls.append(("delete", list(ids), remaining))
        self.ids.difference_update(ids)

    def replace(self, old_ids, new_ids, contents, *, scope=None, metadatas=None):
        # stub mirrors the base compensating impl (add then delete-old): if add fails it raises and doesn't delete old -- same semantics as HybridRetriever.replace
        self.add(new_ids, contents, scope=scope)
        stale = [i for i in old_ids if i not in set(new_ids)]
        if stale:
            self.delete(stale, scope=scope)


# ---------- SourceStore: composite primary key (chunk_id, scope) ----------

def test_source_store_scope_isolation_and_get():
    """Same chunk_id in different scopes stores one row each without clobbering; get fetches the chunk for its scope."""
    s = SourceStore()
    s.save_chunks([Chunk(content="alice", chunk_id="dup", doc_id="d")], scope=Scope(base="rag", user="alice"))
    s.save_chunks([Chunk(content="bob", chunk_id="dup", doc_id="d")], scope=Scope(base="rag", user="bob"))
    assert s.get("dup", scope=Scope(base="rag", user="alice")).content == "alice"
    assert s.get("dup", scope=Scope(base="rag", user="bob")).content == "bob"


def test_source_store_delete_chunks_keeps_sibling_scope():
    """delete_chunks matches all dimensions exactly, deleting only this scope's chunk and sparing the sibling scope with the same id."""
    s = SourceStore()
    s.save_chunks([Chunk(content="a", chunk_id="dup", doc_id="d")], scope=Scope(base="rag", user="alice"))
    s.save_chunks([Chunk(content="b", chunk_id="dup", doc_id="d")], scope=Scope(base="rag", user="bob"))
    s.delete_chunks(["dup"], scope=Scope(base="rag", user="alice"))
    assert s.get("dup", scope=Scope(base="rag", user="alice")) is None
    assert s.get("dup", scope=Scope(base="rag", user="bob")).content == "b"


def test_source_store_schema_guard_rejects_old_pk(tmp_path):
    """Opening a DB with the old schema (single-column chunk_id PK) fails loud instead of silently using the wrong schema."""
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT, content TEXT)")  # old single-column PK
    conn.commit()
    conn.close()
    with pytest.raises(RetrievalError):
        SourceStore(db)


def test_source_store_fresh_db_passes_guard(tmp_path):
    """A fresh DB has the correct PK and passes the self-check; reopening the same DB doesn't raise either."""
    db = str(tmp_path / "fresh.db")
    s = SourceStore(db)
    s.save_chunks([Chunk(content="hi", chunk_id="c1", doc_id="d")], scope=RAG)
    assert s.get("c1", scope=RAG).content == "hi"
    s.close()
    SourceStore(db).close()  # reopen (schema is already the composite PK) without raising


# ---------- splitter ----------

def test_splitter_rejects_invalid_params():
    """Invalid chunking params (chunk<=0, overlap>=chunk) raise ValueError."""
    for ct, ov in [(0, 0), (-1, 0), (100, 100), (100, 150)]:
        with pytest.raises(ValueError):
            TextSplitter(ct, ov)


def test_splitter_valid_params_ok():
    """Valid params construct fine."""
    TextSplitter(100, 20)
    TextSplitter(512, 0)


def test_splitter_merges_doc_metadata():
    """doc.metadata merges into every chunk's metadata (custom fields aren't lost)."""
    doc = Document(content="hello world foo bar", format="txt", source="n.txt", metadata={"author": "jason"})
    chunks = split_document(doc)
    assert chunks[0].metadata["author"] == "jason"
    assert chunks[0].metadata["source"] == "n.txt"


def test_splitter_uses_injected_counter():
    """split_document's chunking budget uses the injected counter: a more "expensive" counter -> the same doc splits into more chunks."""
    doc = Document(content="\n\n".join(f"段落{i}的内容文字" for i in range(12)), format="txt", source="d.txt")
    default = split_document(doc, chunk_tokens=50, overlap_tokens=0)
    inflated = split_document(doc, chunk_tokens=50, overlap_tokens=0, token_counter=lambda s: count_tokens(s) * 5)
    assert len(inflated) > len(default)        # each paragraph counts as more "expensive" -> fewer paragraphs fit per chunk -> more chunks


# ---------- loader: normalize to RetrievalError ----------

def test_loader_missing_file_raises_retrieval_error():
    """A missing file is normalized to RetrievalError."""
    with pytest.raises(RetrievalError):
        load_file("/no/such/file.txt")


def test_loader_bad_jsonl_raises_retrieval_error(tmp_path):
    """A bad JSONL line is normalized to RetrievalError (not a raw JSONDecodeError)."""
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\n{not json}\n', encoding="utf-8")
    with pytest.raises(RetrievalError):
        load_file(str(p))


# ---------- IngestionPipeline: upsert / delete consistency ----------

def test_ingest_text_derives_title_from_source():
    """When ingest_text gets no title it derives one from the source filename, into the chunk's heading_path."""
    store = SourceStore()
    pipe = IngestionPipeline(retriever=_FakeRetriever(), source_store=store)
    res = pipe.ingest_text("hello world this is body text", source="/x/手册.txt")
    cid = store.chunk_ids_of_doc(res.doc_id, scope=pipe.scope)[0]
    assert "手册" in store.get(cid, scope=pipe.scope).heading_path


def test_ingest_uses_injected_counter():
    """IngestionPipeline threads the injected counter into chunking: a more "expensive" counter -> the same text ingests into more chunks."""
    text = "\n\n".join(f"段落{i}的内容文字" for i in range(12))
    cfg = ChunkingConfig(chunk_tokens=50, overlap_tokens=0)
    r_default = IngestionPipeline(retriever=_FakeRetriever(), source_store=SourceStore(), config=cfg).ingest_text(text, source="d.txt")
    r_big = IngestionPipeline(retriever=_FakeRetriever(), source_store=SourceStore(), config=cfg,
                              token_counter=lambda s: count_tokens(s) * 5).ingest_text(text, source="d.txt")
    assert r_big.chunks > r_default.chunks


def test_reimport_preserves_old_version_on_failure():
    """If indexing fails on reimport -> roll back the new chunks and keep the old version (don't delete the doc entirely)."""
    store = SourceStore()
    fake = _FakeRetriever()
    pipe = IngestionPipeline(retriever=fake, source_store=store)
    pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")
    old = set(store.chunk_ids_of_doc("DOC", scope=pipe.scope))
    fake.fail_add = True
    with pytest.raises(RuntimeError):
        pipe.ingest_text("# A\nNEW one.\n\n## B\nNEW two.", source="d.md", fmt="md", doc_id="DOC")
    assert set(store.chunk_ids_of_doc("DOC", scope=pipe.scope)) == old   # old version kept intact
    assert old <= fake.ids                                               # the index still holds the old version too


def test_reimport_success_replaces_old():
    """Successful reimport -> only brand-new chunk_ids remain; the old version is cleared from both the source of truth and the index."""
    store = SourceStore()
    fake = _FakeRetriever()
    pipe = IngestionPipeline(retriever=fake, source_store=store)
    pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")
    old = set(store.chunk_ids_of_doc("DOC", scope=pipe.scope))
    pipe.ingest_text("# A\nNEW one.\n\n## B\nNEW two.", source="d.md", fmt="md", doc_id="DOC")
    new = set(store.chunk_ids_of_doc("DOC", scope=pipe.scope))
    assert new.isdisjoint(old)              # all-new ids
    assert not (old & fake.ids)             # old ids already cleared from the index
    assert new <= fake.ids


def test_delete_document_deletes_index_before_source():
    """delete_document removes the search index first, then the source of truth (even on failure, no searchable orphan is left)."""
    store = SourceStore()
    fake = _FakeRetriever(source_store=store, watch_doc="DOC")
    pipe = IngestionPipeline(retriever=fake, source_store=store)
    pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")
    n_chunks = len(store.chunk_ids_of_doc("DOC", scope=pipe.scope))
    fake.calls.clear()
    removed = pipe.delete_document("DOC")
    assert removed == n_chunks
    op, _ids, src_remaining = fake.calls[0]
    assert op == "delete" and src_remaining == n_chunks   # source of truth still present when the index is deleted -> index goes first
    assert store.chunk_ids_of_doc("DOC", scope=pipe.scope) == []
    assert fake.ids == set()


def test_ingest_routes_writes_through_injected_index_sync():
    """The ingest write path goes through the injected IndexSync (proof the sync default can be swapped for an async / distributed impl: memory and rag share one seam)."""
    calls = []

    class _RecordingSync:  # duck-typed stub seam: just records which methods were called
        def index(self, ids, contents, *, scope=None): calls.append("index")
        def replace(self, old, new, contents, *, scope=None, metadatas=None): calls.append(("replace", len(new)))
        def drop(self, ids, *, scope=None): calls.append(("drop", list(ids)))
        def reconcile(self, items, *, scope=None, batch_size=256): return len(list(items))
        def pending(self, *, scope=None): return set()

    pipe = IngestionPipeline(retriever=_FakeRetriever(), source_store=SourceStore(), index_sync=_RecordingSync())
    pipe.ingest_text("hello there body text", source="x.txt", doc_id="D")
    assert any(isinstance(c, tuple) and c[0] == "replace" for c in calls)   # ingest goes through the seam's atomic replace


def test_retrieve_filters_and_self_heals_orphan():
    """A retrieval hit whose source of truth is gone (orphan) -> the index's stale content isn't returned, and a read-time self-heal removes the orphan from the index (like memory.search)."""
    from agentmaker.rag import RagRetriever
    from agentmaker.retrieval.types import RetrievalResult

    class _OrphanRetriever:
        def __init__(self): self.deleted = []
        def search(self, query, *, top_k=5, candidate_pool=20, scope=None):
            return [RetrievalResult(content="STALE 已删残留内容", score=1.0, source="vector", id="ghost")]
        def delete(self, ids, *, scope=None): self.deleted += list(ids)

    class _EmptyStore:
        def get(self, chunk_id, *, scope=None): return None   # not present in the source of truth

    fr = _OrphanRetriever()
    res = RagRetriever(fr, _EmptyStore(), _FakeLLM("")).retrieve("Q", top_k=2)
    assert res == []                 # never leak the index's stale content as a result
    assert fr.deleted == ["ghost"]   # read-time self-heal: orphan cleared from the index


# ---------- RAGTool: action-level confirmation gate ----------

def test_ragtool_confirms_only_add_document():
    """needs_confirmation returns True only for add_document; every other action passes (through the unified confirmation gate)."""
    tool = RAGTool(pipeline=object(), rag_retriever=object())
    assert tool.needs_confirmation({"action": "add_document"}) is True
    for action in ["add_text", "search", "ask", "stats", ""]:
        assert tool.needs_confirmation({"action": action}) is False


# ---------- count_tokens ----------

def test_count_tokens_cn_en():
    """CJK counts per character, the rest at ~4 chars/token; empty string is 0."""
    assert count_tokens("你好世界") == 4
    assert count_tokens("hello world") == 3          # 11 chars -> (11+3)//4 = 3
    assert count_tokens("我有 3 个 apple") == 6       # 3 CJK chars + 9 other chars -> 3 + (9+3)//4 = 6
    assert count_tokens("") == 0


# ---------- RagRetriever: injectable anti-hallucination prompt ----------

def test_ragretriever_system_prompt_injectable():
    """system_prompt can be injected to override the default; _build_messages uses the instance prompt and assembles chunks into numbered sources + the question."""
    from agentmaker.rag.retriever import RagRetriever, DEFAULT_ASK_PROMPT
    from agentmaker.retrieval.types import RetrievalResult
    r = RagRetriever(retriever=object(), source_store=object(), llm=object())
    assert r.system_prompt is DEFAULT_ASK_PROMPT                        # not passed = framework default
    r2 = RagRetriever(retriever=object(), source_store=object(), llm=object(), system_prompt="X")
    msgs = r2._build_messages("q", [RetrievalResult(content="hi", score=1.0, source="rag", id="c1")])
    assert msgs[0] == {"role": "system", "content": "X"}               # uses the injected prompt
    assert "[1] hi" in msgs[1]["content"] and "[Question]\nq" in msgs[1]["content"]


def test_component_prompts_overridable():
    """Each RAG component's built-in prompt can be wholesale-overridden via a constructor arg; omitted, it uses the public DEFAULT_* default (behavior unchanged)."""
    from agentmaker.rag import (DEFAULT_CONTEXT_PROMPT, DEFAULT_HYDE_PROMPT, DEFAULT_MQE_PROMPT,
                            HyDETransformer, LLMContextualizer, MultiQueryExpander)
    # omitted = public default constant; passed = your own
    assert MultiQueryExpander(object()).expand_prompt is DEFAULT_MQE_PROMPT
    assert HyDETransformer(object()).hyde_prompt is DEFAULT_HYDE_PROMPT
    assert LLMContextualizer(object()).context_prompt is DEFAULT_CONTEXT_PROMPT
    assert MultiQueryExpander(object(), expand_prompt="MQE!").expand_prompt == "MQE!"
    assert HyDETransformer(object(), hyde_prompt="HYDE!").hyde_prompt == "HYDE!"
    assert LLMContextualizer(object(), context_prompt="CTX!").context_prompt == "CTX!"

    # end-to-end: the custom prompt is actually sent to the LLM as the system message
    rec = {}

    class _RecLLM:
        async def chat(self, messages):
            rec["system"] = messages[0]["content"]
            return _Reply("改写A\n改写B")

    MultiQueryExpander(_RecLLM(), expand_prompt="MQE!").transform("住宿能报多少")
    assert rec["system"] == "MQE!"


# ---------- Query expansion MQE / HyDE (§5, off by default) ----------

class _Reply:
    def __init__(self, content): self.content = content


class _FakeLLM:
    """Stub LLM: chat returns preset text; when fail=True it raises (exercises the fallback)."""
    def __init__(self, content):
        self._content = content
        self.fail = False
    async def chat(self, messages):
        if self.fail:
            raise RuntimeError("llm down")
        return _Reply(self._content)


class _SearchRetriever:
    """Stub retriever: records searched queries; search is single, search_many is batch, each query returns 1 result (the id embeds the query, to make fan-out assertions easy)."""
    def __init__(self): self.queries = []
    def _hit(self, query):
        from agentmaker.retrieval.types import RetrievalResult
        self.queries.append(query)
        return [RetrievalResult(content="c", score=1.0, source="vector", id="id-" + query)]
    def search(self, query, *, top_k=5, candidate_pool=20, scope=None):
        return self._hit(query)
    def search_many(self, queries, *, top_k=5, candidate_pool=20, scope=None):
        return [self._hit(q) for q in queries]


class _GetStore:
    """Stub source of truth: get returns one Chunk."""
    def get(self, chunk_id, *, scope=None):
        return Chunk(content="chunk-" + chunk_id, chunk_id=chunk_id, doc_id="d")


def test_mqe_transform():
    """MQE: original query + rewrites; dirty responses get symbols stripped / blanks filtered / truncated to n; on LLM failure falls back to [query]."""
    from agentmaker.rag import MultiQueryExpander
    qs = MultiQueryExpander(_FakeLLM("住宿费上限\n酒店费用"), n=2).transform("住宿能报多少")
    assert qs[0] == "住宿能报多少" and "住宿费上限" in qs and "酒店费用" in qs
    # list markers + blank lines + more than n items -> strip -•*, drop blanks, truncate to n (no empty queries, no unbounded fan-out)
    assert MultiQueryExpander(_FakeLLM("- A\n\n  \n• B\n* C"), n=2).transform("Q") == ["Q", "A", "B"]
    boom = _FakeLLM("x")
    boom.fail = True
    assert MultiQueryExpander(boom).transform("Q") == ["Q"]
    with pytest.raises(ValueError):       # n < 1 raises at construction
        MultiQueryExpander(_FakeLLM("x"), n=0)


def test_search_sanitizes_transformer_output():
    """_search sanitizes a custom transformer's output: drops empty / non-string, truncates to the cap, and never feeds a dirty query to the backend."""
    from agentmaker.rag import QueryTransformer, RagRetriever

    class _Junk(QueryTransformer):
        def transform(self, query):
            # empty / whitespace / non-string / unhashable(list) / over-cap -- sanitizing must handle all without crashing (an unhashable value would break dict.fromkeys)
            return ["Q", "", "  ", 123, ["unhashable"], "v"] + [f"x{i}" for i in range(20)]

    fr = _SearchRetriever()
    RagRetriever(fr, _GetStore(), _FakeLLM(""), query_transformer=_Junk()).retrieve("Q", top_k=2)
    assert "" not in fr.queries and "  " not in fr.queries and 123 not in fr.queries   # dirty values dropped
    assert len(fr.queries) <= 8                                                         # truncated to _MQ_MAX_QUERIES


def test_hyde_transform():
    """HyDE: original query + hypothetical document; on LLM failure falls back to [query]."""
    from agentmaker.rag import HyDETransformer
    assert HyDETransformer(_FakeLLM("公司住宿费每晚 500 元。")).transform("住宿能报多少") \
        == ["住宿能报多少", "公司住宿费每晚 500 元。"]
    boom = _FakeLLM("x")
    boom.fail = True
    assert HyDETransformer(boom).transform("Q") == ["Q"]


def test_retrieve_default_off_single_query():
    """No query_transformer by default: retrieve searches once with the original query."""
    from agentmaker.rag import RagRetriever
    fr = _SearchRetriever()
    RagRetriever(fr, _GetStore(), _FakeLLM("")).retrieve("住宿能报多少", top_k=3)
    assert fr.queries == ["住宿能报多少"]


def test_retrieve_mqe_fans_out_and_merges():
    """With MQE on: the original query + rewrites each get searched, results merged by RRF."""
    from agentmaker.rag import MultiQueryExpander, RagRetriever
    fr = _SearchRetriever()
    rag = RagRetriever(fr, _GetStore(), _FakeLLM(""),
                       query_transformer=MultiQueryExpander(_FakeLLM("v1\nv2"), n=2))
    res = rag.retrieve("Q", top_k=5)
    assert set(fr.queries) == {"Q", "v1", "v2"}    # fan-out: original query + 2 rewrites
    assert res                                      # non-empty after RRF merge


# ---------- filters pass-through + multi-query fusion seam + RAGTool filter fields ----------

def test_retrieve_passes_filters_only_when_given():
    """retrieve passes filters through only when given (doesn't force stubs to grow the parameter); when given, they reach the backend as-is."""
    from agentmaker.rag import RagRetriever
    from agentmaker.retrieval import MetadataFilter
    from agentmaker.retrieval.types import RetrievalResult

    class _SpyRetriever:
        def __init__(self): self.kwargs = None
        def search(self, query, *, top_k=5, candidate_pool=20, scope=None, **kw):
            self.kwargs = kw
            return [RetrievalResult(content="c", score=1.0, source="vector", id="i1")]

    spy = _SpyRetriever()
    r = RagRetriever(spy, _GetStore(), _FakeLLM(""))
    r.retrieve("Q", top_k=1)
    assert spy.kwargs == {}                                            # no filters given -> absent from the call
    f = [MetadataFilter("doc_id", "D1")]
    r.retrieve("Q", top_k=1, filters=f)
    assert spy.kwargs == {"filters": f}                                # given -> passed through as-is


def test_multi_query_fusion_uses_retriever_fusion_seam():
    """The multi-query path (with a query_transformer) uses the backend's fusion seam (injecting a custom fusion bypasses the hardwired RRF)."""
    from agentmaker.rag import QueryTransformer, RagRetriever
    from agentmaker.retrieval.types import RetrievalResult

    class _TwoQueries(QueryTransformer):
        def transform(self, query):
            return [query, query + " 改写"]

    class _MarkerFusion:                      # duck-typed fusion stub: returns one marker result to prove it was used
        def fuse(self, result_lists, *, top_k):
            return [RetrievalResult(content="FUSED", score=1.0, source="vector", id="fused")]

    class _ManyRetriever:
        fusion = _MarkerFusion()
        def search_many(self, queries, *, top_k=5, candidate_pool=20, scope=None, **kw):
            return [[RetrievalResult(content=q, score=1.0, source="vector", id="id-" + q)] for q in queries]

    r = RagRetriever(_ManyRetriever(), _GetStore(), _FakeLLM(""), query_transformer=_TwoQueries())
    hits = r.retrieve("Q", top_k=2)
    assert [h.id for h in hits] == ["fused"]                           # fusion went through the injected seam


def test_ragtool_filter_fields_become_params_and_filters():
    """RAGTool(filter_fields=) exposes filter fields as parameters; a value the model fills is assembled into a MetadataFilter and passed to retrieval."""
    from agentmaker.rag import RAGTool, RagRetriever
    from agentmaker.retrieval.types import RetrievalResult

    class _SpyRetriever:
        def __init__(self): self.filters = "UNSET"
        def search(self, query, *, top_k=5, candidate_pool=20, scope=None, filters=None, **kw):
            self.filters = filters
            return [RetrievalResult(content="hit", score=1.0, source="vector", id="i1")]

    spy = _SpyRetriever()
    rr = RagRetriever(spy, _GetStore(), _FakeLLM(""))
    tool = RAGTool(pipeline=object(), rag_retriever=rr, filter_fields=("doc_id",))
    assert any(p.name == "doc_id" for p in tool.get_parameters())      # exposed as an optional parameter
    tool.run({"action": "search", "query": "Q", "doc_id": "D1"})
    assert spy.filters and spy.filters[0].key == "doc_id" and spy.filters[0].value == "D1"
    tool.run({"action": "search", "query": "Q"})                       # unfilled -> no filter
    assert spy.filters is None


# ---------- doc-hash short-circuit / rebuild / neighbor expansion / persistent bookkeeping ----------

def test_reimport_unchanged_doc_short_circuits():
    """Reimporting an unchanged doc short-circuits the whole thing (no replace call / no re-embedding); only changed content re-ingests; changing the chunking params counts as changed too."""
    store = SourceStore()
    fake = _FakeRetriever()
    pipe = IngestionPipeline(retriever=fake, source_store=store)
    r1 = pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")
    n_calls = len(fake.calls)
    r2 = pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")          # unchanged -> short-circuit
    assert r2.skipped is True and r2.chunks == r1.chunks
    assert len(fake.calls) == n_calls                                          # no index calls at all
    r3 = pipe.ingest_text(MD + "\n新增一段。", source="d.md", fmt="md", doc_id="DOC")   # changed -> re-ingest
    assert r3.skipped is False and len(fake.calls) > n_calls
    r4 = pipe.ingest_text(MD + "\n新增一段。", source="d.md", fmt="md", doc_id="DOC",
                          chunk_tokens=64, overlap_tokens=8)                   # changed chunking params -> not wrongly skipped
    assert r4.skipped is False


def test_ingest_file_stable_doc_id_and_skip(tmp_path):
    """ingest_file derives a stable doc_id from the file path: reimporting the same file doesn't re-ingest (short-circuit); only an edited file replaces."""
    p = tmp_path / "n.md"
    p.write_text(MD, encoding="utf-8")
    store = SourceStore()
    pipe = IngestionPipeline(retriever=_FakeRetriever(), source_store=store)
    r1 = pipe.ingest_file(str(p))
    r2 = pipe.ingest_file(str(p))                                              # same file unchanged -> short-circuit
    assert r2.doc_id == r1.doc_id and r2.skipped is True
    assert pipe.stats()["documents"] == 1                                      # no duplicate documents pile up
    p.write_text(MD + "\n更新了。", encoding="utf-8")
    r3 = pipe.ingest_file(str(p))                                              # file changed -> replace the same doc
    assert r3.doc_id == r1.doc_id and r3.skipped is False
    assert pipe.stats()["documents"] == 1


def test_delete_document_clears_doc_hash():
    """Deleting a doc also clears its ingest fingerprint: reimporting the same content after deletion should truly ingest, not be short-circuited by the stale fingerprint."""
    store = SourceStore()
    pipe = IngestionPipeline(retriever=_FakeRetriever(), source_store=store)
    pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")
    pipe.delete_document("DOC")
    r = pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")
    assert r.skipped is False and r.chunks > 0


# ---------- ingest/ask return frozen dataclasses (IngestReport / AskResult / SourceRef) ----------

def test_ingest_report_shape():
    """ingest_text/ingest_file return an IngestReport: has doc_id/chunks, and skipped is always present (default False, True when short-circuited)."""
    from agentmaker.rag import IngestReport
    pipe = IngestionPipeline(retriever=_FakeRetriever(), source_store=SourceStore())
    r = pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")
    assert isinstance(r, IngestReport) and r.skipped is False and r.chunks > 0 and r.doc_id == "DOC"
    r2 = pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")          # reimport unchanged -> short-circuit
    assert r2.skipped is True


def test_ask_result_shape():
    """ask returns an AskResult: answer text + sources (list[SourceRef], n starts at 1); no hits -> sources == []."""
    import asyncio

    from agentmaker.rag import AskResult, RagRetriever, SourceRef
    res = asyncio.run(RagRetriever(_SearchRetriever(), _GetStore(), _FakeLLM("答案")).ask("Q", top_k=2))
    assert isinstance(res, AskResult) and res.answer == "答案"
    assert res.sources and all(isinstance(s, SourceRef) for s in res.sources) and res.sources[0].n == 1

    class _EmptyRetriever:
        def search(self, query, *, top_k=5, candidate_pool=20, scope=None):
            return []
    empty = asyncio.run(RagRetriever(_EmptyRetriever(), _GetStore(), _FakeLLM("x")).ask("Q"))
    assert empty.sources == [] and empty.answer == "Not mentioned in the sources."   # no hits -> the rag.no_hits fallback


def test_rag_dataclasses_exported_from_top_level():
    """The three RAG output types are importable from the agentmaker top level (public API)."""
    from agentmaker import AskResult, IngestReport, SourceRef
    assert IngestReport and AskResult and SourceRef


def test_rag_rebuild_index_from_source():
    """rebuild_index: after the index is lost entirely, re-populate it in full from the source of truth (symmetric with Memory.rebuild_index)."""
    store = SourceStore()
    fake = _FakeRetriever()
    pipe = IngestionPipeline(retriever=fake, source_store=store)
    pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")
    ids = set(store.chunk_ids_of_doc("DOC", scope=pipe.scope))
    fake.ids.clear()                                                           # simulate index loss / swapping to an empty backend
    n = pipe.rebuild_index()
    assert n == len(ids) and ids <= fake.ids                                   # everything re-populated into the index


def test_source_store_get_doc_chunks_ordered_and_ranged():
    """get_doc_chunks is ordered by idx; index_range takes neighbor chunks over a closed interval."""
    s = SourceStore()
    s.save_chunks([Chunk(content=f"c{i}", chunk_id=f"id{i}", doc_id="D", index=i) for i in (2, 0, 3, 1)], scope=RAG)
    assert [c.index for c in s.get_doc_chunks("D", scope=RAG)] == [0, 1, 2, 3]
    assert [c.content for c in s.get_doc_chunks("D", index_range=(1, 2), scope=RAG)] == ["c1", "c2"]


def test_neighbor_window_expander_merges_and_dedupes():
    """Neighbor-window expansion: a hit chunk expands to merge ±1 neighbors; when two hit windows overlap they dedupe by (doc_id, idx), no repeated content."""
    from agentmaker.rag import NeighborWindowExpander
    from agentmaker.retrieval.types import RetrievalResult
    s = SourceStore()
    s.save_chunks([Chunk(content=f"c{i}", chunk_id=f"id{i}", doc_id="D", index=i) for i in range(4)], scope=RAG)
    ex = NeighborWindowExpander(window=1)
    hits = [RetrievalResult(content="c2", score=0.9, source="rag", id="id2", metadata={"doc_id": "D", "index": 2}),
            RetrievalResult(content="c1", score=0.5, source="rag", id="id1", metadata={"doc_id": "D", "index": 1})]
    out = ex.expand(hits, source_store=s, scope=RAG)
    assert out[0].content == "c1\nc2\nc3"                       # the high-score hit gets the full window (1..3)
    assert len(out) == 2 and out[1].content == "c0"             # the low-score hit's window (0..2) keeps only the still-free c0
    out2 = ex.expand([hits[0], hits[0]], source_store=s, scope=RAG)
    assert len(out2) == 1                                       # a fully overlapping window is skipped


def test_retrieve_applies_expander():
    """With RagRetriever(expander=) injected, retrieve's hits are expanded (metadata carries index to support locating them)."""
    from agentmaker.rag import NeighborWindowExpander, RagRetriever
    from agentmaker.retrieval.types import RetrievalResult
    s = SourceStore()
    s.save_chunks([Chunk(content=f"c{i}", chunk_id=f"id{i}", doc_id="D", index=i) for i in range(3)], scope=RAG)

    class _HitRetriever:
        def search(self, query, *, top_k=5, candidate_pool=20, scope=None, **kw):
            return [RetrievalResult(content="c1", score=1.0, source="vector", id="id1")]

    r = RagRetriever(_HitRetriever(), s, _FakeLLM(""), scope=RAG, expander=NeighborWindowExpander(window=1))
    hits = r.retrieve("Q", top_k=1)
    assert hits[0].content == "c0\nc1\nc2"                      # hit id1 (idx=1) -> expanded and merged to 0..2


# ---------- Markdown code fences + loader size cap + fingerprint includes format ----------

def test_markdown_code_fence_lines_not_treated_as_headings():
    """A line starting with # inside a code fence is a code comment, not a heading -- not mis-split, code kept intact, real heading levels undisturbed."""
    from agentmaker.rag.splitter import MarkdownSplitter
    md = (
        "# Title\n\nIntro.\n\n"
        "```python\n"
        "# not a heading, a comment\n"
        "def f():\n"
        "    ## also not a heading\n"
        "    return 1\n"
        "```\n\n"
        "## Real Section\n\nBody."
    )
    secs = MarkdownSplitter()._split_by_heading(md)
    assert [p for p, _ in secs] == ["Title", "Title > Real Section"]   # only two real headings
    title_body = next(b for p, b in secs if p == "Title")
    assert "# not a heading, a comment" in title_body                  # the code comment is kept verbatim, not swallowed as a heading
    assert "## also not a heading" in title_body


def test_markdown_unclosed_fence_extends_to_end():
    """An unclosed code fence extends to end-of-document per CommonMark; subsequent # lines are no longer treated as headings."""
    from agentmaker.rag.splitter import MarkdownSplitter
    md = "# Title\n\n```\n# still code\n## still code\nmore"
    secs = MarkdownSplitter()._split_by_heading(md)
    assert [p for p, _ in secs] == ["Title"]                          # no new section after the fence


def test_load_file_rejects_oversize(tmp_path):
    """load_file over max_bytes fails loud (RetrievalError), not reading an oversized file wholesale into memory."""
    big = tmp_path / "big.txt"
    big.write_text("x" * 5000)
    with pytest.raises(RetrievalError):
        load_file(str(big), max_bytes=1000)
    assert load_file(str(big), max_bytes=10000).content == "x" * 5000  # within the cap it reads normally


def test_fingerprint_includes_format_reimport_not_skipped():
    """Reimporting the same doc_id with a different format isn't short-circuited by the fingerprint (txt->md split differently, so it must re-split and re-index)."""
    store = SourceStore()
    pipe = IngestionPipeline(retriever=_FakeRetriever(), source_store=store)
    body = "# A\nalpha.\n\n## B\nbeta."
    pipe.ingest_text(body, source="d", fmt="txt", doc_id="DOC")
    assert pipe.ingest_text(body, source="d", fmt="txt", doc_id="DOC").skipped is True   # identical -> short-circuit
    assert pipe.ingest_text(body, source="d", fmt="md", doc_id="DOC").skipped is False   # only format changed -> not skipped


# ---------- single-query candidate_pool pass-through + verify() divergence detection ----------

def test_retrieve_large_top_k_passes_candidate_pool():
    """The single-query path passes candidate_pool >= top_k: a top_k larger than the default candidate pool no longer raises RetrievalError."""
    from agentmaker.rag import RagRetriever
    from agentmaker.retrieval.hybrid import require_valid_top_k
    from agentmaker.retrieval.types import RetrievalResult

    class _StrictRetriever:
        def search(self, query, *, top_k=5, candidate_pool=20, scope=None):
            require_valid_top_k(top_k, candidate_pool=candidate_pool)   # same check as the real backend: pool < top_k raises
            return [RetrievalResult(content=f"c{i}", score=1.0, source="vector", id=f"c{i}") for i in range(top_k)]

    res = RagRetriever(_StrictRetriever(), _GetStore(), _FakeLLM("")).retrieve("Q", top_k=25)
    assert len(res) == 25       # before the fix the single-query path dropped candidate_pool -> top_k=25 > default pool -> RetrievalError


def test_verify_detects_source_index_divergence():
    """A failed old-chunk cleanup leaves stale source-of-truth rows -> verify()'s cross-check reports divergence (source_only holds the old-version ids)."""
    store = SourceStore()
    pipe = IngestionPipeline(retriever=_FakeRetriever(), source_store=store)
    pipe.ingest_text(MD, source="d.md", fmt="md", doc_id="DOC")
    old = set(store.chunk_ids_of_doc("DOC", scope=pipe.scope))
    assert pipe.verify()["consistent"] is True                         # consistent after a normal ingest

    def _boom(ids, *, scope=None):                                     # simulate the source-of-truth delete failing during old-chunk cleanup
        raise RuntimeError("delete boom")
    store.delete_chunks = _boom
    pipe.ingest_text("# A\nNEW one.\n\n## B\nNEW two.", source="d.md", fmt="md", doc_id="DOC")   # best-effort old-chunk cleanup fails
    report = pipe.verify()
    assert report["consistent"] is False
    assert set(report["source_only"]) >= old                           # old-version chunks linger in the source of truth; the index no longer tracks them
