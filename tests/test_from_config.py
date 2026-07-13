"""from_config classmethod tests (hermetic: fake embedder + SQLite memory database, no network).

Locks down the assembly convenience layer: from_config on Memory / RagRetriever / IngestionPipeline.
It defaults to the sqlite backend (one-line stack, correct sub-config slicing), backends are injectable
(the assembly root lives in the app, so swapping in pgvector etc. needs no framework source changes),
and a missing embedder with no injected backend raises a clear error.
"""

import pytest

from agentmaker import (ChunkingConfig, IngestionPipeline, AgentmakerConfig, Memory, MemoryConfig,
                    RagConfig, RagRetriever, RetrievalConfig, Scope)
from agentmaker.core.exceptions import RetrievalError
from agentmaker.retrieval import HybridRetriever


class _FakeEmbedder:
    """Fake embedder: returns fixed-dimension vectors, no network."""
    dim = 3

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class _AssemblyResource:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class _AssemblySync(_AssemblyResource):
    def __init__(self, bookkeeping):
        super().__init__()
        self.bookkeeping = bookkeeping

    def close(self):
        super().close()
        self.bookkeeping.close()


# ---- default sqlite backend: one-line stack + correct sub-config slicing ----

def test_memory_from_config_default_sqlite():
    m = Memory.from_config(AgentmakerConfig(memory=MemoryConfig(search_top_k=9)), embedder=_FakeEmbedder())
    assert isinstance(m, Memory)
    assert isinstance(m.retriever, HybridRetriever)   # default built the sqlite backend
    assert m.cfg.search_top_k == 9                    # config.memory sliced out correctly


def test_rag_from_config_default_and_shared_backend():
    cfg = AgentmakerConfig(retrieval=RetrievalConfig(top_k=8), rag=RagConfig(mq_max_queries=3))
    rag = RagRetriever.from_config(cfg, embedder=_FakeEmbedder())
    assert isinstance(rag, RagRetriever) and rag.cfg.top_k == 8 and rag.rag_cfg.mq_max_queries == 3
    # ingestion reuses rag's same backend (the app's shared-assembly pattern: read and write the same data)
    ingestor = IngestionPipeline.from_config(cfg, retriever=rag.retriever, source_store=rag.source_store)
    assert ingestor.retriever is rag.retriever and ingestor.source_store is rag.source_store
    assert ingestor.cfg.chunk_tokens == ChunkingConfig().chunk_tokens   # config.chunking slice


@pytest.mark.parametrize("cls", [RagRetriever, IngestionPipeline])
def test_rag_from_config_close_owns_its_default_stack(cls):
    instance = cls.from_config(AgentmakerConfig(), embedder=_FakeEmbedder())
    closed = []
    sync_close = instance._sync.close
    retriever_close = instance.retriever.close
    store_close = instance.source_store.close
    instance._sync.close = lambda: closed.append("sync")
    instance.retriever.close = lambda: closed.append("retriever")
    instance.source_store.close = lambda: closed.append("store")

    instance.close()
    instance.close()

    assert closed == ["sync", "retriever", "store"]
    sync_close()
    retriever_close()
    store_close()


def test_ingestion_from_config_does_not_close_shared_rag_resources():
    config = AgentmakerConfig()
    rag = RagRetriever.from_config(config, embedder=_FakeEmbedder())
    pipeline = IngestionPipeline.from_config(
        config, retriever=rag.retriever, source_store=rag.source_store)
    shared_closed = []
    sync_closed = []
    pipeline_sync_close = pipeline._sync.close
    rag_retriever_close = rag.retriever.close
    rag_store_close = rag.source_store.close
    pipeline._sync.close = lambda: sync_closed.append(True)
    rag.retriever.close = lambda: shared_closed.append("retriever")
    rag.source_store.close = lambda: shared_closed.append("store")

    pipeline.close()
    pipeline.close()

    assert sync_closed == [True]
    assert shared_closed == []
    pipeline_sync_close()
    rag._sync.close()
    rag_retriever_close()
    rag_store_close()


@pytest.mark.parametrize(
    "module_name,class_name,store_name,bad_base",
    [
        ("agentmaker.memory.memory", "Memory", "MemoryStore", "rag"),
        ("agentmaker.rag.retriever", "RagRetriever", "SourceStore", "memory"),
        ("agentmaker.rag.ingest", "IngestionPipeline", "SourceStore", "memory"),
    ],
)
def test_from_config_failure_closes_only_resources_created_during_assembly(
        monkeypatch, module_name, class_name, store_name, bad_base):
    import importlib

    module = importlib.import_module(module_name)
    backends = importlib.import_module("agentmaker.retrieval.backends")
    index_sync_module = importlib.import_module("agentmaker.retrieval.index_sync")
    retriever = _AssemblyResource()
    store = _AssemblyResource()
    bookkeeping = _AssemblyResource()
    sync = _AssemblySync(bookkeeping)
    monkeypatch.setattr(backends, "build_sqlite_hybrid", lambda *args, **kwargs: retriever)
    monkeypatch.setattr(module, store_name, lambda *args, **kwargs: store)
    monkeypatch.setattr(index_sync_module, "SqliteBookkeeping", lambda *args, **kwargs: bookkeeping)
    monkeypatch.setattr(index_sync_module, "SyncIndexSync", lambda *args, **kwargs: sync)
    monkeypatch.setattr(module, "SyncIndexSync", lambda *args, **kwargs: sync)

    with pytest.raises(RetrievalError, match="scope.base"):
        getattr(module, class_name).from_config(
            AgentmakerConfig(), embedder=_FakeEmbedder(), scope=Scope(base=bad_base))

    assert (sync.closed, bookkeeping.closed, retriever.closed, store.closed) == (1, 1, 1, 1)

    injected_retriever = _AssemblyResource()
    injected_store = _AssemblyResource()
    injected_sync = _AssemblyResource()
    kwargs = {"retriever": injected_retriever, "index_sync": injected_sync,
              "scope": Scope(base=bad_base)}
    kwargs["store" if class_name == "Memory" else "source_store"] = injected_store
    with pytest.raises(RetrievalError, match="scope.base"):
        getattr(module, class_name).from_config(AgentmakerConfig(), **kwargs)
    assert (injected_retriever.closed, injected_store.closed, injected_sync.closed) == (0, 0, 0)


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("agentmaker.memory.memory", "Memory"),
        ("agentmaker.rag.retriever", "RagRetriever"),
        ("agentmaker.rag.ingest", "IngestionPipeline"),
    ],
)
def test_from_config_closes_unattached_bookkeeping_when_sync_construction_fails(
        monkeypatch, module_name, class_name):
    import importlib

    module = importlib.import_module(module_name)
    backends = importlib.import_module("agentmaker.retrieval.backends")
    index_sync_module = importlib.import_module("agentmaker.retrieval.index_sync")
    retriever = _AssemblyResource()
    bookkeeping = _AssemblyResource()
    monkeypatch.setattr(backends, "build_sqlite_hybrid", lambda *args, **kwargs: retriever)
    monkeypatch.setattr(index_sync_module, "SqliteBookkeeping", lambda *args, **kwargs: bookkeeping)
    monkeypatch.setattr(
        index_sync_module, "SyncIndexSync",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )
    monkeypatch.setattr(
        module, "SyncIndexSync",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        getattr(module, class_name).from_config(
            AgentmakerConfig(), embedder=_FakeEmbedder())

    assert bookkeeping.closed == 1
    assert retriever.closed == 1


# ---- injectable backend: skip embedder, pass a self-built backend (the pgvector-swap path, no framework source changes) ----

def test_from_config_injects_backend_without_embedder():
    retr, store = object(), object()                  # stand in for a "custom backend"; duck typing is enough
    m = Memory.from_config(AgentmakerConfig(), retriever=retr, store=store)
    assert m.retriever is retr and m.store is store    # injected and used as-is, no embedder needed


# ---- missing embedder with no injection -> clear error (consistent across all three entry points) ----

@pytest.mark.parametrize("cls", [Memory, RagRetriever, IngestionPipeline])
def test_from_config_requires_embedder_or_retriever(cls):
    with pytest.raises(ValueError):
        cls.from_config(AgentmakerConfig())


# ---- validation gate / falsy store / HistoryCompactor.from_config ----

def test_from_config_validates_used_slice():
    """from_config validates the sub-config it actually uses before handing it off: an invalid rrf_k is caught at assembly time, not deferred to retrieval."""
    bad = AgentmakerConfig(retrieval=RetrievalConfig(rrf_k=0))      # rrf_k must be >= 1
    with pytest.raises(ValueError):
        RagRetriever.from_config(bad, retriever=object())       # RagRetriever always uses config.retrieval -> always caught
    with pytest.raises(ValueError):
        Memory.from_config(bad, embedder=_FakeEmbedder())       # the default backend-building path also uses config.retrieval -> also caught


def test_from_config_keeps_falsy_store():
    """A custom store whose __bool__ is False isn't clobbered by an `or` fallback (the code checks `is not None`)."""
    class _Falsy:
        def __bool__(self): return False
    s = _Falsy()
    assert Memory.from_config(AgentmakerConfig(), retriever=object(), store=s).store is s


def test_history_compactor_from_config():
    """HistoryCompactor.from_config slices config.compaction (the auto-assembly entry point for CompactionConfig)."""
    from agentmaker import CompactionConfig, HistoryCompactor, LLMClient
    cfg = AgentmakerConfig(compaction=CompactionConfig(keep_recent=7, trigger_tokens=999))
    hc = HistoryCompactor.from_config(LLMClient("deepseek", api_key="x"), cfg)
    assert hc.keep_recent == 7 and hc.trigger_tokens == 999
