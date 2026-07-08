"""from_config classmethod regression (hermetic: fake embedder + sqlite :memory:, no network).

Locks down the assembly convenience layer: from_config on Memory / RagRetriever / IngestionPipeline.
It defaults to the sqlite backend (one-line stack, correct sub-config slicing), backends are injectable
(the assembly root lives in the app, so swapping in pgvector etc. needs no framework source changes),
and a missing embedder with no injected backend raises a clear error.
"""

import pytest

from agentmaker import (ChunkingConfig, IngestionPipeline, AgentmakerConfig, Memory, MemoryConfig,
                    RagConfig, RagRetriever, RetrievalConfig)
from agentmaker.retrieval import HybridRetriever


class _FakeEmbedder:
    """Fake embedder: returns fixed-dimension vectors, no network."""
    dim = 3

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


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
