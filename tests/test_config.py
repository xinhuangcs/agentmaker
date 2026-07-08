"""agentmaker configuration-system regression (hermetic: no key, offline).

Locks in the design invariants from config-design.md: frozen / no module-level singleton / deep-freeze
against leak-through / hashable / serialization round-trip / from_dict policy / three-level resolution
(None does not swallow 0) / defaults don't drift / components accept a config / LLMClient profile extension point.
"""

import json
import os
import re
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from agentmaker import (ChunkingConfig, CompactionConfig, ContextConfig, AgentmakerConfig, LLMClient,
                    MemoryConfig, ProviderProfile, RagConfig, ReducerConfig, RetrievalConfig,
                    ToolRetrievalConfig, WindowBudgetConfig)

_SUBS = [ChunkingConfig, RetrievalConfig, RagConfig, MemoryConfig, ReducerConfig, CompactionConfig, ContextConfig,
         WindowBudgetConfig, ToolRetrievalConfig]


# ---------- 1. frozen ----------

@pytest.mark.parametrize("cls", _SUBS + [AgentmakerConfig])
def test_all_configs_frozen(cls):
    obj = cls()
    field = next(iter(vars(obj)))
    with pytest.raises(FrozenInstanceError):
        setattr(obj, field, 123)


# ---------- 1b. Sub-config list is single-sourced (dataclass fields are the one source of truth, preventing drift across three hand-copied places) ----------

def test_sub_config_single_source():
    """_sub_config_classes must map one-to-one with AgentmakerConfig's dataclass fields, and every sub-config must have validate."""
    from dataclasses import fields as _fields

    from agentmaker.config import _sub_config_classes
    sub = _sub_config_classes(AgentmakerConfig)
    assert set(sub) == {f.name for f in _fields(AgentmakerConfig)}      # field set == derived list
    assert all(hasattr(cls, "validate") for cls in sub.values())    # a sub-config missing validate would blow up validate() on the spot


# ---------- 2. default_factory in effect (two instances don't share sub-configs) ----------

def test_default_factory_distinct_instances():
    assert AgentmakerConfig().retrieval is not AgentmakerConfig().retrieval
    assert AgentmakerConfig().context is not AgentmakerConfig().context
    assert ContextConfig().source_ratios is not ContextConfig().source_ratios


# ---------- 3. Gate: no module-level config instance inside agentmaker/ (catches misuse like "DEFAULTS = AgentmakerConfig()") ----------

def test_no_module_level_config_singleton():
    root = os.path.dirname(os.path.dirname(__file__))
    agentmaker_dir = os.path.join(root, "src", "agentmaker")   # src layout: code lives under src/agentmaker/
    pat = re.compile(r"^\w+\s*=\s*(AgentmakerConfig|RetrievalConfig|ChunkingConfig|RagConfig|"
                     r"MemoryConfig|ReducerConfig|CompactionConfig|ContextConfig|WindowBudgetConfig|"
                     r"ToolRetrievalConfig)\s*\(")
    offenders = []
    for dirpath, _, files in os.walk(agentmaker_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if pat.match(line):
                        offenders.append(f"{os.path.join(dirpath, f)}:{n} {line.strip()}")
    assert not offenders, "agentmaker must not have module-level config instances (they degrade into a global singleton):\n" + "\n".join(offenders)


# ---------- 4. Deep freeze: no leak-through + read-only ----------

def test_source_ratios_deep_frozen_no_leak():
    orig = {"history": 0.9, "rag": 0.1, "memory": 0.0, "tool": 0.0}
    cfg = ContextConfig(source_ratios=orig)
    orig["history"] = 0.0                              # mutate the original dict that was passed in
    assert cfg.source_ratios["history"] == 0.9        # config is unaffected (copied, then frozen)
    assert isinstance(cfg.source_ratios, MappingProxyType)
    with pytest.raises(TypeError):
        cfg.source_ratios["history"] = 0.5            # read-only


def test_source_ratios_value_coercion_and_clear_error():
    cfg = ContextConfig(source_ratios={"history": "0.5", "rag": "0.5", "memory": "0", "tool": "0"})
    assert cfg.source_ratios["history"] == 0.5 and isinstance(cfg.source_ratios["history"], float)
    with pytest.raises(ValueError):
        ContextConfig(source_ratios={"history": "abc"})


# ---------- 5. Hashable (frozen + mapping fields still go into a set / act as dict keys) ----------

def test_configs_hashable():
    assert hash(ContextConfig()) is not None
    assert len({AgentmakerConfig(), AgentmakerConfig()}) == 1   # equal by eq -> dedups to 1
    assert {ContextConfig(): "v"}[ContextConfig()] == "v"


# ---------- 6. Three-level resolution: None doesn't swallow 0 (a per-call explicit 0 isn't silently overridden by the default) ----------

def test_default_resolution_does_not_swallow_zero():
    from agentmaker.memory import Memory, MemoryStore
    from agentmaker.retrieval.scope import Scope

    class _Fake:
        def search(self, q, *, top_k=5, candidate_pool=20, scope=None):
            return []
        def add(self, *a, **k): pass
        def delete(self, *a, **k): pass
        def close(self): pass

    m = Memory(_Fake(), MemoryStore(), scope=Scope(base="memory"), config=MemoryConfig(search_top_k=7))
    with pytest.raises(Exception):     # explicit top_k=0 isn't treated as "not passed" -> hits validation error (RetrievalError)
        m.search("q", top_k=0)
    assert m.search("q") == []         # omitted -> uses cfg.search_top_k=7 (_Fake returns empty; no error proves the default was used)


# ---------- 7. Serialization round-trip (including ContextConfig's MappingProxyType) ----------

def test_to_dict_from_dict_roundtrip():
    kc = AgentmakerConfig()
    d = kc.to_dict()
    assert isinstance(d["context"]["source_ratios"], dict)   # _to_plain converts MappingProxyType back to a dict
    assert json.dumps(d, ensure_ascii=False)                 # JSON-serializable (asdict would crash; the hand-written _to_plain doesn't)
    assert AgentmakerConfig.from_dict(d) == kc


# ---------- 8. from_dict policy: unknown key fails loud, missing key falls back, str->int/float coercion ----------

def test_from_dict_policies():
    with pytest.raises(ValueError):
        AgentmakerConfig.from_dict({"bogus": {}})            # unknown top-level key
    with pytest.raises(ValueError):
        AgentmakerConfig.from_dict({"retrieval": {"nope": 1}})  # unknown sub-level key
    assert AgentmakerConfig.from_dict({}).retrieval.top_k == 5                    # missing key falls back to default
    assert AgentmakerConfig.from_dict({"retrieval": {"top_k": "8"}}).retrieval.top_k == 8       # str->int
    assert AgentmakerConfig.from_dict({"window_budget": {"rag_ratio": "0.25"}}).window_budget.rag_ratio == 0.25  # str->float


# ---------- 9. validate: illegal values raise ----------

@pytest.mark.parametrize("cfg_call", [
    lambda: RetrievalConfig(top_k=0),
    lambda: RetrievalConfig(candidate_pool=2, top_k=5),
    lambda: ChunkingConfig(overlap_tokens=999, chunk_tokens=100),
    lambda: MemoryConfig(relevance_weight=-1),
    lambda: MemoryConfig(summary_top_k=0),
    lambda: WindowBudgetConfig(rag_ratio=1.5),       # rag block ratio out of [0,1]
    lambda: WindowBudgetConfig(max_output_fraction=0),  # guardrail fraction must be in (0,1]
    lambda: CompactionConfig(keep_recent=0),
    lambda: ContextConfig(max_tokens="x"),          # non-int max_tokens raises clearly
])
def test_validate_rejects_bad(cfg_call):
    with pytest.raises(ValueError):
        cfg_call().validate()   # each sub-config range-checks via validate(); AgentmakerConfig.validate calls them one by one


# ---------- 10. Defaults == existing constants (drift guard) ----------

def test_defaults_match_existing_constants():
    from agentmaker.rag import splitter
    from agentmaker.retrieval import hybrid
    assert (ChunkingConfig().chunk_tokens, ChunkingConfig().overlap_tokens) == (splitter._CHUNK_TOKENS, splitter._OVERLAP_TOKENS)
    assert RetrievalConfig().rrf_k == hybrid._RRF_K
    assert RetrievalConfig().top_k == 5 and RetrievalConfig().candidate_pool == 20
    assert MemoryConfig().summary_top_k == 20 and MemoryConfig().rebuild_batch_size == 256
    assert CompactionConfig() == CompactionConfig(keep_recent=4, trigger_tokens=2000)


# ---------- 11. Components accept a config (not hardcoded) + for_window ----------

def test_components_accept_and_store_config():
    from agentmaker.retrieval.hybrid import HybridRetriever
    assert HybridRetriever(None, None, None, config=RetrievalConfig(top_k=9, rrf_k=11)).config.rrf_k == 11
    from agentmaker.rag.ingest import IngestionPipeline
    assert IngestionPipeline(None, None, config=ChunkingConfig(chunk_tokens=321)).cfg.chunk_tokens == 321
    from agentmaker.rag.retriever import RagRetriever
    r = RagRetriever(None, None, None, config=RetrievalConfig(top_k=9), rag_config=RagConfig(mq_max_queries=3))
    assert r.cfg.top_k == 9 and r.rag_cfg.mq_max_queries == 3


def test_agentmakerconfig_for_window_sets_max_tokens_and_keeps_fields():
    kc = AgentmakerConfig(context=ContextConfig(mmr_lambda=0.3)).for_window(8000)
    assert kc.context.max_tokens == 4000 and kc.context.mmr_lambda == 0.3   # max_tokens set, other fields preserved
    kc.validate()                                                            # passes overall validation once the window is set


def test_agentmakerconfig_validate_passes_without_for_window():
    """Out of the box, AgentmakerConfig() (context.max_tokens=None) passes overall validation: when wired to an Agent the RAG block budget rides the window ledger, so max_tokens isn't required."""
    AgentmakerConfig().validate()                                               # passes if it doesn't raise (no longer requires for_window first)


# ---------- 12. LLMClient profile injection extension point (offline) ----------

def test_llmclient_profile_injection_offline():
    p = ProviderProfile(base_url="http://x/v1", key_envs=("K",), default_model="m",
                        context_window=128000, structured_output="json_object")
    c = LLMClient(provider="myvendor", profile=p, model="m", api_key="k")
    assert c.provider == "myvendor" and c.protocol == "openai"
    assert c.model == "m" and c.context_window == 128000 and c.base_url == "http://x/v1"
    with pytest.raises(Exception):     # not in _PROFILES and no profile given -> still fails loud
        LLMClient("definitely-not-a-provider")


# ---------- 14. Non-default model catalog (_KNOWN_MODELS): a swapped model still resolves window/output (offline) ----------

def test_known_models_catalog_resolution():
    """A non-default but known model resolves window/output automatically via _KNOWN_MODELS; the default model still uses the profile; unknown -> None."""
    # non-default model (leaves the default alone, opt-in) -> catalog resolution
    c = LLMClient("openai", model="gpt-5.4-nano", api_key="x")
    assert c.context_window == 400_000 and c.max_output_tokens == 128_000
    # default model unaffected by the catalog (still takes the profile value)
    assert LLMClient("openai", api_key="x").context_window == 1_000_000
    # explicit value takes precedence over the catalog
    assert LLMClient("openai", model="gpt-5.4-nano", api_key="x", context_window=123).context_window == 123
    # unknown model -> None (the window ledger skips it, no crash)
    u = LLMClient("openai", model="totally-made-up-xyz", api_key="x")
    assert u.context_window is None and u.max_output_tokens is None
    # a model whose output cap the vendor hasn't published -> max_output_tokens is None (honestly left blank)
    k = LLMClient("openai_compatible", model="kimi-k2.6", api_key="x", base_url="http://x/v1")
    assert k.context_window == 262_144 and k.max_output_tokens is None


# ---------- 13. ReducerConfig flows end to end through AgentSpec -> build_agent -> strategy -> Harness ----------

@pytest.mark.parametrize("strategy,tools", [("chat", "calc"), ("react", "calc"), ("plan", "calc"), ("reflection", None)])
def test_reducer_threads_through_build_agent(strategy, tools):
    from agentmaker import AgentSpec, CalculatorTool, build_agent
    rc = ReducerConfig(plan_keep_recent=1)
    spec = AgentSpec(name="t", strategy=strategy, model=LLMClient("deepseek", api_key="x"),
                     tools=[CalculatorTool()] if tools else None, reducer=rc)
    assert build_agent(spec).harness.reducer is rc   # the injected ReducerConfig flows all the way to Harness


# ---------- 14. WindowBudgetConfig flows end to end through AgentSpec -> build_agent -> strategy -> Harness (four paradigms) ----------

@pytest.mark.parametrize("strategy,tools", [("chat", "calc"), ("react", "calc"), ("plan", "calc"), ("reflection", None)])
def test_window_budget_threads_through_build_agent(strategy, tools):
    from agentmaker import AgentSpec, CalculatorTool, build_agent
    wb = WindowBudgetConfig(rag_ratio=0.25)
    spec = AgentSpec(name="t", strategy=strategy, model=LLMClient("deepseek", api_key="x"),
                     tools=[CalculatorTool()] if tools else None, window_budget=wb)
    assert build_agent(spec).harness.window_budget is wb   # the injected WindowBudgetConfig flows all the way to Harness


# ---------- 15. from_dict edge cases: string-bool parsing / non-Mapping rejection ----------

def test_from_dict_bool_string_and_non_mapping():
    assert AgentmakerConfig.from_dict({"context": {"allow_borrow": "false"}}).context.allow_borrow is False  # "false" is no longer treated as truthy
    assert AgentmakerConfig.from_dict({"context": {"allow_borrow": "true"}}).context.allow_borrow is True
    with pytest.raises(ValueError):
        AgentmakerConfig.from_dict({"context": {"allow_borrow": "maybe"}})   # garbage value fails loud
    with pytest.raises(ValueError):
        AgentmakerConfig.from_dict("notadict")                              # non-Mapping at top level -> clear error
