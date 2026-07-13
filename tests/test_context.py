"""Context subsystem tests (hermetic: local hand-built data, no key, offline).

Locks in the hardened edges and core paths of this subsystem:
ContextBuilder assembly (custom source-name fallback rendering / duplicate source-name error / tokens within budget / unknown source-name error),
ContextConfig.validate range checks, mmr_select parameter checks with two-layer dedup, _cosine returning 0 on dimension mismatch,
CallableSource's three scope-passing modes, HistoryCompactor construction checks and compaction behavior (using a stub LLM, offline).
"""

import asyncio

import pytest

from agentmaker.context.builder import ContextBuilder
from agentmaker.context.history_compactor import HistoryCompactor
from agentmaker.context.mmr import _cosine, _normalize, mmr_select
from agentmaker.context.reducer import reduce_agent, reduce_plan, reduce_reflection, tokens_of
from agentmaker.context.window_budget import WindowBudget, WindowBudgetConfig
from agentmaker.context.sources import CallableSource
from agentmaker.context.types import ContextConfig, ContextSource
from agentmaker.core.exceptions import ContextWindowExceeded
from agentmaker.core.message import Message
from agentmaker.core.text import count_tokens
from agentmaker.retrieval import RetrievalResult
from agentmaker.runtime.harness import Harness
from agentmaker.core.aio import run_sync


def r(content, score, vec=None):
    """Build one RetrievalResult (id defaults to content for easy identification), matching each file's __main__ self-test."""
    return RetrievalResult(content=content, score=score, source="t", id=content, embedding=vec)


class FakeSource(ContextSource):
    """Fake source: fetch returns preset candidates directly, no retrieval, offline."""

    def __init__(self, name, items):
        self.name = name
        self._items = items

    def fetch(self, query, scope=None):
        return self._items


class _Reply:
    """Fake LLM response: exposes only the .content used by compact."""

    def __init__(self, content):
        self.content = content


class _StubLLM:
    """Fake LLMClient: chat returns a response with fixed content and records the call count; offline, no key needed."""

    def __init__(self, content):
        self._content = content
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return _Reply(self._content)


# ---------- ContextBuilder assembly ----------

def test_builder_renders_custom_source_name():
    """A custom source name (not memory/rag/history/tool), as long as it has a quota in source_ratios, is rendered as a "[name]" fallback rather than silently dropped; known sources come first, custom sources after."""
    cfg = ContextConfig(max_tokens=400, source_ratios={"memory": 0.5, "scratch": 0.5})
    builder = ContextBuilder(cfg, min_chunk_tokens=5)
    # deliberately put the custom source first in the input: render order is decided by the skeleton (known sources first), not input order
    sources = [
        FakeSource("scratch", [r("草稿区的一条笔记", 0.9)]),
        FakeSource("memory", [r("用户对花生过敏", 0.8)]),
    ]
    ctx = builder.build("问题", sources=sources, system_prompt="你是助手。")
    assert "[scratch]" in ctx
    assert "草稿区的一条笔记" in ctx
    assert ctx.index("[Memory]") < ctx.index("[scratch]")   # known source first, custom source falls in after


def test_builder_duplicate_source_name_raises():
    """The same source name passed twice -> ValueError (same-named sources would overwrite each other at assembly, silently dropping candidates)."""
    builder = ContextBuilder(ContextConfig(max_tokens=400), min_chunk_tokens=10)
    with pytest.raises(ValueError):
        builder.build("问题", sources=[
            FakeSource("memory", [r("a", 0.9)]),
            FakeSource("memory", [r("b", 0.8)]),
        ])


def test_builder_unknown_source_name_raises():
    """A source name not in source_ratios -> ValueError (otherwise it gets a 0 quota and silently never enters the context)."""
    builder = ContextBuilder(ContextConfig(max_tokens=400), min_chunk_tokens=10)
    with pytest.raises(ValueError):
        builder.build("问题", sources=[FakeSource("unknown_src", [r("x", 0.5)])])


def test_builder_respects_token_budget():
    """Final text tokens stay within max_tokens: the budget already accounts for the structural overhead of block titles and the "- " list prefix, so low-relevance tail candidates are kept out."""
    builder = ContextBuilder(ContextConfig(max_tokens=200), min_chunk_tokens=10)
    memory = FakeSource("memory", [r(f"记忆编号{i}的一段较长描述内容", 0.90 - i * 0.01) for i in range(30)])
    rag = FakeSource("rag", [r(f"知识编号{i}的一段较长片段内容", 0.85 - i * 0.01) for i in range(30)])
    ctx = builder.build("用户当前的问题文本", sources=[memory, rag], system_prompt="你是企业助手。")
    assert count_tokens(ctx) <= 200            # final text, including structural overhead, still within the total budget
    assert "记忆编号29" not in ctx              # the least-relevant tail candidate is kept out by the budget (the budget really is tightening)


# ---------- ContextConfig.validate ----------

def test_config_validate_accepts_legal_default():
    """A valid config (default source_ratios + a large enough window) passes validation."""
    ContextConfig(max_tokens=100_000).validate()   # passes if it doesn't raise


def test_config_validate_rejects_bad_values():
    """Illegal values each raise ValueError: negative reserve, lambda out of range, negative ratio, ratio sum <= 0, max_tokens <= 0."""
    with pytest.raises(ValueError):
        ContextConfig(max_tokens=1000, output_reserve_ratio=-1).validate()
    with pytest.raises(ValueError):
        ContextConfig(max_tokens=1000, mmr_lambda=2).validate()
    with pytest.raises(ValueError):
        ContextConfig(max_tokens=1000, source_ratios={"memory": -0.5}).validate()
    with pytest.raises(ValueError):
        ContextConfig(max_tokens=1000, source_ratios={"memory": 0.0}).validate()   # sum <= 0
    with pytest.raises(ValueError):
        ContextConfig(max_tokens=0).validate()     # 0 takes the "unset" branch
    with pytest.raises(ValueError):
        ContextConfig(max_tokens=-5).validate()    # negative takes the "must be positive" branch


def test_config_validate_allows_zero_ratio_source():
    """Explicitly setting a source ratio to 0 is allowed (intentionally no budget: not allocated in the first pass, may only borrow idle space): validate skips that source and doesn't error."""
    # some sources 0, sum > 0 -> the 0 source skips quota validation, normal sources validate as usual
    ContextConfig(max_tokens=10_000, source_ratios={"memory": 0.5, "rag": 0.0}).validate(min_chunk_tokens=64)
    # End-to-end construction accepts a deliberately unallocated source.
    ContextBuilder(ContextConfig(max_tokens=10_000, source_ratios={"memory": 0.5, "rag": 0.0}), min_chunk_tokens=64)


def test_builder_constructs_without_max_tokens_for_agent_path():
    """Agent-wiring case: constructs even without max_tokens (the budget rides the window ledger via budget=); only a standalone build/build_block call fails loud."""
    class _Src(ContextSource):
        name = "memory"
        def fetch(self, query, scope=None):
            return [RetrievalResult(content="候选A", score=0.9, source="m", id="1")]
    builder = ContextBuilder(ContextConfig())
    # with a budget= override (window-ledger case) it emits a block normally
    assert "候选A" in builder.build_block("q", sources=[_Src()], budget=500)
    # a standalone call without budget and missing max_tokens -> clear error
    with pytest.raises(ValueError):
        builder.build_block("q", sources=[_Src()])
    with pytest.raises(ValueError):
        builder.build("q", sources=[_Src()])


# ---------- mmr_select ----------

def test_mmr_rejects_bad_params():
    """Negative top_k / lambda out of [0,1] / negative dedup_threshold all raise ValueError."""
    cands = [r("a", 0.9, [1.0, 0.0])]
    with pytest.raises(ValueError):
        mmr_select(cands, top_k=-1)
    with pytest.raises(ValueError):
        mmr_select(cands, lambda_=2)
    with pytest.raises(ValueError):
        mmr_select(cands, lambda_=-0.5)
    with pytest.raises(ValueError):
        mmr_select(cands, dedup_threshold=-0.1)


def test_mmr_exact_dedup_without_embedding():
    """Two items with identical content and embedding=None: exact dedup keeps only one, and keeps the highest-score one (independent of input order)."""
    low_first = r("完全相同的一句话", 0.5, None)    # appears first, low score
    high_later = r("完全相同的一句话", 0.9, None)   # appears later, high score
    out = mmr_select([low_first, high_later], lambda_=0.7)
    assert len(out) == 1
    assert out[0].content == "完全相同的一句话"
    assert out[0].score == 0.9    # keeps the high score, not the first-seen low score (robust to unsorted input)


def test_mmr_near_duplicate_removed():
    """Near-duplicates (cosine >= 0.95) are dropped, keeping the most relevant one plus another topic."""
    a = r("住宿费上限500元", 0.90, [1.0, 0.0, 0.0])
    b = r("住宿报销不超过500", 0.85, [0.99, 0.10, 0.0])   # cosine with a ~= 0.995
    c = r("餐补每天80元", 0.60, [0.0, 1.0, 0.0])
    out = [x.content for x in mmr_select([a, b, c], top_k=3, lambda_=0.7)]
    assert "住宿报销不超过500" not in out
    assert out == ["住宿费上限500元", "餐补每天80元"]


def test_mmr_missing_embedding_not_suppressed():
    """Different content with a missing embedding isn't suppressed as a duplicate (similarity counted as 0 -> no diversity penalty), so it should be selected."""
    a = r("住宿费上限500元", 0.90, [1.0, 0.0, 0.0])
    d = r("另一条不同的住宿信息", 0.80, None)
    out = [x.content for x in mmr_select([a, d], top_k=2, lambda_=0.5)]
    assert "另一条不同的住宿信息" in out


# ---------- _cosine ----------

def test_cosine_dimension_mismatch_returns_zero():
    """Dimension mismatch returns 0.0 (no silent truncation producing a fake score); missing likewise returns 0; identical same-dimension vectors give a positive control of ~=1."""
    assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine(None, [1.0, 0.0]) == 0.0
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ---------- CallableSource: three scope-passing modes ----------

def test_callable_source_autodetect_positional_arity():
    """By default detects arity from positional params: (query, scope) receives scope; (query) doesn't, and doesn't error."""
    got = []
    CallableSource("memory", lambda q, s: got.append(s) or []).fetch("q", scope="S1")
    assert got == ["S1"]

    got_one = []
    CallableSource("memory", lambda q: got_one.append("called") or []).fetch("q", scope="S2")
    assert got_one == ["called"]   # single positional param -> scope ignored, not mis-passed into a TypeError


def test_callable_source_pass_scope_keyword_only():
    """pass_scope=True passes scope by keyword to a fetch with keyword-only scope; auto-detection wouldn't recognize it and wouldn't pass it (a documented gotcha)."""
    got = {}

    def f(query, *, scope=None):
        got["scope"] = scope
        return []

    CallableSource("memory", f, pass_scope=True).fetch("q", scope="S3")
    assert got["scope"] == "S3"

    got_auto = {}

    def g(query, *, scope=None):
        got_auto["scope"] = scope
        return []

    CallableSource("memory", g).fetch("q", scope="S3")   # auto-detection only counts positional params; keyword-only doesn't receive it
    assert got_auto["scope"] is None


def test_callable_source_pass_scope_false_forces_none():
    """pass_scope=False forces scope not to be passed, even if fetch has a second positional param (overrides auto-detection)."""
    got = []

    def f(query, scope="DEFAULT"):
        got.append(scope)
        return []

    CallableSource("memory", f, pass_scope=False).fetch("q", scope="S4")
    assert got == ["DEFAULT"]   # scope not passed, default value retained


# ---------- HistoryCompactor ----------

def test_history_compactor_rejects_bad_init():
    """keep_recent=0 (a negative-zero slicing bug) / keep_recent=-1 / trigger_tokens<0 all raise ValueError."""
    llm = _StubLLM("摘要")
    with pytest.raises(ValueError):
        HistoryCompactor(llm, keep_recent=0)
    with pytest.raises(ValueError):
        HistoryCompactor(llm, keep_recent=-1)
    with pytest.raises(ValueError):
        HistoryCompactor(llm, trigger_tokens=-1)


def test_history_compactor_compacts_old_keeps_recent():
    """Compacts when over threshold: [recap (system)] + the most recent keep_recent originals; the summary uses a stub LLM, offline."""
    llm = _StubLLM("这是前情提要")
    compactor = HistoryCompactor(llm, keep_recent=2, trigger_tokens=10)
    history = [Message(f"第{i}轮的内容很长很长很长很长", "user" if i % 2 == 0 else "assistant")
               for i in range(8)]
    out = compactor.compact(history)
    assert llm.calls == 1                           # summary triggered once
    assert out[0].role == "system"
    assert out[0].content == "[Recap] 这是前情提要"
    assert len(out) == 1 + 2                         # recap + 2 most recent
    assert out[-1].content == history[-1].content    # most recent original preserved verbatim
    assert out[-2].content == history[-2].content


def test_history_compactor_incremental_running_summary():
    """Incremental cache: in a long conversation each turn's old turns are a prefix of the previous turn's old turns -> a prefix hit only summarizes the few newly slid-out items plus a merge instruction (no full re-summarization)."""
    seen = []

    class _RecLLM:
        provider = "stub"
        calls = 0
        async def chat(self, messages, **kw):
            seen.append(messages[0]["content"][:6])   # record whether the summary or the merge instruction was used
            from agentmaker.core.llm_response import LLMResponse
            return LLMResponse(content="提要")

    c = HistoryCompactor(_RecLLM(), keep_recent=2, trigger_tokens=1)
    h = [Message(f"第{i}轮内容", "user" if i % 2 == 0 else "assistant") for i in range(10)]
    c.compact(h[:6])                                   # old turns h[:4]: first full summary (summary instruction)
    c.compact(h[:8])                                   # old turns h[:6]: prefix h[:4] hits -> merge instruction
    assert seen[0] == "Condense the following"[:6]      # first uses context.summary
    assert seen[1] == "Below is an existing"[:6]        # incremental uses context.summary_merge (reuses the old summary)


def test_history_compactor_skips_when_below_trigger():
    """Below threshold: returns unchanged, doesn't call the LLM (saves a summary)."""
    llm = _StubLLM("不该被用到")
    compactor = HistoryCompactor(llm, keep_recent=2, trigger_tokens=100_000)
    history = [Message("短", "user"), Message("短", "assistant")]
    out = compactor.compact(history)
    assert out is history


def test_history_compactor_summary_prompt_overridable():
    """summary_prompt can be overridden via a constructor arg; if omitted it uses the public default DEFAULT_SUMMARY_PROMPT, and it really is passed as the instruction to the summarize callback."""
    from agentmaker.context import DEFAULT_SUMMARY_PROMPT
    llm = _StubLLM("摘要")
    assert HistoryCompactor(llm).summary_prompt is DEFAULT_SUMMARY_PROMPT           # omitted = public default
    c = HistoryCompactor(llm, keep_recent=2, trigger_tokens=10, summary_prompt="SUM!")
    history = [Message(f"第{i}轮内容很长很长很长很长", "user" if i % 2 == 0 else "assistant") for i in range(8)]
    seen = {}

    def rec_summarize(text, instruction):
        seen["instruction"] = instruction
        return "X"

    c.compact(history, summarize=rec_summarize)
    assert seen["instruction"] == "SUM!"
    assert llm.calls == 0


# ---------- reducer: four-paradigm loss-aware trajectory/history reduction (overflow protection) ----------

class _BudgetLLM:
    """Fake LLMClient: carries context_window, chat returns fixed summary text; for reducer / Harness reduction tests."""

    def __init__(self, content="（摘要）", window=None):
        self._content = content
        self.context_window = window
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return _Reply(self._content)


class _Summ:
    """Fake summarize callback (async) (text, instruction) -> str: returns fixed text, records the call count (reduction functions are async-native)."""

    def __init__(self, text="（摘要）"):
        self.text = text
        self.calls = 0

    async def __call__(self, text, instruction):
        self.calls += 1
        return self.text


def _reduce(fn, *args, **kwargs):
    """Synchronously drive an async reduction function (test convenience: reduce_agent/plan/reflection are all async-native)."""
    return asyncio.run(fn(*args, **kwargs))


def test_window_budget_reserve_clamps_and_partition():
    """WindowBudget: output reserve = min(desired, model cap, window guardrail); the three streams sum to <= window; unknown window -> None."""
    cfg = WindowBudgetConfig()
    # large window: desired(4096) < model cap(32K) -> take desired; desired above the model cap -> clamped to the model cap
    assert cfg.output_reserve(window=1_000_000, model_max_output=32_768) == 4096
    assert WindowBudgetConfig(desired_output_tokens=100_000).output_reserve(window=1_000_000, model_max_output=65_536) == 65_536
    # small window: desired is huge but clamped by the "window x max_output_fraction" guardrail (8192x0.5)
    assert WindowBudgetConfig(desired_output_tokens=999_999).output_reserve(window=8_192, model_max_output=None) == 4096
    # in fragment terms: output reserve + fixed + rag block + trajectory = window (exactly partitioned, no overflow)
    wb = WindowBudget.for_run(llm=_BudgetLLM(window=1_000_000), cfg=cfg, tool_tokens=2000)
    assert wb.output_reserve + wb.fixed + wb.rag_budget + wb.trajectory_budget(rag_in_scope=False) <= wb.window
    # agent path (rag_in_scope=True): the trajectory budget deducts the tool schema (not in messages, so the reducer can't count it, otherwise it overflows the window)
    assert wb.trajectory_budget(rag_in_scope=True) == wb.window - wb.output_reserve - wb.tool_tokens
    # rag_ratio=0 override (harness passes this when there's no retrieval source): rag block budget goes to 0, doesn't waste trajectory budget
    assert WindowBudget.for_run(llm=_BudgetLLM(window=1_000_000), cfg=cfg, rag_ratio=0.0).rag_budget == 0
    assert WindowBudget.for_run(llm=_BudgetLLM(window=None), cfg=cfg) is None   # unknown window -> None


def test_reduce_plan_preserves_recent():
    """Plan over budget: keeps the most recent keep_recent step results verbatim, summarizing earlier steps into one."""
    history = [f"步骤{i}：做了X{i}\n结果：值{i}" for i in range(1, 7)]
    out = _reduce(reduce_plan, history, summarize=_Summ("更早摘要：关键数=42"), budget=tokens_of(*history) - 1, keep_recent=3)
    assert len(out) == 1 + 3
    assert out[0].startswith("[Earlier Step Results Summary]") and "关键数=42" in out[0]
    assert out[-3:] == history[-3:]                     # recent results preserved verbatim


def test_reduce_reflection_keeps_latest_answer_and_critique_digest():
    """Reflection over budget: keeps the latest answer verbatim + condenses past critiques into "prior suggestion points", dropping old drafts."""
    # entries are enlarged so the fixed "marker" overhead is negligible vs. budget and the summary has room (under a small budget the markers crowd out space, which is normal)
    entries = [{"kind": "draft", "text": "初稿正文" * 6}, {"kind": "critique", "text": "批评一的详细意见" * 4},
               {"kind": "refine", "text": "改进稿一内容" * 6}, {"kind": "critique", "text": "批评二的详细意见" * 4},
               {"kind": "refine", "text": "最新答案正文内容" * 6}]
    out = _reduce(reduce_reflection, entries, summarize=_Summ("要点：1.补来源 2.改措辞"), budget=tokens_of(*(e["text"] for e in entries)) - 1)
    assert out[-1] == entries[-1]                                  # latest answer preserved verbatim
    assert out[0]["kind"] == "critique" and out[0]["text"].startswith("[Past Critique Points]")
    assert "补来源" in out[0]["text"]


def test_reduce_caps_oversized_summary_within_budget():
    """When the summary LLM output is too long, the reduced total is capped to roughly within budget (shared _summary_block fallback, verified via reduce_agent)."""
    msgs = [{"role": "user", "content": "原始问题"}] + [{"role": "assistant", "content": f"第{i}步的内容描述"} for i in range(8)]
    budget = tokens_of(*(m["content"] for m in msgs)) - 1
    out = _reduce(reduce_agent, msgs, summarize=_Summ("摘" * 5000), budget=budget, keep_recent_steps=2, turn_start=1)   # deliberately over-long summary
    assert tokens_of(*(m["content"] for m in out)) <= budget   # hard guarantee within budget (the +1 token from the CJK ellipsis is already covered)


def test_reduce_fail_loud_when_mandatory_exceeds_budget():
    """When the must-keep portion itself exceeds budget -> ContextWindowExceeded (fail loud, never silently truncate)."""
    summ = _Summ()
    with pytest.raises(ContextWindowExceeded):
        _reduce(reduce_agent, [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "很长的一步" * 3},
                               {"role": "assistant", "content": "再一步" * 3}], summarize=summ, budget=1, turn_start=1)
    with pytest.raises(ContextWindowExceeded):
        _reduce(reduce_reflection, [{"kind": "refine", "text": "很长的最新答案" * 5}], summarize=summ, budget=1)


def test_harness_reduce_budget_none_is_noop():
    """Harness.reduce: unknown window (budget=None) -> returns unchanged, untouched, no LLM call."""
    llm = _BudgetLLM(window=None)
    msgs = [{"role": "user", "content": "q"}] + [{"role": "assistant", "content": f"step{i}"} for i in range(6)]
    assert asyncio.run(Harness(llm).areduce("agent", msgs, turn_start=1)) is msgs and llm.calls == 0


def test_harness_reduce_dispatches_by_kind():
    """Harness.reduce dispatches by kind to the matching reducer; a small enough window triggers reduction."""
    llm = _BudgetLLM("摘要")
    history = [f"步骤{i}：...\n结果：{i}" for i in range(1, 7)]
    llm.context_window = tokens_of(*history) - 1   # slightly under the full size -> triggers reduction, the 3 recent items still fit
    # turn off output reserve and rag block ratio so the plan trajectory budget = the whole window, landing exactly between "full vs recent"
    h = Harness(llm, window_budget=WindowBudgetConfig(desired_output_tokens=0, rag_ratio=0.0))
    out = asyncio.run(h.areduce("plan", history))
    assert len(out) < len(history)


# ---------- TokenCounter pluggable injection point (default count_tokens, swappable for tiktoken etc. in production) ----------

def test_builder_uses_injected_counter():
    """ContextBuilder's budget uses the injected counter: _prefix_tokens is computed with it; a counter that counts each item as more expensive -> fewer candidates kept at the same budget."""
    def src():
        return [FakeSource("memory", [r(f"记忆编号{i}的一段较长描述内容", 0.90 - i * 0.01) for i in range(30)]),
                FakeSource("rag", [r(f"知识编号{i}的一段较长片段内容", 0.85 - i * 0.01) for i in range(30)])]
    triple = ContextBuilder(ContextConfig(max_tokens=200), min_chunk_tokens=10, token_counter=lambda s: count_tokens(s) * 3)
    assert triple._prefix_tokens == count_tokens("- ") * 3        # the "- " prefix overhead is computed with the injected counter
    kept_triple = triple.build("用户当前的问题文本", sources=src())
    kept_default = ContextBuilder(ContextConfig(max_tokens=200), min_chunk_tokens=10).build("用户当前的问题文本", sources=src())
    assert kept_triple.count("编号") < kept_default.count("编号")   # a "more expensive" counter -> fewer candidates kept at the same budget


def test_compactor_uses_injected_counter():
    """HistoryCompactor's token counting uses the injected counter: constant 0 -> never triggers compaction; scaled up -> triggers earlier."""
    llm = _StubLLM("摘要")
    history = [Message(f"第{i}轮", "user" if i % 2 == 0 else "assistant") for i in range(8)]
    c_zero = HistoryCompactor(llm, keep_recent=2, trigger_tokens=10, token_counter=lambda s: 0)
    assert c_zero.compact(history) is history and llm.calls == 0    # total constant 0 <= trigger -> returns unchanged, no LLM call
    c_big = HistoryCompactor(llm, keep_recent=2, trigger_tokens=10, token_counter=lambda s: 100)
    out = c_big.compact(history)
    assert len(out) == 1 + 2 and out[0].role == "system"           # total inflated > trigger -> [recap]+recent


def test_reducer_counter_threaded_through_harness():
    """Harness.token_counter is threaded into areduce's reduction function: a scaled-up counter makes the same history trigger reduction earlier."""
    history = [f"步骤{i}：...\n结果：{i}" for i in range(1, 7)]
    window = tokens_of(*history)                                    # exactly equals the full size under default counting
    cfg = WindowBudgetConfig(desired_output_tokens=0, rag_ratio=0.0)
    h_default = Harness(_BudgetLLM("摘要", window=window), window_budget=cfg)
    assert asyncio.run(h_default.areduce("plan", history)) is history   # default counting: full size == budget -> no reduction
    h_big = Harness(_BudgetLLM("摘要", window=window), window_budget=cfg, token_counter=lambda s: count_tokens(s) * 2)
    out = asyncio.run(h_big.areduce("plan", history))
    assert len(out) < len(history)                                 # scaled-up counting: measured double > budget -> triggers reduction


def test_run_context_layering_and_reexports():
    """Execution owns governance while observability re-exports correlation helpers."""
    from agentmaker.runtime import current_run_id, current_scope, governed_chat   # sidecar governance entry point exposed via runtime
    from agentmaker.runtime.execution import governed_chat as gc                  # execution is the home
    from agentmaker.runtime.observability import correlation, current_run_id as crid   # re-exported in reverse on the trace side
    assert governed_chat is gc and current_run_id is crid and callable(correlation) and current_scope


def test_harness_summary_counts_toward_run_policy():
    """Reduction summaries consume RunPolicy quota and propagate RunLimitExceeded."""
    from agentmaker.core.exceptions import RunLimitExceeded
    from agentmaker.runtime.execution.run_policy import RunPolicy
    from agentmaker.runtime.execution.run_context import record_llm, reset_run, start_run
    history = [f"步骤{i}：...\n结果：{i}" for i in range(1, 7)]
    llm = _BudgetLLM("摘要")
    llm.context_window = tokens_of(*history) - 1                                 # triggers reduction, the 3 recent items fit -> reaches the summary's call_llm
    h = Harness(llm, window_budget=WindowBudgetConfig(desired_output_tokens=0, rag_ratio=0.0))
    tok = start_run("r", policy=RunPolicy(max_llm_calls=1))
    try:
        record_llm(None)                                      # the single LLM quota is already used up
        with pytest.raises(RunLimitExceeded):                 # the summary's call_llm trips the limit and propagates
            asyncio.run(h.areduce("plan", history))
    finally:
        reset_run(tok)


def test_harness_areduce_summary_counts_toward_run_policy():
    """Async path: areduce's summary (acall_llm inside the async-native reduction function) is likewise counted against RunPolicy - completed within the same event loop, so context is naturally shared."""
    import asyncio

    from agentmaker.core.exceptions import RunLimitExceeded
    from agentmaker.runtime.execution.run_policy import RunPolicy
    from agentmaker.runtime.execution.run_context import record_llm, reset_run, start_run
    history = [f"步骤{i}：...\n结果：{i}" for i in range(1, 7)]
    llm = _BudgetLLM("摘要")
    llm.context_window = tokens_of(*history) - 1                                 # same as above: triggers reduction, reaches the summary's call_llm

    async def go():
        tok = start_run("r", policy=RunPolicy(max_llm_calls=1))
        try:
            record_llm(None)
            with pytest.raises(RunLimitExceeded):
                await Harness(llm, window_budget=WindowBudgetConfig(desired_output_tokens=0, rag_ratio=0.0)).areduce("plan", history)
        finally:
            reset_run(tok)

    asyncio.run(go())


# ---------- Harness half-wired assembly fails loud (context_builder and sources must come as a pair) ----------

def test_harness_rejects_half_wired_context_injection():
    """Context builders and sources must be supplied together or both omitted."""
    llm = _BudgetLLM()
    with pytest.raises(ValueError):
        Harness(llm, context_builder=object())            # builder but no sources
    with pytest.raises(ValueError):
        Harness(llm, sources=[object()])                  # sources but no builder
    Harness(llm, context_builder=object(), sources=[object()])   # paired -> constructs normally
    Harness(llm)                                          # neither -> constructs normally


# ---------- Retrieval observability + sidecar LLM governance ----------

def test_governed_chat_counts_into_run_policy():
    """Sidecar LLM calls (governed_chat) count against the RunPolicy limit: inside a run it raises on excess; outside a run it's a no-op."""
    from agentmaker.core.exceptions import RunLimitExceeded
    from agentmaker.runtime.execution.run_policy import RunPolicy
    from agentmaker.runtime.execution.run_context import governed_chat
    from agentmaker.runtime.execution.run_context import reset_run, start_run

    class _Resp:
        usage = {"total_tokens": 1}
        model = "stub"

    class _LLM:
        async def chat(self, messages, **kw): return _Resp()

    run_sync(governed_chat(_LLM(), []))                        # outside a run: unlimited, no raise
    tok = start_run("r", policy=RunPolicy(max_llm_calls=1))
    try:
        run_sync(governed_chat(_LLM(), []))                    # 1st call: allowed and counted
        with pytest.raises(RunLimitExceeded):
            run_sync(governed_chat(_LLM(), []))                # 2nd call: exceeds max_llm_calls=1
    finally:
        reset_run(tok)


def test_governed_chat_emits_finish_reason():
    """The sidecar llm_call event matches Harness._llm_event: carries finish_reason / has_tool_calls (sidecar truncation is observable too)."""
    from agentmaker.runtime.execution.run_context import governed_chat

    class _Resp:
        usage = {"total_tokens": 1}
        model = "stub"
        finish_reason = "max_tokens"
        tool_calls = None

    class _LLM:
        async def chat(self, messages, **kw): return _Resp()

    class _Spy:
        def __init__(self): self.events = []
        def emit(self, e): self.events.append(e)

    spy = _Spy()
    run_sync(governed_chat(_LLM(), [], tracer=spy, origin="memory.summary"))
    evt = [e for e in spy.events if e["type"] == "llm_call"][0]
    assert evt["finish_reason"] == "max_tokens" and evt["has_tool_calls"] is False and evt["origin"] == "memory.summary"


def test_memory_search_emits_trace_event():
    """A Memory.search with a tracer attached emits a memory_search event (query / hit count / latency)."""
    from agentmaker.memory import Memory, MemoryStore
    from agentmaker.retrieval.scope import Scope as _S

    class _Spy:
        def __init__(self): self.events = []
        def emit(self, e): self.events.append(e)

    class _Empty:
        def search(self, q, *, top_k=5, candidate_pool=20, scope=None, **kw): return []
        def add(self, ids, contents, *, scope=None, **kw): pass
        def delete(self, ids, *, scope=None): pass

    spy = _Spy()
    m = Memory(retriever=_Empty(), store=MemoryStore(), scope=_S(base="memory"), tracer=spy)
    m.search("有什么")
    assert spy.events and spy.events[0]["type"] == "memory_search" and spy.events[0]["hits"] == 0


def test_rag_retrieve_emits_trace_event():
    """A RagRetriever.retrieve with a tracer attached emits a rag_retrieve event."""
    from agentmaker.rag import RagRetriever, SourceStore

    class _Spy:
        def __init__(self): self.events = []
        def emit(self, e): self.events.append(e)

    class _Empty:
        def search(self, q, *, top_k=5, candidate_pool=20, scope=None, **kw): return []

    class _NoLLM:
        pass

    spy = _Spy()
    r = RagRetriever(_Empty(), SourceStore(), _NoLLM(), tracer=spy)
    r.retrieve("问点什么")
    assert spy.events and spy.events[0]["type"] == "rag_retrieve"


def test_harness_skips_empty_tools_param():
    """An empty tool list is omitted from the provider call."""
    class _SpyLLM:
        def __init__(self):
            self.calls = []

        async def chat(self, messages, **kw):
            self.calls.append(kw)
            return _Reply("ok")

    llm = _SpyLLM()
    h = Harness(llm)
    asyncio.run(h.acall_llm([{"role": "user", "content": "hi"}], tools=[]))
    assert "tools" not in llm.calls[0]                          # empty list -> no tools parameter


# ---------- Anthropic output cap wired into the budget (option B) + observable truncation ----------


class _CapLLM:
    """Fake LLM: records the kwargs seen on each chat call (to check whether max_tokens is passed down per budget); protocol / window / output cap / finish_reason are configurable."""

    def __init__(self, *, protocol="anthropic", context_window=200_000, max_output_tokens=64_000,
                 model="claude-x", finish_reason=None):
        self.protocol = protocol
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.model = model
        self._finish_reason = finish_reason
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        reply = _Reply("部分回复")
        reply.model = self.model
        reply.finish_reason = self._finish_reason
        return reply


def test_anthropic_max_tokens_from_budget():
    """Anthropic protocol: when the caller doesn't pass max_tokens explicitly, the harness passes down the window ledger's output_reserve,
    making the desired_output_tokens knob take effect for Claude."""
    llm = _CapLLM(protocol="anthropic", context_window=200_000, max_output_tokens=64_000)
    cfg = WindowBudgetConfig(desired_output_tokens=8192, rag_ratio=0.0)
    asyncio.run(Harness(llm, window_budget=cfg).acall_llm([{"role": "user", "content": "写篇长文"}]))
    assert llm.calls[0]["max_tokens"] == 8192                   # min(8192, 200000*0.5, 64000) = 8192


def test_non_anthropic_no_max_tokens_injected():
    """OpenAI, DeepSeek, and Gemini retain the model's server-side output default."""
    llm = _CapLLM(protocol="openai", context_window=128_000, max_output_tokens=16_000)
    cfg = WindowBudgetConfig(desired_output_tokens=8192, rag_ratio=0.0)
    asyncio.run(Harness(llm, window_budget=cfg).acall_llm([{"role": "user", "content": "hi"}]))
    assert "max_tokens" not in llm.calls[0]


def test_explicit_max_tokens_not_overridden_by_budget():
    """If the caller passes max_tokens explicitly it's respected, not overridden by the budget (even for Anthropic)."""
    llm = _CapLLM(protocol="anthropic")
    cfg = WindowBudgetConfig(desired_output_tokens=8192, rag_ratio=0.0)
    asyncio.run(Harness(llm, window_budget=cfg).acall_llm([{"role": "user", "content": "hi"}], max_tokens=333))
    assert llm.calls[0]["max_tokens"] == 333


def test_anthropic_no_injection_when_window_unknown():
    """Unknown window (context_window=None) -> no ledger -> no max_tokens injected (the adapter still uses its own fallback)."""
    llm = _CapLLM(protocol="anthropic", context_window=None)
    asyncio.run(Harness(llm, window_budget=WindowBudgetConfig()).acall_llm([{"role": "user", "content": "hi"}]))
    assert "max_tokens" not in llm.calls[0]


def test_warn_and_trace_on_truncation(caplog):
    """When finish_reason indicates length truncation, log a warning (don't treat it as a complete answer); finish_reason also goes into the llm_call trace."""
    import logging

    class _Spy:
        def __init__(self): self.events = []
        def emit(self, e): self.events.append(e)

    spy = _Spy()
    llm = _CapLLM(protocol="openai", context_window=None, finish_reason="length")  # turn off budget injection, test only truncation
    h = Harness(llm, tracer=spy)
    with caplog.at_level(logging.WARNING, logger="agentmaker.runtime.harness"):
        asyncio.run(h.acall_llm([{"role": "user", "content": "hi"}]))
    assert any("truncated" in r.message for r in caplog.records)                        # truncation warning logged
    evt = [e for e in spy.events if e["type"] == "llm_call"][0]
    assert evt["finish_reason"] == "length"                                        # finish_reason in the trace


def test_no_warn_on_normal_finish(caplog):
    """Normal finish (finish_reason=stop) logs no truncation warning."""
    import logging
    llm = _CapLLM(protocol="openai", context_window=None, finish_reason="stop")
    with caplog.at_level(logging.WARNING, logger="agentmaker.runtime.harness"):
        asyncio.run(Harness(llm).acall_llm([{"role": "user", "content": "hi"}]))
    assert not any("truncated" in r.message for r in caplog.records)


def test_stream_truncation_observable(caplog):
    """Streaming truncation is observable too (same as non-streaming): finish_reason goes into the stream llm_call event + a warning is logged."""
    import logging
    from agentmaker.core.llm_response import StreamStats

    class _StreamTruncLLM:
        model = "m"

        async def stream(self, messages, *, on_stats=None, **kw):
            yield "部分"
            if on_stats:                                         # on finish, report stats carrying finish_reason (the real adapter contract)
                on_stats(StreamStats(model="m", finish_reason="length", usage=None, latency_ms=1))

    class _Spy:
        def __init__(self): self.events = []
        def emit(self, e): self.events.append(e)

    spy = _Spy()
    h = Harness(_StreamTruncLLM(), tracer=spy)

    async def _drive():
        return [p async for p in h.astream_llm([{"role": "user", "content": "hi"}])]
    with caplog.at_level(logging.WARNING, logger="agentmaker.runtime.harness"):
        pieces = asyncio.run(_drive())
    assert pieces == ["部分"]
    evt = [e for e in spy.events if e["type"] == "llm_call"][0]
    assert evt["finish_reason"] == "length" and evt.get("streamed") is True       # finish_reason in the streaming event
    assert any("truncated" in r.message for r in caplog.records)                        # streaming truncation logs a warning too


# ---------- reduce_agent (unified-loop fc trajectory reduction) ----------


def _fc_trail(n, result_pad=5):
    """Build n atomic units: assistant(tool_calls) + the corresponding role:"tool" result."""
    out = []
    for i in range(n):
        out += [{"role": "assistant", "content": f"想{i}",
                 "tool_calls": [{"id": f"c{i}", "type": "function",
                                 "function": {"name": "t", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": f"结果{i}" * result_pad}]
    return out


def _fc_tokens(msgs):
    """Message token estimate matching the implementation (content + tool_calls JSON payload)."""
    import json as _json
    n = 0
    for m in msgs:
        n += tokens_of(m.get("content") or "")
        if m.get("tool_calls"):
            n += tokens_of(_json.dumps(m["tool_calls"], ensure_ascii=False))
    return n


def test_reduce_agent_protects_initial_assembly_and_pairs():
    """turn_start identifies the protected question boundary even when history contains earlier user turns;
    assistant(tool_calls) and tool results stay paired and atomic (no orphan tool_call_id in the kept region)."""
    head = [{"role": "system", "content": "角色"},
            {"role": "user", "content": "旧历史问题"},
            {"role": "assistant", "content": "旧历史回答"},
            {"role": "user", "content": "当前问题XYZ"}]
    msgs = head + _fc_trail(5)
    budget = _fc_tokens(msgs) - 1                            # slightly below the payload-inclusive total -> triggers reduction and leaves room for the summary
    out = _reduce(reduce_agent, msgs, summarize=_Summ("步骤摘要"), budget=budget,
                  keep_recent_steps=2, turn_start=4)
    assert out[:4] == head                                   # protected region unchanged (includes the current question; the old user turn wasn't mis-protected as a boundary)
    assert out[4]["role"] == "system" and "步骤摘要" in out[4]["content"]   # summary block inserted after the protected region
    ids_assist = {c["id"] for m in out if m.get("tool_calls") for c in m["tool_calls"]}
    ids_tool = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
    assert ids_tool == ids_assist == {"c3", "c4"}            # the 2 most recent units kept as pairs, no orphans


def test_reduce_agent_counts_tool_calls_payload():
    """The tool_calls JSON payload counts toward the token estimate - reduction can trigger even when arguments are huge (and content is tiny).
    Note the payload must be incompressible (a repeated single char would be squeezed to single-digit tokens by the BPE tokenizer)."""
    args = "，".join(f"字段{i}=值{i}" for i in range(300))     # large incompressible argument
    big = {"role": "assistant", "content": "",
           "tool_calls": [{"id": "c", "type": "function",
                           "function": {"name": "t", "arguments": args}}]}
    msgs = [{"role": "user", "content": "q"}, big, {"role": "tool", "tool_call_id": "c", "content": "r"}]
    assert _fc_tokens(msgs) > tokens_of("q", "r") + 60       # precondition: the payload really pushes the total over budget
    out = _reduce(reduce_agent, msgs, summarize=_Summ("摘"), budget=tokens_of("q", "r") + 60,
                  keep_recent_steps=0, turn_start=1)
    assert out is not msgs                                   # reduction triggered (payload was counted)
    assert out[0] == msgs[0] and not any(m.get("tool_calls") for m in out)   # the giant call was summarized away


def test_reduce_agent_noop_under_budget():
    """Under budget: returns the same object unchanged, no summarize call."""
    summ = _Summ()
    msgs = [{"role": "user", "content": "q"}] + _fc_trail(2)
    assert _reduce(reduce_agent, msgs, summarize=summ, budget=10**9, turn_start=1) is msgs
    assert summ.calls == 0


# ---------- MMR negative-score normalization / numpy vectors / budget newlines / summary cap ----------

def test_normalize_preserves_order_with_negative_scores():
    """All-negative scores (e.g. some rerankers' logits) are shift-normalized, preserving relative order (dividing by the max would flip / flatten them)."""
    out = _normalize([-0.1, -0.5, -0.9])
    assert out == [1.0, 0.5, 0.0]                                  # highest -0.1->1, lowest -0.9->0, descending order intact
    assert _normalize([0.9, 0.6, 0.3]) == [1.0, 2 / 3, 1 / 3]      # the non-negative normal case is still "divide by the max"
    assert _normalize([5.0, 5.0]) == [1.0, 1.0]                    # all equal: diversity only


def test_mmr_select_orders_by_relevance_when_negative_scores():
    """With negative-score candidates MMR still orders by relevance (lambda=1, pure relevance), not scrambled by normalization flattening."""
    a = r("最相关", -0.1, [1.0, 0.0])
    b = r("次相关", -0.5, [0.0, 1.0])
    c = r("最不相关", -0.9, [0.0, 0.0, 1.0])                        # different dimension -> similarity counted as 0, no mutual suppression
    picked = [x.content for x in mmr_select([c, b, a], top_k=3, lambda_=1.0)]
    assert picked == ["最相关", "次相关", "最不相关"]


def test_cosine_empty_and_numpy_no_truth_ambiguity():
    """_cosine returns 0 for empty vectors; for numpy arrays it doesn't raise the "truth value is ambiguous" error."""
    assert _cosine([], [1.0, 2.0]) == 0.0
    assert _cosine([1.0, 2.0], []) == 0.0
    np = pytest.importorskip("numpy")
    v = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine(v, v) - 1.0) < 1e-9                         # numpy vector doesn't raise, cosine is normal


def test_builder_budget_holds_under_char_counter():
    """With structural overhead counting newlines / inter-block blank lines, assembly with an exact "1 token per char" counter still stays within budget."""
    char_count = len                                              # 1 token per char (including newlines), more precise than the default
    cfg = ContextConfig(max_tokens=600, source_ratios={"memory": 0.5, "rag": 0.5})
    builder = ContextBuilder(cfg, min_chunk_tokens=5, token_counter=char_count)
    memory = FakeSource("memory", [r(f"记忆{i}的一段内容文字abcdef", 0.9 - i * 0.02) for i in range(20)])
    rag = FakeSource("rag", [r(f"知识{i}的一段片段文字ghijkl", 0.85 - i * 0.02) for i in range(20)])
    block = builder.build_block("问题", sources=[memory, rag], budget=300)
    assert char_count(block) <= 300                              # the real char count including newlines stays within the given budget


def test_history_compactor_caps_accumulated_summary():
    """The incrementally merged summary is hard-capped at max_summary_tokens, preventing unbounded growth across turns."""
    hc = HistoryCompactor(llm=None, keep_recent=2, trigger_tokens=1, max_summary_tokens=20)
    history = [Message(content=f"消息{i}", role="user") for i in range(10)]
    out = hc.compact(history, summarize=lambda text, instr: "很" * 5000)   # very long summary
    assert out[0].role == "system"
    prefix = hc.prompts.text("context.summary_prefix")
    summary_part = out[0].content[len(prefix):]
    assert count_tokens(summary_part) <= hc.max_summary_tokens   # the summary body is truncated to within the cap
