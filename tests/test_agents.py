"""Regression net for agentmaker.agents / agentmaker.multi_agent fixes (hermetic: stub LLM / tools, no key, offline).

Locks in these fixes:
① resume finalization records first, clears the checkpoint after (an output-guardrail trip / persist failure doesn't lose the checkpoint);
② AgentTool normalizes a child Agent's return into a ToolResponse (Interrupt -> error, non-string textualized, str(resp) doesn't crash);
③ ChatStrategy's text-protocol _find_calls uses raw_decode (a { } ] inside a string value no longer drops the whole call);
④ the ReAct Action regex accepts hyphen / dot tool names (common with MCP / CLI);
⑤ build_agent: strategy='react' must have tools (errors at construction, not deferred to runtime);
⑥ build_agent: max_turns distinguishes None from a number and errors on <=0 (doesn't treat 0 as unset and silently fall back);
⑦ _checkpoint clears stale pending in passing, leaving none in the checkpoint;
⑧ PlanAgent no longer treats an empty plan as the single step ['[]'], falling back to "the original question as a single step";
plus end-to-end ReAct / Plan HITL resume, verifying fixes ① and ⑦ don't regress on the real resume path.
"""

import asyncio

import pytest

from agentmaker.agents.base import BaseAgent
from agentmaker.agents.agent import Agent as UnifiedAgent
from agentmaker.agents.workflows import PlanAgent, ReflectionAgent
from agentmaker.agents.spec import AgentSpec, _turns, build_agent
from agentmaker.core.exceptions import GuardrailTripwireError, RunLimitExceeded, SessionError
from agentmaker.core.llm_clients import LLMClient
from agentmaker.core.llm_response import LLMResponse
from agentmaker.runtime.execution import CheckpointStore, ExecutionState, RunPolicy
from agentmaker.runtime.harness import Harness
from agentmaker.runtime.execution.run_context import new_run_id, reset_run, start_run
from agentmaker.runtime.hitl import Interrupt, Interrupt as _Interrupt, PendingAction, PendingAction as _PendingAction
from agentmaker.agents.multi_agent import AgentTool
from agentmaker.retrieval import Scope
from agentmaker.tools import CalculatorTool, ToolRegistry
from agentmaker.tools.base import Tool, ToolParameter
from agentmaker.tools.response import ToolResponse


# ---------- Test doubles (offline, no key needed) ----------

class ScriptLLM:
    """Fake LLM that yields preset responses in call order: elements are a content string or a ready-made LLMResponse (with tool_calls)."""

    provider = "stub"

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        r = self._scripted[self.calls]
        self.calls += 1
        return r if isinstance(r, LLMResponse) else LLMResponse(content=r)


class MemCheckpoint(CheckpointStore):
    """In-process dict CheckpointStore (stores one ExecutionState JSON per scope)."""

    def __init__(self):
        self._d = {}

    def save(self, state_json, *, scope=None):
        self._d[scope] = state_json

    def load(self, *, scope=None):
        return self._d.get(scope)

    def clear(self, *, scope=None):
        self._d.pop(scope, None)


class DangerTool(Tool):
    """High-risk stub tool (requires_confirmation=True), used to trigger a HITL suspend."""

    requires_confirmation = True

    def __init__(self):
        super().__init__("danger", "高风险删除（测试用）")

    def get_parameters(self):
        return [ToolParameter("x", "string", "目标")]

    def run(self, parameters):
        return ToolResponse.ok(f"已删除 {parameters.get('x')}")


class StubAgent(BaseAgent):
    """Minimal Agent: _run returns a fixed value (or an Interrupt) and records the scope it received."""

    def __init__(self, name, ret):
        super().__init__(name, llm=None)
        self._ret = ret
        self.seen_scope = "UNSET"

    def _run(self, input_text, *, scope, **kwargs):
        self.seen_scope = scope
        return self._ret


class TripGuard:
    """Guardrail that always trips (duck-typed: only needs check(text) -> a result with passed/message)."""

    def check(self, text):
        return type("R", (), {"passed": False, "message": "拦截"})()


def _dummy_llm():
    """Build an LLMClient that doesn't trigger key validation (only for build_agent to reach parameter validation, never actually called)."""
    return LLMClient("deepseek", api_key="dummy")


# ---------- ⑧ PlanAgent plan parsing (line-by-line fallback salvage when structured parsing fails) ----------

def test_parse_plan_empty_list():
    """An empty list [] parses to empty (including when wrapped in a code fence), no longer degrading to ['[]']."""
    assert PlanAgent._parse_plan("[]") == []
    assert PlanAgent._parse_plan("```python\n[]\n```") == []


def test_parse_plan_normal_and_fallback():
    """A normal list is kept as-is; non-list output still goes through line-by-line fallback."""
    assert PlanAgent._parse_plan('["a", "b"]') == ["a", "b"]
    assert PlanAgent._parse_plan("1. 第一步\n2. 第二步") == ["第一步", "第二步"]


def test_plan_falls_back_to_question_on_structured_empty():
    """Structured planning returns empty steps -> _aplan falls back to running the original question as a single step ([question])."""
    import asyncio
    agent = PlanAgent("p", ScriptLLM(['{"steps": []}']))
    assert asyncio.run(agent._aplan("帮我做一件事")) == ["帮我做一件事"]


# ---------- ⑤⑥ AgentSpec / build_agent validation ----------

def test_spec_react_requires_tools():
    """strategy='react' with no tools (None or empty list) errors at construction (and before LLMClient construction, so it doesn't depend on a key)."""
    with pytest.raises(ValueError, match="react"):
        build_agent(AgentSpec(name="r", strategy="react", tools=None))
    with pytest.raises(ValueError, match="react"):
        build_agent(AgentSpec(name="r", strategy="react", tools=[]))


def test_spec_react_is_unified_agent_preset():
    """react = a preset of the unified-loop Agent: tools required, max_turns defaults to 5, system prompt injects react.persona/react.style (think before acting)."""
    agent = build_agent(AgentSpec(name="r", strategy="react", model=_dummy_llm(), tools=[DangerTool()]))
    assert isinstance(agent, UnifiedAgent)
    assert agent.max_turns == 5
    assert agent.tool_registry is not None
    # preset prompts in place: persona (role) + style (the "think before acting" keywords)
    assert "Think before acting" in agent.system_prompt
    # a user's custom instructions still replace the persona, with style appended as usual
    custom = build_agent(AgentSpec(name="r2", strategy="react", model=_dummy_llm(),
                                   tools=[DangerTool()], instructions="你是运维助手。"))
    assert custom.system_prompt.startswith("你是运维助手。") and "Think before acting" in custom.system_prompt


def test_agent_reflection_construct_positive_turns():
    """Directly constructing a unified Agent / ReflectionAgent validates that turns is a positive integer (covers the entry points that bypass build_agent)."""
    llm = ScriptLLM([])
    with pytest.raises(ValueError):
        UnifiedAgent("c", llm, max_turns=0)
    with pytest.raises(ValueError):
        ReflectionAgent("f", llm, max_turns=0)
    assert UnifiedAgent("c", llm, max_turns=2).max_turns == 2
    assert ReflectionAgent("f", llm, max_turns=2).max_turns == 2


def test_turns_helper():
    """_turns: None -> default; positive -> as-is; <=0 -> error (doesn't treat 0 as unset)."""
    assert _turns(None, 3) == 3
    assert _turns(7, 5) == 7
    for bad in (0, -1):
        with pytest.raises(ValueError):
            _turns(bad, 3)


def test_spec_max_turns_zero_raises():
    """build_agent: max_turns=0 errors rather than silently falling back to the default."""
    with pytest.raises(ValueError):
        build_agent(AgentSpec(name="c", strategy="chat", model=_dummy_llm(), max_turns=0))


def test_spec_max_turns_none_and_value():
    """max_turns=None uses the paradigm default (chat=10); a positive value is passed through. chat now maps to the unified-loop Agent (max_turns attribute)."""
    assert build_agent(AgentSpec(name="c", strategy="chat", model=_dummy_llm())).max_turns == 10
    assert build_agent(AgentSpec(name="c", strategy="chat", model=_dummy_llm(), max_turns=2)).max_turns == 2


def test_build_agent_plan_max_turns_maps_to_executor():
    """build_agent(plan, max_turns=N) maps to the max_turns of PlanAgent's internal executor (no longer silently ignored)."""
    agent = build_agent(AgentSpec(name="p", strategy="plan", model=_dummy_llm(), max_turns=7))
    assert agent._executor.max_turns == 7


# ---------- Naming unification: PlanAgent signature alignment + Harness structural check + AgentSpec.model provider:model ----------

def test_plan_agent_signature_aligned():
    """PlanAgent's 3rd positional arg is system_prompt (aligned with Agent/ReflectionAgent), and tool_registry becomes keyword-only."""
    p = PlanAgent("p", _dummy_llm(), "你是助手")                  # 3rd position -> system_prompt, no longer treated as the registry
    assert p.system_prompt == "你是助手"
    p2 = PlanAgent("p", _dummy_llm(), tool_registry=_reg(CalculatorTool()))
    assert p2.harness.tool_registry is not None


def test_harness_rejects_non_registry_tool_registry():
    """Harness construction-time duck-type check: passing a non-registry object as tool_registry -> TypeError (not deferred to a runtime AttributeError)."""
    with pytest.raises(TypeError) as e:
        UnifiedAgent("a", _dummy_llm(), tool_registry="不是registry")
    assert "to_openai_schema" in str(e.value)
    UnifiedAgent("a", _dummy_llm(), tool_registry=None)          # None allowed
    UnifiedAgent("a", _dummy_llm(), tool_registry=_reg(CalculatorTool()))   # a real registry allowed


def test_resolve_llm_provider_model(monkeypatch):
    """AgentSpec.model accepts 'provider:model': splits provider + model; a bare provider name still works; illegal types fail loud."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")                  # LLMClient construction validates the key; inject a fake key under hermetic
    from agentmaker.agents.spec import _resolve_llm
    a = _resolve_llm("deepseek:deepseek-v4-pro")
    assert a.provider == "deepseek" and a.model == "deepseek-v4-pro"
    b = _resolve_llm("deepseek")                                  # bare provider name -> default model
    assert b.provider == "deepseek"
    c = _resolve_llm("deepseek:")                                 # empty right half -> model falls back to default
    assert c.provider == "deepseek" and c.model == b.model
    assert _resolve_llm(_dummy_llm()).api_key == "dummy"          # an LLMClient instance is returned as-is
    with pytest.raises(TypeError):
        _resolve_llm(123)


def test_build_agent_with_provider_model_string(monkeypatch):
    """build_agent constructed via a 'provider:model' string: agent.llm's provider/model are correct."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    agent = build_agent(AgentSpec(name="c", strategy="chat", model="deepseek:deepseek-v4-pro"))
    assert agent.llm.provider == "deepseek" and agent.llm.model == "deepseek-v4-pro"


def test_unknown_provider_error_hints_provider_model_format():
    """An unknown provider errors with a hint about the 'provider:model' format (guidance for new users passing a model name)."""
    from agentmaker.core.exceptions import LLMConfigError
    with pytest.raises(LLMConfigError) as e:
        LLMClient("gpt-5")
    assert "provider:model" in str(e.value)


# ---------- A nested run's run_policy is ignored -> observable warning (no longer silent) ----------

def test_nested_run_policy_ignored_warns(caplog):
    """With an outer run context already present, an inner start_run carrying run_policy inherits the outer (returns None) and warns."""
    outer = start_run(new_run_id())                         # outer run (no policy)
    try:
        with caplog.at_level("WARNING"):
            inner = start_run(new_run_id(), policy=RunPolicy(max_llm_calls=1))
        assert inner is None                                # inherits the outer, doesn't start another
        assert any("run_policy is ignored" in r.message for r in caplog.records)
    finally:
        reset_run(outer)


def test_top_level_run_policy_does_not_warn(caplog):
    """The outermost run with a policy takes effect normally, no warning."""
    with caplog.at_level("WARNING"):
        tok = start_run(new_run_id(), policy=RunPolicy(max_llm_calls=1))
    reset_run(tok)
    assert not any("run_policy is ignored" in r.message for r in caplog.records)


# ---------- _make_harness auto-injects _harness_hooks / prompts (eliminates the hidden contract) ----------

def test_make_harness_injects_hooks_and_prompts():
    """The three paradigms assemble their own harness via _make_harness: injecting self.prompts (shared reference) + self._harness_hooks (not self.hooks)."""
    class _H:                                                    # observe-only stub hook
        def __call__(self, *a, **k): ...
    h = _H()
    for agent in (UnifiedAgent("a", _dummy_llm(), hooks=[h]),
                  PlanAgent("p", _dummy_llm(), hooks=[h]),
                  ReflectionAgent("r", _dummy_llm(), hooks=[h])):
        assert agent.harness.prompts is agent.prompts            # prompts shared reference injected correctly
        assert h in agent.harness.hooks                          # model/tool-level hook injected into _harness_hooks (non-empty, contains h)


def test_make_harness_inner_executor_keeps_harness_hooks():
    """An as_child inner executor's harness still carries the parent's _harness_hooks (model/tool-level observation stays connected), even when run-level self.hooks=[]."""
    class _H:
        def __call__(self, *a, **k): ...
    h = _H()
    p = PlanAgent("p", _dummy_llm(), hooks=[h])
    assert p._executor._as_child is True and p._executor.hooks == []    # run-level hooks don't fire in the child layer
    assert h in p._executor.harness.hooks                               # but model/tool-level _harness_hooks reach the inner harness


def test_update_prompts_propagates_to_harness_after_make():
    """update_prompts changes one key and the agent's harness sees it immediately (shares the same registry; _make_harness injects it correctly)."""
    a = UnifiedAgent("a", _dummy_llm())
    a.update_prompts({"chat.persona": "X"})
    assert a.harness.prompts.text("chat.persona") == "X"


# ---------- ① resume finalization order / ⑦ clearing pending (unit) ----------

def test_finish_resume_records_before_clear():
    """resume finalization records first (including output guardrails), then clears the checkpoint: on a guardrail trip the checkpoint is kept and can be resumed again."""
    cp = MemCheckpoint()
    scope = Scope(user="u")
    cp.save("{}", scope=scope)
    agent = StubAgent("a", "out")
    agent.checkpoint_store = cp
    agent.output_guardrails = [TripGuard()]
    with pytest.raises(GuardrailTripwireError):
        asyncio.run(agent._finish_resume("out", ExecutionState(messages=[], input_text="q"), scope))
    assert cp.load(scope=scope) is not None     # checkpoint not cleared (before the fix it would clear first and be lost after a trip)


def test_checkpoint_clears_stale_pending():
    """_checkpoint (a per-step save = currently no suspension) clears stale pending in passing, leaving none in the checkpoint."""
    cp = MemCheckpoint()
    scope = Scope(user="u")
    agent = StubAgent("a", "x")
    agent.checkpoint_store = cp
    state = ExecutionState(messages=[], input_text="q", pending=[PendingAction("danger", {"x": "/a"}, "c1")])
    asyncio.run(agent._checkpoint(state, scope))
    assert state.pending == []
    assert '"pending": []' in cp.load(scope=scope)


# ---------- ①⑦ HITL resume end to end ----------

def test_plan_hitl_resume_propagates_decision():
    """Plan: a child executor suspends on a high-risk tool -> resume(True) passes the decision to the executor to continue (guards against fix ⑦'s pending-clear breaking Plan)."""
    reg = ToolRegistry()
    reg.register(DangerTool())
    cp = MemCheckpoint()
    danger_call = LLMResponse(content="", tool_calls=[
        {"id": "call1", "type": "function",
         "function": {"name": "danger", "arguments": '{"x": "/tmp/a"}'}}])
    llm = ScriptLLM([
        '{"steps": ["删除文件", "汇报结果"]}',   # 0 plan (structured PlanSteps output)
        danger_call,                  # 1 executor step1 -> calls danger -> suspends
        "已删除完成",                  # 2 executor step1 finalizes after the tool result
        "已汇报",                      # 3 executor step2
        "全部完成",                    # 4 synthesis
    ])
    agent = PlanAgent("p", llm, tool_registry=reg, checkpoint_store=cp)
    scope = Scope(user="u")

    out = agent.run("处理文件", scope=scope, verbose=False)
    assert out.interrupted and out.interrupt.pending.tool_name == "danger"

    # if fix ⑦ wrongly cleared pending before the decision was read, Plan would get False (reject) -> different result; here we assert the decision was honored
    assert agent.resume(True, scope=scope, verbose=False).final_output == "全部完成"


def test_plan_hitl_partial_decision_reprompts_on_delegation_path():
    """A Plan sub-step requests two high-risk tools in one turn -> one suspend (two pending); a parent partial decision resume({a:True}) -> a executes,
    b re-suspends (without crashing); then resume({b:True}) -> done. Guards partial decisions on the delegation path (_child_decision only passes decided items, never stuffs None)."""
    reg = ToolRegistry()
    reg.register(DangerTool())
    cp = MemCheckpoint()
    two_danger = LLMResponse(content="", tool_calls=[
        {"id": "a", "type": "function", "function": {"name": "danger", "arguments": '{"x": "/f1"}'}},
        {"id": "b", "type": "function", "function": {"name": "danger", "arguments": '{"x": "/f2"}'}}])
    llm = ScriptLLM(['{"steps": ["删两个文件", "汇报"]}', two_danger, "已删两个", "已汇报", "全部完成"])
    agent = PlanAgent("p", llm, tool_registry=reg, checkpoint_store=cp)
    scope = Scope(user="u")

    out = agent.run("处理", scope=scope, verbose=False)
    assert out.interrupted and {p.call_id for p in out.interrupt.pendings} == {"a", "b"}   # one suspend, two awaiting approval
    out2 = agent.resume({"a": True}, scope=scope, verbose=False)   # partial decision: approve only a (the old implementation raised TypeError here)
    assert out2.interrupted and {p.call_id for p in out2.interrupt.pendings} == {"b"}       # b re-suspends awaiting approval
    assert agent.resume({"b": True}, scope=scope, verbose=False).final_output == "全部完成"


# ---------- ②A AgentTool normalization / scope injection ----------

def test_agenttool_textualizes_string():
    """A child Agent returning a plain string -> ToolResponse.ok(text)."""
    resp = AgentTool(StubAgent("w", "结果42")).run({"task": "x"})
    assert isinstance(resp, ToolResponse) and resp.status == "success" and resp.text == "结果42"


def test_agenttool_interrupt_becomes_error():
    """A child Agent suspending returns an Interrupt -> converted to an error result; text is a string, str(resp) no longer TypeErrors."""
    resp = AgentTool(StubAgent("w", Interrupt(PendingAction("danger", {}, "c1"), None))).run({"task": "x"})
    assert resp.status == "error"
    assert isinstance(resp.text, str)
    assert isinstance(str(resp), str)


def test_agenttool_scope_injection():
    """Passing scope at construction -> the delegated child Agent uses it; default None -> the child Agent uses its own default scope (backward-compatible)."""
    s = StubAgent("w", "ok")
    AgentTool(s, scope=Scope(user="alice")).run({"task": "x"})
    assert s.seen_scope == Scope(user="alice")

    s2 = StubAgent("w", "ok")
    AgentTool(s2).run({"task": "x"})
    assert s2.seen_scope is None


def test_agenttool_inherits_parent_run_scope():
    """With no explicit scope, it takes the parent Agent's current run scope (current_scope) - sharing one instance across sessions doesn't cross-contaminate."""
    child = StubAgent("w", "ok")
    tok = start_run("rid", scope=Scope(user="alice"))   # simulate the parent Agent's current run: scope=alice
    try:
        AgentTool(child).run({"task": "x"})             # AgentTool with no explicit scope
    finally:
        reset_run(tok)
    assert child.seen_scope == Scope(user="alice")      # the child Agent received the parent run's scope, not its own default None


def test_agenttool_explicit_scope_overrides_parent_run_scope():
    """An explicit scope at construction takes precedence over the parent run scope (advanced override, pinning the child Agent to a fixed scope)."""
    child = StubAgent("w", "ok")
    tok = start_run("rid", scope=Scope(user="alice"))
    try:
        AgentTool(child, scope=Scope(user="bob")).run({"task": "x"})
    finally:
        reset_run(tok)
    assert child.seen_scope == Scope(user="bob")


# ---------- Harness streaming counts the call even on early break (record_llm in finally) ----------

class _StreamLLM:
    """Minimal streaming LLM double: stream yields segment by segment, offline."""
    model = "m"

    async def stream(self, messages, **kwargs):
        for i in range(10):
            yield str(i)


def test_harness_stream_counts_call_even_when_consumer_breaks_early():
    """When the consumer closes the stream early, record_llm still counts in finally - otherwise the RunPolicy limit could be bypassed repeatedly.
    The harness is fully async: aio.iter_sync synchronously drives astream_llm (the real facade path for sync consumption)."""
    from agentmaker.core.aio import iter_sync
    h = Harness(_StreamLLM())
    policy = RunPolicy(max_llm_calls=1)
    tok = start_run("rid", policy=policy)
    try:
        gen = iter_sync(h.astream_llm([{"role": "user", "content": "hi"}]))
        next(gen)            # take only one segment
        gen.close()          # early break -> GeneratorExit -> record_llm counts in finally
        # the second streaming call should be blocked for reaching max_llm_calls=1 (proving the previous one was counted)
        with pytest.raises(RunLimitExceeded):
            next(iter_sync(h.astream_llm([{"role": "user", "content": "hi"}])))
    finally:
        reset_run(tok)


# ---------- Unified context engineering across four paradigms: memory/RAG injection + Tool-RAG + Reflection first-class tools + HITL ----------

class StubBuilder:
    """Fake ContextBuilder: build_block / abuild_block return an identifiable block carrying the query (no retrieval, offline)."""

    def build_block(self, query, *, sources, scope=None, budget=None):
        return f"[MEM:{query}]"

    async def abuild_block(self, query, *, sources, scope=None, budget=None):
        return f"[MEM:{query}]"


class StubRetriever:
    """Fake ToolRetriever: returns a single tool's description / schema, proving a Tool-RAG subset was used rather than the full set."""

    def description_for(self, query, **kw):
        return "calc: 仅计算器"

    def schema_for(self, query, **kw):
        return [{"type": "function", "function": {"name": "calc", "parameters": {}}}]


def _reg(*tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def test_memory_injection_react_preset():
    """react preset (= unified-loop Agent): the initial messages inject the memory/RAG block as system (via harness.aassemble)."""
    import asyncio
    a = build_agent(AgentSpec(name="r", strategy="react", model=_dummy_llm(), tools=[CalculatorTool()],
                              context_builder=StubBuilder(), sources=[object()]))
    msgs = asyncio.run(a._initial_messages("天气", a.scope))
    assert any(m["role"] == "system" and "[MEM:天气]" in m["content"] for m in msgs)
    assert msgs[-1] == {"role": "user", "content": "天气"}


def test_memory_injection_reflection():
    """Reflection: the draft / refine prompt injects the memory/RAG block."""
    a = ReflectionAgent("r", ScriptLLM([]), context_builder=StubBuilder(), sources=[object()])
    assert "[MEM:天气]" in a._initial_prompt("天气", a.harness.context_block("天气", a.scope))


def test_memory_injection_plan_and_executor_passthrough():
    """Plan: the planning prompt injects the memory/RAG block; the executor also gets the full set (per-step Tool-RAG + injection)."""
    a = PlanAgent("p", ScriptLLM([]), tool_registry=_reg(CalculatorTool()),
                     context_builder=StubBuilder(), sources=[object()], tool_retriever=StubRetriever())
    assert "[MEM:天气]" in a._with_context(a._planner_prompt("天气"), a.harness.context_block("天气", a.scope))
    assert a._executor.harness.context_builder is not None and a._executor.harness.tool_retriever is not None


def test_reflection_pure_self_critique_unchanged():
    """Reflection without tools: pure self-critique (draft->critique->refine->critique(pass)), behavior unchanged."""
    out = ReflectionAgent("r", ScriptLLM(["初稿", "批评一", "改进稿", "GOOD ENOUGH"]), max_turns=2).run("t", verbose=False)
    assert out.final_output == "改进稿"


def test_reflection_iteration_bound_survives_trajectory_reduction():
    """After the Reflection trajectory is collapsed by overflow reduction, iteration still stops at max_turns by the independent counter `rounds` - no infinite loop from a reduction reset."""
    answers = [f"内容{i}" for i in range(12)]             # enough preset replies (an infinite loop would exhaust them and IndexError)
    llm = ScriptLLM(answers)
    out = ReflectionAgent("r", llm, max_turns=3).run("t", verbose=False)
    # stops at 3 rounds (rounds is an independent meta counter, unrelated to the reduction-collapsed trajectory) -> no infinite loop, doesn't exhaust the 12 replies.
    # 7 = draft + 3 critique + 3 refine (with a tool-less critic, each critique = 1 LLM call).
    assert isinstance(out.final_output, str) and llm.calls == 7


def test_reflection_each_nonpassing_critique_is_refined():
    """One round = critique + refine: max_turns=2 and never passing -> exactly 2 critiques + 2 refines, returning the last refined draft (the final round's critique isn't discarded)."""
    llm = ScriptLLM([f"内容{i}" for i in range(8)])           # never returns the "already optimal" pass signal
    out = ReflectionAgent("r", llm, max_turns=2).run("t", verbose=False)
    # draft + crit1 + refine1 + crit2 + refine2 = 5 calls; the last refined draft is the 5th (index 4)
    assert llm.calls == 5 and out.final_output == "内容4"


def test_reflection_tool_critique_hitl_suspend_resume():
    """Reflection with tools: a critique hits a high-risk tool -> suspends with an Interrupt; resume(True) -> runs the verification -> done."""
    tc = {"id": "c1", "function": {"name": "danger", "arguments": '{"x": "算式"}'}}
    # one round = critique + refine: after resuming and finishing the critique it still refines, so a 4th reply is needed
    llm = ScriptLLM(["初稿答案", LLMResponse(content="", tool_calls=[tc]), "批评：已核验，无误", "改进：最终版"])
    agent = ReflectionAgent("r", llm, max_turns=1, tool_registry=_reg(DangerTool()), checkpoint_store=MemCheckpoint())
    out = agent.run("t", verbose=False)
    assert out.interrupted and out.interrupt.pending.tool_name == "danger"
    assert agent.resume(True, verbose=False).final_output == "改进：最终版"   # resume: run verification -> critique completes -> refine -> return the refined draft


def test_nested_executor_checkpoint_survives_parent_commit_crash():
    """If the parent's commit crashes at the instant of resume -> the child checkpoint is still there (the critic's defer doesn't self-clear) -> recoverable, not stuck in the unrecoverable "parent awaiting but child cleared" state."""
    class CrashCP(CheckpointStore):
        def __init__(self):
            self._d = {}
            self.crash_scope = None
            self.armed = False

        def save(self, state_json, *, scope=None):
            if self.armed and scope == self.crash_scope:
                raise RuntimeError("simulated crash at the parent-commit moment")
            self._d[scope] = state_json

        def load(self, *, scope=None):
            return self._d.get(scope)

        def clear(self, *, scope=None):
            self._d.pop(scope, None)

    tc = {"id": "c1", "function": {"name": "danger", "arguments": "{}"}}
    llm = ScriptLLM(["初稿答案", LLMResponse(content="", tool_calls=[tc]),
                     "批评：已核验", "批评：恢复后重判", "改进：最终版", "兜底"])
    cp = CrashCP()
    agent = ReflectionAgent("r", llm, max_turns=1, tool_registry=_reg(DangerTool()),
                               checkpoint_store=cp, scope=Scope(user="u"))
    crit_scope = agent._derive_scope(agent.scope, "reflect_crit")
    assert agent.run("t", verbose=False).interrupted
    assert cp.load(scope=crit_scope) is not None              # after suspend the child checkpoint exists
    cp.crash_scope, cp.armed = agent.scope, True              # make the parent's commit (save to the parent scope) crash
    with pytest.raises(RuntimeError):
        agent.resume(True, verbose=False)
    assert cp.load(scope=crit_scope) is not None              # ★ after the crash the child checkpoint is still there (old code would have the critic self-clear it to None -> unrecoverable)
    cp.armed = False                                          # disarm the crash
    assert isinstance(agent.resume(None, verbose=False).final_output, str)  # crash-recoverable, doesn't raise SessionError


def test_build_agent_accepts_injection_and_tools_on_all_paradigms():
    """build_agent's new boundaries: reflection accepts tools/context; react/plan accept the full set; compactor still fails loud;
    the use_function_calling field is removed (fc is the unified loop's only mechanism) - setting it TypeErrors at AgentSpec construction."""
    refl = build_agent(AgentSpec(name="r", strategy="reflection", model=_dummy_llm(),
                                 tools=[CalculatorTool()], context_builder=StubBuilder(), sources=[object()]))
    assert isinstance(refl, ReflectionAgent) and refl._critic.tool_registry is not None
    react = build_agent(AgentSpec(name="r", strategy="react", model=_dummy_llm(),
                                  tools=[CalculatorTool()], tool_retriever=StubRetriever()))
    assert react.harness.tool_retriever is not None
    # compactor only fails loud for plan/reflection; chat/react (the unified-loop Agent) is legal (has conversation history to compact)
    assert build_agent(AgentSpec(name="rc", strategy="react", model=_dummy_llm(),
                                 tools=[CalculatorTool()], compactor=object())).harness.compactor is not None
    for s in ("reflection", "plan"):
        with pytest.raises(ValueError):
            build_agent(AgentSpec(name="x", strategy=s, model=_dummy_llm(), tools=[CalculatorTool()], compactor=object()))
    with pytest.raises(TypeError):
        AgentSpec(name="x", strategy="chat", use_function_calling=True)


# ---------- Tool-RAG fallback / tool search as a tool / mid-run expansion of the available set ----------

def _tool_rag_registry():
    """A registry of three fake tools (shared by the Tool-RAG tests)."""
    from agentmaker.tools.base import ToolParameter
    from agentmaker.tools.registry import ToolRegistry
    reg = ToolRegistry()
    for name, desc in [("calc", "算数"), ("mail", "发邮件"), ("ask_user", "向用户提问澄清")]:
        reg.register_function(lambda p: "ok", name, desc, [ToolParameter("x", "string", "参数")])
    return reg


class _HitsRetriever:
    """Stub backend: search returns preset hits (with ids); can be set to zero hits."""

    def __init__(self, names=()):
        self.names = list(names)

    def search(self, query, *, top_k=5, scope=None, **kw):
        class _H:
            def __init__(self, i): self.id = i
        return [_H(n) for n in self.names[:top_k]]


def test_tool_retriever_always_include_and_on_empty():
    """always_include stays pinned first; zero hits defaults to falling back to the full catalog (never zero tools); on_empty can switch the strategy."""
    from agentmaker.tools.tool_retriever import ToolRetriever
    reg = _tool_rag_registry()
    tr = ToolRetriever(reg, _HitsRetriever(["calc"]), always_include=("ask_user",))
    assert tr.retrieve("算一下") == ["ask_user", "calc"]            # pinned first + hits following
    empty = ToolRetriever(reg, _HitsRetriever([]))                  # zero hits
    assert set(empty.retrieve("???")) == {"calc", "mail", "ask_user"}   # defaults to full fallback
    only = ToolRetriever(reg, _HitsRetriever([]), always_include=("ask_user",), on_empty="always_include")
    assert only.retrieve("???") == ["ask_user"]                     # falls back to the pinned list
    none = ToolRetriever(reg, _HitsRetriever([]), on_empty="none")
    assert none.retrieve("???") == []                               # explicitly chooses the empty set
    with pytest.raises(ValueError):
        ToolRetriever(reg, _HitsRetriever([]), on_empty="bogus")    # enum validation


def test_tool_retriever_selector_seam():
    """selector truncation seam: once injected, the callback decides which hits to take (e.g. a score threshold); the default remains a fixed top-k."""
    from agentmaker.tools.tool_retriever import ToolRetriever
    reg = _tool_rag_registry()
    tr = ToolRetriever(reg, _HitsRetriever(["calc", "mail"]), selector=lambda q, hits: [hits[0].id])
    assert tr.retrieve("发邮件并算账") == ["calc"]                   # the callback only lets the top hit through


def test_tool_retrieval_config_in_agentmaker_config():
    """ToolRetrievalConfig as AgentmakerConfig's 9th sub-config: from_dict restore + from_config assembly."""
    from agentmaker import AgentmakerConfig, ToolRetrievalConfig
    from agentmaker.tools.tool_retriever import ToolRetriever
    kc = AgentmakerConfig.from_dict({"tool_retrieval": {"top_k": 3, "always_include": ["ask_user"], "on_empty": "none"}})
    assert kc.tool_retrieval == ToolRetrievalConfig(top_k=3, always_include=("ask_user",), on_empty="none")
    tr = ToolRetriever.from_config(kc, _tool_rag_registry(), _HitsRetriever(["calc"]))
    assert tr.top_k == 3 and tr.always_include == ("ask_user",) and tr.on_empty == "none"


def test_tool_search_tool_returns_discovered():
    """ToolSearchTool: returns catalog text + data.discovered tool names (excluding itself)."""
    from agentmaker.tools.tool_retriever import ToolRetriever, ToolSearchTool
    reg = _tool_rag_registry()
    tr = ToolRetriever(reg, _HitsRetriever(["mail", "calc"]))
    tool = ToolSearchTool(tr, top_k=2)
    out = tool.run({"query": "发邮件"})
    assert out.data["discovered"] == ["mail", "calc"] and "mail" in out.text
    assert tool.run({"query": ""}).status == "error"


def test_fc_loop_expands_tools_from_discovery():
    """Mid-run expansion of the available set on the fc path: after tool_search discovers a new tool, the next LLM call's tools include its schema."""
    from agentmaker.tools.base import ToolParameter
    from agentmaker.tools.registry import ToolRegistry
    from agentmaker.tools.response import ToolResponse
    from agentmaker.tools.base import Tool

    class _Searcher(Tool):
        def __init__(self):
            super().__init__(name="tool_search", description="搜工具")
        def get_parameters(self): return [ToolParameter("query", "string", "q")]
        def run(self, parameters): return ToolResponse.ok("找到 mail", data={"discovered": ["mail"]})

    reg = ToolRegistry()
    reg.register(_Searcher())
    reg.register_function(lambda p: "已发送", "mail", "发邮件", [ToolParameter("to", "string", "收件人")])

    seen_tools = []                                  # record the tool names sent to the LLM each turn

    class _ScriptedLLM:
        """Turn 1 calls tool_search, turn 2 answers directly; records the tools seen each turn."""
        provider = "stub"
        def __init__(self): self.turn = 0
        async def chat(self, messages, tools=None, **kw):
            seen_tools.append({t["function"]["name"] for t in (tools or [])})
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(content="", model="stub", tool_calls=[
                    {"id": "c1", "type": "function", "function": {"name": "tool_search", "arguments": '{"query": "发邮件"}'}}])
            return LLMResponse(content="完成", model="stub")

    agent = UnifiedAgent("扩集", _ScriptedLLM(), tool_registry=reg)
    # initially expose only tool_search (simulating a Tool-RAG preselected subset): narrow harness.atools_for (the real async impl) via monkeypatch
    async def _subset(q):
        return reg.to_openai_schema(names=["tool_search"])
    agent.harness.atools_for = _subset
    assert agent.run("帮我发个邮件").final_output == "完成"
    assert seen_tools[0] == {"tool_search"}                       # turn 1 sees only the search tool
    assert seen_tools[1] == {"tool_search", "mail"}               # after discovery, turn 2 has mail in the available set


def test_discovered_tools_survive_hitl_resume():
    """The discovered list is persisted with the checkpoint: turn1 discovers -> turn2 suspends on a high-risk tool -> after resume the model still sees the discovered tools' schemas."""
    from agentmaker.tools.base import Tool, ToolParameter
    from agentmaker.tools.registry import ToolRegistry
    from agentmaker.tools.response import ToolResponse

    class _Searcher(Tool):
        def __init__(self):
            super().__init__(name="tool_search", description="搜工具")
        def get_parameters(self): return [ToolParameter("query", "string", "q")]
        def run(self, parameters): return ToolResponse.ok("找到 mail", data={"discovered": ["mail"]})

    class _Mail(Tool):
        def __init__(self):
            super().__init__(name="mail", description="发邮件")
            self.requires_confirmation = True                     # high-risk -> HITL suspend
        def get_parameters(self): return [ToolParameter("to", "string", "收件人")]
        def run(self, parameters): return ToolResponse.ok("已发送")

    reg = ToolRegistry()
    reg.register(_Searcher())
    reg.register(_Mail())
    seen_tools = []

    class _ScriptedLLM:
        """turn1 calls tool_search; turn2 calls high-risk mail (suspends); after resume turn 3 answers."""
        provider = "stub"
        def __init__(self): self.turn = 0
        async def chat(self, messages, tools=None, **kw):
            seen_tools.append({t["function"]["name"] for t in (tools or [])})
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(content="", model="stub", tool_calls=[
                    {"id": "c1", "type": "function", "function": {"name": "tool_search", "arguments": '{"query": "发邮件"}'}}])
            if self.turn == 2:
                return LLMResponse(content="", model="stub", tool_calls=[
                    {"id": "c2", "type": "function", "function": {"name": "mail", "arguments": '{"to": "a@x.com"}'}}])
            return LLMResponse(content="完成", model="stub")

    agent = UnifiedAgent("续跑扩集", _ScriptedLLM(), tool_registry=reg,
                         checkpoint_store=MemCheckpoint(), max_turns=5)
    async def _subset(q):
        return reg.to_openai_schema(names=["tool_search"])        # initial preselection gives only the search tool
    agent.harness.atools_for = _subset
    interrupt = agent.run("发邮件给 a@x.com")
    assert interrupt.interrupted                                  # turn2 suspends on the high-risk tool
    assert agent.resume(True).final_output == "完成"              # approve and continue
    assert seen_tools[-1] == {"tool_search", "mail"}              # after resume, turn 3 still sees the discovered mail


class _NoFcLLM:
    """An LLM stub that declares no native fc support (chat should never be reached - it should be blocked at construction)."""
    provider = "stub"
    supports_function_calling = False
    async def chat(self, *a, **k):
        raise AssertionError("a model without function calling must not reach the tool-call path")


def test_agent_with_tools_rejects_no_fc_model():
    """Tools present + a model that declares no fc support -> fail loud at construction (rather than tools silently failing at runtime)."""
    from agentmaker.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register_function(lambda p: "ok", "noop", "无操作工具")
    with pytest.raises(ValueError, match="function calling"):
        UnifiedAgent("x", _NoFcLLM(), tool_registry=reg)


def test_agent_no_tools_allows_no_fc_model():
    """Pure Q&A (no tools) + a model without fc support -> allowed (pure Q&A doesn't need fc, so don't over-block)."""
    UnifiedAgent("x", _NoFcLLM())   # passes if it doesn't raise


# ---------- BaseAgent base (__init_subclass__ check + three child-delegation methods) ----------


class _BareLLM:
    """Minimal LLM placeholder (the base unit tests don't call the model)."""
    provider = "stub"
    model = "stub"


def test_init_subclass_requires_run_or_arun():
    """A subclass must implement at least one of _run (sync) or _arun (async) - fail loud at definition, not halfway through a run."""
    with pytest.raises(TypeError, match="_run"):
        class _Bad(BaseAgent):
            pass

    class _SyncOK(BaseAgent):                       # implements only sync _run: legal (the default _arun offloads to a thread pool)
        def _run(self, input_text, *, scope, **kw):
            return "ok-sync"

    class _AsyncOK(BaseAgent):                      # implements only async _arun: legal
        async def _arun(self, input_text, *, scope, **kw):
            return "ok-async"

    assert _SyncOK("s", _BareLLM()).run("hi").final_output == "ok-sync"
    assert _AsyncOK("a", _BareLLM()).run("hi").final_output == "ok-async"


def test_derive_scope_suffix_and_none_fallback():
    """_derive_scope: appends "::"+suffix to the agent dimension (byte-for-byte identical to the old _exec_scope/_crit_scope); None falls back to an empty Scope."""
    from agentmaker.retrieval import Scope
    assert BaseAgent._derive_scope(Scope(agent="A"), "plan_exec").agent == "A::plan_exec"
    assert BaseAgent._derive_scope(None, "reflect_crit").agent == "::reflect_crit"


def test_child_decision_none_when_missing():
    """When the decision table has no decision for a pending call (parent resume(None) crash recovery) -> pass None so the child re-suspends;
    with decisions -> collect a multi-decision dict keyed by call_id. Fixes the old semantic flaw where "no decision" was read as False=reject, silently swallowing the awaiting action."""
    st = ExecutionState(messages=[], input_text="t")
    assert BaseAgent._child_decision(st) is None                 # no pending
    st.pending = [_PendingAction("tool", {}, "c1")]
    assert BaseAgent._child_decision(st) is None                 # has pending, no decision -> None (not False!)
    st.decisions["c1"] = False
    assert BaseAgent._child_decision(st) == {"c1": False}        # has a decision -> {call_id: bool} multi-decision dict
    st.decisions["c1"] = True
    assert BaseAgent._child_decision(st) == {"c1": True}
    # multiple awaiting actions: decided individually
    st.pending = [_PendingAction("t1", {}, "c1"), _PendingAction("t2", {}, "c2")]
    st.decisions = {"c1": True, "c2": False}
    assert BaseAgent._child_decision(st) == {"c1": True, "c2": False}
    # partial decision (only c1 decided): contains only decided items, never stuffs c2:None - otherwise the child's aresume "values must all be bool" check would break
    st.decisions = {"c1": True}
    result = BaseAgent._child_decision(st)
    assert result == {"c1": True} and all(isinstance(v, bool) for v in result.values())


def test_absorb_child_order_contract():
    """_absorb_child completion-branch ordering contract: awaiting is reset first -> on_complete records -> parent _checkpoint -> child cleanup
    ("parent commit before child cleanup"); suspend branch: awaiting=True + repackaged as a parent-scope Interrupt (doesn't leak the child scope)."""
    from agentmaker.retrieval import Scope

    events = []

    class _Parent(BaseAgent):
        def _run(self, input_text, *, scope, **kw):
            return "x"

    class _SpyStore(CheckpointStore):
        def save(self, state_json, *, scope=None):
            events.append(("parent_save", scope))
        def load(self, *, scope=None):
            return None
        def clear(self, *, scope=None):
            events.append(("parent_clear", scope))

    class _Child:
        async def clear_checkpoint(self, scope):       # _absorb_child now awaits child.clear_checkpoint
            events.append(("child_clear", scope))

    parent_scope = Scope(agent="P")
    child_scope = BaseAgent._derive_scope(parent_scope, "sub")
    p = _Parent("p", _BareLLM(), checkpoint_store=_SpyStore())
    st = ExecutionState(messages=[], input_text="t")
    st.meta["awaiting"] = True                                   # simulate "child already finished after resume"

    def on_complete(r):
        events.append(("record", r, st.meta["awaiting"]))        # awaiting must already be reset to False when recording

    from agentmaker.agents.result import RunResult
    # _absorb_child now consumes the child's RunResult (on_complete gets final_output)
    assert asyncio.run(p._absorb_child(RunResult(final_output="结果", status="completed"), st, parent_scope,
                                       child=_Child(), child_scope=child_scope, on_complete=on_complete)) is None
    assert events == [("record", "结果", False), ("parent_save", parent_scope), ("child_clear", child_scope)]

    events.clear()                                               # suspend branch
    pend = _PendingAction("danger", {}, "c9")
    out = asyncio.run(p._absorb_child(RunResult(final_output=None, status="interrupted",
                                                interrupt=_Interrupt(pend, child_scope)), st, parent_scope,
                                      child=_Child(), child_scope=child_scope, on_complete=lambda r: None))
    # _absorb_child still returns a bare parent-scope Interrupt (an internal resume signal, not wrapped in a RunResult)
    assert isinstance(out, _Interrupt) and out.scope == parent_scope and out.pending is pend
    assert st.meta["awaiting"] is True


# ---------- Unified-loop Agent (agentmaker/agents/agent.py) ----------


def test_unified_agent_plain_chat_and_history():
    """No tools = pure Q&A (the loop ends on the first turn); after completing, the turn atomically enters history."""
    a = UnifiedAgent("u", ScriptLLM(["你好！"]))
    assert a.run("hi").final_output == "你好！"
    assert [m.role for m in a.get_history()] == ["user", "assistant"]


def test_run_result_envelope_fields():
    """RunResult envelope: the completed state has final_output/status/new_messages/run_id/usage all present, and __str__ equals the final output text;
    the suspended state has interrupted=True + an interrupt field, and __str__ isn't a bare None."""
    from agentmaker import RunResult
    a = UnifiedAgent("u", ScriptLLM(["你好！"]))
    r = a.run("hi")
    assert isinstance(r, RunResult) and r.status == "completed" and r.interrupted is False
    assert r.final_output == "你好！" and str(r) == "你好！"           # __str__ soft migration: used as a string it still yields the answer
    assert [m.role for m in r.new_messages] == ["user", "assistant"]   # this turn's new messages
    assert r.run_id and r.usage.llm_calls == 1                         # run_id + usage snapshot

    b = UnifiedAgent("u", ScriptLLM([_danger_call(), "完成"]),
                     tool_registry=_reg(DangerTool()), checkpoint_store=MemCheckpoint())
    s = b.run("删")
    assert s.interrupted and s.status == "interrupted" and s.final_output is None
    assert s.interrupt.pending.tool_name == "danger" and "danger" in str(s)   # suspended __str__ is readable, not None


def test_unified_agent_fc_loop_and_empty_nudge():
    """fc loop: after tool results are fed back as role:"tool" the model answers; an empty reply is nudged once, then a second empty falls back to a final canned message (never returns an empty string)."""
    calc = LLMResponse(content="", model="s", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}}])
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    a = UnifiedAgent("u", ScriptLLM([calc, "答案是 2"]), tool_registry=reg)
    assert a.run("算1+1").final_output == "答案是 2"

    b = UnifiedAgent("u2", ScriptLLM(["", ""]))             # two empties -> invalid_reply fallback
    assert b.run("hi").final_output == b.prompts.text("agent.invalid_reply")


def test_unified_agent_exhausted_text():
    """Turns exhausted (the model keeps calling tools without answering) -> the agent.exhausted message."""
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    calc = LLMResponse(content="", model="s", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}}])
    a = UnifiedAgent("u", ScriptLLM([calc]), tool_registry=reg, max_turns=1)
    assert a.run("x").final_output == a.prompts.text("agent.exhausted")


def test_unified_agent_hitl_suspend_resume_and_callid_rewrite():
    """A high-risk tool suspends -> resume(True) continues and executes; the next turn the server reuses the same call_id -> it collides with the decision table and is rewritten,
    re-suspending for approval (rather than riding the old approval and executing directly); after resume(False) rejects and feeds back, the model reroutes and answers."""
    reg = ToolRegistry()
    reg.register(DangerTool())

    def call(cid):
        return LLMResponse(content="", model="s", tool_calls=[
            {"id": cid, "type": "function", "function": {"name": "danger", "arguments": '{"x": "f"}'}}])

    a = UnifiedAgent("u", ScriptLLM([call("call_0"), call("call_0"), "完成"]),
                     tool_registry=reg, checkpoint_store=MemCheckpoint())
    out = a.run("删两次")
    assert out.interrupted and out.interrupt.pending.call_id == "call_0"
    out2 = a.resume(True)                                    # approve the first -> execute; the second turn's duplicate id is rewritten -> suspends again
    assert out2.interrupted and out2.interrupt.pending.call_id != "call_0"
    assert out2.interrupt.pending.call_id.startswith("call_0#")        # rewrite rule: original id + turn suffix
    assert a.resume(False).final_output == "完成"                          # reject -> feed back and reroute -> answer


# ---------- Re-running on a scope with a pending suspend doesn't silently overwrite ----------

def _danger_call(cid="c1"):
    return LLMResponse(content="", model="s", tool_calls=[
        {"id": cid, "type": "function", "function": {"name": "danger", "arguments": '{"x": "f"}'}}])


def test_run_on_pending_scope_raises_by_default():
    """Starting a new run on a scope that has a pending suspend checkpoint -> SessionError by default (prevents silently overwriting the awaiting action and the approval request vanishing)."""
    reg = _reg(DangerTool())
    a = UnifiedAgent("u", ScriptLLM([_danger_call(), "完成"]), tool_registry=reg, checkpoint_store=MemCheckpoint())
    assert a.run("删").interrupted                            # suspended, awaiting approval
    with pytest.raises(SessionError):
        a.run("换个话题")                                      # a new run on the same scope is blocked, doesn't overwrite the pending checkpoint
    assert a.resume(False).final_output == "完成"             # the original pending is still resumable (the checkpoint wasn't lost)


def test_run_on_pending_scope_discard_policy():
    """on_pending='discard': discards the old pending and continues the new run (chat UX: the user ignores the approval and just changes topic)."""
    reg = _reg(DangerTool())
    a = UnifiedAgent("u", ScriptLLM([_danger_call(), "新话题答案"]), tool_registry=reg,
                     checkpoint_store=MemCheckpoint(), on_pending="discard")
    assert a.run("删").interrupted
    assert a.run("换个话题").final_output == "新话题答案"        # discard: old pending dropped, new run completes normally


# ---------- resume decision type check + approval-gate defense in depth ----------

def test_resume_rejects_non_bool_decision():
    """A non-bool resume decision (e.g. mistakenly passing scope as a positional arg) -> TypeError, not silently injected into the decision table as an approval."""
    a = UnifiedAgent("u", ScriptLLM(["x"]), checkpoint_store=MemCheckpoint())
    with pytest.raises(TypeError):
        a.resume(Scope(user="alice"))                         # mistakenly passing scope positionally as the decision
    with pytest.raises(TypeError):
        a.resume("yes")


def test_approval_gate_requires_explicit_true():
    """Approval-gate defense in depth: a truthy-but-not-True value in the decision table (dirty data that bypassed the type check) does not release the high-risk action; it re-suspends."""
    from agentmaker.runtime.hitl import ApprovalRequired
    h = Harness(ScriptLLM([]), tool_registry=_reg(DangerTool()))
    with pytest.raises(ApprovalRequired):
        h._approval_gate("danger", {"x": "f"}, "c1", {"c1": "approved"})   # truthy but not True -> re-suspend
    assert h._approval_gate("danger", {"x": "f"}, "c1", {"c1": True}) is None      # only an explicit True releases it
    assert h._approval_gate("danger", {"x": "f"}, "c1", {"c1": False}).status == "error"  # False rejects


def test_parallel_independent_runs_each_suspend_and_resume():
    """A developer runs multiple independent sessions in parallel (each with its own scope) -> each HITL-suspends and resumes without interfering.
    pending is "one per ExecutionState (per run)" and isolated by scope - not a global singleton, so independent parallelism is naturally supported."""
    import asyncio as _aio
    reg = ToolRegistry()
    reg.register(DangerTool())

    class _StatelessLLM:
        """Stateless: if messages have no tool result -> call danger (suspend); once present -> answer. Doesn't rely on incrementing state, so it's safe under concurrency."""
        provider = "stub"
        model = "s"
        async def chat(self, messages, tools=None, **kw):
            if any(m.get("role") == "tool" for m in messages):
                return LLMResponse(content="完成", model="s")
            return LLMResponse(content="", model="s", tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "danger", "arguments": '{"x": "f"}'}}])

    a = UnifiedAgent("u", _StatelessLLM(), tool_registry=reg, checkpoint_store=MemCheckpoint())
    s1, s2 = Scope(user="u1"), Scope(user="u2")

    async def drive():
        i1, i2 = await _aio.gather(a.arun("删 a", scope=s1), a.arun("删 b", scope=s2))
        assert i1.interrupted and i2.interrupted
        assert i1.interrupt.scope == s1 and i2.interrupt.scope == s2   # each suspend carries its own scope, no crossover
        return await _aio.gather(a.aresume(True, scope=s1), a.aresume(True, scope=s2))

    r1, r2 = _aio.run(drive())
    assert r1.final_output == "完成" and r2.final_output == "完成"   # each session approves and resumes to completion independently


def test_unified_agent_stream_history_semantics():
    """Streaming: natural exhaustion -> stores one turn of history; early close -> doesn't store (the output-side responsibility lives inside the generator, a break doesn't trigger it)."""
    class _SLLM:
        provider = "stub"
        model = "s"
        async def stream(self, messages, **kw):
            yield "你"
            yield "好"

    a = UnifiedAgent("u", _SLLM())
    assert "".join(a.stream_run("hi")) == "你好"
    assert len(a.get_history()) == 2
    g = a.stream_run("again")
    next(g)
    g.close()                                                # early break
    assert len(a.get_history()) == 2                         # didn't store a half turn


def test_unified_agent_stream_run_context_carries_scope():
    """The streaming run context carries scope (fixes the old ChatStrategy.stream_run drift of not passing scope)."""
    from agentmaker.runtime.execution.run_context import current_scope
    seen = {}

    class _SLLM:
        provider = "stub"
        model = "s"
        async def stream(self, messages, **kw):
            seen["scope"] = current_scope()
            yield "x"

    sc = Scope(agent="A1")
    a = UnifiedAgent("u", _SLLM())
    assert "".join(a.stream_run("hi", scope=sc)) == "x"
    assert seen["scope"] == sc


def test_unified_agent_verbose_not_leaked_to_llm():
    """verbose is a framework parameter and must not leak into LLM kwargs (**kwargs reaches the SDK directly along the whole chain; a leak means an API 400)."""
    class _KwLLM:
        provider = "stub"
        model = "s"
        def __init__(self):
            self.kws = []
        async def chat(self, messages, tools=None, **kw):
            self.kws.append(kw)
            return LLMResponse(content="好")

    llm = _KwLLM()
    UnifiedAgent("u", llm).run("hi", verbose=True)
    assert llm.kws and all("verbose" not in kw for kw in llm.kws)


def test_unified_agent_output_schema():
    """output_schema takes the structured path (pure Q&A, no tools), returning the validated instance."""
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    assert UnifiedAgent("u", ScriptLLM(['{"x": 7}'])).run("给我 x", output_schema=Out).final_output.x == 7


# ---------- Checkpoint format version: "discard on mismatch" ----------

def test_incompatible_checkpoint_version_discarded():
    """resume hits an incompatible old checkpoint (a genuinely old checkpoint lacking the "v" field) -> auto-clears + SessionError so the user restarts."""
    import json as _json
    cp = MemCheckpoint()
    scope = Scope(user="u")
    # simulate an old checkpoint persisted "before the framework upgrade": hand-build JSON missing the "v" field (replaces the retired _LEGACY_META_KEYS blacklist path)
    old_raw = _json.dumps({"messages": [{"role": "user", "content": "x"}], "input_text": "x",
                           "remaining": 1, "decisions": {}, "meta": {}, "pending": None})
    cp.save(old_raw, scope=scope)
    agent = UnifiedAgent("u", ScriptLLM([]), tool_registry=_reg(DangerTool()), checkpoint_store=cp)
    with pytest.raises(SessionError, match="incompatible"):
        agent.resume(True, scope=scope)
    assert cp.load(scope=scope) is None                         # the invalid suspended state was auto-cleared


def test_new_run_clears_incompatible_pending_checkpoint():
    """The suspended-state gate meets an incompatible old checkpoint -> doesn't block a new run (clears the unresumable old checkpoint, then allows it)."""
    import json as _json
    cp = MemCheckpoint()
    scope = Scope(user="u")
    old_raw = _json.dumps({"messages": [], "input_text": "x", "remaining": 1, "decisions": {}, "meta": {},
                           "pending": {"tool_name": "danger", "arguments": {"x": "a"}, "call_id": "c1"}})
    cp.save(old_raw, scope=scope)
    agent = UnifiedAgent("u", ScriptLLM(["答"]), checkpoint_store=cp)   # on_pending defaults to error
    assert agent.run("新任务", scope=scope).final_output == "答"        # old checkpoint cleared, new run allowed, no raise


def test_unified_agent_checkpoint_json_roundtrip_resume():
    """New checkpoint JSON literal replay -> resume(True) continues: the serialization format and the resume path are consistent (pending_calls/turn_start unchanged)."""
    cp = MemCheckpoint()
    scope = Scope(user="u")
    danger = LLMResponse(content="", model="s", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "danger", "arguments": '{"x": "f"}'}}])
    agent = UnifiedAgent("u", ScriptLLM([danger, "完成"]), tool_registry=_reg(DangerTool()), checkpoint_store=cp)
    out = agent.run("删", scope=scope)
    assert out.interrupted
    raw = cp.load(scope=scope)                                  # read back the persisted JSON literal
    restored = ExecutionState.from_json(raw)                    # restores faithfully (including the pending list / meta.pending_calls)
    assert restored.pending[0].call_id == "c1" and "pending_calls" in restored.meta
    assert agent.resume(True, scope=scope).final_output == "完成"   # resumes from that checkpoint to completion


# ---------- Reflection pass-signal word-boundary matching (not substring in) ----------

def test_reflection_passed_uses_word_boundary():
    """The pass signal uses word-boundary matching: appearing as part of a longer word (e.g. GOOD ENOUGH within GOOD ENOUGHNESS) is no longer misjudged as passing."""
    from agentmaker.agents.workflows.reflection import ReflectionAgent
    from agentmaker.prompts import DEFAULT_PROMPTS
    from agentmaker.prompts.packs import chinese_registry

    class _FakeSelf:
        pass

    def passed(prompts, text):
        fs = _FakeSelf()
        fs.prompts = prompts
        return ReflectionAgent._passed(fs, [{"kind": "critique", "text": text}])

    # default (English) pass_signal = "GOOD ENOUGH"
    assert passed(DEFAULT_PROMPTS, "GOOD ENOUGH") is True                  # pure signal -> passes
    assert passed(DEFAULT_PROMPTS, "The answer is GOOD ENOUGH.") is True   # standalone phrase -> passes
    assert passed(DEFAULT_PROMPTS, "GOOD ENOUGHNESS is different") is False  # part of a longer word -> substring would misjudge, word boundary doesn't
    assert passed(DEFAULT_PROMPTS, "not enough detail") is False           # no signal -> doesn't pass

    zh = chinese_registry()                                    # Chinese pack pass_signal = "已达最佳"
    assert passed(zh, "已达最佳") is True          # Chinese pure signal -> passes
    assert passed(zh, "还没达到要求") is False      # no signal -> doesn't pass
    assert passed(zh, "已达最佳，无需再改") is True  # signal at the start -> passes


# ---------- Child-agent checkpoint cascade cleanup ----------

def test_child_agents_default_on_pending_discard():
    """Plan / Reflection's internal child agents default to on_pending='discard' (on a leftover checkpoint they discard and retry, not deadlock with SessionError)."""
    assert PlanAgent("p", _dummy_llm())._executor._on_pending == "discard"
    assert ReflectionAgent("r", _dummy_llm())._critic._on_pending == "discard"


def test_plan_clear_checkpoint_cascades_to_executor():
    """A Plan step suspends -> clear_checkpoint(scope) cascades to clear the child executor's checkpoint -> a subsequent run doesn't hit SessionError and can re-run."""
    from agentmaker.core.aio import run_sync
    reg = ToolRegistry()
    reg.register(DangerTool())
    cp = MemCheckpoint()
    danger = LLMResponse(content="", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "danger", "arguments": '{"x": "/tmp/a"}'}}])
    llm = ScriptLLM([
        '{"steps": ["删文件"]}',        # run1 plan
        danger,                          # run1 executor step1 -> danger -> suspend
        '{"steps": ["改走安全路径"]}',   # run2 plan
        "安全完成",                      # run2 executor step1 (no dangerous tool)
        "全部完成",                      # run2 synthesis
    ])
    agent = PlanAgent("p", llm, tool_registry=reg, checkpoint_store=cp)
    scope = Scope(user="u")
    exec_scope = agent._derive_scope(scope, "plan_exec")

    assert agent.run("处理文件", scope=scope).interrupted
    assert cp.load(scope=exec_scope) is not None                  # child executor checkpoint exists (suspended)

    run_sync(agent.clear_checkpoint(scope))                       # user clears the parent per the error hint - should cascade to clear the child too
    assert cp.load(scope=scope) is None
    assert cp.load(scope=exec_scope) is None                      # ★ cascade-cleared the child checkpoint (otherwise an orphan deadlocks)

    assert agent.run("换个安全办法", scope=scope).final_output == "全部完成"  # a fresh run no longer hits SessionError


def test_agenttool_clears_child_checkpoint_on_interrupt():
    """AgentTool clears the child checkpoint after a delegation suspend -> a second delegation on the same scope still runs normally (doesn't hit the child's _guard_pending)."""
    reg = ToolRegistry()
    reg.register(DangerTool())
    cp = MemCheckpoint()
    danger = LLMResponse(content="", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "danger", "arguments": "{}"}}])
    child = UnifiedAgent("worker", ScriptLLM([danger, "安全完成"]), tool_registry=reg,
                         checkpoint_store=cp, scope=Scope(user="w"))
    tool = AgentTool(child, scope=Scope(user="w"))

    r1 = tool.run({"task": "危险任务"})
    assert r1.status == "error"                                   # suspend converted to error
    assert cp.load(scope=Scope(user="w")) is None                # ★ child checkpoint cleared (otherwise leftover deadlocks)

    r2 = tool.run({"task": "安全任务"})
    assert r2.status == "success" and r2.text == "安全完成"        # the second delegation runs normally, doesn't hit SessionError


# ---------- resume idempotency (guards against re-execution / double accounting) ----------

def test_resume_midbatch_checkpoint_prevents_reexecution():
    """A batch of two high-risk calls approved at once (batch approval): after executing the first, per-call save immediately; after executing the second, the save crashes at that instant,
    and re-resume only re-runs the second, never the first (per-call checkpoints shrink the double-execution window to a single tool)."""
    counter = {}   # x -> execution count (tracked per argument)

    class CountingDanger(Tool):
        requires_confirmation = True

        def __init__(self):
            super().__init__("danger", "高风险（计数）")

        def get_parameters(self):
            return [ToolParameter("x", "string", "目标")]

        def run(self, parameters):
            x = parameters.get("x")
            counter[x] = counter.get(x, 0) + 1
            return ToolResponse.ok(f"done {x}")

    class CrashAfterSecond(CheckpointStore):
        def __init__(self):
            self._d = {}
            self.armed = False

        def save(self, state_json, *, scope=None):
            if self.armed and counter.get("two") == 1:   # b finished, crash while saving b's per-call checkpoint (a's checkpoint already persisted)
                self.armed = False                        # crash only once
                raise RuntimeError("simulated crash while persisting after hr2 ran")
            self._d[scope] = state_json

        def load(self, *, scope=None):
            return self._d.get(scope)

        def clear(self, *, scope=None):
            self._d.pop(scope, None)

    reg = ToolRegistry()
    reg.register(CountingDanger())
    two = LLMResponse(content="", tool_calls=[
        {"id": "a", "type": "function", "function": {"name": "danger", "arguments": '{"x": "one"}'}},
        {"id": "b", "type": "function", "function": {"name": "danger", "arguments": '{"x": "two"}'}}])
    cp = CrashAfterSecond()
    agent = UnifiedAgent("a", ScriptLLM([two, "完成"]), tool_registry=reg, checkpoint_store=cp)
    scope = Scope(user="u")

    out = agent.run("go", scope=scope)
    assert out.interrupted and len(out.interrupt.pendings) == 2  # one suspend, two awaiting actions (batch approval)
    cp.armed = True
    with pytest.raises(RuntimeError):
        agent.resume(True, scope=scope)                        # approve all: a executes+saves -> b executes -> b's save crashes
    assert counter == {"one": 1, "two": 1}                     # at crash time each executed once
    st = ExecutionState.from_json(cp.load(scope=scope))
    assert st.pending == []                                    # the checkpoint is a's post-completion per-call state (not the old two-action suspended state)
    assert st.meta.get("pending_calls") == [two.tool_calls[1]]  # only the unpersisted hr2 remains
    assert agent.resume(True, scope=scope).final_output == "完成"  # re-resume: only re-runs b
    assert counter == {"one": 1, "two": 2}                     # a never re-runs (=1); b re-runs once because the crash was before its save (at-least-once)


def _two_danger():
    """An LLMResponse with two high-risk danger calls (a="one" / b="two")."""
    return LLMResponse(content="", model="s", tool_calls=[
        {"id": "a", "type": "function", "function": {"name": "danger", "arguments": '{"x": "one"}'}},
        {"id": "b", "type": "function", "function": {"name": "danger", "arguments": '{"x": "two"}'}}])


class _CountDanger(Tool):
    """High-risk tool that records the arguments actually executed (verifies batch-approval routing)."""
    requires_confirmation = True

    def __init__(self, log):
        super().__init__("danger", "hr")
        self._log = log

    def get_parameters(self):
        return [ToolParameter("x", "string", "")]

    def run(self, parameters):
        self._log.append(parameters.get("x"))
        return ToolResponse.ok("ok")


def test_batch_approval_multiple_high_risk_one_turn():
    """One turn requests multiple high-risk tools -> one suspend, pendings contains all of them; resume approves / rejects individually by call_id."""
    executed = []
    agent = UnifiedAgent("a", ScriptLLM([_two_danger(), "完成"]),
                         tool_registry=_reg(_CountDanger(executed)), checkpoint_store=MemCheckpoint())
    scope = Scope(user="u")
    out = agent.run("删两个", scope=scope)
    assert out.interrupted and {p.call_id for p in out.interrupt.pendings} == {"a", "b"}   # surfaces all at once
    assert agent.resume({"a": True, "b": False}, scope=scope).final_output == "完成"        # a approved / b rejected
    assert executed == ["one"]                                  # only a executed, b was rejected and didn't run


def test_resume_bool_applies_to_all_pending():
    """resume(bool) applies a single decision to all awaiting actions."""
    executed = []
    agent = UnifiedAgent("a", ScriptLLM([_two_danger(), "完成"]),
                         tool_registry=_reg(_CountDanger(executed)), checkpoint_store=MemCheckpoint())
    scope = Scope(user="u")
    assert agent.run("删两个", scope=scope).interrupted
    assert agent.resume(True, scope=scope).final_output == "完成"
    assert set(executed) == {"one", "two"}                      # both approved and executed


def test_resume_dict_partial_reprompts_remaining():
    """resume decides only some -> the decided ones execute, the undecided re-suspend for approval."""
    executed = []
    agent = UnifiedAgent("a", ScriptLLM([_two_danger(), "完成"]),
                         tool_registry=_reg(_CountDanger(executed)), checkpoint_store=MemCheckpoint())
    scope = Scope(user="u")
    assert len(agent.run("删两个", scope=scope).interrupt.pendings) == 2
    out2 = agent.resume({"a": True}, scope=scope)               # approve only a
    assert out2.interrupted and {p.call_id for p in out2.interrupt.pendings} == {"b"}   # b re-suspends
    assert executed == ["one"]
    assert agent.resume({"b": True}, scope=scope).final_output == "完成"
    assert executed == ["one", "two"]


def test_interrupt_backcompat_and_pending_list_roundtrip():
    """Interrupt.pending returns the first (backward-compatible) + a single construction is normalized to a list; ExecutionState pending list to_json/from_json round-trip."""
    p1, p2 = PendingAction("t1", {"x": 1}, "c1"), PendingAction("t2", {}, "c2")
    it = Interrupt([p1, p2], None)
    assert it.pending is p1 and it.pendings == [p1, p2]
    it2 = Interrupt(p1)                                          # a single PendingAction also works
    assert it2.pendings == [p1] and it2.pending is p1
    restored = ExecutionState.from_json(ExecutionState(messages=[], input_text="q", pending=[p1, p2]).to_json())
    assert [p.call_id for p in restored.pending] == ["c1", "c2"]


def test_resume_dict_value_must_be_bool():
    """A resume dict's decision values must all be bool (fail loud at construction)."""
    agent = UnifiedAgent("a", ScriptLLM([]), tool_registry=_reg(DangerTool()), checkpoint_store=MemCheckpoint())
    with pytest.raises(TypeError):
        agent.resume({"c1": "yes"}, scope=Scope(user="u"))


def test_resume_no_duplicate_history_on_crash_between_record_and_clear():
    """A crash between "store history" and "clear checkpoint" after resume completes -> the leftover checkpoint is marked completed; a subsequent resume(None) doesn't re-run or append duplicate history."""
    class CrashOnClear(CheckpointStore):
        def __init__(self):
            self._d = {}
            self.armed = False

        def save(self, state_json, *, scope=None):
            self._d[scope] = state_json

        def load(self, *, scope=None):
            return self._d.get(scope)

        def clear(self, *, scope=None):
            if self.armed:
                raise RuntimeError("simulated crash at the checkpoint-clear moment")
            self._d.pop(scope, None)

    reg = ToolRegistry()
    reg.register(DangerTool())
    danger = LLMResponse(content="", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "danger", "arguments": '{"x": "/tmp/a"}'}}])
    cp = CrashOnClear()
    agent = UnifiedAgent("a", ScriptLLM([danger, "已完成"]), tool_registry=reg, checkpoint_store=cp)
    scope = Scope(user="u")

    assert agent.run("处理", scope=scope).interrupted
    cp.armed = True
    with pytest.raises(RuntimeError):
        agent.resume(True, scope=scope)                        # complete -> mark completed -> store history -> clear-checkpoint crash
    assert len(agent.get_history(scope)) == 2                  # one user+assistant pair already stored
    st = ExecutionState.from_json(cp.load(scope=scope))
    assert st.completed is True                                # the leftover checkpoint is marked completed

    cp.armed = False
    with pytest.raises(SessionError):
        agent.resume(None, scope=scope)                        # a completed checkpoint -> treated as done, cleared and fails loud, no re-run
    assert len(agent.get_history(scope)) == 2                  # ★ no second pair (no double accounting)


# ---------- Concurrent same-scope lock ----------

def test_concurrent_same_scope_run_serialized():
    """Two concurrent runs on the same scope both suspend -> the per-scope lock serializes them: one suspends, the other hits _guard_pending and raises SessionError, and the suspended state isn't overwritten."""
    import asyncio as _aio
    from agentmaker.core.aio import run_sync

    class AlwaysDanger:
        provider = "stub"
        model = "stub"
        context_window = None
        supports_function_calling = True

        async def chat(self, messages, tools=None, **kwargs):
            return LLMResponse(content="", tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "danger", "arguments": "{}"}}])

    reg = ToolRegistry()
    reg.register(DangerTool())
    cp = MemCheckpoint()
    agent = UnifiedAgent("a", AlwaysDanger(), tool_registry=reg, checkpoint_store=cp)
    scope = Scope(user="u")

    async def _both():
        return await _aio.gather(agent.arun("A", scope=scope), agent.arun("B", scope=scope),
                                 return_exceptions=True)

    results = run_sync(_both())
    interrupts = [r for r in results if not isinstance(r, Exception) and getattr(r, "interrupted", False)]
    errors = [r for r in results if isinstance(r, SessionError)]
    assert len(interrupts) == 1 and len(errors) == 1    # one suspends, one is gated (the suspended state isn't overwritten)
    st = ExecutionState.from_json(cp.load(scope=scope))
    assert st.pending and st.pending[0].tool_name == "danger"   # the checkpoint holds only one suspended state


# ---------- Streaming output-guardrail buffer mode ----------

def test_stream_buffer_mode_guardrail_blocks_before_output():
    """buffer_output=True: when the output guardrail trips, content is blocked before reaching the consumer (nothing emitted); default streaming has already emitted it and can't hold it back."""
    class _SLLM:
        provider = "stub"
        model = "s"

        async def stream(self, messages, **kw):
            yield "禁"
            yield "词"

    class BlockGuard:
        def check(self, text):
            return type("R", (), {"passed": "禁词" not in text, "message": "含禁词"})()

    a = UnifiedAgent("u", _SLLM(), output_guardrails=[BlockGuard()])

    received = []
    with pytest.raises(GuardrailTripwireError):
        for piece in a.stream_run("hi", buffer_output=True):
            received.append(piece)
    assert received == []                                    # ★ buffer mode: content never reached the consumer

    received2 = []
    with pytest.raises(GuardrailTripwireError):
        for piece in a.stream_run("hi"):
            received2.append(piece)
    assert received2 == ["禁", "词"]                          # default streaming: content already emitted, the guardrail raises only afterward


def test_stream_buffer_mode_passes_through_when_ok():
    """buffer_output=True with compliant output: each segment is released fully, history stored as usual."""
    class _SLLM:
        provider = "stub"
        model = "s"

        async def stream(self, messages, **kw):
            yield "你"
            yield "好"

    a = UnifiedAgent("u", _SLLM())
    assert "".join(a.stream_run("hi", buffer_output=True)) == "你好"
    assert len(a.get_history()) == 2


# ---------- Invalid arguments also go into the trace ----------

def test_invalid_tool_args_emits_trace():
    """A tool-argument parse failure (invalid JSON) still emits a tool_call trace (status=invalid_args) - the audit sees this invalid call."""
    from agentmaker.runtime.observability.tracer import Tracer
    tracer = Tracer()
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    bad = LLMResponse(content="", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": "{not json"}}])
    agent = UnifiedAgent("a", ScriptLLM([bad, "完成"]), tool_registry=reg, tracer=tracer)
    assert agent.run("go").final_output == "完成"
    inv = [e for e in tracer.events if e.get("type") == "tool_call" and e.get("status") == "invalid_args"]
    assert len(inv) == 1 and inv[0]["tool"] == "calculator"
