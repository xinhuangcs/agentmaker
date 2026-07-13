"""Prompt registry rendering, override validation, and Agent propagation tests.

Hermetic (no network, no key): builds an Agent with a stub LLM and only checks prompt get / list / override and protocol protection.
"""

import pytest

from agentmaker.prompts import DEFAULT_PROMPTS, PromptError, PromptRegistry, PromptTemplate


class _StubLLM:
    provider = "stub"
    model = "stub"


# ---------- PromptTemplate ----------

def test_render_substitutes_declared_vars_only():
    """render substitutes only declared {var}; non-placeholder braces like {...} in a JSON example are left as-is."""
    t = PromptTemplate("hi {name}, raw {\"k\":1}", variables=("name",))
    assert t.render(name="A") == 'hi A, raw {"k":1}'


def test_render_missing_var_raises():
    """A missing declared placeholder raises PromptError."""
    t = PromptTemplate("{a}{b}", variables=("a", "b"))
    with pytest.raises(PromptError):
        t.render(a="x")


def test_with_text_validates_vars_and_protected():
    """Override text: missing a placeholder or a protected marker is rejected; a complete one passes."""
    t = PromptTemplate("用 {x}；记号 Action:", variables=("x",), protected=("Action:",))
    with pytest.raises(PromptError):
        t.with_text("没有占位也没有记号")           # missing both {x} and Action:
    with pytest.raises(PromptError):
        t.with_text("有 {x} 但没记号")              # missing Action:
    ok = t.with_text("new {x} keeps Action:")        # complete
    assert ok.variables == ("x",) and ok.protected == ("Action:",)


# ---------- PromptRegistry ----------

def test_registry_basic_ops():
    r = PromptRegistry({"a.b": PromptTemplate("hello {n}", variables=("n",))})
    assert "a.b" in r and r.text("a.b") == "hello {n}"
    assert r.render("a.b", n=1) == "hello 1"
    assert r.keys() == ["a.b"] and r.as_dict() == {"a.b": "hello {n}"}
    with pytest.raises(PromptError):
        r.get("missing")


def test_with_overrides_is_immutable_and_validated():
    """with_overrides returns a new registry without mutating the original; an unknown key raises."""
    base = PromptRegistry({"k": PromptTemplate("zh")})
    eng = base.with_overrides({"k": "en"})
    assert eng.text("k") == "en" and base.text("k") == "zh"      # original unchanged
    with pytest.raises(PromptError):
        base.with_overrides({"nope": "x"})


def test_override_mutates_in_place_shared():
    """override mutates in place: every holder of the same registry sees it (used for agent -> harness propagation)."""
    r = PromptRegistry({"k": PromptTemplate("zh")})
    alias = r
    r.override({"k": "en"})
    assert alias.text("k") == "en"


def test_register_adds_new_key_and_enters_override_system():
    """register adds a new key: it can be listed / rendered / overridden via with_overrides (third-party components join the same localization system)."""
    r = DEFAULT_PROMPTS.copy()
    r.register("myapp.greet", "你好 {name}", variables=("name",))
    assert "myapp.greet" in r and "myapp.greet" in r.keys()
    assert r.render("myapp.greet", name="李雷") == "你好 李雷"
    en = r.with_overrides({"myapp.greet": "hello {name}"})            # an app's own language pack can override it
    assert en.render("myapp.greet", name="Lei") == "hello Lei"


def test_register_rejects_existing_key():
    """An existing key can't be re-registered (guards against clobbering framework built-ins); an override missing placeholders is still caught by with_text validation."""
    r = DEFAULT_PROMPTS.copy()
    with pytest.raises(PromptError):
        r.register("chat.persona", "重复登记")                        # already in the framework
    r.register("myapp.tpl", "需要 {x}", variables=("x",))
    with pytest.raises(PromptError):                                  # override drops the placeholder -> caught by validation
        r.with_overrides({"myapp.tpl": "没有占位了"})


# ---------- DEFAULT_PROMPTS completeness ----------

def test_default_prompts_has_expected_keys():
    keys = set(DEFAULT_PROMPTS.keys())
    for k in ("memory.extract", "memory.reconcile", "context.summary", "rag.ask", "react.persona", "react.style",
              "agent.empty_reply", "plan.planner", "reflection.reflect", "harness.context_guard",
              "harness.schema_instruction", "tool.error.not_found", "context.section.memory"):
        assert k in keys, k


def test_protocol_protected_defaults():
    """Protocol markers are present by default: reconcile carries the four ops, the schema carries JSON (prompts with a protected marker must keep it when overridden)."""
    assert all(op in DEFAULT_PROMPTS.text("memory.reconcile") for op in ("ADD", "UPDATE", "DELETE", "NOOP"))
    assert "JSON" in DEFAULT_PROMPTS.text("harness.schema_instruction")


# ---------- Agent.get_prompts / update_prompts propagation ----------

def test_agent_get_and_update_prompts_propagate():
    """update_prompts mutates in place; the agent and its harness share one registry and it takes effect immediately; a missing protocol marker is rejected."""
    from agentmaker.agents.agent import Agent
    a = Agent("t", _StubLLM(), prompts=DEFAULT_PROMPTS.copy())          # isolate with a copy so the global isn't polluted
    d = a.get_prompts()
    assert "chat.persona" in d and len(d) > 40
    a.update_prompts({"chat.persona": "You are terse."})
    assert a.prompts.text("chat.persona") == "You are terse."           # persona takes effect (with no system_prompt, chat.persona is the default)
    assert a.harness.prompts.text("chat.persona") == "You are terse."   # harness stays in sync (shares the same registry)
    with pytest.raises(PromptError):                                    # breaking the reconcile protocol markers (four ops) is rejected
        a.update_prompts({"memory.reconcile": "丢了四个操作标记"})


def test_agent_default_prompts_isolated_per_instance():
    """Without prompts=, each Agent copies DEFAULT_PROMPTS at construction: update_prompts calls don't cross-contaminate and don't touch the global."""
    from agentmaker.agents.agent import Agent
    original = DEFAULT_PROMPTS.text("chat.persona")
    a = Agent("a", _StubLLM())                                       # no prompts= -> isolated by default
    b = Agent("b", _StubLLM())
    a.update_prompts({"chat.persona": "A persona"})
    assert a.prompts.text("chat.persona") == "A persona"
    assert a.harness.prompts.text("chat.persona") == "A persona"     # this agent's whole chain (harness) follows
    assert b.prompts.text("chat.persona") == original                # the other agent is unaffected
    assert DEFAULT_PROMPTS.text("chat.persona") == original          # the global singleton is unaffected


def test_override_with_prompt_template_still_validated():
    """A PromptTemplate override value must still satisfy the target key's protocol markers -- you can't bypass protection by passing a PromptTemplate."""
    bad = PromptTemplate("broken")                                  # drops memory.reconcile's four op markers
    with pytest.raises(PromptError):
        DEFAULT_PROMPTS.with_overrides({"memory.reconcile": bad})
    ok = DEFAULT_PROMPTS.with_overrides(                            # keeps the protocol markers -> passes
        {"memory.reconcile": PromptTemplate("新规则 ADD UPDATE DELETE NOOP")})
    assert "ADD" in ok.text("memory.reconcile")


def test_build_agent_threads_prompts():
    """The declarative path build_agent threads spec.prompts through to the agent and its harness (isolated overrides work via spec too)."""
    from agentmaker import AgentSpec, LLMClient, build_agent
    eng = DEFAULT_PROMPTS.copy()
    a = build_agent(AgentSpec(name="a", model=LLMClient("deepseek", api_key="x"), prompts=eng))
    assert a.prompts is eng and a.harness.prompts is eng


def test_update_prompts_does_not_touch_separately_built_tool():
    """Boundary: agent.update_prompts only affects the agent's own chain; a separately constructed tool is unaffected --
    to localize such a tool, pass it prompts= or override globally before constructing it."""
    from agentmaker import CalculatorTool
    from agentmaker.agents.agent import Agent
    a = Agent("a", _StubLLM(), prompts=DEFAULT_PROMPTS.copy())          # isolate with a copy so the global isn't polluted
    independent = CalculatorTool()                                  # independent tool: reads the global DEFAULT_PROMPTS (unchanged)
    a.update_prompts({"tool.desc.calculator": "EN calc desc"})      # only mutates the agent's copy
    assert independent.description != "EN calc desc"                # independent tool is unaffected (the boundary)
    shared = CalculatorTool(prompts=a.prompts)                      # recommended: the tool shares the agent's registry
    assert shared.description == "EN calc desc"


def test_chinese_pack_complete_and_valid():
    """The ready-made Chinese pack agentmaker.prompts.packs.CHINESE_PROMPTS covers every key and passes override validation (placeholders/protocol markers intact); and the default (English) catalog itself contains no Chinese."""
    import re
    from agentmaker.prompts.packs import CHINESE_PROMPTS, chinese_registry
    assert set(CHINESE_PROMPTS) == set(DEFAULT_PROMPTS.keys())      # one-to-one with the registry: nothing missing / nothing extra
    chinese_registry()                                              # overriding validates placeholders + protocol markers; invalid ones raise PromptError
    cjk = re.compile(r"[一-鿿　-〿]")
    leaked = [k for k, v in DEFAULT_PROMPTS.as_dict().items() if cjk.search(v)]
    assert not leaked, f"default catalog (English) still contains Chinese: {leaked}"


def test_tool_status_messages_go_through_prompts(tmp_path):
    """Tool error / status messages all come from prompts (not hardcoded): after switching to pure ASCII, every action's error / fallback text is free of Chinese.

    Covers dependency-free error paths for calculator, search, notes, shell, RAG, memory, tool search,
    and conversation search.
    """
    import re
    cjk = re.compile(r"[一-鿿　-〿]")
    from agentmaker import (CalculatorTool, CLITool, MemoryTool, NotesTool, RAGTool, SearchTool)
    from agentmaker.runtime.sessions import ConversationSearchTool
    from agentmaker.tools.tool_retriever import ToolSearchTool
    eng = {}
    for k in DEFAULT_PROMPTS.keys():
        t = DEFAULT_PROMPTS.get(k)
        eng[k] = " ".join(["EN", k] + ["{" + v + "}" for v in t.variables] + list(t.protected))
    reg = DEFAULT_PROMPTS.with_overrides(eng)

    calc = CalculatorTool(prompts=reg)
    notes = NotesTool(str(tmp_path / "notes_i18n"), prompts=reg)
    rag = RAGTool(pipeline=object(), rag_retriever=object(), prompts=reg)
    mem = MemoryTool(memory=object(), prompts=reg)
    out = [
        calc.run({"expression": ""}).text,                          # calc.empty
        calc.run({"expression": "foo+1"}).text,                     # calc.eval_failed ← bad_name
        calc.run({"expression": "1/0"}).text,                       # calc.div_zero
        calc.run({"expression": "round(1, ndigits=2)"}).text,       # calc.eval_failed ← no_kwargs
        SearchTool(prompts=reg).run({"query": ""}).text,            # search.empty
        notes.run({"action": "delete", "path": "p"}).text,          # notes.bad_action
        notes.run({"action": "read", "path": ""}).text,             # notes.empty_path
        notes.run({"action": "read", "path": "../../etc/passwd"}).text,  # notes.path_escape
        notes.run({"action": "append", "path": "p.md", "content": " "}).text,  # notes.append_empty
        CLITool([], prompts=reg).run({"command": ""}).text,         # shell.empty_cmd
        CLITool(["ls"], prompts=reg).run({"command": "rm -rf /"}).text,  # shell.not_allowed
        rag.run({"action": "add_text"}).text,                       # rag.need_text
        rag.run({"action": "nope"}).text,                           # rag.unknown_action
        mem.run({"action": "recall"}).text,                         # mem.need_query
        mem.run({"action": "nope"}).text,                           # mem.unknown_action
        ToolSearchTool(tool_retriever=object(), prompts=reg).run({"query": ""}).text,  # tool_search.need_query
        ConversationSearchTool(object(), prompts=reg).run({"query": ""}).text,         # conv.need_query
    ]
    leaks = {i: "".join(cjk.findall(s)) for i, s in enumerate(out) if s and cjk.search(s)}
    assert not leaks, f"tool copy still contains Chinese (not routed through prompts): {leaks}"
    assert all(s.startswith("EN ") for s in out), "all should be rendered via the master registry (with an EN prefix)"


def test_full_english_switch_leaves_no_chinese(tmp_path):
    """After swapping the whole registry to pure ASCII, tool schemas + the four-paradigm system prompts must contain no Chinese -- proving the full English switch leaves no hardcoded residue."""
    import json
    import re
    cjk = re.compile(r"[一-鿿　-〿]")
    from agentmaker import (CLITool, CalculatorTool, Harness, NotesTool, PlanAgent,
                        ReflectionAgent, SearchTool, ToolPermissions, ToolRegistry)
    from agentmaker.runtime.harness import _validate_structured
    eng = {}
    for k in DEFAULT_PROMPTS.keys():                      # each key -> pure ASCII, keeping placeholders + protocol markers
        t = DEFAULT_PROMPTS.get(k)
        eng[k] = " ".join(["EN", k] + ["{" + v + "}" for v in t.variables] + list(t.protected))
    reg = DEFAULT_PROMPTS.with_overrides(eng)

    treg = ToolRegistry(prompts=reg)
    for tool in (CalculatorTool(prompts=reg), SearchTool(prompts=reg),
                    NotesTool(str(tmp_path / "notes"), prompts=reg), CLITool([], prompts=reg)):   # empty allowlist exercises tool.none
        treg.register(tool)
    out = [json.dumps(treg.to_openai_schema(), ensure_ascii=False),    # tool schema (descriptions + params)
           treg.get_tools_description(),                               # text-protocol tool listing (with required/optional)
           treg.execute_tool("nope", {}).text,                        # tool error: not found
           treg.execute_tool("calculator", {"expression": 123}).text]  # param validation: type mismatch (path present, triggers validation_field)
    h = Harness(_StubLLM(), tool_registry=treg,
                permissions=ToolPermissions(deny=["calculator"], prompts=reg), prompts=reg)
    import asyncio
    out.append(asyncio.run(h.aexec_tool("calculator", {"expression": "1"})).text)   # permission denial (denial_reason flows into {reason})

    class _M(__import__("pydantic").BaseModel):
        x: int
    out.append(_validate_structured(None, "", reg)[1])                # structured: empty content
    out.append(_validate_structured(_M, '{"x":"nope"}', reg)[1])      # structured: validation failure (flows into retry_note's {err})
    p = PlanAgent("p", _StubLLM(), prompts=reg)
    f = ReflectionAgent("f", _StubLLM(), prompts=reg)
    out += [reg.text("chat.persona"), reg.text("react.persona"), reg.text("react.style"),   # chat default persona + react presets
            reg.text("agent.empty_reply"), reg.text("agent.invalid_reply"), reg.text("agent.exhausted"),
            p._planner_prompt("q"), p._executor_prompt("s", "q", ["a"], []), p._synthesize_prompt("q", ["h"]),
            f._initial_prompt("t"), f._reflect_prompt("t", [{"kind": "draft", "text": "d"}]),
            f._refine_prompt("t", [{"kind": "refine", "text": "x"}])]
    leaks = {i: "".join(cjk.findall(s)) for i, s in enumerate(out) if s and cjk.search(s)}
    assert not leaks, f"Chinese still leaks after full English switch: {leaks}"


def test_reflection_pass_signal_tracks_override():
    """Localization: after overriding reflection.pass_signal, both the reflect prompt and the _passed check use the same new signal (no hardcoded "good enough")."""
    from agentmaker import ReflectionAgent
    a = ReflectionAgent("r", _StubLLM(), prompts=DEFAULT_PROMPTS.copy())
    a.update_prompts({"reflection.pass_signal": "DONE"})
    prompt = a._reflect_prompt("task", [{"kind": "draft", "text": "d"}])
    assert "DONE" in prompt and "GOOD ENOUGH" not in prompt
    assert a._passed([{"kind": "critique", "text": "looks DONE to me"}]) is True
