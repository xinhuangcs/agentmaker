"""Regression suite for @tool / Tool.from_callable / register_callable / Agent(tools=) (hermetic: no key / no network).

Locks: signature -> ToolParameter inference (type / default / required), parameter descriptions from Annotated or a Chinese docstring, calling by signature-expanded kwargs (extra keys filtered), native async await + sync-entry fail-loud, register-time fail-loud on missing annotations / var-args / unmappable annotations, requires_confirmation / external_content passthrough, and Agent(tools=) vs. tool_registry= being mutually exclusive yet equivalent.

Note: tests/ is gitignored, so new files need `git add -f`.
"""

import asyncio
from typing import Annotated, Optional

import pytest

from agentmaker import Tool, ToolRegistry, tool
from agentmaker.core.exceptions import ToolRegistrationError
from agentmaker.prompts import DEFAULT_PROMPTS


class _StubLLM:
    provider = "stub"
    model = "stub"


# ---------- signature inference: type / default / required ----------

def test_tool_infers_params_from_signature():
    """@tool infers params from the signature: no default -> required; has a default -> optional with that default; types map correctly."""
    @tool
    def forecast(city: str, days: int = 3, metric: bool = True) -> str:
        """查询天气。"""
        return f"{city}/{days}/{metric}"

    assert isinstance(forecast, Tool)
    params = {p.name: p for p in forecast.get_parameters()}
    assert params["city"].type == "string" and params["city"].required is True
    assert params["days"].type == "integer" and params["days"].required is False and params["days"].default == 3
    assert params["metric"].type == "boolean" and params["metric"].required is False
    reg = ToolRegistry()
    reg.register(forecast)
    assert reg.to_openai_schema()[0]["function"]["parameters"]["required"] == ["city"]


def test_optional_annotation_is_not_required():
    """An Optional[T] / T | None annotation means optional (even without a default)."""
    @tool
    def f(a: Optional[str] = None, b: int | None = None) -> str:
        return "x"

    params = {p.name: p for p in f.get_parameters()}
    assert params["a"].required is False and params["a"].type == "string"
    assert params["b"].required is False and params["b"].type == "integer"


# ---------- parameter description source: Annotated > Chinese docstring "参数" section ----------

def test_annotated_description_wins():
    """The string metadata in Annotated[T, '...'] is used as the parameter description."""
    @tool
    def f(city: Annotated[str, "城市名"]) -> str:
        return city

    assert f.get_parameters()[0].description == "城市名"


def test_docstring_param_section_parsed():
    """Without Annotated, parameter descriptions come from the Chinese "参数" section by matching name; Chinese colons / parentheses do not break parsing, and a missing section does not crash."""
    @tool
    def f(city: str, days: int = 1) -> str:
        """查询天气。

        参数：
            city: 城市名（如：北京）
            days: 预报天数
        """
        return city

    params = {p.name: p for p in f.get_parameters()}
    assert params["city"].description == "城市名（如：北京）"   # contains a Chinese colon / parens; takes everything after the first colon
    assert params["days"].description == "预报天数"

    @tool
    def g(x: int) -> str:
        """没有参数段的工具。"""
        return str(x)
    assert g.get_parameters()[0].description == ""            # no "参数" section -> empty description, no crash


# ---------- call by signature-expanded kwargs: extra keys filtered ----------

def test_call_expands_kwargs_and_filters_extras():
    """execute_tool calls the function with signature-expanded kwargs; extra keys in the input are filtered out without error."""
    @tool
    def f(city: str, days: int = 2) -> str:
        return f"{city}-{days}"

    reg = ToolRegistry()
    reg.register(f)
    assert reg.execute_tool("f", {"city": "北京", "days": 5, "extra": "ignored"}).text == "北京-5"
    assert reg.execute_tool("f", {"city": "上海"}).text == "上海-2"   # missing days -> uses the function default


# ---------- async: native await + sync-entry fail-loud ----------

def test_async_tool_runs_via_async_entry_and_sync_fails_loud():
    """An async @tool executes via aexecute_tool's await; the sync run entry fails loud (never feeds a coroutine back to the model as a result)."""
    @tool
    async def af(x: int) -> str:
        return f"async-{x}"

    reg = ToolRegistry()
    reg.register(af)
    assert asyncio.run(reg.aexecute_tool("af", {"x": 7})).text == "async-7"
    with pytest.raises(TypeError):
        af.run({"x": 1})                                       # sync entry rejects an async tool


# ---------- register-time fail-loud: missing annotation / var-args / unmappable annotation ----------

def test_missing_annotation_fails_loud():
    with pytest.raises(ToolRegistrationError):
        @tool
        def f(city) -> str:                                    # no type annotation
            return city


def test_var_args_fails_loud():
    with pytest.raises(ToolRegistrationError):
        @tool
        def f(*args: int) -> str:                              # var-args cannot map to a named schema
            return "x"


def test_unmappable_annotation_fails_loud():
    class Custom:
        pass
    with pytest.raises(ToolRegistrationError):
        @tool
        def f(x: Custom) -> str:                               # custom-class annotation cannot map
            return "x"


def test_bad_name_fails_loud():
    with pytest.raises(ToolRegistrationError):
        Tool.from_callable(lambda x: x, name="bad name!")      # invalid name (space / exclamation)


# ---------- requires_confirmation / external_content passthrough ----------

def test_requires_confirmation_threads_and_gate_denies_without_confirm():
    """requires_confirmation=True passthrough: execute_tool safely rejects when no confirm is given."""
    @tool(requires_confirmation=True)
    def danger(x: str) -> str:
        return "did " + x

    reg = ToolRegistry()
    reg.register(danger)
    assert danger.needs_confirmation({"x": "a"}) is True
    assert reg.execute_tool("danger", {"x": "a"}).status == "error"           # no confirm -> rejected
    assert reg.execute_tool("danger", {"x": "a"}, confirm=lambda t, p: True).text == "did a"


def test_external_content_threads_to_instance():
    """external_content=True is set on the instance attribute (so the framework's anti-injection wrapper can read it via getattr)."""
    @tool(external_content=True)
    def fetch(url: str) -> str:
        return "content"
    assert getattr(fetch, "external_content", False) is True


# ---------- from_callable / register_callable equivalent paths ----------

def test_from_callable_and_register_callable():
    """Tool.from_callable and ToolRegistry.register_callable use the same inference, equivalent to @tool."""
    def reverse(text: str) -> str:
        """反转文本。"""
        return text[::-1]

    t = Tool.from_callable(reverse)
    assert isinstance(t, Tool) and t.name == "reverse" and t.run({"text": "abc"}).text == "cba"

    reg = ToolRegistry()
    reg.register_callable(reverse)
    assert reg.execute_tool("reverse", {"text": "xy"}).text == "yx"


def test_register_function_legacy_path_unchanged():
    """register_function's legacy contract (function receives the whole dict, hand-written parameter list) is unchanged (backward compatible)."""
    from agentmaker import ToolParameter
    reg = ToolRegistry()
    reg.register_function(lambda p: p["city"] + "晴", "weather", "查天气",
                          [ToolParameter("city", "string", "城市名")])
    assert reg.execute_tool("weather", {"city": "北京"}).text == "北京晴"


# ---------- Agent(tools=): convenience entry + mutually exclusive with tool_registry + equivalent to AgentSpec ----------

def test_agent_accepts_tools_list():
    """Agent(tools=[...]) accepts a list of Tools (including @tool-decorated objects), normalizes to a registry, and can execute."""
    from agentmaker import CalculatorTool
    from agentmaker.agents.agent import Agent

    @tool
    def echo(text: str) -> str:
        return text

    a = Agent("a", _StubLLM(), tools=[CalculatorTool(), echo])
    assert {t.name for t in a.tool_registry.list_tools()} == {"calculator", "echo"}


def test_agent_tools_registry_inherits_agent_prompts():
    """Agent(tools=[...])'s internal registry inherits the agent's prompts: registry-level errors (unknown tool, etc.) match the agent's language."""
    from agentmaker import CalculatorTool
    from agentmaker.agents.agent import Agent
    reg_prompts = DEFAULT_PROMPTS.copy()
    a = Agent("a", _StubLLM(), tools=[CalculatorTool()], prompts=reg_prompts)
    assert a.tool_registry.prompts is a.prompts


def test_agent_tools_and_tool_registry_mutually_exclusive():
    """Passing both tools= and tool_registry= -> ValueError."""
    from agentmaker.agents.agent import Agent

    @tool
    def echo(text: str) -> str:
        return text
    with pytest.raises(ValueError):
        Agent("a", _StubLLM(), tools=[echo], tool_registry=ToolRegistry())


def test_agent_tools_equiv_to_spec_tools():
    """Agent(tools=[...]) and AgentSpec(tools=[...]) produce the same tool set through build_agent (single source of truth)."""
    from agentmaker import AgentSpec, LLMClient, build_agent
    from agentmaker.agents.agent import Agent

    @tool
    def echo(text: str) -> str:
        return text

    direct = Agent("a", LLMClient("deepseek", api_key="x"), tools=[echo])
    declarative = build_agent(AgentSpec(name="a", model=LLMClient("deepseek", api_key="x"),
                                        strategy="chat", tools=[echo]))
    assert {t.name for t in direct.tool_registry.list_tools()} == {t.name for t in declarative.tool_registry.list_tools()}
