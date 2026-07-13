"""Hermetic contract tests for core framework components (offline, no key).

Covers: ① history-compaction trace event; ② Gemini default retry; ③ MCP remote transport construction;
④ OTel traceparent propagation; ⑤ SessionStore.list_scopes enumeration; ⑥ tool-level parallel dispatch.
"""
import asyncio
import inspect
import json

import pytest

import agentmaker.core.trace_events as ev
from agentmaker import (Agent, Harness, Message, OTelExporter, Scope, ScopeSummary,
                          SqliteSessionStore, Tool, ToolParameter, ToolRegistry, ToolResponse,
                          Tracer, current_trace_carrier)
from agentmaker.core.llm_response import LLMResponse
from agentmaker.runtime.execution import new_run_id, reset_run, start_run
from agentmaker.testing import ScriptedLLM


# ========== ① History compaction emits EVENT_CONTEXT_COMPACT ==========

class _ShrinkCompactor:
    """Real compaction: returns two fewer items than the input (fires the event)."""
    async def acompact(self, history, *, asummarize):
        return history[:-2] if len(history) > 2 else history


class _NoopCompactor:
    """No compaction: returns the same object unchanged (should not fire the event)."""
    async def acompact(self, history, *, asummarize):
        return history


class _SameLenCompactor:
    """Real compaction but the count is unchanged (keep_recent+1 boundary: 1 old turn -> 1 summary): returns a new list of the same length."""
    async def acompact(self, history, *, asummarize):
        return [Message("摘要", "system"), *history[1:]]     # new object, same length


def test_compaction_emits_context_compact_event():
    hist = [Message(str(i), "user") for i in range(5)]
    tr = Tracer()
    asyncio.run(Harness(object(), compactor=_ShrinkCompactor(), tracer=tr).aassemble(hist))
    compacts = [e for e in tr.events if e["type"] == ev.EVENT_CONTEXT_COMPACT]
    assert len(compacts) == 1
    assert compacts[0]["before"] == 5 and compacts[0]["after"] == 3


def test_no_compaction_no_event():
    hist = [Message(str(i), "user") for i in range(5)]
    tr = Tracer()
    asyncio.run(Harness(object(), compactor=_NoopCompactor(), tracer=tr).aassemble(hist))
    assert not [e for e in tr.events if e["type"] == ev.EVENT_CONTEXT_COMPACT]


def test_compaction_emits_even_when_length_unchanged():
    """Real compaction with an unchanged count (keep_recent+1 boundary) must still fire the event: the guard uses object identity, not a length delta."""
    hist = [Message(str(i), "user") for i in range(5)]
    tr = Tracer()
    asyncio.run(Harness(object(), compactor=_SameLenCompactor(), tracer=tr).aassemble(hist))
    compacts = [e for e in tr.events if e["type"] == ev.EVENT_CONTEXT_COMPACT]
    assert len(compacts) == 1
    assert compacts[0]["before"] == 5 and compacts[0]["after"] == 5


def test_context_compact_registered():
    assert ev.EVENT_CONTEXT_COMPACT == "context_compact"
    assert ev.EVENT_CONTEXT_COMPACT in ev.ALL_EVENT_TYPES


# ========== ② Gemini default retry alignment ==========

def test_gemini_configures_default_retry(monkeypatch):
    pytest.importorskip("google.genai")
    import google.genai as genai

    from agentmaker.core.adapters.gemini import _DEFAULT_RETRY_ATTEMPTS, GeminiAdapter

    captured = {}
    monkeypatch.setattr(genai, "Client",
                        lambda *, api_key, http_options: captured.update(ho=http_options) or object())
    adapter = GeminiAdapter(model="gemini-2.5-flash", api_key="k", base_url=None,
                            timeout=30, default_temperature=None)

    async def go():
        adapter._ensure_client()

    asyncio.run(go())
    ho = captured["ho"]
    assert ho.retry_options is not None
    assert ho.retry_options.attempts == _DEFAULT_RETRY_ATTEMPTS == 3   # 3 = 2 retries, matching the openai/anthropic default


# ========== ③ MCP remote StreamableHTTP construction + transport mutual exclusion ==========

def test_mcp_stdio_and_http_construction():
    pytest.importorskip("mcp")
    from agentmaker.tools.integrations.mcp import MCPClient
    stdio = MCPClient(command="python", args=["s.py"], namespace="a")
    assert stdio.url is None and stdio.command == "python"
    http = MCPClient(url="https://x/mcp", namespace="a", headers={"k": "v"}, auth=None)
    assert http.command is None and http.url.endswith("/mcp")


@pytest.mark.parametrize("bad", [
    {"namespace": "a"},                                              # neither given
    {"command": "p", "url": "http://x", "namespace": "a"},           # both given
    {"url": "http://x", "args": ["a"], "namespace": "a"},            # http + stdio-only args
    {"command": "p", "headers": {"a": "b"}, "namespace": "a"},       # stdio + http-only headers
])
def test_mcp_transport_mutual_exclusion(bad):
    pytest.importorskip("mcp")
    from agentmaker.tools.integrations.mcp import MCPClient
    with pytest.raises(ValueError):
        MCPClient(**bad)


# ========== ④ OTel traceparent propagation ==========

def test_trace_carrier_stored_and_inherited():
    carrier = {"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
    token = start_run(new_run_id(), trace_carrier=carrier)
    try:
        assert current_trace_carrier() == carrier
        inner = start_run(new_run_id(), trace_carrier={"traceparent": "other"})  # nested run
        assert inner is None                                    # inherits the outer run, doesn't start a new one
        assert current_trace_carrier() == carrier               # and isn't overwritten by the inner one
    finally:
        reset_run(token)
    assert current_trace_carrier() is None                      # None outside a run


def test_otel_exporter_accepts_carrier_provider():
    assert "carrier_provider" in inspect.signature(OTelExporter.__init__).parameters


def test_run_api_accepts_trace_carrier():
    from agentmaker.agents.base import BaseAgent
    for method in ("arun", "run", "aresume", "resume"):
        assert "trace_carrier" in inspect.signature(getattr(BaseAgent, method)).parameters


# ========== ⑤ SessionStore.list_scopes enumeration ==========

def _seed_store():
    store = SqliteSessionStore()
    store.append_many([Message("hi", "user"), Message("yo", "assistant")],
                      scope=Scope(base="session", user="alice", session="c1"))
    store.append(Message("q", "user"), scope=Scope(base="session", user="alice", session="c2"))
    store.append(Message("b", "user"), scope=Scope(base="session", user="bob", session="c1"))
    return store


def test_list_scopes_enumeration_and_b_semantics():
    store = _seed_store()
    assert {x.value: x.message_count for x in store.list_scopes(along="user")} == {"alice": 3, "bob": 1}
    # B semantics: constrain user=alice -> list only sessions under alice
    assert {x.value: x.message_count
            for x in store.list_scopes(along="session", scope=Scope(user="alice"))} == {"c1": 2, "c2": 1}
    summary = store.list_scopes(along="user")[0]
    assert isinstance(summary, ScopeSummary)
    assert summary.first_at is not None and summary.last_at is not None
    # async version agrees
    assert {x.value for x in asyncio.run(store.alist_scopes(along="user"))} == {"alice", "bob"}
    store.close()


def test_list_scopes_rejects_unknown_dimension():
    store = SqliteSessionStore()
    with pytest.raises(ValueError):
        store.list_scopes(along="not_a_dimension")     # allowlist check doubles as an SQL-injection guard
    store.close()


def test_list_scopes_unsupported_backend_raises():
    from agentmaker.runtime.sessions import SessionStore

    class _Bare(SessionStore):
        def append(self, message, *, scope=None): ...
        def load(self, *, scope=None, all_scopes=False): return []
        def clear(self, *, scope=None, all_scopes=False): ...

    with pytest.raises(NotImplementedError):
        _Bare().list_scopes()


# ========== ⑥ Tool-level parallel dispatch ==========

class _Recording(ScriptedLLM):
    """Records the messages seen on each chat call, to observe the order tool results are fed back to the LLM."""
    def __init__(self, script):
        super().__init__(script)
        self.seen = []

    async def chat(self, messages, *, tools=None, **kwargs):
        self.seen.append([dict(m) for m in messages])
        return await super().chat(messages, tools=tools, **kwargs)

    def tool_results(self):
        return [m["content"] for m in self.seen[-1] if m.get("role") == "tool"]


def _multi_call(pairs):
    """An LLMResponse carrying multiple tool_calls."""
    return LLMResponse(content="", model="test", tool_calls=[
        {"id": cid, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        for (n, a, cid) in pairs])


def test_parallel_tools_run_concurrently_and_in_order():
    probe = {"active": 0, "max": 0}

    class _Probe(Tool):
        supports_parallel = True

        def __init__(self, name):
            super().__init__(name=name, description=name)

        def get_parameters(self):
            return [ToolParameter("x", "string", "")]

        async def arun(self, parameters):
            probe["active"] += 1
            probe["max"] = max(probe["max"], probe["active"])   # both running at once -> max==2 (proves concurrency deterministically, not by timing)
            await asyncio.sleep(0.03)
            probe["active"] -= 1
            return ToolResponse.ok(f"{self.name}:{parameters.get('x')}")

    reg = ToolRegistry()
    reg.register(_Probe("a"))
    reg.register(_Probe("b"))
    llm = _Recording([_multi_call([("a", {"x": "1"}, "c1"), ("b", {"x": "2"}, "c2")]), "done"])
    asyncio.run(Agent("t", llm, tool_registry=reg).arun("go"))
    assert probe["max"] == 2                                     # genuinely concurrent
    assert llm.tool_results() == ["a:1", "b:2"]                  # results backfilled in original call order


class _SerialTool(Tool):
    """Doesn't set supports_parallel (defaults to False): never enters a parallel batch."""
    def __init__(self, name):
        super().__init__(name=name, description=name)

    def get_parameters(self):
        return [ToolParameter("x", "string", "")]

    async def arun(self, parameters):
        return ToolResponse.ok(f"{self.name}:{parameters.get('x')}")


class _ParallelTool(_SerialTool):
    supports_parallel = True


def test_mixed_batch_preserves_call_order():
    reg = ToolRegistry()
    reg.register(_ParallelTool("p1"))
    reg.register(_SerialTool("s1"))
    reg.register(_ParallelTool("p2"))
    llm = _Recording([_multi_call([("p1", {"x": "a"}, "c1"), ("s1", {"x": "b"}, "c2"),
                                   ("p2", {"x": "c"}, "c3")]), "done"])
    asyncio.run(Agent("t", llm, tool_registry=reg).arun("go"))
    assert llm.tool_results() == ["p1:a", "s1:b", "p2:c"]        # parallel/serial/parallel interleaving still preserves order


def test_high_risk_tool_never_parallelized():
    class _RiskyParallel(_ParallelTool):
        requires_confirmation = True                            # high-risk: not parallelized even though supports_parallel is set

    reg = ToolRegistry()
    reg.register(_RiskyParallel("risky"))
    reg.register(_ParallelTool("safe"))
    llm = _Recording([_multi_call([("risky", {"x": "1"}, "c1"), ("safe", {"x": "2"}, "c2")]), "done"])
    asyncio.run(Agent("t", llm, tool_registry=reg).arun("go"))  # no confirm -> risky is safely rejected, doesn't hang
    results = llm.tool_results()
    assert len(results) == 2 and results[1] == "safe:2"


def test_supports_parallel_flag_propagates_through_callable_paths():
    from agentmaker import tool

    @tool(supports_parallel=True)
    def lookup(city: str) -> str:
        """Look up read-only information."""
        return city

    assert lookup.supports_parallel is True
    reg = ToolRegistry()
    reg.register_function(lambda p: p.get("q", ""), name="echo", description="echo",
                          parameters=[ToolParameter("q", "string", "")], supports_parallel=True)
    assert reg.get("echo").supports_parallel is True
