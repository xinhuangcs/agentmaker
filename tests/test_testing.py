"""Contract + usage regression for agentmaker.testing (hermetic: no key / no network).

Proves third parties can test their own agents with these doubles **for free and offline**: ScriptedLLM drives plain chat / tool loop / streaming / HITL, FakeEmbedder yields deterministic vectors that plug into the real sqlite retrieval backend, MemoryCheckpointStore resumes a suspended run, and RecordingHook captures events.
"""

import pytest

from agentmaker import Agent
from agentmaker.tools import Tool, ToolParameter, ToolResponse
from agentmaker.testing import FakeEmbedder, MemoryCheckpointStore, RecordingHook, ScriptedLLM


class _EchoTool(Tool):
    def __init__(self):
        super().__init__("echo", "回显输入")

    def get_parameters(self):
        return [ToolParameter("text", "string", "要回显的内容")]

    def run(self, parameters):
        return ToolResponse.ok(f"echo:{parameters.get('text')}")


class _DangerTool(Tool):
    requires_confirmation = True

    def __init__(self):
        super().__init__("danger", "高风险删除")

    def get_parameters(self):
        return [ToolParameter("x", "string", "目标")]

    def run(self, parameters):
        return ToolResponse.ok(f"已删除 {parameters.get('x')}")


# ---------- ScriptedLLM ----------

def test_scripted_llm_duck_contract():
    """ScriptedLLM exposes LLMClient's duck-type contract attributes (tool-enabled Agent / window budget / trace all rely on them)."""
    llm = ScriptedLLM(["x"], context_window=1000)
    for attr in ("provider", "model", "supports_function_calling", "context_window", "chat", "stream"):
        assert hasattr(llm, attr)


def test_scripted_llm_plain_chat():
    """Plain chat: emits replies in scripted order so third parties test their own agent at zero cost."""
    agent = Agent("t", ScriptedLLM(["你好", "再见"]))
    assert agent.run("hi").final_output == "你好"
    assert agent.run("bye").final_output == "再见"


def test_scripted_llm_tool_loop():
    """Tool loop: the script emits a tool_call then a final answer -> the Agent drives tool execution and produces the final output."""
    llm = ScriptedLLM([ScriptedLLM.tool_call("echo", {"text": "hi"}), "回显完成"])
    agent = Agent("t", llm, tools=[_EchoTool()])
    assert agent.run("回显 hi").final_output == "回显完成"
    assert llm.calls == 2                                   # one call to request the tool + one to give the final answer


def test_scripted_llm_stream():
    """Streaming: yields the next scripted content in fragments."""
    agent = Agent("t", ScriptedLLM(["流式回复内容"]))
    pieces = list(agent.stream_run("hi"))
    assert "".join(pieces) == "流式回复内容"


def test_scripted_llm_exhausted_raises():
    """Calling past the end of the script -> AssertionError (prompts you to extend it, so tests never silently swallow unexpected behavior)."""
    agent = Agent("t", ScriptedLLM(["仅一条"]))
    agent.run("a")
    with pytest.raises(AssertionError, match="script exhausted"):
        agent.run("b")


def test_scripted_llm_stream_calls_on_stats():
    """The stream's on_stats finalizer returns StreamStats (matching the real adapter contract: harness.astream_llm relies on it to record usage)."""
    import asyncio
    from agentmaker.core.llm_response import LLMResponse, StreamStats
    llm = ScriptedLLM([LLMResponse(content="你好世界", model="test", finish_reason="stop",
                                   usage={"total_tokens": 5})])
    got = {}

    async def _drive():
        async for _piece in llm.stream([{"role": "user", "content": "hi"}], on_stats=lambda s: got.update(s=s)):
            pass
    asyncio.run(_drive())
    assert isinstance(got["s"], StreamStats)
    assert got["s"].usage == {"total_tokens": 5} and got["s"].finish_reason == "stop"
    assert llm.last_stream_stats is got["s"]                 # also written to the convenience attribute


def test_scripted_llm_stream_empty_content_yields_no_chunk():
    """Empty content yields no chunk (the old `or [0]` squeezed out an "", inconsistent with the real adapter's `if piece` semantics)."""
    import asyncio
    llm = ScriptedLLM([""])

    async def _drive():
        return [p async for p in llm.stream([{"role": "user", "content": "hi"}])]
    assert asyncio.run(_drive()) == []                       # yields nothing (not [""])


def test_scripted_llm_supports_fc_false_gates_tools():
    """supports_function_calling=False + tools -> Agent fails loud at construction (the capability flag is enforced)."""
    with pytest.raises(ValueError, match="function calling"):
        Agent("t", ScriptedLLM([], supports_function_calling=False), tools=[_EchoTool()])


# ---------- FakeEmbedder ----------

def test_fake_embedder_deterministic_and_discriminating():
    """Same text -> same vector, different text -> different vector; dim / model_id present (enough to plug into the real retrieval backend)."""
    emb = FakeEmbedder(dim=8)
    assert emb.dim == 8 and emb.model_id == "fake-embedder-8"
    [v1], [v2] = emb.embed(["上海"]), emb.embed(["上海"])
    assert v1 == v2 and len(v1) == 8                        # deterministic
    assert emb.embed(["北京"])[0] != v1                     # discriminates different text


def test_fake_embedder_in_real_retriever():
    """FakeEmbedder plugs into the real sqlite retrieval backend: write two entries, search for one's topic and hit it (offline)."""
    pytest.importorskip("sqlite_vec")
    from agentmaker.retrieval import Scope
    from agentmaker.retrieval.backends import build_sqlite_hybrid
    r = build_sqlite_hybrid(FakeEmbedder(dim=16))
    sc = Scope(base="t")
    r.add(["a", "b"], ["猫喜欢吃鱼", "今天天气晴"], scope=sc)
    hits = r.search("猫喜欢吃鱼", top_k=1, scope=sc)        # identical text -> identical vector -> stable hit
    assert hits and hits[0].id == "a"


# ---------- MemoryCheckpointStore (HITL) ----------

def test_memory_checkpoint_hitl_roundtrip():
    """High-risk tool -> run suspends, resume(True) continues to completion; entirely in-process memory, nothing persisted to disk."""
    llm = ScriptedLLM([ScriptedLLM.tool_call("danger", {"x": "/tmp/a"}), "删完了"])
    agent = Agent("t", llm, tools=[_DangerTool()], checkpoint_store=MemoryCheckpointStore())
    r = agent.run("删 /tmp/a")
    assert r.interrupted and r.interrupt.pending.tool_name == "danger"
    assert agent.resume(True, scope=r.interrupt.scope).final_output == "删完了"


# ---------- RecordingHook ----------

def test_recording_hook_captures_events():
    """RecordingHook captures run lifecycle events (on_run_start / before_model / after_model / on_run_end)."""
    hook = RecordingHook()
    Agent("t", ScriptedLLM(["答"]), hooks=[hook]).run("问")
    names = [e[0] for e in hook.events]
    assert names[0] == "on_run_start" and "before_model" in names and "after_model" in names
    assert names[-1] == "on_run_end" and hook.events[-1][1] == "答"
