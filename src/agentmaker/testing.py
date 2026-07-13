"""agentmaker.testing: official test utilities for testing agents built on this framework, with no cost and no network.

When third parties write agents, the deterministic doubles here swap the LLM / embedding / checkpoint / hook
for local implementations so hermetic unit tests can run (mirroring Pydantic AI's TestModel). Not part of the
top-level `agentmaker.__all__`; import on demand via `from agentmaker.testing import ScriptedLLM`.

    from agentmaker import Agent
    from agentmaker.testing import ScriptedLLM

    agent = Agent("test", ScriptedLLM(["hello"]))
    assert agent.run("hi").final_output == "hello"

    # Tool call: the script first emits a "call calculator" response, then the final answer.
    llm = ScriptedLLM([ScriptedLLM.tool_call("calculator", {"expression": "1+1"}), "equals 2"])
"""

import hashlib
import json
import math
from typing import List, Optional

from .core.llm_response import LLMResponse, StreamStats
from .retrieval.base import Embedder
from .runtime.execution.checkpoint import CheckpointStore
from .runtime.hooks import Hook


class ScriptedLLM:
    """Test LLM that emits preset responses in call order (duck-typed; does NOT inherit LLMClient, avoiding key validation / network).

    Each script element is a `str` (a plain-text reply) or a ready-made `LLMResponse` (with tool_calls / usage /
    etc.). `chat` consumes them in order; calling again after the script is exhausted raises `AssertionError`
    (noting how many entries are missing). `stream` slices the next response's content. Use
    `ScriptedLLM.tool_call(...)` to conveniently build a "request to call a tool" response. The duck-typed
    contract mirrors LLMClient: provider / model / supports_function_calling / context_window / chat / stream.
    """

    def __init__(self, script: Optional[List[str | LLMResponse]] = None, *, model: str = "test", provider: str = "test",
                 supports_function_calling: bool = True, context_window: Optional[int] = None):
        """
        Args:
            script: List of response entries, each a str (text reply) or an LLMResponse (for precise control over tool_calls / usage / etc.).
            supports_function_calling: Model capability flag (a tool-enabled Agent validates against it at construction time); pass False to test the no-function-calling path.
            context_window: Context window (None means unknown, so no window-budget reduction is triggered); pass a concrete value to test reduction / window budget.
        """
        self._script = list(script or [])
        self.model = model
        self.provider = provider
        self.supports_function_calling = supports_function_calling
        self.context_window = context_window
        self.calls = 0          # Number of script entries already consumed by chat / stream (for asserting call count).
        self.last_stream_stats = None   # Mirrors real adapters: stats from the most recent stream() call.

    @staticmethod
    def tool_call(name: str, arguments: Optional[dict] = None, *, call_id: str = "call_1",
                  content: str = "") -> LLMResponse:
        """Build an LLMResponse representing "the model requests calling tool name(arguments)" (so you need not hand-craft the OpenAI tool_calls structure)."""
        return LLMResponse(content=content, model="test", tool_calls=[{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments or {}, ensure_ascii=False)}}])

    def _next(self) -> LLMResponse:
        if self.calls >= len(self._script):
            raise AssertionError(
                f"ScriptedLLM script exhausted (this is call #{self.calls + 1}, but the script has only {len(self._script)} entries). "
                "Add more entries to script (a string reply or an LLMResponse).")
        item = self._script[self.calls]
        self.calls += 1
        return item if isinstance(item, LLMResponse) else LLMResponse(content=str(item), model=self.model)

    async def chat(self, messages, *, tools=None, **kwargs) -> LLMResponse:
        """Return the next response from the script (ignoring messages / tools, since the script determines test behavior)."""
        return self._next()

    async def stream(self, messages, *, tools=None, on_stats=None, **kwargs):
        """Yield the next response's content in slices (roughly 8 characters each); at the end, invoke on_stats and write last_stream_stats.

        Mirrors the real adapter contract: empty content yields NO empty chunk (an empty range simply skips the
        loop); on completion, regardless of whether there was content, assemble a StreamStats from the response's
        model / finish_reason / usage and hand it back, since harness.astream_llm relies on on_stats to collect
        usage for accounting. With tools passed, the full LLMResponse is yielded as the terminal item after the
        text slices (same contract as the real adapters), making the streaming tool loop hermetically testable.
        """
        resp = self._next()
        text = resp.content
        for i in range(0, len(text), 8):                 # Empty string means an empty range means no empty chunk is yielded (a plain `or [0]` would squeeze out a "").
            yield text[i:i + 8]
        # resp is guaranteed to be an LLMResponse (via _next); finish_reason/usage are declared fields taken directly; model defaults to empty string, hence the `or` fallback.
        stats = StreamStats(model=resp.model or self.model, finish_reason=resp.finish_reason,
                            usage=resp.usage, latency_ms=0)
        self.last_stream_stats = stats
        if on_stats:
            on_stats(stats)
        if tools:
            yield resp


class FakeEmbedder(Embedder):
    """Deterministic fake embedder (no network): same text yields the same vector, different text yields different vectors (sha256-derived + L2-normalized), so retrieval can genuinely distinguish them."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Turn each text into a deterministic vector."""
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> List[float]:
        raw = hashlib.sha256(text.encode("utf-8")).digest()           # 32 bytes, deterministic.
        vals = [raw[i % len(raw)] - 128 for i in range(self._dim)]    # Fill all dim dimensions, centered into [-128, 127].
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]                               # L2-normalize, so cosine similarity is meaningful.

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> Optional[str]:
        return f"fake-embedder-{self._dim}"


class MemoryCheckpointStore(CheckpointStore):
    """In-process in-memory checkpoint store (one ExecutionState JSON per scope): test HITL suspend / resume / crash recovery, without persisting to disk.

    Implementing synchronous save / load / clear is enough: the base class's asave / aload / aclear wrap them
    with to_thread by default, so the async path comes for free.
    """

    def __init__(self):
        self._d: dict = {}

    def save(self, state_json: str, *, scope=None) -> None:
        self._d[scope] = state_json

    def load(self, *, scope=None) -> Optional[str]:
        return self._d.get(scope)

    def clear(self, *, scope=None) -> None:
        self._d.pop(scope, None)


class RecordingHook(Hook):
    """Record every triggered lifecycle event into `self.events` (`[(event_name, key_param), ...]`), for asserting whether hook dispatch happened as expected."""

    def __init__(self):
        self.events: list = []

    def on_run_start(self, input_text, *, scope=None): self.events.append(("on_run_start", input_text))
    def before_model(self, messages): self.events.append(("before_model", len(messages)))
    def after_model(self, response): self.events.append(("after_model", getattr(response, "content", "")))
    def before_tool(self, name, parameters): self.events.append(("before_tool", name))
    def after_tool(self, name, parameters, result): self.events.append(("after_tool", name))
    def on_guardrail_trip(self, stage, message): self.events.append(("on_guardrail_trip", stage))
    def on_interrupt(self, pendings, *, scope=None): self.events.append(("on_interrupt", pendings[0].tool_name if pendings else None))
    def on_error(self, error): self.events.append(("on_error", type(error).__name__))
    def on_run_end(self, output, *, scope=None): self.events.append(("on_run_end", output))


__all__ = ["ScriptedLLM", "FakeEmbedder", "MemoryCheckpointStore", "RecordingHook"]
