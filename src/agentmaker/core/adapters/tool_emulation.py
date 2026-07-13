"""agentmaker.core.adapters.tool_emulation: tool-call translation shim for models without function calling (bidirectional translation at the adapter layer).

Wraps any underlying adapter: on a chat with tools, it writes the tool catalog into system, flattens the
tool trace into plain text, and does not send the native tools parameter; then it parses "which tool it
wants to call" out of the model's plain-text reply and translates that back into unified tool_calls. This
lets models without native function calling (a few pure-reasoning / older / quantized local models) use
tools with zero changes to the agent loop / harness: to the upper layers it still looks like standard
tool_calls. Enabled only with `LLMClient(emulate_tools=True)` (presenting supports_function_calling=True
to the Agent).

Boundaries: one round parses only a single tool call (no parallel); the adapter's direct streaming path
does no tool emulation, and Agent fails fast when asked for a streaming tool loop; with tools it does not also send a
structured-output output_schema (the two are different paths). This is only "text emulation", inherently
less reliable than native function calling: if native function calling is available, use that instead.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Optional

from ..exceptions import LLMConfigError, LLMError
from ..llm_response import LLMResponse, StreamStats
from ..multimodal import messages_have_images
from ...prompts import DEFAULT_PROMPTS


class ToolEmulationAdapter:
    """A translation shim that emulates tool calls for models without native function calling (wraps one underlying adapter, duck-typing chat/stream)."""

    supports_streaming_tools = False

    def __init__(self, delegate, *, prompts=None):
        """
        Args:
            delegate: The wrapped underlying adapter (any wire protocol, e.g. OpenAIAdapter); it is what
                actually sends the request.
            prompts: Optional prompt registry (PromptRegistry); the model-visible tool catalog / call-format
                instructions / trace-flattening copy are taken from it, defaulting to DEFAULT_PROMPTS when
                not passed (a whole language pack can be swapped in).
        """
        self._delegate = delegate
        self.prompts = prompts or DEFAULT_PROMPTS
        self._call_seq = 0   # Emulated calls have no native id; auto-increment a unique call_id (emu_N).

    async def chat(self, messages, *, temperature=None, max_tokens=None, output_schema=None, **kwargs) -> LLMResponse:
        """With tools, run translation (flatten the trace + inject the catalog + do not send tools + parse the call from text); without tools, pass straight through to the underlying adapter."""
        tools = kwargs.pop("tools", None)
        if not tools:
            return await self._delegate.chat(messages, temperature=temperature, max_tokens=max_tokens,
                                             output_schema=output_schema, **kwargs)
        prepared = self._flatten(messages, tools)   # Flatten the tool trace + inject the tool catalog into system.
        resp = await self._delegate.chat(prepared, temperature=temperature, max_tokens=max_tokens, **kwargs)  # Do not send tools / output_schema.
        return self._parse(resp)

    async def stream(self, messages, *, temperature=None, max_tokens=None, on_stats=None,
                     **kwargs) -> AsyncIterator[str]:
        """Stream plain text without tool emulation; Agent rejects tool loops via the capability flag."""
        if kwargs.pop("tools", None):
            raise LLMConfigError(
                "Text tool emulation does not support streaming tool calls; use chat() instead.")
        async for piece in self._delegate.stream(messages, temperature=temperature, max_tokens=max_tokens,
                                                 on_stats=on_stats, **kwargs):
            yield piece

    @property
    def last_stream_stats(self) -> Optional[StreamStats]:
        """Pass through the underlying adapter's most recent streaming stats."""
        return self._delegate.last_stream_stats

    async def aclose(self) -> None:
        """Close the wrapped adapter's client for the current event loop."""
        await self._delegate.aclose()

    # Outbound: flatten the tool trace into plain text + inject the tool catalog and call-format instructions into system.

    def _flatten(self, messages, tools) -> list:
        """Flatten the OpenAI-shaped tool trace (assistant.tool_calls / role:"tool") into plain-text messages a model without function calling can read,
        and merge the tool catalog + call-format instructions into system (insert a new one if there is no system).

        Raises:
            LLMError: When the messages carry image content parts. Text emulation flattens
                everything into a plain-text prompt, and an image cannot survive that: fail loud
                instead of silently dropping it.
        """
        if messages_have_images(messages):
            raise LLMError("emulate_tools is text-only: image content parts cannot be flattened into a "
                           "plain-text prompt. Use a natively multimodal, function-calling model for image inputs.")
        catalog = "\n".join(
            self.prompts.render("emulation.catalog_item",
                                name=(t.get("function") or {}).get("name", ""),
                                description=(t.get("function") or {}).get("description", ""),
                                schema=json.dumps((t.get("function") or {}).get("parameters", {}), ensure_ascii=False))
            for t in tools)
        instruction = self.prompts.render("emulation.instruction", catalog=catalog)
        id_to_name: dict = {}
        out: list = []
        injected = False
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                out.append({"role": "system", "content": f"{content}\n\n{instruction}" if not injected else content})
                injected = True
            elif role == "assistant" and m.get("tool_calls"):
                parts = [content] if content else []
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    id_to_name[tc["id"]] = fn["name"]
                    parts.append(self.prompts.render("emulation.assistant_call",
                                                     name=fn["name"], arguments=fn["arguments"]))
                out.append({"role": "assistant", "content": "\n".join(parts)})
            elif role == "tool":
                out.append({"role": "user", "content": self.prompts.render(
                    "emulation.tool_result", name=id_to_name.get(m.get("tool_call_id"), ""), content=content)})
            else:
                out.append({"role": role, "content": content})
        if not injected:
            out.insert(0, {"role": "system", "content": instruction})
        return out

    # Inbound: parse the tool-call directive out of the plain-text reply -> translate back into unified tool_calls.

    def _parse(self, resp) -> LLMResponse:
        """Look for a tool-call directive in the model's reply text: if found, translate it into tool_calls (remove the directive JSON, keep only the thinking text); if not, it stays a plain text answer."""
        text = resp.content or ""
        directive = self._extract(text)
        if directive is None:
            return resp
        name, arguments, start, end = directive
        self._call_seq += 1
        call = {"id": f"emu_{self._call_seq}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
        content = (text[:start] + text[end:]).strip()   # Cut out the directive JSON precisely by the indices located by _extract, keeping the rest of the thinking text (not replace, to avoid stripping the wrong spot when the same substring appears earlier).
        return replace(resp, content=content, tool_calls=[call])

    def _extract(self, text: str):
        """Find the first valid JSON object containing a "tool" key (a tool-call directive) in the text; return (name, arguments dict, start, end) or None.

        Uses json.JSONDecoder().raw_decode to try parsing a complete JSON value starting from each `{`: it is a
        real JSON parser and naturally handles strings / escapes / nested braces correctly (even a { or } inside
        a parameter value will not mismatch), more robust than hand-written brace balancing.
        If some `{` does not parse into valid JSON, shift right by one and retry; if it parses but is not a tool
        directive, skip it and continue searching from its end.
        Returns the (start, end) indices for _parse to precisely excise the directive (text[start:end] is the
        directive JSON substring).
        """
        decoder = json.JSONDecoder()
        i, n = 0, len(text)
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            try:
                obj, end = decoder.raw_decode(text, i)
            except (json.JSONDecodeError, ValueError):
                i += 1
                continue
            if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
                arguments = obj.get("arguments")
                return obj["tool"], (arguments if isinstance(arguments, dict) else {}), i, end
            i = end   # Parsed valid JSON but not a tool directive -> skip the whole segment and continue.
