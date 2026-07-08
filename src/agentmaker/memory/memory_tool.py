"""agentmaker.memory.memory_tool: wraps memory as a tool so an Agent can actively remember / recall (agentic memory).

Built on agentmaker.tools.base.Tool. Once registered on an Agent, the Agent can decide on its own to
call the various memory operations mid-conversation.
"""

import asyncio
from typing import List, Optional

from ..prompts import DEFAULT_PROMPTS
from ..core.aio import run_sync
from ..tools.base import Tool, ToolParameter
from ..tools.response import ToolResponse
from .memory import Memory
from .smart_writer import SmartWriter


class MemoryTool(Tool):
    """A tool letting an Agent manage long-term memory: remember / recall / forget / summary / stats / consolidate."""

    # destructive actions: deleting low-scoring memories / whole-block rewrite, requiring human confirmation before firing (action-level, see needs_confirmation)
    _CONFIRM_ACTIONS = {"forget", "consolidate"}

    def __init__(self, memory: Memory, writer: Optional[SmartWriter] = None, *, top_k: int = 5,
                 confirm_writer_edits: bool = True, prompts=None):
        """Initialize the memory tool.

        Args:
            memory: The memory manager (provides search / forget / summary / stats / consolidate / add).
            writer: Optional smart writer; when provided, remember goes through SmartWriter (auto
                de-duplicate / rewrite), otherwise it calls add directly.
            top_k: Number of results recall returns.
            confirm_writer_edits: When a writer is attached, whether remember also passes through the
                confirmation gate (default True). SmartWriter may decide, per the LLM, to UPDATE / DELETE
                existing memories (soft-invalidate, rewrite), which modifies existing data, so by default
                it goes through the confirmation gate like forget / consolidate (fail-safe). Pure add (no
                writer) never gates. Pass False to disable in low-friction scenarios.
        """
        self.prompts = prompts or DEFAULT_PROMPTS
        super().__init__(name="memory", description=self.prompts.text("tool.desc.memory"))
        self.memory = memory
        self.writer = writer
        self.top_k = top_k
        self.confirm_writer_edits = confirm_writer_edits

    def get_parameters(self) -> List[ToolParameter]:
        """Declare parameters: one action + content / query depending on the operation."""
        return [
            ToolParameter("action", "string", self.prompts.text("tool.param.memory.action")),
            ToolParameter("content", "string", self.prompts.text("tool.param.memory.content"), required=False),
            ToolParameter("query", "string", self.prompts.text("tool.param.memory.query"), required=False),
        ]

    def needs_confirmation(self, parameters: dict) -> bool:
        """Data-deleting / data-modifying actions require human confirmation: forget / consolidate always do; remember also does when a writer is attached and confirm_writer_edits is set (SmartWriter may UPDATE / DELETE existing memories). recall / summary / stats and pure-add remember pass through."""
        action = (parameters.get("action") or "").strip()
        if action in self._CONFIRM_ACTIONS:
            return True
        if action == "remember" and self.writer is not None and self.confirm_writer_edits:
            return True
        return False

    def run(self, parameters: dict) -> ToolResponse:
        """Dispatch by action to the memory operations, returning a result for the LLM to read (errors as status="error")."""
        action = (parameters.get("action") or "").strip()

        if action == "remember":
            text = parameters.get("content", "")
            if not text:
                return ToolResponse.error(self.prompts.text("tool.msg.mem.need_content"))
            if self.writer is not None:
                recs = run_sync(self.writer.write(text))
                if not recs:
                    return ToolResponse.ok(self.prompts.text("tool.msg.mem.nothing_extracted"))
                return ToolResponse.ok(self._format_recs(recs))
            item = self.memory.add(text)
            return ToolResponse.ok(self.prompts.render("tool.msg.mem.remembered", content=item.content))

        if action == "recall":
            query = parameters.get("query", "")
            if not query:
                return ToolResponse.error(self.prompts.text("tool.msg.mem.need_query"))
            hits = self.memory.search(query, top_k=self.top_k)
            if not hits:
                return ToolResponse.ok(self.prompts.text("tool.msg.mem.no_recall"))
            return ToolResponse.ok(self.prompts.text("tool.msg.mem.found_prefix") + "\n"
                                   + "\n".join(f"- {h.content}" for h in hits))

        if action == "summary":
            return ToolResponse.ok(run_sync(self.memory.summary(parameters.get("query") or None)))

        if action == "stats":
            s = self.memory.stats()
            return ToolResponse.ok(self.prompts.render("tool.msg.mem.stats", total=s["total"], by_type=s["by_type"]), data=s)

        if action == "forget":
            ids = self.memory.forget()  # default: forget low-scoring items by importance
            return ToolResponse.ok(self.prompts.render("tool.msg.mem.forgotten", n=len(ids)))

        if action == "consolidate":
            r = run_sync(self.memory.consolidate())
            return ToolResponse.ok(self.prompts.render("tool.msg.mem.consolidated", before=r["before"], after=r["after"]))

        return ToolResponse.error(self.prompts.render("tool.msg.mem.unknown_action", action=action))

    def _format_recs(self, recs: list) -> str:
        """Format SmartWriter.write's result ([{fact, op}, ...]) into a "remembered" list (shared by run / arun, to avoid copy drift)."""
        items = "\n".join(self.prompts.render("tool.msg.mem.remembered_item", fact=r["fact"], op=r["op"]) for r in recs)
        return self.prompts.text("tool.msg.mem.remembered_list") + "\n" + items

    async def arun(self, parameters: dict) -> ToolResponse:
        """Native async variant of run: LLM-calling actions (remember/summary/consolidate) go async via await; non-LLM ones (recall/stats/forget) reuse the sync logic directly (no network wait, no need for async)."""
        action = (parameters.get("action") or "").strip()
        if action == "remember":
            text = parameters.get("content", "")  # consistent with run / get_parameters, uniformly using content
            if not text:
                return ToolResponse.error(self.prompts.text("tool.msg.mem.need_content"))
            if self.writer is not None:
                recs = await self.writer.write(text)
                if not recs:
                    return ToolResponse.ok(self.prompts.text("tool.msg.mem.nothing_extracted"))
                return ToolResponse.ok(self._format_recs(recs))
            item = await self.memory.aadd(text)   # add involves a store write + embedding network call, so go through memory's own async entry
            return ToolResponse.ok(self.prompts.render("tool.msg.mem.remembered", content=item.content))
        if action == "summary":
            return ToolResponse.ok(await self.memory.summary(parameters.get("query") or None))
        if action == "consolidate":
            r = await self.memory.consolidate()
            return ToolResponse.ok(self.prompts.render("tool.msg.mem.consolidated", before=r["before"], after=r["after"]))
        return await asyncio.to_thread(self.run, parameters)  # recall/stats/forget/unknown: no LLM involved, run the sync logic in a thread pool so it does not block the event loop


