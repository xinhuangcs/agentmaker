"""agentmaker.rag.rag_tool: wrap RAG as a tool so an Agent can manage a knowledge base and answer questions (agentic RAG).

Built on agentmaker.tools.base.Tool. Actions: add_text / add_document / search / ask / stats.
Among them add_document reads a file from disk (high risk), so the needs_confirmation override
returns True only for it: before execution the unified confirmation gate (ToolRegistry / harness)
confirms it, the same mechanism as MemoryTool's forget / consolidate (action-level confirmation).
"""

import asyncio
from typing import List

from ..prompts import DEFAULT_PROMPTS
from ..core.aio import run_sync
from ..tools.base import Tool, ToolParameter
from ..tools.response import ToolResponse
from .ingest import IngestionPipeline
from .retriever import RagRetriever
from .types import AskResult


class RAGTool(Tool):
    """Tool that lets an Agent manage a knowledge base and answer questions: add_text / add_document / search / ask / stats."""

    external_content = True   # search / ask return the raw text of the knowledge base (content from an external source): wrap it in anti-injection delimiting guards before feeding it back to the model.
    # High-risk action: reads a file from disk, requires human confirmation before execution (action-level, see needs_confirmation).
    _CONFIRM_ACTIONS = {"add_document"}

    def __init__(self, pipeline: IngestionPipeline, rag_retriever: RagRetriever, *, top_k: int = 5,
                 filter_fields: tuple = (), prompts=None):
        """
        Args:
            pipeline: Ingestion pipeline (add_text / add_document / stats).
            rag_retriever: Retrieval and question answering (search / ask).
            top_k: How many chunks search / ask retrieve.
            filter_fields: Optional; which metadata filter fields to expose to the model (e.g.
                ("doc_id", "tag"), contract in retrieval/types.py). These fields become optional
                parameters of search / ask, and the model fills their values via function calling,
                which is equivalent to self-query (natural language -> structured filter). The
                retrieval backend must have declared filterable columns of the same name at index
                build time (build_sqlite_hybrid's metadata_columns=). Not exposed by default.
        """
        self.prompts = prompts or DEFAULT_PROMPTS
        super().__init__(name="rag", description=self.prompts.text("tool.desc.rag"))
        self.pipeline = pipeline
        self.rag_retriever = rag_retriever
        self.top_k = top_k
        self.filter_fields = tuple(filter_fields)

    def needs_confirmation(self, parameters: dict) -> bool:
        """Only add_document (reads disk, a red-line action) needs human confirmation; add_text / search / ask / stats do not touch disk and are allowed through.

        The confirmation gate (ToolRegistry.execute_tool and HITL's harness.exec_tool) always
        reads this: without a confirm passed, it defaults to reject, so the Agent cannot read disk
        on its own. The safe default is guaranteed by the unified confirmation gate, and this tool
        need not hold its own callback.
        """
        return (parameters.get("action") or "").strip() in self._CONFIRM_ACTIONS

    def get_parameters(self) -> List[ToolParameter]:
        """Declare parameters; each filter field in filter_fields becomes an optional parameter of search / ask (model fills it = self-query)."""
        return [
            ToolParameter("action", "string", self.prompts.text("tool.param.rag.action")),
            ToolParameter("text", "string", self.prompts.text("tool.param.rag.text"), required=False),
            ToolParameter("format", "string", self.prompts.text("tool.param.rag.format"), required=False),
            ToolParameter("file_path", "string", self.prompts.text("tool.param.rag.file_path"), required=False),
            ToolParameter("query", "string", self.prompts.text("tool.param.rag.query"), required=False),
            *[ToolParameter(f, "string", self.prompts.render("tool.param.rag.filter", field=f), required=False)
              for f in self.filter_fields],
        ]

    def _filters(self, parameters: dict):
        """Collect the filter-field values filled in by the model from the parameters and assemble them into a MetadataFilter list; return None if no field was filled."""
        from ..retrieval.types import MetadataFilter
        fs = [MetadataFilter(f, str(v)) for f in self.filter_fields if (v := parameters.get(f))]
        return fs or None

    def run(self, parameters: dict) -> ToolResponse:
        """Dispatch by action and return a result for the LLM to read (errors have status="error")."""
        action = (parameters.get("action") or "").strip()

        if action == "add_text":
            text = parameters.get("text", "")
            if not text:
                return ToolResponse.error(self.prompts.text("rag.msg.need_text"))
            fmt = parameters.get("format") or "txt"  # Pass md to split by heading, defaults to plain text.
            res = self.pipeline.ingest_text(text, source="agent_input", fmt=fmt)  # Tag the source for retrieval traceability.
            return ToolResponse.ok(self.prompts.render("rag.msg.ingested", chunks=res.chunks, doc_id=res.doc_id[:8]))

        if action == "add_document":
            path = parameters.get("file_path", "")
            if not path:
                return ToolResponse.error(self.prompts.text("rag.msg.need_file"))
            # Reading disk is high-risk: whether to allow it is already decided by the unified confirmation gate per needs_confirmation, blocked before run (a red-line action).
            res = self.pipeline.ingest_file(path)
            return ToolResponse.ok(self.prompts.render("rag.msg.imported", path=path, chunks=res.chunks))

        if action == "search":
            query = parameters.get("query", "")
            if not query:
                return ToolResponse.error(self.prompts.render("rag.msg.need_query", action=action))
            hits = self.rag_retriever.retrieve(query, top_k=self.top_k, filters=self._filters(parameters))
            if not hits:
                return ToolResponse.ok(self.prompts.text("rag.msg.search_empty"))
            body = "\n".join(
                f"- {h.content}" + (self.prompts.render("rag.msg.source_suffix", path=h.metadata.get("heading_path"))
                                    if h.metadata.get("heading_path") else "") for h in hits)
            return ToolResponse.ok(self.prompts.text("rag.msg.found_prefix") + "\n" + body)

        if action == "ask":
            query = parameters.get("query", "")
            if not query:
                return ToolResponse.error(self.prompts.render("rag.msg.need_query", action=action))
            return self._format_ask(run_sync(self.rag_retriever.ask(query, top_k=self.top_k, filters=self._filters(parameters))))

        if action == "stats":
            s = self.pipeline.stats()
            return ToolResponse.ok(self.prompts.render("rag.msg.stats", documents=s["documents"], chunks=s["chunks"]), data=s)

        return ToolResponse.error(self.prompts.render("rag.msg.unknown_action", action=action))

    async def arun(self, parameters: dict) -> ToolResponse:
        """Native async version of run: ask (which calls the LLM) awaits the async ask directly; other actions run the synchronous logic via to_thread without blocking the event loop.

        Note: add_text/add_document/search involve embedding (network) but are not multi-turn LLM
        calls, so they still go through the synchronous run, just dispatched to a thread pool (same
        as the base Tool.arun default), avoiding a synchronous embedding request on the event loop.
        """
        action = (parameters.get("action") or "").strip()
        if action == "ask":
            query = parameters.get("query", "")
            if not query:
                return ToolResponse.error(self.prompts.render("rag.msg.need_query", action=action))
            return self._format_ask(await self.rag_retriever.ask(query, top_k=self.top_k,
                                                                  filters=self._filters(parameters)))
        return await asyncio.to_thread(self.run, parameters)  # add_text/add_document/search/stats: run the synchronous logic in the thread pool.

    def _format_ask(self, res: "AskResult") -> ToolResponse:
        """Assemble ask's AskResult(answer, sources) into a ToolResponse with a "sources" footer (shared by run / arun to avoid two copies drifting apart)."""
        sep = self.prompts.text("rag.msg.source_sep")
        src = sep.join(f"[{s.n}]{s.heading_path or s.content[:20]}" for s in res.sources)
        return ToolResponse.ok(res.answer + (self.prompts.render("rag.msg.source_label", src=src) if src else ""))


