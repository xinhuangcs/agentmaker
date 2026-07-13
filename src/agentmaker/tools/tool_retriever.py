"""agentmaker.tools.tool_retriever: Tool-RAG, semantic retrieval over a batch of tools (tier 3 of progressive disclosure).

Once there are many tools, stuffing all their full schemas into the prompt is both expensive
and degrades accuracy (empirically fine at around 50 tools, collapsing to 0-20% at 700+).
Tool-RAG feeds each tool's name+description into the shared retrieval base, retrieves the top-k
most relevant tool names for a user query, and pairs with `ToolRegistry`'s `names` subset
rendering to expand only those few: the same base as memory / rag, isolated by `scope=tools`.

The reliability trio (aligned with Anthropic's official practice of keeping common tools
always-on):
    - always_include: tools in this list bypass retrieval and are always in the subset (things like clarify / ask-for-help / ToolSearchTool that must not be squeezed out by top-k).
    - on_empty: the fallback on zero hits (defaults to falling back to the full catalog, never giving the model 0 tools, since tools=[] makes some providers return 400 outright).
    - selector: the truncation-strategy seam (defaults to a fixed top-k; inject a callback yourself for a score threshold / knee-point algorithm; agentmaker does not bet on a specific algorithm).

Also includes `ToolSearchTool` (tool search made into a tool itself): preselection handles the
first round, search handles mid-course; the model can re-search on demand mid-execution, and the
function-calling path dynamically merges the discovered tools into the usable set (see the class
docstring and Agent._expand_tools).

agentmaker provides the mechanism; whether to use it, and how large top_k is, is decided by the
app based on tool scale (few tools: use them all directly; only retrieve once there are many; the
knobs can be assembled in one line via AgentmakerConfig.tool_retrieval + from_config). This
module is not exported at the `agentmaker.tools` top level: basic tool imports stay free of a
dependency on the retrieval stack; import explicitly when needed via
`from agentmaker.tools.tool_retriever import ToolRetriever, ToolSearchTool`.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

from ..prompts import DEFAULT_PROMPTS
from ..retrieval.index_sync import IndexSync, SyncIndexSync
from ..retrieval.scope import Scope
from .base import Tool, ToolParameter
from .response import ToolResponse

if TYPE_CHECKING:
    from ..retrieval.hybrid import HybridRetriever
    from .registry import ToolRegistry

_ON_EMPTY = ("all", "always_include", "none")


@dataclass(frozen=True)
class ToolRetrievalConfig:
    """Tool-RAG knobs (a sub-config of AgentmakerConfig).

    Fields:
        top_k: how many tools to retrieve each round.
        always_include: tool names that bypass retrieval and are always in the subset (such as clarify / ask-for-help / tool_search).
        on_empty: the zero-hit fallback: "all" (default, fall back to the full catalog) / "always_include" (give only the resident list) / "none" (empty set, left to the harness to handle as "no tools").
    """
    top_k: int = 8
    always_include: tuple = ()
    on_empty: str = "all"

    def __post_init__(self):
        """Normalize: coerce always_include into a tuple (JSON / from_dict yields a list; a uniform shape keeps to_dict/from_dict round-trip equal)."""
        object.__setattr__(self, "always_include", tuple(self.always_include))

    def validate(self) -> None:
        """Range / enum validation."""
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.on_empty not in _ON_EMPTY:
            raise ValueError(f"on_empty must be one of {_ON_EMPTY}, got {self.on_empty!r}")


# A minimal source-of-truth-item adapter outside of Chunk: reconcile expects elements with .id / .content
class _ToolItem:
    """The form a tool takes in the index's source-of-truth snapshot (id=tool name, content=retrievable text)."""

    __slots__ = ("id", "content")

    def __init__(self, id: str, content: str):
        self.id = id
        self.content = content


class ToolRetriever:
    """Semantic retrieval over the tools in a registry: `index()` loads the base, `retrieve()` retrieves the top-k tool names for a query."""

    def __init__(self, registry: "ToolRegistry", retriever: "HybridRetriever", *,
                 scope: Optional[Scope] = None, top_k: int = 8, always_include: tuple = (),
                 on_empty: str = "all", selector: Optional[Callable] = None,
                 index_sync: Optional[IndexSync] = None):
        """
        Args:
            registry: the tool registry (retrieval-hit tool names are resolved against it to fetch schemas).
            retriever: the shared hybrid retrieval base (the same one as memory / rag, isolated by scope).
            scope: where the tool data belongs, default Scope(base="tools") (does not leak across scopes with memory / rag).
            top_k: how many tools to retrieve each round (the default for retrieve / schema_for / description_for, overridable per call).
            always_include: tool names that bypass retrieval and are always in the subset (placed first in stable order); names not in the registry are ignored.
            on_empty: the fallback strategy on zero retrieval hits (see ToolRetrievalConfig.on_empty); default "all", preferring to fall back to all rather than give the model 0 tools (tools=[] returns 400 for some providers, and behavior is inconsistent across them).
            selector: an optional truncation-strategy callback `(query, hits) -> List[str]` (hits is a list of RetrievalResult, in descending relevance); default None = a fixed top-k. Inject your own for a score threshold / knee-point truncation; agentmaker does not build in a specific algorithm.
            index_sync: an optional index-sync seam; SyncIndexSync(retriever) if not passed. The registry is the source of truth: index() reconciles (removes stale indexes of removed tools + force-reloads the current tools), so calling index() once after a restart rebuilds it.
        """
        cfg = ToolRetrievalConfig(top_k=top_k, always_include=tuple(always_include), on_empty=on_empty)
        cfg.validate()
        self.registry = registry
        self.retriever = retriever
        self.scope = scope or Scope(base="tools")
        self.top_k = cfg.top_k
        self.always_include = cfg.always_include
        self.on_empty = cfg.on_empty
        self.selector = selector
        self._sync = index_sync if index_sync is not None else SyncIndexSync(retriever)

    @classmethod
    def from_config(cls, config, registry: "ToolRegistry", retriever: "HybridRetriever", *,
                    scope: Optional[Scope] = None, selector: Optional[Callable] = None,
                    index_sync: Optional[IndexSync] = None) -> "ToolRetriever":
        """Assemble from an AgentmakerConfig (reads config.tool_retrieval's top_k / always_include / on_empty)."""
        trc = config.tool_retrieval
        trc.validate()
        return cls(registry, retriever, scope=scope, top_k=trc.top_k,
                   always_include=trc.always_include, on_empty=trc.on_empty,
                   selector=selector, index_sync=index_sync)

    def index(self) -> None:
        """Load the name+description (including parameter names) of all current registry tools into the retrieval base; idempotent to call repeatedly.

        Via IndexSync.reconcile with the registry as source of truth: stale indexes of removed
        tools are cleaned up as orphans (so they do not take up retrieval slots), and current tools
        are force-reloaded; after a restart (when bookkeeping is in-process), calling this method
        once rebuilds fully from the registry.
        """
        items = [_ToolItem(t.name, self._doc(t)) for t in self.registry.list_tools()]
        self._sync.reconcile(items, scope=self.scope)

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> List[str]:
        """Retrieve the most relevant tool names for a query (descending relevance) + the resident always_include (placed first in stable order).

        Returns only those still present in the registry; on zero retrieval hits, fall back per on_empty (default the full catalog).
        """
        k = top_k if top_k is not None else self.top_k
        hits = self.retriever.search(query, top_k=k, scope=self.scope)
        names = self.selector(query, hits) if self.selector is not None else [h.id for h in hits]
        names = [n for n in names if self.registry.get(n) is not None]
        if not names:                                    # zero-hit fallback: never let the model get 0 tools
            if self.on_empty == "all":
                names = [t.name for t in self.registry.list_tools()]
            elif self.on_empty == "always_include":
                names = []
            else:                                        # "none": empty set (harness handles it as "no tools")
                return []
        always = [n for n in self.always_include if self.registry.get(n) is not None]
        seen = set(always)
        return always + [n for n in names if n not in seen]   # resident first (stable order), retrieval hits follow by relevance

    def schema_for(self, query: str, *, top_k: Optional[int] = None) -> List[dict]:
        """Tool-RAG in one step: retrieve the top-k tools (including resident / fallback) and give this batch's OpenAI function-calling schema."""
        return self.registry.to_openai_schema(names=self.retrieve(query, top_k=top_k))

    def description_for(self, query: str, *, top_k: Optional[int] = None) -> str:
        """Same as schema_for but rendered as a textual tool description (for offline inspection / an app assembling its own prompt)."""
        return self.registry.get_tools_description(names=self.retrieve(query, top_k=top_k))

    @staticmethod
    def _doc(tool) -> str:
        """The tool's retrievable text: name + description + parameter names (parameter names are a retrieval signal too)."""
        params = " ".join(p.name for p in tool.get_parameters())
        return f"{tool.name}: {tool.description} {params}".strip()


# Tool search as a tool (retrieval as a tool)
# One-shot preselection from the initial query has a systematic blind spot: in a multi-step task
# "which tool to use in step 2" depends on the output of step 1. The fix is to hand the search
# capability to the model (Anthropic's Tool Search Tool is already an official API feature;
# langgraph-bigtool is a same-shape reference implementation). agentmaker does a client-side,
# cross-provider implementation: the model calls tool_search(query) -> catalog text (model-readable)
# + ToolResponse.data["discovered"] (a list of tool names); the unified-loop Agent, seeing
# discovered, merges those tools' schemas into this round's usable set (aexec_tool already executes
# the whole registry by name, zero changes). This tool itself should be an always_include resident
# item (it must not be squeezed out by top-k).


class ToolSearchTool(Tool):
    """Let the model search available tools on demand mid-execution: returns catalog text + data.discovered tool names (the function-calling loop expands the usable set from this)."""

    def __init__(self, tool_retriever: "ToolRetriever", *, top_k: int = 5, prompts=None):
        """
        Args:
            tool_retriever: the Tool-RAG retriever (retrieves relevant tools for a query).
            top_k: how many tools each search returns.
            prompts: an optional prompt registry; DEFAULT_PROMPTS if not passed.
        """
        self.prompts = prompts or DEFAULT_PROMPTS
        super().__init__(name="tool_search", description=self.prompts.text("tool.desc.tool_search"))
        self.tool_retriever = tool_retriever
        self.top_k = top_k

    def get_parameters(self) -> List[ToolParameter]:
        """Declare parameters: just a description of what capability to look for."""
        return [ToolParameter("query", "string", self.prompts.text("tool.param.tool_search.query"))]

    def run(self, parameters: dict) -> ToolResponse:
        """Retrieve relevant tools, returning catalog text (for the model to read) + the discovered list (for the strategy loop to expand the usable set)."""
        query = (parameters.get("query") or "").strip()
        if not query:
            return ToolResponse.error(self.prompts.text("tool.msg.tool_search.need_query"))
        names = [n for n in self.tool_retriever.retrieve(query, top_k=self.top_k) if n != self.name]
        if not names:
            return ToolResponse.ok(self.prompts.text("tool.msg.tool_search.no_match"))
        catalog = self.tool_retriever.registry.get_tools_description(names=names)
        return ToolResponse.ok(self.prompts.render("tool.tool_search_result", catalog=catalog),
                               data={"discovered": names})


