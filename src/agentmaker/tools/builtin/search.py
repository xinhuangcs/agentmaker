"""agentmaker.tools.builtin.search: multi-source web search tool.

Falls back in a fixed order: Tavily -> DuckDuckGo -> Brave -> SerpAPI.
If the previous source has no library installed / no key configured / fails, it
automatically falls back to the next; only if all fail does it error.
Each source's SDK is imported lazily, so a missing source does not affect the rest.
Keys are read from environment variables: TAVILY_API_KEY / BRAVE_API_KEY / SERPAPI_API_KEY
(DuckDuckGo needs no key).
"""

import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from ...prompts import DEFAULT_PROMPTS
from ..base import Tool, ToolParameter
from ..response import ToolResponse

if TYPE_CHECKING:
    from ...prompts import PromptRegistry

# Fallback order: each item is (source_name, method_name); change the order here.
# Balancing "free first but quality-preserving": Tavily has a large free quota (1000/month),
# is optimized for agents, and has good quality, so it goes first; DuckDuckGo is entirely free
# as a fallback (when Tavily has no key / quota is exhausted); Brave and SerpAPI come last.
_SOURCES = [
    ("Tavily", "_via_tavily"),
    ("DuckDuckGo", "_via_duckduckgo"),
    ("Brave", "_via_brave"),
    ("SerpAPI", "_via_serpapi"),
]


class SearchTool(Tool):
    """Web search. Multi-source automatic fallback (Tavily->DuckDuckGo->Brave->SerpAPI), returning a summary of the top results."""

    external_content = True   # Search results are external content (attackers can manipulate them via SEO): wrap them with an anti-injection delimiter guardrail before feeding them back to the model.
    supports_parallel = True  # Read-only with no shared mutable state (each call issues an independent HTTP request): the model can run multiple searches in one turn concurrently.

    def __init__(self, max_results: int = 5, *, prompts: "Optional[PromptRegistry]" = None):
        """
        Args:
            max_results: Maximum number of results returned per call.
            prompts: Optional prompt registry (PromptRegistry); the tool description / parameter text come from it, defaulting to DEFAULT_PROMPTS if not passed.
        """
        if max_results <= 0:
            raise ValueError(f"max_results must be a positive integer, got {max_results}")
        self.prompts = prompts or DEFAULT_PROMPTS
        super().__init__("search", self.prompts.text("tool.desc.search"))
        self.max_results = max_results

    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("query", "string", self.prompts.text("tool.param.search.query"))]

    def run(self, parameters: dict) -> ToolResponse:
        """Try each source in fallback order and return the first successful result; if all fail, summarize each source's reason.

        Args:
            parameters: Contains "query".

        Returns:
            ToolResponse: ok on success (data carries the raw entries {source, answer, results});
                error if all sources are unavailable.
        """
        query = (parameters.get("query") or "").strip()
        if not query:
            return ToolResponse.error(self.prompts.text("tool.msg.search.empty"))

        errors = []
        for source_name, method_name in _SOURCES:
            try:
                answer, results = getattr(self, method_name)(query)
            except Exception as e:
                errors.append(f"{source_name}: {e}")
                continue
            # Filter out placeholder entries where every field is empty: when an upstream library / API
            # field contract drifts (e.g. a key is renamed), get() yields None and produces a non-empty
            # list of empty shells. Judge success by "non-empty after cleaning"; otherwise treat this
            # source as having no valid results and keep falling back to the next source.
            results = [r for r in results if r.get("title") or r.get("snippet") or r.get("url")]
            if results:
                # Leave a signal on multi-source fallback: when this source succeeds but earlier sources
                # failed, put the failure reasons in data.degraded (visible via trace); text still gives
                # the model only clean results. Otherwise a paid source (e.g. an invalid Tavily key)
                # degrading long-term to a free source would be entirely silent and undetectable.
                return ToolResponse.ok(
                    self._format(source_name, answer, results),
                    data={"source": source_name, "answer": answer, "results": results,
                          **({"degraded": errors} if errors else {})})
            errors.append(self.prompts.render("tool.msg.search.no_result", source=source_name))
        return ToolResponse.error(self.prompts.render("tool.msg.search.all_failed", errors="\n".join(errors)))

    def _format(self, source_name: str, answer: Optional[str], results: List[dict]) -> str:
        """Format results into readable text: if there is an answer (Tavily's direct AI answer), put it first, then list each result."""
        lines = [self.prompts.render("tool.msg.search.source_label", source=source_name)]
        if answer:
            lines.append(self.prompts.render("tool.msg.search.ai_answer", answer=answer) + "\n")
        for i, r in enumerate(results[: self.max_results], 1):
            # Use `or ''` (not a get default): in the adapters the keys always exist but the value may be None; get's default only applies to a missing key and does not stop None, which would render the literal "None".
            lines.append(f"{i}. {r.get('title') or ''}\n   {r.get('snippet') or ''}\n   {r.get('url') or ''}")
        return "\n".join(lines)

    # Per-source implementations: lazily import the corresponding library; a missing library / key raises, caught by run() to fall back.
    # Returns (answer, results): answer is that source's "direct AI answer", None if none; results is a list of uniformly-shaped results.

    def _via_duckduckgo(self, query: str) -> Tuple[Optional[str], List[dict]]:
        """DuckDuckGo: free, no key needed (uses the ddgs library). No AI answer."""
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = ddgs.text(query, max_results=self.max_results)
        return None, [{"title": h.get("title"), "snippet": h.get("body"), "url": h.get("href")} for h in hits]

    def _via_tavily(self, query: str) -> Tuple[Optional[str], List[dict]]:
        """Tavily: needs TAVILY_API_KEY. With include_answer it returns a "direct AI answer"."""
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY not configured")
        from tavily import TavilyClient

        data = TavilyClient(api_key=key).search(query, max_results=self.max_results, include_answer=True)
        results = [{"title": r.get("title"), "snippet": r.get("content"), "url": r.get("url")}
                   for r in data.get("results", [])]
        return data.get("answer"), results

    def _via_brave(self, query: str) -> Tuple[Optional[str], List[dict]]:
        """Brave Search: needs BRAVE_API_KEY, uses REST."""
        key = os.getenv("BRAVE_API_KEY")
        if not key:
            raise RuntimeError("BRAVE_API_KEY not configured")
        import requests

        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            params={"q": query, "count": self.max_results},
            timeout=10,
        )
        resp.raise_for_status()
        web = resp.json().get("web", {}).get("results", [])
        return None, [{"title": r.get("title"), "snippet": r.get("description"), "url": r.get("url")} for r in web]

    def _via_serpapi(self, query: str) -> Tuple[Optional[str], List[dict]]:
        """SerpAPI: needs SERPAPI_API_KEY."""
        key = os.getenv("SERPAPI_API_KEY")
        if not key:
            raise RuntimeError("SERPAPI_API_KEY not configured")
        import serpapi  # Lazy import (official serpapi-python package, serpapi.Client.search).

        # Official new client: parameters are keyword-based and engine="google" must be explicit; num caps the number of results (10-100; Google may recently cap it back to 10, which is their own limit).
        data = serpapi.Client(api_key=key).search(q=query, engine="google", num=self.max_results)
        return None, [{"title": r.get("title"), "snippet": r.get("snippet"), "url": r.get("link")}
                      for r in data.get("organic_results", [])]


# With TAVILY_API_KEY set it uses the preferred Tavily; without it, it falls back to the key-free DuckDuckGo.
