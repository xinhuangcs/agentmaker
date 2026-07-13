"""agentmaker.context.builder: the ContextBuilder main pipeline.

Assembles candidates from sources like memory / RAG / history into the final context fed to the model, kept
within the token budget.
Pipeline: Gather (collect candidates per source) -> MMR (per-source dedup + diversity selection) -> Budget
(three-region budget + two-round quota borrowing) -> Structure (fixed skeleton: system -> memory -> RAG ->
history -> question).

It does not run a second business-level rerank: the baseline relevance score comes from the retrieval
foundation; this layer only does what assembly alone can do: de-duplicate, select within budget via MMR,
allocate the budget, and structure the layout.
"""

import asyncio
from typing import Dict, List, Optional

from ..core.text import TokenCounter, count_tokens
from ..retrieval.types import RetrievalResult
from .mmr import mmr_select
from ..prompts import DEFAULT_PROMPTS
from .types import ContextConfig, ContextSource

# The display order of each source and the "set of known names" (the header text is provided by the central
# registry context.section.*; here we only define order and membership).
_SECTION_ORDER = ["memory", "rag", "history", "tool"]
_SECTION_NAMES = set(_SECTION_ORDER)


def _header_for(name: str, prompts) -> str:
    """The section header for a source name: known names (memory/rag/history/tool) take it from the central registry context.section.*, custom names fall back to `[name]`."""
    key = f"context.section.{name}"
    return prompts.text(key) if key in prompts else f"[{name}]"


class ContextBuilder:
    """Context assembler: assembles multi-source candidates into the final context text within budget."""

    def __init__(self, config: Optional[ContextConfig] = None, *, min_chunk_tokens: Optional[int] = None,
                 prompts=None, token_counter: TokenCounter = count_tokens):
        """Build a ContextBuilder.

        Args:
            config: Budget config; if omitted, the default ContextConfig is used.
            min_chunk_tokens: The maximum possible tokens of a single candidate's body, used for quota sanity
                checking; if omitted, config.min_chunk_tokens is used (default 64). If you mainly feed RAG
                (chunks can reach 512), pass 512 explicitly at construction (or set it in config) for stricter
                checking.

        The quota is validated at construction: if a source's quota cannot hold one complete candidate block,
        it raises immediately (guarding against misconfiguration). Before validating, the body-measure
        min_chunk_tokens is augmented with the list-prefix overhead (_PREFIX_TOKENS) to convert it to the
        rendering measure, matching _greedy's _item_tokens at the assembly stage. This eliminates the "validate
        passes but assembly cannot fit" measure mismatch.

        max_tokens may be omitted at construction (require_max_tokens=False): when attached to an Agent, the
        retrieval-block budget is supplied by the window-wide accounting WindowBudget.rag_budget via
        build_block(budget=...), max_tokens is not involved and may be left unset; only standalone build() /
        build_block(budget=None) need it, and it fails loud in those two places if missing (see
        _max_tokens_or_raise).
        """
        self.config = config or ContextConfig()
        self.prompts = prompts or DEFAULT_PROMPTS        # section headers (context.section.* / context.current_question) come from it
        self._count = token_counter                      # pluggable token counter (default count_tokens; can inject tiktoken in production)
        self._prefix_tokens = self._count("- ")          # fixed overhead of the list-item "- " prefix (must be computed before validate below)
        eff_min = self.config.min_chunk_tokens if min_chunk_tokens is None else min_chunk_tokens
        self.config.validate(min_chunk_tokens=eff_min + self._prefix_tokens, require_max_tokens=False)

    def _item_tokens(self, r: RetrievalResult) -> int:
        """The rendering overhead of a single candidate in the final list: '- ' prefix + body + trailing newline (list items are joined by '\\n', and assembly appends a newline after each item).

        The trailing newline is included to match the actual rendering in _sections: counting only '- body' and
        omitting the newline would, when swapping in a more precise counter (e.g. tiktoken counts each newline
        as 1 token), accumulate an underestimate and slightly overrun the budget.
        """
        return self._count(f"- {r.content}\n")

    def _structure_overhead(self, sources: List[ContextSource]) -> int:
        """Structural rendering overhead: each source's section header (including the newline between header and body) + inter-section blank lines ('\\n\\n'). The budget deducts this first, to avoid underestimating the final text by counting only header bodies.

        Conservative trade-off: reserve even for sources that may ultimately be empty (no candidates, header not
        rendered): we do not know in advance which sources are non-empty, so we would rather recall less than
        overrun the budget.
        """
        headers = sum(self._count(f"{_header_for(s.name, self.prompts)}\n") for s in sources)
        separators = self._count("\n\n") * max(len(sources) - 1, 0)   # sections joined by '\n\n': n sections have n-1 blank lines
        return headers + separators

    def _max_tokens_or_raise(self) -> int:
        """Standalone-call convention (build / build_block without budget) reads max_tokens; fails loud if missing (Agent-attached use overrides via budget= and does not go here)."""
        if not self.config.max_tokens:
            raise ValueError(
                "A standalone ContextBuilder.build() / build_block() call (without budget=) needs "
                "config.max_tokens. Use ContextConfig.for_window(llm.context_window) or pass max_tokens "
                "explicitly. When attached to an Agent you need not set it (the retrieval-block budget is "
                "supplied by the window-wide accounting WindowBudget via build_block(budget=)).")
        return self.config.max_tokens

    def build(self, query: str, *, sources: List[ContextSource], system_prompt: str = "", scope=None) -> str:
        """Assemble the final context text (system -> sections -> question), flattened into one string, suitable for single-shot / RAG-style calls.

        For multi-turn conversation use build_block instead: once history is flattened into a string, the
        user/assistant roles are lost.

        Args:
            query: The current user question.
            sources: The various sources (memory / rag / history / ...).
            system_prompt: The system prompt: a fixed reservation, always present, does not compete.

        Returns:
            str: The assembled context text (system -> sections -> question).
        """
        cfg = self.config
        max_tokens = self._max_tokens_or_raise()        # standalone-call convention: max_tokens must be present
        # Dynamic-region budget = total budget - output reserve - fixed footprint - structure overhead.
        reserve = int(max_tokens * cfg.output_reserve_ratio)
        # Fixed footprint: system prompt + current question (including its section header [Current question]).
        fixed = self._count(system_prompt) + self._count(self.prompts.render("context.current_question", query=query))
        # Structure overhead: each source's section header (rendered as [Memory] etc.); reserve its tokens to avoid underestimating the final text by counting only bodies.
        structure = self._structure_overhead(sources)
        kept = self._select(query, sources, max(max_tokens - reserve - fixed - structure, 0), scope)
        parts = [system_prompt] if system_prompt else []
        parts.extend(self._sections(kept))
        parts.append(self.prompts.render("context.current_question", query=query))
        return "\n\n".join(parts)

    def build_block(self, query: str, *, sources: List[ContextSource], scope=None,
                    budget: Optional[int] = None) -> str:
        """Assemble only the dynamic-source (memory / RAG / ...) context block: no system prompt, no [Current question].

        For multi-turn conversation scenarios: inject this block as a system message, while the conversation
        history is passed separately as role-carrying messages. Returns an empty string if there are no
        candidates.

        Args:
            query: The current user question (used for retrieval + budget estimation, but not written into the returned block).
            sources: The dynamic sources (memory / rag / ...).
            budget: An optional "retrieval-block net amount" override (from the Harness's window-wide accounting
                WindowBudget.rag_budget). If given: used directly as the retrieval-block budget: the
                window-level accounting has already deducted the output reserve and fixed overhead, so here we
                only further deduct the structure overhead, not re-reserve for output. If not given (None,
                standalone builder call): fall back to the builder's own convention max_tokens - output reserve
                - question - structure overhead.

        Returns:
            str: The sections assembled into a context block (empty string if no candidates).
        """
        cfg = self.config
        structure = self._structure_overhead(sources)
        if budget is None:
            max_tokens = self._max_tokens_or_raise()    # standalone-call convention: max_tokens must be present
            avail = max_tokens - int(max_tokens * cfg.output_reserve_ratio) - self._count(query)
        else:
            avail = budget          # already the net amount the window-wide accounting allotted to the retrieval block; do not deduct output reserve again (that is the accounting's job)
        kept = self._select(query, sources, max(avail - structure, 0), scope)
        return "\n\n".join(self._sections(kept))

    async def abuild_block(self, query: str, *, sources: List[ContextSource], scope=None,
                           budget: Optional[int] = None) -> str:
        """The async version of build_block (Harness.acontext_block uses it): the budget convention matches
        build_block, but multiple sources fetch concurrently via _aselect's asyncio.gather afetch (replacing
        serial fetch); the rest (structure overhead / MMR / allocation / joining) shares the sync implementation."""
        cfg = self.config
        structure = self._structure_overhead(sources)
        if budget is None:
            max_tokens = self._max_tokens_or_raise()
            avail = max_tokens - int(max_tokens * cfg.output_reserve_ratio) - self._count(query)
        else:
            avail = budget
        kept = await self._aselect(query, sources, max(avail - structure, 0), scope)
        return "\n\n".join(self._sections(kept))

    def _validate_sources(self, sources: List[ContextSource]) -> None:
        """Two source checks (both fail loud, before the expensive fetch): (1) every source.name is in
        config.source_ratios, otherwise it gets a 0 quota and silently never enters the context; (2) source
        names do not duplicate, otherwise same-named sources overwrite each other in the fetched dict and
        silently drop candidates."""
        seen = set()
        for src in sources:
            if src.name not in self.config.source_ratios:
                raise ValueError(
                    f"source '{src.name}' is not in ContextConfig.source_ratios (have "
                    f"{list(self.config.source_ratios)}); it would get a 0 quota and silently never enter the context. "
                    "Add it to source_ratios (set its budget share), or use an existing source name.")
            if src.name in seen:
                raise ValueError(
                    f"source name '{src.name}' passed in more than once: same-named sources overwrite each other during assembly and silently drop candidates. "
                    "Merge them into one source, or give them different source names.")
            seen.add(src.name)

    def _mmr(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Run MMR dedup over one source's candidates (pure CPU, shared by the sync / async paths)."""
        return mmr_select(candidates, top_k=None, lambda_=self.config.mmr_lambda,
                          dedup_threshold=self.config.dedup_threshold)

    def _select(self, query: str, sources: List[ContextSource],
                dynamic_budget: int, scope=None) -> Dict[str, List[RetrievalResult]]:
        """Per-source fetch (serial) -> MMR dedup -> three-region budget with two-round borrowing; returns the final kept candidates per source."""
        self._validate_sources(sources)
        fetched = {src.name: self._mmr(src.fetch(query, scope)) for src in sources}
        return self._allocate(fetched, dynamic_budget)

    async def _aselect(self, query: str, sources: List[ContextSource],
                       dynamic_budget: int, scope=None) -> Dict[str, List[RetrievalResult]]:
        """The async version of _select: each source fetches candidates concurrently via afetch under asyncio.gather (each on its own thread / natively async); the rest (MMR / allocation) is sync."""
        self._validate_sources(sources)
        raw = await asyncio.gather(*(src.afetch(query, scope) for src in sources))
        fetched = {src.name: self._mmr(cands) for src, cands in zip(sources, raw)}
        return self._allocate(fetched, dynamic_budget)

    def _allocate(self, fetched: Dict[str, List[RetrievalResult]],
                  dynamic_budget: int) -> Dict[str, List[RetrievalResult]]:
        """Two-round quota allocation: round one, each source takes within its own quota; round two, idle quota is given to sources that still want to place more."""
        cfg = self.config
        ratio_sum = sum(cfg.source_ratios.get(name, 0.0) for name in fetched) or 1.0
        # Round one: each source's quota = dynamic region * that source's ratio (normalized over the sources present).
        quota = {name: int(dynamic_budget * cfg.source_ratios.get(name, 0.0) / ratio_sum)
                 for name in fetched}
        kept: Dict[str, List[RetrievalResult]] = {}
        used: Dict[str, int] = {}
        leftover_items: Dict[str, List[RetrievalResult]] = {}  # candidates each source could not place (wanted to but exceeded quota)
        for name, results in fetched.items():
            kept[name], used[name], leftover_items[name] = self._greedy(results, quota[name])

        # Round two: distribute the idle quota (each source's quota - used) fairly, by "amount wanted", to
        # sources that still have leftovers. (Not by dict order, which would let earlier sources eat it all,
        # avoiding bias + a result that depends on the sources' input order.)
        if cfg.allow_borrow:
            free = sum(quota[n] - used[n] for n in fetched)
            wanting = {n: sum(self._item_tokens(r) for r in leftover_items[n])
                       for n in fetched if leftover_items[n]}  # how many more tokens each source wants to place (by rendered form)
            want_sum = sum(wanting.values())
            if free > 0 and want_sum > 0:
                for name, want in wanting.items():
                    share = int(free * want / want_sum)  # allocate the idle amount by share of amount wanted
                    extra, _, leftover_items[name] = self._greedy(leftover_items[name], share)
                    kept[name].extend(extra)
        return kept

    def _greedy(self, results: List[RetrievalResult], budget: int):
        """Collect candidates in MMR order within budget, stopping at the first that does not fit (cutoff); returns (placed, tokens used, remaining).

        Uses a cutoff rather than "skip the big block and keep stuffing small ones": the candidates have
        already been through MMR's relevance/dedup/diversity trade-off, so keep them strictly in order:
        "place the right ones" takes priority over "fill it up" (shuffling the order for utilization would let
        later candidates crowd out earlier ones).
        """
        kept, used = [], 0
        for i, r in enumerate(results):
            t = self._item_tokens(r)
            if used + t > budget:
                return kept, used, results[i:]  # stop at the first that does not fit; keep the rest (including later ones) as remaining
            kept.append(r)
            used += t
        return kept, used, []

    def _sections(self, kept: Dict[str, List[RetrievalResult]]) -> List[str]:
        """Format each source's candidates into a list of section strings, skipping empty sources.

        Known sources (memory/rag/history/tool) come first in the recommended _SECTION_ORDER; custom source
        names fall back to `[name]` and are ordered after by kept's insertion order, ensuring a custom source
        that made it into the budget is not silently omitted from rendering (matching _select admitting custom
        names).
        """
        known = [name for name in _SECTION_ORDER if name in kept]
        custom = [name for name in kept if name not in _SECTION_NAMES]
        parts = []
        for name in known + custom:
            items = kept.get(name) or []
            if items:
                body = "\n".join(f"- {r.content}" for r in items)
                parts.append(f"{_header_for(name, self.prompts)}\n{body}")
        return parts


# constructed sources, no network / key needed).
