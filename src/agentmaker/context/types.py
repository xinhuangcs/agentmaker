"""agentmaker.context.types: config and source interface for context engineering.

ContextConfig: budget config (ratio-based and tunable; the industry best practice is "explicit allocation"
rather than passive accumulation).
ContextSource: the unified supplier interface that wraps memory / rag / history / etc. into sources the
builder can treat uniformly.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import List, Mapping, Optional

from ..retrieval.types import RetrievalResult


def _deep_freeze(v):
    """Recursively make a value immutable: Mapping -> read-only MappingProxyType, list/tuple -> tuple, values
    recursed too; scalars returned as is.

    Guards against shallow-freeze leakage: a shallow dict() copy does not stop a value from being a mutable
    list/dict, so recursion is required.
    """
    if isinstance(v, Mapping):
        return MappingProxyType({k: _deep_freeze(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_deep_freeze(x) for x in v)
    return v


@dataclass(frozen=True)
class ContextConfig:
    """Context budget config (frozen and immutable; hashable, injectable, derivable via dataclasses.replace).

    The budget uses "ratios" rather than absolute numbers: when you switch models (the window grows) you just
    scale by ratio, which is more stable than absolute numbers.

    Attributes:
        max_tokens: Total budget (for the whole context). There is NO hard-coded default: it must be set from
            the model's real window. Prefer ContextConfig.for_window(llm.context_window), or explicitly pass a
            value you know to be correct. Leaving it None and building a budget anyway raises in validate(),
            preventing "silently use some arbitrary default".
        output_reserve_ratio: The ratio reserved for output + the current question (does not compete for
            candidates).
        source_ratios: The quota ratio of each source within the dynamic region (keys are source names, e.g.
            history/rag/memory/tool); the values are recommended to sum to 1, though other sums are allowed
            (they allocate the dynamic region by relative proportion). After construction, __post_init__
            coerces values to float and deep-freezes them read-only.
        mmr_lambda: MMR's relevance vs diversity trade-off, passed to mmr_select.
        dedup_threshold: MMR near-duplicate threshold (cosine >= it is treated as a duplicate and dropped; > 1
            effectively disables dedup), passed to mmr_select.
        allow_borrow: Whether to enable quota borrowing (a source's unused idle quota is given, in a second
            round, to sources that still want to place more).
        min_chunk_tokens: The token count of a single body item that a quota must at least be able to hold.
    """
    max_tokens: Optional[int] = None
    output_reserve_ratio: float = 0.2
    # hash=False excludes this mutable mapping field from the auto __hash__ so the whole frozen config stays
    # hashable; compare stays True and eq still compares it.
    source_ratios: Mapping[str, float] = field(
        default_factory=lambda: {"history": 0.35, "rag": 0.30, "memory": 0.20, "tool": 0.15}, hash=False)
    mmr_lambda: float = 0.7
    dedup_threshold: float = 0.95
    allow_borrow: bool = True
    min_chunk_tokens: int = 64

    def __post_init__(self):
        # Coerce values to float ("0.5" -> 0.5; a non-numeric value raises a clear ValueError instead of
        # deferring the crash to sum()) -> deep copy -> read-only.
        # Placed in __post_init__ because for_window(**kwargs) constructs directly and bypasses any factory;
        # under frozen, mutating a field requires object.__setattr__.
        coerced = {k: float(v) for k, v in dict(self.source_ratios).items()}
        object.__setattr__(self, "source_ratios", _deep_freeze(coerced))

    @classmethod
    def for_window(cls, context_window: Optional[int], *, use_ratio: float = 0.5,
                   fallback_window: Optional[int] = None, **kwargs) -> "ContextConfig":
        """Derive the budget from the model's real context window (max_tokens = window * use_ratio) so the
        budget tracks the real window instead of being hard-coded.

        The window comes from the llm source-of-truth (LLMClient.context_window, i.e. the vendor data verified
        in the profile). use_ratio defaults to 0.5: the context takes only half the window, leaving ample room
        for output and safety margin (a larger window need not be fully used; overly long context dilutes the
        signal and adds cost).

        Args:
            context_window: The model context window in tokens (from LLMClient.context_window).
            use_ratio: The ratio the context takes of the window, default 0.5.
            fallback_window: The fallback window used when the window is unknown (local / self-built models,
                context_window is None); it must be given explicitly (no default, forcing the caller to pin a
                conservative value), otherwise None raises in validate().
            **kwargs: Passed through to other ContextConfig fields (source_ratios / mmr_lambda / etc.).

        Example:
            ContextConfig.for_window(LLMClient("deepseek").context_window)   # 1M -> max_tokens=500,000
            ContextConfig.for_window(None, fallback_window=8000)             # local unknown model, explicit fallback
        """
        window = context_window or fallback_window
        max_tokens = int(window * use_ratio) if window else None
        return cls(max_tokens=max_tokens, **kwargs)

    def validate(self, *, min_chunk_tokens: Optional[int] = None, require_max_tokens: bool = True) -> None:
        """Sanity-check the config: first validate the value ranges of the numeric parameters, then ensure
        each source's quota can hold one complete candidate block.

        Guards against two kinds of "misconfiguration": (1) a ratio / budget parameter takes an illegal value
        (negative reserve, out-of-range lambda, all-negative ratios, etc.), which makes the budget compute
        counter-intuitive results (e.g. a negative reserve inflating the context beyond the window); (2) a
        quota < a single candidate's size -> even the most relevant item cannot fit (except sources with an
        explicit ratio=0: treated as intentionally unbudgeted, this check is skipped).
        (Note: (2) only guards "a single item cannot fit"; "total candidates exceed the budget" is a normal
        trade-off, handled by the builder keeping the top few by relevance, and is not covered here.)

        Args:
            min_chunk_tokens: The token count of a single item that a quota must at least hold; None (default)
                uses the field's own value. ContextBuilder passes in the rendering measure of "body + list
                prefix" to override it, aligning the check with the actual assembly footprint.
            require_max_tokens: Whether to require max_tokens to have a value. True (default, the standalone
                builder call convention): missing max_tokens raises. False (the Agent-attached convention): the
                retrieval-block budget is given by the window-wide accounting WindowBudget.rag_budget,
                max_tokens is not involved, so a missing max_tokens is legal here and the max_tokens-related
                checks are skipped (including the per-source quota check, which depends on max_tokens). The
                structural checks (dedup / output_reserve_ratio / mmr_lambda / source_ratios) run under both
                conventions.
        """
        min_chunk_tokens = self.min_chunk_tokens if min_chunk_tokens is None else min_chunk_tokens
        # Structural checks (unrelated to max_tokens, run under both conventions).
        if self.dedup_threshold < 0:
            raise ValueError(f"dedup_threshold must be >= 0, got {self.dedup_threshold} (> 1 disables dedup).")
        if not 0.0 <= self.output_reserve_ratio < 1.0:
            raise ValueError(
                f"output_reserve_ratio must be in [0, 1), got {self.output_reserve_ratio}: "
                "a negative value inflates the budget beyond the window, and >=1 makes the dynamic region zero "
                "so no candidate can enter.")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {self.mmr_lambda} (1=pure relevance, 0=pure diversity).")
        if any(r < 0 for r in self.source_ratios.values()) or sum(self.source_ratios.values()) <= 0:
            raise ValueError(f"source_ratios must be non-negative with a sum > 0, got {self.source_ratios}.")
        # max_tokens-related checks: a type error always raises; a missing value raises only under
        # require_max_tokens; the per-source quota check depends on max_tokens.
        if self.max_tokens is not None and (not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool)):
            raise ValueError(f"max_tokens must be int or None, got {self.max_tokens!r} ({type(self.max_tokens).__name__}).")
        if not self.max_tokens:
            if require_max_tokens:
                raise ValueError(
                    "max_tokens is not set: configure it from the model's real window. "
                    "Use ContextConfig.for_window(llm.context_window), or pass max_tokens explicitly. "
                    "(When calling the builder standalone the budget must be tied to the real window; when "
                    "attached to an Agent it is supplied by window-wide accounting and may be left unset.)")
            return  # Agent-attached convention: the budget comes from WindowBudget; skip max_tokens-related checks.
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}.")
        dynamic = self.max_tokens * (1.0 - self.output_reserve_ratio)
        ratio_sum = sum(self.source_ratios.values()) or 1.0
        for name, ratio in self.source_ratios.items():
            if ratio == 0:
                continue  # Explicit 0 quota = intentionally unbudgeted (not allotted in round one, only borrowable), not a misconfiguration.
            quota = dynamic * ratio / ratio_sum
            if quota < min_chunk_tokens:
                raise ValueError(
                    f"source '{name}' has a quota of about {int(quota)} tokens < a single candidate's {min_chunk_tokens} tokens, "
                    f"so it cannot hold one complete block. Raise max_tokens, raise this source's ratio, or reduce the splitter's chunk_tokens.")


@dataclass(frozen=True)
class ReducerConfig:
    """The "how much recent original text to keep" knob for loss-aware trimming of paradigm trajectories (used by Harness.reduce).

    The trajectory's token budget is not here: it is given by the window-wide accounting
    WindowBudget.trajectory_budget (see window_budget.py). This config only controls how many trailing
    steps/items of each paradigm are kept as original text without summarization.

    Attributes:
        agent_keep_recent_steps: How many trailing atomic units (assistant + its tool results) of the unified-loop
            Agent tool trajectory to keep as original text.
        plan_keep_recent: How many trailing Plan step results to keep as original text.
    """
    agent_keep_recent_steps: int = 3
    plan_keep_recent: int = 3

    def validate(self) -> None:
        """Range check: keep_recent >= 0."""
        if self.agent_keep_recent_steps < 0 or self.plan_keep_recent < 0:
            raise ValueError("agent_keep_recent_steps / plan_keep_recent must be >= 0")


@dataclass(frozen=True)
class CompactionConfig:
    """Knobs for cross-session Chat history compression (HistoryCompactor).

    Attributes:
        keep_recent: How many recent items to keep as original text (not compressed, ensuring the current
            topic stays exact and coherent).
        trigger_tokens: The token count the history must exceed before compression is triggered.
    """
    keep_recent: int = 4
    trigger_tokens: int = 2000

    def validate(self) -> None:
        """Range check: both must be >= 1."""
        if self.keep_recent < 1 or self.trigger_tokens < 1:
            raise ValueError(f"keep_recent / trigger_tokens must be >= 1, got {self.keep_recent}, {self.trigger_tokens}")


class ContextSource(ABC):
    """The unified interface for a context source: a class of supplier (memory / rag / history / ...).

    name decides which quota it uses (corresponding to a key of ContextConfig.source_ratios).
    The system prompt is not a source: it is a fixed reservation, always present, does not compete, and is
    handled separately by ContextBuilder.build.
    """

    name: str

    @abstractmethod
    def fetch(self, query: str, scope=None) -> List[RetrievalResult]:
        """Fetch this source's candidates for the query (already carrying score and optional embedding).

        scope: the session identifier threaded through the run (optional). Which of its dimensions to use is up
        to the implementation: e.g. memory uses the user dimension (user-level, cross-conversation), rag might
        use user/app; agentmaker only threads scope down to here and does not prescribe its semantics.
        """

    async def afetch(self, query: str, scope=None) -> List[RetrievalResult]:
        """The async version of fetch (called by ContextBuilder.abuild_block when fanning out over multiple
        sources via asyncio.gather).

        Defaults to wrapping the sync fetch in to_thread (retrieval involves embedding / DB IO, and each source
        occupies its own thread concurrently); an async retrieval backend can override this to await natively.
        """
        return await asyncio.to_thread(lambda: self.fetch(query, scope))
