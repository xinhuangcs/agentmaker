"""agentmaker.context.history_compactor: conversation history compaction (Anthropic compaction).

Conversations get longer and longer, and dozens of turns of history will exceed the budget while diluting the
signal (context rot). The approach: LLM-summarize the older turns into a single "recap" and keep only the most
recent keep_recent turns verbatim. Recent conversation must stay precise (the model continues answering from
it); distant history only needs a summary.

Why LLM summarization rather than truncation: the history is "one big object with scattered highlights", and
truncation would drop key interactions in the middle; compressing one big object is worth a single LLM call.
This is a different matter from "compressing scattered candidates at assembly time" (which should not be done,
and is instead guaranteed by upstream chunking to control size).
"""

import hashlib
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

from ..core.llm_clients import LLMClient
from ..core.message import Message
from ..core.multimodal import content_text, content_tokens
from ..core.text import TokenCounter, count_tokens
from ..prompts import DEFAULT_PROMPTS

_logger = logging.getLogger(__name__)   # summary-failure degradation warnings go through it (the library configures no handler; the host takes over, see the NullHandler in agentmaker/__init__)

# The default prompt is now provided by the central registry agentmaker.prompts (single source of truth); this
# constant is a convenience alias snapshotted at import time (it does not track later DEFAULT_PROMPTS.override; for
# the currently effective value use DEFAULT_PROMPTS.text(key)).
DEFAULT_SUMMARY_PROMPT = DEFAULT_PROMPTS.text("context.summary")


class HistoryCompactor:
    """Conversation history compactor: LLM-summarize old turns + keep the most recent few turns verbatim."""

    def __init__(self, llm: LLMClient, *, keep_recent: int = 4, trigger_tokens: int = 2000,
                 max_summary_tokens: int = 1000,
                 summary_prompt: Optional[str] = None, prompts=None, token_counter: TokenCounter = count_tokens):
        """
        Args:
            llm: the LLM used for summarization (a cheap model such as deepseek is recommended).
            keep_recent: how many recent turns to keep verbatim (uncompressed, ensuring the current topic stays
                precise and coherent). Default 4, must be >= 1.
            trigger_tokens: compress only when the total history token count exceeds this, otherwise return
                unchanged (not wasting an LLM call). Must be >= 0.
            max_summary_tokens: the hard cap (in tokens) of the recap summary, truncated if exceeded. Default
                1000, must be >= 1. This prevents the incrementally merged summary from growing ever larger
                across hundreds of turns: the LLM's merge output length is uncontrollable, and the cached summary
                gets fed back as input on the next turn.
            summary_prompt: the summary instruction used to compact history; if omitted, the framework default is
                used (central registry context.summary). Pass your own to switch language or adjust summary style.
            prompts: optional prompt registry (PromptRegistry, see agentmaker.prompts); if omitted,
                DEFAULT_PROMPTS is used. summary_prompt is a local shortcut override of it (equivalent to
                overriding context.summary). The recap prefix is taken from context.summary_prefix.
        """
        if keep_recent < 1:
            raise ValueError(
                f"keep_recent must be >= 1, currently {keep_recent}: "
                "=0 would degenerate, via the negative-zero slice history[:-0], into \"summarize empty content + "
                "keep the entire history\", the opposite of the compaction intent.")
        if trigger_tokens < 0:
            raise ValueError(f"trigger_tokens cannot be negative, currently {trigger_tokens}.")
        if max_summary_tokens < 1:
            raise ValueError(f"max_summary_tokens must be >= 1, currently {max_summary_tokens}.")
        self.llm = llm
        self.keep_recent = keep_recent
        self.trigger_tokens = trigger_tokens
        self.max_summary_tokens = max_summary_tokens
        self._count = token_counter                       # pluggable token counter (defaults to count_tokens)
        base = prompts or DEFAULT_PROMPTS
        self.prompts = base.with_overrides({"context.summary": summary_prompt}) if summary_prompt else base
        self.summary_prompt = self.prompts.text("context.summary")        # convenience alias = the actual value after injection
        self.merge_prompt = self.prompts.text("context.summary_merge")    # incremental merge instruction (old summary + the few newly slid-out turns -> new summary)
        # Incremental cache: digest(old turns) -> (old-turn count, that turn's summary). In a long conversation
        # each turn's history tail grows by +2 while keep_recent stays fixed, so this turn's old turns are a
        # prefix of the previous turn's old turns. A prefix hit therefore means only summarizing the few newly
        # slid-out turns + merging the old summary, reducing the O(n^2) of re-summarizing the whole segment to
        # O(delta) per turn. A bounded LRU prevents unbounded growth.
        self._cache: "OrderedDict[str, Tuple[int, str]]" = OrderedDict()
        self._cache_max = 32

    @classmethod
    def from_config(cls, llm: LLMClient, config, *, summary_prompt: Optional[str] = None,
                    prompts=None, token_counter: TokenCounter = count_tokens) -> "HistoryCompactor":
        """Assemble a HistoryCompactor from an AgentmakerConfig: slice config.compaction (keep_recent / trigger_tokens).

        Same mindset as each retrieval class's from_config (set defaults in one place, assemble in one line); the
        compactor needs the llm, so llm is passed separately.

        Args:
            llm: the LLM used for summarization.
            config: AgentmakerConfig (reads config.compaction).
            summary_prompt: optional summary-instruction override; if omitted, the framework default is used.
            prompts: optional prompt registry (PromptRegistry); passed through to the constructor, and if omitted
                DEFAULT_PROMPTS is used (supports isolated overrides that do not pollute the global).
        """
        config.compaction.validate()                          # validate the slice it uses before applying it
        c = config.compaction
        return cls(llm, keep_recent=c.keep_recent, trigger_tokens=c.trigger_tokens,
                   summary_prompt=summary_prompt, prompts=prompts, token_counter=token_counter)

    def compact(self, history: List[Message], *, summarize=None) -> List[Message]:
        """When history exceeds trigger_tokens, compress to [recap (system)] + the most recent keep_recent turns; otherwise return unchanged.

        Args:
            history: the full conversation history.
            summarize: optional (text, instruction) -> str summary callback (for an app's custom summary
                channel). If omitted, the built-in llm is used (standalone, e.g. this file's self-test). The
                Harness path goes through acompact (whose asummarize is governed via acall_llm). A summary
                failure should return an empty string, and `_assemble` then does not compress and keeps the
                original (no history lost).

        Returns:
            List[Message]: the compacted history (fewer turns, fewer tokens).
        """
        split = self._split(history)
        if split is None:
            return history                # below the threshold, or few turns to begin with: no compaction
        old, recent = split
        convo, instruction, key = self._plan_summary(old)
        summary = self._cap((summarize or self._summarize)(convo, instruction))
        self._store(key, len(old), summary)
        return self._assemble(summary, recent, history)

    async def acompact(self, history: List[Message], *, asummarize=None) -> List[Message]:
        """Async version of compact: summarization goes through the asummarize callback (governed via acall_llm if Harness passes it) or the built-in llm.chat (async); splitting / assembly is shared with compact."""
        split = self._split(history)
        if split is None:
            return history
        old, recent = split
        convo, instruction, key = self._plan_summary(old)
        summary = self._cap(await (asummarize or self._asummarize)(convo, instruction))
        self._store(key, len(old), summary)
        return self._assemble(summary, recent, history)

    @staticmethod
    def _digest(old: List[Message]) -> str:
        """Compute a stable fingerprint for a segment of old turns (each turn as role\\x1f content, joined with \\x1e, then sha256): the prefix-comparison key for the incremental cache."""
        return hashlib.sha256("\x1e".join(f"{m.role}\x1f{m.content}" for m in old).encode("utf-8")).hexdigest()

    def _plan_summary(self, old: List[Message]) -> Tuple[str, str, str]:
        """Plan this summarization: if the cache holds an old summary for a true prefix of this turn's old turns, only summarize the few newly slid-out turns + the merge instruction (incremental); otherwise re-summarize the whole segment. Returns (text to summarize, instruction used, cache key to write this turn)."""
        old_len = len(old)
        key = self._digest(old)
        for k, (cached_len, cached_summary) in reversed(self._cache.items()):   # try the most recently stored (longest prefix) first
            if cached_len < old_len and self._digest(old[:cached_len]) == k:
                return cached_summary + "\n" + self._convo(old[cached_len:]), self.merge_prompt, key
        return self._convo(old), self.summary_prompt, key

    def _cap(self, summary: str) -> str:
        """Hard-truncate the summary to within max_summary_tokens (if exceeded, truncate by character ratio + ellipsis): prevents the incrementally merged summary from growing ever larger across hundreds of turns.

        The cached summary is fed back as input for the next turn's merge; without a cap it would grow
        monotonically and blow the "recap" past the budget. Follows the same truncation approach as
        reducer._summary_block.
        """
        if not summary or self._count(summary) <= self.max_summary_tokens:
            return summary
        ell = "…"
        room = self.max_summary_tokens - self._count(ell)     # reserve tokens for the ellipsis
        if room < 1:
            return ell
        ratio = len(summary) / max(self._count(summary), 1)   # estimate how many characters per token
        return summary[:max(1, int(room * ratio))].rstrip() + ell

    def _store(self, key: str, old_len: int, summary: str) -> None:
        """Write this turn's summary into the bounded LRU cache (an empty summary is not stored, so failures do not pollute the cache)."""
        if not summary:
            return
        self._cache[key] = (old_len, summary)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def _summarize(self, text: str, instruction: str) -> str:
        """Standalone default summarization (uses the built-in llm directly, not governed by the Harness); returns an empty string on failure.

        llm.chat is now async, so it is driven via aio.run_sync (keeping compact's synchronous interface; the
        Harness path uses _asummarize and does not go through here).
        """
        from ..core.aio import run_sync
        try:
            return run_sync(self.llm.chat([{"role": "system", "content": instruction},
                                           {"role": "user", "content": text}])).content.strip()
        except Exception as e:  # noqa: BLE001  on summary failure, prefer not to compress and keep the original rather than lose history; but leave a signal
            _logger.warning("History compaction summary failed, skipping compaction this turn (persistent failures let history keep accumulating and the token bill quietly rise): %r", e)
            return ""

    async def _asummarize(self, text: str, instruction: str) -> str:
        """Async version of _summarize."""
        try:
            return (await self.llm.chat([{"role": "system", "content": instruction},
                                         {"role": "user", "content": text}])).content.strip()
        except Exception as e:  # noqa: BLE001  on failure, keep the original and do not lose history; but leave a signal
            _logger.warning("History compaction summary failed, skipping compaction this turn (persistent failures let history keep accumulating and the token bill quietly rise): %r", e)
            return ""

    def _split(self, history: List[Message]):
        """Decide whether to compress and split: return None to not compress, otherwise return (old turns, the most recent keep_recent turns)."""
        total = sum(content_tokens(m.content, self._count) for m in history)   # multimodal-safe (flat estimate per image)
        if total <= self.trigger_tokens or len(history) <= self.keep_recent:
            return None
        return history[:-self.keep_recent], history[-self.keep_recent:]

    @staticmethod
    def _convo(messages: List[Message]) -> str:
        """Join the old turns to be summarized into text (image parts render as "[image: ...]" placeholders)."""
        return "\n".join(f"{m.role}: {content_text(m.content)}" for m in messages)

    def _assemble(self, summary: str, recent: List[Message], history: List[Message]) -> List[Message]:
        """Non-empty summary: [recap (system)] + recent verbatim; empty summary: return unchanged (no history lost). The prefix is taken from the registry's context.summary_prefix.

        Note: if the most recent keep_recent turns are themselves long, the total may still be > trigger_tokens.
        This is an intentional trade-off: precision of recent conversation takes priority, and they are not
        compressed further.
        """
        if not summary:
            return history
        prefix = self.prompts.text("context.summary_prefix")
        return [Message(content=f"{prefix}{summary}", role="system"), *recent]


