"""agentmaker.rag.contextualizer: Contextual Retrieval, adding context to chunks before ingestion.

Inspired by Anthropic's Contextual Retrieval: after a chunk is split out it often loses context (e.g. a
"Lodging" chunk's body only reads "tier-1 cities capped at 500 per night", while the word "lodging" is in
the heading and never made it into the chunk), which hurts search accuracy. The fix is to add one
sentence of "its background in the original document" to each chunk before embedding / indexing.
Anthropic measured a reduction in retrieval failures: 35% on its own, 49% with BM25, 67% with rerank.

Key point: the enhanced text is used for retrieval only (pushed into the vector store / keyword store);
what is stored in the source of truth and ultimately returned to the user / LLM is still the original
chunk.

    HeadingContextualizer: prepends the heading path to the chunk (zero cost, no LLM; fixes "lost
        heading word").
    LLMContextualizer: the original Anthropic version, an LLM generates one sentence of context per chunk
        (stronger, but one LLM call per chunk).
"""

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from ..core.exceptions import RunCancelled, RunLimitExceeded
from ..core.llm_clients import LLMClient
from ..prompts import DEFAULT_PROMPTS
from .types import Chunk, Document
from ..core.trace_events import EVENT_RAG_CONTEXTUALIZE_FAILED

if TYPE_CHECKING:
    from ..prompts import PromptRegistry
    from ..runtime.observability import Tracer


class Contextualizer(ABC):
    """Turn a chunk into "enhanced text for retrieval" (original chunk + context)."""

    @abstractmethod
    def contextualize(self, chunk: Chunk, doc: Document) -> str:
        """Return the enhanced retrieval text; does not modify the chunk itself. On LLM failure it should fall back to the original text / heading path on its own; but governance-class exceptions (RunLimitExceeded / RunCancelled) must propagate and not be swallowed, otherwise RunPolicy's limit / cancel stops applying to per-chunk enhancement."""

    def fingerprint(self) -> str:
        """The identity of this enhancer (goes into the doc ingestion fingerprint): defaults to the class name. A subclass whose behavior is affected by prompt / model / params (like LLMContextualizer) should override this and include them, otherwise re-ingesting a document after swapping the prompt / model would be wrongly skipped as "fingerprint unchanged" and the index would keep the stale enhanced text."""
        return type(self).__name__


class HeadingContextualizer(Contextualizer):
    """Prepend the chunk's heading path to its body (zero LLM cost).

    Example: heading_path="Expense Policy > Lodging" + body "tier-1 cities capped at 500 per night"
        -> "Expense Policy > Lodging\ntier-1 cities capped at 500 per night"
    This way the keyword in the heading (Lodging) enters the searched text, directly mitigating "chunking
    lost the heading".
    """

    def contextualize(self, chunk: Chunk, doc: Document) -> str:
        """Heading path + body; returns the body as-is if there is no heading path."""
        if chunk.heading_path:
            return f"{chunk.heading_path}\n{chunk.content}"
        return chunk.content


DEFAULT_CONTEXT_PROMPT = DEFAULT_PROMPTS.text("rag.contextualize")


class LLMContextualizer(Contextualizer):
    """Use an LLM to generate one sentence of context per chunk (the original Anthropic version, stronger but one LLM call per chunk).

    Note: cost = document chunk count x one LLM call. For large documents, pair with caching / a cheap
    model (e.g. deepseek).
    """

    def __init__(self, llm: LLMClient, *, max_doc_chars: int = 4000, context_prompt: Optional[str] = None,
                 prompts: "Optional[PromptRegistry]" = None, tracer: "Optional[Tracer]" = None):
        """
        Args:
            llm: The LLM that generates context (a cheap model is recommended).
            max_doc_chars: The maximum number of characters of the whole document fed to the LLM
                (truncated if too long, to control cost).
            context_prompt: The system prompt for generating the chunk-context annotation; if omitted,
                use the framework default (registry key rag.contextualize). Pass your own to switch
                language.
            prompts: Optional prompt registry (PromptRegistry); defaults to DEFAULT_PROMPTS if omitted.
                context_prompt is a local shortcut override of it.
        """
        self.llm = llm
        self.tracer = tracer   # optional tracer: the per-chunk annotation LLM calls go into the trace and RunPolicy governance (governed_chat)
        self.max_doc_chars = max_doc_chars
        base = prompts or DEFAULT_PROMPTS
        self.prompts = base.with_overrides({"rag.contextualize": context_prompt}) if context_prompt else base
        self.context_prompt = self.prompts.text("rag.contextualize")

    def contextualize(self, chunk: Chunk, doc: Document) -> str:
        """Call the LLM to generate one sentence of context and prepend it to the body; on failure fall back to the heading path / original text (without losing searchability).

        The LLM sub-step during ingestion: driven via aio.run_sync over async governed_chat (keeping
        contextualize a synchronous interface, so as not to make the whole IngestionPipeline chain async;
        ingestion is an offline batch-processing path, consistent with the synchronous embedder).
        """
        doc_text = doc.content[:self.max_doc_chars]
        user = self.prompts.render("rag.contextualize_user", doc_text=doc_text, chunk=chunk.content)
        try:
            from ..core.aio import run_sync
            from ..runtime.execution.run_context import governed_chat   # lazy import: only the LLM version needs it
            ctx = run_sync(governed_chat(self.llm, [{"role": "system", "content": self.context_prompt},
                                                    {"role": "user", "content": user}],
                                         tracer=self.tracer, origin="rag.contextualize")).content.strip()
        except (RunLimitExceeded, RunCancelled):
            raise                                       # governance-class control-flow exceptions pass straight through (otherwise RunPolicy's limit/cancel stops applying to per-chunk enhancement)
        except Exception:  # noqa: BLE001
            if self.tracer is not None:                 # a silent degradation still leaves one observable event (distinguishing "not enhanced" from "enhancement keeps failing")
                from ..runtime.execution.run_context import correlation
                self.tracer.emit({"type": EVENT_RAG_CONTEXTUALIZE_FAILED, **correlation()})
            ctx = ""
        prefix = ctx or chunk.heading_path  # fall back to the heading path on LLM failure
        return f"{prefix}\n{chunk.content}" if prefix else chunk.content

    def fingerprint(self) -> str:
        """Include class name + prompt + model + max_doc_chars: changing any one counts as "changed", so a re-ingest is not wrongly skipped (overrides the base class's plain class name)."""
        raw = f"{type(self).__name__}\x00{self.context_prompt}\x00{getattr(self.llm, 'model', '')}\x00{self.max_doc_chars}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
