"""agentmaker.context.window_budget: the window budget (single bookkeeper).

Solves the "two half-windows each deciding on their own" problem: the retrieval block
(ContextConfig.for_window's use_ratio=0.5) and the paradigm trajectory (reducer's
budget_fraction=0.5) each defaulted to half the window, unaware of each other, so in the
extreme case system + tool schemas + retrieval block + trajectory + output would approach
or exceed the full window. This module funnels allocation of the whole window into a single
ledger:

    whole window = output reserve + fixed overhead (system + tool schemas) + retrieval block + trajectory

All streams that compete for the window draw their quota from the same WindowBudget; none
decides on its own.

Two objects:
    - WindowBudgetConfig: configuration knobs (part of AgentmakerConfig / AgentSpec,
      serializable): output reserve + retrieval block ratio.
    - WindowBudget: a value object computed once per run (not part of config): carves each
      stream's quota out of the real window plus the measured fixed overhead.

Why the output reserve is min(three) rather than a fixed percentage: the per-call output cap
is decoupled from the window. The window can reach 1M, yet output is commonly only 8K~128K.
Reserving by window fraction (say 20%) would set aside 200K tokens on a 1M window as a dead
zone the model can never fill, wasting usable input; a fixed absolute number would eat up
input on an 8K small window. So take min(desired, model hard cap, window guardrail), three
clamps each guarding against one failure mode.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WindowBudgetConfig:
    """Window allocation knobs (single source-of-truth, replacing the scattered ContextConfig.output_reserve_ratio + ReducerConfig.budget_fraction).

    Attributes:
        desired_output_tokens: How many tokens at most you want the model to generate this
            time (absolute number, the main knob for the output reserve). The default is
            fine; raise it if you want long answers. It is further clamped by the "model hard
            cap" and the "window guardrail" (see output_reserve).
        max_output_fraction: Small-window guardrail: the output reserve takes at most this
            fraction of the window. It only really engages on small windows (e.g. 8K),
            preventing the output reserve from eating up input space; on large windows it
            almost never triggers (clamped first by desired / model cap).
        rag_ratio: The fraction of the "allocatable balance" that the retrieval block
            (memory/RAG) takes; the trajectory gets the rest. Only this one ratio is set,
            the trajectory no longer lists a second number, structurally ruling out "two
            ratios summing past the window".
    """
    desired_output_tokens: int = 4096
    max_output_fraction: float = 0.5
    rag_ratio: float = 0.35

    def validate(self) -> None:
        """Range check: output reserve >= 0, and both the guardrail and retrieval block ratio within valid ranges."""
        if self.desired_output_tokens < 0:
            raise ValueError(f"desired_output_tokens must be >= 0, got {self.desired_output_tokens}")
        if not 0.0 < self.max_output_fraction <= 1.0:
            raise ValueError(f"max_output_fraction must be within (0, 1], got {self.max_output_fraction}")
        if not 0.0 <= self.rag_ratio <= 1.0:
            raise ValueError(f"rag_ratio must be within [0, 1], got {self.rag_ratio}")

    def output_reserve(self, *, window: int, model_max_output: Optional[int]) -> int:
        """Output reserve = min(desired, model per-call hard cap, window guardrail), three clamps each guarding against one failure mode.

        - desired_output_tokens: do not reserve too much (the default is already small).
        - model_max_output: do not leave a dead zone on large windows: the model outputs at
          most this many tokens, so reserving beyond it is pure waste (None means unknown,
          does not participate in the clamp).
        - window * max_output_fraction: do not eat up input on small windows.

        Args:
            window: The model's real context window in tokens.
            model_max_output: The model's max output tokens per call (from
                LLMClient.max_output_tokens); None means do not clamp on it.

        Returns:
            int: The token count reserved for model output this time.
        """
        caps = [self.desired_output_tokens, int(window * self.max_output_fraction)]
        if model_max_output:
            caps.append(model_max_output)
        return max(min(caps), 0)


@dataclass(frozen=True)
class WindowBudget:
    """The window budget for one run (a value object, not part of config files): allocates the whole window among competing streams.

    Attributes:
        window: The model's real window in tokens.
        output_reserve: The output reserve already computed as min(three) (see
            WindowBudgetConfig.output_reserve).
        system_tokens: The measured system-prompt tokens (one part of fixed overhead).
        tool_tokens: The measured tool-schema tokens (one part of fixed overhead); they
            ride in the request's tools= payload, not in messages, so trajectory trimming
            cannot account for them. They must be subtracted separately, otherwise the
            unified-loop Agent's trajectory could grow large enough to push the tool schemas
            out of the window.
        rag_ratio: The retrieval block's fraction of the "allocatable balance" (passed
            through from config).
    """
    window: int
    output_reserve: int
    system_tokens: int
    tool_tokens: int
    rag_ratio: float

    @property
    def fixed(self) -> int:
        """Total fixed overhead = system prompt + tool schemas (convenience property)."""
        return self.system_tokens + self.tool_tokens

    @property
    def spendable(self) -> int:
        """The balance actually divisible between retrieval block / trajectory after subtracting output reserve + fixed overhead (system + tool schemas), non-negative."""
        return max(self.window - self.output_reserve - self.fixed, 0)

    @property
    def rag_budget(self) -> int:
        """Retrieval block cap = allocatable balance * rag_ratio."""
        return int(self.spendable * self.rag_ratio)

    def trajectory_budget(self, *, rag_in_scope: bool) -> int:
        """The trimming budget for the paradigm trajectory (the two paradigm classes differ in semantics, hence the branch).

        Args:
            rag_in_scope: Whether the data being trimmed already contains the retrieval block.
                True (unified-loop Agent: the retrieval block sits as a system message in the
                    protected head of state.messages) => gives "window - output reserve - tool schemas":
                    the reducer counts the retrieval block / system into the head it protects,
                    but the tool schemas hang off the tools= payload where the reducer cannot
                    see them, so they are subtracted here separately.
                False (Plan/Reflection: what is trimmed is step results / draft fragments, with
                    the retrieval block counted separately in the prompt) => gives "balance - retrieval block".

        Returns:
            int: The token budget for trajectory trimming.
        """
        if rag_in_scope:
            return max(self.window - self.output_reserve - self.tool_tokens, 0)
        return max(self.spendable - self.rag_budget, 0)

    @classmethod
    def for_run(cls, *, llm, cfg: WindowBudgetConfig, system_tokens: int = 0,
                tool_tokens: int = 0, rag_ratio: Optional[float] = None) -> Optional["WindowBudget"]:
        """Compute one ledger from this run's real window plus the measured fixed overhead; returns None when the window is unknown (llm.context_window is None).

        Args:
            llm: The LLM client (source of context_window and max_output_tokens).
            cfg: The window allocation knobs.
            system_tokens: The system-prompt tokens (optional; the unified-loop Agent path
                does not use it, see trajectory_budget).
            tool_tokens: The tool-schema tokens (counts tool usage into fixed overhead).
            rag_ratio: Overrides cfg.rag_ratio (optional); when no retrieval source is
                configured, the caller passes 0 so an empty retrieval does not waste
                trajectory budget (without mutating the frozen cfg).

        Returns:
            Optional[WindowBudget]: The ledger; None when the window is unknown (the caller
            then falls back to "no cap / no trimming").
        """
        window = getattr(llm, "context_window", None)
        if not window:
            return None
        reserve = cfg.output_reserve(window=window, model_max_output=getattr(llm, "max_output_tokens", None))
        return cls(window=window, output_reserve=reserve, system_tokens=system_tokens, tool_tokens=tool_tokens,
                   rag_ratio=cfg.rag_ratio if rag_ratio is None else rag_ratio)
