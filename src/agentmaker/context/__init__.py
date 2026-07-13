"""agentmaker.context: context engineering that assembles memory / RAG retrieval candidates into the model window.

memory and RAG are the "suppliers"; this subsystem does "final assembly + quality control": it picks the
candidates that most deserve a place in the window, orders them, controls the budget, and compresses when they
overflow. It does not re-rank (it trusts upstream retrieval to already be ordered) and only does what assembly
alone can do: de-duplicate (MMR), budget per source, structure the layout, and compress as a fallback.

Provides:
    - mmr_select: MMR selection, relevant yet mutually distinct (reuses the vectors retrieval brings back).
    - count_tokens: token estimation (mixed Chinese/English).
    - ContextConfig / ContextSource: budget config (with quota validation) and the source interface.
    - CallableSource: adapts any "(query) -> candidates" callable into a ContextSource.
    - ContextBuilder: the main pipeline: Gather -> MMR -> three-region budget (two-round borrowing) ->
      skeleton assembly; build flattens everything, build_block emits only the dynamic-source block.
    - HistoryCompactor: conversation history compression (cross-session Chat history: LLM summary of old
      turns + keep the most recent few).
    - reducer: loss-aware trimming of trajectories/history (overflow protection for ReAct / Plan / Reflection
      trajectories, each preserving its own lifeline).
    - WindowBudget / WindowBudgetConfig: window-wide accounting that allocates the whole window across output
      reserve / fixed overhead / retrieval block / trajectory in one place.
"""

from ..core.text import count_tokens
from .builder import ContextBuilder
from .history_compactor import DEFAULT_SUMMARY_PROMPT, HistoryCompactor
from .mmr import mmr_select
from .reducer import REDUCERS, reduce_agent, reduce_plan, reduce_reflection, tokens_of
from .sources import CallableSource
from .types import CompactionConfig, ContextConfig, ContextSource, ReducerConfig
from .window_budget import WindowBudget, WindowBudgetConfig

__all__ = ["mmr_select", "count_tokens", "ContextConfig", "ReducerConfig", "CompactionConfig", "ContextSource",
           "CallableSource", "ContextBuilder", "HistoryCompactor", "DEFAULT_SUMMARY_PROMPT",
           "WindowBudget", "WindowBudgetConfig",
           "REDUCERS", "tokens_of",
           "reduce_agent", "reduce_plan", "reduce_reflection"]
