"""agentmaker.context.reducer: loss-aware trimming of trajectories/history for per-paradigm overflow protection.

Complementary to HistoryCompactor (which specifically compresses cross-session conversation history): this
module trims the paradigm's own trajectory for the unified-loop Agent's tool trajectory, Plan step results, and
Reflection drafts. Core principle: only trigger when approaching the context window budget, and drop the least
important signals first while preserving the paradigm's lifeline. A generic, undifferentiated summary would
strip out Reflection's "past critique points" and Plan's "exact step numbers", and losing those breaks the
paradigm.

The trajectory token budget is provided by the overall window accounting (Harness.reduce ->
WindowBudget.trajectory_budget, see window_budget.py); this module no longer estimates it by proportion on its
own. It is only responsible for how to trim loss-aware once the budget is known.

Shared primitives (write once):
    - tokens_of: estimate the total token count of several texts.
Per-paradigm preserve policies (reduce_agent / reduce_plan / reduce_reflection): decide what to keep and what
to summarize.

The three trim functions are async-native (the framework has only a single async implementation internally): the
summary is injected by the caller as an async callback `summarize(text, instruction) -> str`. Harness passes in
a version that goes through `acall_llm`, so the compression cost is included in hooks / tracer / RunPolicy (see
Harness._asummarize). On failure it should return an empty string (each paradigm decides its own degraded
wording, and it must never raise and break the flow).

Fallback: if the parts that must be kept (protected head + the most recent few units) themselves exceed the
budget, raise ContextWindowExceeded (an actionable error, never a silent truncation).
"""

import json
from typing import Awaitable, Callable, List

from ..core.exceptions import ContextWindowExceeded
from ..core.multimodal import content_tokens
from ..core.text import TokenCounter, count_tokens

# Summary callback type (async): (text to compress, summary instruction) -> summary text (empty string on failure)
Summarize = Callable[[str, str], Awaitable[str]]

# Per-paradigm summary instructions: preserve targets differ, so each has its own. This is the key to being
# loss-aware: the summary must know what to keep.
_AGENT_INSTRUCTION = (
    "Compress the following executed tool-call trajectory (model-initiated calls, tool return results) into a "
    "concise summary. Preserve the key facts already found, the calls already attempted and failed, and the "
    "intermediate conclusions already reached, and drop redundancy. Output only the summary itself.")
_PLAN_INSTRUCTION = (
    "Compress the results of the following completed steps into a summary. Be sure to preserve verbatim all key "
    "numbers, dates, IDs, names, and explicit conclusions (later steps and the final synthesis will reference "
    "them), removing only narrative redundancy. Output only the summary itself.")
_REFLECTION_INSTRUCTION = (
    "Merge the following past review comments into a de-duplicated list of \"suggestions already made\". Keep "
    "each concrete, actionable suggestion, without repetition. Output only the list itself.")

# Marker added to the summary block after trimming (so a human/model reading it knows it has been compressed).
_AGENT_MARK = "[Attempted Steps Summary]"
_PLAN_MARK = "[Earlier Step Results Summary]"
_REFLECTION_MARK = "[Past Critique Points]"
_FAIL_MARK = "(compression of earlier content failed, omitted)"


def tokens_of(*texts: str, counter: TokenCounter = count_tokens) -> int:
    """Estimate the total token count of several texts (None treated as empty string); counter is pluggable (defaults to count_tokens)."""
    return sum(counter(t or "") for t in texts)


def _mandatory_or_raise(head_recent_tokens: int, budget: int, what: str) -> None:
    """Raise ContextWindowExceeded when the must-keep parts (protected head + recent units) already exceed the budget (fail loud, no silent truncation)."""
    if head_recent_tokens > budget:
        raise ContextWindowExceeded(
            f"{what}: the parts that must be kept are about {head_recent_tokens} tokens, exceeding the available "
            f"budget for this model of {budget} tokens. This task is too large for this model: please split the "
            "task, reduce single-step output length, or switch to a model with a larger context window.")


def _summary_block(mark: str, summary: str, room: int, counter: TokenCounter = count_tokens) -> str:
    """Build a "marker + summary" text and truncate it to within room tokens, guaranteeing the trimmed total stays under budget (the summary LLM's output length is uncontrollable and must be bounded).

    room <= marker tokens: return an empty string (not even the marker fits, so the caller skips this block and
    keeps only the mandatory parts). If the summary is short enough it fits whole; if too long, reserve tokens
    for the ellipsis, then truncate by character ratio and append the ellipsis (`…` at the tail of Chinese text
    can be tokenized as a separate token, so without reserving it the result would exceed room by 1). Finally do
    one more token-estimate check at the end; if it still exceeds, drop the block. This hard-guarantees the
    returned text's token count <= room.
    """
    mark_tokens = tokens_of(mark, counter=counter)
    if room <= mark_tokens:
        return ""
    if tokens_of(summary, counter=counter) <= room - mark_tokens:    # short enough: fits whole, no ellipsis needed
        return mark + summary
    ell = "…"
    room_for_body = room - mark_tokens - tokens_of(ell, counter=counter)     # reserve tokens for the ellipsis
    if room_for_body < 1:                                   # not even "marker + 1 char + ellipsis" fits: skip this block
        return ""
    ratio = len(summary) / max(tokens_of(summary, counter=counter), 1)       # estimate how many characters per token
    content = mark + summary[:max(1, int(room_for_body * ratio))].rstrip() + ell
    return content if tokens_of(content, counter=counter) <= room else ""    # estimate safety net: if still over, drop it, never exceed room


def _msg_tokens(m: dict, counter: TokenCounter = count_tokens) -> int:
    """Estimate the token count of one message: content + the JSON payload of tool_calls (the arguments in a function-call trajectory can be large, so ignoring them systematically underestimates)."""
    n = content_tokens(m.get("content"), counter)   # multimodal-safe (flat estimate per image part)
    if m.get("tool_calls"):
        n += tokens_of(json.dumps(m["tool_calls"], ensure_ascii=False), counter=counter)
    return n


def _agent_units(tail: List[dict]) -> List[List[dict]]:
    """Slice this turn's tool trajectory into atomic units: an assistant (with tool_calls) absorbs the consecutive role:"tool" results that follow it into one group.

    The two cannot be split (an orphan tool_call_id is rejected by the vendor protocol); every other message
    (a nudge user, etc.) forms its own group.
    """
    units: List[List[dict]] = []
    for m in tail:
        if m.get("role") == "tool" and units and units[-1][0].get("tool_calls"):
            units[-1].append(m)
        else:
            units.append([m])
    return units


def _render_msg(m: dict) -> str:
    """Render one function-call message into a single line of text for the summary LLM to read (tool_calls appended as JSON)."""
    calls = f" tool_calls={json.dumps(m['tool_calls'], ensure_ascii=False)}" if m.get("tool_calls") else ""
    return f"{m['role']}: {m.get('content') or ''}{calls}"


async def reduce_agent(messages: List[dict], *, summarize: Summarize, budget: int,
                       keep_recent_steps: int = 3, turn_start: int = 0,
                       counter: TokenCounter = count_tokens) -> List[dict]:
    """Trim the unified-loop (Agent) function-call trajectory.

    Protected region = all initially assembled messages (messages[:turn_start]: system + compacted conversation
    history + RAG blocks + current user, delimited by index rather than by "find the first user". The unified
    loop's trajectory contains conversation history, so the first user is an old history message and delimiting
    by it would summarize away the current question). The trimmable region is the tool trajectory newly added
    this turn afterward.

    The trimmable region is sliced into atomic units (see _agent_units): the most recent keep_recent_steps units
    are kept verbatim, and earlier units are summarized into a single system "Attempted Steps Summary". If not
    over budget, returns unchanged (the same object, so the caller can tell whether trimming happened).

    Args:
        messages: the Agent's state.messages.
        summarize: async summary callback (see module header).
        budget: trajectory token budget (WindowBudget.trajectory_budget, includes the protected region).
        keep_recent_steps: how many atomic units to keep at the tail, default 3.
        turn_start: this turn's start index (ExecutionState.meta["turn_start"], the count of initially assembled messages).
    """
    if keep_recent_steps < 0:
        raise ValueError(f"keep_recent_steps must be >= 0, got {keep_recent_steps}")
    if sum(_msg_tokens(m, counter) for m in messages) <= budget:
        return messages
    head = messages[:turn_start]
    units = _agent_units(messages[turn_start:])
    recent_units = units[-keep_recent_steps:] if keep_recent_steps else []
    old_units = units[:-keep_recent_steps] if keep_recent_steps else units
    recent = [m for u in recent_units for m in u]
    protected = sum(_msg_tokens(m, counter) for m in head + recent)     # protected token count (shared by the check and the room calculation)
    _mandatory_or_raise(protected, budget, "Agent tool trajectory")
    if not old_units:                                          # protected part is already everything and not over budget (should not reach here): return unchanged as a safeguard
        return messages
    convo = "\n".join(_render_msg(m) for u in old_units for m in u)
    content = _summary_block(_AGENT_MARK, await summarize(convo, _AGENT_INSTRUCTION) or _FAIL_MARK, budget - protected, counter)
    mid = [{"role": "system", "content": content}] if content else []   # if there is no room for the summary block, keep only the protected region + recent units
    return head + mid + recent


async def reduce_plan(history: List[str], *, summarize: Summarize, budget: int, keep_recent: int = 3,
                      counter: TokenCounter = count_tokens) -> List[str]:
    """Trim Plan step results: keep the most recent keep_recent step results verbatim and summarize earlier ones into a single entry (preserving key numbers/conclusions).

    Returns unchanged if not over budget. Each history entry looks like "Step N: ...\\nResult: ...". See the
    module header for summarize.
    """
    if keep_recent < 0:                                 # a negative value would make history[-keep_recent:] slice in reverse, counterintuitive behavior
        raise ValueError(f"keep_recent must be >= 0, got {keep_recent}")
    if tokens_of(*history, counter=counter) <= budget:
        return history
    recent = history[-keep_recent:] if keep_recent else []
    old = history[:-keep_recent] if keep_recent else history
    protected = tokens_of(*recent, counter=counter)    # protected (recent) token count, shared by the check and the room calculation
    _mandatory_or_raise(protected, budget, "Plan step results")
    if not old:
        return history
    content = _summary_block(_PLAN_MARK, await summarize("\n\n".join(old), _PLAN_INSTRUCTION) or _FAIL_MARK, budget - protected, counter)
    return [content, *recent] if content else list(recent)   # if there is no room for the summary, keep only the recent step results


async def reduce_reflection(entries: List[dict], *, summarize: Summarize, budget: int,
                            counter: TokenCounter = count_tokens) -> List[dict]:
    """Trim the Reflection trajectory: keep the latest answer verbatim + summarize past critiques into a de-duplicated list of "suggestions already made", dropping superseded old drafts.

    entries is a list of {"kind": "draft"|"critique"|"refine", "text": ...} (dict rather than tuple, so it passes
    JSON checkpoints); returns a shorter list of the same structure. The paradigm's lifeline is the "latest
    answer" (which it keeps refining) + the "past critique points" (to avoid re-raising suggestions already
    made); both are kept and old drafts are dropped. See the module header for summarize.
    """
    if tokens_of(*(e["text"] for e in entries), counter=counter) <= budget:
        return entries
    answers = [e for e in entries if e["kind"] in ("draft", "refine")]
    critiques = [e["text"] for e in entries if e["kind"] == "critique"]
    latest = answers[-1] if answers else entries[-1]   # latest answer (fall back to the last entry if none)
    protected = tokens_of(latest["text"], counter=counter)   # protected (latest answer) token count, shared by the check and the room calculation
    _mandatory_or_raise(protected, budget, "Reflection latest answer")
    out: List[dict] = []
    if critiques:
        content = _summary_block(_REFLECTION_MARK, await summarize("\n\n".join(critiques), _REFLECTION_INSTRUCTION) or _FAIL_MARK, budget - protected, counter)
        if content:                                          # if there is no room for the critique points, keep only the latest answer
            out.append({"kind": "critique", "text": content})
    out.append(latest)
    return out


# Dispatch table from Harness.areduce's kind -> trim function (async).
REDUCERS: dict[str, Callable] = {"agent": reduce_agent, "plan": reduce_plan, "reflection": reduce_reflection}
