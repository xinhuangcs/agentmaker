"""agentmaker.prompts: the framework's master prompt registry.

Collects every built-in prompt the framework feeds to the LLM into one **enumerable, per-entry / whole-set overridable,
validatable** catalog, replacing constants that were scattered around, privately hard-coded, and unchangeable downstream.
The design mirrors LlamaIndex's PromptMixin (get_prompts/update_prompts) plus first-class template objects:

- `PromptTemplate`: one prompt = template text + declared placeholders `variables` + protocol tokens that must be
  preserved verbatim `protected`. Rendering only substitutes the declared `{var}` placeholders (non-placeholder braces
  such as `{...}` in JSON examples are left as-is); overriding validates that "placeholders + protocol tokens are still present".
- `PromptRegistry`: holds a set of PromptTemplates keyed by `subsystem.name` (e.g. `memory.extract` / `react.style` / `tool.error.not_found`).
  Supports `keys()` to list all, `text(key)` to read, `render(key, **kw)` to render, `with_overrides({...})` for a whole-set swap, `override({...})` for in-place change.
- `DEFAULT_PROMPTS`: the framework's default catalog (English). Each component **reads prompts by key from the injected registry**, so "what is listed" == "what is actually used" and they never drift.

Downstream usage (import after release, no source changes):
    agent.get_prompts()                      # list every prompt this agent actually uses
    agent.update_prompts({"memory.extract": "..."})   # override one entry (validates placeholders / protocol tokens)
    eng = DEFAULT_PROMPTS.with_overrides(my_english_pack)    # swap the whole language, then feed it to components / agents
"""

from .defaults import DEFAULT_PROMPTS
from .registry import PromptError, PromptRegistry, PromptTemplate

__all__ = ["PromptTemplate", "PromptRegistry", "PromptError", "DEFAULT_PROMPTS"]
