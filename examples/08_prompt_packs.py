"""Prompt packs: English is the default; switch the whole set to Chinese.

Every built-in prompt lives in a registry. The framework defaults to English; the bundled
Chinese pack overrides all of them in one call, preserving each prompt's placeholders and
protocol tokens. No LLM call, so this runs instantly.

    uv run python examples/08_prompt_packs.py
"""
from agentmaker import DEFAULT_PROMPTS
from agentmaker.prompts.packs import CHINESE_PROMPTS, chinese_registry

# The default registry is English.
print("default (English):", DEFAULT_PROMPTS.text("context.section.memory"))

# Option A: build a Chinese registry and pass it explicitly to an Agent/Tool via prompts=,
# without touching the global default.
zh = chinese_registry()
print("chinese_registry():", zh.text("context.section.memory"))

# Option B (process-wide): apply the Chinese pack before creating any agent or tool, so
# everything that uses the default becomes Chinese. Uncomment to switch globally:
#     DEFAULT_PROMPTS.override(CHINESE_PROMPTS)
print(f"(the Chinese pack has {len(CHINESE_PROMPTS)} entries, one per default prompt)")
