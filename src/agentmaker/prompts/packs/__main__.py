"""Chinese-pack self-test: uv run python -m agentmaker.prompts.packs (purely local). Checks the keys map one-to-one and that the whole-set override passes."""

from .. import DEFAULT_PROMPTS
from . import CHINESE_PROMPTS, chinese_registry

missing = set(DEFAULT_PROMPTS.keys()) - set(CHINESE_PROMPTS)
extra = set(CHINESE_PROMPTS) - set(DEFAULT_PROMPTS.keys())
print(f"🔧 default registry: {len(DEFAULT_PROMPTS.keys())} entries; Chinese pack: {len(CHINESE_PROMPTS)} entries")
print(f"🔧 missing keys: {missing or 'none'}; extra keys: {extra or 'none'}")
chinese_registry()   # applying the override validates placeholders + protocol tokens; invalid ones raise PromptError
print("🔧 ✅ Chinese pack override validation passed (placeholders / protocol tokens all present)")
