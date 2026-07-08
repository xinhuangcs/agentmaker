"""Prompt-engine self-test: uv run python -m agentmaker.prompts (purely local, no network / key needed)."""

from . import DEFAULT_PROMPTS, PromptError

print(f"🔧 registered {len(DEFAULT_PROMPTS.keys())} built-in prompts")
print("🔧 render example:", DEFAULT_PROMPTS.render("memory.reconcile_user", fact="The user now lives in Beijing", listing="1. The user now lives in Shanghai"))
overridden = DEFAULT_PROMPTS.with_overrides({"chat.persona": "You are a helpful assistant."})
print("🔧 after override:", overridden.text("chat.persona"))
try:
    # memory.reconcile's protocol tokens are ADD/UPDATE/DELETE/NOOP (operation names the model must emit verbatim);
    # overriding it with text that drops those tokens triggers the protocol-token validation error (not an "unknown key" error).
    DEFAULT_PROMPTS.with_overrides({"memory.reconcile": "arbitrary text with the protocol tokens removed"})
except PromptError as e:
    print(f"🔧 ✅ invalid override rejected (missing protocol tokens): {e}")
