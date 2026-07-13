"""agentmaker.prompts.registry: prompt engine with PromptTemplate (template + validation) and PromptRegistry (catalog).

Handles only how prompts are stored, rendered, and validated on override. It contains no concrete prompt
content (default content lives in defaults.py, prompt packs live in packs/).
"""

import re

from ..core.exceptions import AgentmakerError

# Extract {name} placeholders in a template. Ignore {{ }} escapes and non-identifier braces (e.g. JSON examples
# like {"to":...}), so protocol examples are never matched by mistake.
_VAR_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")


class PromptError(AgentmakerError):
    """Prompt-related error: a missing placeholder at render time, or a lost required placeholder / protocol token on override. Inherits AgentmakerError so callers can catch it uniformly."""


class PromptTemplate:
    """A single built-in prompt: template text + declared placeholders (variables) + protocol tokens that must be preserved verbatim (protected).

    Rendering only substitutes the declared `{var}` placeholders; other braces (JSON examples, etc.) are left
    as-is. Overriding enforces that placeholders and protocol tokens are still present, moving "missing placeholder /
    broken protocol token" failures from run time forward to override time.
    """

    __slots__ = ("template", "variables", "protected")

    def __init__(self, template: str, *, variables: tuple = (), protected: tuple = ()):
        """
        Args:
            template: The template text.
            variables: Placeholder names that must appear (validated on render, required to remain on override).
            protected: Protocol tokens that must be preserved verbatim (required to remain on override, e.g. ReAct's "Action:", Chat's "[TOOL_CALL:").
        """
        self.template = template
        self.variables = tuple(variables)
        self.protected = tuple(protected)

    def render(self, **kwargs) -> str:
        """Fill placeholders from keyword arguments and return the final text; raises PromptError if any declared variable is missing.

        Only the declared `{var}` placeholders are substituted; other braces in the template (e.g. the JSON example
        `{"to":...}`) are left as-is.
        """
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise PromptError(f"Missing placeholder {missing} when rendering prompt (required: {list(self.variables)})")
        return _VAR_RE.sub(lambda m: str(kwargs[m.group(1)]) if m.group(1) in kwargs else m.group(0), self.template)

    def with_text(self, new_template: str) -> "PromptTemplate":
        """Return a new PromptTemplate with different text but the same variables / protected constraints; raises PromptError if the new text drops a placeholder or protocol token."""
        present = set(_VAR_RE.findall(new_template))
        miss_var = [v for v in self.variables if v not in present]
        miss_prot = [p for p in self.protected if p not in new_template]
        if miss_var or miss_prot:
            parts = []
            if miss_var:
                parts.append(f"missing placeholder {miss_var}")
            if miss_prot:
                parts.append(f"missing required protocol token {miss_prot} (changing it would break parsing)")
            raise PromptError("Invalid prompt override: " + "; ".join(parts))
        return PromptTemplate(new_template, variables=self.variables, protected=self.protected)


class PromptRegistry:
    """A catalog of built-in prompts: list all / read / render / override by key.

    with_overrides returns a new catalog with the overrides applied (the original is untouched); override mutates in
    place (all holders of the same instance see it); register adds a new key.
    """

    def __init__(self, prompts: dict):
        """prompts: {key: PromptTemplate}."""
        self._p: dict = dict(prompts)

    def __contains__(self, key: str) -> bool:
        """Support `key in registry`: whether the prompt is registered."""
        return key in self._p

    def get(self, key: str) -> PromptTemplate:
        """Return the PromptTemplate for a key; raises PromptError if it does not exist."""
        try:
            return self._p[key]
        except KeyError:
            raise PromptError(f"Unknown prompt key: {key!r}") from None

    def text(self, key: str) -> str:
        """Return the current template text for a key."""
        return self.get(key).template

    def render(self, key: str, **kwargs) -> str:
        """Render (fill placeholders) by key."""
        return self.get(key).render(**kwargs)

    def keys(self) -> list:
        """All keys (every registered prompt name)."""
        return list(self._p)

    def as_dict(self) -> dict:
        """A {key: current template text} snapshot, for inspection / printing / export."""
        return {k: v.template for k, v in self._p.items()}

    def copy(self) -> "PromptRegistry":
        """Shallow copy (PromptTemplate is immutable, so copying the dict suffices): give each Agent its own copy so they never cross-mutate."""
        return PromptRegistry(self._p)

    def register(self, key: str, template: str, *, variables: tuple = (), protected: tuple = ()) -> None:
        """Register a new prompt (only allowed if the key does not exist), letting third parties bring their custom strategy / tool prompts into the same enumerable / overridable / re-translatable system.

        Use a namespace prefix (e.g. 'myapp.greeting') to avoid clashing with the framework's built-in keys. An
        existing key cannot be re-registered with this method (to prevent accidental overwrite): change an existing
        prompt via override / with_overrides. Typical usage: reg = DEFAULT_PROMPTS.copy(); reg.register('myapp.x', '...');
        agent = Agent(..., prompts=reg), so that get_prompts() lists it fully and your own prompt pack can reach it via with_overrides.

        Args:
            key: The prompt key (subsystem.name).
            template: The template text.
            variables: Placeholder names that must appear (validated on render, required to remain on override).
            protected: Protocol tokens that must be preserved verbatim (required to remain on override).
        """
        if key in self._p:
            raise PromptError(f"Prompt key already exists: {key!r} (use override / with_overrides to change an existing one; use register only to add a new entry)")
        self._p[key] = PromptTemplate(template, variables=variables, protected=protected)

    def _apply(self, target: dict, updates) -> None:
        """Validate updates ({key: text or PromptTemplate}, or another PromptRegistry) against **each target key's existing constraints**, then write them into target.

        Whether a string or a PromptTemplate is passed, only its text is taken and validated through the target key's
        `with_text` (placeholders + protocol tokens must remain), so **you cannot bypass protocol protection by passing
        a PromptTemplate** (any variables/protected carried by the passed PromptTemplate are ignored).
        """
        items = updates.as_dict().items() if isinstance(updates, PromptRegistry) else dict(updates).items()
        for key, val in items:
            if key not in target:
                raise PromptError(f"Unknown prompt key: {key!r} (to override an existing entry use with_overrides / override and check keys() first; to add a new entry use register())")
            text = val.template if isinstance(val, PromptTemplate) else str(val)
            target[key] = target[key].with_text(text)

    def with_overrides(self, updates) -> "PromptRegistry":
        """Return a new catalog with updates applied (the original is unchanged); each override passes with_text validation."""
        merged = dict(self._p)
        self._apply(merged, updates)
        return PromptRegistry(merged)

    def override(self, updates) -> None:
        """**Mutate this registry instance in place**: every component holding the same instance sees the change. Used by Agent.update_prompts to propagate to already-built
        harness / sub-agents after construction. Calling it on the shared singleton DEFAULT_PROMPTS takes effect process-wide (for a global language switch), so when calling on DEFAULT_PROMPTS make
        sure it happens **before** creating any agent / tool (otherwise tool schemas are construction-time snapshots and you get a mix of languages)."""
        self._apply(self._p, updates)
