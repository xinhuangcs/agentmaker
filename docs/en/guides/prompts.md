# Prompt registry

Every prompt the framework sends to the model, from the memory extractor to the ReAct thinking style to each tool's error text, lives in one catalog you can list, read, and override. Nothing is hard-coded out of reach. Reach for this page when you want to see exactly what the framework tells the model, reword a built-in prompt to fit your product's voice, switch the whole set to another language, or add your own prompts to the same enumerable system.

## The catalog

The prompt engine has two pieces. A `PromptTemplate` is a single prompt: its template text plus the `{placeholder}` slots it fills at render time. A `PromptRegistry` is a catalog of those templates, keyed by a `subsystem.name` string such as `memory.extract`, `react.style`, or `tool.error.not_found`. The framework ships one ready-made catalog, `DEFAULT_PROMPTS`, which holds the default (English) text for every built-in prompt.

The design is drift-proof: each subsystem reads its prompts by key from an injected registry, so what the catalog lists is exactly what runs. There is no second copy of the text hidden inside the code.

All four names come straight off the top-level package:

```python
from agentmaker import DEFAULT_PROMPTS, PromptRegistry, PromptTemplate, PromptError
```

A registry answers a few direct questions. `keys()` lists every registered prompt name, `text(key)` returns the current template text, `render(key, **kw)` fills its placeholders, and `as_dict()` returns a `{key: text}` snapshot for printing or export:

```python
print(len(DEFAULT_PROMPTS.keys()), "prompts registered")
print(DEFAULT_PROMPTS.text("chat.persona"))          # You are a helpful assistant.
print(DEFAULT_PROMPTS.render("context.current_question", query="What time is it?"))
```

## Inspect what your agent says

Every agent carries its own registry. `agent.get_prompts()` returns it as a plain `{key: text}` dict, so you can see which prompts the framework builds in, what their keys are, and what they currently say before deciding which to change:

```python
prompts = agent.get_prompts()
print(prompts["react.style"])
for key in sorted(prompts):
    if key.startswith("tool.error."):
        print(key, "->", prompts[key])
```

An agent's registry is its own copy. When you do not pass `prompts=` at construction, the agent makes a copy of `DEFAULT_PROMPTS`, so inspecting or overriding one agent's prompts never touches the global catalog or any other agent.

## Override a prompt

`agent.update_prompts(updates)` rewrites built-in prompts in place. Pass a `{key: new-text}` dict to change specific entries:

```python
agent.update_prompts({"chat.persona": "You are a terse, factual assistant. No small talk."})
```

The change propagates across the whole agent: its harness (the internal layer wrapping every model call) and any internal sub-agents share the same registry copy, so one call updates them together. Because the registry is per-instance, this affects only this agent, not other agents, separately constructed tools, or the process-wide default.

You can also hand `update_prompts` an entire `PromptRegistry` to swap the whole set at once. That is how you change language or wording wholesale, covered below.

!!! note
    Overriding an agent's prompts changes its wording live, with one boundary: a tool's overall description is captured when the tool is constructed, so switching language or tool-related prompts on an already-built agent leaves that snapshot in the old wording (parameter descriptions still update live). To change tool wording cleanly, set the registry before you build the tools. See the language-pack section for the pattern.

## Placeholders and protocol tokens

Two kinds of literal text inside a prompt are load-bearing, and the registry protects both when you override.

A **placeholder** is a `{name}` slot the framework fills at render time, like `{query}` in `context.current_question` or `{schema}` in `harness.schema_instruction`. Rendering substitutes only the declared placeholders; any other braces in the template (a JSON example such as `{"op": "ADD"}`, for instance) are left exactly as written, so protocol examples are never mangled. Rendering with a declared placeholder missing raises `PromptError`.

A **protocol token** is a literal string the framework's parser depends on, such as the `ADD` / `UPDATE` / `DELETE` / `NOOP` operations the memory reconciler emits, or the `JSON` keyword in the structured-output instruction. These are declared as `protected` and must survive any override verbatim, because changing them would break parsing downstream.

When you override a prompt, the registry validates that every declared placeholder and every protocol token is still present in your new text. If one is missing, the override is rejected up front with `PromptError`, moving a failure that would otherwise only surface at run time forward to the moment you make the change:

```python
from agentmaker import PromptError

# Valid: the required {query} placeholder is preserved, so this override is accepted.
agent.update_prompts({"context.current_question": "[User asks]\n{query}"})

# Rejected: dropping {query} would break rendering, so it fails immediately.
try:
    agent.update_prompts({"context.current_question": "The question is above."})
except PromptError as exc:
    print(exc)   # Invalid prompt override: missing placeholder ['query']
```

`PromptError` inherits from `AgentmakerError`, so you can catch it alongside the framework's other errors.

If you build templates directly, `PromptTemplate` exposes the same guarantees. Its constructor takes the text plus optional `variables` and `protected` tuples; `render(**kwargs)` fills the placeholders, and `with_text(new_template)` returns a copy with different text but the same constraints, raising `PromptError` if the new text drops a placeholder or protocol token:

```python
tpl = PromptTemplate("Answer this question:\n{query}", variables=("query",))
print(tpl.render(query="How tall is Everest?"))
reworded = tpl.with_text("Please answer the following.\n{query}")   # keeps {query}, so this is fine
```

## The Chinese language pack

The framework defaults to English, and ships a complete Chinese pack that overrides every built-in prompt in one call while preserving each prompt's placeholders and protocol tokens. Applying it is instant: it is pure text substitution, with no model call. The pack lives in `agentmaker.prompts.packs` as `CHINESE_PROMPTS` (the raw `{key: text}` mapping) and `chinese_registry()` (a helper that returns a fresh Chinese registry). This is [`examples/08_prompt_packs.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/08_prompt_packs.py), copied verbatim:

```python
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
```

The two options reflect a real distinction between the registry's two override methods:

- **`with_overrides` (per registry, isolated).** `chinese_registry()` calls `DEFAULT_PROMPTS.with_overrides(CHINESE_PROMPTS)`, which returns a brand-new catalog with the pack applied and leaves the global `DEFAULT_PROMPTS` untouched. Pass that registry to an agent via `prompts=`, and only that agent speaks Chinese:

    ```python
    from agentmaker import Agent
    from agentmaker.prompts.packs import chinese_registry

    agent = Agent("assistant", llm, prompts=chinese_registry())
    ```

- **`override` (in place, process-wide).** `DEFAULT_PROMPTS.override(CHINESE_PROMPTS)` mutates the shared global catalog, so every component that later reads the default becomes Chinese. Call it once, before you create any agent or tool (the tool-description snapshot caveat above is why timing matters).

You can also switch an already-built agent by handing its whole registry across, since `update_prompts` accepts a `PromptRegistry`:

```python
agent.update_prompts(chinese_registry())
```

To support another language, copy the Chinese pack file and translate each value, keeping the keys, `{placeholder}` names, and protocol tokens verbatim. The same override validation that guards individual edits will reject a pack that drops any of them, so a mistranslation that loses a placeholder fails loudly rather than at run time.

## Register your own prompts

Third-party strategies and tools can bring their own prompts into the same enumerable, overridable, translatable system. `register(key, template, *, variables=(), protected=())` adds a new entry. Use a namespace prefix on the key to avoid clashing with the framework's built-in names, and register on a copy so you do not mutate the global default:

```python
from agentmaker import Agent, DEFAULT_PROMPTS

reg = DEFAULT_PROMPTS.copy()
reg.register("myapp.greeting", "Greet {name} warmly and offer help.", variables=("name",))
agent = Agent("assistant", llm, prompts=reg)

# Your prompt now shows up alongside the built-ins and renders like any other.
print("myapp.greeting" in agent.get_prompts())        # True
print(reg.render("myapp.greeting", name="Ada"))
```

`register` only adds new keys: registering a key that already exists raises `PromptError`, so you cannot overwrite a built-in by accident. To change an existing entry, use `update_prompts` (or the registry's `override` / `with_overrides`) instead.

## Where to go next

- [Agents & workflows](agents.md) for the `prompts=` constructor argument and how an agent's registry flows to its harness and sub-agents.
- [Tools](tools.md) for tool descriptions and error text, all of which are registry entries you can reword.
- [Structured output](structured-output.md), [Memory](memory.md), and [Retrieval & RAG](retrieval-and-rag.md) for the subsystems whose prompts (schema instruction, fact extraction, source-grounded answering) live in this catalog.
- The [Prompts API reference](../reference/prompts.md) for exact signatures.
