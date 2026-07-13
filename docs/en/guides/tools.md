# Tools

Tools are the functions your agent can call. A tool exposes a name, a description, and a typed parameter list; the model uses **function calling** (the mechanism by which the model emits a structured request to run a named tool with arguments) to invoke it, and the framework runs the tool and feeds the result back. This guide covers defining tools (the one-line `@tool` decorator or a `Tool` subclass), returning results with `ToolResponse`, grouping tools in a `ToolRegistry`, the built-in tools, permission and confirmation gates, connecting external MCP servers, and selecting from a large tool set at runtime with Tool-RAG.

Everything on this page runs with the [Agent](agents.md) loop. The full runnable example is [`examples/02_tools_and_registry.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/02_tools_and_registry.py):

```python
from agentmaker import Agent, CalculatorTool, ToolRegistry, tool
from agentmaker.testing import ScriptedLLM


@tool
def to_upper(text: str) -> str:
    """Uppercase a string.

    Args:
        text: The input text.
    """
    return text.upper()


registry = ToolRegistry()
registry.register(CalculatorTool())   # built-in: safe arithmetic evaluation
registry.register(to_upper)           # your custom tool

# Script the model's decision to call the calculator, then its final answer.
llm = ScriptedLLM([
    ScriptedLLM.tool_call("calculator", {"expression": "(3 + 4) * 5"}),
    "The result is 35.",
])
agent = Agent("assistant", llm, tool_registry=registry)
print(agent.run("Compute (3 + 4) * 5").final_output)
```

`ScriptedLLM` is a test double that replays a fixed script, so the example needs no API key and no network. With a real model, replace it with `LLMClient(...)` and the model decides when to call each tool.

## Define a tool with `@tool`

The name produced by decorating a type-annotated function with `@tool` is a `Tool` object, ready to pass to `Agent(tools=[...])` or `registry.register(...)`. The schema the model sees is inferred for you:

- **Parameter names, types, defaults, and required-ness** come from the function signature. Python types map to JSON Schema types: `str` to `string`, `int` to `integer`, `float` to `number`, `bool` to `boolean`, `list`/`tuple` to `array`, `dict` to `object`.
- **The tool description** is the first paragraph of the docstring.
- **Parameter descriptions** come from `Annotated` metadata if present, otherwise from the docstring `Args:` section entry of the same name.

Both styles work. Use `Annotated[T, "description"]` to attach a per-parameter description inline:

```python
from typing import Annotated
from agentmaker import tool


@tool
def get_weather(city: Annotated[str, "city name"], days: int = 3) -> str:
    """Query the weather for a city over the next few days."""
    ...
```

A parameter with a default (like `days=3` above) is treated as optional; a parameter without one is required.

The decorator also takes optional keyword flags:

```python
@tool(requires_confirmation=True)
def delete_file(path: str) -> str:
    """Delete a file at the given path."""
    ...
```

- `requires_confirmation`: set `True` for high-risk actions (writes, deletes, sending requests) so the call passes through the confirmation gate before execution.
- `external_content`: set `True` when the result is content from an external source, so the framework wraps it in an anti-injection guardrail before feeding it back to the model.
- `supports_parallel`: set `True` for read-only, concurrency-safe tools so they may run concurrently with other parallelizable calls in the same turn. A run with any numeric LLM, tool, or token cap executes tool calls serially.

Parallel batching is used only when the active run has no exact `max_tool_calls` cap. If `RunPolicy.max_tool_calls` is set, eligible calls execute serially so the framework can perform an exact admission check before each real execution.

You can also name a tool explicitly with `@tool(name=..., description=...)`; the name defaults to the function name.

!!! note "Async tools work the same way"
    `@tool` supports `async def` functions natively. The framework awaits an async tool and dispatches a synchronous one onto a thread pool, so the same tool works in both sync and async agent loops.

`@tool` fails loud at definition time rather than degrading silently: a missing type annotation, a variadic parameter (`*args`/`**kwargs`), or an annotation that cannot be mapped to a JSON Schema type raises `ToolRegistrationError`. If you hit that limit, write a `Tool` subclass instead.

!!! note "Registering a plain function without decorating"
    If you would rather not decorate, `registry.register_callable(func)` infers the schema from the signature exactly like `@tool`. For a function that takes the whole parameter dict and hand-written parameters, use `registry.register_function(func, name, description, parameters)`.

## Subclass `Tool`

When you need state, a custom schema, or logic the decorator cannot express, subclass `Tool` directly. Implement `get_parameters()` (return a list of `ToolParameter`) and `run()` (return a `ToolResponse`):

```python
from agentmaker import Tool, ToolParameter, ToolResponse


class ReverseTool(Tool):
    def __init__(self):
        super().__init__("reverse", "Reverse a string.")

    def get_parameters(self):
        return [ToolParameter("text", "string", "The text to reverse.")]

    def run(self, parameters: dict) -> ToolResponse:
        return ToolResponse.ok(parameters["text"][::-1])
```

A subclass must implement at least one of `run` (synchronous) or `arun` (native async). The default `arun` dispatches `run` onto a thread pool, so a synchronous tool works unchanged in an async agent. Native async tools (such as an HTTP or subprocess call) override `arun` and await the real call. The class-level flags `requires_confirmation`, `external_content`, and `supports_parallel` mean the same as the decorator keywords above.

!!! note "Threading contract"
    The execution chain may dispatch `run` onto a different worker thread each time. Do not hold thread-bound resources (like a single shared `sqlite3` connection) on the tool instance; build them lazily per thread, or create them with `check_same_thread=False` plus your own lock.

### `ToolParameter`

`ToolParameter` describes one parameter. Its fields:

| Field | Meaning |
| --- | --- |
| `name` | Parameter name. |
| `type` | JSON Schema type string (`string`, `integer`, `number`, `boolean`, `array`, `object`). |
| `description` | Parameter description shown to the model. |
| `required` | Whether the parameter is required (default `True`). |
| `default` | Default value, meaningful only when not required. |
| `schema` | A full JSON Schema for this parameter; when given it is used verbatim, preserving `enum`, array `items`, or nested objects. |

Use `schema` for anything a plain `type` cannot express, such as an enum:

```python
ToolParameter("action", "string", "The action to run",
              schema={"type": "string", "enum": ["read", "append"], "description": "The action to run"})
```

## Return values: `ToolResponse`

Every tool returns a `ToolResponse`. If a `@tool` function returns a plain `str`, the framework wraps it in a success response for you; otherwise construct one explicitly. It has three fields:

- `text`: the result text the model reads (always present).
- `status`: `"success"`, `"partial"` (succeeded but incomplete, for example truncated output), or `"error"`.
- `data`: optional structured data for programmatic use; the model reads only `text`, not `data`.

Three constructors cover the common cases:

```python
ToolResponse.ok("42", data=42)                 # status="success"
ToolResponse.partial("first 4000 chars ...")   # status="partial"
ToolResponse.error("query must not be empty")  # status="error"
```

Returning `ToolResponse.error(...)` is the idiomatic way to report a recoverable failure: the error text is fed back to the model so it can adjust its arguments and retry. A raised exception does not crash the run either, it is caught at the execution layer and fed back the same way, but returning an explicit error is clearer and preserves the `status` and `data` fields you control.

## The registry

A `ToolRegistry` holds the tools an agent may call, keyed by name. `register` one at a time, or `register_all` in a batch:

```python
from agentmaker import ToolRegistry, CalculatorTool, SearchTool

registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register_all([SearchTool(), to_upper])
```

Tool names must match the function-calling name rule `^[a-zA-Z0-9_-]{1,64}$` (shared by OpenAI and Anthropic); an illegal name raises `ToolRegistrationError`. Registering a duplicate name raises by default; pass `overwrite=True` (or `register_all(..., on_conflict="skip"/"overwrite")`) when replacement is intended.

The registry renders the tools into the forms the loop needs:

- `get_catalog()`: a cheap `- name: description` catalog, one line per tool.
- `get_tools_description()`: the full textual description including the parameter list.
- `to_openai_schema()`: the `tools` argument for function calling.

`get_tools_description()` and `to_openai_schema()` accept an optional `names` list to render only a subset in a given order, which is what Tool-RAG uses (see [Runtime tool selection](#runtime-tool-selection-tool-rag) below).

To run a tool directly (the [Agent](agents.md) loop does this for you), use `execute_tool`, which validates the arguments against the schema first and returns a `ToolResponse`:

```python
registry.execute_tool("calculator", {"expression": "2 + 2"})
```

Both the schema sent to the model and the schema used to validate incoming arguments come from the same source, so there is no drift: an argument that fails validation is returned as an error `ToolResponse` for the model to fix, not raised.

### Wiring a registry into an agent

The `Agent` accepts either a `tools` list (a convenience entry point, normalized into a registry internally) or a `tool_registry` you built yourself. They are mutually exclusive:

```python
agent = Agent("assistant", llm, tools=[to_upper, CalculatorTool()])   # convenience
agent = Agent("assistant", llm, tool_registry=registry)               # explicit registry
```

An agent with no tools is plain question-answering.

## Built-in tools

The framework ships a few general-purpose tools with no business logic.

### `CalculatorTool`

Evaluates math expressions safely by parsing them into an abstract syntax tree and evaluating only whitelisted operators, so there is no `eval` and no arbitrary code execution. It supports `+ - * / // % **`, unary sign, the functions `sqrt`, `abs`, `round`, `log`, `sin`, `cos`, and the constants `pi` and `e`. It has one parameter, `expression`, and needs no constructor arguments (like every built-in, it accepts an optional `prompts=` to localize its user-facing strings):

```python
from agentmaker import CalculatorTool

registry.register(CalculatorTool())   # tool name: "calculator"
```

### `SearchTool`

Web search with automatic multi-source fallback: it tries Tavily, then DuckDuckGo, then Brave, then SerpAPI, moving to the next source whenever one has no library installed, no key configured, or fails. Only if all fail does it return an error. Keys are read from the environment (`TAVILY_API_KEY`, `BRAVE_API_KEY`, `SERPAPI_API_KEY`); DuckDuckGo needs no key. It has one parameter, `query`.

```python
from agentmaker import SearchTool

registry.register(SearchTool(max_results=5))   # tool name: "search"
```

`SearchTool` sets `external_content = True` (results are external and are wrapped in the anti-injection guardrail) and `supports_parallel = True` (each call is an independent read-only request, so the model can run several searches in one turn concurrently).

### `CLITool`

Wraps "run one allowlisted local command" as a tool. Because a CLI is high-risk, safety is the core design: it is deny-by-default (only programs you list are allowed), resolves those programs to absolute paths at construction, never uses `shell=True` (arguments are tokenized with `shlex` and unquoted shell operators are refused), applies a dangerous-argument gate against high-risk interpreter, Git, network, and filesystem flags, and passes only a minimal environment (`PATH`, `HOME`, `LANG`). Timeout, cancellation, excessive output, or descendants holding output pipes kills and reaps the captured process group; output is bounded while it is read. CLI output is treated as external content and receives anti-injection delimiters before the model sees it. The process-group lifecycle contract relies on POSIX `setsid` / `killpg`, so `CLITool` is supported on POSIX hosts; use an application-specific sandboxed tool for Windows command execution. It is marked `requires_confirmation = True`. Its tool name is `shell` and it takes one parameter, `command`.

```python
from agentmaker import CLITool

registry.register(CLITool(allowed_commands=["git", "ls", "grep"], timeout=10.0, max_output_chars=4000))
```

You can override the dangerous-argument gate with the `arg_policy` callback, and the subprocess environment with `env`. An allowlist and argument denylist are not an OS sandbox: commands such as Git can execute hooks or application/user configuration, so the confirmation callback must review the complete command and high-risk deployments should add a container or platform sandbox.

### `NotesTool`

Lets an agent read and append a note file within a restricted directory, so it can keep progress, plans, and decisions across sessions. Construction creates a missing `root` with mode `0700`; an existing root must be a real directory owned by the current user and grant no group/other permissions (e.g. `0700`). Absolute paths and `..` are refused; every parent component and final path is opened without following symlinks; non-regular files and files with additional hard links are rejected. Its tool name is `notes`, and its parameters are `action` (`read` or `append`), `path` (relative to `root`), and `content` (for `append`).

```python
from agentmaker import NotesTool

registry.register(NotesTool(root="./agent_notes"))
```

`NotesTool` requires a POSIX environment with directory-relative file operations and `O_NOFOLLOW`; construction raises `OSError` when those facilities are unavailable. Appends take a non-blocking per-file `flock`, so separate cooperating instances serialize the size check and write against `max_file_bytes`; lock contention returns a tool error instead of waiting indefinitely. The lock is advisory, so unrelated writers remain the application's responsibility.

`NotesTool` uses action-level confirmation: `append` writes to disk and requires confirmation, while `read` is read-only and runs without a confirmation prompt. A successful `read` is marked as external content, so the note body is wrapped in anti-injection delimiters before the model sees it; the local acknowledgement returned by `append` is not marked external.

## High-risk actions: the confirmation gate

Tools flagged `requires_confirmation` (or those, like `NotesTool`, that decide per action) must clear a confirmation callback before they run. The callback has the signature `(tool, parameters) -> bool`; the tool runs only when it returns `True`. Pass it to the agent as `confirm`:

```python
from agentmaker import Agent, cli_confirm

agent = Agent("assistant", llm, tools=[CLITool(allowed_commands=["ls"])], confirm=cli_confirm)
```

`cli_confirm` is the built-in command-line prompt (a `y/n` question on stdin). If you pass no `confirm`, a high-risk call is safely refused by default (the model receives a readable error rather than the action running unconfirmed). For server or asynchronous approval flows, use human-in-the-loop (HITL, the pattern where a run pauses for a person to approve or edit a pending action); see [Guardrails & HITL](guardrails-and-hitl.md).

## Tool permissions

`ToolPermissions` declares which tools an agent may call, as allow and deny lists. It is judged along two dimensions, the tool **name** and the tool **origin**. Origin is the true root of trust: a name can be spoofed by a remote server (naming a malicious tool `search` to piggyback on your allowlist), whereas origin is stamped by the framework (`"builtin"`, or `"mcp:{namespace}"` for MCP tools) and cannot be forged by the tool definition.

The decision rule is "deny wins, then an allowlist restricts":

- A match on `deny` or `deny_origins` denies immediately (highest priority).
- If an allowlist is enabled (either `allow` or `allow_origins` is set), a tool must match the allowed names or origins to be permitted.
- If no allowlist is enabled, the tool is permitted (subject only to the deny lists).

`allow=None` means that dimension enables no allowlist ("no restriction"); `allow=[]` means an empty allowlist that denies everything. Pass a `ToolPermissions` to the agent as `permissions`:

```python
from agentmaker import Agent, ToolPermissions

permissions = ToolPermissions(allow_origins={"builtin"}, deny={"shell"})
agent = Agent("assistant", llm, tool_registry=registry, permissions=permissions)
```

Permissions are enforced at the execution gate: a denied tool is rejected outright and is not even offered for confirmation.

## MCP integration

MCP (Model Context Protocol, Anthropic's open standard for exposing tools to models, sometimes called "USB-C for AI") lets you connect a server that publishes a set of tools and adapt each one into an agentmaker `Tool`. `MCPClient` manages the connection and lists the tools; each becomes an `MCPTool` you register like any other tool. Install the lazily imported integration with `uv add "agentmaker[mcp]"`.

Two transports are supported. Use `async with` to manage the connection lifecycle, and call the tools while the block is alive:

```python
from agentmaker import MCPClient, ToolRegistry

registry = ToolRegistry()

# stdio: run a local server as a subprocess
async with MCPClient(command="python", args=["my_server.py"], namespace="calc") as client:
    tools = await client.load_tools()               # [MCPTool, ...], one per server tool
    registry.register_all(tools, on_conflict="skip")
    # ... use the tools while the connection is alive ...
```

For a remote server, pass `url` instead of `command` (the two are mutually exclusive), plus optional `headers` or an `httpx.Auth` via `auth` for OAuth:

```python
async with MCPClient(url="https://mcp.example.com/mcp", namespace="calendar", auth=my_oauth) as client:
    ...
```

Key safety points, all handled for you:

- `namespace` is **required** and is the trust root. Each tool's display name becomes `"{namespace}_{original name}"` and its origin is stamped `"mcp:{namespace}"`. The namespace is chosen by you, never derived from the server's self-reported name (which is attacker-controlled). This also prevents collisions when two servers each expose a `search` tool.
- `requires_confirmation` defaults to `True` for loaded MCP tools, since remote tools are untrusted; downgrade to `False` only after you have vetted the server.
- `MCPTool` sets `external_content = True`, so results are wrapped in the anti-injection guardrail.
- A valid root `inputSchema` is preserved for both model exposure and local argument validation, including local `$ref`/`$defs` and root constraints. Oversized, excessively deep, or externally-referenced schemas are refused before use.
- Each tool definition gets a fingerprint (a sha256 over its remote name, description, and input schema). `expected_fingerprints` is an exact `{display_name: sha256}` pin set: a mismatch, any returned tool absent from the set, or any pinned tool absent from the server makes `load_tools` fail.
- `max_tools` rejects an oversized remote tool catalog. `max_result_chars` bounds retained text; structured content is copied under a related byte/depth/node budget and is replaced by `{"truncated": true}` when it cannot fit. No unbounded serializer is invoked before these checks.
- Remote URLs require HTTPS by default. Cleartext HTTP is accepted for loopback endpoints; a non-loopback endpoint requires the explicit `allow_insecure_http=True` opt-in. Embedded URL credentials are rejected, and descriptions/results have ASCII control characters and Unicode format controls removed (the joiners and soft hyphen that emoji and complex scripts need are kept).
- `timeout` covers initialization, `list_tools`, and every `call_tool`; `None` disables it and should be reserved for an application that supplies its own cancellation policy.

Registering with `on_conflict="skip"` (rather than raising on the first name collision) keeps a single duplicate from aborting the whole load loop.

## Runtime tool selection (Tool-RAG)

Once an agent has many tools, putting every tool's full schema in the prompt is expensive and degrades accuracy. Tool-RAG (RAG is retrieval-augmented generation, retrieving relevant items instead of sending everything) retrieves only the most relevant tools for the current input and expands just that subset. `ToolRetriever` indexes each tool's name, description, and parameter names into a shared retriever and returns the top matches:

```python
from agentmaker import ToolRetriever

# `retriever` is a HybridRetriever; see the Retrieval & RAG guide for how to build one.
tool_retriever = ToolRetriever(registry, retriever, top_k=8, always_include=("tool_search",))
tool_retriever.index()                                     # load every tool's name + description

names = tool_retriever.retrieve("convert between currencies")     # list of tool names, most relevant first
schema = tool_retriever.schema_for("convert between currencies")  # function-calling schema for that subset
```

Three knobs keep it reliable:

- `always_include`: tool names that bypass retrieval and are always in the subset (things that must never be squeezed out by the top-k cutoff).
- `on_empty`: the zero-hit fallback, defaulting to `"all"` (fall back to the full catalog) so the model is never handed zero tools.
- `selector`: an optional truncation-strategy callback for a score threshold or knee-point cutoff instead of a fixed top-k.

Pass a retriever to the agent as `tool_retriever` and it selects the relevant subset for each turn's input automatically:

```python
agent = Agent("assistant", llm, tool_registry=registry, tool_retriever=tool_retriever)
```

One-shot preselection has a blind spot: in a multi-step task, which tool step two needs may depend on step one's output. `ToolSearchTool` closes that gap by making tool search itself a tool the model can call mid-run. It returns a catalog of matching tools plus a `discovered` list, and the loop merges those tools into the usable set for the rest of the run:

```python
from agentmaker import ToolSearchTool

registry.register(ToolSearchTool(tool_retriever, top_k=5))   # tool name: "tool_search"
```

Make `tool_search` an `always_include` entry (as in the retriever above) so it is always available. See [Retrieval & RAG](retrieval-and-rag.md) for building the underlying retriever.
