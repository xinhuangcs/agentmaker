"""agentmaker.tools.registry: the tool registry.

Centrally manages a group of Tools: register, look up, list, and render them into
machine-readable forms:
get_catalog() (a name+description catalog only, cheap, kept resident in the system prompt,
tier 1 of progressive disclosure),
get_tools_description() (full textual description including parameters: for offline rendering
and context-budget estimation),
to_openai_schema() (the tools argument for OpenAI function calling).
The latter two can render a subset by names: when there are hundreds of tools, pair with
Tool-RAG to expand only the retrieved top-k (the seam lives here; the retrieval implementation
is added on demand).
"""

import asyncio
import inspect
import logging
import re
from typing import Callable, Dict, List, Optional

from ..core.exceptions import ToolRegistrationError
from ..prompts import DEFAULT_PROMPTS
from .base import ConfirmCallback, Tool, ToolParameter, _TOOL_NAME_RE   # single source of truth for the name rule lives in base (shared by from_callable / decorator)
from .response import ToolResponse

logger = logging.getLogger(__name__)   # full tracebacks of tool-execution exceptions go here (the library configures no handler; the host takes over, see the NullHandler in agentmaker/__init__)

_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_tool_name(name: str) -> str:
    """Normalize a name into ^[a-zA-Z0-9_-]{1,64}$: replace illegal chars (dot / space / slash, etc.) with _, truncate over 64, fall back if empty.

    Use for uncontrolled tool names from external sources (such as an MCP server), best-effort
    normalizing them into legal names; tools registered directly within this framework should
    use a legal name from the start: register validates and raises on an illegal one.
    """
    cleaned = _INVALID_NAME_CHARS.sub("_", name or "")[:64]
    return cleaned or "tool"


def validation_error(schema: dict, parameters: dict, *, prompts=None) -> Optional[str]:
    """Validate LLM input parameters against the tool's JSON Schema before execution; return a readable error on mismatch, None on pass.

    It uses the exact same schema (produced by `_param_schema`) that `to_openai_schema` sends
    to the LLM, so there is zero drift: validate against the same shape the model was told to
    produce. additionalProperties is not set, so extra keys do not error (tolerating the LLM
    occasionally passing more); it only checks for missing required fields and type mismatches.
    A failed validation is a soft failure (_resolve returns an error fed back for the model to
    fix its parameters, no raise). jsonschema is imported lazily. Error text (parameter-path
    prefix / separator) is taken from prompts (DEFAULT_PROMPTS if not passed), so it can be
    swapped to another language along with the whole registry.
    """
    from jsonschema import validators
    p = prompts or DEFAULT_PROMPTS
    validator = validators.validator_for(schema)(schema)
    errors = sorted(validator.iter_errors(parameters), key=lambda e: list(e.path))
    if not errors:
        return None
    return p.text("tool.validation_sep").join(_format_validation_error(e, p) for e in errors)


def _format_validation_error(error, prompts) -> str:
    """Format one jsonschema error into readable text: include the parameter path if present; for a missing required field (error at the root) use the original message (which already names the parameter)."""
    path = ".".join(str(p) for p in error.path)
    return prompts.render("tool.validation_field", path=path, message=error.message) if path else error.message


class _FunctionTool(Tool):
    """Wrap a plain function as a Tool so function tools and class tools are treated uniformly in the registry.

    Internal type, used only by ToolRegistry.register_function; not exposed publicly.
    """

    def __init__(self, func: Callable[[dict], str], name: str, description: str,
                 parameters: List[ToolParameter], requires_confirmation: bool = False,
                 supports_parallel: bool = False):
        """
        Args:
            func: the wrapped function; takes a parameter dict and returns a string result (or a ToolResponse); may be a sync or async function.
            name: the tool name.
            description: the tool description.
            parameters: the list of parameter definitions.
            requires_confirmation: whether it is high-risk and needs confirmation before execution (functions that write to disk / send requests / run commands should set True).
            supports_parallel: set True for read-only, concurrency-safe functions to allow concurrent execution with other parallel tools in the same round (see the Tool class attribute).
        """
        super().__init__(name, description)
        self._func = func
        self._is_async = inspect.iscoroutinefunction(func)   # async functions go through arun (await), sync ones through run (thread pool), to avoid feeding str(coroutine) to the model as a result
        self._parameters = parameters
        self.requires_confirmation = requires_confirmation
        self.supports_parallel = supports_parallel

    def run(self, parameters: dict) -> ToolResponse:
        """Call the wrapped sync function; a returned str is wrapped into a success ToolResponse, an already-ToolResponse is returned as-is."""
        if self._is_async:
            raise TypeError(f"tool '{self.name}' was registered from an async function; execute it via the async entry point (aexecute_tool)")
        result = self._func(parameters)
        if inspect.isawaitable(result):                      # a sync signature that returns an awaitable: reject, to avoid feeding <coroutine ...> to the model as a result
            if inspect.iscoroutine(result):
                result.close()                               # close the un-awaited coroutine to silence the "coroutine was never awaited" warning
            raise TypeError(f"the sync function for tool '{self.name}' returned an awaitable; register it with async def and use the async entry point")
        return result if isinstance(result, ToolResponse) else ToolResponse.ok(str(result))

    async def arun(self, parameters: dict) -> ToolResponse:
        """Async functions are awaited directly; sync functions go through the base-class thread pool (via run). Always returns a ToolResponse."""
        if self._is_async:
            result = await self._func(parameters)
            return result if isinstance(result, ToolResponse) else ToolResponse.ok(str(result))
        return await asyncio.to_thread(self.run, parameters)

    def get_parameters(self) -> List[ToolParameter]:
        """Return the parameter definitions declared at registration time."""
        return self._parameters


class ToolRegistry:
    """Tool registry: manage a batch of Tools by name and produce the tool list for the LLM."""

    def __init__(self, *, prompts=None):
        """Initialize an empty registry. prompts: optional prompt registry (PromptRegistry, see agentmaker.prompts); tool error text is taken from it, DEFAULT_PROMPTS if not passed."""
        self._tools: Dict[str, Tool] = {}
        self.prompts = prompts or DEFAULT_PROMPTS

    def register(self, tool: Tool, *, overwrite: bool = False):
        """Register one tool, keyed by tool.name.

        The tool name must match the function-calling rule ^[a-zA-Z0-9_-]{1,64}$ (same for
        OpenAI / Anthropic); an illegal one raises, otherwise to_openai_schema would emit a
        function name the server rejects; for external sources (MCP), normalize with
        sanitize_tool_name first. Duplicate names raise by default (to prevent accidental
        re-registration / a name collision being silently overwritten); pass overwrite=True
        explicitly when replacement is intended.

        Args:
            tool: the Tool instance to register.
            overwrite: whether to allow overwriting on a duplicate name; default False (duplicate raises ValueError).
        """
        if not _TOOL_NAME_RE.match(tool.name):
            raise ToolRegistrationError(
                f"tool name {tool.name!r} is illegal: must match ^[a-zA-Z0-9_-]{{1,64}}$ (the function-calling function-name rule)")
        if not overwrite and tool.name in self._tools:
            raise ToolRegistrationError(f"tool name '{tool.name}' is already registered; pass overwrite=True explicitly to replace it")
        self._tools[tool.name] = tool

    def register_all(self, tools, *, on_conflict: str = "error") -> List[str]:
        """Register a batch of tools, returning the list of tool names actually registered.

        on_conflict controls duplicate-name behavior: "error" (default, a duplicate raises
        ToolRegistrationError) / "skip" (skip duplicates, keep existing) / "overwrite" (overwrite
        existing). Use "skip" when batch-attaching tools after MCP load_tools, to prevent a
        malicious server from squatting an existing name so a single collision raises and blows
        up the whole load loop (DoS).

        Args:
            tools: the iterable of Tools to register.
            on_conflict: duplicate-name policy, error / skip / overwrite.
        """
        if on_conflict not in ("error", "skip", "overwrite"):
            raise ValueError(f"on_conflict must be error / skip / overwrite, got {on_conflict!r}")
        registered = []
        for tool in tools:
            if on_conflict == "skip" and tool.name in self._tools:
                continue
            self.register(tool, overwrite=(on_conflict == "overwrite"))
            registered.append(tool.name)
        return registered

    def register_function(self, func: Callable[[dict], str], name: str, description: str,
                          parameters: Optional[List[ToolParameter]] = None, *,
                          requires_confirmation: bool = False, supports_parallel: bool = False,
                          overwrite: bool = False):
        """Register a plain function as a tool (a shortcut that avoids writing a Tool subclass first).

        The function is wrapped into a uniform Tool stored in the same table, so like class tools
        it can be listed / rendered to a description / converted to a schema.

        Args:
            func: the tool function; takes a parameter dict and returns a string result.
            name: the tool name.
            description: the tool description.
            parameters: the list of parameter definitions; omit if there are no parameters (defaults to empty).
            requires_confirmation: whether it is high-risk and needs confirmation before execution; set True when quick-registering functions that write to disk / send requests / run commands.
            supports_parallel: set True for read-only, concurrency-safe functions to allow concurrent execution with other parallel tools in the same round (see the Tool class attribute).
            overwrite: whether to allow overwriting on a duplicate name; default False (duplicate raises ValueError).

        func may be a sync or async function: an async function is awaited via the async entry
        point (aexecute_tool), so it is never fed as str(coroutine) garbage to the model.

        Example:
            reg.register_function(lambda p: p["city"] + " sunny", "weather", "check the weather",
                                  [ToolParameter("city", "string", "city name")])
        """
        self.register(_FunctionTool(func, name, description, parameters or [], requires_confirmation,
                                    supports_parallel),
                      overwrite=overwrite)

    def register_callable(self, func: Callable, *, name: Optional[str] = None, description: Optional[str] = None,
                          requires_confirmation: bool = False, external_content: bool = False,
                          supports_parallel: bool = False, overwrite: bool = False):
        """Register a type-annotated function as a tool: the parameter schema is inferred automatically from the signature (via the same implementation as Tool.from_callable / @tool).

        Division of labor with register_function: use register_function to hand-write parameters
        with a function that receives the whole dict; use this method (or @tool-decorate then
        register) for a type-annotated function whose schema you want inferred automatically and
        which is called by expanding kwargs from the signature.

        Args:
            func: the type-annotated function (sync or async).
            name / description: default to the function name / the first line of the docstring.
            requires_confirmation: set True for high-risk actions.
            external_content: set True when the result is content from an external source.
            supports_parallel: set True for read-only, concurrency-safe functions to allow concurrent execution with other parallel tools in the same round (see the Tool class attribute).
            overwrite: whether to allow overwriting on a duplicate name.
        """
        self.register(Tool.from_callable(func, name=name, description=description,
                                         requires_confirmation=requires_confirmation,
                                         external_content=external_content,
                                         supports_parallel=supports_parallel), overwrite=overwrite)

    @classmethod
    def from_tools(cls, tools, *, prompts=None) -> "Optional[ToolRegistry]":
        """The single-source-of-truth entry that normalizes tools into a ToolRegistry (list[Tool]->registry / ToolRegistry->as-is / None->None).

        Both Agent(tools=) and build_agent(spec.tools=) convert through it, avoiding two copies
        of normalization code drifting apart. The list may contain @tool-decorated function
        objects (which are themselves Tools).

        Args:
            tools: None / ToolRegistry / an iterable of Tools (including @tool-decorated function objects).
            prompts: the prompt registry used when building a new registry; DEFAULT_PROMPTS if not passed.
        """
        if tools is None or isinstance(tools, ToolRegistry):
            return tools
        registry = cls(prompts=prompts)
        for tool in tools:
            registry.register(tool)
        return registry

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name, returning None if it does not exist.

        Args:
            name: the tool name.
        """
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def execute_tool(self, name: str, parameters: dict,
                     confirm: Optional[ConfirmCallback] = None) -> ToolResponse:
        """The unified execution entry: locate a tool by name and run it; the sole channel through which an Agent / tool chain calls tools.

        Tools that need confirmation (tool.needs_confirmation(parameters), which by default is
        requires_confirmation; multi-action tools can decide per action) first go through the
        confirm callback to ask consent (a §8 red line); when no confirm is passed, execution is
        denied by default for safety (only allowed once the upper layer explicitly provides a
        confirmation mechanism).

        Args:
            name: the tool name.
            parameters: the parameter dict passed to the tool's run().
            confirm: the confirmation callback for high-risk tools, signature (tool, parameters) -> bool; executes only when it returns True.

        Returns:
            ToolResponse: the tool execution result; a readable status="error" explanation when the tool does not exist or execution is denied.
        """
        result, _executed = self.execute_tool_checked(name, parameters, confirm=confirm)
        return result

    async def aexecute_tool(self, name: str, parameters: dict,
                            confirm: Optional[ConfirmCallback] = None) -> ToolResponse:
        """The async version of execute_tool: tool lookup + confirmation logic is shared with the sync version (_resolve); execution goes through await tool.arun.

        For sync tools, arun offloads run to the thread pool by default; for natively async tools
        (such as MCPTool), arun directly awaits the real async call.
        """
        result, _executed = await self.aexecute_tool_checked(name, parameters, confirm=confirm)
        return result

    def execute_tool_checked(self, name: str, parameters: dict,
                             confirm: Optional[ConfirmCallback] = None) -> tuple[ToolResponse, bool]:
        """Execute a tool and return `(result, whether it actually entered tool.run)`.

        Normal calls keep using `execute_tool()`; the Harness's RunPolicy needs to distinguish
        "stopped at the parameter-validation / confirmation gate" from "the tool actually ran and
        then returned an error", so it counts via this method. It keeps the public return value
        backward-compatible rather than stuffing an execution flag into the ToolResponse.

        Exceptions raised inside the tool body are caught here in place and turned into an error
        ToolResponse (executed=True, since it did enter tool.run): the contract is that a tool
        returns ToolResponse.error itself and does not raise, but custom / MCP / AgentTool tools
        may not honor it, so we catch a layer at the sole execution channel, feeding the exception
        back for the model to reroute rather than letting one tool crash the whole run. HITL's
        ApprovalRequired is raised at a higher layer (harness) and never reaches here.
        """
        tool, err = self._resolve(name, parameters, confirm)
        if err is not None:
            return err, False
        try:
            return tool.run(parameters), True
        except Exception as e:
            return self._exec_error(name, e), True

    async def aexecute_tool_checked(self, name: str, parameters: dict,
                                    confirm: Optional[ConfirmCallback] = None) -> tuple[ToolResponse, bool]:
        """The async version of execute_tool_checked, returning `(result, whether it actually entered tool.arun)`; tool exceptions are likewise caught in place into an error.

        The confirmation callback is offloaded to the thread pool by _aresolve: this keeps a
        blocking confirm (such as the default command-line y/n input) from stalling the event
        loop, and also makes re-entering a sync facade inside the callback (such as using another
        agent to judge approval) legal (the worker thread has no running loop).
        """
        tool, err = await self._aresolve(name, parameters, confirm)
        if err is not None:
            return err, False
        try:
            return await tool.arun(parameters), True
        except Exception as e:
            return self._exec_error(name, e), True

    def _exec_error(self, name: str, e: Exception) -> ToolResponse:
        """Catch an exception raised inside the tool body into a readable error ToolResponse (fed back to the LLM); the full traceback goes to logger.exception,

        otherwise a bug inside a custom / MCP tool only shows up as "the LLM received an error
        message and rerouted", and the developer would see neither the stack nor perhaps even know
        an error occurred. The text fed back to the LLM still gives only a short type + message
        (it does not leak the code path to the model)."""
        logger.exception("tool '%s' raised during execution", name)
        return ToolResponse.error(self.prompts.render("tool.error.exec_failed", name=name,
                                                      err=f"{type(e).__name__}: {e}"))

    def _locate(self, name, parameters):
        """Tool lookup + parameter validation (the common part before the confirmation gate, shared by _resolve / _aresolve).

        Returns (tool, None) on pass; (None, error ToolResponse) when the tool does not exist / parameters are invalid.
        """
        tool = self._tools.get(name)
        if tool is None:
            return None, ToolResponse.error(self.prompts.render(
                "tool.error.not_found", name=name, available=", ".join(self._tools) or self.prompts.text("tool.none")))
        try:
            err = validation_error(self._param_schema(tool), parameters, prompts=self.prompts)
        except Exception as e:
            # schema construction / the validator itself raises (commonly from a bad schema given by an untrusted MCP server): soft-fail back to the model, do not crash the whole run
            logger.exception("tool '%s' parameter-schema construction or validation raised", name)
            return None, ToolResponse.error(self.prompts.render(
                "tool.error.validation", name=name, err=f"{type(e).__name__}: {e}"))
        if err is not None:
            return None, ToolResponse.error(self.prompts.render("tool.error.validation", name=name, err=err))
        return tool, None

    def _resolve(self, name, parameters, confirm):
        """Tool lookup + parameter validation + high-risk confirmation (for the sync execution path).

        Returns (tool, None) when executable; (None, error ToolResponse) when the tool does not exist / parameters are invalid / confirmation was not obtained.
        Order: locate -> validate parameters (a mismatch soft-fails, fed back for the model to fix its parameters, no confirmation needed) -> high-risk confirmation.
        """
        tool, err = self._locate(name, parameters)
        if err is not None:
            return None, err
        if tool.needs_confirmation(parameters):
            if confirm is None:
                approved = False                             # no confirm passed: deny by default for safety (the library's dangerous path "fails safe")
            elif inspect.iscoroutinefunction(confirm):       # an async confirm cannot be awaited on the sync path (the resulting coroutine is always truthy -> silently allowing a high-risk tool): fail loud
                raise TypeError(
                    f"the confirm for tool '{name}' is an async function, but this is the sync execution path; use the async entry point aexecute_tool / aexecute_tool_checked instead")
            else:
                approved = confirm(tool, parameters)
                if inspect.isawaitable(approved):            # a sync signature that returns an awaitable (such as an object with an async __call__): reject, so bool(coroutine) is not always truthy
                    if inspect.iscoroutine(approved):
                        approved.close()
                    raise TypeError(f"the confirm for tool '{name}' returned an awaitable; use the async entry point (aexecute_tool) or return a bool")
            if not approved:
                return None, ToolResponse.error(self.prompts.render("tool.error.needs_confirmation", name=name))
        return tool, None

    async def _aresolve(self, name, parameters, confirm):
        """The async version of _resolve: the confirmation callback is offloaded to the thread pool (for the async execution path).

        confirm runs in a worker thread with no running loop, so a blocking callback (input) does not stall the event loop, and re-entering a sync facade inside the callback is legal too.
        """
        tool, err = self._locate(name, parameters)
        if err is not None:
            return None, err
        if tool.needs_confirmation(parameters):
            if confirm is None:
                approved = False                             # no confirm passed: deny by default for safety
            elif inspect.iscoroutinefunction(confirm):       # async confirm: await directly (do not offload to the thread pool, otherwise the un-awaited coroutine is always truthy -> silently allowing a high-risk tool)
                approved = await confirm(tool, parameters)
            else:
                approved = await asyncio.to_thread(confirm, tool, parameters)
                if inspect.isawaitable(approved):            # a callable object's async __call__: to_thread yields a coroutine, then await it
                    approved = await approved
            if not approved:
                return None, ToolResponse.error(self.prompts.render("tool.error.needs_confirmation", name=name))
        return tool, None

    def get_catalog(self) -> str:
        """Tool catalog: one line per tool "- name: description", no parameters, for keeping resident in the system prompt (tier 1 of progressive disclosure).

        It contrasts with get_tools_description(): the catalog is cheap and resident; the full
        parameter schema is rendered on demand / by subset by the latter. Same shape as
        SkillLoader.catalog(), keeping the "catalog layer" of tools and skills consistent.

        Returns:
            str: one line per tool "- name: description"; a placeholder message when there are no tools.
        """
        if not self._tools:
            return self.prompts.text("tool.empty_catalog")
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def _select(self, names: Optional[List[str]]) -> List[Tool]:
        """Select the tools to render: names=None returns all (preserving prior behavior); if given, take the existing tools in that order.

        For progressive disclosure: after Tool-RAG retrieves the top-k tool names, expand only
        this batch's full schema while preserving the retrieved relevance order; names not present
        are skipped and duplicates are deduplicated (dict.fromkeys preserves first-occurrence order).
        """
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in dict.fromkeys(names) if n in self._tools]

    def get_tools_description(self, names: Optional[List[str]] = None) -> str:
        """Assemble tools into a readable block of text (including a parameter list): for offline rendering / inspection; the context-budget accounting also uses it to estimate tool-schema cost.

        Args:
            names: render only these tools, in the given order; None for all. Used for expanding a subset in progressive disclosure (paired with Tool-RAG).

        Returns:
            str: one block per tool with name, description, and parameter list; a placeholder message when there are no tools.
        """
        tools = self._select(names)
        if not tools:
            return self.prompts.text("tool.empty_catalog")
        lines = []
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
            for p in tool.get_parameters():
                required = self.prompts.text("tool.label.required" if p.required else "tool.label.optional")
                default = "" if p.required or p.default is None else self.prompts.render("tool.label.default", value=p.default)
                lines.append(f"    - {p.name} ({p.type}, {required}{default}): {p.description}")
        return "\n".join(lines)

    def to_openai_schema(self, names: Optional[List[str]] = None) -> List[dict]:
        """Convert tools into the tools list for OpenAI function calling.

        Args:
            names: render only these tools, in the given order; None for all. Used for expanding a subset in progressive disclosure (paired with Tool-RAG).

        Returns:
            List[dict]: each element shaped like {"type": "function", "function": {...}}, the parameter sub-schema produced by _param_schema.
        """
        return [{"type": "function",
                 "function": {"name": tool.name, "description": tool.description,
                              "parameters": self._param_schema(tool)}}
                for tool in self._select(names)]

    @staticmethod
    def _param_schema(tool: Tool) -> dict:
        """The parameter JSON Schema for a single tool ({type:object, properties, required}).

        The one sent to the LLM (to_openai_schema) and the one used to validate input parameters
        before execution are the same, so there is zero drift. If p.schema is given, use it as-is
        (faithfully preserving MCP's enum / array items / nested object); otherwise generate from
        type + description; for a non-required parameter that declared a default, write the
        default into that parameter's schema too (a standard JSON Schema field, given to the model
        as a default hint).
        """
        properties, required = {}, []
        for p in tool.get_parameters():
            if p.schema is not None:
                properties[p.name] = p.schema
            else:
                prop = {"type": p.type, "description": p.description}
                if not p.required and p.default is not None:
                    prop["default"] = p.default
                properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {"type": "object", "properties": properties, "required": required}


