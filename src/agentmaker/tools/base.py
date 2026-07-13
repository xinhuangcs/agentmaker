"""agentmaker.tools.base: tool base class and parameter definition.

Every tool subclasses the Tool abstract base class, implementing a uniform run() execution entry point and a
get_parameters() self-description. This lets the registry auto-generate the tool listing / function-calling schema,
and lets the Agent invoke tools in a consistent way.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from .response import ToolResponse

# Function-calling function-name rule (OpenAI / Anthropic share it: ^[a-zA-Z0-9_-]{1,64}$): the single source of
# truth for tool names, living in the lowest-level base so Tool.from_callable / registry / decorator can share it.
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Confirmation callback type for high-risk tools: (tool, parameters dict) -> whether to approve execution. The
# registry / harness / each recipe's confirm= parameter share this alias instead of restating Callable[[Tool, dict], bool].
# It lives in the same module as Tool with zero circular imports ("Tool" uses a forward-reference string).
ConfirmCallback = Callable[["Tool", dict], bool]


@dataclass
class ToolParameter:
    """A single parameter definition for a tool, used for self-description (generating the parameter description / schema shown to the LLM).

    Attributes:
        name: Parameter name.
        type: Parameter type (OpenAI schema style, e.g. string / integer / number / boolean / array / object).
        description: Parameter description.
        required: Whether the parameter is required, defaults to True.
        default: Default value, meaningful only when not required, defaults to None.
        schema: The full JSON Schema for this parameter; if given, to_openai_schema uses it verbatim (preserving enum / array items / nested object),
            otherwise it is generated from type + description. Tools with complex schemas (such as MCP) use this; ordinary tools leave it empty.
    """
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    schema: Optional[dict] = None


class Tool(ABC):
    """Abstract base class for tools. Subclasses must implement run() and get_parameters().

    Class attributes:
        requires_confirmation: Whether this is a high-risk action (send email / delete file / shell, etc.);
            when True, execution requires human confirmation (the confirmation logic is handled by the Agent execution stage, see project red line section 8).
        external_content: Whether the result is "content from an external source" (web search / knowledge base / third-party MCP tool). When True, the framework
            wraps the result in an anti-injection delimiting guardrail before feeding it back to the model (same as memory/RAG's context_guard), reducing the risk of
            indirect prompt injection (OWASP LLM01: external text hiding "ignore previous instructions..." to steer the model). The builtin SearchTool / RAGTool / MCPTool set this True.
        supports_parallel: Whether this tool may run concurrently with other tools in the same turn (when the model emits multiple independent read-only
            calls in one turn, the Agent runs adjacent parallelizable calls together with asyncio.gather and backfills results in original order). Defaults to False
            (strictly serial); setting it True promises "safe under concurrent invocation, no side-effect ordering dependency, does not depend on results
            of other calls in the same turn": typically a read-only tool with no shared mutable state (such as web search). High-risk tools requiring confirmation never run
            in parallel (enforced separately by the framework), so a write-type tool is not run concurrently even if it sets this True.
    """

    requires_confirmation: bool = False
    external_content: bool = False
    supports_parallel: bool = False

    def __init__(self, name: str, description: str, *, origin: str = "builtin"):
        """
        Args:
            name: Tool name (unique identifier, used by the registry as a key).
            description: Description of the tool's function (shown to the LLM, determines when to call it). Fixed at construction time and does not follow prompt changes:
                a builtin tool's description is read from the prompt registry at the moment of creation and stored; later update_prompts / language switches do not change an
                already-built tool's description (whereas parameter descriptions are read live and do change). To switch language wholesale, override before creating the tool
                (see approach one in the packs).
            origin: Tool origin identifier (trust root): "builtin" for framework builtins / app custom tools may set their own / MCP tools are stamped by MCPClient as
                "mcp:{namespace}" (stamped by the framework, cannot be overridden by the tool definition). Used by ToolPermissions for origin-based authorization
                (allow_origins / deny_origins): a name can be spoofed by a remote, the origin is the root of trust.
        """
        self.name = name
        self.description = description
        self.origin = origin

    def needs_confirmation(self, parameters: dict) -> bool:
        """Whether this specific call needs human confirmation; defaults to the static requires_confirmation (whole-tool, all-or-nothing).

        Multi-action tools (such as MemoryTool's forget / consolidate) can override this method to return True only for destructive actions,
        implementing action-level confirmation. The confirmation gate (registry / harness) always reads this method rather than requires_confirmation directly.
        """
        return self.requires_confirmation

    def is_external_content(self, parameters: dict) -> bool:
        """Whether this call returns untrusted external content.

        Action-based tools can override this method when only their read operations return external
        content. The default preserves the class-level ``external_content`` contract.

        Args:
            parameters: The parameters for this call.

        Returns:
            bool: Whether the result must be wrapped as external content.
        """
        return self.external_content

    def get_input_schema(self) -> Optional[dict]:
        """Return a complete root JSON Schema for tool arguments, when one exists.

        Ordinary tools return ``None`` and let the registry build a schema from
        :meth:`get_parameters`. Protocol adapters such as MCP can override this seam to preserve
        root-level JSON Schema keywords in the schema exposed to models and used for validation.

        Returns:
            Optional[dict]: A root argument schema, or ``None`` to use parameter synthesis.
        """
        return None

    def run(self, parameters: dict) -> ToolResponse:
        """Execute the tool (synchronous). Synchronous tools implement this method.

        A subclass must implement at least one of run (synchronous tools) or arun (native async tools); a tool that implements only arun
        has its synchronous run raise by default to steer callers to arun.

        Threading contract: the framework execution chain (harness -> registry -> arun default implementation) dispatches run onto a thread pool,
        possibly a different worker thread each time. Do not hold thread-bound resources on the tool instance (such as a default-argument sqlite3
        connection, signal, or GUI handle); either build connections lazily per-thread with threading.local, or create them with
        check_same_thread=False and add your own lock (the framework's builtin stores do exactly this).

        Args:
            parameters: The call parameter dict (keys correspond to the parameter names declared by get_parameters()).

        Returns:
            ToolResponse: The execution result (text is read by the model, status marks success/partial/error, data is an optional structured result).
        """
        raise NotImplementedError(f"Tool '{self.name}' does not implement synchronous run; it is a native async tool, use arun.")

    async def arun(self, parameters: dict) -> ToolResponse:
        """Execute the tool (async entry point).

        The default implementation dispatches synchronous run onto a thread pool (asyncio.to_thread), so synchronous tools do not block the event loop in an async
        environment: existing synchronous tools can be called by an async Agent with zero changes. Native async tools (such as MCPTool) should override this method and
        await the real async call internally (no need to implement run). The semantics match run: awaiting it means "wait until execution completes, get the result".

        Args and Returns: same as run.
        """
        return await asyncio.to_thread(self.run, parameters)

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Declare the parameters this tool accepts, for the registry to generate a description / schema.

        Returns:
            List[ToolParameter]: The list of parameter definitions; returns an empty list if there are no parameters.
        """

    @classmethod
    def from_callable(cls, func: Callable, *, name: Optional[str] = None, description: Optional[str] = None,
                      requires_confirmation: bool = False, external_content: bool = False,
                      supports_parallel: bool = False, origin: str = "builtin") -> "Tool":
        """Wrap an ordinary type-annotated function into a Tool in one line: parameter names / types / defaults / required-ness are all inferred from the function
        signature, the first docstring line becomes the tool description, and parameter descriptions come from Annotated or the docstring "Args:" section (same implementation as the @tool decorator).

        Difference from register_function: register_function's function takes the whole dict and requires a hand-written parameter list; this path expands kwargs by
        signature and auto-infers the schema (so a parameter-name drift surfaces immediately at the call site rather than a silent KeyError). async functions are awaited natively.

        Args:
            func: The function to wrap (synchronous or async); must have parameter type annotations, otherwise it fails loud at registration time.
            name: Tool name, defaults to func.__name__ (must match the function-calling name rule, raises ToolRegistrationError if invalid).
            description: Tool description, defaults to the first paragraph of the docstring.
            requires_confirmation: Set True for high-risk actions (disk writes / sending requests, etc.) to pass through the confirmation gate before execution.
            external_content: Set True when the result is content from an external source, to wrap it in an anti-injection guardrail before feeding it back to the model.
            supports_parallel: Set True for read-only, concurrency-safe tools to allow concurrent execution with other parallelizable tools in the same turn (see the Tool class attributes).
            origin: Origin identifier (trust root), defaults to "builtin".

        Returns:
            Tool: A Tool instance that calls func by signature.
        """
        from ..core.exceptions import ToolRegistrationError
        from .decorator import _CallableTool, _params_from_signature, _parse_docstring
        resolved_name = name or getattr(func, "__name__", None)
        if not resolved_name or not _TOOL_NAME_RE.match(resolved_name):
            raise ToolRegistrationError(
                f"Cannot infer a valid tool name from {func!r} (got {resolved_name!r}): pass name= explicitly, must match ^[a-zA-Z0-9_-]{{1,64}}$")
        doc_desc, param_docs = _parse_docstring(func)
        params, param_names = _params_from_signature(func, param_docs)
        return _CallableTool(func, resolved_name, description or doc_desc, params, param_names,
                             requires_confirmation=requires_confirmation, external_content=external_content,
                             supports_parallel=supports_parallel, origin=origin)
