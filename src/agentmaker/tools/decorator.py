"""agentmaker.tools.decorator: the @tool decorator, turns an ordinary type-annotated function into a Tool in one line.

It auto-infers the parameter list and JSON Schema from the function signature (inspect.signature) + type annotations
(typing.get_type_hints) + docstring, saving you from hand-writing ToolParameter. Aligned with OpenAI's @function_tool /
Pydantic AI's @agent.tool: the decorated function name becomes a Tool object directly, usable as
`Agent(tools=[get_weather])` or `registry.register(...)`.

Technical terms:
    - function-calling: the mechanism that makes the LLM emit a structured instruction of "which tool to call, with what arguments"; the schema describes what a tool looks like.
    - get_type_hints: evaluates a function's type annotations (including string forward references) into real type objects.
    - Annotated[T, "description"]: attaches a piece of metadata (here used as a parameter description) to type T, readable at runtime.

Parameter description source priority: Annotated's string metadata > the docstring "Args:" section entry of the same name > empty.
Fail-loud boundary: no type annotation / variadic parameters (*args/**kwargs) / an unresolvable annotation -> raise ToolRegistrationError at registration time,
no silent degradation (a vague schema is more of a trap than an error).
"""

import asyncio
import inspect
import re
import types
from typing import Optional, Union, get_args, get_origin, get_type_hints

from ..core.exceptions import ToolRegistrationError
from .base import Tool, ToolParameter
from .response import ToolResponse

# Python annotation -> JSON Schema type string (same convention as ToolParameter.type / _param_schema)
_TYPE_MAP = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", tuple: "array", dict: "object",
}
# Each line of the parameter section: `param_name: description` or Google-style `param_name (type): description`
# (snake_case names; the optional `(type)` is ignored; accepts both the ASCII ':' and the full-width '：' colon).
_PARAM_LINE_RE = re.compile(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*[:：]\s*(.+)$")
# Header that opens the parameter section: English Google-style "Args:" (also Arguments/Parameters) or Chinese "参数:".
_PARAM_HEADER_RE = re.compile(r"^\s*(?:Args|Arguments|Parameters|参数)\s*[:：]\s*$")
# Any section header, English Google-style (Args/Returns/Raises/Example/...) or Chinese (参数/返回/示例), used to detect where the parameter section ends.
_SECTION_HEADER_RE = re.compile(r"^\s*(?:Args|Arguments|Parameters|Returns?|Yields|Raises|Examples?|Notes?|参数|返回|示例)\s*[:：]")


def _annotation_to_json_type(ann, func_name: str, pname: str) -> str:
    """Map a Python annotation (already stripped of Annotated / Optional) to a JSON Schema type string; fails loud if unrecognized."""
    if ann in _TYPE_MAP:
        return _TYPE_MAP[ann]
    origin = get_origin(ann)                       # generics like list[str] / dict[str,int] take origin (list / dict)
    if origin in _TYPE_MAP:
        return _TYPE_MAP[origin]
    raise ToolRegistrationError(
        f"Tool function '{func_name}' parameter '{pname}' annotation {ann!r} cannot be mapped to a JSON Schema type; "
        "use str / int / float / bool / list / dict (or their Optional / Annotated wrappers), or construct a Tool subclass explicitly")


def _resolve_annotation(ann, func_name: str, pname: str) -> tuple:
    """Strip Annotated (extract description) and Optional/Union-None (mark optional), return (json_type, optional, description)."""
    description = ""
    optional = False
    for _ in range(4):                             # strip at most a few layers (Annotated[Optional[...]] / Optional[Annotated[...]])
        if hasattr(ann, "__metadata__"):           # Annotated[T, "description", ...]
            description = description or next((m for m in ann.__metadata__ if isinstance(m, str)), "")
            ann = get_args(ann)[0]
            continue
        origin = get_origin(ann)
        if origin is Union or origin is getattr(types, "UnionType", None):   # Optional[T] / T | None
            non_none = [a for a in get_args(ann) if a is not type(None)]
            if len(get_args(ann)) > len(non_none):
                optional = True
            if len(non_none) == 1:
                ann = non_none[0]
                continue
            raise ToolRegistrationError(
                f"Tool function '{func_name}' parameter '{pname}' is a multi-type Union {ann!r}, cannot be mapped to a single JSON Schema type")
        break
    return _annotation_to_json_type(ann, func_name, pname), optional, description


def _params_from_signature(func, param_docs: dict) -> tuple:
    """Infer (parameter list list[ToolParameter], parameter name order list[str]) from the function signature.

    Skips a leading self / cls; fails loud on *args / **kwargs, missing annotation, or unresolvable annotation. Description comes from Annotated > param_docs > empty.
    """
    func_name = getattr(func, "__name__", repr(func))
    try:
        hints = get_type_hints(func, include_extras=True)   # include_extras keeps Annotated metadata
    except (NameError, TypeError) as e:                      # forward references / names not visible outside the module, etc.: wrap into a clear error, do not hand third parties a bare NameError
        raise ToolRegistrationError(
            f"Tool function '{func_name}' type annotations cannot be resolved ({e}): ensure the annotated types are visible in module scope, or construct a Tool subclass explicitly") from e
    params, names = [], []
    sig = inspect.signature(func)
    for i, (pname, param) in enumerate(sig.parameters.items()):
        if i == 0 and pname in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolRegistrationError(
                f"Tool function '{func_name}' has variadic parameter '{pname}' (*args / **kwargs), cannot be mapped to a named schema parameter")
        if pname not in hints:
            raise ToolRegistrationError(
                f"Tool function '{func_name}' parameter '{pname}' is missing a type annotation; @tool infers the schema from annotations, add the annotation or construct a Tool subclass explicitly")
        json_type, optional, ann_desc = _resolve_annotation(hints[pname], func_name, pname)
        has_default = param.default is not inspect.Parameter.empty
        required = not optional and not has_default
        params.append(ToolParameter(pname, json_type, ann_desc or param_docs.get(pname, ""),
                                    required=required, default=(param.default if has_default else None)))
        names.append(pname)
    return params, names


def _parse_docstring(func) -> tuple:
    """Extract (tool description, {param_name: description}) from the docstring: the first paragraph becomes the description, the "Args:" section is parsed line by line for parameter descriptions.

    Docstrings are brittle, so a parse failure just means "descriptions come up empty" rather than an error (the description also has Annotated and explicit arguments as fallbacks).
    """
    doc = inspect.getdoc(func) or ""
    if not doc:
        return "", {}
    lines = doc.splitlines()
    # description = the first paragraph, up to the first blank line or the first section header
    # (so a Google-style docstring with no blank line before "Args:" does not swallow the Args block).
    desc_lines = []
    for line in lines:
        if not line.strip() or _SECTION_HEADER_RE.match(line):
            break
        desc_lines.append(line.strip())
    description = " ".join(desc_lines)
    # "Args:" section: after the header line, up to the next blank line or a "Returns/Example" header, extract `name: description`
    # line by line. Continuations are told apart from new params by indentation: a line more indented than the parameter
    # lines is appended to the previous description (so a wrapped line that looks like "word: ..." is not mistaken for a new param).
    param_docs, in_params, last_key, param_indent = {}, False, None, None
    for line in lines:
        if _PARAM_HEADER_RE.match(line):
            in_params = True
            continue
        if in_params:
            if not line.strip() or (_SECTION_HEADER_RE.match(line) and not _PARAM_HEADER_RE.match(line)):
                break
            indent = len(line) - len(line.lstrip())
            m = _PARAM_LINE_RE.match(line)
            if m and (param_indent is None or indent <= param_indent):
                last_key, param_indent = m.group(1), indent
                param_docs[last_key] = m.group(2).strip()
            elif last_key is not None:
                param_docs[last_key] += " " + line.strip()
    return description, param_docs


class _CallableTool(Tool):
    """Wraps a type-annotated callable into a Tool: expands kwargs by signature to call it (parameter-name drift surfaces immediately), with native async support.

    Internal type, produced by @tool / Tool.from_callable, not constructed directly by callers.
    """

    def __init__(self, func, name: str, description: str, parameters: list, param_names: list, *,
                 requires_confirmation: bool = False, external_content: bool = False,
                 supports_parallel: bool = False, origin: str = "builtin"):
        super().__init__(name, description, origin=origin)
        self._func = func
        self._parameters = parameters
        self._param_names = param_names
        self._is_async = inspect.iscoroutinefunction(func)   # async goes through arun (await), synchronous goes through run (thread pool)
        self.requires_confirmation = requires_confirmation   # instance attribute shadows the class attribute (same as _FunctionTool)
        self.external_content = external_content
        self.supports_parallel = supports_parallel

    def get_parameters(self) -> list:
        """Return the parameter definitions computed from the signature at construction time."""
        return self._parameters

    def _kwargs(self, parameters: dict) -> dict:
        """Extract the parameters declared by the signature from the input dict (filter out extra keys, leave missing keys to function defaults), for expansion by name in the call."""
        return {k: parameters[k] for k in self._param_names if k in parameters}

    def run(self, parameters: dict) -> ToolResponse:
        """Call the wrapped synchronous function (expanding kwargs by signature); str -> success ToolResponse, an already-ToolResponse is returned as-is."""
        if self._is_async:
            raise TypeError(f"Tool '{self.name}' is defined by an async function, use the async entry point (aexecute_tool) to run it")
        result = self._func(**self._kwargs(parameters))
        if inspect.isawaitable(result):                      # synchronous signature yet returned an awaitable: reject, do not feed a <coroutine> to the model as a result
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError(f"Tool '{self.name}' synchronous function returned an awaitable, define it with async def and use the async entry point")
        return result if isinstance(result, ToolResponse) else ToolResponse.ok(str(result))

    async def arun(self, parameters: dict) -> ToolResponse:
        """An async function is awaited directly; a synchronous function goes through the base class thread pool (via run). Returns a ToolResponse uniformly."""
        if self._is_async:
            result = await self._func(**self._kwargs(parameters))
            return result if isinstance(result, ToolResponse) else ToolResponse.ok(str(result))
        return await asyncio.to_thread(self.run, parameters)


def tool(_func=None, *, name: Optional[str] = None, description: Optional[str] = None,
         requires_confirmation: bool = False, external_content: bool = False, supports_parallel: bool = False):
    """Decorator: turns a type-annotated function into a Tool object. Supports both bare @tool and parameterized @tool(name=..., ...) forms.

    The decorated name is a Tool, usable as `Agent(tools=[f])` / `registry.register(f)`.
    Parameters / types / defaults / required-ness are inferred from the signature, the tool description comes from the first docstring line, and parameter descriptions come from Annotated or the docstring "Args:" section.

    Args:
        name: Tool name, defaults to the function name.
        description: Tool description, defaults to the first docstring line.
        requires_confirmation: Set True for high-risk actions, to pass through the confirmation gate before execution.
        external_content: Set True when the result is content from an external source, to wrap it in an anti-injection guardrail before feeding it back to the model.
        supports_parallel: Set True for read-only, concurrency-safe tools to allow concurrent execution with other parallelizable tools in the same turn (see the Tool class attributes).

    Example:
        @tool
        def get_weather(city: Annotated[str, "city name"], days: int = 3) -> str:
            \"\"\"Query the weather for a city over the next few days.\"\"\"
            ...
    """
    def wrap(func) -> Tool:
        return Tool.from_callable(func, name=name, description=description,
                                  requires_confirmation=requires_confirmation, external_content=external_content,
                                  supports_parallel=supports_parallel)
    return wrap(_func) if _func is not None else wrap
