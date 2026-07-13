"""agentmaker.tools: the tool system: base class, response protocol, registry, permissions, and builtin / advanced tools.

Exports: Tool (base class) / tool (@tool decorator, turns a function into a tool in one line) / ToolParameter / ToolResponse / ToolRegistry /
ToolPermissions / ConfirmCallback (confirmation callback type for high-risk actions) / CalculatorTool / SearchTool / CLITool / NotesTool / MCPClient / MCPTool.
"""

from .base import ConfirmCallback, Tool, ToolParameter
from .decorator import tool
from .response import ToolResponse
from .registry import ToolRegistry
from .permissions import ToolPermissions
from .builtin import CalculatorTool, SearchTool
from .integrations import CLITool, NotesTool, MCPClient, MCPTool  # advanced / integration tools (mcp is imported lazily)

__all__ = [
    "Tool", "tool", "ToolParameter", "ToolResponse", "ToolRegistry", "ToolPermissions",
    "ConfirmCallback",
    "CalculatorTool", "SearchTool",
    "CLITool", "NotesTool",
    "MCPClient", "MCPTool",
]
