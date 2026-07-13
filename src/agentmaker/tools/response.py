"""agentmaker.tools.response: ToolResponse, the uniform return type for tool execution results.

A tool's run / arun uniformly returns a ToolResponse: text for the model to read + a machine-checkable status + structured data.
This lets the Agent / main loop both feed text back to the model and, by status, distinguish success / partial success / failure and take the structured result by data.
"""

from dataclasses import dataclass
from typing import Any, Literal

# Execution status discriminant values: tightened with Literal (machines branch on this, and a misspelled literal is caught by static checking rather than at runtime).
ToolStatus = Literal["success", "partial", "error"]


@dataclass
class ToolResponse:
    """A tool execution result.

    Attributes:
        text: The result text for the model to read (always present; this is what gets spliced into the conversation / scratchpad).
        status: Execution status: "success" normal / "partial" succeeded but incomplete (e.g. output was truncated) / "error" failed.
        data: Structured data (optional), for programmatic use (such as raw search entries, a computed value); the model only reads text, not data.
    """

    text: str
    status: ToolStatus = "success"
    data: Any = None

    def __str__(self) -> str:
        """Let an f-string / string concatenation take the result text directly."""
        return self.text

    @classmethod
    def ok(cls, text: str, data: Any = None) -> "ToolResponse":
        """Construct a success result (optionally with structured data)."""
        return cls(text=text, status="success", data=data)

    @classmethod
    def error(cls, text: str, data: Any = None) -> "ToolResponse":
        """Construct a failure result (text is a readable error description, optionally with structured data).

        Most failures have no structured result and leave data as None; but some failures still carry usable context (such as an MCP tool's
        structuredContent / raw content when isError), pass data when needed.
        """
        return cls(text=text, status="error", data=data)

    @classmethod
    def partial(cls, text: str, data: Any = None) -> "ToolResponse":
        """Construct a "succeeded but incomplete" result (such as truncated output / some sources failing)."""
        return cls(text=text, status="partial", data=data)
