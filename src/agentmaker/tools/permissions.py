"""agentmaker.tools.permissions: tool-call permissions (allow / deny lists).

Declares which tools this agent is permitted to call, taking effect at the harness execution
gate (same layer as HITL, before HITL approval). It is another tool-governance gate after
requires_confirmation/HITL (whether to approve) and harness.context_guard (down-weighting
retrieved content, a prompt-registry key): this gate only decides whether a call is permitted,
and a denied tool is not even asked for approval. See ../doc/tools/permissions.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional, cast

from ..prompts import DEFAULT_PROMPTS

if TYPE_CHECKING:
    from ..prompts import PromptRegistry
    from .base import Tool


@dataclass
class ToolPermissions:
    """Allow / deny lists for tool calls, judged along two dimensions: the tool name and the origin.

    Origin is the true root of trust: a name can be spoofed by a remote MCP server (naming a
    malicious tool "search" to piggyback on the allowlist), whereas origin is stamped by the
    framework ("builtin" / "mcp:{namespace}") and cannot be forged by the tool definition. So the
    more robust allowlist is by origin (allow_origins).

    Decision semantics (aligned with Claude permissions' "deny wins" + a subagent's "allow
    restricts to a usable subset"):
        - matches the deny list or deny_origins -> deny (highest priority).
        - an allowlist is enabled (either allow or allow_origins is non-None) -> the tool must match the allow list or one of allow_origins to be permitted.
        - neither is enabled -> permit (constrained only by deny).

    allow / allow_origins=None means that dimension enables no allowlist; allow=[] (an empty
    allowlist) means that dimension denies everything: use None to mean "no restriction". Exact
    match (no parameter-level / wildcard matching; that is an upper-layer policy). denial_reason
    accepts a tool-name str (judged by name only) or a Tool object (additionally judged by origin).
    """
    allow: Optional[Iterable[str]] = None
    deny: Iterable[str] = ()
    allow_origins: Optional[Iterable[str]] = None      # origin allowlist (such as {"builtin", "mcp:calendar"}); None = this dimension not enabled
    deny_origins: Iterable[str] = ()                   # origin denylist (such as {"mcp:untrusted"})
    prompts: object = field(default=None, compare=False, repr=False)   # prompt registry; the denial reason (fed back to the model) is taken from it

    def __post_init__(self) -> None:
        """Normalize each list into a set to speed up judgment; keep None for allow / allow_origins (distinct from an empty set "deny everything"). prompts defaults to DEFAULT_PROMPTS."""
        self.allow = set(self.allow) if self.allow is not None else None
        self.deny = set(self.deny)
        self.allow_origins = set(self.allow_origins) if self.allow_origins is not None else None
        self.deny_origins = set(self.deny_origins)
        self.prompts = self.prompts or DEFAULT_PROMPTS

    def denial_reason(self, tool: str | Tool) -> Optional[str]:
        """Decide whether a call is allowed: return the denial reason (readable, can be fed back to the model); return None to permit.

        Args:
            tool: A tool-name string, judged by name only, or a Tool object, also judged by trusted origin.

        Returns:
            Optional[str]: the denial-reason text, None to permit.
        """
        name = tool if isinstance(tool, str) else tool.name
        origin = None if isinstance(tool, str) else getattr(tool, "origin", None)
        prompts = cast("PromptRegistry", self.prompts or DEFAULT_PROMPTS)
        if name in self.deny:                                          # deny wins: a name match denies immediately
            return prompts.render("tool.permission.in_deny", name=name)
        if origin is not None and origin in self.deny_origins:         # an origin match on deny_origins denies immediately
            return prompts.render("tool.permission.origin_in_deny", name=name, origin=origin)
        if self.allow is not None or self.allow_origins is not None:   # allowlist enabled: must match either the name or the origin
            name_ok = self.allow is not None and name in self.allow
            origin_ok = self.allow_origins is not None and origin is not None and origin in self.allow_origins
            if not (name_ok or origin_ok):
                if self.allow_origins is not None and origin is not None:
                    return prompts.render("tool.permission.origin_not_allowed", name=name, origin=origin)
                return prompts.render("tool.permission.not_in_allow", name=name)
        return None
