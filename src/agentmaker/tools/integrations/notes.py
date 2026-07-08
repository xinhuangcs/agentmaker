"""agentmaker.tools.integrations.notes: a file-notes tool (read / append within a restricted path).

Lets an agent write progress / plans / decisions into a note file under a restricted directory and read them back
across sessions, completing the "keep long-running notes in the filesystem" pattern (Anthropic / OpenCode's
AGENTS.md / progress.txt). Complementary to memory's semantic recall: one is structured progress the agent
actively maintains (notes), the other is automatically extracted vector memory (memory).

Security:
    1. Restricted root directory: all reads and writes are confined to the root given at construction; anything
       that escapes after resolution (.., absolute paths, symlink escape) is refused.
    2. Writes require confirmation: append (writing to disk, high-risk) goes through ToolRegistry's confirm; read
       is read-only and is allowed natively by needs_confirmation without the confirmation gate (action-level
       confirmation, so the app need not write a confirm that distinguishes by action).

File reads and writes are fast local IO, so a native arun is not implemented; the base-class default (thread-pool
bridging) suffices, since the standard library has no native async file IO and we do not pull in the third-party
aiofiles. This differs from CLITool: a subprocess has a native async API and commands may run long, which is why
that one is natively async.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from ...prompts import DEFAULT_PROMPTS
from ..base import Tool, ToolParameter
from ..response import ToolResponse

# Supported actions: read the whole file / append to the end.
_ACTIONS = ("read", "append")


class NotesTool(Tool):
    """Read / append note files under a restricted directory (append writes to disk, requires confirmation)."""

    # High-risk posture by default: it can write to disk. Which action requires confirmation is refined by needs_confirmation.
    requires_confirmation = True

    def needs_confirmation(self, parameters: dict) -> bool:
        """Only disk-writing actions require confirmation; read is a read-only operation within the restricted root and is allowed natively (action-level confirmation).

        Uses "anything but read requires confirmation" rather than "only append requires confirmation": consistent
        with the high-risk default posture, so any new write action added later also automatically requires confirmation (fail-safe).
        """
        return (parameters.get("action") or "").strip() != "read"

    def __init__(self, root: str, *, max_read_chars: int = 8000,
                 max_append_chars: int = 50_000, max_file_bytes: int = 2_000_000, prompts=None):
        """Build a NotesTool confined to a root directory.

        Args:
            root: Notes root directory; all reads and writes are confined within it (if it does not exist, parent directories are created on demand on the first append).
            max_read_chars: Maximum characters returned by read, truncated beyond it (to protect the context).
            max_append_chars: Maximum characters of content per append, refused beyond it (to prevent a runaway single write).
            max_file_bytes: Maximum bytes for a single note file; an append that would exceed it is refused (to prevent unbounded writes filling the disk).
        """
        if max_read_chars <= 0:
            raise ValueError(f"max_read_chars must be a positive integer, got {max_read_chars}")
        if max_append_chars <= 0:
            raise ValueError(f"max_append_chars must be a positive integer, got {max_append_chars}")
        if max_file_bytes <= 0:
            raise ValueError(f"max_file_bytes must be a positive integer, got {max_file_bytes}")
        self._root = Path(root).resolve()
        self.max_read_chars = max_read_chars
        self.max_append_chars = max_append_chars
        self.max_file_bytes = max_file_bytes
        self.prompts = prompts or DEFAULT_PROMPTS
        super().__init__(name="notes", description=self.prompts.render("tool.desc.notes", root=self._root))

    def get_parameters(self) -> List[ToolParameter]:
        """Declare parameters: action (read/append), path (relative path under root), content (text to append when appending)."""
        action_desc = self.prompts.text("tool.param.notes.action")
        return [
            ToolParameter("action", "string", action_desc,
                          schema={"type": "string", "enum": list(_ACTIONS), "description": action_desc}),
            ToolParameter("path", "string", self.prompts.text("tool.param.notes.path")),
            ToolParameter("content", "string", self.prompts.text("tool.param.notes.content"), required=False),
        ]

    def run(self, parameters: dict) -> ToolResponse:
        """Read or append a note per action; an invalid action, a path escape, or an empty append content all return a readable error."""
        action = (parameters.get("action") or "").strip()
        if action not in _ACTIONS:
            return ToolResponse.error(self.prompts.render("tool.msg.notes.bad_action",
                                                          actions=" / ".join(_ACTIONS), got=repr(action)))
        target, err = self._resolve_path(parameters.get("path", ""))
        if err is not None:
            return ToolResponse.error(err)
        if action == "read":
            return self._read(target)
        return self._append(target, parameters.get("content", ""))

    def _resolve_path(self, path: str) -> Tuple[Optional[Path], Optional[str]]:
        """Resolve a relative path into an absolute path within root; empty or escaping (.., absolute path, symlink escape) returns an error.

        First (root / path), then resolve(): an absolute path would override root, `..` truly steps back, and
        symlinks are resolved to their real target, then a single is_relative_to(root) check decides whether it is
        still within the restricted root. All three escape classes are blocked at this one gate.
        """
        path = (path or "").strip()
        if not path:
            return None, self.prompts.text("tool.msg.notes.empty_path")
        target = (self._root / path).resolve()
        if not target.is_relative_to(self._root):
            return None, self.prompts.render("tool.msg.notes.path_escape", root=self._root)
        return target, None

    def _read(self, target: Path) -> ToolResponse:
        """Read the full note (truncating if too long); a missing file is treated as "no notes yet" and returns a successful empty hint (not an error)."""
        rel = self._rel(target)
        if not target.is_file():
            return ToolResponse.ok(self.prompts.render("tool.msg.notes.empty_note", rel=rel),
                                   data={"path": rel, "exists": False})
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResponse.error(self.prompts.render("tool.msg.notes.read_failed", rel=rel, err=e))
        if len(text) > self.max_read_chars:
            clipped = text[:self.max_read_chars] + self.prompts.render("tool.msg.notes.truncated", max=self.max_read_chars)
            return ToolResponse.partial(clipped, data={"path": rel, "exists": True, "truncated": True})
        return ToolResponse.ok(text, data={"path": rel, "exists": True})

    def _append(self, target: Path, content: str) -> ToolResponse:
        """Append content to the end of the note (auto-adding a trailing newline, creating parent directories on demand); empty content, over the limit, or a symlink target all return an error.

        Writes use os.open with O_NOFOLLOW: even if the target is swapped for a symlink between resolution and open
        (a TOCTOU race), following is refused so nothing is written outside the restricted root, narrowing the time
        window after _resolve_path. The file-size limit is checked via fstat after opening (same fd, no second race).
        """
        rel = self._rel(target)
        if not (content or "").strip():
            return ToolResponse.error(self.prompts.text("tool.msg.notes.append_empty"))
        if len(content) > self.max_append_chars:
            return ToolResponse.error(self.prompts.render("tool.msg.notes.append_too_large", max=self.max_append_chars))
        chunk = content if content.endswith("\n") else content + "\n"
        data = chunk.encode("utf-8")
        # Where O_NOFOLLOW is absent (e.g. Windows) it is 0 (degrading to a plain open), without affecting the narrowing on POSIX.
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, flags, 0o600)
        except OSError as e:
            return ToolResponse.error(self.prompts.render("tool.msg.notes.write_failed", rel=rel, err=e))
        try:
            if os.fstat(fd).st_size + len(data) > self.max_file_bytes:
                return ToolResponse.error(self.prompts.render("tool.msg.notes.file_too_large", max=self.max_file_bytes))
            os.write(fd, data)
        except OSError as e:
            return ToolResponse.error(self.prompts.render("tool.msg.notes.write_failed", rel=rel, err=e))
        finally:
            os.close(fd)
        return ToolResponse.ok(self.prompts.render("tool.msg.notes.appended", rel=rel, n=len(chunk)),
                               data={"path": rel, "appended_chars": len(chunk)})

    def _rel(self, target: Path) -> str:
        """Display path of target relative to root (used in result text / data)."""
        return str(target.relative_to(self._root))


