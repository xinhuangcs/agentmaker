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
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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

    def is_external_content(self, parameters: dict) -> bool:
        """Guard note bodies on read without wrapping append status messages."""
        return (parameters.get("action") or "").strip() == "read"

    def __init__(self, root: str, *, max_read_chars: int = 8000,
                 max_append_chars: int = 50_000, max_file_bytes: int = 2_000_000, prompts=None):
        """Build a NotesTool confined to a root directory.

        Args:
            root: Notes root directory; all reads and writes are confined within it. Missing directories are created during construction.
            max_read_chars: Maximum characters returned by read, truncated beyond it (to protect the context).
            max_append_chars: Maximum characters of content per append, refused beyond it (to prevent a runaway single write).
            max_file_bytes: Maximum bytes for a single note file; an append that would exceed it is refused (to prevent unbounded writes filling the disk).
        """
        if (os.name != "posix" or not hasattr(os, "O_NOFOLLOW")
                or os.open not in os.supports_dir_fd or os.mkdir not in os.supports_dir_fd):
            raise OSError("NotesTool requires POSIX dirfd and O_NOFOLLOW support")
        if max_read_chars <= 0:
            raise ValueError(f"max_read_chars must be a positive integer, got {max_read_chars}")
        if max_append_chars <= 0:
            raise ValueError(f"max_append_chars must be a positive integer, got {max_append_chars}")
        if max_file_bytes <= 0:
            raise ValueError(f"max_file_bytes must be a positive integer, got {max_file_bytes}")
        raw_root = os.fspath(root)
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise ValueError("notes root must be a non-empty path")
        self._root = Path(os.path.abspath(raw_root))
        created = False
        try:
            self._root.mkdir(mode=0o700, parents=True)
            created = True
        except FileExistsError:
            pass
        if created:
            os.chmod(self._root, 0o700)
        info = os.lstat(self._root)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("notes root must be a directory and must not be a symlink")
        if info.st_uid != os.geteuid():
            raise PermissionError("notes root must be owned by the current user")
        mode = stat.S_IMODE(info.st_mode)
        if mode & 0o077:
            raise PermissionError("notes root must not be accessible to group or others (e.g. chmod 0700)")
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
        parts, err = self._resolve_path(parameters.get("path", ""))
        if err is not None:
            return ToolResponse.error(err)
        assert parts is not None
        if action == "read":
            return self._read(parts)
        return self._append(parts, parameters.get("content", ""))

    def _resolve_path(self, path: str) -> Tuple[Optional[tuple[str, ...]], Optional[str]]:
        """Validate a relative note path and return its components."""
        path = (path or "").strip()
        if not path:
            return None, self.prompts.text("tool.msg.notes.empty_path")
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return None, self.prompts.render("tool.msg.notes.path_escape", root=self._root)
        parts = tuple(part for part in candidate.parts if part not in ("", "."))
        if not parts:
            return None, self.prompts.text("tool.msg.notes.empty_path")
        return parts, None

    @contextmanager
    def _parent_dir(self, parts: tuple[str, ...], *, create: bool) -> Iterator[tuple[int, str]]:
        """Open each parent directory without following symlinks."""
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        current = os.open(self._root, flags)
        try:
            for component in parts[:-1]:
                if create:
                    try:
                        os.mkdir(component, 0o700, dir_fd=current)
                    except FileExistsError:
                        pass
                next_fd = os.open(component, flags, dir_fd=current)
                os.close(current)
                current = next_fd
            yield current, parts[-1]
        finally:
            os.close(current)

    @staticmethod
    def _relative(parts: tuple[str, ...]) -> str:
        return "/".join(parts)

    @staticmethod
    def _require_private_regular(fd: int) -> None:
        """Require a regular file with no additional hard links."""
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("note path must be a singly linked regular file")

    def _read(self, parts: tuple[str, ...]) -> ToolResponse:
        """Read the full note (truncating if too long); a missing file is treated as "no notes yet" and returns a successful empty hint (not an error)."""
        rel = self._relative(parts)
        try:
            with self._parent_dir(parts, create=False) as (parent_fd, name):
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent_fd)
            try:
                self._require_private_regular(fd)
                with os.fdopen(fd, encoding="utf-8") as stream:
                    fd = -1
                    text = stream.read(self.max_read_chars + 1)
            finally:
                if fd >= 0:
                    os.close(fd)
        except FileNotFoundError:
            return ToolResponse.ok(self.prompts.render("tool.msg.notes.empty_note", rel=rel),
                                   data={"path": rel, "exists": False})
        except (OSError, UnicodeError) as e:
            return ToolResponse.error(self.prompts.render("tool.msg.notes.read_failed", rel=rel, err=e))
        if len(text) > self.max_read_chars:
            clipped = text[:self.max_read_chars] + self.prompts.render("tool.msg.notes.truncated", max=self.max_read_chars)
            return ToolResponse.partial(clipped, data={"path": rel, "exists": True, "truncated": True})
        return ToolResponse.ok(text, data={"path": rel, "exists": True})

    def _append(self, parts: tuple[str, ...], content: str) -> ToolResponse:
        """Append one bounded text block without following path symlinks."""
        rel = self._relative(parts)
        if not (content or "").strip():
            return ToolResponse.error(self.prompts.text("tool.msg.notes.append_empty"))
        if len(content) > self.max_append_chars:
            return ToolResponse.error(self.prompts.render("tool.msg.notes.append_too_large", max=self.max_append_chars))
        chunk = content if content.endswith("\n") else content + "\n"
        data = chunk.encode("utf-8")
        if len(data) > self.max_file_bytes:
            return ToolResponse.error(self.prompts.render(
                "tool.msg.notes.file_too_large", max=self.max_file_bytes))
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            with self._parent_dir(parts, create=True) as (parent_fd, name):
                fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
        except OSError as e:
            return ToolResponse.error(self.prompts.render("tool.msg.notes.write_failed", rel=rel, err=e))
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._require_private_regular(fd)
            if os.fstat(fd).st_size + len(data) > self.max_file_bytes:
                return ToolResponse.error(self.prompts.render("tool.msg.notes.file_too_large", max=self.max_file_bytes))
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        except OSError as e:
            return ToolResponse.error(self.prompts.render("tool.msg.notes.write_failed", rel=rel, err=e))
        finally:
            os.close(fd)
        return ToolResponse.ok(self.prompts.render("tool.msg.notes.appended", rel=rel, n=len(chunk)),
                               data={"path": rel, "appended_chars": len(chunk)})
