"""Progressive-disclosure Skill loader.

A Skill is a folder plus a SKILL.md (YAML frontmatter: name + description; the body holds the
steps/knowledge). Unlike a Tool (an action), a Skill is workflow knowledge about "how to do
something well", and the model decides on its own when to activate it (model-invoked).

Progressive disclosure:
    1. discover/catalog: at startup read only each skill's name + description (building a "directory"
       placed in the system prompt).
    2. load: when the model decides from the directory that it needs a skill, read its full body
       (into the context) only then.
    3. (placeholder) extra files referenced from the body are read only when used.
"""

import logging
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from yaml.events import (AliasEvent, MappingEndEvent, MappingStartEvent, ScalarEvent,
                         SequenceEndEvent, SequenceStartEvent)


_log = logging.getLogger(__name__)

_DEFAULT_MAX_FRONTMATTER_BYTES = 64 * 1024
_DEFAULT_MAX_BODY_BYTES = 1024 * 1024


@dataclass
class Skill:
    """A single skill. name/description are known at startup (directory layer); body is lazy-loaded (read only when used).

    Attributes:
        name: Unique identifier (kebab-case, e.g. daily-planning).
        description: What it does plus when to use it. The model decides whether to activate it from
            this, so it is the key to triggering.
        path: Path to SKILL.md.
        body: Full body; empty at discover time, populated after load() (progressive disclosure).
    """
    name: str
    description: str
    path: str
    body: str = ""


def _parse_meta(front: str, *, path: str = "") -> dict:
    """Parse frontmatter text into a dict.

    YAML aliases and non-string name/description values are rejected before construction.
    """
    try:
        depth = 0
        nodes = 0
        for event in yaml.parse(front, Loader=yaml.SafeLoader):
            if isinstance(event, AliasEvent):
                raise ValueError(
                    f"SKILL.md frontmatter must not contain YAML aliases: {path or '?'}")
            if isinstance(event, (MappingStartEvent, SequenceStartEvent, ScalarEvent)):
                nodes += 1
                if nodes > 4096:
                    raise ValueError(f"SKILL.md frontmatter has too many YAML nodes: {path or '?'}")
            if isinstance(event, (MappingStartEvent, SequenceStartEvent)):
                depth += 1
                if depth > 32:
                    raise ValueError(f"SKILL.md frontmatter is nested too deeply: {path or '?'}")
            elif isinstance(event, (MappingEndEvent, SequenceEndEvent)):
                depth -= 1
        loaded = yaml.safe_load(front)
    except yaml.YAMLError as error:
        raise ValueError(
            f"SKILL.md frontmatter is invalid YAML: {path or '?'}: {error}") from error
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"SKILL.md frontmatter must be a YAML object: {path or '?'}")
    meta = {}
    for key in ("name", "description"):
        if key not in loaded:
            continue
        value = loaded[key]
        if isinstance(value, (bool, int, float)):
            value = str(value)   # unquoted scalars such as `name: 2048` parse as YAML numbers
        if not isinstance(value, str):
            raise ValueError(f"SKILL.md frontmatter '{key}' must be a string: {path or '?'}")
        meta[key] = value
    return meta


def _parse_frontmatter(text: str, *, path: str = "",
                       max_frontmatter_bytes: Optional[int] = None,
                       max_body_bytes: Optional[int] = None) -> Tuple[dict, str]:
    """Parse frontmatter plus body, returning (metadata dict, body str); with no frontmatter the metadata is empty and the whole text is the body.

    Used by load() (which needs the body); discover() only needs the frontmatter and uses the leaner
    _read_frontmatter. Scanning is line-by-line and the closing line must be exactly
    `strip() == '---'`, the same rule used by _read_frontmatter.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        if max_body_bytes is not None and len(text.encode("utf-8")) > max_body_bytes:
            raise ValueError(f"SKILL.md body exceeds {max_body_bytes} bytes: {path or '?'}")
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":             # exact close
            front = "".join(lines[1:i])
            body = "".join(lines[i + 1:]).lstrip("\n")
            if max_frontmatter_bytes is not None and len(front.encode("utf-8")) > max_frontmatter_bytes:
                raise ValueError(
                    f"SKILL.md frontmatter exceeds {max_frontmatter_bytes} bytes: {path or '?'}")
            if max_body_bytes is not None and len(body.encode("utf-8")) > max_body_bytes:
                raise ValueError(f"SKILL.md body exceeds {max_body_bytes} bytes: {path or '?'}")
            return _parse_meta(front, path=path), body
    if max_body_bytes is not None and len(text.encode("utf-8")) > max_body_bytes:
        raise ValueError(f"SKILL.md body exceeds {max_body_bytes} bytes: {path or '?'}")
    return {}, text                               # no closing ---: treat the whole text as the body


def _read_frontmatter_fd(fd: int, path: str, limit: int) -> dict:
    """Read only the frontmatter block at the head of the file (reading lines up to the second ---), without loading the body into memory.

    Used by discover(): a skill body can be very long, and the directory layer only needs
    name/description. With no frontmatter (first line is not ---) or a missing closing ---, return an
    empty dict, matching _parse_frontmatter's behavior and avoiding mistaking a `key: value`-shaped
    line in the body for frontmatter (otherwise a body line `name: ...` would override the real
    name).
    """
    lines: list[bytes] = []
    with os.fdopen(os.dup(fd), "rb", buffering=0) as stream:
        first = stream.readline(limit + 1)
        if len(first) > limit:
            if first.lstrip().startswith(b"---"):
                raise ValueError(f"SKILL.md frontmatter exceeds {limit} bytes: {path}")
            return {}
        if first.strip() != b"---":
            return {}
        used = len(first)
        closed = False
        while True:
            remaining = limit - used
            if remaining <= 0:
                raise ValueError(f"SKILL.md frontmatter exceeds {limit} bytes: {path}")
            line = stream.readline(remaining + 1)
            if not line:
                break
            used += len(line)
            if used > limit:
                raise ValueError(f"SKILL.md frontmatter exceeds {limit} bytes: {path}")
            if line.strip() == b"---":
                closed = True
                break
            lines.append(line)
    if not closed:
        return {}
    try:
        front = b"".join(lines).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"SKILL.md frontmatter is not valid UTF-8: {path}") from error
    return _parse_meta(front, path=path)


def _read_snapshot_fd(fd: int, path: str, limit: int) -> str:
    """Read a regular file from one descriptor with a hard byte limit."""
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"SKILL.md must be a regular file: {path}")
    if info.st_size > limit:
        raise ValueError(f"SKILL.md exceeds the {limit}-byte snapshot limit: {path}")
    chunks = []
    total = 0
    while total <= limit:
        chunk = os.read(fd, min(64 * 1024, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total > limit:
        raise ValueError(f"SKILL.md exceeds the {limit}-byte snapshot limit: {path}")
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"SKILL.md is not valid UTF-8: {path}") from error


class SkillLoader:
    """Scans the skills directory, parses SKILL.md, and provides progressive-disclosure discovery and loading."""

    def __init__(self, skills_dir: str, *,
                 max_frontmatter_bytes: int = _DEFAULT_MAX_FRONTMATTER_BYTES,
                 max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES):
        """
        Args:
            skills_dir: The skills root directory; each subdirectory containing a SKILL.md is one
                skill. (The directory is passed in by the caller/app; agentmaker does not hardcode
                a path.)
        """
        for label, value in (("max_frontmatter_bytes", max_frontmatter_bytes),
                             ("max_body_bytes", max_body_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer, got {value!r}")
        self.skills_dir = str(Path(skills_dir).resolve())
        self.max_frontmatter_bytes = max_frontmatter_bytes
        self.max_body_bytes = max_body_bytes
        self._warned_paths: set = set()

    def _warn_skip(self, reason: str, target, error: Optional[OSError] = None) -> None:
        """Warn once per (reason, target) that an entry was skipped, then stay quiet on repeat scans."""
        key = (reason, str(target))
        if key in self._warned_paths:
            return
        self._warned_paths.add(key)
        if error is not None:
            _log.warning("%s: %s (%s)", reason, target, error)
        else:
            _log.warning("%s: %s", reason, target)

    @contextmanager
    def _open_skill_file(self, entry: str):
        """Open one skill file without accepting symlink or path-swap targets."""
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        base_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if not nofollow or os.open not in os.supports_dir_fd:
            path = str(Path(self.skills_dir) / entry / "SKILL.md")
            directory_path = str(Path(path).parent)
            try:
                directory_info = os.lstat(directory_path)
            except FileNotFoundError:
                yield None
                return
            if stat.S_ISLNK(directory_info.st_mode):
                self._warn_skip("skipping symlinked skill directory", directory_path)
                yield None
                return
            if not stat.S_ISDIR(directory_info.st_mode):
                yield None
                return
            try:
                file_info = os.lstat(path)
            except FileNotFoundError:
                yield None
                return
            if stat.S_ISLNK(file_info.st_mode):
                self._warn_skip("skipping skill with symlinked SKILL.md", path)
                yield None
                return
            if not stat.S_ISREG(file_info.st_mode):
                yield None
                return
            fd = os.open(path, base_flags | getattr(os, "O_BINARY", 0)
                         | getattr(os, "O_NONBLOCK", 0))
            try:
                opened_info = os.fstat(fd)
                if (not stat.S_ISREG(opened_info.st_mode)
                        or (opened_info.st_dev, opened_info.st_ino)
                        != (file_info.st_dev, file_info.st_ino)):
                    self._warn_skip("skipping skill whose SKILL.md changed while opening", path)
                    yield None
                    return
                yield fd, path
            finally:
                os.close(fd)
            return
        root_fd = os.open(self.skills_dir, base_flags | os.O_DIRECTORY | nofollow)
        directory_fd = -1
        file_fd = -1
        path = str(Path(self.skills_dir) / entry / "SKILL.md")
        try:
            try:
                entry_info = os.stat(entry, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                yield None
                return
            if stat.S_ISLNK(entry_info.st_mode):
                self._warn_skip("skipping symlinked skill directory", Path(path).parent)
                yield None
                return
            if not stat.S_ISDIR(entry_info.st_mode):
                yield None
                return
            try:
                directory_fd = os.open(
                    entry, base_flags | os.O_DIRECTORY | nofollow, dir_fd=root_fd)
            except OSError as error:
                self._warn_skip("skipping skill directory that could not be opened", Path(path).parent, error)
                yield None
                return
            try:
                file_info = os.stat("SKILL.md", dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                yield None
                return
            if stat.S_ISLNK(file_info.st_mode):
                self._warn_skip("skipping skill with symlinked SKILL.md", path)
                yield None
                return
            try:
                file_fd = os.open(
                    "SKILL.md", base_flags | nofollow | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=directory_fd)
            except OSError as error:
                self._warn_skip("skipping skill whose SKILL.md could not be opened", path, error)
                yield None
                return
            opened_info = os.fstat(file_fd)
            if not stat.S_ISREG(opened_info.st_mode):
                yield None
                return
            yield file_fd, path
        finally:
            if file_fd >= 0:
                os.close(file_fd)
            if directory_fd >= 0:
                os.close(directory_fd)
            os.close(root_fd)

    def discover(self) -> List[Skill]:
        """Scan the directory, reading only each SKILL.md's frontmatter (name + description), not the body.

        Progressive-disclosure layer 1: fetch only the "directory", leaving body empty to be read
        later on use (load). Reads lines up to the closing ---, so the body never enters memory.
        Skill names must be unique: a duplicate raises ValueError (otherwise load would take the
        first and catalog would list a duplicate, a silent ambiguity).

        Returns:
            List[Skill]: The discovered skills (with empty body), sorted by directory name.
        """
        skills = []
        seen = {}
        if not os.path.isdir(self.skills_dir):
            return skills
        for entry in sorted(os.listdir(self.skills_dir)):
            with self._open_skill_file(entry) as opened:
                if opened is None:
                    continue
                fd, path = opened
                meta = _read_frontmatter_fd(fd, path, self.max_frontmatter_bytes)
                name = meta.get("name") or entry
                if name in seen:
                    raise ValueError(f"Duplicate skill name: '{name}' appears in both {seen[name]} and {path}; rename one to keep names unique")
                seen[name] = path
                skills.append(Skill(name=name, description=meta.get("description", ""), path=path))
        return skills

    def catalog(self) -> str:
        """Join every skill's name + description into a "directory" text, placed in the system prompt for the model to decide which to use.

        Returns:
            str: Something like "- daily-planning: Organize scattered todos into a plan for the day. Use it when the user says 'plan today'.".
        """
        skills = self.discover()
        if not skills:
            return ""
        return "\n".join(f"- {s.name}: {s.description}" for s in skills)

    def load(self, name: str) -> Optional[str]:
        """Read a skill's full body (SKILL.md body); None if it does not exist.

        Progressive-disclosure layer 2: called only when the model decides from the directory to
        activate a skill, so the body enters the context only at that moment.

        Args:
            name: The skill name (the name from discover/catalog).

        Returns:
            Optional[str]: The full body; None if not found.
        """
        if not os.path.isdir(self.skills_dir):
            return None
        seen = {}
        matched = None
        for entry in sorted(os.listdir(self.skills_dir)):
            with self._open_skill_file(entry) as opened:
                if opened is None:
                    continue
                fd, path = opened
                meta = _read_frontmatter_fd(fd, path, self.max_frontmatter_bytes)
                skill_name = meta.get("name") or entry
                if skill_name in seen:
                    raise ValueError(
                        f"Duplicate skill name: '{skill_name}' appears in both {seen[skill_name]} "
                        f"and {path}; rename one to keep names unique")
                seen[skill_name] = path
                if skill_name != name:
                    continue
                os.lseek(fd, 0, os.SEEK_SET)
                text = _read_snapshot_fd(
                    fd, path, self.max_frontmatter_bytes + self.max_body_bytes + 16)
                _, matched = _parse_frontmatter(
                    text, path=path,
                    max_frontmatter_bytes=self.max_frontmatter_bytes,
                    max_body_bytes=self.max_body_bytes)
        return matched
