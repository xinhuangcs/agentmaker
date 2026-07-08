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

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


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

    If pyyaml is installed, use yaml.safe_load (which supports the multi-line / folded scalars that
    are valid in the official ecosystem, such as `description: >-`). Otherwise fall back to a
    hand-written "single-line key: value" subset, but on a multi-line form it cannot handle (a value
    starting with > or |, or an indented continuation line) raise a ValueError with the file path,
    turning silent corruption into a locatable hard error (with a hint to install pyyaml or rewrite
    as a single line).
    """
    try:
        import yaml   # pyyaml is a core dependency (see pyproject); if it is ever trimmed, fall back to the single-line subset below
        loaded = yaml.safe_load(front)
        return {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}
    except ImportError:
        pass
    meta = {}
    for line in front.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):      # skip blank lines and full-line comments
            continue
        if line[:1] in (" ", "\t"):         # an indented continuation line is part of a multi-line scalar, which the hand-written subset cannot handle
            raise ValueError(f"SKILL.md frontmatter contains multi-line / indented YAML ({path or '?'}): without a pyyaml fallback only single-line scalars are supported; install pyyaml or rewrite as a single line.")
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip()
            if v[:1] in (">", "|"):         # folded (>) / literal-block (|) scalar
                raise ValueError(f"SKILL.md frontmatter contains a folded / block scalar ({path or '?'}: '{k.strip()}: {v[:2]}...'): without a pyyaml fallback only single-line scalars are supported; install pyyaml or rewrite as a single line.")
            meta[k.strip()] = v.strip('"').strip("'")
    return meta


def _parse_frontmatter(text: str, *, path: str = "") -> Tuple[dict, str]:
    """Parse frontmatter plus body, returning (metadata dict, body str); with no frontmatter the metadata is empty and the whole text is the body.

    Used by load() (which needs the body); discover() only needs the frontmatter and uses the leaner
    _read_frontmatter. Scanning is line-by-line and the closing line must be exactly
    `strip() == '---'` (same rule as _read_frontmatter, eliminating the prefix-match difference of a
    find('\\n---') so that '----' / '--- x' no longer diverge between the two paths).
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":             # exact close
            body = "".join(lines[i + 1:]).lstrip("\n")
            return _parse_meta("".join(lines[1:i]), path=path), body
    return {}, text                               # no closing ---: treat the whole text as the body


def _read_frontmatter(path: str) -> dict:
    """Read only the frontmatter block at the head of the file (reading lines up to the second ---), without loading the body into memory.

    Used by discover(): a skill body can be very long, and the directory layer only needs
    name/description. With no frontmatter (first line is not ---) or a missing closing ---, return an
    empty dict, matching _parse_frontmatter's behavior and avoiding mistaking a `key: value`-shaped
    line in the body for frontmatter (otherwise a body line `name: ...` would override the real
    name).
    """
    lines = []
    with open(path, encoding="utf-8") as f:
        if f.readline().strip() != "---":
            return {}
        closed = False
        for line in f:
            if line.strip() == "---":
                closed = True
                break
            lines.append(line)
    return _parse_meta("".join(lines), path=path) if closed else {}


class SkillLoader:
    """Scans the skills directory, parses SKILL.md, and provides progressive-disclosure discovery and loading."""

    def __init__(self, skills_dir: str):
        """
        Args:
            skills_dir: The skills root directory; each subdirectory containing a SKILL.md is one
                skill. (The directory is passed in by the caller/app; agentmaker does not hardcode
                a path.)
        """
        self.skills_dir = skills_dir

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
            md = os.path.join(self.skills_dir, entry, "SKILL.md")
            if not os.path.isfile(md):
                continue
            meta = _read_frontmatter(md)            # read only the frontmatter; the body stays out of memory
            name = meta.get("name") or entry
            if name in seen:
                raise ValueError(f"Duplicate skill name: '{name}' appears in both {seen[name]} and {md}; rename one to keep names unique")
            seen[name] = md
            skills.append(Skill(name=name, description=meta.get("description", ""), path=md))
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
        for s in self.discover():
            if s.name == name:
                with open(s.path, encoding="utf-8") as f:
                    _, body = _parse_frontmatter(f.read(), path=s.path)
                return body
        return None


