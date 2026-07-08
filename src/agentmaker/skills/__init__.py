"""Progressive-disclosure loading of Skills (one of the agent capability layers: workflow knowledge).

A Skill is a folder plus a SKILL.md (YAML frontmatter: name + description, then a body). Unlike a
Tool (an action), a Skill is knowledge about "how to do something well", and the model decides on
its own when to activate it (aligned with the Claude Code / OpenClaw / official format).

    - Skill: a single skill (name/description known at startup, body lazy-loaded).
    - SkillLoader: discover (directory layer) / catalog (the directory shown to the model) / load
      (read the body on demand).
"""

from .loader import Skill, SkillLoader

__all__ = ["Skill", "SkillLoader"]
