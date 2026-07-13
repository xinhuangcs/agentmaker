"""Skills: progressive disclosure of instructions the agent can look up on demand.

A Skill is a SKILL.md file (frontmatter name/description + a body). SkillLoader reads the catalog
(name + description) cheaply, and loads a skill's full body only when it's actually needed, so a
large skill library does not bloat every prompt. Hermetic (writes to a temp directory).

    uv run python examples/16_skills.py
"""
import tempfile
from pathlib import Path

from agentmaker import SkillLoader

# Create a couple of skills on disk; each is a folder containing a SKILL.md.
root = Path(tempfile.mkdtemp())
(root / "greet").mkdir()
(root / "greet" / "SKILL.md").write_text(
    "---\nname: greet\ndescription: greet the user warmly by name\n---\n"
    "Step 1: address the user by name.\nStep 2: ask how you can help.\n")
(root / "summarize").mkdir()
(root / "summarize" / "SKILL.md").write_text(
    "---\nname: summarize\ndescription: condense a long text into bullet points\n---\n"
    "Keep only the key facts, decisions, and open questions.\n")

loader = SkillLoader(str(root))

# The catalog is cheap (name + description only): show it to the model so it can pick a skill.
print("catalog:")
print(loader.catalog())

# Load the full body only when the chosen skill is actually needed.
print("\nloaded 'greet' body:")
print(loader.load("greet"))
