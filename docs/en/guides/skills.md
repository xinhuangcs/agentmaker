# Skills

A skill is a piece of know-how the agent can look up on demand: a named, described block of instructions for "how to do something well" (draft a weekly plan, summarize a transcript, run a code review). Unlike a tool, which is an action the model calls, a skill is knowledge the model reads. The model decides on its own when a skill is relevant, so skills are model-invoked.

Skills use progressive disclosure: at startup the agent sees only each skill's name and one-line description (a cheap catalog that fits in the system prompt), and the full body of a skill is loaded into context only when the model actually reaches for it. A large skill library therefore costs almost nothing until a skill is used.

## Quickstart

This is [`examples/16_skills.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/16_skills.py) verbatim. It is hermetic (it writes to a temp directory, needs no API key and no network):

```python
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
```

## What a skill is

Each skill is a directory containing a `SKILL.md` file. The file has YAML frontmatter with a `name` and a `description`, followed by the body:

```text
---
name: greet
description: greet the user warmly by name
---
Step 1: address the user by name.
Step 2: ask how you can help.
```

The `name` is a unique identifier (kebab-case by convention). The `description` says what the skill does and when to use it; because the model chooses a skill from the catalog on this line alone, the description is what makes the skill discoverable. The body holds the actual steps or knowledge.

Frontmatter parsing uses `pyyaml` (a core dependency), so multi-line and folded scalars (for example `description: >-`) are supported. YAML aliases are rejected, as are nesting deeper than 32 levels and documents with more than 4,096 parsed nodes. `name` and `description`, when present, must be strings; unrelated keys are ignored.

## The loader

`SkillLoader(skills_dir)` scans a directory where each subdirectory that contains a `SKILL.md` is one skill. The path is supplied by your application; the framework hardcodes no location. It exposes three methods:

- `discover()` returns a list of `Skill` objects, reading only each file's frontmatter (name and description), sorted by directory name. The body is left empty at this stage. Skill names must be unique; a duplicate raises `ValueError`.
- `catalog()` joins every skill's name and description into a directory string, ready to place in the system prompt so the model can pick one. Each line looks like `- greet: greet the user warmly by name`.
- `load(name)` reads and returns a skill's full body, or `None` if no skill by that name exists.

A `Skill` is a dataclass with `name`, `description`, `path`, and `body` (empty until `load()` reads it).

The loader accepts `max_frontmatter_bytes` (64 KiB by default) and `max_body_bytes` (1 MiB by default). Discovery stops at the closing frontmatter delimiter and never reads the body; loading reads one bounded snapshot from the same opened descriptor. On POSIX, skill directories and `SKILL.md` files are opened relative to directory descriptors with `O_NOFOLLOW`. On Windows or a platform without those primitives, the fallback uses `lstat`, opens one descriptor, then compares its `fstat` device/inode identity with the validated file. Neither path follows symlinks or reopens a checked path. A symlinked skill directory or `SKILL.md`, or one swapped or made unopenable mid-check, is skipped with a warning; a non-regular or missing `SKILL.md` is skipped silently. Either way the remaining skills keep loading.

## Progressive disclosure

The split between `catalog()` and `load()` is the point of the design, and it happens in two layers:

1. **Catalog (cheap, always present).** At startup, `discover()` / `catalog()` read only the frontmatter of every skill, so only the names and descriptions enter the system prompt. A body, however long, never loads at this stage.
2. **Load (on demand).** When the model decides from the catalog that it needs a skill, you call `load(name)` and put the returned body into context, only then.

This keeps the prompt small no matter how many skills you have: the cost you always pay is one line per skill, and you pay for a full body only when it is used.

## Skills vs tools

Use a **tool** ([Tools](tools.md)) when the agent needs to *do* something with an effect: call an API, run a calculation, read a file. Use a **skill** when the agent needs to *know* how to do something: a procedure, a checklist, a house style. Tools are called and return a result; skills are read and shape how the model proceeds. The two compose: a skill's body can instruct the model to use particular tools.
