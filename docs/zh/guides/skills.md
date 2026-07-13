# 技能（Skills）

技能（skill）是 agent 可以按需查阅的一份「怎么把某件事做好」的知识：一段有名字、有描述的操作说明（拟一份周计划、总结一段对话记录、跑一次代码评审）。与工具（tool，一个供模型调用的动作）不同，技能是供模型阅读的知识。是否用某个技能由模型自己判断，因此技能是模型自选的（model-invoked）。

技能采用渐进式披露（progressive disclosure，先只给概要、用到了再给全文）：启动时 agent 只看到每个技能的名字和一行描述（一份塞进系统提示词的廉价目录），只有当模型真的要用某个技能时，它的完整正文才会被载入上下文。因此一个庞大的技能库在被用到之前几乎不产生任何开销。

## 快速上手

下面是 [`examples/16_skills.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/16_skills.py) 的原样内容。它是自洽的（写入一个临时目录，无需 API key、无需联网）：

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

## 技能是什么

每个技能是一个目录，里面有一个 `SKILL.md` 文件。该文件有一段 YAML frontmatter（文件头部由 `---` 括起的元数据块），含 `name` 与 `description`，其后是正文：

```text
---
name: greet
description: greet the user warmly by name
---
Step 1: address the user by name.
Step 2: ask how you can help.
```

`name` 是唯一标识（约定用 kebab-case，即小写加连字符）。`description` 说明这个技能做什么、何时该用；由于模型仅凭目录里的这一行来挑选技能，描述正是让技能可被发现的关键。正文承载真正的步骤或知识。

frontmatter 的解析使用 `pyyaml`（一个核心依赖），因此支持多行与折叠标量（例如 `description: >-`）。YAML alias 会被拒绝，嵌套不能超过 32 层，解析节点不能超过 4096 个；`name` 与 `description` 若存在必须是字符串，其它键会被忽略。

## 加载器

`SkillLoader(skills_dir)` 扫描一个目录，其中每个含有 `SKILL.md` 的子目录即为一个技能。路径由你的应用提供，框架不写死任何位置。它暴露三个方法：

- `discover()` 返回一个 `Skill` 对象列表，只读取每个文件的 frontmatter（name 与 description），按目录名排序。此阶段正文留空。技能名必须唯一，重名会抛出 `ValueError`。
- `catalog()` 把每个技能的名字与描述拼成一段目录字符串，可直接放进系统提示词供模型挑选。每一行形如 `- greet: greet the user warmly by name`。
- `load(name)` 读取并返回某个技能的完整正文，若不存在该名字的技能则返回 `None`。

`Skill` 是一个数据类，含 `name`、`description`、`path` 和 `body`（在 `load()` 读取之前为空）。

加载器接受 `max_frontmatter_bytes`（默认 64 KiB）与 `max_body_bytes`（默认 1 MiB）。发现阶段读到 frontmatter 的闭合分隔符即停止，绝不读取正文；加载阶段从同一个已打开描述符读取一份有界快照。POSIX 路径通过目录描述符相对打开并使用 `O_NOFOLLOW`；Windows 或缺少这些原语的平台会使用 `lstat` 校验，再打开一个描述符并用 `fstat` 比对设备号与 inode 身份。两条路径都不跟随软链、也不会重新按已检查路径打开。软链的技能目录或 `SKILL.md`、以及在检查途中被替换或变得无法打开的条目，会记一条警告后跳过；非普通文件或缺失的 `SKILL.md` 则静默跳过。无论哪种，其余技能都照常加载。

## 渐进式披露

`catalog()` 与 `load()` 的分工正是这套设计的用意所在，它分两层发生：

1. **目录（廉价、始终在场）。** 启动时 `discover()` / `catalog()` 只读取每个技能的 frontmatter，因此只有名字和描述进入系统提示词。正文无论多长，此阶段都不加载。
2. **加载（按需）。** 当模型从目录判断出它需要某个技能，你才调用 `load(name)`，并把返回的正文放入上下文。

这样一来，无论你有多少技能，提示词都能保持精简：始终付出的成本是每个技能一行，而完整正文只在被用到时才付费。

## 技能与工具的取舍

当 agent 需要「做」某件有副作用的事（调用 API、跑一次计算、读取文件）时，用**工具**（见 [工具](tools.md)）。当 agent 需要「知道」怎么做某件事（一套流程、一份清单、一种风格约定）时，用**技能**。工具被调用并返回结果；技能被阅读并影响模型接下来怎么走。二者可以组合：一个技能的正文可以指示模型去使用特定的工具。
