# 提示词注册表

框架发给模型的每一条提示词（prompt，即喂给模型的指令文本），从记忆抽取器到 ReAct 的思考风格，再到每个工具的错误文案，全部收录在同一份目录里，你可以列出、读取并覆盖它们。没有任何一条被硬编码在触及不到的地方。当你想看清框架究竟对模型说了什么、想把某条内置提示词改写成贴合自家产品的口吻、想把整套文案切换到另一种语言，或想把自己的提示词加入这套可枚举的系统时，就来看这一页。

## 目录

提示词引擎由两部分组成。`PromptTemplate` 是一条单独的提示词：它包含模板文本，以及渲染时要填充的 `{placeholder}` 占位槽。`PromptRegistry` 是这些模板的目录，以 `subsystem.name` 形式的字符串为键，例如 `memory.extract`、`react.style` 或 `tool.error.not_found`。框架自带一份现成目录 `DEFAULT_PROMPTS`，其中收录了每一条内置提示词的默认（英文）文本。

这套设计天然防漂移：每个子系统都通过键从注入的注册表里读取自己的提示词，因此目录里列出的，正是实际运行的内容。代码里不存在藏着第二份文本副本的情况。

这四个名字全部直接来自顶层包：

```python
from agentmaker import DEFAULT_PROMPTS, PromptRegistry, PromptTemplate, PromptError
```

注册表能直接回答几个问题。`keys()` 列出所有已注册的提示词名字，`text(key)` 返回当前的模板文本，`render(key, **kw)` 填充其占位符，`as_dict()` 返回一份 `{key: text}` 快照，供打印或导出：

```python
print(len(DEFAULT_PROMPTS.keys()), "prompts registered")
print(DEFAULT_PROMPTS.text("chat.persona"))          # You are a helpful assistant.
print(DEFAULT_PROMPTS.render("context.current_question", query="What time is it?"))
```

## 查看你的 agent 说了什么

每个 agent 都携带自己的注册表。`agent.get_prompts()` 会把它作为一个普通的 `{key: text}` 字典返回，这样在决定要改哪一条之前，你就能看清框架内置了哪些提示词、它们的键是什么、当前又说了些什么：

```python
prompts = agent.get_prompts()
print(prompts["react.style"])
for key in sorted(prompts):
    if key.startswith("tool.error."):
        print(key, "->", prompts[key])
```

agent 的注册表是它自己的一份副本。当你在构造时不传入 `prompts=`，agent 会复制一份 `DEFAULT_PROMPTS`，因此查看或覆盖某个 agent 的提示词，绝不会碰到全局目录或任何其他 agent。

## 覆盖一条提示词

`agent.update_prompts(updates)` 会就地改写内置提示词。传入一个 `{key: new-text}` 字典即可修改特定条目：

```python
agent.update_prompts({"chat.persona": "You are a terse, factual assistant. No small talk."})
```

这项改动会在整个 agent 内传播：它的 harness（包裹每一次模型调用的内部层）以及任何内部子 agent 共享同一份注册表副本，因此一次调用就把它们一起更新了。由于注册表是按实例独立的，这只影响当前这个 agent，不会波及其他 agent、单独构造的工具，或进程范围内的默认值。

你也可以把整个 `PromptRegistry` 交给 `update_prompts`，一次替换整套。切换语言或整体改写文案，正是这样做的，下文会讲。

!!! note
    覆盖 agent 的提示词会实时改变其措辞，但有一处边界：工具的总体描述是在工具构造时定格的，因此在一个已经构建好的 agent 上切换语言或与工具相关的提示词，会让那份快照停留在旧措辞上（参数描述仍会实时更新）。要干净地改变工具文案，请在构建工具之前就设定好注册表。具体写法见语言包一节。

## 占位符与协议标记

提示词内部有两类字面文本是承重的，覆盖时注册表会对两者都加以保护。

**占位符**（placeholder）是框架在渲染时填充的 `{name}` 槽，比如 `context.current_question` 里的 `{query}`，或 `harness.schema_instruction` 里的 `{schema}`。渲染只替换已声明的占位符；模板里的其他花括号（例如 `{"op": "ADD"}` 这样的 JSON 示例）会原样保留，因此协议示例绝不会被破坏。渲染时若缺少某个已声明的占位符，会抛出 `PromptError`。

**协议标记**（protocol token）是框架解析器所依赖的字面字符串，例如记忆调和器发出的 `ADD` / `UPDATE` / `DELETE` / `NOOP` 操作，或结构化输出指令里的 `JSON` 关键字。这些会被声明为 `protected`，任何覆盖都必须逐字保留它们，因为改动它们会破坏下游的解析。

当你覆盖一条提示词时，注册表会校验每一个已声明的占位符和每一个协议标记在你的新文本里是否仍然存在。只要缺了一个，覆盖就会当场以 `PromptError` 被拒绝，从而把一个本来只会在运行时才暴露的失败，提前到你做出改动的这一刻：

```python
from agentmaker import PromptError

# Valid: the required {query} placeholder is preserved, so this override is accepted.
agent.update_prompts({"context.current_question": "[User asks]\n{query}"})

# Rejected: dropping {query} would break rendering, so it fails immediately.
try:
    agent.update_prompts({"context.current_question": "The question is above."})
except PromptError as exc:
    print(exc)   # Invalid prompt override: missing placeholder ['query']
```

`PromptError` 继承自 `AgentmakerError`，因此你可以把它和框架的其他错误一起捕获。

如果你直接构建模板，`PromptTemplate` 也提供同样的保证。它的构造函数接收文本，外加可选的 `variables` 和 `protected` 元组；`render(**kwargs)` 填充占位符，`with_text(new_template)` 返回一份文本不同但约束相同的副本，若新文本丢掉了某个占位符或协议标记，就抛出 `PromptError`：

```python
tpl = PromptTemplate("Answer this question:\n{query}", variables=("query",))
print(tpl.render(query="How tall is Everest?"))
reworded = tpl.with_text("Please answer the following.\n{query}")   # keeps {query}, so this is fine
```

## 中文语言包

框架默认使用英文，同时自带一套完整的中文包，一次调用即可覆盖每一条内置提示词，同时保留每条提示词的占位符和协议标记。应用它是瞬时的：它纯粹是文本替换，不涉及任何模型调用。该包位于 `agentmaker.prompts.packs`，包含 `CHINESE_PROMPTS`（原始的 `{key: text}` 映射）和 `chinese_registry()`（一个返回全新中文注册表的辅助函数）。以下是 [`examples/08_prompt_packs.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/08_prompt_packs.py)，逐字照录：

```python
from agentmaker import DEFAULT_PROMPTS
from agentmaker.prompts.packs import CHINESE_PROMPTS, chinese_registry

# The default registry is English.
print("default (English):", DEFAULT_PROMPTS.text("context.section.memory"))

# Option A: build a Chinese registry and pass it explicitly to an Agent/Tool via prompts=,
# without touching the global default.
zh = chinese_registry()
print("chinese_registry():", zh.text("context.section.memory"))

# Option B (process-wide): apply the Chinese pack before creating any agent or tool, so
# everything that uses the default becomes Chinese. Uncomment to switch globally:
#     DEFAULT_PROMPTS.override(CHINESE_PROMPTS)
print(f"(the Chinese pack has {len(CHINESE_PROMPTS)} entries, one per default prompt)")
```

这两种方式，反映了注册表两个覆盖方法之间的真实区别：

- **`with_overrides`（按注册表、相互隔离）。** `chinese_registry()` 调用 `DEFAULT_PROMPTS.with_overrides(CHINESE_PROMPTS)`，它返回一份应用了该包的全新目录，而全局的 `DEFAULT_PROMPTS` 保持不变。把这份注册表通过 `prompts=` 传给某个 agent，就只有那个 agent 说中文：

    ```python
    from agentmaker import Agent
    from agentmaker.prompts.packs import chinese_registry

    agent = Agent("assistant", llm, prompts=chinese_registry())
    ```

- **`override`（就地、进程范围）。** `DEFAULT_PROMPTS.override(CHINESE_PROMPTS)` 会改动共享的全局目录，因此之后每个读取默认值的组件都会变成中文。请在创建任何 agent 或工具之前调用它一次（上文关于工具描述是构造时快照的告诫，正是时机要紧的原因）。

你也可以把整个注册表交给一个已经构建好的 agent 来切换，因为 `update_prompts` 接受一个 `PromptRegistry`：

```python
agent.update_prompts(chinese_registry())
```

要支持另一种语言，复制中文包文件并翻译每个值，同时逐字保留键、`{placeholder}` 名字和协议标记。守护单条编辑的那套覆盖校验，同样会拒绝任何丢掉了它们之一的语言包，因此一处丢失占位符的误译会响亮地失败，而不是拖到运行时才暴露。

## 注册你自己的提示词

第三方策略和工具可以把它们自己的提示词带入这套可枚举、可覆盖、可翻译的系统。`register(key, template, *, variables=(), protected=())` 会添加一个新条目。请在键上使用命名空间前缀，以避免和框架的内置名字冲突，并在副本上注册，这样就不会改动全局默认值：

```python
from agentmaker import Agent, DEFAULT_PROMPTS

reg = DEFAULT_PROMPTS.copy()
reg.register("myapp.greeting", "Greet {name} warmly and offer help.", variables=("name",))
agent = Agent("assistant", llm, prompts=reg)

# Your prompt now shows up alongside the built-ins and renders like any other.
print("myapp.greeting" in agent.get_prompts())        # True
print(reg.render("myapp.greeting", name="Ada"))
```

`register` 只会添加新键：注册一个已经存在的键会抛出 `PromptError`，因此你不会不小心覆盖掉某条内置提示词。要修改一个已有条目，请改用 `update_prompts`（或注册表的 `override` / `with_overrides`）。

## 下一步去哪里

- [Agent 与工作流](agents.md)：了解 `prompts=` 构造参数，以及一个 agent 的注册表如何流向它的 harness 和子 agent。
- [工具](tools.md)：了解工具描述和错误文案，它们全都是你可以改写的注册表条目。
- [结构化输出](structured-output.md)、[记忆](memory.md) 和 [检索与 RAG](retrieval-and-rag.md)：了解这些子系统，它们的提示词（schema 指令、事实抽取、基于来源的作答）都收录在这份目录里。
- [提示词 API 参考](../reference/prompts.md)：查看确切的签名（该节由英文 docstring 生成）。
