# 工具

工具（tool）就是你的 Agent 能调用的函数。一个工具对外暴露名称、描述和带类型的参数列表；模型通过 **function calling**（函数调用，即模型输出一个结构化请求，要求以给定参数运行某个具名工具的机制）来调用它，框架随后运行该工具并把结果回传给模型。本指南涵盖：定义工具（一行式的 `@tool` 装饰器或 `Tool` 子类）、用 `ToolResponse` 返回结果、用 `ToolRegistry` 归集工具、内置工具、权限与确认关卡、接入外部 MCP 服务器，以及用 Tool-RAG 在运行时从大量工具中动态挑选。

本页所有内容都运行在 [Agent](agents.md) 循环之上。完整可运行示例见 [`examples/02_tools_and_registry.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/02_tools_and_registry.py)：

```python
from agentmaker import Agent, CalculatorTool, ToolRegistry, tool
from agentmaker.testing import ScriptedLLM


@tool
def to_upper(text: str) -> str:
    """Uppercase a string.

    Args:
        text: The input text.
    """
    return text.upper()


registry = ToolRegistry()
registry.register(CalculatorTool())   # built-in: safe arithmetic evaluation
registry.register(to_upper)           # your custom tool

# Script the model's decision to call the calculator, then its final answer.
llm = ScriptedLLM([
    ScriptedLLM.tool_call("calculator", {"expression": "(3 + 4) * 5"}),
    "The result is 35.",
])
agent = Agent("assistant", llm, tool_registry=registry)
print(agent.run("Compute (3 + 4) * 5").final_output)
```

`ScriptedLLM` 是一个测试替身（test double），会回放一段固定脚本，所以这个示例既不需要 API key，也不需要联网。换成真实模型时，把它替换为 `LLMClient(...)`，由模型自己决定何时调用每个工具。

## 用 `@tool` 定义工具

用 `@tool` 装饰带类型注解的函数后，对应名称是一个 `Tool` 对象，可以直接传给 `Agent(tools=[...])` 或 `registry.register(...)`。模型看到的 schema（描述工具入参结构的 JSON 定义）会自动推断出来：

- **参数名、类型、默认值和是否必填**来自函数签名。Python 类型映射到 JSON Schema 类型：`str` 到 `string`，`int` 到 `integer`，`float` 到 `number`，`bool` 到 `boolean`，`list`/`tuple` 到 `array`，`dict` 到 `object`。
- **工具描述**取自 docstring 的第一段。
- **各参数描述**优先取自 `Annotated` 元数据；若没有，则取自 docstring `Args:` 小节中同名参数的条目。

两种写法都可用。用 `Annotated[T, "description"]` 可以就地给单个参数附上描述：

```python
from typing import Annotated
from agentmaker import tool


@tool
def get_weather(city: Annotated[str, "city name"], days: int = 3) -> str:
    """Query the weather for a city over the next few days."""
    ...
```

带默认值的参数（如上面的 `days=3`）视为可选；没有默认值的参数则为必填。

装饰器还接受几个可选的关键字开关：

```python
@tool(requires_confirmation=True)
def delete_file(path: str) -> str:
    """Delete a file at the given path."""
    ...
```

- `requires_confirmation`：对高风险动作（写入、删除、发送请求）设为 `True`，这样该调用在执行前会先经过确认关卡。
- `external_content`：当结果是来自外部来源的内容时设为 `True`，框架会先用一层防注入护栏（anti-injection guardrail）把它包起来，再回传给模型。
- `supports_parallel`：对只读、并发安全的工具设为 `True`，它就可以在同一轮里与其他可并行的调用并发执行。配置了任一 LLM、工具或 token 数值上限的运行会改为串行执行工具调用。

只有当当前运行没有精确的 `max_tool_calls` 上限时，框架才会做并行批处理。一旦设置 `RunPolicy.max_tool_calls`，符合并行条件的调用也会串行执行，使框架能在每次真实执行前做精确的准入检查。

你也可以用 `@tool(name=..., description=...)` 显式命名工具；不指定时，名称默认取函数名。

!!! note "异步工具用法完全相同"
    `@tool` 原生支持 `async def` 函数。框架会 await 异步工具，并把同步工具派发到线程池，因此同一个工具在同步和异步的 Agent 循环里都能用。

`@tool` 遵循「定义时就大声报错、绝不悄悄降级」的原则：缺少类型注解、含可变参数（`*args`/`**kwargs`），或注解无法映射到 JSON Schema 类型，都会抛出 `ToolRegistrationError`。如果你撞上了这个限制，请改写成 `Tool` 子类。

!!! note "不加装饰器直接注册普通函数"
    如果你不想加装饰器，`registry.register_callable(func)` 会像 `@tool` 一样从签名推断 schema。对于一个接收整个参数字典、且需要手写参数定义的函数，用 `registry.register_function(func, name, description, parameters)`。

## 继承 `Tool` 子类

当你需要保存状态、自定义 schema，或表达装饰器无法表达的逻辑时，直接继承 `Tool`。实现 `get_parameters()`（返回一个 `ToolParameter` 列表）和 `run()`（返回一个 `ToolResponse`）：

```python
from agentmaker import Tool, ToolParameter, ToolResponse


class ReverseTool(Tool):
    def __init__(self):
        super().__init__("reverse", "Reverse a string.")

    def get_parameters(self):
        return [ToolParameter("text", "string", "The text to reverse.")]

    def run(self, parameters: dict) -> ToolResponse:
        return ToolResponse.ok(parameters["text"][::-1])
```

子类至少要实现 `run`（同步）或 `arun`（原生异步）二者之一。默认的 `arun` 会把 `run` 派发到线程池，因此一个同步工具在异步 Agent 里也能原样工作。原生异步工具（比如一次 HTTP 或子进程调用）会重写 `arun` 并 await 真正的调用。类级别的开关 `requires_confirmation`、`external_content`、`supports_parallel` 含义与上文装饰器的关键字相同。

!!! note "线程约定"
    执行链每次可能把 `run` 派发到不同的工作线程。不要在工具实例上持有绑定线程的资源（比如一个共享的 `sqlite3` 连接）；应按线程惰性创建，或用 `check_same_thread=False` 加上你自己的锁来创建。

### `ToolParameter`

`ToolParameter` 描述一个参数。它的字段：

| 字段 | 含义 |
| --- | --- |
| `name` | 参数名。 |
| `type` | JSON Schema 类型字符串（`string`、`integer`、`number`、`boolean`、`array`、`object`）。 |
| `description` | 展示给模型看的参数描述。 |
| `required` | 该参数是否必填（默认 `True`）。 |
| `default` | 默认值，仅在非必填时有意义。 |
| `schema` | 该参数的完整 JSON Schema；给定时会被原样使用，从而保留 `enum`、数组的 `items` 或嵌套对象。 |

对于普通 `type` 无法表达的情形（例如枚举），使用 `schema`：

```python
ToolParameter("action", "string", "The action to run",
              schema={"type": "string", "enum": ["read", "append"], "description": "The action to run"})
```

## 返回值：`ToolResponse`

每个工具都返回一个 `ToolResponse`。如果一个 `@tool` 函数返回普通 `str`，框架会替你把它包成一个成功响应；否则请显式构造。它有三个字段：

- `text`：模型读取的结果文本（始终存在）。
- `status`：`"success"`、`"partial"`（成功但不完整，例如输出被截断）或 `"error"`。
- `data`：可选的结构化数据，供程序化使用；模型只读 `text`，不读 `data`。

三个构造函数覆盖常见场景：

```python
ToolResponse.ok("42", data=42)                 # status="success"
ToolResponse.partial("first 4000 chars ...")   # status="partial"
ToolResponse.error("query must not be empty")  # status="error"
```

返回 `ToolResponse.error(...)` 是上报可恢复失败的惯用做法：错误文本会回传给模型，让它调整参数后重试。抛出异常同样不会让整个运行崩溃，它会在执行层被捕获、以同样的方式回喂给模型；但返回一个显式的错误更清晰，也能保留由你掌控的 `status` 和 `data` 字段。

## 注册表

`ToolRegistry` 按名称保存一个 Agent 可调用的工具。可以用 `register` 逐个注册，也可以用 `register_all` 批量注册：

```python
from agentmaker import ToolRegistry, CalculatorTool, SearchTool

registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register_all([SearchTool(), to_upper])
```

工具名必须符合函数调用的命名规则 `^[a-zA-Z0-9_-]{1,64}$`（OpenAI 与 Anthropic 通用）；非法名称会抛出 `ToolRegistrationError`。重复注册同名工具默认会报错；当确实想替换时，传 `overwrite=True`（或 `register_all(..., on_conflict="skip"/"overwrite")`）。

注册表会把工具渲染成循环所需的各种形态：

- `get_catalog()`：一份廉价的 `- name: description` 目录，每个工具一行。
- `get_tools_description()`：包含参数列表的完整文字描述。
- `to_openai_schema()`：用于函数调用的 `tools` 参数。

`get_tools_description()` 和 `to_openai_schema()` 都接受一个可选的 `names` 列表，只按给定顺序渲染其中的子集，这正是 Tool-RAG 所使用的（见下文 [运行时工具挑选](#运行时工具挑选tool-rag)）。

要直接运行一个工具（[Agent](agents.md) 循环会替你做这件事），用 `execute_tool`，它会先按 schema 校验参数，再返回一个 `ToolResponse`：

```python
registry.execute_tool("calculator", {"expression": "2 + 2"})
```

发给模型的 schema 与用于校验入参的 schema 出自同一来源，因此不会漂移：校验不通过的参数会作为错误 `ToolResponse` 返回给模型去修正，而不是抛出异常。

### 把注册表接入 Agent

`Agent` 既接受一个 `tools` 列表（便捷入口，内部会归一化成一个注册表），也接受你自己构建的 `tool_registry`。二者互斥：

```python
agent = Agent("assistant", llm, tools=[to_upper, CalculatorTool()])   # convenience
agent = Agent("assistant", llm, tool_registry=registry)               # explicit registry
```

一个没有工具的 Agent 就是纯粹的问答。

## 内置工具

框架自带几个不含任何业务逻辑的通用工具。

### `CalculatorTool`

安全地求解数学表达式：把表达式解析成抽象语法树（AST），只对白名单内的运算符求值，因此没有 `eval`、也不存在任意代码执行。它支持 `+ - * / // % **`、一元正负号，以及函数 `sqrt`、`abs`、`round`、`log`、`sin`、`cos` 和常量 `pi`、`e`。它只有一个参数 `expression`，构造时无需任何参数（与所有内置工具一样，它接受一个可选的 `prompts=`，用于本地化其面向用户的字符串）：

```python
from agentmaker import CalculatorTool

registry.register(CalculatorTool())   # tool name: "calculator"
```

### `SearchTool`

带自动多源回退的网页搜索：它先试 Tavily，再试 DuckDuckGo，然后 Brave，最后 SerpAPI；只要某个源没装对应库、没配置 key 或调用失败，就切到下一个源。只有全部失败时才返回错误。key 从环境变量读取（`TAVILY_API_KEY`、`BRAVE_API_KEY`、`SERPAPI_API_KEY`）；DuckDuckGo 不需要 key。它只有一个参数 `query`。

```python
from agentmaker import SearchTool

registry.register(SearchTool(max_results=5))   # tool name: "search"
```

`SearchTool` 设置了 `external_content = True`（结果来自外部，会被防注入护栏包裹）和 `supports_parallel = True`（每次调用都是独立的只读请求，所以模型可以在一轮里并发跑好几次搜索）。

### `CLITool`

把「运行一条白名单内的本地命令」封装成一个工具。由于命令行本身高风险，安全是其核心设计：它默认拒绝（deny-by-default，只有你列出的程序才被允许），构造时把这些程序解析并固定为绝对路径，从不使用 `shell=True`（参数用 `shlex` 分词，未加引号的 shell 运算符会被拒绝），针对解释器、Git、网络和文件系统的高风险标志施加危险参数关卡，并只传入最小环境（`PATH`、`HOME`、`LANG`）。超时、取消、输出过量或后代进程持续占用输出管道时，会终止并回收启动时捕获的进程组；输出在读取时即受硬上限约束。CLI 输出按外部内容处理，回传模型前会加入防注入定界。进程组生命周期契约依赖 POSIX 的 `setsid` / `killpg`，因此 `CLITool` 支持的是 POSIX 主机；Windows 命令执行请使用应用自有的沙箱工具。它被标记为 `requires_confirmation = True`。它的工具名是 `shell`，只有一个参数 `command`。

```python
from agentmaker import CLITool

registry.register(CLITool(allowed_commands=["git", "ls", "grep"], timeout=10.0, max_output_chars=4000))
```

你可以用 `arg_policy` 回调覆盖危险参数关卡，用 `env` 覆盖子进程环境。白名单和参数拒绝规则并不是操作系统沙箱；Git 等命令仍可能执行 hook 或应用/用户配置，因此确认回调必须审查完整命令，高风险部署还应增加容器或平台沙箱。

### `NotesTool`

让 Agent 在一个受限目录内读取和追加笔记文件，从而跨会话保留进度、计划和决策。构造时会以 `0700` 创建尚不存在的 `root`；已有 root 必须是真实目录、归当前用户所有，且不得对同组或其他用户开放任何权限位（例如 `0700`）。绝对路径与 `..` 会被拒绝；笔记路径的每一级父目录和最终文件都不跟随符号链接；非普通文件及带额外硬链接的文件也会被拒绝。它的工具名是 `notes`，参数为 `action`（`read` 或 `append`）、`path`（相对于 `root`）和 `content`（用于 `append`）。

```python
from agentmaker import NotesTool

registry.register(NotesTool(root="./agent_notes"))
```

`NotesTool` 要求 POSIX 环境提供目录相对文件操作与 `O_NOFOLLOW`；缺少这些能力时，构造会抛出 `OSError`。追加操作会尝试取得非阻塞的逐文件 `flock`，协作实例会串行完成 `max_file_bytes` 检查与写入；锁竞争会返回工具错误，而不会无限等待。该锁是建议锁；其它写入者仍需由应用负责协调。

`NotesTool` 采用按动作确认：`append` 会写入磁盘、需要确认，而 `read` 是只读的、无需确认提示即可运行。成功的 `read` 会被标成外部内容，因此模型看到笔记正文前，框架会先用防注入定界符包裹它；`append` 返回的本地确认信息不会被标成外部内容。

## 高风险动作：确认关卡

被标记 `requires_confirmation` 的工具（以及像 `NotesTool` 那样按动作各自决定的工具）必须先通过一个确认回调才能运行。该回调的签名是 `(tool, parameters) -> bool`；只有返回 `True` 时工具才会运行。把它作为 `confirm` 传给 Agent：

```python
from agentmaker import Agent, cli_confirm

agent = Agent("assistant", llm, tools=[CLITool(allowed_commands=["ls"])], confirm=cli_confirm)
```

`cli_confirm` 是内置的命令行提示（在 stdin 上问一个 `y/n` 问题）。如果你不传 `confirm`，高风险调用会默认被安全地拒绝（模型收到一条可读的错误，而不是让动作在未确认的情况下执行）。对于服务端或异步的审批流程，请使用 human-in-the-loop（HITL，人在回路，即让一次运行暂停、等人来批准或修改某个待定动作的模式）；见 [护栏与人在回路](guardrails-and-hitl.md)。

## 工具权限

`ToolPermissions` 以允许列表（allow）和拒绝列表（deny）声明一个 Agent 可以调用哪些工具。它从两个维度来裁决：工具的**名称**和工具的**来源**（origin）。来源才是真正的信任根：名称可以被远程服务器冒充（把一个恶意工具命名为 `search`，蹭你的允许列表），而来源由框架盖章（`"builtin"`，或 MCP 工具的 `"mcp:{namespace}"`），无法被工具定义伪造。

裁决规则是「拒绝优先，然后由允许列表进一步收窄」：

- 命中 `deny` 或 `deny_origins` 立即拒绝（最高优先级）。
- 若启用了某个允许列表（设置了 `allow` 或 `allow_origins`），工具必须匹配被允许的名称或来源才放行。
- 若未启用任何允许列表，则放行工具（只受拒绝列表约束）。

`allow=None` 表示该维度不启用允许列表（「不设限制」）；`allow=[]` 表示一个空的允许列表，拒绝一切。把一个 `ToolPermissions` 作为 `permissions` 传给 Agent：

```python
from agentmaker import Agent, ToolPermissions

permissions = ToolPermissions(allow_origins={"builtin"}, deny={"shell"})
agent = Agent("assistant", llm, tool_registry=registry, permissions=permissions)
```

权限在执行关卡处强制执行：被拒绝的工具直接被驳回，连确认环节都不会进入。

## MCP 集成

MCP（Model Context Protocol，模型上下文协议，Anthropic 提出的、用于向模型暴露工具的开放标准，有时被称为「AI 的 USB-C」）让你可以连接一个发布了一组工具的服务器，并把其中每个工具适配成 agentmaker 的 `Tool`。`MCPClient` 负责管理连接并列出工具；每个工具会变成一个 `MCPTool`，像其他工具一样注册。用 `uv add "agentmaker[mcp]"` 安装这个惰性导入的集成。

支持两种传输方式。用 `async with` 管理连接生命周期，并在该代码块存活期间调用这些工具：

```python
from agentmaker import MCPClient, ToolRegistry

registry = ToolRegistry()

# stdio: run a local server as a subprocess
async with MCPClient(command="python", args=["my_server.py"], namespace="calc") as client:
    tools = await client.load_tools()               # [MCPTool, ...], one per server tool
    registry.register_all(tools, on_conflict="skip")
    # ... use the tools while the connection is alive ...
```

连接远程服务器时，传 `url` 而非 `command`（二者互斥），并可选地通过 `headers` 传请求头，或通过 `auth` 传一个用于 OAuth 的 `httpx.Auth`：

```python
async with MCPClient(url="https://mcp.example.com/mcp", namespace="calendar", auth=my_oauth) as client:
    ...
```

关键的安全要点，全部已替你处理好：

- `namespace` 是**必填**的，也是信任根。每个工具的展示名会变成 `"{namespace}_{original name}"`，其来源被盖章为 `"mcp:{namespace}"`。namespace 由你自己选定，绝不从服务器自报的名称派生（那是攻击者可控的）。这同时也避免了两个服务器各自暴露一个 `search` 工具时的冲突。
- 对加载进来的 MCP 工具，`requires_confirmation` 默认为 `True`，因为远程工具不可信；只有在你审查过该服务器之后，才把它降为 `False`。
- `MCPTool` 设置了 `external_content = True`，所以结果会被防注入护栏包裹。
- 合法的根 `inputSchema` 会原样保留其结构，同时用于模型暴露和本地参数校验，包括本地 `$ref`/`$defs` 与根约束。过大、过深或引用外部地址的 schema 会在使用前被拒绝。
- 每个工具定义都会得到一个指纹（对其远程名称、描述和输入 schema 计算的 sha256）。`expected_fingerprints` 是精确的 `{展示名称: sha256}` 固定集合：指纹不符、服务器返回集合中未列出的工具，或服务器漏掉集合中已固定的工具，都会使 `load_tools` 失败。
- `max_tools` 会拒绝过大的远程工具目录；`max_result_chars` 限制保留文本。结构化内容在相关的字节、深度与节点预算内复制，无法容纳时替换为 `{"truncated": true}`；预算检查前不会调用无界序列化器。
- 远程 URL 默认必须使用 HTTPS；回环地址可使用明文 HTTP，非回环 HTTP 必须显式传入 `allow_insecure_http=True`。URL 中嵌入的凭据会被拒绝，描述和结果中的 ASCII 控制字符与 Unicode 格式控制符会被移除（emoji 与复杂文字所需的连接符和软连字符会保留）。
- `timeout` 同时覆盖初始化、`list_tools` 和每次 `call_tool`；设为 `None` 会禁用超时，只适合由应用自行提供取消策略的场景。

用 `on_conflict="skip"`（而不是在第一次名称冲突时报错）来注册，可以避免一个重复项就中断整个加载循环。

## 运行时工具挑选（Tool-RAG）

一旦一个 Agent 有很多工具，把每个工具的完整 schema 都塞进 prompt 既昂贵又会拉低准确率。Tool-RAG（RAG 即 retrieval-augmented generation，检索增强生成，只检索相关条目而不是把一切都发过去）只为当前输入检索出最相关的工具，并只展开那一个子集。`ToolRetriever` 把每个工具的名称、描述和参数名索引进一个共享的检索器，并返回最匹配的若干项：

```python
from agentmaker import ToolRetriever

# `retriever` is a HybridRetriever; see the Retrieval & RAG guide for how to build one.
tool_retriever = ToolRetriever(registry, retriever, top_k=8, always_include=("tool_search",))
tool_retriever.index()                                     # load every tool's name + description

names = tool_retriever.retrieve("convert between currencies")     # list of tool names, most relevant first
schema = tool_retriever.schema_for("convert between currencies")  # function-calling schema for that subset
```

三个旋钮保证它可靠：

- `always_include`：绕过检索、始终留在子集里的工具名（那些绝不能被 top-k 截断挤掉的工具）。
- `on_empty`：零命中时的回退，默认为 `"all"`（回退到完整目录），确保模型永远不会被交给零个工具。
- `selector`：一个可选的截断策略回调，用分数阈值或拐点（knee-point）截断来替代固定的 top-k。

把一个检索器作为 `tool_retriever` 传给 Agent，它就会为每一轮的输入自动挑出相关子集：

```python
agent = Agent("assistant", llm, tool_registry=registry, tool_retriever=tool_retriever)
```

一次性预选有个盲区：在多步任务里，第二步需要哪个工具，可能取决于第一步的输出。`ToolSearchTool` 补上了这个缺口，它把工具检索本身做成一个模型可以在运行途中调用的工具。它返回一份匹配工具的目录外加一个 `discovered` 列表，循环会把这些工具并入本次运行剩余部分的可用工具集：

```python
from agentmaker import ToolSearchTool

registry.register(ToolSearchTool(tool_retriever, top_k=5))   # tool name: "tool_search"
```

把 `tool_search` 设为一个 `always_include` 条目（如上文检索器所示），让它始终可用。构建底层检索器见 [检索与 RAG](retrieval-and-rag.md)。
