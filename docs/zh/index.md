# agentmaker

一个通用的 Python 框架，用于构建 LLM Agent（大模型智能体）与多 agent 系统，内置工具（tool）、记忆、检索 / RAG（Retrieval-Augmented Generation，检索增强生成，即先检索资料再让模型作答）、上下文工程、护栏、human-in-the-loop（HITL，人在环中，即关键步骤交由人工确认）以及可观测性。

## 安装

```bash
pip install agentmaker            # core batteries, works out of the box
pip install "agentmaker[all]"     # every optional capability
```

需要 Python 3.12+。完整的可选附加项矩阵见 [安装](installation.md)。

## 30 秒示例

无需任何配置即可运行（无需 API key，无需联网），与 [`examples/01_quickstart.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/01_quickstart.py) 中随包提供的代码完全一致：

```python
from agentmaker import Agent, tool
from agentmaker.testing import ScriptedLLM


@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: sunny, 24C"


# With a real model the LLM decides when to call the tool. Here we script that decision:
# first it asks to call get_weather(city="Copenhagen"), then it writes the final answer.
llm = ScriptedLLM([
    ScriptedLLM.tool_call("get_weather", {"city": "Copenhagen"}),
    "It's sunny and 24C in Copenhagen today.",
])

agent = Agent("assistant", llm, tools=[get_weather])
result = agent.run("What's the weather in Copenhagen?")
print(result.final_output)
```

要使用真实模型，把 `ScriptedLLM(...)` 换成 `LLMClient("deepseek")`（或 `"openai"` / `"anthropic"` / `"gemini"`），并在环境中设置对应的 API key；此后由模型自己决定何时调用工具。

## 下一步去哪里

- 初次接触？从 [快速上手](guides/quickstart.md) 开始。
- 从 **指南** 中挑一个能力：[LLM 客户端](guides/llm-clients.md)、[Agent 与工作流](guides/agents.md)、[工具](guides/tools.md)、[结构化输出](guides/structured-output.md)、[记忆](guides/memory.md)、[检索与 RAG](guides/retrieval-and-rag.md)、[上下文工程](guides/context-engineering.md)、[护栏与 HITL](guides/guardrails-and-hitl.md)、[可观测性](guides/observability.md)、[提示词注册表](guides/prompts.md)、[技能](guides/skills.md)。
- 在 [API 参考](reference.md)（由源码 docstring 生成）中查阅确切的函数签名。

## 版本策略

1.0 之前：次版本号可能引入破坏性变更，修订版本号只做修复。请锁定 `agentmaker>=0.1,<0.2`。参见 [更新日志](https://github.com/xinhuangcs/agentmaker/blob/main/CHANGELOG.md)。

## 许可证

[MIT](https://github.com/xinhuangcs/agentmaker/blob/main/LICENSE)。
