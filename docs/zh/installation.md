# 安装

```bash
pip install agentmaker
```

需要 **Python 3.12+**。

## 核心安装包含哪些内容

基础包开箱即用，无需任何可选附加项：

- 基于 OpenAI 兼容协议的多厂商 LLM 调用（`openai`）
- 通过 Pydantic 实现结构化输出（`pydantic`）
- 对工具参数进行 JSON Schema 校验（`jsonschema`）
- 本地向量库 `SqliteVecStore`（`sqlite-vec`）
- 面向 FTS5 索引（SQLite 内置的全文检索引擎）的中日韩（CJK）分词感知关键词检索（`jieba`）
- 解析技能（skill）的 `SKILL.md` 前置元数据（frontmatter）（`pyyaml`）

## 可选附加项

只安装你需要的能力。每个后端都是惰性导入的，因此未安装的附加项在导入时不会产生任何开销。

```bash
pip install "agentmaker[anthropic]"        # one extra
pip install "agentmaker[anthropic,rag]"    # several
pip install "agentmaker[all]"              # everything below
```

| 附加项 | 新增能力 |
|---|---|
| `anthropic` | Anthropic 原生协议适配器 |
| `gemini` | Google Gemini 原生协议适配器 |
| `search` | `SearchTool` 后端：DuckDuckGo（无需密钥）、Tavily、Brave、SerpAPI |
| `rag` | 面向 RAG（检索增强生成，先检索资料再交给 LLM 作答）的文档加载：PDF / DOCX / HTML 转 Markdown |
| `rerank` | Cohere 多语言重排器（reranker） |
| `mcp` | MCP（Model Context Protocol，模型上下文协议）工具集成 |
| `otel` | OpenTelemetry（OTel，一套开放的可观测性追踪标准）trace 导出 |
| `devtools` | Trace Detective：用于诊断 agent 运行过程的本地网页界面 |

## 厂商 API 密钥

`LLMClient` 会从环境变量中读取对应厂商的 API 密钥。设置与你所用厂商相匹配的变量，例如：

| 厂商（`LLMClient` 的参数） | 环境变量 |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `gemini` | `GEMINI_API_KEY`（或 `GOOGLE_API_KEY`） |
| `dashscope` | `DASHSCOPE_API_KEY` |
| `moonshot` | `MOONSHOT_API_KEY` |
| `zhipu` | `ZHIPUAI_API_KEY` |
| `modelscope` | `MODELSCOPE_API_KEY` |

运行前先在 shell 里设置变量：

```bash
export DEEPSEEK_API_KEY="sk-..."          # macOS / Linux
```

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."          # Windows PowerShell
```

更喜欢用 `.env` 文件？`.env` 里的变量名就用上表中的名字——`.env` 只是设置同一批环境变量的另一种方式：

```text
# .env
DEEPSEEK_API_KEY=sk-...
```

agentmaker 有意不自己读取 `.env`（加载环境文件是应用层的决定），但标准的 [python-dotenv](https://pypi.org/project/python-dotenv/) 照常可用：

```python
from dotenv import load_dotenv
load_dotenv()                             # reads .env from the working directory

from agentmaker import Agent, LLMClient
agent = Agent("assistant", LLMClient("deepseek"))
```

记得把 `.env` 排除在版本控制之外（加进 `.gitignore`）。

本地引擎（`ollama`、`vllm`、`sglang`）无需密钥。完整的厂商列表以及 `provider:model` 语法，见 [LLM 客户端与厂商](guides/llm-clients.md)。

## 使用 uv 安装

如果你使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv add agentmaker
uv add "agentmaker[all]"
```
