# Installation

```bash
pip install agentmaker
```

Requires **Python 3.12+**.

## What the core install includes

The base package works out of the box, with no optional extras:

- Multi-provider LLM calling over the OpenAI-compatible protocol (`openai`)
- Structured output via Pydantic (`pydantic`)
- JSON Schema validation of tool arguments (`jsonschema`)
- A local vector store, `SqliteVecStore` (`sqlite-vec`)
- CJK-aware keyword search for the FTS5 index (`jieba`)
- `SKILL.md` frontmatter parsing for skills (`pyyaml`)

## Optional extras

Install only the capabilities you need. Each backend is imported lazily, so an extra you do not install costs nothing at import time.

```bash
pip install "agentmaker[anthropic]"        # one extra
pip install "agentmaker[anthropic,rag]"    # several
pip install "agentmaker[all]"              # everything below
```

| Extra | Adds |
|---|---|
| `anthropic` | Anthropic native protocol adapter |
| `gemini` | Google Gemini native protocol adapter |
| `search` | `SearchTool` backends: DuckDuckGo (no key needed), Tavily, Brave, SerpAPI |
| `rag` | Document loading for RAG: PDF / DOCX / HTML to Markdown |
| `rerank` | Cohere multilingual reranker |
| `mcp` | MCP (Model Context Protocol) tool integration |
| `otel` | OpenTelemetry trace export |
| `devtools` | Trace Detective: local web UI for diagnosing agent runs |

## Provider API keys

`LLMClient` reads the API key for a provider from your environment. Set the variable that matches the provider you use, for example:

| Provider (argument to `LLMClient`) | Environment variable |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `gemini` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| `dashscope` | `DASHSCOPE_API_KEY` |
| `moonshot` | `MOONSHOT_API_KEY` |
| `zhipu` | `ZHIPUAI_API_KEY` |
| `modelscope` | `MODELSCOPE_API_KEY` |

Set the variable in your shell before running:

```bash
export DEEPSEEK_API_KEY="sk-..."          # macOS / Linux
```

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."          # Windows PowerShell
```

Prefer a `.env` file? The variable names inside `.env` are exactly the ones in the table above — a `.env` file is just another way to set the same environment variables:

```text
# .env
DEEPSEEK_API_KEY=sk-...
```

agentmaker deliberately does not read `.env` itself (loading environment files is an application-level decision), but the standard [python-dotenv](https://pypi.org/project/python-dotenv/) works as usual:

```python
from dotenv import load_dotenv
load_dotenv()                             # reads .env from the working directory

from agentmaker import Agent, LLMClient
agent = Agent("assistant", LLMClient("deepseek"))
```

Keep `.env` out of version control (add it to `.gitignore`).

Local engines (`ollama`, `vllm`, `sglang`) need no key. See [LLM clients & providers](guides/llm-clients.md) for the full provider list and the `provider:model` syntax.

## Installing with uv

If you use [uv](https://docs.astral.sh/uv/):

```bash
uv add agentmaker
uv add "agentmaker[all]"
```
