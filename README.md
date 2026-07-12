# agentmaker

[![CI](https://github.com/xinhuangcs/agentmaker/actions/workflows/ci.yml/badge.svg)](https://github.com/xinhuangcs/agentmaker/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agentmaker)](https://pypi.org/project/agentmaker/)
[![Python](https://img.shields.io/pypi/pyversions/agentmaker)](https://pypi.org/project/agentmaker/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/xinhuangcs/agentmaker/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://xinhuangcs.github.io/agentmaker/)

A general-purpose Python framework for building LLM agents and multi-agent systems, with tools, memory, retrieval / RAG, context engineering, guardrails, human-in-the-loop, and observability built in. Async-first, fully typed, and easy to debug: a built-in LLM debugger pinpoints a failed run's first bad step, root cause, and fix.

<p>
  <a href="https://agentmaker.xinhuang.me/"><img src="docs/assets/readme-website-button.svg" alt="Website" height="42"></a>
  <a href="https://xinhuangcs.github.io/agentmaker/"><img src="docs/assets/readme-docs-en-button.svg" alt="English documentation" height="42"></a>
  <a href="https://xinhuangcs.github.io/agentmaker/zh/"><img src="docs/assets/readme-docs-zh-button.svg" alt="Chinese documentation" height="42"></a>
</p>

## Highlights

- **One agent, many recipes**: a single agent loop for chat and tool use, plus plan-and-execute and reflection workflows, declarative agent specs, and multi-agent orchestration.
- **Any LLM provider**: native OpenAI, Anthropic, and Gemini, plus DeepSeek, Moonshot, Zhipu, local models (Ollama, vLLM, SGLang), and any OpenAI-compatible endpoint, with function calling, streaming, structured output, multimodal, and prompt caching.
- **Tools**: turn any typed function into a tool with a one-line decorator, use the built-in tools, connect MCP servers, and auto-select the relevant tools at runtime when there are many.
- **Retrieval, RAG, memory**: hybrid retrieval (vectors, keywords, and rank fusion) with no external services, a full RAG pipeline with source citations, and long-term memory that extracts and updates facts.
- **Batteries included, every backend swappable**: local SQLite defaults with no database to run, and every backend (embeddings, vector store, reranker, session and checkpoint stores, trace exporter, chunker) sits behind an interface you can swap, for example to pgvector.
- **Context engineering**: assembles each prompt under an explicit token budget, with history compaction, relevance-based selection, and pluggable token counting.
- **Guardrails & human-in-the-loop**: input and output guardrails, an approve-before-run gate for high-risk tools, lifecycle hooks, searchable sessions, checkpoints, run limits, and cancellation.
- **Observability & Trace Detective**: trace every run to JSONL, SQLite, or OpenTelemetry, then have a built-in LLM debugger pinpoint a failed run's first bad step, root cause, and fix, in your terminal or a local web UI.
- **Overridable prompts**: list and replace any built-in prompt; English defaults with a Chinese language pack.
- **Multi-tenant**: a single scope label isolates retrieval, memory, and sessions across users, agents, and apps.
- **Test-friendly**: a built-in LLM test double runs your agents in CI with no API key and no network.

## Installation

```bash
pip install agentmaker            # core batteries, works out of the box
pip install "agentmaker[all]"     # every optional extra below
```

Requires Python 3.12+. The core install already covers multi-provider LLM calls, structured output, tool-argument validation, and local hybrid retrieval (vectors plus CJK-aware keyword search). The optional extras below add the rest:

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

## Quickstart

Define a tool, hand it to an agent, and the model calls it when it needs to:

```python
from agentmaker import Agent, LLMClient, tool


@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: sunny, 24C"


agent = Agent("assistant", LLMClient("deepseek"), tools=[get_weather])
print(agent.run("What's the weather in Copenhagen?").final_output)
```

### Mount more capabilities

Every capability is a few more arguments to the same constructor. Here is that agent given semantic long-term memory, a model-invoked skill library, retrieved context, and an input guardrail:

```python
from agentmaker import (Agent, LLMClient, Memory, MemoryStore, ContextBuilder, CallableSource, SkillLoader, CallableGuardrail)
from agentmaker.retrieval import build_sqlite_hybrid, OpenAIEmbedder

llm = LLMClient("openai")
memory = Memory(build_sqlite_hybrid(OpenAIEmbedder()), MemoryStore())
skills = SkillLoader("./skills")

agent = Agent(
    "assistant", llm,
    tools=[get_weather],  # function calling
    sources=[CallableSource("memory", memory.search)],  # memory pulled into context each turn
    context_builder=ContextBuilder(),  # assemble context under a token budget
    system_prompt=f"You are a helpful assistant.\nSkills:\n{skills.catalog()}",  # model-invoked skills
    input_guardrails=[CallableGuardrail(lambda t: len(t) < 4000, message="input too long")],  # validate input
)
print(agent.run("Plan a day in Copenhagen, and remember I'm vegetarian.").final_output)
```

Every argument past `llm` is optional, so you add capabilities one at a time, and the same pattern reaches the rest of the framework: RAG retrieval as another `sources=` entry, MCP servers and sub-agents (`AgentTool`) as `tools=`, a `SmartWriter` that extracts and diffs memories instead of storing raw text, structured output and streaming, the `PlanAgent` / `ReflectionAgent` workflow recipes, plus sessions, checkpoints (human-in-the-loop), permissions, and history compaction. See the [Highlights](#highlights) above for the full list.

### Debug it with an agent

For development, attach the trace-based agent debugger. When a run fails, `DoctorHook` prints an LLM-written diagnosis (first bad step, root cause, suggested fix) straight to your terminal:

```python
from agentmaker import Agent, Tracer
from agentmaker.devtools import DoctorHook

tracer = Tracer()
agent = Agent("assistant", llm, tools=[get_weather], tracer=tracer, hooks=[DoctorHook(tracer)])
print(agent.run("What's the weather in Copenhagen?").final_output)
```

`DoctorHook` prints inline; for the full picture, `python -m agentmaker.devtools` opens Trace Detective, a local web page that visualizes a recorded run step by step (its LLM calls, tool calls, and guardrails) and, on demand, has an LLM pinpoint the first bad step, root cause, and fix. Both are themselves agentmaker agents, so the framework debugs its own runs. You debug agents with an agent.

## Learn more

- [`examples/`](https://github.com/xinhuangcs/agentmaker/tree/main/examples): sixteen runnable, numbered examples, from quickstart to skills.
- [`CHANGELOG.md`](https://github.com/xinhuangcs/agentmaker/blob/main/CHANGELOG.md)
- Versioning: pre-1.0, minor versions may introduce breaking changes and patch versions only fix. Pin `agentmaker>=0.1,<0.2`.

## License

[MIT](https://github.com/xinhuangcs/agentmaker/blob/main/LICENSE)
