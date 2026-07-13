# Changelog

## 0.2.0

- Hardened reliability across the agent lifecycle (per-scope serialization, cooperative deadlines, at-most-once checkpoint recovery), client shutdown, and retrieval/memory/session consistency; custom backends built on the 0.1.0 interfaces keep working through fallbacks.
- Tightened tool security (anti-injection wrapping, Notes/CLI/MCP hardening, bounded file and skill loading), refreshed model limits from official vendor docs, preserved provider continuation state across tool turns, and added PDF/DOCX converters to the `rag` extra.

## 0.1.0

Initial public release.

- A single-loop `Agent` (chat / react), plus `PlanAgent` and `ReflectionAgent` workflow recipes, declarative `AgentSpec` / `build_agent`, and `AgentTool` for multi-agent composition.
- A multi-provider `LLMClient` over OpenAI-compatible, Anthropic, and Gemini protocols, with function calling, streaming, structured output, and multimodal input.
- Tools via `@tool` or the `Tool` base class, a tool registry, built-in tools, MCP integration, and runtime tool retrieval.
- Hybrid retrieval (vectors + keywords + rank fusion), a RAG ingestion pipeline, and a memory subsystem.
- Context engineering, guardrails, human-in-the-loop, sessions, checkpoints, and observability (tracing with pluggable exporters).
- **Trace Detective** (optional `[devtools]`): record a run, then get an LLM-written diagnosis of what went wrong (first bad step, root cause, fix) in a local web UI (`python -m agentmaker.devtools`) or inline via `DoctorHook`. It is itself an agentmaker agent, so the framework diagnoses its own runs.
- Ships `py.typed`; includes a test double (`ScriptedLLM`) for hermetic, no-network testing.
