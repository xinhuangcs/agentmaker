"""Async: the framework is async to the core, every capability has an `a*` coroutine twin.

Use `await agent.arun(...)` instead of `agent.run(...)`; memory exposes asearch / aadd / aupdate,
RAG exposes aingest_text / aingest_file, etc. Token streaming lives one layer down on the LLM
client (`async for chunk in llm.stream(...)`). Hermetic via ScriptedLLM.

    uv run python examples/09_async.py
"""
import asyncio

from agentmaker import Agent
from agentmaker.testing import ScriptedLLM


async def main():
    # The async twin of agent.run(...).
    agent = Agent("assistant", ScriptedLLM(["Hello from an async run."]))
    result = await agent.arun("hi")
    print("arun:", result.final_output)

    # Token streaming is exposed on the LLM client as an async generator.
    llm = ScriptedLLM(["streamed piece by piece"])
    chunks = [chunk async for chunk in llm.stream([{"role": "user", "content": "hi"}])]
    print("stream chunks:", chunks)


asyncio.run(main())
