"""Orchestration recipes: Reflection (self-critique) and Plan-and-Solve.

Both are built on the same single-loop Agent, so they take the same LLM and tools.
Hermetic via ScriptedLLM: the scripts stand in for what a real model would generate.

    uv run python examples/06_workflows.py
"""
from agentmaker import PlanAgent, ReflectionAgent
from agentmaker.testing import ScriptedLLM

# Reflection: draft -> critique -> refine, looping until the critic replies "GOOD ENOUGH"
# (the default English pass signal; the Chinese pack uses a Chinese one).
reflection = ReflectionAgent("writer", ScriptedLLM([
    "The Earth orbits the Sun.",                              # draft
    "Add that one orbit takes about 365 days.",              # critique
    "The Earth orbits the Sun once every ~365 days.",        # refine
    "GOOD ENOUGH",                                            # critique -> pass, stop
]), max_turns=3)
print("Reflection:", reflection.run("Explain Earth's orbit in one sentence.").final_output)

# Plan-and-Solve: break the task into an ordered plan, execute each step, then synthesize.
plan = PlanAgent("solver", ScriptedLLM([
    '{"steps": ["Name the capital of Denmark", "State its approximate population"]}',  # plan (structured)
    "The capital of Denmark is Copenhagen.",                  # step 1 execution
    "Copenhagen has roughly 660,000 residents.",             # step 2 execution
    "Copenhagen is Denmark's capital, home to about 660,000 people.",  # synthesis
]))
print("Plan:", plan.run("Tell me about Denmark's capital.").final_output)
