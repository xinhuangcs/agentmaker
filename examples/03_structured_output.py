"""Structured output: get a validated Pydantic object back instead of free text.

Pass an output_schema (a Pydantic model) to run(); the model is asked to emit JSON, which
the framework parses and validates into an instance. Hermetic via ScriptedLLM (which
returns the JSON directly).

    uv run python examples/03_structured_output.py
"""
from pydantic import BaseModel

from agentmaker import Agent
from agentmaker.testing import ScriptedLLM


class Person(BaseModel):
    name: str
    age: int


llm = ScriptedLLM(['{"name": "Ada", "age": 36}'])
agent = Agent("extractor", llm)

person = agent.run("Extract the person from: Ada is 36.", output_schema=Person).final_output
print(f"{type(person).__name__}(name={person.name!r}, age={person.age})")
