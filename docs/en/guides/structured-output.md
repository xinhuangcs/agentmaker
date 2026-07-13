# Structured output

Structured output makes an agent hand back a typed, validated object instead of a free-text string. You describe the shape you want as a Pydantic model (Pydantic is a Python library that validates data against a declared schema), pass it to `run()`, and get back a `RunResult` whose `.final_output` is an instance of that model, already parsed and checked. Reach for this whenever the reply feeds code rather than a human: extracting fields from text, classifying into a fixed set of labels, or producing a payload another system will consume.

## The basics

Pass a Pydantic model as `output_schema` to `run()`. The completed `RunResult` carries the validated instance in `.final_output`:

```python
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
```

This is [`examples/03_structured_output.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/03_structured_output.py), copied verbatim. It runs with no API key and no network: `ScriptedLLM` is the framework's test double (a stand-in that replays canned replies instead of calling a real model), and here it returns the JSON object directly, which the framework parses and validates into a `Person`. `person` is a real `Person` instance, so `person.name` and `person.age` are typed attributes, not dictionary lookups.

To run against a real model, swap the test double for a live client and keep everything else the same:

```python
from agentmaker import LLMClient

agent = Agent("extractor", LLMClient("deepseek"))
person = agent.run("Extract the person from: Ada is 36.", output_schema=Person).final_output
```

Set the matching API key in your environment (see [LLM clients](llm-clients.md)). The model now decides what to emit, and the framework holds it to your schema.

!!! note
    Structured output runs as plain question-answering: no tool loop, non-streaming. If you build the agent with tools, they are not offered on a `run(..., output_schema=...)` call. Use a schema when you want a single validated object back, and the tool loop (see [Tools](tools.md)) when you want the model to take actions.

## What happens under the hood

When you pass `output_schema`, the agent routes the call to its harness (the internal layer that wraps every model call), which does the following:

1. **Derives a JSON Schema** from your model via `model_json_schema()`. JSON Schema is a standard, language-neutral way to describe the fields and types a JSON object must have.
2. **Prepends a system instruction** telling the model to return only a single JSON object that conforms to that schema, with no explanatory text and no markdown code fences.
3. **Calls the model** with the schema attached, then extracts JSON from the reply. Extraction is lenient: it takes everything from the first `{` to the last `}`, so a stray ```json fence or surrounding prose does not break parsing.
4. **Validates** the extracted JSON against your model with `model_validate_json()`. On success you get the instance.
5. **Retries on failure.** If validation fails, the invalid output plus a short correction note are fed back to the model and the call is repeated. The default is one retry; if it still fails after the retries are exhausted, the framework raises `LLMResponseError`.

Because validation is Pydantic's own, every guarantee your model declares (required fields, types, constraints, nested models) is enforced. A reply that is well-formed JSON but the wrong shape is treated as a failure and retried, not silently returned.

### How providers enforce the schema

The schema is sent to the model through each provider's native structured-output path when one exists, in addition to being written into the system prompt. For the OpenAI-compatible protocol this depends on the provider's declared capability:

- `json_schema`: the schema is sent as a `response_format` of type `json_schema`, constraining the model at the API level.
- `json_object`: the request asks for valid JSON, and the schema itself is carried by the prompt.
- `none`: nothing is sent at the API level; the prompt instruction is the only guide.

The Anthropic and Gemini protocols use their own native structured paths. Whichever path applies, the result is always Pydantic-validated at the end, so the prompt-only case (`json_object` / `none`) and the native-constrained case converge on the same checked instance. See [LLM clients](llm-clients.md) for how a provider's capabilities are configured.

## Working with the result

`run()` returns a `RunResult`. On a completed run, `.final_output` holds the validated model instance. It is the same object your schema defines, so you can use it directly:

```python
result = agent.run("Extract the person from: Ada is 36.", output_schema=Person)
person = result.final_output
send_to_database(person.name, person.age)
```

Guardrails and conversation history still apply on the structured path. When the framework needs a text form of the output (to check an output guardrail, or to persist the turn to history), it serializes the model with `model_dump_json()`, so what lands in history is the JSON form of your object. See [Guardrails & HITL](guardrails-and-hitl.md) for output guardrails.

## Tuning retries

Extra keyword arguments to `run()` are forwarded down the structured path, so you can raise the retry budget when a model needs more room to correct itself:

```python
person = agent.run(
    "Extract the person from: Ada is 36.",
    output_schema=Person,
    retries=3,
).final_output
```

`retries` counts corrections after the first attempt, so `retries=3` allows up to four model calls in total.

## When it fails

If the model cannot produce a valid object within the retry budget, the framework raises `LLMResponseError` with the last validation error. Handle it where you would handle any other model-call failure:

```python
from agentmaker import LLMResponseError

try:
    person = agent.run("...", output_schema=Person).final_output
except LLMResponseError as exc:
    ...  # log, fall back, or surface the error
```

A raised error means no guessed or partial object is returned. You either get a fully validated instance or an explicit failure, never a half-filled result.
