"""agentmaker.devtools.diagnose: the LLM half of Trace Detective.

Feeds the deterministic timeline from trace_parser to an LLM and gets back a validated three-part verdict:
where the run first went wrong, the root cause, and how to fix it. Dogfoods the framework itself: the
diagnosis runs through a plain agentmaker Agent with output_schema (pydantic-validated structured output),
so any LLMClient the framework supports works here unchanged. The system prompt lives in the prompt catalog
(key "devtools.diagnose"), so language packs and app overrides cover it like any other built-in prompt.
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from ..agents import Agent
from ..prompts import DEFAULT_PROMPTS
from .trace_parser import TraceRun, parse_trace, pick_run, render_run

# Languages the web UI offers; free-form language strings are accepted by diagnose() too.
LANGUAGES = {"en": "English", "zh": "Simplified Chinese"}


class TraceDiagnosis(BaseModel):
    """The structured verdict Trace Detective returns (also the exact JSON contract the web API serves)."""
    healthy: bool = Field(description="True when the run shows no real failure")
    first_bad_step: Optional[int] = Field(None, description="#N of the earliest step failing the counterfactual test; null when healthy")
    what_went_wrong: str = Field(description="The causal chain from the first failure to the final symptom, each link citing its #N step")
    root_cause: str = Field(description="Why it happened, as specifically as the evidence supports; name missing evidence instead of inventing")
    suggested_fix: str = Field(description="The smallest change that removes the root cause, naming the exact framework knob; end with one sentence on how to verify the fix")
    confidence: Literal["low", "medium", "high"] = Field(description="Evidence strength: high = fact-backed complete chain, medium = one inferred link, low = key evidence missing")


def diagnose(run: TraceRun, llm, *, language: Optional[str] = None, max_chars: int = 20_000,
             extra_context: Optional[str] = None, prompts=None, **kwargs) -> TraceDiagnosis:
    """Diagnose one parsed run with an LLM: earliest failure, root cause, fix.

    Args:
        run: A TraceRun from parse_trace / pick_run.
        llm: Any LLMClient-compatible client (ScriptedLLM works for tests).
        language: Language for the three text fields; a LANGUAGES key ("en"/"zh") or a free-form
            language name passed straight into the prompt. None (default) follows the active prompt
            catalog: each language pack declares its own output language in the
            "devtools.diagnose_language" entry, so installing a pack switches the verdict language too.
        max_chars: Character budget for the rendered timeline (bounds the prompt size).
        extra_context: Optional out-of-trace evidence appended after the timeline, e.g. the text of an
            uncaught exception that ended the run (used by DoctorHook's on_error path).
        prompts: Optional PromptRegistry; defaults to the global DEFAULT_PROMPTS. The system prompt is
            the catalog entry "devtools.diagnose", so a language pack (chinese_registry()) or a
            process-wide DEFAULT_PROMPTS.override(...) switches it like any built-in prompt; the
            registry is also handed to the inner Agent, so its harness-level prompts follow too.
        **kwargs: Passed through to Agent.run (e.g. temperature).

    Returns:
        TraceDiagnosis: Validated instance; first_bad_step is clamped to None if the model cites a
        step number outside the run (so UI step links never dangle).
    """
    registry = prompts if prompts is not None else DEFAULT_PROMPTS
    language_name = (registry.text("devtools.diagnose_language") if language is None
                     else LANGUAGES.get(language, language))
    prompt = registry.render("devtools.diagnose", language=language_name)
    timeline = render_run(run, max_chars=max_chars)
    if extra_context:
        timeline += ("\n\nThe run terminated with an uncaught exception (the timeline may end abruptly "
                     f"at the failure point):\n{extra_context}")
    agent = Agent("trace-detective", llm, system_prompt=prompt, prompts=registry)
    result = agent.run(timeline, output_schema=TraceDiagnosis, **kwargs)
    verdict: TraceDiagnosis = result.final_output
    if verdict.first_bad_step is not None and not 0 <= verdict.first_bad_step < len(run.steps):
        verdict.first_bad_step = None
    return verdict


def diagnose_trace(source: Union[str, list], llm, *, run_id: Optional[str] = None,
                   **kwargs) -> tuple[TraceRun, TraceDiagnosis]:
    """One-call convenience: parse a whole trace, pick one run (run_id, or the most recent), diagnose it.

    Returns:
        tuple[TraceRun, TraceDiagnosis]: The run that was diagnosed (for rendering) and its verdict.
    """
    run = pick_run(parse_trace(source), run_id)
    return run, diagnose(run, llm, **kwargs)
