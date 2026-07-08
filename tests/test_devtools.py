"""Trace Detective (agentmaker.devtools) regression: deterministic parsing / health-check rules / rendering / LLM diagnosis / Web API.

Fully hermetic: the LLM uses the official ScriptedLLM stand-in, and trace events are hand-built to the
trace_events field conventions. Two integration tests cross-check against the real pipeline
(JsonlExporter to disk -> load_trace, and a real Tracer collecting events -> parse_trace) to keep the
devtools/framework event contract from drifting. The web tests skip automatically when fastapi is absent.
"""

import asyncio
import json

import pytest

from agentmaker import Agent, CalculatorTool, JsonlExporter, Tracer
from agentmaker.devtools import (DoctorHook, TraceDiagnosis, TraceParseError, diagnose,
                                   diagnose_trace, load_trace, parse_trace, pick_run, render_run)
from agentmaker.testing import ScriptedLLM


def _events() -> list[dict]:
    """Build sample events covering the main health-check rules: run-a with four steps (tool failure / empty retrieval / truncation), run-b with one step, and one bare event with no run_id."""
    return [
        {"type": "llm_call", "model": "m", "latency_ms": 1200, "finish_reason": "tool_calls",
         "usage": {"prompt_tokens": 700, "completion_tokens": 100, "total_tokens": 800},
         "has_tool_calls": True, "run_id": "run-a", "step_index": 0},
        {"type": "tool_call", "tool": "calculator", "params": {"expression": "1/0"}, "latency_ms": 3,
         "status": "error", "result": "division by zero", "run_id": "run-a", "step_index": 0},
        {"type": "rag_retrieve", "query": "blood oxygen baseline", "hits": 0, "latency_ms": 40,
         "run_id": "run-a", "step_index": 1},
        {"type": "llm_call", "model": "m", "latency_ms": 900, "finish_reason": "length",
         "usage": {"total_tokens": 700}, "run_id": "run-a", "step_index": 1},
        {"type": "llm_call", "model": "m", "latency_ms": 500, "usage": None, "streamed": True,
         "finish_reason": "stop", "run_id": "run-b"},
        {"type": "memory_search", "query": "q", "hits": 3},
    ]


def _diagnosis_json(**overrides) -> str:
    """A valid TraceDiagnosis JSON (ScriptedLLM's scripted reply); fields can be overridden as needed."""
    payload = {"healthy": False, "first_bad_step": 1, "what_went_wrong": "#1 tool failed",
               "root_cause": "bad expression", "suggested_fix": "guard divide by zero",
               "confidence": "high", **overrides}
    return json.dumps(payload)


# ---------- parsing and grouping ----------

def test_parse_groups_runs_in_order():
    runs = parse_trace(_events())
    assert [r.run_id for r in runs] == ["run-a", "run-b", None]   # first-seen order, bare events last
    assert [len(r.steps) for r in runs] == [4, 1, 1]
    assert [s.index for s in runs[0].steps] == [0, 1, 2, 3]       # contiguous within-run indices, i.e. the #N the diagnosis references
    assert runs[0].steps[1].step_index == 0                        # the framework's own step_index preserved verbatim


def test_findings_and_stats():
    run = parse_trace(_events())[0]
    assert [f.code for f in run.steps[1].findings] == ["tool_error"]
    assert [f.code for f in run.steps[2].findings] == ["empty_retrieval"]
    assert [f.code for f in run.steps[3].findings] == ["llm_truncated"]
    assert run.steps[0].findings == []                             # a normal llm_call yields no findings
    assert (run.stats.llm_calls, run.stats.tool_calls) == (2, 1)
    assert run.stats.total_tokens == 1500
    assert (run.stats.errors, run.stats.warnings) == (2, 1)        # tool_error+llm_truncated / empty_retrieval


def test_more_finding_rules():
    events = [
        {"type": "tool_call", "tool": "t", "params": {}, "status": "denied", "result": "banned"},
        {"type": "tool_call", "tool": "t", "params": {}, "status": "invalid_args", "result": "bad params"},
        {"type": "summarize_failed", "run_id": "r"},
        {"type": "index_sync_reconcile", "items": 5, "pending_after": 2},
        {"type": "made_up_event", "x": 1},
    ]
    codes = {step.findings[0].code: step.findings[0].severity for step in parse_trace(events)[-1].steps
             if step.findings}
    assert codes["tool_blocked"] == "warn"
    assert codes["index_not_converged"] == "warn"
    assert codes["unknown_event"] == "warn"
    all_steps = [s for r in parse_trace(events) for s in r.steps]
    assert any(f.code == "tool_error" and f.severity == "error"
               for s in all_steps for f in s.findings if s.event.get("status") == "invalid_args")
    assert any(f.code == "compaction_degraded" for s in all_steps for f in s.findings)


def test_parse_jsonl_text_and_errors():
    text = "\n\n".join(json.dumps(e, ensure_ascii=False) for e in _events())   # reads fine even with blank lines interspersed
    assert [r.run_id for r in parse_trace(text)] == ["run-a", "run-b", None]
    with pytest.raises(TraceParseError, match="line 2"):
        parse_trace('{"type": "llm_call"}\nnot json')
    with pytest.raises(TraceParseError, match="empty"):
        parse_trace("   \n  ")
    with pytest.raises(TraceParseError, match="JSON object"):
        parse_trace("[1, 2]")
    with pytest.raises(TraceParseError, match="type"):
        parse_trace('{"model": "m"}')


def test_pick_run():
    runs = parse_trace(_events())
    assert pick_run(runs).run_id is None                # default picks the last (newest) run
    assert pick_run(runs, "run-a").run_id == "run-a"
    with pytest.raises(TraceParseError, match="nope"):
        pick_run(runs, "nope")
    with pytest.raises(TraceParseError, match="no runs"):
        pick_run([])                                     # empty list gives a clear error, not a bare IndexError


# ---------- rendering ----------

def test_render_run_contains_steps_and_findings():
    text = render_run(parse_trace(_events())[0])
    assert text.startswith("run run-a: 4 steps, 2 llm_calls, 1 tool_calls")
    assert "#1 tool_call tool=calculator status=error" in text
    assert "!! error tool_error" in text


def test_render_run_elides_but_keeps_findings():
    events = [{"type": "llm_call", "model": "m", "latency_ms": 10, "usage": {"total_tokens": 5},
               "finish_reason": "stop", "run_id": "r"} for _ in range(200)]
    events[100] = {"type": "tool_call", "tool": "boom", "params": {}, "status": "error",
                   "result": "kaboom", "run_id": "r"}
    text = render_run(parse_trace(events)[0], max_chars=4000)
    assert len(text) <= 4000                             # the budget is a hard cap (including the fallback truncation marker)
    assert "steps omitted" in text                       # the middle section really is elided
    assert "#100 tool_call" in text and "tool_error" in text   # but the step carrying findings must survive


# ---------- cross-check against the real pipeline (contract drift guard) ----------

def test_load_trace_roundtrip_with_jsonl_exporter(tmp_path):
    path = str(tmp_path / "run.jsonl")
    exporter = JsonlExporter(path)
    for event in _events():
        exporter.export(event)
    exporter.close()
    runs = load_trace(path)
    assert [r.run_id for r in runs] == ["run-a", "run-b", None]
    assert runs[0].stats.total_tokens == 1500


def test_parse_events_from_real_tracer():
    tracer = Tracer()
    agent = Agent("t", ScriptedLLM(["hello"]), tracer=tracer)
    assert agent.run("hi").final_output == "hello"
    runs = parse_trace(tracer.events)                    # events emitted by the real Harness must parse directly
    assert len(runs) == 1 and runs[0].run_id             # carries a real run_id
    llm_steps = [s for s in runs[0].steps if s.type == "llm_call"]
    assert llm_steps and "model=test" in llm_steps[0].summary


# ---------- LLM diagnosis ----------

def test_diagnose_with_scripted_llm():
    run = parse_trace(_events())[0]
    verdict = diagnose(run, ScriptedLLM([_diagnosis_json()]))
    assert isinstance(verdict, TraceDiagnosis)
    assert (verdict.healthy, verdict.first_bad_step, verdict.confidence) == (False, 1, "high")
    assert verdict.suggested_fix == "guard divide by zero"


def test_diagnose_clamps_out_of_range_step():
    run = parse_trace(_events())[0]
    verdict = diagnose(run, ScriptedLLM([_diagnosis_json(first_bad_step=99)]))
    assert verdict.first_bad_step is None                # out-of-range reference is clamped to None so the UI link can't dangle


def test_diagnose_trace_convenience():
    run, verdict = diagnose_trace(_events(), ScriptedLLM([_diagnosis_json()]), run_id="run-a")
    assert run.run_id == "run-a" and verdict.first_bad_step == 1


def test_diagnose_prompt_follows_language_pack():
    """The diagnosis system prompt is centralized in the prompt registry (key devtools.diagnose): English by default, the Chinese pack swaps the whole set, and prompts= injection flows through.
    When language is omitted, the output language follows the pack's self-declared devtools.diagnose_language meta entry."""
    from agentmaker import DEFAULT_PROMPTS
    from agentmaker.prompts.packs import chinese_registry

    en = DEFAULT_PROMPTS.render("devtools.diagnose", language="English")
    zh = chinese_registry().render("devtools.diagnose", language="简体中文")
    assert "Trace Detective" in en and "in English" in en
    assert "轨迹侦探" in zh and "#N" in zh and "first_bad_step" in zh   # Chinese pack in effect, protocol tokens preserved
    for text in (en, zh):                                               # framework knowledge injected in both: event semantics + failure handbook + real knobs
        assert "WindowBudgetConfig" in text and "RunPolicy" in text and "Scope" in text and "max_turns" in text
    assert DEFAULT_PROMPTS.text("devtools.diagnose_language") == "English"
    assert chinese_registry().text("devtools.diagnose_language") == "简体中文"   # the pack self-declares its output language
    run = parse_trace(_events())[0]
    verdict = diagnose(run, ScriptedLLM([_diagnosis_json()]), prompts=chinese_registry())
    assert verdict.first_bad_step == 1                                  # injecting a non-default registry with language omitted works end-to-end


def test_diagnose_default_language_reaches_prompt():
    """Content-level assertion: with language omitted, the Chinese pack's system prompt and its self-declared output language actually reach the messages sent to the model."""
    from agentmaker.prompts.packs import chinese_registry

    class SpyLLM(ScriptedLLM):
        """Records the messages chat receives so the prompt content can be asserted."""
        async def chat(self, messages, **kwargs):
            self.seen = messages
            return await super().chat(messages, **kwargs)

    spy = SpyLLM([_diagnosis_json()])
    diagnose(parse_trace(_events())[0], spy, prompts=chinese_registry())
    joined = "\n".join(m["content"] for m in spy.seen if isinstance(m.get("content"), str))
    assert "轨迹侦探" in joined     # Chinese system prompt in effect
    assert "简体中文" in joined     # the pack's self-declared output language is filled into {language}


# ---------- DoctorHook (auto-diagnose to terminal on error) ----------

def _failing_agent(tracer, hooks):
    """Real-pipeline failure case: the calculator genuinely fails on 1/0 and the LLM ignores the failure and fabricates an answer."""
    llm = ScriptedLLM([ScriptedLLM.tool_call("calculator", {"expression": "1/0"}), "it is 0"])
    return Agent("t", llm, tools=[CalculatorTool()], tracer=tracer, hooks=hooks)


def test_doctor_hook_prints_diagnosis_on_error_finding(capsys):
    tracer = Tracer()
    doctor_llm = ScriptedLLM([_diagnosis_json()])
    _failing_agent(tracer, [DoctorHook(tracer, doctor_llm)]).run("1/0?")
    out = capsys.readouterr().out
    assert "Trace Detective: run" in out and "diagnosing with" in out   # announce before spending on the paid LLM call
    assert "guard divide by zero" in out                                # the three-part verdict actually prints
    assert doctor_llm.calls == 1


def test_doctor_hook_stays_silent_on_healthy_run(capsys):
    tracer = Tracer()
    doctor_llm = ScriptedLLM([])                                        # must not be called, so the script is left empty
    agent = Agent("t", ScriptedLLM(["hello"]), tracer=tracer, hooks=[DoctorHook(tracer, doctor_llm)])
    assert agent.run("hi").final_output == "hello"
    assert "Trace Detective" not in capsys.readouterr().out
    assert doctor_llm.calls == 0


def test_doctor_hook_diagnoses_on_exception(capsys):
    tracer = Tracer()
    doctor_llm = ScriptedLLM([_diagnosis_json()])
    # round 1 requests a tool (emits llm_call/tool_call events); round 2 exhausts the script -> AssertionError -> on_error path
    llm = ScriptedLLM([ScriptedLLM.tool_call("calculator", {"expression": "1+1"})])
    agent = Agent("t", llm, tools=[CalculatorTool()], tracer=tracer, hooks=[DoctorHook(tracer, doctor_llm)])
    with pytest.raises(AssertionError):
        agent.run("1+1?")
    out = capsys.readouterr().out
    assert "raised AssertionError" in out                               # the exception reaches the header
    assert doctor_llm.calls == 1                                        # an exception run triggers unconditionally


def test_doctor_hook_disables_without_llm(capsys, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)               # ensure the lazily-constructed default LLM must fail
    tracer = Tracer()
    hook = DoctorHook(tracer)
    _failing_agent(tracer, [hook]).run("1/0?")
    _failing_agent(tracer, [hook]).run("1/0?")                          # a second trigger must not repeat the notice
    out = capsys.readouterr().out
    assert out.count("DoctorHook disabled") == 1
    assert "diagnosing with" not in out                                 # never reached the paid step


def test_doctor_hook_warn_severity_and_run_fallback(capsys):
    tracer = Tracer()
    tracer.emit({"type": "rag_retrieve", "query": "q", "hits": 0, "run_id": "r"})   # only a warn-level finding
    strict = DoctorHook(tracer, ScriptedLLM([]))                        # default tier: warn does not trigger
    asyncio.run(strict.on_run_end("out"))                               # called outside a run context, also covering the "fall back to newest run" path
    from agentmaker.prompts.packs import chinese_registry
    aggressive = DoctorHook(tracer, ScriptedLLM([_diagnosis_json(healthy=True, first_bad_step=None)]),
                            severity="warn", prompts=chinese_registry())   # also covers prompts pass-through
    asyncio.run(aggressive.on_run_end("out"))
    out = capsys.readouterr().out
    assert out.count("Trace Detective: run") == 1                       # only the warn-tier run spoke up
    assert "run looks healthy" in out
    with pytest.raises(ValueError, match="severity"):
        DoctorHook(tracer, severity="bogus")


# ---------- Web API (skipped when fastapi is absent) ----------

@pytest.fixture
def client_factory():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from agentmaker.devtools import create_app

    def make(llm=None):
        return TestClient(create_app(llm))
    return make


def _jsonl() -> str:
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in _events())


def test_webapp_index_and_parse(client_factory):
    client = client_factory()
    page = client.get("/")
    assert page.status_code == 200 and "Trace Detective" in page.text
    resp = client.post("/api/parse", json={"trace": _jsonl()})
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert [r["run_id"] for r in runs] == ["run-a", "run-b", None]
    assert runs[0]["steps"][1]["findings"][0]["code"] == "tool_error"


def test_webapp_client_errors(client_factory, monkeypatch):
    client = client_factory()
    assert client.post("/api/parse", json={"trace": "not json"}).status_code == 400
    assert client.post("/api/diagnose", json={"trace": _jsonl()}).status_code == 503   # no LLM = parse-only mode
    import agentmaker.devtools.webapp as webapp_module
    monkeypatch.setattr(webapp_module, "MAX_TRACE_CHARS", 10)
    assert client.post("/api/parse", json={"trace": _jsonl()}).status_code == 413


def test_webapp_diagnose(client_factory):
    client = client_factory(llm=ScriptedLLM([_diagnosis_json()]))
    resp = client.post("/api/diagnose", json={"trace": _jsonl(), "run_id": "run-a", "language": "zh"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run-a" and body["steps"] == 4
    assert body["diagnosis"]["root_cause"] == "bad expression"
    assert client.post("/api/diagnose", json={"trace": _jsonl(), "run_id": "ghost"}).status_code == 400


def test_webapp_providers_and_model_choice(client_factory, monkeypatch):
    import agentmaker.devtools.webapp as webapp_module
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-real")
    client = client_factory()                                     # no server-side default LLM
    data = client.get("/api/providers").json()
    assert data["default"] is None
    assert "deepseek" in [p["provider"] for p in data["available"]]
    assert "sk-test-not-real" not in json.dumps(data)             # report provider names only, never echo the key value
    # no provider chosen -> 503 (no default); unknown provider -> 400 clear error
    assert client.post("/api/diagnose", json={"trace": _jsonl()}).status_code == 503
    assert client.post("/api/diagnose", json={"trace": _jsonl(), "provider": "nope"}).status_code == 400
    # valid provider: on-demand construction goes through webapp's LLMClient reference, swapped here for a scripted stand-in
    monkeypatch.setattr(webapp_module, "LLMClient", lambda *a, **k: ScriptedLLM([_diagnosis_json()]))
    resp = client.post("/api/diagnose", json={"trace": _jsonl(), "provider": "deepseek", "run_id": "run-a"})
    assert resp.status_code == 200 and resp.json()["diagnosis"]["first_bad_step"] == 1


def test_doctor_hook_provider_choice(capsys, monkeypatch):
    import agentmaker.devtools.doctor as doctor_module
    built = {}

    def fake_client(provider=None, model=None, **_):
        built["args"] = (provider, model)
        return ScriptedLLM([_diagnosis_json()])

    monkeypatch.setattr(doctor_module, "LLMClient", fake_client)
    tracer = Tracer()
    _failing_agent(tracer, [DoctorHook(tracer, provider="zhipu", model="glm-x")]).run("1/0?")
    assert built["args"] == ("zhipu", "glm-x")                    # lazy construction used the developer-chosen provider/model
    assert "guard divide by zero" in capsys.readouterr().out
