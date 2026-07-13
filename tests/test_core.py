"""Core contract tests (hermetic: no key or network).

Covers the AgentmakerError exception hierarchy, count_tokens handling of CJK/Kana/Hangul,
calculator rejecting keyword arguments, and reducer keep_recent lower-bound validation.
"""

import pytest

from agentmaker import (AgentmakerError, LLMConfigError, LLMError, LLMRequestError, LLMResponseError,
                    RetrievalError, RunLimitExceeded, SessionError)
from agentmaker.core.exceptions import GuardrailTripwireError, RunCancelled, ContextWindowExceeded
from agentmaker.core.text import count_tokens
from agentmaker.tools import CalculatorTool


# ---------- AgentmakerError shared base ----------

@pytest.mark.parametrize("exc", [
    LLMError, LLMConfigError, LLMRequestError, LLMResponseError, RetrievalError, SessionError,
    GuardrailTripwireError, RunLimitExceeded, RunCancelled, ContextWindowExceeded,
])
def test_all_exceptions_subclass_agentmaker_error(exc):
    assert issubclass(exc, AgentmakerError)
    assert issubclass(exc, RuntimeError)


def test_agentmaker_error_catches_any_framework_exception():
    try:
        raise RetrievalError("boom")
    except AgentmakerError as e:
        assert str(e) == "boom"


def test_llm_error_subtypes_and_retryable():
    """LLMError splits into Config/Request/Response subtypes (except LLMError still catches all); LLMRequestError carries retryable for precise retries."""
    from agentmaker.core.adapters import _request_error
    for sub in (LLMConfigError, LLMRequestError, LLMResponseError):
        assert issubclass(sub, LLMError)
    assert _request_error("openai", "m", type("E", (Exception,), {"status_code": 429})()).retryable is True   # rate limited -> retry
    assert _request_error("openai", "m", type("E", (Exception,), {"status_code": 401})()).retryable is False  # auth failure -> no retry
    assert _request_error("openai", "m", TimeoutError()).retryable is True       # timeout -> retry
    assert _request_error("openai", "m", type("E", (Exception,), {"status_code": 503})()).status_code == 503


def test_llm_response_keeps_legacy_positional_order():
    """Provider continuation state is appended without shifting existing positional fields."""
    from agentmaker.core.llm_response import LLMResponse

    raw = object()
    response = LLMResponse("text", "stop", "model", {"total_tokens": 1}, "reason", [], 12, raw)
    assert response.latency_ms == 12
    assert response.raw is raw
    assert response.assistant_message is None


# ---------- open_sqlite: file DBs enable WAL + busy_timeout, memory DBs skip it (SQLite concurrency) ----------

def test_open_sqlite_sets_wal_for_file(tmp_path):
    """A file-backed DB gets WAL + busy_timeout=5000 after open_sqlite."""
    from agentmaker.core.sqlite_util import open_sqlite
    conn = open_sqlite(str(tmp_path / "t.db"))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_open_sqlite_memory_skips_wal():
    """An in-memory DB (:memory:) skips WAL (meaningless there) without error."""
    from agentmaker.core.sqlite_util import open_sqlite
    conn = open_sqlite()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"   # in-memory reports "memory"
    finally:
        conn.close()


# ---------- count_tokens: CJK/Kana/Hangul per character, everything else ~4 chars/token ----------

def test_count_tokens_chinese_and_english_unchanged():
    assert count_tokens("你好世界") == 4
    assert count_tokens("hello world") == 3          # 11 chars -> (11+3)//4 = 3
    assert count_tokens("") == 0


def test_count_tokens_no_space_blob_not_underestimated():
    """A long no-space blob is estimated at chars/4, not collapsed to 1 token (the base64 / compressed-data / long-URL undercount hole)."""
    assert count_tokens("A" * 10000) == 2500         # (10000+3)//4 = 2500
    assert count_tokens("A" * 200000) == 50000


@pytest.mark.parametrize("text,expected", [
    ("こんにちは", 5),     # hiragana, 5 chars
    ("カタカナ", 4),       # katakana, 4 chars
    ("안녕하세요", 5),     # hangul, 5 chars
])
def test_count_tokens_counts_cjk_kana_hangul(text, expected):
    assert count_tokens(text) == expected


# ---------- calculator: positional args work, keyword args explicitly rejected ----------

def test_calculator_round_positional_works():
    assert CalculatorTool().run({"expression": "round(3.14159, 2)"}).text == "3.14"


def test_calculator_rejects_keyword_arguments():
    r = CalculatorTool().run({"expression": "round(3.14159, ndigits=2)"})
    assert r.status == "error"                      # kwargs would be silently ignored -> reject explicitly rather than return a wrong result


# ---------- reducer: keep_recent lower-bound validation ----------

def test_reducer_rejects_negative_keep_recent():
    import asyncio
    from agentmaker.context.reducer import reduce_agent, reduce_plan
    with pytest.raises(ValueError):                     # reduce fns are natively async, driven via asyncio.run (validation raises in-body, never reaching summarization)
        asyncio.run(reduce_plan(["步骤1：x\n结果：y"], summarize=None, budget=1, keep_recent=-1))
    with pytest.raises(ValueError):
        asyncio.run(reduce_agent([{"role": "user", "content": "q"}], summarize=None, budget=1, keep_recent_steps=-1))


# ---------- streaming tool calls: accumulated, never silently dropped ----------

def test_openai_stream_feed_accumulates_tool_calls():
    """OpenAI stream chunks carrying tool_calls deltas accumulate by index (streaming tool loop) instead of raising."""
    from types import SimpleNamespace

    from agentmaker.core.adapters import _StreamState
    frag = SimpleNamespace(index=0, id="c1",
                           function=SimpleNamespace(name="f", arguments="{}"))
    delta = SimpleNamespace(tool_calls=[frag], content=None)
    chunk = SimpleNamespace(model="m", usage=None,
                            choices=[SimpleNamespace(delta=delta, finish_reason=None)])
    st = _StreamState("m")
    assert st.feed(chunk) == ""            # tool deltas emit no text
    assert st.final_tool_calls() == [{"id": "c1", "type": "function",
                                      "function": {"name": "f", "arguments": "{}"}}]


def test_gemini_stream_feed_collects_function_call():
    """Gemini stream chunks carrying a function_call part collect into unified tool_calls instead of raising."""
    from types import SimpleNamespace

    from agentmaker.core.adapters import _GemStreamState
    fc = SimpleNamespace(id=None, name="f", args={"a": 1})
    part = SimpleNamespace(function_call=fc, thought_signature=None)
    content = SimpleNamespace(parts=[part])
    cand = SimpleNamespace(finish_reason=None, content=content)
    chunk = SimpleNamespace(usage_metadata=None, candidates=[cand], text=None)
    st = _GemStreamState("m")
    assert st.feed(chunk) == ""
    assert st.tool_calls[0]["function"]["name"] == "f"
    assert st.tool_calls[0]["function"]["arguments"] == '{"a": 1}'


def test_stream_feed_plain_text_not_misfired():
    """Plain text chunks emit normally with correct stats (no false trigger when there are no tool_calls)."""
    from agentmaker.core.adapters import _StreamState
    delta = type("D", (), {"tool_calls": None, "content": "hello"})()
    chunk = type("C", (), {"model": "m", "usage": None,
                           "choices": [type("Ch", (), {"delta": delta, "finish_reason": "stop"})()]})()
    st = _StreamState("m")
    assert st.feed(chunk) == "hello" and st.finish_reason == "stop"
