"""agentmaker.devtools.webapp: the Trace Detective local web app (FastAPI factory).

Serves the self-contained UI (static/index.html) plus three JSON endpoints: /api/parse (deterministic,
LLM-free), /api/providers (which vendors this environment holds keys for; names only, never key values)
and /api/diagnose (LLM verdict, optionally on a per-request provider/model choice). fastapi is an optional
dependency: install `agentmaker[devtools]`; importing this module stays cheap and dependency-free until
create_app is called. Start it with `python -m agentmaker.devtools` (see __main__.py) or mount
create_app() yourself.
"""

import os
from dataclasses import asdict
from importlib.resources import files
from typing import Optional

from ..core.exceptions import AgentmakerError
from ..core.llm_clients import _PROFILES, LLMClient   # _PROFILES: vendor table (framework-internal; devtools ships with the framework)
from .diagnose import diagnose
from .trace_parser import TraceParseError, parse_trace, pick_run

# Upper bound on the pasted trace, in characters (a devtool guard against accidental giant payloads;
# read at request time via the module global, so tests and apps can override it).
MAX_TRACE_CHARS = 5_000_000


def _index_html() -> str:
    """Load the bundled single-file UI (packaged data, works from a wheel install too).

    Deliberately NOT cached: this is a local devtool, re-reading ~15KB per page load is free, and it makes
    frontend edits visible on refresh without restarting the server.
    """
    return files(__package__).joinpath("static/index.html").read_text(encoding="utf-8")


def create_app(llm=None):
    """Build the Trace Detective FastAPI app.

    Args:
        llm: Optional LLMClient-compatible client powering /api/diagnose. None starts parse-only mode:
            the timeline works and /api/diagnose answers 503 with a hint.

    Returns:
        FastAPI: The configured app (serve it with uvicorn).

    Raises:
        ImportError: fastapi is not installed (install `agentmaker[devtools]`).
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as e:
        raise ImportError("Trace Detective's web app requires fastapi: pip install 'agentmaker[devtools]'") from e
    from pydantic import BaseModel

    class TraceRequest(BaseModel):
        """Request body shared by both endpoints (run_id / language / provider / model only matter for
        /api/diagnose). language None = follow the active prompt catalog / language pack (see diagnose);
        provider/model None = the server's default client."""
        trace: str
        run_id: Optional[str] = None
        language: Optional[str] = None
        provider: Optional[str] = None
        model: Optional[str] = None

    def _parse_or_http_error(text: str):
        """Shared request validation: size cap then parse; parse problems become client-side 4xx."""
        if len(text) > MAX_TRACE_CHARS:
            raise HTTPException(413, f"trace too large ({len(text)} chars > {MAX_TRACE_CHARS})")
        try:
            return parse_trace(text)
        except TraceParseError as e:
            raise HTTPException(400, str(e)) from None

    llm_cache: dict = {}   # (provider, model) -> LLMClient built for a request's choice, reused across requests

    def _resolve_llm(request: TraceRequest):
        """Pick the diagnosis client: the request's provider/model choice wins, else the server default.

        A requested client is built from that vendor's environment keys (same rules as LLMClient) and
        cached; an unknown vendor or a missing key becomes a clear 400 instead of a crash.
        """
        if request.provider is None and request.model is None:
            if llm is None:
                raise HTTPException(503, "this server was started without a default LLM (parse-only mode); "
                                         "pick a provider in the request, or restart with an API key / --provider")
            return llm
        cache_key = (request.provider, request.model)
        client = llm_cache.get(cache_key)
        if client is None:
            try:
                client = (LLMClient(request.provider, model=request.model) if request.provider is not None
                          else LLMClient(model=request.model))
            except Exception as e:
                raise HTTPException(400, f"cannot build LLM for provider={request.provider!r} "
                                         f"model={request.model!r}: {e}") from None
            llm_cache[cache_key] = client
        return client

    app = FastAPI(title="Trace Detective", docs_url=None, redoc_url=None)  # local devtool: no OpenAPI UI surface

    # Endpoints are sync on purpose: FastAPI runs them in a worker thread, where the blocking
    # Agent.run facade (which starts its own event loop) is safe; async def would break it.
    @app.get("/", response_class=HTMLResponse)
    def index():
        """Serve the single-file UI."""
        return _index_html()

    @app.post("/api/parse")
    def api_parse(request: TraceRequest):
        """Deterministic half: parse the trace into runs (timeline + findings + stats), no LLM cost."""
        runs = _parse_or_http_error(request.trace)
        return {"runs": [asdict(run) for run in runs]}

    @app.get("/api/providers")
    def api_providers():
        """Which vendors this environment can power a diagnosis with (an API key is present), plus the
        server's default client. Returns names only, never key values."""
        available = [{"provider": name, "default_model": profile.default_model}
                     for name, profile in _PROFILES.items()
                     if profile.default_model and any(os.environ.get(env) for env in profile.key_envs)]
        default = ({"provider": getattr(llm, "provider", "?"), "model": getattr(llm, "model", "?")}
                   if llm is not None else None)
        return {"default": default, "available": available}

    @app.post("/api/diagnose")
    def api_diagnose(request: TraceRequest):
        """LLM half: pick one run (run_id, or the most recent) and return the three-part verdict."""
        client = _resolve_llm(request)
        runs = _parse_or_http_error(request.trace)
        try:
            run = pick_run(runs, request.run_id)
        except TraceParseError as e:
            raise HTTPException(400, str(e)) from None
        try:
            verdict = diagnose(run, client, language=request.language)
        except AgentmakerError as e:
            raise HTTPException(502, f"LLM diagnosis failed: {e}") from None
        return {"run_id": run.run_id, "steps": run.stats.steps, "diagnosis": verdict.model_dump()}

    return app
