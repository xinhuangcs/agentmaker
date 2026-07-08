"""python -m agentmaker.devtools: start the local Trace Detective server.

Builds an LLMClient from --provider/--model (keys come from environment variables, same as everywhere in
the framework); if that fails (e.g. missing key) the server still starts in parse-only mode so the timeline
remains usable. Binds 127.0.0.1 by default: this is a local debugging tool, not something to expose.
"""

import argparse

from ..core.exceptions import AgentmakerError
from ..core.llm_clients import LLMClient
from .webapp import create_app


def main() -> None:
    """Parse CLI args, assemble the app, serve it with uvicorn."""
    parser = argparse.ArgumentParser(prog="python -m agentmaker.devtools",
                                     description="Trace Detective: local web UI to debug agentmaker traces")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; keep it local)")
    parser.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    parser.add_argument("--provider", default="deepseek", help="LLM provider for diagnosis (default deepseek)")
    parser.add_argument("--model", default=None, help="model name (default: the provider's default model)")
    parser.add_argument("--no-llm", action="store_true", help="parse-only mode: skip the LLM, disable /api/diagnose")
    args = parser.parse_args()

    llm = None
    if not args.no_llm:
        try:
            llm = LLMClient(args.provider, model=args.model)
        except AgentmakerError as e:  # Missing key / unknown provider: degrade instead of refusing to start.
            print(f"🔧 LLM unavailable ({e}); starting in parse-only mode (timeline works, diagnosis disabled)")

    try:
        import uvicorn
    except ImportError as e:
        raise ImportError("serving Trace Detective requires uvicorn: pip install 'agentmaker[devtools]'") from e

    print(f"🔧 Trace Detective: http://{args.host}:{args.port}"
          + (f" (diagnosing with {llm.provider}/{llm.model})" if llm is not None else " (parse-only mode)"))
    uvicorn.run(create_app(llm), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
