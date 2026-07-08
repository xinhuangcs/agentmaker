"""agentmaker.core.llm_response: unified response types for LLM calls.

Adapters translate each provider's raw response into these types, and higher layers deal only with them:
    - LLMResponse: the unified result of a single non-streaming call.
    - StreamStats: the statistics after a streaming call finishes (streaming yields only text segments, so the meta information lives here separately).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    """The unified result of a single (non-streaming) call. Every protocol adapter translates its raw response into this."""
    content: str = ""
    finish_reason: Optional[str] = None
    model: str = ""
    usage: Optional[Dict[str, Any]] = None  # values are not all int: OpenAI model_dump() includes nested structures such as *_tokens_details
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list] = None  # function-calling tool calls (OpenAI standard format, can be fed straight back into messages); None if absent
    latency_ms: int = 0
    raw: Any = None

    def __str__(self) -> str:
        """Let print(response) / f"{response}" show the reply text directly, for easier debugging.

        Returns:
            str: The content text.
        """
        return self.content


@dataclass
class StreamStats:
    """Statistics after a streaming call finishes.

    Streaming yields only text segments, so the meta information (usage / latency / finish reason) lives here,
    for cost accounting and trace use; obtained via `LLMClient.last_stream_stats`.

    Attributes:
        model: The actual model name used.
        finish_reason: The finish reason.
        usage: Token usage; may be None by default for streaming (OpenAI-family requires stream_options={"include_usage": True}).
        latency_ms: Total latency of this streaming call (milliseconds).
    """
    model: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None  # values are not all int: OpenAI model_dump() includes nested detail fields
    latency_ms: int = 0
