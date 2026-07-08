"""agentmaker.core: framework foundation (LLM client, unified response types, message types, and unified exceptions), with no dependency on any agentmaker subsystem.

The abstract Agent base class lives in agentmaker.agents.base (it orchestrates hooks, guardrails, persistence, and other higher-level subsystems, so it belongs to the orchestration layer rather than the foundation layer).
"""

from .exceptions import (
    AgentmakerError, LLMError, LLMConfigError, LLMRequestError, LLMResponseError, ContextWindowExceeded,
    RetrievalError, SessionError, GuardrailTripwireError,
    RunLimitExceeded, RunCancelled, ToolError, ToolRegistrationError,
)
from .llm_response import LLMResponse
from .llm_clients import LLMClient, ModelInfo, ProviderProfile
from .message import Message, MessageRole
from .multimodal import (content_text, content_tokens, image_part_from_bytes, image_part_from_file,
                         image_part_from_url, messages_have_images, text_part)
from .text import TokenCounter, count_tokens

__all__ = ["LLMClient", "LLMResponse", "ProviderProfile", "ModelInfo", "Message", "MessageRole",
           "AgentmakerError", "LLMError", "LLMConfigError", "LLMRequestError", "LLMResponseError",
           "ContextWindowExceeded", "RetrievalError", "SessionError",
           "GuardrailTripwireError", "RunLimitExceeded", "RunCancelled", "ToolError", "ToolRegistrationError",
           "TokenCounter", "count_tokens",
           "text_part", "image_part_from_bytes", "image_part_from_file", "image_part_from_url",
           "content_text", "content_tokens", "messages_have_images"]
