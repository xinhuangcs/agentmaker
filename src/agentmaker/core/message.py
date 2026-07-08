"""agentmaker.core.message: the unified type for conversation messages.

Conversation history is the most critical context for an Agent's interaction with the model. Here a simple Message specifies a single message's role, content, and accompanying information; adapters only recognize the {"role", "content"} shape produced by to_dict().
(For more sophisticated context management, see the agentmaker.context subsystem.)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, get_args

from .clock import now_utc as _now_utc   # unified time source (aware UTC): the single source of truth is core/clock.py
from .multimodal import MessageContent, content_text

# Message role: restricted to four values; a wrong one is caught by the type checker
MessageRole = Literal["user", "assistant", "system", "tool"]
_VALID_ROLES = frozenset(get_args(MessageRole))  # for runtime validation (Literal only constrains statically)


@dataclass
class Message:
    """A single conversation message.

    Attributes:
        content: The message body: a plain str, or a list of multimodal content parts
            (text / image, see core.multimodal for the neutral part shapes). Consumers
            that need text must go through content_text() instead of assuming str.
        role: The role; see MessageRole for allowed values.
        timestamp: The creation time, defaulting to the current UTC time (timezone-aware).
        metadata: Accompanying information (e.g. source, token count), an empty dict by default, for logging / future feature extension.
    """
    content: MessageContent
    role: MessageRole
    timestamp: datetime = field(default_factory=_now_utc)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that role is legal after construction: Literal only takes effect at type-check time, and an illegal value may still be passed at runtime, so add a lightweight runtime check here."""
        if self.role not in _VALID_ROLES:
            raise ValueError(f"invalid role={self.role!r}, must be one of {sorted(_VALID_ROLES)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to an OpenAI-style {"role", "content"} dict for direct consumption by adapters
        (multimodal part lists pass through as-is; each adapter translates them to its wire format).

        Returns:
            Dict[str, Any]: Contains only role and content, without timestamp / metadata.
        """
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        """Let print(message) show "[role] body" directly, for easier debugging (image parts
        render as "[image: ...]" placeholders, see content_text).

        Returns:
            str: Of the form "[user] hello".
        """
        return f"[{self.role}] {content_text(self.content)}"
