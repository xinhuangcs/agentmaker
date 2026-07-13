"""agentmaker.core.adapters: LLM wire-protocol adapter subpackage (one file per protocol).

Split along the physical boundary "one adapter = one wire protocol": base (interface + streaming base
class + exception normalization) / openai_compat / anthropic / gemini. LLMClient selects an adapter by
looking up ProviderProfile.protocol in _ADAPTERS; third parties add a new protocol via register_adapter
(the front door).
"""

from .base import BaseAdapter, _BaseStreamState, _request_error
from .openai_compat import OpenAIAdapter, _StreamState
from .anthropic import AnthropicAdapter, _close_objects
from .gemini import GeminiAdapter, _GemStreamState, _gemini_usage

# protocol -> adapter class.
_ADAPTERS = {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "gemini": GeminiAdapter}


def register_adapter(protocol: str, adapter_cls: type) -> None:
    """Register a mapping from a wire protocol to an adapter class so third parties can add new provider protocols (the front door).

    Combined with ProviderProfile(protocol=...) this attaches a new protocol: first call
    register_adapter("myproto", MyAdapter), then
    LLMClient(provider, profile=ProviderProfile(protocol="myproto", ...)). adapter_cls must be a
    BaseAdapter subclass.

    Args:
        protocol: Protocol name (the value of ProviderProfile.protocol).
        adapter_cls: A BaseAdapter subclass.
    """
    if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, BaseAdapter)):
        raise TypeError(f"adapter_cls must be a BaseAdapter subclass, got {adapter_cls!r}")
    _ADAPTERS[protocol] = adapter_cls


__all__ = [
    "BaseAdapter", "OpenAIAdapter", "AnthropicAdapter", "GeminiAdapter",
    "register_adapter",
    "_ADAPTERS", "_BaseStreamState", "_StreamState", "_GemStreamState",
    "_request_error", "_close_objects", "_gemini_usage",
]
