"""agentmaker: a general-purpose agent framework.

Aggregates the public API of every subsystem in one place (`from agentmaker import ...`); each
subpackage can also be imported directly (e.g. `from agentmaker.retrieval import VectorStore`).
The `__all__` in this file is agentmaker's complete public contract: the stable surface lives here,
while internals (modules outside each subpackage's `__all__`) may change freely.

Lazy exports (PEP 562 module-level `__getattr__`): `import agentmaker` does not eagerly pull in every
subsystem (including heavy dependencies like jieba / sqlite-vec / pydantic). The first access to a symbol
imports its owning submodule and caches it, so both `from agentmaker import Agent` and `agentmaker.Agent`
work. `__all__` (the runtime contract), `_LAZY` (name to source-module mapping), and the `TYPE_CHECKING`
block (for static checkers / IDEs) correspond one-to-one; a module-level assert catches any missing entry
on the spot.
"""

import importlib
import logging as _logging
from typing import TYPE_CHECKING

_logging.getLogger("agentmaker").addHandler(_logging.NullHandler())   # A library configures no handler/level; the host application takes over (standard Python library practice).

# Symbol name to source submodule (relative to this package). __all__ is the key set; the assert below locks the two to be identical.
_LAZY = {
    # Core foundation: LLM client / response / message / unified exceptions
    "LLMClient": ".core", "LLMResponse": ".core", "ProviderProfile": ".core", "ModelInfo": ".core",
    "Message": ".core", "MessageRole": ".core",
    # Multimodal content parts (text + image in one message; adapters translate per protocol)
    "text_part": ".core", "image_part_from_bytes": ".core", "image_part_from_file": ".core",
    "image_part_from_url": ".core", "content_text": ".core", "messages_have_images": ".core",
    "AgentmakerError": ".core", "LLMError": ".core", "LLMConfigError": ".core", "LLMRequestError": ".core",
    "LLMResponseError": ".core", "ContextWindowExceeded": ".core", "RetrievalError": ".core",
    "SessionError": ".core", "GuardrailTripwireError": ".core", "RunLimitExceeded": ".core",
    "RunCancelled": ".core", "ToolError": ".core",
    # Agent base + unified loop + orchestration recipes + declarative config
    "Agent": ".agents", "BaseAgent": ".agents", "PlanAgent": ".agents", "ReflectionAgent": ".agents",
    "AgentSpec": ".agents", "build_agent": ".agents",
    "RunResult": ".agents", "RunStatus": ".agents", "RunUsage": ".agents",
    "AgentTool": ".agents.multi_agent",
    # Tool system (tool: @tool decorator; Tool: tool base class)
    "Tool": ".tools", "tool": ".tools", "ToolParameter": ".tools", "ToolResponse": ".tools",
    "ToolRegistry": ".tools", "ToolPermissions": ".tools",
    "CalculatorTool": ".tools", "SearchTool": ".tools", "CLITool": ".tools", "NotesTool": ".tools",
    "MCPClient": ".tools", "MCPTool": ".tools",
    "ToolRetrievalConfig": ".tools.tool_retriever", "ToolRetriever": ".tools.tool_retriever",
    "ToolSearchTool": ".tools.tool_retriever",
    # Runtime layer: harness coordination + cross-cutting capabilities + run-level context
    "Harness": ".runtime", "cli_confirm": ".runtime",
    "Guardrail": ".runtime", "GuardrailResult": ".runtime", "CallableGuardrail": ".runtime",
    "Interrupt": ".runtime", "PendingAction": ".runtime", "ApprovalRequired": ".runtime",
    "Hook": ".runtime",
    "SessionStore": ".runtime", "SqliteSessionStore": ".runtime", "ConversationSearch": ".runtime",
    "ConversationSearchTool": ".runtime", "ScopeSummary": ".runtime",
    "ExecutionState": ".runtime", "CheckpointStore": ".runtime", "SqliteCheckpointStore": ".runtime",
    "RunPolicy": ".runtime",
    "JsonlExporter": ".runtime", "MemoryExporter": ".runtime", "OTelExporter": ".runtime",
    "SqliteExporter": ".runtime", "Tracer": ".runtime", "TraceExporter": ".runtime",
    "current_run_id": ".runtime", "current_scope": ".runtime", "current_step": ".runtime",
    "current_trace_carrier": ".runtime", "governed_chat": ".runtime",
    # Isolation labels + retrieval foundation
    "Scope": ".retrieval", "RetrievalResult": ".retrieval", "RetrievalConfig": ".retrieval",
    "Embedder": ".retrieval", "OpenAIEmbedder": ".retrieval", "VectorStore": ".retrieval",
    "SqliteVecStore": ".retrieval", "KeywordIndex": ".retrieval", "Fts5KeywordIndex": ".retrieval",
    "Reranker": ".retrieval", "CohereReranker": ".retrieval", "HybridRetriever": ".retrieval",
    "reciprocal_rank_fusion": ".retrieval", "IndexSync": ".retrieval", "SyncIndexSync": ".retrieval",
    "MetadataFilter": ".retrieval", "FusionStrategy": ".retrieval", "RRFFusion": ".retrieval",
    "SyncBookkeeping": ".retrieval", "InMemoryBookkeeping": ".retrieval", "SqliteBookkeeping": ".retrieval",
    # Memory subsystem
    "KVMemory": ".memory", "KVStore": ".memory", "Memory": ".memory", "MemoryConfig": ".memory",
    "MemoryItem": ".memory", "MemoryStore": ".memory", "MemoryTool": ".memory", "SmartWriter": ".memory",
    # RAG subsystem
    "Document": ".rag", "Chunk": ".rag", "ChunkingConfig": ".rag", "RagConfig": ".rag",
    "IngestReport": ".rag", "AskResult": ".rag", "SourceRef": ".rag",
    "DocumentLoader": ".rag", "load_file": ".rag", "register_loader": ".rag", "Splitter": ".rag",
    "split_document": ".rag", "SourceStore": ".rag", "IngestionPipeline": ".rag", "RagRetriever": ".rag",
    "RAGTool": ".rag", "Contextualizer": ".rag", "HeadingContextualizer": ".rag", "LLMContextualizer": ".rag",
    "QueryTransformer": ".rag", "MultiQueryExpander": ".rag", "HyDETransformer": ".rag",
    "ChunkExpander": ".rag", "NeighborWindowExpander": ".rag",
    # Context engineering
    "ContextBuilder": ".context", "ContextConfig": ".context", "ReducerConfig": ".context",
    "CompactionConfig": ".context", "ContextSource": ".context", "CallableSource": ".context",
    "HistoryCompactor": ".context", "WindowBudget": ".context", "WindowBudgetConfig": ".context",
    "mmr_select": ".context", "count_tokens": ".context",
    # Config aggregation
    "AgentmakerConfig": ".config",
    # Skills (progressive disclosure)
    "Skill": ".skills", "SkillLoader": ".skills",
    # Prompt registry
    "DEFAULT_PROMPTS": ".prompts", "PromptRegistry": ".prompts", "PromptTemplate": ".prompts",
    "PromptError": ".prompts",
}


def __getattr__(name: str):
    """PEP 562 lazy top-level export: on first access to agentmaker.X, import the submodule per the _LAZY table, fetch X, cache it in module globals, and return it."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'agentmaker' has no attribute {name!r}")
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value   # Cache: subsequent access hits the normal attribute and skips __getattr__ (zero extra overhead).
    return value


def __dir__() -> list[str]:
    """List every lazy symbol for dir(agentmaker) / IDE completion (PEP 562 companion)."""
    return sorted(__all__)


if TYPE_CHECKING:   # Static checkers / IDEs: after lazification the top-level symbols are invisible, so import them here by their real paths (zero runtime cost).
    from .agents import (Agent, AgentSpec, BaseAgent, PlanAgent, ReflectionAgent, RunResult, RunStatus,
                        RunUsage, build_agent)
    from .agents.multi_agent import AgentTool
    from .config import AgentmakerConfig
    from .context import (CallableSource, CompactionConfig, ContextBuilder, ContextConfig, ContextSource,
                          HistoryCompactor, ReducerConfig, WindowBudget, WindowBudgetConfig, count_tokens, mmr_select)
    from .core import (ContextWindowExceeded, GuardrailTripwireError, AgentmakerError, LLMClient, LLMConfigError,
                       LLMError, LLMRequestError, LLMResponse, LLMResponseError, Message, MessageRole, ModelInfo,
                       ProviderProfile, RetrievalError, RunCancelled, RunLimitExceeded, SessionError, ToolError,
                       content_text, image_part_from_bytes, image_part_from_file, image_part_from_url,
                       messages_have_images, text_part)
    from .memory import KVMemory, KVStore, Memory, MemoryConfig, MemoryItem, MemoryStore, MemoryTool, SmartWriter
    from .prompts import DEFAULT_PROMPTS, PromptError, PromptRegistry, PromptTemplate
    from .rag import (AskResult, Chunk, ChunkExpander, ChunkingConfig, Contextualizer, Document, DocumentLoader,
                     HeadingContextualizer, HyDETransformer, IngestionPipeline, IngestReport, LLMContextualizer,
                     MultiQueryExpander, NeighborWindowExpander, QueryTransformer, RAGTool, RagConfig, RagRetriever,
                     SourceRef, SourceStore, Splitter, load_file, register_loader, split_document)
    from .retrieval import (CohereReranker, Embedder, Fts5KeywordIndex, FusionStrategy, HybridRetriever, IndexSync,
                           InMemoryBookkeeping, KeywordIndex, MetadataFilter, OpenAIEmbedder, RRFFusion, Reranker,
                           RetrievalConfig, RetrievalResult, Scope, SqliteBookkeeping, SqliteVecStore, SyncBookkeeping,
                           SyncIndexSync, VectorStore, reciprocal_rank_fusion)
    from .runtime import (ApprovalRequired, CallableGuardrail, CheckpointStore, ConversationSearch,
                         ConversationSearchTool, ExecutionState, Guardrail, GuardrailResult, Harness, Hook, Interrupt,
                         JsonlExporter, MemoryExporter, OTelExporter, PendingAction, RunPolicy, SessionStore,
                         ScopeSummary, SqliteCheckpointStore, SqliteExporter, SqliteSessionStore, TraceExporter, Tracer, cli_confirm,
                         current_run_id, current_scope, current_step, current_trace_carrier, governed_chat)
    from .skills import Skill, SkillLoader
    from .tools import (CLITool, CalculatorTool, MCPClient, MCPTool, NotesTool, SearchTool, Tool, ToolParameter,
                       ToolPermissions, ToolRegistry, ToolResponse, tool)
    from .tools.tool_retriever import ToolRetrievalConfig, ToolRetriever, ToolSearchTool


__all__ = [
    # core
    "LLMClient", "LLMResponse", "ProviderProfile", "ModelInfo", "Message", "MessageRole", "Agent",
    "text_part", "image_part_from_bytes", "image_part_from_file", "image_part_from_url",
    "content_text", "messages_have_images",
    "AgentmakerError", "LLMError", "LLMConfigError", "LLMRequestError", "LLMResponseError",
    "ContextWindowExceeded", "RetrievalError", "SessionError", "GuardrailTripwireError",
    "RunLimitExceeded", "RunCancelled", "ToolError",
    # agents
    "BaseAgent", "PlanAgent", "ReflectionAgent", "AgentSpec", "build_agent",
    "RunResult", "RunStatus", "RunUsage",
    # tools (tool: @tool decorator; Tool: tool base class)
    "Tool", "tool", "ToolParameter", "ToolResponse", "ToolRegistry", "ToolPermissions",
    "CalculatorTool", "SearchTool", "CLITool", "NotesTool", "MCPClient", "MCPTool",
    "ToolRetriever", "ToolSearchTool",
    # harness / observability / run-level context
    "Harness", "cli_confirm", "Tracer", "current_run_id", "current_scope", "current_step",
    "current_trace_carrier", "governed_chat",
    "TraceExporter", "MemoryExporter", "JsonlExporter", "SqliteExporter", "OTelExporter",
    # retrieval
    "Scope", "RetrievalResult", "RetrievalConfig", "Embedder", "OpenAIEmbedder", "VectorStore", "SqliteVecStore",
    "KeywordIndex", "Fts5KeywordIndex", "Reranker", "CohereReranker", "HybridRetriever", "reciprocal_rank_fusion",
    "IndexSync", "SyncIndexSync", "MetadataFilter", "FusionStrategy", "RRFFusion",
    "SyncBookkeeping", "InMemoryBookkeeping", "SqliteBookkeeping", "ChunkExpander", "NeighborWindowExpander",
    # memory
    "MemoryItem", "MemoryConfig", "MemoryStore", "Memory",
    "SmartWriter", "MemoryTool", "KVStore", "KVMemory",
    # rag
    "Document", "Chunk", "ChunkingConfig", "RagConfig", "IngestReport", "AskResult", "SourceRef",
    "DocumentLoader", "load_file", "register_loader", "Splitter", "split_document",
    "SourceStore", "IngestionPipeline", "RagRetriever", "RAGTool",
    "Contextualizer", "HeadingContextualizer", "LLMContextualizer",
    "QueryTransformer", "MultiQueryExpander", "HyDETransformer",
    # context
    "ContextBuilder", "ContextConfig", "ReducerConfig", "CompactionConfig", "ContextSource", "CallableSource", "HistoryCompactor",
    "WindowBudget", "WindowBudgetConfig", "mmr_select", "count_tokens",
    # config aggregation
    "AgentmakerConfig", "ToolRetrievalConfig",
    # sessions
    "SessionStore", "SqliteSessionStore", "ConversationSearch", "ConversationSearchTool", "ScopeSummary",
    # guardrails
    "Guardrail", "GuardrailResult", "CallableGuardrail",
    # hitl
    "Interrupt", "PendingAction", "ApprovalRequired",
    # hooks
    "Hook",
    # execution
    "ExecutionState", "CheckpointStore", "SqliteCheckpointStore", "RunPolicy",
    # skills
    "Skill", "SkillLoader",
    # multi_agent
    "AgentTool",
    # prompts (prompt registry)
    "DEFAULT_PROMPTS", "PromptRegistry", "PromptTemplate", "PromptError",
]
__version__ = "0.2.0"

assert set(_LAZY) == set(__all__), (   # Dual-list drift guard: _LAZY and __all__ must be strictly identical; a single missing entry blows up on `import agentmaker`.
    f"_LAZY and __all__ disagree: only in _LAZY={sorted(set(_LAZY) - set(__all__))}, only in __all__={sorted(set(__all__) - set(_LAZY))}")
