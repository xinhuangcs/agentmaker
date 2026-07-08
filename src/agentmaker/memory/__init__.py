"""agentmaker.memory: memory subsystem (built on top of the agentmaker.retrieval base).

Provided:
    - MemoryItem: the data type for a single memory (type is a free-form label).
    - MemoryStore: the memory source-of-truth store (full memories persisted by id in SQLite).
    - Memory: semantic memory combining the source-of-truth store with the retrieval base;
      add / search / update / delete + forget / stats / summary / consolidate
      (the source-to-derived-index sync seam IndexSync / SyncIndexSync is shared with RAG and lives in agentmaker.retrieval).
    - SmartWriter: Mem0-style smart writing (extract facts -> ADD/UPDATE/DELETE/NOOP).
    - MemoryTool: wraps memory as a tool so an agent can actively record / recall (agentic memory).
    - KVStore / KVMemory: key-value memory, structured facts stored and read by exact key (complements semantic memory).
"""

from .kv import KVMemory, KVStore
from .memory import Memory
from .memory_tool import MemoryTool
from .smart_writer import DEFAULT_EXTRACT_PROMPT, DEFAULT_RECONCILE_PROMPT, SmartWriter
from .store import MemoryStore
from .types import MemoryConfig, MemoryItem

__all__ = ["MemoryItem", "MemoryConfig", "MemoryStore", "Memory",
           "SmartWriter", "MemoryTool", "KVStore", "KVMemory",
           "DEFAULT_EXTRACT_PROMPT", "DEFAULT_RECONCILE_PROMPT"]
