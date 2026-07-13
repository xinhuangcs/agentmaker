"""Long-term memory: store durable facts and retrieve them by meaning.

Memory pairs a source-of-truth store (MemoryStore) with a retrieval index. Hermetic here:
FakeEmbedder is deterministic and offline, and build_sqlite_hybrid uses a local in-memory
SQLite backend. In production, swap FakeEmbedder() for OpenAIEmbedder() (needs OPENAI_API_KEY).

    uv run python examples/04_memory.py
"""
from agentmaker import Memory, MemoryStore
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder

memory = Memory(retriever=build_sqlite_hybrid(FakeEmbedder()), store=MemoryStore())

memory.add("I am allergic to peanuts")
memory.add("I like oat milk in the evening")
memory.add("I work as a backend engineer")

# Note: FakeEmbedder is a deterministic hash-based stand-in, so ranking is stable but NOT
# semantic. With a real embedder (OpenAIEmbedder), the allergy fact would rank on top here.
print("Top matches for 'what food should I avoid':")
for hit in memory.search("what food should I avoid", top_k=2):
    print("  -", hit.content)
