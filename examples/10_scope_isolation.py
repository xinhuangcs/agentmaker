"""Scope isolation: one shared backend, many tenants that cannot see each other.

Every read and write carries a Scope (base / user / agent / session / app dimensions). Below,
two users share the exact same store and index, yet each retrieves only their own data. This is
how the framework keeps memories / documents / sessions isolated per user without separate
databases. Hermetic.

    uv run python examples/10_scope_isolation.py
"""
from agentmaker import Memory, MemoryStore, Scope
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder

# One shared store + index; the only thing separating the two users is their Scope.
store = MemoryStore()
index = build_sqlite_hybrid(FakeEmbedder())
alice = Memory(retriever=index, store=store, scope=Scope(base="memory", user="alice"))
bob = Memory(retriever=index, store=store, scope=Scope(base="memory", user="bob"))

alice.add("Alice loves tea")
bob.add("Bob loves coffee")

print("alice sees:", [h.content for h in alice.search("favorite drink", top_k=5)])
print("bob sees:  ", [h.content for h in bob.search("favorite drink", top_k=5)])
