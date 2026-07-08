"""Shared test fixtures: expose agentmaker.testing's official doubles as fixtures for reuse across tests.

Prefer these fixtures (or importing from agentmaker.testing) over hand-rolling a fake LLM / embedder / checkpoint in each test.
"""

import pytest

from agentmaker.testing import FakeEmbedder, MemoryCheckpointStore, RecordingHook, ScriptedLLM


@pytest.fixture
def fake_embedder():
    """Deterministic fake embedder (dim=8, offline)."""
    return FakeEmbedder(dim=8)


@pytest.fixture
def mem_checkpoint():
    """In-process checkpoint store (for HITL / resume tests)."""
    return MemoryCheckpointStore()


@pytest.fixture
def recording_hook():
    """Hook that records lifecycle events."""
    return RecordingHook()


@pytest.fixture
def scripted_llm():
    """Return the ScriptedLLM class itself; tests build with `scripted_llm(["reply", ...])`."""
    return ScriptedLLM
