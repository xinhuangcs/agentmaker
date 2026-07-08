"""Sessions: multi-turn conversations with persisted history.

Attach a SqliteSessionStore and a Scope; the agent loads prior turns and appends new ones, so a
later turn can reference an earlier one. Hermetic (in-memory SQLite).

    uv run python examples/12_sessions.py
"""
from agentmaker import Agent, Scope, SqliteSessionStore
from agentmaker.testing import ScriptedLLM

store = SqliteSessionStore()                       # in-memory by default
scope = Scope(user="alice", session="chat-1")      # which conversation this is

# Turn 1: the user introduces themselves.
Agent("assistant", ScriptedLLM(["Nice to meet you, Alice!"]),
      session_store=store, scope=scope).run("Hi, I'm Alice.")

# Turn 2: a fresh Agent with the SAME store + scope replays the earlier turn as context.
turn2 = Agent("assistant", ScriptedLLM(["Your name is Alice."]),
              session_store=store, scope=scope).run("What's my name?")

print("turn 2 answer:", turn2.final_output)
print("messages stored:", len(store.load(scope=scope)))   # 2 turns x (user + assistant) = 4
