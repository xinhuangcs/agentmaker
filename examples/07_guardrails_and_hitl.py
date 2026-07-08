"""Guardrails and human-in-the-loop (HITL).

Guardrails screen input/output and trip a run when a rule is violated. A tool marked
requires_confirmation suspends the run for approval; you grant it with resume(). Hermetic
via ScriptedLLM.

    uv run python examples/07_guardrails_and_hitl.py
"""
from agentmaker import Agent, CallableGuardrail, GuardrailTripwireError, tool
from agentmaker.testing import MemoryCheckpointStore, ScriptedLLM

# 1) Guardrail: reject any input that mentions a password.
no_secrets = CallableGuardrail(lambda text: "password" not in text.lower(),
                               message="input mentions a password")
guarded = Agent("assistant", ScriptedLLM(["ok"]), input_guardrails=[no_secrets])
try:
    guarded.run("my password is 1234")
except GuardrailTripwireError as e:
    print("Guardrail blocked the input:", e)


# 2) HITL: a high-risk tool suspends the run until a human approves.
@tool(requires_confirmation=True)
def delete_file(path: str) -> str:
    """Delete a file (high-risk, requires confirmation).

    Args:
        path: File path to delete.
    """
    return f"deleted {path}"


ops = Agent("ops", ScriptedLLM([
    ScriptedLLM.tool_call("delete_file", {"path": "/tmp/old.log"}),
    "Done, the file was deleted.",
]), tools=[delete_file], checkpoint_store=MemoryCheckpointStore())

result = ops.run("please delete /tmp/old.log")
if result.interrupt:                                    # run paused, awaiting approval
    pending = result.interrupt.pendings[0]
    print(f"Approval needed for: {pending.tool_name}({pending.arguments})")
    approved = ops.resume(True, scope=result.interrupt.scope)   # resume(False) would reject
    print("After approval:", approved.final_output)
