"""agentmaker.memory.smart_writer: Mem0-style smart writing.

Rather than blindly appending: first use the LLM to extract atomic facts from the input, then for each
fact search for similar existing memories, let the LLM decide ADD / UPDATE / DELETE / NOOP, and finally
execute. Keeps memory clean, non-duplicated, and non-contradictory.
"""

import asyncio
import json
from typing import TYPE_CHECKING, List, Optional

from ..core.llm_clients import LLMClient
from ..prompts import DEFAULT_PROMPTS
from ..runtime.execution.run_context import governed_chat
from ..retrieval.scope import Scope
from .memory import Memory

if TYPE_CHECKING:
    from ..prompts import PromptRegistry
    from ..runtime.observability import Tracer


DEFAULT_EXTRACT_PROMPT = DEFAULT_PROMPTS.text("memory.extract")
DEFAULT_RECONCILE_PROMPT = DEFAULT_PROMPTS.text("memory.reconcile")


def _parse_json(text: str) -> Optional[object]:
    """Best-effort parse of JSON from an LLM reply (tolerating ```json fencing and surrounding stray text); returns None on failure."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):                 # strip the markdown code fence
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]   # cut from the first { or [
    ends = [i for i in (s.rfind("}"), s.rfind("]")) if i != -1]   # to the last } or ]
    if starts and ends:
        s = s[min(starts):max(ends) + 1]
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


class SmartWriter:
    """Mem0-style smart writing: extract facts -> compare against existing memories -> decide ADD/UPDATE/DELETE/NOOP -> execute."""

    def __init__(self, memory: Memory, llm: LLMClient, *, similar_k: Optional[int] = None, fail_open: bool = True,
                 extract_prompt: Optional[str] = None, reconcile_prompt: Optional[str] = None,
                 prompts: "Optional[PromptRegistry]" = None,
                 tracer: "Optional[Tracer]" = None):
        """Initialize the smart writer.

        Args:
            memory: The memory manager (provides search / add / update / delete).
            llm: LLM client, used for fact extraction and reconcile decisions (ideally a cheap model such as deepseek).
            similar_k: How many similar existing memories to fetch per fact for the comparison; defaults to
                the injected memory's cfg.similar_k when None.
            fail_open: Strategy when extraction parsing fails. True (default) degrades to storing the whole
                original text as a single fact, losing no information; False drops the segment and does not
                store it. Set False for chit-chat / long text / sensitive content you don't want stored wholesale.
            extract_prompt: System prompt for fact extraction; defaults to the framework default (registry
                memory.extract). Pass your own to change language or tune the extraction categories.
            reconcile_prompt: System prompt for the reconcile decision (ADD/UPDATE/DELETE/NOOP); defaults to
                the framework default (registry memory.reconcile).
            prompts: Optional prompt registry (PromptRegistry, see agentmaker.prompts); defaults to the
                framework default DEFAULT_PROMPTS. extract_prompt / reconcile_prompt are its nearby shortcut
                overrides (equivalent to overriding the memory.extract / memory.reconcile keys).
            tracer: Optional tracer; defaults to following the injected memory's tracer. The extraction /
                reconcile LLM calls flow through governance (into trace and RunPolicy limits, see
                governed_chat in runtime/execution/run_context.py).
        """
        self.memory = memory
        self.llm = llm
        self.tracer = tracer if tracer is not None else getattr(memory, "tracer", None)   # defaults to memory's tracer
        self.similar_k = memory.cfg.similar_k if similar_k is None else similar_k
        self.fail_open = fail_open
        overrides = {}
        if extract_prompt:
            overrides["memory.extract"] = extract_prompt
        if reconcile_prompt:
            overrides["memory.reconcile"] = reconcile_prompt
        base = prompts or DEFAULT_PROMPTS
        self.prompts = base.with_overrides(overrides) if overrides else base
        self.extract_prompt = self.prompts.text("memory.extract")        # convenience alias = the actual value after injection
        self.reconcile_prompt = self.prompts.text("memory.reconcile")

    async def write(self, text: str, *, scope: Optional[Scope] = None) -> List[dict]:
        """Smartly write a piece of input into memory (async); returns a processing record per fact (with op, so you can see what was done).

        Extraction / reconcile await chat; retrieval uses memory.asearch and CRUD runs in a thread pool.
        Sync callers go through agentmaker.core.aio.run_sync.

        Writing / reconciling / edits+deletes all land within the injected memory's scope: in multi-user
        scenarios, give each user its own Memory(scope=...) instance each with its own SmartWriter and
        isolation is natural (consistent with the alice/bob pattern).
        """
        records = []
        sc = self.memory._scope(scope, "SmartWriter.write")
        for fact in await self._extract(text):
            similar = await self.memory.asearch(fact, top_k=self.similar_k, scope=sc)
            decision = {"op": "ADD"} if not similar else await self._reconcile(fact, similar)
            await asyncio.to_thread(self._execute, decision, fact, sc)
            # build the return record from a whitelist: take only known fields, do not splat the raw LLM dict (which could carry a "fact" key overwriting the real fact, or pass arbitrary keys through)
            records.append({"fact": fact, "op": decision.get("op"),
                            "id": decision.get("id"), "content": decision.get("content")})
        return records

    async def _extract(self, text: str) -> List[str]:
        """Call the LLM to extract atomic facts (async), returning a list of strings (on parse failure, fail_open decides whether to degrade to the original text or drop it)."""
        resp = await governed_chat(self.llm, [{"role": "system", "content": self.extract_prompt},
                                              {"role": "user", "content": text}],
                                   tracer=self.tracer, origin="memory.smart_writer.extract")
        return self._parse_extract(resp.content, text, self.fail_open)

    @staticmethod
    def _parse_extract(content: str, text: str, fail_open: bool = True) -> List[str]:
        """Parse the extraction result (shared by sync/async): a JSON array is taken item by item; when not an array, fail_open degrades to one entry of the original text, otherwise it drops and returns empty.

        Only string items are accepted: calling str() on a non-string would turn a dict / number / null
        into garbage in the store; dict items take the fact / content / text field by convention, and the
        rest (numbers / null / nested lists) are skipped, not force-converted.
        """
        data = _parse_json(content)
        if isinstance(data, list):
            out = []
            for x in data:
                if isinstance(x, str):
                    if x.strip():
                        out.append(x)
                elif isinstance(x, dict):
                    fact = x.get("fact") or x.get("content") or x.get("text")
                    if isinstance(fact, str) and fact.strip():
                        out.append(fact)
            return out
        return [text] if fail_open else []

    async def _reconcile(self, fact: str, similar: List) -> dict:
        """Call the LLM to decide ADD/UPDATE/DELETE/NOOP for this fact (async); map the index back to the real id. Degrades to ADD on parse failure."""
        resp = await governed_chat(self.llm, [{"role": "system", "content": self.reconcile_prompt},
                                              {"role": "user", "content": self._reconcile_user(fact, similar)}],
                                   tracer=self.tracer, origin="memory.smart_writer.reconcile")
        return self._parse_reconcile(resp.content, similar)

    def _reconcile_user(self, fact: str, similar: List) -> str:
        """Assemble the user message for the comparison (shared by sync/async): rendered from registry memory.reconcile_user."""
        listing = "\n".join(f"{i}. {r.content}" for i, r in enumerate(similar, start=1))
        return self.prompts.render("memory.reconcile_user", fact=fact, listing=listing)

    @staticmethod
    def _parse_reconcile(content: str, similar: List) -> dict:
        """Parse the reconcile result (shared by sync/async): validate op, map the index back to the real id, and fall back to ADD if invalid (never a wrong delete)."""
        data = _parse_json(content)
        if not (isinstance(data, dict) and data.get("op") in ("ADD", "UPDATE", "DELETE", "NOOP")):
            return {"op": "ADD"}
        if data["op"] in ("UPDATE", "DELETE"):
            idx = data.get("index")
            if isinstance(idx, str) and idx.strip().lstrip("-").isdigit():   # the LLM occasionally returns a numeric string like "2", normalize it
                idx = int(idx)
            # crucial: bool is a subclass of int, so `{"index": true}` would be treated as index=1 and wrongly delete similar[0]; explicitly exclude bool
            if not (isinstance(idx, int) and not isinstance(idx, bool) and 1 <= idx <= len(similar)):
                return {"op": "ADD"}
            data["id"] = similar[idx - 1].id
        return data

    def _execute(self, decision: dict, fact: str, scope: Optional[Scope] = None) -> None:
        """Apply a reconciliation decision within one memory scope."""
        op = decision.get("op")
        sc = self.memory._scope(scope, "SmartWriter.execute")
        if op == "UPDATE" and decision.get("id"):
            old = self.memory.store.get(decision["id"], scope=sc)
            if old is None:
                new_item = self.memory.add(decision.get("content") or fact, scope=sc)
            else:
                new_item = self.memory.add(decision.get("content") or fact, scope=sc, type=old.type,
                                           importance=old.importance, metadata=old.metadata)
            self.memory.invalidate(decision["id"], superseded_by=new_item.id, scope=sc)
        elif op == "DELETE" and decision.get("id"):
            self.memory.invalidate(decision["id"], scope=sc)
        elif op == "NOOP":
            pass
        else:                                   # ADD, and the fallback for UPDATE/DELETE missing an id
            self.memory.add(fact, scope=sc)
