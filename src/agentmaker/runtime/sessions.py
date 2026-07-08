"""agentmaker.runtime.sessions: conversation-history persistence (append-only, isolated by Scope).

Lets a long-running daemon survive a restart without losing conversations: it persists the Agent's conversation
history (Message) into SQLite per session. Append-only: each message is stored as one row, only appended and never
rewritten (cheap writes, natural auditability, restore in insertion order). Session identity reuses the same Scope
as retrieval / memory (its session dimension is the session identifier), so the whole framework has one isolation model.

Also includes `ConversationSearch` (episodic memory): it wraps any SessionStore to make past conversations
semantically searchable. Extractive memory (SmartWriter) inevitably loses detail on compaction, and the industry
consensus is to keep the raw conversation as an unloseable source-of-truth layer that can be searched back (Letta recall
memory / conversation_search, Zep's episode layer, ChatGPT / Claude conversation-history search, same shape). The Agent
attaches it directly (it is itself a SessionStore, no changes needed); the consumption layer is symmetric with memory / rag:
a third CallableSource + ConversationSearchTool.
"""

import asyncio
import json
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Optional

from ..core.clock import ensure_utc
from ..core.exceptions import SessionError
from ..core.message import Message
from ..core.multimodal import content_text
from ..core.sqlite_util import open_sqlite, require_columns
from ..retrieval.index_sync import IndexSync, SyncIndexSync

# Metadata flag marking that the TEXT content column holds JSON-encoded multimodal parts
# (the flag is stripped again on load, so callers see their original metadata untouched).
_CONTENT_FORMAT_KEY = "_content_format"


def _encode_content(message: Message):
    """Return (storable content string, metadata dict) for one message: multimodal part lists
    are JSON-encoded into the TEXT column with a metadata flag so load() restores them faithfully."""
    if isinstance(message.content, list):
        metadata = {**(message.metadata or {}), _CONTENT_FORMAT_KEY: "parts"}
        return json.dumps(message.content, ensure_ascii=False), metadata
    return message.content, message.metadata
from ..retrieval.scope import Scope
from ..retrieval.scope_sql import (scope_column_for, scope_column_names, scope_exact_where_clause,
                                   scope_store_values, scope_where_clause)
from ..retrieval.types import RetrievalResult


@dataclass(frozen=True)
class ScopeSummary:
    """A session overview of one value (one "bucket") along a scope dimension: who, how many messages, time span. The element type returned by list_scopes."""
    dimension: str                    # The dimension being enumerated (e.g. "session")
    value: str                        # This dimension's value ("" = the default / unnamed bucket)
    message_count: int                # Number of messages in this bucket
    first_at: Optional[datetime]      # Earliest message time (UTC); None for an empty bucket
    last_at: Optional[datetime]       # Latest message time (UTC); None for an empty bucket


class SessionStore(ABC):
    """Conversation-history storage interface: append / read / clear conversation messages by Scope.

    Isolation semantics: load / clear match all scope dimensions exactly (an empty scope only hits the "all-empty" default
    session bucket, never crossing into other sessions); a truly global operation across all sessions requires an explicit
    `all_scopes=True` (unlike the B semantics of memory / rag retrieval: misreading / mis-deleting a session is far more
    dangerous than a retrieval miss, so the default is tightened and the dangerous path made explicit).
    """

    @abstractmethod
    def append(self, message: Message, *, scope: Optional[Scope] = None) -> None:
        """Append one message to the session identified by scope (append-only)."""

    def append_many(self, messages: List[Message], *, scope: Optional[Scope] = None) -> None:
        """Atomically append multiple messages (a turn's user+assistant land together, avoiding a half-turn of history).

        The default implementation appends one by one (non-atomic, a fallback only); a backend with transaction support should override it as a single transaction (see SqliteSessionStore).
        """
        for m in messages:
            self.append(m, scope=scope)

    @abstractmethod
    def load(self, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> List[Message]:
        """Read all messages of the scope's session in insertion order; an empty session returns [].

        By default it matches all scope dimensions exactly (an empty scope reads only the "all-empty" default bucket, not crossing into other sessions);
        `all_scopes=True` reads across all sessions (for ops / export, use with care).
        """

    @abstractmethod
    def clear(self, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> None:
        """Clear all messages of the scope's session (matching all scope dimensions exactly); `all_scopes=True` clears all sessions (use with care)."""

    # a* async dual track: the framework's async execution layer (BaseAgent's per-turn load / wrap-up append) calls the a* forms;
    #    the default wraps the sync version in to_thread (DB IO really exists), so a sync implementation gets it for free; an async backend may override with a native await.
    async def aappend(self, message: Message, *, scope: Optional[Scope] = None) -> None:
        """Async version of append (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.append(message, scope=scope))

    async def aappend_many(self, messages: List[Message], *, scope: Optional[Scope] = None) -> None:
        """Async version of append_many (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.append_many(messages, scope=scope))

    async def aload(self, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> List[Message]:
        """Async version of load (defaults to to_thread)."""
        return await asyncio.to_thread(lambda: self.load(scope=scope, all_scopes=all_scopes))

    async def aclear(self, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> None:
        """Async version of clear (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.clear(scope=scope, all_scopes=all_scopes))

    # Optional capability: enumerate which values a dimension has (e.g. "list which sessions exist"). Not supported by the base class by default, overridden by backends that can aggregate efficiently.
    def list_scopes(self, *, along: str = "session", scope: Optional[Scope] = None) -> List[ScopeSummary]:
        """Enumerate which values exist along the `along` dimension, giving each bucket a message count and first/last time (e.g. list "which sessions exist" for the app to build a conversation list).

        Args:
            along: The dimension to enumerate (base/user/agent/session/app), default "session".
            scope: Uses non-empty dimensions (B semantics) to constrain the "other dimensions": e.g. Scope(user="alice") lists
                only sessions under alice; passing None / Scope() enumerates globally. This is a read-only discovery operation
                that returns no message bodies and deletes no data, so it uses B semantics (unlike load/clear's exact match of all
                dimensions). The enumerated `along` dimension is not itself used as a filter condition.

        Returns:
            List[ScopeSummary]: descending by most recent activity (last_at); `value=""` denotes the default bucket where that dimension is unspecified.

        Raises NotImplementedError by default: not every backend can DISTINCT + aggregate efficiently; SqliteSessionStore implements it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support list_scopes (enumerating scope dimension values)")

    async def alist_scopes(self, *, along: str = "session", scope: Optional[Scope] = None) -> List[ScopeSummary]:
        """Async version of list_scopes (defaults to to_thread)."""
        return await asyncio.to_thread(lambda: self.list_scopes(along=along, scope=scope))


class SqliteSessionStore(SessionStore):
    """Conversation history stored in SQLite: one row per message (scope columns + role / content / time / metadata), ordered by rowid."""

    def __init__(self, db_path: str = ":memory:"):
        """Open the connection and create the table if needed.

        Args:
            db_path: SQLite file path; the default ":memory:" is for self-tests only, use a file path in production to persist
                (may share the database with memory / the vector store by passing the same file path).
        """
        scope_cols = ", ".join(f"{c} TEXT" for c in scope_column_names())
        self._lock = threading.Lock()  # Serialize cross-thread access: a single connection with check_same_thread=False does not guarantee concurrency safety (low concurrency of a personal daemon is enough)
        try:
            self._db = open_sqlite(db_path)
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS session_messages("
                f"{scope_cols}, role TEXT, content TEXT, created_at TEXT, metadata TEXT)")
            # Composite index over the scope dimensions (aligned with checkpoints): load / clear / prune all filter exactly by scope, and a full-table scan gets slower as sessions accumulate.
            self._db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_session_messages_scope ON session_messages({', '.join(scope_column_names())})")
            # Startup self-check: the append-only conversation source-of-truth is the least reconstructable user data. When an old
            # database lacks a new scope dimension column, turn the cryptic "no such column" into a clear error with migration
            # guidance (scope columns cannot be backfilled automatically: NULL in old rows breaks empty-string exact matching).
            require_columns(self._db, "session_messages", set(scope_column_names()), error_cls=SessionError)
            self._db.commit()
        except sqlite3.Error as e:
            raise SessionError(f"Failed to open / initialize the session store: {e}") from e

    def append(self, message: Message, *, scope: Optional[Scope] = None) -> None:
        """Append one message (append-only: only INSERT, never modify old rows). metadata is stored as JSON, and time as a UTC-normalized isoformat."""
        sv = scope_store_values(scope or Scope())
        cols = ", ".join(scope_column_names())
        placeholders = ", ".join("?" for _ in scope_column_names())
        content_value, metadata = _encode_content(message)
        metadata_json = self._dump_metadata(metadata)
        with self._lock:
            try:
                self._db.execute(
                    f"INSERT INTO session_messages({cols}, role, content, created_at, metadata) "
                    f"VALUES ({placeholders}, ?, ?, ?, ?)",
                    (*sv, message.role, content_value, ensure_utc(message.timestamp).isoformat(), metadata_json))
                self._db.commit()
            except sqlite3.Error as e:
                self._db.rollback()   # Roll back the half-finished transaction so the next write does not carry this dirty state out (append_many uses with self._db which already rolls back, left untouched)
                raise SessionError(f"Failed to write the session message: {e}") from e

    def append_many(self, messages: List[Message], *, scope: Optional[Scope] = None) -> None:
        """Write multiple messages in a single transaction (all succeed or all fail), avoiding a half-turn of history where the user write succeeds but the assistant write fails."""
        if not messages:
            return
        sv = scope_store_values(scope or Scope())
        cols = ", ".join(scope_column_names())
        placeholders = ", ".join("?" for _ in scope_column_names())
        encoded = [(m, *_encode_content(m)) for m in messages]
        rows = [(*sv, m.role, content_value, ensure_utc(m.timestamp).isoformat(), self._dump_metadata(metadata))
                for m, content_value, metadata in encoded]
        with self._lock:
            try:
                with self._db:  # The connection as a context manager: commit on success, roll back on exception (the transaction guarantees atomicity)
                    self._db.executemany(
                        f"INSERT INTO session_messages({cols}, role, content, created_at, metadata) "
                        f"VALUES ({placeholders}, ?, ?, ?, ?)", rows)
            except sqlite3.Error as e:
                raise SessionError(f"Failed to batch-write session messages: {e}") from e

    def load(self, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> List[Message]:
        """Read all messages of the scope's session by rowid (i.e. insertion order).

        By default it matches all scope dimensions exactly (an empty scope reads only the "all-empty" default bucket); all_scopes=True reads across all sessions.
        """
        where, params = self._filter(scope, all_scopes)
        with self._lock:
            try:
                cur = self._db.execute(
                    "SELECT role, content, created_at, metadata FROM session_messages"
                    f"{where} ORDER BY rowid ASC", params)
                return [self._row_to_message(row) for row in cur.fetchall()]
            except sqlite3.Error as e:
                raise SessionError(f"Failed to read session messages: {e}") from e

    def clear(self, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> None:
        """Delete all messages of the scope's session (by default matching all scope dimensions exactly); all_scopes=True clears all sessions (use with care)."""
        where, params = self._filter(scope, all_scopes)
        with self._lock:
            try:
                self._db.execute(f"DELETE FROM session_messages{where}", params)
                self._db.commit()
            except sqlite3.Error as e:
                self._db.rollback()
                raise SessionError(f"Failed to clear the session: {e}") from e

    def prune(self, *, scope: Optional[Scope] = None, all_scopes: bool = False,
              keep_last: Optional[int] = None, before: Optional[datetime] = None) -> int:
        """Truncate a session's history and return the number of messages deleted: `keep_last=N` keeps only the most recent N of this scope, `before=time` deletes messages earlier than that time.

        Sessions accumulated over the long term make each turn's load full-table scan and grow slower; the app can prune periodically to control single-session length / clean up stale sessions.
        At least one of `keep_last` and `before` must be given (giving neither raises, to guard against an accidental wipe; to clear the whole session use `clear`); if both are given, it first deletes messages earlier than
        `before`, then truncates the remainder by `keep_last`. By default it matches all scope dimensions exactly (like load / clear); `all_scopes=True` spans all sessions
        (in which case `keep_last` means "the most recent N across all sessions combined", mixing across users, use with care).

        Args:
            scope: Target session; defaults to the "all-empty" bucket.
            all_scopes: Span all sessions (use with care).
            keep_last: Keep only this many most recent messages (must be >= 0).
            before: Delete messages earlier than this time (naive is treated as UTC).
        """
        if keep_last is None and before is None:
            raise SessionError("prune needs at least one of keep_last or before (to guard against an accidental wipe of the whole session; to clear, use clear)")
        if keep_last is not None and keep_last < 0:
            raise SessionError(f"prune's keep_last must be >= 0, got {keep_last}")
        where, params = self._filter(scope, all_scopes)
        deleted = 0
        with self._lock:
            try:
                if before is not None:
                    # created_at is TEXT, and `< ?` is a lexicographic comparison: it is equivalent to a chronological comparison
                    # only when both the stored values and the bound are the same canonical UTC isoformat. The write side already
                    # ensure_utc's (append/append_many), and the bound here is ensure_utc'd too, so both sides have the same width
                    # and suffix, lexicographic == chronological, and the strict `<` boundary is exact (rows equal to the bound are not deleted).
                    clause = f"{where} AND created_at < ?" if where else " WHERE created_at < ?"
                    cur = self._db.execute(f"DELETE FROM session_messages{clause}",
                                           (*params, ensure_utc(before).isoformat()))
                    deleted += cur.rowcount
                if keep_last is not None:
                    inner = f"SELECT rowid FROM session_messages{where} ORDER BY rowid DESC LIMIT ?"
                    clause = f"{where} AND rowid NOT IN ({inner})" if where else f" WHERE rowid NOT IN ({inner})"
                    cur = self._db.execute(f"DELETE FROM session_messages{clause}", (*params, *params, keep_last))
                    deleted += cur.rowcount
                self._db.commit()
                return deleted
            except sqlite3.Error as e:
                self._db.rollback()
                raise SessionError(f"Failed to truncate session history: {e}") from e

    def list_scopes(self, *, along: str = "session", scope: Optional[Scope] = None) -> List[ScopeSummary]:
        """DISTINCT + aggregate along the `along` dimension to get each bucket's message count and first/last time (B semantics constrain the other dimensions). See the base class list_scopes."""
        col = scope_column_for(along)                    # Validate that along is a legal dimension + get the fixed column name (the return is always a whitelisted column name, doubling as a SQL-injection guardrail)
        # B semantics filter only the "other non-empty dimensions"; clear along itself so it is not used as a filter condition (it is the dimension being enumerated)
        where, params = scope_where_clause(replace(scope or Scope(), **{along: None}))
        with self._lock:
            try:
                cur = self._db.execute(
                    f"SELECT {col}, COUNT(*), MIN(created_at), MAX(created_at) FROM session_messages"
                    f"{where} GROUP BY {col} ORDER BY MAX(created_at) DESC", params)
                return [ScopeSummary(dimension=along, value=row[0] or "", message_count=row[1],
                                     first_at=self._parse_ts(row[2]), last_at=self._parse_ts(row[3]))
                        for row in cur.fetchall()]
            except sqlite3.Error as e:
                raise SessionError(f"Failed to enumerate scope dimensions: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()

    @staticmethod
    def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
        """Restore the TEXT time from MIN/MAX(created_at) into a UTC datetime; NULL / empty returns None."""
        return ensure_utc(datetime.fromisoformat(ts)) if ts else None

    @staticmethod
    def _filter(scope: Optional[Scope], all_scopes: bool):
        """Build the WHERE + params for load / clear.

        all_scopes=True: no filter (across all sessions); otherwise match all scope dimensions exactly (empty dimensions compared
        as empty strings, an empty scope hitting only the "all-empty" default bucket), preventing misreading / mis-deleting all
        sessions when scope is accidentally omitted.
        """
        if all_scopes:
            return "", []
        return scope_exact_where_clause(scope or Scope())

    @staticmethod
    def _dump_metadata(metadata) -> str:
        """Serialize message.metadata to JSON; raise a clear error (rather than a bare exception bubbling up) when it contains non-serializable content.

        Catches both TypeError (unsupported types) and ValueError (a circular reference raises `ValueError: Circular reference detected`, which used to escape as a bare exception).
        """
        try:
            return json.dumps(metadata, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise SessionError(f"Session message metadata cannot be JSON-serialized (store basic types, no circular references): {e}") from e

    @staticmethod
    def _row_to_message(row) -> Message:
        """Restore one record row into a Message (faithfully preserving role / content / timestamp / metadata;
        multimodal part lists are decoded back from JSON and the internal flag is stripped)."""
        role, content, created_at, metadata = row
        metadata_dict = json.loads(metadata)
        if metadata_dict.pop(_CONTENT_FORMAT_KEY, None) == "parts":
            content = json.loads(content)
        return Message(content=content, role=role,
                       timestamp=ensure_utc(datetime.fromisoformat(created_at)), metadata=metadata_dict)


class ConversationSearch(SessionStore):
    """A semantically searchable session store: wraps any SessionStore (the source of truth) + a retrieval backbone (the derived index).

    Write path: append / append_many stamp each message with a stable message_id (stored in message.metadata, persisted along with
    the source of truth); after landing in the source of truth, they are fed into the index best-effort via IndexSync.index (an index
    failure does not affect the source of truth and is marked pending, the same "source of truth is authoritative, eventually
    consistent" semantics as Memory). On clear, it first fetches all message ids of the session and drops them from the index too.
    """

    def __init__(self, store: SessionStore, retriever, *, index_sync: Optional[IndexSync] = None):
        """
        Args:
            store: The source of truth (any SessionStore, e.g. SqliteSessionStore); this class is still a SessionStore to the outside, attached directly by the Agent.
            retriever: The shared retrieval backbone (HybridRetriever); may share the same instance as memory / rag (isolated by base="conversation").
            index_sync: Optional index-sync seam; if omitted, SyncIndexSync(retriever) is used (in-process bookkeeping; to persist bookkeeping, pass an instance with SqliteBookkeeping).
        """
        self.store = store
        self.retriever = retriever
        self._sync = index_sync if index_sync is not None else SyncIndexSync(retriever)

    @staticmethod
    def _conv_scope(scope: Optional[Scope]) -> Scope:
        """A session message's placement in the shared backbone: fix the base dimension to "conversation" (isolated the same way as memory/"memory", rag/"rag")."""
        return replace(scope or Scope(), base="conversation")

    @staticmethod
    def _stamp(message: Message) -> str:
        """Stamp the message with a stable id (idempotent: reuse an existing one). Stored in metadata and persisted with the source of truth, so the index aligns upsert / delete by it."""
        mid = message.metadata.get("message_id")
        if not mid:
            mid = uuid.uuid4().hex
            message.metadata["message_id"] = mid
        return mid

    def _feed(self, messages: List[Message], ids: List[str], scope) -> None:
        """Feed several messages into the index (role-prefixed content + role/time metadata). Best-effort: on failure mark pending, don't drag down writing the source of truth."""
        contents = [f"{m.role}: {content_text(m.content)}" for m in messages]   # image parts index as "[image: ...]" placeholders
        mds = [{"role": m.role, "created_at": ensure_utc(m.timestamp).isoformat()} for m in messages]
        self._sync.index(ids, contents, scope=self._conv_scope(scope), metadatas=mds)

    # SessionStore protocol: delegate to the source of truth + feed the index on the side.

    def append(self, message: Message, *, scope: Optional[Scope] = None) -> None:
        """Append one message: stamp id, land in the source of truth, feed the index (best-effort)."""
        mid = self._stamp(message)
        self.store.append(message, scope=scope)
        self._feed([message], [mid], scope)

    def append_many(self, messages: List[Message], *, scope: Optional[Scope] = None) -> None:
        """Atomically append multiple: stamp ids, land in the source of truth in a single transaction, feed the whole batch to the index (best-effort)."""
        if not messages:
            return
        ids = [self._stamp(m) for m in messages]
        self.store.append_many(messages, scope=scope)
        self._feed(messages, ids, scope)

    def load(self, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> List[Message]:
        """Read session messages in insertion order (pure delegation to the source of truth)."""
        return self.store.load(scope=scope, all_scopes=all_scopes)

    def clear(self, *, scope: Optional[Scope] = None, all_scopes: bool = False) -> None:
        """Clear the session: first drop the index by the message_ids in the source of truth (drop is best-effort), then clear the source of truth.

        Old messages predating this class have no message_id (not in the index), so just skip them.
        """
        ids = [m.metadata.get("message_id") for m in self.store.load(scope=scope, all_scopes=all_scopes)]
        ids = [i for i in ids if i]
        if ids:
            self._sync.drop(ids, scope=self._conv_scope(scope))
        self.store.clear(scope=scope, all_scopes=all_scopes)

    def list_scopes(self, *, along: str = "session", scope: Optional[Scope] = None) -> List[ScopeSummary]:
        """Enumerate a dimension's values in the source of truth (pure delegation; the index is not involved). See SessionStore.list_scopes."""
        return self.store.list_scopes(along=along, scope=scope)

    # Retrieval (episodic: search what was discussed in the past).

    def search(self, query: str, *, top_k: int = 5, scope: Optional[Scope] = None) -> List[RetrievalResult]:
        """Semantically search past conversation messages, returning the most relevant top_k (content like "user: ...", metadata with role / time).

        Args:
            query: The query text.
            top_k: Number of results to return.
            scope: Session placement (the same scope as at write time); isolated retrieval in the shared backbone by base="conversation".

        Returns:
            List[RetrievalResult]: from most to least relevant.
        """
        return self.retriever.search(query, top_k=top_k, scope=self._conv_scope(scope))

    async def asearch(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Async version of search (to_thread)."""
        return await asyncio.to_thread(lambda: self.search(query, **kwargs))

    def pending_reindex(self, *, scope: Optional[Scope] = None) -> set:
        """The set of message ids the index sync marked pending (index writes that failed and have not converged), for app monitoring (same semantics as Memory.pending_reindex)."""
        return self._sync.pending(scope=self._conv_scope(scope))

    def close(self) -> None:
        """Close the source of truth (the backbone is closed uniformly by whoever shares it)."""
        close = getattr(self.store, "close", None)
        if callable(close):
            close()


class ConversationSearchTool:
    """Wraps conversation retrieval as a tool: the model can proactively "search what we discussed before" (agentic episodic recall, aligned with Letta conversation_search).

    Same shape as MemoryTool / RAGTool (based on the Tool base class); read-only, no confirmation gate. scope is bound at
    construction (matching the served Agent's session scope) and not exposed to the model: the model only supplies the query, never touching the tenant boundary.
    """

    def __new__(cls, conversation_search: ConversationSearch, *, scope: Optional[Scope] = None,
                top_k: int = 5, prompts=None):
        """Construct the actual Tool instance (lazily import the Tool base class to avoid the runtime top level dragging in the tools package)."""
        from ..prompts import DEFAULT_PROMPTS
        from ..tools.base import Tool, ToolParameter
        from ..tools.response import ToolResponse

        p = prompts or DEFAULT_PROMPTS

        class _Tool(Tool):
            """The conversation_search tool entity."""

            def __init__(self):
                super().__init__(name="conversation_search", description=p.text("tool.desc.conversation_search"))

            def get_parameters(self):
                """Declare parameters: only the query."""
                return [ToolParameter("query", "string", p.text("tool.param.conversation_search.query"))]

            def run(self, parameters: dict) -> ToolResponse:
                """Search past conversations and return a readable list."""
                query = (parameters.get("query") or "").strip()
                if not query:
                    return ToolResponse.error(p.text("tool.msg.conv.need_query"))
                hits = conversation_search.search(query, top_k=top_k, scope=scope)
                if not hits:
                    return ToolResponse.ok(p.text("tool.msg.conv.no_match"))
                lines = [f"- [{h.metadata.get('created_at', '')[:16]}] {h.content}" for h in hits]
                return ToolResponse.ok(p.text("tool.msg.conv.found_prefix") + "\n" + "\n".join(lines))

        return _Tool()


