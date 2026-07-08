"""agentmaker.rag.ingest: the document ingestion orchestrator.

Wires the loader / splitter together with the retrieval base and the source-of-truth store:
read file -> chunk -> ingest. Ingesting means (1) store into the source-of-truth store (the full chunks)
plus (2) push into the retrieval index (vector + keyword, so it becomes searchable); both sides are
aligned by the same chunk_id.

Deduplicates via upsert by doc_id: when the same document is re-ingested, the new version is fully
ingested first, then the old version is deleted by its old chunk_ids (on failure the old version is
kept, so no data is lost). Uses scope.base="rag" to share the same retrieval base with memory
(base="memory") while keeping the data of the two isolated.
"""

import asyncio
import hashlib
import logging
import os
from collections import namedtuple
from typing import TYPE_CHECKING, List, Optional

from ..core.text import TokenCounter, count_tokens
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.index_sync import IndexSync, SyncIndexSync
from ..retrieval.scope import Scope
from .types import Chunk, Document, IngestReport
from .loader import load_file
from .source_store import SourceStore
from .splitter import split_document
from .types import ChunkingConfig

_RebuildItem = namedtuple("_RebuildItem", ("id", "content", "metadata"))   # adapts Chunk to the (.id / .content / .metadata) shape reconcile expects

if TYPE_CHECKING:
    from .contextualizer import Contextualizer


class IngestionPipeline:
    """Document ingestion orchestrator: coordinates loader -> splitter -> SourceStore + HybridRetriever."""

    def __init__(self, retriever: HybridRetriever, source_store: SourceStore, *,
                 contextualizer: Optional["Contextualizer"] = None, scope: Optional[Scope] = None,
                 config: Optional[ChunkingConfig] = None, index_sync: Optional[IndexSync] = None,
                 token_counter: TokenCounter = count_tokens):
        """
        Args:
            retriever: The retrieval base (vector + keyword + RRF + optional rerank).
            source_store: The RAG source-of-truth store.
            contextualizer: Optional Contextual Retrieval enhancer; if provided, the "enhanced text"
                is pushed into the index while the original chunks are still stored in the source-of-truth
                store. If omitted, the original chunks are used directly (backward compatible).
            scope: The default ownership, isolating RAG data from memory; defaults to Scope(base="rag").
            config: Optional chunking knobs (chunk_tokens / overlap_tokens); defaults to ChunkingConfig()
                if omitted. The per-call kwargs of ingest_file / ingest_text can override these
                (three-level resolution).
            index_sync: Optional derived-index sync seam (IndexSync, shared with memory, see
                agentmaker.retrieval.index_sync); ingestion goes through it for the atomic-replace
                plus reconciliation. Defaults to SyncIndexSync(retriever) if omitted. For async /
                distributed use, implement and inject your own.
            token_counter: Pluggable token counter (defaults to count_tokens); chunking budgets are
                estimated with it, so it is recommended to use the same ruler as the context budget.
        """
        self.retriever = retriever
        self.source_store = source_store
        self.contextualizer = contextualizer
        self.scope = scope or Scope(base="rag")
        self.cfg = config or ChunkingConfig()
        self._sync = index_sync if index_sync is not None else SyncIndexSync(retriever)
        self._count = token_counter       # passed through to split_document; chunking estimates budgets with it

    @classmethod
    def from_config(cls, config, *, embedder=None, retriever: Optional[HybridRetriever] = None,
                    source_store: Optional[SourceStore] = None, db_path: str = ":memory:",
                    reranker=None, contextualizer: Optional["Contextualizer"] = None,
                    index_sync: Optional[IndexSync] = None,
                    token_counter: TokenCounter = count_tokens) -> "IngestionPipeline":
        """Assemble an IngestionPipeline from an AgentmakerConfig in one line: defaults to the sqlite backend; pass retriever / source_store to inject a custom backend.

        Must share the same base with RagRetriever (the assembly root is in the app). Typical usage:
        first rag = RagRetriever.from_config(...), then
        IngestionPipeline.from_config(config, retriever=rag.retriever, source_store=rag.source_store),
        so the two read and write the same data.

        Args:
            config: AgentmakerConfig (reads config.chunking).
            embedder: Required when using the default sqlite base; not needed if a retriever is injected.
            retriever / source_store: Inject a custom backend (typically reuse the same instances built
                by RagRetriever); if omitted, a default sqlite one is built.
        """
        config.chunking.validate()                            # validate the slice we actually use before dispatch
        if retriever is None:
            if embedder is None:
                raise ValueError("IngestionPipeline.from_config needs an embedder for the default sqlite base; or pass retriever= to inject a custom backend")
            config.retrieval.validate()
            from ..retrieval.backends import build_sqlite_hybrid   # lazy import: sqlite is the default, do not couple it into this module's top level
            retriever = build_sqlite_hybrid(embedder, db_path=db_path, reranker=reranker, config=config.retrieval)
        if index_sync is None:
            # The assembly path defaults to persistent bookkeeping (same DB as the source of truth): the repair set survives across processes (direct __init__ construction is still the in-process default)
            from ..retrieval.index_sync import SqliteBookkeeping, SyncIndexSync
            index_sync = SyncIndexSync(retriever, bookkeeping=SqliteBookkeeping(db_path))
        return cls(retriever, source_store if source_store is not None else SourceStore(db_path),   # is not None: avoid being tripped up by a custom store whose __bool__ is False
                   contextualizer=contextualizer, config=config.chunking, index_sync=index_sync,
                   token_counter=token_counter)

    def ingest_file(self, path: str, *, doc_id: Optional[str] = None,
                    chunk_tokens: Optional[int] = None, overlap_tokens: Optional[int] = None) -> "IngestReport":
        """Read file -> chunk -> ingest. Returns IngestReport(doc_id / chunks / skipped); skipped=True when the content is unchanged and the run short-circuits. chunk/overlap default to self.cfg if omitted.

        If doc_id is omitted, it is stably derived from the file's absolute path: re-ingesting the same
        file yields the same doc_id, which combined with the content fingerprint means an unchanged file
        short-circuits the whole run (no chunking, no per-chunk LLM enhancement, no embedding); a changed
        file atomically replaces the old version (upsert dedup).
        """
        chunk_tokens = self.cfg.chunk_tokens if chunk_tokens is None else chunk_tokens
        overlap_tokens = self.cfg.overlap_tokens if overlap_tokens is None else overlap_tokens
        doc = load_file(path)
        doc.doc_id = doc_id or hashlib.sha1(("file:" + os.path.abspath(path)).encode("utf-8")).hexdigest()
        skipped = self._maybe_skip(doc, chunk_tokens, overlap_tokens)
        if skipped is not None:
            return skipped
        chunks = split_document(doc, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens, token_counter=self._count)
        return self._index(doc, chunks, fingerprint=self._fingerprint(doc, chunk_tokens, overlap_tokens))

    def ingest_text(self, text: str, *, source: str = "", title: Optional[str] = None,
                    doc_id: Optional[str] = None, fmt: str = "txt",
                    chunk_tokens: Optional[int] = None, overlap_tokens: Optional[int] = None) -> "IngestReport":
        """Ingest a piece of text directly (no file). If title is omitted it is derived from source (its filename); if doc_id is omitted one is auto-generated. chunk/overlap default to self.cfg if omitted.

        The concept of "re-ingest" only applies when doc_id is passed: unchanged content short-circuits
        the whole run (skipped=True), changed content atomically replaces the old version. Returns IngestReport.
        """
        chunk_tokens = self.cfg.chunk_tokens if chunk_tokens is None else chunk_tokens
        overlap_tokens = self.cfg.overlap_tokens if overlap_tokens is None else overlap_tokens
        doc = Document(content=text, source=source, format=fmt)
        if title is not None:
            doc.title = title
        elif source:
            doc.title = os.path.splitext(os.path.basename(source))[0]  # derive the title from source, giving chunks "which file this belongs to" context
        if doc_id is not None:
            doc.doc_id = doc_id
            skipped = self._maybe_skip(doc, chunk_tokens, overlap_tokens)
            if skipped is not None:
                return skipped
        chunks = split_document(doc, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens, token_counter=self._count)
        # Only register a fingerprint when doc_id was explicitly passed: otherwise doc_id is a random uuid, the short-circuit never hits, and registering would just accumulate never-reused junk fingerprint rows in the docs table
        fp = self._fingerprint(doc, chunk_tokens, overlap_tokens) if doc_id is not None else None
        return self._index(doc, chunks, fingerprint=fp)

    def _fingerprint(self, doc: Document, chunk_tokens: int, overlap_tokens: int) -> str:
        """Doc-level ingestion fingerprint: body + title + source + chunking params + enhancer fingerprint.

        Changing any one of these (including swapping the LLM enhancer's prompt / model, or changing the
        title) counts as "changed", so a re-ingest is not wrongly skipped (otherwise a new title never
        makes it into the chunk's heading_path, and after a prompt swap the index keeps the stale
        enhanced text).
        """
        ctx = self.contextualizer.fingerprint() if self.contextualizer is not None else "none"
        # Include doc.format: re-ingesting under the same doc_id with a different format (e.g. txt -> md, where chunking goes from plain-text to heading-aware) must change the fingerprint and not be wrongly skipped
        raw = f"{doc.content}\x00{doc.title}\x00{doc.source}\x00{doc.format}\x00{chunk_tokens}|{overlap_tokens}|{ctx}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _maybe_skip(self, doc: Document, chunk_tokens: int, overlap_tokens: int) -> Optional["IngestReport"]:
        """Re-ingest short-circuit: if the fingerprint matches the last ingestion, skip the whole run (no chunking, no per-chunk LLM enhancement, no embedding) and return IngestReport(skipped=True); otherwise None."""
        if self.source_store.get_doc_hash(doc.doc_id, scope=self.scope) != \
                self._fingerprint(doc, chunk_tokens, overlap_tokens):
            return None
        n = len(self.source_store.chunk_ids_of_doc(doc.doc_id, scope=self.scope))
        return IngestReport(doc_id=doc.doc_id, chunks=n, skipped=True)

    def delete_document(self, doc_id: str) -> int:
        """Delete a whole document (first the retrieval index, then the source-of-truth store, and clear the ingestion fingerprint); returns the number of chunks deleted."""
        ids = self.source_store.chunk_ids_of_doc(doc_id, scope=self.scope)
        self._delete_ids(ids)
        self.source_store.delete_doc_hash(doc_id, scope=self.scope)
        return len(ids)

    def rebuild_index(self, *, scope: Optional[Scope] = None, batch_size: int = 256) -> int:
        """Fully rebuild the retrieval index from the source-of-truth store (symmetric with Memory.rebuild_index): iterate over all chunks in the scope and, through the seam's reconcile, "delete orphans + force re-push in batches". Returns the number of chunks re-pushed.

        Uses: migrating data when switching retrieval backends (sqlite -> pgvector), fully re-embedding
        after switching embedding models, and recovering from index corruption / a pending-repair set.

        Boundary: re-pushing uses the original chunk text from the source-of-truth store. For a store
        with a Contextualizer attached, the enhanced text is not in the source of truth (intentional: the
        source of truth stays clean), so after a rebuild the index reverts to un-enhanced text; to rebuild
        with enhancement, re-ingest the source document.

        Args:
            scope: Which ownership scope to rebuild; defaults to this pipeline's scope.
            batch_size: The batch size for re-pushing.
        """
        sc = scope or self.scope
        # Carry the filterable metadata (same shape as _index), otherwise a reconcile re-push would blank out filterable columns like md_doc_id and filters would silently return zero hits
        items = [_RebuildItem(c.chunk_id, c.content, {"doc_id": c.doc_id, **c.metadata})
                 for c in self.source_store.all_chunks(scope=sc)]
        return self._sync.reconcile(items, scope=sc, batch_size=batch_size)

    def verify(self, *, scope: Optional[Scope] = None) -> dict:
        """Cross-check consistency between the source-of-truth store and the retrieval index: return a divergence report without auto-repairing.

        After a normal ingest, the chunk_id sets of the two are equal. If clearing old chunks fails or a
        crash leaves stale rows in the source of truth (the retrieval index was already atomically
        replaced but the source of truth was not cleaned), you get "source of truth is a superset of the
        indexed set"; those stale rows would be revived from the source of truth by rebuild_index. This
        method surfaces the divergence (and logs a warning), leaving the app to decide to re-ingest that
        document to fix it (one clean ingest converges it via replace).

        Args:
            scope: Which ownership scope to check; defaults to this pipeline's scope.

        Returns:
            dict: {"scope", "source_only" (ids present in the source of truth but not tracked by the
                   index), "index_only" (ids tracked by the index but no longer in the source of truth),
                   "consistent" (True if the two are equal; None if the seam does not support
                   enumeration)}.
        """
        sc = scope or self.scope
        source_ids = {c.chunk_id for c in self.source_store.all_chunks(scope=sc)}
        try:
            indexed_ids = self._sync.tracked_ids(scope=sc)
        except NotImplementedError:
            return {"scope": sc, "source_only": [], "index_only": [], "consistent": None}
        source_only = sorted(source_ids - indexed_ids)
        index_only = sorted(indexed_ids - source_ids)
        consistent = not source_only and not index_only
        if not consistent:
            logging.getLogger(__name__).warning(
                "RAG index diverges from source of truth (scope=%s): source of truth has %d extra chunks, index has %d extra chunks; consider re-ingesting the affected documents to converge",
                sc, len(source_only), len(index_only))
        return {"scope": sc, "source_only": source_only, "index_only": index_only, "consistent": consistent}

    def stats(self) -> dict:
        """Return {documents, chunks}: the document count and total chunk count."""
        docs = self.source_store.list_docs(scope=self.scope)
        return {"documents": len(docs), "chunks": sum(docs.values())}

    def _index(self, doc: Document, chunks: List[Chunk], *, fingerprint: Optional[str] = None) -> "IngestReport":
        """Core: store new chunks into the source of truth and "atomically replace" the old chunks in the retrieval index (upsert; on failure keep the old and lose no data).

        The source of truth and the retrieval index live in different SQLite connections and cannot share
        a single cross-database transaction. On the retrieval-index side, the seam `IndexSync.replace`
        (by default delegating to retriever.replace) collects "push new chunks + delete old chunks" into a
        single atomic transaction commit (SqliteHybridRetriever): retrieval either sees all old or all
        new, ruling out both the concurrency window where old and new coexist and get hit together, and
        residue from a failed old-chunk delete. `replace` raises on failure (aside from the best-effort
        index/drop, it is the only atomic, fail-loud entry in the seam), handing this method the rollback.
        On the source-of-truth side, the new chunks are stored first (their ids differ from the old ones,
        so no overwriting), and the old orphan chunks are cleared last (at which point the retrieval index
        no longer points at them, they are unsearchable, cleanup failure is harmless and self-heals on
        re-ingest). If any step fails, only this batch of new chunks is rolled back and the old version is
        kept (rather than the document being deleted entirely).
        """
        new_ids = [c.chunk_id for c in chunks]
        old_ids = self.source_store.chunk_ids_of_doc(doc.doc_id, scope=self.scope)  # old version: record now, clear last
        if not chunks:
            self._delete_ids(old_ids)  # empty document = delete this doc (no new chunks to ingest, just clear the old version)
            self.source_store.delete_doc_hash(doc.doc_id, scope=self.scope)
            return IngestReport(doc_id=doc.doc_id, chunks=0)
        try:
            # (1) Store into the source of truth (original chunks, the authoritative copy: kept clean, no enhancement context)
            self.source_store.save_chunks(chunks, scope=self.scope)
            # (2) Push into the retrieval index: with a contextualizer, use the "enhanced text" (improves searchability), otherwise the original text.
            #     The enhanced text only goes into the index, not the source of truth: after a retrieval hit, the source of truth still returns the clean original chunk.
            if self.contextualizer is not None:
                texts = [self.contextualizer.contextualize(c, doc) for c in chunks]
            else:
                texts = [c.content for c in chunks]
            # Per-chunk filterable metadata (doc_id + the chunk's own metadata); the base only stores declared fields (see MetadataFilter in retrieval/types.py) and ignores undeclared ones
            metadatas = [{"doc_id": c.doc_id, **c.metadata} for c in chunks]
            # Atomic replace through the seam (push new chunks + delete old chunks, no concurrent double-hit window); raises on failure -> roll back new chunks below and keep the old version
            self._sync.replace(old_ids, new_ids, texts, scope=self.scope, metadatas=metadatas)
        except Exception:
            self._best_effort_delete(new_ids, what="roll back new chunks after ingest failure", level="error")  # consistency is compromised (residual new chunks) -> error level
            raise
        # (3) The retrieval index is atomically swapped to the new chunks; clear the old-version orphan chunks in the source of truth (old ids not in new). Cleanup failure leaves a warning: residual old chunks are unsearchable,
        #     but rebuild_index would revive them from the source of truth as stale rows, so raise it to warning for monitoring visibility, paired with verify()'s cross-check to detect divergence.
        new_set = set(new_ids)
        self._best_effort_delete([cid for cid in old_ids if cid not in new_set], what="clear old-version orphan chunks", level="warning")
        if fingerprint is not None:
            self.source_store.set_doc_hash(doc.doc_id, fingerprint, scope=self.scope)   # register only on ingest success: on failure keep the old fingerprint and retry next time
        return IngestReport(doc_id=doc.doc_id, chunks=len(chunks))

    def _delete_ids(self, ids: List[str]) -> None:
        """Delete by id from both the retrieval index and the source of truth (index first, then source of truth).

        The index delete goes through the seam `drop` (best-effort: on failure mark for repair, do not
        raise); if the index delete fails and leaves an "in index, not in source of truth" orphan,
        RagRetriever.retrieve filters and self-heals on read (it does not return deleted content). A
        source-of-truth delete failure raises as usual (a real delete failure is a real error).
        """
        if not ids:
            return
        self._sync.drop(ids, scope=self.scope)
        self.source_store.delete_chunks(ids, scope=self.scope)

    def close(self) -> None:
        """Close the index-sync seam _sync (including the SqliteBookkeeping connection installed by default in from_config). source_store is owned by the caller and released per ownership rules."""
        self._sync.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _best_effort_delete(self, ids: List[str], *, what: str = "cleanup", level: str = "debug") -> None:
        """Compensating / cleanup delete (best-effort: does not re-raise and does not mask the original exception); on failure leave a signal per level.

        level: use "error" for compensation where consistency is compromised (e.g. rolling back new
        chunks after an ingest failure, where the residue is dirty index data); use "debug" for harmless
        cleanup (orphan chunks).
        """
        try:
            self._delete_ids(ids)
        except Exception as e:  # noqa: BLE001  best-effort: do not re-raise and do not mask the original exception, but no longer stay fully silent
            getattr(logging.getLogger(__name__), level)("%s failed (best-effort): %r", what, e)

    # a* async versions (chunking + Contextualizer + embedding are all sync / network operations, wrapped in to_thread, awaited directly inside the event loop)

    async def aingest_file(self, path: str, **kwargs) -> "IngestReport":
        """Async version of ingest_file (to_thread)."""
        return await asyncio.to_thread(lambda: self.ingest_file(path, **kwargs))

    async def aingest_text(self, text: str, **kwargs) -> "IngestReport":
        """Async version of ingest_text (to_thread)."""
        return await asyncio.to_thread(lambda: self.ingest_text(text, **kwargs))

    async def adelete_document(self, doc_id: str) -> int:
        """Async version of delete_document (to_thread)."""
        return await asyncio.to_thread(self.delete_document, doc_id)


