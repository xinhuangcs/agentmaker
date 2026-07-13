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
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING, List, Optional

from ..core.text import TokenCounter, count_tokens
from ..retrieval.hybrid import HybridRetriever
from ..retrieval._coordination import shared_coordinator
from ..retrieval.index_sync import IndexSync, SyncIndexSync
from ..retrieval.scope import Scope, canonical_scope
from .types import Chunk, Document, IngestReport
from .loader import load_file
from .source_store import SourceStore
from .splitter import split_document
from .types import ChunkingConfig

_RebuildItem = namedtuple("_RebuildItem", ("id", "content", "metadata"))   # adapts Chunk to the (.id / .content / .metadata) shape reconcile expects

if TYPE_CHECKING:
    from ..config import AgentmakerConfig
    from ..retrieval.base import Embedder, Reranker
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
                store. If omitted, the original chunks are used directly.
            scope: The default ownership, isolating RAG data from memory; defaults to Scope(base="rag").
            config: Optional chunking knobs (chunk_tokens / overlap_tokens); defaults to ChunkingConfig()
                if omitted. The per-call kwargs of ingest_file / ingest_text can override these
                (three-level resolution).
            index_sync: Optional derived-index sync seam (IndexSync, shared with memory, see
                agentmaker.retrieval.index_sync); ingestion goes through it for fail-loud replacement
                plus reconciliation. Defaults to SyncIndexSync(retriever) if omitted. For async /
                distributed use, implement and inject your own.
            token_counter: Pluggable token counter (defaults to count_tokens); chunking budgets are
                estimated with it, so it is recommended to use the same ruler as the context budget.
        """
        self.retriever = retriever
        self.source_store = source_store
        self.contextualizer = contextualizer
        self.scope = canonical_scope(scope, "rag", "IngestionPipeline construction")
        self.cfg = config or ChunkingConfig()
        self._sync = index_sync if index_sync is not None else SyncIndexSync(retriever)
        self._count = token_counter       # passed through to split_document; chunking estimates budgets with it
        self._mutations = shared_coordinator(source_store)
        self._owns_source_store = False
        self._owns_retriever = False
        self._owns_sync = index_sync is None
        self._closed = False

    def _scope(self, scope: Optional[Scope], action: str) -> Scope:
        """Resolve a per-call scope while preserving the RAG subsystem boundary."""
        return self.scope if scope is None else canonical_scope(scope, "rag", action)

    def _doc_lock(self, scope: Scope, doc_id: str):
        """Serialize a document mutation across pipelines sharing the source store."""
        return self._mutations.hold([(scope, doc_id)])

    @classmethod
    def from_config(cls, config: "AgentmakerConfig", *, embedder: "Optional[Embedder]" = None,
                    retriever: Optional[HybridRetriever] = None,
                    source_store: Optional[SourceStore] = None, db_path: str = ":memory:",
                    reranker: "Optional[Reranker]" = None, contextualizer: Optional["Contextualizer"] = None,
                    index_sync: Optional[IndexSync] = None, scope: Optional[Scope] = None,
                    token_counter: TokenCounter = count_tokens) -> "IngestionPipeline":
        """Assemble an IngestionPipeline from an AgentmakerConfig in one line: defaults to the sqlite backend; pass retriever / source_store to inject a custom backend.

        Must share the same base with RagRetriever (the assembly root is in the app). Typical usage:
        first rag = RagRetriever.from_config(...), then
        IngestionPipeline.from_config(config, retriever=rag.retriever, source_store=rag.source_store),
        so the two read and write the same data.

        Args:
            config: AgentmakerConfig (reads config.chunking).
            embedder: Required when using the default sqlite base; not needed if a retriever is injected.
            retriever: Inject a custom retrieval backend (typically reuse the instance built by
                RagRetriever); if omitted, a default sqlite one is built.
            source_store: Inject a custom source-of-truth store (typically reuse the instance built by
                RagRetriever); if omitted, a default sqlite one is built.
            scope: Fixed ownership scope; per-call scopes may replace its non-base dimensions.
        """
        config.chunking.validate()                            # validate the slice we actually use before dispatch
        owns_retriever = retriever is None
        owns_source_store = source_store is None
        owns_sync = index_sync is None
        unattached_bookkeeping = None
        try:
            if retriever is None:
                if embedder is None:
                    raise ValueError("IngestionPipeline.from_config needs an embedder for the default sqlite base; or pass retriever= to inject a custom backend")
                config.retrieval.validate()
                from ..retrieval.backends import build_sqlite_hybrid   # lazy import: sqlite is the default, do not couple it into this module's top level
                retriever = build_sqlite_hybrid(embedder, db_path=db_path, reranker=reranker, config=config.retrieval)
            if index_sync is None:
                from ..retrieval.index_sync import SqliteBookkeeping, SyncIndexSync
                unattached_bookkeeping = SqliteBookkeeping(db_path)
                index_sync = SyncIndexSync(retriever, bookkeeping=unattached_bookkeeping)
                unattached_bookkeeping = None
            if source_store is None:
                source_store = SourceStore(db_path)
            pipeline = cls(
                retriever,
                source_store,
                contextualizer=contextualizer,
                scope=scope,
                config=config.chunking,
                index_sync=index_sync,
                token_counter=token_counter,
            )
        except BaseException as construction_error:
            seen = set()
            resources = [
                (owns_sync, index_sync),
                (owns_retriever, retriever),
                (owns_source_store, source_store),
            ]
            if unattached_bookkeeping is not None:
                resources.insert(0, (True, unattached_bookkeeping))
            for owned, resource in resources:
                if not owned or resource is None or id(resource) in seen:
                    continue
                seen.add(id(resource))
                try:
                    resource.close()
                except BaseException as cleanup_error:
                    construction_error.add_note(f"IngestionPipeline construction cleanup also failed: {cleanup_error}")
            raise
        pipeline._owns_source_store = owns_source_store
        pipeline._owns_retriever = owns_retriever
        pipeline._owns_sync = owns_sync
        return pipeline

    def ingest_file(self, path: str, *, doc_id: Optional[str] = None,
                    chunk_tokens: Optional[int] = None, overlap_tokens: Optional[int] = None,
                    max_bytes: Optional[int] = None, max_output_chars: Optional[int] = None,
                    max_expanded_bytes: Optional[int] = None,
                    scope: Optional[Scope] = None) -> "IngestReport":
        """Read file -> chunk -> ingest. Returns IngestReport(doc_id / chunks / skipped); skipped=True when the content is unchanged and the run short-circuits. chunk/overlap default to self.cfg if omitted.

        If doc_id is omitted, it is stably derived from the file's absolute path: re-ingesting the same
        file yields the same doc_id, which combined with the content fingerprint means an unchanged file
        short-circuits the whole run (no chunking, no per-chunk LLM enhancement, no embedding); a changed
        file replaces the old version (atomically when the backend provides transactional replacement).
        max_bytes / max_output_chars / max_expanded_bytes raise load_file's corresponding limits for
        legitimately large files (for example wide CSVs whose parsed form expands past the default cap).
        """
        chunk_tokens = self.cfg.chunk_tokens if chunk_tokens is None else chunk_tokens
        overlap_tokens = self.cfg.overlap_tokens if overlap_tokens is None else overlap_tokens
        load_kwargs = {name: value for name, value in (
            ("max_bytes", max_bytes), ("max_output_chars", max_output_chars),
            ("max_expanded_bytes", max_expanded_bytes)) if value is not None}
        doc = load_file(path, **load_kwargs)
        doc.doc_id = doc_id or hashlib.sha1(("file:" + os.path.abspath(path)).encode("utf-8")).hexdigest()
        sc = self._scope(scope, "IngestionPipeline.ingest_file")
        with self._doc_lock(sc, doc.doc_id):
            skipped = self._maybe_skip(doc, chunk_tokens, overlap_tokens, scope=sc)
            if skipped is not None:
                return skipped
            chunks = split_document(doc, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens, token_counter=self._count)
            return self._index(doc, chunks, fingerprint=self._fingerprint(doc, chunk_tokens, overlap_tokens), scope=sc)

    def ingest_text(self, text: str, *, source: str = "", title: Optional[str] = None,
                    doc_id: Optional[str] = None, fmt: str = "txt",
                    chunk_tokens: Optional[int] = None, overlap_tokens: Optional[int] = None,
                    scope: Optional[Scope] = None) -> "IngestReport":
        """Ingest a piece of text directly (no file). If title is omitted it is derived from source (its filename); if doc_id is omitted one is auto-generated. chunk/overlap default to self.cfg if omitted.

        The concept of "re-ingest" only applies when doc_id is passed: unchanged content short-circuits
        the whole run (skipped=True); changed content uses the backend's replacement contract. Returns IngestReport.
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
        sc = self._scope(scope, "IngestionPipeline.ingest_text")
        with self._doc_lock(sc, doc.doc_id):
            if doc_id is not None:
                skipped = self._maybe_skip(doc, chunk_tokens, overlap_tokens, scope=sc)
                if skipped is not None:
                    return skipped
            chunks = split_document(doc, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens, token_counter=self._count)
            fp = self._fingerprint(doc, chunk_tokens, overlap_tokens) if doc_id is not None else None
            return self._index(doc, chunks, fingerprint=fp, scope=sc)

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

    def _maybe_skip(self, doc: Document, chunk_tokens: int, overlap_tokens: int, *,
                    scope: Optional[Scope] = None) -> Optional["IngestReport"]:
        """Re-ingest short-circuit: if the fingerprint matches the last ingestion, skip the whole run (no chunking, no per-chunk LLM enhancement, no embedding) and return IngestReport(skipped=True); otherwise None."""
        sc = self._scope(scope, "IngestionPipeline.skip_check")
        if self.source_store.get_doc_hash(doc.doc_id, scope=sc) != \
                self._fingerprint(doc, chunk_tokens, overlap_tokens):
            return None
        n = len(self.source_store.chunk_ids_of_doc_exact(doc.doc_id, scope=sc))
        return IngestReport(doc_id=doc.doc_id, chunks=n, skipped=True)

    def delete_document(self, doc_id: str, *, scope: Optional[Scope] = None) -> int:
        """Delete a whole document (first the retrieval index, then the source-of-truth store, and clear the ingestion fingerprint); returns the number of chunks deleted."""
        sc = self._scope(scope, "IngestionPipeline.delete_document")
        with self._mutations.hold_all():
            grouped = defaultdict(list)
            for chunk_id, exact_scope in self.source_store.chunk_ids_with_scopes_of_doc(
                    doc_id, scope=sc):
                grouped[exact_scope].append(chunk_id)
            exact_scopes = self.source_store.scopes_of_doc(doc_id, scope=sc)
            if grouped:
                drop_range = getattr(self._sync, "drop_range", None)
                if drop_range is None:
                    self._sync.drop(
                        sorted({id_ for ids in grouped.values() for id_ in ids}), scope=sc)
                else:
                    drop_range(grouped, scope=sc)
            for exact_scope, ids in grouped.items():
                self.source_store.delete_chunks(ids, scope=exact_scope)
            for exact_scope in exact_scopes:
                self.source_store.delete_doc_hash(doc_id, scope=exact_scope)
            return sum(len(ids) for ids in grouped.values())

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
        sc = self._scope(scope, "IngestionPipeline.rebuild_index")
        with self._mutations.hold_all():
            rows = self.source_store.all_chunks_with_scopes(scope=sc)
            grouped = defaultdict(list)
            for chunk, exact_scope in rows:
                grouped[exact_scope].append(_RebuildItem(
                    chunk.chunk_id, chunk.content, {"doc_id": chunk.doc_id, **chunk.metadata}))
            try:
                footprints = set(self._sync.exact_scopes(scope=sc)) | set(grouped)
            except (AttributeError, NotImplementedError):
                footprints = set(grouped)
            if footprints:
                rebuilt = sum(
                    self._sync.reconcile(grouped.get(exact_scope, []), scope=exact_scope,
                                         batch_size=batch_size)
                    for exact_scope in footprints
                )
            else:
                rebuilt = self._sync.reconcile([], scope=sc, batch_size=batch_size)
            if self.contextualizer is not None:
                self.source_store.delete_doc_hashes_in_scope(
                    sorted({chunk.doc_id for chunk, _ in rows}), scope=sc)
            return rebuilt

    def verify(self, *, scope: Optional[Scope] = None) -> dict:
        """Cross-check consistency between the source-of-truth store and the retrieval index: return a divergence report without auto-repairing.

        After a normal ingest, the chunk_id sets of the two are equal. If clearing old chunks fails or a
        crash leaves stale rows in the source of truth (the retrieval replacement completed but source
        cleanup did not), you get "source of truth is a superset of the
        indexed set"; those stale rows would be revived from the source of truth by rebuild_index. This
        method surfaces the divergence (and logs a warning), leaving the app to decide to re-ingest that
        document to fix it (one clean ingest converges it via replace).

        Args:
            scope: Which ownership scope to check; defaults to this pipeline's scope.

        Returns:
            dict: {"scope", "source_only" (ids present in the source of truth but not tracked by the
                   index), "index_only" (ids tracked by the index but absent from the source of truth),
                   "consistent" (True if the two are equal; None if the seam does not support
                   enumeration)}.
        """
        sc = self._scope(scope, "IngestionPipeline.verify")
        rows = self.source_store.all_chunks_with_scopes(scope=sc)
        grouped = defaultdict(set)
        for chunk, exact_scope in rows:
            grouped[exact_scope].add(chunk.chunk_id)
        try:
            source_only = []
            index_only = []
            footprints = set(self._sync.exact_scopes(scope=sc)) | set(grouped)
            if not footprints:
                footprints = {sc}
            for exact_scope in footprints:
                source_ids = grouped.get(exact_scope, set())
                indexed_ids = self._sync.tracked_ids(scope=exact_scope)
                source_only.extend(source_ids - indexed_ids)
                index_only.extend(indexed_ids - source_ids)
        except (AttributeError, NotImplementedError):
            return {"scope": sc, "source_only": [], "index_only": [], "consistent": None}
        source_only = sorted(source_only)
        index_only = sorted(index_only)
        consistent = not source_only and not index_only
        if not consistent:
            logging.getLogger(__name__).warning(
                "RAG index diverges from source of truth (scope=%s): source of truth has %d extra chunks, index has %d extra chunks; consider re-ingesting the affected documents to converge",
                sc, len(source_only), len(index_only))
        return {"scope": sc, "source_only": source_only, "index_only": index_only, "consistent": consistent}

    def stats(self, *, scope: Optional[Scope] = None) -> dict:
        """Return {documents, chunks}: the document count and total chunk count."""
        docs = self.source_store.list_docs(scope=self._scope(scope, "IngestionPipeline.stats"))
        return {"documents": len(docs), "chunks": sum(docs.values())}

    def _index(self, doc: Document, chunks: List[Chunk], *, fingerprint: Optional[str] = None,
               scope: Optional[Scope] = None) -> "IngestReport":
        """Store new source chunks, replace the retrieval batch, and compensate failures.

        The source of truth and the retrieval index live in different SQLite connections and cannot share
        a single cross-database transaction. On the retrieval-index side, the seam `IndexSync.replace`
        delegates to the backend replacement contract. SqliteHybridRetriever performs one transaction,
        so retrieval sees either the old or new batch. Generic backends may expose a short overlap window
        but still raise on failure, handing this method the compensation path.
        On the source-of-truth side, the new chunks are stored first (their ids differ from the old ones,
        so no overwriting), and the old orphan chunks are cleared last (at which point the retrieval index
        does not point at them, they are unsearchable, cleanup failure is harmless and self-heals on
        re-ingest). If any step fails, only this batch of new chunks is rolled back and the old version is
        kept (rather than the document being deleted entirely).
        """
        sc = self._scope(scope, "IngestionPipeline.index")
        new_ids = [c.chunk_id for c in chunks]
        old_ids = self.source_store.chunk_ids_of_doc_exact(doc.doc_id, scope=sc)
        if not chunks:
            self._delete_ids(old_ids, scope=sc)
            self.source_store.delete_doc_hash(doc.doc_id, scope=sc)
            return IngestReport(doc_id=doc.doc_id, chunks=0)
        try:
            # (1) Store into the source of truth (original chunks, the authoritative copy: kept clean, no enhancement context)
            self.source_store.save_chunks(chunks, scope=sc)
            # (2) Push into the retrieval index: with a contextualizer, use the "enhanced text" (improves searchability), otherwise the original text.
            #     The enhanced text only goes into the index, not the source of truth: after a retrieval hit, the source of truth still returns the clean original chunk.
            if self.contextualizer is not None:
                texts = [self.contextualizer.contextualize(c, doc) for c in chunks]
            else:
                texts = [c.content for c in chunks]
            # Per-chunk filterable metadata (doc_id + the chunk's own metadata); the base only stores declared fields (see MetadataFilter in retrieval/types.py) and ignores undeclared ones
            metadatas = [{"doc_id": c.doc_id, **c.metadata} for c in chunks]
            self._sync.replace(old_ids, new_ids, texts, scope=sc, metadatas=metadatas)
        except Exception:
            self._best_effort_delete(new_ids, scope=sc, what="roll back new chunks after ingest failure", level="error")
            raise
        new_set = set(new_ids)
        self._best_effort_delete([cid for cid in old_ids if cid not in new_set], scope=sc,
                                 what="clear old-version orphan chunks", level="warning")
        if fingerprint is not None:
            self.source_store.set_doc_hash(doc.doc_id, fingerprint, scope=sc)
        return IngestReport(doc_id=doc.doc_id, chunks=len(chunks))

    def _delete_ids(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Delete by id from both the retrieval index and the source of truth (index first, then source of truth).

        The index delete goes through the seam `drop` (best-effort: on failure mark for repair, do not
        raise); if the index delete fails and leaves an "in index, not in source of truth" orphan,
        RagRetriever.retrieve filters and self-heals on read (it does not return deleted content). A
        source-of-truth delete failure raises as usual (a real delete failure is a real error).
        """
        if not ids:
            return
        sc = self._scope(scope, "IngestionPipeline.delete_ids")
        drop_exact = getattr(self._sync, "drop_exact", None)
        if drop_exact is None:
            self._sync.drop(ids, scope=sc)
        else:
            drop_exact(ids, scope=sc)
        self.source_store.delete_chunks(ids, scope=sc)

    def close(self) -> None:
        """Close resources created by this ingestion pipeline."""
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        closed_ids: set[int] = set()
        resources = (
            (self._owns_sync, self._sync),
            (self._owns_retriever, self.retriever),
            (self._owns_source_store, self.source_store),
        )
        for owned, resource in resources:
            if not owned or id(resource) in closed_ids:
                continue
            closed_ids.add(id(resource))
            try:
                resource.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, exc, _tb):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"IngestionPipeline cleanup also failed: {cleanup_error}")
        return False

    def _best_effort_delete(self, ids: List[str], *, scope: Optional[Scope] = None,
                            what: str = "cleanup", level: str = "debug") -> None:
        """Compensating / cleanup delete (best-effort: does not re-raise and does not mask the original exception); on failure leave a signal per level.

        level: use "error" for compensation where consistency is compromised (e.g. rolling back new
        chunks after an ingest failure, where the residue is dirty index data); use "debug" for harmless
        cleanup (orphan chunks).
        """
        try:
            self._delete_ids(ids, scope=scope)
        except Exception as e:  # noqa: BLE001
            getattr(logging.getLogger(__name__), level)("%s failed (best-effort): %r", what, e)

    # a* async versions (chunking + Contextualizer + embedding are all sync / network operations, wrapped in to_thread, awaited directly inside the event loop)

    async def aingest_file(self, path: str, **kwargs) -> "IngestReport":
        """Async version of ingest_file (to_thread)."""
        return await asyncio.to_thread(lambda: self.ingest_file(path, **kwargs))

    async def aingest_text(self, text: str, **kwargs) -> "IngestReport":
        """Async version of ingest_text (to_thread)."""
        return await asyncio.to_thread(lambda: self.ingest_text(text, **kwargs))

    async def adelete_document(self, doc_id: str, **kwargs) -> int:
        """Async version of delete_document (to_thread)."""
        return await asyncio.to_thread(lambda: self.delete_document(doc_id, **kwargs))
