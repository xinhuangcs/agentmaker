"""agentmaker.rag: retrieval-augmented generation (RAG) subsystem, built on top of the agentmaker.retrieval base.

Reads files of various formats into a Document, splits them into Chunks, ingests them (source-of-truth store + retrieval index), performs retrieval-based question answering, and optionally applies Contextual Retrieval enhancement.
    - Document / Chunk: source document and text chunk
    - DocumentLoader / load_file: extension-dispatched file loading (txt/md/json/csv/pdf/docx/html)
    - Splitter / split_document: format-dispatched chunking (Markdown heading-aware / structured by record / plain text by token)
    - SourceStore / IngestionPipeline: chunk source-of-truth store and ingestion orchestrator (doc_id upsert dedup)
    - RagRetriever / RAGTool: retrieval-based question answering and agentic RAG tool
    - Contextualizer: adds context to chunks before ingestion (used for retrieval only)
    - QueryTransformer: query expansion before retrieval (MQE / HyDE, opt-in, off by default)
    - ChunkExpander / NeighborWindowExpander: post-retrieval chunk expansion (small-to-big, opt-in; lives in retriever.py alongside QueryTransformer)
"""

from .contextualizer import Contextualizer, DEFAULT_CONTEXT_PROMPT, HeadingContextualizer, LLMContextualizer
from .types import AskResult, ChunkingConfig, Chunk, Document, IngestReport, RagConfig, SourceRef
from .ingest import IngestionPipeline
from .loader import DocumentLoader, load_file, register_loader
from .rag_tool import RAGTool
from .retriever import (DEFAULT_ASK_PROMPT, DEFAULT_HYDE_PROMPT, DEFAULT_MQE_PROMPT,
                        ChunkExpander, HyDETransformer, MultiQueryExpander, NeighborWindowExpander,
                        QueryTransformer, RagRetriever)
from .source_store import SourceStore
from .splitter import Splitter, split_document

__all__ = ["Document", "Chunk", "ChunkingConfig", "RagConfig", "IngestReport", "AskResult", "SourceRef",
           "DocumentLoader", "load_file", "register_loader",
           "Splitter", "split_document", "SourceStore", "IngestionPipeline",
           "RagRetriever", "RAGTool",
           "Contextualizer", "HeadingContextualizer", "LLMContextualizer",
           "ChunkExpander", "NeighborWindowExpander",
           "QueryTransformer", "MultiQueryExpander", "HyDETransformer",
           "DEFAULT_ASK_PROMPT", "DEFAULT_CONTEXT_PROMPT", "DEFAULT_MQE_PROMPT", "DEFAULT_HYDE_PROMPT"]
