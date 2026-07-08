"""agentmaker.rag.splitter: splits a Document into Chunks.

Different formats are split differently: Markdown is split by heading level (preserving the heading path),
structured data (json/csv) is split by record, and other plain text is split by token + overlap. Dispatch is by
doc.format (split_document). Chunking exists so retrieval can pinpoint precisely and vectors can express meaning
accurately; a short document that does not exceed chunk_tokens stays as one whole chunk and is not split.
"""

import re
from abc import ABC, abstractmethod
from typing import List

from ..core.text import TokenCounter, count_tokens
from .types import Chunk, Document

# Target token count and overlap token count for a chunk (2026 baseline-optimal defaults; adjustable in split_document)
_CHUNK_TOKENS = 512
_OVERLAP_TOKENS = 64

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")  # Markdown heading line
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*(.*)$")  # code fence delimiter line (3+ backticks / tildes, optional language label)


class Splitter(ABC):
    """Abstract base class for splitters. Subclasses implement split(): split a Document into a number of Chunks."""

    def __init__(self, chunk_tokens: int = _CHUNK_TOKENS, overlap_tokens: int = _OVERLAP_TOKENS,
                 token_counter: TokenCounter = count_tokens):
        # Parameter validation: an invalid config (e.g. overlap >= chunk) would produce over-budget chunks,
        # so reject it up front rather than silently emitting bad chunks.
        if chunk_tokens <= 0:
            raise ValueError(f"chunk_tokens must be a positive integer, got {chunk_tokens}")
        if not 0 <= overlap_tokens < chunk_tokens:
            raise ValueError(
                f"overlap_tokens must satisfy 0 <= overlap < chunk_tokens, got overlap={overlap_tokens}, chunk={chunk_tokens}")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self._count = token_counter       # pluggable token counter (default count_tokens); the chunking budget is estimated with it

    @abstractmethod
    def split(self, doc: Document) -> List[Chunk]:
        """Split the document into a list of Chunks."""

    def _pack(self, units: List[str], doc: Document, *, heading_path: str = "",
              start_index: int = 0) -> List[Chunk]:
        """Pack a number of "text units" into chunks by token budget (split only when over-long, adjacent chunks overlap by overlap_tokens).

        Args:
            units: The smallest units already split semantically (paragraphs / sentences).
            doc: The source document (used for doc_id / source / format).
            heading_path: The heading path these units belong to.
            start_index: The starting chunk index.

        Returns:
            List[Chunk]: The packed chunks.
        """
        # Preprocess: first break "large units" that already exceed chunk_tokens into smaller units,
        # otherwise such a unit would occupy a whole chunk and far exceed the budget (chunking fails).
        units = self._ensure_units(units)
        chunks, buf, buf_tok = [], [], 0
        idx = start_index
        for unit in units:
            ut = self._count(unit)
            if buf and buf_tok + ut > self.chunk_tokens:
                chunks.append(self._make(buf, doc, heading_path, idx))
                idx += 1
                # Overlap: keep a few units from the tail, then continue building the next chunk
                buf, buf_tok = self._tail_overlap(buf)
            buf.append(unit)
            buf_tok += ut
        if buf:
            chunks.append(self._make(buf, doc, heading_path, idx))
        return chunks

    def _ensure_units(self, units: List[str]) -> List[str]:
        """Break large units exceeding chunk_tokens into smaller ones: first split by sentence (Chinese and English punctuation), and if still over-long, hard-split by character."""
        out = []
        for unit in units:
            if self._count(unit) <= self.chunk_tokens:
                out.append(unit)
                continue
            sentences = re.split(r"(?<=[。！？.!?\n])", unit)  # split at sentence-ending punctuation, keeping the punctuation
            for sent in sentences:
                if not sent:
                    continue
                if self._count(sent) <= self.chunk_tokens:
                    out.append(sent)
                else:
                    out.extend(self._char_split(sent))  # a single sentence still over-long (a large span with no punctuation) -> hard-split by character
        return out

    def _char_split(self, text: str) -> List[str]:
        """Hard-split text by character into pieces each not exceeding chunk_tokens (the final fallback, guaranteeing the limit is not exceeded)."""
        step = max(self.chunk_tokens, 1)
        return [text[i:i + step] for i in range(0, len(text), step)]

    def _tail_overlap(self, buf: List[str]) -> tuple:
        """Take the units at the tail of buf whose cumulative size does not exceed overlap_tokens, as the start of the next chunk (to keep context continuous).

        Note: only keep the tail units whose "cumulative size does not exceed overlap_tokens"; if the last unit
        itself exceeds overlap_tokens, do not overlap (return empty), to avoid stuffing a large unit into the next
        chunk and causing the next chunk to exceed chunk_tokens.
        """
        kept, tok = [], 0
        for unit in reversed(buf):
            t = self._count(unit)
            if tok + t > self.overlap_tokens:
                break
            kept.insert(0, unit)
            tok += t
        return kept, tok

    def _make(self, units: List[str], doc: Document, heading_path: str, index: int) -> Chunk:
        """Join a batch of units into one Chunk; doc.title (the file name) is used as the leading prefix of the heading path so "which file it belongs to" is retrievable."""
        full_path = " > ".join(p for p in (doc.title, heading_path) if p)  # file name > in-document heading
        # chunk metadata = document-level metadata (author / upload time, etc.) plus source / format; the latter overrides the former on name clashes
        metadata = {**doc.metadata, "source": doc.source, "format": doc.format}
        return Chunk(content="\n".join(units).strip(), doc_id=doc.doc_id, heading_path=full_path,
                     index=index, metadata=metadata)


class TextSplitter(Splitter):
    """Plain text: split into paragraphs by blank lines, then pack by token budget + overlap."""

    def split(self, doc: Document) -> List[Chunk]:
        """Chunk by paragraph + token."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.content) if p.strip()]
        return self._pack(paragraphs, doc)


class MarkdownSplitter(Splitter):
    """Markdown: split into sections by heading level (preserving the heading path), then pack each section internally by token."""

    def split(self, doc: Document) -> List[Chunk]:
        """Split at #/##/### headings, recording heading_path; over-long sections are further split by token."""
        sections = self._split_by_heading(doc.content)   # [(heading_path, body), ...]
        chunks, idx = [], 0
        for heading_path, body in sections:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            if not paragraphs:
                continue
            packed = self._pack(paragraphs, doc, heading_path=heading_path, start_index=idx)
            chunks.extend(packed)
            idx += len(packed)
        return chunks

    @staticmethod
    def _split_by_heading(text: str) -> List[tuple]:
        """Split at heading lines, maintaining a heading stack to generate each section's heading path, e.g. 'Chapter 1 > 1.1'.

        Sections with body text are emitted normally. A "heading-only section" (no body under the heading) is
        kept only when it is a leaf heading (no deeper subheading immediately follows): its own heading is used as
        the body, to avoid losing leaf-heading information; a "parent heading with subsections" is not emitted on
        its own (its content lives in its subsections), to avoid noise chunks.

        Lines inside a code fence (``` / ~~~) are always treated as body: a line starting with `#` inside a fence
        is actually a code comment, not a Markdown heading; otherwise it would be mistaken for a heading, splitting
        it out, dropping content and scrambling levels, and any document containing code would be split badly. The
        fence stays open until a matching closing delimiter is met (same character, length >= the opener, no
        trailing info after the delimiter); an unclosed fence extends to end of text per CommonMark.
        """
        sections, stack, body = [], [], []
        fence = None  # None or (character, length): whether currently inside a code fence

        def flush(emit_empty_leaf: bool):
            path = " > ".join(h for _, h in stack)
            text_body = "\n".join(body).strip()
            if text_body:
                sections.append((path, text_body))
            elif path and emit_empty_leaf:  # leaf heading-only section: kept even without body, using the heading as body
                sections.append((path, stack[-1][1]))

        for line in text.splitlines():
            fence_match = _FENCE_RE.match(line)
            if fence is not None:
                # Inside a fence: everything counts as body; exit the fence only on a matching closing delimiter
                if (fence_match and fence_match.group(1)[0] == fence[0]
                        and len(fence_match.group(1)) >= fence[1] and not fence_match.group(2).strip()):
                    fence = None
                body.append(line)
                continue
            if fence_match:
                # Fence opening line: enter the fence, this line counts as body and is not parsed as a heading
                fence = (fence_match.group(1)[0], len(fence_match.group(1)))
                body.append(line)
                continue
            m = _HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                # A new heading deeper than the stack top -> the stack top is a "parent heading with subsections", its empty body is not emitted
                parent_has_child = bool(stack) and level > stack[-1][0]
                flush(emit_empty_leaf=not parent_has_child)
                body.clear()
                while stack and stack[-1][0] >= level:  # pop headings at the same or deeper level
                    stack.pop()
                stack.append((level, m.group(2).strip()))
            else:
                body.append(line)
        flush(emit_empty_leaf=True)  # the last section at end of text: if heading-only, keep it as a leaf
        return sections


class RecordSplitter(Splitter):
    """Structured (json/csv): one chunk per record; oversized records are further split by token.

    Prefers the record list doc.records passed directly by the loader (structured data does not round-trip through
    "blank-line separated" text, which avoids mis-splitting records whose body contains blank lines / CSV
    multi-line cells); only when doc.records is None (e.g. a hand-constructed Document) does it fall back to
    splitting content on blank lines.
    """

    def split(self, doc: Document) -> List[Chunk]:
        """One chunk per record; a single record exceeding chunk_tokens is split into multiple chunks (to avoid oversized chunks)."""
        source = doc.records if doc.records is not None else re.split(r"\n\s*\n", doc.content)
        records = [r.strip() for r in source if r.strip()]
        chunks, idx = [], 0
        for record in records:
            for unit in self._ensure_units([record]):  # normal records as-is; oversized records broken down
                chunks.append(self._make([unit], doc, "", idx))
                idx += 1
        return chunks


def split_document(doc: Document, *, chunk_tokens: int = _CHUNK_TOKENS,
                   overlap_tokens: int = _OVERLAP_TOKENS,
                   token_counter: TokenCounter = count_tokens) -> List[Chunk]:
    """Automatically pick a splitter by doc.format and chunk.

    Args:
        doc: The source document.
        chunk_tokens: Target token count per chunk.
        overlap_tokens: Overlap token count between adjacent chunks.
        token_counter: Pluggable token counter (default count_tokens); the chunking budget is estimated with it,
            and using the same ruler as the context budget is more accurate.

    Returns:
        List[Chunk]: The resulting chunks; an empty document returns an empty list.
    """
    if not doc.content.strip():
        return []
    fmt = doc.format.lower()
    if fmt in ("md", "markdown"):
        splitter: Splitter = MarkdownSplitter(chunk_tokens, overlap_tokens, token_counter)
    elif fmt in ("json", "jsonl", "csv"):
        splitter = RecordSplitter(chunk_tokens, overlap_tokens, token_counter)
    else:
        splitter = TextSplitter(chunk_tokens, overlap_tokens, token_counter)
    return splitter.split(doc)


