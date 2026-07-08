"""agentmaker.rag.loader: reads files of various formats into a Document.

Different formats are read differently: txt/md are read directly, json/csv are parsed into structures, and
pdf/docx/html are converted to Markdown via MarkItDown. Dispatch is by extension (load_file); the MarkItDown
formats are lazily imported, so not installing it does not affect txt/md/json/csv.
"""

import csv
import json
import os
from abc import ABC, abstractmethod

from ..core.exceptions import RetrievalError
from .types import Document


class DocumentLoader(ABC):
    """Abstract base class for document loaders. Subclasses implement load(): read one file into a Document."""

    @abstractmethod
    def load(self, path: str) -> Document:
        """Read the file and return a Document."""


class TextLoader(DocumentLoader):
    """Plain text / Markdown: read directly as UTF-8."""

    def load(self, path: str) -> Document:
        """Read a txt / md file."""
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return Document(content=content, title=_title(path), source=path, format=_ext(path))


class JsonLoader(DocumentLoader):
    """JSON / JSONL: after parsing, serialize "each record" into a list of texts and pass them via Document.records to RecordSplitter, one record per chunk.

    Array -> one per element; JSONL -> one per line; single object -> one record. Each record is turned back into
    readable text with json.dumps. Records are passed as a list (not separated by blank lines), so a record whose
    body contains blank lines is not mis-split.
    """

    def load(self, path: str) -> Document:
        """Read a json / jsonl file."""
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        records = [json.dumps(r, ensure_ascii=False) for r in self._parse(raw, path)]
        # content serves only as readable body / archive; chunking is driven by the records list (RecordSplitter
        # prefers it and does not reconstruct from blank lines)
        return Document(content="\n\n".join(records), title=_title(path), source=path, format=_ext(path),
                        metadata={"record_count": len(records)}, records=records)

    @staticmethod
    def _parse(raw: str, path: str) -> list:
        """Parse raw text into a record list (array -> elements; JSONL -> lines; object -> single record). On parse failure, raise RetrievalError."""
        try:
            if path.lower().endswith(".jsonl"):
                return [json.loads(line) for line in raw.splitlines() if line.strip()]
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RetrievalError(f"JSON parse failed ({path}): {e}") from e
        return data if isinstance(data, list) else [data]


class CsvLoader(DocumentLoader):
    """CSV: turn each row into "column: value" text and pass it via the Document.records list to RecordSplitter, one row per chunk.

    Records are passed as a list (not separated by blank lines), so multi-line cells (values containing blank
    lines) are not mis-split.
    """

    def load(self, path: str) -> Document:
        """Read a csv file."""
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        records = ["\n".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
        return Document(content="\n\n".join(records), title=_title(path), source=path, format="csv",
                        metadata={"record_count": len(rows)}, records=records)


class MarkItDownLoader(DocumentLoader):
    """PDF / Word / HTML, etc.: convert uniformly to Markdown via Microsoft MarkItDown (lazily imported, optional dependency)."""

    def load(self, path: str) -> Document:
        """Convert pdf / docx / html, etc. to Markdown. Requires installing first: uv add markitdown"""
        try:
            from markitdown import MarkItDown
        except ImportError as e:
            raise RetrievalError("processing PDF/Word/HTML requires installing first: uv add markitdown") from e
        try:
            content = MarkItDown().convert(path).text_content
        except Exception as e:  # noqa: BLE001
            raise RetrievalError(f"MarkItDown conversion failed ({path}): {e}") from e
        # MarkItDown outputs Markdown, so format is recorded as "md" (rather than the original extension pdf/docx):
        # this lets the splitter use the heading-aware MarkdownSplitter and preserve the converted document's
        # heading structure; the original file type is available in source.
        return Document(content=content or "", title=_title(path), source=path, format="md")


def _ext(path: str) -> str:
    """Return the lowercased extension (without the dot), e.g. 'md'."""
    return os.path.splitext(path)[1].lstrip(".").lower()


def _title(path: str) -> str:
    """Return the file name (without directory and extension) as the document title, e.g. '/x/handbook.md' -> 'handbook'."""
    return os.path.splitext(os.path.basename(path))[0]


# extension -> loader instance. Adding a new format only needs one registration line here.
_LOADERS = {
    "txt": TextLoader(), "md": TextLoader(), "markdown": TextLoader(),
    "json": JsonLoader(), "jsonl": JsonLoader(),
    "csv": CsvLoader(),
    "pdf": MarkItDownLoader(), "docx": MarkItDownLoader(), "doc": MarkItDownLoader(),
    "html": MarkItDownLoader(), "htm": MarkItDownLoader(),
}


def load_file(path: str, *, max_bytes: int = 10 * 1024 * 1024) -> Document:
    """Automatically pick a loader by extension to read the file; unknown extensions are treated as plain text.

    All loaders dispatch through here, and the various file read / parse errors (IO, encoding, CSV, JSON, etc.)
    are normalized here to RetrievalError for a consistent external contract; a RetrievalError already raised
    inside a loader is passed through unchanged and not re-wrapped.

    Before reading, set an upper bound on file size (max_bytes, default 10MB): exceeding it fails loud, to avoid
    one oversized file being read entirely into memory and blowing it up.

    Args:
        path: The file path.
        max_bytes: The upper bound on file size (bytes); exceeding it raises RetrievalError.

    Returns:
        Document: The document after reading and normalization.
    """
    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise RetrievalError(f"reading file failed ({path}): {e}") from e
    if size > max_bytes:
        raise RetrievalError(f"file too large ({path}): {size} bytes, exceeds the limit of {max_bytes} bytes")
    loader = _LOADERS.get(_ext(path), TextLoader())
    try:
        return loader.load(path)
    except RetrievalError:
        raise
    except Exception as e:  # noqa: BLE001  normalize the various read / parse exceptions to RetrievalError
        raise RetrievalError(f"reading file failed ({path}): {e}") from e


def register_loader(ext: str, loader: DocumentLoader) -> None:
    """Register / override the loader for an extension (lets users extend custom formats)."""
    _LOADERS[ext.lstrip(".").lower()] = loader
