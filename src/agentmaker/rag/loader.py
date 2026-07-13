"""agentmaker.rag.loader: reads files of various formats into a Document.

Different formats are read differently: txt/md are read directly, json/csv are parsed into structures, and
pdf/docx/html are converted to Markdown via MarkItDown. Dispatch is by extension (load_file); the MarkItDown
formats are lazily imported, so not installing it does not affect txt/md/json/csv.
"""

import csv
import io
import json
import os
import stat
import tempfile
import zipfile
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
            return self.from_text(f.read(), path)

    @staticmethod
    def from_text(content: str, path: str) -> Document:
        """Build a text document from a bounded snapshot."""
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
            return self.from_text(f.read(), path)

    def from_text(self, raw: str, path: str) -> Document:
        """Build a structured document from a bounded JSON snapshot."""
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
            return self.from_text(f.read(), path)

    @staticmethod
    def from_text(raw: str, path: str) -> Document:
        """Build a structured document from a bounded CSV snapshot."""
        rows = list(csv.DictReader(io.StringIO(raw)))
        records = ["\n".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
        return Document(content="\n\n".join(records), title=_title(path), source=path, format="csv",
                        metadata={"record_count": len(rows)}, records=records)


class MarkItDownLoader(DocumentLoader):
    """PDF / Word / HTML, etc.: convert uniformly to Markdown via Microsoft MarkItDown (lazily imported, optional dependency)."""

    def load(self, path: str) -> Document:
        """Convert PDF, DOCX, or HTML to Markdown with the RAG extra installed."""
        try:
            from markitdown import MarkItDown, MissingDependencyException
        except ImportError as e:
            raise RetrievalError(
                "processing PDF/DOCX/HTML requires installing the RAG extra: "
                "uv add 'agentmaker[rag]'") from e
        try:
            content = MarkItDown().convert(path).text_content
        except MissingDependencyException as e:
            raise RetrievalError(
                "processing PDF/DOCX/HTML requires installing the RAG extra: "
                "uv add 'agentmaker[rag]'") from e
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


_TEXT_LOADER = TextLoader()
_JSON_LOADER = JsonLoader()
_CSV_LOADER = CsvLoader()
_MARKDOWN_LOADER = MarkItDownLoader()

# extension -> loader instance. Adding a new format only needs one registration line here.
_LOADERS = {
    "txt": _TEXT_LOADER, "md": _TEXT_LOADER, "markdown": _TEXT_LOADER,
    "json": _JSON_LOADER, "jsonl": _JSON_LOADER,
    "csv": _CSV_LOADER,
    "pdf": _MARKDOWN_LOADER, "docx": _MARKDOWN_LOADER,
    "html": _MARKDOWN_LOADER, "htm": _MARKDOWN_LOADER,
}


def _read_snapshot(path: str, max_bytes: int) -> bytes:
    """Read one regular file through a single descriptor with a hard byte bound."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise RetrievalError(f"reading file failed ({path}): {e}") from e
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RetrievalError(f"reading file failed ({path}): path is not a regular file")
        if info.st_size > max_bytes:
            raise RetrievalError(
                f"file too large ({path}): {info.st_size} bytes, exceeds the limit of {max_bytes} bytes")
        chunks = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > max_bytes:
            raise RetrievalError(
                f"file too large ({path}): exceeds the limit of {max_bytes} bytes while reading")
        return b"".join(chunks)
    except OSError as e:
        raise RetrievalError(f"reading file failed ({path}): {e}") from e
    finally:
        os.close(fd)


def _validate_docx_snapshot(data: bytes, path: str, max_expanded_bytes: int) -> None:
    """Reject malformed or excessively expanded DOCX containers before conversion."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            expanded = 0
            for info in archive.infolist():
                expanded += info.file_size
                if expanded > max_expanded_bytes:
                    raise RetrievalError(
                        f"DOCX expanded data exceeds {max_expanded_bytes} bytes ({path})")
                if info.file_size and info.compress_size == 0:
                    raise RetrievalError(f"DOCX contains an invalid compressed entry ({path})")
                if info.compress_size and info.file_size / info.compress_size > 1000:
                    raise RetrievalError(f"DOCX compression ratio is unsafe ({path})")
    except zipfile.BadZipFile as e:
        raise RetrievalError(f"DOCX container is invalid ({path}): {e}") from e


def _load_snapshot(loader: DocumentLoader, data: bytes, path: str) -> Document:
    """Dispatch a bounded snapshot to a built-in or registered loader."""
    try:
        text = data.decode("utf-8") if loader in (_TEXT_LOADER, _JSON_LOADER, _CSV_LOADER) else None
        if loader is _TEXT_LOADER:
            return _TEXT_LOADER.from_text(text or "", path)
        if loader is _JSON_LOADER:
            return _JSON_LOADER.from_text(text or "", path)
        if loader is _CSV_LOADER:
            return _CSV_LOADER.from_text(text or "", path)
        suffix = os.path.splitext(path)[1]
        snapshot_fd, snapshot_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(snapshot_fd, "wb") as snapshot:
                snapshot_fd = -1
                snapshot.write(data)
            try:
                document = loader.load(snapshot_path)
            except RetrievalError as e:
                # Converter errors name the temp snapshot; report the user's file instead.
                raise RetrievalError(str(e).replace(snapshot_path, path)) from e
        finally:
            if snapshot_fd >= 0:
                os.close(snapshot_fd)
            try:
                os.unlink(snapshot_path)
            except FileNotFoundError:
                pass
        document.source = path
        if not document.title or document.title == _title(snapshot_path):
            document.title = _title(path)
        return document
    except RetrievalError:
        raise
    except Exception as e:  # noqa: BLE001
        raise RetrievalError(f"reading file failed ({path}): {e}") from e


def load_file(path: str, *, max_bytes: int = 10 * 1024 * 1024,
              max_output_chars: int = 10_000_000,
              max_expanded_bytes: int = 100 * 1024 * 1024) -> Document:
    """Automatically pick a loader by extension to read the file; unknown extensions are treated as plain text.

    All loaders dispatch through here, and the various file read / parse errors (IO, encoding, CSV, JSON, etc.)
    are normalized here to RetrievalError for a consistent external contract; a RetrievalError already raised
    inside a loader is passed through, except that a converter error naming its internal temp snapshot has that
    path rewritten back to the original file.

    Before reading, set an upper bound on file size (max_bytes, default 10MB): exceeding it fails loud, to avoid
    one oversized file being read entirely into memory and blowing it up.

    Args:
        path: The file path.
        max_bytes: The upper bound on file size (bytes); exceeding it raises RetrievalError.
        max_output_chars: Maximum characters retained after parsing or conversion.
        max_expanded_bytes: Maximum total uncompressed size accepted from a DOCX container.

    Returns:
        Document: The document after reading and normalization.
    """
    for name, value in (("max_bytes", max_bytes), ("max_output_chars", max_output_chars),
                        ("max_expanded_bytes", max_expanded_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    extension = _ext(path)
    loader = _LOADERS.get(extension)
    if loader is None:
        if extension == "doc":
            raise RetrievalError("the .doc format is not supported; convert the file to .docx")
        loader = _TEXT_LOADER
    data = _read_snapshot(path, max_bytes)
    if loader is _MARKDOWN_LOADER and data[:4] == b"PK\x03\x04":
        # MarkItDown routes by sniffed content, so every zip container it may expand is validated.
        _validate_docx_snapshot(data, path, max_expanded_bytes)
    document = _load_snapshot(loader, data, path)
    if len(document.content) > max_output_chars:
        raise RetrievalError(
            f"loaded document exceeds {max_output_chars} characters ({path})")
    return document


def register_loader(ext: str, loader: DocumentLoader) -> None:
    """Register / override the loader for an extension (lets users extend custom formats)."""
    _LOADERS[ext.lstrip(".").lower()] = loader
