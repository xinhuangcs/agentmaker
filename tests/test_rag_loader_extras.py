import zipfile
import os

import pytest

from agentmaker.core.exceptions import RetrievalError
from agentmaker.rag.loader import load_file


def test_doc_format_is_rejected_explicitly(tmp_path):
    path = tmp_path / "sample.doc"
    path.write_bytes(b"not a docx container")
    with pytest.raises(RetrievalError, match="not supported.*docx"):
        load_file(str(path))


@pytest.mark.parametrize("kwargs", [
    {"max_bytes": 0},
    {"max_output_chars": -1},
    {"max_expanded_bytes": "100"},
])
def test_loader_rejects_invalid_bounds(tmp_path, kwargs):
    path = tmp_path / "sample.txt"
    path.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(str(path), **kwargs)


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO behavior")
def test_loader_rejects_fifo_without_blocking(tmp_path):
    path = tmp_path / "pipe.txt"
    os.mkfifo(path)
    with pytest.raises(RetrievalError, match="regular file"):
        load_file(str(path))


def test_loader_bounds_parsed_output(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("abcdef", encoding="utf-8")
    with pytest.raises(RetrievalError, match="loaded document"):
        load_file(str(path), max_output_chars=5)


def test_docx_loader_smoke(tmp_path):
    pytest.importorskip("markitdown")
    pytest.importorskip("mammoth")
    path = tmp_path / "sample.docx"
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    relationships = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>Hello DOCX</w:t></w:r></w:p><w:sectPr/></w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)

    loaded = load_file(str(path))
    assert loaded.format == "md"
    assert "Hello DOCX" in loaded.content


def test_pdf_loader_smoke(tmp_path):
    pytest.importorskip("markitdown")
    pdfium = pytest.importorskip("pypdfium2")
    path = tmp_path / "sample.pdf"
    document = pdfium.PdfDocument.new()
    document.new_page(612, 792)
    document.save(path)
    document.close()

    loaded = load_file(str(path))
    assert loaded.format == "md"
    assert loaded.source == str(path)


def test_zip_container_guard_applies_regardless_of_extension(tmp_path):
    """A zip container headed for the converter is validated by content, so renaming a DOCX bomb to .html cannot skip the expansion checks."""
    bomb = tmp_path / "bomb.html"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"0" * 200_000)
    with pytest.raises(RetrievalError, match="expanded data exceeds"):
        load_file(str(bomb), max_expanded_bytes=100_000)


def test_register_loader_for_doc_extension_is_honored(tmp_path):
    """A custom loader registered for .doc runs; without one, .doc still fails loud."""
    from agentmaker.rag.loader import Document, _LOADERS, register_loader

    path = tmp_path / "old.doc"
    path.write_text("legacy", encoding="utf-8")
    with pytest.raises(RetrievalError, match="convert the file to .docx"):
        load_file(str(path))

    class DocLoader:
        def load(self, snapshot_path):
            return Document(content="converted", title="old", source=snapshot_path, format="txt")

    register_loader("doc", DocLoader())
    try:
        loaded = load_file(str(path))
        assert loaded.content == "converted" and loaded.source == str(path)
    finally:
        _LOADERS.pop("doc", None)


def test_converter_errors_name_the_original_file(tmp_path):
    """A converter failure reports the user's path, not the mkstemp snapshot."""
    from agentmaker.rag.loader import register_loader, _LOADERS

    class BoomLoader:
        def load(self, snapshot_path):
            raise RetrievalError(f"conversion failed ({snapshot_path}): boom")

    path = tmp_path / "report.custom"
    path.write_text("x", encoding="utf-8")
    register_loader("custom", BoomLoader())
    try:
        with pytest.raises(RetrievalError, match="report.custom"):
            load_file(str(path))
    finally:
        _LOADERS.pop("custom", None)


def test_text_starting_with_pk_letters_is_not_mistaken_for_zip(tmp_path):
    """Only the full zip local-file-header magic triggers container validation; text that merely starts with "PK" loads normally."""
    pytest.importorskip("markitdown")
    path = tmp_path / "note.html"
    path.write_text("PK Chopra homepage bio", encoding="utf-8")
    assert "PK Chopra" in load_file(str(path)).content
