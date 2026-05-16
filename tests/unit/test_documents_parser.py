"""Unit tests for the document intake tool."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.tools.documents import (
    DocumentParseError,
    ParsedDocument,
    parse_pdf,
    parse_plain_text,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "uploaded_pdfs"


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_parse_pdf_extracts_text_from_real_8k() -> None:
    """The small Jan 28 2026 MSFT 8-K extracts a non-empty body."""
    raw = _read("0001193125-26-027198.pdf")
    parsed = parse_pdf(raw)
    assert isinstance(parsed, ParsedDocument)
    assert parsed.char_count > 1000
    assert "Microsoft" in parsed.text
    assert parsed.page_count is not None and parsed.page_count >= 1
    assert parsed.content_sha256 == hashlib.sha256(raw).hexdigest()


def test_parse_pdf_rejects_zero_extracted_text() -> None:
    """A PDF whose text extraction yields nothing is treated as scanned-image."""
    empty_pdf = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000055 00000 n \n0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n156\n%%EOF"
    )
    with pytest.raises(DocumentParseError, match=r"scanned|no extractable text"):
        parse_pdf(empty_pdf)


def test_parse_pdf_rejects_wrong_magic_bytes() -> None:
    with pytest.raises(DocumentParseError, match="not a PDF"):
        parse_pdf(b"hello, world")


def test_parse_plain_text_decodes_utf8() -> None:
    raw = b"Microsoft reported revenue of $X."
    parsed = parse_plain_text(raw)
    assert parsed.text == "Microsoft reported revenue of $X."
    assert parsed.char_count == 33
    assert parsed.page_count is None
    assert parsed.content_sha256 == hashlib.sha256(raw).hexdigest()


def test_parse_plain_text_rejects_empty() -> None:
    with pytest.raises(DocumentParseError, match="empty"):
        parse_plain_text(b"   \n\t  ")
