"""Upload-document parser.

Handles the two content types accepted by ``POST /api/upload``:

* ``application/pdf`` -- decoded with ``pypdf``. PDFs whose pages contain no
  embedded text (typical of scanned images) are rejected with a clear error;
  OCR is intentionally out of scope.
* ``text/plain`` -- decoded as UTF-8. Whitespace-only payloads are rejected.

Every successful parse returns a :class:`ParsedDocument` carrying the
extracted text, character count, page count (PDFs only), and the SHA-256
of the raw bytes. The hash dedupes re-uploads of identical content via
:meth:`Repository.get_uploaded_document_by_sha256`.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Final

from pypdf import PdfReader

_PDF_MAGIC: Final[bytes] = b"%PDF-"


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be parsed.

    The message is user-facing -- it goes back through the API to the
    uploader, so it must be specific and actionable.
    """


@dataclass(frozen=True)
class ParsedDocument:
    """Normalised view of an uploaded document."""

    text: str
    char_count: int
    page_count: int | None
    content_sha256: str


def parse_pdf(raw: bytes) -> ParsedDocument:
    """Extract text from a PDF byte string.

    Raises :class:`DocumentParseError` if the bytes do not begin with the
    ``%PDF-`` magic header, or if the extracted text is empty (a strong
    signal the PDF is a scan and would require OCR).
    """
    if not raw.startswith(_PDF_MAGIC):
        raise DocumentParseError(
            "Uploaded file is not a PDF (missing %PDF- magic bytes)."
        )
    reader = PdfReader(io.BytesIO(raw))
    page_count = len(reader.pages)
    chunks = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(chunk for chunk in chunks if chunk).strip()
    if not text:
        raise DocumentParseError(
            "This PDF has no extractable text -- it looks like a scanned image. "
            "Paste the text directly or supply a text-extractable PDF."
        )
    return ParsedDocument(
        text=text,
        char_count=len(text),
        page_count=page_count,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def parse_plain_text(raw: bytes, *, encoding: str = "utf-8") -> ParsedDocument:
    """Decode a plain-text upload.

    Raises :class:`DocumentParseError` if decoding fails or the decoded body
    is empty after stripping whitespace.
    """
    try:
        decoded = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise DocumentParseError(
            f"Plain-text upload is not valid {encoding}: {exc!s}."
        ) from exc
    text = decoded.strip()
    if not text:
        raise DocumentParseError("Plain-text upload is empty.")
    return ParsedDocument(
        text=text,
        char_count=len(text),
        page_count=None,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )
