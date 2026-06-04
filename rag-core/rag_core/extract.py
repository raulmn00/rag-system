"""Text extraction with per-extension dispatch.

Centralizes the "given a path, give me clean text" step so the ingest
loop doesn't branch on extension every time a new format gets added.
Adding a new format = one new private extractor + one branch in
extract_text. The chunking / embedding / storage stages downstream
stay format-agnostic — they only ever see normalized text.
"""

import re
from pathlib import Path

from pypdf import PdfReader


# Below this many post-cleanup characters the document is treated as
# "effectively empty" — typically a scanned image-only PDF with no
# embedded text layer. The threshold is intentionally generous: a real
# document chunk is usually hundreds of characters minimum, and a
# header-only stub doesn't make a useful RAG source.
MIN_TEXT_LENGTH = 50

# Extensions extract_text() knows how to handle. Kept here so callers
# can share the same source of truth (e.g. the directory walker filters
# by this set before calling extract_text).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt", ".pdf"})


def extract_text(path: Path) -> str:
    """Read the file at `path` and return its plain-text content with
    whitespace normalized. Dispatches by extension.

    Raises ValueError for unsupported extensions — callers should
    filter by SUPPORTED_EXTENSIONS before calling, so reaching this
    path is a programming error rather than expected input.
    """
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        raw = _extract_text_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}")
    return clean_whitespace(raw)


def is_empty(text: str) -> bool:
    """True if the (cleaned) text is too short to be a useful document.

    Used to detect image-only PDFs and other degenerate cases where
    extraction succeeded technically but produced nothing the agent
    could ever cite. The threshold is MIN_TEXT_LENGTH; tune there.
    """
    return len(text.strip()) < MIN_TEXT_LENGTH


def clean_whitespace(text: str) -> str:
    """Normalize whitespace in extracted text without flattening structure.

    PDF text extraction produces a mess: column-wrapped mid-sentence
    line breaks, runs of three or four blank lines between paragraphs,
    trailing spaces. This collapses the noise but preserves real
    paragraph breaks (a single blank line between blocks).

    The same function runs on .md / .txt too — it's a no-op on clean
    input and a useful defense if a file got weird newline encoding.
    """
    # Normalize CR / CRLF to plain LF so the regex below sees a uniform
    # newline character.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of horizontal whitespace within a line.
    text = re.sub(r"[ \t]+", " ", text)
    # Drop trailing horizontal whitespace at end of lines (PDFs love it).
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Cap consecutive newlines at two — keeps paragraph separation, kills
    # the "five blank lines between sections" PDF habit.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_text_pdf(path: Path) -> str:
    """Concatenate the text layer of every page in a PDF.

    pypdf does NOT do OCR — scanned image PDFs return an empty string
    here, which extract_text + is_empty turn into the "skipped" path
    downstream. Pages are separated by a blank line so clean_whitespace
    treats them as paragraph boundaries.
    """
    reader = PdfReader(str(path))
    pages = (page.extract_text() or "" for page in reader.pages)
    return "\n\n".join(pages)
