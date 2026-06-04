"""Tests for the pure, no-API parts of the system."""

from pathlib import Path

import pytest

from rag_core.chunking import chunk_text
from rag_core.extract import (
    SUPPORTED_EXTENSIONS,
    clean_whitespace,
    extract_text,
    is_empty,
)
from rag_core.fusion import reciprocal_rank_fusion
from eval.metrics import hit_rate_at_k, mrr_at_k, recall_at_k


# --- chunking --------------------------------------------------------------

def test_chunking_produces_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, doc_id="d", source="d.txt", chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.doc_id == "d" for c in chunks)
    assert all(c.chunk_id.startswith("d::") for c in chunks)


def test_chunking_rejects_bad_overlap():
    try:
        chunk_text("hello world", doc_id="d", source="d", chunk_size=10, overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- fusion ----------------------------------------------------------------

def test_rrf_rewards_items_ranked_high_in_both_lists():
    a = ["x", "y", "z"]
    b = ["x", "z", "y"]
    fused = reciprocal_rank_fusion([a, b])
    # x is first in both -> should top the fused ranking.
    assert max(fused, key=fused.get) == "x"


# --- metrics ---------------------------------------------------------------

def test_hit_rate():
    assert hit_rate_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], {"z"}, 3) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k(["a", "b", "c"], {"b"}, 3) == 0.5   # rank 2 -> 1/2
    assert mrr_at_k(["a", "b", "c"], {"a"}, 3) == 1.0


def test_recall():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5


# --- whitespace normalization ----------------------------------------------

def test_clean_whitespace_collapses_horizontal_runs():
    # Multiple spaces and tabs within a line collapse to one space.
    assert clean_whitespace("alpha   beta\t\tgamma") == "alpha beta gamma"


def test_clean_whitespace_preserves_paragraph_breaks():
    # A single blank line between paragraphs is intentional and must
    # survive — chunking will treat it as a soft separator later.
    assert (
        clean_whitespace("first paragraph\n\nsecond paragraph")
        == "first paragraph\n\nsecond paragraph"
    )


def test_clean_whitespace_caps_excessive_newlines():
    # PDFs love five blank lines between sections. Cap at two.
    assert (
        clean_whitespace("first\n\n\n\n\nsecond") == "first\n\nsecond"
    )


def test_clean_whitespace_strips_trailing_spaces_per_line():
    assert clean_whitespace("alpha   \nbeta\t \ngamma") == "alpha\nbeta\ngamma"


def test_clean_whitespace_normalizes_crlf():
    # Windows line endings -> plain LF, then the rest of the rules apply.
    assert clean_whitespace("alpha\r\nbeta") == "alpha\nbeta"


def test_clean_whitespace_idempotent_on_clean_input():
    text = "a clean paragraph.\n\nAnother one."
    assert clean_whitespace(text) == text


# --- is_empty (post-extraction sanity check) ------------------------------

def test_is_empty_true_for_short_or_blank():
    assert is_empty("") is True
    assert is_empty("    ") is True
    assert is_empty("short") is True  # under MIN_TEXT_LENGTH (50)


def test_is_empty_false_for_real_content():
    text = "This is a paragraph long enough to clear the empty threshold."
    assert is_empty(text) is False


# --- extract_text dispatch -------------------------------------------------

def test_extract_text_reads_txt(tmp_path: Path):
    path = tmp_path / "hello.txt"
    path.write_text("plain text content")
    assert extract_text(path) == "plain text content"


def test_extract_text_reads_md(tmp_path: Path):
    path = tmp_path / "doc.md"
    path.write_text("# Heading\n\nSome **bold** text.")
    # Whitespace is normalized but markdown markup is preserved as-is —
    # we feed the raw markdown to the embedder; the model handles it.
    assert extract_text(path) == "# Heading\n\nSome **bold** text."


def test_extract_text_applies_whitespace_cleanup_for_plain_files(tmp_path: Path):
    # extract_text runs the same clean_whitespace pass for every format,
    # not just PDFs — defense for weirdly-encoded .txt files.
    path = tmp_path / "messy.txt"
    path.write_text("alpha   beta\n\n\n\ngamma\r\ndelta")
    assert extract_text(path) == "alpha beta\n\ngamma\ndelta"


def test_extract_text_rejects_unsupported_extension(tmp_path: Path):
    path = tmp_path / "doc.docx"
    path.write_bytes(b"PK\x03\x04")  # docx is a zip; bytes don't matter for routing
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(path)


def test_supported_extensions_matches_dispatch():
    # The exported frozenset is the contract callers (e.g. backend
    # /upload, CLI directory walker) rely on. Sanity-check it lists
    # exactly what the dispatch knows how to handle.
    assert SUPPORTED_EXTENSIONS == frozenset({".md", ".txt", ".pdf"})
