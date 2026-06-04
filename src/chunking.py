"""
Document chunking.

Chunking is the single most underrated lever in a RAG system: chunks too large
dilute the embedding signal and waste context; too small lose the surrounding
meaning. We use token-aware fixed-size chunks with overlap, which is a robust
default. The overlap keeps a sentence that straddles a boundary retrievable from
both chunks.

Trade-offs worth stating in an interview:
- Fixed-size + overlap: simple, predictable, works on any text. Can cut
  mid-thought.
- Structure-aware (by heading/paragraph): preserves meaning but needs clean
  structure in the source.
We default to the first and expose the knobs.
"""

from dataclasses import dataclass

import tiktoken

try:
    ENCODER = tiktoken.get_encoding("cl100k_base")  # matches OpenAI embedding models
    _HAS_ENCODER = True
except Exception:  # offline / vocab download blocked — fall back to word tokens
    ENCODER = None
    _HAS_ENCODER = False


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source: str  # human-readable origin (filename, title)


def count_tokens(text: str) -> int:
    if _HAS_ENCODER:
        return len(ENCODER.encode(text))
    return len(text.split())


def _encode(text: str) -> list:
    return ENCODER.encode(text) if _HAS_ENCODER else text.split()


def _decode(tokens: list) -> str:
    return ENCODER.decode(tokens) if _HAS_ENCODER else " ".join(tokens)


def chunk_text(
    text: str,
    doc_id: str,
    source: str,
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[Chunk]:
    """Split `text` into overlapping token-windows of ~chunk_size tokens."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = _encode(text)
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunk_str = _decode(window).strip()
        if chunk_str:
            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::{idx}",
                text=chunk_str,
                source=source,
            ))
            idx += 1
        start += step

    return chunks
