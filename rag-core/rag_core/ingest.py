"""
Ingestion: read source files -> chunk -> embed -> store.

Supports .txt and .md out of the box. Run as a script:

    python -m rag_core.ingest data/docs
"""

# `from . import config` triggers .env loading at module import.
# `config.require()` is deliberately NOT called here — this module is
# also imported as a library (e.g. from the backend's /upload route),
# and a library shouldn't sys.exit() its host process at import time.
# The CLI entry point validates the key inside main() instead.
from . import config

import sys
from pathlib import Path

from .chunking import chunk_text
from .embeddings import Embedder
from .vector_store import VectorStore


def ingest_file(path: Path, store: VectorStore, embedder: Embedder) -> int:
    """Chunk a single file, embed every chunk in one batched API call, and
    persist to the store. Returns the number of chunks added (0 if the file
    decoded to empty/whitespace).

    The caller is responsible for filtering by extension; this function
    trusts whatever path it gets. Used by both the CLI loop and the
    backend's /upload route, so the chunk-embed-store sequence stays in
    one place.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    doc_id = path.stem
    chunks = chunk_text(text, doc_id=doc_id, source=path.name)
    if not chunks:
        return 0
    # Batch-embed (one API call per file regardless of chunk count).
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.add(chunks, embeddings)
    return len(chunks)


def ingest_directory(docs_dir: str, store: VectorStore, embedder: Embedder) -> int:
    paths = sorted(Path(docs_dir).glob("**/*"))
    files = [p for p in paths if p.suffix.lower() in {".txt", ".md"}]
    if not files:
        print(f"No .txt/.md files found in {docs_dir}")
        return 0

    total_chunks = 0
    for path in files:
        added = ingest_file(path, store, embedder)
        total_chunks += added
        print(f"  ingested {path.name}: {added} chunks")

    print(f"Done. {total_chunks} chunks from {len(files)} files. "
          f"Collection now holds {store.count()} chunks.")
    return total_chunks


def main():
    # Fail fast with a clear message at the CLI boundary only.
    config.require("OPENAI_API_KEY")
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "data/docs"
    ingest_directory(docs_dir, VectorStore(), Embedder())


if __name__ == "__main__":
    main()
