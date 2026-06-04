"""
Ingestion: read source files -> chunk -> embed -> store.

Supports .txt and .md out of the box. Run as a script:

    python -m src.ingest data/docs
"""

# Import config first so `.env` is loaded before Embedder (which reads
# OPENAI_API_KEY) is constructed.
from . import config
config.require("OPENAI_API_KEY")

import sys
from pathlib import Path

from .chunking import chunk_text
from .embeddings import Embedder
from .vector_store import VectorStore


def ingest_directory(docs_dir: str, store: VectorStore, embedder: Embedder) -> int:
    paths = sorted(Path(docs_dir).glob("**/*"))
    files = [p for p in paths if p.suffix.lower() in {".txt", ".md"}]
    if not files:
        print(f"No .txt/.md files found in {docs_dir}")
        return 0

    total_chunks = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_id = path.stem
        chunks = chunk_text(text, doc_id=doc_id, source=path.name)
        if not chunks:
            continue
        # Batch-embed (one API call per document).
        embeddings = embedder.embed_documents([c.text for c in chunks])
        store.add(chunks, embeddings)
        total_chunks += len(chunks)
        print(f"  ingested {path.name}: {len(chunks)} chunks")

    print(f"Done. {total_chunks} chunks from {len(files)} files. "
          f"Collection now holds {store.count()} chunks.")
    return total_chunks


def main():
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "data/docs"
    ingest_directory(docs_dir, VectorStore(), Embedder())


if __name__ == "__main__":
    main()
