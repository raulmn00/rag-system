"""
Ingestion: read source files -> chunk -> embed -> store.

Supports .txt, .md, and .pdf. Run as a script:

    python -m rag_core.ingest data/docs
"""

# `from . import config` triggers .env loading at module import.
# `config.require()` is deliberately NOT called here — this module is
# also imported as a library (e.g. from the backend's /upload route),
# and a library shouldn't sys.exit() its host process at import time.
# The CLI entry point validates the key inside main() instead.
from . import config

import sys
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_text
from .embeddings import Embedder
from .extract import SUPPORTED_EXTENSIONS, extract_text, is_empty
from .vector_store import VectorStore


@dataclass
class IngestResult:
    """Outcome of ingesting one file.

    Three valid shapes:
    - chunks_added=N, skipped=False, reason=None   → success
    - chunks_added=0, skipped=True,  reason=...    → file recognized
      but produced no usable text (e.g. image-only PDF). The caller
      should report `reason` to the end user instead of silently
      dropping the file.
    - chunks_added=0, skipped=False, reason=None   → never happens
      with the current code paths, but kept representable so the
      dataclass stays neutral.
    """
    chunks_added: int = 0
    skipped: bool = False
    reason: str | None = None


def ingest_file(
    path: Path,
    store: VectorStore,
    embedder: Embedder,
) -> IngestResult:
    """Extract text, chunk, embed, and persist a single file.

    Reads via rag_core.extract.extract_text, which dispatches by
    extension (.md/.txt/.pdf) and applies whitespace normalization.
    If the extracted text is too short to be useful (image-only PDF,
    blank file), returns a skipped IngestResult with a human-readable
    reason instead of writing empty chunks to the store.

    The caller is responsible for filtering by extension; this
    function trusts whatever path it gets. Used by both the CLI loop
    and the backend's /upload route, so the chunk-embed-store
    sequence stays in one place.
    """
    text = extract_text(path)
    if is_empty(text):
        return IngestResult(
            skipped=True,
            reason=(
                "Não foi possível extrair texto deste arquivo. "
                "PDFs escaneados (apenas imagem) precisam de OCR — "
                "esta ingestão lê só a camada de texto."
            ),
        )

    doc_id = path.stem
    chunks = chunk_text(text, doc_id=doc_id, source=path.name)
    if not chunks:
        # Belt-and-suspenders: is_empty above should have already
        # caught this. Surface as a skip rather than silently returning
        # 0 chunks so the caller's UI doesn't look like it succeeded.
        return IngestResult(skipped=True, reason="Chunking produziu zero chunks.")

    # Batch-embed (one API call per file regardless of chunk count).
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.add(chunks, embeddings)
    return IngestResult(chunks_added=len(chunks))


def ingest_directory(
    docs_dir: str,
    store: VectorStore,
    embedder: Embedder,
) -> int:
    """Walk `docs_dir`, ingest every supported file, print a summary.

    The extensions accepted match SUPPORTED_EXTENSIONS exported by
    rag_core.extract — same source of truth the backend /upload route
    uses, so adding a format updates both consumers at once.
    """
    paths = sorted(Path(docs_dir).glob("**/*"))
    files = [p for p in paths if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        suffixes = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"No files with supported extensions ({suffixes}) found in {docs_dir}")
        return 0

    total_chunks = 0
    skipped_count = 0
    for path in files:
        result = ingest_file(path, store, embedder)
        if result.skipped:
            skipped_count += 1
            print(f"  skipped {path.name}: {result.reason}")
        else:
            total_chunks += result.chunks_added
            print(f"  ingested {path.name}: {result.chunks_added} chunks")

    skipped_note = f" ({skipped_count} skipped)" if skipped_count else ""
    print(
        f"Done. {total_chunks} chunks from {len(files) - skipped_count} files"
        f"{skipped_note}. Collection now holds {store.count()} chunks."
    )
    return total_chunks


def main():
    # Fail fast with a clear message at the CLI boundary only.
    config.require("OPENAI_API_KEY")
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "data/docs"
    ingest_directory(docs_dir, VectorStore(), Embedder())


if __name__ == "__main__":
    main()
