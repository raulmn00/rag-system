"""POST /upload — runtime ingestion of .md, .txt, and .pdf files via multipart."""

import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag_core.embeddings import Embedder
from rag_core.extract import SUPPORTED_EXTENSIONS
from rag_core.ingest import ingest_file
from rag_core.vector_store import VectorStore


router = APIRouter()

# Per-file size cap. PDFs are heavier than plain text, so the default
# (15 MiB) is bigger than the 5 MiB that covered .md/.txt-only. Tune
# at deploy time via the RAG_MAX_UPLOAD_MB env var without redeploying
# code — useful for tightening down in a public demo or loosening for
# trusted users.
MAX_BYTES_PER_FILE: int = int(os.environ.get("RAG_MAX_UPLOAD_MB", "15")) * 1024 * 1024

UPLOAD_DIR = Path("data/uploads")  # resolved against cwd (run uvicorn from rag-core/)


class FileIngested(BaseModel):
    filename: str
    chunks: int


class SkippedFile(BaseModel):
    filename: str
    reason: str


class UploadResponse(BaseModel):
    files: list[FileIngested]
    skipped: list[SkippedFile]
    total_chunks_in_collection: int


def _safe_filename(raw: str) -> str:
    """Strip path components and reject anything that smells like a traversal.

    A FastAPI UploadFile reports whatever filename the client sent, so it
    has to be treated as untrusted input. We never honor a directory part,
    and we refuse hidden / empty names so an upload can't end up outside
    UPLOAD_DIR or silently overwrite a dotfile.
    """
    if not raw:
        raise HTTPException(status_code=400, detail="Filename missing.")
    base = os.path.basename(raw)  # handles both / and \
    if not base or base.startswith(".") or "/" in base or "\\" in base:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {raw!r}")
    return base


def _format_extension_error(filename: str) -> str:
    pretty = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return (
        f"Tipo de arquivo não suportado: {filename!r}. "
        f"Apenas {pretty} são aceitos."
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
    """Receive one or more .md/.txt/.pdf files, persist them under
    data/uploads/, and ingest them into the vector store.

    Two outcomes per file:
    - Ingested successfully → listed in `files` with the chunk count.
    - Recognized but produced no extractable text (e.g. scanned PDF
      with only an image layer) → listed in `skipped` with a human-
      readable reason. The request as a whole still returns 200; the
      caller can show both lists.

    Validation up-front (extension, filename, size) still fails the
    whole request with 4xx if anything is wrong — partial-state retries
    are worse than a clean redo when the input itself is invalid.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo recebido.")

    # First pass: validate extension + sanitize names. Nothing on disk yet.
    # SUPPORTED_EXTENSIONS comes from rag_core.extract so backend and
    # domain layer can't drift apart on what's accepted.
    safe_names: list[str] = []
    for upload_file in files:
        suffix = Path(upload_file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=_format_extension_error(upload_file.filename or ""),
            )
        safe_names.append(_safe_filename(upload_file.filename or ""))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Constructing these per request matches the /ask handler's pattern.
    # A real prod deploy would inject them via FastAPI dependencies.
    store = VectorStore()
    embedder = Embedder()

    ingested: list[FileIngested] = []
    skipped: list[SkippedFile] = []
    for upload_file, safe_name in zip(files, safe_names):
        contents = await upload_file.read()
        if len(contents) > MAX_BYTES_PER_FILE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Arquivo {safe_name} excede o limite de "
                    f"{MAX_BYTES_PER_FILE // (1024 * 1024)} MiB."
                ),
            )

        destination = UPLOAD_DIR / safe_name
        destination.write_bytes(contents)

        result = ingest_file(destination, store, embedder)
        if result.skipped:
            skipped.append(
                SkippedFile(
                    filename=safe_name,
                    reason=result.reason or "Arquivo pulado sem motivo declarado.",
                )
            )
        else:
            ingested.append(
                FileIngested(filename=safe_name, chunks=result.chunks_added)
            )

    return UploadResponse(
        files=ingested,
        skipped=skipped,
        total_chunks_in_collection=store.count(),
    )
