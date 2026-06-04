"""POST /upload — runtime ingestion of .md and .txt files via multipart."""

import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag_core.embeddings import Embedder
from rag_core.ingest import ingest_file
from rag_core.vector_store import VectorStore


router = APIRouter()

ALLOWED_SUFFIXES = {".md", ".txt"}
MAX_BYTES_PER_FILE = 5 * 1024 * 1024  # 5 MiB — covers anything textual
UPLOAD_DIR = Path("data/uploads")  # resolved against cwd (run uvicorn from rag-core/)


class FileIngested(BaseModel):
    filename: str
    chunks: int


class UploadResponse(BaseModel):
    files: list[FileIngested]
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


@router.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
    """Receive one or more .md/.txt files, persist them under
    data/uploads/, and ingest them into the vector store. Returns the
    per-file chunk counts plus the total now in the collection.

    All-or-nothing on validation: extension + filename are checked
    up-front for every file before anything is written. Size is checked
    per-file as we read it. A bad file in a batch fails the whole
    request — partial state would be worse than a clean retry.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo recebido.")

    # First pass: validate extension + sanitize names. Nothing on disk yet.
    safe_names: list[str] = []
    for upload_file in files:
        suffix = Path(upload_file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Tipo de arquivo não suportado: {upload_file.filename!r}. "
                    "Apenas .md e .txt são aceitos."
                ),
            )
        safe_names.append(_safe_filename(upload_file.filename or ""))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Constructing these per request matches the /ask handler's pattern.
    # A real prod deploy would inject them via FastAPI dependencies.
    store = VectorStore()
    embedder = Embedder()

    ingested: list[FileIngested] = []
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
        chunks_added = ingest_file(destination, store, embedder)
        ingested.append(FileIngested(filename=safe_name, chunks=chunks_added))

    return UploadResponse(
        files=ingested,
        total_chunks_in_collection=store.count(),
    )
