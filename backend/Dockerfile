FROM python:3.12-slim

WORKDIR /app

# Copy both packages first. Build context is the repo root:
#   docker build -f backend/Dockerfile -t rag-backend .
COPY rag-core/ ./rag-core/
COPY backend/ ./backend/

# Install rag-core (editable) so backend imports work, then install the
# backend's own deps. Editable install keeps the rag_core package
# importable as `rag_core` without copying it under site-packages.
RUN pip install --no-cache-dir -e ./rag-core \
 && pip install --no-cache-dir -e ./backend

# Pre-cache the cross-encoder so the first /ask after a cold start doesn't
# pay a ~90 MB HuggingFace download on the user's request thread. Adds ~90 MB
# to the image; saves 10-15s off the first warm-up request. The model name
# is duplicated from rag_core/retriever.py:RERANK_MODEL — small price for
# keeping this Dockerfile self-contained and trivially auditable.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Run from rag-core/ so the default relative paths used by the pipeline
# (data/chroma, data/docs) resolve against the right directory.
WORKDIR /app/rag-core

EXPOSE 8000
CMD uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}
