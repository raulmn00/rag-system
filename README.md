# RAG System — Monorepo

A retrieval-augmented generation system that answers questions over a
document collection **with citations**, built to make every stage of
the RAG pipeline explicit: chunking, embeddings, hybrid retrieval,
cross-encoder re-ranking, grounded generation, and a quantitative
evaluation harness.

This repo is a **monorepo** with three apps:

```
.
├── rag-core/     domain logic — chunking, embeddings, retrieval,
│                 generation, the pipeline. Importable Python package
│                 (`rag_core`). No web framework. No HTTP.
├── backend/     FastAPI transport layer. Depends on rag-core via
│                 editable install. Holds every web dependency
│                 (fastapi, uvicorn, python-multipart) so rag-core
│                 stays framework-agnostic.
└── frontend/    React + TypeScript SPA (added later in development).
```

Separating these three lets the same domain code be consumed by
different transports — a CLI, an HTTP API, a notebook — without
dragging FastAPI around. It also keeps the dependency surface honest:
`pip install rag-core` won't pull uvicorn; `npm install` in
`frontend/` doesn't pull Python.

## Setup

Requires Python 3.10+. Set `OPENAI_API_KEY` in the repo-root `.env`
(see `.env.example`).

```bash
# 1. Make a virtualenv and activate it
python3 -m venv .venv
source .venv/bin/activate

# 2. Install rag-core first, then the backend that depends on it.
#    Order matters because the backend imports rag_core; both are
#    editable so changes take effect without re-installing.
pip install -e ./rag-core
pip install -e ./backend

# 3. Optional: dev tooling and the heavier Ragas-based eval extras.
pip install -e "./rag-core[dev]"          # pytest
pip install -e "./rag-core[eval]"         # ragas + datasets
```

`python-dotenv` loads the `.env` from the repo root on package import
(see `rag_core/config.py`), so `OPENAI_API_KEY` is available to every
entry point without a manual `export`.

## Running each part

### Ingest the sample corpus

```bash
cd rag-core
python -m rag_core.ingest data/docs
```

Reads `.md` / `.txt` files, chunks them, embeds in batches, and stores
in `data/chroma/` (persistent on disk, gitignored).

### Backend (FastAPI)

```bash
cd rag-core
uvicorn app.api:app --reload --port 8000
```

Run from `rag-core/` so the default relative paths (`data/chroma`,
`data/docs`) resolve correctly. The `app` package is importable from
anywhere thanks to the editable install — uvicorn finds it via the
Python path.

Endpoints:

- `GET /` — health check.
- `POST /ask` — `{ question, use_keyword, use_rerank, top_k }` →
  `{ answer, sources[] }`.

### Evaluation

```bash
cd rag-core
python -m eval.run_eval                   # retrieval metrics, no extra cost
python -m eval.ragas_eval                 # LLM-judged answer quality (needs `eval` extra)
```

### Frontend

Added in a later step. Will live in `frontend/` as a Vite + React + TS
project. Will read `VITE_API_URL` for the backend's base URL.

### Container (backend only)

```bash
# Build with the REPO ROOT as the context so both packages get copied.
docker build -f backend/Dockerfile -t rag-backend .
docker run --rm -p 8000:8000 --env-file .env rag-backend
```

## Tests

```bash
cd rag-core
pytest tests/ -v
```

Tests cover the pure parts (chunking, fusion, metrics) — no API key,
no model downloads, no network. Run in well under a second.

## What lives where, briefly

| File | What it is | Why it's there |
|---|---|---|
| `rag-core/rag_core/chunking.py` | Token-aware chunker | tiktoken windows + overlap |
| `rag-core/rag_core/embeddings.py` | OpenAI embedding client | One method per direction (docs, query) |
| `rag-core/rag_core/vector_store.py` | Chroma wrapper | Persistent, in-process |
| `rag-core/rag_core/retriever.py` | Hybrid retrieval + rerank | BM25 + semantic + RRF + cross-encoder |
| `rag-core/rag_core/generation.py` | Grounded answer + citations | Refuses if context is empty |
| `rag-core/rag_core/pipeline.py` | The 5-line orchestrator | retrieve → generate |
| `rag-core/eval/metrics.py` | hit_rate, MRR, recall | Pure functions, no API |
| `rag-core/eval/run_eval.py` | Retrieval A/B harness | semantic vs hybrid vs +rerank |
| `rag-core/eval/ragas_eval.py` | Answer-quality eval | Faithfulness, relevancy, precision, recall |
| `backend/app/api.py` | FastAPI app | Thin transport over `rag_core.pipeline` |
| `backend/Dockerfile` | Production container | Installs both packages editably |

## Roadmap (in progress)

- **Part 1 (next)**: add `POST /upload` for runtime ingestion of `.md`/`.txt` files, and `POST /evaluate` exposing on-demand Ragas faithfulness + answer-relevancy.
- **Part 2 (after Part 1)**: React + TypeScript frontend in `frontend/` — upload area, ask form, sources panel, and a Ragas-driven metrics dashboard.
