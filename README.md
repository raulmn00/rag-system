# RAG System — Monorepo

A retrieval-augmented generation system that answers questions over a
document collection **with citations**, built to make every stage of
the RAG pipeline explicit: chunking, embeddings, hybrid retrieval,
cross-encoder re-ranking, grounded generation, and a quantitative
evaluation harness.

## Live demo

| | URL |
|---|---|
| **Frontend** | https://rag-system-delta-five.vercel.app |
| **Backend API** | https://rag-system-api-909428365094.us-central1.run.app |
| **Health check** | [`GET /`](https://rag-system-api-909428365094.us-central1.run.app/) |
| **Repo** | https://github.com/raulmn00/rag-system |

> Deployed on Vercel (frontend) + Google Cloud Run (backend, scale-to-zero).
> Storage is ephemeral on the free tier — the vector store and uploaded
> documents are reset whenever Cloud Run hibernates after idle. Re-upload
> a file to ask questions about it in a fresh session.

## Architecture

This repo is a **monorepo** with three apps:

```
.
├── rag-core/     domain logic — chunking, embeddings, retrieval,
│                 generation, the pipeline. Importable Python package
│                 (`rag_core`). No web framework. No HTTP.
├── backend/     FastAPI transport layer. Depends on rag-core via
│                 editable install. Holds every web dependency
│                 (fastapi, uvicorn, python-multipart, slowapi) so
│                 rag-core stays framework-agnostic.
└── frontend/    React + TypeScript SPA (Vite, no UI library).
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

- `GET  /`         — health check.
- `POST /ask`      — `{ question, use_keyword, use_rerank, top_k }` →
  `{ answer, sources[] }`.
- `POST /upload`   — multipart `.md`/`.txt` files → ingest into the
  vector store. Returns per-file chunk counts plus the collection total.
- `POST /evaluate` — reference-free quality scoring of a single
  `{ question, answer, contexts }`. Returns `{ faithfulness,
  answer_relevancy }`. Implemented in-tree (no Ragas / no extras
  required) — see `rag-core/rag_core/evaluation.py` and "Evaluation
  metrics" below.

### Evaluation (offline harness)

```bash
cd rag-core
python -m eval.run_eval                   # retrieval metrics, no API cost
python -m eval.quality_eval               # LLM-judged answer quality
```

`run_eval.py` measures **retrieval** quality (Hit Rate@k, MRR@k,
Recall@k — pure functions, no LLM cost). `quality_eval.py` measures
**answer** quality using the same Evaluator the backend's
`/evaluate` route uses.

### Evaluation metrics — how they work

Two LLM-judged metrics, both implemented in-tree against the OpenAI
SDK (no Ragas, no third-party eval framework). See
[`rag-core/rag_core/evaluation.py`](rag-core/rag_core/evaluation.py)
for the prompts and the math; unit-tested in
[`rag-core/tests/test_rag.py`](rag-core/tests/test_rag.py) with the
OpenAI client mocked out.

| Metric | What it measures | Implementation |
|---|---|---|
| `faithfulness` | Every factual claim in the answer is supported by the retrieved contexts (anti-hallucination) | 1) LLM extracts atomic claims from the answer. 2) LLM judges each claim against the contexts in a batched call. Score = supported / total. |
| `answer_relevancy` | The answer actually addresses the question that was asked | 1) LLM generates K=4 questions the answer plausibly answers. 2) Cosine similarity between the embedding of each generated question and the embedding of the original question. Score = mean, clamped to [0, 1]. |

Cost per `/evaluate` call: ~3 LLM judge calls + 1 batched embeddings
call on `gpt-4o-mini` + `text-embedding-3-small` ≈ **$0.0005**.
Latency: 3–8 seconds.

`context_precision` and `context_recall` (which Ragas surfaces) are
deliberately not included — both need a ground-truth reference, which
isn't available at request time.

### Frontend

```bash
cd frontend
npm install
npm run dev                                # :5173
```

Vite + React 19 + TypeScript (strict). Single page wired with three
flows: drag-and-drop upload, ask with citations, and an LLM-judged
metrics panel (animated SVG gauges driven by the in-tree evaluator).
Reads the backend URL from `VITE_API_URL` — see
`frontend/.env.example`.

```bash
cd frontend
npm run build                              # tsc -b && vite build
```

### Container (backend only)

```bash
# Build with the REPO ROOT as the context so both packages get copied.
docker build -t rag-backend .
docker run --rm -p 8000:8000 --env-file .env rag-backend
```

## Tests

```bash
cd rag-core
pytest tests/ -v
```

Tests cover the pure parts (chunking, fusion, retrieval metrics) plus
the LLM-judged evaluator (with the OpenAI client mocked out) — no API
key, no model downloads, no network. 31 tests, all in well under a
second.

## What lives where, briefly

| File | What it is | Why it's there |
|---|---|---|
| `rag-core/rag_core/chunking.py` | Token-aware chunker | tiktoken windows + overlap |
| `rag-core/rag_core/embeddings.py` | OpenAI embedding client | One method per direction (docs, query) |
| `rag-core/rag_core/vector_store.py` | Chroma wrapper | Persistent, in-process |
| `rag-core/rag_core/retriever.py` | Hybrid retrieval + rerank | BM25 + semantic + RRF + cross-encoder |
| `rag-core/rag_core/generation.py` | Grounded answer + citations | Refuses if context is empty |
| `rag-core/rag_core/pipeline.py` | The 5-line orchestrator | retrieve → generate |
| `rag-core/rag_core/evaluation.py` | LLM-judged Evaluator | Self-contained faithfulness + answer relevancy |
| `rag-core/eval/metrics.py` | hit_rate, MRR, recall | Pure functions, no API |
| `rag-core/eval/run_eval.py` | Retrieval A/B harness | semantic vs hybrid vs +rerank |
| `rag-core/eval/quality_eval.py` | Offline answer-quality harness | Loops the Evaluator over the gold set |
| `backend/app/api.py` | FastAPI app entry | Wires CORS + the per-route routers |
| `backend/app/routes/ask.py` | POST /ask | Thin transport over `rag_core.pipeline` |
| `backend/app/routes/upload.py` | POST /upload | Multipart + sanitize + reuse `ingest_file` |
| `backend/app/routes/evaluate.py` | POST /evaluate | Wraps `rag_core.evaluation.Evaluator` |
| `Dockerfile` (root) | Production container | Builds rag-core + backend into one image |
| `frontend/src/App.tsx` | SPA state machine | Three independent flows (upload/ask/evaluate) |
| `frontend/src/components/MetricsPanel.tsx` | Ragas gauges | Animated SVG arcs, color by threshold |
