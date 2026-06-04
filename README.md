# RAG System — Chat with your docs, with evaluation

A retrieval-augmented generation system that answers questions over a document
collection **with citations**, built to make every stage of the RAG pipeline
explicit: chunking, embeddings, hybrid retrieval, cross-encoder re-ranking,
grounded generation, and — the part most demos skip — a **retrieval evaluation
harness** that quantifies the contribution of each stage.

## Why this project

Anyone can call `index.query()`. This project demonstrates understanding of the
decisions that actually determine RAG quality, and backs them with numbers:

- **Chunking** with token-aware windows + overlap, and the trade-offs documented.
- **Hybrid retrieval**: dense vector search (semantic) fused with BM25 (keyword)
  via Reciprocal Rank Fusion — because embeddings miss exact terms and keywords
  miss paraphrase.
- **Cross-encoder re-ranking** of the candidate shortlist (the retrieve-then-
  rerank pattern).
- **Grounded generation** that answers only from retrieved context and cites
  passage numbers, reducing hallucination.
- **Evaluation** of retrieval quality (Hit Rate, MRR, Recall@k) comparing three
  configurations, so improvements are measured, not asserted.

## Architecture

```
INGEST (offline)                          QUERY (online)
─────────────────                         ──────────────
docs/*.md,*.txt                           user question
     │ chunk (token windows + overlap)         │ embed
     ▼                                          ▼
  embeddings (OpenAI text-embedding-3-small)  ┌──────────────────────────┐
     │                                        │ 1. semantic (dense)      │
     ▼                                        │ 2. BM25 (keyword)        │
  ChromaDB (persistent, cosine)  ───────────► │ 3. fuse (RRF)            │
                                              │ 4. cross-encoder re-rank │
                                              └────────────┬─────────────┘
                                                           ▼  top-k chunks
                                              generation (cites [1],[2],...)
                                                           ▼
                                                  grounded answer + sources
```

## Module map

| File | Responsibility |
|------|----------------|
| `src/chunking.py` | Token-aware chunking with overlap |
| `src/embeddings.py` | OpenAI embeddings (batch + query) |
| `src/vector_store.py` | ChromaDB persistence + vector query |
| `src/fusion.py` | Reciprocal Rank Fusion (pure, testable) |
| `src/retriever.py` | Hybrid search + cross-encoder re-ranking |
| `src/generation.py` | Cited, context-only answer generation |
| `src/pipeline.py` | Orchestrates retrieve → generate |
| `src/ingest.py` | CLI: read → chunk → embed → store |
| `src/api.py` | FastAPI endpoint `/ask` |
| `eval/metrics.py` | Hit Rate / MRR / Recall@k (pure functions) |
| `eval/run_eval.py` | Compares retriever configs on a gold set |
| `eval/ragas_eval.py` | LLM-judged answer quality (faithfulness, relevancy, ...) |

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# 1. Ingest the sample docs (or drop your own .md/.txt into data/docs/)
python -m src.ingest data/docs

# 2. Ask a question (API)
uvicorn src.api:app --reload
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?"}'

# 3. Evaluate retrieval — prints the comparison table below
python -m eval.run_eval
```

## Evaluation

The system is evaluated on **two complementary fronts**, because a fluent answer
built on the wrong context is still wrong, and perfect retrieval wasted by a bad
prompt is also wrong.

### 1. Retrieval quality — `eval/run_eval.py` (no LLM calls)

Runs three retriever configurations over a labeled gold set (`eval/gold.json`)
and reports retrieval metrics at k=5:

```
config                   hit_rate@5         mrr@5      recall@5
------------------------------------------------------------------
semantic_only                 ...            ...           ...
hybrid                        ...            ...           ...
hybrid_plus_rerank            ...            ...           ...
```

Run it after ingestion and paste your real numbers here. The expected story —
and the thing to talk through in an interview — is that adding keyword fusion
lifts recall (exact-term questions get found), and adding the cross-encoder
re-ranker lifts MRR (the right chunk moves to the top).

### 2. Answer quality — `eval/ragas_eval.py` (LLM-judged, via Ragas)

Measures the *generated answer*, not just retrieval, on four metrics:

- **faithfulness** — is every claim in the answer supported by the retrieved
  context? (the direct hallucination measure)
- **answer_relevancy** — does the answer actually address the question?
- **context_precision** — are the retrieved chunks relevant and well-ordered?
- **context_recall** — does the retrieved context cover the reference answer?
  (uses the `ground_truth` field in `eval/gold.json`)

```bash
pip install ragas datasets        # optional extras
export OPENAI_API_KEY=sk-...
python -m eval.ragas_eval
```

This one **does** call the judge LLM several times per question (a few cents on
the 8-item sample set). Flip the `use_keyword` / `use_rerank` flags in
`build_samples` to measure how much faithfulness depends on each retrieval
stage — e.g. showing faithfulness drops when re-ranking is off is a strong,
concrete result to present.

## Sample data

`data/docs/` ships with two original explainer documents (the Transformer
architecture and RAG itself) so the demo works immediately. For a richer demo,
add real papers as `.txt`/`.md` — e.g. export arXiv papers — and re-run
ingestion. The gold set targets document-level relevance, so it stays valid as
you change chunking parameters.

## Design decisions & trade-offs

- **ChromaDB over a hosted vector DB**: runs in-process with on-disk
  persistence, nothing to provision for a demo. Production swap: Qdrant or
  pgvector — only `vector_store.py` changes.
- **RRF over weighted score fusion**: no score-scale tuning between cosine and
  BM25; combines on rank alone.
- **Retrieve-then-rerank**: the cross-encoder is accurate but too slow for the
  whole corpus, so it runs only on the ~20-candidate shortlist.
- **Document-level eval labels**: stable across chunking changes, so the harness
  keeps working as you tune chunk size/overlap.

## Tests

```bash
pytest tests/ -v
```

Covers chunking, rank fusion, and all retrieval metrics — pure logic, no API
calls or model downloads.

## Tech stack

Python · FastAPI · OpenAI embeddings · ChromaDB · rank-bm25 ·
sentence-transformers (cross-encoder) · tiktoken
