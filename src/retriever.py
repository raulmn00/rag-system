"""
Retrieval: hybrid search + cross-encoder re-ranking.

This is the heart of the system and where retrieval quality is won or lost.

Pipeline:
  1. SEMANTIC search (dense vectors) — captures meaning/paraphrase.
  2. KEYWORD search (BM25, sparse) — captures exact terms, names, acronyms that
     embeddings sometimes wash out.
  3. FUSE the two ranked lists with Reciprocal Rank Fusion (RRF) — a simple,
     robust way to combine rankings without tuning score scales.
  4. RE-RANK the fused candidates with a cross-encoder, which scores each
     (query, chunk) pair jointly and is far more accurate than the first-stage
     retrievers — but too slow to run over the whole corpus, hence the
     retrieve-then-rerank pattern.

Each stage is independently switchable so the eval harness can measure the
contribution of each one.
"""

from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .embeddings import Embedder
from .fusion import reciprocal_rank_fusion
from .vector_store import VectorStore

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class Retrieved:
    chunk_id: str
    text: str
    source: str
    score: float


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        use_keyword: bool = True,
        use_rerank: bool = True,
    ):
        self.store = store
        self.embedder = embedder
        self.use_keyword = use_keyword
        self.use_rerank = use_rerank

        self._reranker = CrossEncoder(RERANK_MODEL) if use_rerank else None

        # Build the BM25 index once from the stored chunks.
        self._corpus = store.all_documents() if use_keyword else []
        if self._corpus:
            tokenized = [d["text"].lower().split() for d in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int = 5, candidates: int = 20) -> list[Retrieved]:
        # 1. Semantic.
        q_emb = self.embedder.embed_query(query)
        semantic = self.store.query(q_emb, top_k=candidates)
        sem_ids = [r["chunk_id"] for r in semantic]
        by_id = {r["chunk_id"]: r for r in semantic}

        # 2 + 3. Keyword + fusion (or semantic-only).
        if self.use_keyword and self._bm25 is not None:
            scores = self._bm25.get_scores(query.lower().split())
            ranked = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:candidates]
            kw_ids = [self._corpus[i]["chunk_id"] for i in ranked]
            for i in ranked:
                d = self._corpus[i]
                by_id.setdefault(d["chunk_id"], {**d, "score": 0.0})
            fused = reciprocal_rank_fusion([sem_ids, kw_ids])
            order = sorted(fused, key=lambda c: fused[c], reverse=True)
        else:
            order = sem_ids

        candidate_docs = [by_id[c] for c in order if c in by_id][:candidates]

        # 4. Re-rank.
        if self.use_rerank and self._reranker is not None and candidate_docs:
            pairs = [(query, d["text"]) for d in candidate_docs]
            rr_scores = self._reranker.predict(pairs)
            for d, s in zip(candidate_docs, rr_scores):
                d["score"] = float(s)
            candidate_docs.sort(key=lambda d: d["score"], reverse=True)

        return [
            Retrieved(
                chunk_id=d["chunk_id"], text=d["text"],
                source=d["source"], score=d["score"],
            )
            for d in candidate_docs[:top_k]
        ]
