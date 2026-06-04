"""Rank fusion — a pure helper with no heavy dependencies.

Kept separate from retriever.py so it can be imported (and unit-tested) without
pulling in sentence-transformers and the cross-encoder model.
"""


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Combine several ranked id-lists into one fused score per id.

    RRF avoids having to normalize score scales between, say, cosine similarity
    and BM25: it uses only the *rank* of each item in each list. An item ranked
    near the top of multiple lists accumulates the most weight.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused
